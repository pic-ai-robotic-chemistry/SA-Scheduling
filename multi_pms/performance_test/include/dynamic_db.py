from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime, timedelta
from typing import Any

from obj.db_tools import SchedulerDatabase


# start is immediately completed; exclude start/lock/unlock from frozen-row counts.
def count_dynamic_task_operations(task: dict[str, Any]) -> int:
    steps = task.get("steps", [])
    if not isinstance(steps, list):
        return 0
    return sum(
        1
        for step in steps
        if isinstance(step, dict)
        and step.get("workstationType") not in {"start", "lock", "unlock"}
    )


# The dynamic client and service use the same stable frozen-row digest.
def canonical_schedule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (copy.deepcopy(row) for row in rows),
        key=lambda row: (str(row.get("expr_no")), int(row.get("fjspb_index", -1))),
    )


# Hash complete task_scheduled rows so machine and step changes cannot escape time-only validation.
def schedule_rows_digest(rows: list[dict[str, Any]]) -> str:
    canonical = canonical_schedule_rows(rows)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class DynamicReplayDatabase(SchedulerDatabase):
    """Global timeline database used only for dynamic replay."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dynamic_simulation_origin: datetime | None = None

    def set_dynamic_simulation_origin(self, origin: datetime) -> None:
        # The controller sets this only for batch 0 of a new run.
        self.dynamic_simulation_origin = origin.replace(microsecond=0)

    def construct_fespb_jobs_data_from_db(
        self,
        cur_ptr: int,
        expr_no_list: list,
        is_plot: bool = False,
    ) -> dict:
        jobs_data = copy.deepcopy(super().construct_fespb_jobs_data_from_db(cur_ptr, expr_no_list, is_plot=is_plot))
        if not is_plot:
            if self.dynamic_simulation_origin is None:
                raise RuntimeError("dynamic simulation origin is not initialized")
            # Surviving tasks may have only a started final step; it occupies a machine but is not a job to reschedule.
            # Keep machine availability to preserve occupancy; trim only job state that must map to details.
            schedulable_expr_nos = set(jobs_data["detail"])
            for info_key in ("job_avail_time", "job_lock", "job_lock_periods", "job_rightnow"):
                info_value = jobs_data["info"].get(info_key, {})
                jobs_data["info"][info_key] = {
                    expr_no: value
                    for expr_no, value in info_value.items()
                    if expr_no in schedulable_expr_nos
                }
            # Classify tariffs from the fixed simulation origin plus the global minute pointer.
            jobs_data["info"]["sim_cur_time_stamp"] = (
                self.dynamic_simulation_origin + timedelta(minutes=int(cur_ptr))
            ).strftime("%Y-%m-%d %H:%M:%S")
        return jobs_data

    def has_reschedulable_operations(self, cur_ptr: int, expr_no_list: list[str]) -> bool:
        """Whether the current dynamic window still contains steps that start after the pointer and need to be rescheduled."""
        if not expr_no_list:
            return False
        placeholders = ",".join("?" for _ in expr_no_list)
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 1
                FROM task_scheduled
                WHERE expr_no IN ({placeholders})
                  AND start_time >= ?
                  AND status != 'completed'
                LIMIT 1
                """,
                (*expr_no_list, int(cur_ptr)),
            )
            return cursor.fetchone() is not None

    @staticmethod
    def _flatten_ws_table(ws_table: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], dict[str, Any]]:
        rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for tasks in ws_table.values():
            for task in tasks:
                key = (str(task["expr_no"]), int(task["fjspb_index"]))
                if key in rows_by_key:
                    raise RuntimeError(f"duplicate schedule row in ws_table: {key}")
                rows_by_key[key] = copy.deepcopy(task)
        return rows_by_key

    def _fetch_finite_times(self, expr_no_list: list[str]) -> dict[tuple[str, int], tuple[int, int]]:
        if not expr_no_list:
            return {}
        placeholders = ",".join("?" for _ in expr_no_list)
        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT expr_no, fjspb_index, start_time, end_time
                FROM task_scheduled
                WHERE expr_no IN ({placeholders})
                  AND start_time < ?
                  AND end_time IS NOT NULL
                """,
                (*expr_no_list, sys.maxsize),
            )
            return {
                (str(expr_no), int(fjspb_index)): (int(start_time), int(end_time))
                for expr_no, fjspb_index, start_time, end_time in cursor.fetchall()
            }

    @staticmethod
    def _build_lock_periods(rows: list[tuple[int, int, int, int]]) -> list[list[int]]:
        lock_periods: list[list[int]] = []
        in_lock = False
        lock_start: int | None = None
        previous_end: int | None = None
        for _, start_time, end_time, is_lock in rows:
            if is_lock:
                if not in_lock:
                    in_lock = True
                    lock_start = start_time
            elif in_lock:
                if lock_start is not None and previous_end is not None:
                    lock_periods.append([lock_start, previous_end])
                in_lock = False
                lock_start = None
            previous_end = end_time
        if in_lock and lock_start is not None and previous_end is not None:
            lock_periods.append([lock_start, previous_end])
        return lock_periods

    def write_ws_table_to_db(self, task_group: str, expr_no_list: list, ws_table: dict) -> int:
        incoming_rows = self._flatten_ws_table(ws_table)
        expr_nos = [str(expr_no) for expr_no in expr_no_list]
        # The parent shifts active-task tables; save history absent from ws_table to protect surviving tasks' past steps.
        previous_times = self._fetch_finite_times(expr_nos)
        super().write_ws_table_to_db(task_group, expr_no_list, ws_table)

        with self.lock, self._connect() as conn:
            cursor = conn.cursor()
            conn.execute("BEGIN")

            for key, (start_time, end_time) in previous_times.items():
                if key in incoming_rows:
                    continue
                cursor.execute(
                    """
                    UPDATE task_scheduled
                    SET start_time = ?, end_time = ?
                    WHERE expr_no = ? AND fjspb_index = ?
                    """,
                    (start_time, end_time, key[0], key[1]),
                )

            for (expr_no, fjspb_index), task in incoming_rows.items():
                cursor.execute(
                    """
                    UPDATE task_scheduled
                    SET start_time = ?, end_time = ?, duration = ?
                    WHERE expr_no = ? AND fjspb_index = ?
                    """,
                    (
                        int(task["start_time"]),
                        int(task["end_time"]),
                        int(task["duration"]),
                        expr_no,
                        fjspb_index,
                    ),
                )

            # The parent derives lock_periods from shifted times; recompute only current-window tasks after restoring global time.
            for expr_no in expr_nos:
                cursor.execute(
                    """
                    SELECT fjspb_index, start_time, end_time, lock
                    FROM task_scheduled
                    WHERE expr_no = ? AND start_time < ? AND end_time IS NOT NULL
                    ORDER BY fjspb_index
                    """,
                    (expr_no, sys.maxsize),
                )
                lock_rows = [
                    (int(index), int(start), int(end), int(bool(is_lock)))
                    for index, start, end, is_lock in cursor.fetchall()
                ]
                cursor.execute(
                    "UPDATE task_scheduled SET lock_periods = ? WHERE expr_no = ?",
                    (json.dumps(self._build_lock_periods(lock_rows)), expr_no),
                )

            conn.commit()

            for (expr_no, fjspb_index), task in incoming_rows.items():
                cursor.execute(
                    """
                    SELECT start_time, end_time, duration
                    FROM task_scheduled
                    WHERE expr_no = ? AND fjspb_index = ?
                    """,
                    (expr_no, fjspb_index),
                )
                actual = cursor.fetchone()
                expected = (
                    int(task["start_time"]),
                    int(task["end_time"]),
                    int(task["duration"]),
                )
                if actual is None or tuple(map(int, actual)) != expected:
                    raise RuntimeError(
                        f"dynamic global time verification failed for {(expr_no, fjspb_index)}: "
                        f"expected={expected}, actual={actual}"
                    )

            placeholders = ",".join("?" for _ in expr_nos)
            cursor.execute(
                f"SELECT MAX(end_time) FROM task_scheduled WHERE expr_no IN ({placeholders})",
                expr_nos,
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                raise RuntimeError("dynamic schedule did not produce a finite makespan")
            return int(row[0])
