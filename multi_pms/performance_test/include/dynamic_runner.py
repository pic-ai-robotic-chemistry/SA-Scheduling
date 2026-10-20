from __future__ import annotations

from algorithm.utils.tariff import daily_tariff, tariff_snapshot

import copy
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from obj.db_tools import SchedulerDatabase

from .dynamic_db import canonical_schedule_rows, count_dynamic_task_operations, schedule_rows_digest
from .gantt import DEFAULT_COLOR_BY_TASK, ExperimentSnapshot, HorizonGanttPlotter
from .runner import (
    HorizonReplayRunner,
    build_schedule_method_output_name,
    format_datetime,
    load_json_file,
    parse_optional_float,
)

# Protocol v3 completes a run once the final active window has been solved.
DYNAMIC_PROTOCOL_VERSION = 3
DYNAMIC_ROUTE = "/scheduling_dynamic_test"
RL_CP_RESULT_SOURCES = {"rl_incumbent", "cp_optimized"}  # Allowed final sources for RL+CP dynamic batches.


@dataclass(slots=True)
class DynamicReplayResult:
    snapshot: ExperimentSnapshot | None
    total_makespan: int | None
    total_carbon_objective_scaled: int | None
    total_electricity_cost_yuan: float | None
    total_algo_cal_time: float | None
    algo_cal_times: list[float | None]
    total_operation_count: int
    schedule_method: str
    schedule_status: str
    output_path: Path | None
    total_tasks: int
    used_files: list[Path]
    start_time: datetime
    end_time: datetime | None
    dynamic_batches: list[dict[str, Any]]
    # Store cumulative admitted-task metrics for task-count plots.
    dynamic_task_metrics: list[dict[str, Any]]
    run_id: str
    server_session_id: str | None
    schedule_time_limit: float | None  # Actual per-batch solve time limit used by the dynamic service.
    error_message: str | None = None


class DynamicHorizonReplayRunner:
    """Sliding-window client and frozen-history accumulator for dynamic replay."""

    def __init__(
        self,
        database_path: str | Path,
        output_dir: str | Path,
        logger: Any,
        schedule_port: int = 5082,
        request_timeout: float = 2100.0,
        replay_interval: float = 5.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.base_url = f"http://127.0.0.1:{int(schedule_port)}{DYNAMIC_ROUTE}"
        self.request_timeout = float(request_timeout)
        self.replay_interval = float(replay_interval)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None

    def _health_handshake(self) -> dict[str, Any]:
        while True:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=min(15.0, self.request_timeout))
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                # Keep waiting for service connectivity.
                print(
                    f"[dynamic_health_handshake] service unavailable, retrying in "
                    f"{self.replay_interval:.0f}s: {exc}",
                    flush=True,
                )
                time.sleep(self.replay_interval)

        expected = {
            "controller": "dynamic_replay",
            "protocol_version": DYNAMIC_PROTOCOL_VERSION,
            "route": DYNAMIC_ROUTE,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(
                    f"dynamic health mismatch for {field}: expected={value!r}, actual={payload.get(field)!r}"
                )
        session_id = str(payload.get("server_session_id") or "").strip()
        if not session_id:
            raise RuntimeError("dynamic health response has no server_session_id")
        actual_db_path = Path(str(payload.get("database_path") or "")).expanduser().resolve()
        if actual_db_path != self.database_path:
            raise RuntimeError(f"dynamic database mismatch: expected={self.database_path}, actual={actual_db_path}")
        return payload

    @staticmethod
    def _http_json(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"dynamic service returned non-JSON HTTP {response.status_code}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("dynamic service response must be a JSON object")
        return payload

    def _submit_and_wait(
        self,
        payload: dict[str, Any],
        run_id: str,
        batch_index: int,
        server_session_id: str,
    ) -> dict[str, Any]:
        submitted = False
        while True:
            try:
                if not submitted:
                    response = requests.post(
                        self.base_url,
                        json=payload,
                        timeout=min(30.0, self.request_timeout),
                    )
                else:
                    response = requests.get(
                        f"{self.base_url}/status/{run_id}/{batch_index}",
                        timeout=min(30.0, self.request_timeout),
                    )
            except (requests.ConnectionError, requests.Timeout):
                # Retry temporary connection failures at the configured replay interval.
                print(
                    f"[dynamic_submit_and_wait] batch {batch_index} service unavailable, retrying in "
                    f"{self.replay_interval:.0f}s",
                    flush=True,
                )
                time.sleep(self.replay_interval)
                continue

            body = self._http_json(response)
            if 400 <= response.status_code < 500:
                if submitted and response.status_code == 404:
                    # Service restarts often make status return 404; handshake again to detect session changes explicitly.
                    refreshed_health = self._health_handshake()
                    if refreshed_health.get("server_session_id") != server_session_id:
                        raise RuntimeError("dynamic service session changed during experiment")
                raise RuntimeError(
                    f"dynamic batch rejected: HTTP {response.status_code}, {body.get('message') or body}"
                )
            if response.status_code >= 500:
                time.sleep(self.replay_interval)
                continue

            submitted = True
            status = str(body.get("dynamic_status") or "").strip().lower()
            if status == "pending":
                time.sleep(self.replay_interval)
                continue
            if status not in {"ready", "error"}:
                raise RuntimeError(f"unknown dynamic_status: {status!r}")
            if body.get("dynamic_run_id") != run_id or body.get("dynamic_batch_index") != batch_index:
                raise RuntimeError("dynamic response run/batch token mismatch")
            if body.get("server_session_id") != server_session_id:
                raise RuntimeError("dynamic service session changed during experiment")
            if status == "error":
                raise RuntimeError(str(body.get("message") or "dynamic scheduling failed"))
            return body

    def _save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        file_repeat_index: dict[Path, int],
    ) -> None:
        if self.checkpoint_path is None:
            return
        current: dict[str, Any] = {}
        if self.checkpoint_path.exists():
            try:
                loaded = load_json_file(self.checkpoint_path)
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, ValueError, json.JSONDecodeError):
                current = {}
        current.update(copy.deepcopy(checkpoint))
        current["fileRepeatIndex"] = {
            str(source_file.resolve()): int(repeat_index) for source_file, repeat_index in file_repeat_index.items()
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.checkpoint_path.with_suffix(f"{self.checkpoint_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(current, file, ensure_ascii=False, indent=2)
        temp_path.replace(self.checkpoint_path)

    @staticmethod
    def _row_identity(row: dict[str, Any]) -> str:
        return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    # Independently validate full window schedules to prevent redraws with missing or mismatched rows.
    def _validate_window_schedule(
        self,
        response: dict[str, Any],
        active_ids: list[str],
        expected_operation_counts: dict[str, int],
    ) -> list[dict[str, Any]]:
        raw_rows = response.get("dynamic_window_schedule_rows")
        if not isinstance(raw_rows, list):
            raise RuntimeError("dynamic window schedule rows must be a list")
        rows = canonical_schedule_rows(raw_rows)
        if schedule_rows_digest(rows) != response.get("dynamic_window_schedule_digest"):
            raise RuntimeError("dynamic window schedule digest mismatch")

        active_set = set(active_ids)
        rows_by_expr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_keys: set[tuple[str, int]] = set()
        for row in rows:
            expr_no = str(row.get("expr_no") or "")
            if expr_no not in active_set:
                raise RuntimeError("dynamic window schedule contains an inactive task")
            try:
                fjspb_index = int(row["fjspb_index"])
                start_time = int(row["start_time"])
                end_time = int(row["end_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("dynamic window schedule has invalid key/time fields") from exc
            if start_time < 0 or start_time >= sys.maxsize or end_time < start_time:
                raise RuntimeError("dynamic window schedule contains a non-finite row")
            key = (expr_no, fjspb_index)
            if key in seen_keys:
                raise RuntimeError(f"duplicate dynamic window schedule key: {key}")
            seen_keys.add(key)
            rows_by_expr[expr_no].append(row)

        for expr_no in active_ids:
            expected_count = expected_operation_counts[expr_no]
            if len(rows_by_expr[expr_no]) != expected_count:
                raise RuntimeError(
                    f"dynamic window task {expr_no} has {len(rows_by_expr[expr_no])} rows, "
                    f"expected {expected_count}"
                )
        return rows

    @staticmethod
    def _rows_to_tables(
        rows: list[dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
        raw_ws_table: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_bottle_table: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in canonical_schedule_rows(rows):
            ws_code = str(row.get("ws_code_fjspb") or "")
            expr_no = str(row["expr_no"])
            if not ws_code:
                raise RuntimeError(f"schedule row has no workstation: {(expr_no, row.get('fjspb_index'))}")
            raw_ws_table[ws_code].append(copy.deepcopy(row))
            raw_bottle_table[expr_no].append(copy.deepcopy(row))
        return dict(raw_ws_table), dict(raw_bottle_table)

    def _validate_and_freeze(
        self,
        response: dict[str, Any],
        active_ids: list[str],
        expected_operation_counts: dict[str, int],
        completed_ids: set[str],
        frozen_rows: dict[tuple[str, int], dict[str, Any]],
        ptr_before: int,
        requested_count: int,
    ) -> tuple[list[str], int, list[dict[str, Any]]]:
        if int(response.get("dynamic_ptr_before", -1)) != ptr_before:
            raise RuntimeError("dynamic_ptr_before does not match the client pointer")
        ptr_after = int(response.get("dynamic_ptr_after", -1))
        if ptr_after <= ptr_before:
            raise RuntimeError("dynamic batch made no pointer progress")
        if int(response.get("dynamic_requested_completion_count", -1)) != requested_count:
            raise RuntimeError("dynamic requested completion count mismatch")

        window_rows = self._validate_window_schedule(response, active_ids, expected_operation_counts)

        completed_items = response.get("dynamic_completed_tasks")
        if not isinstance(completed_items, list) or not completed_items:
            raise RuntimeError("dynamic batch returned no completed tasks")
        batch_completed_ids = [str(item.get("expr_no") or "") for item in completed_items]
        if len(set(batch_completed_ids)) != len(batch_completed_ids):
            raise RuntimeError("dynamic batch returned duplicate completed tasks")
        if not set(batch_completed_ids).issubset(set(active_ids)):
            raise RuntimeError("dynamic batch returned an unknown completed task")
        if set(batch_completed_ids) & completed_ids:
            raise RuntimeError("dynamic batch completed a task more than once")
        if int(response.get("dynamic_actual_completion_count", -1)) != len(batch_completed_ids):
            raise RuntimeError("dynamic actual completion count mismatch")
        for item in completed_items:
            if int(item.get("completion_time", -1)) > ptr_after:
                raise RuntimeError("completed task extends past dynamic_ptr_after")

        raw_rows = response.get("dynamic_completed_schedule_rows")
        if not isinstance(raw_rows, list):
            raise RuntimeError("dynamic completed schedule rows must be a list")
        rows = canonical_schedule_rows(raw_rows)
        if schedule_rows_digest(rows) != response.get("dynamic_completed_schedule_digest"):
            raise RuntimeError("dynamic completed schedule digest mismatch")

        rows_by_expr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        pending_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            expr_no = str(row.get("expr_no") or "")
            if expr_no not in batch_completed_ids:
                raise RuntimeError("a frozen row belongs to a non-completed task")
            try:
                fjspb_index = int(row["fjspb_index"])
                start_time = int(row["start_time"])
                end_time = int(row["end_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("a frozen row has invalid key/time fields") from exc
            if start_time < 0 or start_time >= sys.maxsize or end_time < start_time or end_time > ptr_after:
                raise RuntimeError("a frozen row is outside the finite completed horizon")
            key = (expr_no, fjspb_index)
            if key in pending_rows:
                raise RuntimeError(f"duplicate frozen primary key in one batch: {key}")
            previous = frozen_rows.get(key)
            if previous is not None and self._row_identity(previous) != self._row_identity(row):
                raise RuntimeError(f"frozen primary key changed across an idempotent replay: {key}")
            pending_rows[key] = copy.deepcopy(row)
            rows_by_expr[expr_no].append(row)

        for expr_no in batch_completed_ids:
            expected_count = expected_operation_counts[expr_no]
            if len(rows_by_expr[expr_no]) != expected_count:
                raise RuntimeError(
                    f"completed task {expr_no} has {len(rows_by_expr[expr_no])} rows, expected {expected_count}"
                )
            if max(int(row["end_time"]) for row in rows_by_expr[expr_no]) > ptr_after:
                raise RuntimeError(f"completed task {expr_no} extends past the pointer")

        # Validate the whole batch before merging it atomically into authoritative frozen history.
        frozen_rows.update(pending_rows)
        return batch_completed_ids, ptr_after, window_rows

    def run(
        self,
        experiment_files: list[Path],
        experiment_file_selection: list[Path],
        payload_cache: dict[Path, dict[str, Any]],
        file_repeat_index: dict[Path, int],
        window_size: int,
        advance_count: int,
        schedule_method: str,
        peak_price: float | None,
        valley_price: float | None,
        makespan_60min_cost_in_valley_price_units: float | None,
        simulation_start_time: datetime,
        output_name: str,
        experiment_name: str,
        time_scale: float = 1.0,
        color_by_task: bool = DEFAULT_COLOR_BY_TASK,
        objective_limit: int | None = None,
        makespan_limit: int | None = None,
        carbon_limit: int | None = None,
        tariff_info: dict[str, Any] | None = None,
    ) -> DynamicReplayResult:
        if window_size < 1 or advance_count < 1 or advance_count > window_size:
            raise ValueError("dynamic window/advance values are invalid")

        # Prepare all k tasks at once without sending taskList yet.
        payload_builder = HorizonReplayRunner(None, None, schedule_url="dynamic://unused")
        (
            prepared_payload,
            _,
            total_tasks,
            used_files,
            expr_nos_by_source_file,
        ) = payload_builder._build_experiment_payload(
            experiment_files=experiment_file_selection,
            payload_cache=payload_cache,
            target_start=simulation_start_time.replace(microsecond=0),
            file_repeat_index=file_repeat_index,
            time_scale=time_scale,
            remaining_tasks=None,
        )
        prepared_tasks = [copy.deepcopy(task) for task in prepared_payload.get("taskList", [])]
        if not prepared_tasks:
            raise ValueError("dynamic experiment contains no tasks")
        prepared_ids = [str(task.get("expr_no") or "") for task in prepared_tasks]
        if any(not expr_no for expr_no in prepared_ids) or len(set(prepared_ids)) != len(prepared_ids):
            raise ValueError("dynamic prepared tasks must have unique non-empty expr_no values")
        # Prefer business task names for event labels; fall back to expr_no.
        task_display_names = {
            str(task["expr_no"]): str(task.get("name") or task["expr_no"]) for task in prepared_tasks
        }
        # start produces no schedule row; exclude it from expected frozen counts consistently with the service.
        expected_operation_counts = {
            str(task["expr_no"]): count_dynamic_task_operations(task) for task in prepared_tasks
        }
        if any(count <= 0 for count in expected_operation_counts.values()):
            raise ValueError("dynamic prepared task has no schedulable operations")

        run_id = uuid.uuid4().hex
        server_session_id: str | None = None
        effective_schedule_time_limit: float | None = None
        frozen_rows: dict[tuple[str, int], dict[str, Any]] = {}
        dynamic_batches: list[dict[str, Any]] = []
        # For window 6 and advance 1, build cumulative makespan/cost metrics for task counts 6..N.
        dynamic_task_metrics: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        algo_cal_times: list[float | None] = []
        completed_ids: set[str] = set()
        active = prepared_tasks[: min(window_size, total_tasks)]
        next_index = len(active)
        ptr = 0
        batch_index = 0
        effective_peak_price = peak_price
        effective_valley_price = valley_price
        effective_makespan_cost_units = makespan_60min_cost_in_valley_price_units
        dynamic_output_name = build_schedule_method_output_name(output_name, schedule_method)
        output_path: Path | None = None
        # The final global snapshot combines frozen history with the complete final active window.
        final_window_rows: list[dict[str, Any]] | None = None

        try:
            health = self._health_handshake()
            server_session_id = str(health["server_session_id"])
            effective_schedule_time_limit = parse_optional_float(health.get("schedule_time_limit"))
            # Open the dynamic database and plotter only after handshake to avoid side effects from invalid service paths.
            placeholder_db = SchedulerDatabase(str(self.database_path), self.logger, recover=True)
            plotter = HorizonGanttPlotter(placeholder_db.power_info, self.output_dir)
            while active:
                active_ids_before = [str(task["expr_no"]) for task in active]
                # When all tasks have entered, solve the final window once and finish the experiment.
                is_final_window = next_index >= total_tasks
                requested_count = min(advance_count, len(active))
                request_payload = copy.deepcopy(prepared_payload)
                request_payload["taskList"] = copy.deepcopy(active)
                request_payload.update(
                    {
                        "dynamic_mode": True,
                        "dynamic_protocol_version": DYNAMIC_PROTOCOL_VERSION,
                        "dynamic_run_id": run_id,
                        "dynamic_batch_index": batch_index,
                        "dynamic_total_task_count": total_tasks,
                        "dynamic_window_size": window_size,
                        "dynamic_advance_count": advance_count,
                        "dynamic_simulation_origin": format_datetime(simulation_start_time),
                        "schedule_method": str(schedule_method).strip().lower(),

                        "objective_limit": objective_limit,
                        "makespan_limit": makespan_limit,
                        "carbon_limit": carbon_limit,
                    }
                )
                if peak_price is not None:
                    request_payload["peak_price"] = peak_price
                if valley_price is not None:
                    request_payload["valley_price"] = valley_price
                if tariff_info is not None:
                    request_payload["tariff_info"] = tariff_snapshot(tariff_info)
                if makespan_60min_cost_in_valley_price_units is not None:
                    request_payload["makespan_60min_cost_in_valley_price_units"] = (
                        makespan_60min_cost_in_valley_price_units
                    )

                response = self._submit_and_wait(
                    request_payload,
                    run_id,
                    batch_index,
                    server_session_id,
                )
                response_schedule_time_limit = parse_optional_float(
                    response.get("schedule_time_limit")
                )
                if response_schedule_time_limit is not None:
                    effective_schedule_time_limit = response_schedule_time_limit
                response_source = response.get("source")
                if str(schedule_method).strip().lower() in {"rl_or", "rl_release_or"}:
                    if response_source not in RL_CP_RESULT_SOURCES:
                        raise ValueError(
                            f"dynamic batch {batch_index} missing valid RL/CP result source: "
                            f"expected one of {sorted(RL_CP_RESULT_SOURCES)}, got {response_source!r}"
                        )  # Reject RL+CP dynamic batches with unknown provenance.
                # Record actual service tariff/time-value defaults consistently with the static runner.
                if tariff_info is not None:
                    actual_tariff = response.get("tariff_info")
                    requested_day = daily_tariff(tariff_info)
                    if (
                        not isinstance(actual_tariff, dict)
                        or daily_tariff(actual_tariff) != requested_day
                        or any(
                            float(actual_tariff[key]) != float(tariff_info[key])
                            for key in {"peak_price", "valley_price"} | {key for _, _, key in requested_day}
                        )
                    ):
                        raise ValueError("The dynamic service tariff prices or periods do not match the simulation configuration")
                if response.get("tariff_info") is not None:
                    plotter.power_info.update(copy.deepcopy(response["tariff_info"]))
                effective_peak_price = parse_optional_float(response.get("peak_price"))
                effective_valley_price = parse_optional_float(response.get("valley_price"))
                effective_makespan_cost_units = parse_optional_float(
                    response.get("makespan_60min_cost_in_valley_price_units")
                )
                batch_completed, ptr_after, window_rows = self._validate_and_freeze(
                    response=response,
                    active_ids=active_ids_before,
                    expected_operation_counts=expected_operation_counts,
                    completed_ids=completed_ids,
                    frozen_rows=frozen_rows,
                    ptr_before=ptr,
                    requested_count=requested_count,
                )
                completed_ids.update(batch_completed)
                completed_set = set(batch_completed)
                survivors = [task for task in active if str(task["expr_no"]) not in completed_set]
                add_count = min(len(batch_completed), total_tasks - next_index)
                added = prepared_tasks[next_index : next_index + add_count]
                added_ids = [str(task["expr_no"]) for task in added]
                active = survivors + added
                next_index += add_count
                if is_final_window:
                    final_window_rows = [copy.deepcopy(row) for row in window_rows]

                algo_time = parse_optional_float(response.get("algo_cal_time"))
                algo_cal_times.append(algo_time)
                solver_diagnostics = response.get("solver_diagnostics") or {}
                batch_record = {
                    "batchIndex": batch_index,
                    "activeTaskIdsBefore": active_ids_before,
                    "completedTaskIds": batch_completed,
                    "completedTaskNames": [task_display_names[expr_no] for expr_no in batch_completed],
                    "survivorTaskIds": [str(task["expr_no"]) for task in survivors],
                    "addedTaskIds": added_ids,
                    "addedTaskNames": [task_display_names[expr_no] for expr_no in added_ids],
                    "activeTaskIdsAfter": [str(task["expr_no"]) for task in active],
                    "requestedCompletionCount": requested_count,
                    "actualCompletionCount": len(batch_completed),
                    "pointerBefore": ptr,
                    "pointerAfter": ptr_after,
                    "windowMakespan": int(response["dynamic_window_makespan"]),
                    "algoCalTime": algo_time,
                    "scheduleTimeLimit": response_schedule_time_limit,  # Record this batch's actual service time limit.
                    "solverObjectiveValue": response.get("solver_objective_value"),
                    "solverObjectiveCarbonScaled": response.get("solver_carbon_objective_scaled"),
                    "source": response_source,  # Persist validated final provenance for accurate statistics.
                    "solverDiagnostics": solver_diagnostics,
                    # Promote RL-incumbent/CP-candidate comparisons to top-level batch fields for direct aggregation.
                    "objectiveComparison": solver_diagnostics.get("objective_comparison"),
                    "rlIncumbentObjective": solver_diagnostics.get("incumbent_objective"),
                    "cpCandidateObjective": solver_diagnostics.get("cp_candidate_objective"),
                    "objectiveDifferenceFromRlIncumbent": solver_diagnostics.get(
                        "objective_difference_from_incumbent"
                    ),
                    "objectiveImprovedBy": solver_diagnostics.get("objective_improvement"),
                    "objectiveWorseBy": solver_diagnostics.get("objective_worsening"),
                    "beta": response.get("beta"),
                }
                dynamic_batches.append(batch_record)
                timeline_events.append(
                    {
                        "time": ptr_after,
                        "completed": batch_record["completedTaskNames"],
                        "added": batch_record["addedTaskNames"],
                    }
                )
                ptr = ptr_after

                # Combine frozen history and this full window into the cumulative admitted-task snapshot.
                # window_rows contains all active tasks before this request; combined with frozen_rows,
                # x=6/y=1 produces cumulative task counts 6, 7, ..., total_tasks.
                progressive_rows_by_key = {
                    key: copy.deepcopy(row) for key, row in frozen_rows.items()
                }
                for row in window_rows:
                    key = (str(row["expr_no"]), int(row["fjspb_index"]))
                    progressive_rows_by_key[key] = copy.deepcopy(row)
                progressive_rows = canonical_schedule_rows(list(progressive_rows_by_key.values()))
                progressive_task_ids = list(
                    dict.fromkeys(str(row["expr_no"]) for row in progressive_rows)
                )
                progressive_task_count = len(progressive_task_ids)
                previous_task_count = (
                    int(dynamic_task_metrics[-1]["taskCount"])
                    if dynamic_task_metrics
                    else 0
                )
                if progressive_task_count > previous_task_count:
                    progressive_ws_table, progressive_bottle_table = self._rows_to_tables(progressive_rows)
                    progressive_makespan = max(int(row["end_time"]) for row in progressive_rows)
                    progressive_machine_power = placeholder_db.find_power_by_ws_code_from_db(
                        list(progressive_ws_table.keys())
                    )
                    progressive_snapshot = ExperimentSnapshot(
                        label=experiment_name,
                        source_file=str(used_files[0]),
                        source_files=[str(path) for path in used_files],
                        start_time=simulation_start_time.replace(microsecond=0),
                        makespan=progressive_makespan,
                        expr_nos=[expr_no for expr_no in prepared_ids if expr_no in progressive_task_ids],
                        expr_nos_by_source_file=expr_nos_by_source_file,
                        raw_ws_table=progressive_ws_table,
                        raw_bottle_table=progressive_bottle_table,
                        operation_count=len(progressive_rows),
                        tariff_info=tariff_snapshot(plotter.power_info),
                        peak_price=effective_peak_price,
                        valley_price=effective_valley_price,
                        makespan_60min_cost_in_valley_price_units=effective_makespan_cost_units,
                        beta=response.get("beta"),
                        machine_power_by_ws_code=progressive_machine_power,
                    )
                    (
                        progressive_carbon,
                        progressive_electricity,
                        progressive_carbon_peak,
                        progressive_carbon_valley,
                        progressive_electricity_peak,
                        progressive_electricity_valley,
                    ) = plotter.calculate_experiment_energy_metrics(progressive_snapshot)
                    dynamic_task_metrics.append(
                        {
                            "batchIndex": batch_index,
                            "taskCount": progressive_task_count,
                            "taskIds": progressive_task_ids,
                            "activeTaskCountBefore": len(active_ids_before),
                            "pointerAfter": ptr_after,
                            "makespan": progressive_makespan,
                            "recomputedCarbonObjectiveScaled": progressive_carbon,
                            "recomputedCarbonObjectiveScaledPeak": progressive_carbon_peak,
                            "recomputedCarbonObjectiveScaledValley": progressive_carbon_valley,
                            "totalElectricityCostYuan": progressive_electricity,
                            "totalElectricityCostYuanPeak": progressive_electricity_peak,
                            "totalElectricityCostYuanValley": progressive_electricity_valley,
                            "totalOperationCount": len(progressive_rows),
                            "scheduleTimeLimit": response_schedule_time_limit,  # Retain the batch time limit in task-count metrics.
                            "source": response_source,  # RL/CP source adopted in the current sliding window.
                            "solverStatus": solver_diagnostics.get("solver_status"),
                            "solverObjectiveValue": solver_diagnostics.get("solver_objective_value"),
                            "solverBestObjectiveBound": solver_diagnostics.get("solver_best_bound"),
                            "optimalityGap": solver_diagnostics.get("optimality_gap"),
                            "incumbentObjective": solver_diagnostics.get("incumbent_objective"),
                            "cpCandidateObjective": solver_diagnostics.get("cp_candidate_objective"),
                            "objectiveComparison": solver_diagnostics.get("objective_comparison"),
                            "objectiveDifferenceFromRlIncumbent": solver_diagnostics.get(
                                "objective_difference_from_incumbent"
                            ),
                            "objectiveImprovedBy": solver_diagnostics.get("objective_improvement"),
                            "objectiveWorseBy": solver_diagnostics.get("objective_worsening"),
                        }
                    )

                # Merge frozen history and surviving-task schedules each batch, then atomically replace the same SVG.
                preview_rows_by_key = {key: copy.deepcopy(row) for key, row in frozen_rows.items()}
                survivor_ids = {str(task["expr_no"]) for task in survivors}
                for row in window_rows:
                    expr_no = str(row["expr_no"])
                    if expr_no not in survivor_ids:
                        continue
                    key = (expr_no, int(row["fjspb_index"]))
                    preview_rows_by_key[key] = copy.deepcopy(row)
                preview_rows = canonical_schedule_rows(list(preview_rows_by_key.values()))
                preview_ws_table, preview_bottle_table = self._rows_to_tables(preview_rows)
                preview_makespan = max(int(row["end_time"]) for row in preview_rows)
                preview_machine_power = placeholder_db.find_power_by_ws_code_from_db(
                    list(preview_ws_table.keys())
                )
                preview_snapshot = ExperimentSnapshot(
                    label=experiment_name,
                    source_file=str(used_files[0]),
                    source_files=[str(path) for path in used_files],
                    start_time=simulation_start_time.replace(microsecond=0),
                    makespan=preview_makespan,
                    expr_nos=prepared_ids,
                    expr_nos_by_source_file=expr_nos_by_source_file,
                    raw_ws_table=preview_ws_table,
                    raw_bottle_table=preview_bottle_table,
                    algo_cal_time=sum(value for value in algo_cal_times if value is not None),
                    operation_count=len(preview_rows),
                    tariff_info=tariff_snapshot(plotter.power_info),
                    peak_price=effective_peak_price,
                    valley_price=effective_valley_price,
                    makespan_60min_cost_in_valley_price_units=effective_makespan_cost_units,
                    beta=response.get("beta"),
                    solver_objective_value=response.get("solver_objective_value"),
                    solver_carbon_objective_scaled=response.get("solver_objective_carbon_scaled"),
                    machine_power_by_ws_code=preview_machine_power,
                )
                output_path = plotter.render(
                    preview_snapshot,
                    output_name=dynamic_output_name,
                    color_by_task=color_by_task,
                    timeline_events=timeline_events,
                    atomic_write=True,
                )
                self.logger.info(
                    f"dynamic gantt refreshed: batch={batch_index}, "
                    f"pointer={ptr_after}, path={output_path}"
                )

                self._save_checkpoint(
                    {
                        "dynamicRunId": run_id,
                        "serverSessionId": server_session_id,
                        "scheduleStatus": "success" if is_final_window else "running",
                        "nextBatchIndex": batch_index + 1,
                        "pointer": ptr,
                        "nextTaskIndex": next_index,
                        "completedTaskIds": sorted(completed_ids),
                        "frozenRows": canonical_schedule_rows(list(frozen_rows.values())),
                        # Keep the final window separately in success checkpoints so unfinished tasks are not marked frozen.
                        "finalWindowRows": (
                            canonical_schedule_rows(final_window_rows)
                            if is_final_window and final_window_rows is not None
                            else []
                        ),
                        "dynamicBatches": dynamic_batches,
                        "dynamicTaskMetrics": dynamic_task_metrics,
                    },
                    file_repeat_index,
                )
                batch_index += 1
                if is_final_window:
                    break

            if final_window_rows is None or next_index != total_tasks:
                raise RuntimeError("dynamic run ended before the final active window was scheduled")

            # Merge frozen rows with the authoritative final-window plans for tail tasks.
            final_rows_by_key = {key: copy.deepcopy(row) for key, row in frozen_rows.items()}
            for row in final_window_rows:
                key = (str(row["expr_no"]), int(row["fjspb_index"]))
                previous = final_rows_by_key.get(key)
                if previous is not None and self._row_identity(previous) != self._row_identity(row):
                    raise RuntimeError(f"final window changed an already frozen primary key: {key}")
                final_rows_by_key[key] = copy.deepcopy(row)

            expected_total_rows = sum(expected_operation_counts.values())
            if len(final_rows_by_key) != expected_total_rows:
                raise RuntimeError(
                    f"dynamic final schedule has {len(final_rows_by_key)} rows, expected {expected_total_rows}"
                )
            final_task_ids = {str(row["expr_no"]) for row in final_rows_by_key.values()}
            if final_task_ids != set(prepared_ids):
                raise RuntimeError("dynamic final schedule does not contain every prepared task")
            makespan = max(int(row["end_time"]) for row in final_rows_by_key.values())
            if makespan < ptr:
                raise RuntimeError(f"dynamic final makespan precedes pointer: makespan={makespan}, pointer={ptr}")

            raw_ws_table, raw_bottle_table = self._rows_to_tables(list(final_rows_by_key.values()))

            machine_power = placeholder_db.find_power_by_ws_code_from_db(list(raw_ws_table.keys()))
            beta = next(
                (int(record["beta"]) for record in reversed(dynamic_batches) if record.get("beta") is not None),
                None,
            )
            snapshot = ExperimentSnapshot(
                label=experiment_name,
                source_file=str(used_files[0]),
                source_files=[str(path) for path in used_files],
                start_time=simulation_start_time.replace(microsecond=0),
                makespan=makespan,
                expr_nos=prepared_ids,
                expr_nos_by_source_file=expr_nos_by_source_file,
                raw_ws_table=raw_ws_table,
                raw_bottle_table=raw_bottle_table,
                algo_cal_time=sum(value for value in algo_cal_times if value is not None),
                operation_count=expected_total_rows,
                tariff_info=tariff_snapshot(plotter.power_info),
                peak_price=effective_peak_price,
                valley_price=effective_valley_price,
                makespan_60min_cost_in_valley_price_units=effective_makespan_cost_units,
                beta=beta,
                machine_power_by_ws_code=machine_power,
            )
            # Recompute total energy/carbon once from frozen history plus the final window.
            (
                snapshot.recomputed_carbon_objective_scaled,
                snapshot.electricity_cost_yuan,
                snapshot.recomputed_carbon_objective_scaled_peak,
                snapshot.recomputed_carbon_objective_scaled_valley,
                snapshot.electricity_cost_yuan_peak,
                snapshot.electricity_cost_yuan_valley,
            ) = plotter.calculate_experiment_energy_metrics(snapshot)
            output_path = plotter.render(
                snapshot,
                output_name=dynamic_output_name,
                color_by_task=color_by_task,
                timeline_events=timeline_events,
                atomic_write=True,
            )
            total_algo_time = sum(value for value in algo_cal_times if value is not None)
            return DynamicReplayResult(
                snapshot=snapshot,
                total_makespan=makespan,
                total_carbon_objective_scaled=snapshot.recomputed_carbon_objective_scaled,
                total_electricity_cost_yuan=snapshot.electricity_cost_yuan,
                total_algo_cal_time=total_algo_time,
                algo_cal_times=algo_cal_times,
                total_operation_count=expected_total_rows,
                schedule_method=str(schedule_method).strip().lower(),
                schedule_status="success",
                output_path=output_path,
                total_tasks=total_tasks,
                used_files=used_files,
                start_time=simulation_start_time.replace(microsecond=0),
                end_time=simulation_start_time.replace(microsecond=0) + timedelta(minutes=makespan),
                dynamic_batches=dynamic_batches,
                dynamic_task_metrics=dynamic_task_metrics,
                run_id=run_id,
                server_session_id=server_session_id,
                schedule_time_limit=effective_schedule_time_limit,
            )
        except Exception as exc:  # noqa: BLE001 - Persist diagnostic results and a checkpoint on mid-run failure.
            self._save_checkpoint(
                {
                    "dynamicRunId": run_id,
                    "serverSessionId": server_session_id,
                    "scheduleStatus": "error",
                    "errorMessage": str(exc),
                    "nextBatchIndex": batch_index,
                    "pointer": ptr,
                    "nextTaskIndex": next_index,
                    "completedTaskIds": sorted(completed_ids),
                    "frozenRows": canonical_schedule_rows(list(frozen_rows.values())),
                    "dynamicBatches": dynamic_batches,
                    "dynamicTaskMetrics": dynamic_task_metrics,
                },
                file_repeat_index,
            )
            return DynamicReplayResult(
                snapshot=None,
                total_makespan=None,
                total_carbon_objective_scaled=None,
                total_electricity_cost_yuan=None,
                total_algo_cal_time=sum(value for value in algo_cal_times if value is not None),
                algo_cal_times=algo_cal_times,
                total_operation_count=sum(expected_operation_counts.values()),
                schedule_method=str(schedule_method).strip().lower(),
                schedule_status="error",
                output_path=output_path,
                total_tasks=total_tasks,
                used_files=used_files,
                start_time=simulation_start_time.replace(microsecond=0),
                end_time=None,
                dynamic_batches=dynamic_batches,
                dynamic_task_metrics=dynamic_task_metrics,
                run_id=run_id,
                server_session_id=server_session_id,
                schedule_time_limit=effective_schedule_time_limit,
                error_message=str(exc),
            )
