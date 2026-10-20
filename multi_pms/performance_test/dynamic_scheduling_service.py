from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pprint import pprint  # Format startup arguments for logging.
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

SCRIPT_DIR = Path(__file__).resolve().parent
MULTI_PMS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MULTI_PMS_DIR.parent
for path in (str(MULTI_PMS_DIR), str(PROJECT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from include.dynamic_db import (
    DynamicReplayDatabase,
    canonical_schedule_rows,
    count_dynamic_task_operations,
    schedule_rows_digest,
)
from rl.utils.args import set_torch_params
from scheduler.scheduler import Scheduler
from scheduling import TaskScheduling
from algorithm.utils.tariff import TARIFF_FIELDS, daily_tariff, tariff_snapshot

DYNAMIC_ROUTE = "/scheduling_dynamic_test"
# Finish once all tasks have entered and the final window has been solved.
DYNAMIC_PROTOCOL_VERSION = 3


class DynamicProtocolError(RuntimeError):
    """Can be mapped directly to an HTTP error in the dedicated dynamic protocol."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class DynamicBatchEntry:
    request_hash: str
    request_payload: dict[str, Any]
    status: str
    response: dict[str, Any]


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _parse_origin(value: Any) -> datetime:
    if not isinstance(value, str):
        raise DynamicProtocolError("dynamic_simulation_origin must be a datetime string")
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise DynamicProtocolError("dynamic_simulation_origin must use YYYY-MM-DD HH:MM:SS") from exc


def _task_identity(task: dict[str, Any]) -> str:
    # Identity covers full task and step definitions; clients may not silently modify surviving tasks.
    return _canonical_json_hash(task)


class DynamicReplayController:
    """The sole control plane for dynamic replay; all solves run serially on one worker."""

    def __init__(
        self,
        db: DynamicReplayDatabase,
        scheduler: Scheduler,
        logger: Any,
        schedule_time_limit: float,
        cuda_device: int,
        database_path: Path,
    ) -> None:
        self.db = db
        self.scheduler = scheduler
        self.logger = logger
        self.schedule_time_limit = float(schedule_time_limit)
        self.cuda_device = int(cuda_device)
        self.database_path = database_path.resolve()
        self.default_tariff_info = tariff_snapshot(db.power_info)
        self.tariff_override_active = False
        self.server_session_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dynamic-replay")
        self.batch_cache: dict[tuple[str, int], DynamicBatchEntry] = {}

        self.active_run_id: str | None = None
        self.run_finished_successfully = False
        self.expected_batch_index = 0
        self.current_ptr = 0
        self.window_size = 0
        self.advance_count = 0
        self.total_task_count = 0
        self.simulation_origin: datetime | None = None
        self.schedule_method = ""
        self.seen_tasks: set[str] = set()
        self.completed_tasks: set[str] = set()
        self.previous_survivors: set[str] = set()
        self.previous_actual_completion_count = 0
        self.task_identities: dict[str, str] = {}

    def _release_active_run(self) -> None:
        """Release the current dynamic run state so one failure does not permanently lock later experiments with a 409 response."""
        self.active_run_id = None
        self.run_finished_successfully = False
        self.expected_batch_index = 0
        self.current_ptr = 0
        self.window_size = 0
        self.advance_count = 0
        self.total_task_count = 0
        self.simulation_origin = None
        self.schedule_method = ""
        self.seen_tasks = set()
        self.completed_tasks = set()
        self.previous_survivors = set()
        self.previous_actual_completion_count = 0
        self.task_identities = {}

    def health(self) -> dict[str, Any]:
        return {
            "controller": "dynamic_replay",
            "protocol_version": DYNAMIC_PROTOCOL_VERSION,
            "server_session_id": self.server_session_id,
            "database_path": str(self.database_path),
            "route": DYNAMIC_ROUTE,
            "schedule_time_limit": self.schedule_time_limit,  # Expose the actual per-batch service time limit.
        }

    def _base_response(self, run_id: str, batch_index: int, status: str) -> dict[str, Any]:
        return {
            "dynamic_status": status,
            "dynamic_run_id": run_id,
            "dynamic_batch_index": batch_index,
            "server_session_id": self.server_session_id,
        }

    @staticmethod
    def _require_int(payload: dict[str, Any], field: str) -> int:
        raw_value = payload.get(field)
        if isinstance(raw_value, bool):
            raise DynamicProtocolError(f"{field} must be an integer")
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise DynamicProtocolError(f"{field} must be an integer") from exc

    def _validate_common_payload(self, payload: dict[str, Any]) -> tuple[str, int, list[dict[str, Any]]]:
        if payload.get("dynamic_mode") is not True:
            raise DynamicProtocolError("dynamic_mode must be true")
        if self._require_int(payload, "dynamic_protocol_version") != DYNAMIC_PROTOCOL_VERSION:
            raise DynamicProtocolError("unsupported dynamic_protocol_version")
        for field in ("taskList", "robotList", "workstationList", "workstationDefinitionList"):
            if not isinstance(payload.get(field), list):
                raise DynamicProtocolError(f"{field} must be a list")

        run_id = str(payload.get("dynamic_run_id") or "").strip()
        if not run_id:
            raise DynamicProtocolError("dynamic_run_id must be non-empty")
        batch_index = self._require_int(payload, "dynamic_batch_index")
        if batch_index < 0:
            raise DynamicProtocolError("dynamic_batch_index must be >= 0")

        tasks = payload["taskList"]
        expr_nos = [str(task.get("expr_no") or "").strip() for task in tasks if isinstance(task, dict)]
        if len(expr_nos) != len(tasks) or any(not expr_no for expr_no in expr_nos):
            raise DynamicProtocolError("every task must be an object with a non-empty expr_no")
        if len(set(expr_nos)) != len(expr_nos):
            raise DynamicProtocolError("taskList contains duplicate expr_no values")
        return run_id, batch_index, tasks

    def _start_new_run(self, payload: dict[str, Any], run_id: str, tasks: list[dict[str, Any]]) -> None:
        window_size = self._require_int(payload, "dynamic_window_size")
        advance_count = self._require_int(payload, "dynamic_advance_count")
        total_task_count = self._require_int(payload, "dynamic_total_task_count")
        if window_size < 1 or advance_count < 1 or advance_count > window_size:
            raise DynamicProtocolError("dynamic window/advance values are invalid")
        if total_task_count < 1:
            raise DynamicProtocolError("dynamic_total_task_count must be >= 1")
        if len(tasks) != min(window_size, total_task_count):
            raise DynamicProtocolError("batch 0 task count does not match min(window, total)")
        schedule_method = str(payload.get("schedule_method") or "").strip().lower()
        if not schedule_method:
            raise DynamicProtocolError("schedule_method must be non-empty")

        self.active_run_id = run_id
        self.run_finished_successfully = False
        self.expected_batch_index = 0
        self.current_ptr = 0
        self.window_size = window_size
        self.advance_count = advance_count
        self.total_task_count = total_task_count
        self.simulation_origin = _parse_origin(payload.get("dynamic_simulation_origin"))
        self.schedule_method = schedule_method
        self.seen_tasks = set()
        self.completed_tasks = set()
        self.previous_survivors = set()
        self.previous_actual_completion_count = 0
        self.task_identities = {}

    def _validate_window(self, payload: dict[str, Any], tasks: list[dict[str, Any]], batch_index: int) -> None:
        if (
            self._require_int(payload, "dynamic_window_size") != self.window_size
            or self._require_int(payload, "dynamic_advance_count") != self.advance_count
            or self._require_int(payload, "dynamic_total_task_count") != self.total_task_count
            or _parse_origin(payload.get("dynamic_simulation_origin")) != self.simulation_origin
            or str(payload.get("schedule_method") or "").strip().lower() != self.schedule_method
        ):
            raise DynamicProtocolError("run-level dynamic parameters changed", status_code=409)

        current_ids = {str(task["expr_no"]) for task in tasks}
        if current_ids & self.completed_tasks:
            raise DynamicProtocolError("completed tasks cannot reappear", status_code=409)

        if batch_index > 0:
            if not self.previous_survivors.issubset(current_ids):
                raise DynamicProtocolError("next window omitted previous survivors", status_code=409)
            new_ids = current_ids - self.previous_survivors
            if new_ids & self.seen_tasks:
                raise DynamicProtocolError("next window contains a previously seen non-survivor", status_code=409)
            expected_new_count = min(
                self.previous_actual_completion_count,
                self.total_task_count - len(self.seen_tasks),
            )
            if len(new_ids) != expected_new_count:
                raise DynamicProtocolError(
                    f"next window must add {expected_new_count} new tasks, got {len(new_ids)}",
                    status_code=409,
                )
            expected_window_count = len(self.previous_survivors) + expected_new_count
            if len(tasks) != expected_window_count:
                raise DynamicProtocolError("next window task count is inconsistent", status_code=409)

        for task in tasks:
            expr_no = str(task["expr_no"])
            identity = _task_identity(task)
            previous_identity = self.task_identities.get(expr_no)
            if previous_identity is not None and previous_identity != identity:
                raise DynamicProtocolError(f"task identity changed for {expr_no}", status_code=409)

    def submit(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        run_id, batch_index, tasks = self._validate_common_payload(payload)
        if "tariff_info" in payload:
            info = payload["tariff_info"]
            try:
                if not isinstance(info, dict) or not {"peak_price", "valley_price"} <= info.keys():
                    raise ValueError("tariff_info must explicitly define peak_price and valley_price")
                day = daily_tariff(info)
                if any(key not in info for _, _, key in day):
                    raise ValueError("tariff_info must explicitly define all prices used by its periods")
                for field in ("peak_price", "valley_price"):
                    if field in payload and float(payload[field]) != float(info[field]):
                        raise ValueError(f"{field} does not match tariff_info")
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise DynamicProtocolError(f"invalid tariff_info: {exc}") from exc
        request_hash = _canonical_json_hash(payload)
        token = (run_id, batch_index)

        with self.lock:
            existing = self.batch_cache.get(token)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise DynamicProtocolError(
                        "same run/batch token was submitted with a different payload",
                        status_code=409,
                    )
                return copy.deepcopy(existing.response), 202 if existing.status == "pending" else 200

            if self.active_run_id != run_id:
                if batch_index != 0:
                    raise DynamicProtocolError("a new run must start from batch 0", status_code=409)
                if self.active_run_id is not None and not self.run_finished_successfully:
                    raise DynamicProtocolError("another dynamic run is still active", status_code=409)
                self._start_new_run(payload, run_id, tasks)
            elif self.run_finished_successfully:
                raise DynamicProtocolError("the dynamic run is already complete", status_code=409)

            if batch_index != self.expected_batch_index:
                raise DynamicProtocolError(
                    f"expected batch {self.expected_batch_index}, got {batch_index}",
                    status_code=409,
                )
            if any(
                entry.status == "pending"
                for (cached_run_id, _), entry in self.batch_cache.items()
                if cached_run_id == run_id
            ):
                raise DynamicProtocolError("the previous batch is still pending", status_code=409)

            self._validate_window(payload, tasks, batch_index)
            pending = self._base_response(run_id, batch_index, "pending")
            self.batch_cache[token] = DynamicBatchEntry(
                request_hash=request_hash,
                request_payload=copy.deepcopy(payload),
                status="pending",
                response=pending,
            )
            self.executor.submit(self._run_batch, token, request_hash)
            return copy.deepcopy(pending), 202

    def status(self, run_id: str, batch_index: int) -> tuple[dict[str, Any], int]:
        with self.lock:
            entry = self.batch_cache.get((run_id, batch_index))
            if entry is None:
                raise DynamicProtocolError("unknown dynamic run/batch", status_code=404)
            return copy.deepcopy(entry.response), 202 if entry.status == "pending" else 200

    def _publish_error(self, token: tuple[str, int], message: str) -> None:
        with self.lock:
            entry = self.batch_cache.get(token)
            if entry is None or entry.status != "pending":
                return
            response = self._base_response(token[0], token[1], "error")
            response["message"] = message
            entry.status = "error"
            entry.response = response
            if self.active_run_id == token[0]:
                # Release the active slot on failure to avoid persistent 409 responses for later runs.
                self._release_active_run()

    def _run_batch(self, token: tuple[str, int], request_hash: str) -> None:
        run_id, batch_index = token
        try:
            with self.lock:
                entry = self.batch_cache.get(token)
                if entry is None or entry.status != "pending" or entry.request_hash != request_hash:
                    return
                payload = copy.deepcopy(entry.request_payload)
                ptr_before = self.current_ptr
                if self.active_run_id != run_id or batch_index != self.expected_batch_index:
                    raise RuntimeError("pending batch is no longer the active generation")

            if batch_index == 0:
                # A new run resets only its dedicated dynamic database, preserving scheduling_param_tb startup settings.
                self.db.reset_db()
                self.db.is_db_init = True
                if self.simulation_origin is None:
                    raise RuntimeError("dynamic simulation origin is missing")
                self.db.set_dynamic_simulation_origin(self.simulation_origin)

                if "tariff_info" in payload or self.tariff_override_active:
                    for field in TARIFF_FIELDS:
                        self.db.power_info.pop(field, None)
                    self.db.power_info.update(copy.deepcopy(self.default_tariff_info))
                    self.tariff_override_active = False

            if "tariff_info" in payload:
                for field in TARIFF_FIELDS:
                    self.db.power_info.pop(field, None)
                self.db.power_info.update(tariff_snapshot(payload["tariff_info"]))
                self.tariff_override_active = True

            for field in ("peak_price", "valley_price", "makespan_60min_cost_in_valley_price_units"):
                if field in payload:
                    self.db.power_info[field] = payload[field]

            self.db.update_db(payload)
            expr_nos = [str(task["expr_no"]) for task in payload["taskList"]]
            task_group = f"dynamic_replay:{run_id}"
            # Restore existing PyTorch thread/device settings before each single-worker solve.
            set_torch_params(device_id=self.cuda_device)
            started_at = time.perf_counter()
            if self.db.has_reschedulable_operations(ptr_before, expr_nos):
                # Only dynamic-compare *_green_limit_* requests carry RL-derived limits; ordinary requests use None.
                objective_limit = payload.get("objective_limit")
                makespan_limit = payload.get("makespan_limit")
                carbon_limit = payload.get("carbon_limit")
                window_makespan = self.scheduler.do_schedule(
                    task_group=task_group,
                    cur_ptr=ptr_before,
                    expr_no_list=expr_nos,
                    time_limit=self.schedule_time_limit,
                    sim=True,
                    method=self.schedule_method,
                    objective_limit=objective_limit,
                    makespan_limit=makespan_limit,
                    carbon_limit=carbon_limit,
                )
                if window_makespan is None:
                    # Forward Scheduler failure stages and raw CP-SAT statuses into dynamic result records.
                    failed_metrics = self.scheduler.get_last_schedule_metrics(task_group)
                    detailed_error = failed_metrics.get("error_message")
                    raise RuntimeError(
                        detailed_error or "scheduler returned no solution without diagnostics"
                    )
                raw_ws_table, raw_bottle_table, _, _ = self.db.fetch_scheduling_table(expr_nos)
                schedule_metrics = self.scheduler.get_last_schedule_metrics(task_group)
            else:
                # If surviving tasks only have their last started step, do not pass empty jobs to the solver.
                raw_ws_table, raw_bottle_table, _, _ = self.db.fetch_scheduling_table(expr_nos)
                scheduled_rows = [row for rows in raw_bottle_table.values() for row in rows]
                if not scheduled_rows:
                    raise RuntimeError("dynamic window has no existing schedule rows to advance")
                window_makespan = max(int(row["end_time"]) for row in scheduled_rows)
                schedule_metrics = {}
            algo_cal_time = time.perf_counter() - started_at
            completion_rows: dict[str, list[dict[str, Any]]] = {}
            completion_times: list[tuple[int, str]] = []
            tasks_by_id = {str(task["expr_no"]): task for task in payload["taskList"]}
            for expr_no in expr_nos:
                rows = [copy.deepcopy(row) for row in raw_bottle_table.get(expr_no, [])]
                expected_operations = count_dynamic_task_operations(tasks_by_id[expr_no])
                if len(rows) != expected_operations:
                    raise RuntimeError(f"task {expr_no} has {len(rows)} scheduled rows, expected {expected_operations}")
                for row in rows:
                    start_time = row.get("start_time")
                    end_time = row.get("end_time")
                    if (
                        start_time is None
                        or end_time is None
                        or int(start_time) < 0
                        or int(start_time) >= sys.maxsize
                        or int(end_time) < int(start_time)
                    ):
                        raise RuntimeError(f"task {expr_no} contains a non-finite schedule row")
                completion_rows[expr_no] = rows
                completion_times.append((max(int(row["end_time"]) for row in rows), expr_no))

            completion_times.sort(key=lambda item: (item[0], item[1]))
            requested_count = min(self.advance_count, len(completion_times))
            cutoff = completion_times[requested_count - 1][0]
            completed_items = [item for item in completion_times if item[0] <= cutoff]
            completed_ids = {expr_no for _, expr_no in completed_items}
            # Return the full window for client redraws; keep completed rows separate for authoritative freezing.
            window_rows = canonical_schedule_rows(
                [row for expr_no in expr_nos for row in completion_rows[expr_no]]
            )
            completed_rows = canonical_schedule_rows(
                [row for expr_no in completed_ids for row in completion_rows[expr_no]]
            )
            ready = {
                **self._base_response(run_id, batch_index, "ready"),
                "dynamic_ptr_before": ptr_before,
                "dynamic_ptr_after": cutoff,
                "dynamic_requested_completion_count": requested_count,
                "dynamic_actual_completion_count": len(completed_items),
                "dynamic_completed_tasks": [
                    {
                        "expr_no": expr_no,
                        "name": tasks_by_id[expr_no].get("name"),
                        "completion_time": completion_time,
                    }
                    for completion_time, expr_no in completed_items
                ],
                "dynamic_window_schedule_rows": window_rows,
                "dynamic_window_schedule_digest": schedule_rows_digest(window_rows),
                "dynamic_completed_schedule_rows": completed_rows,
                "dynamic_completed_schedule_digest": schedule_rows_digest(completed_rows),
                "dynamic_window_makespan": int(window_makespan),
                "algo_cal_time": float(algo_cal_time),
                "schedule_method": self.schedule_method,
                "schedule_time_limit": self.schedule_time_limit,  # Return the actual time limit with batch results.
                "tariff_info": tariff_snapshot(self.db.power_info),
                "peak_price": self.db.get_power_info("peak_price", 0.6),
                "valley_price": self.db.get_power_info("valley_price", 0.3),
                "makespan_60min_cost_in_valley_price_units": self.db.get_power_info(
                    "makespan_60min_cost_in_valley_price_units", 2.0
                ),
                "beta": schedule_metrics.get("beta"),
                "solver_objective_value": schedule_metrics.get("solver_objective_value"),
                "solver_carbon_objective_scaled": schedule_metrics.get("solver_carbon_objective_scaled"),
                "source": schedule_metrics.get("source"),  # Return the RL/CP source adopted for this batch.
                # Forward LNS objective comparisons so the client records all four outcomes accurately.
                "solver_diagnostics": schedule_metrics.get("solver_diagnostics") or {},
            }

            with self.lock:
                current_entry = self.batch_cache.get(token)
                if current_entry is None or current_entry.status != "pending":
                    return
                if self.current_ptr != ptr_before or batch_index != self.expected_batch_index:
                    raise RuntimeError("dynamic pointer changed while a batch was solving")
                current_ids = set(expr_nos)
                for task in payload["taskList"]:
                    expr_no = str(task["expr_no"])
                    self.task_identities.setdefault(expr_no, _task_identity(task))
                self.seen_tasks.update(current_ids)
                self.completed_tasks.update(completed_ids)
                self.previous_survivors = current_ids - completed_ids
                self.previous_actual_completion_count = len(completed_ids)
                self.current_ptr = cutoff
                self.expected_batch_index += 1
                # Release the run after solving its final active window; tail tasks retain the final schedule.
                self.run_finished_successfully = len(self.seen_tasks) == self.total_task_count
                current_entry.status = "ready"
                current_entry.response = ready
        except Exception as exc:  # noqa: BLE001 - The worker must persist every solve failure as an immutable error.
            self.logger.info_red_bright(f"dynamic batch {run_id}/{batch_index} failed: {exc}")
            self._publish_error(token, str(exc))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class DynamicTaskScheduling(TaskScheduling):
    """Reuse model loading only; explicitly override production route registration and tracker startup."""

    def setup_routes(self) -> None:
        # Do not register production routes when called from super().__init__.
        return

    def load_model(self):
        # Resolve configured model paths relative to multi_pms/, independent of the startup directory.
        model_path = Path(self.model_name).expanduser()
        if not model_path.is_absolute():
            self.model_name = str((MULTI_PMS_DIR / model_path).resolve())
        return super().load_model()

    def __init__(self, startup_args: argparse.Namespace, argv: list[str] | None = None) -> None:
        # Forward dynamic service arguments to the parent.
        original_argv = sys.argv
        parent_argv = list(original_argv[1:] if argv is None else argv)
        parent_argv.extend(
            [
                "--debug",
                "--debug_flask_port",
                str(startup_args.debug_flask_port),
                "--database_file",
                str(startup_args.database_file),
                "--schedule_method",
                str(startup_args.schedule_method),
                "--schedule_time_limit",
                str(startup_args.schedule_time_limit),
            ]
        )
        try:
            sys.argv = [original_argv[0], *parent_argv]
            super().__init__()
        finally:
            sys.argv = original_argv
        self.db = DynamicReplayDatabase(self.database_path, self.logger, recover=True)
        self.scheduler = Scheduler(self.logger, self.db, self.model, self.memories, self.env_params)
        self.controller = DynamicReplayController(
            db=self.db,
            scheduler=self.scheduler,
            logger=self.logger,
            schedule_time_limit=self.schedule_time_limit,
            cuda_device=self.rl_args.cuda,
            database_path=Path(self.database_path),
        )
        self.app = Flask("dynamic_replay_service")
        self._setup_dynamic_routes()

    def _setup_dynamic_routes(self) -> None:
        @self.app.get(f"{DYNAMIC_ROUTE}/health")
        def dynamic_health():
            return jsonify(self.controller.health()), 200

        @self.app.post(DYNAMIC_ROUTE)
        def dynamic_submit():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"dynamic_status": "error", "message": "JSON object required"}), 400
            try:
                response, status_code = self.controller.submit(payload)
                return jsonify(response), status_code
            except DynamicProtocolError as exc:
                return jsonify({"dynamic_status": "error", "message": str(exc)}), exc.status_code

        @self.app.get(f"{DYNAMIC_ROUTE}/status/<run_id>/<int:batch_index>")
        def dynamic_status(run_id: str, batch_index: int):
            try:
                response, status_code = self.controller.status(run_id, batch_index)
                return jsonify(response), status_code
            except DynamicProtocolError as exc:
                return jsonify({"dynamic_status": "error", "message": str(exc)}), exc.status_code

    def run(self) -> None:
        # Bind only localhost and do not start TaskScheduling.tracker_loop().
        self.logger.info_green_bright(
            f"dynamic replay service: http://127.0.0.1:{self.flask_port}{DYNAMIC_ROUTE}"
        )
        try:
            self.app.run(
                host="127.0.0.1",
                port=self.flask_port,
                debug=False,
                use_reloader=False,
                threaded=True,
            )
        finally:
            self.controller.shutdown()


def build_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated dynamic replay scheduling service")
    parser.add_argument("--debug_flask_port", "--debug_port", dest="debug_flask_port", type=int, default=5082)
    parser.add_argument("--database_file", default="dynamic_replay_paper")  # Use the publication dynamic-replay database name by default.
    parser.add_argument("--schedule_method", default="rl_or")
    parser.add_argument("--schedule_time_limit", type=float, default=10.0)
    return parser


def validate_startup_gate(argv: list[str] | None = None) -> argparse.Namespace:
    args, _ = build_gate_parser().parse_known_args(argv)
    database_file = str(args.database_file).strip()
    if Path(database_file).name != database_file or not database_file.startswith("dynamic_replay_"):
        raise SystemExit("--database_file must be a basename starting with dynamic_replay_")
    if database_file == "task" or database_file.endswith(".sqlite"):
        raise SystemExit("--database_file must be an isolated basename without .sqlite")
    if args.debug_flask_port in {5060, 5081}:
        raise SystemExit("dynamic service cannot use port 5060 or 5081")
    if not (1 <= args.debug_flask_port <= 65535):
        raise SystemExit("dynamic service port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> int:
    # Check the guard before loading models or creating database objects.
    startup_args = validate_startup_gate(argv)
    print("dynamic service args:", flush=True)
    pprint(vars(startup_args), sort_dicts=False)
    service = DynamicTaskScheduling(startup_args, argv)
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
