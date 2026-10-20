from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

from obj.db_tools import SchedulerDatabase

from .gantt import DEFAULT_COLOR_BY_TASK, ExperimentSnapshot, HorizonGanttPlotter


TIMESTAMP_KEYS = {
    "stamp",
    "startTime",
    "endTime",
    "real_start_time",
    "real_end_time",
    "cur_time_stamp",
    "curTime",
}

TASK_NAME_REGEX = re.compile(r"^(.*?)(\d+)$")


REQUEST_TIMEOUT = 2100.0  # Exceed the usual 1800-second *_green_limit_* cap so replay waits for explicit service success/failure.
REPLAY_INTERVAL = 5.0


# -------------------------
# Basic utility modules.
# -------------------------

def load_json_file(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)



def dump_json_file(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_jsonl_line(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
        file.write("\n")


def load_replay_state(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        return {}

    payload = load_json_file(file_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a JSON object")

    return payload


def extract_file_repeat_index(state: dict[str, Any], experiment_files: list[Path]) -> dict[Path, int]:
    raw_mapping = state.get("fileRepeatIndex")
    if (
        raw_mapping is None
        and state
        and all(isinstance(value, (int, float, str)) for value in state.values())
    ):
        raw_mapping = state

    if not isinstance(raw_mapping, dict):
        return {}

    file_repeat_index: dict[Path, int] = {}
    for source_file in experiment_files:
        for key in (str(source_file.resolve()), source_file.name):
            if key not in raw_mapping:
                continue

            try:
                value = int(raw_mapping[key])
            except (TypeError, ValueError):
                break

            if value < 0:
                break

            file_repeat_index[source_file] = value
            break

    return file_repeat_index


def save_replay_state(file_path: Path, state: dict[str, Any], file_repeat_index: dict[Path, int]) -> None:
    payload = dict(state) if state else {}

    existing_mapping = payload.get("fileRepeatIndex")
    merged_mapping: dict[str, int] = {}
    if isinstance(existing_mapping, dict):
        for key, value in existing_mapping.items():
            try:
                merged_mapping[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    for source_file, repeat_index in file_repeat_index.items():
        merged_mapping[str(source_file.resolve())] = int(repeat_index)

    payload["fileRepeatIndex"] = merged_mapping
    dump_json_file(file_path, payload)


def round_optional_float(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        if abs(value) > 10_000_000_000:
            return datetime.fromtimestamp(value / 1000.0)
        return datetime.fromtimestamp(value)

    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    return None


def shift_timestamp_scalar(value: Any, delta_seconds: float) -> Any:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if abs(value) > 10_000_000_000:
            return int(value + delta_seconds * 1000)
        return int(value + delta_seconds)

    if isinstance(value, str):
        dt = parse_datetime(value)
        if dt is None:
            return value
        return format_datetime(dt + timedelta(seconds=delta_seconds))

    return value


def build_schedule_method_output_name(output_name: str, schedule_method: str | None) -> str:
    method_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(schedule_method or "unknown").strip())
    method_dir = method_dir.strip("_") or "unknown"
    return str(Path(method_dir) / Path(output_name).name)


def shift_payload_timestamps(node: Any, delta_seconds: float, key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: shift_payload_timestamps(v, delta_seconds, k) for k, v in node.items()}

    if isinstance(node, list):
        return [shift_payload_timestamps(item, delta_seconds, key) for item in node]

    if key in TIMESTAMP_KEYS:
        return shift_timestamp_scalar(node, delta_seconds)

    return node


def find_experiment_start_time(payload: dict[str, Any]) -> datetime:
    task_list = payload.get("taskList", [])
    candidates: list[datetime] = []
    for task in task_list:
        dt = parse_datetime(task.get("stamp"))
        if dt is not None:
            candidates.append(dt)

    if candidates:
        return min(candidates)

    raise ValueError("Cannot determine source start time from payload.")


def build_rebased_payload(
    payload: dict[str, Any], source_start: datetime, target_start: datetime
) -> dict[str, Any]:
    delta = (target_start - source_start).total_seconds()
    rebased = copy.deepcopy(payload)
    return shift_payload_timestamps(rebased, delta)


def scale_time_value(value: Any, scale: float) -> Any:
    if value is None or scale == 1.0:
        return value

    if isinstance(value, (int, float)):
        return max(0, int(round(float(value) * scale)))

    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return value
        return str(max(0, int(round(number * scale))))

    return value


def scale_payload_step_times(payload: dict[str, Any], scale: float) -> dict[str, Any]:
    if scale == 1.0:
        return payload

    updated = copy.deepcopy(payload)
    for task in updated.get("taskList", []):
        for step in task.get("steps", []):
            if "time" in step:
                step["time"] = scale_time_value(step.get("time"), scale)
    return updated


def shift_name_with_offset(name: str, offset: int, seq_index: int) -> str:
    match = TASK_NAME_REGEX.match(name)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(number) + offset}"
    return f"{name}_{offset + seq_index}"


def shift_expr_no(expr_no: Any, offset: int, seq_index: int) -> Any:
    if expr_no is None:
        return expr_no

    if isinstance(expr_no, (int, float)):
        return int(expr_no) + offset

    expr_text = str(expr_no)
    if expr_text.isdigit():
        return str(int(expr_text) + offset)

    match = TASK_NAME_REGEX.match(expr_text)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(number) + offset}"

    return f"{expr_text}_{offset + seq_index}"


def parse_non_zero_makespan(value: Any) -> int | None:
    if value is None:
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    if parsed <= 0:
        return None
    return parsed


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def count_task_operations(task: dict[str, Any]) -> int:
    steps = task.get("steps", [])
    if not isinstance(steps, list):
        return 0

    return sum(
        1
        for step in steps
        if isinstance(step, dict) and step.get("workstationType") not in {"lock", "unlock"}
    )


def resolve_experiment_files(source_dir: Path) -> list[Path]:
    # Sort naturally by numeric filename prefix for stable ordering before grouped sampling.
    def file_sort_key(file_path: Path) -> tuple[int, str]:
        match = re.match(r"^(\d+)", file_path.name)
        if match:
            return int(match.group(1)), file_path.name
        return 10**9, file_path.name

    files = sorted(source_dir.glob("*.json"), key=file_sort_key)

    result: list[Path] = []
    for file_path in files:
        resolved = file_path if file_path.is_absolute() else (source_dir / file_path).resolve()
        if resolved.is_file() and resolved.suffix == ".json":
            if resolved.name not in {"template.json", "base_steps.json"}:
                # ========== Keep only JSON filenames starting with digits ==========
                if resolved.name[0].isdigit():
                    result.append(resolved)
                # ========== End filtering ==========

    return result

# Format source filenames independently of the experiment label.
def format_experiment_file_list(experiment_files: list[Path]) -> str:
    return repr([file_path.stem for file_path in experiment_files])


@dataclass(slots=True)
class ReplayResult:
    snapshot: ExperimentSnapshot | None
    total_makespan: int | None
    total_carbon_objective_scaled: int | None
    total_electricity_cost_yuan: float | None
    total_algo_cal_time: float | None
    total_operation_count: int
    schedule_method: str | None
    schedule_status: str
    output_path: Path | None
    total_tasks: int
    used_files: list[Path]
    start_time: datetime
    end_time: datetime | None


# -------------------------
# Long-horizon replay execution module.
# -------------------------


class HorizonReplayRunner:
    def __init__(
        self,
        db: SchedulerDatabase,
        plotter: HorizonGanttPlotter,
        schedule_url: str,
        request_timeout: float = 300.0,
        replay_interval: float = 5.0,
    ) -> None:
        self.db = db
        self.plotter = plotter
        self.schedule_url = schedule_url
        self.request_timeout = request_timeout
        self.replay_interval = replay_interval

    def _post_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        deadline = time.time() + self.request_timeout
        retry_interval = 2.0

        while True:
            try:
                print(f"[post_experiment] POST {self.schedule_url}", flush=True)
                response = requests.post(self.schedule_url, json=payload, timeout=self.request_timeout)
            except requests.ConnectionError:
                print("[post_experiment] connection error, waiting to retry...", flush=True)
                if time.time() >= deadline:
                    return {}
                time.sleep(retry_interval)
                continue
            except requests.Timeout:
                print(
                    "[post_experiment] request timeout, returning empty response for makespan wait",
                    flush=True,
                )
                return {}

            if response.status_code >= 500:
                return {}

            response.raise_for_status()
            data = response.json()
            print(f"[post_experiment] body={data}", flush=True)
            return data

    def _response_matches_experiment(self, response: dict[str, Any], expected_task_names: list[str]) -> bool:
        response_task_names = response.get("all_tasks")
        if not isinstance(response_task_names, list):
            return False

        parsed_makespan = parse_non_zero_makespan(response.get("makespan"))
        if parsed_makespan is None:
            return False

        return sorted(map(str, response_task_names)) == sorted(map(str, expected_task_names))

    def _collect_experiment_snapshot(
        self,
        label: str,
        source_file: Path,
        source_files: list[str],
        payload: dict[str, Any],
        makespan: int,
        algo_cal_time: float | None,
        operation_count: int,
        peak_price: float | None,
        valley_price: float | None,
        makespan_60min_cost_in_valley_price_units: float | None,
        beta: int | None,
        expr_nos_by_source_file: dict[str, list[str]],
    ) -> ExperimentSnapshot:
        expr_nos = [task["expr_no"] for task in payload.get("taskList", [])]
        raw_ws_table, raw_bottle_table, _, _ = self.db.fetch_scheduling_table(expr_nos)
        start_time = find_experiment_start_time(payload)
        machine_power_by_ws_code = self.db.find_power_by_ws_code_from_db(list(raw_ws_table.keys()))

        return ExperimentSnapshot(
            label=label,
            source_file=str(source_file),
            source_files=source_files,
            start_time=start_time,
            makespan=int(makespan),
            expr_nos=expr_nos,
            expr_nos_by_source_file=expr_nos_by_source_file,
            raw_ws_table={key: list(value) for key, value in raw_ws_table.items()},
            raw_bottle_table={key: list(value) for key, value in raw_bottle_table.items()},
            algo_cal_time=algo_cal_time,
            operation_count=operation_count,
            peak_price=peak_price,
            valley_price=valley_price,
            makespan_60min_cost_in_valley_price_units=makespan_60min_cost_in_valley_price_units,
            beta=beta,
            machine_power_by_ws_code=machine_power_by_ws_code,
        )

    def _build_experiment_payload(
        self,
        experiment_files: list[Path],
        payload_cache: dict[Path, dict[str, Any]],
        target_start: datetime,
        file_repeat_index: dict[Path, int],
        time_scale: float,
        remaining_tasks: int | None,
    ) -> tuple[dict[str, Any], list[str], int, list[Path], dict[str, list[str]]]:
        merged: dict[str, Any] | None = None
        expected_task_names: list[str] = []
        used_files: list[Path] = []
        expr_nos_by_source_file: dict[str, list[str]] = {}
        tasks_used = 0
        remaining = remaining_tasks

        for source_file in experiment_files:
            payload = payload_cache[source_file]
            source_start_time = find_experiment_start_time(payload)
            task_list = payload.get("taskList", [])
            if not task_list:
                raise ValueError(f"No tasks found in {source_file}")

            repeat_index = file_repeat_index.get(source_file, 0)
            # Add each source file once; repeat index only produces IDs unique across experiments.
            offset = repeat_index * len(task_list)
            rebased_payload = build_rebased_payload(payload, source_start_time, target_start)
            rebased_payload = scale_payload_step_times(rebased_payload, time_scale)
            rebased_tasks: list[dict[str, Any]] = []

            for idx, task in enumerate(rebased_payload.get("taskList", []), start=1):
                updated = dict(task)
                name = updated.get("name")
                if name is not None:
                    updated["name"] = shift_name_with_offset(str(name), offset, idx)
                updated["expr_no"] = shift_expr_no(updated.get("expr_no"), offset, idx)
                rebased_tasks.append(updated)

            if remaining is not None and remaining < len(rebased_tasks):
                rebased_tasks = rebased_tasks[:remaining]

            if not rebased_tasks:
                break

            current_expr_nos = [
                str(item["expr_no"]) for item in rebased_tasks if item.get("expr_no") is not None
            ]
            expr_nos_by_source_file.setdefault(str(source_file), []).extend(current_expr_nos)

            for item in rebased_tasks:
                if item.get("name") is not None:
                    expected_task_names.append(str(item["name"]))

            rebased_payload["taskList"] = rebased_tasks
            tasks_used += len(rebased_tasks)
            if remaining is not None:
                remaining -= len(rebased_tasks)

            if merged is None:
                merged = copy.deepcopy(rebased_payload)
            else:
                merged_tasks = merged.get("taskList", [])
                merged_tasks.extend(rebased_tasks)
                merged["taskList"] = merged_tasks

            file_repeat_index[source_file] = repeat_index + 1
            used_files.append(source_file)

            if remaining is not None and remaining <= 0:
                break

        if merged is None:
            raise ValueError("No payloads to merge.")

        return merged, expected_task_names, tasks_used, used_files, expr_nos_by_source_file

    def run(
        self,
        experiment_files: list[Path],
        experiment_file_selection: list[Path],
        payload_cache: dict[Path, dict[str, Any]] | None = None,
        file_repeat_index: dict[Path, int] | None = None,
        task_limit: int | None = None,
        time_scale: float = 1.0,
        output_name: str = "gantt.svg",
        color_by_task: bool = DEFAULT_COLOR_BY_TASK,
        experiment_name: str | None = None,
        experiment_index: int | None = None,
        on_experiment_snapshot: Callable[[ReplayResult], None] | None = None,
        # ========== Peak/off-peak tariff parameters ==========
        peak_price: float | None = None,
        valley_price: float | None = None,
        # ========== End tariff parameters ==========
        makespan_60min_cost_in_valley_price_units: float | None = None,
        objective_limit: int | None = None,
        makespan_limit: int | None = None,
        carbon_limit: int | None = None,
        simulation_start_time: datetime | None = None,
    ) -> ReplayResult:
        if not experiment_files:
            raise ValueError("experiment_files is empty")
        if not experiment_file_selection:
            raise ValueError("experiment_file_selection is empty")
        if task_limit is not None and task_limit < 1:
            raise ValueError("task_limit must be >= 1")

        payload_cache = payload_cache or {path: load_json_file(path) for path in experiment_files}
        if file_repeat_index is None:
            file_repeat_index = {}

        # Allow one fixed simulation origin for all experiments; default to current time when unspecified.
        simulated_start_time = (
            simulation_start_time.replace(microsecond=0)
            if simulation_start_time is not None
            else datetime.now().replace(microsecond=0)
        )
        print(
            f"[experiment] align experiment start to {format_datetime(simulated_start_time)}",
            flush=True,
        )

        payload, expected_task_names, tasks_used, used_files, expr_nos_by_source_file = (
            self._build_experiment_payload(
                experiment_files=experiment_file_selection,
                payload_cache=payload_cache,
                target_start=simulated_start_time,
                file_repeat_index=file_repeat_index,
                time_scale=time_scale,
                remaining_tasks=task_limit,
            )
        )
        if tasks_used <= 0:
            raise RuntimeError("No tasks were added to the experiment payload.")

        operation_count = sum(count_task_operations(task) for task in payload.get("taskList", []))
        source_files = [str(path) for path in used_files]
        experiment_file_list = format_experiment_file_list(used_files)
        experiment_start_time = find_experiment_start_time(payload)

        if experiment_name:
            if experiment_index is not None:
                print(
                    f"\033[92m[experiment {experiment_index:03d}][{experiment_name}]\033[0m"
                    f"\033[93m{experiment_file_list}\033[0m send tasks={expected_task_names}",
                    flush=True,
                )
            else:
                print(
                    f"\033[92m[{experiment_name}]\033[0m"
                    f"\033[93m{experiment_file_list}\033[0m send tasks={expected_task_names}",
                    flush=True,
                )
        else:
            print(
                f"\033[93m{experiment_file_list}\033[0m send tasks={expected_task_names}",
                flush=True,
            )

        if peak_price is not None:
            payload["peak_price"] = peak_price
        if valley_price is not None:
            payload["valley_price"] = valley_price
        if makespan_60min_cost_in_valley_price_units is not None:
            # Allow replay_config to override the valley_price multiplier for 60 minutes of makespan.
            payload["makespan_60min_cost_in_valley_price_units"] = makespan_60min_cost_in_valley_price_units
        payload["objective_limit"] = objective_limit  # Use reference objectives from the configured source method for *_green_limit_*; send null if absent.
        payload["makespan_limit"] = makespan_limit  # Use historical reference makespan for *_green_limit_strict_*.
        payload["carbon_limit"] = carbon_limit  # Use historical reference carbon for *_green_limit_strict_*.

        makespan: int | None = None
        algo_cal_time: float | None = None
        response_peak_price: float | None = None
        response_valley_price: float | None = None
        response_makespan_60min_cost_units: float | None = None
        response_beta: int | None = None
        response_solver_objective_value: int | None = None
        response_solver_carbon_objective_scaled: int | None = None
        schedule_method: str | None = None
        schedule_status = "success"
        # Include experiment index/name in polling logs to distinguish concurrent or sequential experiments.
        if experiment_name and experiment_index is not None:
            retry_log_prefix = f"\033[93m[experiment {experiment_index:03d}][{experiment_name}]\033[0m"
        elif experiment_name:
            retry_log_prefix = f"\033[93m[{experiment_name}]\033[0m"
        else:
            retry_log_prefix = "\033[93m[experiment]\033[0m"
        # Poll empty/not-ready responses at replay_interval; only explicit schedule_status=error fails the batch.
        while makespan is None:
            result = self._post_experiment(payload)
            response_schedule_status = str(result.get("schedule_status") or "").strip().lower()
            if response_schedule_status == "error":
                schedule_status = "error"
                response_schedule_method = result.get("schedule_method")
                if response_schedule_method is not None:
                    schedule_method = str(response_schedule_method)
                break

            parsed_makespan = parse_non_zero_makespan(result.get("makespan"))
            if self._response_matches_experiment(result, expected_task_names) and parsed_makespan is not None:
                makespan = parsed_makespan
                algo_cal_time = parse_optional_float(result.get("algo_cal_time"))
                response_schedule_method = result.get("schedule_method")
                response_peak_price = parse_optional_float(result.get("peak_price"))
                response_valley_price = parse_optional_float(result.get("valley_price"))
                response_makespan_60min_cost_units = parse_optional_float(
                    result.get("makespan_60min_cost_in_valley_price_units")
                )
                response_beta = parse_optional_int(result.get("beta"))
                response_solver_objective_value = parse_optional_int(result.get("solver_objective_value"))
                response_solver_carbon_objective_scaled = parse_optional_int(
                    result.get("solver_carbon_objective_scaled")
                )
                if response_peak_price is None:
                    response_peak_price = parse_optional_float(self.db.get_power_info("peak_price", 0.6))
                if response_valley_price is None:
                    response_valley_price = parse_optional_float(self.db.get_power_info("valley_price", 0.3))
                if response_makespan_60min_cost_units is None:
                    response_makespan_60min_cost_units = parse_optional_float(
                        makespan_60min_cost_in_valley_price_units
                    )
                if response_makespan_60min_cost_units is None:
                    response_makespan_60min_cost_units = parse_optional_float(
                        self.db.get_power_info("makespan_60min_cost_in_valley_price_units", 2.0)
                    )
                non_green_methods = {"cp_or", "pure_rl"}  # Keep the nongreen method allowlist aligned with the scheduling entry point.
                if (
                    response_beta is None
                    and response_valley_price is not None
                    and str(response_schedule_method).strip().lower() not in non_green_methods
                ):
                    if response_makespan_60min_cost_units is not None:
                        response_beta = max(
                            0, int(round(response_makespan_60min_cost_units * response_valley_price * 1000 / 60))
                        )
                if response_schedule_method is not None:
                    schedule_method = str(response_schedule_method)
                break

            print(
                f"{retry_log_prefix} response not ready, retrying in "
                f"{self.replay_interval:.0f}s",
                flush=True,
            )
            time.sleep(self.replay_interval)

        if schedule_status == "error":
            replay_result = ReplayResult(
                snapshot=None,
                total_makespan=None,
                total_carbon_objective_scaled=None,
                total_electricity_cost_yuan=None,
                total_algo_cal_time=None,
                total_operation_count=operation_count,
                schedule_method=schedule_method,
                schedule_status="error",
                output_path=None,
                total_tasks=tasks_used,
                used_files=used_files,
                start_time=experiment_start_time,
                end_time=None,
            )
            if on_experiment_snapshot is not None:
                on_experiment_snapshot(replay_result)
            return replay_result

        snapshot_label = str(experiment_name or "experiment")
        snapshot = self._collect_experiment_snapshot(
            label=snapshot_label,
            source_file=used_files[0],
            source_files=source_files,
            payload=payload,
            makespan=makespan,
            algo_cal_time=algo_cal_time,
            operation_count=operation_count,
            peak_price=response_peak_price,
            valley_price=response_valley_price,
            makespan_60min_cost_in_valley_price_units=response_makespan_60min_cost_units,
            beta=response_beta,
            expr_nos_by_source_file=expr_nos_by_source_file,
        )
        snapshot.solver_objective_value = response_solver_objective_value
        snapshot.solver_carbon_objective_scaled = response_solver_carbon_objective_scaled
        (
            snapshot.recomputed_carbon_objective_scaled,
            snapshot.electricity_cost_yuan,
            snapshot.recomputed_carbon_objective_scaled_peak,
            snapshot.recomputed_carbon_objective_scaled_valley,
            snapshot.electricity_cost_yuan_peak,
            snapshot.electricity_cost_yuan_valley,
        ) = self.plotter.calculate_experiment_energy_metrics(snapshot)
        output_path = self.plotter.render(
            snapshot,
            output_name=build_schedule_method_output_name(output_name, schedule_method),
            color_by_task=color_by_task,
        )
        end_time = snapshot.start_time + timedelta(minutes=snapshot.makespan)
        replay_result = ReplayResult(
            snapshot=snapshot,
            total_makespan=snapshot.makespan,
            total_carbon_objective_scaled=snapshot.recomputed_carbon_objective_scaled,
            total_electricity_cost_yuan=snapshot.electricity_cost_yuan,
            total_algo_cal_time=algo_cal_time,
            total_operation_count=operation_count,
            schedule_method=schedule_method,
            schedule_status="success",
            output_path=output_path,
            total_tasks=tasks_used,
            used_files=used_files,
            start_time=snapshot.start_time,
            end_time=end_time,
        )
        if on_experiment_snapshot is not None:
            on_experiment_snapshot(replay_result)
        return replay_result
