from __future__ import annotations

import argparse
import json
import math
from pprint import pprint  # Used to format startup arguments for printing.
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MULTI_PMS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MULTI_PMS_DIR.parent
REAL_EXPR_DIR = MULTI_PMS_DIR / "dms_link" / "real_expr"

OUTPUT_DIR = MULTI_PMS_DIR / "expr_data" / "data"
# Store dynamic summary/detail records in the dedicated jsonl directory.
JSONL_OUTPUT_DIR = OUTPUT_DIR / "jsonl"
DEFAULT_CONFIG_NAME = "replay_experiment_config.json"
DEFAULT_CASE_CONFIG_NAME = "replay_test_cases.json"  # Default dynamic-replay scenario configuration.
# All ordinary dynamic methods share summary and detail files.
DYNAMIC_RESULTS_NAME = "metrics_dynamic.jsonl"
DYNAMIC_DETAILS_RESULTS_NAME = "metrics_dynamic_details.jsonl"
DYNAMIC_REPLAY_STATE_NAME = "replay_state_dynamic.json"
# Unique-task and unique-compare share summary/details but use separate checkpoints per mode.
UNIQUE_TASK_RESULTS_NAME = "metrics_dynamic_unique_task.jsonl"
UNIQUE_TASK_DETAILS_RESULTS_NAME = "metrics_dynamic_unique_task_details.jsonl"
UNIQUE_TASK_REPLAY_STATE_NAME = "replay_state_dynamic_unique_task.json"
# Unique-compare uses all scenarios, task counts 6-12, and its own checkpoint.
UNIQUE_COMPARE_REPLAY_STATE_NAME = "replay_state_dynamic_unique_compare.json"
UNIQUE_COMPARE_SCHEDULE_METHODS = (
    "cp_green_or",
    "rl_release_or",
    "cp_green_limit_or",
)
UNIQUE_COMPARE_LIMIT_METHODS = frozenset(
    {
        "cp_green_limit_or",
        "cp_green_limit_strict_or",
    }
)
UNIQUE_COMPARE_TASK_COUNTS = tuple(range(6, 13))
# Use one simulation origin across experiments so real startup times cannot shift tariff phases.
FIXED_SIMULATION_START_TIME = "2026-06-22 12:00:00"
# Select historical objective_limit references for *_green_limit_* from rl_or or rl_release_or.
OBJECTIVE_LIMIT_SOURCE_METHOD = "rl_release_or"

for path in (str(MULTI_PMS_DIR), str(PROJECT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.async_log import AsyncLog

from include.runner import (
    REPLAY_INTERVAL,
    REQUEST_TIMEOUT,
    append_jsonl_line,
    extract_file_repeat_index,
    format_datetime,
    load_json_file,
    parse_datetime,
    parse_optional_float,
    load_replay_state,
    resolve_experiment_files,
    round_optional_float,
    save_replay_state,
)

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a config-driven dynamic replay parameter sweep")
    parser.add_argument(
        "--config",
        default=str(OUTPUT_DIR / DEFAULT_CONFIG_NAME),
        help="Path to the replay experiment config JSON (default: expr_data/data/replay_experiment_config.json)",
    )
    parser.add_argument("--results-file", help="Summary JSONL filename; uses the current mode default if omitted")
    parser.add_argument("--details-results-file", help="Details JSONL filename; defaults to the --results-file stem plus _details when specified")
    parser.add_argument(
        "--source-dir",
        default=str(REAL_EXPR_DIR),
        help="Directory that contains experiment JSON files",
    )
    parser.add_argument(
        "--case-config",
        help="Path to the dynamic scenario config; overrides caseConfig in the main config",
    )
    parser.add_argument(
        "--tasks-per-scene",
        type=int,
        help="Run only scenarios with the specified tasksPerScene; runs all configured scenarios if omitted",
    )
    parser.add_argument(
        "--dynamic-window-size",
        type=int,
        help="Override the configured dynamic active window size (fallback: 6)",
    )
    parser.add_argument(
        "--dynamic-advance-count",
        type=int,
        help="Requested task completions per dynamic batch (default: 1)",
    )
    parser.add_argument(
        "--dynamic-schedule-port",
        type=int,
        default=5082,  # The dynamic service defaults to isolated port 5082.
        help="Port of the isolated dynamic scheduling service (default: 5082)",
    )
    parser.add_argument(
        "--dynamic-db-path",
        help="Explicit SQLite path used by the isolated dynamic scheduling service",
    )
    unique_task_group = parser.add_mutually_exclusive_group()
    unique_task_group.add_argument(
        "--unique-task-repeat",
        "--unique_task_repeat",
        dest="unique_task_repeat",
        action="store_true",
        help="Replay config.uniqueTask.scenarioName repeatCount times using repeatWindowAdvancePairs",
    )
    unique_task_group.add_argument(
        "--unique-task-combine",
        "--unique_task_combine",
        dest="unique_task_combine",
        action="store_true",
        help="Run only config.uniqueTask.scenarioName and evaluate the configured window combinations",
    )
    unique_task_group.add_argument(
        "--unique-compare",
        "--unique_compare",
        dest="unique_compare",
        action="store_true",
        help="Compare the first 6--12 tasks of every scenario using only the three specified algorithms",
    )
    return parser


def resolve_fixed_simulation_start_time() -> datetime:
    fixed_start_time = parse_datetime(FIXED_SIMULATION_START_TIME)
    if fixed_start_time is None:
        raise SystemExit(
            "FIXED_SIMULATION_START_TIME must use format 'YYYY-MM-DD HH:MM:SS', "
            f"got: {FIXED_SIMULATION_START_TIME}"
        )
    return fixed_start_time.replace(microsecond=0)


def resolve_dynamic_options(
    config_path: Path,
    args: argparse.Namespace,
    unique_task_mode: str | None = None,
    unique_compare_mode: bool = False,
) -> dict[str, Any]:
    # Isolation priority is CLI > main config > defaults; do not reuse production ports or dbPath.
    config = load_json_file(config_path)
    configured_mode = str(config.get("mode") or "").strip().lower()
    if configured_mode and configured_mode != "dynamic":
        raise SystemExit("Only dynamic replay is supported; config mode must be 'dynamic'")

    schedule_port = int(
        args.dynamic_schedule_port
        if args.dynamic_schedule_port is not None
        else config.get("dynamicSchedulePort", 5082)
    )

    dynamic_db_path: Path | None = None
    unique_task_config = config.get("uniqueTask")
    if not isinstance(unique_task_config, dict):
        unique_task_config = {}
    unique_compare_config = config.get("uniqueCompare")
    if not isinstance(unique_compare_config, dict):
        unique_compare_config = {}
    raw_dynamic_db_path = (
        args.dynamic_db_path
        or (
            unique_task_config.get("dynamicDbPath")
            if unique_task_mode is not None
            else None
        )
        or (
            unique_compare_config.get("dynamicDbPath")
            if unique_compare_mode
            else None
        )
        or config.get("dynamicDbPath")
    )
    if schedule_port in {5060, 5081} or not (1 <= schedule_port <= 65535):
        raise SystemExit("dynamicSchedulePort must be valid and cannot be 5060 or 5081")
    if not raw_dynamic_db_path:
        raise SystemExit(
            "dynamic replay requires dynamicDbPath/--dynamic-db-path; dbPath is not a fallback"
        )
    dynamic_db_path = Path(str(raw_dynamic_db_path)).expanduser()
    if not dynamic_db_path.is_absolute():
        dynamic_db_path = PROJECT_DIR / dynamic_db_path
    dynamic_db_path = dynamic_db_path.resolve()
    if not dynamic_db_path.stem.startswith("dynamic_replay_"):
        raise SystemExit("dynamicDbPath basename must start with dynamic_replay_")

    return {
        "schedulePort": schedule_port,
        "databasePath": dynamic_db_path,
    }


def sanitize_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "experiment"


# Append all ordinary dynamic methods to the same summary file.
def build_dynamic_results_path(output_dir: Path) -> Path:
    return output_dir / DYNAMIC_RESULTS_NAME


def format_numeric_label(value: float) -> str:
    # Normalize numeric parameters into short stable labels so 3.0 and 3 do not produce different filenames.
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def load_experiment_definitions(
    config_path: Path,
    mode_override: str | None = None,
    case_config_override: str | None = None,
    dynamic_window_size_override: int | None = None,
    dynamic_advance_count_override: int | None = None,
    tasks_per_scene_override: int | None = None,
    unique_task_mode: str | None = None,
    unique_compare_mode: bool = False,
) -> tuple[list[dict[str, Any]], str, Path]:
    payload = load_json_file(config_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{config_path} must contain a JSON object")

    # Only explicit uniqueTask replaces the ordinary Cartesian product; regular replay uses top-level settings.
    raw_unique_task_config = payload.get("uniqueTask")
    if raw_unique_task_config is None:
        unique_task_config: dict[str, Any] = {}
    elif isinstance(raw_unique_task_config, dict):
        unique_task_config = raw_unique_task_config
    else:
        raise SystemExit(f"{config_path} field 'uniqueTask' must be an object")
    if unique_task_mode is not None and unique_compare_mode:
        raise SystemExit("unique task modes and --unique-compare are mutually exclusive")
    if unique_compare_mode:
        raw_unique_compare_config = payload.get("uniqueCompare")
        if not isinstance(raw_unique_compare_config, dict):
            raise SystemExit(f"{config_path} field 'uniqueCompare' must be an object")
        unique_compare_config = raw_unique_compare_config
    else:
        unique_compare_config = {}
    parameter_payload = (
        unique_compare_config
        if unique_compare_mode
        else unique_task_config
        if unique_task_mode is not None
        else payload
    )

    def get_config_value(field_name: str, source_payload: dict[str, Any] | None = None) -> Any:
        source = parameter_payload if source_payload is None else source_payload
        return source.get(field_name)

    def normalize_value_list(
        field_name: str,
        source_payload: dict[str, Any] | None = None,
    ) -> list[int | float]:
        raw_value = get_config_value(field_name, source_payload)
        if raw_value is None and field_name in {"peakPrices", "valleyPrices"}:
            source = parameter_payload if source_payload is None else source_payload
            tariff = source.get("tariffInfo")
            if isinstance(tariff, dict):
                raw_value = tariff.get("peak_price" if field_name == "peakPrices" else "valley_price")
        if raw_value is None:
            raise SystemExit(f"{config_path} is missing required field '{field_name}'")
        if field_name in {"peakPrices", "valleyPrices"}:
            raw_prices = raw_value if isinstance(raw_value, list) else [raw_value]
            if not raw_prices:
                raise SystemExit(f"{config_path} field '{field_name}' cannot be empty")
            try:
                prices = [float(value) for value in raw_prices]
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{field_name} must contain numeric prices") from exc
            if any(not math.isfinite(value) or value < 0 for value in prices):
                raise SystemExit(f"{field_name} must contain finite non-negative prices")

            return [int(value) if value.is_integer() else value for value in prices]

        if isinstance(raw_value, int):
            values = [raw_value]
        elif isinstance(raw_value, list):
            values = []
            for index, item in enumerate(raw_value, start=1):
                try:
                    values.append(int(item))
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"{config_path} field '{field_name}' item #{index} must be an integer"
                    ) from exc
        else:
            raise SystemExit(f"{config_path} field '{field_name}' must be an int or list of ints")

        if not values:
            raise SystemExit(f"{config_path} field '{field_name}' cannot be empty")
        if any(value < 1 for value in values):
            raise SystemExit(f"{config_path} field '{field_name}' values must be >= 1")
        return values

    def normalize_optional_float_list(
        field_name: str,
        source_payload: dict[str, Any] | None = None,
    ) -> list[float | None]:
        raw_value = get_config_value(field_name, source_payload)
        if raw_value is None:
            return [None]

        if isinstance(raw_value, (int, float)):
            values = [float(raw_value)]
        elif isinstance(raw_value, list):
            values = []
            for index, item in enumerate(raw_value, start=1):
                try:
                    values.append(float(item))
                except (TypeError, ValueError) as exc:
                    raise SystemExit(
                        f"{config_path} field '{field_name}' item #{index} must be a number"
                    ) from exc
        else:
            raise SystemExit(f"{config_path} field '{field_name}' must be a number or list of numbers")

        if not values:
            raise SystemExit(f"{config_path} field '{field_name}' cannot be empty")
        if any(value < 0 for value in values):
            raise SystemExit(f"{config_path} field '{field_name}' values must be >= 0")
        return values

    def normalize_dynamic_window_pairs(
        field_name: str,
        source_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, int]]:
        raw_value = get_config_value(field_name, source_payload)
        if raw_value is None:
            return []
        if not isinstance(raw_value, list):
            raise SystemExit(f"{config_path} field '{field_name}' must be a list")
        if not raw_value:
            raise SystemExit(f"{config_path} field '{field_name}' cannot be empty")

        pairs: list[dict[str, int]] = []
        for index, item in enumerate(raw_value, start=1):
            window_size: Any
            advance_count: Any
            if isinstance(item, dict):
                window_size = item.get("windowSize", item.get("x"))
                advance_count = item.get("advanceCount", item.get("y"))
            elif isinstance(item, list) and len(item) == 2:
                window_size, advance_count = item
            else:
                raise SystemExit(
                    f"{config_path} field '{field_name}' item #{index} must be an object "
                    "with windowSize/advanceCount (or x/y), or a [x, y] list"
                )

            try:
                normalized_window_size = int(window_size)
                normalized_advance_count = int(advance_count)
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{config_path} field '{field_name}' item #{index} must contain integers"
                ) from exc

            if normalized_window_size < 1 or normalized_advance_count < 1:
                raise SystemExit(
                    f"{config_path} field '{field_name}' item #{index} requires x >= 1 and y >= 1"
                )
            if normalized_advance_count > normalized_window_size:
                raise SystemExit(
                    f"{config_path} field '{field_name}' item #{index} requires y <= x"
                )
            pairs.append(
                {
                    "windowSize": normalized_window_size,
                    "advanceCount": normalized_advance_count,
                }
            )
        return pairs

    replay_mode = str(mode_override or payload.get("mode") or "dynamic").strip().lower()
    if replay_mode != "dynamic":
        raise SystemExit(f"Only dynamic replay is supported, got mode={replay_mode!r}")

    if unique_task_mode is not None:
        if unique_task_mode not in {"repeat", "combine"}:
            raise SystemExit(f"Unknown unique task mode: {unique_task_mode}")
    if unique_task_mode is not None and not unique_task_config:
        raise SystemExit(f"{config_path} requires a non-empty 'uniqueTask' object")
    if unique_compare_mode and not unique_compare_config:
        raise SystemExit(f"{config_path} requires a non-empty 'uniqueCompare' object")

    file_scenarios: list[dict[str, Any]] = []
    raw_case_config = case_config_override or payload.get("caseConfig") or DEFAULT_CASE_CONFIG_NAME
    case_config_path = Path(str(raw_case_config)).expanduser()
    if not case_config_path.is_absolute():
        case_config_path = config_path.parent / case_config_path
    case_config_path = case_config_path.resolve()
    case_payload = load_json_file(case_config_path)
    raw_scenarios = case_payload.get("scenarios") if isinstance(case_payload, dict) else None
    if raw_scenarios is None and isinstance(case_payload, dict):
        # Expand scenarioSets grouped by tasksPerScene during replay.
        raw_scenario_sets = case_payload.get("scenarioSets")
        if isinstance(raw_scenario_sets, list):
            raw_scenarios = []
            matched_set_count = 0
            for set_index, scenario_set in enumerate(raw_scenario_sets, start=1):
                if not isinstance(scenario_set, dict):
                    raise SystemExit(
                        f"{case_config_path} scenario set #{set_index} must be an object"
                    )
                set_tasks_per_scene = scenario_set.get("tasksPerScene")
                if not isinstance(set_tasks_per_scene, int) or set_tasks_per_scene < 1:
                    raise SystemExit(
                        f"{case_config_path} scenario set #{set_index} has invalid tasksPerScene"
                    )
                if (
                    tasks_per_scene_override is not None
                    and set_tasks_per_scene != tasks_per_scene_override
                ):
                    continue
                set_scenarios = scenario_set.get("scenarios")
                if not isinstance(set_scenarios, list):
                    raise SystemExit(
                        f"{case_config_path} scenario set #{set_index} field 'scenarios' "
                        "must be a list"
                    )
                raw_scenarios.extend(set_scenarios)
                matched_set_count += 1
            if tasks_per_scene_override is not None and matched_set_count == 0:
                raise SystemExit(
                    f"{case_config_path} has no scenario set with "
                    f"tasksPerScene={tasks_per_scene_override}"
                )
    elif tasks_per_scene_override is not None and isinstance(raw_scenarios, list):

        filtered_scenarios = []
        for scenario in raw_scenarios:
            if not isinstance(scenario, dict):
                continue
            scenario_tasks_per_scene = scenario.get("tasksPerScene")
            if not isinstance(scenario_tasks_per_scene, int):
                files = scenario.get("files")
                scenario_tasks_per_scene = len(files) if isinstance(files, list) else None
            if scenario_tasks_per_scene == tasks_per_scene_override:
                filtered_scenarios.append(scenario)
        raw_scenarios = filtered_scenarios
        if not raw_scenarios:
            raise SystemExit(
                f"{case_config_path} has no scenario with "
                f"tasksPerScene={tasks_per_scene_override}"
            )
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise SystemExit(
            f"{case_config_path} field 'scenarios' or 'scenarioSets' must contain scenarios"
        )
    for index, scenario in enumerate(raw_scenarios, start=1):
        if not isinstance(scenario, dict):
            raise SystemExit(f"{case_config_path} scenario #{index} must be an object")
        raw_files = scenario.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise SystemExit(f"{case_config_path} scenario #{index} field 'files' must be a non-empty list")
        files = [str(item).strip() for item in raw_files]
        if any(not item for item in files):
            raise SystemExit(f"{case_config_path} scenario #{index} contains an empty file name")
        file_scenarios.append(
            {
                "name": sanitize_label(str(scenario.get("name") or f"scenario_{index:03d}")),
                "files": files,
            }
        )
    # Select scenario_36_001 by name, preserving its original 36-file order and duplicates.
    unique_scenario_name: str | None = None
    if unique_task_mode is not None:
        raw_unique_scenario_name = unique_task_config.get("scenarioName")
        if not isinstance(raw_unique_scenario_name, str) or not raw_unique_scenario_name.strip():
            raise SystemExit(
                f"{config_path} uniqueTask.scenarioName must be a non-empty string"
            )
        unique_scenario_name = sanitize_label(raw_unique_scenario_name)
        file_scenarios = [
            scenario
            for scenario in file_scenarios
            if scenario["name"] == unique_scenario_name
        ]
        if not file_scenarios:
            raise SystemExit(
                f"{case_config_path} has no scenario named {raw_unique_scenario_name!r}"
            )
    # ========== Read peak/off-peak tariff parameters ==========
    allowed_peak_prices = normalize_value_list("peakPrices")
    allowed_valley_prices = normalize_value_list("valleyPrices")
    # ========== End reading tariff parameters ==========
    compare_task_counts: list[int] = []
    if unique_compare_mode:
        compare_task_counts = normalize_value_list("taskCounts")
        if (
            len(compare_task_counts) != len(UNIQUE_COMPARE_TASK_COUNTS)
            or set(compare_task_counts) != set(UNIQUE_COMPARE_TASK_COUNTS)
        ):
            raise SystemExit(
                f"{config_path} uniqueCompare.taskCounts must contain 6, 7, 8, 9, 10, 11, 12"
            )
        if dynamic_window_size_override is not None or dynamic_advance_count_override is not None:
            raise SystemExit(
                "--unique-compare fixes task counts/windows to 6..12; "
                "do not combine it with --dynamic-window-size/--dynamic-advance-count"
            )

    # If unspecified, do not override the service's schedule_params.jsonc default.
    allowed_makespan_cost_units = normalize_optional_float_list(
        "makespan_60min_cost_in_valley_price_units"
    )
    # repeat uses x12/y1; combine uses configured x6..x11/y1 combinations.
    if unique_compare_mode:
        configured_dynamic_window_pairs = [
            {"windowSize": task_count, "advanceCount": 1}
            for task_count in compare_task_counts
        ]
    elif unique_task_mode is None:
        configured_dynamic_window_pairs = normalize_dynamic_window_pairs(
            "dynamicWindowAdvancePairs"
        )
    else:
        unique_window_field = (
            "repeatWindowAdvancePairs"
            if unique_task_mode == "repeat"
            else "combineWindowAdvancePairs"
        )
        configured_dynamic_window_pairs = normalize_dynamic_window_pairs(
            unique_window_field,
            unique_task_config,
        )
        if not configured_dynamic_window_pairs:
            raise SystemExit(
                f"{config_path} uniqueTask.{unique_window_field} cannot be empty"
            )

    unique_repeat_count = 1
    if unique_task_mode == "repeat":
        raw_repeat_count = unique_task_config.get("repeatCount")
        try:
            unique_repeat_count = int(raw_repeat_count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{config_path} uniqueTask.repeatCount must be an integer"
            ) from exc
        if unique_repeat_count < 1:
            raise SystemExit(f"{config_path} uniqueTask.repeatCount must be >= 1")

    def resolve_dynamic_window_pairs() -> list[dict[str, int]]:
        if unique_compare_mode:
            return configured_dynamic_window_pairs

        # If either CLI dimension is supplied, replace config lists with one combination; CLI overrides config.
        if dynamic_window_size_override is not None or dynamic_advance_count_override is not None:
            window_size = int(
                dynamic_window_size_override
                if dynamic_window_size_override is not None
                else parameter_payload.get("dynamicWindowSize", 6)
            )
            advance_count = int(
                dynamic_advance_count_override
                if dynamic_advance_count_override is not None
                else parameter_payload.get("dynamicAdvanceCount", 1)
            )
            if window_size < 1 or advance_count < 1 or advance_count > window_size:
                raise SystemExit("dynamic requires x >= 1 and 1 <= y <= x")
            return [{"windowSize": window_size, "advanceCount": advance_count}]

        if configured_dynamic_window_pairs:
            return configured_dynamic_window_pairs

        window_size = int(parameter_payload.get("dynamicWindowSize", 6))
        advance_count = int(parameter_payload.get("dynamicAdvanceCount", 1))
        if window_size < 1 or advance_count < 1 or advance_count > window_size:
            raise SystemExit("dynamic requires x >= 1 and 1 <= y <= x")
        return [{"windowSize": window_size, "advanceCount": advance_count}]

    dynamic_window_pairs = resolve_dynamic_window_pairs()

    def normalize_schedule_methods() -> list[str]:
        field_name = (
            "uniqueCompare.scheduleMethods"
            if unique_compare_mode
            else "uniqueTask.scheduleMethods"
            if unique_task_mode is not None
            else "schedule_method"
        )
        raw_value = (
            unique_compare_config.get("scheduleMethods")
            if unique_compare_mode
            else unique_task_config.get("scheduleMethods")
            if unique_task_mode is not None
            else payload.get("schedule_method")
        )
        if raw_value is None:
            raise SystemExit(f"{config_path} is missing required field '{field_name}'")

        if isinstance(raw_value, str):
            methods = [raw_value.strip()]
        elif isinstance(raw_value, list):
            methods = []
            for index, item in enumerate(raw_value, start=1):
                if not isinstance(item, str) or not item.strip():
                    raise SystemExit(
                        f"{config_path} field '{field_name}' item #{index} must be a non-empty string"
                    )
                methods.append(item.strip())
        else:
            raise SystemExit(f"{config_path} field '{field_name}' must be a string or list of strings")

        if not methods:
            raise SystemExit(f"{config_path} field '{field_name}' cannot be empty")
        if unique_compare_mode:
            normalized_methods = [method.strip().lower() for method in methods]
            expected_methods = set(UNIQUE_COMPARE_SCHEDULE_METHODS)
            if len(normalized_methods) != len(expected_methods) or set(normalized_methods) != expected_methods:
                raise SystemExit(
                    f"{config_path} uniqueCompare.scheduleMethods must contain exactly: "
                    f"{', '.join(UNIQUE_COMPARE_SCHEDULE_METHODS)}"
                )
            # Compare runs two unconstrained baselines before the RL-dependent objective-limit method, regardless of config order.
            methods = list(UNIQUE_COMPARE_SCHEDULE_METHODS)
        return methods

    allowed_schedule_methods = normalize_schedule_methods()

    # Give each repeat run an independent index; run each combine/compare window once.
    repeat_indices: tuple[int | None, ...] = (
        tuple(range(1, unique_repeat_count + 1))
        if unique_task_mode == "repeat"
        else (None,)
    )
    experiments: list[dict[str, Any]] = []
    for schedule_method in allowed_schedule_methods:
        for makespan_cost_units in allowed_makespan_cost_units:
            # Use fixed scenarios in the dynamic-replay parameter Cartesian product.
            for selection_dimension in file_scenarios:
                for peak_price in allowed_peak_prices:
                    for valley_price in allowed_valley_prices:
                        for pair_index, dynamic_window_pair in enumerate(
                            dynamic_window_pairs,
                            start=1,
                        ):
                            compare_task_count: int | None = None
                            scenario_files = selection_dimension["files"]
                            if unique_compare_mode:
                                compare_task_count = int(dynamic_window_pair["windowSize"])
                                if compare_task_count > len(selection_dimension["files"]):
                                    raise SystemExit(
                                        f"{case_config_path} scenario {selection_dimension['name']!r} "
                                        f"has fewer than {compare_task_count} files"
                                    )
                                # Each compare experiment uses the first N scenario tasks, with N equal to its window; the first window is final.
                                scenario_files = selection_dimension["files"][:compare_task_count]
                            for repeat_index in repeat_indices:
                                selection_label = selection_dimension["name"]
                                makespan_cost_label = (
                                    f"_mc{format_numeric_label(makespan_cost_units)}"
                                    if makespan_cost_units is not None
                                    else ""
                                )
                                unique_label = ""
                                if unique_task_mode == "repeat":
                                    unique_label = f"_unique_repeat_r{int(repeat_index):02d}"
                                elif unique_task_mode == "combine":
                                    unique_label = f"_unique_combine_c{pair_index:02d}"
                                elif unique_compare_mode:
                                    assert compare_task_count is not None
                                    unique_label = f"_unique_compare_n{compare_task_count:02d}"
                                experiment = {
                                    # Experiment names include only parameters participating in the combination.
                                    "experimentName": sanitize_label(
                                        f"{selection_label}"
                                        f"{unique_label}"
                                        f"{makespan_cost_label}_pp{peak_price}_vp{valley_price}"
                                    ),
                                    "scheduleMethod": schedule_method,
                                    "filesPerBatch": len(scenario_files),
                                    "scenarioFiles": list(scenario_files),
                                    "peakPrice": peak_price,
                                    "valleyPrice": valley_price,
                                    "makespan_60min_cost_in_valley_price_units": makespan_cost_units,
                                }
                                if unique_task_mode is not None:
                                    experiment.update(
                                        {
                                            "uniqueTaskMode": unique_task_mode,
                                            "uniqueTaskScenarioName": unique_scenario_name,
                                            "uniqueTaskRepeatIndex": repeat_index,
                                            "uniqueTaskCombinationIndex": (
                                                pair_index
                                                if unique_task_mode == "combine"
                                                else None
                                            ),
                                        }
                                    )
                                elif unique_compare_mode:
                                    experiment.update(
                                        {
                                            "uniqueCompareMode": "compare",
                                            "uniqueCompareScenarioName": selection_label,
                                            "uniqueCompareTaskCount": compare_task_count,
                                        }
                                    )
                                experiment["dynamicWindowSize"] = dynamic_window_pair["windowSize"]
                                experiment["dynamicAdvanceCount"] = dynamic_window_pair["advanceCount"]
                                experiments.append(experiment)

    return experiments, replay_mode, case_config_path


def resolve_scenario_files(source_dir: Path, configured_files: list[str]) -> list[Path]:
    # Scenario filenames may include or omit the .json suffix.
    selected: list[Path] = []
    for configured_name in configured_files:
        file_name = configured_name if configured_name.lower().endswith(".json") else f"{configured_name}.json"
        file_path = (source_dir / file_name).resolve()
        if file_path.parent != source_dir or not file_path.is_file():
            raise SystemExit(f"Scenario experiment file not found in {source_dir}: {configured_name}")
        selected.append(file_path)
    return selected


def build_output_name(index: int, label: str) -> str:
    return f"{index:03d}_{label}.svg"


def _extract_record_objective_value(record: dict[str, Any]) -> int | None:
    raw_solver_objective = record.get("solverObjectiveValue")
    if raw_solver_objective is not None:
        try:
            return int(round(float(raw_solver_objective)))
        except (TypeError, ValueError):
            return None

    beta = record.get("beta")
    # JSONL metrics use makespan; keep this only as a recomputation fallback.
    makespan = record.get("makespan")
    carbon = record.get("recomputedCarbonObjectiveScaled")
    if beta is None or makespan is None or carbon is None:
        return None

    try:
        return int(round(float(beta) * float(makespan) + float(carbon)))
    except (TypeError, ValueError):
        return None


def _extract_record_int(record: dict[str, Any], field_name: str) -> int | None:
    raw_value = record.get(field_name)
    if raw_value is None:
        return None
    try:
        return int(round(float(raw_value)))
    except (TypeError, ValueError):
        return None


def _same_file_stem_list(left: list[Any], right: list[Any]) -> bool:
    return sorted(str(item) for item in left) == sorted(str(item) for item in right)


def _same_numeric_value(left: Any, right: Any, tol: float = 1e-6) -> bool:
    left_value = parse_optional_float(left)
    right_value = parse_optional_float(right)
    if left_value is None or right_value is None:
        return False
    return abs(left_value - right_value) <= tol


def find_matching_rl_limit_values(
    results_path: Path,
    experiment_name: str,
    experiment_file_list: list[str],
    peak_price: float,
    valley_price: float,
    makespan_60min_cost_in_valley_price_units: float | None,
    tariff_info: dict[str, Any] | None = None,
) -> dict[str, int | None]:
    from algorithm.utils.tariff import tariff_identity

    if not results_path.exists():
        return {"objective_limit": None, "makespan_limit": None, "carbon_limit": None}

    matched_limits = {"objective_limit": None, "makespan_limit": None, "carbon_limit": None}
    with results_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                str(record.get("scheduleMethod", "")).strip().lower()
                != OBJECTIVE_LIMIT_SOURCE_METHOD
            ):
                continue
            if str(record.get("experimentName")) != str(experiment_name):
                continue

            record_file_list = record.get("experimentFileList")
            if not isinstance(record_file_list, list):
                continue
            if not _same_file_stem_list(record_file_list, experiment_file_list):
                continue
            if not _same_numeric_value(record.get("peakPrice"), peak_price):
                continue
            if not _same_numeric_value(record.get("valleyPrice"), valley_price):
                continue
            if (
                makespan_60min_cost_in_valley_price_units is not None
                and not _same_numeric_value(
                    record.get("makespan_60min_cost_in_valley_price_units"),
                    makespan_60min_cost_in_valley_price_units,
                )
            ):
                continue

            if tariff_identity(record.get("tariffInfo") or {}) != tariff_identity(tariff_info or {}):
                continue
            objective_value = _extract_record_objective_value(record)
            if objective_value is not None:
                matched_limits["objective_limit"] = objective_value
            makespan_value = _extract_record_int(record, "makespan")
            if makespan_value is not None:
                matched_limits["makespan_limit"] = makespan_value
            carbon_value = _extract_record_int(record, "recomputedCarbonObjectiveScaled")
            if carbon_value is not None:
                matched_limits["carbon_limit"] = carbon_value

    return matched_limits


def read_replay_tariff(database_path):
    import json5
    from algorithm.utils.tariff import load_tariff_config
    directory = Path(database_path).parent
    with (directory / "schedule_params.jsonc").open(encoding="utf-8") as file:
        params = json5.load(file)
    return load_tariff_config(params.get("power_info", {}), directory)


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.tasks_per_scene is not None and args.tasks_per_scene < 1:
        parser.error("--tasks-per-scene must be >= 1")
    unique_task_mode = (
        "repeat"
        if args.unique_task_repeat
        else "combine"
        if args.unique_task_combine
        else None
    )
    unique_compare_mode = bool(args.unique_compare)

    # Print raw CLI arguments first, then report effective dynamic-window settings after loading configuration.
    print("replay experiment args:", flush=True)
    pprint(vars(args), sort_dicts=False)

    config_path = Path(args.config).expanduser().resolve()
    replay_config = load_json_file(config_path)
    configured_tariff = replay_config.get("tariffInfo")
    if configured_tariff is not None:
        from algorithm.utils.tariff import daily_tariff, TARIFF_FIELDS
        if (not isinstance(configured_tariff, dict)
                or not {"peak_price", "valley_price", "tariff_periods"} <= configured_tariff.keys()
                or configured_tariff.keys() - set(TARIFF_FIELDS)):
            parser.error("tariffInfo must contain only tariff fields, including prices and tariff_periods")
        daily_tariff(configured_tariff)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output_dir = JSONL_OUTPUT_DIR
    jsonl_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_definitions, replay_mode, case_config_path = (
        load_experiment_definitions(
            config_path,
            "dynamic",  # Dynamic replay is the supported experiment mode.
            args.case_config,
            args.dynamic_window_size,
            args.dynamic_advance_count,
            args.tasks_per_scene,
            unique_task_mode,
            unique_compare_mode,
        )
    )
    if any(
        experiment.get("dynamicWindowSize") is not None
        and experiment.get("dynamicAdvanceCount") is not None
        for experiment in experiment_definitions
    ):
        resolved_dynamic_pairs = sorted(
            {
                (
                    int(experiment["dynamicWindowSize"]),
                    int(experiment["dynamicAdvanceCount"]),
                )
                for experiment in experiment_definitions
                if experiment.get("dynamicWindowSize") is not None
                and experiment.get("dynamicAdvanceCount") is not None
            }
        )
        print(
            "resolved dynamicWindowSize/dynamicAdvanceCount from config: "
            f"{resolved_dynamic_pairs}",
            flush=True,
        )
    dynamic_options = resolve_dynamic_options(
        config_path,
        args,
        unique_task_mode,
        unique_compare_mode,
    )
    is_unique_task_replay = unique_task_mode is not None
    is_unique_compare_replay = unique_compare_mode
    details_results_path = jsonl_output_dir / (
        UNIQUE_TASK_DETAILS_RESULTS_NAME
        if is_unique_task_replay or is_unique_compare_replay
        else DYNAMIC_DETAILS_RESULTS_NAME
    )
    for name in (args.results_file, args.details_results_file):
        if name is not None and (Path(name).name != name or not name.endswith(".jsonl")):
            parser.error("output must be a JSONL filename without a directory")
    if args.details_results_file:
        details_results_path = jsonl_output_dir / args.details_results_file
    elif args.results_file:
        details_results_path = jsonl_output_dir / (Path(args.results_file).stem + "_details.jsonl")
    dynamic_results_paths: set[Path] = set()
    svg_output_dir = output_dir / "temp" / "dynamic"
    svg_output_dir.mkdir(parents=True, exist_ok=True)
    replay_state_path = (
        output_dir
        / "temp"
        / (
            UNIQUE_TASK_REPLAY_STATE_NAME
            if is_unique_task_replay
            else UNIQUE_COMPARE_REPLAY_STATE_NAME
            if is_unique_compare_replay
            else DYNAMIC_REPLAY_STATE_NAME
        )
    )
    if not experiment_definitions:
        raise SystemExit(f"No enabled experiments found in {config_path}")

    experiment_files = resolve_experiment_files(source_dir)
    if not experiment_files:
        raise SystemExit(f"No experiment files found in {source_dir}")

    payload_cache = {file_path: load_json_file(file_path) for file_path in experiment_files}

    # fileRepeatIndex is runtime resume state used by dynamic replay.
    replay_state = load_replay_state(replay_state_path)
    file_repeat_index = extract_file_repeat_index(replay_state, experiment_files)
    if replay_state.get("fileRepeatIndex"):
        print(f"Loaded replay state: {replay_state_path}", flush=True)
    print(
        "Replay mode: dynamic",
        flush=True,
    )
    print(f"Replay case config: {case_config_path}", flush=True)

    logger = AsyncLog("HorizonReplayExperiment")
    from include.dynamic_runner import DynamicHorizonReplayRunner

    dynamic_database_path = dynamic_options["databasePath"]
    runner = DynamicHorizonReplayRunner(
        database_path=dynamic_database_path,
        output_dir=svg_output_dir,
        logger=logger,
        schedule_port=dynamic_options["schedulePort"],
        request_timeout=REQUEST_TIMEOUT,
        replay_interval=REPLAY_INTERVAL,
        checkpoint_path=replay_state_path,
    )
    fixed_simulation_start_time = resolve_fixed_simulation_start_time()
    if replay_config.get("simulationStartTime"):
        fixed_simulation_start_time = parse_datetime(replay_config["simulationStartTime"])
        if fixed_simulation_start_time is None:
            parser.error("simulationStartTime must use YYYY-MM-DD HH:MM:SS")
    print(
        f"Replay fixed simulation start time: {format_datetime(fixed_simulation_start_time)}",
        flush=True,
    )
    for config_index, experiment in enumerate(experiment_definitions, start=1):
        base_label = experiment["experimentName"]
        dynamic_window_size = experiment.get("dynamicWindowSize")
        dynamic_advance_count = experiment.get("dynamicAdvanceCount")
        label = sanitize_label(
            f"{base_label}_dyn_x{dynamic_window_size}"
            f"_y{dynamic_advance_count}"
        )
        schedule_method = str(experiment["scheduleMethod"])
        dynamic_results_path = (
            jsonl_output_dir / UNIQUE_TASK_RESULTS_NAME
            if is_unique_task_replay or is_unique_compare_replay
            else build_dynamic_results_path(jsonl_output_dir)
        )
        if args.results_file:
            dynamic_results_path = jsonl_output_dir / args.results_file
        dynamic_results_paths.add(dynamic_results_path)
        experiment_file_count = int(experiment["filesPerBatch"])
        # ========== Read peak/off-peak tariff configuration ==========
        peak_price = float(experiment.get("peakPrice", 2))
        valley_price = float(experiment.get("valleyPrice", 1))
        # ========== End reading tariff configuration ==========
        # Use current experiment scan prices; tariff periods and mid-peak prices come from the selected replay config.
        effective_tariff = None
        if configured_tariff is not None:
            effective_tariff = dict(configured_tariff, peak_price=peak_price, valley_price=valley_price)
            daily_tariff(effective_tariff)
        raw_makespan_cost_units = experiment.get("makespan_60min_cost_in_valley_price_units")
        makespan_cost_units = (
            parse_optional_float(raw_makespan_cost_units)
            if raw_makespan_cost_units is not None
            else None
        )
        selected_experiment_files = resolve_scenario_files(source_dir, experiment["scenarioFiles"])
        output_name = build_output_name(config_index, label)

        print(
            f"\033[92m[experiment {config_index:03d}]\033[0m "
            f"start name={label} method={schedule_method} "
            f"experimentFiles={experiment_file_count} peakPrice={peak_price} valleyPrice={valley_price} "
            f"makespanCostUnits={makespan_cost_units}",
            flush=True,
        )
        objective_limit = None
        makespan_limit = None
        carbon_limit = None
        if is_unique_compare_replay and schedule_method in UNIQUE_COMPARE_LIMIT_METHODS:
            limit_values = find_matching_rl_limit_values(
                tariff_info=effective_tariff if effective_tariff is not None else read_replay_tariff(dynamic_database_path),
                results_path=dynamic_results_path,
                experiment_name=label,
                experiment_file_list=[path.stem for path in selected_experiment_files],
                peak_price=peak_price,
                valley_price=valley_price,
                makespan_60min_cost_in_valley_price_units=makespan_cost_units,
            )
            objective_limit = limit_values["objective_limit"]
            makespan_limit = limit_values["makespan_limit"]
            carbon_limit = limit_values["carbon_limit"]
            required_limit_fields = (
                ("objective_limit",)
                if schedule_method == "cp_green_limit_or"
                else ("makespan_limit", "carbon_limit")
            )
            missing_limit_fields = [
                field for field in required_limit_fields if limit_values[field] is None
            ]
            if missing_limit_fields:
                raise RuntimeError(
                    f"unique-compare requires completed {OBJECTIVE_LIMIT_SOURCE_METHOD} result "
                    f"before {schedule_method}; missing {', '.join(missing_limit_fields)} "
                    f"for {label}"
                )
            print(
                f"unique-compare limits for {schedule_method}: "
                f"objective={objective_limit} makespan={makespan_limit} carbon={carbon_limit}",
                flush=True,
            )
        dynamic_result = runner.run(
            experiment_files=experiment_files,
            experiment_file_selection=selected_experiment_files,
            payload_cache=payload_cache,
            file_repeat_index=file_repeat_index,
            window_size=int(dynamic_window_size),
            advance_count=int(dynamic_advance_count),
            schedule_method=schedule_method,
            peak_price=peak_price,
            valley_price=valley_price,
            makespan_60min_cost_in_valley_price_units=makespan_cost_units,
            objective_limit=objective_limit,
            makespan_limit=makespan_limit,
            carbon_limit=carbon_limit,
            simulation_start_time=fixed_simulation_start_time,
            output_name=output_name,
            experiment_name=label,
            time_scale=1.0,
            tariff_info=effective_tariff,
        )
        snapshot = dynamic_result.snapshot
        recomputed_objective_value = None
        if (
            snapshot is not None
            and snapshot.beta is not None
            and snapshot.recomputed_carbon_objective_scaled is not None
        ):
            recomputed_objective_value = int(
                snapshot.beta * snapshot.makespan + snapshot.recomputed_carbon_objective_scaled
            )
        total_algo_time = round_optional_float(dynamic_result.total_algo_cal_time)
        source_counts: dict[str, int] = {}
        objective_comparison_counts: dict[str, int] = {}
        for batch in dynamic_result.dynamic_batches:
            source = batch.get("source")
            if source:
                source_counts[str(source)] = source_counts.get(str(source), 0) + 1
            comparison = batch.get("objectiveComparison")
            if comparison:
                comparison_key = str(comparison)
                objective_comparison_counts[comparison_key] = (
                    objective_comparison_counts.get(comparison_key, 0) + 1
                )  # Count better/equal/worse/no_cp_solution CP outcomes relative to RL incumbents.
        unique_task_record_fields = (
            {
                "uniqueTaskMode": experiment.get("uniqueTaskMode"),
                "uniqueTaskScenarioName": experiment.get("uniqueTaskScenarioName"),
                "uniqueTaskRepeatIndex": experiment.get("uniqueTaskRepeatIndex"),
                "uniqueTaskCombinationIndex": experiment.get("uniqueTaskCombinationIndex"),
            }
            if is_unique_task_replay
            else {}
        )
        unique_compare_record_fields = (
            {
                "uniqueCompareMode": experiment.get("uniqueCompareMode"),
                "uniqueCompareScenarioName": experiment.get("uniqueCompareScenarioName"),
                "uniqueCompareTaskCount": experiment.get("uniqueCompareTaskCount"),
            }
            if is_unique_compare_replay
            else {}
        )
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configIndex": config_index,
            "experimentName": label,
            "experimentFileList": [path.stem for path in dynamic_result.used_files],
            "scheduleMethod": dynamic_result.schedule_method,
            "configuredScheduleMethod": schedule_method,
            "scheduleTimeLimit": round_optional_float(
                dynamic_result.schedule_time_limit
            ),  # Record the actual per-batch dynamic service time limit.
            "schedule_status": dynamic_result.schedule_status,
            "filesPerExperiment": experiment_file_count,
            "replayMode": "dynamic",
            "simulationStartTime": format_datetime(dynamic_result.start_time),
            "beta": snapshot.beta if snapshot else None,
            "objectiveLimit": objective_limit,
            "makespanLimit": makespan_limit,
            "carbonLimit": carbon_limit,
            # Window-level raw solver values are not whole-run values; keep them only in dynamicBatches.
            "solverObjectiveValue": None,
            "solverObjectiveCarbonScaled": None,
            "recomputedObjectiveValue": recomputed_objective_value,
            "makespan": dynamic_result.total_makespan,
            "recomputedCarbonObjectiveScaled": dynamic_result.total_carbon_objective_scaled,
            "recomputedCarbonObjectiveScaledPeak": (
                snapshot.recomputed_carbon_objective_scaled_peak if snapshot else None
            ),
            "recomputedCarbonObjectiveScaledValley": (
                snapshot.recomputed_carbon_objective_scaled_valley if snapshot else None
            ),
            "totalElectricityCostYuan": round_optional_float(
                dynamic_result.total_electricity_cost_yuan, digits=6
            ),
            "totalElectricityCostYuanPeak": round_optional_float(
                snapshot.electricity_cost_yuan_peak if snapshot else None,
                digits=6,
            ),
            "totalElectricityCostYuanValley": round_optional_float(
                snapshot.electricity_cost_yuan_valley if snapshot else None,
                digits=6,
            ),
            "totalAlgoCalTime": total_algo_time,
            "algoCalTimes": dynamic_result.algo_cal_times,
            "totalOperationCount": dynamic_result.total_operation_count,
            "tariffInfo": snapshot.tariff_info if snapshot else None,
            "tariffCostBreakdown": snapshot.tariff_cost_breakdown if snapshot else None,
            "peakPrice": round_optional_float(snapshot.peak_price if snapshot else peak_price),
            "valleyPrice": round_optional_float(snapshot.valley_price if snapshot else valley_price),
            "makespan_60min_cost_in_valley_price_units": round_optional_float(
                snapshot.makespan_60min_cost_in_valley_price_units if snapshot else makespan_cost_units
            ),
            "dynamicWindowSize": dynamic_window_size,
            "dynamicAdvanceCount": dynamic_advance_count,
            "totalTaskCount": dynamic_result.total_tasks,
            "dynamicBatchCount": len(dynamic_result.dynamic_batches),
            "sourceCounts": source_counts,
            "objectiveComparisonCounts": objective_comparison_counts,
            "dynamicTaskMetrics": dynamic_result.dynamic_task_metrics,
            "dynamicRunId": dynamic_result.run_id,
            "serverSessionId": dynamic_result.server_session_id,
            "errorMessage": dynamic_result.error_message,
            **({"countryScenario": replay_config["countryScenario"]} if "countryScenario" in replay_config else {}),
            **unique_task_record_fields,
            **unique_compare_record_fields,
        }
        append_jsonl_line(dynamic_results_path, record)
        append_jsonl_line(
            details_results_path,
            {
                **record,
                "dynamicBatches": dynamic_result.dynamic_batches,
                "ws_table": snapshot.raw_ws_table if snapshot else None,
            },
        )
        # The dynamic runner writes dedicated checkpoints atomically per batch; never overwrite them with generic state.
        print(
            f"\033[92m[experiment {config_index:03d}]\033[0m done "
            f"method={dynamic_result.schedule_method} status={dynamic_result.schedule_status} "
            f"makespan={dynamic_result.total_makespan} "
            f"objective_carbon={dynamic_result.total_carbon_objective_scaled} "
            f"electricity_yuan={round_optional_float(dynamic_result.total_electricity_cost_yuan, digits=6)} "
            f"batches={len(dynamic_result.dynamic_batches)} algo={total_algo_time}",
            flush=True,
        )

    for dynamic_results_path in sorted(dynamic_results_paths):
        print(f"Experiment results saved to: {dynamic_results_path}")
    print(f"Replay state saved to: {replay_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
