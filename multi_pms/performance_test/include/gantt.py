from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any

from algorithm.utils.tariff import daily_tariff

import builtins

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Plot feature toggles (quickly editable)
# True colors by task; False by workstation. Change replay plot colors here.
DEFAULT_COLOR_BY_TASK = False
# True adds a workstation panel; False keeps experiment and tariff/cumulative-cost panels only.
SHOW_WORKSTATION_PLOT = False
# Show machine names on the workstation y axis; disable for dense task sets.
SHOW_MACHINE_YTICK_LABELS = False
# Show experiment task labels; disable for crowded long-replay plots.
SHOW_EXPERIMENT_YTICK_LABELS = True
# Show the workstation legend; it may be long with many machine types.
SHOW_MACHINE_LEGEND = False
# Show the tariff/cumulative-cost legend; normally useful for identifying curves.
SHOW_TOU_LEGEND = True
# Overlay task lock brackets in the experiment view, matching scheduler.plot_schedule().
SHOW_EXPERIMENT_LOCKS = True
# Horizontal cap length of lock brackets; increase for more visible brackets.
LOCK_BRACKET_ARM_LENGTH = 5.0


@dataclass(slots=True)
class ExperimentSnapshot:
    label: str
    source_file: str
    source_files: list[str]
    start_time: datetime
    makespan: int
    expr_nos: list[str]
    expr_nos_by_source_file: dict[str, list[str]]
    raw_ws_table: dict[str, list[dict[str, Any]]]
    raw_bottle_table: dict[str, list[dict[str, Any]]]
    algo_cal_time: float | None = None
    operation_count: int = 0
    peak_price: float | None = None
    valley_price: float | None = None
    makespan_60min_cost_in_valley_price_units: float | None = None  # Record the makespan time-value configuration for replay metrics.
    tariff_info: dict[str, Any] | None = None
    tariff_cost_breakdown: dict[str, Any] | None = None
    beta: int | None = None
    solver_objective_value: int | None = None
    solver_carbon_objective_scaled: int | None = None
    recomputed_carbon_objective_scaled: int | None = None
    recomputed_carbon_objective_scaled_peak: int | None = None
    recomputed_carbon_objective_scaled_valley: int | None = None
    electricity_cost_yuan: float | None = None
    electricity_cost_yuan_peak: float | None = None
    electricity_cost_yuan_valley: float | None = None
    machine_power_by_ws_code: dict[str, float] | None = None


class HorizonGanttPlotter:
    def __init__(self, power_info: dict[str, Any] | None, output_dir: str | Path):
        self.power_info = power_info or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_total_carbon: int | None = None
        self.green_objective_scale_indicator = 60  # Use scale_indicator=60 consistently with the green solver when recomputing replay metrics.

        whites = {
            "white",
            "whitesmoke",
            "snow",
            "seashell",
            "linen",
            "ivory",
            "honeydew",
            "floralwhite",
            "ghostwhite",
            "aliceblue",
            "azure",
            "mintcream",
            "oldlace",
            "antiquewhite",
        }
        self.colors = [name for name in mcolors.CSS4_COLORS.keys() if name not in whites]

    def _color_for(self, key: str) -> str:
        return mcolors.CSS4_COLORS[self.colors[hash(key) % len(self.colors)]]

    def _max_overlap(self, intervals: list[tuple[int, int]]) -> int:
        events: list[tuple[int, int]] = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))

        active = 0
        peak = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            active += delta
            peak = max(peak, active)
        return max(1, peak)

    # Share tariff classification between energy splits and price curves to align boundaries.
    def _is_valley_minute(self, minute_of_day: int) -> bool:
        return self._tariff_key_at(minute_of_day) == "valley_price"

    def _tariff_key_at(self, minute):
        day = getattr(self, "_effective_tariff_day", None) or daily_tariff(self.power_info)
        return next(key for start, end, key in day if start <= minute % 1440 < end)

    def _price_at(self, minute_of_day: int, peak_price: float, valley_price: float) -> float:
        key = self._tariff_key_at(minute_of_day)
        return {"peak_price": peak_price, "valley_price": valley_price,
                "flat_price": getattr(self, "_effective_flat_price", self.power_info.get("flat_price"))}[key]

    def _resolve_prices(self, snapshot: ExperimentSnapshot) -> tuple[float, float]:
        if snapshot.peak_price is not None and snapshot.valley_price is not None:
            return float(snapshot.peak_price), float(snapshot.valley_price)

        return (
            float(self.power_info.get("peak_price", 0.6)),
            float(self.power_info.get("valley_price", 0.3)),
        )

    def _resolve_machine_power_info(self, snapshot: ExperimentSnapshot) -> dict[str, float]:

        if snapshot.machine_power_by_ws_code:
            return {str(key): float(value) for key, value in snapshot.machine_power_by_ws_code.items()}

        machine_power_info = self.power_info.get("machine_power_info")
        if isinstance(machine_power_info, dict):
            return {str(key): float(value) for key, value in machine_power_info.items()}

        legacy_power_info = self.power_info.get("power_info")
        if isinstance(legacy_power_info, dict):
            return {str(key): float(value) for key, value in legacy_power_info.items()}

        return {}

    def _calculate_tou_cost(
        self,
        snapshot: ExperimentSnapshot,
        machine_tasks: dict[str, list[dict[str, Any]]],
        start_time: datetime,
        horizon_end: int,
        peak_price: float,
        valley_price: float,
    ) -> tuple[
        int | None,
        float | None,
        int | None,
        int | None,
        float | None,
        float | None,
        list[float],
        list[float],
    ]:
        effective_info = dict(snapshot.tariff_info or self.power_info)
        effective_info.update(peak_price=peak_price, valley_price=valley_price)
        self._effective_tariff_day = daily_tariff(effective_info)
        self._effective_flat_price = effective_info.get("flat_price")
        x_max = max(horizon_end, 1)

        dedup_experiments: dict[tuple[str, int, int], bool] = {}
        for machine_name, tasks in machine_tasks.items():
            for task in tasks:
                key = (machine_name, int(task["start_time"]), int(task["end_time"]))
                dedup_experiments[key] = True

        machine_power_info = self._resolve_machine_power_info(snapshot)
        objective_cost_delta = [0] * x_max
        electricity_cost_delta = [0.0] * x_max
        origin_min = start_time.hour * 60 + start_time.minute
        for machine_name, start, end in dedup_experiments.keys():
            power_w = machine_power_info.get(machine_name, 1000.0)
            power_int = int(builtins.round(power_w))
            for minute in range(max(0, start), min(x_max, end)):
                minute_of_day = (origin_min + minute) % 1440
                price = self._price_at(minute_of_day, peak_price, valley_price)
                objective_cost_delta[minute] += int(
                    builtins.round(price * power_int / self.green_objective_scale_indicator)
                )
                electricity_cost_delta[minute] += price * power_int / 60000.0

        price_curve = [
            self._price_at((origin_min + minute) % 1440, peak_price, valley_price)
            for minute in range(x_max + 1)
        ]
        cum_cost_scaled = [0.0] * (x_max + 1)
        for minute in range(x_max):
            cum_cost_scaled[minute + 1] = cum_cost_scaled[minute] + electricity_cost_delta[minute]

        total_carbon = sum(objective_cost_delta) if objective_cost_delta else None
        total_electricity_cost_yuan = cum_cost_scaled[-1] if cum_cost_scaled else None
        # Split metrics by each minute's tariff using the same increments as total metrics.
        peak_minutes = [
            self._tariff_key_at((origin_min + minute) % 1440) == "peak_price"
            for minute in range(x_max)
        ]
        carbon_peak = sum(
            value for minute, value in enumerate(objective_cost_delta) if peak_minutes[minute]
        )
        valley_minutes = [self._is_valley_minute((origin_min + minute) % 1440) for minute in range(x_max)]
        self.tariff_cost_breakdown = {
            key.removesuffix("_price"): {
                "objective_scaled": sum(value for minute, value in enumerate(objective_cost_delta)
                                        if self._tariff_key_at(origin_min + minute) == key),
                "electricity_cost": sum(value for minute, value in enumerate(electricity_cost_delta)
                                        if self._tariff_key_at(origin_min + minute) == key),
            } for key in ("peak_price", "valley_price", "flat_price")
        }
        carbon_valley = sum(
            value for minute, value in enumerate(objective_cost_delta) if valley_minutes[minute]
        )
        electricity_cost_peak = sum(
            value for minute, value in enumerate(electricity_cost_delta) if peak_minutes[minute]
        )
        electricity_cost_valley = sum(
            value for minute, value in enumerate(electricity_cost_delta) if valley_minutes[minute]
        )
        snapshot.tariff_cost_breakdown = self.tariff_cost_breakdown
        return (
            total_carbon,
            total_electricity_cost_yuan,
            carbon_peak,
            carbon_valley,
            electricity_cost_peak,
            electricity_cost_valley,
            price_curve,
            cum_cost_scaled,
        )

    def calculate_experiment_energy_metrics(
        self,
        snapshot: ExperimentSnapshot,
    ) -> tuple[int | None, float | None, int | None, int | None, float | None, float | None]:
        machine_tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for ws_code, tasks in snapshot.raw_ws_table.items():
            for task in tasks:
                start_time = task.get("start_time")
                end_time = task.get("end_time")
                if start_time is None or end_time is None or start_time >= end_time:
                    continue

                machine_tasks[ws_code].append(
                    {
                        "start_time": int(start_time),
                        "end_time": int(end_time),
                        "duration": int(task.get("duration") or (int(end_time) - int(start_time))),
                    }
                )

        peak_price, valley_price = self._resolve_prices(snapshot)
        (
            total_carbon,
            total_electricity_cost_yuan,
            carbon_peak,
            carbon_valley,
            electricity_cost_peak,
            electricity_cost_valley,
            _,
            _,
        ) = self._calculate_tou_cost(
            snapshot=snapshot,
            machine_tasks=dict(machine_tasks),
            start_time=snapshot.start_time,
            horizon_end=snapshot.makespan,
            peak_price=peak_price,
            valley_price=valley_price,
        )
        return (
            total_carbon,
            total_electricity_cost_yuan,
            carbon_peak,
            carbon_valley,
            electricity_cost_peak,
            electricity_cost_valley,
        )

    def _machine_tasks(self, snapshot: ExperimentSnapshot) -> dict[str, list[dict[str, Any]]]:
        machine_tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for ws_code, tasks in snapshot.raw_ws_table.items():
            for task in tasks:
                start_time = task.get("start_time")
                end_time = task.get("end_time")
                if start_time is None or end_time is None or start_time >= end_time:
                    continue

                machine_tasks[ws_code].append(
                    {
                        "start_time": int(start_time),
                        "end_time": int(end_time),
                        "duration": int(task.get("duration") or (int(end_time) - int(start_time))),
                        "expr_no": task.get("expr_no"),
                        "step_id": task.get("step_id"),
                        "ws_code_fjspb": task.get("ws_code_fjspb", ws_code),
                        "label": f"{snapshot.label}:{task.get('expr_no')}",
                        "experiment_label": snapshot.label,
                        "snapshot": snapshot,
                        "task": task,
                    }
                )

        return machine_tasks

    def _experiment_tasks(self, snapshot: ExperimentSnapshot) -> dict[str, list[dict[str, Any]]]:
        experiment_tasks: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for expr_no, tasks in snapshot.raw_bottle_table.items():
            row_key = f"{snapshot.label}:{expr_no}"
            for task in tasks:
                start_time = task.get("start_time")
                end_time = task.get("end_time")
                if start_time is None or end_time is None or start_time >= end_time:
                    continue

                experiment_tasks[row_key].append(
                    {
                        "start_time": int(start_time),
                        "end_time": int(end_time),
                        "duration": int(task.get("duration") or (int(end_time) - int(start_time))),
                        "expr_no": expr_no,
                        "step_id": task.get("step_id"),
                        "ws_code_fjspb": task.get("ws_code_fjspb"),
                        "experiment_label": snapshot.label,
                        "snapshot": snapshot,
                        "task": task,
                    }
                )

        return experiment_tasks

    # Prefer experiment names on the y axis; fall back to task names or expr_no.
    def _resolve_experiment_display_label(
        self,
        snapshot: ExperimentSnapshot,
        row_key: str,
        row_tasks: list[dict[str, Any]],
    ) -> str:
        if not row_tasks:
            return row_key

        expr_no = row_tasks[0].get("expr_no")
        if expr_no is None:
            return row_key

        raw_tasks = snapshot.raw_bottle_table.get(expr_no) or snapshot.raw_bottle_table.get(str(expr_no), [])
        for raw_task in raw_tasks:
            expr_name = raw_task.get("expr_name") or raw_task.get("name")
            if expr_name:
                return str(expr_name)

        return str(expr_no)

    # Accept string/list lock_periods and normalize to [(start, end), ...].
    def _parse_lock_periods(self, raw_lock_periods: Any) -> list[tuple[int, int]]:
        if raw_lock_periods in (None, "", "[]"):
            return []

        parsed = raw_lock_periods
        if isinstance(raw_lock_periods, str):
            try:
                parsed = json.loads(raw_lock_periods)
            except json.JSONDecodeError:
                return []

        if not isinstance(parsed, list):
            return []

        lock_periods: list[tuple[int, int]] = []
        for period in parsed:
            if not isinstance(period, (list, tuple)) or len(period) != 2:
                continue

            try:
                lock_start = int(period[0])
                lock_end = int(period[1])
            except (TypeError, ValueError):
                continue

            if lock_start >= sys.maxsize or lock_end <= lock_start:
                continue
            lock_periods.append((lock_start, lock_end))

        return lock_periods

    # Overlay experiment-view lock brackets matching scheduler.plot_schedule().
    def _draw_lock_bracket(
        self,
        axis: Any,
        lock_start: int,
        lock_end: int,
        bar_y: float,
        bar_height: float,
        color_name: str = "FireBrick",
    ) -> None:
        across = LOCK_BRACKET_ARM_LENGTH
        lower_y = bar_y
        upper_y = bar_y + bar_height

        axis.plot([lock_start, lock_start], [lower_y, upper_y], color=color_name, lw=3)
        axis.plot([lock_start, lock_start + across], [lower_y, lower_y], color=color_name, lw=3)
        axis.plot([lock_start, lock_start + across], [upper_y, upper_y], color=color_name, lw=3)

        axis.plot([lock_end, lock_end], [lower_y, upper_y], color=color_name, lw=3)
        axis.plot([lock_end - across, lock_end], [lower_y, lower_y], color=color_name, lw=3)
        axis.plot([lock_end - across, lock_end], [upper_y, upper_y], color=color_name, lw=3)

    # Draw dynamic events only for task insertions; omit lines and labels for done-only tail batches.
    def _draw_timeline_events(
        self,
        axes: list[Any],
        annotation_axis: Any,
        timeline_events: list[dict[str, Any]],
        horizon_end: int,
    ) -> None:
        def format_task_group(title: str, labels: list[str]) -> list[str]:
            if not labels:
                return [f"{title}(0): -"]
            chunks = [", ".join(labels[index : index + 2]) for index in range(0, len(labels), 2)]
            return [f"{title}({len(labels)}): {chunks[0]}", *[f"  {chunk}" for chunk in chunks[1:]]]

        valid_events: list[tuple[int, dict[str, Any]]] = []
        for event in timeline_events:
            added = [str(label) for label in (event.get("added") or [])]
            if not added:
                continue
            try:
                event_time = int(event["time"])
            except (KeyError, TypeError, ValueError):
                continue
            if event_time < 0 or event_time > horizon_end:
                continue
            valid_events.append((event_time, event))

        # Draw insertion timelines first; sort labels by time and layer using actual pixel bounds.
        for event_time, _ in valid_events:
            for axis in axes:
                axis.axvline(
                    x=event_time,
                    color="red",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=20,
                )

        placed_bboxes = []
        for event_time, event in sorted(valid_events, key=lambda item: item[0]):
            completed = [str(label) for label in (event.get("completed") or [])]
            added = [str(label) for label in (event.get("added") or [])]
            annotation_lines = [
                f"t={event_time}",
                *format_task_group("Done", completed),
                *format_task_group("Added", added),
            ]
            if event_time < horizon_end * 0.12:
                horizontal_alignment = "left"
            elif event_time > horizon_end * 0.88:
                horizontal_alignment = "right"
            else:
                horizontal_alignment = "center"
            annotation = annotation_axis.text(
                event_time,
                0.025,
                "\n".join(annotation_lines),
                transform=annotation_axis.get_xaxis_transform(),
                ha=horizontal_alignment,
                va="bottom",
                fontsize=7,
                color="darkred",
                zorder=21,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "red", "alpha": 0.82},
            )

            # Nonoverlapping labels share the lowest layer; move up only on pixel-bound conflicts.
            for lane_index in range(len(valid_events) + 1):
                annotation_axis.figure.canvas.draw()
                renderer = annotation_axis.figure.canvas.get_renderer()
                axis_bbox = annotation_axis.get_window_extent(renderer=renderer)
                annotation_bbox = annotation.get_window_extent(renderer=renderer)
                lane_step = max(0.04, (annotation_bbox.height + 5.0) / max(axis_bbox.height, 1.0))
                annotation.set_position((event_time, 0.025 + lane_index * lane_step))
                annotation_axis.figure.canvas.draw()
                annotation_bbox = annotation.get_window_extent(
                    renderer=annotation_axis.figure.canvas.get_renderer()
                )
                if not any(annotation_bbox.overlaps(placed_bbox) for placed_bbox in placed_bboxes):
                    placed_bboxes.append(annotation_bbox.frozen())
                    break

    def render(
        self,
        snapshot: ExperimentSnapshot,
        output_name: str = "gantt.svg",
        color_by_task: bool = DEFAULT_COLOR_BY_TASK,
        timeline_events: list[dict[str, Any]] | None = None,
        atomic_write: bool = False,
    ) -> Path:
        machine_tasks = self._machine_tasks(snapshot)
        experiment_tasks = self._experiment_tasks(snapshot)
        horizon_end = max(snapshot.makespan, 1)
        show_workstation_plot = SHOW_WORKSTATION_PLOT

        if show_workstation_plot:
            fig, (ax_machine, ax_experiment, ax_tou) = plt.subplots(
                3,
                1,
                figsize=(22, 14),
                sharex=True,
                gridspec_kw={"height_ratios": [2.4, 2.0, 1.0]},
            )
        else:
            fig, (ax_experiment, ax_tou) = plt.subplots(
                2,
                1,
                figsize=(22, 10),
                sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.0]},
            )

        task_height = 0.28

        if show_workstation_plot:
            machine_names = [name for name in sorted(machine_tasks.keys()) if machine_tasks[name]]
            machine_y_start: dict[str, float] = {}
            machine_y_height: dict[str, float] = {}
            current_y = 0.0
            for machine_name in machine_names:
                intervals = [(task["start_time"], task["end_time"]) for task in machine_tasks[machine_name]]
                lane_count = self._max_overlap(intervals)
                machine_y_start[machine_name] = current_y
                machine_y_height[machine_name] = lane_count * task_height
                current_y += machine_y_height[machine_name]

            for machine_name in machine_names:
                lanes_end: list[int] = []
                current_base = machine_y_start[machine_name]
                tasks = sorted(
                    machine_tasks[machine_name], key=lambda item: (item["start_time"], item["end_time"])
                )

                for task in tasks:
                    lane_index = None
                    for idx, lane_end in enumerate(lanes_end):
                        if lane_end <= task["start_time"]:
                            lane_index = idx
                            lanes_end[idx] = task["end_time"]
                            break

                    if lane_index is None:
                        lane_index = len(lanes_end)
                        lanes_end.append(task["end_time"])

                    color_key = task["expr_no"] if color_by_task else task["ws_code_fjspb"]
                    color = self._color_for(color_key)
                    ax_machine.broken_barh(
                        [(task["start_time"], task["duration"])],
                        (current_base + lane_index * task_height, task_height),
                        facecolors=color,
                        edgecolor="none",
                    )

        if show_workstation_plot:
            ax_machine.axvline(x=0, color="tab:blue", linestyle="-", linewidth=1.1, label="Simulation Start")
            ax_machine.axvline(
                x=horizon_end, color="tab:red", linestyle="-", linewidth=1.1, label="Simulation End"
            )

            machine_ticks = [machine_y_start[name] + machine_y_height[name] / 2 for name in machine_names]
            machine_labels = [
                name if name != "manual_workstation" else "muffle_station" for name in machine_names
            ]
            if machine_ticks:
                ax_machine.set_yticks(machine_ticks)


                if SHOW_MACHINE_YTICK_LABELS:
                    ax_machine.set_yticklabels(machine_labels)
            ax_machine.set_ylabel("Machines")
            ax_machine.set_title("Long-Horizon Workstation Perspective")
            ax_machine.set_ylim(0, current_y if current_y > 0 else 1)
            ax_machine.set_xlim(xmin=0, xmax=max(horizon_end, 1))

            machine_legend = [
                mpatches.Patch(color=self._color_for(name), label=name) for name in machine_names[:20]
            ]
            if SHOW_MACHINE_LEGEND and machine_legend:
                ax_machine.legend(
                    handles=machine_legend, loc="upper left", fontsize=9, bbox_to_anchor=(1.01, 1.01)
                )

        # Dynamic snapshot.expr_nos preserves insertion order; arrange experiment rows in that order,
        # avoiding lexical expr_no order placing later tasks first; retain existing ordering without timeline_events.
        non_empty_experiment_rows = [
            row_key for row_key in experiment_tasks if experiment_tasks[row_key]
        ]
        if timeline_events is not None:
            insertion_order = {
                str(expr_no): index for index, expr_no in enumerate(snapshot.expr_nos)
            }
            experiment_rows = sorted(
                non_empty_experiment_rows,
                key=lambda row_key: (
                    insertion_order.get(str(experiment_tasks[row_key][0].get("expr_no")), sys.maxsize),
                    row_key,
                ),
            )
        else:
            experiment_rows = sorted(non_empty_experiment_rows)
        experiment_y_start: dict[str, float] = {}
        experiment_y_height: dict[str, float] = {}
        current_y = 0.0
        for row_key in experiment_rows:
            experiment_y_start[row_key] = current_y
            experiment_y_height[row_key] = task_height
            current_y += task_height

        for row_key in experiment_rows:
            row_tasks = sorted(experiment_tasks[row_key], key=lambda item: (item["start_time"], item["end_time"]))
            base = experiment_y_start[row_key]
            # Share y/height values between task bars and lock brackets to keep them aligned.
            bar_height = task_height * 0.8
            bar_y = base + (task_height - bar_height) / 2.0
            if SHOW_EXPERIMENT_LOCKS and row_tasks:
                expr_no = row_tasks[0].get("expr_no")
                raw_tasks = []
                if expr_no is not None:
                    raw_tasks = snapshot.raw_bottle_table.get(expr_no) or snapshot.raw_bottle_table.get(str(expr_no), [])
                raw_lock_periods = next(
                    (task.get("lock_periods") for task in raw_tasks if task.get("lock_periods") is not None),
                    None,
                )
                for lock_start, lock_end in self._parse_lock_periods(raw_lock_periods):
                    self._draw_lock_bracket(ax_experiment, lock_start, lock_end, bar_y, bar_height)

            for task in row_tasks:
                if task["start_time"] >= task["end_time"]:
                    continue

                ax_experiment.broken_barh(
                    [(task["start_time"], task["duration"])],
                    (bar_y, bar_height),
                    facecolors=self._color_for(
                        str(task["expr_no"]) if color_by_task else str(task["ws_code_fjspb"])
                    ),
                    edgecolor="none",
                )

        experiment_ticks = [
            experiment_y_start[row_key] + experiment_y_height[row_key] / 2 for row_key in experiment_rows
        ]
        if experiment_ticks:
            ax_experiment.set_yticks(experiment_ticks)
            experiment_labels = [
                self._resolve_experiment_display_label(snapshot, row_key, experiment_tasks[row_key])
                for row_key in experiment_rows
            ]


            if SHOW_EXPERIMENT_YTICK_LABELS:
                ax_experiment.set_yticklabels(experiment_labels)
        ax_experiment.set_ylabel("Experiment / Task")
        ax_experiment.set_title("Experiment Perspective")
        ax_experiment.set_ylim(-0.05, current_y + 0.05 if current_y > 0 else 1)
        ax_experiment.set_xlim(xmin=0, xmax=max(horizon_end, 1))
        ax_experiment.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        x_max = max(horizon_end, 1)
        peak_price, valley_price = self._resolve_prices(snapshot)
        (
            total_carbon,
            total_electricity_cost_yuan,
            _,
            _,
            _,
            _,
            price_curve,
            cum_cost_scaled,
        ) = self._calculate_tou_cost(
            snapshot=snapshot,
            machine_tasks=machine_tasks,
            start_time=snapshot.start_time,
            horizon_end=horizon_end,
            peak_price=peak_price,
            valley_price=valley_price,
        )
        self.last_total_carbon = total_carbon

        ax_tou.step(range(x_max + 1), price_curve, where="post", color="tab:orange", linewidth=1.8, label="TOU Price")
        ax_tou.set_ylabel("TOU Price (yuan/kWh)", color="tab:orange")
        ax_tou.tick_params(axis="y", labelcolor="tab:orange")
        ax_tou.grid(axis="x", linestyle="--", alpha=0.3)

        ax_tou_right = ax_tou.twinx()
        ax_tou_right.plot(range(x_max + 1), cum_cost_scaled, color="tab:green", linewidth=2.0, label="Electricity Cost")
        ax_tou_right.set_ylabel("Electricity Cost (yuan)", color="tab:green")
        ax_tou_right.tick_params(axis="y", labelcolor="tab:green")

        ax_tou.axvline(x=0, color="tab:blue", linestyle="-")
        ax_tou.axvline(x=horizon_end, color="tab:red", linestyle="-")
        ax_tou.set_xlabel("Simulation Minutes")
        if total_carbon is not None and total_electricity_cost_yuan is not None:
            title_suffix = (
                f"recomputed objective carbon={total_carbon}, electricity={total_electricity_cost_yuan:.3f} yuan"
            )
            if snapshot.solver_carbon_objective_scaled is not None:
                title_suffix = (
                    f"solver carbon={snapshot.solver_carbon_objective_scaled}, "
                    f"recomputed carbon={total_carbon}, electricity={total_electricity_cost_yuan:.3f} yuan"
                )
            ax_tou.set_title(
                f"TOU Curve & Electricity Cost from {snapshot.start_time:%Y-%m-%d %H:%M:%S}\n"
                f"{title_suffix}"
            )
        else:
            ax_tou.set_title(f"TOU Curve & Electricity Cost from {snapshot.start_time:%Y-%m-%d %H:%M:%S}")

        left_handles, left_labels = ax_tou.get_legend_handles_labels()
        right_handles, right_labels = ax_tou_right.get_legend_handles_labels()
        if SHOW_TOU_LEGEND:
            ax_tou.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left")

        plt.tight_layout()
        if timeline_events:
            event_axes = [ax_experiment, ax_tou]
            if show_workstation_plot:
                event_axes.insert(0, ax_machine)
            self._draw_timeline_events(event_axes, ax_experiment, timeline_events, horizon_end)
        output_path = self.output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            if atomic_write:
                # Write each dynamic redraw to a complete temporary SVG, then atomically replace the open target.
                temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
                output_format = output_path.suffix.lstrip(".") or None
                fig.savefig(temp_path, format=output_format, dpi=180, bbox_inches="tight")
                temp_path.replace(output_path)
            else:
                fig.savefig(output_path, dpi=180, bbox_inches="tight")
            return output_path
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            # Release figures after dynamic redraws to avoid memory growth in long replays.
            plt.close(fig)
