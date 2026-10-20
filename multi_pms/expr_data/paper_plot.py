
"""Generate paper figures from replay metrics.

Fig S1 pairs cp_or and rl_release_or by the exact source-file cohort,
so every reported change compares results within the same scenario.
Dynamic Fig. 3 integrates the energy-share and
trade-off panels, cumulative-cost panels, and the Fig S2 summary row into one
seven-panel publication figure. Dynamic Fig S1 expands the five paired cohorts
behind the Fig. 3 D/E bands, one scenario per row, dynamic Fig S2 shows the
unique-compare scaling curves across scheduling methods and task counts.
Fig S3 compares completed country tariff scenarios
using the Fig. 3e five-cohort curves. Dynamic Fig. 5 compares regional per-task
savings vs simulated Manual (China actual, other regions simulated) above the map.
"""

from dataclasses import dataclass, replace
import json
import json5
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D  # Invisible placeholders split the dynamic Fig. 3 legend into columns.
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator  # Use integer task-count ticks in the dynamic task-count panel.

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
# Ordinary dynamic methods share a summary; unique details retain their default read order.
JSONL_DIR = SCRIPT_DIR / "data" / "jsonl"
DYNAMIC_INPUT_GLOB = "metrics_dynamic*.jsonl"
DEFAULT_DYNAMIC_DETAILS_INPUT = JSONL_DIR / "metrics_dynamic_details.jsonl"
# Unique modes share files; default Fig. 3 inputs include unique details but exclude the unique summary.
UNIQUE_TASK_DYNAMIC_INPUT = JSONL_DIR / "metrics_dynamic_unique_task.jsonl"
UNIQUE_TASK_DYNAMIC_DETAILS_INPUT = JSONL_DIR / "metrics_dynamic_unique_task_details.jsonl"
DEFAULT_DYNAMIC_OUTPUT_DIR = SCRIPT_DIR / "metrics"
FIG5_WORLD_MAP_PATH = ASSETS_DIR / "World_map_blank_without_borders.svg.webp"
FIG5_MAP_REFERENCE_PATH = ASSETS_DIR / "Global_SDL_118_Wikimedia_georeferenced.png"
WORKSTATION_POWER_PATH = SCRIPT_DIR.parent / "dms_link" / "station_power.jsonc"
BASE_STEPS_PATH = SCRIPT_DIR.parent / "dms_link" / "base_steps.json"
FIG4_A_VARIANTS = (
    ("PBAs", 6), ("PET", 2), ("IM", 7),
    ("SMC", 1), ("HEC", 5), ("LDH", 6),
)
# Align Fig. 4a station names with SI; omit liquid-station instance numbers.
FIG4_A_STATION_LABELS = {
    "starting_station": "Sample rack",
    "CAFF": "Liquid",
    "CAAV": "Magnetic stirring",
    "CAFQ": "Standing",
    "CACT": "Centrifugation",
    "CAGM": "Liquid pouring",
    "CADQ": "Electrochemical",
    "CAFK": "Gas-liquid mass-transfer",
    "CACH": "Drying",
    "CAAR": "Ultrasonic disperser station",
    "CAFT": "XRD",
    "CAEY": "Solid dispensing",
    "CAFD": "Furnace",
    "CACS": "Cooling",
    "CAAI": "Dark-box",
    "CACB": "Photocatalysis",
    "CAFV": "Microplate reader",
    "CAGS": "Washing liquid station",
    "CADZ": "Liquid chromatography",
}
# Database instance names distinguish uses and equipment power levels required by SI.
FIG4_A_INSTANCE_LABELS = {
    "物料站-耐热": "Heat-resistant sample rack",
    "四通道液体进样": "Four-channel Liquid",
    "超声液体进样": "Ultrasonic Liquid",
    "加热移瓶": "Heated magnetic stirring",
    "电化学置物": "Electrochemical transfer",
    "马弗炉": "Muffle furnace",
    "箱式炉": "Box furnace",
    "智能催化中转": "Intelligent transfer",
    "中转工作站": "Transfer",
    "回收工作站": "Recycling",
    "光照压膜": "Light-press",
}
# Make the Fig. 4 power colorbar span the figure height; constrained_layout uses the left 90%,
# and an explicit cax positions the full-height bar at the right.
FIG4_LAYOUT_RECT = (0.04, 0.0, 0.86, 1.0)
FIG4_COLORBAR_RECT = (0.925, 0.004, 0.013, 0.992)
# Fig. 4 uses six rows: rows 3 and 5 separate the two b panels and b from c;
# a-b keeps default spacing; remaining row ratios are a:b:b:c = 2.0:3.6:3.6:2.3.
FIG4_FIGSIZE = (14.5, 17.0)
FIG4_ROW_RATIOS = (2.0, 3.6, 0.08, 3.6, 0.08, 2.3)
# With spacer rows, constrained_layout ignores hspace; keep its default.
FIG4_ROW_SPACING = 0.02
FIG5_REFERENCE_PRICE_RATIO = 2.0
# Scale tick and axis-label text only in Fig. 5 Panel A.
FIG5_AXIS_FONT_SCALE = 1.35
FIG5_COUNTRY_COLORS = {
    "China": "#54A24B",
    "United Kingdom": "#BDD7E7",
    "United States": "#6BAED6",
    "Canada": "#2171B5",
    "South Korea": "#08306B",
}
# Native-pixel locations of five case-study cities in the fixed reference map; highlight specific sites only.
FIG5_CASE_STUDY_MARKERS = (
    ("China", (1520.5, 326.5)),
    ("United Kingdom", (888.5, 190.5)),
    ("United States", (263.0, 289.5)),
    ("Canada", (485.5, 252.0)),
    ("South Korea", (1573.5, 297.5)),
)
FIG4_RL_DATABASE_DEFAULT = (
    SCRIPT_DIR.parent
    / "expr_data"
    / "rl_real_expr.sqlite"
)
FIG4_CP_DATABASE_DEFAULT = (
    SCRIPT_DIR.parent
    / "expr_data"
    / "cp_real_expr.sqlite"
)
BASELINE_METHOD = "cp_or"
# The integrated dynamic figure is Fig. 3; supplementary figures are Fig S1/S2/S3.
CP_GREEN_METHOD = "cp_green_or"  # Green CP-SAT comparison method for dynamic Fig. 3.
GREEN_METHOD = "rl_release_or"
HUMAN_METHOD = "human"  # Work-calendar scheduling method for dynamic Fig. 3.
# Available dynamic series include CP/Human controls and parameterized green CP/RL results.
DYNAMIC_FIG2_METHODS = (BASELINE_METHOD, CP_GREEN_METHOD, GREEN_METHOD, HUMAN_METHOD)
# Fig. 3 D/E and Fig S1 aggregate the five paired cohorts used in Fig. 3a/b/c.
DYNAMIC_FIG2_BAND_METHODS = (BASELINE_METHOD, GREEN_METHOD, HUMAN_METHOD)
DYNAMIC_FIG2_BAND_COHORT_COUNT = 5
# Dynamic Fig S2 and Fig. 3f use only the blue, green, and orange compare methods.
UNIQUE_COMPARE_METHODS = (
    CP_GREEN_METHOD,
    GREEN_METHOD,
    "cp_green_limit_or",
)
UNIQUE_COMPARE_TASK_COUNTS = tuple(range(6, 13))
UNIQUE_COMPARE_METHOD_COLORS = {
    CP_GREEN_METHOD: "#4C78A8",
    GREEN_METHOD: "#4F8A5B",
    "cp_green_limit_or": "#F58518",
}
# Fig. 3f and Fig S2 use the same three methods.
DYNAMIC_FIG2_COMPARE_METHODS = UNIQUE_COMPARE_METHODS
FIG2_COMPARE_DISPLAY_LABELS = {
    CP_GREEN_METHOD: "CP (fixed budget)",
    GREEN_METHOD: "RL-initialized CP",
    "cp_green_limit_or": "CP (matched objective)",
}
FIG2_COMPARE_GREEN_DASHES = (2.0, 9.0)
FIG2_COMPARE_GREEN_LINEWIDTH = 2.6
# Publication display names; filtering, JSONL, and internal method keys retain their original names.
METHOD_DISPLAY_LABELS = {
    BASELINE_METHOD: "Sustainability unaware",
    GREEN_METHOD: "Sustainability aware",
    CP_GREEN_METHOD: "Sustainability aware",
    "cp_green_limit_or": "Sustainability aware",
}


def display_method_label(method: str) -> str:
    """Return the publication-facing label without changing the source method key."""
    return METHOD_DISPLAY_LABELS.get(method, method)


# Dynamic Fig. 3 uses 36 tasks.
DYNAMIC_FIG2_TASK_COUNT = 36
# Integrated dynamic Fig. 3 defaults to 36 tasks, v=128, and x12/y1.
DYNAMIC_FIG2_DEFAULT_V = 128.0
DEFAULT_DYNAMIC_FIG2_WINDOW_SIZE = 12
DEFAULT_DYNAMIC_FIG2_ADVANCE_COUNT = 1
# Scale cumulative electricity costs in Fig. 3 D/E only for plotting.
DYNAMIC_FIG2_COST_PLOT_SCALE = 0.3
DEFAULT_MAKESPAN_COST_UNITS = 2.0
# Emphasize the 2:1 peak/off-peak ratio in dynamic Fig. 3c as a common Chinese tariff ratio.
DEFAULT_FIG4_PRIMARY_RATIO = 2.0
# Dynamic Fig. 3 D/E use a 2:1 tariff ratio by default.
DEFAULT_FIG2_PRICE_RATIO = 2.0
# For dynamic Fig. 3, cp_or/human ignore v and use the newest cohort record; green methods show all v values.
EPSILON = 1e-9
# end is a virtual terminal node, not equipment; exclude it from power and facility-load calculations.
VIRTUAL_WORKSTATIONS = frozenset({"end"})

FIG4_V_VALUES = (1.0, 2.0, 4.0, 16.0, 32.0, 64.0, 128.0)
FIG4_TRAJECTORY_ORDER = (128.0, 64.0, 32.0, 16.0, 4.0, 2.0, 1.0)
# Fig. 3c uses sequential color families for tariff ratios and shades within each family for v.
FIG4B_RATIO_CMAPS = ("Purples", "Blues", "Greens", "Oranges", "Reds", "Greys")

# Fig. 3b peak/off-peak bar colors: lower ALPHA is paler, higher ALPHA darker (0-1).
FIG3B_PEAK_COLOR = "#D9828D"
FIG3B_VALLEY_COLOR = "#78A981"
FIG3B_BAR_ALPHA = 0.35

COLORS = {
    "baseline": "#4C78A8",
    "green": "#F58518",
    "human": "#54A24B",  # Human colors distinct from CP/RL.
    "peak": "#D1495B",
    "valley": "#3A7D44",
}
# Shared colors for Fig. 3d/e, Fig S1 cohorts, and Fig S3 legends: CP blue, RL dark green, Human pink from panel b.
DYNAMIC_FIG2_PRIMARY_COLORS = {
    BASELINE_METHOD: "#4C78A8",
    GREEN_METHOD: "#4F8A5B",
    HUMAN_METHOD: FIG3B_PEAK_COLOR,
}
# Only Fig S3 uses peak/mid/off-peak background colors; other figures retain tariff icons.
FIGS3_TARIFF_BACKGROUNDS = {
    "peak_price": (COLORS["peak"], 0.22, "Peak"),
    "flat_price": ("#7B8492", 0.25, "Mid-peak"),
    "valley_price": (COLORS["valley"], 0.22, "Off-peak"),
}
# configure_plot_style() populates all named font sizes centrally.
FONT_SIZES: dict[str, float] = {}
FONT_WEIGHTS: dict[str, str] = {}


@dataclass
class PlotConfig:
    """Fixed publication settings used when generating the complete figure set."""

    dynamic_window_size: int = DEFAULT_DYNAMIC_FIG2_WINDOW_SIZE
    dynamic_advance_count: int = DEFAULT_DYNAMIC_FIG2_ADVANCE_COUNT
    output_dir: Path = DEFAULT_DYNAMIC_OUTPUT_DIR
    makespan_cost_units: float = DYNAMIC_FIG2_DEFAULT_V
    fig2_price_ratio: float = DEFAULT_FIG2_PRICE_RATIO
    power_file: Path = WORKSTATION_POWER_PATH
    fig4_rl_database: Path = FIG4_RL_DATABASE_DEFAULT
    fig4_cp_database: Path = FIG4_CP_DATABASE_DEFAULT
    countries_input: Path = JSONL_DIR / "metircs_dynamic_countries.jsonl"
    dpi: int = 600


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Input jsonl does not exist: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


# Read the unified summary and unique details by default; exclude shared details from summary scans.
def default_dynamic_input_paths() -> list[Path]:
    input_paths = sorted(
        path
        for path in JSONL_DIR.glob(DYNAMIC_INPUT_GLOB)
        if path
        not in {
            DEFAULT_DYNAMIC_DETAILS_INPUT,
            UNIQUE_TASK_DYNAMIC_INPUT,
        }
    )
    if not input_paths:
        raise SystemExit(f"No dynamic summary jsonl files found in {JSONL_DIR}")
    return input_paths


def read_jsonl_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [record for path in paths for record in read_jsonl(path)]


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def scenario_key(
    record: dict[str, Any], expected_task_count: int
) -> tuple[str, ...] | None:
    """Validate the publication task count and construct a scenario key."""
    files = record.get("experimentFileList")
    if not isinstance(files, list) or len(files) != expected_task_count:
        return None
    # Pair records by file membership; this key does not validate dynamic arrival order.
    return tuple(sorted(str(item) for item in files))


def experiment_file_key(record: dict[str, Any]) -> tuple[str, ...] | None:
    """Return the complete file cohort key, including non-Fig. 3 cases."""
    files = record.get("experimentFileList")
    if not isinstance(files, list) or not files:
        return None
    return tuple(sorted(str(item) for item in files))


def is_dynamic_record(record: dict[str, Any]) -> bool:
    """Check whether a metrics record represents dynamic replay."""
    if str(record.get("replayMode") or "").strip().lower() == "dynamic":
        return True

    if str(record.get("fileSubmode") or "").strip().lower() == "dynamic":
        return True
    dynamic_fields = (
        "dynamicWindowSize",
        "dynamicAdvanceCount",
        "dynamicRunId",
        "dynamicBatches",
    )
    return any(record.get(field) is not None for field in dynamic_fields)


def record_dynamic_dimension(record: dict[str, Any], field: str) -> int | None:
    """Read an integer dynamic dimension without conflating missing x/y values."""
    value = as_float(record.get(field))
    if value is None or not value.is_integer():
        return None
    return int(value)


def filter_dynamic_records(
    records: Iterable[dict[str, Any]],
    window_size: int,
    advance_count: int,
) -> list[dict[str, Any]]:
    """Keep dynamic records matching the requested window x/y."""
    materialized = list(records)
    if any(not is_dynamic_record(record) for record in materialized):
        raise SystemExit(
            "Input contains non-dynamic records; only dynamic replay metrics are supported"
        )

    if window_size < 1 or advance_count < 1 or advance_count > window_size:
        raise SystemExit("Dynamic plotting requires x >= 1 and 1 <= y <= x")

    selected: list[dict[str, Any]] = []
    for record in materialized:
        if record_dynamic_dimension(record, "dynamicWindowSize") != window_size:
            continue
        if record_dynamic_dimension(record, "dynamicAdvanceCount") != advance_count:
            continue
        selected.append(record)
    return selected


def filter_dynamic_fig2_task_count(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep dynamic Fig. 3 records with the required total task count."""
    selected: list[dict[str, Any]] = []
    for record in records:
        task_count = as_float(record.get("totalTaskCount"))
        if task_count is None:
            task_count = as_float(record.get("filesPerExperiment"))
        if task_count is None:
            files = record.get("experimentFileList")
            task_count = float(len(files)) if isinstance(files, list) else None
        if (
            task_count is not None
            and task_count.is_integer()
            and int(task_count) == DYNAMIC_FIG2_TASK_COUNT
        ):
            selected.append(record)
    return selected


def price_ratio(record: dict[str, Any]) -> float | None:
    peak = as_float(record.get("peakPrice"))
    valley = as_float(record.get("valleyPrice"))
    if peak is None or valley is None or peak <= 0 or valley <= 0:
        return None
    return peak / valley


def normalized_ratio(value: float) -> float:
    return round(value, 8)


def same_number(left: Any, right: float) -> bool:
    value = as_float(left)
    return value is not None and math.isclose(value, right, rel_tol=1e-8, abs_tol=1e-8)


def select_records(
    records: Iterable[dict[str, Any]], makespan_cost_units: float
) -> dict[tuple[tuple[str, ...], str, float], dict[str, Any]]:
    selected: dict[tuple[tuple[str, ...], str, float], dict[str, Any]] = {}
    for record in records:
        method = str(record.get("scheduleMethod") or record.get("configuredScheduleMethod") or "")
        if method not in {BASELINE_METHOD, GREEN_METHOD}:
            continue
        if record.get("schedule_status") not in (None, "success"):
            continue
        key = scenario_key(record, DYNAMIC_FIG2_TASK_COUNT)
        ratio = price_ratio(record)
        if key is None or ratio is None:
            continue
        if method == GREEN_METHOD and not same_number(
            record.get("makespan_60min_cost_in_valley_price_units"), makespan_cost_units
        ):
            continue

        # If a resumed replay duplicated a combination, the last jsonl row is authoritative.
        selected[(key, method, normalized_ratio(ratio))] = record
    return selected


def energy_components(record: dict[str, Any]) -> tuple[float, float] | None:
    peak_price = as_float(record.get("peakPrice"))
    valley_price = as_float(record.get("valleyPrice"))
    peak_cost = as_float(record.get("totalElectricityCostYuanPeak"))
    valley_cost = as_float(record.get("totalElectricityCostYuanValley"))
    if (
        peak_price is None
        or valley_price is None
        or peak_cost is None
        or valley_cost is None
        or peak_price <= 0
        or valley_price <= 0
    ):
        return None

    peak_energy = peak_cost / peak_price
    valley_energy = valley_cost / valley_price
    # Compute energy for the project's two-tier peak/off-peak tariff.
    return peak_energy, valley_energy


def paired_rows(
    selected: dict[tuple[tuple[str, ...], str, float], dict[str, Any]]
) -> dict[float, list[dict[str, Any]]]:
    ratios = sorted({item[2] for item in selected})
    pairs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    scenario_keys = sorted({item[0] for item in selected})
    for ratio in ratios:
        for key in scenario_keys:
            baseline = selected.get((key, BASELINE_METHOD, ratio))
            green = selected.get((key, GREEN_METHOD, ratio))
            if baseline is None or green is None:
                continue
            baseline_cost = as_float(baseline.get("totalElectricityCostYuan"))
            green_cost = as_float(green.get("totalElectricityCostYuan"))
            baseline_makespan = as_float(baseline.get("makespan"))
            green_makespan = as_float(green.get("makespan"))
            baseline_energy = energy_components(baseline)
            green_energy = energy_components(green)
            if (
                baseline_cost is None
                or green_cost is None
                or baseline_makespan is None
                or green_makespan is None
                or baseline_cost <= 0
                or baseline_makespan <= 0
                or baseline_energy is None
                or green_energy is None
            ):
                continue

            baseline_total_energy = sum(baseline_energy)
            green_total_energy = sum(green_energy)
            if baseline_total_energy <= 0 or green_total_energy <= 0:
                continue
            baseline_shares = np.asarray(baseline_energy) / baseline_total_energy * 100.0
            green_shares = np.asarray(green_energy) / green_total_energy * 100.0
            pairs[ratio].append(
                {
                    "scenario_key": key,
                    "baseline": baseline,
                    "green": green,
                    "cost_reduction": (baseline_cost - green_cost) / baseline_cost * 100.0,
                    "makespan_penalty": (green_makespan - baseline_makespan) / baseline_makespan * 100.0,
                    "baseline_energy": baseline_total_energy,
                    "baseline_makespan": baseline_makespan,
                    "baseline_shares": baseline_shares,
                    "green_shares": green_shares,
                    "peak_share_change": baseline_shares[0] - green_shares[0],
                }
            )
    return dict(pairs)


def parameter_value(record: dict[str, Any]) -> float | None:
    """Read and normalize the soft objective parameter v for Fig. 3c."""
    value = as_float(record.get("makespan_60min_cost_in_valley_price_units"))
    if value is None:
        return None
    for expected in FIG4_V_VALUES:
        if math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-8):
            return expected
    return None


def select_parameter_scan_records(
    records: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[tuple[str, ...], float], dict[str, Any]],
    dict[tuple[tuple[str, ...], float, float], dict[str, Any]],
]:
    """Index CP baselines and green results at each v separately for Fig. 3c."""
    baselines: dict[tuple[tuple[str, ...], float], dict[str, Any]] = {}
    green_records: dict[tuple[tuple[str, ...], float, float], dict[str, Any]] = {}
    for record in records:
        method = str(record.get("scheduleMethod") or record.get("configuredScheduleMethod") or "")
        if method not in {BASELINE_METHOD, GREEN_METHOD}:
            continue
        if record.get("schedule_status") not in (None, "success"):
            continue
        # Restrict all dynamic Fig. 3c v scans to 36-task cohorts.
        key = scenario_key(record, DYNAMIC_FIG2_TASK_COUNT)
        ratio = price_ratio(record)
        if key is None or ratio is None:
            continue
        normalized_price_ratio = normalized_ratio(ratio)
        if method == BASELINE_METHOD:
            # For duplicate resumed records at the same scenario and tariff ratio, use the last successful record.
            baselines[(key, normalized_price_ratio)] = record
            continue
        v = parameter_value(record)
        if v is not None:
            green_records[(key, normalized_price_ratio, v)] = record
    return baselines, green_records


def build_parameter_scan_rows(
    baselines: dict[tuple[tuple[str, ...], float], dict[str, Any]],
    green_records: dict[tuple[tuple[str, ...], float, float], dict[str, Any]],
) -> dict[float, dict[float, list[dict[str, Any]]]]:
    """Convert green results to bivariate trade-off coordinates relative to matching CP baselines."""
    rows: dict[float, dict[float, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (key, ratio, v), green in sorted(green_records.items()):
        baseline = baselines.get((key, ratio))
        if baseline is None:
            continue
        baseline_cost = as_float(baseline.get("totalElectricityCostYuan"))
        green_cost = as_float(green.get("totalElectricityCostYuan"))
        baseline_makespan = as_float(baseline.get("makespan"))
        green_makespan = as_float(green.get("makespan"))
        if (
            baseline_cost is None
            or green_cost is None
            or baseline_makespan is None
            or green_makespan is None
            or baseline_cost <= 0
            or baseline_makespan <= 0
        ):
            continue
        rows[ratio][v].append(
            {
                "scenario_key": key,
                "ratio": ratio,
                "v": v,
                "baseline": baseline,
                "green": green,
                "cost_reduction": (baseline_cost - green_cost) / baseline_cost * 100.0,
                "makespan_penalty": (
                    (green_makespan - baseline_makespan) / baseline_makespan * 100.0
                ),
            }
        )
    return {
        ratio: {v: value for v, value in values_by_v.items()}
        for ratio, values_by_v in rows.items()
    }


def ratio_label(ratio: float) -> str:
    value = str(int(round(ratio))) if math.isclose(ratio, round(ratio)) else f"{ratio:g}"
    return f"{value}:1"


def style_axis(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def draw_distribution_boxplot(
    axis: Any,
    values_by_ratio: list[list[float]],
    ratios: list[float],
    ylabel: str,
    panel_label: str,
    show_raw_points: bool = False,
) -> None:
    # Dynamic Fig. 3a uses light-green boxes, black lines, and orange raw points.
    box_edge_color = "#4F8F80" if show_raw_points else "#4C78A8"
    whisker_color = "#202020" if show_raw_points else "#4C78A8"
    axis.boxplot(
        values_by_ratio,
        positions=np.arange(len(ratios)),
        widths=0.55,
        patch_artist=True,
        showfliers=not show_raw_points,
        medianprops={"color": "#202020", "linewidth": 1.8},
        boxprops={
            "facecolor": "#A9DCCF" if show_raw_points else "#A9C5E8",
            "edgecolor": box_edge_color,
            "alpha": 0.78 if show_raw_points else 1.0,
        },
        whiskerprops={"color": whisker_color},
        capprops={"color": whisker_color},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.45, "markeredgecolor": "#4C78A8"},
    )
    if show_raw_points:
        # Seed horizontal jitter for reproducible points in dynamic Fig. 3a.
        rng = np.random.default_rng(20260805)
        for position, values in enumerate(values_by_ratio):
            jitter = rng.uniform(-0.085, 0.085, size=len(values))
            axis.scatter(
                position + jitter,
                values,
                s=26,
                color="#F58518",
                alpha=0.78,
                edgecolors="#E27612",
                linewidths=0.45,
                zorder=3,
            )
    axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    axis.set_xticks(np.arange(len(ratios)), [ratio_label(value) for value in ratios])
    axis.set_xlabel("Peak-to-off-peak price ratio")
    axis.set_ylabel(ylabel)
    axis.text(-0.12, 1.05, panel_label, transform=axis.transAxes, fontsize=FONT_SIZES["panel_label"], fontweight=FONT_WEIGHTS["panel_label"])
    style_axis(axis)


def draw_energy_shares(
    axis: Any,
    pairs: dict[float, list[dict[str, Any]]],
    ratios: list[float],
    panel_label: str = "",
) -> None:
    width = 0.38
    x_values = np.arange(len(ratios), dtype=float)
    method_offsets = ((-(width + 0.04) / 2, "baseline_shares", BASELINE_METHOD), ((width + 0.04) / 2, "green_shares", GREEN_METHOD))
    # Energy-share bars use muted colors darker than the translucent tariff backgrounds.
    share_colors = [FIG3B_PEAK_COLOR, FIG3B_VALLEY_COLOR]
    for offset, field, _ in method_offsets:
        medians = np.asarray(
            [np.median([row[field] for row in pairs[ratio]], axis=0) for ratio in ratios]
        )
        # Renormalize component medians of peak/off-peak shares to 100% to account for numeric error.
        medians = medians / np.sum(medians, axis=1, keepdims=True) * 100.0
        bottoms = np.zeros(len(ratios))
        for component_index, color in enumerate(share_colors):
            heights = medians[:, component_index]
            axis.bar(
                x_values + offset,
                heights,
                width=width,
                bottom=bottoms,
                color=color,
                alpha=FIG3B_BAR_ALPHA,
                edgecolor="white",
                linewidth=0.4,
            )
            # Label energy shares inside each peak/off-peak bar segment.
            for x_value, height, bottom in zip(x_values + offset, heights, bottoms):
                axis.text(
                    x_value,
                    bottom + height / 2.0,
                    f"{height:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZES["small"] * 1.2,
                    color="#303030",
                    zorder=3,
                )
            bottoms += heights

    axis.set_xticks(x_values)
    axis.set_xticklabels([ratio_label(value) for value in ratios])
    axis.set_ylim(0, 112)
    axis.set_xlabel("Peak-to-off-peak price ratio")
    axis.set_ylabel("Energy share (%)")
    # The integrated dynamic energy-share panel is Fig. 3b.
    if panel_label:
        axis.text(
            -0.12,
            1.05,
            panel_label,
            transform=axis.transAxes,
            fontsize=FONT_SIZES["panel_label"],
            fontweight=FONT_WEIGHTS["panel_label"],
        )
    axis.legend(
        handles=[
            Patch(facecolor=color, alpha=FIG3B_BAR_ALPHA, label=label)
            for color, label in zip(share_colors, ("Peak", "Off-peak"))
        ],
        # Reserve space above bars for a legend separate from the title and 100% bar tops.
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=2,
        frameon=False,
        fontsize=FONT_SIZES["small"] * 1.15,
    )
    style_axis(axis)


def record_identity(
    record: dict[str, Any],
) -> tuple[
    tuple[str, ...] | None,
    str,
    float | None,
    float | None,
    int | None,
    int | None,
]:
    """Include the dynamic x/y policy in summary/detail identity."""
    method = str(record.get("scheduleMethod") or record.get("configuredScheduleMethod") or "")
    ratio = price_ratio(record)
    weight = as_float(record.get("makespan_60min_cost_in_valley_price_units"))
    return (
        experiment_file_key(record),
        method,
        normalized_ratio(ratio) if ratio is not None else None,
        weight,
        record_dynamic_dimension(record, "dynamicWindowSize"),
        record_dynamic_dimension(record, "dynamicAdvanceCount"),
    )


def details_lookup(records: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if isinstance(record.get("ws_table"), dict):
            lookup[record_identity(record)] = record
    return lookup


def resolve_detailed_record(
    summary: dict[str, Any], lookup: dict[tuple[Any, ...], dict[str, Any]]
) -> dict[str, Any] | None:
    if isinstance(summary.get("ws_table"), dict):
        return summary
    exact = lookup.get(record_identity(summary))
    if exact is not None:
        return exact
    identity = record_identity(summary)
    # Baselines may omit the green objective weight, so permit a unique match without it.
    candidates = [
        value
        for key, value in lookup.items()
        if key[:3] == identity[:3] and key[4:] == identity[4:]
    ]
    return candidates[0] if len(candidates) == 1 else None


def load_workstation_powers(power_path: Path) -> dict[str, float]:
    """Read workstation-code to power (W) mappings from JSONC."""
    if not power_path.is_file():
        return {}
    payload = json5.loads(power_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workstation powers must be a JSONC object: {power_path}")
    powers: dict[str, float] = {}
    for ws_code, power in payload.items():
        if (not isinstance(ws_code, str) or not ws_code.strip()
                or isinstance(power, bool) or not isinstance(power, (int, float))
                or not math.isfinite(power) or power < 0):
            raise ValueError(f"invalid workstation power for {ws_code!r}: {power!r}")
        powers[ws_code] = float(power)
    return powers


def record_powers(record: dict[str, Any], power_path: Path) -> dict[str, float]:
    ws_table = record.get("ws_table")
    if not isinstance(ws_table, dict):
        return {}
    return load_workstation_powers(power_path)


def load_curve(record: dict[str, Any], power_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    ws_table = record.get("ws_table")
    if not isinstance(ws_table, dict):
        raise ValueError("ws_table is missing")
    makespan = int(as_float(record.get("makespan")) or 0)
    if makespan <= 0:
        raise ValueError("makespan is missing")
    if not power_path.is_file():
        raise ValueError(f"workstation power file is missing: {power_path}")
    powers = record_powers(record, power_path)
    missing = sorted(
        code for code in ws_table if code not in VIRTUAL_WORKSTATIONS and code not in powers
    )
    if missing:
        raise ValueError(
            f"power is missing for {len(missing)} workstation(s) in {power_path}: "
            f"{', '.join(missing[:4])}"
        )

    delta = np.zeros(makespan + 1, dtype=float)
    intervals: set[tuple[str, int, int]] = set()
    for ws_code, tasks in ws_table.items():
        if ws_code in VIRTUAL_WORKSTATIONS:
            continue
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            start = max(0, min(makespan, int(as_float(task.get("start_time")) or 0)))
            end = max(0, min(makespan, int(as_float(task.get("end_time")) or 0)))
            if end > start:
                intervals.add((str(ws_code), start, end))
    # One powered workstation is counted once per identical batch interval.
    for ws_code, start, end in intervals:
        power_kw = powers.get(ws_code, 0.0) / 1000.0
        delta[start] += power_kw
        delta[end] -= power_kw
    curve = np.cumsum(delta[:-1])
    return np.arange(makespan, dtype=float) / 60.0, curve, makespan


def shade_tariff_periods(axis: Any, record: dict[str, Any], start: int, end: int) -> None:
    raw_start_time = record.get("simulationStartTime")
    try:
        schedule_start = datetime.fromisoformat(str(raw_start_time))
    except (TypeError, ValueError):
        schedule_start = datetime(2000, 1, 1, 0, 0)

    # Shade mutually exclusive tariff intervals to avoid blending pink and green backgrounds.
    origin_minute = schedule_start.hour * 60 + schedule_start.minute
    valley_flags = [
        ((origin_minute + minute) % 1440 >= 22 * 60)
        or ((origin_minute + minute) % 1440 < 8 * 60)
        for minute in range(start, end)
    ]
    if not valley_flags:
        return
    segment_start = start
    segment_is_valley = valley_flags[0]
    for offset, is_valley in enumerate(valley_flags[1:], start=1):
        if is_valley != segment_is_valley:
            segment_end = start + offset
            axis.axvspan(
                segment_start / 60.0,
                segment_end / 60.0,
                color=COLORS["valley"] if segment_is_valley else COLORS["peak"],
                alpha=0.08 if segment_is_valley else 0.055,
                linewidth=0,
                zorder=0,
            )
            segment_start = segment_end
            segment_is_valley = is_valley
    axis.axvspan(
        segment_start / 60.0,
        end / 60.0,
        color=COLORS["valley"] if segment_is_valley else COLORS["peak"],
        alpha=0.08 if segment_is_valley else 0.055,
        linewidth=0,
        zorder=0,
    )


def tariff_keys_per_minute(record: dict[str, Any], start: int, end: int) -> np.ndarray:

    raw_start = record.get("simulationStartTime")
    try:
        origin = datetime.fromisoformat(str(raw_start))
    except (TypeError, ValueError):
        origin = datetime(2000, 1, 1)
    phase = (origin.hour * 60 + origin.minute + np.arange(start, end)) % 1440
    periods = (record.get("tariffInfo") or {}).get("tariff_periods")
    if periods is None:
        return np.where((phase >= 22 * 60) | (phase < 8 * 60),
                        "valley_price", "peak_price")

    day = np.full(1440, "", dtype="U12")
    coverage = np.zeros(1440, dtype=int)
    for period in periods:
        def clock_minutes(value: str) -> int:
            hour, minute = map(int, value.split(":"))
            if not (0 <= hour <= 24 and 0 <= minute < 60) or (hour == 24 and minute):
                raise ValueError(f"Invalid tariff clock: {value}")
            return hour * 60 + minute
        left, right = clock_minutes(period["start"]), clock_minutes(period["end"])
        key = period["price_key"]
        if key not in {"peak_price", "valley_price", "flat_price"} or left == right:
            raise ValueError(f"Invalid tariff period: {period}")
        intervals = [(left, right)] if left < right else [(left, 1440), (0, right)]
        for lo, hi in intervals:
            day[lo:hi] = key
            coverage[lo:hi] += 1
    if not np.all(coverage == 1):
        raise ValueError("tariff_periods must cover each minute of the day exactly once")
    return day[phase]


def tariff_price_per_minute(record: dict[str, Any], start: int, end: int) -> np.ndarray:
    """Return replay prices, including country-specific peak/flat/valley periods."""
    peak_price = as_float(record.get("peakPrice"))
    valley_price = as_float(record.get("valleyPrice"))
    if peak_price is None or valley_price is None or peak_price <= 0 or valley_price <= 0:
        raise ValueError("peakPrice and valleyPrice are required for cumulative cost")
    keys = tariff_keys_per_minute(record, start, end)
    prices = {**(record.get("tariffInfo") or {}),
              "peak_price": peak_price, "valley_price": valley_price}
    result = np.empty(len(keys), dtype=float)
    for key in np.unique(keys):
        price = as_float(prices.get(str(key)))
        if price is None or not math.isfinite(price) or price < 0:
            raise ValueError(f"Missing or invalid tariff price: {key}")
        result[keys == key] = price
    return result


def shade_figs3_tariff_periods(axis: Any, record: dict[str, Any], horizon: int) -> None:
    """Shade mutually exclusive peak/mid/off-peak periods using country replay clocks, without day/night icons."""
    keys = tariff_keys_per_minute(record, 0, horizon)
    if not len(keys):
        return
    changes = (np.flatnonzero(keys[1:] != keys[:-1]) + 1).tolist()
    boundaries = [0, *changes, horizon]
    for left, right in zip(boundaries, boundaries[1:]):
        color, alpha, _ = FIGS3_TARIFF_BACKGROUNDS[str(keys[left])]
        axis.axvspan(left / 60.0, right / 60.0, color=color, alpha=alpha,
                     linewidth=0, zorder=0)


def cumulative_electricity_cost(
    record: dict[str, Any], load_kw: np.ndarray, start: int = 0
) -> np.ndarray:
    """Integrate final-snapshot facility load into cumulative yuan cost."""
    prices = tariff_price_per_minute(record, start, start + len(load_kw))
    # load is kW and each sample spans one minute, hence price/60 gives yuan/minute.
    return np.cumsum(load_kw * prices / 60.0)


def fig4b_ratio_v_colors(ratios: list[float]) -> dict[float, dict[float, Any]]:
    """Assign point colors by tariff ratio, using the same color for different v values within a group."""
    ratio_colors: dict[float, dict[float, Any]] = {}
    for index, ratio in enumerate(ratios):
        cmap = plt.get_cmap(FIG4B_RATIO_CMAPS[index % len(FIG4B_RATIO_CMAPS)])
        color = cmap(0.52)
        ratio_colors[ratio] = {v: color for v in FIG4_TRAJECTORY_ORDER}
    return ratio_colors


def median_summary(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Compute the currently displayed bivariate median points for Fig. 3c."""
    x_values = np.asarray([row["makespan_penalty"] for row in rows], dtype=float)
    y_values = np.asarray([row["cost_reduction"] for row in rows], dtype=float)
    return float(np.median(x_values)), float(np.median(y_values))


def fitted_tradeoff_curve(
    trajectory: list[tuple[float, float]],
    sample_count: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot shape-preserving or monotone saturating Fig. 3c trends against time penalties."""
    points = np.asarray(trajectory, dtype=float)
    if len(points) < 2:
        return points[:, 0], points[:, 1]

    # Merge duplicate x values; quadratic regression can create artificial bumps between sparse nonmonotone points.
    unique_x, inverse_indices = np.unique(points[:, 0], return_inverse=True)
    if len(unique_x) < 2:
        return unique_x, np.asarray([float(np.mean(points[:, 1]))])
    unique_y = np.asarray(
        [
            float(np.mean(points[inverse_indices == index, 1]))
            for index in range(len(unique_x))
        ]
    )
    fitted_x = np.linspace(float(unique_x[0]), float(unique_x[-1]), sample_count)
    if len(unique_x) == 2:
        return fitted_x, np.interp(fitted_x, unique_x, unique_y)

    shifted_x = unique_x - unique_x[0]
    fitted_shifted_x = fitted_x - unique_x[0]
    x_span = float(shifted_x[-1])
    best_fit: tuple[float, float, float] | None = None
    best_error = math.inf
    for rate in np.geomspace(0.08 / x_span, 12.0 / x_span, 240):
        basis = -np.expm1(-rate * shifted_x)
        design = np.column_stack((np.ones_like(basis), basis))
        intercept, amplitude = np.linalg.lstsq(design, unique_y, rcond=None)[0]
        if amplitude <= 0.0:
            continue
        residuals = unique_y - (intercept + amplitude * basis)
        error = float(np.dot(residuals, residuals))
        if error < best_error:
            best_error = error
            best_fit = (float(intercept), float(amplitude), float(rate))
    if best_fit is not None:
        intercept, amplitude, rate = best_fit
        fitted_y = intercept - amplitude * np.expm1(-rate * fitted_shifted_x)
        return fitted_x, fitted_y

    # Use Fritsch-Carlson shape-preserving cubic Hermite interpolation to avoid overshoot.
    intervals = np.diff(unique_x)
    secants = np.diff(unique_y) / intervals
    slopes = np.zeros(len(unique_x), dtype=float)
    for index in range(1, len(unique_x) - 1):
        left_slope = secants[index - 1]
        right_slope = secants[index]
        if left_slope * right_slope > 0.0:
            interval_sum = intervals[index] + intervals[index - 1]
            weight_left = (2.0 * intervals[index] + intervals[index - 1]) / interval_sum
            weight_right = (intervals[index] + 2.0 * intervals[index - 1]) / interval_sum
            slopes[index] = (weight_left + weight_right) / (
                weight_left / left_slope + weight_right / right_slope
            )
    slopes[0] = (
        (2.0 * intervals[0] + intervals[1]) * secants[0]
        - intervals[0] * secants[1]
    ) / (intervals[0] + intervals[1])
    slopes[-1] = (
        (2.0 * intervals[-1] + intervals[-2]) * secants[-1]
        - intervals[-1] * secants[-2]
    ) / (intervals[-1] + intervals[-2])
    # Limit endpoint slopes using adjacent intervals to prevent reversal or overshoot.
    endpoint_secants = ((0, secants[0]), (len(slopes) - 1, secants[-1]))
    for index, adjacent_secant in endpoint_secants:
        if slopes[index] * adjacent_secant <= 0.0:
            slopes[index] = 0.0
        elif abs(slopes[index]) > 3.0 * abs(adjacent_secant):
            slopes[index] = 3.0 * adjacent_secant

    segment_indices = np.minimum(
        np.searchsorted(unique_x, fitted_x, side="right") - 1,
        len(intervals) - 1,
    )
    local_x = fitted_x - unique_x[segment_indices]
    normalized_x = local_x / intervals[segment_indices]
    y0 = unique_y[segment_indices]
    y1 = unique_y[segment_indices + 1]
    h00 = 2.0 * normalized_x**3 - 3.0 * normalized_x**2 + 1.0
    h10 = normalized_x**3 - 2.0 * normalized_x**2 + normalized_x
    h01 = -2.0 * normalized_x**3 + 3.0 * normalized_x**2
    h11 = normalized_x**3 - normalized_x**2
    fitted_y = h00 * y0 + h10 * intervals[segment_indices] * slopes[segment_indices]
    fitted_y += (
        h01 * y1
        + h11 * intervals[segment_indices] * slopes[segment_indices + 1]
    )
    return fitted_x, fitted_y


def draw_fig4b(
    axis: Any,
    scan_rows: dict[float, dict[float, list[dict[str, Any]]]],
    panel_label: str = "b",
) -> None:
    """Aggregate scenarios by tariff ratio and v, plotting bivariate medians."""
    ratios = sorted(scan_rows)
    ratio_v_colors = fig4b_ratio_v_colors(ratios)

    for ratio_index, ratio in enumerate(ratios):
        # Omit all 1:1 points, error bars, and fitted paths from dynamic Fig. 3c.
        if same_number(ratio, 1.0):
            continue
        v_colors = ratio_v_colors[ratio]
        # Lines are slightly paler than points at the same ratio; line color does not depend on v.
        line_color = plt.get_cmap(
            FIG4B_RATIO_CMAPS[ratio_index % len(FIG4B_RATIO_CMAPS)]
        )(0.44)
        trajectory: list[tuple[float, float]] = []
        for v in FIG4_TRAJECTORY_ORDER:
            rows = scan_rows.get(ratio, {}).get(v, [])
            if not rows:
                continue
            color = v_colors[v]
            x_median, y_median = median_summary(rows)
            trajectory.append((x_median, y_median))
            is_default = same_number(v, DEFAULT_MAKESPAN_COST_UNITS)
            # Use uniform point sizes across v values in dynamic Fig. 3c.
            # Central points represent bivariate medians.
            axis.plot(
                x_median,
                y_median,
                marker="o",
                linestyle="None",
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=4 if is_default else 3,
            )
        if len(trajectory) >= 2:
            # Summarize each tariff ratio's trend with a low-order fit in its color family.
            fitted_x, fitted_y = fitted_tradeoff_curve(
                trajectory,
            )
            axis.plot(
                fitted_x,
                fitted_y,
                color=line_color,
                # Thicken the 2:1 response curve in dynamic Fig. 3c.
                linewidth=2.8 if same_number(ratio, DEFAULT_FIG4_PRIMARY_RATIO) else 1.6,
                alpha=1.0,
                zorder=2,
            )

        # Keep one legend handle per tariff ratio, using the same circular markers as Fig. 3c.
        axis.plot(
            [],
            [],
            color=line_color,
            marker="o",
            markerfacecolor=v_colors[DEFAULT_MAKESPAN_COST_UNITS],
            markeredgecolor=v_colors[DEFAULT_MAKESPAN_COST_UNITS],
            linewidth=2.4 if same_number(ratio, DEFAULT_FIG4_PRIMARY_RATIO) else 1.2,
            label=ratio_label(ratio),
        )

    # Ordinary cp_or is insensitive to tariff ratio and v; draw one set of baseline points.
    axis.scatter(
        0,
        0,
        marker="*",
        s=210,
        facecolor="#202020",
        edgecolor="white",
        linewidth=1.1,
        zorder=10,
        clip_on=False,
        label=display_method_label(BASELINE_METHOD),
    )

    # Place the tariff-ratio legend inside the lower-right part of Fig. 3c.
    axis.legend(
        title="Price ratio",
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        fontsize=FONT_SIZES["small"],
        ncols=2,
        loc="lower right",
        # Lower the Fig. 3c legend slightly to avoid covering the rightmost blue point.
        bbox_to_anchor=(0.98, 0.30),
        borderaxespad=0,
    )
    if not ratios:
        axis.text(
            0.5,
            0.5,
            "All-scenario parameter scan is not available yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Completion-time increase (%)")
    axis.set_ylabel("Electricity-cost reduction (%)")
    axis.set_title("All-scenario parameter response")
    # The caller labels this parameter-response panel dynamic Fig. 3c.
    if panel_label:
        axis.text(
            -0.14,
            1.05,
            panel_label,
            transform=axis.transAxes,
            fontsize=FONT_SIZES["panel_label"],
            fontweight=FONT_WEIGHTS["panel_label"],
        )
    style_axis(axis)
    # Hide horizontal grids in dynamic Fig. 3c but keep the dashed zero reference.
    axis.grid(False, axis="y")


def draw_fig2_placeholder(axis: Any, panel_label: str, title: str, message: str) -> None:
    """Reserve blank placeholders for dynamic Fig. 3 panels without available plots."""
    axis.text(-0.08, 1.04, panel_label, transform=axis.transAxes, fontsize=FONT_SIZES["panel_label"], fontweight=FONT_WEIGHTS["panel_label"])
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#777777",
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D0D0")


def _dynamic_method_candidates(
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    method_name: str,
    *,
    require_task_metrics: bool,
    required_v: float | None = None,
) -> list[dict[str, Any]]:
    """Collect candidate timeline or task-count records for a dynamic Fig. 3 method."""
    candidates: list[dict[str, Any]] = []
    materialized = list(records)
    for record in materialized:
        method = str(record.get("scheduleMethod") or record.get("configuredScheduleMethod") or "")
        if method != method_name or not is_dynamic_record(record):
            continue
        if record.get("schedule_status") not in (None, "success"):
            continue
        detailed = resolve_detailed_record(record, details)
        candidate = detailed if isinstance(detailed, dict) else record
        # Filter v only when explicitly requested; otherwise retain all parameter records.
        if required_v is not None:
            candidate_v = candidate.get("makespan_60min_cost_in_valley_price_units")
            if candidate_v is None:
                candidate_v = record.get("makespan_60min_cost_in_valley_price_units")
            if not same_number(candidate_v, required_v):
                continue
        if require_task_metrics:
            valid = isinstance(candidate.get("dynamicTaskMetrics"), list) and bool(
                candidate.get("dynamicTaskMetrics")
            )
        else:
            valid = isinstance(candidate.get("dynamicBatches"), list) and isinstance(
                candidate.get("ws_table"), dict
            )
        if valid:
            candidates.append(candidate)

    # Allow details-only input; construct candidates from details when summaries are absent.
    if not candidates:
        for record in details.values():
            method = str(record.get("scheduleMethod") or record.get("configuredScheduleMethod") or "")
            if (
                method != method_name
                or not is_dynamic_record(record)
                or record.get("schedule_status") not in (None, "success")
            ):
                continue
            if required_v is not None and not same_number(
                record.get("makespan_60min_cost_in_valley_price_units"),
                required_v,
            ):
                continue
            if require_task_metrics:
                valid = isinstance(record.get("dynamicTaskMetrics"), list) and bool(
                    record.get("dynamicTaskMetrics")
                )
            else:
                valid = isinstance(record.get("dynamicBatches"), list) and isinstance(
                    record.get("ws_table"), dict
                )
            if valid:
                candidates.append(record)
    return candidates


def dynamic_fig2_task_count(record: dict[str, Any]) -> int | None:
    """Read the total task count of a dynamic Fig. 3 cohort."""
    value = as_float(record.get("totalTaskCount"))
    if value is None:
        value = as_float(record.get("filesPerExperiment"))
    if value is None and isinstance(record.get("experimentFileList"), list):
        value = float(len(record["experimentFileList"]))
    return int(value) if value is not None and value.is_integer() else None


def dynamic_fig2_cohort_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    """Build a strict pairing key from files, window, tariff ratio, and task count."""
    files = experiment_file_key(record)
    window_size = record_dynamic_dimension(record, "dynamicWindowSize")
    advance_count = record_dynamic_dimension(record, "dynamicAdvanceCount")
    ratio = price_ratio(record)
    task_count = dynamic_fig2_task_count(record)
    if (
        files is None
        or window_size is None
        or advance_count is None
        or ratio is None
        or task_count is None
    ):
        return None
    return (
        files,
        window_size,
        advance_count,
        normalized_ratio(ratio),
        task_count,
    )


def dynamic_fig2_cohort_label(cohort_key: tuple[Any, ...] | None) -> str:
    """Format a compact dynamic Fig. 3 cohort key for logging."""
    if cohort_key is None:
        return "<none>"
    files, window_size, advance_count, ratio, task_count = cohort_key
    return (
        f"files={len(files)},x{window_size}/y{advance_count},"
        f"ratio={ratio:g},tasks={task_count}"
    )


def dynamic_band_cohort_name(record: dict[str, Any], fallback_index: int) -> str:
    """Read a paired cohort's scenario name, falling back to cohort_<index> without experimentName."""
    return str(record.get("experimentName") or f"cohort_{fallback_index}").split(
        "_mc", maxsplit=1
    )[0]


def select_dynamic_fig2_method_records(
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    *,
    require_task_metrics: bool,
    preferred_cohort_key: tuple[Any, ...] | None = None,
    preferred_scenario_key: tuple[str, ...] | None = None,
    preferred_ratio: float | None = None,
    green_v: float | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[Any, ...] | None]:
    """Select the same dynamic Fig. 3 cohort for CP, one RL curve, and Human."""
    grouped: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for method in DYNAMIC_FIG2_METHODS:
        by_cohort: dict[tuple[Any, ...], dict[str, Any]] = {}
        for candidate in _dynamic_method_candidates(
            records,
            details,
            method,
            require_task_metrics=require_task_metrics,
            required_v=green_v,
        ):
            cohort_key = dynamic_fig2_cohort_key(candidate)
            if cohort_key is None:
                continue
            previous = by_cohort.get(cohort_key)
            # Dynamic Fig. 3 CP/Human controls ignore v and prioritize the newest timestamp.
            if method in {BASELINE_METHOD, HUMAN_METHOD}:
                candidate_rank = (
                    str(candidate.get("timestamp") or ""),
                    int(as_float(candidate.get("dynamicBatchCount")) or len(candidate.get("dynamicBatches") or [])),
                    len(candidate.get("dynamicTaskMetrics") or []),
                )
                previous_rank = (
                    str(previous.get("timestamp") or ""),
                    int(as_float(previous.get("dynamicBatchCount")) or len(previous.get("dynamicBatches") or [])),
                    len(previous.get("dynamicTaskMetrics") or []),
                ) if previous is not None else None
            elif method == GREEN_METHOD:
                candidate_v = as_float(
                    candidate.get("makespan_60min_cost_in_valley_price_units")
                )
                candidate_rank = (
                    int(
                        candidate_v is not None
                        and same_number(candidate_v, DYNAMIC_FIG2_DEFAULT_V)
                    ),
                    str(candidate.get("timestamp") or ""),
                    int(as_float(candidate.get("dynamicBatchCount")) or len(candidate.get("dynamicBatches") or [])),
                    len(candidate.get("dynamicTaskMetrics") or []),
                )
                previous_v = (
                    as_float(previous.get("makespan_60min_cost_in_valley_price_units"))
                    if previous is not None
                    else None
                )
                previous_rank = (
                    int(
                        previous_v is not None
                        and same_number(previous_v, DYNAMIC_FIG2_DEFAULT_V)
                    ),
                    str(previous.get("timestamp") or ""),
                    int(as_float(previous.get("dynamicBatchCount")) or len(previous.get("dynamicBatches") or [])),
                    len(previous.get("dynamicTaskMetrics") or []),
                ) if previous is not None else None
            else:
                candidate_rank = (
                    int(as_float(candidate.get("dynamicBatchCount")) or len(candidate.get("dynamicBatches") or [])),
                    len(candidate.get("dynamicTaskMetrics") or []),
                    str(candidate.get("timestamp") or ""),
                )
                previous_rank = (
                    int(as_float(previous.get("dynamicBatchCount")) or len(previous.get("dynamicBatches") or [])),
                    len(previous.get("dynamicTaskMetrics") or []),
                    str(previous.get("timestamp") or ""),
                ) if previous is not None else None
            if previous is None or candidate_rank >= previous_rank:
                by_cohort[cohort_key] = candidate
        grouped[method] = by_cohort

    cohort_keys = set().union(*(set(values) for values in grouped.values()))
    if preferred_cohort_key is not None:
        # Restrict selection to the requested cohort; individual methods may be absent.
        cohort_keys &= {preferred_cohort_key}
    if preferred_scenario_key is not None:
        matched = {key for key in cohort_keys if key[0] == preferred_scenario_key}
        if matched:
            cohort_keys = matched
    if preferred_ratio is not None:
        matched = {
            key for key in cohort_keys if math.isclose(key[3], preferred_ratio, rel_tol=1e-8, abs_tol=1e-8)
        }
        if matched:
            cohort_keys = matched
    if not cohort_keys:
        return {}, None

    def cohort_rank(key: tuple[Any, ...]) -> tuple[Any, ...]:
        present = sum(key in grouped[method] for method in DYNAMIC_FIG2_METHODS)
        complete_cp_rl = int(key in grouped[BASELINE_METHOD] and key in grouped[GREEN_METHOD])
        has_human = int(key in grouped[HUMAN_METHOD])
        # Keep the core CP/RL pair and prefer cohorts that also contain Human.
        return (complete_cp_rl, present, has_human, key[4], key[3], key[0])

    selected_key = max(cohort_keys, key=cohort_rank)
    selected = {
        method: grouped[method][selected_key]
        for method in DYNAMIC_FIG2_METHODS
        if selected_key in grouped[method]
    }
    return selected, selected_key


def available_dynamic_fig2_cohorts(
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    preferred_ratio: float,
) -> dict[tuple[int, int], tuple[Any, ...]]:
    """Find all dynamic window combinations with at least one plottable method."""
    materialized = list(records)
    timeline_by_method: dict[str, set[tuple[Any, ...]]] = {}
    task_metrics_by_method: dict[str, set[tuple[Any, ...]]] = {}
    for method in DYNAMIC_FIG2_METHODS:
        timeline_by_method[method] = {
            key
            for candidate in _dynamic_method_candidates(
                materialized,
                details,
                method,
                require_task_metrics=False,
            )
            if (key := dynamic_fig2_cohort_key(candidate)) is not None
        }
        task_metrics_by_method[method] = {
            key
            for candidate in _dynamic_method_candidates(
                materialized,
                details,
                method,
                require_task_metrics=True,
            )
            if (key := dynamic_fig2_cohort_key(candidate)) is not None
        }

    available_keys = set().union(
        *(timeline_by_method[method] | task_metrics_by_method[method] for method in DYNAMIC_FIG2_METHODS)
    )
    cohorts_by_pair: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
    for cohort_key in available_keys:
        cohorts_by_pair[(cohort_key[1], cohort_key[2])].append(cohort_key)

    def cohort_rank(key: tuple[Any, ...]) -> tuple[Any, ...]:
        timeline_count = sum(key in timeline_by_method[method] for method in DYNAMIC_FIG2_METHODS)
        task_metric_count = sum(
            key in task_metrics_by_method[method] for method in DYNAMIC_FIG2_METHODS
        )
        method_count = sum(
            key in timeline_by_method[method] or key in task_metrics_by_method[method]
            for method in DYNAMIC_FIG2_METHODS
        )
        # At the same x/y, maximize method coverage in both panels before considering tariff preference and scenario stability.
        return (
            min(timeline_count, task_metric_count),
            timeline_count + task_metric_count,
            method_count,
            int(math.isclose(key[3], preferred_ratio, rel_tol=1e-8, abs_tol=1e-8)),
            key[4],
            key[3],
            key[0],
        )

    return {
        pair: max(cohort_keys, key=cohort_rank)
        for pair, cohort_keys in sorted(cohorts_by_pair.items())
    }


def select_dynamic_fig2_series_records(
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    *,
    require_task_metrics: bool,
    cohort_key: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Select newest cohort records for cp_or/Human and newest records per v for green methods."""
    selected: list[dict[str, Any]] = []

    def latest(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        matching = [
            candidate
            for candidate in candidates
            if dynamic_fig2_cohort_key(candidate) == cohort_key
        ]
        if not matching:
            return None
        return max(
            matching,
            key=lambda candidate: (
                str(candidate.get("timestamp") or ""),
                int(
                    as_float(candidate.get("dynamicBatchCount"))
                    or len(candidate.get("dynamicBatches") or [])
                ),
                len(candidate.get("dynamicTaskMetrics") or []),
            ),
        )

    baseline = latest(
        _dynamic_method_candidates(
            records,
            details,
            BASELINE_METHOD,
            require_task_metrics=require_task_metrics,
        )
    )
    if baseline is not None:
        # Use v only for internal color/pairing keys, with a default if absent; do not filter newest CP records by v.
        baseline_v = as_float(baseline.get("makespan_60min_cost_in_valley_price_units"))
        selected.append(
            {
                "method": BASELINE_METHOD,
                "v": baseline_v if baseline_v is not None else DYNAMIC_FIG2_DEFAULT_V,
                "label": display_method_label(BASELINE_METHOD),
                "record": baseline,
            }
        )

    # cp_green_or reuses the current cohort and retains the newest successful record for each v.
    cp_green_by_v: dict[float, dict[str, Any]] = {}
    for candidate in _dynamic_method_candidates(
        records,
        details,
        CP_GREEN_METHOD,
        require_task_metrics=require_task_metrics,
    ):
        if dynamic_fig2_cohort_key(candidate) != cohort_key:
            continue
        value = as_float(candidate.get("makespan_60min_cost_in_valley_price_units"))
        if value is None:
            continue
        previous = cp_green_by_v.get(value)
        chosen = latest(item for item in (previous, candidate) if item is not None)
        if chosen is not None:
            cp_green_by_v[value] = chosen
    for value in sorted(cp_green_by_v):
        selected.append(
            {
                "method": CP_GREEN_METHOD,
                "v": value,
                "label": f"{CP_GREEN_METHOD} (v={value:g})",
                "record": cp_green_by_v[value],
            }
        )

    green_by_v: dict[float, dict[str, Any]] = {}
    for candidate in _dynamic_method_candidates(
        records,
        details,
        GREEN_METHOD,
        require_task_metrics=require_task_metrics,
    ):
        if dynamic_fig2_cohort_key(candidate) != cohort_key:
            continue
        value = as_float(candidate.get("makespan_60min_cost_in_valley_price_units"))
        if value is None:
            continue
        previous = green_by_v.get(value)
        chosen = latest(item for item in (previous, candidate) if item is not None)
        if chosen is not None:
            green_by_v[value] = chosen
    for value in sorted(green_by_v):
        selected.append(
            {
                "method": GREEN_METHOD,
                "v": value,
                "label": f"{display_method_label(GREEN_METHOD)} (v={value:g})",
                "record": green_by_v[value],
            }
        )

    human = latest(
        _dynamic_method_candidates(
            records,
            details,
            HUMAN_METHOD,
            require_task_metrics=require_task_metrics,
        )
    )
    if human is not None:
        # Like CP, Human uses recorded v only as an internal key, not a filter.
        human_v = as_float(human.get("makespan_60min_cost_in_valley_price_units"))
        selected.append(
            {
                "method": HUMAN_METHOD,
                "v": human_v if human_v is not None else DYNAMIC_FIG2_DEFAULT_V,
                "label": HUMAN_METHOD,
                "record": human,
            }
        )
    return selected


def is_dynamic_fig2_primary_series(item: dict[str, Any]) -> bool:
    """Show only three directly comparable curves in the primary dynamic view."""
    method = str(item.get("method") or "")
    return method in {BASELINE_METHOD, HUMAN_METHOD} or (
        method == GREEN_METHOD
        and same_number(item.get("v"), DYNAMIC_FIG2_DEFAULT_V)
    )


def dynamic_fig2_series_colors(
    series: Iterable[dict[str, Any]],
) -> dict[tuple[str, float], Any]:
    """Assign stable sequential colors to cp_green_or / RL v values."""
    materialized = list(series)
    cp_green_vs = sorted(
        {
            float(item["v"])
            for item in materialized
            if item["method"] == CP_GREEN_METHOD
        }
    )
    cp_green_colors = {
        value: plt.get_cmap("Greys")(0.48 + 0.42 * index / max(len(cp_green_vs) - 1, 1))
        for index, value in enumerate(cp_green_vs)
    }
    green_vs = sorted(
        {
            float(item["v"])
            for item in materialized
            if item["method"] == GREEN_METHOD
        }
    )
    green_colors = {
        value: plt.get_cmap("plasma")(0.12 + 0.76 * index / max(len(green_vs) - 1, 1))
        for index, value in enumerate(green_vs)
    }
    colors: dict[tuple[str, float], Any] = {}
    for item in materialized:
        key = (str(item["method"]), float(item["v"]))
        if item["method"] == BASELINE_METHOD:
            colors[key] = DYNAMIC_FIG2_PRIMARY_COLORS[BASELINE_METHOD]
        elif item["method"] == CP_GREEN_METHOD:
            colors[key] = cp_green_colors[float(item["v"])]
        elif item["method"] == HUMAN_METHOD:
            colors[key] = DYNAMIC_FIG2_PRIMARY_COLORS[HUMAN_METHOD]
        elif same_number(item["v"], DYNAMIC_FIG2_DEFAULT_V):
            colors[key] = DYNAMIC_FIG2_PRIMARY_COLORS[GREEN_METHOD]
        else:
            colors[key] = green_colors[float(item["v"])]
    return colors


def draw_dynamic_fig2_primary_legend(
    axis: Any,
    entries: Iterable[tuple[str, Any, str]],
    *,
    loc: str,
    bbox_to_anchor: tuple[float, float],
    ncols: int | None = None,
) -> Any | None:
    """Draw a compact three-method legend within the panel bounds."""
    primary_labels = {
        BASELINE_METHOD: display_method_label(BASELINE_METHOD),
        GREEN_METHOD: f"{display_method_label(GREEN_METHOD)} (v={DYNAMIC_FIG2_DEFAULT_V:g})",
        HUMAN_METHOD: HUMAN_METHOD,
    }
    entries_by_method = {
        method: (handle, primary_labels[method])
        for method, handle, _ in entries
        if method in primary_labels
    }
    ordered_entries = [
        entries_by_method[method]
        for method in (BASELINE_METHOD, GREEN_METHOD, HUMAN_METHOD)
        if method in entries_by_method
    ]
    if not ordered_entries:
        return None
    handles, labels = zip(*ordered_entries)
    return axis.legend(
        handles,
        labels,
        frameon=False,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=axis.transAxes,
        fontsize=FONT_SIZES["small"],
        ncols=ncols or len(ordered_entries),
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )


def annotate_dynamic_task_counts(
    axis: Any,
    curve: np.ndarray,
    events: list[tuple[float, int]],
    color: str,
    curve_minutes: np.ndarray,
    occupied_boxes: list[tuple[float, float, float, float]] | None = None,
    avoid_overlaps: bool = True,
    above_curve: bool = False,
    below_curve: bool = False,
    max_offset_points: int = 96,
) -> None:
    """Label task counts at completion times, resolving overlaps vertically only."""
    if not len(curve):
        return
    if occupied_boxes is None:
        occupied_boxes = []
    point_to_pixel = axis.figure.dpi / 72.0
    for event_minute, completed_count in events:
        event_time = min(max(event_minute, 0.0), float(curve_minutes[-1]))
        event_x = event_time / 60.0
        event_y = float(np.interp(event_time, curve_minutes, curve))
        label = str(completed_count)
        label_width = (8.0 + 4.5 * len(label)) * point_to_pixel
        label_height = 10.0 * point_to_pixel
        # Anchor label x positions at task completion; resolve overlaps vertically only,
        # so labels stay at actual breakpoints.
        candidate_offsets = [0]
        if avoid_overlaps:
            for distance in range(8, 49, 8):
                candidate_offsets.extend((distance, -distance))
        directed_max_offset = max(8, (max_offset_points // 8) * 8)
        if above_curve or below_curve:
            direction = -1 if below_curve else 1
            candidate_offsets = [
                direction * distance
                for distance in range(8, directed_max_offset + 1, 8)
            ]
        chosen_box: tuple[float, float, float, float] | None = None
        chosen_offset = 0
        base_center_x, base_center_y = axis.transData.transform((event_x, event_y))
        for offset in candidate_offsets:
            center_x = base_center_x
            center_y = base_center_y + offset * point_to_pixel
            candidate_box = (
                center_x - label_width / 2.0,
                center_y - label_height / 2.0,
                center_x + label_width / 2.0,
                center_y + label_height / 2.0,
            )
            if above_curve and candidate_box[3] > axis.bbox.ymax - 2.0:
                continue
            if below_curve and candidate_box[1] < axis.bbox.ymin + 2.0:
                continue
            overlaps = any(
                candidate_box[0] < occupied[2] + 2.0
                and candidate_box[2] + 2.0 > occupied[0]
                and candidate_box[1] < occupied[3] + 2.0
                and candidate_box[3] + 2.0 > occupied[1]
                for occupied in occupied_boxes
            )
            if not overlaps:
                chosen_offset = offset
                chosen_box = candidate_box
                break
        if chosen_box is None:
            center_x = base_center_x
            if above_curve:
                center_y = min(
                    base_center_y + directed_max_offset * point_to_pixel,
                    axis.bbox.ymax - label_height / 2.0 - 2.0,
                )
            elif below_curve:
                center_y = max(
                    base_center_y - directed_max_offset * point_to_pixel,
                    axis.bbox.ymin + label_height / 2.0 + 2.0,
                )
            else:
                center_y = base_center_y
            chosen_offset = (center_y - base_center_y) / point_to_pixel
            chosen_box = (
                center_x - label_width / 2.0,
                center_y - label_height / 2.0,
                center_x + label_width / 2.0,
                center_y + label_height / 2.0,
            )
        occupied_boxes.append(chosen_box)
        annotation_kwargs: dict[str, Any] = {
            "xy": (event_x, event_y),
            "xytext": (0, chosen_offset),
            "textcoords": "offset points",
            "ha": "center",
            "va": "center",
            "fontsize": FONT_SIZES["small"],
            "color": color,
            "bbox": {
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.88,
            },
            "zorder": 6,
        }
        if chosen_offset:
            annotation_kwargs["arrowprops"] = {
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.45,
                "alpha": 0.55,
            }
        axis.annotate(
            label,
            **annotation_kwargs,
        )


def draw_fig2_dynamic_savings(
    axis: Any,
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    power_path: Path,
    preferred_cohort_key: tuple[Any, ...] | None = None,
    panel_label: str = "a",
    cost_scale: float = 1.0,
) -> None:
    """Draw cumulative electricity cost over time for one selected cohort."""
    if panel_label:
        axis.text(-0.08, 1.04, panel_label, transform=axis.transAxes, fontsize=FONT_SIZES["panel_label"], fontweight=FONT_WEIGHTS["panel_label"])
    # All curves share the same file/window/tariff/task-count cohort.
    selected_series = (
        select_dynamic_fig2_series_records(
            records,
            details,
            require_task_metrics=False,
            cohort_key=preferred_cohort_key,
        )
        if preferred_cohort_key is not None
        else []
    )
    # Select a complete cohort, then retain CP, Human, and RL v=128;
    # Fig. 3c already shows parameter responses, avoiding repetition and an oversized legend in d.
    selected_series = [
        item for item in selected_series if is_dynamic_fig2_primary_series(item)
    ]
    print(
        f"dynamic Fig. 3d: cohort="
        f"{dynamic_fig2_cohort_label(preferred_cohort_key)}; "
        f"series={','.join(str(item['label']) for item in selected_series) or 'none'}"
    )
    if not selected_series:
        axis.text(
            0.5,
            0.5,
            "No successful dynamic scheduling record is available",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        axis.set_title("Dynamic cumulative electricity cost", pad=12)
        style_axis(axis)
        return

    series: list[dict[str, Any]] = []
    series_colors = dynamic_fig2_series_colors(selected_series)
    for series_index, selected in enumerate(selected_series):
        method = str(selected["method"])
        value = float(selected["v"])
        record = selected["record"]
        color = series_colors[(method, value)]
        try:
            _, load_values, makespan = load_curve(record, power_path)
            # Apply the caller's electricity-cost scale; Fig S1 uses the Fig. 3 D/E scale
            # so values and y axes are directly comparable.
            cost_values = cumulative_electricity_cost(record, load_values) * cost_scale
        except ValueError:
            # Keep other available methods when one method has incomplete details.
            continue
        if makespan > 0 and len(cost_values):
            series.append(
                {
                    "method": method,
                    "label": selected["label"],
                    "v": value,
                    "series_index": series_index,
                    "record": record,
                    "color": color,
                    "cost": cost_values,
                    "makespan": makespan,
                }
            )

    if not series:
        axis.text(0.5, 0.5, "Dynamic cost curve unavailable\nNo valid schedule details", ha="center", va="center")
        axis.set_title("Dynamic cumulative electricity cost", pad=12)
        style_axis(axis)
        return

    full_horizon = max(int(item["makespan"]) for item in series)
    if full_horizon <= 0:
        axis.text(0.5, 0.5, "Dynamic cost curve unavailable\nNo valid schedule horizon", ha="center", va="center")
        axis.set_title("Dynamic cumulative electricity cost", pad=12)
        style_axis(axis)
        return

    title_record = series[0]["record"]
    for item in series:
        # End each curve at its own completion time.
        cost_values = item["cost"]
        axis.step(
            np.arange(1, len(cost_values) + 1, dtype=float) / 60.0,
            cost_values,
            where="post",
            color=item["color"],
            linewidth=2.4 if item["method"] == GREEN_METHOD else 1.2,
            label="Human-centric" if item["method"] == HUMAN_METHOD else item["label"],
            zorder=3,
        )
    axis.set_xlabel("Hours from simulation start")
    axis.set_ylabel("Cumulative electricity cost (¥)")
    horizon_hours = max(full_horizon / 60.0, 1.0 / 60.0)
    axis.set_xlim(0, horizon_hours)
    axis.relim()
    axis.autoscale_view(scalex=False, scaley=True)
    # Fix y limits before endpoint projections so projections do not change padding.
    axis.set_ylim(*axis.get_ylim())
    human_final_cost = None
    for item in series:
        end_x = float(item["makespan"]) / 60.0
        end_cost = float(item["cost"][-1])
        if item["method"] == HUMAN_METHOD:
            human_final_cost = end_cost
        axis.hlines(
            end_cost, 0, end_x,
            color=item["color"], linestyles=(0, (4, 3)), linewidth=0.9, zorder=2,
        )
        axis.vlines(
            end_x, axis.get_ylim()[0], end_cost,
            color="#888888", linestyles=(0, (3, 3)), linewidth=0.9, zorder=2,
        )
        axis.annotate(
            f"¥{end_cost:.1f}",
            (end_x, end_cost),
            xytext=(-5, 5),
            textcoords="offset points",
            ha="right", va="bottom",
            color="black", fontsize=FONT_SIZES["small"] * 1.2 * 1.3,
            fontweight="bold", zorder=7, gid="figs1-endpoint-cost",
        )

    ratio = price_ratio(title_record)
    ratio_text = ratio_label(ratio) if ratio is not None else "unknown"
    task_count = int(
        as_float(title_record.get("totalTaskCount"))
        or len(title_record.get("experimentFileList") or [])
    )
    window_size = record_dynamic_dimension(title_record, "dynamicWindowSize")
    advance_count = record_dynamic_dimension(title_record, "dynamicAdvanceCount")
    window_text = f"x{window_size}/y{advance_count}" if window_size and advance_count else "dynamic"
    axis.set_title(
        f"Dynamic cumulative electricity cost ({task_count} tasks; {window_text}; {ratio_text} price ratio)",
        pad=12,
    )
    # Fig S1's shared top legend identifies methods; curve endpoints show cost only.
    style_axis(axis)
    axis.grid(False)
    axis.set_ylim(*axis.get_ylim())
    draw_tariff_period_icons(
        axis, title_record, full_horizon,
        top_cost=human_final_cost,
        icon_height=16.0, keep_last_moon_size=False,
    )


def dynamic_task_metric_points(
    record: dict[str, Any],
    metric_name: str,
) -> list[tuple[float, float]]:
    """Normalize dynamicTaskMetrics into curve points sorted by task count."""
    raw_metrics = record.get("dynamicTaskMetrics")
    if not isinstance(raw_metrics, list):
        return []
    points_by_task_count: dict[float, float] = {}
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, dict):
            continue
        task_count = as_float(raw_metric.get("taskCount"))
        metric_value = as_float(raw_metric.get(metric_name))
        if task_count is None or metric_value is None:
            continue
        points_by_task_count[task_count] = metric_value
    return sorted(points_by_task_count.items())


def draw_fig2_dynamic_task_count_cost(
    axis: Any,
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    preferred_cohort_key: tuple[Any, ...] | None = None,
    panel_label: str = "b",
    cost_scale: float = 1.0,
) -> None:
    """Plot dynamic task-count versus electricity-cost trends."""
    if panel_label:
        axis.text(-0.08, 1.04, panel_label, transform=axis.transAxes, fontsize=FONT_SIZES["panel_label"], fontweight=FONT_WEIGHTS["panel_label"])
    selected_series = (
        select_dynamic_fig2_series_records(
            records,
            details,
            require_task_metrics=True,
            cohort_key=preferred_cohort_key,
        )
        if preferred_cohort_key is not None
        else []
    )
    # Show the same primary methods as the cumulative-cost timeline: CP, Human, and RL v=128.
    selected_series = [
        item for item in selected_series if is_dynamic_fig2_primary_series(item)
    ]
    print(
        f"dynamic task-cost view: cohort="
        f"{dynamic_fig2_cohort_label(preferred_cohort_key)}; "
        f"series={','.join(str(item['label']) for item in selected_series) or 'none'}"
    )
    if not selected_series:
        axis.text(
            0.5,
            0.5,
            "Dynamic scheduling task metrics are not available",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        axis.set_title("Electricity cost by cumulative task count", pad=12)
        style_axis(axis)
        return

    series: list[dict[str, Any]] = []
    series_colors = dynamic_fig2_series_colors(selected_series)
    for selected in selected_series:
        method = str(selected["method"])
        value = float(selected["v"])
        record = selected["record"]
        color = series_colors[(method, value)]
        points = [
            (task_count, cost * cost_scale)
            for task_count, cost in dynamic_task_metric_points(
                record, "totalElectricityCostYuan"
            )
        ]
        if points:
            series.append(
                {
                    "method": method,
                    "label": selected["label"],
                    "v": value,
                    "record": record,
                    "color": color,
                    "points": points,
                }
            )
    if not series:
        axis.text(
            0.5,
            0.5,
            "Dynamic task-count electricity metrics are empty",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        axis.set_title("Electricity cost by cumulative task count", pad=12)
        style_axis(axis)
        return

    legend_entries: list[tuple[str, Any, str]] = []
    all_x: list[float] = []
    fitted_series: dict[tuple[str, float], dict[str, Any]] = {}
    # Seed confidence-band fitting for reproducible plots.
    bootstrap_rng = np.random.default_rng(20260713)
    for item in series:
        points_array = np.asarray(item["points"], dtype=float)
        x_values = points_array[:, 0]
        y_values = points_array[:, 1]
        scatter = axis.scatter(
            x_values,
            y_values,
            s=28,
            color=item["color"],
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        handle = scatter
        if len(x_values) >= 2 and np.ptp(x_values) > EPSILON:
            # Prefer quadratic fits for nonlinear growth; fall back to linear with fewer than three distinct x values.
            fit_degree = min(2, len(np.unique(x_values)) - 1)
            x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
            coefficients = np.polyfit(x_values, y_values, fit_degree)
            fitted_values = np.polyval(coefficients, x_grid)
            bootstrap_lines: list[np.ndarray] = []
            for _ in range(500):
                sample_indices = bootstrap_rng.integers(0, len(x_values), len(x_values))
                sample_x = x_values[sample_indices]
                if len(np.unique(sample_x)) <= fit_degree:
                    continue
                sample_coefficients = np.polyfit(
                    sample_x,
                    y_values[sample_indices],
                    fit_degree,
                )
                bootstrap_lines.append(np.polyval(sample_coefficients, x_grid))
            if bootstrap_lines:
                lower, upper = np.percentile(np.asarray(bootstrap_lines), [2.5, 97.5], axis=0)
                # Use translucent confidence bands matching the curve colors to keep points visible.
                axis.fill_between(
                    x_grid,
                    lower,
                    upper,
                    color=item["color"],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            series_key = (item["method"], item["v"])
            fitted_series[series_key] = {
                "x": x_grid,
                "y": fitted_values,
                "color": item["color"],
            }
            handle, = axis.plot(
                x_grid,
                fitted_values,
                linestyle="-",
                # Thicken the dark-green RL fit in the dynamic task-count view.
                linewidth=2.6 if item["method"] == GREEN_METHOD else 1.5,
                color=item["color"],
                alpha=0.9,
                zorder=2,
            )
        legend_entries.append(
            (
                str(item["method"]),
                handle,
                f"{item['label']} (final {y_values[-1]:.2f} ¥)",
            )
        )
        all_x.extend(x_values)
    # CP ignores v and uses the newest record; locate the selected fit by method.
    baseline_key = next((key for key in fitted_series if key[0] == BASELINE_METHOD), None)
    baseline_fit = fitted_series.get(baseline_key) if baseline_key is not None else None
    preferred_green_key = next(
        (
            key
            for key in fitted_series
            if key[0] == GREEN_METHOD
            and same_number(key[1], DYNAMIC_FIG2_DEFAULT_V)
        ),
        next((key for key in fitted_series if key[0] == GREEN_METHOD), None),
    )
    if baseline_fit is not None and preferred_green_key is not None:
        # For multiple v values, compare CP against RL v=128 first, or the smallest available v otherwise.
        first_fit = baseline_fit
        second_fit = fitted_series[preferred_green_key]
        overlap_start = max(float(first_fit["x"][0]), float(second_fit["x"][0]))
        overlap_end = min(float(first_fit["x"][-1]), float(second_fit["x"][-1]))
        if overlap_end > overlap_start:
            fill_x = np.linspace(overlap_start, overlap_end, 100)
            first_y = np.interp(fill_x, first_fit["x"], first_fit["y"])
            second_y = np.interp(fill_x, second_fit["x"], second_fit["y"])
            midpoint = (first_y + second_y) / 2.0
            axis.fill_between(
                fill_x,
                first_y,
                midpoint,
                color=first_fit["color"],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
            axis.fill_between(
                fill_x,
                second_y,
                midpoint,
                color=second_fit["color"],
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
    axis.set_xlabel("Task Count")
    axis.set_ylabel("Electricity Cost (Yuan)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlim(left=min(all_x))
    title_record = series[0]["record"]
    ratio = price_ratio(title_record)
    ratio_text = ratio_label(ratio) if ratio is not None else "unknown"
    window_size = record_dynamic_dimension(title_record, "dynamicWindowSize")
    advance_count = record_dynamic_dimension(title_record, "dynamicAdvanceCount")
    window_text = f"x{window_size}/y{advance_count}" if window_size and advance_count else "dynamic"
    axis.set_title(
        f"Electricity cost by task count ({window_text}; {ratio_text} price ratio)",
        pad=12,
    )
    draw_dynamic_fig2_primary_legend(
        axis,
        legend_entries,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
        ncols=1,
    )
    style_axis(axis)


def plot_dynamic_band_cohort_figs1(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
) -> list[Path]:
    """Generate a 5x2 Fig S1 from the five paired cohorts used in integrated Fig. 3 bands."""
    # Primary curves use v=128; pair cohorts with the same v,
    # so each RL curve corresponds to the same run as the Fig. 3 D/E band member.
    if not same_number(args.makespan_cost_units, DYNAMIC_FIG2_DEFAULT_V):
        raise SystemExit(
            "Dynamic Fig S1 renders the five Fig. 3 band cohorts at "
            f"v={DYNAMIC_FIG2_DEFAULT_V:g}; got PlotConfig.makespan_cost_units="
            f"{args.makespan_cost_units:g}"
        )
    # Match draw_dynamic_fig2_bands(): select the same five cohorts from the same records/details.
    materialized_records = list(records)
    ratio_key = normalized_ratio(args.fig2_price_ratio)
    paired_cohorts = paired_rows(
        select_records(materialized_records, args.makespan_cost_units)
    ).get(ratio_key, [])
    if len(paired_cohorts) != DYNAMIC_FIG2_BAND_COHORT_COUNT:
        raise SystemExit(
            "Dynamic Fig S1 requires the five paired Fig. 3a/b/c cohorts "
            f"at ratio={ratio_key:g}, found {len(paired_cohorts)}"
        )

    # Sort row titles by scenario name so Scenario 1..5 identify the same scenarios in Fig S1 and Fig S2.
    ordered_cohorts = sorted(
        (
            (dynamic_band_cohort_name(pair["baseline"], index), pair)
            for index, pair in enumerate(paired_cohorts, start=1)
        ),
        key=lambda item: item[0],
    )
    # Reserve a top legend row to avoid overlapping first-row curves and titles.
    figure = plt.figure(figsize=(13.6, 17.4), constrained_layout=True)
    # Spacing includes axis text; this value roughly halves the visible column gap.
    figure.get_layout_engine().set(wspace=0.04)
    grid = figure.add_gridspec(
        DYNAMIC_FIG2_BAND_COHORT_COUNT + 1,
        2,
        height_ratios=(0.28, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    legend_axis = figure.add_subplot(grid[0, :])
    legend_axis.axis("off")

    for row_index, (scenario_name, pair) in enumerate(ordered_cohorts, start=1):
        cohort_key = dynamic_fig2_cohort_key(pair["baseline"])
        if cohort_key is None:
            raise SystemExit(
                f"Dynamic Fig S1 found an invalid paired cohort: {scenario_name}"
            )
        # Like the bands, require CP/RL curves and task-count details; only warn if Human is missing.
        curve_methods, selected_key = select_dynamic_fig2_method_records(
            materialized_records,
            details,
            require_task_metrics=False,
            preferred_cohort_key=cohort_key,
            green_v=args.makespan_cost_units,
        )
        metric_methods, _ = select_dynamic_fig2_method_records(
            materialized_records,
            details,
            require_task_metrics=True,
            preferred_cohort_key=cohort_key,
            green_v=args.makespan_cost_units,
        )
        missing_methods = [
            method
            for method in (BASELINE_METHOD, GREEN_METHOD)
            if method not in curve_methods or method not in metric_methods
        ]
        if selected_key != cohort_key or missing_methods:
            raise SystemExit(
                "Dynamic Fig S1 is missing drawable core records for cohort "
                f"{scenario_name}: {', '.join(missing_methods) or 'cohort key mismatch'}"
            )
        if HUMAN_METHOD not in curve_methods:
            print(f"Dynamic Fig S1 cohort {scenario_name} has no Human record")

        row_title = f"Scenario {row_index}"
        cost_axis = figure.add_subplot(grid[row_index, 0])
        task_cost_axis = figure.add_subplot(grid[row_index, 1])
        draw_fig2_dynamic_savings(
            cost_axis,
            materialized_records,
            details,
            args.power_file,
            preferred_cohort_key=cohort_key,
            panel_label="",
            # Use the Fig. 3d/e display scale for directly comparable electricity costs.
            cost_scale=DYNAMIC_FIG2_COST_PLOT_SCALE,
        )
        draw_fig2_dynamic_task_count_cost(
            task_cost_axis,
            materialized_records,
            details,
            preferred_cohort_key=cohort_key,
            panel_label="",
            cost_scale=DYNAMIC_FIG2_COST_PLOT_SCALE,
        )
        # Use one shared legend for the 5x2 grid to preserve plotting space.
        for axis in (cost_axis, task_cost_axis):
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            axis.set_title(
                (
                    f"{row_title} — cumulative electricity cost"
                    if axis is cost_axis
                    else f"{row_title} — electricity cost by task count"
                ),
                fontsize=FONT_SIZES["body"],
                pad=5,
            )
            axis.xaxis.label.set_size(FONT_SIZES["small"])
            axis.yaxis.label.set_size(FONT_SIZES["small"])
            axis.tick_params(labelsize=FONT_SIZES["small"])

    figure.suptitle(
        "Five paired dynamic experimental settings",
        fontsize=FONT_SIZES["figure_title"],
        fontweight=FONT_WEIGHTS["figure_title"],
        y=1.0,
    )
    legend_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=DYNAMIC_FIG2_PRIMARY_COLORS[BASELINE_METHOD],
                linewidth=2.2,
                label=display_method_label(BASELINE_METHOD),
            ),
            Line2D(
                [],
                [],
                color=DYNAMIC_FIG2_PRIMARY_COLORS[GREEN_METHOD],
                linewidth=2.6,
                label=display_method_label(GREEN_METHOD),
            ),
            Line2D(
                [],
                [],
                color=DYNAMIC_FIG2_PRIMARY_COLORS[HUMAN_METHOD],
                linewidth=1.2,
                label="Human-centric",
            ),
            OffsetImage(plt.imread(ASSETS_DIR / "peak.png"), label="Peak"),
            OffsetImage(plt.imread(ASSETS_DIR / "off-peak.png"), label="Off-peak"),
        ],
        labels=[display_method_label(BASELINE_METHOD), display_method_label(GREEN_METHOD),
                "Human-centric", "Peak", "Off-peak"],
        handler_map={OffsetImage: TariffIconLegendHandler()},
        loc="center",
        bbox_to_anchor=(0.5, 0.28),
        ncols=5,
        frameon=False,
        fontsize=FONT_SIZES["small"],
        columnspacing=1.2,
        handletextpad=0.6,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "Fig S1.png"
    enlarge_axis_and_legend_fonts(figure)
    # Scale endpoint-cost text by 1.10 and other text by 1.10 * 1.3 after shared font scaling.
    figure.canvas.draw()
    for text in figure.findobj(match=matplotlib.text.Text):
        scale = 1.10 if text.get_gid() == "figs1-endpoint-cost" else 1.10 * 1.3
        text.set_fontsize(text.get_fontsize() * scale)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(
        "Dynamic Fig S1 cohorts: "
        + ", ".join(
            f"Scenario {index}={name}"
            for index, (name, _) in enumerate(ordered_cohorts, start=1)
        )
    )
    return [png_path]


def unique_compare_task_count(record: dict[str, Any]) -> int | None:
    """Read the compare task count, preferring its dedicated field."""
    for field in (
        "uniqueCompareTaskCount",
        "totalTaskCount",
        "filesPerExperiment",
        "dynamicWindowSize",
    ):
        value = as_float(record.get(field))
        if value is not None and value.is_integer():
            return int(value)
    return None


def select_dynamic_unique_compare_records(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Select successful compare summaries and deduplicate by scenario, method, and task count."""
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        if not is_dynamic_record(record):
            continue
        if str(record.get("uniqueCompareMode") or "").strip().lower() != "compare":
            continue
        method = str(
            record.get("scheduleMethod") or record.get("configuredScheduleMethod") or ""
        ).strip()
        if method not in UNIQUE_COMPARE_METHODS:
            continue
        if record.get("schedule_status") not in (None, "success"):
            continue
        scenario_name = str(record.get("uniqueCompareScenarioName") or "").strip()
        task_count = unique_compare_task_count(record)
        if (
            not scenario_name
            or task_count not in UNIQUE_COMPARE_TASK_COUNTS
            or as_float(record.get("totalAlgoCalTime")) is None
            or as_float(record.get("recomputedObjectiveValue")) is None
        ):
            continue
        # For duplicate combinations, use the last successful plottable JSONL record.
        selected[(scenario_name, method, task_count)] = record
    return selected


def dynamic_unique_compare_series(
    selected: dict[tuple[str, str, int], dict[str, Any]],
    scenario_names: list[str],
    method: str,
    metric_field: str,
    scenario_name: str | None,
) -> np.ndarray:
    """Return single-scenario or scenario-averaged curves, keeping missing points as NaN."""
    values: list[float] = []
    for task_count in UNIQUE_COMPARE_TASK_COUNTS:
        if scenario_name is None:
            metric_values = [
                as_float(selected[(name, method, task_count)].get(metric_field))
                for name in scenario_names
                if (name, method, task_count) in selected
            ]
            finite_values = [value for value in metric_values if value is not None]
            values.append(float(np.mean(finite_values)) if finite_values else np.nan)
            continue

        record = selected.get((scenario_name, method, task_count))
        value = as_float(record.get(metric_field)) if record is not None else None
        values.append(value if value is not None else np.nan)
    return np.asarray(values, dtype=float)


def draw_dynamic_unique_compare_metric(
    axis: Any,
    selected: dict[tuple[str, str, int], dict[str, Any]],
    scenario_names: list[str],
    scenario_name: str | None,
    metric_field: str,
    ylabel: str,
    title: str,
    methods: tuple[str, ...] = UNIQUE_COMPARE_METHODS,
    reference_time_limit: float | None = None,
    fig2_green_style: bool = False,
) -> None:
    """Draw a multi-method compare panel for one scenario or scenario averages."""
    x_values = np.asarray(UNIQUE_COMPARE_TASK_COUNTS, dtype=float)
    has_data = False
    # RL and limited-CP objectives may differ by only hundreds; draw CP first and RL dashed second
    # so both colors remain visible in overlaps.
    plot_methods = tuple(method for method in methods if method != GREEN_METHOD)
    if GREEN_METHOD in methods:
        plot_methods += (GREEN_METHOD,)
    for method in plot_methods:
        y_values = dynamic_unique_compare_series(
            selected,
            scenario_names,
            method,
            metric_field,
            scenario_name,
        )
        if np.isfinite(y_values).any():
            has_data = True
            axis.plot(
                x_values,
                y_values,
                color=UNIQUE_COMPARE_METHOD_COLORS[method],
                marker="o",
                markersize=4.0,
                linewidth=(
                    FIG2_COMPARE_GREEN_LINEWIDTH
                    if fig2_green_style and method == GREEN_METHOD
                    else 1.8
                ),
                linestyle=(
                    (0, FIG2_COMPARE_GREEN_DASHES)
                    if fig2_green_style and method == GREEN_METHOD
                    else "--" if method == GREEN_METHOD else "-"
                ),
                markerfacecolor=(
                    "none" if fig2_green_style else "white"
                ) if method == GREEN_METHOD else None,
                label=display_method_label(method),
                zorder=4 if method == GREEN_METHOD else 3,
            )
    if not has_data:
        axis.text(
            0.5,
            0.5,
            "No successful compare result",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#777777",
        )
    if metric_field == "totalAlgoCalTime" and reference_time_limit is not None:
        # Mark the nominal compare time limit to show deviations in actual solve times.
        axis.axhline(
            reference_time_limit,
            color="#555555",
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    axis.set_xticks(x_values)
    axis.set_xlabel("Parallel task count")
    axis.set_ylabel(ylabel)
    if metric_field == "recomputedObjectiveValue":
        # Use scientific notation for large objective values to preserve horizontal space.
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # Align titles at a fixed height above the scientific-notation offsets.
    axis.set_title(title, fontsize=FONT_SIZES["body"], pad=5, y=1.10)
    style_axis(axis)


def prepare_dynamic_unique_compare_summary(
    records: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[str],
    float | None,
]:
    """Prepare shared data for dynamic unique-compare Fig S2 and Fig. 3f."""
    materialized = list(records)
    compare_records = [
        record
        for record in materialized
        if str(record.get("uniqueCompareMode") or "").strip().lower() == "compare"
        and record_dynamic_dimension(record, "dynamicAdvanceCount") == 1
        and str(
            record.get("scheduleMethod") or record.get("configuredScheduleMethod") or ""
        ).strip() in UNIQUE_COMPARE_METHODS
    ]
    scenario_names = sorted(
        {
            str(record.get("uniqueCompareScenarioName") or "").strip()
            for record in compare_records
            if str(record.get("uniqueCompareScenarioName") or "").strip()
        }
    )
    if len(scenario_names) != 5:
        raise SystemExit(
            "Dynamic unique-compare summary requires five scenarios, "
            f"found {len(scenario_names)}: {', '.join(scenario_names) or 'none'}"
        )

    selected = select_dynamic_unique_compare_records(compare_records)
    if not selected:
        raise SystemExit("No successful dynamic unique-compare records are available")

    missing = [
        (scenario_name, method, task_count)
        for scenario_name in scenario_names
        for method in UNIQUE_COMPARE_METHODS
        for task_count in UNIQUE_COMPARE_TASK_COUNTS
        if (scenario_name, method, task_count) not in selected
    ]
    if missing:
        missing_text = ", ".join(
            f"{scenario}/{method}/n{task_count}"
            for scenario, method, task_count in missing
        )
        print(
            "Dynamic unique-compare missing successful records "
            f"({len(missing)}): {missing_text}"
        )

    schedule_limits = sorted(
        {
            value
            for record in compare_records
            if (value := as_float(record.get("scheduleTimeLimit"))) is not None
        }
    )
    nominal_time_limit = schedule_limits[0] if len(schedule_limits) == 1 else None
    return selected, scenario_names, nominal_time_limit


def plot_dynamic_unique_compare_figs2(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
) -> list[Path]:
    """Generate dynamic unique-compare Fig S2 with five scenario rows and two columns."""
    # Integrated Fig. 3f handles cross-scenario averages; nominal_time_limit is unused here.
    selected, scenario_names, _ = prepare_dynamic_unique_compare_summary(records)

    row_count = len(scenario_names)
    figure = plt.figure(figsize=(13.0, 16.2), constrained_layout=True)
    grid = figure.add_gridspec(
        row_count + 1,
        1,
        height_ratios=(0.30, *([1.0] * row_count)),
    )
    legend_axis = figure.add_subplot(grid[0, 0])
    legend_axis.axis("off")

    for row_index, scenario_name in enumerate(scenario_names, start=1):
        # Label the five scenarios as Scenario 1..5 in publication figures.
        row_title = f"Scenario {row_index}"
        row_grid = grid[row_index, 0].subgridspec(1, 2)
        draw_dynamic_unique_compare_metric(
            figure.add_subplot(row_grid[0, 0]),
            selected,
            scenario_names,
            scenario_name,
            "totalAlgoCalTime",
            "Solving time (s)",
            f"{row_title} — solving time",
            methods=DYNAMIC_FIG2_COMPARE_METHODS,
            fig2_green_style=True,
        )
        draw_dynamic_unique_compare_metric(
            figure.add_subplot(row_grid[0, 1]),
            selected,
            scenario_names,
            scenario_name,
            "recomputedObjectiveValue",
            "Total objective value",
            f"{row_title} — total objective",
            methods=DYNAMIC_FIG2_COMPARE_METHODS,
            fig2_green_style=True,
        )

    legend_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=UNIQUE_COMPARE_METHOD_COLORS[method],
                linewidth=(
                    FIG2_COMPARE_GREEN_LINEWIDTH if method == GREEN_METHOD else 2.0
                ),
                marker="o",
                markersize=4.0,
                linestyle=(0, (3.0, 2.0)) if method == GREEN_METHOD else "-",
                markerfacecolor="none" if method == GREEN_METHOD else None,
                label=FIG2_COMPARE_DISPLAY_LABELS[method],
            )
            for method in DYNAMIC_FIG2_COMPARE_METHODS
        ],
        loc="center",
        ncols=len(DYNAMIC_FIG2_COMPARE_METHODS),
        frameon=False,
        fontsize=FONT_SIZES["small"],
        columnspacing=1.4,
        handlelength=4.0,
        handletextpad=0.6,
    )
    figure.suptitle(
        "Scenario-level scaling of scheduling performance",
        fontsize=FONT_SIZES["figure_title"],
        fontweight=FONT_WEIGHTS["figure_title"],
        y=1.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "Fig S2.png"
    enlarge_axis_and_legend_fonts(figure)
    # Render complete tick labels before scaling all Fig S2 text by 1.2.
    figure.canvas.draw()
    for text in figure.findobj(match=matplotlib.text.Text):
        text.set_fontsize(text.get_fontsize() * 1.2)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(
        "Dynamic Fig S2 unique-compare: "
        f"scenarios={len(scenario_names)}, successful_records={len(selected)}"
    )
    return [png_path]


def plot_fig3(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
    details: dict[tuple[Any, ...], dict[str, Any]],
    dynamic_cohort_key: tuple[Any, ...] | None = None,
    dynamic_compare_records: Iterable[dict[str, Any]] | None = None,
) -> list[Path]:
    """Generate dynamic Fig. 3 with panels a-e and the two-panel f group."""
    # Materialize records because dynamic panels iterate them repeatedly; accept generator inputs.
    records = list(records)
    # Dynamic Fig. 3f uses the scenario-average definitions from unique-compare Fig S2.
    compare_selected, compare_scenario_names, compare_time_limit = (
        prepare_dynamic_unique_compare_summary(dynamic_compare_records or [])
    )
    # Restrict all panels to current x/y; D/E aggregate the same five paired cohorts as a/b/c.
    aggregate_records = [
        record
        for record in records
        if dynamic_cohort_key is None
        or (
            record_dynamic_dimension(record, "dynamicWindowSize")
            == dynamic_cohort_key[1]
            and record_dynamic_dimension(record, "dynamicAdvanceCount")
            == dynamic_cohort_key[2]
        )
    ]
    # Use V=16 pairs at all tariff ratios in A/B.
    selected = select_records(aggregate_records, 16.0)
    figure_pairs = paired_rows(selected)
    figure_ratios = sorted(
        ratio for ratio, ratio_rows in figure_pairs.items() if ratio_rows
    )
    baselines, green_records = select_parameter_scan_records(aggregate_records)
    scan_rows = build_parameter_scan_rows(baselines, green_records)

    # Displayed panel d (axis_e) is on the right of row 2; displayed panel e (axis_d) spans row 3.
    # Panel f's solve-time/objective plots and shared legend occupy row 4.
    figure = plt.figure(figsize=(13.0, 15.7), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=(3.0, 3.0, 2.7, 2.25))
    first_row = grid[0].subgridspec(1, 2, width_ratios=(1.0, 1.0))
    second_row = grid[1].subgridspec(1, 2, width_ratios=(1.0, 1.0))
    cumulative_cost_row = grid[2].subgridspec(1, 1)
    axis_a = figure.add_subplot(first_row[0, 0])
    axis_b = figure.add_subplot(first_row[0, 1])
    axis_c = figure.add_subplot(second_row[0, 0])
    axis_e = figure.add_subplot(second_row[0, 1])
    axis_d = figure.add_subplot(cumulative_cost_row[0, 0])

    if figure_ratios:
        draw_distribution_boxplot(
            axis_a,
            [
                [row["cost_reduction"] for row in figure_pairs[ratio]]
                for ratio in figure_ratios
            ],
            figure_ratios,
            "Relative electricity-cost reduction (%)",
            "a",
            show_raw_points=True,
        )
        axis_a.set_title("Relative electricity-cost reduction")
    else:
        draw_fig2_placeholder(
            axis_a,
            "a",
            "Relative electricity-cost reduction",
            "Paired dynamic cost-reduction data are not available",
        )
    if figure_ratios:
        draw_energy_shares(axis_b, figure_pairs, figure_ratios, panel_label="b")
        axis_b.set_title("Energy share shifts from peak to off-peak", pad=8)
    else:
        draw_fig2_placeholder(
            axis_b,
            "b",
            "Energy share shifts from peak to off-peak",
            "Paired dynamic energy-share data are not available",
        )

    draw_fig4b(axis_c, scan_rows, panel_label="c")
    axis_c.set_title("Cost–completion-time trade-off")
    draw_dynamic_fig2_bands(
        args,
        aggregate_records,
        details.values(),
        axes=(axis_d, axis_e),
        tariff_period_icons=True,
    )
    compare_row = grid[3].subgridspec(
        2,
        2,
        height_ratios=(0.18, 1.0),
    )
    compare_title_axis = figure.add_subplot(compare_row[0, :])
    compare_title_axis.axis("off")
    compare_title_axis.text(
        0.5,
        0.5,
        "RL initialization improves solution quality and search efficiency",
        ha="center",
        va="center",
        fontsize=FONT_SIZES["axes_title"],
        fontweight="normal",
    )
    fig2_f_label = compare_title_axis.text(
        -0.06,
        0.5,
        "f",
        ha="left",
        va="center",
        fontsize=FONT_SIZES["panel_label"],
        fontweight=FONT_WEIGHTS["panel_label"],
    )
    fig2_f_label.set_in_layout(False)
    compare_legend_handles = [
        Line2D(
            [],
            [],
            color=UNIQUE_COMPARE_METHOD_COLORS[method],
            linewidth=(
                FIG2_COMPARE_GREEN_LINEWIDTH if method == GREEN_METHOD else 2.0
            ),
            marker="o",
            markersize=4.0,
            # Use compact dashes in short legend handles; retain longer gaps in plotted curves.
            linestyle=(0, (3.0, 2.0)) if method == GREEN_METHOD else "-",
            markerfacecolor="none" if method == GREEN_METHOD else None,
            label=FIG2_COMPARE_DISPLAY_LABELS[method],
        )
        for method in DYNAMIC_FIG2_COMPARE_METHODS
    ]
    compare_time_axis = figure.add_subplot(compare_row[1, 0])
    compare_objective_axis = figure.add_subplot(compare_row[1, 1])
    for axis, metric_field, ylabel in (
        (compare_time_axis, "totalAlgoCalTime", "Solving time (s)"),
        (compare_objective_axis, "recomputedObjectiveValue", "Total objective value"),
    ):
        draw_dynamic_unique_compare_metric(
            axis,
            compare_selected,
            compare_scenario_names,
            None,
            metric_field,
            ylabel,
            "",
            methods=DYNAMIC_FIG2_COMPARE_METHODS,
            reference_time_limit=compare_time_limit,
            fig2_green_style=True,
        )
        axis.legend(
            handles=compare_legend_handles, loc="upper left",
            ncols=1, frameon=False,
            fontsize=FONT_SIZES["small"],
            columnspacing=0.7, handlelength=1.8, handletextpad=0.4,
            borderaxespad=0.4,
        )
    # Hide grids in integrated panels a, b, d, e, and both f plots.
    for axis in (axis_a, axis_b, axis_d, axis_e, compare_time_axis, compare_objective_axis):
        axis.grid(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "Fig 3.png"
    # Scale the six panel labels a-f while preserving their font family, weight, and positions.
    for axis in figure.axes:
        for text in axis.texts:
            if text.get_text() in {"a", "b", "c", "d", "e", "f"}:
                text.set_fontsize(text.get_fontsize() * 1.2)
    enlarge_axis_and_legend_fonts(figure)
    figure.savefig(
        png_path, dpi=args.dpi, bbox_inches="tight",
        bbox_extra_artists=[a for a in axis_d.artists if isinstance(a, AnnotationBbox)],
    )
    plt.close(figure)
    return [png_path]


def draw_dynamic_fig2_bands(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
    detail_records: Iterable[dict[str, Any]],
    *,
    axes: tuple[Any, Any],
    cost_scale: float = DYNAMIC_FIG2_COST_PLOT_SCALE,
    cost_unit: str = "¥",
    tariff_period_icons: bool = False,
    tariff_background: bool = False,
    reference_scenario: str | None = None,
    actual_curves: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Draw the five paired Fig. 3a/b/c cohorts as D/E band panels."""
    materialized_records = list(records)
    ratio_key = normalized_ratio(args.fig2_price_ratio)
    paired_cohorts = paired_rows(
        select_records(materialized_records, args.makespan_cost_units)
    ).get(ratio_key, [])
    if len(paired_cohorts) != DYNAMIC_FIG2_BAND_COHORT_COUNT:
        raise SystemExit(
            "Dynamic Fig. 3 requires the five paired Fig. 3a/b/c cohorts "
            f"at ratio={ratio_key:g}, found {len(paired_cohorts)}"
        )

    details = details_lookup(detail_records)
    curves_by_method: dict[str, list[np.ndarray]] = defaultdict(list)
    makespans_by_method: dict[str, list[int]] = defaultdict(list)
    cohort_names_by_method: dict[str, list[str]] = defaultdict(list)
    task_curves_by_method: dict[str, list[dict[int, float]]] = defaultdict(list)
    missing_cohorts_by_method: dict[str, list[str]] = defaultdict(list)
    missing_task_cohorts_by_method: dict[str, list[str]] = defaultdict(list)
    representative_record: dict[str, Any] | None = None

    for cohort_index, pair in enumerate(paired_cohorts, start=1):
        cohort_key = dynamic_fig2_cohort_key(pair["baseline"])
        if cohort_key is None:
            raise SystemExit("Dynamic Fig. 3 found an invalid paired cohort")
        scenario_name = dynamic_band_cohort_name(pair["baseline"], cohort_index)
        selected, selected_key = select_dynamic_fig2_method_records(
            materialized_records,
            details,
            require_task_metrics=False,
            preferred_cohort_key=cohort_key,
            green_v=args.makespan_cost_units,
        )
        if selected_key != cohort_key:
            raise SystemExit(
                "Dynamic Fig. 3 could not resolve the Fig. 3a/b/c cohort "
                f"{scenario_name} from its detailed records"
            )

        for method in DYNAMIC_FIG2_BAND_METHODS:
            method_record = selected.get(method)
            if method_record is None:
                if method in {BASELINE_METHOD, GREEN_METHOD}:
                    raise SystemExit(
                        "Dynamic Fig. 3 is missing a drawable core curve for "
                        f"{method}, cohort {scenario_name}"
                    )
                missing_cohorts_by_method[method].append(scenario_name)
                continue
            record = method_record
            task_points = dynamic_task_metric_points(
                record, "totalElectricityCostYuan"
            )
            task_curve = {
                int(task_count): float(cost) * cost_scale
                for task_count, cost in task_points
                if task_count.is_integer()
            }
            if task_curve:
                task_curves_by_method[method].append(task_curve)
            else:
                missing_task_cohorts_by_method[method].append(scenario_name)
            try:
                _, load_values, makespan = load_curve(record, args.power_file)
                cost_values = cumulative_electricity_cost(record, load_values)
            except ValueError as exc:
                if method in {BASELINE_METHOD, GREEN_METHOD}:
                    raise SystemExit(
                        "Dynamic Fig. 3 cannot load "
                        f"{method}, cohort {scenario_name}: {exc}"
                    ) from exc
                missing_cohorts_by_method[method].append(scenario_name)
                continue
            if makespan <= 0 or not len(cost_values):
                if method in {BASELINE_METHOD, GREEN_METHOD}:
                    raise SystemExit(
                        "Dynamic Fig. 3 has no valid cost curve for "
                        f"{method}, cohort {scenario_name}"
                    )
                missing_cohorts_by_method[method].append(scenario_name)
                continue
            curves_by_method[method].append(
                np.asarray(cost_values, dtype=float) * cost_scale
            )
            makespans_by_method[method].append(int(makespan))
            cohort_names_by_method[method].append(scenario_name)
            if representative_record is None:
                representative_record = record

    if representative_record is None:
        raise SystemExit("Dynamic Fig. 3 has no valid cohort curve")
    full_horizon = max(
        int(max(makespans)) for makespans in makespans_by_method.values()
    )
    if actual_curves:
        full_horizon = max(full_horizon, math.ceil(max(
            float(minutes[-1]) for minutes, _ in actual_curves.values()
        )))
    if full_horizon <= 0:
        raise SystemExit("Dynamic Fig. 3 has no positive schedule horizon")

    axis_d, axis_e = axes
    x_values = np.arange(full_horizon + 1, dtype=float) / 60.0
    if tariff_background:
        shade_figs3_tariff_periods(axis_d, representative_record, full_horizon)
    elif not tariff_period_icons:
        shade_tariff_periods(axis_d, representative_record, 0, full_horizon)
    emphasize_endpoints = tariff_period_icons or tariff_background
    method_labels = {
        BASELINE_METHOD: display_method_label(BASELINE_METHOD),
        GREEN_METHOD: display_method_label(GREEN_METHOD),
        HUMAN_METHOD: "Human-centric",
    }
    method_line_widths = {
        BASELINE_METHOD: 2.0,
        GREEN_METHOD: 2.4,
        HUMAN_METHOD: 1.8,
    }
    curve_endpoints = []
    human_final_cost = None
    for method in DYNAMIC_FIG2_BAND_METHODS:
        if not curves_by_method[method]:
            continue
        method_curves = curves_by_method[method]
        method_makespans = makespans_by_method[method]
        band_horizon = max(method_makespans)
        # Fig S3 fixes the Fig. 4 task cohort; other panels select the median-completion instance.
        if reference_scenario is not None:
            matches = [i for i, name in enumerate(cohort_names_by_method[method])
                       if name.partition("_unique_repeat_")[0] == reference_scenario]
            if len(matches) != 1:
                raise ValueError(f"Expected one {reference_scenario} cohort for {method}")
            representative_index = matches[0]
        else:
            representative_index = sorted(
                range(len(method_makespans)), key=method_makespans.__getitem__,
            )[len(method_makespans) // 2]
        representative_horizon = method_makespans[representative_index]
        representative_curve = np.concatenate((
            [0.0], method_curves[representative_index][:representative_horizon],
        ))
        representative_minutes = np.arange(representative_horizon + 1, dtype=float)
        if actual_curves and method in actual_curves:
            representative_minutes, actual_cost = actual_curves[method]
            representative_curve = np.asarray(actual_cost, dtype=float) * cost_scale
        # Bands cover all five instances; completed instances retain their final costs.
        # cumsum starts at minute 1; prepend zero to plot at minute boundaries.
        matrix = np.vstack(
            [
                np.pad(
                    np.concatenate(([0.0], values[:band_horizon])),
                    (0, max(0, band_horizon - len(values))),
                    mode="constant",
                    constant_values=float(values[-1]),
                )
                for values in method_curves
            ]
        )
        lower = np.min(matrix, axis=0)
        upper = np.max(matrix, axis=0)
        color = DYNAMIC_FIG2_PRIMARY_COLORS[method]
        axis_d.fill_between(
            x_values[:band_horizon + 1],
            lower,
            upper,
            step="post",
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        plot_line = axis_d.plot if actual_curves and method in actual_curves else axis_d.step
        plot_line(
            representative_minutes / 60.0,
            representative_curve,
            **({} if actual_curves and method in actual_curves else {"where": "post"}),
            color=color,
            linewidth=method_line_widths[method],
            label=method_labels[method],
            zorder=3,
        )
        end_x = float(representative_minutes[-1]) / 60.0
        end_cost = float(representative_curve[-1])
        print(
            f"Representative {'actual-execution' if actual_curves and method in actual_curves else 'fixed-scenario' if reference_scenario else 'makespan-median'} curve: method={method}; "
            f"cohort={cohort_names_by_method[method][representative_index]}; "
            f"makespans={method_makespans}; selected={representative_minutes[-1]}; "
            f"final_cost={cost_unit}{end_cost:.6f}"
        )
        curve_endpoints.append((end_x, end_cost, color))
        if method == HUMAN_METHOD:
            human_final_cost = end_cost
        axis_d.hlines(
            end_cost, 0, end_x,
            color=color, linestyles=(0, (4, 3)), linewidth=0.9, zorder=2,
        )
        axis_d.annotate(
            f"{cost_unit}{end_cost:.1f}",
            (end_x, end_cost),
            xytext=(-5, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="black" if emphasize_endpoints else color,
            fontsize=FONT_SIZES["small"] * (1.2 * 1.3 if emphasize_endpoints else 1.0),
            fontweight="bold" if emphasize_endpoints else "normal",
            zorder=5,
        )

    task_curve_endpoints: dict[str, tuple[float, float]] = {}
    fit_task_ranges_by_method: dict[str, tuple[float, float]] = {}
    for method in DYNAMIC_FIG2_BAND_METHODS:
        if axis_e is None:
            continue
        method_task_curves = task_curves_by_method[method]
        if not method_task_curves:
            continue
        fit_start = max(min(curve) for curve in method_task_curves)
        fit_end = min(max(curve) for curve in method_task_curves)
        if fit_end <= fit_start:
            missing_task_cohorts_by_method[method].append(
                "no shared task-count range"
            )
            continue
        fit_x = np.linspace(fit_start, fit_end, 200)
        fitted_cohort_curves: list[np.ndarray] = []
        for task_curve in method_task_curves:
            observed_x = np.asarray(sorted(task_curve), dtype=float)
            observed_y = np.asarray(
                [task_curve[int(task_count)] for task_count in observed_x],
                dtype=float,
            )
            degree = min(2, len(np.unique(observed_x)) - 1)
            coefficients = np.polyfit(observed_x, observed_y, degree)
            fitted_cohort_curves.append(np.polyval(coefficients, fit_x))
        matrix = np.vstack(fitted_cohort_curves)
        lower = np.min(matrix, axis=0)
        center = np.median(matrix, axis=0)
        upper = np.max(matrix, axis=0)
        # Pointwise medians can kink when representative cohorts change; smooth only the displayed centerline.
        sigma = 0.8 / (fit_x[1] - fit_x[0])
        radius = min(int(math.ceil(3 * sigma)), len(center) - 1)
        offsets = np.arange(-radius, radius + 1, dtype=float)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        smooth_center = np.convolve(
            np.pad(center, radius, mode="edge"), weights, mode="valid"
        )
        smooth_center += np.linspace(
            center[0] - smooth_center[0],
            center[-1] - smooth_center[-1],
            len(center),
        )
        color = DYNAMIC_FIG2_PRIMARY_COLORS[method]
        if len(method_task_curves) > 1:
            axis_e.fill_between(
                fit_x,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        axis_e.plot(
            fit_x,
            smooth_center,
            color=color,
            linewidth=method_line_widths[method],
            label=method_labels[method],
            zorder=3,
        )
        fit_task_ranges_by_method[method] = (fit_start, fit_end)
        task_curve_endpoints[method] = (float(fit_x[-1]), float(smooth_center[-1]))

    axis_d.text(
        -0.08,
        1.04,
        "e",
        transform=axis_d.transAxes,
        fontsize=FONT_SIZES["panel_label"],
        fontweight=FONT_WEIGHTS["panel_label"],
    )
    axis_d.set_xlabel("Hours from simulation start")
    axis_d.set_ylabel(f"Cumulative electricity cost ({cost_unit})")
    axis_d.set_xlim(0, max(full_horizon / 60.0, 1.0 / 60.0))
    axis_d.set_title(
        f"Cumulative electricity cost of {reference_scenario}" if reference_scenario
        else "Cumulative electricity cost of median-completion-time instances", pad=8,
    )
    annotation_lines: list[str] = []
    for method, missing_cohorts in missing_cohorts_by_method.items():
        if missing_cohorts:
            annotation_lines.append(
                f"Missing {method}: {', '.join(missing_cohorts)}"
            )
    if annotation_lines:
        axis_d.text(
            0.99,
            0.02,
            "\n".join(annotation_lines),
            transform=axis_d.transAxes,
            ha="right",
            fontsize=FONT_SIZES["small"],
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0},
            zorder=5,
        )
    legend_handles, legend_labels = axis_d.get_legend_handles_labels()
    if tariff_background:
        present_keys = set(tariff_keys_per_minute(representative_record, 0, 1440))
        for key, (color, alpha, label) in FIGS3_TARIFF_BACKGROUNDS.items():
            if key in present_keys:
                # Use stronger legend colors than backgrounds to distinguish red, green, and gray tariff periods.
                legend_handles.append(Patch(facecolor=color, alpha=0.65,
                                            edgecolor=color, linewidth=0.7))
                legend_labels.append(label)
    if tariff_period_icons:
        for filename, label in (("peak.png", "Peak"), ("off-peak.png", "Off-peak")):
            legend_handles.append(OffsetImage(plt.imread(ASSETS_DIR / filename)))
            legend_labels.append(label)
        if "flat_price" in tariff_keys_per_minute(representative_record, 0, 1440):
            legend_handles.append(Patch(facecolor="#dddddd", edgecolor="#888888"))
            legend_labels.append("Flat")
    axis_d.legend(
        legend_handles, legend_labels,
        handler_map={OffsetImage: TariffIconLegendHandler()} if tariff_period_icons else {},
        frameon=False,
        loc="upper left",
        fontsize=FONT_SIZES["small"],
    )
    style_axis(axis_d)
    # Keep space below zero; extend vertical projections to the x axis and use standard completion-time ticks.
    y_bottom, y_top = axis_d.get_ylim()
    if reference_scenario is not None:
        y_top += (y_top - y_bottom) * 0.12  # Reserve space for vertically staggered nearby Fig S3 endpoint labels.
    axis_d.set_ylim(y_bottom, y_top)
    if tariff_period_icons:
        draw_tariff_period_icons(
            axis_d, representative_record, full_horizon, top_cost=human_final_cost,
        )
    for end_x, end_cost, color in curve_endpoints:
        axis_d.vlines(
            end_x, axis_d.get_ylim()[0], end_cost,
            color="#888888", linestyles=(0, (3, 3)), linewidth=0.9, zorder=2,
        )
        print(f"Fig. 3 E endpoint: completion={end_x:.4f} h; cost={cost_unit}{end_cost:.4f}")

    completion_ticks = [end_x for end_x, _, _ in curve_endpoints]
    regular_ticks = [x for x in axis_d.get_xticks() if 0 <= x <= full_horizon / 60.0
                     and all(abs(x - end_x) > max(6, full_horizon / 60.0 * 0.075)
                             for end_x in completion_ticks)]
    ticks = sorted(set(regular_ticks + completion_ticks))
    axis_d.set_xticks(ticks, [f"{x:.2f}" if x in completion_ticks else f"{x:g}" for x in ticks])
    axis_d.tick_params(axis="x", colors="black")
    time_alignments = dict(zip(sorted(set(completion_ticks)), ("right", "left", "right")))
    for x, label in zip(ticks, axis_d.get_xticklabels()):
        if x in time_alignments:
            label.set_horizontalalignment(time_alignments[x])

    if axis_e is None:
        return

    axis_e.text(
        -0.08,
        1.04,
        "d",
        transform=axis_e.transAxes,
        fontsize=FONT_SIZES["panel_label"],
        fontweight=FONT_WEIGHTS["panel_label"],
    )
    axis_e.set_xlabel("Cumulative task count")
    axis_e.set_ylabel("Cumulative electricity cost (¥)")
    axis_e.xaxis.set_major_locator(MaxNLocator(integer=True))
    task_x_values = [
        bound
        for task_range in fit_task_ranges_by_method.values()
        for bound in task_range
    ]
    if task_x_values:
        axis_e.set_xlim(min(task_x_values), max(task_x_values))
    for reference_method, compared_method in (
        (HUMAN_METHOD, BASELINE_METHOD), (BASELINE_METHOD, GREEN_METHOD),
    ):
        if reference_method not in task_curve_endpoints or compared_method not in task_curve_endpoints:
            continue
        reference_x, reference_cost = task_curve_endpoints[reference_method]
        compared_x, compared_cost = task_curve_endpoints[compared_method]
        if not math.isclose(reference_x, compared_x) or reference_cost <= 0:
            continue
        saving_percent = (reference_cost - compared_cost) / reference_cost * 100.0
        axis_e.annotate(
            "", xy=(reference_x, compared_cost), xytext=(reference_x, reference_cost),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0,
                        "shrinkA": 4, "shrinkB": 4},
            annotation_clip=False, zorder=5,
        )
        axis_e.annotate(
            f"{saving_percent:.1f}%", xy=(reference_x, (reference_cost + compared_cost) / 2),
            xytext=(-4, -7.5), textcoords="offset points", ha="right", va="center",
            fontsize=FONT_SIZES["small"] * 1.2 * 1.3, color="black", zorder=6,
        )
        print(f"Fig. 3d endpoint reduction {reference_method} -> {compared_method}: {saving_percent:.6f}%")
    axis_e.set_title("Electricity cost by task count", pad=8)
    task_note_lines: list[str] = []
    for method, missing_cohorts in missing_task_cohorts_by_method.items():
        if missing_cohorts:
            task_note_lines.append(
                f"Missing {method}: {', '.join(missing_cohorts)}"
            )
    if task_note_lines:
        axis_e.text(
            0.99,
            0.02,
            "\n".join(task_note_lines),
            transform=axis_e.transAxes,
            ha="right",
            fontsize=FONT_SIZES["small"],
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0},
            zorder=5,
        )
    axis_e.legend(
        frameon=False,
        loc="upper left",
        fontsize=FONT_SIZES["small"],
    )
    style_axis(axis_e)

    curve_counts = ", ".join(
        f"{method}={len(curves_by_method[method])}"
        for method in DYNAMIC_FIG2_BAND_METHODS
    )
    task_curve_counts = ", ".join(
        f"{method}={len(task_curves_by_method[method])}"
        for method in DYNAMIC_FIG2_BAND_METHODS
    )
    output_label = "Integrated dynamic Fig. 3 D/E bands"
    print(
        f"{output_label}: cohorts={len(paired_cohorts)}; curves={curve_counts}; "
        f"missing={dict(missing_cohorts_by_method)}"
    )
    print(
        f"{output_label} task-cost fits: curves={task_curve_counts}; "
        f"fit_ranges={fit_task_ranges_by_method}; "
        f"missing={dict(missing_task_cohorts_by_method)}"
    )


def load_fig4_schedule_database(path: Path, expected_method: str) -> dict[str, Any]:
    """Read actual timestamps without modifying the database; use the earliest actual start as zero."""
    if not path.is_file():
        raise SystemExit(f"Fig. 4 SQLite database does not exist: {path}")

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        method_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower())) & {"cp", "rl"}
        method = (
            {"cp": BASELINE_METHOD, "rl": GREEN_METHOD}[next(iter(method_tokens))]
            if len(method_tokens) == 1 else None
        )
        if method != expected_method:
            raise ValueError(
                f"expected filename method={expected_method}, found {method!r} in {path}; "
                "database filename must contain exactly one cp or rl token"
            )

        workstations = {
            str(row["workstationCode"]): {
                "name": str(row["workstationName"] or row["workstationCode"]),
                "power_watts": as_float(row["powerWatts"]),
            }
            for row in connection.execute(
                "SELECT workstationCode, workstationName, powerWatts FROM ws_info"
            )
        }
        scheduled_rows = connection.execute(
            """
            SELECT expr_no, expr_name, fjspb_index, step_id, ws_type,
                   ws_code_fjspb, real_start_time, real_end_time
            FROM task_scheduled
            WHERE task_group = 'main_group' AND fjspb_index >= 1
            ORDER BY real_start_time, expr_no, fjspb_index
            """
        )
        operations: list[dict[str, Any]] = []
        intervals: set[tuple[str, float, float]] = set()
        actual_rows = list(scheduled_rows)
        if not actual_rows:
            raise ValueError("no actual schedule operations are available")
        parsed_times = [
            (datetime.fromisoformat(str(row["real_start_time"])),
             datetime.fromisoformat(str(row["real_end_time"])))
            for row in actual_rows
        ]
        origin = min(start for start, _ in parsed_times)
        for row, (real_start, real_end) in zip(actual_rows, parsed_times):
            workstation_code = str(row["ws_code_fjspb"] or "")
            workstation = workstations.get(workstation_code)
            if workstation is None:
                raise ValueError(
                    f"workstation {workstation_code!r} is missing from ws_info in {path}"
                )
            if row["fjspb_index"] is None:
                continue
            if real_end <= real_start:
                raise ValueError(f"invalid actual interval for {row['expr_no']}: {row['step_id']}")
            start = (real_start - origin).total_seconds() / 60.0
            end = (real_end - origin).total_seconds() / 60.0
            operation = {
                "expr_no": str(row["expr_no"]),
                "expr_name": str(row["expr_name"] or ""),
                "fjspb_index": int(row["fjspb_index"]),
                "step_id": str(row["step_id"] or ""),
                "ws_type": str(row["ws_type"] or ""),
                "ws_code": workstation_code,
                "ws_name": workstation["name"],
                "power_watts": float(workstation["power_watts"] or 0.0),
                "start": start,
                "end": end,
            }
            operations.append(operation)
            intervals.add((workstation_code, start, end))

        task_rows = list(
            connection.execute(
                """
                SELECT expr_no, expr_name FROM task_info
                WHERE task_group = 'main_group'
                ORDER BY rowid
                """
            )
        )
        task_order = [str(row["expr_no"]) for row in task_rows]
        task_name_order = [str(row["expr_name"] or "") for row in task_rows]
        if (
            any(row["expr_no"] is None for row in task_rows)
            or len(set(task_order)) != len(task_order)
            or not all(task_name_order)
            or len(set(task_name_order)) != len(task_name_order)
        ):
            raise ValueError("task_info must contain unique task numbers and nonempty unique task names")
        names_by_id = dict(zip(task_order, task_name_order))
        if any(names_by_id.get(op["expr_no"]) != op["expr_name"] for op in operations):
            raise ValueError("task_scheduled task numbers/names do not match task_info")
        task_ids = set(task_order)
        makespan = max((operation["end"] for operation in operations), default=0)
        if not operations or makespan <= 0:
            raise ValueError(f"no completed schedule is available in {path}")

        return {
            "path": path,
            "method": method,
            "origin": origin,
            "task_ids": task_ids,
            "task_order": task_order,
            "task_names": set(task_name_order),
            "task_name_order": task_name_order,
            "workstations": workstations,
            "operations": operations,
            "operation_keys": {
                (operation["expr_name"], operation["fjspb_index"])
                for operation in operations
            },
            "intervals": intervals,
            "makespan": makespan,
        }
    except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read Fig. 4 SQLite database {path}: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()


def fig4_database_cost_curve(
    database: dict[str, Any],
    tariff_record: dict[str, Any],
    horizon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate actual operations exactly; off-peak hours are 10-20 after zero, repeating every 24 hours."""
    horizon = database["makespan"] if horizon is None else min(
        max(0.0, float(horizon)), database["makespan"]
    )
    peak_price = as_float(tariff_record.get("peakPrice"))
    valley_price = as_float(tariff_record.get("valleyPrice"))
    if peak_price is None or valley_price is None or peak_price <= 0 or valley_price <= 0:
        raise ValueError("peakPrice and valleyPrice must be positive")
    # Count physical-station power once for overlapping or same-batch actual records.
    by_station: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for code, start, end in database["intervals"]:
        if code not in VIRTUAL_WORKSTATIONS and start < horizon:
            by_station[code].append((start, min(end, horizon)))
    events: dict[float, float] = defaultdict(float)
    for code, intervals in by_station.items():
        power_kw = float(database["workstations"][code]["power_watts"] or 0.0) / 1000.0
        if power_kw <= 0:
            continue
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            events[start] += power_kw
            events[end] -= power_kw
    boundaries = {0.0, float(horizon), *events}
    for day in range(int(horizon // 1440) + 1):
        for offset in (600, 1200):
            boundary = float(day * 1440 + offset)
            if boundary < horizon:
                boundaries.add(boundary)
    minutes = np.asarray(sorted(boundaries), dtype=float)
    delta = np.asarray([events.get(float(minute), 0.0) for minute in minutes])
    power = np.maximum(np.cumsum(delta)[:-1], 0.0)
    phase = ((minutes[:-1] + minutes[1:]) / 2.0) % 1440
    prices = np.where((phase >= 600) & (phase < 1200), valley_price, peak_price)
    costs = np.concatenate(([0.0], np.cumsum(power * prices * np.diff(minutes) / 60.0)))
    return minutes, costs


def fig4_database_completion_events(database: dict[str, Any]) -> list[tuple[float, int]]:
    """Count completed tasks using each SQLite task's final operation end time."""
    completion_times: dict[str, float] = {}
    for operation in database["operations"]:
        task_id = operation["expr_no"]
        completion_times[task_id] = max(
            completion_times.get(task_id, 0), operation["end"]
        )
    return [
        (end_time, count)
        for count, end_time in enumerate(sorted(completion_times.values()), start=1)
    ]


def load_fig4_selected_variants(
    path: Path, cp_database: dict[str, Any], rl_database: dict[str, Any]
) -> list[tuple[str, list[tuple[float, float, str, str]]]]:
    """Load six selected variants in template-generator order with consistent workstation powers."""
    try:
        templates = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read Fig. 4 task templates {path}: {exc}") from exc
    if not isinstance(templates, dict) or set(templates) != {
        task_type for task_type, _ in FIG4_A_VARIANTS
    }:
        raise SystemExit("Fig. 4 a requires six task types in base_steps.json")

    variants: list[tuple[str, list[tuple[float, float, str, str]]]] = []
    for task_type, variant_index in FIG4_A_VARIANTS:
        steps = templates[task_type]
        if not isinstance(steps, list):
            raise SystemExit(f"Fig. 4 a has invalid steps for {task_type}")
        choices = [
            (step_index, duration)
            for step_index, step in enumerate(steps)
            if isinstance(step.get("time"), list)
            for duration in step["time"]
        ]
        if not 1 <= variant_index <= len(choices):
            raise SystemExit(f"Fig. 4 a has no {task_type}_{variant_index} variant")
        selected_step_index, selected_duration = choices[variant_index - 1]
        operations: list[tuple[float, float, str, str]] = []
        for step_index, step in enumerate(steps):
            if step.get("workstationType") in {"lock", "unlock"}:
                continue
            raw_duration = step.get("time")
            if isinstance(raw_duration, list):
                raw_duration = raw_duration[0] if raw_duration else None
            if step_index == selected_step_index:
                raw_duration = selected_duration
            duration = as_float(raw_duration)
            if duration is None or not math.isfinite(duration) or duration <= 0:
                raise SystemExit(f"Fig. 4 a has invalid operation time for {task_type}")
            code = str(step.get("workstationCode") or "").removesuffix("__simu")
            cp_station = cp_database["workstations"].get(code)
            rl_station = rl_database["workstations"].get(code)
            if cp_station is None or rl_station is None:
                raise SystemExit(f"Fig. 4 a is missing power for {task_type}: {code}")
            cp_power = cp_station["power_watts"]
            rl_power = rl_station["power_watts"]
            if cp_power is None or rl_power is None or not math.isclose(
                cp_power, rl_power, abs_tol=EPSILON
            ):
                raise SystemExit(f"Fig. 4 a has inconsistent power for {code}")
            operations.append(
                (duration, float(rl_power), str(step["workstationType"]), rl_station["name"])
            )
        if not operations:
            raise SystemExit(f"Fig. 4 a has no operations for {task_type}")
        variants.append((f"{task_type}_{variant_index}", operations))
    return variants


def plot_dynamic_fig4(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
    detail_records: Iterable[dict[str, Any]],
) -> list[Path]:
    """Plot five-cohort ranges, actual SQLite schedules and matched Manual simulation."""
    materialized_records = list(records)
    ratio_key = normalized_ratio(args.fig2_price_ratio)
    paired_cohorts = paired_rows(
        select_records(materialized_records, args.makespan_cost_units)
    ).get(ratio_key, [])
    if len(paired_cohorts) != DYNAMIC_FIG2_BAND_COHORT_COUNT:
        raise SystemExit(
            "Dynamic Fig. 4 requires the five paired Fig. 3a/b/c cohorts at "
            f"ratio={ratio_key:g}, found {len(paired_cohorts)}"
        )

    cp_database = load_fig4_schedule_database(
        args.fig4_cp_database, BASELINE_METHOD
    )
    rl_database = load_fig4_schedule_database(
        args.fig4_rl_database, GREEN_METHOD
    )
    # Experiment IDs may be generated independently; pair databases by stable task names and operation indices.
    if cp_database["task_names"] != rl_database["task_names"]:
        raise SystemExit("Fig. 4 CP OR and RL Release OR databases have different task sets")
    if cp_database["task_name_order"] != rl_database["task_name_order"]:
        raise SystemExit(
            "Fig. 4 CP OR and RL Release OR databases have different task order"
        )
    if cp_database["operation_keys"] != rl_database["operation_keys"]:
        raise SystemExit(
            "Fig. 4 CP OR and RL Release OR databases have different scheduled operations"
        )
    details = details_lookup(detail_records)
    curves_by_method: dict[str, list[np.ndarray]] = defaultdict(list)
    makespans_by_method: dict[str, list[int]] = defaultdict(list)
    tariff_record: dict[str, Any] | None = None
    found_reference_cohort = False
    manual_reference: tuple[np.ndarray, int] | None = None
    for cohort_index, pair in enumerate(paired_cohorts, start=1):
        cohort_key = dynamic_fig2_cohort_key(pair["baseline"])
        if cohort_key is None:
            raise SystemExit("Dynamic Fig. 4 found an invalid paired cohort")
        selected, selected_key = select_dynamic_fig2_method_records(
            materialized_records,
            details,
            require_task_metrics=False,
            preferred_cohort_key=cohort_key,
            green_v=args.makespan_cost_units,
        )
        if selected_key != cohort_key:
            raise SystemExit(
                "Dynamic Fig. 4 could not resolve a paired integrated Fig. 3 cohort"
            )
        scenario_name = dynamic_band_cohort_name(pair["baseline"], cohort_index)
        is_reference_cohort = (
            scenario_name.partition("_unique_repeat_")[0] == "scenario_36_001"
        )
        if tariff_record is None or is_reference_cohort:
            tariff_record = pair["baseline"]
        if is_reference_cohort:
            found_reference_cohort = True
        for method in DYNAMIC_FIG2_BAND_METHODS:
            record = selected.get(method)
            if record is None:
                if method in {BASELINE_METHOD, GREEN_METHOD}:
                    raise SystemExit(
                        f"Dynamic Fig. 4 is missing {method} in cohort {scenario_name}"
                    )
                continue
            try:
                _, load_values, makespan = load_curve(record, args.power_file)
                cost_values = cumulative_electricity_cost(record, load_values)
            except ValueError as exc:
                if method in {BASELINE_METHOD, GREEN_METHOD}:
                    raise SystemExit(
                        f"Dynamic Fig. 4 cannot load {method}, cohort {scenario_name}: {exc}"
                    ) from exc
                continue
            # Use the Fig. 3 D/E scale for consistent bands and solid curves in panel c.
            curves_by_method[method].append(
                np.asarray(cost_values, dtype=float) * DYNAMIC_FIG2_COST_PLOT_SCALE
            )
            makespans_by_method[method].append(int(makespan))
            if is_reference_cohort and method == HUMAN_METHOD:
                manual_reference = (curves_by_method[method][-1], int(makespan))

    if tariff_record is None or not found_reference_cohort:
        raise SystemExit("Dynamic Fig. 4 requires scenario_36_001 price and clock metadata")
    if manual_reference is None:
        raise SystemExit("Dynamic Fig. 4 requires scenario_36_001 Manual simulation")
    manual_values, manual_makespan = manual_reference
    manual_cost = np.concatenate(([0.0], manual_values[:manual_makespan]))
    manual_minutes = np.arange(manual_makespan + 1, dtype=float)
    peak_price = as_float(tariff_record.get("peakPrice"))
    valley_price = as_float(tariff_record.get("valleyPrice"))
    if peak_price is None or valley_price is None or peak_price <= valley_price:
        raise SystemExit("Dynamic Fig. 4 requires valid peak and valley prices")
    source_start = tariff_record.get("simulationStartTime")
    try:
        source_origin = datetime.fromisoformat(str(source_start))
    except (TypeError, ValueError) as exc:
        raise SystemExit("Dynamic Fig. 4 is missing a valid simulationStartTime") from exc
    if (
        not math.isclose(peak_price / valley_price, ratio_key, rel_tol=1e-8, abs_tol=1e-8)
        or not curves_by_method[BASELINE_METHOD]
        or not curves_by_method[GREEN_METHOD]
    ):
        raise SystemExit("Dynamic Fig. 4 source price ratio or core cost curves are invalid")
    full_horizon = math.ceil(max(
        max(max(spans) for spans in makespans_by_method.values() if spans),
        cp_database["makespan"],
        rl_database["makespan"],
    ))
    def pad_curve(values: np.ndarray, horizon: int) -> np.ndarray:
        if len(values) >= horizon:
            return values[:horizon]
        return np.pad(
            values,
            (0, horizon - len(values)),
            mode="constant",
            constant_values=float(values[-1]),
        )

    # Panel c curves and panel b schedules use the same two final SQLite schedules, scaled by
    # the Fig. 3 D/E factor to match the five-cohort bands above.
    cp_minutes, cp_cost = fig4_database_cost_curve(cp_database, tariff_record)
    rl_minutes, rl_cost = fig4_database_cost_curve(rl_database, tariff_record)
    cp_cost *= DYNAMIC_FIG2_COST_PLOT_SCALE
    rl_cost *= DYNAMIC_FIG2_COST_PLOT_SCALE
    # Use one GridSpec to preserve row proportions; insert spacer rows between b panels and b-c,
    # controlling those gaps through row ratios while retaining default a-b spacing.
    figure = plt.figure(figsize=FIG4_FIGSIZE, constrained_layout=True)
    grid = figure.add_gridspec(
        len(FIG4_ROW_RATIOS), 1,
        height_ratios=FIG4_ROW_RATIOS,
        hspace=FIG4_ROW_SPACING,
    )
    axis_a = figure.add_subplot(grid[0])
    axis_b_cp = figure.add_subplot(grid[1])
    axis_b_rl = figure.add_subplot(grid[3])
    axis_cost = figure.add_subplot(grid[5])
    # Most operation powers are 0-1000 W; expand low-power colors to distinguish common equipment.
    power_norm = PowerNorm(gamma=0.25, vmin=0, vmax=9000)
    power_cmap = plt.get_cmap("turbo")
    x_values = np.arange(full_horizon, dtype=float) / 60.0
    for method in DYNAMIC_FIG2_BAND_METHODS:
        method_curves = curves_by_method[method]
        if not method_curves:
            continue
        matrix = np.vstack(
            [pad_curve(values, full_horizon) for values in method_curves]
        )
        color = DYNAMIC_FIG2_PRIMARY_COLORS[method]
        band_horizon = min(max(makespans_by_method[method]), full_horizon)
        axis_cost.fill_between(
            x_values[:band_horizon],
            np.min(matrix[:, :band_horizon], axis=0),
            np.max(matrix[:, :band_horizon], axis=0),
            step="post",
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

    cp_color = DYNAMIC_FIG2_PRIMARY_COLORS[BASELINE_METHOD]
    rl_color = DYNAMIC_FIG2_PRIMARY_COLORS[GREEN_METHOD]
    axis_cost.plot(
        cp_minutes / 60.0,
        cp_cost,
        color=cp_color,
        linewidth=2.5,
        label="Sustainability Unaware",
        zorder=4,
    )
    axis_cost.plot(
        rl_minutes / 60.0,
        rl_cost,
        color=rl_color,
        linewidth=2.5,
        label="Sustainability Aware",
        zorder=4,
    )
    manual_color = DYNAMIC_FIG2_PRIMARY_COLORS[HUMAN_METHOD]
    axis_cost.plot(
        manual_minutes / 60.0, manual_cost,
        color=manual_color, linewidth=2.5, label="Human-centric", zorder=4,
    )
    line_handles = [
        Line2D(
            [], [], color=cp_color, linewidth=2.5,
            label="Sustainability Unaware",
        ),
        Line2D(
            [], [], color=rl_color, linewidth=2.5,
            label="Sustainability Aware",
        ),
        Line2D([], [], color=manual_color, linewidth=2.5, label="Human-centric"),
    ]
    axis_cost.legend(
        handles=line_handles,
        loc="upper left",
        ncols=3,
        frameon=False,
        fontsize=FONT_SIZES["body"] * 0.9,
    )
    axis_cost.set_xlabel("Hours from experiment start")
    axis_cost.set_ylabel("Cumulative electricity\ncost (¥)")
    axis_cost.set_xlim(0, max(full_horizon / 60.0, 1.0 / 60.0))
    axis_cost.set_ylim(*axis_cost.get_ylim())  # Keep lower padding and fix limits so dashed lines do not expand them.
    axis_cost.set_title(
        "Cumulative electricity cost",
        pad=8,
        fontweight=FONT_WEIGHTS["axes_title"],
    )
    axis_cost.text(-0.055, 1.13, "c", transform=axis_cost.transAxes,
                   fontsize=FONT_SIZES["panel_label"],
                   fontweight=FONT_WEIGHTS["panel_label"])
    style_axis(axis_cost)
    axis_cost.grid(False)
    # As in Fig. 3, project final costs and actual completion times from curve endpoints.
    endpoint_labels = []
    for method, makespan, cost, color in (
        (BASELINE_METHOD, cp_database["makespan"], cp_cost, cp_color),
        (GREEN_METHOD, rl_database["makespan"], rl_cost, rl_color),
        (HUMAN_METHOD, manual_makespan, manual_cost, manual_color),
    ):
        end_hour = makespan / 60.0
        end_cost = float(cost[-1])
        axis_cost.vlines(end_hour, axis_cost.get_ylim()[0], end_cost, color="#888888",
                         linestyles=(0, (3, 3)), linewidth=0.9, zorder=2)
        axis_cost.hlines(end_cost, 0, end_hour, color=color,
                         linestyles=(0, (4, 3)), linewidth=0.9, zorder=2)
        endpoint_labels.append(axis_cost.annotate(
            f"¥{end_cost:.1f}", (end_hour, end_cost),
            xytext=(-12 if method == HUMAN_METHOD else 9 if method == BASELINE_METHOD else 6, 6),
            textcoords="offset points", ha="left", va="bottom", color="black",
            fontsize=FONT_SIZES["small"], fontweight="bold", zorder=6,
        ))
        print(f"Fig. 4 C {method}: completion={end_hour:.4f} h; cost=¥{end_cost:.4f}")

    completion_ticks = [makespan / 60.0 for makespan in (
        cp_database["makespan"], rl_database["makespan"], manual_makespan,
    )]
    regular_ticks = [x for x in axis_cost.get_xticks() if 0 <= x <= full_horizon / 60.0
                     and all(abs(x - end_x) > 4 for end_x in completion_ticks)]
    ticks = sorted(set(regular_ticks + completion_ticks))
    axis_cost.set_xticks(ticks, [f"{x:.2f}" if x in completion_ticks else f"{x:g}" for x in ticks])
    axis_cost.tick_params(axis="x", colors="black")

    origin = source_origin  # Align only the displayed simulation origin; preserve actual operation gaps.
    # Each b row is one task, with operations colored by workstation power.
    task_order = cp_database["task_order"]
    if len(task_order) != 36 or len(set(task_order)) != 36:
        raise SystemExit("Fig. 4 b requires 36 distinct tasks in SQLite task_info")
    full_schedule_horizon = math.ceil(max(cp_database["makespan"], rl_database["makespan"]))
    for axis_b, database, method_title in (
        (axis_b_cp, cp_database, "Sustainability unaware scheduling result"),
        (axis_b_rl, rl_database, "Sustainability aware scheduling result"),
    ):
        task_order = database["task_order"]
        task_rows = {task_id: index for index, task_id in enumerate(task_order)}
        task_names = {}
        for operation in database["operations"]:
            if operation["ws_code"] in VIRTUAL_WORKSTATIONS:
                continue
            task_id = operation["expr_no"]
            if task_id not in task_rows:
                raise SystemExit(f"Fig. 4 b found an unknown task {task_id} in {database['path']}")
            task_names.setdefault(task_id, operation["expr_name"] or task_id)
            axis_b.barh(
                task_rows[task_id], (operation["end"] - operation["start"]) / 60.0,
                left=operation["start"] / 60.0, height=0.72,
                color=power_cmap(power_norm(operation["power_watts"])),
                edgecolor="white", linewidth=0.25,
            )
        axis_b.set_yticks(
            range(len(task_order)),
            [f"{i + 1:02d} {task_names.get(task_id, task_id)}"
             for i, task_id in enumerate(task_order)],
        )
        axis_b.tick_params(axis="y", labelsize=7)
        axis_b.invert_yaxis()
        axis_b.set_ylim(len(task_order) - 0.4, -0.6)
        axis_b.set_xlim(0, full_schedule_horizon / 60.0)
        # Center and bold titles consistently with a/b.
        axis_b.set_title(
            method_title,
            pad=8,
            fontweight=FONT_WEIGHTS["axes_title"],
        )
        axis_b.set_ylabel("Task (arrival order)")
        # Label both schedule x axes with elapsed experiment hours.
        axis_b.set_xlabel("Hours from experiment start")
        style_axis(axis_b)
        axis_b.grid(False)
        # Show tariff periods using the Fig. 3e boundaries and sun/moon icons.
        draw_tariff_period_icons(
            axis_b, {"simulationStartTime": origin.isoformat()},
            full_schedule_horizon,
        )
    axis_b_cp.text(-0.055, 1.02, "b", transform=axis_b_cp.transAxes,
                   fontsize=FONT_SIZES["panel_label"],
                   fontweight=FONT_WEIGHTS["panel_label"])

    # Panel a shows six selected variants as continuous operation sequences without scheduling waits.
    selected_variants = load_fig4_selected_variants(BASE_STEPS_PATH, cp_database, rl_database)
    longest_duration = max(
        sum(duration for duration, _, _, _ in operations)
        for _, operations in selected_variants
    )
    station_legend_entries: dict[tuple[str, float], Any] = {}
    for row, (_, operations) in enumerate(selected_variants):
        elapsed = 0.0
        for duration, power_watts, station_type, station_name in operations:
            station_label = next(
                (label for fragment, label in FIG4_A_INSTANCE_LABELS.items()
                 if fragment in station_name),
                FIG4_A_STATION_LABELS.get(station_type, station_type),
            )
            station_display_name = (
                station_label if station_label.endswith(" station")
                else f"{station_label} Station"
            )
            color = power_cmap(power_norm(power_watts))
            station_legend_entries.setdefault(
                (station_label, power_watts),
                Patch(
                    facecolor=color, edgecolor="white",
                    label=f"{station_display_name} ({power_watts:g} W)",
                ),
            )
            axis_a.barh(
                row, duration / 60.0, left=elapsed / 60.0, height=0.72,
                color=color,
                edgecolor="white", linewidth=0.5,
            )
            elapsed += duration
    axis_a.set_yticks(
        range(len(selected_variants)),
        [name.rsplit("_", maxsplit=1)[0] for name, _ in selected_variants],
    )
    axis_a.invert_yaxis()
    axis_a.set_xlim(0, longest_duration / 60.0)
    # Match b/c: same default font, axes_title size, centered and bold.
    axis_a.set_title(
        "Duration and power distributions of six experimental categories",
        pad=8,
        fontweight=FONT_WEIGHTS["axes_title"],
    )
    axis_a.set_xlabel("Cumulative operation time (h)")
    axis_a.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(axis_a)
    axis_a.grid(False)
    axis_a.text(-0.055, 1.03, "a", transform=axis_a.transAxes,
                fontsize=FONT_SIZES["panel_label"],
                fontweight=FONT_WEIGHTS["panel_label"])

    axis_a.legend(
        handles=[
            station_legend_entries[key]
            for key in sorted(station_legend_entries, key=lambda item: (item[1], item[0]))
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.25),
        ncols=3,
        frameon=False,
        fontsize=FONT_SIZES["small"],
        handlelength=1.3,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0,
    )

    # Make the Fig. 4 colorbar span the figure height by laying out only the left area,
    # then placing an explicit cax at the right to avoid ax=[...] and shrink shortening it.
    layout_engine = figure.get_layout_engine()
    if layout_engine is None:
        raise SystemExit("Dynamic Fig. 4 requires constrained_layout for its full-height colorbar")
    layout_engine.set(rect=FIG4_LAYOUT_RECT)
    colorbar_axis = figure.add_axes(FIG4_COLORBAR_RECT)
    colorbar_axis.set_in_layout(False)
    power_colorbar = figure.colorbar(
        ScalarMappable(norm=power_norm, cmap=power_cmap),
        cax=colorbar_axis,
        orientation="vertical",
    )
    power_colorbar.set_label("Station power (W)")
    power_colorbar.set_ticks([0, 100, 360, 1000, 3000, 6000, 9000])
    # Reserve top space for the enlarged legend to avoid completed-task labels.
    axis_cost.set_ylim(top=axis_cost.get_ylim()[1] * 1.12)
    draw_tariff_period_icons(axis_cost, tariff_record, full_horizon)
    # Apply final fonts before avoiding task-label overlap using actual legend dimensions.
    enlarge_axis_and_legend_fonts(figure)
    # Fig. 4: enlarge A legends, B/C axis text, and power-colorbar ticks by 20%.
    for text in axis_a.get_legend().get_texts():
        text.set_fontsize(text.get_fontsize() * 1.2)
    for panel in (axis_b_cp, axis_b_rl, axis_cost):
        for label in (panel.xaxis.label, panel.yaxis.label):
            label.set_fontsize(label.get_fontsize() * 1.2)
        panel.tick_params(axis="x", labelsize=panel.get_xticklabels()[0].get_fontsize() * 1.2)
    # Enlarge B axis labels and colorbar ticks by a further 20%.
    for panel in (axis_b_cp, axis_b_rl):
        for label in (panel.xaxis.label, panel.yaxis.label):
            label.set_fontsize(label.get_fontsize() * 1.2)
    axis_cost.yaxis.label.set_fontsize(axis_b_cp.yaxis.label.get_fontsize())
    # Align panel labels to the figure's left edge while following each panel vertically.
    panel_labels = []
    for panel, name in ((axis_a, "a"), (axis_b_cp, "b"), (axis_cost, "c")):
        panel_label = next(text for text in panel.texts if text.get_text() == name)
        panel_label.set_transform(matplotlib.transforms.blended_transform_factory(
            figure.transFigure, panel.transAxes,
        ))
        panel_label.set_x(0.032)
        panel_label.set_fontsize(panel_label.get_fontsize() * 1.3)
        panel_label.set_in_layout(False)
        panel_labels.append(panel_label)
    colorbar_axis.tick_params(axis="y", labelsize=colorbar_axis.get_yticklabels()[0].get_fontsize() * 1.2 * 1.2)
    power_colorbar.ax.yaxis.label.set_fontsize(colorbar_axis.get_yticklabels()[0].get_fontsize())
    figure.canvas.draw()
    legend_box = axis_cost.get_legend().get_window_extent()
    occupied_boxes = [(legend_box.x0, legend_box.y0, legend_box.x1, legend_box.y1)]
    for label in endpoint_labels:
        box = label.get_window_extent()
        occupied_boxes.append((box.x0, box.y0, box.x1, box.y1))
    for database, minutes, cost, color, above_curve, below_curve in (
        (cp_database, cp_minutes, cp_cost, cp_color, True, False),
        (rl_database, rl_minutes, rl_cost, rl_color, False, True),
    ):
        existing_text_count = len(axis_cost.texts)
        annotate_dynamic_task_counts(
            axis_cost, cost, fig4_database_completion_events(database), color,
            curve_minutes=minutes,
            occupied_boxes=occupied_boxes,
            above_curve=above_curve,
            below_curve=below_curve,
            max_offset_points=24,
        )
        for annotation in axis_cost.texts[existing_text_count:]:
            annotation.set_color("black")
            if database is cp_database and annotation.get_text() == "36":
                offset_x, offset_y = annotation.get_position()
                annotation.set_position((offset_x, offset_y - 12))
        if database is cp_database:
            labels = {text.get_text(): text for text in axis_cost.texts[existing_text_count:]}
            label_two, label_four = labels["2"], labels["4"]
            anchor_two_y = axis_cost.transData.transform(label_two.xy)[1]
            anchor_four_y = axis_cost.transData.transform(label_four.xy)[1]
            offset_x, _ = label_four.get_position()
            aligned_offset_y = label_two.get_position()[1] + (
                anchor_two_y - anchor_four_y
            ) * 72.0 / figure.dpi
            label_four.set_position((offset_x, aligned_offset_y))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "Fig 4.png"
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", bbox_extra_artists=panel_labels)
    plt.close(figure)
    print(
        "Dynamic Fig. 4: five-cohort bands end at each method's last completion; "
        "cohort center lines omitted; "
        "paired SQLite curves=CP OR/RL Release OR;"
        " Manual simulation=scenario_36_001;"
        f"output={png_path}"
    )
    return [png_path]


def fig5_task_type_counts_from_files(record: dict[str, Any]) -> Counter[str]:
    """Return the 36-task mix encoded by the selected dynamic cohort files."""
    files = record.get("experimentFileList")
    if not isinstance(files, list) or not files:
        raise SystemExit("Fig. 5 reference Human record has no experimentFileList")
    counts: Counter[str] = Counter()
    for value in files:
        parts = re.split(r"[-_]", Path(str(value)).stem)
        if len(parts) < 3:
            raise SystemExit(f"Fig. 5 cannot parse task type from cohort file {value!r}")
        counts[f"{parts[-2]}_{parts[-1]}"] += 1
    return counts


def fig5_task_type_counts_from_database(database: dict[str, Any]) -> Counter[str]:
    """Return one normalized task type per Fig. 4 SQLite task."""
    first_operation_names = {
        operation["expr_no"]: operation["expr_name"]
        for operation in database["operations"]
        if operation["fjspb_index"] == 1
    }
    if set(first_operation_names) != database["task_ids"]:
        raise SystemExit("Fig. 5 green SQLite schedule does not cover every reference task")
    return Counter(
        "_".join(str(name).split("_")[:2])
        for name in first_operation_names.values()
    )


def draw_fig5_world_map(axis: Any) -> int:
    """Render neutral SDL markers and highlight five representative case locations."""
    for path in (FIG5_WORLD_MAP_PATH, FIG5_MAP_REFERENCE_PATH):
        if not path.is_file():
            raise SystemExit(f"Fig. 5 map asset is missing: {path}")
    with Image.open(FIG5_WORLD_MAP_PATH) as source:
        base = np.asarray(source.convert("RGBA"))
    with Image.open(FIG5_MAP_REFERENCE_PATH) as source:
        reference = np.asarray(source.convert("RGBA"))
    if base.shape != reference.shape:
        raise SystemExit("Fig. 5 map base and reference must have identical pixel dimensions")
    if base.shape[:2] != (1068, 1920):
        raise SystemExit("Fig. 5 requires the original 1920×1068 reference map")

    rgb = reference[:, :, :3].astype(np.int16)
    blue_mask = (
        (rgb[:, :, 2] > rgb[:, :, 0] + 40)
        & (rgb[:, :, 2] > rgb[:, :, 1] + 20)
        & (reference[:, :, 3] > 0)
    )
    blue_pixel_count = int(np.count_nonzero(blue_mask))
    if blue_pixel_count == 0:
        raise SystemExit("Fig. 5 map reference contains no blue marker pixels")
    if not np.array_equal(base[~blue_mask], reference[~blue_mask]):
        raise SystemExit("Fig. 5 map base and reference do not align outside the blue markers")
    # Recover centers from local distance peaks inside original diamonds, preserving dense markers.
    distance = np.zeros(blue_mask.shape, dtype=int)
    interior = blue_mask.copy()
    while interior.any():
        distance += interior
        padded = np.pad(interior, 1, mode="constant")
        interior &= (padded[:-2, 1:-1] & padded[2:, 1:-1]
                     & padded[1:-1, :-2] & padded[1:-1, 2:])
    padded = np.pad(distance, 1, mode="constant")
    local_max = np.maximum.reduce([
        padded[dy:dy + distance.shape[0], dx:dx + distance.shape[1]]
        for dy in range(3) for dx in range(3)
    ])
    centers = set(map(tuple, np.argwhere((distance == local_max) & (distance >= 2))))
    # Use neutral colors for ordinary sites to avoid implying national or regional groups.
    overlay = np.zeros_like(reference)
    while centers:
        pending = [centers.pop()]
        component = []
        while pending:
            y, x = pending.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in centers:
                        centers.remove(neighbor)
                        pending.append(neighbor)
        center_y, center_x = np.mean(component, axis=0)

        radius = 7.5
        left, right = max(0, int(center_x - radius)), min(blue_mask.shape[1], int(center_x + radius) + 2)
        top, bottom = max(0, int(center_y - radius)), min(blue_mask.shape[0], int(center_y + radius) + 2)
        yy, xx = np.ogrid[top:bottom, left:right]
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        circle = squared_distance <= radius ** 2
        core = squared_distance <= 6.5 ** 2
        overlay[top:bottom, left:right][circle] = (255, 255, 255, 255)
        overlay[top:bottom, left:right][core] = (85, 85, 85, 255)
    axis.imshow(base, origin="upper", interpolation="nearest", zorder=1)
    axis.imshow(overlay, origin="upper", interpolation="nearest", zorder=2)
    for country, (x, y) in FIG5_CASE_STUDY_MARKERS:
        color = FIG5_COUNTRY_COLORS[country]
        axis.scatter([x], [y], s=110, color=color, edgecolors="white",
                     linewidths=1.1, zorder=4)
    axis.set_axis_off()
    return blue_pixel_count


def fig5_country_saving_rows(
    args: PlotConfig,
    reference_human: dict[str, Any],
    china_reference: dict[str, float],
) -> list[dict[str, Any]]:
    """Compare the Fig. 4 task cohort under each region's own replay tariff."""
    rows = [{**china_reference, "region": "China\nHefei", "actual": True}]
    records = filter_dynamic_fig2_task_count(filter_dynamic_records(
        read_jsonl(args.countries_input), args.dynamic_window_size, args.dynamic_advance_count,
    ))
    details_path = args.countries_input.with_name(args.countries_input.stem + "_details.jsonl")
    detail_records = read_jsonl(details_path)
    for country_id, label in (
        ("uk_liverpool", "United Kingdom\nLiverpool"),
        ("us_berkeley", "United States\nBerkeley"),
        ("ca_toronto", "Canada\nToronto"),
        ("kr_daejeon", "South Korea\nDaejeon"),
    ):
        country_records = [r for r in records
                           if (r.get("countryScenario") or {}).get("id") == country_id]
        lookup = details_lookup([r for r in detail_records
                                 if (r.get("countryScenario") or {}).get("id") == country_id])
        reference_pairs = [pair for pairs in paired_rows(
            select_records(country_records, args.makespan_cost_units)
        ).values() for pair in pairs
            if dynamic_band_cohort_name(pair["baseline"], 1) == "scenario_36_001"]
        if len(reference_pairs) != 1:
            raise SystemExit(f"Fig. 5 requires one complete scenario_36_001 for {country_id}")
        selected, _ = select_dynamic_fig2_method_records(
            country_records, lookup, require_task_metrics=False,
            preferred_cohort_key=dynamic_fig2_cohort_key(reference_pairs[0]["baseline"]),
            green_v=args.makespan_cost_units,
        )
        human, green = selected.get(HUMAN_METHOD), selected.get(GREEN_METHOD)
        if human is None or green is None:
            raise SystemExit(f"Fig. 5 requires Manual and green schedules for {country_id}")
        for record in (human, green):
            if record.get("experimentFileList") != reference_human.get("experimentFileList"):
                raise SystemExit(f"Fig. 5 task cohort differs from Fig. 4 for {country_id}")
        if (human.get("tariffInfo") != green.get("tariffInfo")
                or human.get("simulationStartTime") != green.get("simulationStartTime")):
            raise SystemExit(f"Fig. 5 Manual and green tariffs/clocks differ for {country_id}")
        costs = []
        for record in (human, green):
            _, load, _ = load_curve(record, args.power_file)
            costs.append(float(cumulative_electricity_cost(record, load)[-1])
                         / float(record["valleyPrice"]))
        saving = costs[0] - costs[1]
        rows.append({"region": label, "actual": False, "human_cost": costs[0],
                     "green_cost": costs[1], "total_saving": saving,
                     "human_vs_aware": saving / DYNAMIC_FIG2_TASK_COUNT})
    return rows


def plot_dynamic_fig5(
    args: PlotConfig,
    records: Iterable[dict[str, Any]],
    detail_records: Iterable[dict[str, Any]],
) -> list[Path]:
    """Plot regional per-task savings (A) and the global SDL map (B)."""
    materialized_records = list(records)
    paired_cohorts = paired_rows(
        select_records(materialized_records, args.makespan_cost_units)
    ).get(normalized_ratio(FIG5_REFERENCE_PRICE_RATIO), [])
    if len(paired_cohorts) != DYNAMIC_FIG2_BAND_COHORT_COUNT:
        raise SystemExit(
            "Fig. 5 requires the five paired Fig. 3 cohorts at "
            f"ratio={FIG5_REFERENCE_PRICE_RATIO:g}, found {len(paired_cohorts)}"
        )

    details = details_lookup(detail_records)
    reference_human: dict[str, Any] | None = None
    task_count = DYNAMIC_FIG2_TASK_COUNT
    for cohort_index, pair in enumerate(paired_cohorts, start=1):
        # Actual Fig. 4 schedules correspond to scenario_36_001; use only its matching Manual simulation.
        if dynamic_band_cohort_name(pair["baseline"], cohort_index).partition(
            "_unique_repeat_"
        )[0] != "scenario_36_001":
            continue
        cohort_key = dynamic_fig2_cohort_key(pair["baseline"])
        if cohort_key is None:
            raise SystemExit("Fig. 5 found an invalid paired cohort")
        selected, selected_key = select_dynamic_fig2_method_records(
            materialized_records,
            details,
            require_task_metrics=False,
            preferred_cohort_key=cohort_key,
            green_v=args.makespan_cost_units,
        )
        if selected_key != cohort_key:
            raise SystemExit("Fig. 5 could not resolve the paired Fig. 3 cohort")
        record = selected.get(HUMAN_METHOD)
        if record is None or len(record.get("experimentFileList", [])) != task_count:
            raise SystemExit(
                f"Fig. 5 requires a complete {task_count}-task Manual schedule "
                f"for {dynamic_band_cohort_name(pair['baseline'], cohort_index)}"
            )
        reference_human = record

    # The green reference uses Fig. 4 actual operation times, workstation power, and tariff integration.
    green_database = load_fig4_schedule_database(args.fig4_rl_database, GREEN_METHOD)
    if len(green_database["task_ids"]) != task_count:
        raise SystemExit(f"Fig. 5 requires {task_count} tasks in the actual green schedule")
    if reference_human is None or (
        fig5_task_type_counts_from_files(reference_human)
        != fig5_task_type_counts_from_database(green_database)
    ):
        raise SystemExit("Fig. 5 actual green schedule does not match the reference Manual task mix")

    _, manual_load, manual_makespan = load_curve(reference_human, args.power_file)
    if manual_makespan <= 0 or not len(manual_load):
        raise SystemExit("Fig. 5 has no valid load curve for the matched Manual schedule")
    print(
        f"Fig. 5 matched Manual: scenario={dynamic_band_cohort_name(reference_human, 1)}; "
        f"makespan={manual_makespan} min; task_mix="
        f"{dict(sorted(fig5_task_type_counts_from_files(reference_human).items()))}"
    )
    # China retains Fig. 5's 2:1 values; other countries use matching Fig S3 simulations and full tariffs.
    china_manual_cost = float(cumulative_electricity_cost(
        {**reference_human, "peakPrice": FIG5_REFERENCE_PRICE_RATIO, "valleyPrice": 1.0},
        manual_load,
    )[-1])
    china_green_cost = float(fig4_database_cost_curve(
        green_database, {"peakPrice": FIG5_REFERENCE_PRICE_RATIO, "valleyPrice": 1.0},
    )[1][-1])
    china_reference = {"human_cost": china_manual_cost, "green_cost": china_green_cost,
                       "total_saving": china_manual_cost - china_green_cost,
                       "human_vs_aware": (china_manual_cost - china_green_cost) / task_count}
    country_rows = fig5_country_saving_rows(args, reference_human, china_reference)
    figure = plt.figure(figsize=(15.5, 14.5), constrained_layout=True)
    # Use a spacer row between A country labels and the B map.
    grid = figure.add_gridspec(3, 1, height_ratios=(5.4, 0.7, 7.4))
    country_axis = figure.add_subplot(grid[0, 0])
    map_axis = figure.add_subplot(grid[2, 0])
    fig5_axis_font_size = FONT_SIZES["body"] * FIG5_AXIS_FONT_SCALE * 1.20
    country_axis.set_box_aspect(1.0 / 2.5)
    positions = np.arange(len(country_rows))
    country_bars = country_axis.bar(
        positions, [row["human_vs_aware"] for row in country_rows],
        width=0.60, color=[FIG5_COUNTRY_COLORS[row["region"].split("\n")[0]]
                           for row in country_rows],
        edgecolor="#555555", linewidth=0.7, zorder=3,
    )
    for row, bar in zip(country_rows, country_bars):
        if not row["actual"]:
            bar.set_hatch("//")
        value = row["human_vs_aware"]
        country_axis.annotate(
            f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5 if value >= 0 else -5), textcoords="offset points",
            ha="center", va="bottom" if value >= 0 else "top",
            fontsize=fig5_axis_font_size,
        )
        print(f"Fig. 5 region={row['region'].replace(chr(10), ' / ')}; "
              f"green-source={'actual' if row['actual'] else 'simulation'}; "
              f"manual={row['human_cost']:.6f}; green={row['green_cost']:.6f}; "
              f"saving-per-task={value:.6f} kWh equiv./task")
    country_axis.set_xticks(positions, [row["region"] for row in country_rows])
    country_axis.set_xlabel("Representative tariff region", fontsize=fig5_axis_font_size, labelpad=plt.rcParams["axes.labelpad"] + 4.0)
    country_axis.set_ylabel(
        "Off-peak price normalized saving\n(kWh equiv. / experimental workflow)",
        fontsize=fig5_axis_font_size,
    )
    country_axis.set_title("Saving vs. simulated manual scheduling", pad=14,
                           fontsize=fig5_axis_font_size)
    country_axis.tick_params(axis="both", labelsize=fig5_axis_font_size)
    country_axis.yaxis.set_major_formatter(lambda value, _position: f"{value:,.1f}")
    country_axis.margins(x=0.08, y=0.25)
    country_axis.axhline(0, color="#555555", linewidth=0.8)
    country_axis.legend(handles=[
        Patch(facecolor=COLORS["human"], edgecolor="#376B32", label="Actual sustainability aware execution (China)"),
        Patch(facecolor="#D0D0D0", edgecolor="#555555", hatch="//",
              label="Simulated sustainability aware scheduling (other representative regions)"),
    ], frameon=False, loc="upper right", fontsize=FONT_SIZES["small"] * 1.50,
       handlelength=2.2, handleheight=1.3, labelspacing=0.6)
    style_axis(country_axis)
    country_axis.grid(False)

    blue_pixel_count = draw_fig5_world_map(map_axis)
    print(
        f"Fig. 5 map: neutral SDL markers with white edges; five colored case-study cities; source blue pixels={blue_pixel_count}; "
        f"base={FIG5_WORLD_MAP_PATH.name}; reference={FIG5_MAP_REFERENCE_PATH.name}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "Fig 5.png"
    enlarge_axis_and_legend_fonts(figure)
    figure.canvas.draw()
    panel_box = country_axis.get_window_extent()
    print(f"Fig. 5A plot width/height={panel_box.width / panel_box.height:.4f}")
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(
        f"Dynamic Fig. 5: five regional savings vs simulated Manual; China actual, other regions simulated; "
        f"{task_count} tasks; actual green database={args.fig4_rl_database}; output={png_path}"
    )
    return [png_path]



def enlarge_axis_and_legend_fonts(figure: Any) -> None:
    """Before export, enlarge axis titles by 10% and legends by 21%, preserving tick and main-title sizes."""
    axis_scale = 1.10
    legend_scale = 1.21
    for axis in figure.axes:
        for label in (axis.xaxis.label, axis.yaxis.label):
            label.set_fontsize(label.get_fontsize() * axis_scale)
        legend = axis.get_legend()
        if legend is not None:
            for text in [*legend.get_texts(), legend.get_title()]:
                text.set_fontsize(text.get_fontsize() * legend_scale)
    for legend in figure.legends:
        for text in [*legend.get_texts(), legend.get_title()]:
            text.set_fontsize(text.get_fontsize() * legend_scale)

def configure_plot_style() -> None:
    """Configure all font sizes centrally and scale all text with zoom."""
    # zoom scales all fonts; other entries define unscaled base sizes by text role.
    font_config: dict[str, Any] = {
        "zoom": 1.1,
        "font.family": "Arial",
        "font.size": 11.0,
        "axes.titlesize": 15.0,
        "figure.titlesize": 13.0,
        "panel_label.size": 15.0,
        "small.size": 8.0,
        "detail.size": 6.5,
        "endpoint.size": 9.0,
        "saving.size": 8.2,
        "bar_value.size": 8.0,
        "figure_title.weight": "bold",
        "axes_title.weight": "bold",
        "panel_label.weight": "bold",
        "endpoint.weight": "bold",
    }
    zoom = float(font_config["zoom"])
    if not math.isfinite(zoom) or zoom <= 0.0:
        raise SystemExit(f"font zoom must be a positive finite number, got: {zoom}")

    # Plotting functions use only the scaled named fonts generated here.
    size_sources = {
        "body": "font.size",
        "axes_title": "axes.titlesize",
        "figure_title": "figure.titlesize",
        "panel_label": "panel_label.size",
        "small": "small.size",
        "detail": "detail.size",
        "endpoint": "endpoint.size",
        "saving": "saving.size",
        "bar_value": "bar_value.size",
    }
    FONT_SIZES.clear()
    FONT_SIZES.update(
        {
            name: float(font_config[source]) * zoom
            for name, source in size_sources.items()
        }
    )
    FONT_WEIGHTS.clear()
    FONT_WEIGHTS.update(
        {
            "figure_title": str(font_config["figure_title.weight"]),
            "axes_title": str(font_config["axes_title.weight"]),
            "panel_label": str(font_config["panel_label.weight"]),
            "endpoint": str(font_config["endpoint.weight"]),
        }
    )
    # Check Arial explicitly to avoid silent font fallback on servers.
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family=font_config["font.family"], weight=weight),
            fallback_to_default=False,
        )
    plt.rcParams.update(
        {
            "font.family": font_config["font.family"],
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.bfit": "Arial:italic:bold",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.cal": "Arial",
            "mathtext.fallback": None,
            "font.size": FONT_SIZES["body"],
            "axes.titlesize": FONT_SIZES["axes_title"],
            "axes.labelsize": FONT_SIZES["body"],
            "xtick.labelsize": FONT_SIZES["body"],
            "ytick.labelsize": FONT_SIZES["body"],
            "legend.fontsize": FONT_SIZES["small"],
            "legend.title_fontsize": FONT_SIZES["small"],
            "figure.titlesize": FONT_SIZES["figure_title"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )



def plot_dynamic_countries_figs3(
    args: PlotConfig,
    china_records: Iterable[dict[str, Any]],
    china_details: Iterable[dict[str, Any]],
) -> list[Path]:
    """Plot solid curves for each country's matched Fig. 4 task cohort and five-cohort min-max ranges using replay tariffs."""
    def filtered(rows):
        return filter_dynamic_fig2_task_count(filter_dynamic_records(
            rows, args.dynamic_window_size, args.dynamic_advance_count,
        ))

    china_records = filtered(china_records)
    china_records = [r for r in china_records if not r.get("countryScenario")]
    china_details = filtered(china_details)
    country_records = filtered(read_jsonl(args.countries_input))
    country_details_path = args.countries_input.with_name(
        args.countries_input.stem + "_details.jsonl"
    )
    country_details = filtered(read_jsonl(country_details_path))
    groups = [("China", china_records, china_details, args.fig2_price_ratio)]
    for country_id, label in (
        ("uk_liverpool", "United Kingdom"), ("us_berkeley", "United States"),
        ("ca_toronto", "Canada"), ("kr_daejeon", "South Korea"),
    ):
        rows = [r for r in country_records
                if (r.get("countryScenario") or {}).get("id") == country_id]
        details = [r for r in country_details
                   if (r.get("countryScenario") or {}).get("id") == country_id]
        ratios = {price_ratio(r) for r in rows if r.get("schedule_status") == "success"}
        if not ratios:
            print(f"Fig S3: skip {label}; no completed records")
            continue
        if len(ratios) != 1:
            raise SystemExit(f"Fig S3: {label} contains multiple tariff ratios")
        ratio = ratios.pop()
        pairs = paired_rows(select_records(rows, args.makespan_cost_units)).get(
            normalized_ratio(ratio), []
        )
        lookup = details_lookup(details)
        complete = len(pairs) == DYNAMIC_FIG2_BAND_COHORT_COUNT
        for pair in pairs:
            selected, _ = select_dynamic_fig2_method_records(
                rows, lookup, require_task_metrics=False,
                preferred_cohort_key=dynamic_fig2_cohort_key(pair["baseline"]),
                green_v=args.makespan_cost_units,
            )
            complete = complete and all(m in selected for m in DYNAMIC_FIG2_BAND_METHODS)
        if not complete:
            print(f"Fig S3: skip {label}; five complete three-method cohorts required")
            continue
        groups.append((label, rows, details, ratio))

    figure, axes = plt.subplots(
        len(groups), 1, figsize=(13.0, 3.9 * len(groups)),
        constrained_layout=True, squeeze=False,
    )
    for index, (label, rows, details, ratio) in enumerate(groups):
        row_args = replace(args, fig2_price_ratio=ratio)
        axis = axes[index, 0]
        actual_curves = None
        if label == "China":
            databases = {
                BASELINE_METHOD: load_fig4_schedule_database(args.fig4_cp_database, BASELINE_METHOD),
                GREEN_METHOD: load_fig4_schedule_database(args.fig4_rl_database, GREEN_METHOD),
            }
            cp_database, rl_database = databases[BASELINE_METHOD], databases[GREEN_METHOD]
            if (len(cp_database["task_ids"]) != DYNAMIC_FIG2_TASK_COUNT
                    or cp_database["task_name_order"] != rl_database["task_name_order"]
                    or cp_database["operation_keys"] != rl_database["operation_keys"]):
                raise SystemExit("Fig S3 China requires matching Fig. 4 actual 36-task schedules")
            actual_curves = {
                method: fig4_database_cost_curve(
                    database, {"peakPrice": ratio, "valleyPrice": 1.0},
                ) for method, database in databases.items()
            }
        draw_dynamic_fig2_bands(
            row_args, rows, details, axes=(axis, None), cost_scale=1.0, cost_unit="",
            tariff_background=True,
            reference_scenario="scenario_36_001",
            actual_curves=actual_curves,
        )
        for text in list(axis.texts):
            if text.get_text() == "e":
                text.set_text(chr(ord("a") + index))
        axis.set_title(f"{label} — peak/off-peak ratio {ratio:.2f}", pad=8)
        axis.set_ylabel("Off-peak price normalized cost\n(kWh equiv.)")
        axis.grid(False)
        # Stagger nearby bold endpoint-cost labels vertically to prevent overlap.
        cost_labels = [text for text in axis.texts
                       if isinstance(text, matplotlib.text.Annotation)
                       and re.fullmatch(r"\d+\.\d+", text.get_text())]
        for first_index, first in enumerate(cost_labels):
            for second in cost_labels[first_index + 1:]:
                if (abs(first.xy[0] - second.xy[0]) < np.ptp(axis.get_xlim()) * 0.05
                        and abs(first.xy[1] - second.xy[1]) < np.ptp(axis.get_ylim()) * 0.08):
                    higher = first if first.xy[1] >= second.xy[1] else second
                    higher.set_position((-5, 18))
        print(f"Fig S3 {label}: five paired settings, three methods, V={args.makespan_cost_units:g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "Fig S3.png"
    enlarge_axis_and_legend_fonts(figure)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return [output]



class TariffIconLegendHandler(HandlerBase):
    """Render sun/moon images as standard legend entries at the configured legend position."""

    def create_artists(self, legend, handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        side = height * 1.6
        bounds = Bbox.from_bounds(
            -xdescent + (width - side) / 2,
            -ydescent + (height - side) / 2, side, side,
        )
        icon = BboxImage(TransformedBbox(bounds, trans), interpolation="hanning")
        icon.set_data(handle.get_data())
        return [icon]

def draw_tariff_period_icons(
    axis: Any, record: dict[str, Any], horizon: int, *, top_cost: float | None = None,
    icon_height: float = 24.0, keep_last_moon_size: bool = True,
) -> None:
    """Shared tariff images and gray boundaries for Fig. 3e / Fig. 4b/c / Fig S1 / Fig S3."""
    keys = tariff_keys_per_minute(record, 0, horizon)
    changes = (np.flatnonzero(keys[1:] != keys[:-1]) + 1).tolist()
    y_bottom, y_top = axis.get_ylim()
    separator_top = (top_cost - y_bottom) / (y_top - y_bottom) if top_cost is not None else 1.0
    for minute in changes:
        axis.axvline(
            minute / 60.0, ymax=separator_top,
            color="#a8a8a8", linestyle=(0, (3, 4)),
            linewidth=0.8, zorder=0,
        )
    icons = {
        "peak_price": plt.imread(ASSETS_DIR / "peak.png"),
        "valley_price": plt.imread(ASSETS_DIR / "off-peak.png"),
    }
    boundaries = [0, *changes, horizon]
    axis_width_points = axis.get_position().width * axis.figure.get_figwidth() * 72
    for left, right in zip(boundaries, boundaries[1:]):
        if keys[left] == "flat_price":
            label = axis.text(
                (left + right) / 120.0, 0.5, "Flat", ha="center", va="center",
                transform=axis.get_xaxis_transform(), fontsize=FONT_SIZES["small"] * 0.8,
                color="#666666", rotation=90,
                bbox={"facecolor": "#eeeeee", "edgecolor": "none", "pad": 1.5},
                zorder=0.5,
            )
            label.set_in_layout(False)
            continue
        pixels = icons[str(keys[left])]
        # Standard icon height is 24 pt, including the final moon.
        available_width = axis_width_points * (right - left) / horizon
        zoom = min(icon_height / pixels.shape[0], available_width * 0.65 / pixels.shape[1])
        if keep_last_moon_size and right == horizon and keys[left] == "valley_price":
            zoom = icon_height / pixels.shape[0]
        icon = AnnotationBbox(
            OffsetImage(pixels, zoom=zoom),
            ((left + right) / 120.0, 0.5),
            xybox=(4, 0) if keep_last_moon_size and right == horizon and keys[left] == "valley_price" else (0, 0),
            boxcoords="offset points",
            xycoords=axis.get_xaxis_transform(), frameon=False,
            box_alignment=(0.5, 0.5), pad=0, zorder=0.5,
        )
        icon.set_in_layout(False)
        axis.add_artist(icon)


def main() -> int:
    args = PlotConfig()
    input_paths = default_dynamic_input_paths()
    raw_records = read_jsonl_paths(input_paths)
    records = filter_dynamic_records(
        raw_records, args.dynamic_window_size, args.dynamic_advance_count,
    )
    if not records:
        raise SystemExit(
            "No dynamic records remain after replay filter: "
            f"x{args.dynamic_window_size}/y{args.dynamic_advance_count}"
        )
    raw_details = (
        read_jsonl(DEFAULT_DYNAMIC_DETAILS_INPUT)
        if DEFAULT_DYNAMIC_DETAILS_INPUT.is_file() else raw_records
    )

    def figure_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return filter_dynamic_fig2_task_count(
            record for record in rows
            if is_dynamic_record(record)
            and record_dynamic_dimension(record, "dynamicWindowSize") == args.dynamic_window_size
            and record_dynamic_dimension(record, "dynamicAdvanceCount") == args.dynamic_advance_count
        )

    fig2_records = figure_records(raw_records)
    fig2_details = figure_records(raw_details)
    fig2_detail_lookup = details_lookup(fig2_details)
    available_cohorts = available_dynamic_fig2_cohorts(
        fig2_records, fig2_detail_lookup, normalized_ratio(args.fig2_price_ratio),
    )
    if not available_cohorts:
        raise SystemExit(
            "No dynamic Fig. 3 cohort contains a drawable method for "
            f"x{args.dynamic_window_size}/y{args.dynamic_advance_count}: "
            f"{', '.join(map(str, input_paths))}"
        )
    cohort_key = available_cohorts[
        (args.dynamic_window_size, args.dynamic_advance_count)
    ]
    if not UNIQUE_TASK_DYNAMIC_INPUT.is_file():
        raise SystemExit(
            "Dynamic Fig. 3 requires the unique-compare summary jsonl: "
            f"{UNIQUE_TASK_DYNAMIC_INPUT}"
        )
    compare_records = [
        record for record in read_jsonl(UNIQUE_TASK_DYNAMIC_INPUT)
        if is_dynamic_record(record)
    ]
    if not compare_records:
        raise SystemExit(
            "No dynamic unique-compare records found: "
            f"{UNIQUE_TASK_DYNAMIC_INPUT}"
        )
    configure_plot_style()
    print(
        f"Generate all publication figures for x{args.dynamic_window_size}"
        f"/y{args.dynamic_advance_count}: {dynamic_fig2_cohort_label(cohort_key)}"
    )
    output_paths = plot_fig3(
        args, fig2_records, fig2_detail_lookup,
        dynamic_cohort_key=cohort_key, dynamic_compare_records=compare_records,
    )
    output_paths.extend(plot_dynamic_fig4(args, fig2_records, fig2_details))
    output_paths.extend(plot_dynamic_fig5(args, fig2_records, fig2_details))
    output_paths.extend(plot_dynamic_band_cohort_figs1(args, fig2_records, fig2_detail_lookup))
    output_paths.extend(plot_dynamic_unique_compare_figs2(args, compare_records))
    output_paths.extend(plot_dynamic_countries_figs3(args, fig2_records, fig2_details))
    for output_path in output_paths:
        print(f"Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
