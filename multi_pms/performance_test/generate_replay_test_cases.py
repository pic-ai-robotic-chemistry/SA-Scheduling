from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR.parent / "dms_link" / "real_expr"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parent / "expr_data" / "data" / "replay_test_cases.json"
DEFAULT_SCENE_COUNT = 60
DEFAULT_RANDOM_SEED = 20260712
# The main test set uses six experiment categories with equal task allocation per scenario.
REQUIRED_GROUPS = ("pbas", "pet", "im", "smc", "hec", "ldh")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate balanced replay test scenarios")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path (default: expr_data/data/replay_test_cases.json)",
    )
    parser.add_argument("--scene-count", type=int, default=DEFAULT_SCENE_COUNT)
    parser.add_argument(
        "--tasks-per-scene",
        type=int,
        default=len(REQUIRED_GROUPS) * 2,
        help="Tasks per scenario; must be evenly divisible among the six experimental categories (default: 12)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def extract_group_name(file_path: Path) -> str:
    parts = file_path.stem.lower().split("-")
    return parts[1] if len(parts) >= 3 else file_path.stem.lower()


def allocate_balanced_samples(
    variants: list[Path], scene_count: int, samples_per_scene: int, rng: random.Random
) -> list[list[Path]]:
    if not variants:
        raise ValueError("Each experiment group needs at least one variant")

    # Allocate quotas exactly; randomize remainders across variants so usage counts differ by at most one.
    total_draws = scene_count * samples_per_scene
    base_count, remainder = divmod(total_draws, len(variants))
    shuffled_variants = list(variants)
    rng.shuffle(shuffled_variants)
    remaining = {
        variant: base_count + (index < remainder)
        for index, variant in enumerate(shuffled_variants)
    }

    samples: list[list[Path]] = []
    for _ in range(scene_count):
        scene_samples: list[Path] = []
        for _ in range(samples_per_scene):
            # Avoid repeats within a scenario when possible; pet has only three variants,
            # but a 36-task scenario needs six tasks per category, so repeats are allowed when necessary.
            available = [
                variant
                for variant, count in remaining.items()
                if count > 0 and variant not in scene_samples
            ]
            if not available:
                available = [variant for variant, count in remaining.items() if count > 0]
            if not available:
                raise RuntimeError("Unable to allocate the requested tasks")
            rng.shuffle(available)
            available.sort(key=lambda variant: remaining[variant], reverse=True)
            selected = available[0]
            remaining[selected] -= 1
            scene_samples.append(selected)
        samples.append(scene_samples)
    return samples


def infer_tasks_per_scene(scenario: dict[str, object]) -> int:
    configured_count = scenario.get("tasksPerScene")
    if isinstance(configured_count, int) and configured_count > 0:
        return configured_count
    files = scenario.get("files")
    if isinstance(files, list) and files:
        return len(files)
    raise SystemExit("Existing scenario must contain a non-empty files list")


def make_scenario_set(
    scenarios: list[dict[str, object]], tasks_per_scene: int, random_seed: object | None
) -> dict[str, object]:
    return {
        "tasksPerScene": tasks_per_scene,
        "sceneCount": len(scenarios),
        "randomSeed": random_seed,
        "scenarios": scenarios,
    }


def load_existing_scenario_sets(
    output_path: Path,
) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Existing output is not valid JSON: {output_path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Existing output must be a JSON object: {output_path}")

    raw_sets = payload.get("scenarioSets")
    if isinstance(raw_sets, list) and any(
        isinstance(item, dict) and isinstance(item.get("scenarios"), list)
        for item in raw_sets
    ):
        scenario_sets = []
        for set_index, scenario_set in enumerate(raw_sets, start=1):
            if not isinstance(scenario_set, dict):
                raise SystemExit(f"Existing scenario set #{set_index} must be an object")
            raw_scenarios = scenario_set.get("scenarios")
            if not isinstance(raw_scenarios, list) or any(
                not isinstance(scenario, dict) for scenario in raw_scenarios
            ):
                raise SystemExit(f"Existing scenario set #{set_index} has invalid scenarios")
            configured_count = scenario_set.get("tasksPerScene")
            if not isinstance(configured_count, int) or configured_count < 1:
                raise SystemExit(f"Existing scenario set #{set_index} has invalid tasksPerScene")
            scenario_sets.append(
                make_scenario_set(
                    raw_scenarios,
                    configured_count,
                    scenario_set.get("randomSeed", payload.get("randomSeed")),
                )
            )
        return scenario_sets


    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or any(not isinstance(scenario, dict) for scenario in scenarios):
        raise SystemExit(f"Existing output must contain scenarioSets or scenarios: {output_path}")
    grouped: dict[int, list[dict[str, object]]] = {}
    for scenario in scenarios:
        grouped.setdefault(infer_tasks_per_scene(scenario), []).append(scenario)
    return [
        make_scenario_set(group_scenarios, tasks_per_scene, payload.get("randomSeed"))
        for tasks_per_scene, group_scenarios in grouped.items()
    ]


def upsert_scenario_set(
    scenario_sets: list[dict[str, object]], new_set: dict[str, object]
) -> list[dict[str, object]]:
    target_tasks_per_scene = new_set["tasksPerScene"]
    replaced = False
    updated_sets = []
    for scenario_set in scenario_sets:
        if scenario_set.get("tasksPerScene") != target_tasks_per_scene:
            updated_sets.append(scenario_set)
            continue
        if not replaced:
            # Detect overlap by tasksPerScene only; update sceneCount from this generation.
            updated_sets.append(new_set)
            replaced = True
    if replaced:
        return updated_sets
    # Append new tasksPerScene values at the end.
    return scenario_sets + [new_set]


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.scene_count < 1:
        raise SystemExit("--scene-count must be >= 1")
    if args.tasks_per_scene < 1 or args.tasks_per_scene % len(REQUIRED_GROUPS) != 0:
        raise SystemExit(
            f"--tasks-per-scene must be a positive multiple of {len(REQUIRED_GROUPS)}"
        )
    tasks_per_group = args.tasks_per_scene // len(REQUIRED_GROUPS)

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    grouped: dict[str, list[Path]] = {group: [] for group in REQUIRED_GROUPS}
    for file_path in sorted(source_dir.glob("*.json")):
        group_name = extract_group_name(file_path)
        if group_name in grouped:
            grouped[group_name].append(file_path)

    missing = [group for group, variants in grouped.items() if not variants]
    if missing:
        raise SystemExit(f"Groups without variants: {', '.join(missing)}")

    rng = random.Random(args.seed)
    samples_by_group = {
        group: allocate_balanced_samples(variants, args.scene_count, tasks_per_group, rng)
        for group, variants in grouped.items()
    }
    scenarios = []
    for scene_index in range(args.scene_count):
        # Interleave categories by adding one task of each of the six categories per round.
        files = [
            samples_by_group[group][scene_index][sample_index].name
            for sample_index in range(tasks_per_group)
            for group in REQUIRED_GROUPS
        ]
        scenario_name = (
            f"scenario_{args.tasks_per_scene}_{scene_index + 1:03d}"
            if args.tasks_per_scene != len(REQUIRED_GROUPS) * 2
            else f"scenario_{scene_index + 1:03d}"
        )
        scenarios.append(
            {
                "name": scenario_name,
                "tasksPerScene": args.tasks_per_scene,
                "tasksPerGroup": tasks_per_group,
                "randomSeed": args.seed,
                "files": files,
            }
        )

    new_set = make_scenario_set(scenarios, args.tasks_per_scene, args.seed)
    scenario_sets = load_existing_scenario_sets(output_path)

    for scenario_set in scenario_sets:
        historical_seed = scenario_set.get("randomSeed")
        if not isinstance(historical_seed, int):
            historical_seed = args.seed
        scenario_set["randomSeed"] = historical_seed
        scenario_set["scenarios"] = [
            {**scenario, "randomSeed": scenario.get("randomSeed", historical_seed)}
            for scenario in scenario_set["scenarios"]
        ]
    scenario_sets = upsert_scenario_set(scenario_sets, new_set)

    # Store scenarios per tasksPerScene; sceneCount records the current generation count.
    payload = {
        "scenarioSets": scenario_sets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {args.scene_count} replay scenarios: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
