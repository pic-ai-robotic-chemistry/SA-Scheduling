import collections
import time

from ortools.sat.python import cp_model

from algorithm.utils.cp_model_core import (
    OptionalTask,
    Task,
    add_makespan_variable,
    build_cp_model_core,
    extract_ws_table as _extract_ws_table,
)
from algorithm.utils.cp_solver_utils import configure_cp_solver_workers
from algorithm.utils.green_objective import (
    GREEN_TARIFF_CYCLE_MINUTES,
    OBJECTIVE_LIMIT_MAX_TIME_SECONDS,
    _build_price_segments_for_horizon as _shared_build_price_segments_for_horizon,
    _get_green_time_value_info as _shared_get_green_time_value_info,
    add_green_objective_terms,
    compute_green_schedule_horizon,
)


# =========================================================
# 1) Parse the RL solution.
# =========================================================
def parse_rl_ws(ws_table):
    """
    Convert the RL output `ws_table` into a `{(job_id, task_id): (machine, start, duration)}` mapping.

    Args:
    ws_table: Schedule table extracted from RL or CP, grouped by machine.

    Returns:
    rl_map: Lightweight index structure shared by later LNS / probe / batch hints.
    """
    rl_map = {}
    for m_id, tasks in ws_table.items():
        for t in tasks:
            job_id = int(t["expr_no"])
            task_id = int(t["fjspb_index"])
            rl_map[(job_id, task_id)] = (int(m_id), int(t["start_time"]), int(t["duration"]))
    return rl_map


def _sum_max_operation_durations(jobs_data):
    """Sum each operation’s maximum processing time across its candidate machines to bound the schedule tail."""
    return sum(
        max(durations) for job in jobs_data["detail"].values() for _, durations, _ in job if durations
    )


def _max_indexed_value(value):
    """Support release information in list/dict form and extract the largest nonempty time value."""
    if not value:
        return 0
    values = value.values() if isinstance(value, dict) else value
    return max((int(v) for v in values), default=0)


def _max_rl_end_from_map(rl_map):
    """Extract the latest end time from the RL/incumbent map to ensure the horizon covers the known feasible solution."""
    if not rl_map:
        return 0
    return max((int(start) + int(duration) for _, start, duration in rl_map.values()), default=0)


def _compute_horizon(jobs_data, rl_map=None, max_rl_end=None):
    """
    Compute the time upper bound from jobs_data using the same horizon formula as cp_green_or.
    Upper bound = maximum release + the sum of each operation’s maximum processing time + one full peak/off-peak cycle.
    """
    horizon, _release_upper, _processing_upper = compute_green_schedule_horizon(jobs_data)
    return int(horizon)


# =========================================================
# 2) Build the model (variables and constraints).
# =========================================================
def build_model(
    jobs_data,
    beta=None,
    scale_indicator=60,
    include_objective_terms=True,
    incumbent_rl_map=None,
):
    """
    Build a CP-SAT model.
    With `include_objective_terms=True`, build the full green model;
    with `include_objective_terms=False`, build only scheduling constraints and a makespan skeleton for quick probe diagnostics.

    Shared variables and base constraints are built by `algorithm.utils.cp_model_core.build_cp_model_core`.
    """
    horizon = _compute_horizon(jobs_data, rl_map=incumbent_rl_map)
    (
        model,
        all_tasks,
        tasks_to_machine,
        machines_list,
        batch_capacities,
        batch_context,
    ) = build_cp_model_core(
        jobs_data,
        horizon=horizon,
        lock=True,
        rightnow=True,
        require_lock_info=True,
        require_rightnow_info=True,
    )

    if not include_objective_terms:
        # The probe checks scheduling feasibility of the fixed RL solution without green overlap variables.
        beta = 1
        carbon_emission = model.NewConstant(0)
        makespan_var = add_makespan_variable(model, jobs_data, all_tasks, horizon)
        return (
            model,
            all_tasks,
            tasks_to_machine,
            machines_list,
            batch_capacities,
            beta,
            makespan_var,
            carbon_emission,
            batch_context,
        )

    # Preserve LNS creation order: green overlap variables first, then makespan.
    carbon_emission, green_overlap_hints = add_green_objective_terms(
        model,
        jobs_data,
        batch_context["machine_batches"],
        horizon,
        scale_indicator=scale_indicator,
    )
    if beta is None:
        raise ValueError("beta must be provided by Scheduler for green objective build_model")

    makespan_var = add_makespan_variable(model, jobs_data, all_tasks, horizon)
    batch_context["green_overlap_hints"] = green_overlap_hints
    return (
        model,
        all_tasks,
        tasks_to_machine,
        machines_list,
        batch_capacities,
        beta,
        makespan_var,
        carbon_emission,
        batch_context,
    )



def apply_neighborhood_guidance(
    model,
    incumbent_map,
    tasks_to_machine,
    all_tasks,
    relaxed_tasks=None,
    delta_fixed=0,
    delta_relaxed=None,
):
    """
    Guide the LNS neighborhood: fixed tasks keep the incumbent machine, while relaxed tasks may change machines.

    Args:
    incumbent_map: `{(job_id, task_id): (machine, start, duration)}`, from RL or the current best incumbent.
    relaxed_tasks: Tasks allowed to change machines or be rescheduled; if empty, fall back to the original fixed-machine, time-window behavior.
    delta_fixed: Start-time window radius for fixed tasks.
    delta_relaxed: Start-time window radius for relaxed tasks; if None, do not further restrict the start window.
    """
    relaxed_tasks = set(relaxed_tasks or [])
    for (job_id, task_id), m_optional_tasks in tasks_to_machine.items():
        if (job_id, task_id) not in incumbent_map:
            continue
        m_rl, s_rl, d_rl = incumbent_map[(job_id, task_id)]
        task_key = (job_id, task_id)
        if task_key not in relaxed_tasks:
            # Fixed tasks retain their machines and use narrow time windows.
            for m_id, opt in m_optional_tasks.items():
                model.Add(opt.presence == (1 if m_id == m_rl else 0))
            model.Add(all_tasks[job_id, task_id].start >= s_rl - delta_fixed)
            model.Add(all_tasks[job_id, task_id].start <= s_rl + delta_fixed)
        elif delta_relaxed is not None:
            # Relaxed tasks may change machines but can retain broad time windows to limit the search space.
            model.Add(all_tasks[job_id, task_id].start >= s_rl - delta_relaxed)
            model.Add(all_tasks[job_id, task_id].start <= s_rl + delta_relaxed)

        # Hint: guide search toward the incumbent solution.
        model.AddHint(all_tasks[job_id, task_id].start, s_rl)
        model.AddHint(all_tasks[job_id, task_id].end, s_rl + d_rl)
        if all_tasks[job_id, task_id].duration is not None:
            model.AddHint(all_tasks[job_id, task_id].duration, d_rl)
        for m_id, opt in m_optional_tasks.items():
            model.AddHint(opt.presence, 1 if m_id == m_rl else 0)
            if m_id == m_rl:
                model.AddHint(opt.start, s_rl)
                model.AddHint(opt.end, s_rl + d_rl)


def _build_rl_batch_groups(jobs_data, rl_map):
    """
    Group the RL solution using CP batch semantics: tasks on the same machine with the same start/end and param form one batch.

    Args:
    jobs_data: Used to read each operation’s `param`.
    rl_map: Lightweight mapping of the RL solution.

    Returns:
    `{machine_id: [group, ...]}`, where each group contains `start/end/duration/param/tasks`.
    """
    groups_by_machine = collections.defaultdict(dict)
    for job_id, job in jobs_data["detail"].items():
        for task_id, (_machines, _durations, task_param) in enumerate(job):
            task_key = (job_id, task_id)
            if task_key not in rl_map:
                continue
            m_id, start, duration = rl_map[task_key]
            end = start + duration
            batch_key = (start, end, task_param)
            groups_by_machine[m_id].setdefault(batch_key, []).append(task_key)

    sorted_groups = {}
    for m_id, groups in groups_by_machine.items():
        sorted_groups[m_id] = [
            {"start": key[0], "end": key[1], "duration": key[1] - key[0], "param": key[2], "tasks": tasks}
            for key, tasks in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
        ]
    return sorted_groups


def _build_price_segments_for_horizon(jobs_data, horizon_upper):
    """Generate price segments over [0, horizon_upper) using the current peak/off-peak tariff configuration."""
    return _shared_build_price_segments_for_horizon(jobs_data, horizon_upper)

def _get_green_time_value_info(jobs_data, scale_indicator=60):
    """Collect green-objective time-value settings for beta calculation, fixed-solution evaluation, and log reuse."""
    return _shared_get_green_time_value_info(jobs_data, scale_indicator)

def _evaluate_fixed_solution(jobs_data, rl_map, ws_table, beta, scale_indicator=60):
    """Evaluate makespan, carbon, and objective for a fixed RL/incumbent solution directly, without an additional green solve."""
    machine_power_info = {m_id: p for m_id, p in enumerate(jobs_data["info"]["machine_power_info"])}
    makespan = _max_rl_end_from_map(rl_map)
    price_segments = _build_price_segments_for_horizon(jobs_data, makespan)
    batch_groups = _build_rl_batch_groups(jobs_data, rl_map)
    carbon = 0
    for m_id, groups in batch_groups.items():
        power_int = int(round(machine_power_info.get(m_id, 1000.0)))
        for group in groups:
            start = int(group["start"])
            end = int(group["end"])
            for seg_start, seg_end, price in price_segments:
                overlap = max(0, min(end, seg_end) - max(start, seg_start))
                if overlap > 0:
                    carbon += overlap * int(round(price * power_int / scale_indicator))
    machines_list = sorted({int(m) for job in jobs_data["detail"].values() for machines, _, _ in job for m in machines})
    batch_capacities = {m_idx: capacity for m_idx, capacity in enumerate(jobs_data["info"]["machine_capacities"])}
    return {
        "obj": beta * makespan + carbon,
        "makespan": makespan,
        "carbon": carbon,
        "ws_table": ws_table,
        "machines_list": machines_list,
        "batch_capacities": batch_capacities,
        "beta": beta,
        "delta": None,
    }


def _configure_solver_timebox(solver, time_limit_sec, cpu_threads="full"):
    """Unify solver time-limit and search-thread parameters."""
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    return configure_cp_solver_workers(solver, cpu_threads)


def apply_rl_batch_hints(
    model,
    jobs_data,
    rl_map,
    tasks_to_machine,
    batch_context,
    fix=False,
    stage_tag="optimize",
    logger=None,
):
    """
    Add hints to CP-SAT batch and is_in_batch variables using the RL batch groups.
    In probe mode, `fix=True` also fixes these groups to check whether the complete RL solution can be extended.

    Args:
    model: Model to solve.
    jobs_data: Used to rebuild the RL batch groups.
    rl_map: Lightweight mapping of the RL solution.
    tasks_to_machine: Operation-to-machine optional-task mapping in the current model.
    batch_context: Batch context returned by `build_model()`, including batch intervals and membership variables.
    fix:
        False: Add hints only; do not shrink the feasible region.
        True: Fix the RL batch groups as hard constraints for probe / complete-solution validation.
    stage_tag:
        Stage name used in logs; the main flow uses `probe-green-full-fixed` / `optimize`.
    """
    machine_batches = batch_context["machine_batches"]
    task_to_batch = batch_context["task_to_batch"]
    batch_groups = _build_rl_batch_groups(jobs_data, rl_map)

    hinted_batches = 0
    for m_id, batch_list in machine_batches.items():
        groups = batch_groups.get(m_id, [])
        for batch_idx, batch_var in enumerate(batch_list):
            if batch_idx < len(groups):
                group = groups[batch_idx]
                hinted_batches += 1
                model.AddHint(batch_var.presence, 1)
                model.AddHint(batch_var.start, group["start"])
                model.AddHint(batch_var.end, group["end"])
                if batch_var.duration is not None:
                    model.AddHint(batch_var.duration, group["duration"])
                if fix:
                    model.Add(batch_var.presence == 1)
                    model.Add(batch_var.start == group["start"])
                    model.Add(batch_var.end == group["end"])
                    if batch_var.duration is not None:
                        model.Add(batch_var.duration == group["duration"])
            else:
                model.AddHint(batch_var.presence, 0)
                # Use zero hints for absent candidate batches to derive green overlap hints directly.
                model.AddHint(batch_var.start, 0)
                model.AddHint(batch_var.end, 0)
                if batch_var.duration is not None:
                    model.AddHint(batch_var.duration, 0)
                if fix:
                    # The fully fixed fast-check also fixes time variables of empty batches;
                    # otherwise, batch start/end values remain free even with fixed green auxiliary variables.
                    model.Add(batch_var.presence == 0)
                    model.Add(batch_var.start == 0)
                    model.Add(batch_var.end == 0)
                    if batch_var.duration is not None:
                        model.Add(batch_var.duration == 0)
                    continue

        for group_idx, group in enumerate(groups):
            for task_key in group["tasks"]:
                if task_key not in task_to_batch or m_id not in task_to_batch[task_key]:
                    continue
                for batch_idx, is_in_batch in enumerate(task_to_batch[task_key][m_id]):
                    selected = 1 if batch_idx == group_idx else 0
                    model.AddHint(is_in_batch, selected)
                    if fix:
                        model.Add(is_in_batch == selected)

    for task_key, machine_map in task_to_batch.items():
        if task_key not in rl_map:
            continue
        m_rl, _, _ = rl_map[task_key]
        for m_id, batch_vars in machine_map.items():
            if m_id == m_rl:
                continue
            for is_in_batch in batch_vars:
                model.AddHint(is_in_batch, 0)
                if fix:
                    model.Add(is_in_batch == 0)

    if stage_tag.startswith("probe"):
        log_func = logger.info_orange
    else:
        log_func = logger.info_blue

    if fix:
        log_func(f"[lns][{stage_tag}][batch-hint] fix=true")
    else:
        log_func(f"[lns][{stage_tag}][batch-hint] mode=hint batches={hinted_batches}")


def apply_computed_green_hints(
    model,
    jobs_data,
    incumbent_map,
    makespan_var,
    carbon_emission,
    batch_context,
    fix=False,
    stage_tag="computed-green",
    logger=None,
):
    """
    Compute auxiliary-variable hints / fixed values for the full green model directly from the known incumbent schedule.

    Reuse the batch numbering rules from `_build_rl_batch_groups()` and `build_model()`
    to compute min_end/max_start/diff/overlap for each candidate batch and tariff segment.
    Also hint the total makespan and carbon, avoiding another CP-SAT run just to fill in these deterministic values.
    With `fix=True`, hard-fix these green auxiliary variables for the full-green-model fast check.
    """
    batch_groups = _build_rl_batch_groups(jobs_data, incumbent_map)
    green_overlap_hints = batch_context.get("green_overlap_hints", [])
    total_carbon = 0
    present_overlap_count = 0

    for hint_context in green_overlap_hints:
        machine_id = hint_context["machine_id"]
        batch_idx = hint_context["batch_idx"]
        groups = batch_groups.get(machine_id, [])
        if batch_idx < len(groups):
            group = groups[batch_idx]
            batch_start = int(group["start"])
            batch_end = int(group["end"])
            present_overlap_count += 1
        else:
            # Match the start=end=0 hints for absent batches in apply_rl_batch_hints().
            batch_start = 0
            batch_end = 0

        seg_start = int(hint_context["seg_start"])
        seg_end = int(hint_context["seg_end"])
        min_end = min(batch_end, seg_end)
        max_start = max(batch_start, seg_start)
        diff = min_end - max_start
        overlap = max(0, diff) if batch_idx < len(groups) else 0

        model.AddHint(hint_context["min_end"], min_end)
        model.AddHint(hint_context["max_start"], max_start)
        model.AddHint(hint_context["diff"], diff)
        model.AddHint(hint_context["overlap"], overlap)
        if fix:
            # Fix green linearization variables so fast-check validates the RL schedule, batches, and carbon segments
            # together in the complete green model.
            model.Add(hint_context["min_end"] == min_end)
            model.Add(hint_context["max_start"] == max_start)
            model.Add(hint_context["diff"] == diff)
            model.Add(hint_context["overlap"] == overlap)
        total_carbon += overlap * int(hint_context["carbon_coefficient"])

    makespan = _max_rl_end_from_map(incumbent_map)
    model.AddHint(makespan_var, makespan)
    model.AddHint(carbon_emission, total_carbon)
    if fix:
        # Also hard-fix makespan and carbon, the aggregate green objective variables.
        model.Add(makespan_var == makespan)
        model.Add(carbon_emission == total_carbon)
    if logger is not None:
        log_func = logger.info_orange if stage_tag.startswith("probe") else logger.info_blue
        mode = "fix" if fix else "hint"
        log_func(
            f"[lns][{stage_tag}] mode={mode} batches={sum(len(groups) for groups in batch_groups.values())} "
            f"overlap_values={present_overlap_count} makespan={makespan} carbon={total_carbon}"
        )
    return {"makespan": makespan, "carbon": total_carbon}


def apply_fixed_rl_solution(model, rl_map, tasks_to_machine, all_tasks, jobs_data=None):
    """
    Fix the RL machine choices and primary time variables as hard constraints.
    If the RL solution is missing tasks or uses a machine outside the candidate set, add an explicit contradiction so the probe returns INFEASIBLE.

    Args:
    model: Current probe or full-green-model fast-check model.
    rl_map: Lightweight mapping of the RL solution.
    tasks_to_machine: Operation-to-machine optional-task mapping in the current model.
    all_tasks: Mapping of primary task intervals in the current model.
    jobs_data: Optional; when provided, also fix optional start/end values for unselected machines to valid placeholders.
    """
    machine_available_times = {}
    if jobs_data is not None:
        machine_available_times = {
            m_id: int(time_value)
            for m_id, time_value in enumerate(jobs_data.get("info", {}).get("machines_available_time", []))
        }
    for task_key, m_optional_tasks in tasks_to_machine.items():
        if task_key not in rl_map:
            model.Add(0 == 1)
            continue
        duration_by_machine = {}
        if jobs_data is not None:
            job_id, task_id = task_key
            machines, durations, _task_param = jobs_data["detail"][job_id][task_id]
            duration_by_machine = {m_id: int(duration) for m_id, duration in zip(machines, durations)}
        m_rl, s_rl, duration_rl = rl_map[task_key]
        if m_rl not in m_optional_tasks:
            model.Add(0 == 1)
            continue
        end_rl = s_rl + duration_rl
        for m_id, opt in m_optional_tasks.items():
            selected = 1 if m_id == m_rl else 0
            model.Add(opt.presence == selected)
            model.AddHint(opt.presence, selected)
            if selected:
                # The fully fixed fast-check fixes optional interval times on selected machines too.
                model.Add(opt.start == s_rl)
                model.Add(opt.end == end_rl)
                model.AddHint(opt.start, s_rl)
                model.AddHint(opt.end, end_rl)
            elif jobs_data is not None:
                # Optional intervals on unselected machines do not affect the schedule,
                # but their start/end remain model variables; fix them to placeholders satisfying availability.
                placeholder_start = machine_available_times.get(m_id, 0)
                placeholder_end = placeholder_start + duration_by_machine.get(m_id, 0)
                model.Add(opt.start == placeholder_start)
                model.Add(opt.end == placeholder_end)
                model.AddHint(opt.start, placeholder_start)
                model.AddHint(opt.end, placeholder_end)
        model.Add(all_tasks[task_key].start == s_rl)
        model.AddHint(all_tasks[task_key].start, s_rl)
        model.Add(all_tasks[task_key].end == end_rl)
        model.AddHint(all_tasks[task_key].end, end_rl)
        if all_tasks[task_key].duration is not None:
            model.Add(all_tasks[task_key].duration == duration_rl)
            model.AddHint(all_tasks[task_key].duration, duration_rl)


def _log_fixed_rl_release_diagnostics(jobs_data, rl_map, logger):
    """Record differences between the first RL operation and release/rightnow constraints before the full-fixed fast check."""
    if logger is None:
        return

    whole_info = jobs_data.get("info", {})
    job_avail_time = whole_info.get("job_avail_time", {})
    job_rightnow = whole_info.get("job_rightnow", {})

    # Scheduling normally supplies a dict with consecutive integer keys; also accept lists and string keys
    # so diagnostics cannot alter or interrupt fast-check.
    def indexed_value(values, index, default=0):
        if isinstance(values, dict):
            return values.get(index, values.get(str(index), default))
        if isinstance(values, (list, tuple)) and 0 <= int(index) < len(values):
            return values[int(index)]
        return default

    rightnow_count = 0
    rightnow_mismatch_count = 0
    release_early_count = 0
    missing_first_operation_count = 0
    for job_id in jobs_data.get("detail", {}):
        required_start = int(indexed_value(job_avail_time, job_id, 0))
        must_start_now = bool(indexed_value(job_rightnow, job_id, False))
        rl_entry = rl_map.get((job_id, 0))
        if rl_entry is None:
            missing_first_operation_count += 1
            if must_start_now:
                rightnow_count += 1
                rightnow_mismatch_count += 1
            logger.info_orange(
                f"[lns][probe-green-full-fixed][release-check] job={job_id} "
                f"rightnow={must_start_now} required_start={required_start} rl_start=missing "
                f"delta=missing matches=false"
            )
            continue

        _machine_id, rl_start, _duration = rl_entry
        rl_start = int(rl_start)
        delta = rl_start - required_start
        release_early = delta < 0
        rightnow_matches = (not must_start_now) or delta == 0
        if must_start_now:
            rightnow_count += 1
            rightnow_mismatch_count += int(not rightnow_matches)
        release_early_count += int(release_early)
        if must_start_now or release_early:
            logger.info_orange(
                f"[lns][probe-green-full-fixed][release-check] job={job_id} "
                f"rightnow={must_start_now} required_start={required_start} rl_start={rl_start} "
                f"delta={delta} matches={str(rightnow_matches and not release_early).lower()}"
            )

    logger.info_orange(
        f"[lns][probe-green-full-fixed][release-check-summary] "
        f"rightnow_jobs={rightnow_count} rightnow_mismatches={rightnow_mismatch_count} "
        f"release_early_mismatches={release_early_count} "
        f"missing_first_operations={missing_first_operation_count}"
    )


def fast_check_fixed_rl_green_solution(
    jobs_data,
    ws_table_rl,
    rl_map,
    beta,
    scale_indicator=60,
    time_limit_sec=1.0,
    logger=None,
    cpu_threads="full",  # Fast-check and formal LNS use the caller's CP-SAT worker count.
):







    _log_fixed_rl_release_diagnostics(jobs_data, rl_map, logger)
    (
        model,
        all_tasks,
        tasks_to_machine,
        machines_list,
        batch_capacities,
        model_beta,
        makespan_var,
        carbon_emission,
        batch_context,
    ) = build_model(
        jobs_data,
        beta=beta,
        scale_indicator=scale_indicator,
        include_objective_terms=True,
        incumbent_rl_map=rl_map,
    )

    apply_fixed_rl_solution(
        model=model,
        rl_map=rl_map,
        tasks_to_machine=tasks_to_machine,
        all_tasks=all_tasks,
        jobs_data=jobs_data,
    )
    apply_rl_batch_hints(
        model=model,
        jobs_data=jobs_data,
        rl_map=rl_map,
        tasks_to_machine=tasks_to_machine,
        batch_context=batch_context,
        fix=True,
        stage_tag="probe-green-full-fixed",
        logger=logger,
    )
    green_values = apply_computed_green_hints(
        model=model,
        jobs_data=jobs_data,
        incumbent_map=rl_map,
        makespan_var=makespan_var,
        carbon_emission=carbon_emission,
        batch_context=batch_context,
        fix=True,
        stage_tag="probe-green-full-fixed",
        logger=logger,
    )

    objective_expr = model_beta * makespan_var + carbon_emission
    model.Minimize(objective_expr)

    solver = cp_model.CpSolver()
    _configure_solver_timebox(
        solver,
        max(0.1, float(time_limit_sec)),
        cpu_threads=cpu_threads,
    )
    started_at = time.monotonic()
    status = solver.Solve(model)
    elapsed = time.monotonic() - started_at
    status_name = solver.StatusName(status).lower()

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ws_table = extract_ws_table(solver, jobs_data, tasks_to_machine)
        ret = {
            "obj": solver.ObjectiveValue(),
            "makespan": solver.Value(makespan_var),
            "carbon": solver.Value(carbon_emission),
            "ws_table": ws_table,
            "machines_list": machines_list,
            "batch_capacities": batch_capacities,
            "beta": model_beta,
            "delta": None,
            "source": "rl_incumbent",
            "solver_status": status_name,  # Keep the fast-check status to identify RL incumbent feasibility results.
            "solver_best_bound": solver.BestObjectiveBound(),
            "hint_values": _collect_solution_hint_by_index(model, solver),
        }
        if logger is not None:
            logger.info_green(
                f"[lns][probe-green-full-fixed][fast-check] status={status_name} "
                f"complete_hint=true obj={ret['obj']} makespan={ret['makespan']} "
                f"carbon={ret['carbon']} elapsed={elapsed:.3f}s"
            )
        return {
            "status": status,
            "status_name": status_name,
            "extendable": "true",
            "ret": ret,
            "hint_values": ret["hint_values"],
        }

    extendable = "false" if status == cp_model.INFEASIBLE else "unknown"
    fallback_ret = _evaluate_fixed_solution(
        jobs_data=jobs_data,
        rl_map=rl_map,
        ws_table=ws_table_rl,
        beta=beta,
        scale_indicator=scale_indicator,
    )
    fallback_ret["source"] = "rl_incumbent"
    if logger is not None:
        logger.info_orange(
            f"[lns][probe-green-full-fixed][fast-check] status={status_name} "
            f"fixed_rl_solution_extendable={extendable} complete_hint=false "
            f"time_limit={max(0.1, float(time_limit_sec)):.2f}s elapsed={elapsed:.3f}s "
            f"computed_makespan={green_values['makespan']} computed_carbon={green_values['carbon']}"
        )
    return {
        "status": status,
        "status_name": status_name,
        "extendable": extendable,
        "ret": fallback_ret,
        "hint_values": None,
    }


def _collect_solution_hint_by_index(model, solver):
    """
    Collect all variable values from a solution accepted by CP-SAT.
    This returns the raw values for a “complete-solution hint”, which will later be injected into the formal optimization model
    so formal optimization starts with a complete assignment already accepted by CP.
    """
    values = []
    for var_idx in range(len(model.Proto().variables)):
        var = model.GetIntVarFromProtoIndex(var_idx)
        values.append(solver.Value(var))
    return values


def _apply_solution_hint_by_index(model, hint_values):
    """
    Inject the “complete-solution hint” by variable proto index.
    This first calls ClearHints(), then writes all variable values from the complete solution back into the model,
    replacing the scattered primary-task / batch hints added earlier.
    """
    model.ClearHints()
    for var_idx, value in enumerate(hint_values):
        if var_idx >= len(model.Proto().variables):
            break
        model.AddHint(model.GetIntVarFromProtoIndex(var_idx), int(value))
    return {
        "hint_values": len(hint_values),
        "model_variables": len(model.Proto().variables),
        "applied": min(len(hint_values), len(model.Proto().variables)),
    }


# =========================================================
# 4) Extract the solution.
# =========================================================
def extract_ws_table(solver, jobs_data, tasks_to_machine):
    """Extract the scheduler's ws_table format from the shared CP model layer."""
    return _extract_ws_table(solver, jobs_data, tasks_to_machine)

def _is_strictly_better(candidate, incumbent):
    """Unify incumbent comparison: use obj when a green objective is available, and fall back to makespan otherwise."""
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if candidate.get("obj") is not None and incumbent.get("obj") is None:
        return True
    if candidate.get("obj") is not None and incumbent.get("obj") is not None:
        return candidate["obj"] < incumbent["obj"] - 1e-6
    if candidate.get("makespan") is not None and incumbent.get("makespan") is not None:
        return candidate["makespan"] < incumbent["makespan"]
    return False


def _prioritize_delta_list(delta0, delta_list):
    """In multi-round search, try the automatically estimated delta0 first, then expand to smaller and larger windows."""
    candidates = set(int(delta) for delta in delta_list)
    candidates.add(int(delta0))
    return tuple(sorted(candidates, key=lambda delta: (0 if delta == delta0 else 1, abs(delta - delta0), delta)))


def _expand_lock_chain(task_key, jobs_data):
    """Treat a lock chain as an atomic neighborhood unit so releasing only part of it does not break the no-wait chain."""
    job_id, task_id = task_key
    locks = jobs_data["info"]["job_lock"][job_id]
    left = task_id
    right = task_id
    while left - 1 >= 0 and bool(locks[left]) and bool(locks[left - 1]):
        left -= 1
    while right + 1 < len(locks) and bool(locks[right]) and bool(locks[right + 1]):
        right += 1
    return {(job_id, idx) for idx in range(left, right + 1)}


def _task_param_map(jobs_data):
    """Pre-fetch task -> param mappings for reuse by the batch neighborhood."""
    param_map = {}
    for job_id, job in jobs_data["detail"].items():
        for task_id, (_machines, _durations, task_param) in enumerate(job):
            param_map[(job_id, task_id)] = task_param
    return param_map


def _select_batch_repack_tasks(jobs_data, incumbent_map, limit=12):
    """Release tasks on the same machine with the same parameter and nearby times, allowing batches to be rebuilt and machines to change locally."""
    param_map = _task_param_map(jobs_data)
    grouped = collections.defaultdict(list)
    for task_key, (m_id, start, duration) in incumbent_map.items():
        grouped[(m_id, param_map.get(task_key))].append((task_key, int(start), int(duration)))

    best_group = []
    best_span = None
    for (_m_id, _param), tasks in grouped.items():
        if len(tasks) < 2:
            continue
        tasks = sorted(tasks, key=lambda item: (item[1], item[2]))
        span = tasks[-1][1] - tasks[0][1]
        if best_span is None or len(tasks) > len(best_group) or (len(tasks) == len(best_group) and span < best_span):
            best_group = tasks
            best_span = span

    selected = set()
    for task_key, _start, _duration in best_group[:limit]:
        selected.update(_expand_lock_chain(task_key, jobs_data))
        if len(selected) >= limit:
            break
    return selected


def _build_neighborhood_rounds(jobs_data, incumbent_map, delta0, delta_list, total_time_limit=None):

    if total_time_limit is not None and float(total_time_limit) <= 10:
        batch_limit = 0
        max_rounds = 2
    else:
        batch_limit = 12
        max_rounds = 4

    prioritized_deltas = _prioritize_delta_list(delta0, delta_list)
    rounds = []

    rounds.append({"delta": prioritized_deltas[0], "name": "time_window", "relaxed_tasks": set()})
    batch_tasks = _select_batch_repack_tasks(jobs_data, incumbent_map, limit=batch_limit) if batch_limit > 0 else set()
    if batch_tasks and len(rounds) < max_rounds:
        rounds.append({"delta": prioritized_deltas[0], "name": "batch_repack", "relaxed_tasks": batch_tasks})

    for delta in prioritized_deltas[1:]:
        rounds.append({"delta": delta, "name": "time_window", "relaxed_tasks": set()})
        if len(rounds) >= max_rounds:
            break
    return tuple(rounds[:max_rounds])


def _should_run_final_full_release(total_time_limit, remaining_time):
    """
    By default, append a full_release round only when enough budget remains.

    This round releases the machines and time windows of all tasks, keeping only the incumbent/full hint as soft guidance.
    Since its search space is much larger than a local neighborhood, require a conservative minimum remaining budget.
    """
    total_time_limit = float(total_time_limit)
    remaining_time = float(remaining_time)
    min_required = max(6.0, min(12.0, total_time_limit * 0.4))
    return remaining_time >= min_required, min_required


# =========================================================
# 5) Adaptive solving.
# =========================================================
# =========================================================
# A) Collect features and compute delta0 / delta_list.
# =========================================================
def _estimate_delta0_and_list(
    jobs_data,
    rl_map,
    multipliers=(0.5, 1.0, 1.5, 2.0),
    alpha=0.8,  # Base coefficient.
    c_lock=0.8,  # Lock-ratio weight.
    relax_factor=2,  # Overall relaxation factor (>1 increases delta).
    delta_min_coef=0.25,  # delta_min = max(5, round(delta_min_coef * avg_dur))
    delta_max_horizon_coef=0.3,  # Horizon coefficient in the delta_max bound.
    delta_max_avgdur_coef=10,  # Average-duration coefficient in the delta_max bound.
):
    """
    Estimate the formal-optimization time window `delta0` from the RL solution’s duration distribution and lock ratio.

    Args:
    jobs_data: Current problem instance.
    rl_map: Lightweight mapping of the RL solution.
    multipliers: Multipliers for generating candidate delta values; formal optimization tries them in priority order.
    alpha/c_lock/relax_factor/...: Heuristic coefficients controlling the width and bounds of delta0.

    Returns:
    `(delta0, delta_list, stats)`, where `stats` supports log diagnostics.
    """
    # 1) Average operation duration avg_dur from the RL solution.
    rl_durations = [d for (_, _, d) in rl_map.values()]
    avg_dur = sum(rl_durations) / len(rl_durations)
    # 2) Lock ratio: adjacent operation pairs with both lock flags set.
    lock_pairs = 0
    adjacent_pairs = 0
    for job_id, job in jobs_data["detail"].items():
        for t in range(len(job) - 1):
            adjacent_pairs += 1
            if jobs_data["info"]["job_lock"][job_id][t] and jobs_data["info"]["job_lock"][job_id][t + 1]:
                lock_pairs += 1
    lock_ratio = lock_pairs / max(1, adjacent_pairs)
    # 3) Estimate horizon H for clipping delta.
    horizon_upper = _compute_horizon(jobs_data, rl_map=rl_map)
    # 4) Delta clipping bounds.
    delta_min = max(5, int(round(delta_min_coef * avg_dur)))
    delta_max = max(
        delta_min + 1,
        int(min(delta_max_horizon_coef * horizon_upper, delta_max_avgdur_coef * avg_dur)),
    )
    # 5) delta0
    raw_delta0 = avg_dur * (alpha + c_lock * lock_ratio) * relax_factor
    delta0 = int(round(raw_delta0))
    delta0 = max(delta_min, min(delta0, delta_max))
    # 6) delta_list
    delta_list = sorted({max(delta_min, min(int(round(m * delta0)), delta_max)) for m in multipliers})
    stats = {
        "avg_dur": avg_dur,
        "lock_ratio": lock_ratio,
        "horizon_upper": horizon_upper,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "raw_delta0": raw_delta0,
        "relax_factor": relax_factor,
        "delta_min_coef": delta_min_coef,
        "delta_max_horizon_coef": delta_max_horizon_coef,
        "delta_max_avgdur_coef": delta_max_avgdur_coef,
    }
    return delta0, tuple(delta_list), stats


# =========================================================
# B) Adaptive solving.
# =========================================================
def fespb_or_green_adaptive_lns(
    jobs_data,
    ws_table_rl,
    beta,
    time_limit=30,  # Total budget in seconds.
    scale_indicator=60,
    fast_check_time_limit=5.0,  # Time limit for fast-check of the fully fixed green model.
    full_release_only_after_incumbent=False,  # If True, skip neighborhoods and run full_release directly after establishing the incumbent.
    logger=None,  # AsyncLog supplied by Scheduler; LNS does not print directly.
    cpu_threads="full",  # CP-SAT workers shared by fast-check and all formal LNS rounds.
):
    """
    Multi-delta optimization: after RL produces a feasible solution, hard-fix it in the full green model for a fast check,
    then use the complete variable hints accepted by CP-SAT to search for improvements in each neighborhood.

    Args:
    jobs_data: Current problem instance.
    ws_table_rl: Schedule table produced by RL.
    time_limit: Total time limit for formal optimization; the full-fixed fast check has a separate small budget.
    scale_indicator: Green-objective scaling parameter.
    fast_check_time_limit: Time limit for the full-fixed green-model fast check.
    full_release_only_after_incumbent:
        If True, after establishing the incumbent, skip time_window / batch_repack
        and use all remaining formal-optimization time for one full_release rerun.
    """

    if time_limit <= 10:
        # Fast-check of a fully fixed green model needs enough time for large instances;
        # keep a short budget, but allow at least 1 second for validating all variables.
        fast_check_time_limit = min(float(fast_check_time_limit), 1.0)
    else:
        fast_check_time_limit = float(fast_check_time_limit)

    # Record each formal CP round's status; final source depends only on strict improvement over RL,
    # not on the fast-check status.
    cp_attempt_diagnostics = []
    initial_incumbent_objective = None

    def _run_once(
        delta,
        time_limit_sec,
        incumbent_map,
        full_solution_hint_values=None,
        neighborhood_name="time_window",
        relaxed_tasks=None,
        full_release=False,
    ):
        """
        Solve one formal optimization round.

        Args:
        delta: Allowed start-time deviation window for primary tasks relative to the RL solution.
        time_limit_sec: Solve time limit for this formal optimization round.
        full_solution_hint_values:
            Complete set of variable values from the full green model; when present, it replaces scattered hints
            so formal optimization starts from a complete solution accepted by CP.
        neighborhood_name:
            Neighborhood used in this round, for distinguishing log entries.
        relaxed_tasks:
            Tasks whose machine choices may be released in this round.
        full_release:
            If True, release the machines and time windows of all tasks and keep only the soft hint,
            effectively performing a warm-started global rerun from the incumbent.
        """
        (
            model,
            all_tasks,
            tasks_to_machine,
            machines_list,
            batch_capacities,
            model_beta,
            makespan_var,
            carbon_emission,
            batch_context,
        ) = build_model(
            jobs_data,
            beta=beta,
            scale_indicator=scale_indicator,
            incumbent_rl_map=incumbent_map,
        )

        relaxed_tasks = set(relaxed_tasks or [])
        effective_delta_relaxed = max(delta * 2, delta)
        if full_release:
            # full_release: relax all tasks and omit start-time windows.
            relaxed_tasks = set(incumbent_map.keys())
            effective_delta_relaxed = None
        apply_neighborhood_guidance(
            model=model,
            incumbent_map=incumbent_map,
            tasks_to_machine=tasks_to_machine,
            all_tasks=all_tasks,
            relaxed_tasks=relaxed_tasks,
            delta_fixed=delta,
            delta_relaxed=effective_delta_relaxed,
        )
        apply_rl_batch_hints(
            model=model,
            jobs_data=jobs_data,
            rl_map=incumbent_map,
            tasks_to_machine=tasks_to_machine,
            batch_context=batch_context,
            fix=False,
            stage_tag="optimize",
            logger=logger,
        )
        if full_solution_hint_values:
            # Replace partial hints with the complete RL hint accepted by the green model to help establish feasibility.
            hint_stats = _apply_solution_hint_by_index(model, full_solution_hint_values)
            logger.info_blue(
                f"[lns][optimize][full-hint] applied={hint_stats['applied']} "
                f"hint_values={hint_stats['hint_values']} model_variables={hint_stats['model_variables']} "
                f"complete={hint_stats['hint_values'] == hint_stats['model_variables']}"
            )
        else:
            # If fast-check provides no complete hint, derive green auxiliary hints directly from the schedule.
            apply_computed_green_hints(
                model=model,
                jobs_data=jobs_data,
                incumbent_map=incumbent_map,
                makespan_var=makespan_var,
                carbon_emission=carbon_emission,
                batch_context=batch_context,
                logger=logger,
            )
        objective_expr = model_beta * makespan_var + carbon_emission
        model.Minimize(objective_expr)

        solver = cp_model.CpSolver()
        _configure_solver_timebox(
            solver,
            max(0.5, float(time_limit_sec)),
            cpu_threads=cpu_threads,
        )
        # This OR-Tools version can trigger a native assertion on the repair_hint/hint_conflict_limit path:
        # `heuristics.fixed_search != nullptr`; use AddHint alone here.
        started_at = time.monotonic()
        status = solver.Solve(model)
        elapsed = time.monotonic() - started_at
        status_name = solver.StatusName(status).lower()
        solver_best_bound = solver.BestObjectiveBound()
        attempt_diagnostics = {
            "solver_status": status_name,
            "solver_best_bound": solver_best_bound,
            "solver_stage": neighborhood_name,
            "bound_scope": "global" if full_release else "neighborhood",
            "cp_candidate_objective": None,
        }
        cp_attempt_diagnostics.append(attempt_diagnostics)
        # Preserve the actual formal CP status even for UNKNOWN or no improvement; for feasible candidates,
        # also record the objective to distinguish equal and worse not_better results.
        delta_log = "free" if full_release else str(delta)
        logger.info_blue(
            f"[lns][optimize][cp-solve] neighborhood={neighborhood_name} relaxed={len(relaxed_tasks)} "
            f"delta={delta_log} status={status_name} "
            f"time_limit={max(1.0, float(time_limit_sec)):.2f}s elapsed={elapsed:.3f}s"
        )

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if full_solution_hint_values:
                # AddHint is soft guidance; it does not register the incumbent as a returned solution.
                # UNKNOWN means this round returned no new CP solution; the caller still retains the checked incumbent.
                logger.info_yellow(
                    f"[lns][optimize][cp-solve] no_cp_solution_returned "
                    f"status={status_name} complete_hint=true incumbent_retained=true"
                )
            return None

        obj = solver.ObjectiveValue()
        makespan = solver.Value(makespan_var)
        carbon = solver.Value(carbon_emission)
        attempt_diagnostics["cp_candidate_objective"] = obj
        ws_table = extract_ws_table(solver, jobs_data, tasks_to_machine)

        return {
            "obj": obj,
            "makespan": makespan,
            "carbon": carbon,
            "ws_table": ws_table,
            "machines_list": machines_list,
            "batch_capacities": batch_capacities,
            "beta": model_beta,
            "delta": None if full_release else delta,
            "neighborhood": neighborhood_name,
            "source": "cp_optimized",
            "solver_status": status_name,  # Dynamic replay records the CP-SAT status for each window.
            "solver_best_bound": solver_best_bound,
            "solver_stage": neighborhood_name,
            "bound_scope": "global" if full_release else "neighborhood",
            "incumbent_objective": initial_incumbent_objective,  # Always refers to the RL incumbent before CP optimization.
            "hint_values": _collect_solution_hint_by_index(model, solver),
        }

    # ---------- Stages 1-3: RL solution, green auxiliary values, and fully fixed fast-check ----------
    # Combine RL preprocessing, lightweight probing, and Python evaluation:
    # accept the RL schedule, derive green auxiliary values, and hard-fix times, machines, batches,
    # and green variables in the full model to obtain a complete CP-SAT-accepted hint.
    rl_map = parse_rl_ws(ws_table_rl)
    green_time_value_info = _get_green_time_value_info(
        jobs_data,
        scale_indicator=scale_indicator,
    )
    logger.info(
        "[lns][green-weight] "
        f"60min_time_value = "
        f"{green_time_value_info['makespan_60min_cost_in_valley_price_units']:.3f} * "
        f"valley_price({green_time_value_info['valley_price']:.3f}) = "
        f"{green_time_value_info['hourly_time_value_yuan']:.3f} yuan, "
        f"beta = {green_time_value_info['beta']}"
    )
    fixed_check = fast_check_fixed_rl_green_solution(
        jobs_data=jobs_data,
        ws_table_rl=ws_table_rl,
        rl_map=rl_map,
        beta=beta,
        scale_indicator=scale_indicator,
        time_limit_sec=fast_check_time_limit,
        logger=logger,
        cpu_threads=cpu_threads,
    )
    full_solution_hint_values = fixed_check.get("hint_values")
    best_ret = fixed_check.get("ret")
    initial_incumbent_objective = best_ret.get("obj") if best_ret is not None else None
    logger.info_green(
        f"[lns][incumbent] source=full_fixed_green_fast_check "
        f"extendable={fixed_check.get('extendable')} complete_hint={bool(full_solution_hint_values)} "
        f"obj={best_ret['obj']} makespan={best_ret['makespan']} carbon={best_ret['carbon']}"
    )
    # ---------- End of stages 1-3 ----------

    # ---------- Stage 4: organize formal optimization rounds ----------
    # Only choose round counts, deltas, and neighborhoods here; Solve has not started.
    delta0, delta_list, stats = _estimate_delta0_and_list(jobs_data, rl_map)
    logger.info(
        f"[lns] delta0={delta0}, delta_list={delta_list}, "
        f"avg_dur={stats['avg_dur']:.2f}, lock_ratio={stats['lock_ratio']:.3f}"
    )
    logger.info(
        f"[lns] stage_time_limits fast_check={fast_check_time_limit:.2f}s "
        f"green_hint=full_fixed_fast_check optimize_total={float(time_limit):.2f}s",
    )

    # ---------- Multiple deltas: seek better incumbents within the formal optimization budget ----------
    if full_release_only_after_incumbent:
        search_rounds = tuple()
        logger.info("[lns] mode=full_release_only_after_incumbent skip_neighborhood_rounds=true")
    else:
        incumbent_map_for_rounds = parse_rl_ws(best_ret["ws_table"]) if best_ret is not None else rl_map
        search_rounds = _build_neighborhood_rounds(
            jobs_data,
            incumbent_map_for_rounds,
            delta0,
            delta_list,
            total_time_limit=time_limit,
        )
    # ---------- End of stage 4 ----------

    # ---------- Stage 5: formal green optimization ----------
    # Optimize the full green objective using the incumbent as a search hint.
    # Prefer the complete solver hint generated by fully fixed fast-check for the initial RL incumbent;
    # otherwise derive green auxiliary hints directly from the incumbent.
    optimize_started_at = time.monotonic()
    for round_idx, round_spec in enumerate(search_rounds, start=1):
        delta = round_spec["delta"]
        neighborhood_name = round_spec["name"]
        relaxed_tasks = round_spec["relaxed_tasks"]
        incumbent_map = parse_rl_ws(best_ret["ws_table"]) if best_ret is not None else rl_map
        elapsed_total = time.monotonic() - optimize_started_at
        remaining_time = float(time_limit) - elapsed_total
        remaining_rounds = len(search_rounds) - round_idx + 1
        if remaining_time <= 0:
            logger.info_yellow(
                f"[lns][round] idx={round_idx} neighborhood={neighborhood_name} delta={delta} "
                "skipped reason=budget_exhausted"
            )
            break
        per_round_limit = max(1.0, remaining_time / max(1, remaining_rounds))
        logger.info(
            f"[lns][round] idx={round_idx}/{len(search_rounds)} neighborhood={neighborhood_name} "
            f"relaxed={len(relaxed_tasks)} delta={delta} time_limit={per_round_limit:.2f}s"
        )
        candidate = _run_once(
            delta=delta,
            time_limit_sec=per_round_limit,
            incumbent_map=incumbent_map,
            full_solution_hint_values=full_solution_hint_values,
            neighborhood_name=neighborhood_name,
            relaxed_tasks=relaxed_tasks,
        )
        if _is_strictly_better(candidate, best_ret):
            best_ret = candidate
            full_solution_hint_values = candidate.get("hint_values")
            logger.info_green(
                f"[lns][round] idx={round_idx} neighborhood={neighborhood_name} delta={delta} improve=true "
                f"obj={candidate['obj']} makespan={candidate['makespan']} carbon={candidate['carbon']}"
            )
        else:
            status_note = (
                "no_cp_solution_returned_hint_incumbent_retained"
                if candidate is None and full_solution_hint_values
                else "no_cp_solution_returned" if candidate is None
                else "not_better"
            )
            logger.info(
                f"[lns][round] idx={round_idx} neighborhood={neighborhood_name} delta={delta} "
                f"improve=false reason={status_note}"
            )

    # By default, append a full_release round after local neighborhoods if enough budget remains.
    # If full_release_only_after_incumbent is enabled, skip neighborhoods and use all remaining budget here.
    elapsed_total = time.monotonic() - optimize_started_at
    remaining_time = float(time_limit) - elapsed_total
    should_run_final_release, min_required = _should_run_final_full_release(time_limit, remaining_time)
    force_final_release = bool(full_release_only_after_incumbent)
    if force_final_release or should_run_final_release:
        incumbent_map = parse_rl_ws(best_ret["ws_table"]) if best_ret is not None else rl_map
        logger.info(
            f"[lns][final-full-release] trigger={'forced' if force_final_release else 'budget_ok'} "
            f"remaining_time={remaining_time:.2f}s threshold={min_required:.2f}s"
        )
        candidate = _run_once(
            delta=delta0,
            time_limit_sec=max(1.0, remaining_time),
            incumbent_map=incumbent_map,
            full_solution_hint_values=full_solution_hint_values,
            neighborhood_name="full_release",
            relaxed_tasks=None,
            full_release=True,
        )
        if _is_strictly_better(candidate, best_ret):
            best_ret = candidate
            full_solution_hint_values = candidate.get("hint_values")
            logger.info_green(
                f"[lns][final-full-release] improve=true obj={candidate['obj']} "
                f"makespan={candidate['makespan']} carbon={candidate['carbon']}"
            )
        else:
            status_note = (
                "no_cp_solution_returned_hint_incumbent_retained"
                if candidate is None and full_solution_hint_values
                else "no_cp_solution_returned" if candidate is None
                else "not_better"
            )
            logger.info(f"[lns][final-full-release] improve=false reason={status_note}")
    else:
        logger.info(
            f"[lns][final-full-release] skipped reason=insufficient_budget "
            f"remaining_time={remaining_time:.2f}s threshold={min_required:.2f}s"
        )
    # ---------- End of stage 5 ----------

    # ---------- Stage 6: finalize and fall back ----------
    # Return a better green solution from formal optimization, or retain the incumbent if none improves it.
    ret = best_ret
    if ret is None:
        logger.info_red_bright("[lns][fallback] unavailable reason=missing_incumbent")
        return (None, None, None, None, None, None, None, {})  # Match the per-window diagnostic return values.

    if ret["source"] == "cp_optimized":
        logger.info(
            f"[lns] source={ret['source']}, delta={ret.get('delta')}, beta={ret['beta']}, "
            f"obj={ret['obj']}, makespan={ret['makespan']}, carbon={ret['carbon']}"
        )
    elif ret["source"] == "rl_incumbent":
        # Explicitly identify RL fallback in service logs, distinguishing it from a green incumbent return.
        logger.info_yellow(
            f"[lns][fallback] source=rl_incumbent makespan={ret['makespan']} "
            f"obj={ret['obj']} carbon={ret['carbon']}"
        )
    # source is the authoritative replay field: cp_optimized requires a formal CP solution
    # that strictly improves the incumbent and is adopted; otherwise retain RL and use rl_incumbent.
    final_source = "cp_optimized" if ret.get("source") == "cp_optimized" else "rl_incumbent"
    solver_objective = ret.get("obj")
    last_cp_diagnostics = cp_attempt_diagnostics[-1] if cp_attempt_diagnostics else {}
    cp_candidate_objective = (
        solver_objective
        if final_source == "cp_optimized"
        else last_cp_diagnostics.get("cp_candidate_objective")
    )
    objective_difference_from_incumbent = None
    objective_comparison = "no_cp_solution"
    objective_improvement = None
    objective_worsening = None
    if cp_candidate_objective is not None and initial_incumbent_objective is not None:
        # Difference is CP - RL incumbent: positive means worse, zero equal, negative better.
        objective_difference_from_incumbent = float(cp_candidate_objective) - float(
            initial_incumbent_objective
        )
        if abs(objective_difference_from_incumbent) <= 1e-6:
            objective_difference_from_incumbent = 0.0
            objective_comparison = "equal"
            objective_improvement = 0.0
            objective_worsening = 0.0
        elif objective_difference_from_incumbent < 0:
            objective_comparison = "better"
            objective_improvement = -objective_difference_from_incumbent
            objective_worsening = 0.0
        else:
            objective_comparison = "worse"
            objective_improvement = 0.0
            objective_worsening = objective_difference_from_incumbent
    solver_best_bound = (
        ret.get("solver_best_bound")
        if final_source == "cp_optimized"
        else last_cp_diagnostics.get("solver_best_bound")
    )
    optimality_gap = None
    if solver_objective is not None and solver_best_bound is not None:
        optimality_gap = abs(float(solver_objective) - float(solver_best_bound)) / max(
            abs(float(solver_objective)), 1.0
        )
    solver_diagnostics = {
        "solver_status": (
            ret.get("solver_status")
            if final_source == "cp_optimized"
            else last_cp_diagnostics.get("solver_status")
        ),
        "solver_objective_value": solver_objective,
        "solver_best_bound": solver_best_bound,
        "optimality_gap": optimality_gap,
        "incumbent_objective": initial_incumbent_objective,
        "cp_candidate_objective": cp_candidate_objective,  # Feasible CP candidate objective used in the final comparison.
        "objective_comparison": objective_comparison,  # better/equal/worse/no_cp_solution。
        "objective_difference_from_incumbent": objective_difference_from_incumbent,
        "objective_improvement": objective_improvement,  # Nonnegative objective improvement over the RL incumbent.
        "objective_worsening": objective_worsening,  # Nonnegative objective deterioration relative to the RL incumbent.
        "solver_stage": (
            ret.get("solver_stage")
            if final_source == "cp_optimized"
            else last_cp_diagnostics.get("solver_stage")
        ),
        "bound_scope": (
            ret.get("bound_scope")
            if final_source == "cp_optimized"
            else last_cp_diagnostics.get("bound_scope")
        ),
    }
    return (
        ret["makespan"],
        ret["obj"],
        ret["carbon"],
        ret["ws_table"],
        ret["batch_capacities"],
        ret["machines_list"],
        final_source,  # Final source must be rl_incumbent or cp_optimized.
        solver_diagnostics,  # Forward per-window CP/RL provenance and solution quality to dynamic replay.
    )
    # ---------- End of stage 6 ----------
