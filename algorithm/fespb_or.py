from ortools.sat.python import cp_model

from algorithm.utils.cp_model_core import (
    OptionalTask,
    Task,
    add_makespan_variable,
    build_cp_model_core,
    extract_ws_table,
)
from algorithm.utils.cp_solver_utils import configure_cp_solver_workers


def fespb_or(
    jobs_data,
    time_limit=120,
    lock=True,
    rightnow=True,
    solver_diagnostics=None,
    cpu_threads="full",  # CP-SAT workers; full uses all CPU cores available to this process.
):
    """


    lock: Whether to enable the Job Lock constraint (locked segments must remain contiguous).
    rightnow: Whether to enable the job_rightnow / job_avail_time constraints.

    Returns: makespan, ws_table, batch_capacities, machines_list
    """
    horizon = int(1e9)
    (
        model,
        all_tasks,
        tasks_to_machine,
        machines_list,
        batch_capacities,
        _batch_context,
    ) = build_cp_model_core(
        jobs_data,
        horizon=horizon,
        lock=lock,
        rightnow=rightnow,
    )
    makespan_var = add_makespan_variable(model, jobs_data, all_tasks, horizon)
    model.Minimize(makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    resolved_cpu_threads = configure_cp_solver_workers(solver, cpu_threads)
    result = solver.Solve(model)

    # Optional diagnostics preserve the return tuple and distinguish timeouts, infeasibility, and model errors.
    if solver_diagnostics is not None:
        solver_diagnostics.clear()
        solver_diagnostics.update(
            {
                "status": solver.StatusName(result),
                "status_code": int(result),
                "wall_time_seconds": float(solver.WallTime()),
                "num_conflicts": int(solver.NumConflicts()),
                "num_branches": int(solver.NumBranches()),
                "time_limit_seconds": float(time_limit),
                "num_search_workers": resolved_cpu_threads,
            }
        )

    if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        makespan = solver.ObjectiveValue()
        ws_table = extract_ws_table(solver, jobs_data, tasks_to_machine)
        return makespan, ws_table, batch_capacities, machines_list

    # Print the raw CP-SAT status to distinguish UNKNOWN, INFEASIBLE, and MODEL_INVALID.
    print(
        "No solution returned by cp_or: "
        f"status={solver.StatusName(result)}, status_code={int(result)}, "
        f"wall_time={solver.WallTime():.3f}s, time_limit={float(time_limit):.3f}s."
    )
    return None, None, None, None
