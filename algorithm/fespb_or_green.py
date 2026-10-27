from ortools.sat.python import cp_model

from algorithm.utils.cp_model_core import (
    OptionalTask,
    Task,
    add_makespan_variable,
    build_cp_model_core,
    extract_ws_table,
)
from algorithm.utils.cp_solver_utils import configure_cp_solver_workers
from algorithm.utils.green_objective import (
    GREEN_TARIFF_CYCLE_MINUTES,
    OBJECTIVE_LIMIT_MAX_TIME_SECONDS,
    _get_green_time_value_info,
    add_green_objective_terms,
    compute_green_schedule_horizon,
)


def fespb_or_green(
    jobs_data,
    time_limit=120,
    scale_indicator=60,
    beta=None,
    objective_limit=None,
    makespan_limit=None,
    carbon_limit=None,
    limit_mode="objective",
    stop_at_first_solution_within_objective_limit=False,
    solver_diagnostics=None,  # Optionally record the raw CP-SAT status in the diagnostics mapping.
    cpu_threads="full",  # CP-SAT search threads; full uses all CPU cores available to this process.
):
    """Pure CP-SAT green scheduling solver (makespan + carbon-emission bi-objective).




    """
    if beta is None:
        raise ValueError("beta must be provided by Scheduler for fespb_or_green")

    horizon, release_upper, processing_upper = compute_green_schedule_horizon(jobs_data)
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
    )

    # Preserve variable creation order: makespan first, then carbon-overlap variables.
    makespan_var = add_makespan_variable(model, jobs_data, all_tasks, horizon)
    carbon_emission, green_overlap_hints = add_green_objective_terms(
        model,
        jobs_data,
        batch_context["machine_batches"],
        horizon,
        scale_indicator=scale_indicator,
    )
    batch_context["green_overlap_hints"] = green_overlap_hints

    objective_expr = beta * makespan_var + carbon_emission
    objective_limit_value = None
    if objective_limit is not None:
        objective_limit_value = int(round(float(objective_limit)))
    makespan_limit_value = None
    if makespan_limit is not None:
        makespan_limit_value = int(round(float(makespan_limit)))
    carbon_limit_value = None
    if carbon_limit is not None:
        carbon_limit_value = int(round(float(carbon_limit)))

    limit_mode_value = str(limit_mode or "objective").strip().lower()
    if limit_mode_value not in {"objective", "makespan_carbon"}:
        raise ValueError(f"Unsupported cp_green_limit mode: {limit_mode}")
    if stop_at_first_solution_within_objective_limit:
        if limit_mode_value == "objective" and objective_limit_value is None:
            raise ValueError("objective_limit must be provided when limit_mode='objective'")
        if limit_mode_value == "makespan_carbon" and (
            makespan_limit_value is None or carbon_limit_value is None
        ):
            raise ValueError(
                "makespan_limit and carbon_limit must be provided when limit_mode='makespan_carbon'"
            )

    # In limit mode, optimize normally and stop via callback once the target is met; tight bounds can make feasibility search difficult.
    model.Minimize(objective_expr)

    class ObjectiveLimitStopper(cp_model.CpSolverSolutionCallback):
        """Stop the search when a solution meets the target during ordinary optimization."""

        def __init__(
            self,
            mode: str,
            objective_limit_value: int | None,
            makespan_limit_value: int | None,
            carbon_limit_value: int | None,
        ):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.mode = mode
            self.objective_limit_value = objective_limit_value
            self.makespan_limit_value = makespan_limit_value
            self.carbon_limit_value = carbon_limit_value
            self.reached_limit = False
            self.best_objective_value = None

        def on_solution_callback(self):
            objective_value = self.ObjectiveValue()
            makespan_value = self.Value(makespan_var)
            carbon_value = self.Value(carbon_emission)
            self.best_objective_value = objective_value
            if self.mode == "objective":
                reached_limit = objective_value <= self.objective_limit_value
                limit_message = (
                    f"objective={objective_value:.0f}, limit={self.objective_limit_value}"
                )
            else:
                reached_limit = (
                    makespan_value <= self.makespan_limit_value
                    and carbon_value <= self.carbon_limit_value
                )
                limit_message = (
                    f"makespan={makespan_value}, makespan_limit={self.makespan_limit_value}, "
                    f"carbon={carbon_value}, carbon_limit={self.carbon_limit_value}, "
                    f"objective={objective_value:.0f}"
                )
            if reached_limit:
                self.reached_limit = True
                print(
                    "cp_green_limit callback reached limit: "
                    f"mode={self.mode}, {limit_message}"
                )
                self.StopSearch()

    solver = cp_model.CpSolver()
    resolved_cpu_threads = configure_cp_solver_workers(solver, cpu_threads)
    objective_limit_callback = None
    if stop_at_first_solution_within_objective_limit:
        objective_limit_time_cap = float(OBJECTIVE_LIMIT_MAX_TIME_SECONDS)
        print(
            "cp_green_limit mode: "
            f"limit_mode={limit_mode_value}, "
            f"objective_limit={objective_limit_value}, "
            f"makespan_limit={makespan_limit_value}, "
            f"carbon_limit={carbon_limit_value}, "
            f"max_time_in_seconds={objective_limit_time_cap}, "
            "optimize normally and stop by callback after first solution within limit."
        )
        solver.parameters.max_time_in_seconds = objective_limit_time_cap
        objective_limit_callback = ObjectiveLimitStopper(
            limit_mode_value,
            objective_limit_value,
            makespan_limit_value,
            carbon_limit_value,
        )
    else:
        solver.parameters.max_time_in_seconds = time_limit
    result = solver.Solve(model, objective_limit_callback) if objective_limit_callback else solver.Solve(model)

    if solver_diagnostics is not None:
        solver_diagnostics.clear()
        solver_diagnostics.update(
            {
                "status": solver.StatusName(result),
                "status_code": int(result),
                "wall_time_seconds": float(solver.WallTime()),
                "num_conflicts": int(solver.NumConflicts()),
                "num_branches": int(solver.NumBranches()),
                "time_limit_seconds": float(solver.parameters.max_time_in_seconds),
                "schedule_horizon": int(horizon),
                "release_upper": int(release_upper),
                "processing_upper": int(processing_upper),
                "num_search_workers": resolved_cpu_threads,
            }
        )

    if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        makespan = solver.Value(makespan_var)
        carbon = solver.Value(carbon_emission)
        objective_value = int(beta * makespan + carbon)
        if limit_mode_value == "objective":
            limit_reached = objective_limit_value is not None and objective_value <= objective_limit_value
        else:
            limit_reached = (
                makespan_limit_value is not None
                and carbon_limit_value is not None
                and makespan <= makespan_limit_value
                and carbon <= carbon_limit_value
            )
        if stop_at_first_solution_within_objective_limit and not limit_reached:
            print(
                "cp_green_limit timed out before reaching limit: "
                f"limit_mode={limit_mode_value}, "
                f"best_objective={objective_value}, objective_limit={objective_limit_value}, "
                f"best_makespan={makespan}, makespan_limit={makespan_limit_value}, "
                f"best_carbon={carbon}, carbon_limit={carbon_limit_value}"
            )
            return None, None, None, None, None

        green_time_value_info = _get_green_time_value_info(
            jobs_data,
            scale_indicator=scale_indicator,
        )
        print(
            "Solution found: \n"
            f"green_weight: 60min_time_value = "
            f"{green_time_value_info['makespan_60min_cost_in_valley_price_units']:.3f} * "
            f"valley_price({green_time_value_info['valley_price']:.3f}) = "
            f"{green_time_value_info['hourly_time_value_yuan']:.3f} yuan, "
            f"beta = {green_time_value_info['beta']}\n"
            f"makespan: {makespan}\ncarbon: {carbon}\nobjective: {objective_value}\n"
            f"limit_mode: {limit_mode_value}\n"
            f"objective_limit: {objective_limit_value}\n"
            f"makespan_limit: {makespan_limit_value}\n"
            f"carbon_limit: {carbon_limit_value}"
        )
        ws_table = extract_ws_table(solver, jobs_data, tasks_to_machine)
        return makespan, carbon, ws_table, batch_capacities, machines_list

    print(
        "No solution returned by cp_green: "
        f"status={solver.StatusName(result)}, status_code={int(result)}, "
        f"wall_time={solver.WallTime():.3f}s, "
        f"time_limit={float(solver.parameters.max_time_in_seconds):.3f}s."
    )
    return None, None, None, None, None
