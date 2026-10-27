import re
import collections
import sqlite3
import builtins
import hashlib  # Generate a reproducible recycling-operation shuffle seed stable across processes.
import random  # Shuffle task priorities when competing for the same recycling station.
from collections import defaultdict
from pprint import pprint, pformat
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import copy
import json
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from multi_pms.obj.db_tools import SchedulerDatabase
from multi_pms.utils.async_log import AsyncLog

from algorithm.fespb_rl import fespb_rl
from algorithm.fjspb_human import fespb_human  # Human work-calendar scheduling branch.

from algorithm.fespb_or import fespb_or
from algorithm.fespb_or_green import fespb_or_green
from algorithm.fespb_or_green_lns import fespb_or_green_adaptive_lns
from algorithm.utils.green_objective import build_price_segments_for_horizon


CP_OR_METHODS = {
    "cp_or",
    "cp_green_or",
    "cp_green_limit_or",
    "cp_green_limit_strict_or",
}
CP_GREEN_METHODS = {
    "cp_green_or",
    "cp_green_limit_or",
    "cp_green_limit_strict_or",
}
CP_GREEN_LIMIT_METHODS = {
    "cp_green_limit_or",
    "cp_green_limit_strict_or",
}
CP_GREEN_LIMIT_STRICT_METHODS = {
    "cp_green_limit_strict_or",
}
RL_OR_LNS_METHODS = {
    "rl_or",
    "rl_release_or",
}
RL_FULL_RELEASE_METHODS = {
    "rl_release_or",
}


class Scheduler:
    def __init__(self, logger: AsyncLog, db: SchedulerDatabase, model: Any, memories: Any, env_params: dict):
        self.logger = logger
        self.db = db
        self.green_objective_scale_indicator = 60.0
        self.last_schedule_metrics_by_group: dict[str, dict[str, Any]] = {}  # Cache actual green-objective metrics per task group for HTTP responses and replay records.

        # Reinforcement learning.
        self.model = model
        self.memories = memories
        self.env_params = env_params

        # Plot colors.
        self.colors = self.color_filter()

    def color_filter(self):
        colors = set(mcolors.CSS4_COLORS.keys())
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
        return list(colors - whites)

    def compute_green_beta(
        self,
        jobs_data: dict | None = None,
        method: str | None = None,
        scale_indicator: float | None = None,
    ) -> int | None:

        whole_info = (jobs_data or {}).get("info", {})
        effective_scale = float(
            self.green_objective_scale_indicator if scale_indicator is None else scale_indicator
        )
        valley_price = float(whole_info.get("valley_price", self.db.get_power_info("valley_price", 0.3)))
        makespan_60min_cost_units = float(
            whole_info.get(
                "makespan_60min_cost_in_valley_price_units",
                self.db.get_power_info("makespan_60min_cost_in_valley_price_units", 2.0),
            )
        )
        return max(0, int(round(makespan_60min_cost_units * valley_price * 1000 / effective_scale)))

    def get_last_schedule_metrics(self, task_group: str | None) -> dict[str, Any]:
        # Return an empty dict before the group's first schedule so callers can fall back to None.
        if task_group is None:
            return {}
        return dict(self.last_schedule_metrics_by_group.get(task_group, {}))

    def do_schedule(
        self,
        task_group,
        cur_ptr,
        expr_no_list,
        time_limit=120,
        sim=False,
        method="rl_or",
        objective_limit=None,
        makespan_limit=None,
        carbon_limit=None,
        threads="full",  # Forward to every OR-Tools CP-SAT solve stage.
        yolo=False,
    ):
        self.logger.debug(f"Start loading task data, cur_ptr={cur_ptr}, expr_no_list={expr_no_list}")
        # Route scheduler messages through AsyncLog for consistent formatting and persistence.
        # Find unfinished task data with start_time >= cur_ptr.
        jobs_data = self.db.construct_fespb_jobs_data_from_db(cur_ptr, expr_no_list)
        if sim:
            jobs_data["info"]["cur_time_stamp"] = jobs_data["info"].get(
                "sim_cur_time_stamp", jobs_data["info"].get("cur_time_stamp")
            )
        elif yolo:
            # YOLO uses fixed in-process tariff boundaries, with the current Shanghai wall clock as each round's origin.
            jobs_data["info"]["cur_time_stamp"] = datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
        self.logger.debug(f"Task data construction completed")
        # self.logger.debug(f"Task data constructed: {pformat(jobs_data)}")

        # 1. shifted_jobs_data: rebase fjspb_index to zero and shift machine availability in info left by cur_ptr.
        # 2. mapped: map string job and machine IDs to numbers.
        shifted_jobs_data, shifted_mapping = self.shift_fjspb_index(cur_ptr, jobs_data)
        mapped_jobs_data, mappings = self.map_jobs_data(shifted_jobs_data)
        method = str(method or "rl_or").strip().lower()
        green_beta = self.compute_green_beta(mapped_jobs_data, method=method)
        schedule_metrics = {
            "method": method,
            "beta": None,
            "scale_indicator": None,
            "solver_objective_value": None,
            "solver_carbon_objective_scaled": None,
            "source": None,  # Save adopted schedule provenance for per-batch replay statistics.
            "failure_stage": None,  # Record the precise failure stage for the dynamic service.
            "error_message": None,  # Save readable errors including raw CP-SAT statuses.
            "solver_diagnostics": {},  # Store wall time, status codes, and search statistics by stage.
        }






        if method in CP_OR_METHODS:
            # Subtract 1 from machine IDs for CP methods to align with RL results.
            # ----------------cp methods----------------
            cp_start_time = time.time()
            for job in mapped_jobs_data["detail"].values():
                for task_id, (machines, durations, _) in enumerate(job):
                    machines[:] = [m - 1 for m in machines]

            if method in CP_GREEN_METHODS:
                limit_mode_value = "makespan_carbon" if method in CP_GREEN_LIMIT_STRICT_METHODS else "objective"
                use_objective_limit_mode = (
                    method in CP_GREEN_LIMIT_METHODS
                    and (
                        (limit_mode_value == "objective" and objective_limit is not None)
                        or (
                            limit_mode_value == "makespan_carbon"
                            and makespan_limit is not None
                            and carbon_limit is not None
                        )
                    )
                )
                effective_objective_limit = objective_limit if use_objective_limit_mode else None
                effective_makespan_limit = makespan_limit if use_objective_limit_mode else None
                effective_carbon_limit = carbon_limit if use_objective_limit_mode else None
                if use_objective_limit_mode:
                    self.logger.info_green(
                        f"{method} received limit parameters, "
                        f"limit_mode={limit_mode_value}, "
                        f"objective_limit={effective_objective_limit}, "
                        f"makespan_limit={effective_makespan_limit}, "
                        f"carbon_limit={effective_carbon_limit}; "
                        "Search with the same time limit (internal parameters); return immediately after finding the first feasible solution that meets the limits."
                    )
                if method in CP_GREEN_LIMIT_METHODS and not use_objective_limit_mode:
                    self.logger.info_yellow(
                        f"{method} did not receive the limit required by this branch; falling back to standard green CP time-limited optimization."
                    )
                cp_green_makespan, cp_green_carbon, mapped_ws_table, mapped_batch_capacities, mapped_machines_list = fespb_or_green(
                    mapped_jobs_data,
                    time_limit=time_limit,
                    scale_indicator=self.green_objective_scale_indicator,
                    beta=green_beta,
                    objective_limit=effective_objective_limit,  # Pass objective bounds to callbacks/listeners only in *_green_limit mode; ordinary green disables limits.
                    makespan_limit=effective_makespan_limit,  # Forward historical makespan references in makespan/carbon mode.
                    carbon_limit=effective_carbon_limit,  # Forward historical carbon references in makespan/carbon mode.
                    limit_mode=limit_mode_value,
                    stop_at_first_solution_within_objective_limit=use_objective_limit_mode,
                    cpu_threads=threads,
                )
                schedule_metrics.update(
                    {
                        "beta": green_beta,
                        "scale_indicator": self.green_objective_scale_indicator,
                    }
                )
                if cp_green_makespan is not None and cp_green_carbon is not None:
                    schedule_metrics.update(
                        {
                            "solver_objective_value": int(green_beta * cp_green_makespan + cp_green_carbon),
                            "solver_carbon_objective_scaled": int(cp_green_carbon),
                        }
                    )  # Record objective values only on green-solve success to avoid type errors on None.
                schedule_metrics["source"] = "cp_solver"  # The pure CP branch has no RL incumbent.
                self.logger.info(f"CP OR green calc time: {time.time() - cp_start_time:.2f}s")
            else:
                _, mapped_ws_table, mapped_batch_capacities, mapped_machines_list = fespb_or(
                    mapped_jobs_data, time_limit=time_limit, cpu_threads=threads
                )
                schedule_metrics["source"] = "cp_solver"  # The pure CP branch has no RL incumbent.
                self.logger.info(f"CP OR calc time: {time.time() - cp_start_time:.2f}s")
            # --------------------------------------------
        elif method == "human":
            # Human uses deterministic work-calendar scheduling, bypassing RL and CP/LNS branches.
            human_start_time = time.time()
            _, mapped_ws_table, mapped_batch_capacities, mapped_machines_list, _ = fespb_human(
                mapped_jobs_data,
                self.env_params,
            )
            schedule_metrics["source"] = "human"
            self.logger.info(f"Human calc time: {time.time() - human_start_time:.2f}s")
        elif method in ["pure_rl"]:
            rl_start_time = time.time()
            _, mapped_ws_table, mapped_batch_capacities, mapped_machines_list, _ = fespb_rl(
                mapped_jobs_data, self.env_params, self.model, self.memories
            )
            schedule_metrics["source"] = "pure_rl"
            self.logger.info(f"Pure RL calc time: {time.time() - rl_start_time:.2f}s")
        elif method in RL_OR_LNS_METHODS:
            # ----------------rl methods----------------
            self.logger.info(
                f"Starting RL-guided OR LNS scheduling with time limit {time_limit}s..."
            )
            rl_start_time = time.time()
            _, mapped_ws_table, mapped_batch_capacities, mapped_machines_list, _ = fespb_rl(
                mapped_jobs_data, self.env_params, self.model, self.memories
            )
            rl_elapsed_time = time.time() - rl_start_time

            guided_cp_start_time = time.time()
            for job in mapped_jobs_data["detail"].values():
                for task_id, (machines, durations, _) in enumerate(job):
                    machines[:] = [m - 1 for m in machines]
            full_release_only_after_incumbent = method in RL_FULL_RELEASE_METHODS  # *_release_* skips neighborhoods and runs global release search after establishing the incumbent.
            # Reserve at least 10 seconds for CP optimization so RL cannot consume its entire budget.
            (
                guided_makespan,
                guided_obj,
                guided_carbon,
                mapped_ws_table,
                mapped_batch_capacities,
                mapped_machines_list,
                guided_source,
                guided_diagnostics,
            ) = fespb_or_green_adaptive_lns(
                mapped_jobs_data,
                mapped_ws_table,
                beta=green_beta,
                time_limit=max(time_limit - rl_elapsed_time, 10.0),
                scale_indicator=self.green_objective_scale_indicator,
                full_release_only_after_incumbent=full_release_only_after_incumbent,  # Enable direct full_release after incumbent only for *_release_* methods.
                logger=self.logger,  # Route LNS logging through the scheduler's AsyncLog.
                cpu_threads=threads,
            )
            self.logger.info(
                f"RL-guided OR makespan: {guided_makespan}, obj: {guided_obj}, carbon: {guided_carbon}, rl_time: {rl_elapsed_time:.2f}s, cp_time: {time.time() - guided_cp_start_time:.2f}s"
            )
            schedule_metrics.update(
                {
                    "beta": green_beta,
                    "scale_indicator": self.green_objective_scale_indicator,
                    "source": guided_source,  # Final LNS provenance: rl_incumbent or cp_optimized.
                    "solver_diagnostics": guided_diagnostics,  # Save the current window status, bound, and gap.
                }
            )
            if guided_obj is not None and guided_carbon is not None:
                schedule_metrics.update(
                    {
                        "solver_objective_value": int(guided_obj),
                        "solver_carbon_objective_scaled": int(guided_carbon),
                    }
                )  # Record metrics only for valid LNS green objectives to avoid None type errors during failure fallback.
            # --------------------------------------------
        else:
            # Use an explicit method allowlist so deleted or misspelled names cannot silently enter RL.
            self.logger.info_red_bright(f"scheduler: unsupported schedule method: {method}")
            self.last_schedule_metrics_by_group[task_group] = schedule_metrics
            return None





        if mapped_ws_table is None:
            # Prefer staged algorithm errors, with generic fallback for other methods.
            self.logger.info_red_bright(
                schedule_metrics.get("error_message") or "scheduler: failed."
            )
            self.last_schedule_metrics_by_group[task_group] = schedule_metrics
            return None

        # 1. Remap numeric job/machine IDs to strings; 2. restore fjspb_index and shift tasks forward by cur_ptr.
        shifted_ws_table, machines_list, batch_capacities = self.remap_ws_table(
            mapped_ws_table, mapped_machines_list, mapped_batch_capacities, mappings
        )
        ws_table = self.reshift_ws_table(cur_ptr, shifted_ws_table, shifted_mapping)
        # This is the shared production finalization path for every algorithm, not only dynamic replay
        # or debug; postprocessing directly changes the schedule written to the production database.
        # Move the recycling operation immediately before end into its earliest feasible gap, then tighten end;
        # preserve physical-station assignments and durations from the solver solution.
        self.advance_terminal_storage_operations(
            ws_table,
            cur_ptr,
            jobs_data,
            task_group,
        )
        self.normalize_terminal_end_operations(ws_table, cur_ptr)






        if ws_table:
            makespan = self.db.write_ws_table_to_db(task_group, expr_no_list, ws_table)
            self.last_schedule_metrics_by_group[task_group] = schedule_metrics
            return makespan
        else:
            self.logger.info_red_bright("No solution found.")
            self.last_schedule_metrics_by_group[task_group] = schedule_metrics
            return None

    def advance_terminal_storage_operations(
        self,
        ws_table: dict,
        cur_ptr: int,
        jobs_data: dict,
        task_group: str,
    ) -> None:
        """Move the final cleanup operation before ``end`` to the earliest feasible gap on a physical workstation.

        Only operations actually assigned to ``elec_chem_storage_workstation*`` and directly preceding ``end``
        are treated as cleanup candidates. Static operations on the same workstation type that occur in the middle of a task serve only as
        fixed occupied intervals for conflict checks; this method does not modify them.

        To ensure post-processing never delays any solver operation, each unprocessed cleanup operation reserves its original
        interval. Candidates are processed in a stable randomized order based on ``task_group``, workstation, and task set.
         Advance a candidate only when a gap is found no later than its original start time; otherwise, keep the solver's original time.

        This method uses a conservative capacity policy and does not create overlapping batches. Legal overlapping batches already present in the solver's solution
        are preserved as-is, but newly advanced cleanup operations are inserted only into completely idle intervals. This cannot exceed workstation
        capacity or conflict with static or other fixed operations, but it may skip more aggressive advances that use spare batch capacity.
        """
        terminal_tasks = ws_table.get("end", [])
        if not terminal_tasks:
            return

        tasks_by_expr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for tasks in ws_table.values():
            for task in tasks:
                tasks_by_expr[str(task["expr_no"])].append(task)

        # Keep lock metadata in the current subproblem; job_rightnow covers locked first operations across windows.
        lock_by_expr_index: dict[tuple[str, int], bool] = {}
        remaining_indices_by_expr: defaultdict[str, list[int]] = defaultdict(list)
        for expr_no, operations in jobs_data.get("detail", {}).items():
            for operation in operations:
                metadata = operation[3]
                normalized_expr_no = str(expr_no)
                operation_index = int(metadata["fjspb_index"])
                lock_by_expr_index[(normalized_expr_no, operation_index)] = bool(
                    metadata.get("lock", False)
                )
                remaining_indices_by_expr[normalized_expr_no].append(operation_index)

        job_available_times = jobs_data.get("info", {}).get("job_avail_time", {})
        machine_available_times = jobs_data.get("info", {}).get("machines_available_time", {})
        job_rightnow = jobs_data.get("info", {}).get("job_rightnow", {})
        candidates_by_machine: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        for end_task in terminal_tasks:
            expr_no = str(end_task["expr_no"])
            recovery_index = int(end_task["fjspb_index"]) - 1
            expr_tasks = tasks_by_expr[expr_no]
            recovery_task = next(
                (
                    task
                    for task in expr_tasks
                    if int(task["fjspb_index"]) == recovery_index
                ),
                None,
            )
            if recovery_task is None:
                # Recycling operations already completed or running are outside this solve result and must not be retimed.
                continue

            ws_code = str(recovery_task["ws_code_fjspb"])
            if not ws_code.startswith("elec_chem_storage_workstation"):
                continue

            predecessor_index = recovery_index - 1
            predecessor = next(
                (
                    task
                    for task in expr_tasks
                    if int(task["fjspb_index"]) == predecessor_index
                ),
                None,
            )
            predecessor_end = (
                int(predecessor["end_time"])
                if predecessor is not None
                else self.db.find_finite_step_end_time(expr_no, predecessor_index)
            )
            release_time = max(
                int(cur_ptr),
                int(job_available_times.get(expr_no, cur_ptr)),
                int(machine_available_times.get(ws_code, cur_ptr)),
                int(predecessor_end) if predecessor_end is not None else int(cur_ptr),
            )

            original_start = int(recovery_task["start_time"])
            original_end = int(recovery_task["end_time"])
            duration = original_end - original_start
            if duration < 0:
                raise ValueError(
                    f"invalid terminal storage duration for {expr_no}: {duration}"
                )
            if release_time > original_start:
                raise ValueError(
                    f"terminal storage release exceeds solver start for {expr_no}: "
                    f"release={release_time}, start={original_start}"
                )

            # Lock-adjacent segments and dynamic rightnow first remaining operations must match release times exactly.
            is_first_remaining_operation = (
                recovery_index == min(remaining_indices_by_expr.get(expr_no, [recovery_index]))
            )
            requires_exact_release = (
                bool(job_rightnow.get(expr_no, False)) and is_first_remaining_operation
            ) or (
                lock_by_expr_index.get((expr_no, predecessor_index), False)
                and lock_by_expr_index.get((expr_no, recovery_index), False)
            )
            candidates_by_machine[ws_code].append(
                {
                    "task": recovery_task,
                    "expr_no": expr_no,
                    "release_time": release_time,
                    "original_start": original_start,
                    "original_end": original_end,
                    "duration": duration,
                    "requires_exact_release": requires_exact_release,
                }
            )

        moved_count = 0
        total_advance = 0
        candidate_task_ids = {
            id(candidate["task"])
            for candidates in candidates_by_machine.values()
            for candidate in candidates
        }

        for ws_code, candidates in candidates_by_machine.items():
            # Fixed occupancy includes resting and all nonterminal recycling operations on the physical station.
            fixed_intervals = [
                (int(task["start_time"]), int(task["end_time"]))
                for task in ws_table.get(ws_code, [])
                if id(task) not in candidate_task_ids
            ]
            seed_text = (
                f"{task_group}|{ws_code}|"
                + "|".join(sorted(str(candidate["expr_no"]) for candidate in candidates))
            )
            stable_seed = int.from_bytes(
                hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            random.Random(stable_seed).shuffle(candidates)

            placed_intervals: list[tuple[int, int]] = []
            for candidate_index, candidate in enumerate(candidates):
                # Keep unprocessed candidates reserved at their original times so shuffled earlier candidates cannot steal their feasible slots.
                reserved_intervals = [
                    (int(other["original_start"]), int(other["original_end"]))
                    for other in candidates[candidate_index + 1 :]
                ]
                earliest_start = self._find_earliest_non_overlapping_start(
                    int(candidate["release_time"]),
                    int(candidate["duration"]),
                    fixed_intervals + placed_intervals + reserved_intervals,
                )
                original_start = int(candidate["original_start"])
                requires_exact_release = bool(candidate["requires_exact_release"])
                can_move = (
                    earliest_start < original_start
                    and (
                        not requires_exact_release
                        or earliest_start == int(candidate["release_time"])
                    )
                )

                if can_move:
                    task = candidate["task"]
                    task["start_time"] = earliest_start
                    task["end_time"] = earliest_start + int(candidate["duration"])
                    moved_count += 1
                    total_advance += original_start - earliest_start

                # Keep original intervals when not advanced, preserving valid batch overlaps in the solver result.
                placed_intervals.append(
                    (
                        int(candidate["task"]["start_time"]),
                        int(candidate["task"]["end_time"]),
                    )
                )

            ws_table[ws_code].sort(
                key=lambda task: (
                    int(task["start_time"]),
                    int(task["end_time"]),
                    str(task["expr_no"]),
                )
            )

        if candidates_by_machine:
            self.logger.info(
                f"Recycling operations moved earlier: candidates="
                f"{sum(len(items) for items in candidates_by_machine.values())}, "
                f"actually moved={moved_count}, total advance={total_advance}."
            )

    @staticmethod
    def _find_earliest_non_overlapping_start(
        release_time: int,
        duration: int,
        occupied_intervals: list[tuple[int, int]],
    ) -> int:
        """Find the first full gap in half-open occupied intervals at or after release_time."""
        candidate_start = int(release_time)
        for occupied_start, occupied_end in sorted(occupied_intervals):
            occupied_start = int(occupied_start)
            occupied_end = int(occupied_end)
            if occupied_end <= candidate_start:
                continue
            if candidate_start + duration <= occupied_start:
                break
            candidate_start = max(candidate_start, occupied_end)
        return candidate_start

    def normalize_terminal_end_operations(self, ws_table: dict, cur_ptr: int) -> None:
        """Tighten each virtual ``end`` so it starts after the last real predecessor in its task.

        Purpose: The solver schedules ``end`` as a regular operation and may leave an unnecessary wait after the last physical
        operation ends. This method keeps the original duration of ``end`` and changes only its
        ``start_time``/``end_time``, and does not move any physical workstation operation.

        Impact on production scheduling: This method runs at the shared database-write entry point and affects results from CP, RL, and RL+LNS
        and Human. Although ``end`` is a virtual workflow node with no physical equipment and is theoretically less likely
        to affect real experiment steps or equipment occupancy, its modified times are written to the database and may still affect:

        1. The makespan calculated and returned by ``write_ws_table_to_db()``;
        2. Downstream checks of task completion time, done events, or bottle status;
        3. Historical plans and time boundaries read in the next rescheduling round;
        4. Consistency among replay Gantt charts, performance metrics, the solver's original objective values, and persisted results.

        Risk boundary: This is in-place post-processing after the solve. The solver is not called again to validate
        precedence, virtual workstation capacity, or the objective function. It assumes ``end`` is always the last operation in a job
        and that the virtual workstation has sufficient capacity. This behavior has not been tested end-to-end with a real DMS, communications backend, or physical equipment. Until
        that testing is complete, treat it as "unlikely to change the physical workflow, but may still affect the semantics of production scheduling data."

        Args:
            ws_table: Schedule table with real job IDs, workstation IDs, and absolute times restored; this method modifies it in place.
            cur_ptr: Current global scheduling pointer; if only ``end`` remains in a subproblem, its earliest start is determined together with the historical predecessor end time.

        Raises:
            ValueError: ``end`` is not the last operation, its direct predecessor is missing, or its duration is negative.
        """
        end_tasks = ws_table.get("end", [])
        if not end_tasks:
            return

        tasks_by_expr: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for tasks in ws_table.values():
            for task in tasks:
                tasks_by_expr[str(task["expr_no"])].append(task)

        for end_task in end_tasks:
            expr_no = str(end_task["expr_no"])
            end_index = int(end_task["fjspb_index"])
            expr_tasks = tasks_by_expr[expr_no]
            if any(int(task["fjspb_index"]) > end_index for task in expr_tasks):
                raise ValueError(f"end is not terminal for {expr_no}: fjspb_index={end_index}")
            predecessor = next(
                (
                    task
                    for task in expr_tasks
                    if int(task["fjspb_index"]) == end_index - 1
                ),
                None,
            )
            if predecessor is None and len(expr_tasks) != 1:
                raise ValueError(f"missing direct predecessor for end of {expr_no}")
            if predecessor is not None:
                target_start = int(predecessor["end_time"])
            else:
                # A cross-window subproblem may contain only virtual end while its frozen direct predecessor
                # ends after cur_ptr; query stored times to preserve precedence rather than assuming completion.
                historical_predecessor_end = self.db.find_finite_step_end_time(expr_no, end_index - 1)
                target_start = max(
                    int(cur_ptr),
                    historical_predecessor_end
                    if historical_predecessor_end is not None
                    else int(cur_ptr),
                )
            duration = int(end_task["end_time"]) - int(end_task["start_time"])
            if duration < 0:
                raise ValueError(f"invalid end duration for {expr_no}: {duration}")
            end_task["start_time"] = target_start
            end_task["end_time"] = target_start + duration

        # Restore virtual-station time ordering after tightening so downstream code does not use stale solver order.
        end_tasks.sort(
            key=lambda task: (
                int(task["start_time"]),
                int(task["end_time"]),
                str(task["expr_no"]),
            )
        )

    # Renumber jobs from zero and record the original-to-new index map.
    def shift_fjspb_index(self, cur_ptr, jobs_data):
        shifted = copy.deepcopy(jobs_data)
        index_map, reverse_map = defaultdict(dict), defaultdict(dict)

        for job_id, job in shifted["detail"].items():
            mapping = {task[3]["fjspb_index"]: idx for idx, task in enumerate(job)}
            reverse = {v: k for k, v in mapping.items()}

            # Update fjspb_index.
            for task in job:
                task[3]["fjspb_index"] = mapping[task[3]["fjspb_index"]]

            index_map[job_id] = mapping
            reverse_map[job_id] = reverse

        # Shift availability times.
        info = shifted["info"]
        info["machines_available_time"] = {k: v - cur_ptr for k, v in info["machines_available_time"].items()}
        info["job_avail_time"] = {k: v - cur_ptr for k, v in info["job_avail_time"].items()}

        return shifted, {"index_mapping": index_map, "reverse_mapping": reverse_map}

    def map_jobs_data(self, jobs_data):
        # Create mapping tables.
        job_mapping = {job: idx for idx, job in enumerate(jobs_data["detail"].keys())}
        reverse_job_mapping = {idx: job for job, idx in job_mapping.items()}

        machine_names = set(jobs_data["info"]["machine_capacities"].keys())
        # RL input uses one-based machine IDs.
        machine_mapping = {m: i + 1 for i, m in enumerate(machine_names)}
        # Both FJSP-B and RL subtract 1 from machine IDs; account for this before restoring machine names.
        reverse_machine_mapping = {idx - 1: m for m, idx in machine_mapping.items()}

        # Parameter mapping.
        param_mapping = {}
        param_counter = 1  # Ordinary parameters are numbered from 1; -1 means no parameter.




        info = copy.deepcopy(jobs_data["info"])
        info["job_avail_time"] = {job_mapping[j]: t for j, t in info["job_avail_time"].items()}
        info["job_lock"] = {job_mapping[j]: l for j, l in info["job_lock"].items()}
        info["job_lock_periods"] = {job_mapping[j]: p for j, p in info["job_lock_periods"].items()}
        info["job_rightnow"] = {job_mapping[j]: r for j, r in info["job_rightnow"].items()}

        machine_capacities = {machine_mapping[m]: cap for m, cap in info["machine_capacities"].items()}
        machines_available_time = {
            machine_mapping[m]: time for m, time in info["machines_available_time"].items()
        }
        machine_power_info = {
            machine_mapping[m]: power for m, power in info.get("machine_power_info", {}).items()
        }
        info["machine_capacities"] = [machine_capacities[key] for key in sorted(machine_capacities)]
        info["machines_available_time"] = [
            machines_available_time[key]
            for key in sorted(machines_available_time)
            # 1 for key in sorted(machines_available_time)
        ]
        info["machine_power_info"] = [machine_power_info[key] for key in sorted(machine_power_info)]

        # Map detail.
        detail_mapped = {}
        for job, tasks in copy.deepcopy(jobs_data["detail"]).items():
            mapped_job = job_mapping[job]
            detail_mapped[mapped_job] = []
            for machine_list, processtimes, machine_details, extra_info in tasks:
                mapped_machine_list = [machine_mapping[m] for m in machine_list]

                # Map parameters using only the first machine_details param.
                first_param = machine_details[0]["param"].copy() if machine_details else {}
                first_param.pop("bottle_count", None)
                param_str = str(first_param) if first_param else "{}"  # Handle empty dictionaries.

                if param_str not in param_mapping:
                    param_mapping[param_str] = -1 if first_param == {} else param_counter
                    mapped_param = param_mapping[param_str]
                    if first_param != {}:
                        param_counter += 1
                else:
                    mapped_param = param_mapping[param_str]  # Reuse an existing ID.

                detail_mapped[mapped_job].append((mapped_machine_list, processtimes, mapped_param))

        mappings = {
            "job_mapping": job_mapping,
            "reverse_job_mapping": reverse_job_mapping,
            "machine_mapping": machine_mapping,
            "reverse_machine_mapping": reverse_machine_mapping,
            "param_mapping": param_mapping,  # Store parameter mappings.
            "reverse_param_mapping": {v: k for k, v in param_mapping.items()},  # Reverse parameter mappings.
        }

        mapped_jobs_data = {"detail": detail_mapped, "info": info}
        return mapped_jobs_data, mappings

    def remap_ws_table(self, mapped_ws_table, mapped_machines_list, mapped_batch_capacities, mappings):
        shifted_ws_table = {}
        for ws_code, tasks in mapped_ws_table.items():
            machine_name = mappings["reverse_machine_mapping"][int(ws_code)]
            updated_tasks = []
            for task in tasks:
                job_id = mappings["reverse_job_mapping"][int(task["expr_no"])]
                # Update name before b_id.
                task["name"] = task["name"].replace(f"M{int(ws_code)}", f"M{machine_name}")
                task["name"] = task["name"].replace(f"B{int(task['expr_no'])}", f"B{job_id}")
                task["ws_code_fjspb"] = machine_name
                task["expr_no"] = job_id
                updated_tasks.append(task)
            shifted_ws_table[machine_name] = updated_tasks

        machines_list = [mappings["reverse_machine_mapping"][i] for i in mapped_machines_list]
        batch_capacities = {
            mappings["reverse_machine_mapping"][i]: capacity
            for i, capacity in mapped_batch_capacities.items()
        }
        return shifted_ws_table, machines_list, batch_capacities

    # Restore fjspb_index and T-suffixed task indices, then update start_time/end_time in ws_table.
    def reshift_ws_table(self, cur_ptr, shifted_ws_table, shifted_mapping):
        ws_table = copy.deepcopy(shifted_ws_table)
        for ws_code, tasks in ws_table.items():
            for task in tasks:
                expr_no = task["expr_no"]
                original_index = shifted_mapping["reverse_mapping"][expr_no][task["fjspb_index"]]
                task["fjspb_index"] = original_index
                task["name"] = re.sub(r"(T)(\d+)", rf"T{original_index}", task["name"])

                task["start_time"] += cur_ptr
                task["end_time"] += cur_ptr
        return ws_table

    def plot_schedule(
        self,
        cur_ptr,
        expr_no_list,
        img_name="schedule.svg",
        yolo=False,
    ):
        jobs_data = self.db.construct_fespb_jobs_data_from_db(-1, expr_no_list, is_plot=True)
        if yolo:
            jobs_data["info"]["cur_time_stamp"] = datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")


        machines_list = []
        for job_id, job in jobs_data["detail"].items():
            for task_id, (machines, durations, _, _) in enumerate(job):
                for m_id in machines:
                    if m_id not in machines_list:
                        machines_list.append(m_id)

        makespan = self.db.find_makespan()
        ws_table, bottle_table, _, _ = self.db.fetch_scheduling_table(expr_no_list)

        # Draw the Gantt chart.
        fig, (ax1, ax2, ax4) = plt.subplots(
            3, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
        )

        # Compute each machine's maximum simultaneous task count.
        machine_max_tasks = {}
        for m_id in machines_list:
            if not ws_table[m_id]:  # Skip machines without tasks.
                continue
            max_time = max(
                task["end_time"] for task in ws_table[m_id] if task["end_time"]
            )  # Find the machine's latest end time.
            time_task_count = {}
            for t in range(max_time + 1):  # Iterate from zero to max_time.
                time_task_count[t] = 0
                for task in ws_table[m_id]:
                    if task["start_time"] == t:  # Count only tasks starting exactly at the current pointer.
                        time_task_count[t] += 1
            machine_max_tasks[m_id] = max(time_task_count.values())

        actual_machines_list = [m_id for m_id in machines_list if m_id in machine_max_tasks]

        # Set task bar height.
        task_height = 0.3

        # Compute each machine's y-axis extent and starting position.
        machine_y_height = {m_id: task_height * machine_max_tasks[m_id] for m_id in actual_machines_list}

        machine_y_start = {}
        cumulative_height = 0
        for m_id in actual_machines_list:
            machine_y_start[m_id] = cumulative_height
            cumulative_height += machine_y_height[m_id]

        expr_name_to_color = {}

        for m_index, m_id in enumerate(actual_machines_list):
            current_y_base = machine_y_start[m_id]
            last_end = -1
            current_y = current_y_base
            for task in ws_table[m_id]:
                if task["start_time"] < sys.maxsize:
                    job_id = task["expr_no"]
                    # Compute bar positions and sizes.
                    bar_height = task_height

                    if last_end <= task["start_time"]:
                        current_y = current_y_base
                    else:
                        current_y += bar_height
                    last_end = task["end_time"]


                    expr_name = job_id
                    if expr_name not in expr_name_to_color:
                        expr_name_to_color[expr_name] = mcolors.CSS4_COLORS[
                            self.colors[hash(expr_name) % len(self.colors)]
                        ]
                    color = expr_name_to_color[expr_name]

                    ax1.broken_barh(
                        [(task["start_time"], task["duration"])],
                        (current_y, bar_height),
                        facecolors=color,
                        edgecolor="none",
                    )

        # Draw vertical lines at cur_ptr and makespan.
        ax1.axvline(x=cur_ptr, color="b", linestyle="-", label="Current Pointer")
        ax1.axvline(x=makespan, color="r", linestyle="-", label="Makespan")

        yticks = [machine_y_start[m_id] + machine_y_height[m_id] / 2 for m_id in actual_machines_list]
        yticklabels = [
            f"{m}" if m != "manual_workstation" else "muffle_station" for m in actual_machines_list
        ]

        # Separate machines.
        for m_id in actual_machines_list:
            ax1.axhline(y=machine_y_start[m_id], color="grey", linestyle="--", linewidth=0.5)

        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticklabels)
        ax1.set_ylim(0, cumulative_height)
        ax1.set_xlim(xmin=0)
        ax1.set_ylabel("Machines")
        ax1.set_title("Workstation Perspective")

        legend_patches = [
            mpatches.Patch(color=color, label=expr_name) for expr_name, color in expr_name_to_color.items()
        ]
        ax1.legend(handles=legend_patches, loc="upper left", fontsize=10, bbox_to_anchor=(1.01, 1.01))

        for idx, (job_id, tasks) in enumerate(bottle_table.items()):
            lock_periods = json.loads(tasks[0].get("lock_periods", "[]"))

            # Draw lock brackets.
            if lock_periods:
                for period in lock_periods:
                    lock_start, lock_end = period
                    if lock_start < sys.maxsize:
                        y = task_height * idx
                        h = task_height
                        across = 0.5
                        lock_color_name = "FireBrick"
                        # Vertical line.
                        ax2.plot(
                            [lock_start, lock_start],
                            [y - h / 2 + 0.1 / 2, y + h / 2 - 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )
                        # Bottom cap.
                        ax2.plot(
                            [lock_start, lock_start + across],
                            [y - h / 2 + 0.1 / 2, y - h / 2 + 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )
                        # Top cap.
                        ax2.plot(
                            [lock_start, lock_start + across],
                            [y + h / 2 - 0.1 / 2, y + h / 2 - 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )

                        ax2.plot(
                            [lock_end, lock_end],
                            [y - h / 2 + 0.1 / 2, y + h / 2 - 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )
                        # Bottom horizontal line
                        ax2.plot(
                            [lock_end - across, lock_end],
                            [y - h / 2 + 0.1 / 2, y - h / 2 + 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )
                        # Top horizontal line
                        ax2.plot(
                            [lock_end - across, lock_end],
                            [y + h / 2 - 0.1 / 2, y + h / 2 - 0.1 / 2],
                            color=lock_color_name,
                            lw=3,
                        )

            for task in tasks:
                if task["start_time"] < sys.maxsize and task["ws_code_fjspb"]:
                    ax2.broken_barh(
                        [(task["start_time"], task["duration"])],
                        (task_height * idx - 0.1, task_height - 0.1),
                        facecolors=mcolors.CSS4_COLORS[
                            self.colors[hash(task["ws_code_fjspb"]) % len(self.colors)]
                        ],
                    )

        ax2.axvline(x=cur_ptr, color="b", linestyle="-", label="Current Pointer")
        ax2.axvline(x=makespan, color="r", linestyle="-", label="Makespan")

        ax2.set_yticks([task_height * idx for idx in range(len(bottle_table))])
        ax2.set_yticklabels([("task {}".format(b_id)) for b_id in bottle_table.keys()])

        # Separate tasks.
        for idx, (job_id, tasks) in enumerate(bottle_table.items()):
            ax2.axhline(y=task_height * idx - task_height / 2, color="grey", linestyle="--", linewidth=0.5)

        ax2.set_ylim(ymin=-task_height / 2, ymax=task_height * len(bottle_table) - task_height / 2)
        ax2.set_xlim(xmin=0)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Experiments")
        ax2.set_title("Experiment Perspective")

        xticks = list(ax2.get_xticks())
        xticks.extend([cur_ptr, makespan])
        ax2.set_xticks(xticks)

        ax1.set_xlim(ax2.get_xlim())

        # Carbon and electricity-price panel.
        info = jobs_data.get("info", {})
        peak_price = info.get("peak_price", 0.6)
        valley_price = info.get("valley_price", 0.3)

        machine_power_info = info.get("machine_power_info", {})  # dict: {m_id: power_w}
        x_max = makespan

        # -------- Deduplicate by machine/start/end to construct batches --------
        # key: (m_id, start, end) -> one batch
        dedup_batches = {}
        for m_id, tasks in ws_table.items():
            for task in tasks:
                s = int(task["start_time"])
                e = int(task["end_time"])
                key = (m_id, s, e)
                dedup_batches[key] = True

        # Total power at each instant from deduplicated batches.
        total_power_scaled = [0] * (x_max + 1)
        for m_id, s, e in dedup_batches.keys():
            power_w = machine_power_info.get(m_id, 1000.0)
            power_int = int(builtins.round(power_w))  # Consistent with A.
            for t in range(max(0, s), min(x_max, e)):
                total_power_scaled[t] += power_int

        price_segments = build_price_segments_for_horizon({"info": info}, int(x_max) + 1)
        price_curve = [peak_price] * (int(x_max) + 1)
        for seg_start, seg_end, price in price_segments:
            price_curve[seg_start:seg_end] = [price] * (seg_end - seg_start)

        # Cumulative electricity-cost/carbon-objective curve.
        # Step increment is approximately overlap_minutes * int(price * power_int).
        # At one-minute resolution: inc = 1 * int(price * total_power_int).
        cum_cost_scaled = [0] * (x_max + 1)
        for t in range(x_max):
            inc = int(price_curve[t] * total_power_scaled[t])
            cum_cost_scaled[t + 1] = cum_cost_scaled[t] + inc

        # Left axis: tariff step curve.
        x = list(range(x_max + 1))
        ax4.step(x, price_curve, where="post", color="tab:orange", linewidth=1.8, label="TOU Price")
        ax4.set_ylabel("Price(scaled)", color="tab:orange")
        ax4.tick_params(axis="y", labelcolor="tab:orange")
        ax4.grid(axis="x", linestyle="--", alpha=0.3)
        # Right axis: cumulative integral.
        ax4_r = ax4.twinx()
        ax4_r.plot(x, cum_cost_scaled, color="tab:green", linewidth=2.0, label="Integrated Cost")
        ax4_r.set_ylabel("Integrated Cost(scaled)", color="tab:green")
        ax4_r.tick_params(axis="y", labelcolor="tab:green")

        # Reference lines.
        ax4.axvline(x=cur_ptr, color="b", linestyle="-")
        ax4.axvline(x=makespan, color="r", linestyle="-")
        ax4.set_xlabel("Time")
        ax4.set_title("TOU Curve & Integrated Cost")
        # Merge legends.
        h1, l1 = ax4.get_legend_handles_labels()
        h2, l2 = ax4_r.get_legend_handles_labels()
        ax4.legend(h1 + h2, l1 + l2, loc="upper left")

        plt.tight_layout()
        plt.savefig(img_name)
