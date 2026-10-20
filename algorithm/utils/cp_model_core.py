import collections
from typing import DefaultDict, Dict, List, Tuple

from ortools.sat.python import cp_model
from pydantic import BaseModel, ConfigDict


class Task(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: cp_model.IntVar
    end: cp_model.IntVar
    interval: cp_model.IntervalVar
    duration: cp_model.IntVar | None = None


class OptionalTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: cp_model.IntVar
    end: cp_model.IntVar
    interval: cp_model.IntervalVar
    presence: cp_model.IntVar
    duration: cp_model.IntVar | None = None


def build_cp_model_core(
    jobs_data,
    horizon,
    lock=True,
    rightnow=True,
    require_lock_info=False,
    require_rightnow_info=False,
):
    """Build the variables, batches, and base constraints shared by the three CP scheduling entry points.

    Upper-level entry points handle green objectives, solver time limits, LNS neighborhoods, and hints; this function keeps only
    the scheduling semantics required by standard CP, green CP, and RL+LNS.
    """
    machines_list = list(
        {m_id for job in jobs_data["detail"].values() for machines, *_ in job for m_id in machines}
    )
    batch_capacities = {
        m_id: capacity for m_id, capacity in enumerate(jobs_data["info"]["machine_capacities"])
    }
    machines_available_time = {
        m_id: time for m_id, time in enumerate(jobs_data["info"]["machines_available_time"])
    }

    model = cp_model.CpModel()
    all_tasks: DefaultDict[Tuple[int, int], Task] = collections.defaultdict(Task)
    tasks_to_machine: DefaultDict[Tuple[int, int], Dict[int, OptionalTask]] = collections.defaultdict(dict)

    for job_id, job in jobs_data["detail"].items():
        for task_id, (machines, durations, _) in enumerate(job):
            # Create interval variables for primary tasks.
            start_var = model.new_int_var(0, horizon, f"start_J{job_id}_T{task_id}")
            end_var = model.new_int_var(0, horizon, f"end_J{job_id}_T{task_id}")
            duration_var = model.new_int_var(0, horizon, f"duration_J{job_id}_T{task_id}")
            interval_var = model.new_interval_var(
                start_var,
                duration_var,
                end_var,
                f"I_J{job_id}_T{task_id}",
            )
            all_tasks[job_id, task_id] = Task(
                start=start_var,
                end=end_var,
                interval=interval_var,
                duration=duration_var,
            )

            # Create interval and presence variables for each machine alternative.
            for m_id, duration in zip(machines, durations):
                presence = model.NewBoolVar(f"presence_J{job_id}_T{task_id}_M{m_id}")
                start_var = model.NewIntVar(
                    0, horizon, f"start_J{job_id}_T{task_id}_M{m_id}"
                )
                end_var = model.NewIntVar(0, horizon, f"end_J{job_id}_T{task_id}_M{m_id}")
                interval_var = model.new_optional_interval_var(
                    start_var,
                    duration,
                    end_var,
                    presence,
                    f"B{job_id}-T{task_id}-M{m_id}-D{duration}",
                )
                tasks_to_machine[job_id, task_id][m_id] = OptionalTask(
                    start=start_var,
                    end=end_var,
                    interval=interval_var,
                    presence=presence,
                )

    # Routing: assign each task to exactly one machine.
    for task_key, m_optional_tasks in tasks_to_machine.items():
        model.add(sum([ot.presence for ot in m_optional_tasks.values()]) == 1)
        for optional_task in m_optional_tasks.values():
            model.add(all_tasks[task_key].start == optional_task.start).OnlyEnforceIf(
                optional_task.presence
            )
            model.add(all_tasks[task_key].end == optional_task.end).OnlyEnforceIf(
                optional_task.presence
            )

    # Sequencing: enforce operation precedence within each job.
    for job_id, job in jobs_data["detail"].items():
        for task_id in range(len(job) - 1):
            model.add(all_tasks[job_id, task_id].end <= all_tasks[job_id, task_id + 1].start)

    # Dynamic batching and capacity constraints.
    tasks_per_machine_count = collections.defaultdict(int)
    for job_id, job in jobs_data["detail"].items():
        for task_id, (machines, _, _) in enumerate(job):
            for m_id in machines:
                tasks_per_machine_count[m_id] += 1

    machine_batches: DefaultDict[int, List[OptionalTask]] = collections.defaultdict(list)
    for m_id in machines_list:
        for i in range(tasks_per_machine_count[m_id]):
            batch_itv_start = model.NewIntVar(0, horizon, f"batch_start_M{m_id}_B{i}")
            batch_itv_duration = model.NewIntVar(0, horizon, f"batch_duration_M{m_id}_B{i}")
            batch_itv_end = model.NewIntVar(0, horizon, f"batch_end_M{m_id}_B{i}")
            batch_itv_presence = model.NewBoolVar(f"batch_presence_M{m_id}_B{i}")
            batch_itv = model.NewOptionalIntervalVar(
                batch_itv_start,
                batch_itv_duration,
                batch_itv_end,
                batch_itv_presence,
                f"batch_I_M{m_id}_B{i}",
            )
            machine_batches[m_id].append(
                OptionalTask(
                    start=batch_itv_start,
                    end=batch_itv_end,
                    interval=batch_itv,
                    presence=batch_itv_presence,
                    duration=batch_itv_duration,
                )
            )

    task_to_batch = collections.defaultdict(dict)
    for (job_id, task_id), m_optional_tasks in tasks_to_machine.items():
        for m_id, optional_task in m_optional_tasks.items():
            presence = optional_task.presence
            is_in_batch = [
                model.NewBoolVar(f"is_in_batch_J{job_id}_T{task_id}_M{m_id}_B{i}")
                for i in range(tasks_per_machine_count[m_id])
            ]
            task_to_batch[job_id, task_id][m_id] = is_in_batch

            model.Add(sum(is_in_batch) == 1).OnlyEnforceIf(presence)
            model.Add(sum(is_in_batch) == 0).OnlyEnforceIf(presence.Not())

            for i in range(tasks_per_machine_count[m_id]):
                batch_itv = machine_batches[m_id][i].interval
                batch_itv_presence = machine_batches[m_id][i].presence
                model.Add(optional_task.interval.size_expr() == batch_itv.size_expr()).OnlyEnforceIf(
                    is_in_batch[i]
                )
                model.Add(optional_task.start == batch_itv.start_expr()).OnlyEnforceIf(
                    is_in_batch[i]
                )
                model.Add(batch_itv_presence == True).OnlyEnforceIf(is_in_batch[i])

    for m_id in machines_list:
        for i in range(tasks_per_machine_count[m_id]):
            batch_itv_presence = machine_batches[m_id][i].presence
            batch_members_count = sum(
                task_to_batch[task_key][m_id][i]
                for task_key in task_to_batch
                if m_id in task_to_batch[task_key]
            )
            model.Add(batch_members_count <= batch_capacities[m_id]).OnlyEnforceIf(batch_itv_presence)

    for m_id in machines_list:
        model.AddNoOverlap(ot.interval for ot in machine_batches[m_id])

    # Availability: tasks must start after the assigned machine becomes available.
    for task_key, m_optional_tasks in tasks_to_machine.items():
        for m_id, optional_task in m_optional_tasks.items():
            model.Add(optional_task.start >= machines_available_time.get(m_id, 0))

    # Workstation parameters: tasks with different parameters cannot overlap on one machine.
    workstation_param_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for job_id, job in jobs_data["detail"].items():
        for task_id, (machines, _, param_id) in enumerate(job):
            for m_id in machines:
                workstation_param_dict[m_id][param_id].append(
                    tasks_to_machine[job_id, task_id][m_id].interval
                )

    for m_id in workstation_param_dict.keys():
        param_ids = list(workstation_param_dict[m_id].keys())
        for i in range(len(param_ids)):
            for j in range(i + 1, len(param_ids)):
                for itv_comb1 in workstation_param_dict[m_id][param_ids[i]]:
                    for itv_comb2 in workstation_param_dict[m_id][param_ids[j]]:
                        model.add_no_overlap([itv_comb1, itv_comb2])

    if lock:
        job_lock = jobs_data["info"]["job_lock"] if require_lock_info else jobs_data["info"].get("job_lock")
        if job_lock is not None:
            for job_id, job in jobs_data["detail"].items():
                for task_id, *_ in enumerate(job):
                    if (
                        task_id + 1 < len(job)
                        and job_lock[job_id][task_id] == True
                        and job_lock[job_id][task_id + 1] == True
                    ):
                        model.add(all_tasks[job_id, task_id].end == all_tasks[job_id, task_id + 1].start)

    if rightnow:
        job_rightnow = (
            jobs_data["info"]["job_rightnow"]
            if require_rightnow_info
            else jobs_data["info"].get("job_rightnow")
        )
        if job_rightnow is not None:
            job_avail_time = jobs_data["info"]["job_avail_time"]
            for job_id, job in jobs_data["detail"].items():
                if job_rightnow[job_id]:
                    model.add(all_tasks[job_id, 0].start == job_avail_time[job_id])
                else:
                    model.add(all_tasks[job_id, 0].start >= job_avail_time[job_id])

    batch_context = {
        "machine_batches": machine_batches,
        "task_to_batch": task_to_batch,
        "tasks_per_machine_count": tasks_per_machine_count,
    }
    return (
        model,
        all_tasks,
        tasks_to_machine,
        machines_list,
        batch_capacities,
        batch_context,
    )


def add_makespan_variable(model, jobs_data, all_tasks, horizon):
    """Create and constrain the makespan variable shared by all entry points."""
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    last_tasks_ends = [all_tasks[job_id, len(job) - 1].end for job_id, job in jobs_data["detail"].items()]
    model.AddMaxEquality(makespan_var, last_tasks_ends)
    return makespan_var


def extract_ws_table(solver, jobs_data, tasks_to_machine):
    """Extract the scheduler's ``ws_table`` format from a solved CP-SAT solution."""
    ws_table = collections.defaultdict(list)
    for job_id, job in jobs_data["detail"].items():
        for task_id, (machines, _, task_param) in enumerate(job):
            task_key = (job_id, task_id)
            for m_id in machines:
                optional_task = tasks_to_machine[task_key][m_id]
                if solver.BooleanValue(optional_task.presence):
                    itv = optional_task.interval
                    ws_table[m_id].append(
                        {
                            "name": itv.name,
                            "ws_code_fjspb": str(m_id),
                            "start_time": solver.Value(itv.start_expr()),
                            "end_time": solver.Value(itv.end_expr()),
                            "duration": solver.Value(itv.size_expr()),
                            "expr_no": itv.name.split("-")[0][1:],
                            "fjspb_index": int(itv.name.split("-")[1][1:]),
                            "param": task_param,
                        }
                    )
    for ws_name in ws_table:
        ws_table[ws_name].sort(key=lambda x: x["start_time"])
    return ws_table
