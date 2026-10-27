import time
import torch
import numpy as np
from pprint import pprint
from collections import OrderedDict, defaultdict

from algorithm.utils.green_objective import _get_tariff_boundary_minutes
from algorithm.utils.tariff import daily_tariff


# Parse standard-format FJSP data
# Input:
# lines: A standard-format test instance
# num_opes: Maximum total number of operations in this batch of instances
# Output:
# matrix_proc_time: Processing-time matrix; processing time of each operation on each eligible machine, [num_opes, num_mas]
# matrix_ope_ma_adj: Operation-machine adjacency matrix, [num_opes, num_mas],
#  a bool matrix where 1 means the operation in this row can run on the machine in this column
# matrix_pre_proc: Operation precedence adjacency matrix, representing precedence between operations, [num_opes, num_opes]
# matrix_pre_proc.t(): Transpose of the matrix above, representing successor relationships, [num_opes, num_opes]
# torch.tensor(job_id_for_each_oper).int(): 1-D array; each operation ID maps to its job row
# torch.tensor(first_oper_id_of_each_job).int(): Index of the first operation in each job
# torch.tensor(num_opers_of_each_job).int(): Array storing the number of operations in each job
# matrix_cal_cumul：[num_opes, num_opes]
#    Cumulative count of matrix_pre_proc above, representing all operation precedence relationships in each job
#    For example, matrix_pre_proc records that 1 precedes 2, 2 precedes 3, 3 precedes 4, and 4 precedes 5
#    matrix_cal_cumul records that 1 precedes 2,3,4,5; 2 precedes 3,4,5; 3 precedes 4,5; and so on
def load_task_info(task_info_dict, num_opes):
    """
    Load the local FJSP instance.
    """





    num_mas = task_info_dict["info"]["num_machines"]

    # ---------------------------------------------------------------------------------------
    # Fill the processing-time matrix with each operation's processing time on each eligible machine, [num_opes, num_mas]
    # The row is the operation ID; columns store that operation's processing time on each machine
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    oper_idx = 0  # Operation IDs start at 0
    for task_id, operations in task_info_dict["detail"].items():
        for machines, proc_times, param in operations:
            for machine, proc_time in zip(machines, proc_times):
                matrix_proc_time[oper_idx, machine - 1] = proc_time  # Machine IDs start at 1; subtract 1 to make them 0-based
            oper_idx += 1  # Move to the next operation


    # ---------------------------------------------------------------------------------------
    # Fill the operation-machine adjacency matrix, with the same shape as matrix_proc_time, [num_opes, num_mas]
    # A bool matrix where 1 means the operation in this row can run on the machine in this column
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)


    # ---------------------------------------------------------------------------------------
    # Operation precedence adjacency matrix; each element represents precedence between operations, [num_opes, num_opes]
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    oper_idx = 0  # Operation IDs start at 0
    for task_id, operations in task_info_dict["detail"].items():
        for i in range(len(operations) - 1):
            # The current operation precedes the next operation
            matrix_pre_proc[oper_idx + i, oper_idx + i + 1] = True
        oper_idx += len(operations)  # Move to the next job's operations


    # ---------------------------------------------------------------------------------------
    # Fill job_id_for_each_oper, [num_opes]; each operation ID maps to its job
    job_id_for_each_oper = np.zeros(num_opes)
    oper_idx = 0  # Operation IDs start at 0
    for task_id, operations in task_info_dict["detail"].items():
        num_operations = len(operations)
        job_id_for_each_oper[oper_idx : oper_idx + num_operations] = task_id  # Fill in the current job's operations
        oper_idx += num_operations  # Move to the next job's operations


    # ---------------------------------------------------------------------------------------
    # Index of the first operation in each job, [num_jobs]
    first_oper_id_of_each_job = []
    oper_idx = 0  # Operation IDs start at 0
    for task_id, operations in task_info_dict["detail"].items():
        first_oper_id_of_each_job.append(oper_idx)  # Record the first operation index of the current job
        oper_idx += len(operations)  # Move to the next job's operations


    # ---------------------------------------------------------------------------------------
    # Array, [num_jobs], storing the number of operations in each job
    num_opers_of_each_job = []
    for task_id, operations in task_info_dict["detail"].items():
        num_opers_of_each_job.append(len(operations))  # Record the number of operations in the current job


    # ---------------------------------------------------------------------------------------
    # Cumulative count of matrix_pre_proc above, representing all operation precedence relationships in each job
    # matrix_cal_cumul records that 1 precedes 2,3,4,5; 2 precedes 3,4,5; 3 precedes 4,5; and so on
    # Fill matrix_cal_cumul
    matrix_cal_cumul = matrix_pre_proc.clone()
    for _ in range(num_opes):
        matrix_cal_cumul_float = matrix_cal_cumul.float()
        matrix_cal_cumul = matrix_cal_cumul | (matrix_cal_cumul_float @ matrix_cal_cumul_float).bool()


    # ---------------------------------------------------------------------------------------
    # Machine capacity
    machine_capacity = task_info_dict["info"]["machine_capacities"]


    # ---------------------------------------------------------------------------------------
    # Parameters for each operation, [num_opes]
    opers_param = torch.zeros(size=(num_opes, num_mas))
    oper_idx = 0
    for task_id, operations in task_info_dict["detail"].items():
        for machines, proc_times, param in operations:
            for machine in machines:
                opers_param[oper_idx, machine - 1] = param
            oper_idx += 1  # Move to the next operation


    # ---------------------------------------------------------------------------------------
    # Machine available times
    if task_info_dict["info"].get("machines_available_time") is None:
        machines_available_time = [0 for _ in range(len(machine_capacity))]
    else:
        machines_available_time = task_info_dict["info"]["machines_available_time"]


    # ---------------------------------------------------------------------------------------

    if task_info_dict["info"].get("job_avail_time") is None:
        job_avail_time = [0 for _ in range(len(num_opers_of_each_job))]
    else:
        job_avail_time = task_info_dict["info"]["job_avail_time"]
        if isinstance(job_avail_time, dict):
            job_avail_time = [job_avail_time[key] for key in sorted(job_avail_time)]

    # ---------------------------------------------------------------------------------------


    if task_info_dict["info"].get("job_rightnow") is None:
        job_rightnow = [False for _ in range(len(num_opers_of_each_job))]
    else:
        job_rightnow = task_info_dict["info"]["job_rightnow"]
        if isinstance(job_rightnow, dict):
            job_rightnow = [job_rightnow[key] for key in sorted(job_rightnow)]

    # ---------------------------------------------------------------------------------------
    # Seamless-link constraints
    if task_info_dict["info"].get("job_lock") is not None:
        d = task_info_dict["info"]["job_lock"]
        job_lock = [x for k in sorted(d) for x in d[k]]
        # Pad the remaining entries up to num_opes with 0
        job_lock += [0] * (num_opes - len(job_lock))
    else:
        job_lock = [0 for _ in range(num_opes)]


    # ---------------------------------------------------------------------------------------
    # Machine power
    machine_power_info = task_info_dict["info"]["machine_power_info"]

    # ---------------------------------------------------------------------------------------
    # Compute beta, the objective weight for makespan and carbon_emission, adaptively scaling them to the same order of magnitude
    machine_option_counts = defaultdict(int)
    for job in task_info_dict["detail"].values():
        for machines, _, _ in job:
            for m in machines:
                machine_option_counts[m - 1] += 1
    total_option_count = sum(machine_option_counts.values())

    # 2) Weighted average machine power (W)
    weighted_power_w = sum(
        machine_option_counts[m] * machine_power_info[m] for m in machine_option_counts
    ) / (total_option_count + 1e-6)

    # 3) Weighted average electricity price
    whole_info = task_info_dict["info"]
    peak_price, valley_price = (
        whole_info.get("peak_price", 0.6),
        whole_info.get("valley_price", 0.3),
    )
    if "tariff_periods" in whole_info:
        minutes = {key: 0 for key in ("valley_price", "peak_price", "flat_price")}
        for start, end, key in daily_tariff(whole_info):
            minutes[key] += end - start

        avg_price = (valley_price * minutes["valley_price"] + peak_price * minutes["peak_price"]
                     + whole_info.get("flat_price", 0) * minutes["flat_price"]) / (24 * 60)
    else:
        # The shared parser accepts YOLO's minute-precision "HH:MM" boundaries.
        valley_start_min = _get_tariff_boundary_minutes(whole_info, "valley_start")
        valley_end_min = _get_tariff_boundary_minutes(whole_info, "valley_end")
        if valley_start_min < valley_end_min:
            valley_minutes = valley_end_min - valley_start_min
        else:
            valley_minutes = (24 * 60 - valley_start_min) + valley_end_min
        peak_minutes = 24 * 60 - valley_minutes
        avg_price = (valley_price * valley_minutes + peak_price * peak_minutes) / (24 * 60)
    # 4) Set beta to avg_price * weighted_power
    beta = max(1, int(round(avg_price * weighted_power_w)))

    return (
        matrix_proc_time,
        matrix_ope_ma_adj,
        matrix_pre_proc,
        matrix_pre_proc.t(),
        torch.tensor(job_id_for_each_oper).int(),
        torch.tensor(first_oper_id_of_each_job).int(),
        torch.tensor(num_opers_of_each_job).int(),
        matrix_cal_cumul,
        torch.tensor(machine_capacity).int(),
        opers_param,
        torch.tensor(machines_available_time).float(),
        torch.tensor(job_lock).int(),
        torch.tensor(machine_power_info).int(),
        torch.tensor(beta).int(),
        torch.tensor(job_avail_time).float(),
        torch.tensor(job_rightnow).bool(),
    )
