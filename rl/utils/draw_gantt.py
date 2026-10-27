import numpy as np
import random
import torch

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors

from pprint import pprint
from collections import defaultdict

colors = list(mcolors.CSS4_COLORS.keys())


# Draw a Gantt chart
# Note: scheduled_result and num_operation_biases are batch-formatted algorithm outputs; select the batch to plot
def draw_beautiful_gantt(ws_table, makespan, mas_capacity_batch, show_render=False, fig_name="gantt.svg"):


    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)

    # Set task height
    task_height = 0.3

    machines_list = sorted(list(ws_table.keys()))

    # Find the maximum number of tasks processed simultaneously on each machine
    machine_max_tasks = {}
    for m_id in machines_list:
        tasks = ws_table[m_id]
        # Use a dictionary to count tasks at each start time
        time_task_count = {}
        for task in tasks:
            start_time = task["start_time"]
            time_task_count[start_time] = time_task_count.get(start_time, 0) + 1

        machine_max_tasks[m_id] = max(time_task_count.values(), default=0)

    machine_y_height = {m_id: task_height * machine_max_tasks[m_id] for m_id in machines_list}

    machine_y_start = {}
    cumulative_height = 0
    for m_id in machines_list:
        machine_y_start[m_id] = cumulative_height
        cumulative_height += machine_y_height[m_id]

    for m_index, m_id in enumerate(ws_table.keys()):
        current_y_base = machine_y_start[m_id]
        last_end = -1
        for task in ws_table[m_id]:
            job_id = task["name"].split("-")[0][1:]  # Extract the job ID from the task name

            # Calculate the position and size of the bar
            bar_height = task_height

            if last_end <= task["start_time"]:
                current_y = current_y_base
            else:
                current_y += bar_height
            last_end = task["end"]

            ax.broken_barh(
                [(task["start_time"], task["duration"])],
                (current_y, bar_height),
                facecolors=mcolors.CSS4_COLORS[colors[hash(job_id) % len(colors)]],
                edgecolor="black",
            )

            # Add text in the center of the bar
            text_x = task["start_time"] + task["duration"] / 2
            text_y = current_y + (bar_height / 2)
            ax.text(
                text_x,
                text_y,
                f"J{job_id}-T{task['fjspb_index']}-M{task['ws_code_fjspb']}-D{task['duration']}",
                ha="center",
                va="center",
            )

    # Draw a vertical line at the makespan
    ax.axvline(x=makespan, color="r", linestyle="--", label="Makespan")

    ax.annotate(
        f"x={makespan}",
        xy=(makespan, 1),
        xytext=(makespan, 0.5),
        arrowprops=dict(arrowstyle="->"),
        va="bottom",
    )

    # Separate the machines
    for m_id in machines_list:
        ax.axhline(y=machine_y_start[m_id], color="grey", linestyle="--", linewidth=0.5)

    yticks = [machine_y_start[m_id] + machine_y_height[m_id] / 2 for m_id in machines_list]
    yticklabels = [f"{m}" for m in machines_list]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Workstation Table - Gantt Chart")

    plt.title("FJSPB")
    plt.tight_layout()

    if show_render:
        plt.show()
    else:
        plt.savefig(fig_name)


# Convert scheduled_result to the data format used to draw a Gantt chart
def transfer_data_form(num_jobs, num_machines, scheduled_result, first_ope_id_in_each_job):

    # Get the job containing the current operation and its index within that job
    def get_ope_idx_by_id(id_ope):
        # Get the job containing the current operation
        idx_job = max([idx for (idx, val) in enumerate(first_ope_id_in_each_job) if id_ope >= val])
        # Get the operation's index within its job
        idx_ope_in_this_job = id_ope - first_ope_id_in_each_job[idx_job]
        return idx_job, idx_ope_in_this_job.item()

    """
    Partial Schedule (state) of operations
        0: Status, whether it is scheduled
        1: Allocated machines
        2: Start time
        3: End time
    """
    schedules = scheduled_result.cpu()

    machine_name = [f"{m_id}" for m_id in range(num_machines)]

    ws_table = defaultdict(list)
    for ope_id, oper_info in enumerate(schedules):
        ws_name = machine_name[int(oper_info[1])]
        duration = oper_info[3].item() - oper_info[2].item()
        job_id, oper_id = get_ope_idx_by_id(ope_id)
        oper_name = f"B{job_id}-T{oper_id}-M{ws_name}-D{duration}"

        ope_feature = {
            "name": oper_name,
            "ws_code_fjspb": int(oper_info[1]),
            "start_time": oper_info[2].item(),
            "end_time": oper_info[3].item(),
            "duration": duration,
            "expr_no": job_id,
            "fjspb_index": oper_id,
            "param": "",
        }
        ws_table[ws_name].append(ope_feature)

    # Sort each machine's tasks by start time
    for ws_name in ws_table:
        ws_table[ws_name].sort(key=lambda x: x["start_time"])

    makespan = torch.max(schedules[:, 3]).item()

    return ws_table, makespan


if __name__ == "__main__":
    pass
