import time
import copy
import sys, os
from pprint import pprint
from colorama import Fore, Style

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import torch
import gymnasium as gym

from rl.ppo_model import Memory, PPO
from rl.env.case_generator import CaseGenerator

from rl.utils.draw_gantt import transfer_data_form, draw_beautiful_gantt
from rl.utils.my_utils import group_args_into_dict
from rl.utils.args import parse_arguments, set_torch_params


def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    i = 0
    while not done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()

    validate_gantt_result = env.validate_gantt()

    # Verify the solution
    gantt_result = validate_gantt_result[0]
    # FJSP scheduling result.
    scheduled_result = validate_gantt_result[1]
    # First operation ID of each job, for plotting.
    first_ope_id_in_each_job = validate_gantt_result[2]

    if not gantt_result:
        print("Scheduling Error!")

    return (
        copy.deepcopy(env.makespan_batch.cpu()),
        validate_gantt_result,
        scheduled_result[:, :, 3].max(dim=1).values.cpu(),
    )


def fespb_rl(jobs_data, env_params, model, memories):
    start = time.time()

    # Convert jobs_data machine IDs from one-based to zero-based indexing.
    machines_list = list(
        {m - 1 for job in jobs_data["detail"].values() for machines, _, _ in job for m in machines}
    )
    batch_capacities = {
        m_idx: capacity for m_idx, capacity in enumerate(jobs_data["info"]["machine_capacities"])
    }




    env_params["num_jobs"], env_params["num_machines"], _ = CaseGenerator.fetch_basic_info(jobs_data)

    env = gym.make(
        "fjsp-v0", case=[jobs_data] * env_params["batch_size"], env_params=env_params, data_source="dict"
    )

    makespans, validate_gantt_results, whole_makespans = schedule(env, model, memories, flag_sample=True)
    env.reset()


    # Find the minimum makespan.
    min_makespan, min_index = torch.min(whole_makespans, dim=0)

    fjsp_result = (
        env_params["num_jobs"],
        env_params["num_machines"],
        validate_gantt_results[1][min_index].cpu(),
        validate_gantt_results[2][min_index].cpu(),
        validate_gantt_results[3][min_index].cpu(),
    )

    print(
        f"RL best makespan: {Fore.YELLOW}{min_makespan}{Style.RESET_ALL}; spend: {Fore.YELLOW}{time.time() - start:.3f}s{Style.RESET_ALL}; Vaild Makespans: {Fore.YELLOW}{whole_makespans}{Style.RESET_ALL}"
    )

    # Convert the data format.
    ws_table, makespan = transfer_data_form(fjsp_result[0], fjsp_result[1], fjsp_result[2], fjsp_result[3])

    return makespan, ws_table, batch_capacities, machines_list, fjsp_result
