import copy
import json
import sys, os
import random
import time
import math
import argparse
from collections import deque
from colorama import init, Fore, Back, Style
from pprint import pprint

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

import h5py
import gymnasium as gym
import pandas as pd
import torch
import numpy as np

import swanlab

import rl.ppo_model as ppo_model
from rl.env.case_generator import CaseGenerator
from rl.validate import validate, get_validate_env
from rl.utils.my_utils import group_args_into_dict
from rl.utils.args import parse_arguments, setup_seed
from rl.utils.draw_gantt import transfer_data_form


def main():
    # Seed and arguments
    setup_seed(62)
    args, device = parse_arguments()

    # Load algorithm parameters
    nested_args_dict = group_args_into_dict(args, device)
    env_params, model_params, train_params, env_valid_params = (
        nested_args_dict["env_params"],
        nested_args_dict["model_params"],
        nested_args_dict["train_params"],
        nested_args_dict["env_valid_params"],
    )

    num_jobs = env_params["num_jobs"]
    num_mas = env_params["num_machines"]
    batch_size = env_params["batch_size"]

    memories = ppo_model.Memory()
    model = ppo_model.PPO(model_params, train_params, num_envs=batch_size)

    best_models = deque()
    makespan_best = float("inf")

    # Data logger
    str_time = time.strftime("%m%d%H%M")
    swanlab.init(
        project="Fespb-rl",
        workspace="Speed",
        experiment_name=f"J{num_jobs}M{num_mas}_{str_time}_lrstep({args.lr_step_size})_lrgamma({args.lr_gamma})",
        config={"algorithm": "cp"},
    )

    print("PyTorch device: ", device)
    print("Generate train cases." if args.gen_case else "Get train cases from file.")

    save_path = f"./data/runs/J{num_jobs}M{num_mas}_{str_time}_lrstep({args.lr_step_size})_lrgamma({args.lr_gamma})"
    os.makedirs(save_path, exist_ok=True)

    start_time = time.time()

    env = None
    env_valid = get_validate_env(env_valid_params)  # Create an environment for validation

    train_file_path = f"./data/data_train/{env_params['num_jobs']:02d}{env_params['num_machines']:02d}/"
    train_files = [os.path.join(train_file_path, f) for f in os.listdir(train_file_path)]

    file_index = 0
    eligible_solutions = []
    train_makespans = []  # Accumulate training makespans within each save_timestep interval
    for i in range(1, args.epoch + 1):
        print(Fore.GREEN + Style.BRIGHT + f"Iteration: {i}" + Style.RESET_ALL)

        # Recreate the training instances every parallel_iter iterations.
        if (i - 1) % train_params["parallel_iter"] == 0:
            if env is not None:
                env.close()

            if args.gen_case:
                # Randomly generate the number of operations for each job as an array
                case_generator = CaseGenerator(num_jobs, num_mas, flag_same_opes=False)
                env = gym.make(
                    "fjsp-v0", case=case_generator, env_params=env_params, data_source="case", train=True
                )
            else:
                start_idx = file_index
                end_idx = start_idx + batch_size

                if end_idx <= len(train_files):
                    selected_files = train_files[start_idx:end_idx]
                else:
                    remaining = batch_size - (len(train_files) - start_idx)
                    selected_files = train_files[start_idx:] + train_files[:remaining]

                file_index = (file_index + batch_size) % len(train_files)

                env = gym.make(
                    "fjsp-v0", case=selected_files, env_params=env_params, data_source="file", train=True
                )
            print(f"num_job: {num_jobs}\tnum_mas: {num_mas}")

        # Get state and completion signal
        state = env.state
        done = False

        # dones is true only when the entire batch is complete
        dones = env.done_batch

        last_time = time.time()

        # Schedule in parallel
        # Let the agent finish the epoch and save the data
        # Accumulate act and step durations
        act_time_total = 0.0
        step_time_total = 0.0
        while not done:
            t_act_start = time.time()
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            t_act_end = time.time()
            act_time_total += t_act_end - t_act_start

            t_step_start = time.time()
            state, rewards, dones = env.step(actions)
            t_step_end = time.time()
            step_time_total += t_step_end - t_step_start

            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)

        print(
            "Collect data spend time: ",
            round(time.time() - last_time, 3),
            f"| act: {act_time_total:.3f}s | step: {step_time_total:.3f}s",
        )  # Print the act and step durations separately

        # Verify the solution
        # Check that the result is valid
        result_set = env.validate_gantt()
        if not result_set[0]:
            print("Scheduling Error!!")


        eligible_solutions.append(result_set[1][:, :, 1].cpu().numpy())
        with h5py.File(f"{save_path}/eligible_solutions.h5", "a") as h5f:
            h5f.create_dataset(f"solution_{i}", data=result_set[1][:, :, 1].cpu().numpy())

        # Capture the training makespan before reset (reset reinitializes makespan_batch)
        train_makespans.append(env.makespan_batch.mean().cpu().item())

        env.reset()
        # Update the policy every update_timestep iterations.
        if i % train_params["update_timestep"] == 0:
            # The update function backpropagates gradients to train the PPO network
            last_time = time.time()
            loss, reward, (policy_loss, value_loss, entropy_loss) = model.update(
                memories, env_params, train_params
            )
            print(f"Update net spend time: {round(time.time() - last_time, 3)}")
            print(
                f"Reward: {reward:.3f}; loss: {loss:.3f} = {policy_loss:.3f} + {value_loss:.3f} + {entropy_loss:.3f}"
            )
            memories.clear_memory()

            if i % (20 * train_params["update_timestep"]) == 0:
                torch.cuda.empty_cache()

            swanlab.log(
                {
                    "Loss/train": loss,
                    "PolicyLoss/train": policy_loss,
                    "ValueLoss/train": value_loss,
                    "EntropyLoss/train": entropy_loss,
                    "Reward/train": reward,
                },
                step=i,
            )

        # Validate the policy every save_timestep iterations.
        if i % train_params["save_timestep"] == 0:
            print("\nStart validating")
            vali_result_ave, _ = validate(env_valid_params, env_valid, model.policy_old)

            # Compute the mean training makespan for this interval and log it with validation
            train_makespan_ave = sum(train_makespans) / len(train_makespans) if train_makespans else 0.0
            swanlab.log(
                {"Makespan/valid": vali_result_ave.item(), "Makespan/train": train_makespan_ave},
                step=i,
            )
            train_makespans.clear()  # Clear the list and start accumulating for the next interval

            # Save the best model
            if vali_result_ave < makespan_best:
                makespan_best = vali_result_ave
                if best_models:  # Save the best model
                    os.remove(best_models.popleft())
                save_file = f"{save_path}/J{num_jobs}_M{num_mas}_iter{i}.pt"
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

    for e in (env, env_valid):
        if e is not None:
            e.close()

    print("total_time: ", time.time() - start_time)

    swanlab.finish()


if __name__ == "__main__":
    main()
