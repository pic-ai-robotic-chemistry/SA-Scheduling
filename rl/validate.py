import gymnasium as gym
import torch
import time
import os
import copy

import rl.env
import rl.ppo_model as ppo_model


def get_validate_env(env_params):
    """
    Generate and return the validation environment from the validation set ()
    """
    file_path = f"./data/data_val/{env_params['num_jobs']:02d}{env_params['num_machines']:02d}/"
    valid_data_files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    return gym.make("fjsp-v0", case=valid_data_files, env_params=env_params, data_source="file")


def validate(env_params, env, model_policy):
    """
    Validate the policy during training, and the process is similar to test
    """
    start = time.time()
    batch_size = env_params["batch_size"]
    memory = ppo_model.Memory()
    print("There are {0} dev instances.".format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while not done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=True, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error!!!!!")
    makespan = env.makespan_batch.mean().cpu().detach().clone()
    makespan_batch = env.makespan_batch.cpu().detach().clone()
    env.reset()
    print("validating time: ", time.time() - start, "\n")
    return makespan, makespan_batch
