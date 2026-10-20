import sys
import time
import gymnasium as gym
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy

from colorama import init, Fore, Back, Style
from pprint import pprint

from dataclasses import dataclass

from rl.env.load_data import load_task_info
from rl.env.case_generator import CaseGenerator


# Use only for floating-point equality of rightnow times and batch times/durations;
# future/current classification, overlap, and inequality bounds retain exact comparisons.
SCHEDULE_COMPAT_EPS = 1e-3


# Environment state dataclass; stores data only.
@dataclass
class EnvState:
    """
    Class for the state of the environment
    """

    # static
    # Static members remain unchanged after initialization.
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    # Dynamic members are refreshed by update().
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    opers_param_batch: torch.Tensor = None
    oper_seamless_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None
    inplace_drop_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_full_batch: torch.Tensor = None
    step_oper_id_of_each_job_batch: torch.Tensor = None

    partial_machines_batch: torch.Tensor = None

    # No-wait chains and future reservations used by find_eligible_om_pair and step.
    chain_first_ops_batch: dict = None
    chain_info_batch: dict = None
    future_reservations_batch: dict = None

    # Feasible chain paths read by ppo_model.select_chain_machines().
    # {b: {chain_start_op: {m_0: [(m_1, m_2, ...), ...]}}}
    # Each tuple is a complete feasible path excluding m_0, computed by _compute_chain_feasible_paths.
    chain_feasible_paths_batch: dict = None

    # ========== Unified eligibility: computed by find_eligible_om_pair, read directly by ppo_model ==========
    eligible_om_pairs_batch: torch.Tensor = None  # [batch_size, num_jobs, num_mas] bool

    def update(
        self,
        batch_idxes,
        feat_opes_batch,
        feat_mas_batch,
        proc_times_batch,
        opers_param_batch,
        oper_seamless_batch,
        ope_ma_adj_batch,
        mask_job_procing_batch,
        mask_job_finish_batch,
        mask_ma_full_batch,
        step_oper_id_of_each_job_batch,
        time,
        inplace_drop,
        partial_machines_batch,
        chain_first_ops_batch=None,
        chain_info_batch=None,
        future_reservations_batch=None,
        # Feasible chain paths.
        chain_feasible_paths_batch=None,
        # ========== Unified eligibility ==========
        eligible_om_pairs_batch=None,
    ):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.opers_param_batch = opers_param_batch
        self.oper_seamless_batch = oper_seamless_batch
        self.oper_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_full_batch = mask_ma_full_batch
        self.step_oper_id_of_each_job_batch = step_oper_id_of_each_job_batch
        self.time_batch = time
        self.inplace_drop_batch = inplace_drop

        self.partial_machines_batch = partial_machines_batch

        # Update chains and reservations.
        if chain_first_ops_batch is not None:
            self.chain_first_ops_batch = chain_first_ops_batch
        if chain_info_batch is not None:
            self.chain_info_batch = chain_info_batch
        if future_reservations_batch is not None:
            self.future_reservations_batch = future_reservations_batch
        # Update feasible chain paths.
        if chain_feasible_paths_batch is not None:
            self.chain_feasible_paths_batch = chain_feasible_paths_batch
        # ========== Unified eligibility ==========
        if eligible_om_pairs_batch is not None:
            self.eligible_om_pairs_batch = eligible_om_pairs_batch


# Inputs:
# feature_job_batch: e.g. nums_ope_batch, the operation count for each job.
# opes_appertain_batch: 1-D array mapping each operation ID to its job row.
# Output:
# Same shape as opes_appertain_batch;
# replace each entry i with feature_job_batch[opes_appertain_batch[i]].
def convert_feature_job_to_oper(feature_job_batch, opes_appertain_batch):
    """
    Convert job features into operation features (such as dimension)
    """
    return feature_job_batch.gather(1, opes_appertain_batch)


# Reinforcement-learning environment.
class FJSPEnv(gym.Env):
    """
    FJSP environment
    """

    def __init__(self, case, env_params, data_source="case", train=False):
        """
        :param case: The instance generator or the addresses of the instances
        :param env_params: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        """
        # ------------------------ Load configuration parameters ------------------------
        self.params = env_params
        self.train = train

        # Number of parallel environments per training batch.
        self.batch_size = self.params["batch_size"]
        # Number of jobs and machines
        self.num_jobs = self.params["num_jobs"]
        self.num_machines = self.params["num_machines"]
        self.reward_punish = self.params["reward_punish"]

        self.no_batch = self.params["no_batch"]
        self.no_drop = self.params["no_drop"]
        # ========== strict_batch: operations in one batch must have identical processing durations ==========
        self.strict_batch = self.params.get("strict_batch", True)

        # ----------------------------------- Load cases -----------------------------------
        # Maximum total operation count among cases in the training batch.
        self.max_num_opers_in_batch = 0

        # Store batch_size cases.
        batch_lines = []
        for i in range(self.batch_size):
            if data_source == "case":
                # case.get_case(i)[1] is the generated case dictionary.
                batch_lines.append(case.get_case(i)[1])
            elif data_source == "file":  # Load cases from files.
                with open(case[i]) as f:
                    if case[i].endswith(".fjs"):
                        task_info = CaseGenerator.convert_fjs_to_json(f.readlines())
                    elif case[i].endswith(".json"):
                        _, _, task_info = CaseGenerator.convert_json_to_json(json.load(f))
                    batch_lines.append(task_info)
            elif data_source == "dict":
                batch_lines.append(case[i])
            else:
                raise ValueError("Invalid data source. Please use 'case', 'file', or 'dict'.")

            # Read job, machine, and total operation counts.
            num_jobs, num_mas, num_opers = CaseGenerator.fetch_basic_info(batch_lines[i])
            # Track the maximum total operation count in the current batch.
            self.max_num_opers_in_batch = max(self.max_num_opers_in_batch, num_opers)

        # ----------------- Parse durations, adjacency matrices, and other static case data -----------------
        # task_feature_dim is the number of values returned by load_task_info.
        task_feature_dim = len(load_task_info(batch_lines[0], self.max_num_opers_in_batch))
        tensors = [[] for _ in range(task_feature_dim)]
        # Parse standard FJSP data and collect load_data results in tensors.
        for i in range(self.batch_size):
            load_data = load_task_info(batch_lines[i], self.max_num_opers_in_batch)
            for j in range(task_feature_dim):
                tensors[j].append(load_data[j])





        # Processing-time matrix.
        # shape: (batch_size, num_opers, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)

        # Operation-machine adjacency changes with the scheduling state.
        # shape: (batch_size, num_opers, num_mas)
        self.oper_ma_adj_batch = torch.stack(tensors[1], dim=0).long()

        # Cumulative matrix_pre_proc encodes all within-job operation precedence relations.
        # shape: (batch_size, num_opers, num_opers),
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # ----------------------------------------------------------------------------
        # Static features.

        # Machine capacities.
        # shape: (batch_size, num_mas)
        if self.no_batch:  # Disable batching.
            self.machines_capacity_batch = torch.ones_like(torch.stack(tensors[8], dim=0)).int()
        else:
            self.machines_capacity_batch = torch.stack(tensors[8], dim=0)

        # Operation parameters.
        # shape: (batch_size, num_opers, num_mas)
        self.opers_param_batch = torch.stack(tensors[9], dim=0)


        # Initial machine availability times.
        self.machines_available_time_batch_init = torch.stack(tensors[10], dim=0)

        # Append initial job availability to load_task_info outputs to preserve existing indices.
        self.job_avail_time_batch_init = torch.stack(tensors[14], dim=0)

        # rightnow is an environment constraint, not an HGNN feature, preserving checkpoint dimensions.
        self.job_rightnow_batch_init = torch.stack(tensors[15], dim=0).bool()

        # Whether each operation connects seamlessly to its successor.
        self.oper_seamless_batch = torch.stack(tensors[11], dim=0).bool()

        # Machine power.
        self.machine_power_batch = torch.stack(tensors[12], dim=0).int()

        # Relative makespan/carbon weight.
        self.beta_batch = torch.stack(tensors[13], dim=0).int()

        # Operation adjacency encoding predecessor relations.
        # shape: (batch_size, num_opers, num_opers)
        self.oper_pre_adj_batch = torch.stack(tensors[2], dim=0)

        # Transpose operation adjacency to encode successor relations.
        # shape: (batch_size, num_opers, num_opers)
        self.oper_sub_adj_batch = torch.stack(tensors[3], dim=0)

        # represents the mapping between operations and jobs
        # 1-D mapping from operation ID to job row.
        # shape: (batch_size, num_opers),
        self.job_id_for_each_oper_batch = torch.stack(tensors[4], dim=0).long()

        # First operation index of each job.
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.first_oper_id_of_each_job_batch = torch.stack(tensors[5], dim=0).long()

        # Operation count of each job.
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.num_opers_of_each_job_batch = torch.stack(tensors[6], dim=0).long()

        # Last operation index of each job.
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_oper_id_of_each_job_batch = (
            self.first_oper_id_of_each_job_batch + self.num_opers_of_each_job_batch - 1
        )

        # Total operations per case.
        # shape: (batch_size), the number of operations for each instance
        self.num_opers_batch = torch.sum(self.num_opers_of_each_job_batch, dim=1)

        # ----------------------------------------------------------------------------
        # Dynamic operation/machine features and their relations.

        # Indices of unfinished cases in the batch.
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances

        # Current and pre-step time for each case in the batch.
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.last_time = torch.zeros(self.batch_size)

        if self.no_drop:  # Disable drop actions.
            self.inplace_drop = torch.ones(self.batch_size).bool()
        else:  # Allow drop actions.
            self.inplace_drop = torch.zeros(self.batch_size).bool()

        # Scheduled-operation counts per case.
        self.num_scheduled_oper_batch = torch.zeros(self.batch_size).int()  # Count scheduled operations

        # ID of each job's next operation.
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.step_oper_id_of_each_job_batch = self.first_oper_id_of_each_job_batch.detach().clone()

        # ----------------------------------------------------------------------------
        # Dynamic operation and machine feature vectors.
        """
        features, dynamic
            ope:
                0: Status
                1: Number of neighboring machines
                2: Processing time
                3: Number of unscheduled operations in the job
                4: Job completion time
                5: Start time
                6: Mean drop count in the instance
                7: Whether the operation is seamless with its successor
            ma:
                0: Number of neighboring operations
                1: Available time
                2: Utilization
                3: Capacity | speed
                4: Cur_processing_machines_nums | speed
                5: Param, parameter of operations in the active batch; -1 means no parameter; 0 means no operation is processing

        """
        # Generate raw feature vectors
        # Construct initial features with the expected dimensions.

        # shape: (batch_size, feature_dims, num_oper/num_machine)
        # Initialize channels 0 (status) and 6 (drop count) to zero.
        feature_opers_batch = torch.zeros(
            size=(self.batch_size, self.params["oper_feat_dim"], self.max_num_opers_in_batch)
        )
        feature_mas_batch = torch.zeros(size=(self.batch_size, self.params["machine_feat_dim"], num_mas))

        # Use operation-machine adjacency to count eligible machines per operation.
        feature_opers_batch[:, 1, :] = torch.count_nonzero(self.oper_ma_adj_batch, dim=2)

        # Initialize average machine processing time per operation as in the paper.
        # Initialize durations from the processing-time matrix; later updates cannot use this matrix directly.
        feature_opers_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(
            feature_opers_batch[:, 1, :] + 1e-9
        )

        # Initialize each operation with its job's operation count.
        feature_opers_batch[:, 3, :] = convert_feature_job_to_oper(
            self.num_opers_of_each_job_batch, self.job_id_for_each_oper_batch
        )

        # Compute operation start times from predecessor start times and durations, as in the paper.
        # unsqueeze changes feature_opers_batch[:, 2, :] from [batch, num_opers] to [batch, 1, num_opers].
        feature_opers_batch[:, 5, :] = torch.bmm(
            feature_opers_batch[:, 2, :].unsqueeze(1), self.cal_cumul_adj_batch
        ).squeeze(1)  # Remove only bmm's middle dimension; preserve the last dimension for single-operation cases.
        # Map each job's initial availability to its operations' initial start times.
        feature_opers_batch[:, 5, :] += convert_feature_job_to_oper(
            self.job_avail_time_batch_init, self.job_id_for_each_oper_batch
        )

        # Job end times.
        end_time_of_each_job_batch = (feature_opers_batch[:, 5, :] + feature_opers_batch[:, 2, :]).gather(
            1, self.end_oper_id_of_each_job_batch
        )

        # Compute completion times of the jobs containing each operation.
        feature_opers_batch[:, 4, :] = convert_feature_job_to_oper(
            end_time_of_each_job_batch, self.job_id_for_each_oper_batch
        )

        # Whether each operation is seamless with its successor.
        feature_opers_batch[:, 7, :] = self.oper_seamless_batch

        # Use operation-machine adjacency to count neighboring operations per machine.
        feature_mas_batch[:, 0, :] = torch.count_nonzero(self.oper_ma_adj_batch, dim=1)

        # Machine availability times.
        feature_mas_batch[:, 1, :] = self.machines_available_time_batch_init

        # Machine capacities.
        feature_mas_batch[:, 3, :] = self.machines_capacity_batch




        # Store machine and operation features as members.
        self.feature_opers_batch = feature_opers_batch
        self.feature_mas_batch = feature_mas_batch

        # Dynamic global scheduling state.

        # Masks of current status, dynamic
        # Current state masks.

        # True if a job is processing.
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(
            size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False
        )

        # True if a job has finished.
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(
            size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False
        )

        # True if a machine has no remaining capacity.
        # shape: (batch_size, num_mas)
        self.mask_ma_full_batch = torch.full(
            size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False
        )

        # True if a machine is processing.
        self.mask_ma_processing_batch = torch.full(
            size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False
        )
        zero_not_available = feature_mas_batch[:, 1, :] != 0
        self.mask_ma_processing_batch[zero_not_available] = True

        # Global partial schedule: four state values per operation, also storing the final FJSP schedule.
        """
        Partial Schedule (state) of jobs/operations, dynamic
            0: Status
            1: Allocated machines
            2: Start time
            3: End time
        """

        # shape: (batch_size, num_opers, 4)
        self.partial_schedules_batch = torch.zeros(size=(self.batch_size, self.max_num_opers_in_batch, 4))

        # Operation start times.
        self.partial_schedules_batch[:, :, 2] = feature_opers_batch[:, 5, :]

        # Operation end times.
        self.partial_schedules_batch[:, :, 3] = feature_opers_batch[:, 5, :] + feature_opers_batch[:, 2, :]

        # Jobs processing on each machine: [batch_size, num_mas, num_jobs].
        self.jobs_status_in_mas_batch = torch.zeros(
            size=(self.batch_size, self.num_machines, self.num_jobs), dtype=torch.bool
        )

        # Global partial machine state with nine features per machine.
        """
        Partial Schedule (state) of machines, dynamic
            0: idle, whether the machine is idle
            1: available_time
            2: utilization_time
            3: id_ope
            4: processing, whether the machine is processing; if so, the active batch start and end times are valid | speed
            5: cur_num_process_opers, number of operations currently processing | speed
            6: batch start time | speed
            7: batch end time | speed
            8: param, parameter of the active operations; -1 means no parameter, -2 means allocation is forbidden (initially occupied), 0 means no operation is processing, and positive integers are other parameters
        """

        # shape: (batch_size, num_machines, 9)
        self.partial_machines_batch = torch.zeros(size=(self.batch_size, self.num_machines, 9))

        # Initialize channel 1 from initial machine availability.
        self.partial_machines_batch[:, :, 1] = feature_mas_batch[:, 1, :]

        # Machines unavailable at time zero.
        zero_not_available = self.partial_machines_batch[:, :, 1] != 0

        # Mark initially available machines idle and initially occupied machines nonidle.
        self.partial_machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 0][zero_not_available] = 0

        # Processing flag: 0 for available machines, 1 for initially occupied machines.
        self.partial_machines_batch[:, :, 4] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 4][zero_not_available] = 1

        # Processing count: 0 for available machines; 1 denotes initial occupancy.
        self.partial_machines_batch[:, :, 5] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 5][zero_not_available] = 1

        # Initially occupied batches start at -1 and end at machine availability.
        self.partial_machines_batch[:, :, 6] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 7] = torch.zeros(size=(self.batch_size, self.num_machines))

        self.partial_machines_batch[:, :, 6][zero_not_available] = -1
        self.partial_machines_batch[:, :, 7] = self.partial_machines_batch[:, :, 1]

        # Parameters are 0 for idle machines and -2 for initially occupied machines.
        self.partial_machines_batch[:, :, 8] = torch.zeros(size=(self.batch_size, self.num_machines))

        self.partial_machines_batch[:, :, 8][zero_not_available] = -2

        # Makespan is the maximum job completion time; shape: (batch_size).
        self.makespan_batch = torch.max(self.feature_opers_batch[:, 4, :], dim=1)[0]

        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        # No-wait chain and future-reservation structures.
        # Parse no-wait chains per batch from oper_seamless_batch.
        # chain_info: {b: {chain_start_op: [op_ids in chain]}}
        # chain_first_ops: {b: set(op_ids)}, identifying chain-start operations.
        # chain_next_of: {b: {op_id: next_op_id}}, identifying successors within chains.
        self.chain_info = {}
        self.chain_first_ops = {}
        self.chain_next_of = {}
        # Future reservations: chain operations occupy future machine intervals, allowing machine reuse.
        # {batch_id: {m_id: [(start_time, end_time, oper_id, job_id, param), ...]}}
        self.future_reservations = {}

        # Cache paths from find_eligible_om_pair for select_chain_machines in act().
        # {b: {chain_start_op: {m_0: [(m_1, m_2, ...), ...]}}}
        self._chain_feasible_paths = {}

        # ========== Unified eligibility: initialize the cache explicitly ==========
        self._cached_eligible_om_pairs = None  # Populated by find_eligible_om_pair.

        # Parse no-wait chains for all batches.
        for b in range(self.batch_size):
            self._parse_no_wait_chains(b)

        # Account for initial machine availability by advancing to the first time with an eligible O-M pair.
        # If no operation-machine pair is currently eligible, advance to the next feasible event time.
        num_eligible_om_pair_batch = self.find_eligible_om_pair()



        while ~((~((num_eligible_om_pair_batch == 0) & (~self.done_batch))).all()):
            self.transfer_to_next_time(num_eligible_om_pair_batch)
            num_eligible_om_pair_batch = self.find_eligible_om_pair()

        # Use a dedicated class for the graph's many state components.
        self.state = EnvState(
            batch_idxes=self.batch_idxes,
            feat_opes_batch=self.feature_opers_batch,
            feat_mas_batch=self.feature_mas_batch,
            proc_times_batch=self.proc_times_batch,
            opers_param_batch=self.opers_param_batch,
            oper_seamless_batch=self.oper_seamless_batch,
            ope_ma_adj_batch=self.oper_ma_adj_batch,
            ope_pre_adj_batch=self.oper_pre_adj_batch,
            ope_sub_adj_batch=self.oper_sub_adj_batch,
            mask_job_procing_batch=self.mask_job_procing_batch,
            mask_job_finish_batch=self.mask_job_finish_batch,
            mask_ma_full_batch=self.mask_ma_full_batch,
            opes_appertain_batch=self.job_id_for_each_oper_batch,
            step_oper_id_of_each_job_batch=self.step_oper_id_of_each_job_batch,
            end_ope_biases_batch=self.end_oper_id_of_each_job_batch,
            time_batch=self.time,
            inplace_drop_batch=self.inplace_drop,
            nums_opes_batch=self.num_opers_batch,
            partial_machines_batch=self.partial_machines_batch,
            # Pass chain state and future reservations.
            chain_first_ops_batch=self.chain_first_ops,
            chain_info_batch=self.chain_info,
            future_reservations_batch=self.future_reservations,
            # Pass feasible chain paths, initially empty.
            chain_feasible_paths_batch=self._chain_feasible_paths,
            # ========== Unified eligibility ==========
            eligible_om_pairs_batch=self._cached_eligible_om_pairs,
        )

        # Save initial data for reset
        # Save the initial state for reset.
        self.old_proc_times_batch = self.proc_times_batch.detach().clone()
        self.old_opers_param_batch = self.opers_param_batch.detach().clone()
        self.old_oper_seamless_batch = self.oper_seamless_batch.detach().clone()
        self.old_ope_ma_adj_batch = self.oper_ma_adj_batch.detach().clone()
        self.old_cal_cumul_adj_batch = self.cal_cumul_adj_batch.detach().clone()
        self.old_feat_opes_batch = self.feature_opers_batch.detach().clone()
        self.old_feat_mas_batch = self.feature_mas_batch.detach().clone()
        self.old_job_rightnow_batch_init = self.job_rightnow_batch_init.detach().clone()

        self.old_state = copy.deepcopy(self.state)

    # Parse no-wait chains.
    def _parse_no_wait_chains(self, batch_id):
        """
        Parse the no-wait operation chains within one batch.
        Chain definition (consistent with the job_lock constraint in the CP solver):
        A consecutive operation pair (op_i, op_{i+1}) has a no-wait constraint if and only if
        oper_seamless[op_i] == True AND oper_seamless[op_{i+1}] == True。

        Populate self.chain_info[batch_id], self.chain_first_ops[batch_id],
        self.chain_next_of[batch_id]。
        """
        oper_seamless = self.oper_seamless_batch[batch_id]  # [num_opers]
        job_ids = self.job_id_for_each_oper_batch[batch_id]  # [num_opers]
        num_opers_total = int(self.num_opers_batch[batch_id])

        chain_info = {}  # {chain_start_op: [op_ids in chain]}
        chain_first_ops = set()
        chain_next_of = {}  # {op_id: next_op_id}

        for job_id in range(self.num_jobs):
            first_op = int(self.first_oper_id_of_each_job_batch[batch_id, job_id])
            end_op = int(self.end_oper_id_of_each_job_batch[batch_id, job_id])

            if end_op <= first_op:  # Single-operation or invalid job.
                continue

            i = first_op
            while i < end_op:
                # Check the no-wait relation between operations i and i+1.
                # Both must have seamless==True.
                if i < num_opers_total and i + 1 < num_opers_total:
                    if oper_seamless[i] and oper_seamless[i + 1]:
                        # Find a chain start.
                        chain_ops = [i]
                        chain_first_ops.add(i)
                        j = i
                        # Use end_op as the chain-end bound for rightnow jobs and end_op - 1 for ordinary jobs.
                        chain_end_bound = (
                            end_op
                            if bool(self.job_rightnow_batch_init[batch_id, job_id].item())
                            else end_op - 1
                        )
                        while j < chain_end_bound:
                            if j + 1 < num_opers_total:
                                if oper_seamless[j] and oper_seamless[j + 1]:
                                    chain_ops.append(j + 1)
                                    chain_next_of[j] = j + 1
                                    j += 1
                                else:
                                    break
                            else:
                                break
                        chain_info[chain_ops[0]] = chain_ops
                        i = j + 1  # Skip the entire chain.
                        continue
                i += 1

        self.chain_info[batch_id] = chain_info
        self.chain_first_ops[batch_id] = chain_first_ops
        self.chain_next_of[batch_id] = chain_next_of
        # Initialize future reservations.
        if batch_id not in self.future_reservations:
            self.future_reservations[batch_id] = {}


    # Internal single-operation scheduling helper.
    def _schedule_single_operation(self, batch_id, oper_id, mas_id, job_id, start_time, duration):
        """
        Schedule one operation on the specified machine.


        Args:
            batch_id: Scalar batch index
            oper_id: Scalar operation ID
            mas_id: Scalar machine ID
            job_id: Scalar job ID
            start_time: Scalar operation start time (may be in the future for chain operations)
            duration: Scalar processing time (may differ from the original value in batch processing)
        """
        b = batch_id
        op = oper_id
        m = mas_id
        j = job_id
        t = start_time
        current_time = self.time[b].item()

        # Treat any start_time > self.time as a future reservation.
        # Current-view principle: machine features describe active batches only.
        # Use exact future/current ordering, without floating-point equality tolerances.
        is_future_operation = t > current_time

        if is_future_operation:
            # Future reservations do not overwrite active batches or update machine features; available_time stays unchanged.
            end_time = t + duration
            oper_param = self.opers_param_batch[b, op, m].item()
            if m not in self.future_reservations[b]:
                self.future_reservations[b][m] = []
            self.future_reservations[b][m].append((t, end_time, op, j, oper_param))
            # Update operation state: the operation itself is scheduled.
            self.partial_schedules_batch[b, op, :2] = torch.tensor([1.0, float(m)])
            self.partial_schedules_batch[b, op, 2] = t
            self.partial_schedules_batch[b, op, 3] = end_time
            self.feature_opers_batch[b, 0, op] = 1.0
            self.feature_opers_batch[b, 1, op] = 1.0
            self.feature_opers_batch[b, 2, op] = duration
            self.feature_opers_batch[b, 5, op] = t
            # Update oper_ma_adj_batch.
            new_adj = torch.zeros(self.num_machines, dtype=torch.int64)
            new_adj[m] = 1
            self.oper_ma_adj_batch[b, op] = new_adj
            self.proc_times_batch[b, op] *= new_adj
            self.opers_param_batch[b, op] *= new_adj
            # Update cal_cumul_adj_batch.
            if op > 0:
                first_op_of_job = self.first_oper_id_of_each_job_batch[b, j].item()
                if op - 1 >= first_op_of_job:
                    self.cal_cumul_adj_batch[b, op - 1, :] = 0
            # Preserve current-view machine features and partial_machines_batch.
            # Mark jobs_status_in_mas_batch only when the reservation becomes active.
            return duration

        # --- Ordinary scheduling: start_time == self.time; update machine state and all features ---
        # Update jobs_status_in_mas_batch.
        self.jobs_status_in_mas_batch[b, m, j] = True

        # Update oper_ma_adj_batch
        new_adj = torch.zeros(self.num_machines, dtype=torch.int64)
        new_adj[m] = 1
        self.oper_ma_adj_batch[b, op] = new_adj
        self.proc_times_batch[b, op] *= new_adj
        self.opers_param_batch[b, op] *= new_adj

        oper_param = self.opers_param_batch[b, op, m].item()

        # Update partial_machines_batch.
        # Cumulative busy time.
        self.partial_machines_batch[b, m, 2] += duration
        # Current operation's job ID.
        self.partial_machines_batch[b, m, 3] = float(j)
        # Increment the processing-operation count.
        self.partial_machines_batch[b, m, 5] += 1
        # Check whether machine capacity is full.
        capacity = self.machines_capacity_batch[b, m].item()
        cur_count = self.partial_machines_batch[b, m, 5].item()
        self.partial_machines_batch[b, m, 0] = 1.0 if cur_count < capacity else 0.0
        # For an idle machine starting a new batch, set start/end times.
        was_processing = self.partial_machines_batch[b, m, 4].item() == 1.0
        if not was_processing:
            self.partial_machines_batch[b, m, 6] = t  # Batch start time.
            self.partial_machines_batch[b, m, 7] = t + duration  # Batch end time.
            self.partial_machines_batch[b, m, 8] = oper_param  # Parameter.
        # available_time equals batch end time.
        self.partial_machines_batch[b, m, 1] = self.partial_machines_batch[b, m, 7]
        # Mark as processing.
        self.partial_machines_batch[b, m, 4] = 1.0

        # Actual processing time with batching.
        actual_duration = self.partial_machines_batch[b, m, 7].item() - t

        # Update the first three feature_opers_batch channels.
        self.feature_opers_batch[b, 0, op] = 1.0
        self.feature_opers_batch[b, 1, op] = 1.0
        self.feature_opers_batch[b, 2, op] = actual_duration
        self.feature_opers_batch[b, 5, op] = t

        # Update cal_cumul_adj_batch by clearing predecessor paths.
        if op > 0:
            first_op_of_job = self.first_oper_id_of_each_job_batch[b, j].item()
            if op - 1 >= first_op_of_job:
                self.cal_cumul_adj_batch[b, op - 1, :] = 0

        # Update partial_schedules_batch.
        self.partial_schedules_batch[b, op, :2] = torch.tensor([1.0, float(m)])
        self.partial_schedules_batch[b, op, 2] = t
        end_time = t + actual_duration
        self.partial_schedules_batch[b, op, 3] = end_time

        # Update feature_mas_batch from the current self.time view.
        self.feature_mas_batch[b, 0, :] = torch.count_nonzero(self.oper_ma_adj_batch[b, :, :], dim=0).float()
        self.feature_mas_batch[b, 1, m] = self.partial_machines_batch[b, m, 7]
        utiliz = self.partial_machines_batch[b, :, 2]
        cur_time_tensor = torch.tensor(max(current_time, 1e-9))
        utiliz_clamped = torch.minimum(utiliz, cur_time_tensor)
        self.feature_mas_batch[b, 2, :] = utiliz_clamped / (cur_time_tensor + 1e-9)
        self.feature_mas_batch[b, 4, :] = self.partial_machines_batch[b, :, 5]
        self.feature_mas_batch[b, 5, :] = self.partial_machines_batch[b, :, 8]

        # Update masks.
        self.mask_job_procing_batch[b, j] = True
        self.mask_ma_full_batch[b, m] = self.partial_machines_batch[b, m, 0].item() == 0.0
        self.mask_ma_processing_batch[b, m] = self.partial_machines_batch[b, m, 4].item() == 1.0

        return actual_duration


    # Enumerate complete feasible chain paths forward using layerwise BFS and prefetched data.
    def _compute_chain_feasible_paths(
        self,
        chain_ops,
        m_0,
        start_time,
        *,
        pm_list,
        mask_full_list,
        ma_is_proc,
        ma_batch_end,
        ma_batch_start,
        ma_batch_param,
        ma_batch_dur,
        ma_capacity,
        res_list_per_m,
        pt_rows,
        param_rows,
    ):
        """
        Enumerate all complete feasible machine-assignment paths for the chain after the starting operation selects m_0.
        All pre-extracted arguments are required (passed by find_eligible_om_pair and reused across m_0).

        Args:
            chain_ops: list of oper_ids, all operations in the chain (including starting operation op_0)
            m_0: Machine selected for the starting operation
            start_time: Start time of the starting operation
            pm_list, mask_full_list, ...: pre-extracted native Python data (required)
            ma_capacity: list[int], capacity of each machine for capacity-aware future-reservation conflict checks
                            ========== Capacity-aware arguments ==========

        Returns:
            list[tuple]: Each path is a tuple (m_1, m_2, ..., m_K),
            where K = len(chain_ops) - 1. Returns [] if no feasible path exists.
        """
        num_ma = self.num_machines
        strict_b = self.strict_batch

        # ========== Precompute eligible machines (pt>0 and not full); _fast_feasible_mask scans candidates only ==========
        # Typical FJSP operations have only 3-5 candidate machines, far fewer than M, reducing inner-loop work.
        feasible_candidates = {}  # {op: [m, ...]}
        for op in chain_ops:
            pt_r = pt_rows[op]
            feasible_candidates[op] = [m for m in range(num_ma) if pt_r[m] > 0 and not mask_full_list[m]]

        # --- Inline feasibility mask: scan prefiltered machines, excluding pt<=0 or full machines ---
        def _fast_feasible_mask(op_id, forced_start):
            """Check feasibility using pre-extracted data; iterate only over feasible_candidates[op_id]."""
            pt_row = pt_rows[op_id]
            param_row = param_rows[op_id]
            result = [False] * num_ma  # Preallocate; noncandidate machines stay False.
            for m in feasible_candidates[op_id]:
                pt = pt_row[m]
                forced_end = forced_start + pt
                pm = pm_list[m]
                if pm[4] == 1.0:  # Machine is processing a batch.
                    if forced_start > ma_batch_start[m]:
                        continue
                    if forced_end > ma_batch_end[m]:
                        continue
                    if param_row[m] != ma_batch_param[m]:
                        continue
                    if strict_b and pt != ma_batch_dur[m]:
                        continue
                # ========== Future-reservation conflicts: capacity-aware sweep line ==========
                # Compute maximum concurrency over [forced_start, forced_end)
                # and block if it exceeds machine capacity.
                # Count active-batch occupancy even when there are no future reservations.


                cap = ma_capacity[m]
                events = [(forced_start, 1), (forced_end, -1)]  # Proposed operation.

                # Include existing active-batch operations.
                if pm[4] == 1.0:
                    active_count = int(pm[5])
                    if active_count > 0 and forced_end > ma_batch_start[m] and forced_start < ma_batch_end[m]:
                        events.append((ma_batch_start[m], active_count))
                        events.append((ma_batch_end[m], -active_count))

                # Include overlapping future reservations.
                if m in res_list_per_m:
                    for res_start, res_end, _res_param in res_list_per_m[m]:
                        if forced_end > res_start and forced_start < res_end:
                            events.append((res_start, 1))
                            events.append((res_end, -1))

                # Sort events by time, with -1 before +1 on ties to avoid false overlap.
                events.sort(key=lambda x: (x[0], x[1]))
                cur = 0
                max_overlap = 0
                for _, delta in events:
                    cur += delta
                    if cur > max_overlap:
                        max_overlap = cur
                if max_overlap > cap:
                    continue
                # ========== Batch compatibility with future reservations on idle machines ==========
                # For idle machines, check proposed and reserved operations for compatible param/duration/start_time.
                # The capacity sweep checks concurrency <= cap, not batch compatibility.
                # ========== find_eligible_om_pair already sorts res_list_per_m[m] by start_time ==========
                # Stop at res_start >= forced_end; later reservations cannot overlap.
                if pm[4] != 1.0 and m in res_list_per_m:
                    compatible = True
                    for res_start, res_end, res_param in res_list_per_m[m]:
                        if res_start >= forced_end:
                            break  # Later reservations cannot overlap.
                        if forced_end <= res_start or forced_start >= res_end:
                            continue  # No overlap.
                        # Overlap requires compatible start_time, param, and duration.
                        if abs(res_start - forced_start) > SCHEDULE_COMPAT_EPS:
                            compatible = False
                            break
                        if res_param != param_row[m]:
                            compatible = False
                            break
                        if strict_b and abs((res_end - res_start) - pt) > SCHEDULE_COMPAT_EPS:
                            compatible = False
                            break
                    if not compatible:
                        continue
                result[m] = True
            return result

        # --- End inline feasibility mask ---

        first_op = chain_ops[0]
        first_pt = pt_rows[first_op][m_0]

        # Initial prev_end.
        if ma_is_proc[m_0]:
            prev_end_init = ma_batch_end[m_0]
        else:
            prev_end_init = start_time + first_pt

        paths_so_far = [()]  # list[tuple]
        prev_ends = [prev_end_init]

        for op in chain_ops[1:]:
            pt_row_op = pt_rows[op]  # Prefetch to reduce dictionary lookups.
            cand_m = feasible_candidates[op]  # Prefilter candidate machines.
            new_paths = []
            new_prev_ends = []
            for path_idx, path in enumerate(paths_so_far):
                forced_start = prev_ends[path_idx]
                mask = _fast_feasible_mask(op, forced_start)
                # ========== Scan prefiltered machines only, skipping pt<=0 and full machines ==========
                for m in cand_m:
                    if mask[m]:
                        new_path = path + (m,)
                        new_paths.append(new_path)
                        if ma_is_proc[m]:
                            new_prev_ends.append(ma_batch_end[m])
                        else:
                            new_prev_ends.append(forced_start + pt_row_op[m])

            if not new_paths:
                return []

            paths_so_far = new_paths
            prev_ends = new_prev_ends

        return paths_so_far


    # Chain scheduling.
    def _schedule_chain(self, batch_id, chain_ops, machine_assignments, start_time):
        """
        Schedule the entire no-wait chain atomically.
        Iterate over each operation in the chain, compute its actual start time, and call _schedule_single_operation.
        Update job progress and features in a batch.

        Args:
            batch_id: Scalar batch index
            chain_ops: list of oper_ids
            machine_assignments: dict {oper_id: mas_id}
            start_time: Start time of the chain's starting operation
        """
        b = batch_id
        current_time = start_time

        # Determine job_id; every operation in the chain belongs to the same job.
        first_op = chain_ops[0]
        job_id = self.job_id_for_each_oper_batch[b, first_op].item()

        for idx, op in enumerate(chain_ops):
            m = machine_assignments[op]
            # Read the processing time.
            proc_time = self.proc_times_batch[b, op, m].item()
            # Schedule one operation.
            actual_dur = self._schedule_single_operation(b, op, m, job_id, current_time, proc_time)
            # Start the next operation exactly when this one ends (no wait).
            current_time = self.partial_schedules_batch[b, op, 3].item()

        # Batch update.
        chain_len = len(chain_ops)

        # Subtract the chain length from feature_opers_batch channel 3 (unscheduled count) at once.
        start_oper_id = self.first_oper_id_of_each_job_batch[b, job_id].item()
        end_oper_id = self.end_oper_id_of_each_job_batch[b, job_id].item()
        for i in range(start_oper_id, end_oper_id + 1):
            self.feature_opers_batch[b, 3, i] -= chain_len

        # _propagate_start_times_batch handles channel 4 (job completion) and start/end propagation.
        # Exact chain times are already stored in feature_opers_batch[:,5] and partial_schedules_batch.

        # Advance step_oper_id_of_each_job_batch past the entire chain.
        self.step_oper_id_of_each_job_batch[b, job_id] += chain_len

        # Keep self.time at the original chain-start time.


    # Vectorized feature propagation across batches.
    def _propagate_start_times_batch(self, batch_indices):
        """
        Vectorize propagation of all operation start times, end times, and job completion times across batches.

        Propagation through cal_cumul_adj_batch must use exact end times of scheduled operations as its basis.



        """

        if isinstance(batch_indices, set):
            if not batch_indices:
                return
            idx_tensor = torch.tensor(sorted(batch_indices), dtype=torch.long)
        elif isinstance(batch_indices, torch.Tensor):
            if batch_indices.numel() == 0:
                return
            idx_tensor = batch_indices.long()
        else:
            if not batch_indices:
                return
            idx_tensor = torch.tensor(list(batch_indices), dtype=torch.long)

        # Propagate start times: [len(idx_list), num_opers].
        is_scheduled = self.feature_opers_batch[idx_tensor, 0, :]
        mean_proc_time = self.feature_opers_batch[idx_tensor, 2, :]
        start_times = self.feature_opers_batch[idx_tensor, 5, :] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_start_times = (
            torch.bmm(
                (start_times + mean_proc_time).unsqueeze(1),
                self.cal_cumul_adj_batch[idx_tensor, :, :],
            ).squeeze(1)
            * un_scheduled
        )
        self.feature_opers_batch[idx_tensor, 5, :] = start_times + estimate_start_times

        # Update job completion times with vectorized operations.
        end_time_of_each_oper = (
            self.feature_opers_batch[idx_tensor, 5, :] + self.feature_opers_batch[idx_tensor, 2, :]
        )
        end_oper_ids = self.end_oper_id_of_each_job_batch[idx_tensor, :]
        end_time_jobs = end_time_of_each_oper.gather(1, end_oper_ids)
        self.feature_opers_batch[idx_tensor, 4, :] = convert_feature_job_to_oper(
            end_time_jobs,
            self.job_id_for_each_oper_batch[idx_tensor, :],
        )

        # Update partial_schedules_batch with vectorized operations.
        self.partial_schedules_batch[idx_tensor, :, 2] = self.feature_opers_batch[idx_tensor, 5, :]
        self.partial_schedules_batch[idx_tensor, :, 3] = (
            self.feature_opers_batch[idx_tensor, 5, :] + self.feature_opers_batch[idx_tensor, 2, :]
        )

        # Update feature_mas_batch with vectorized operations.
        self.feature_mas_batch[idx_tensor, 0, :] = torch.count_nonzero(
            self.oper_ma_adj_batch[idx_tensor, :, :], dim=1
        ).float()
        utiliz = self.partial_machines_batch[idx_tensor, :, 2]
        cur_time = torch.maximum(self.time[idx_tensor, None], torch.tensor(1e-9))
        utiliz_clamped = torch.minimum(utiliz, cur_time)
        self.feature_mas_batch[idx_tensor, 2, :] = utiliz_clamped / (cur_time + 1e-9)
        self.feature_mas_batch[idx_tensor, 4, :] = self.partial_machines_batch[idx_tensor, :, 5]
        self.feature_mas_batch[idx_tensor, 5, :] = self.partial_machines_batch[idx_tensor, :, 8]


    # Apply one action per case to transition the environment state.
    # step() accepts (basic_actions, chain_assignments_dict).
    # basic_actions has shape [4, len(batch_idxes)], with one column per unfinished case.
    # Each action column is [opers_ID, mas_ID, jobs_ID, is_drop].
    # chain_assignments_dict: optional agent-selected {b: {oper_id: mas_id}} chain-machine assignments.
    def step(self, actions):
        """
        Environment transition function
        """
        # Unpack the pair.
        if isinstance(actions, tuple):
            basic_actions, chain_assignments_dict = actions
        else:
            basic_actions = actions
            chain_assignments_dict = {}

        # Only distinguish future-rightnow schedule writes here; eligibility guarantees action feasibility,
        # and validate_gantt() validates final results.
        pending_before = self._get_pending_rightnow_mask()

        # Save the previous step's environment time.
        self.last_time = self.time


        opers = basic_actions[0, :]
        mas = basic_actions[1, :]
        jobs = basic_actions[2, :]
        is_drop = basic_actions[3, :]

        not_drop = is_drop == 0

        # Record scheduled-operation counts.
        self.num_scheduled_oper_batch[self.batch_idxes] += not_drop

        # Update the environment for selected nondrop operation-machine pairs.
        # Use vectorized updates for ordinary operations and individual handling for chain starts.
        t_step_chain = 0.0
        t_step_nonchain = 0.0
        t_step_propagate = 0.0
        if len(self.batch_idxes[not_drop]) > 0:
            # Separate future-rightnow reservations, chain starts, and ordinary operations.
            # is_chain_start and not_drop both have shape [len(batch_idxes)].
            is_chain_start = torch.zeros(len(self.batch_idxes), dtype=torch.bool)
            is_future_rightnow = torch.zeros(len(self.batch_idxes), dtype=torch.bool)
            not_drop_positions = torch.where(not_drop)[0]
            for pos in not_drop_positions:
                b = self.batch_idxes[pos].item()
                op = opers[pos].item()
                job = jobs[pos].item()
                if b in self.chain_first_ops and op in self.chain_first_ops[b]:
                    is_chain_start[pos] = True
                # rightnow may reserve future work now, but its scheduled start
                # must exactly equal the shifted job_avail_time.
                if (
                    bool(pending_before[b, job].item())
                    and self.job_avail_time_batch_init[b, job].item() > self.time[b].item()
                ):
                    is_future_rightnow[pos] = True

            not_drop_future_rightnow = not_drop & is_future_rightnow
            not_drop_chain = not_drop & is_chain_start & ~is_future_rightnow
            not_drop_nonchain = not_drop & ~is_chain_start & ~is_future_rightnow

            # Write future-rightnow reservations first; single-operation jobs reuse singleton chains,
            # sharing future_reservations resource protection with no-wait chains.
            if len(self.batch_idxes[not_drop_future_rightnow]) > 0:
                _t0 = time.time()
                future_positions = torch.where(not_drop_future_rightnow)[0]
                for pos in future_positions:
                    b = int(self.batch_idxes[pos].item())
                    op = int(opers[pos].item())
                    job = int(jobs[pos].item())
                    machine = int(mas[pos].item())
                    target_start = self.job_avail_time_batch_init[b, job].item()
                    chain_ops = self.chain_info.get(b, {}).get(op, [op])
                    assignments = (
                        chain_assignments_dict[b]
                        if len(chain_ops) > 1
                        else {op: machine}
                    )
                    self._schedule_chain(b, chain_ops, assignments, target_start)
                    self.num_scheduled_oper_batch[b] += len(chain_ops) - 1
                t_step_chain += time.time() - _t0

            # ---- 1. Handle chain starts individually; there are few of them ----
            if len(self.batch_idxes[not_drop_chain]) > 0:
                _t0 = time.time()
                for idx in range(len(self.batch_idxes[not_drop_chain])):
                    b = self.batch_idxes[not_drop_chain][idx].item()
                    op = opers[not_drop_chain][idx].item()
                    chain_ops = self.chain_info[b][op]
                    # Use the agent's chain-machine assignments directly.
                    # _compute_chain_feasible_paths in find_eligible_om_pair guarantees feasible complete paths;
                    # the agent's select_chain_machines() filters them at each choice, ensuring every choice
                    # leads to at least one complete feasible path without dead ends.
                    # chain_assignments_dict[b] maps all chain operations, including the first, to machines.
                    assignments = chain_assignments_dict[b]
                    self._schedule_chain(b, chain_ops, assignments, self.time[b].item())
                    self.num_scheduled_oper_batch[b] += len(chain_ops) - 1
                t_step_chain += time.time() - _t0

            # ---- 2. Handle ordinary operations using vectorized updates ----
            if len(self.batch_idxes[not_drop_nonchain]) > 0:
                _t0 = time.time()
                idx_nc = self.batch_idxes[not_drop_nonchain]
                op_nc = opers[not_drop_nonchain]
                ma_nc = mas[not_drop_nonchain]
                job_nc = jobs[not_drop_nonchain]

                self.jobs_status_in_mas_batch[idx_nc, ma_nc, job_nc] = True

                remain_oper_ma_adj = torch.zeros(size=(self.batch_size, self.num_machines), dtype=torch.int64)
                remain_oper_ma_adj[idx_nc, ma_nc] = 1
                self.oper_ma_adj_batch[idx_nc, op_nc] = remain_oper_ma_adj[idx_nc, :]
                self.proc_times_batch *= self.oper_ma_adj_batch
                self.opers_param_batch *= self.oper_ma_adj_batch

                proc_times = self.proc_times_batch[idx_nc, op_nc, ma_nc]
                opers_param = self.opers_param_batch[idx_nc, op_nc, ma_nc]

                # Update partial_machines_batch
                self.partial_machines_batch[idx_nc, ma_nc, 2] += proc_times
                self.partial_machines_batch[idx_nc, ma_nc, 3] = job_nc.float()
                self.partial_machines_batch[idx_nc, ma_nc, 5] += 1
                self.partial_machines_batch[idx_nc, ma_nc, 0] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 5]
                    < self.machines_capacity_batch[idx_nc, ma_nc],
                    1.0,
                    0.0,
                )
                self.partial_machines_batch[idx_nc, ma_nc, 6] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 4] == 0,
                    self.time[idx_nc],
                    self.partial_machines_batch[idx_nc, ma_nc, 6],
                )
                self.partial_machines_batch[idx_nc, ma_nc, 7] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 4] == 0,
                    self.time[idx_nc] + proc_times,
                    self.partial_machines_batch[idx_nc, ma_nc, 7],
                )
                self.partial_machines_batch[idx_nc, ma_nc, 8] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 4] == 0,
                    opers_param,
                    self.partial_machines_batch[idx_nc, ma_nc, 8],
                )
                self.partial_machines_batch[idx_nc, ma_nc, 1] = self.partial_machines_batch[idx_nc, ma_nc, 7]
                self.partial_machines_batch[idx_nc, ma_nc, 4] = torch.ones(idx_nc.size(0))

                proc_times = self.partial_machines_batch[idx_nc, ma_nc, 7] - self.time[idx_nc]

                # Update the first three feature_opers_batch channels.
                self.feature_opers_batch[idx_nc, :3, op_nc] = torch.stack(
                    (
                        torch.ones(idx_nc.size(0), dtype=torch.float),
                        torch.ones(idx_nc.size(0), dtype=torch.float),
                        proc_times,
                    ),
                    dim=1,
                )

                # Update cal_cumul_adj_batch
                last_opes = torch.where(
                    op_nc - 1 < self.first_oper_id_of_each_job_batch[idx_nc, job_nc],
                    self.max_num_opers_in_batch - 1,
                    op_nc - 1,
                )
                self.cal_cumul_adj_batch[idx_nc, last_opes, :] = 0

                # Update channel 3: unscheduled-operation count.
                start_oper_id_nc = self.first_oper_id_of_each_job_batch[idx_nc, job_nc]
                end_oper_id_nc = self.end_oper_id_of_each_job_batch[idx_nc, job_nc]
                for i in range(len(idx_nc)):
                    b_i = idx_nc[i].item()
                    self.feature_opers_batch[b_i, 3, start_oper_id_nc[i] : end_oper_id_nc[i] + 1] -= 1

                # Update start times.
                self.feature_opers_batch[idx_nc, 5, op_nc] = self.time[idx_nc]

                # Update job progress.
                self.step_oper_id_of_each_job_batch[idx_nc, job_nc] += 1

                # Update partial_schedules_batch
                self.partial_schedules_batch[idx_nc, op_nc, :2] = torch.stack(
                    (torch.ones(idx_nc.size(0)), ma_nc),
                    dim=1,
                )

                # Update mask
                self.mask_job_procing_batch[idx_nc, job_nc] = True
                self.mask_ma_full_batch[idx_nc, ma_nc] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 0] == 0.0,
                    True,
                    False,
                )
                self.mask_ma_processing_batch[idx_nc, ma_nc] = torch.where(
                    self.partial_machines_batch[idx_nc, ma_nc, 4] == 1.0,
                    True,
                    False,
                )
                # step timing
                t_step_nonchain = time.time() - _t0

            # ---- 3. Shared vectorized feature propagation ----

            _t0 = time.time()
            self._propagate_start_times_batch(torch.unique(self.batch_idxes[not_drop]))
            t_step_propagate = time.time() - _t0

        # step timing — drop / reward / eligibility
        _t_drop_etc = time.time()

        # The job is complete when its progress reaches its final operation ID.
        self.mask_job_finish_batch = torch.where(
            self.step_oper_id_of_each_job_batch == self.end_oper_id_of_each_job_batch + 1,
            True,
            self.mask_job_finish_batch,
        )

        # Mark a case complete when all its jobs have finished.
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        # Whether all cases have finished.
        self.done = self.done_batch.all()

        # Store mean drop counts so each drop changes the system state.
        drop_ins_batch = torch.zeros(self.batch_size, dtype=torch.bool)
        drop_ins_batch[self.batch_idxes[is_drop == 1]] = 1
        flag_need_drop = (drop_ins_batch == 1) & (~self.done_batch)

        mean_drop = flag_need_drop.float()[:, None].expand_as(self.feature_opers_batch[:, 6, :])

        processing_step_oper_id_of_each_job_batch = torch.where(
            self.mask_job_finish_batch == True, -1, self.step_oper_id_of_each_job_batch
        )
        batch_indices, job_indices = torch.where(self.mask_job_finish_batch == False)
        vaild_oper_indices = processing_step_oper_id_of_each_job_batch[batch_indices, job_indices]

        self.feature_opers_batch[batch_indices, 6, vaild_oper_indices] += mean_drop[
            batch_indices, vaild_oper_indices
        ]



        self.drop_ins_transfer_to_next_time(is_drop)


        # Update step rewards and makespan; apply a small drop penalty.
        true_makespan_batch = self.calculate_true_makespan()
        self.reward_batch = self.makespan_batch - true_makespan_batch
        self.reward_batch -= self.reward_punish * flag_need_drop.float()
        self.makespan_batch = true_makespan_batch

        # step timing: drop_etc ends here; subsequent time belongs to elig.
        t_step_drop_etc = time.time() - _t_drop_etc

        # step timing — find_om / transfer
        t_step_find_om = 0.0
        t_step_transfer = 0.0

        # Check for other feasible operation-machine pairs in the current time step; if none exist, advance the entire environment to the next feasible time step
        # Continue while an unfinished environment has no eligible O-M pair.
        # Stop when every environment has eligible O-M pairs or has finished.
        _t0 = time.time()
        num_eligible_om_pair_batch = self.find_eligible_om_pair()
        t_step_find_om += time.time() - _t0
        while ~((~((num_eligible_om_pair_batch == 0) & (~self.done_batch))).all()):
            _t0 = time.time()
            self.transfer_to_next_time(num_eligible_om_pair_batch)
            t_step_transfer += time.time() - _t0
            _t0 = time.time()
            num_eligible_om_pair_batch = self.find_eligible_om_pair()
            t_step_find_om += time.time() - _t0




        # Keep indices of unfinished cases only.
        mask_not_finish = (self.num_scheduled_oper_batch + 1) <= self.num_opers_batch
        if ~(mask_not_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_not_finish]

        if self.no_drop:
            pass  # Keep drop disabled.
        else:
            # Update state and disable in-place drop.
            self.inplace_drop = flag_need_drop & (self.last_time == self.time)


        pending_after_step = self._get_pending_rightnow_mask().any(dim=1)
        self.inplace_drop = self.inplace_drop | pending_after_step








        # Update the environment state.
        self.state.update(
            self.batch_idxes,
            self.feature_opers_batch,
            self.feature_mas_batch,
            self.proc_times_batch,
            self.opers_param_batch,
            self.oper_seamless_batch,
            self.oper_ma_adj_batch,
            self.mask_job_procing_batch,
            self.mask_job_finish_batch,
            self.mask_ma_full_batch,
            self.step_oper_id_of_each_job_batch,
            self.time,
            self.inplace_drop,
            self.partial_machines_batch,
            # Pass chain state and future reservations
            chain_first_ops_batch=self.chain_first_ops,
            chain_info_batch=self.chain_info,
            future_reservations_batch=self.future_reservations,
            # Pass feasible chain paths.
            chain_feasible_paths_batch=self._chain_feasible_paths,
            # ========== Unified eligibility ==========
            eligible_om_pairs_batch=self._cached_eligible_om_pairs,
        )

        return self.state, self.reward_batch, self.done_batch

    def calculate_true_makespan(self):
        # Extract start and end times.
        intervals_batch = self.partial_schedules_batch[:, :, 2:4]  # [batch_size, num_opers, 2]

        # Sort by start time.
        sorted_indices = torch.argsort(intervals_batch[:, :, 0], dim=1)  # [batch_size, num_opers]
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, 2)
        sorted_intervals = torch.gather(intervals_batch, 1, sorted_indices_expanded)

        # Compute cumulative maximum end times.
        sorted_ends = sorted_intervals[:, :, 1]
        cum_max_ends = torch.cummax(sorted_ends, dim=1).values

        # Previous cumulative maximum (the first position is 0)
        prev_cum_max_ends = torch.zeros_like(cum_max_ends)
        prev_cum_max_ends[:, 1:] = cum_max_ends[:, :-1]

        # Valid start times and contribution times
        effective_starts = torch.maximum(sorted_intervals[:, :, 0], prev_cum_max_ends)
        contributions = torch.clamp(sorted_ends - effective_starts, min=0)

        # Total time
        total_time = contributions.sum(dim=1)

        return total_time

    # rightnow pending is defined as a state where the first remaining operation in the current subproblem has not yet been scheduled;
    # an operation with a future job_avail_time may be recorded as a reservation first and is no longer pending once scheduled.
    def _get_pending_rightnow_mask(self):
        progress_at_first = (
            self.step_oper_id_of_each_job_batch == self.first_oper_id_of_each_job_batch
        )
        pending = (
            self.job_rightnow_batch_init
            & progress_at_first
            & ~self.mask_job_procing_batch
            & ~self.mask_job_finish_batch
        )
        return pending

    # Normalize the active batch and future reservations into machine-batch snapshots for joint search.
    def _build_rightnow_machine_batches(self, batch_id):
        batches = {m: [] for m in range(self.num_machines)}

        def add_batch(machine_id, start, end, param, count=1):
            for item in batches[machine_id]:
                if (
                    abs(item["start"] - start) <= SCHEDULE_COMPAT_EPS
                    and abs(item["end"] - end) <= SCHEDULE_COMPAT_EPS
                    and item["param"] == param
                ):
                    item["count"] += count
                    return
            batches[machine_id].append(
                {"start": float(start), "end": float(end), "param": float(param), "count": int(count)}
            )

        for machine_id in range(self.num_machines):
            machine = self.partial_machines_batch[batch_id, machine_id]
            if machine[4].item() == 1.0 and machine[5].item() > 0:
                add_batch(
                    machine_id,
                    machine[6].item(),
                    machine[7].item(),
                    machine[8].item(),
                    int(machine[5].item()),
                )
        for machine_id, reservations in self.future_reservations.get(batch_id, {}).items():
            for start, end, _oper_id, _job_id, param in reservations:
                add_batch(machine_id, start, end, param)
        return batches

    def _try_add_rightnow_operation(self, batch_id, batches, machine_id, start, proc_time, param):
        """Try placing an operation in the batch snapshot; return (new snapshot, actual end time) or None."""
        intrinsic_end = start + proc_time
        machine_batches = batches[machine_id]
        overlapping = [
            item
            for item in machine_batches
            if item["end"] > start and item["start"] < intrinsic_end
        ]
        capacity = int(self.machines_capacity_batch[batch_id, machine_id].item())

        if overlapping:
            if len(overlapping) != 1:
                return None
            active = overlapping[0]
            batch_duration = active["end"] - active["start"]
            if abs(active["start"] - start) > SCHEDULE_COMPAT_EPS:
                return None
            if active["param"] != param:
                return None
            if self.strict_batch:
                if abs(batch_duration - proc_time) > SCHEDULE_COMPAT_EPS:
                    return None
            elif intrinsic_end > active["end"]:
                return None
            if active["count"] >= capacity:
                return None

            next_batches = {m: [dict(item) for item in values] for m, values in batches.items()}
            for item in next_batches[machine_id]:
                if (
                    abs(item["start"] - active["start"]) <= SCHEDULE_COMPAT_EPS
                    and abs(item["end"] - active["end"]) <= SCHEDULE_COMPAT_EPS
                    and item["param"] == active["param"]
                ):
                    item["count"] += 1
                    break
            return next_batches, active["end"]

        next_batches = {m: [dict(item) for item in values] for m, values in batches.items()}
        next_batches[machine_id].append(
            {"start": float(start), "end": float(intrinsic_end), "param": float(param), "count": 1}
        )
        return next_batches, intrinsic_end

    def _enumerate_rightnow_job_options(
        self,
        batch_id,
        job_id,
        batches,
        *,
        first_machines=None,
    ):
        """Enumerate complete first-action/chain paths for a pending job under the given joint resource state."""
        first_op = int(self.step_oper_id_of_each_job_batch[batch_id, job_id].item())
        chain_ops = self.chain_info.get(batch_id, {}).get(first_op, [first_op])
        proc_rows = self.proc_times_batch[batch_id]
        param_rows = self.opers_param_batch[batch_id]
        options = []

        def visit(op_index, forced_start, current_batches, assignments):
            op_id = chain_ops[op_index]
            if op_index == 0 and first_machines is not None:
                candidates = first_machines
            else:
                candidates = torch.nonzero(proc_rows[op_id] > 0, as_tuple=False).flatten().tolist()

            for machine_id in candidates:
                proc_time = proc_rows[op_id, machine_id].item()
                if proc_time <= 0:
                    continue
                param = param_rows[op_id, machine_id].item()
                result = self._try_add_rightnow_operation(
                    batch_id,
                    current_batches,
                    machine_id,
                    forced_start,
                    proc_time,
                    param,
                )
                if result is None:
                    continue
                next_batches, actual_end = result
                next_assignments = assignments + [machine_id]
                if op_index + 1 == len(chain_ops):
                    path = tuple(next_assignments[1:])
                    options.append(
                        {
                            "machine": next_assignments[0],
                            "path": path,
                            "batches": next_batches,
                        }
                    )
                else:
                    visit(op_index + 1, actual_end, next_batches, next_assignments)

        # The rightnow equality target is each job's own job_avail_time,
        # so pending jobs with different availability times may be scheduled at different future times.
        target_start = self.job_avail_time_batch_init[batch_id, job_id].item()
        visit(0, target_start, batches, [])
        return options

    def _rightnow_remaining_feasible(self, batch_id, remaining_jobs, batches):
        """Run a minimum-candidate-first DFS to determine whether the remaining pending jobs have a complete joint assignment."""
        if not remaining_jobs:
            return True

        best_job = None
        best_options = None
        for job_id in remaining_jobs:
            options = self._enumerate_rightnow_job_options(batch_id, job_id, batches)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_job = job_id
                best_options = options

        next_jobs = [job_id for job_id in remaining_jobs if job_id != best_job]
        for option in best_options:
            if self._rightnow_remaining_feasible(batch_id, next_jobs, option["batches"]):
                return True
        return False

    def _apply_rightnow_joint_guard(self, batch_id, pending_jobs):
        """Keep first actions and chain paths that can extend to future reservations for all pending rightnow jobs."""
        base_batches = self._build_rightnow_machine_batches(batch_id)
        safe_machines = {job_id: set() for job_id in pending_jobs}
        safe_paths = {job_id: {} for job_id in pending_jobs}

        for first_job in pending_jobs:
            first_op = int(self.step_oper_id_of_each_job_batch[batch_id, first_job].item())
            # Future rightnow jobs have not been released and therefore do not appear in ordinary eligibility;
            # get candidate machines from the original O-M adjacency here, then fully validate them against the joint resource snapshot.
            first_machines = torch.nonzero(
                self.proc_times_batch[batch_id, first_op] > 0, as_tuple=False
            ).flatten().tolist()
            first_options = self._enumerate_rightnow_job_options(
                batch_id,
                first_job,
                base_batches,
                first_machines=first_machines,
            )
            remaining = [job_id for job_id in pending_jobs if job_id != first_job]
            for option in first_options:
                if self._rightnow_remaining_feasible(batch_id, remaining, option["batches"]):
                    machine_id = option["machine"]
                    safe_machines[first_job].add(machine_id)
                    safe_paths[first_job].setdefault(machine_id, set()).add(option["path"])

        # When pending jobs exist, mask all ordinary jobs; retain only rightnow first actions that can extend to a joint assignment.
        # Once selected by the policy, an action is immediately recorded as a future reservation at target_start. After pending clears,
        # ordinary operations may use idle gaps before the reservation but must not conflict with it.
        self._cached_eligible_om_pairs[batch_id] = False
        for job_id in pending_jobs:
            for machine_id in safe_machines[job_id]:
                self._cached_eligible_om_pairs[batch_id, job_id, machine_id] = True
            first_op = int(self.step_oper_id_of_each_job_batch[batch_id, job_id].item())
            if first_op in self.chain_first_ops.get(batch_id, set()):
                self._chain_feasible_paths.setdefault(batch_id, {})[first_op] = {
                    machine_id: sorted(paths)
                    for machine_id, paths in safe_paths[job_id].items()
                    if paths
                }

    # Check whether any other O-M pairs can be processed in the current time step
    # Eligibility rule: tasks in one batch must start together; shorter processing times wait for the longest, then all finish together.
    def find_eligible_om_pair(self):
        """
        Check if there are still O-M pairs to be processed
        """
        # Get each job's progress (the ID of its first unscheduled operation)
        # [batch_size, num_jobs]
        step_oper_id_of_each_job_batch = torch.minimum(
            self.step_oper_id_of_each_job_batch, self.end_oper_id_of_each_job_batch
        )

        # Gather the processing time on each machine for the first unscheduled operation of each job
        # [batch_size, num_jobs, num_mas]
        proc_time_of_step_oper = self.proc_times_batch.gather(
            1,
            step_oper_id_of_each_job_batch.unsqueeze(-1).expand(-1, -1, self.proc_times_batch.size(2)),
        )

        # Get the processing parameter for the first unscheduled operation of each job
        # [batch_size, num_jobs, num_mas]
        opers_param_of_step_oper = self.opers_param_batch.gather(
            1,
            step_oper_id_of_each_job_batch.unsqueeze(-1).expand(-1, -1, self.opers_param_batch.size(2)),
        )

        # Determine eligible idle machines from each job's perspective
        # [batch_size, num_jobs, num_mas], with the same num_jobs in every row
        ma_eligible = ~self.mask_ma_full_batch.unsqueeze(1).expand_as(proc_time_of_step_oper)

        # Determine which jobs have not been processed (neither in progress nor complete)
        # [batch_size, num_jobs, num_mas], with the same num_machines in every column
        # A job can enter the eligible O-M pairs only after its initial availability time
        job_released = self.job_avail_time_batch_init <= self.time[:, None]
        job_eligible = (
            (~(self.mask_job_procing_batch + self.mask_job_finish_batch) & job_released)
            .unsqueeze(-1)
            .expand_as(proc_time_of_step_oper)
        )

        # ========== Protect future chain reservations — do not schedule later operations of jobs with unfinished reservations ==========
        # Later chain operations remain future reservations; their job's next operation must not be scheduled until they are activated or completed,
        # otherwise operations within one job can be put out of chronological order (e.g., op_10 end=123, op_11 start=68).
        if self.future_reservations:
            job_has_future = torch.zeros(self.batch_size, self.num_jobs, dtype=torch.bool)
            for b in range(self.batch_size):
                if b not in self.future_reservations:
                    continue
                for reservations in self.future_reservations[b].values():
                    for _, _, _, job_id, _ in reservations:
                        if job_id < self.num_jobs:
                            job_has_future[b, job_id] = True
            job_eligible = job_eligible & ~job_has_future.unsqueeze(-1).expand_as(job_eligible)

        # Processing times of the O-M pairs available to the current task
        possible_o_m_pair_proc_time = torch.where(
            ma_eligible & job_eligible, proc_time_of_step_oper.double(), 0.0
        )

        # Which machines are currently processing
        machine_proc_status = (
            (self.partial_machines_batch[:, :, 4] == 1).unsqueeze(1).expand_as(proc_time_of_step_oper)
        )

        # Current time + processing time must not exceed the batch end time on a busy machine
        proc_time_of_step_oper_new = proc_time_of_step_oper + self.time.unsqueeze(-1).unsqueeze(-1).expand_as(
            proc_time_of_step_oper
        )
        should_set_zero_because_duration_not_equal = machine_proc_status & (
            proc_time_of_step_oper_new
            > self.partial_machines_batch[:, :, 7].unsqueeze(1).expand_as(proc_time_of_step_oper)
        )

        # Current time must not exceed the batch start time on a busy machine
        should_set_zero_because_start_time_not_equal = machine_proc_status & (
            self.time.unsqueeze(-1).unsqueeze(-1).expand_as(proc_time_of_step_oper)
            > self.partial_machines_batch[:, :, 6].unsqueeze(1).expand_as(proc_time_of_step_oper)
        )

        # The current parameter must match the parameter of the active batch on a busy machine
        should_set_zero_because_oper_param_not_equal = machine_proc_status & (
            opers_param_of_step_oper
            != self.partial_machines_batch[:, :, 8].unsqueeze(1).expand_as(opers_param_of_step_oper)
        )

        # ========== strict_batch — require operation processing time to exactly match the batch processing time ==========
        # The duration upper-bound check rejects pt > batch_pt and permits shorter operations to wait for the batch.
        # Strict mode additionally requires pt == batch_pt, so an operation can be neither longer nor shorter
        if self.strict_batch:
            batch_proc_time = (
                self.partial_machines_batch[:, :, 7] - self.partial_machines_batch[:, :, 6]
            )  # [batch_size, num_mas]
            should_set_zero_because_duration_not_strict = machine_proc_status & (
                proc_time_of_step_oper != batch_proc_time.unsqueeze(1).expand_as(proc_time_of_step_oper)
            )
        else:
            should_set_zero_because_duration_not_strict = torch.zeros_like(
                should_set_zero_because_duration_not_equal, dtype=torch.bool
            )

        should_set_zero = (
            should_set_zero_because_duration_not_equal
            | should_set_zero_because_start_time_not_equal
            | should_set_zero_because_oper_param_not_equal
            | should_set_zero_because_duration_not_strict  # ========== strict_batch ==========
        )

        # ========== Future-reservation conflict check (capacity-aware vectorization) ==========
        # Use a sweep line to find the first time future reservations fill machine capacity.
        # This check blocks operations that would extend beyond that time.

        if self.future_reservations:
            future_block_threshold = torch.full((self.batch_size, self.num_machines), float("inf"))
            for b in range(self.batch_size):
                if b not in self.future_reservations or not self.future_reservations[b]:
                    continue
                t_b = self.time[b].item()
                for m_id, reservations in self.future_reservations[b].items():
                    if not reservations:
                        continue
                    cap = self.machines_capacity_batch[b, m_id].item()
                    # Build sweep-line events: (time, delta), where delta=+1 on entry and -1 on exit
                    events = []
                    for res_start, res_end, _, _, _ in reservations:
                        if res_end > t_b:  # Unexpired reservations
                            events.append((max(res_start, t_b), 1))
                            events.append((res_end, -1))
                    if not events:
                        continue
                    # At the same time, process -1 before +1 (finish before start to avoid false overlap)
                    events.sort(key=lambda x: (x[0], x[1]))
                    cur = 0
                    for time_pt, delta in events:
                        cur += delta
                        if cur >= cap:
                            future_block_threshold[b, m_id] = time_pt - t_b
                            break
            # Vectorized comparison: [batch_size, num_jobs, num_mas] vs [batch_size, 1, num_mas]
            should_set_zero |= (proc_time_of_step_oper > 0) & (
                proc_time_of_step_oper > future_block_threshold.unsqueeze(1)
            )


        # For idle machines with future reservations, check whether an operation can join the reserved batch.
        # Key insight: an operation always starts at t_b. A reservation is either at the same time (res_start==t_b, requiring matching param/duration),
        # or in the future (res_start>t_b, when any overlap is incompatible because the start and end times differ).

        if self.future_reservations:
            for b in range(self.batch_size):
                if b not in self.future_reservations or not self.future_reservations[b]:
                    continue
                t_b = self.time[b].item()


                # machine_info[m_id] = (next_start, same_time_values)
                # next_start: float, earliest reservation start with res_start > t_b (inf means none)
                # same_time_values: list[(pt, param)] or list[param], depending on strict_batch
                # None when the machine is processing or has no reservation
                machine_info = {}
                for m_id, reservations in self.future_reservations[b].items():
                    if not reservations:
                        continue
                    if self.partial_machines_batch[b, m_id, 4].item() == 1.0:
                        continue  # Processing machines are covered by the preceding batch-compatibility check
                    next_start = float("inf")
                    same_time_values = []
                    for res_start, res_end, _, _, res_param in reservations:
                        if res_end <= t_b:
                            continue  # Expired reservation
                        if abs(res_start - t_b) < SCHEDULE_COMPAT_EPS:
                            if self.strict_batch:
                                same_time_values.append((res_end - res_start, res_param))
                            else:
                                same_time_values.append(res_param)
                        elif res_start > t_b and res_start < next_start:
                            next_start = res_start
                    machine_info[m_id] = (next_start, same_time_values)
                # --- End precomputation ---

                if not machine_info:
                    continue

                for j in range(self.num_jobs):
                    for m_id, (next_start, same_time_values) in machine_info.items():
                        if should_set_zero[b, j, m_id]:
                            continue
                        pt = proc_time_of_step_oper[b, j, m_id].item()
                        if pt <= 0:
                            continue

                        # Check 1: overlap with a future reservation (different start/end times are never compatible)
                        # next_start is the earliest reservation beginning after t_b.
                        # The operation must finish by next_start to avoid overlapping that reservation.
                        if next_start != float("inf") and t_b + pt > next_start:
                            should_set_zero[b, j, m_id] = True
                            continue

                        # Check 2: compatibility with reservations at the same time
                        # Same-time reservations must be mutually compatible, and the candidate must match them.
                        # Strict batching checks both processing time and parameter; non-strict batching checks the parameter.
                        if same_time_values:
                            oper_param = opers_param_of_step_oper[b, j, m_id].item()
                            if self.strict_batch:
                                ref_pt, ref_param = same_time_values[0]
                                reservations_compatible = all(
                                    abs(res_pt - ref_pt) <= SCHEDULE_COMPAT_EPS
                                    and res_param == ref_param
                                    for res_pt, res_param in same_time_values[1:]
                                )
                                if (
                                    not reservations_compatible
                                    or abs(pt - ref_pt) > SCHEDULE_COMPAT_EPS
                                    or oper_param != ref_param
                                ):
                                    should_set_zero[b, j, m_id] = True
                            else:
                                ref_param = same_time_values[0]
                                reservations_compatible = all(
                                    res_param == ref_param
                                    for res_param in same_time_values[1:]
                                )
                                if not reservations_compatible or oper_param != ref_param:
                                    should_set_zero[b, j, m_id] = True


        # Enumerate complete feasible paths for chain-start operations.
        # For candidate O-M pairs that pass the first five checks, enumerate all complete feasible paths when the operation starts a chain.
        # Keep an O-M pair eligible only if at least one complete feasible path exists.
        # Store the enumeration in self._chain_feasible_paths for select_chain_machines() in act() to read.
        #
        # Extract batch data once per instance and operation data once per chain.
        # Reuse these Python lists across candidate first-machine choices.


        self._chain_feasible_paths = {}  # Recompute on each eligibility pass
        num_ma = self.num_machines
        for b in range(self.batch_size):
            if b not in self.chain_first_ops:
                continue
            t_b = self.time[b].item()
            b_chain_first = self.chain_first_ops[b]
            b_chain_info = self.chain_info[b]
            self._chain_feasible_paths[b] = {}


            pm_list = self.partial_machines_batch[b].tolist()
            mask_full_list = self.mask_ma_full_batch[b].tolist()
            ma_is_proc = [pm[4] == 1.0 for pm in pm_list]
            ma_batch_end = [pm[7] for pm in pm_list]
            ma_batch_start = [pm[6] for pm in pm_list]
            ma_batch_param = [pm[8] for pm in pm_list]
            ma_batch_dur = [pm[7] - pm[6] for pm in pm_list]
            # ========== Capacity-aware: pre-extract machine capacity list ==========
            ma_capacity = self.machines_capacity_batch[b].tolist()
            res_list_per_m = {}
            for m_id, reservations in self.future_reservations.get(b, {}).items():
                # ========== Include param for batch-compatibility checks ==========
                # Sort reservations by start time so early-break checks see chronological order.
                # Natural append order is not necessarily sorted by start_time. A later chain may create a reservation at an earlier time,
                # so an unsorted list can make break skip an overlapping reservation that should be checked, incorrectly marking a conflicting operation FEASIBLE.
                res_list_per_m[m_id] = sorted([(r[0], r[1], r[4]) for r in reservations], key=lambda x: x[0])
            proc_b = self.proc_times_batch[b]
            param_b = self.opers_param_batch[b]
            # --- End batch-level pre-extraction ---

            for j in range(self.num_jobs):
                oper_id = step_oper_id_of_each_job_batch[b, j].item()
                if oper_id not in b_chain_first:
                    continue
                chain_ops = b_chain_info[oper_id]
                self._chain_feasible_paths[b][oper_id] = {}


                pt_rows = {}
                param_rows = {}
                for op in chain_ops:
                    pt_rows[op] = proc_b[op, :].tolist()
                    param_rows[op] = param_b[op, :].tolist()
                # --- End chain-level pre-extraction ---


                sz_row = should_set_zero[b, j, :].tolist()  # [num_mas] bool
                pt_row_j = proc_time_of_step_oper[b, j, :].tolist()  # [num_mas] float
                candidate_machines = [m for m in range(num_ma) if not sz_row[m] and pt_row_j[m] > 0]
                # --- End candidate machine list ---

                if not candidate_machines:
                    continue
                # Enumerate complete feasible paths for each candidate machine (reusing pre-extracted data)
                for m_id in candidate_machines:
                    all_paths = self._compute_chain_feasible_paths(
                        chain_ops,
                        m_id,
                        t_b,
                        pm_list=pm_list,
                        mask_full_list=mask_full_list,
                        ma_is_proc=ma_is_proc,
                        ma_batch_end=ma_batch_end,
                        ma_batch_start=ma_batch_start,
                        ma_batch_param=ma_batch_param,
                        ma_batch_dur=ma_batch_dur,
                        ma_capacity=ma_capacity,
                        res_list_per_m=res_list_per_m,
                        pt_rows=pt_rows,
                        param_rows=param_rows,
                    )
                    if not all_paths:
                        should_set_zero[b, j, m_id] = True
                    else:
                        self._chain_feasible_paths[b][oper_id][m_id] = all_paths

        # An O-M pair is truly feasible only if it is available and the operation can be inserted into the active batch on a batch-processing machine.
        possible_o_m_pair_feasible = torch.where(should_set_zero, 0.0, possible_o_m_pair_proc_time)

        # ========== Unified eligibility — cache the complete eligible matrix for ppo_model ==========
        # eligible: bool [batch_size, num_jobs, num_mas], covering every batch entry, including completed ones
        self._cached_eligible_om_pairs = torch.where(
            should_set_zero, False, (possible_o_m_pair_proc_time > 0)
        )

        # Apply rightnow only when the current batch has a pending job; preserve the results above exactly when there is none.
        # Pending rightnow jobs are first recorded as future reservations at their respective job_avail_time during the current decision,
        # so ordinary operations cannot use required batch capacity or no-wait-chain windows before the reservation time.
        pending_rightnow = self._get_pending_rightnow_mask()
        for b in range(self.batch_size):
            pending_jobs = torch.nonzero(pending_rightnow[b], as_tuple=False).flatten().tolist()
            if not pending_jobs:
                continue
            # DB input and resource constraints guarantee a feasible joint assignment for rightnow; here we only tighten
            # eligibility and add no rightnow-specific exception or runtime fallback.
            self._apply_rightnow_joint_guard(b, pending_jobs)
            self.inplace_drop[b] = True

        # Compute the number of truly feasible O-M pairs
        num_eligible_om_pair_batch = torch.sum(self._cached_eligible_om_pairs, dim=[1, 2])

        # Return the number of truly feasible O-M pairs; advance time if the number is 0
        return num_eligible_om_pair_batch

    # For each batch, get the nearest job availability time strictly after the current time; return inf if none exists
    def _get_next_job_available_time(self):
        future_job_mask = (~self.mask_job_finish_batch) & (
            self.job_avail_time_batch_init > self.time[:, None]
        )
        future_job_available_time = torch.where(
            future_job_mask,
            self.job_avail_time_batch_init,
            torch.full_like(self.job_avail_time_batch_init, float("inf")),
        )
        return torch.min(future_job_available_time, dim=1)[0]

    # Get the nearest future reservation start strictly after the current time for each batch.
    # Future operations in a no-wait chain are usually reached through machine completion events; independent rightnow
    # reservations may be the only next event, so include them explicitly when advancing time.
    def _get_next_future_reservation_time(self):
        next_times = torch.full((self.batch_size,), float("inf"))
        for b, machine_reservations in self.future_reservations.items():
            current = self.time[b].item()
            for reservations in machine_reservations.values():
                for start, end, _oper_id, job_id, _param in reservations:
                    # Add independent rightnow reservation starts as future time events;
                    # no-wait-chain operations advance through machine completion events.
                    if not self.job_rightnow_batch_init[b, job_id]:
                        continue
                    if end > current and start > current:
                        next_times[b] = min(next_times[b].item(), float(start))
        return next_times

    def _get_drop_boundary_adjust_mask(
        self,
        flag_need_trans,
        earliest_available_time,
        fallback_time,
        earliest_real_event_time,
    ):

        del earliest_real_event_time






        return flag_need_trans & (earliest_available_time == fallback_time)

    # Advance instances whose current operation was dropped to the next time
    def drop_ins_transfer_to_next_time(self, is_drop):
        # Determine which instances need to move to the next time step
        drop_ins = torch.zeros(self.batch_size, dtype=torch.bool)
        drop_ins[self.batch_idxes[is_drop == 1]] = 1
        flag_need_trans = (drop_ins == 1) & (~self.done_batch)

        # Available time of all machines—i.e., the end time of the current batch, [batch_size, num_mas]
        available_time_of_machines = self.partial_machines_batch[:, :, 1]

        # Available machine times after the current time (skip all operations feasible now because the current operation was dropped)
        # If a machine's available time is > the current time, keep that time;
        # otherwise set it to a large value to mark the machine unavailable now (it is actually available, but all current O-M pairs must be excluded after drop)
        # Represent absent future machine events with inf when comparing machine, job-release, and reservation times.
        no_future_event = torch.full_like(available_time_of_machines, float("inf"))
        available_time_of_machines_greater_than_cur_time = torch.where(
            available_time_of_machines > self.time[:, None],
            available_time_of_machines,
            no_future_event,
        )

        # Find the earliest available time after the current time; this is the time each instance should advance to
        earliest_machine_available_time = torch.min(
            available_time_of_machines_greater_than_cur_time, dim=1
        )[0]
        # After drop, also consider the nearest future job release and reservation start as the next event.
        next_job_available_time = self._get_next_job_available_time()
        next_reservation_time = self._get_next_future_reservation_time()
        earliest_real_event_time = torch.minimum(
            torch.minimum(earliest_machine_available_time, next_job_available_time),
            next_reservation_time,
        )
        # Use the drop-boundary sentinel only if machines, job releases, and reservations have no real future event.
        # to avoid a sentinel preceding a release and causing an infinite loop at the same time.
        fallback_time = (torch.max(self.feature_opers_batch[:, 4, :]) + 1.0).expand(self.batch_size)
        earliest_available_time = torch.where(
            torch.isfinite(earliest_real_event_time),
            earliest_real_event_time,
            fallback_time,
        )



        # Boundary case: if the dropped O-M pairs are the instance's last operations, there is no earliest available time after now
        # Include equality at the drop boundary to prevent advancing beyond the last operations.
        need_adjust_mask = self._get_drop_boundary_adjust_mask(
            flag_need_trans,
            earliest_available_time,
            fallback_time,
            earliest_real_event_time,
        )
        if need_adjust_mask.any():










            adjusted_available_times = torch.where(
                need_adjust_mask[:, None] & (available_time_of_machines >= self.time[:, None]),
                available_time_of_machines,
                fallback_time[:, None],
            )
            available_time_of_machines_greater_than_cur_time = torch.where(
                need_adjust_mask[:, None],
                adjusted_available_times,
                available_time_of_machines_greater_than_cur_time,
            )
            earliest_machine_available_time = torch.min(
                available_time_of_machines_greater_than_cur_time, dim=1
            )[0]





        # The drop-boundary adjustment above changes only machine candidate times; merge in
        # job release and future reservation events again.
        earliest_available_time = torch.minimum(
            torch.minimum(earliest_machine_available_time, next_job_available_time),
            next_reservation_time,
        )



        # Find the machine with the earliest available time, [batch_size, num_mas]
        earliest_available_machine = torch.where(
            (available_time_of_machines == earliest_available_time[:, None]) & flag_need_trans[:, None],
            True,
            False,
        )




        self.time = torch.where(flag_need_trans, earliest_available_time, self.time)



        # Update intermediate state, variables, and features
        # aa: [batch_size, 9, num_mas]
        aa = self.partial_machines_batch.transpose(1, 2)

        # Set these machines to idle
        aa[:, 0, :][earliest_available_machine] = 1

        # Mark these machines as not processing
        aa[:, 4, :][earliest_available_machine] = 0

        # Set the processing parameter of these machines to 0 to indicate they are not processing
        aa[:, 8, :][earliest_available_machine] = 0

        # Set the number of operations being processed on these machines to 0
        aa[:, 5, :][earliest_available_machine] = 0

        self.partial_machines_batch = aa.transpose(1, 2)

        # Update utilization
        utiliz = self.partial_machines_batch[:, :, 2]
        utiliz = torch.minimum(utiliz, self.time[:, None].expand_as(utiliz))
        self.feature_mas_batch[:, 2, :] = utiliz / (self.time[:, None] + 1e-5)

        # Find the job of the operation being processed on the machine with the earliest start time
        # A batch-processing machine may have multiple jobs in self.partial_machines_batch, so handle more than one job
        # Find the jobs that should be marked as not processing
        job_need_to_set_false = self.jobs_status_in_mas_batch & earliest_available_machine[
            :, :, None
        ].expand_as(self.jobs_status_in_mas_batch)

        # ========== Clear only jobs on the machine that just completed; keep jobs still processing on other machines ==========
        jobs_to_clear = job_need_to_set_false.any(dim=1)  # [batch_size, num_jobs]
        self.mask_job_procing_batch = torch.where(jobs_to_clear, False, self.mask_job_procing_batch)

        # That machine is now available to start processing
        self.mask_ma_full_batch[earliest_available_machine] = False
        self.mask_ma_processing_batch[earliest_available_machine] = 0.0
        self.mask_job_finish_batch = torch.where(
            self.step_oper_id_of_each_job_batch == self.end_oper_id_of_each_job_batch + 1,
            True,
            self.mask_job_finish_batch,
        )

        # Update job processing status on the machine
        self.jobs_status_in_mas_batch = torch.where(
            job_need_to_set_false, False, self.jobs_status_in_mas_batch
        )

        # Handle future reservation activation/cleanup after drop
        self._process_future_reservations_on_time_advance(flag_need_trans)

        return

    # Instances to advance to the next time
    def transfer_to_next_time(self, num_eligible_om_pair_batch):
        """
        Transit to the next time
        """
        # Determine which instances need to advance: unfinished instances with no available O-M pairs, [batch_size]
        # Note: dropped instances have already advanced above, so they must now have available O-M pairs
        # Therefore, num_eligible_om_pair_batch must be nonzero for dropped instances; they do not need another transition
        flag_need_trans = (num_eligible_om_pair_batch == 0) & (~self.done_batch)

        # available_time of machines
        # Available times of all machines — i.e., the end times of the current batches, [batch_size, num_mas]
        available_time_of_machines = self.partial_machines_batch[:, :, 1]

        # remain available_time greater than current time
        # Available machine times after the current time
        # torch.max(self.feature_opers_batch[:, 4, :]) is the estimated maximum completion time for the current instance
        # If a machine's available time is later than now, keep it; otherwise set a large value to mark it unavailable now
        # Prioritize real future machine events; when there are none, use inf so the fallback sentinel cannot appear earlier than
        # job release / future reservation events as a false next event.
        no_future_event = torch.full_like(available_time_of_machines, float("inf"))
        available_time_of_machines_greater_than_cur_time = torch.where(
            available_time_of_machines > self.time[:, None],
            available_time_of_machines,
            no_future_event,
        )

        # Return the minimum value of available_time (the time to transit to)
        # Find the earliest available time after the current time; this is the time each instance should advance to
        earliest_machine_available_time = torch.min(
            available_time_of_machines_greater_than_cur_time, dim=1
        )[0]
        # When no O-M pair is available, the next event may also be a job release or future reservation start.
        next_job_available_time = self._get_next_job_available_time()
        next_reservation_time = self._get_next_future_reservation_time()
        earliest_real_event_time = torch.minimum(
            torch.minimum(earliest_machine_available_time, next_job_available_time),
            next_reservation_time,
        )
        # Use the deadlock-boundary sentinel only when none of the three real event types exists.
        fallback_time = (torch.max(self.feature_opers_batch[:, 4, :]) + 1.0).expand(self.batch_size)
        earliest_available_time = torch.where(
            torch.isfinite(earliest_real_event_time),
            earliest_real_event_time,
            fallback_time,
        )




        earliest_available_machine = torch.where(
            (available_time_of_machines == earliest_available_time[:, None]) & flag_need_trans[:, None],
            True,
            False,
        )

        # The time for each batch to transit to or stay in
        # Determine the time to which each instance should advance, then advance time
        self.time = torch.where(flag_need_trans, earliest_available_time, self.time)

        # Update partial schedule (state), variables and feature vectors
        # Update intermediate state, variables, and features
        # aa: [batch_size, 9, num_mas]
        aa = self.partial_machines_batch.transpose(1, 2)

        # Set these machines to idle
        aa[:, 0, :][earliest_available_machine] = 1

        # Mark these machines as not processing
        aa[:, 4, :][earliest_available_machine] = 0

        # Set the number of operations being processed on these machines to 0
        aa[:, 5, :][earliest_available_machine] = 0

        # Set the processing parameter of these machines to 0 to indicate they are not processing
        aa[:, 8, :][earliest_available_machine] = 0

        self.partial_machines_batch = aa.transpose(1, 2)

        # Update utilization
        utiliz = self.partial_machines_batch[:, :, 2]
        utiliz = torch.minimum(utiliz, self.time[:, None].expand_as(utiliz))
        self.feature_mas_batch[:, 2, :] = utiliz / (self.time[:, None] + 1e-5)

        # Find the job of the operation being processed on the machine with the earliest start time
        # A batch-processing machine may have multiple jobs in self.partial_machines_batch, so handle more than one job
        # Find the jobs that should be marked as not processing
        job_need_to_set_false = self.jobs_status_in_mas_batch & earliest_available_machine[
            :, :, None
        ].expand_as(self.jobs_status_in_mas_batch)

        # ========== Clear only jobs on the machine that just completed; keep jobs still processing on other machines ==========
        jobs_to_clear = job_need_to_set_false.any(dim=1)  # [batch_size, num_jobs]
        self.mask_job_procing_batch = torch.where(jobs_to_clear, False, self.mask_job_procing_batch)

        # That machine is now available to start processing
        self.mask_ma_full_batch[earliest_available_machine] = False
        self.mask_ma_processing_batch[earliest_available_machine] = 0.0
        self.mask_job_finish_batch = torch.where(
            self.step_oper_id_of_each_job_batch == self.end_oper_id_of_each_job_batch + 1,
            True,
            self.mask_job_finish_batch,
        )

        # Update job processing status on the machine
        self.jobs_status_in_mas_batch = torch.where(
            job_need_to_set_false, False, self.jobs_status_in_mas_batch
        )

        # Handle future reservation activation/cleanup after time advances
        self._process_future_reservations_on_time_advance(flag_need_trans)

        return

    # Handle future reservation activation/cleanup after time advances
    def _process_future_reservations_on_time_advance(self, flag_need_trans):
        """
        Activate or clean up future reservations after time advances.
        For each batch that advances, iterate over its future_reservations:
        1. reservation.start_time == new_time → activate as the active batch
        2. reservation.end_time <= new_time → remove
        3. reservation.start_time > new_time → keep

        Args:
            flag_need_trans: [batch_size] bool tensor, indicates which batches just advanced in time
        """
        for b in range(self.batch_size):
            if not flag_need_trans[b]:
                continue
            if b not in self.future_reservations:
                continue
            new_time = self.time[b].item()

            for m_id in list(self.future_reservations[b].keys()):
                reservations = self.future_reservations[b][m_id]
                remaining = []
                for res_start, res_end, oper_id, job_id, param in reservations:
                    if res_end <= new_time:
                        # Reservation expired; remove it
                        # Also clear jobs_status_in_mas_batch
                        if job_id < self.num_jobs:
                            self.jobs_status_in_mas_batch[b, m_id, job_id] = False
                        continue
                    elif abs(res_start - new_time) < SCHEDULE_COMPAT_EPS:
                        # ========== Check for batch-time overwrite risk ==========
                        was_proc_before = self.partial_machines_batch[b, m_id, 4].item() == 1.0
                        if was_proc_before:
                            old_start = self.partial_machines_batch[b, m_id, 6].item()
                            old_end = self.partial_machines_batch[b, m_id, 7].item()
                            old_param = self.partial_machines_batch[b, m_id, 8].item()
                            times_match = (
                                abs(old_start - res_start) < SCHEDULE_COMPAT_EPS
                                and abs(old_end - res_end) < SCHEDULE_COMPAT_EPS
                            )
                            params_match = old_param == param
                            if not (times_match and params_match):



                                print(
                                    f"[WARN] future_res_activate_on_active_machine: b={b} m={m_id} "
                                    f"new_time={new_time:.1f} "
                                    f"old_batch=[{old_start:.1f}, {old_end:.1f}) param={old_param:.1f} "
                                    f"new_batch=[{res_start:.1f}, {res_end:.1f}) param={param:.1f} "
                                    f"oper={oper_id} job={job_id}"
                                )

                        # The active batch has N operations; activating this reservation raises the count to N+1, which must be <= machine capacity.
                        cur_op_count = int(self.partial_machines_batch[b, m_id, 5].item())
                        machine_cap = int(self.machines_capacity_batch[b, m_id].item())
                        if cur_op_count + 1 > machine_cap:
                            print(
                                f"[WARN] future_res_activate_capacity_exceeded: b={b} m={m_id} "
                                f"new_time={new_time:.1f} cur_count={cur_op_count} cap={machine_cap} "
                                f"oper={oper_id} job={job_id} — SKIPPING activation"
                            )
                            continue  # Skip this reservation to prevent exceeding capacity
                        # Reservation activation: set it as the machine's active batch

                        if not was_proc_before:
                            self.partial_machines_batch[b, m_id, 6] = res_start  # Batch start
                            self.partial_machines_batch[b, m_id, 7] = res_end  # Batch end
                            self.partial_machines_batch[b, m_id, 8] = param  # Parameter
                            self.partial_machines_batch[b, m_id, 2] += res_end - res_start  # Utilization
                            self.partial_machines_batch[b, m_id, 1] = res_end  # Available time
                        # When already processing, keep batch times, parameters, utilization, and available time unchanged;
                        # only increment the operation count and update the mask.
                        self.partial_machines_batch[b, m_id, 4] = 1.0  # Processing
                        self.partial_machines_batch[b, m_id, 3] = float(job_id)

                        self.partial_machines_batch[b, m_id, 5] += 1.0  # Processing operation count



                        cur_count = int(self.partial_machines_batch[b, m_id, 5].item())
                        machine_cap = int(self.machines_capacity_batch[b, m_id].item())
                        if cur_count >= machine_cap:
                            self.partial_machines_batch[b, m_id, 0] = 0.0
                            self.mask_ma_full_batch[b, m_id] = True
                        else:
                            self.partial_machines_batch[b, m_id, 0] = 1.0
                            self.mask_ma_full_batch[b, m_id] = False

                        # Update mask
                        self.mask_ma_processing_batch[b, m_id] = True
                        if job_id < self.num_jobs:
                            self.mask_job_procing_batch[b, job_id] = True
                            self.jobs_status_in_mas_batch[b, m_id, job_id] = True

                        # Update feature_mas_batch

                        if not was_proc_before:
                            self.feature_mas_batch[b, 1, m_id] = res_end
                            self.feature_mas_batch[b, 5, m_id] = param
                        # Processing operation count always accumulates
                        self.feature_mas_batch[b, 4, m_id] += 1.0

                        # Neighbor-operation count: oper_ma_adj was already updated in _schedule_single_operation
                        self.feature_mas_batch[b, 0, m_id] = float(
                            torch.count_nonzero(self.oper_ma_adj_batch[b, :, m_id]).item()
                        )
                        # Utilization
                        utiliz_m = self.partial_machines_batch[b, m_id, 2].item()
                        cur_t = max(new_time, 1e-9)
                        self.feature_mas_batch[b, 2, m_id] = min(utiliz_m, cur_t) / cur_t
                    else:
                        # Future reservation; keep it
                        remaining.append((res_start, res_end, oper_id, job_id, param))

                if remaining:
                    self.future_reservations[b][m_id] = remaining
                else:
                    del self.future_reservations[b][m_id]

    # Reset the environment
    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.proc_times_batch = self.old_proc_times_batch.detach().clone()
        self.opers_param_batch = self.old_opers_param_batch.detach().clone()
        self.oper_seamless_batch = self.old_oper_seamless_batch.detach().clone()
        self.oper_ma_adj_batch = self.old_ope_ma_adj_batch.detach().clone()
        self.cal_cumul_adj_batch = self.old_cal_cumul_adj_batch.detach().clone()
        self.feature_opers_batch = self.old_feat_opes_batch.detach().clone()
        self.feature_mas_batch = self.old_feat_mas_batch.detach().clone()
        self.job_rightnow_batch_init = self.old_job_rightnow_batch_init.detach().clone()
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.last_time = torch.zeros(self.batch_size)

        if self.no_drop:  # Do not allow dropping operations
            self.inplace_drop = torch.ones(self.batch_size).bool()
        else:  # Allow dropping operations
            self.inplace_drop = torch.zeros(self.batch_size).bool()

        self.num_scheduled_oper_batch = torch.zeros(self.batch_size)
        self.step_oper_id_of_each_job_batch = self.first_oper_id_of_each_job_batch.detach().clone()
        self.mask_job_procing_batch = torch.full(
            size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False
        )
        self.mask_job_finish_batch = torch.full(
            size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False
        )
        self.mask_ma_full_batch = torch.full(
            size=(self.batch_size, self.num_machines),
            dtype=torch.bool,
            fill_value=False,
        )

        self.partial_schedules_batch = torch.zeros(size=(self.batch_size, self.max_num_opers_in_batch, 4))
        self.partial_schedules_batch[:, :, 2] = self.feature_opers_batch[:, 5, :]
        self.partial_schedules_batch[:, :, 3] = (
            self.feature_opers_batch[:, 5, :] + self.feature_opers_batch[:, 2, :]
        )

        self.partial_machines_batch = torch.zeros(size=(self.batch_size, self.num_machines, 9))

        self.partial_machines_batch[:, :, 1] = self.feature_mas_batch[:, 1, :]
        # Machines unavailable at time 0
        zero_not_available = self.partial_machines_batch[:, :, 1] != 0

        # This variable is a redundant derived cache of partial_machines_batch[:, :, 4] and is not currently used in scheduling decisions.
        # Rebuild it during reset as well, to avoid retaining machine-processing state from the previous episode.
        self.mask_ma_processing_batch = torch.full(
            size=(self.batch_size, self.num_machines),
            dtype=torch.bool,
            fill_value=False,
        )
        self.mask_ma_processing_batch[zero_not_available] = True

        # Mark initially available machines as idle and initially occupied machines as non-idle
        self.partial_machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 0][zero_not_available] = 0

        # Processing flag: 0 for initially available machines, 1 for initially occupied machines
        self.partial_machines_batch[:, :, 4] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 4][zero_not_available] = 1

        # Processing count: 0 for initially available machines, 1 for initially occupied machines
        self.partial_machines_batch[:, :, 5] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 5][zero_not_available] = 1

        # Set the initial occupied batch start time to -1 and its end time to the machine available time
        self.partial_machines_batch[:, :, 6] = torch.zeros(size=(self.batch_size, self.num_machines))
        self.partial_machines_batch[:, :, 7] = torch.zeros(size=(self.batch_size, self.num_machines))

        self.partial_machines_batch[:, :, 6][zero_not_available] = -1
        self.partial_machines_batch[:, :, 7] = self.partial_machines_batch[:, :, 1]

        # Parameter is 0 for idle machines and -2 for initially occupied machines
        self.partial_machines_batch[:, :, 8] = torch.zeros(size=(self.batch_size, self.num_machines))

        self.partial_machines_batch[:, :, 8][zero_not_available] = -2

        self.jobs_status_in_mas_batch = torch.zeros(
            size=(self.batch_size, self.num_machines, self.num_jobs), dtype=torch.bool
        )

        self.makespan_batch = torch.max(self.feature_opers_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)

        # Clear chain-related dynamic state
        self.future_reservations = {}
        for b in range(self.batch_size):
            self.future_reservations[b] = {}
        # Clear the feasible-chain-path cache
        self._chain_feasible_paths = {}
        # chain_info, chain_first_ops, and chain_next_of were parsed in __init__ and do not need to be reset (same as oper_seamless_batch)

        # After reset, repeat the eligibility and time-advance steps from __init__,
        # also replacing caches from the previous episode so initial job/machine availability takes effect again.
        self._cached_eligible_om_pairs = None
        num_eligible_om_pair_batch = self.find_eligible_om_pair()
        while ~((~((num_eligible_om_pair_batch == 0) & (~self.done_batch))).all()):
            self.transfer_to_next_time(num_eligible_om_pair_batch)
            num_eligible_om_pair_batch = self.find_eligible_om_pair()

        # self.state is a deep copy of old_state; after reset rebuilds internal tensors, dynamic state must be rebound,
        # otherwise state.time_batch and other fields still point to the construction-stage or previous-episode copy.
        self.state.update(
            self.batch_idxes,
            self.feature_opers_batch,
            self.feature_mas_batch,
            self.proc_times_batch,
            self.opers_param_batch,
            self.oper_seamless_batch,
            self.oper_ma_adj_batch,
            self.mask_job_procing_batch,
            self.mask_job_finish_batch,
            self.mask_ma_full_batch,
            self.step_oper_id_of_each_job_batch,
            self.time,
            self.inplace_drop,
            self.partial_machines_batch,
            chain_first_ops_batch=self.chain_first_ops,
            chain_info_batch=self.chain_info,
            future_reservations_batch=self.future_reservations,
            chain_feasible_paths_batch=self._chain_feasible_paths,
            eligible_om_pairs_batch=self._cached_eligible_om_pairs,
        )

        return self.state

    def get_idx(self, id_ope, batch_id):
        """
        Get job and operation (relative) index based on instance index and operation (absolute) index
        """
        # Get the job containing the current operation
        idx_job = max(
            [idx for (idx, val) in enumerate(self.first_oper_id_of_each_job_batch[batch_id]) if id_ope >= val]
        )
        # Get the current operation's index within its job
        idx_ope = id_ope - self.first_oper_id_of_each_job_batch[batch_id][idx_job]
        return idx_job, idx_ope


    def validate_gantt(self):
        """
        Check comprehensively whether the schedule satisfies all feasibility constraints:

        1. Every operation is scheduled (none are missing)
        2. Adjacent operations within one job do not overlap (op_i.end <= op_{i+1}.start)
        3. Machine capacity: concurrent operations on a machine never exceed its capacity
        4. Machine batch compatibility: overlapping operations on one machine belong to the same batch
           (same start_time, same end_time, and compatible param)
        5. Machine batches do not overlap: time intervals of different batches do not overlap
        6. Processing-time constraint: scheduled duration >= required processing time on the assigned machine
        7. The first operation of an ordinary job starts no earlier than job_avail_time
        8. The first operation of a rightnow job starts exactly at job_avail_time

        Returns:
            (is_valid, partial_schedules_batch, first_oper_id_of_each_job_batch, machines_capacity_batch)
        """
        total_violations = 0

        self.gantt_validation_failures = []

        for k in range(self.batch_size):
            schedule = self.partial_schedules_batch[k]  # [max_num_opers, 4]
            # Reuse the scheduling tensor precision for processing-duration validation, avoiding repeated lookup in the operation loop
            schedule_float_eps = torch.finfo(schedule.dtype).eps
            num_opers = int(self.num_opers_batch[k])
            num_jobs = self.num_jobs
            num_mas = self.num_machines
            proc_time = self.proc_times_batch[k]  # [num_opers, num_mas]
            opers_param = self.opers_param_batch[k]  # [num_opers, num_mas]
            capacity = self.machines_capacity_batch[k]  # [num_mas]
            nums_ope_job = self.num_opers_of_each_job_batch[k]  # [num_jobs]
            first_ope_job = self.first_oper_id_of_each_job_batch[k]  # [num_jobs]

            # ----- 1. Check that every operation is scheduled -----
            scheduled_count = int((schedule[:num_opers, 0] == 1).sum().item())
            if scheduled_count != num_opers:
                total_violations += 1
                print(f"[VIOLATION] batch={k}: unscheduled ops: {scheduled_count}/{num_opers}")

            # Collect scheduled operations: [(op_id, m_id, start, end, param), ...]
            ops = []
            for op_id in range(num_opers):
                if schedule[op_id, 0] != 1:
                    continue
                m_id = int(schedule[op_id, 1].item())
                start = schedule[op_id, 2].item()
                end = schedule[op_id, 3].item()
                param = opers_param[op_id, m_id].item()
                ops.append((op_id, m_id, start, end, param))

            # ----- 2. Release/rightnow first-operation timing constraints -----
            for job_id in range(num_jobs):
                first_op = int(first_ope_job[job_id].item())
                first_schedule = schedule[first_op]
                if first_schedule[0].item() != 1:
                    continue
                actual_start = first_schedule[2].item()
                required_release = self.job_avail_time_batch_init[k, job_id].item()
                if actual_start < required_release:
                    total_violations += 1
                    failure = {
                        "code": "job_release_early",
                        "batch": k,
                        "job": job_id,
                        "first_oper": first_op,
                        "required": required_release,
                        "actual": actual_start,
                    }
                    self.gantt_validation_failures.append(failure)
                    print(f"[VIOLATION][job_release_early] {failure}")
                if (
                    self.job_rightnow_batch_init[k, job_id]
                    and abs(actual_start - required_release) > SCHEDULE_COMPAT_EPS
                ):
                    total_violations += 1
                    failure = {
                        "code": "job_rightnow_mismatch",
                        "batch": k,
                        "job": job_id,
                        "first_oper": first_op,
                        "required": required_release,
                        "actual": actual_start,
                    }
                    self.gantt_validation_failures.append(failure)
                    print(f"[VIOLATION][job_rightnow_mismatch] {failure}")

            # ----- 3. Adjacent operations within each job do not overlap -----
            for job_id in range(num_jobs):
                n_ops_job = int(nums_ope_job[job_id].item())
                if n_ops_job <= 1:
                    continue
                first_op = int(first_ope_job[job_id].item())
                for j in range(n_ops_job - 1):
                    op_cur = first_op + j
                    op_next = first_op + j + 1
                    s_cur = schedule[op_cur]
                    s_next = schedule[op_next]
                    # Check only adjacent operations when both are scheduled
                    if s_cur[0] != 1 or s_next[0] != 1:
                        continue
                    if s_cur[3].item() > s_next[2].item() + 1e-6:
                        total_violations += 1
                        print(
                            f"[VIOLATION] batch={k} job={job_id}: "
                            f"op_{op_cur} end={s_cur[3].item():.1f} (m={int(s_cur[1].item())}) > "
                            f"op_{op_next} start={s_next[2].item():.1f} (m={int(s_next[1].item())})"
                        )
                        # Print the full operation chain for that job
                        last_op = first_op + n_ops_job - 1
                        chain_str = f"  job_{job_id} chain: "
                        for oi in range(first_op, last_op + 1):
                            s = schedule[oi]
                            status = "S" if s[0] > 0 else "U"
                            chain_str += (
                                f"[op_{oi}: {status}/m{int(s[1].item()):.0f} "
                                f"t={s[2].item():.1f}->{s[3].item():.1f}] "
                            )
                        print(chain_str)

            # ----- 4–6. Per-machine checks: capacity, batch compatibility, and non-overlapping batches -----
            for m_id in range(num_mas):
                cap = int(capacity[m_id].item())
                # Collect operations assigned to this machine
                m_ops = [(op_id, start, end, param) for (op_id, mi, start, end, param) in ops if mi == m_id]
                if not m_ops:
                    continue

                # Sort by start
                m_ops.sort(key=lambda x: x[1])

                # ----- 3. Capacity constraint: sweep-line check for concurrent operations -----
                events = []
                for op_id, start, end, param in m_ops:
                    events.append((start, 1))
                    events.append((end, -1))
                # At the same time, process -1 before +1 (finish before start)
                events.sort(key=lambda x: (x[0], x[1]))
                cur = 0
                max_concurrent = 0
                for _, delta in events:
                    cur += delta
                    if cur > max_concurrent:
                        max_concurrent = cur
                if max_concurrent > cap:
                    total_violations += 1
                    print(
                        f"[VIOLATION] batch={k} m={m_id}: "
                        f"max_concurrent={max_concurrent} > capacity={cap}"
                    )

                # ----- 4. Batch compatibility: overlapping operations must belong to the same batch -----
                # Same batch means (start, end) match within SCHEDULE_COMPAT_EPS tolerance.

                # then op_a cannot overlap any later operation either (because each later start >= op_b.start) ==========
                n_m_ops = len(m_ops)
                for i in range(n_m_ops):
                    op_a = m_ops[i]
                    for jj in range(i + 1, n_m_ops):
                        op_b = m_ops[jj]
                        # If op_a ends before op_b starts, it cannot overlap any later operation; exit the inner loop
                        if op_a[2] <= op_b[1]:
                            break
                        # If op_a and op_b overlap in time (op_a.end > op_b.start)
                        # → They must belong to the same batch
                        same_start = abs(op_a[1] - op_b[1]) < SCHEDULE_COMPAT_EPS
                        same_end = abs(op_a[2] - op_b[2]) < SCHEDULE_COMPAT_EPS
                        same_param = op_a[3] == op_b[3]
                        if not (same_start and same_end and same_param):
                            total_violations += 1
                            print(
                                f"[VIOLATION] batch={k} m={m_id}: overlapping ops in different batches — "
                                f"op_{op_a[0]} [{op_a[1]:.1f}, {op_a[2]:.1f}) vs "
                                f"op_{op_b[0]} [{op_b[1]:.1f}, {op_b[2]:.1f})"
                            )

                # ----- 5. Batches do not overlap: time intervals of different batches must not overlap -----
                # Group by (start, end) into batches and check for overlaps between batches
                batches = []  # [(start, end, [op_ids]), ...], merged using the shared tolerance.
                for op_id, start, end, param in m_ops:
                    matched_batch = None
                    for batch in batches:
                        if (
                            abs(batch[0] - start) <= SCHEDULE_COMPAT_EPS
                            and abs(batch[1] - end) <= SCHEDULE_COMPAT_EPS
                        ):
                            matched_batch = batch
                            break
                    if matched_batch is None:
                        batches.append([start, end, [op_id]])
                    else:
                        matched_batch[2].append(op_id)
                batch_list = sorted(batches, key=lambda x: x[0])
                for bi in range(len(batch_list) - 1):
                    prev_end = batch_list[bi][1]
                    next_start = batch_list[bi + 1][0]
                    if prev_end > next_start:
                        total_violations += 1
                        print(
                            f"[VIOLATION] batch={k} m={m_id}: overlapping batches — "
                            f"batch {batch_list[bi]} vs batch {batch_list[bi+1]}"
                        )

                # ----- 6. Processing-time constraint: scheduled duration >= required processing time -----
                for op_id, start, end, param in m_ops:
                    required_pt = proc_time[op_id, m_id].item()
                    scheduled_dur = end - start
                    # Only compensate for quantization error from subtracting float32 absolute times here,
                    # do not use the 1e-3 batch-equality tolerance.
                    duration_tolerance = max(
                        1e-6,
                        4 * schedule_float_eps * max(abs(start), abs(end), 1.0),
                    )
                    if scheduled_dur < required_pt - duration_tolerance:
                        total_violations += 1
                        # Print full floating-point precision and the difference to distinguish insufficient duration from float32 validation error
                        duration_delta = required_pt - scheduled_dur
                        print(
                            f"[VIOLATION] batch={k} m={m_id} op_{op_id}: "
                            f"start={start:.9f}, end={end:.9f}, "
                            f"scheduled_dur={scheduled_dur:.9f} < required_pt={required_pt:.9f}, "
                            f"delta={duration_delta:.9f}, tolerance={duration_tolerance:.9f}"
                        )

        if total_violations > 0:
            print(f"[validate_gantt] TOTAL VIOLATIONS: {total_violations}")
            return (
                False,
                self.partial_schedules_batch,
                self.first_oper_id_of_each_job_batch,
                self.machines_capacity_batch,
            )
        else:
            return (
                True,
                self.partial_schedules_batch,
                self.first_oper_id_of_each_job_batch,
                self.machines_capacity_batch,
            )
