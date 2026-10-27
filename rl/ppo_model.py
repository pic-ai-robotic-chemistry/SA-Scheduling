import copy
import math
import time
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

from colorama import Fore, Style
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.network.hgnn import GATedge, MLPs
from rl.network.mlp import MLPCritic, MLPActor

from rl.env.fjsp_env import EnvState


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.opers_param = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []

        # Chain-operation trajectories
        # Each step stores a list of length len(batch_idxes), containing chain data for each batch entry
        # Non-chain entries are None; chain entries are the tuple (chain_ops, assignments, feasible_sets)
        self.chain_data = (
            []
        )  # List of lists; one list per step, containing None or (chain_ops, assignments, feasible_sets)

    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.opers_param[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]

        # Clear chain-related Memory fields
        del self.chain_data[:]


# Heterogeneous graph neural network scheduler
class HGNNScheduler(nn.Module):
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras[
            "in_size_machine"
        ]  # Dimension of the raw feature vectors of machine nodes
        self.in_size_ope = model_paras[
            "in_size_oper"
        ]  # Dimension of the raw feature vectors of operation nodes
        self.d = model_paras["d"]  # Dimension of the embedding of machine nodes and operation nodes
        self.hidden_size_ope = model_paras["hidden_size_oper"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations

        self.num_heads: int = model_paras["num_heads"]
        self.L = model_paras["L"]
        self.dropout = model_paras["dropout"]
        self.residual = model_paras["residual"]

        # Machine node embedding
        self.get_machines = nn.ModuleList(
            [
                GATedge(
                    ((self.in_size_ope, self.in_size_ma) if i == 0 else (self.d, self.d)),
                    self.d,
                    self.num_heads,
                    dropout=self.dropout,
                    residual=self.residual,
                )
                for i in range(self.L)
            ]
        )

        # Operation node embedding
        self.get_operations = nn.ModuleList(
            [
                MLPs(
                    [self.d] + [self.in_size_ope if i == 0 else self.d] * 3,
                    self.hidden_size_ope,
                    self.d,
                    self.num_heads,
                    dropout=self.dropout,
                    residual=self.residual,
                )
                for i in range(self.L)
            ]
        )

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(
            self.device
        )
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(
            self.device
        )

    def forward(self):



        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    """
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    """

    def get_normalized(
        self,
        raw_opes,
        raw_mas,
        proc_time,
        opers_param,
        batch_idxes,
        nums_opes,
        flag_sample=False,
        flag_train=False,
    ):
        """
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        """
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        mean_opes = []
        std_opes = []
        for i in range(batch_size):
            # During training, operation feature vectors are padded to the largest operation count in the batch;
            # use only [: nums_opes[i]] because the remaining entries are invalid
            mean_opes.append(torch.mean(raw_opes[i, : nums_opes[i], :], dim=-2, keepdim=True))
            std_opes.append(torch.std(raw_opes[i, : nums_opes[i], :], dim=-2, keepdim=True))


            proc_mask_i = proc_time[i] > 0
            proc_values = proc_time[i][proc_mask_i]
            proc_norm = self.feature_normalize(proc_values)
            proc_time[i][proc_mask_i] = proc_norm

            opers_param_mask_i = opers_param[i] != 0
            opers_param_values = opers_param[i][opers_param_mask_i]
            opers_param_norm = self.feature_normalize(opers_param_values)
            opers_param[i][opers_param_mask_i] = opers_param_norm

        mean_opes = torch.stack(mean_opes, dim=0)
        std_opes = torch.stack(std_opes, dim=0)

        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True)

        proc_time_norm = proc_time
        opers_param_norm = opers_param

        return (
            (raw_opes - mean_opes) / (std_opes + 1e-5),
            (raw_mas - mean_mas) / (std_mas + 1e-5),
            proc_time_norm,
            opers_param_norm,
        )

    # HGNN forward pass shared by action selection and policy evaluation.
    def _forward_hgnn(
        self,
        ope_ma_adj,
        ope_pre_adj,
        ope_sub_adj,
        batch_idxes_for_ops,
        features,
        nums_opes=None,
        per_instance_pool=False,
    ):
        """
        Shared HGNN forward pass (L iterations + pooling), reused by get_action_prob and evaluate.

        Args:
            ope_ma_adj: [B, N_opes, N_mas] operation-machine adjacency
            ope_pre_adj: [B, N_opes, N_opes] operation-predecessor adjacency
            ope_sub_adj: [B, N_opes, N_opes] operation-subsequent adjacency
            batch_idxes_for_ops: [B] batch indices, passed as the fourth argument to get_operations
            features: tuple of (norm_opes, norm_mas, norm_proc, norm_opers_param)
            nums_opes: [B] number of valid operations per instance; required when per_instance_pool=True
            per_instance_pool: True=exact per-instance pooling (exclude padding), False=global average pooling

        Returns:
            h_opes: [B, N_opes, d]  Final operation embeddings
            h_mas:  [B, N_mas, d]   Final machine embeddings
            h_opes_pooled: [B, d]   Pooled operation embeddings
            h_mas_pooled:  [B, d]   Pooled machine embeddings
        """
        for i in range(self.L):
            # Stage 1: machine-node embedding
            h_mas = self.get_machines[i](ope_ma_adj, features)
            features = (features[0], h_mas, features[2], features[3])
            # Stage 2: operation-node embedding
            h_opes = self.get_operations[i](
                ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes_for_ops, features
            )
            features = (h_opes, features[1], features[2], features[3])

        # Pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # [B, d]

        if per_instance_pool:
            # Exact per-instance pooling: average only valid operations and exclude padding
            h_opes_pooled = []
            for i in range(ope_ma_adj.size(0)):
                h_opes_pooled.append(torch.mean(h_opes[i, : nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # [B, d]
        else:
            # Global average pooling (padding operations are zeroed by a mask during training/sampling and do not affect gradients)
            h_opes_pooled = h_opes.mean(dim=-2)  # [B, d]

        return h_opes, h_mas, h_opes_pooled, h_mas_pooled


    # Actor score construction shared by action selection and policy evaluation.
    def _build_action_scores(self, h_opes, h_mas, h_opes_pooled, h_mas_pooled, jobs_gather, num_mas):
        """
        Build raw Actor scores from HGNN outputs (before masking).



        Args:
            h_opes: [B, N_opes, d]
            h_mas: [B, N_mas, d]
            h_opes_pooled: [B, d]
            h_mas_pooled: [B, d]
            jobs_gather: [B, num_jobs, d] indices used to gather each job's next operation from h_opes
            num_mas: int, number of machines

        Returns:
            scores: [B, num_jobs * num_mas + 1] raw Actor scores (including the final drop column, before masking)
        """
        h_step_opes = h_opes.gather(1, jobs_gather)
        h_step_opes_padding = h_step_opes.unsqueeze(-2).expand(-1, -1, num_mas, -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_step_opes_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_step_opes_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_step_opes_padding)

        h_actions = torch.cat(
            (h_step_opes_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
            dim=-1,
        ).transpose(1, 2)

        # Actor MLP input: [features for each O-M pair | global state features] (row-major flattening + global row)
        h_action_state = torch.cat((h_opes_pooled, h_mas_pooled, h_opes_pooled, h_mas_pooled), dim=-1)
        x1_reshaped = h_actions.reshape(h_actions.shape[0], -1, h_actions.shape[-1])
        x2_expanded = h_action_state.unsqueeze(1)
        h_action_new = torch.cat([x1_reshaped, x2_expanded], dim=1)
        return self.actor(h_action_new).squeeze(dim=-1)


    # Get an action from the policy network based on the environment state
    def get_action_prob(self, state: EnvState, memories: Memory, flag_sample=False, flag_train=False):
        """
        Get the probability of selecting each action in decision-making
        """
        # Uncompleted instances
        batch_idxes = state.batch_idxes

        # Raw feats
        # After transpose, shape: (batch_size, num_opers, oper_feat_dim)
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]

        # After transpose, shape: (batch_size, num_machines, machine_feat_dim)
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]

        # shape: (batch_size, num_opers, num_machines)
        proc_time = state.proc_times_batch[batch_idxes]

        # shape: (batch_size, num_opers, num_machines)
        opers_param = state.opers_param_batch[batch_idxes].float()

        # Normalize
        # Normalize the raw features
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(
            raw_opes, raw_mas, proc_time, opers_param, batch_idxes, nums_opes, flag_sample, flag_train
        )
        # ========== Keep references to normalized features (deep-copy once when storing in Memory during training) ==========
        norm_opes = features[0]
        norm_mas = features[1]
        norm_proc = features[2]
        norm_opers_param = features[3]

        # Compute operation and machine embeddings.
        h_opes, h_mas, h_opes_pooled, h_mas_pooled = self._forward_hgnn(
            state.ope_ma_adj_batch[batch_idxes],
            state.ope_pre_adj_batch[batch_idxes],
            state.ope_sub_adj_batch[batch_idxes],
            state.batch_idxes,
            features,
            nums_opes=nums_opes,
            per_instance_pool=(not flag_sample and not flag_train),
        )

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        # Find eligible O-M pairs, gather their feature tensors, and compute actions
        # Shape: (batch_size, num_jobs), with values clipped to a valid range
        step_oper_id_of_each_job_batch = torch.where(
            state.step_oper_id_of_each_job_batch > state.end_ope_biases_batch,
            state.end_ope_biases_batch,
            state.step_oper_id_of_each_job_batch,
        )

        # shape: (len(batch_idxes), num_jobs, d)
        jobs_gather = step_oper_id_of_each_job_batch.unsqueeze(-1).expand(-1, -1, h_opes.size(-1))[
            batch_idxes
        ]

        # Compute scores for scheduling and drop actions.
        num_mas = state.proc_times_batch.size(-1)
        scores = self._build_action_scores(h_opes, h_mas, h_opes_pooled, h_mas_pooled, jobs_gather, num_mas)

        # ========== Unified eligibility — read directly from EnvState ==========
        eligible = state.eligible_om_pairs_batch[batch_idxes]

        if (~eligible).all():
            print(Fore.RED + "No solution found!" + Style.RESET_ALL)
            return

        # Mask infeasible O-M pairs and batches where drop is disabled
        mask = eligible.transpose(1, 2).flatten(1)
        extended_mask = torch.cat([mask, ~state.inplace_drop_batch[batch_idxes][:, None]], dim=1)
        scores[~extended_mask] = float("-inf")

        # Action probabilities, including the drop action
        action_probs = F.softmax(scores, dim=1)

        # shape: [len(batch_idxes), 2 * d]
        h_state_embedding = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        # Store data in memory during training
        # Store independent tensor copies without retaining the computation graph.
        if flag_train == True:
            memories.ope_ma_adj.append(state.ope_ma_adj_batch[batch_idxes].detach().clone())
            memories.ope_pre_adj.append(state.ope_pre_adj_batch[batch_idxes].detach().clone())
            memories.ope_sub_adj.append(state.ope_sub_adj_batch[batch_idxes].detach().clone())
            memories.batch_idxes.append(state.batch_idxes.detach().clone())
            memories.raw_opes.append(norm_opes.detach().clone())
            memories.raw_mas.append(norm_mas.detach().clone())
            memories.proc_time.append(norm_proc.detach().clone())
            memories.opers_param.append(norm_opers_param.detach().clone())
            memories.nums_opes.append(nums_opes.detach().clone())
            memories.jobs_gather.append(jobs_gather.detach().clone())
            memories.eligible.append(eligible.detach().clone())

        # Return HGNN embeddings for reuse when selecting machines along a chain.
        return (
            action_probs,
            step_oper_id_of_each_job_batch,
            h_state_embedding,
            h_opes,
            h_mas,
            h_opes_pooled,
            h_mas_pooled,
        )

    # Build Actor input for a chain operation, [num_feasible, 4d]
    def _build_chain_op_input(self, h_opes_b, h_mas_b, h_opes_pooled_b, h_mas_pooled_b, op_id, feasible_list):
        """
        Build Actor MLP inputs for one chain operation op_id on the given feasible-machine list.

        Input format matches the main action: concatenate [h_op, h_ma, h_opes_pooled, h_mas_pooled]
        Each feasible machine corresponds to one row, for num_feasible rows in total.

        Args:
            h_opes_b: [max_opes, d] operation embeddings for one batch entry
            h_mas_b: [num_mas, d] machine embeddings for one batch entry
            h_opes_pooled_b: [d] pooled operation embedding for one batch entry
            h_mas_pooled_b: [d] pooled machine embedding for one batch entry
            op_id: int, operation ID
            feasible_list: list[int], feasible machine IDs

        Returns:
            Tensor [num_feasible, 4*d]
        """
        n_feasible = len(feasible_list)
        h_op_single = h_opes_b[op_id, :]  # [d]
        h_op_exp = h_op_single.unsqueeze(0).expand(n_feasible, -1)  # [n_feasible, d]
        h_mas_feas = h_mas_b[feasible_list, :]  # [n_feasible, d]
        h_pooled_op_exp = h_opes_pooled_b.unsqueeze(0).expand(n_feasible, -1)  # [n_feasible, d]
        h_pooled_ma_exp = h_mas_pooled_b.unsqueeze(0).expand(n_feasible, -1)  # [n_feasible, d]
        return torch.cat([h_op_exp, h_mas_feas, h_pooled_op_exp, h_pooled_ma_exp], dim=-1)


    # select_chain_machines: neural-network-based machine selection for later operations in a chain
    def select_chain_machines(
        self,
        state: EnvState,
        local_b,  # int, local batch index in the HGNN output tensor (= position in batch_idxes)
        chain_ops,  # list[int], IDs of all operations in the chain
        m_1,  # int, machine selected for the chain's first operation
        all_paths,  # list[tuple], complete feasible paths from state.chain_feasible_paths_batch
        h_opes,  # [len(batch_idxes), max_opes, d]
        h_mas,  # [len(batch_idxes), num_mas, d]
        h_opes_pooled,  # [len(batch_idxes), d]
        h_mas_pooled,  # [len(batch_idxes), d]
        flag_sample=True,
        flag_train=True,
    ):
        """
        Use the Actor MLP to select machines for later operations in a chain.
        Filter complete feasible paths at each step so every choice leads to at least one complete feasible path.

        Core guarantee (two-layer feasibility mechanism):
        - Layer 1: all_paths is enumerated by _compute_chain_feasible_paths; every path is a complete feasible chain
        - Layer 2: project remaining_paths at each step → Actor selection → filter to paths consistent with the choice
        - The feasible set is non-empty at every step, and every machine in it belongs to a complete path consistent with prior choices

        Args:
            state: Environment state
            local_b: Local batch index in the HGNN output
            chain_ops: List of all operation IDs in the chain [op_0, op_1, ..., op_K]
            m_1: Machine selected for the chain's first operation op_0
            all_paths: List of complete feasible paths [(m_1, m_2, ...), ...]
            h_opes, h_mas, h_opes_pooled, h_mas_pooled: HGNN outputs
            flag_sample: Sampling vs. argmax
            flag_train: Training mode

        Returns:
            (assignments_dict, total_log_prob, stored_feasible_sets)
            - assignments_dict: {oper_id: mas_id}
            - total_log_prob: float tensor, sum of log probabilities for all later machine choices in the chain
            - stored_feasible_sets: list[list[int]], feasible machines shown to the Actor at each step (for replay in evaluate)
        """
        remaining = list(all_paths)  # list[tuple], remaining feasible paths
        total_log_prob = torch.tensor(0.0, device=self.device)
        assignments = {chain_ops[0]: m_1}
        stored_feasible_sets = []  # list[list[int]], feasible machine list stored for each step

        h_opes_b = h_opes[local_b]  # [max_opes, d]
        h_mas_b = h_mas[local_b]  # [num_mas, d]
        h_opes_pooled_b = h_opes_pooled[local_b]  # [d]
        h_mas_pooled_b = h_mas_pooled[local_b]  # [d]

        for idx, op in enumerate(chain_ops[1:]):
            # Project the feasible machines for this operation from the remaining paths
            feasible = sorted(set(p[idx] for p in remaining))  # list[int], guaranteed non-empty

            # Build Actor inputs and score them
            h_input = self._build_chain_op_input(
                h_opes_b, h_mas_b, h_opes_pooled_b, h_mas_pooled_b, op, feasible
            )
            scores = self.actor(h_input).squeeze(-1)  # [num_feasible]
            probs = F.softmax(scores, dim=0)
            dist = Categorical(probs)

            if flag_sample:
                chosen_idx = dist.sample()
            else:
                chosen_idx = probs.argmax(dim=0)

            chosen_m = feasible[chosen_idx.item()]
            assignments[op] = chosen_m
            total_log_prob = total_log_prob + dist.log_prob(chosen_idx)
            stored_feasible_sets.append(feasible)  # Store the feasible list (list[int] is serializable)

            # Filter to paths consistent with the current choice
            remaining = [p for p in remaining if p[idx] == chosen_m]
            # remaining is guaranteed non-empty (chosen_m comes from feasible, a projection with a matching path)

        return assignments, total_log_prob, stored_feasible_sets


    # Select a regular action and assign machines along any no-wait chain.
    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        # Get action probabilities and HGNN embeddings.
        result = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)
        if result is None:
            return None
        action_probs, step_oper_id_of_each_job_batch, _, h_opes, h_mas, h_opes_pooled, h_mas_pooled = result

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        is_drop_action = action_indexes == action_probs.size(1) - 1

        # Calculate the machine, job and operation index based on the action index
        num_jobs = state.mask_job_finish_batch.size(1)
        mas = (action_indexes / num_jobs).long()
        jobs = (action_indexes % num_jobs).long()
        opes = step_oper_id_of_each_job_batch[state.batch_idxes, jobs]

        # Chain handling: call select_chain_machines for the first operation in a chain
        batch_idxes = state.batch_idxes
        chain_assignments_dict = {}  # {global_b: {oper_id: mas_id}}
        chain_log_prob_total = torch.zeros(len(batch_idxes), device=self.device)
        step_chain_data = [None] * len(batch_idxes) if flag_train else None  # Do not allocate outside training


        if state.chain_first_ops_batch:
            for pos in range(len(batch_idxes)):
                global_b = batch_idxes[pos].item()
                if is_drop_action[pos]:
                    continue
                oper_id = opes[pos].item()
                if (
                    global_b in state.chain_first_ops_batch
                    and oper_id in state.chain_first_ops_batch[global_b]
                ):
                    chain_ops = state.chain_info_batch[global_b][oper_id]
                    m_1 = mas[pos].item()
                    all_paths = (
                        state.chain_feasible_paths_batch.get(global_b, {}).get(oper_id, {}).get(m_1, [])
                    )
                    if not all_paths:
                        continue

                    chain_assignments, chain_log_prob, feasible_sets = self.select_chain_machines(
                        state=state,
                        local_b=pos,
                        chain_ops=chain_ops,
                        m_1=m_1,
                        all_paths=all_paths,
                        h_opes=h_opes,
                        h_mas=h_mas,
                        h_opes_pooled=h_opes_pooled,
                        h_mas_pooled=h_mas_pooled,
                        flag_sample=flag_sample,
                        flag_train=flag_train,
                    )
                    chain_assignments_dict[global_b] = chain_assignments
                    chain_log_prob_total[pos] = chain_log_prob
                    if flag_train:
                        step_chain_data[pos] = (chain_ops, chain_assignments, feasible_sets)

        # During training, save chain data and record the joint log_prob of regular and chain actions.
        if flag_train:
            memories.chain_data.append(step_chain_data)
            total_log_prob = dist.log_prob(action_indexes) + chain_log_prob_total
            memories.logprobs.append(total_log_prob)
            memories.action_indexes.append(action_indexes)

        return (torch.stack((opes, mas, jobs, is_drop_action), dim=1).t(), chain_assignments_dict)


    # Evaluate the joint log_prob of regular and chain actions.
    def evaluate(
        self,
        ope_ma_adj,
        ope_pre_adj,
        ope_sub_adj,
        raw_opes,
        raw_mas,
        proc_time,
        opers_param,
        jobs_gather,
        eligible,
        action_envs,
        chain_data_flat=None,  # Flattened chain data
        flag_sample=False,
    ):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time, opers_param)

        # Compute operation and machine embeddings.
        h_opes, h_mas, h_opes_pooled, h_mas_pooled = self._forward_hgnn(
            ope_ma_adj,
            ope_pre_adj,
            ope_sub_adj,
            batch_idxes,
            features,
            per_instance_pool=False,
        )

        # Compute scores for scheduling and drop actions.
        scores = self._build_action_scores(
            h_opes, h_mas, h_opes_pooled, h_mas_pooled, jobs_gather, proc_time.size(-1)
        )
        h_state_embedding = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)

        # The eligible matrix was already computed by get_act_prob during data collection
        mask = eligible.transpose(1, 2).flatten(1)
        extended_mask = torch.cat([mask, torch.ones(len(mask), 1, dtype=torch.bool)], dim=1)

        scores[~extended_mask] = float("-inf")

        action_probs = F.softmax(scores, dim=1)
        dist = Categorical(action_probs)  # Keep the minibatch dimension to prevent squeeze from removing it for a single sample.
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()

        state_values = self.critic(h_state_embedding)

        # Compute chain log_prob (use the stored feasible_sets when replaying training)
        # chain_data_flat aligns one-to-one with minibatch rows (including None placeholders) and has length minibatch_size
        minibatch_size = ope_ma_adj.size(0)
        chain_logprobs = torch.zeros(minibatch_size, device=self.device)
        if chain_data_flat is not None and len(chain_data_flat) > 0:
            for i in range(minibatch_size):
                if i >= len(chain_data_flat):
                    break
                cd = chain_data_flat[i]
                if cd is None:
                    continue
                chain_ops, assignments, feasible_sets = cd
                for op_idx, op in enumerate(chain_ops[1:]):
                    feasible = feasible_sets[op_idx]  # list[int], sorted
                    if not feasible:
                        continue
                    # Build Actor inputs for this operation (same as select_chain_machines §5.1)
                    h_input = self._build_chain_op_input(
                        h_opes[i], h_mas[i], h_opes_pooled[i], h_mas_pooled[i], op, feasible
                    )
                    scores_chain = self.actor(h_input).squeeze(-1)  # [num_feasible]
                    probs_chain = F.softmax(scores_chain, dim=0)
                    dist_chain = Categorical(probs_chain)  # Use a separate name to avoid overwriting the outer dist
                    # Find the index of the selected machine in the feasible list
                    chosen_m = assignments[op]
                    try:
                        chosen_idx = feasible.index(chosen_m)
                    except ValueError:
                        # Defensive check: skip if the machine is not feasible (inconsistent data)
                        continue
                    chain_logprobs[i] = chain_logprobs[i] + dist_chain.log_prob(
                        torch.tensor(chosen_idx, device=self.device)
                    )

        action_logprobs = action_logprobs + chain_logprobs

        return action_logprobs, state_values.squeeze(dim=1).double(), dist_entropys


class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        # ========== LR scheduler — decay by lr_gamma after every lr_step_size updates ==========
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_paras.get("lr_step_size", 4000),
            gamma=train_paras.get("lr_gamma", 0.5),
        )
        self.MseLoss = nn.MSELoss()

    # Perform one training update
    def update(self, memory: Memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating

        # Organize samples by environment index and time step to handle different numbers of active environments at each step.
        def cat_by_idx(memory_data, batch_indices):
            """
            Concatenate data by batch index using vectorized operations.

            Parameters:
                memory_data (list of Tensor): data for each step; each element has shape (n_t, ...)
                batch_indices (list of Tensor): batch indices for each step; each element has shape (n_t,)

            Returns:
                Tensor: concatenated tensor with shape (total_samples, ...)
            """
            # Collect all data, batch indices, and step numbers
            data_list = []
            bidx_list = []
            step_list = []
            for step_idx, (data, bidx) in enumerate(zip(memory_data, batch_indices)):
                data_list.append(data)
                bidx_list.append(bidx)
                step_list.append(torch.full((data.size(0),), step_idx, dtype=torch.long, device=data.device))

            # Concatenate all data, batch indices, and step numbers
            all_data = torch.cat(data_list, dim=0)  # (total_samples, ...)
            all_bidx = torch.cat(bidx_list, dim=0)  # (total_samples,)
            all_steps = torch.cat(step_list, dim=0)  # (total_samples,)

            # Create sort keys: batch index first, then step order
            max_step = len(memory_data)
            sort_key = all_bidx * (max_step + 1) + all_steps

            # Sort data by the sort keys
            sorted_indices = torch.argsort(sort_key)
            sorted_data = all_data[sorted_indices]
            sorted_bidx = all_bidx[sorted_indices]

            # Get the starting position and sample count for each batch
            unique_bidx, counts = torch.unique_consecutive(sorted_bidx, return_counts=True)

            # Split and concatenate the data
            splits = torch.split(sorted_data, counts.tolist())
            return torch.cat(splits, dim=0)

        old_ope_ma_adj = cat_by_idx(memory.ope_ma_adj, memory.batch_idxes)
        old_ope_pre_adj = cat_by_idx(memory.ope_pre_adj, memory.batch_idxes)
        old_ope_sub_adj = cat_by_idx(memory.ope_sub_adj, memory.batch_idxes)
        old_raw_opes = cat_by_idx(memory.raw_opes, memory.batch_idxes)
        old_raw_mas = cat_by_idx(memory.raw_mas, memory.batch_idxes)
        old_proc_time = cat_by_idx(memory.proc_time, memory.batch_idxes)
        old_opers_param = cat_by_idx(memory.opers_param, memory.batch_idxes)
        old_jobs_gather = cat_by_idx(memory.jobs_gather, memory.batch_idxes)
        old_eligible = cat_by_idx(memory.eligible, memory.batch_idxes)

        # Flatten chain Memory data and align it with the main data rows
        # cat_by_idx_chain flattens chain data using the same sorting logic as cat_by_idx,
        # preserving None placeholders so each row aligns with one main-data row.
        def cat_by_idx_chain(memory_chain_data, batch_indices):
            """
            Flatten the chain Memory field, sort by batch_idx, and preserve None placeholders.
            Return a list aligned one-to-one with the main data from cat_by_idx.
            memory_chain_data: list of lists, one list per step, containing None or a tuple
            batch_indices: list of Tensor, with the same length as memory_chain_data
            """
            # Collect all data, batch indices, and step numbers (same logic as cat_by_idx)
            data_list = []
            bidx_list = []
            step_list = []
            for step_idx, (data, bidx) in enumerate(zip(memory_chain_data, batch_indices)):
                # data is a list of length len(bidx), with each element either None or a tuple
                for local_i, item in enumerate(data):
                    data_list.append(item)  # None or a tuple
                    bidx_list.append(bidx[local_i].item())
                    step_list.append(step_idx)

            if not data_list:
                return []

            # Create sort keys: batch index first, then step order (exactly as in cat_by_idx)
            max_step = len(memory_chain_data)
            sort_keys = [bidx * (max_step + 1) + step for bidx, step in zip(bidx_list, step_list)]
            sorted_pairs = sorted(zip(sort_keys, data_list), key=lambda x: x[0])
            return [d for _, d in sorted_pairs]

        chain_data_flat = cat_by_idx_chain(memory.chain_data, memory.batch_idxes)

        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)

        old_logprobs = cat_by_idx(memory.logprobs, memory.batch_idxes)
        old_action_envs = cat_by_idx(memory.action_indexes, memory.batch_idxes)

        # Compute and normalize discounted returns
        returns_envs = []
        discounted_returns = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_return = 0

            # Keep only the trajectory up to and including the first termination in this environment.
            first_true_index = next(
                (idx for idx, value in enumerate(memory_is_terminals[i]) if value == True),
                len(memory_is_terminals[i]),
            )
            new_memory_is_terminals_i = memory_is_terminals[i][: first_true_index + 1]
            new_memory_rewards_i = memory_rewards[i][: first_true_index + 1]

            for reward, is_terminal in zip(
                reversed(new_memory_rewards_i), reversed(new_memory_is_terminals_i)
            ):
                if is_terminal:
                    discounted_returns += discounted_return
                    discounted_return = 0
                discounted_return = reward + (self.gamma * discounted_return)
                rewards.insert(0, discounted_return)
            discounted_returns += discounted_return
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            returns_envs.append(rewards)

        returns_envs = torch.cat(returns_envs)

        loss_epochs = 0
        policy_loss_epochs = 0
        value_loss_epochs = 0
        entropy_loss_epochs = 0

        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):

            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size

                if start_idx < end_idx:
                    logprobs, state_values, dist_entropy = self.policy.evaluate(
                        old_ope_ma_adj[start_idx:end_idx, :, :],
                        old_ope_pre_adj[start_idx:end_idx, :, :],
                        old_ope_sub_adj[start_idx:end_idx, :, :],
                        old_raw_opes[start_idx:end_idx, :, :],
                        old_raw_mas[start_idx:end_idx, :, :],
                        old_proc_time[start_idx:end_idx, :, :],
                        old_opers_param[start_idx:end_idx, :, :],
                        old_jobs_gather[start_idx:end_idx, :, :],
                        old_eligible[start_idx:end_idx, :, :],
                        old_action_envs[start_idx:end_idx],
                        # Pass the flattened chain data
                        chain_data_flat=chain_data_flat[start_idx:end_idx] if chain_data_flat else None,
                    )
                else:
                    continue

                advantages = (
                    returns_envs[i * minibatch_size : (i + 1) * minibatch_size] - state_values.detach()
                )

                ratios = torch.exp(
                    logprobs - old_logprobs[i * minibatch_size : (i + 1) * minibatch_size].detach()
                )
                clip_ratios = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)

                policy_loss = -self.A_coeff * torch.min(ratios * advantages, clip_ratios * advantages)
                value_loss = self.vf_coeff * self.MseLoss(
                    state_values, returns_envs[i * minibatch_size : (i + 1) * minibatch_size]
                )
                entropy_loss = -self.entropy_coeff * dist_entropy
                loss = policy_loss + value_loss + entropy_loss

                policy_loss_epochs += policy_loss.mean().detach()
                value_loss_epochs += value_loss.mean().detach()
                entropy_loss_epochs += entropy_loss.mean().detach()
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # ========== Step the LR scheduler at the end of each update cycle ==========
        self.scheduler.step()

        return (
            loss_epochs.item() / self.K_epochs,
            discounted_returns.item() / (self.num_envs * train_paras["update_timestep"]),
            (
                policy_loss_epochs / self.K_epochs,
                value_loss_epochs / self.K_epochs,
                entropy_loss_epochs / self.K_epochs,
            ),
        )
