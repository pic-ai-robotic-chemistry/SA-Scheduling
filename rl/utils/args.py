import argparse
import torch
import numpy as np
import random


# Ensure algorithm reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Configure PyTorch parameters
def set_torch_params(device_id=0):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(threshold=np.inf, sci_mode=False)
    return device


# The RL argument parser keeps only training, inference, and environment parameters.
def build_argument_parser():
    parser = argparse.ArgumentParser(description="Run FJSP-B with specified CUDA device.")

    # ----------------------Normal params--------------------------------
    parser.add_argument("--cuda", default=0, type=int, help="CUDA device ID (default: 0)")
    parser.add_argument("--gen_case", action="store_true", help="Whether to randomly generate train cases")

    # ----------------------Env params--------------------------------
    parser.add_argument("--num_jobs", default=10, type=int, help="")
    parser.add_argument("--num_machines", default=10, type=int, help="")
    parser.add_argument("--reward_punish", default=1.0, type=float, help="")
    parser.add_argument("--batch_size", default=20, type=int, help="")
    parser.add_argument("--oper_feat_dim", default=8, type=int, help="")
    parser.add_argument("--machine_feat_dim", default=6, type=int, help="")
    parser.add_argument("--valid_batch_size", default=100, type=int, help="")

    parser.add_argument(
        "--no_batch", action="store_true", default=False, help="Whether to forbid machine batch abaility"
    )
    parser.add_argument("--no_drop", action="store_true", default=False, help="Whether to forbid drop")
    # ========== strict_batch — strict batch mode: require equal processing times within each batch ==========
    parser.add_argument(
        "--strict_batch",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to require exact equal processing time for batching (True/False, default: True)",
    )
    # ========== End strict_batch ==========

    # ----------------------Model params--------------------------------
    parser.add_argument(
        "--d", default=32, type=int, help="Dimension of the machine and operation nodes embedding"
    )
    parser.add_argument("--residual", action="store_true", help="")
    parser.add_argument("--hidden_size_oper", default=64, type=int, help="")
    parser.add_argument("--num_heads", default=4, type=int, help="")
    parser.add_argument("--L", default=5, type=int, help="")
    parser.add_argument("--dropout", default=0.1, type=float, help="")
    parser.add_argument("--n_latent_actor", default=64, type=int, help="")
    parser.add_argument("--n_latent_critic", default=64, type=int, help="")
    parser.add_argument("--n_hidden_actor", default=5, type=int, help="")
    parser.add_argument("--n_hidden_critic", default=5, type=int, help="")
    parser.add_argument("--action_dim", default=1, type=int, help="")

    # ----------------------Train params--------------------------------
    parser.add_argument("--epoch", default=10000, type=int, help="Num of iteration")
    parser.add_argument("--lr", default=0.0002, type=float, help="")
    parser.add_argument("--betas", default=[0.9, 0.999], type=list, help="")
    parser.add_argument("--gamma", default=1.0, type=float, help="")
    parser.add_argument("--K_epochs", default=3, type=int, help="")
    parser.add_argument("--eps_clip", default=0.2, type=float, help="")
    parser.add_argument("--A_coeff", default=1.0, type=float, help="")
    parser.add_argument("--vf_coeff", default=0.5, type=float, help="")
    parser.add_argument("--entropy_coeff", default=0.01, type=float, help="")
    parser.add_argument("--save_timestep", default=10, type=int, help="")
    parser.add_argument("--update_timestep", default=1, type=int, help="")
    parser.add_argument("--minibatch_size", default=512, type=int, help="")
    parser.add_argument("--parallel_iter", default=50, type=int, help="")
    # ========== LR scheduler parameters ==========
    parser.add_argument("--lr_step_size", default=4000, type=int, help="StepLR step size (epochs per decay)")
    parser.add_argument("--lr_gamma", default=0.5, type=float, help="StepLR decay factor")
    # ========== End LR scheduler parameters ==========

    # ----------------------Test params--------------------------------
    parser.add_argument("--test_iter", default=10, type=int, help="Num of iteration")  # Default number of parallel RL inference instances is set to 10.

    return parser


# When the scheduling service and RL share a command line, allow_unknown can ignore the other side's arguments.
def parse_arguments(argv=None, allow_unknown=False):
    parser = build_argument_parser()
    if allow_unknown:
        args, _ = parser.parse_known_args(argv)
    else:
        args = parser.parse_args(argv)

    return args, set_torch_params(device_id=args.cuda)
