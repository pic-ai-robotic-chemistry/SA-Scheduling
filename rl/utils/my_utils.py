import json


def group_args_into_dict(args, device):

    return {
        "env_params": {
            "num_jobs": args.num_jobs,
            "num_machines": args.num_machines,
            "reward_punish": args.reward_punish,
            "batch_size": args.batch_size,
            "oper_feat_dim": args.oper_feat_dim,
            "machine_feat_dim": args.machine_feat_dim,
            "device": device,
            "no_batch": args.no_batch,
            "no_drop": args.no_drop,
            # ========== strict_batch ==========
            "strict_batch": args.strict_batch,
            # ========== End strict_batch ==========
        },
        "env_valid_params": {
            "num_jobs": args.num_jobs,
            "num_machines": args.num_machines,
            "reward_punish": args.reward_punish,
            "batch_size": args.valid_batch_size,
            "oper_feat_dim": args.oper_feat_dim,
            "machine_feat_dim": args.machine_feat_dim,
            "device": device,
            "no_batch": args.no_batch,
            "no_drop": args.no_drop,
            # ========== strict_batch ==========
            "strict_batch": args.strict_batch,
            # ========== End strict_batch ==========
        },
        "model_params": {
            "device": device,
            "d": args.d,
            "actor_in_dim": 4 * args.d,
            "critic_in_dim": 2 * args.d,
            "in_size_machine": args.machine_feat_dim,
            "in_size_oper": args.oper_feat_dim,
            "hidden_size_oper": args.hidden_size_oper,
            "num_heads": args.num_heads,
            "L": args.L,
            "residual": args.residual,
            "dropout": args.dropout,
            "n_latent_actor": args.n_latent_actor,
            "n_latent_critic": args.n_latent_critic,
            "n_hidden_actor": args.n_hidden_actor,
            "n_hidden_critic": args.n_hidden_critic,
            "action_dim": args.action_dim,
        },
        "train_params": {
            "lr": args.lr,
            "betas": args.betas,
            "gamma": args.gamma,
            "K_epochs": args.K_epochs,
            "eps_clip": args.eps_clip,
            "A_coeff": args.A_coeff,
            "vf_coeff": args.vf_coeff,
            "entropy_coeff": args.entropy_coeff,
            "save_timestep": args.save_timestep,
            "update_timestep": args.update_timestep,
            "minibatch_size": args.minibatch_size,
            "parallel_iter": args.parallel_iter,
            # ========== LR scheduler parameters ==========
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            # ========== End LR scheduler parameters ==========
        },
    }
