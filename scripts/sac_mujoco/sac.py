# Make all the necessary imports for training


import argparse
import gc
import os
from typing import Optional

import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from rlhive.rl_envs import RoboHiveEnv
from rlhive.sim_algos.helpers.rrl_transform import RRLTransform

# from torchrl.objectives import SACLoss
from sac_loss import SACLoss

from torch import nn, optim
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import RandomPolicy
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    Compose,
    FlattenObservation,
    RewardScaling,
    TransformedEnv,
)
from torchrl.envs import ParallelEnv, R3MTransform, SelectTransform, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, NormalParamWrapper, ProbabilisticActor, SafeModule
from torchrl.modules.distributions import TanhNormal

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import SoftUpdate
from torchrl.trainers import Recorder

os.environ["WANDB_MODE"] = "offline"  ## offline sync. TODO: Remove this behavior


def make_env(task, visual_transform, reward_scaling, device):
    assert visual_transform in ("rrl", "r3m")
    base_env = RoboHiveEnv(task, device=device)
    env = make_transformed_env(
        env=base_env, reward_scaling=reward_scaling, visual_transform=visual_transform
    )
    print(env)

    return env


def make_transformed_env(
    env,
    reward_scaling=5.0,
    visual_transform="r3m",
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """
    env = TransformedEnv(env, SelectTransform("solved", "pixels", "observation"))
    if visual_transform == "rrl":
        vec_keys = ["rrl_vec"]
        selected_keys = ["observation", "rrl_vec"]
        env.append_transform(
            Compose(
                RRLTransform("resnet50", in_keys=["pixels"], download=True),
                FlattenObservation(-2, -1, in_keys=vec_keys),
            )
        )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    elif visual_transform == "r3m":
        vec_keys = ["r3m_vec"]
        selected_keys = ["observation", "r3m_vec"]
        env.append_transform(
            Compose(
                R3MTransform("resnet50", in_keys=["pixels"], download=True),
                FlattenObservation(-2, -1, in_keys=vec_keys),
            )
        )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    else:
        raise NotImplementedError
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    #  we normalize the states
    if stats is None:
        _stats = {"loc": 0.0, "scale": 1.0}
    else:
        _stats = stats
    env.append_transform(
        ObservationNorm(**_stats, in_keys=[out_key], standard_normal=True)
    )
    env.append_transform(DoubleToFloat(in_keys=[out_key], in_keys_inv=[]))
    return env


def make_recorder(
    task: str,
    frame_skip: int,
    record_interval: int,
    actor_model_explore: object,
    eval_traj: int,
    env_configs: dict,
):
    test_env = make_env(task=task, **env_configs)
    recorder_obj = Recorder(
        record_frames=eval_traj * test_env.horizon,
        frame_skip=frame_skip,
        policy_exploration=actor_model_explore,
        recorder=test_env,
        exploration_mode="mean",
        record_interval=record_interval,
        log_keys=["reward", "solved"],
        out_keys={"reward": "r_evaluation", "solved": "success"},
    )
    return recorder_obj


def make_replay_buffer(
    prb: bool,
    buffer_size: int,
    buffer_scratch_dir: str,
    device: torch.device,
    make_replay_buffer: int = 3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=make_replay_buffer,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=make_replay_buffer,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
        )
    return replay_buffer


def evaluate_success(env_success_fn, td_record: dict, eval_traj: int):
    td_record["success"] = td_record["success"].reshape((eval_traj, -1))
    paths = []
    for traj, solved_traj in zip(range(eval_traj), td_record["success"]):
        path = {"env_infos": {"solved": solved_traj.data.cpu().numpy()}}
        paths.append(path)
    success_percentage = env_success_fn(paths)
    return success_percentage


@hydra.main(config_name="sac.yaml", config_path="config")
def main(args: DictConfig):
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and args.device == "cuda:0"
        else torch.device("cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Environment
    env_configs = {
        "reward_scaling": args.reward_scaling,
        "visual_transform": args.visual_transform,
        "device": args.device,
    }
    train_env = make_env(task=args.task, **env_configs)

    # Create Agent

    # Define Actor Network
    in_keys = ["observation_vector"]
    action_spec = train_env.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": nn.ReLU,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": False,
    }
    actor_net = NormalParamWrapper(
        actor_net,
        scale_mapping=f"biased_softplus_{1.0}",
        scale_lb=0.1,
    )
    in_keys_actor = in_keys
    actor_module = SafeModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # add forward pass for initialization with proof env
    proof_env = make_env(task=args.task, **env_configs)
    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    proof_env.close()

    actor_model_explore = model[0]

    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        gamma=args.gamma,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, args.target_update_polyak)

    # Make Off-Policy Collector

    collector = MultiaSyncDataCollector(
        create_env_fn=[train_env],
        policy=actor_model_explore,
        total_frames=args.total_frames,
        max_frames_per_traj=args.frames_per_batch,
        frames_per_batch=args.env_per_collector * args.frames_per_batch,
        init_random_frames=args.init_random_frames,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=True,
        devices=[device],  # device for execution
        passing_devices=[device],  # device where data will be stored and passed
        seed=None,
        pin_memory=False,
        update_at_each_batch=False,
        exploration_mode="random",
    )
    collector.set_seed(args.seed)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        prb=args.prb,
        buffer_size=args.buffer_size,
        buffer_scratch_dir=args.buffer_scratch_dir,
        device=device,
    )

    # Trajectory recorder for evaluation
    recorder = make_recorder(
        task=args.task,
        frame_skip=args.frame_skip,
        record_interval=args.record_interval,
        actor_model_explore=actor_model_explore,
        eval_traj=args.eval_traj,
        env_configs=env_configs,
    )

    # Optimizers
    params = list(loss_module.parameters()) + list([loss_module.log_alpha])
    optimizer_actor = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    episodes = 0
    pbar = tqdm.tqdm(total=args.total_frames)
    r0 = None
    loss = None

    with wandb.init(project="SAC_TorchRL", name=args.exp_name, config=args):
        for i, tensordict in enumerate(collector):

            # update weights of the inference policy
            collector.update_policy_weights_()

            if r0 is None:
                r0 = tensordict["reward"].sum(-1).mean().item()
            pbar.update(tensordict.numel())

            # extend the replay buffer with the new data
            if "mask" in tensordict.keys():
                # if multi-step, a mask is present to help filter padded values
                current_frames = tensordict["mask"].sum()
                tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
            else:
                tensordict = tensordict.view(-1)
                current_frames = tensordict.numel()
            collected_frames += current_frames
            episodes += args.env_per_collector
            replay_buffer.extend(tensordict.cpu())

            # optimization steps
            if collected_frames >= args.init_random_frames:
                (
                    total_losses,
                    actor_losses,
                    q_losses,
                    alpha_losses,
                    alphas,
                    entropies,
                ) = ([], [], [], [], [], [])
                for _ in range(
                    args.env_per_collector * args.frames_per_batch * args.utd_ratio
                ):
                    # sample from replay buffer
                    sampled_tensordict = replay_buffer.sample(args.batch_size).clone()

                    loss_td = loss_module(sampled_tensordict)

                    actor_loss = loss_td["loss_actor"]
                    q_loss = loss_td["loss_qvalue"]
                    alpha_loss = loss_td["loss_alpha"]

                    loss = actor_loss + q_loss + alpha_loss
                    optimizer_actor.zero_grad()
                    loss.backward()
                    optimizer_actor.step()

                    # update qnet_target params
                    target_net_updater.step()

                    # update priority
                    if args.prb:
                        replay_buffer.update_priority(sampled_tensordict)

                    total_losses.append(loss.item())
                    actor_losses.append(actor_loss.item())
                    q_losses.append(q_loss.item())
                    alpha_losses.append(alpha_loss.item())
                    alphas.append(loss_td["alpha"].item())
                    entropies.append(loss_td["entropy"].item())

            rewards.append(
                (i, tensordict["reward"].sum().item() / args.env_per_collector)
            )
            wandb.log(
                {
                    "train_reward": rewards[-1][1],
                    "collected_frames": collected_frames,
                    "episodes": episodes,
                }
            )
            if loss is not None:
                wandb.log(
                    {
                        "total_loss": np.mean(total_losses),
                        "actor_loss": np.mean(actor_losses),
                        "q_loss": np.mean(q_losses),
                        "alpha_loss": np.mean(alpha_losses),
                        "alpha": np.mean(alphas),
                        "entropy": np.mean(entropies),
                    }
                )
            td_record = recorder(None)
            success_percentage = evaluate_success(
                env_success_fn=train_env.evaluate_success,
                td_record=td_record,
                eval_traj=args.eval_traj,
            )
            if td_record is not None:
                rewards_eval.append(
                    (
                        i,
                        td_record["total_r_evaluation"]
                        / 1,  # divide by number of eval worker
                    )
                )
                wandb.log({"test_reward": rewards_eval[-1][1]})
                wandb.log({"success": success_percentage})
            if len(rewards_eval):
                pbar.set_description(
                    f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), test reward: {rewards_eval[-1][1]: 4.4f}"
                )
            del tensordict
            gc.collect()

        collector.shutdown()


if __name__ == "__main__":
    main()
