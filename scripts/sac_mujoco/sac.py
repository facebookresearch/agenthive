# Make all the necessary imports for training


import os
import gc
import argparse
import yaml
from typing import Optional

import numpy as np
import torch
import torch.cuda
import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
#from torchrl.objectives import SACLoss
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
from torchrl.envs import EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv, FlattenObservation, Compose
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, NormalParamWrapper, ProbabilisticActor, SafeModule
from torchrl.modules.distributions import TanhNormal

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator

from torchrl.objectives import SoftUpdate
from torchrl.trainers import Recorder

from rlhive.rl_envs import RoboHiveEnv
from torchrl.envs import ParallelEnv, TransformedEnv, R3MTransform

os.environ['WANDB_MODE'] = 'offline' ## offline sync. TODO: Remove this behavior

def make_env():
    """
    Create a base env
    """
    env_args = (args.task,)
    env_library = GymEnv

    env_kwargs = {
        "device": device,
        "frame_skip": args.frame_skip,
        "from_pixels": args.from_pixels,
        "pixels_only": args.from_pixels,
    }
    env = env_library(*env_args, **env_kwargs)

    env_name = args.task
    base_env = RoboHiveEnv(env_name, device=device)
    env = TransformedEnv(base_env, R3MTransform('resnet50', in_keys=["pixels"], download=True))
    assert env.device == device

    return env


def make_transformed_env(
    env,
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """
    env = TransformedEnv(env, Compose(R3MTransform('resnet50', in_keys=["pixels"], download=True), FlattenObservation(-2, -1, in_keys=["r3m_vec"]))) # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    env.append_transform(RewardScaling(loc=0.0, scale=5.0))
    selected_keys = list(env.observation_spec.keys())
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


def parallel_env_constructor(
    stats,
    num_worker=1,
    **env_kwargs,
):
    if num_worker == 1:
        env_creator = EnvCreator(
            lambda: make_transformed_env(make_env(), stats, **env_kwargs)
        )
        return env_creator

    parallel_env = ParallelEnv(
        num_workers=num_worker,
        create_env_fn=EnvCreator(lambda: make_env()),
        create_env_kwargs=None,
        pin_memory=False,
    )
    env = make_transformed_env(parallel_env, stats, **env_kwargs)
    return env


def get_stats_random_rollout(proof_environment, key: Optional[str] = None):
    print("computing state stats")
    n = 0
    td_stats = []
    while n < args.init_env_steps:
        _td_stats = proof_environment.rollout(max_steps=args.init_env_steps)
        n += _td_stats.numel()
        _td_stats_select = _td_stats.to_tensordict().select(key).cpu()
        if not len(list(_td_stats_select.keys())):
            raise RuntimeError(
                f"key {key} not found in tensordict with keys {list(_td_stats.keys())}"
            )
        td_stats.append(_td_stats_select)
        del _td_stats, _td_stats_select
    td_stats = torch.cat(td_stats, 0)

    m = td_stats.get(key).mean(dim=0)
    s = td_stats.get(key).std(dim=0)
    m[s == 0] = 0.0
    s[s == 0] = 1.0

    print(
        f"stats computed for {td_stats.numel()} steps. Got: \n"
        f"loc = {m}, \n"
        f"scale: {s}"
    )
    if not torch.isfinite(m).all():
        raise RuntimeError("non-finite values found in mean")
    if not torch.isfinite(s).all():
        raise RuntimeError("non-finite values found in sd")
    stats = {"loc": m, "scale": s}
    return stats


def get_env_stats():
    """
    Gets the stats of an environment
    """
    proof_env = make_transformed_env(make_env(), None)
    proof_env.set_seed(args.seed)
    stats = get_stats_random_rollout(
        proof_env,
        key="observation_vector",
    )
    # make sure proof_env is closed
    proof_env.close()
    return stats


def make_recorder(
                    task: str,
                    frame_skip: int,
                    record_interval: int,
                    actor_model_explore: object,
                    device: torch.device
                 ):
    _base_env = RoboHiveEnv(task, device=device) # TODO: Move this to make_env() function
    test_env = make_transformed_env(_base_env)
    recorder_obj = Recorder(
        record_frames=1000,
        frame_skip=frame_skip,
        policy_exploration=actor_model_explore,
        recorder=test_env,
        exploration_mode="mean",
        record_interval=record_interval,
    )
    return recorder_obj


def make_replay_buffer(
                        prb: bool,
                        buffer_size: int,
                        buffer_scratch_dir: str,
                        device: torch.device,
                        make_replay_buffer: int = 3
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
    base_env = RoboHiveEnv(args.task, device=args.device) # TODO: Move this to make_env() function
    train_env = make_transformed_env(base_env)

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
    _base_env = RoboHiveEnv(args.task, device=args.device) # TODO: move this to make_env
    proof_env = make_transformed_env(_base_env)
    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_env.reset()
        td = td.to(device)
        #print(td[in_keys[0]].shape)
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
                                device=device
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
            if td_record is not None:
                rewards_eval.append(
                    (
                        i,
                        td_record["total_r_evaluation"]
                        / 1,  # divide by number of eval worker
                    )
                )
                wandb.log({"test_reward": rewards_eval[-1][1]})
            if len(rewards_eval):
                pbar.set_description(
                    f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), test reward: {rewards_eval[-1][1]: 4.4f}"
                )
            del tensordict
            gc.collect()

        collector.shutdown()

if __name__ == "__main__":
    main()
