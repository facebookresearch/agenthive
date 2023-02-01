import os

from torchrl.record import VideoRecorder

os.environ["sim_backend"] = "MUJOCO"

import gc
import os
from copy import deepcopy

import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
from omegaconf import DictConfig
from torchvision.models import ResNet50_Weights
from rlhive.rl_envs import RoboHiveEnv

from sac_loss import SACLoss

# from torchrl.objectives import SACLoss
from tensordict import TensorDict

from torch import nn, optim
from torchrl.data import TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage

# from torchrl.envs import SerialEnv as ParallelEnv, R3MTransform, SelectTransform, TransformedEnv
from torchrl.envs import (
    CatTensors,
    ParallelEnv,
    R3MTransform,
    SelectTransform,
    TransformedEnv,
)
from torchrl.envs.transforms import Compose, FlattenObservation, RewardScaling, Resize, ToTensorImage
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import MLP, NormalParamWrapper, SafeModule
from torchrl.modules.distributions import TanhNormal

from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.objectives import SoftUpdate
from torchrl.record.loggers import WandbLogger
from torchrl.trainers import Recorder

# ===========================================================================================
# Env constructor
# ---------------
# - Use the RoboHiveEnv class to wrap robohive envs in torchrl's GymWrapper
# - Add transforms immediately after that:
#   - SelectTransform: selects the relevant kesy from our output
#   - R3MTransform
#   - FlattenObservation: The images delivered by robohive have a singleton dim to start with, we need to flatten that
#   - RewardScaling
#
# One can also possibly use ObservationNorm.
#
# TIPS:
# - For faster execution, you should follow this abstract scheme, where we reduce the data
#   to be passed from worker to worker to a minimum, we apply R3M to a batch and append the
#   rest of the transforms afterward:
#
#  >>> env = TransformedEnv(
#  ...     ParallelEnv(N, lambda: TransformedEnv(RoboHiveEnv(...), SelectTransform(...))),
#  ...     Compose(
#  ...         R3MTransform(...),
#  ...         FlattenObservation(...),
#  ...         *other_transforms,
#  ...     ))
#

def is_visual_env(task):
    return task.startswith("visual_")

def evaluate_success(env_success_fn, td_record: dict, eval_traj: int):
    td_record["success"] = td_record["success"].reshape((eval_traj, -1))
    paths = []
    for traj, solved_traj in zip(range(eval_traj), td_record["success"]):
        path = {"env_infos": {"solved": solved_traj.data.cpu().numpy()}}
        paths.append(path)
    success_percentage = env_success_fn(paths)
    return success_percentage

def make_env(num_envs, task, visual_transform, reward_scaling, device):
    assert visual_transform in ("rrl", "r3m", "flatten", "state")
    if num_envs > 1:
        base_env = ParallelEnv(num_envs, lambda: RoboHiveEnv(task, device=device))
    else:
        base_env = RoboHiveEnv(task, device=device)
    env = make_transformed_env(
        env=base_env, reward_scaling=reward_scaling, visual_transform=visual_transform
    )

    return env


def make_transformed_env(
    env,
    reward_scaling=5.0,
    visual_transform="r3m",
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """
    if visual_transform != "state":
        env = TransformedEnv(
            env,
            SelectTransform("solved", "pixels", "observation", "rwd_dense", "rwd_sparse"),
        )
        if visual_transform == "r3m":
            vec_keys = ["r3m_vec"]
            selected_keys = ["observation", "r3m_vec"]
            env.append_transform(
                Compose(
                    R3MTransform("resnet50", in_keys=["pixels"], download=True),
                    FlattenObservation(-2, -1, in_keys=vec_keys),
                )
            )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
        elif visual_transform == "rrl":
            vec_keys = ["r3m_vec"]
            selected_keys = ["observation", "r3m_vec"]
            env.append_transform(
                Compose(
                    R3MTransform("resnet50", in_keys=["pixels"], download=ResNet50_Weights.IMAGENET1K_V2),
                    FlattenObservation(-2, -1, in_keys=vec_keys),
                )
            )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
        elif visual_transform == "flatten":
            vec_keys = ["pixels"]
            out_keys = ["pixels"]
            selected_keys = ["observation", "pixels"]
            env.append_transform(
                Compose(
                    ToTensorImage(),
                    Resize(64, 64, in_keys=vec_keys, out_keys=out_keys), ## TODO: Why is resize not working?
                    FlattenObservation(-4, -1, in_keys=out_keys),
                )
            )
        else:
            raise NotImplementedError
    else:
        env = TransformedEnv(env, SelectTransform("solved", "observation", "rwd_dense", "rwd_sparse"))
        selected_keys = ["observation"]

    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))
    return env


# ===========================================================================================
# Making a recorder
# -----------------
#
# A `Recorder` is a dedicated torchrl class that will run the policy in the test env
# once every X steps (eg X=1M).
#


def make_recorder(
    task: str,
    #frame_skip: int,
    record_interval: int,
    actor_model_explore: object,
    eval_traj: int,
    env_configs: dict,
    wandb_logger: WandbLogger,
):
    test_env = make_env(num_envs=1, task=task, **env_configs)
    if is_visual_env(task):## TODO(Rutav): Change this behavior. Record only when using visual env
        test_env.insert_transform(
            0, VideoRecorder(wandb_logger, "test", in_keys=["pixels"])
        )
    recorder_obj = Recorder(
        record_frames=eval_traj * test_env.horizon,
        #frame_skip=frame_skip, ## To maintain consistency and using default env frame_skip values
        frame_skip=1, ## To maintain consistency and using default env frame_skip values
        policy_exploration=actor_model_explore,
        recorder=test_env,
        exploration_mode="mean",
        record_interval=record_interval,
        log_keys=["reward", "solved",  "rwd_dense", "rwd_sparse"],
        out_keys={"reward": "r_evaluation", "solved": "success", "rwd_dense": "rwd_dense", "rwd_sparse": "rwd_sparse"},
    )
    return recorder_obj


# ===========================================================================================
# Relplay buffers
# ---------------
#
# TorchRL also provides prioritized RBs if needed.
#


def make_replay_buffer(
    buffer_size: int,
    buffer_scratch_dir: str,
    device: torch.device,
    make_replay_buffer: int = 3,
):
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


# ===========================================================================================
# Dataloader
# ----------
#
# This is a simplified version of the dataloder
#


@torch.no_grad()
@set_exploration_mode("random")
def dataloader(
    total_frames, fpb, train_env, actor, actor_collection, device_collection
):
    params = TensorDict(
        {k: v for k, v in actor.named_parameters()}, batch_size=[]
    ).unflatten_keys(".")
    params_collection = TensorDict(
        {k: v for k, v in actor_collection.named_parameters()}, batch_size=[]
    ).unflatten_keys(".")
    _prev = None

    collected_frames = 0
    while collected_frames < total_frames:
        params_collection.update_(params)
        batch = TensorDict(
            {}, batch_size=[fpb, *train_env.batch_size], device=device_collection
        )
        for t in range(fpb):
            if _prev is None:
                _prev = train_env.reset()
            _reset = _prev["_reset"] = _prev["done"].clone().squeeze(-1)
            if _reset.any():
                _prev = train_env.reset(_prev)
            _new = train_env.step(actor_collection(_prev))
            batch[t] = _new
            _prev = step_mdp(_new, exclude_done=False)
        collected_frames += batch.numel()
        yield batch


@hydra.main(config_name="sac.yaml", config_path="config")
def main(args: DictConfig):
    assert ((args.visual_transform == "state")^is_visual_env(args.task)), "Please use visual_transform=state if using state environment; else use visual_transform=r3m,rrl"
    # customize device at will
    device = args.device
    device_collection = args.device_collection
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Environment
    env_configs = {
        "reward_scaling": args.reward_scaling,
        "visual_transform": args.visual_transform,
        "device": args.device,
    }
    train_env = make_env(num_envs=args.num_envs, task=args.task, **env_configs).to(
        device_collection
    )

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
        "tanh_loc": True,
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
    proof_env = make_env(num_envs=1, task=args.task, **env_configs)
    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    proof_env.close()

    actor_collection = deepcopy(actor).to(device_collection)

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

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        buffer_size=args.buffer_size,
        buffer_scratch_dir=args.buffer_scratch_dir,
        device=device,
    )

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rewards = []
    rewards_eval = []
    success_percentage_hist = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    episodes = 0
    optim_steps = 0
    pbar = tqdm.tqdm(total=args.total_frames)
    r0 = None
    loss = None

    total_frames = args.total_frames
    frames_per_batch = args.frames_per_batch

    logger = WandbLogger(
        exp_name=args.task,
        project="SAC_TorchRL",
        name=args.exp_name,
        config=args,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )

    # Trajectory recorder for evaluation
    recorder = make_recorder(
        task=args.task,
        #frame_skip=args.frame_skip,
        record_interval=args.record_interval,
        actor_model_explore=actor_model_explore,
        eval_traj=args.eval_traj,
        env_configs=env_configs,
        wandb_logger=logger,
    )

    for i, batch in enumerate(
        dataloader(
            total_frames,
            frames_per_batch,
            train_env,
            actor,
            actor_collection,
            device_collection,
        )
    ):
        if r0 is None:
            r0 = batch["reward"].sum(-1).mean().item()
        pbar.update(batch.numel())

        # extend the replay buffer with the new data
        batch = batch.view(-1)
        current_frames = batch.numel()
        collected_frames += current_frames
        episodes += batch["done"].sum()
        replay_buffer.extend(batch.cpu())

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
                optim_steps += 1
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(args.batch_size).clone()

                loss_td = loss_module(sampled_tensordict)
                ## Not returned in explicit forward loss
                #print(f'value: {loss_td["state_action_value_actor"].mean():4.4f}')
                #print(f'log_prob: {loss_td["action_log_prob_actor"].mean():4.4f}')
                #print(f'next.state_value: {loss_td["state_value"].mean():4.4f}')

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                loss = actor_loss + q_loss + alpha_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

        rewards.append((i, batch["reward"].mean().item()))
        logger.log_scalar("train_reward", rewards[-1][1], step=collected_frames)
        logger.log_scalar("optim_steps", optim_steps, step=collected_frames)
        logger.log_scalar("episodes", episodes, step=collected_frames)

        if loss is not None:
            logger.log_scalar(
                "total_loss", np.mean(total_losses), step=collected_frames
            )
            logger.log_scalar(
                "actor_loss", np.mean(actor_losses), step=collected_frames
            )
            logger.log_scalar("q_loss", np.mean(q_losses), step=collected_frames)
            logger.log_scalar(
                "alpha_loss", np.mean(alpha_losses), step=collected_frames
            )
            logger.log_scalar("alpha", np.mean(alphas), step=collected_frames)
            logger.log_scalar("entropy", np.mean(entropies), step=collected_frames)
        if i % args.eval_interval == 0:
            td_record = recorder(None)
            if td_record is not None:
                rewards_eval.append(
                    (
                        i,
                        td_record["total_r_evaluation"]
                        / 1,  # divide by number of eval worker
                    )
                )
                logger.log_scalar(
                    "test_reward", rewards_eval[-1][1], step=collected_frames
                )
                logger.log_scalar(
                    "reward_sparse", td_record["rwd_sparse"].sum()/args.eval_traj, step=collected_frames
                )
                logger.log_scalar(
                    "reward_dense", td_record["rwd_dense"].sum()/args.eval_traj, step=collected_frames
                )
                success_percentage = evaluate_success(
                    env_success_fn=train_env.evaluate_success,
                    td_record=td_record,
                    eval_traj=args.eval_traj,
                )
                success_percentage_hist.append(success_percentage)
                #solved = float(td_record["success"].any())
                logger.log_scalar(
                    "success_rate", success_percentage, step=collected_frames
                )

        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), test reward: {rewards_eval[-1][1]: 4.4f}, Success: {success_percentage_hist[-1]}"
            )
        del batch
        gc.collect()


if __name__ == "__main__":
    main()
