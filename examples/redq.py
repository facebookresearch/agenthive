# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

from omegaconf import DictConfig

os.environ["sim_backend"] = "MUJOCO"


def main(args: DictConfig):

    import numpy as np
    import torch.cuda
    import tqdm
    from rlhive.rl_envs import RoboHiveEnv

    from tensordict import TensorDict

    from torch import nn, optim
    from torchrl.collectors import MultiaSyncDataCollector
    from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

    from torchrl.data.replay_buffers.storages import LazyMemmapStorage

    # from torchrl.envs import SerialEnv as ParallelEnv, R3MTransform, SelectTransform, TransformedEnv
    from torchrl.envs import (
        CatTensors,
        EnvCreator,
        ParallelEnv,
        R3MTransform,
        SelectTransform,
        TransformedEnv,
    )
    from torchrl.envs.transforms import Compose, FlattenObservation, RewardScaling
    from torchrl.envs.utils import set_exploration_mode, step_mdp
    from torchrl.modules import MLP, NormalParamWrapper, SafeModule
    from torchrl.modules.distributions import TanhNormal

    from torchrl.modules.tensordict_module.actors import (
        ProbabilisticActor,
        ValueOperator,
    )
    from torchrl.objectives import SoftUpdate

    from torchrl.objectives.deprecated import REDQLoss_deprecated as REDQLoss
    from torchrl.record import VideoRecorder
    from torchrl.record.loggers.wandb import WandbLogger
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

    def traj_is_solved(done, solved):
        solved = solved.view_as(done)
        done_cumsum = done.cumsum(-2)
        count = 0
        _i = 0
        for _i, u in enumerate(done_cumsum.unique()):
            is_solved = solved[done_cumsum == u].any()
            count += is_solved
        return count / (_i + 1)

    def traj_total_reward(done, reward):
        reward = reward.view_as(done)
        done_cumsum = done.cumsum(-2)
        count = 0
        _i = 0
        for _i, u in enumerate(done_cumsum.unique()):
            count += reward[done_cumsum == u].sum()
        return count / (_i + 1)

    def make_env(num_envs, task, visual_transform, reward_scaling, device):
        if num_envs > 1:
            base_env = ParallelEnv(
                num_envs, EnvCreator(lambda: RoboHiveEnv(task, device=device))
            )
        else:
            base_env = RoboHiveEnv(task, device=device)
        env = make_transformed_env(
            env=base_env,
            reward_scaling=reward_scaling,
            visual_transform=visual_transform,
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
        env = TransformedEnv(
            env,
            SelectTransform(
                "solved", "pixels", "observation", "rwd_dense", "rwd_sparse"
            ),
        )
        if visual_transform == "r3m":
            vec_keys = ["r3m_vec"]
            selected_keys = ["observation", "r3m_vec"]
            env.append_transform(
                Compose(
                    R3MTransform("resnet50", in_keys=["pixels"], download=True).eval(),
                    FlattenObservation(-2, -1, in_keys=vec_keys),
                )
            )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
        elif visual_transform == "rrl":
            vec_keys = ["r3m_vec"]
            selected_keys = ["observation", "r3m_vec"]
            env.append_transform(
                Compose(
                    R3MTransform(
                        "resnet50", in_keys=["pixels"], download="IMAGENET1K_V2"
                    ).eval(),
                    FlattenObservation(-2, -1, in_keys=vec_keys),
                )
            )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
        elif not visual_transform:
            selected_keys = ["observation"]
        else:
            raise NotImplementedError(visual_transform)
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
        frame_skip: int,
        record_interval: int,
        actor_model_explore: object,
        eval_traj: int,
        env_configs: dict,
        wandb_logger: WandbLogger,
        num_envs: int,
    ):
        test_env = make_env(num_envs=num_envs, task=task, **env_configs)
        if "visual" in task:
            test_env.insert_transform(
                0, VideoRecorder(wandb_logger, "test", in_keys=["pixels"])
            )
        test_env.reset()
        recorder_obj = Recorder(
            record_frames=eval_traj * test_env.horizon,
            frame_skip=frame_skip,
            policy_exploration=actor_model_explore,
            recorder=test_env,
            exploration_mode="mean",
            record_interval=record_interval,
            log_keys=["reward", "solved", "done", "rwd_sparse"],
            out_keys={
                "reward": "r_evaluation",
                "solved": "success",
                "done": "done",
                "rwd_sparse": "rwd_sparse",
            },
        )
        return recorder_obj

    # ===========================================================================================
    # Relplay buffers
    # ---------------
    #
    # TorchRL also provides prioritized RBs if needed.
    #

    def make_replay_buffer(
        prb: bool,
        buffer_size: int,
        buffer_scratch_dir: str,
        device: torch.device,
        prefetch: int = 10,
    ):
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=buffer_scratch_dir,
                    device=device,
                ),
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
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

    # customize device at will
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create Environment
    env_configs = {
        "reward_scaling": args.reward_scaling,
        "visual_transform": args.visual_transform,
        "device": "cpu",
    }
    train_env = make_env(num_envs=args.env_per_collector, task=args.task, **env_configs)
    # add forward pass for initialization with proof env
    proof_env = make_env(num_envs=1, task=args.task, **env_configs)

    # Create Agent
    # Define Actor Network
    in_keys = ["observation_vector"]
    action_spec = proof_env.action_spec
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
        return_log_prob=True,
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

    model = actor, qvalue = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    proof_env.close()

    actor_model_explore = model[0]

    # Create REDQ loss
    loss_module = REDQLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        gamma=args.gamma,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, args.target_update_polyak)

    # Make Replay Buffer
    replay_buffer = make_replay_buffer(
        prb=args.prb,
        buffer_size=args.buffer_size,
        buffer_scratch_dir=args.buffer_scratch_dir,
        device="cpu",
    )

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0
    episodes = 0
    optim_steps = 0
    pbar = tqdm.tqdm(total=args.total_frames)
    r0 = None
    loss = None

    logger = WandbLogger(
        exp_name=args.task,
        project=args.wandb_project,
        name=args.exp_name,
        config=args,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )

    # Trajectory recorder for evaluation
    recorder = make_recorder(
        task=args.task,
        frame_skip=args.frame_skip,
        record_interval=args.record_interval,
        actor_model_explore=actor_model_explore,
        eval_traj=args.eval_traj,
        env_configs=env_configs,
        wandb_logger=logger,
        num_envs=args.num_record_envs,
    )

    collector_device = args.device_collection
    if isinstance(collector_device, str):
        collector_device = [collector_device]
    collector = MultiaSyncDataCollector(
        create_env_fn=[train_env for _ in collector_device],
        policy=actor_model_explore,
        total_frames=args.total_frames,
        max_frames_per_traj=args.frames_per_batch,
        frames_per_batch=args.frames_per_batch,
        init_random_frames=args.init_random_frames,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=False,
        devices=collector_device,  # device for execution
        passing_devices=collector_device,  # device where data will be stored and passed
        seed=args.seed,
        pin_memory=False,
        update_at_each_batch=False,
        exploration_mode="random",
    )

    for i, batch in enumerate(collector):
        collector.update_policy_weights_()
        if r0 is None:
            r0 = batch["reward"].sum(-1).mean().item()
        pbar.update(batch.numel())

        # extend the replay buffer with the new data
        batch = batch.cpu().view(-1)
        current_frames = batch.numel()
        collected_frames += current_frames
        episodes += batch["done"].sum()
        replay_buffer.extend(batch)

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
                max(1, args.frames_per_batch * args.utd_ratio // args.batch_size)
            ):
                optim_steps += 1
                # sample from replay buffer
                sampled_tensordict = (
                    replay_buffer.sample(args.batch_size).clone().to(device)
                )

                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                loss = actor_loss + q_loss + alpha_loss
                optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
                optimizer.step()
                # update qnet_target params
                target_net_updater.step()

                # update priority
                if args.prb:
                    replay_buffer.update_tensordict_priority(sampled_tensordict)

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
            logger.log_scalar("grad_norm", gn, step=collected_frames)
        td_record = recorder(None)
        if td_record is not None:
            rewards_eval.append(
                (
                    i,
                    td_record["r_evaluation"]
                    / recorder.recorder.batch_size.numel(),  # divide by number of eval worker
                )
            )
            logger.log_scalar("test_reward", rewards_eval[-1][1], step=collected_frames)
            solved = traj_is_solved(td_record["done"], td_record["success"])
            logger.log_scalar("success", solved, step=collected_frames)
            rwd_sparse = traj_total_reward(td_record["done"], td_record["rwd_sparse"])
            logger.log_scalar("rwd_sparse", rwd_sparse, step=collected_frames)

        if len(rewards_eval):
            pbar.set_description(
                f"reward: {rewards[-1][1]: 4.4f} (r0 = {r0: 4.4f}), test reward: {rewards_eval[-1][1]: 4.4f}, solved: {solved}"
            )
        del batch
        # gc.collect()


if __name__ == "__main__":
    main()
