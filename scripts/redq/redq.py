# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from rlhive.rl_envs import RoboHiveEnv

from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    ParallelEnv,
    R3MTransform,
    SelectTransform,
    TransformedEnv,
)
from torchrl.envs.transforms import Compose, FlattenObservation, RewardScaling
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    EnvConfig,
    initialize_observation_norm_transforms,
    retrieve_observation_norms_state_dict,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import LossConfig, make_redq_loss
from torchrl.trainers.helpers.models import make_redq_model, REDQModelConfig
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig
from torchrl.trainers.loggers.utils import generate_exp_name, get_logger


def make_env(
    task,
    reward_scaling,
    device,
    obs_norm_state_dict=None,
    action_dim_gsde=None,
    state_dim_gsde=None,
):
    base_env = RoboHiveEnv(task, device=device)
    env = make_transformed_env(env=base_env, reward_scaling=reward_scaling)

    if obs_norm_state_dict is not None:
        obs_norm = ObservationNorm(
            **obs_norm_state_dict, in_keys=["observation_vector"]
        )
        env.append_transform(obs_norm)

    if action_dim_gsde is not None:
        env.append_transform(
            gSDENoise(action_dim=action_dim_gsde, state_dim=state_dim_gsde)
        )
    return env


def make_transformed_env(
    env,
    reward_scaling=5.0,
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """
    env = TransformedEnv(env, SelectTransform("solved", "pixels", "observation"))
    env.append_transform(
        Compose(
            R3MTransform("resnet50", in_keys=["pixels"], download=True),
            FlattenObservation(-2, -1, in_keys=["r3m_vec"]),
        )
    )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))
    selected_keys = ["r3m_vec", "observation"]
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


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        REDQModelConfig,
        LoggerConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]

Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = generate_exp_name("REDQ", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger, logger_name="redq_logging", experiment_name=exp_name
    )

    key, init_env_steps = None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = ("next", "observation_vector")
        init_env_steps = cfg.init_env_steps

    proof_env = make_env(
        task=cfg.env_name,
        reward_scaling=cfg.reward_scaling,
        device=device,
    )
    initialize_observation_norm_transforms(
        proof_environment=proof_env, num_iter=init_env_steps, key=key
    )
    _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]

    print(proof_env)
    model = make_redq_model(
        proof_env,
        cfg=cfg,
        device=device,
        in_keys=["observation_vector"],
    )
    loss_module, target_net_updater = make_redq_loss(model, cfg)

    actor_model_explore = model[0]
    if cfg.ou_exploration:
        if cfg.gSDE:
            raise RuntimeError("gSDE and ou_exploration are incompatible")
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore,
            annealing_num_steps=cfg.annealing_frames,
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    if cfg.gSDE:
        with torch.no_grad(), set_exploration_mode("random"):
            # get dimensions to build the parallel env
            proof_td = actor_model_explore(proof_env.reset().to(device))
        action_dim_gsde, state_dim_gsde = proof_td.get("_eps_gSDE").shape[-2:]
        del proof_td
    else:
        action_dim_gsde, state_dim_gsde = None, None

    proof_env.close()
    create_env_fn = make_env(  # Pass EnvBase instead of the create_env_fn
        task=cfg.env_name,
        reward_scaling=cfg.reward_scaling,
        device=device,
        obs_norm_state_dict=obs_norm_state_dict,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        cfg=cfg,
        # make_env_kwargs=[
        #     {"device": device} if device >= 0 else {}
        #     for device in args.env_rendering_devices
        # ],
    )

    replay_buffer = make_replay_buffer(device, cfg)

    # recorder = transformed_env_constructor(
    #    cfg,
    #    video_tag=video_tag,
    #    norm_obs_only=True,
    #    obs_norm_state_dict=obs_norm_state_dict,
    #    logger=logger,
    #    use_env_creator=False,
    # )()
    recorder = make_env(
        task=cfg.env_name,
        reward_scaling=cfg.reward_scaling,
        device=device,
        obs_norm_state_dict=obs_norm_state_dict,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    # remove video recorder from recorder to have matching state_dict keys
    if cfg.record_video:
        recorder_rm = TransformedEnv(recorder.base_env)
        for transform in recorder.transform:
            if not isinstance(transform, VideoRecorder):
                recorder_rm.append_transform(transform.clone())
    else:
        recorder_rm = recorder

    if isinstance(create_env_fn, ParallelEnv):
        recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
        create_env_fn.close()
    elif isinstance(create_env_fn, EnvCreator):
        recorder_rm.load_state_dict(create_env_fn().state_dict())
    else:
        recorder_rm.load_state_dict(create_env_fn.state_dict())

    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)

    trainer = make_trainer(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        actor_model_explore,
        replay_buffer,
        logger,
        cfg,
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
