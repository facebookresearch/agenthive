# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import uuid
from datetime import datetime

import hydra
import torch
from hydra.core.config_store import ConfigStore
from torchrl.modules import EGreedyWrapper, OrnsteinUhlenbeckProcessWrapper
from torchrl.trainers.helpers import transformed_env_constructor, \
    parallel_env_constructor, correct_for_frame_skip

from rlhive.sim_algos.helpers.collectors import make_collector
from rlhive.sim_algos.helpers.replay_buffer import make_replay_buffer
from rlhive.sim_algos.helpers.trainers import make_trainer
from rlhive.sim_algos.sac.loss import make_sac_loss
from rlhive.sim_algos.sac.models import make_sac_model
from .helpers import \
    TrainerConfig, \
    OffPolicyCollectorConfig, \
    EnvConfig, \
    LossConfig, \
    LoggerConfig, \
    ReplayArgsConfig

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        LoggerConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]

Config = dataclasses.make_dataclass(
    cls_name="Config",
    fields=config_fields
)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def make_env_constructor(cfg):

    stats = None
    if not cfg.vecnorm and cfg.norm_stats:
        proof_env = transformed_env_constructor(cfg=cfg, use_env_creator=False)()
        stats = get_stats_random_rollout(
            cfg,
            proof_env,
            key="next_pixels" if cfg.from_pixels else "next_observation_vector",
        )
        # make sure proof_env is closed
        proof_env.close()
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}

    single_env_constructor = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats
    )

    if cfg.gSDE:
        raise NotImplementedError("gSDE is not supported yet.")
    action_dim_gsde, state_dim_gsde = None, None

    multi_env_constructor = parallel_env_constructor(
        cfg=cfg,
        stats=stats,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    return single_env_constructor, multi_env_constructor

MODEL_BUILDERS = {
    "sac": make_sac_model,
}
def make_model(cfg, device, single_env_constructor):
    # Builds a model
    model = MODEL_BUILDERS[cfg.model.model_entry_point](cfg.model, device, single_env_constructor)
    return model

LOSS_BUILDERS = {
    "sac": make_sac_loss,
}
def make_loss_module(cfg, model):
    # from a model, builds a corresponding loss function
    loss, updater = LOSS_BUILDERS[cfg.model.loss_entry_point](cfg.model)
    return loss, updater

def make_exploration_policy(cfg, model):
    # Extracts a policy and possibly wraps it in an exploration wrapper
    policy = model[0]
    device = policy.device
    if cfg.exploration.type == "eps_greedy":
        policy = EGreedyWrapper(
            policy,
            eps_init=cfg.exploration.eps_init,
            eps_end=cfg.exploration.eps_end,
            annealing_num_steps=cfg.exploration.annealing_num_steps,
            action_key=cfg.exploration.action_key,
        ).to(device)
    elif cfg.exploration.type == "additive_gaussian":
        policy = AdditiveGaussianWrapper(
            policy,
            eps_init=cfg.exploration.eps_init,
            eps_end=cfg.exploration.eps_end,
            annealing_num_steps=cfg.exploration.annealing_num_steps,
            action_key=cfg.exploration.action_key,
        ).to(device)
    elif cfg.exploration.type == "ou_exploration":
        policy = OrnsteinUhlenbeckProcessWrapper(
            policy,
            eps_init=cfg.exploration.eps_init,
            eps_end=cfg.exploration.eps_end,
            annealing_num_steps=cfg.exploration.annealing_num_steps,
            theta=cfg.exploration.theta,
            mu=cfg.exploration.mu,
            sigma=cfg.exploration.sigma,
        ).to(device)
    else:
        raise NotImplementedError(cfg.exploration)
    return policy

def make_logger(cfg):
    if cfg.logger == "tensorboard":
        from torchrl.trainers.loggers.tensorboard import TensorboardLogger

        logger = TensorboardLogger(log_dir="sac_logging", exp_name=exp_name)
    elif cfg.logger == "csv":
        from torchrl.trainers.loggers.csv import CSVLogger

        logger = CSVLogger(log_dir="sac_logging", exp_name=exp_name)
    elif cfg.logger == "wandb":
        from torchrl.trainers.loggers.wandb import WandbLogger

        logger = WandbLogger(log_dir="sac_logging", exp_name=exp_name)
    return logger

def make_exp_name(cfg):
    exp_name = "_".join(
        [
            "SAC",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    return exp_name

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):

    cfg = correct_for_frame_skip(cfg)

    device = (
        torch.device(cfg.model_device)
        if cfg.model_device
        else torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    single_env_constructor, multi_env_constructor = make_env_constructor(cfg)
    model = make_model(cfg, device, single_env_constructor)
    loss_module, target_net_updater = make_loss_module(cfg, model)
    exploration_policy = make_exploration_policy(cfg, model)
    replay_buffer = make_replay_buffer(cfg)
    logger = make_logger(cfg)
    recorder = make_recorder(cfg, single_env_constructor)
    collector = make_collector(
        cfg,
        exploration_policy,
        multi_env_constructor
    )

    trainer = make_trainer(
        collector=collector,
        loss_module=loss_module,
        recorder=recorder,
        target_net_updater=target_net_updater,
        policy_exploration=exploration_policy,
        replay_buffer=replay_buffer,
        logger=logger,
        cfg=cfg,
    )

    trainer.train()

if __name__ == "__main__":
    main()
