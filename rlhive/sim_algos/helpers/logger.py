# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torchrl.envs import RewardScaling
from torchrl.record import VideoRecorder


def make_recorder(cfg, single_env_constructor, logger):
    recorder = single_env_constructor()
    for t in recorder.transform:
        if isinstance(t, VideoRecorder):
            t.logger = logger

    if cfg.vecnorm:
        raise RuntimeError("vecnorm not supported yet")

    # reset reward scaling, as it was just overwritten by state_dict load
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)

    return recorder


@dataclass
class LoggerConfig:
    logger: str = "csv"
    # recorder type to be used. One of 'tensorboard', 'wandb' or 'csv'
    record_video: bool = False
    # whether a video of the task should be rendered during logging.
    no_video: bool = True
    # whether a video of the task should be rendered during logging.
    exp_name: str = ""
    # experiment name. Used for logging directory.
    # A date and uuid will be joined to account for multiple experiments with the same name.
    record_interval: int = 30
    # number of batch collections in between two collections of validation rollouts. Default=1000.
    record_frames: int = 1000
    # number of steps in validation rollouts. " "Default=1000.
    recorder_log_keys: Any = field(default_factory=lambda: ["reward"])
    # Keys to log in the recorder
