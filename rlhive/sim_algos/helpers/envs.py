# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Optional


@dataclass
class EnvConfig:
    env_library: str = "gym"
    # env_library used for the simulated environment. Default=gym
    env_name: str = "Humanoid-v2"
    # name of the environment to be created. Default=Humanoid-v2
    env_task: str = ""
    # task (if any) for the environment. Default=run
    from_pixels: bool = False
    # whether the environment output should be state vector(s) (default) or the pixels.
    frame_skip: int = 1
    # frame_skip for the environment. Note that this value does NOT impact the buffer size,
    # maximum steps per trajectory, frames per batch or any other factor in the algorithm,
    # e.g. if the total number of frames that has to be computed is 50e6 and the frame skip is 4
    # the actual number of frames retrieved will be 200e6. Default=1.
    reward_scaling: Optional[float] = None
    # scale of the reward.
    reward_loc: float = 0.0
    # location of the reward.
    init_env_steps: int = 1000
    # number of random steps to compute normalizing constants
    vecnorm: bool = False
    # Normalizes the environment observation and reward outputs with the running statistics obtained across processes.
    norm_rewards: bool = False
    # If True, rewards will be normalized on the fly. This may interfere with SAC update rule and should be used cautiously.
    norm_stats: bool = True
    # Deactivates the normalization based on random collection of data.
    noops: int = 0
    # number of random steps to do after reset. Default is 0
    catframes: int = 0
    # Number of frames to concatenate through time. Default is 0 (do not use CatFrames).
    center_crop: Any = dataclass_field(default_factory=lambda: [])
    # center crop size.
    grayscale: bool = True
    # Disables grayscale transform.
    max_frames_per_traj: int = 1000
    # Number of steps before a reset of the environment is called (if it has not been flagged as done before).
    batch_transform: bool = False
    # if True, the transforms will be applied to the parallel env, and not to each individual env.
