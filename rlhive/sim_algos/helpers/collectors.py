# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torchrl.trainers.helpers import make_collector_offpolicy


def make_collector(cfg, policy, make_env):
    return make_collector_offpolicy(make_env, policy, cfg)

@dataclass
class OnPolicyCollectorConfig:
    collector_devices: Any = field(default_factory=lambda: ["cpu"])
    # device on which the data collector should store the trajectories to be passed to this script.
    # If the collector device differs from the policy device (cuda:0 if available), then the
    # weights of the collector policy are synchronized with collector.update_policy_weights_().
    pin_memory: bool = False
    # if True, the data collector will call pin_memory before dispatching tensordicts onto the passing device
    init_with_lag: bool = False
    # if True, the first trajectory will be truncated earlier at a random step. This is helpful
    # to desynchronize the environments, such that steps do no match in all collected
    # rollouts. Especially useful for online training, to prevent cyclic sample indices.
    frames_per_batch: int = 1000
    # number of steps executed in the environment per collection.
    # This value represents how many steps will the data collector execute and return in *each*
    # environment that has been created in between two rounds of optimization
    # (see the optim_steps_per_batch above).
    # On the one hand, a low value will enhance the data throughput between processes in async
    # settings, which can make the accessing of data a computational bottleneck.
    # High values will on the other hand lead to greater tensor sizes in memory and disk to be
    # written and read at each global iteration. One should look at the number of frames per second
    # in the log to assess the efficiency of the configuration.
    total_frames: int = 50000000
    # total number of frames collected for training. Does account for frame_skip (i.e. will be
    # divided by the frame_skip). Default=50e6.
    num_workers: int = 32
    # Number of workers used for data collection.
    env_per_collector: int = 8
    # Number of environments per collector. If the env_per_collector is in the range:
    # 1<env_per_collector<=num_workers, then the collector runs
    # ceil(num_workers/env_per_collector) in parallel and executes the policy steps synchronously
    # for each of these parallel wrappers. If env_per_collector=num_workers, no parallel wrapper is created
    seed: int = 42
    # seed used for the environment, pytorch and numpy.
    exploration_mode: str = ""
    # exploration mode of the data collector.
    async_collection: bool = False
    # whether data collection should be done asynchrously. Asynchrounous data collection means
    # that the data collector will keep on running the environment with the previous weights
    # configuration while the optimization loop is being done. If the algorithm is trained
    # synchronously, data collection and optimization will occur iteratively, not concurrently.


@dataclass
class OffPolicyCollectorConfig(OnPolicyCollectorConfig):
    multi_step: bool = False
    # whether or not multi-step rewards should be used.
    n_steps_return: int = 3
    # If multi_step is set to True, this value defines the number of steps to look ahead for the reward computation.
    init_random_frames: int = 50000
