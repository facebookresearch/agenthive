# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any


@dataclass
class LossConfig:
    loss: str = "double"
    # whether double or single SAC loss should be used. Default=double
    hard_update: bool = False
    # whether soft-update should be used with double SAC loss (default) or hard updates.
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
    value_network_update_interval: int = 1000
    # how often the target value network weights are updated (in number of updates).
    # If soft-updates are used, the value is translated into a moving average decay by using
    # the formula decay=1-1/cfg.value_network_update_interval. Default=1000
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    num_q_values: int = 2
    # As suggested in the original SAC paper and in https://arxiv.org/abs/1802.09477, we can
    # use two (or more!) different qvalue networks trained independently and choose the lowest value
    # predicted to predict the state action value. This can be disabled by using this flag.
    # REDQ uses an arbitrary number of Q-value functions to speed up learning in MF contexts.
    target_entropy: Any = None
    # Target entropy for the policy distribution. Default is None (auto calculated as the `target_entropy = -action_dim`)


@dataclass
class PPOLossConfig:
    loss: str = "clip"
    # PPO loss class, either clip or kl or base/<empty>. Default=clip
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    lmbda: float = 0.95
    # lambda factor in GAE (using 'lambda' as attribute is prohibited in python, hence the misspelling)
    entropy_coef: float = 1e-3
    # Entropy factor for the PPO loss
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
    advantage_in_loss: bool = False
    # if True, the advantage is computed on the sub-batch.
