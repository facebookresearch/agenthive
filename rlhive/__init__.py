# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .envs import register_franka_envs, register_kitchen_envs, register_hand_envs

register_franka_envs()
register_kitchen_envs()
register_hand_envs()

from .rl_envs import RoboHiveEnv
