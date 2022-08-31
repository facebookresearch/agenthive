# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

print("importing...")
from .envs import register_franka_envs, register_kitchen_envs
print("done")
register_franka_envs()
register_kitchen_envs()
from .rl_envs import RoboHiveEnv
