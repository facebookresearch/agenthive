# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchrl.envs.transforms import CatTensors, Compose, R3MTransform, \
    TransformedEnv
from torchrl.trainers.helpers.envs import LIBS

from torchrl.envs import RoboHiveEnv  # noqa


def make_r3m_env(env_name, model_name="resnet50", download=True, **kwargs):
    base_env = RoboHiveEnv(env_name, from_pixels=True, pixels_only=False)
    vec_keys = [k for k in base_env.observation_spec.keys() if
                k not in "pixels"]
    env = TransformedEnv(
        base_env,
        Compose(
            R3MTransform(
                model_name,
                keys_in=["pixels"],
                keys_out=["pixel_r3m"],
                download=download,
                **kwargs,
            ),
            CatTensors(
                keys_in=["pixel_r3m", *vec_keys],
                out_key="observation_vector"
            ),
        ),
    )
    return env


LIBS["robohive"] = RoboHiveEnv
