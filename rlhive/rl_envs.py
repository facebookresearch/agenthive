# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchrl.envs import RoboHiveEnv  # noqa
from torchrl.envs.transforms import (
    CatTensors,
    Compose,
    FlattenObservation,
    R3MTransform,
    TransformedEnv,
)
from torchrl.trainers.helpers.envs import LIBS


def make_r3m_env(env_name, model_name="resnet50", download=True, **kwargs):
    base_env = RoboHiveEnv(env_name, from_pixels=True, pixels_only=False)
    vec_keys = [
        k
        for k in base_env.observation_spec.keys()
        if (
            k not in ("pixels", "state", "time")
            and "rwd" not in k
            and "visual" not in k
            and "dict" not in k
        )
    ]
    env = TransformedEnv(
        base_env,
        Compose(
            R3MTransform(
                model_name,
                in_keys=["pixels"],
                out_keys=["pixel_r3m"],
                download=download,
                **kwargs,
            ),
            FlattenObservation(-2, -1, in_keys=["pixel_r3m"]),
            CatTensors(in_keys=["pixel_r3m", *vec_keys], out_key="observation_vector"),
        ),
    )
    return env


LIBS["robohive"] = RoboHiveEnv
