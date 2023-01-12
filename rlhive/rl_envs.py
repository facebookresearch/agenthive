# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import copy

import numpy as np
import torch
from tensordict.tensordict import make_tensordict
from torchrl.data import (
    CompositeSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform, _has_gym, GymEnv
from torchrl.envs.transforms import CatTensors, Compose, R3MTransform, TransformedEnv
from torchrl.trainers.helpers.envs import LIBS

if _has_gym:
    import gym


class RoboHiveEnv(GymEnv):
    info_keys = ["time", "rwd_dense", "rwd_sparse", "solved"]

    def _build_env(
        self,
        env_name: str,
        from_pixels: bool = False,
        pixels_only: bool = False,
        **kwargs,
    ) -> "gym.core.Env":

        self.pixels_only = pixels_only
        try:
            render_device = int(str(self.device)[-1])
        except ValueError:
            render_device = 0
        print(f"rendering device: {render_device}, device is {self.device}")

        if not _has_gym:
            raise RuntimeError(
                f"gym not found, unable to create {env_name}. "
                f"Consider downloading and installing dm_control from"
                f" {self.git_url}"
            )
        try:
            env = self.lib.make(
                env_name,
                frameskip=self.frame_skip,
                device_id=render_device,
                return_dict=True,
                **kwargs,
            )
            self.wrapper_frame_skip = 1
            from_pixels = any("rgb" in key for key in env.obs_keys)
        except TypeError as err:
            if "unexpected keyword argument 'frameskip" not in str(err):
                raise TypeError(err)
            kwargs.pop("framek_skip")
            env = self.lib.make(
                env_name, return_dict=True, device_id=render_device, **kwargs
            )
            self.wrapper_frame_skip = self.frame_skip

        self.from_pixels = from_pixels
        self.render_device = render_device
        self.info_dict_reader = self.read_info
        return env

    def _make_specs(self, env: "gym.Env") -> None:
        if self.from_pixels:
            num_cams = len([key for key in env.obs_keys if key.startswith("rgb")])
            n_pix = 224 * 224 * 3 * num_cams
            env.observation_space = gym.spaces.Box(
                -8 * np.ones(env.obs_dim - n_pix),
                8 * np.ones(env.obs_dim - n_pix),
                dtype=np.float32,
            )
        self.action_spec = _gym_to_torchrl_spec_transform(
            env.action_space, device=self.device
        )
        observation_spec = _gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
        )
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(observation=observation_spec)
        self.observation_spec = observation_spec
        if self.from_pixels:
            self.observation_spec["pixels"] = BoundedTensorSpec(
                torch.zeros(
                    num_cams,
                    224,  # working with 640
                    224,  # working with 480
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                255
                * torch.ones(
                    num_cams,
                    224,
                    224,
                    3,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                torch.Size(torch.Size([num_cams, 224, 224, 3])),
                dtype=torch.uint8,
                device=self.device,
            )

        self.reward_spec = UnboundedContinuousTensorSpec(
            device=self.device,
        )  # default

    def set_from_pixels(self, from_pixels: bool) -> None:
        """Sets the from_pixels attribute to an existing environment.

        Args:
            from_pixels (bool): new value for the from_pixels attribute

        """
        if from_pixels is self.from_pixels:
            return
        self.from_pixels = from_pixels
        self._make_specs(self.env)

    def read_obs(self, observation):
        # the info is missing from the reset
        observations = self.env.obs_dict
        print("info", list(observations.keys()))
        try:
            del observations["t"]
        except KeyError:
            pass
        # recover vec
        obsvec = []
        pixel_list = []
        for key in observations:
            if key.startswith("rgb"):
                pix = observations[key]
                if not pix.shape[0] == 1:
                    pix = pix[None]
                pixel_list.append(pix)
            elif key in self._env.obs_keys:
                obsvec.append(
                    observations[key]
                )  # ravel helps with images
        if obsvec:
            obsvec = np.concatenate(obsvec, 0)
        if self.from_pixels:
            out = {"observation": obsvec, "pixels": np.concatenate(pixel_list, 0)}
        else:
            out = {"observation": obsvec}
        return super().read_obs(out)

    def read_info(
        self,
        info,
        tensordict_out
    ):
        out = {}
        for key, value in info.items():
            if key in ("obs_dict",):
                continue
            if isinstance(value, dict):
                value = make_tensordict(value, batch_size=[])
            out[key] = value
        tensordict_out.update(out)
        return tensordict_out

    def _step(self, td):
        td = super()._step(td)
        return td

    def _reset(self, td=None, **kwargs):
        td = super()._reset(td, **kwargs)
        return td

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        try:
            render_device = int(str(out.device)[-1])
        except ValueError:
            render_device = 0
        if render_device != self.render_device:
            out._build_env(**self._constructor_kwargs)
        return out


def make_r3m_env(env_name, model_name="resnet50", download=True, **kwargs):
    base_env = RoboHiveEnv(env_name, from_pixels=True, pixels_only=False)
    vec_keys = [k for k in base_env.observation_spec.keys() if k not in "pixels"]
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
            CatTensors(keys_in=["pixel_r3m", *vec_keys], out_key="observation_vector"),
        ),
    )
    return env


LIBS["robohive"] = RoboHiveEnv
