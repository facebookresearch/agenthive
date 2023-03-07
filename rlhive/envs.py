# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Custom env reg for RoboHive usage in TorchRL
# Pixel rendering will be queried by torchrl, so we don't include those keys in visual_obs_keys_wt
import os
import warnings
from pathlib import Path

import mj_envs.envs.multi_task.substeps1

from mj_envs.envs.env_variants import register_env_variant

visual_obs_keys_wt = mj_envs.envs.multi_task.substeps1.visual_obs_keys_wt


class set_directory(object):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd
    """

    def __init__(self, path: Path):
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args, **kwargs):
        os.chdir(self.origin)

    def __call__(self, fun):
        def new_fun(*args, **kwargs):
            with set_directory(Path(self.path)):
                return fun(*args, **kwargs)

        return new_fun


CURR_DIR = mj_envs.envs.multi_task.substeps1.CURR_DIR
MODEL_PATH = mj_envs.envs.multi_task.substeps1.MODEL_PATH
CONFIG_PATH = mj_envs.envs.multi_task.substeps1.CONFIG_PATH
RANDOM_ENTRY_POINT = mj_envs.envs.multi_task.substeps1.RANDOM_ENTRY_POINT
FIXED_ENTRY_POINT = mj_envs.envs.multi_task.substeps1.FIXED_ENTRY_POINT
ENTRY_POINT = RANDOM_ENTRY_POINT

override_keys = [
    "objs_jnt",
    "end_effector",
    "knob1_site_err",
    "knob2_site_err",
    "knob3_site_err",
    "knob4_site_err",
    "light_site_err",
    "slide_site_err",
    "leftdoor_site_err",
    "rightdoor_site_err",
    "microhandle_site_err",
    "kettle_site0_err",
    "rgb:right_cam:224x224:2d",
    "rgb:left_cam:224x224:2d",
]


@set_directory(CURR_DIR)
def register_kitchen_envs():
    print("RLHive:> Registering Kitchen Envs")

    env_list = [
        "kitchen_knob1_off-v3",
        "kitchen_knob1_on-v3",
        "kitchen_knob2_off-v3",
        "kitchen_knob2_on-v3",
        "kitchen_knob3_off-v3",
        "kitchen_knob3_on-v3",
        "kitchen_knob4_off-v3",
        "kitchen_knob4_on-v3",
        "kitchen_light_off-v3",
        "kitchen_light_on-v3",
        "kitchen_sdoor_close-v3",
        "kitchen_sdoor_open-v3",
        "kitchen_ldoor_close-v3",
        "kitchen_ldoor_open-v3",
        "kitchen_rdoor_close-v3",
        "kitchen_rdoor_open-v3",
        "kitchen_micro_close-v3",
        "kitchen_micro_open-v3",
        # "kitchen_close-v3",
    ]

    visual_obs_keys_wt = {
        "robot_jnt": 1.0,
        "end_effector": 1.0,
        "rgb:right_cam:224x224:2d": 1.0,
        "rgb:left_cam:224x224:2d": 1.0,
    }
    for env in env_list:
        try:
            new_env_name = "visual_" + env
            register_env_variant(
                env,
                variants={"obs_keys_wt": visual_obs_keys_wt},
                variant_id=new_env_name,
                override_keys=override_keys,
            )
        except AssertionError as err:
            warnings.warn(
                f"Could not register {new_env_name}, the following error was raised: {err}"
            )


@set_directory(CURR_DIR)
def register_franka_envs():
    print("RLHive:> Registering Franka Envs")
    env_list = [
        "franka_slide_random-v3",
        "franka_slide_close-v3",
        "franka_slide_open-v3",
        "franka_micro_random-v3",
        "franka_micro_close-v3",
        "franka_micro_open-v3",
    ]

    # Franka Appliance ======================================================================
    visual_obs_keys_wt = {
        "robot_jnt": 1.0,
        "end_effector": 1.0,
        "rgb:right_cam:224x224:2d": 1.0,
        "rgb:left_cam:224x224:2d": 1.0,
    }
    for env in env_list:
        try:
            new_env_name = "visual_" + env
            register_env_variant(
                env,
                variants={"obs_keys_wt": visual_obs_keys_wt},
                variant_id=new_env_name,
                override_keys=override_keys,
            )
        except AssertionError as err:
            warnings.warn(
                f"Could not register {new_env_name}, the following error was raised: {err}"
            )


@set_directory(CURR_DIR)
def register_hand_envs():
    print("RLHive:> Registering Arm Envs")
    env_list = ["door-v1", "hammer-v1", "pen-v1", "relocate-v1"]

    # Hand Manipulation Suite ======================================================================
    for env in env_list:
        try:
            new_env_name = "visual_" + env
            register_env_variant(
                env,
                variants={
                    "obs_keys": [
                        "hand_jnt",
                        "rgb:vil_camera:224x224:2d",
                        "rgb:fixed:224x224:2d",
                    ]
                },
                variant_id=new_env_name,
            )
        except AssertionError as err:
            warnings.warn(
                f"Could not register {new_env_name}, the following error was raised: {err}"
            )


@set_directory(CURR_DIR)
def register_myo_envs():
    print("RLHive:> Registering Myo Envs")
    env_list = ["motorFingerReachFixed-v0"]

    # Hand Manipulation Suite ======================================================================
    for env in env_list:
        try:
            new_env_name = "visual_" + env
            register_env_variant(
                env,
                variants={
                    "obs_keys": [
                        "hand_jnt",
                        "rgb:vil_camera:224x224:2d",
                        "rgb:fixed:224x224:2d",
                    ]
                },
                variant_id=new_env_name,
            )
        except AssertionError as err:
            warnings.warn(
                f"Could not register {new_env_name}, the following error was raised: {err}"
            )
