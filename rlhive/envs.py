# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Custom env reg for RoboHive usage in TorchRL
# Pixel rendering will be queried by torchrl, so we don't include those keys in visual_obs_keys_wt
import os
from pathlib import Path

import mj_envs.envs.multi_task.substeps1
from gym.envs.registration import register
from mj_envs.envs.multi_task.common.franka_kitchen_v1 import KitchenFrankaFixed

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


@set_directory(CURR_DIR)
def register_kitchen_envs():
    print("RLHive:> Registering Kitchen Envs")

    # ========================================================

    # V3 environments
    # In this version of the environment, the observations consist of the
    # distance between end effector and all relevent objects in the scene

    visual_obs_keys_wt = {
        "robot_jnt": 1.0,
        "end_effector": 1.0,
    }
    obs_keys_wt = visual_obs_keys_wt
    for site in KitchenFrankaFixed.OBJ_INTERACTION_SITES:
        obs_keys_wt[site + "_err"] = 1.0

    # Kitchen
    register(
        id="visual_kitchen_close-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_goal": {},
            "obj_init": {
                "knob1_joint": -1.57,
                "knob2_joint": -1.57,
                "knob3_joint": -1.57,
                "knob4_joint": -1.57,
                "lightswitch_joint": -0.7,
                "slidedoor_joint": 0.44,
                "micro0joint": -1.25,
                "rightdoorhinge": 1.57,
                "leftdoorhinge": -1.25,
            },
            "obs_keys_wt": obs_keys_wt,
        },
    )

    # Microwave door
    register(
        id="visual_kitchen_micro_open-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"micro0joint": 0},
            "obj_goal": {"micro0joint": -1.25},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "microhandle_site",
        },
    )
    register(
        id="visual_kitchen_micro_close-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"micro0joint": -1.25},
            "obj_goal": {"micro0joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "microhandle_site",
        },
    )

    # Right hinge cabinet
    register(
        id="visual_kitchen_rdoor_open-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"rightdoorhinge": 0},
            "obj_goal": {"rightdoorhinge": 1.57},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "rightdoor_site",
        },
    )
    register(
        id="visual_kitchen_rdoor_close-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"rightdoorhinge": 1.57},
            "obj_goal": {"rightdoorhinge": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "rightdoor_site",
        },
    )

    # Left hinge cabinet
    register(
        id="visual_kitchen_ldoor_open-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"leftdoorhinge": 0},
            "obj_goal": {"leftdoorhinge": -1.25},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "leftdoor_site",
        },
    )
    register(
        id="visual_kitchen_ldoor_close-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"leftdoorhinge": -1.25},
            "obj_goal": {"leftdoorhinge": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "leftdoor_site",
        },
    )

    # Slide cabinet
    register(
        id="visual_kitchen_sdoor_open-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"slidedoor_joint": 0},
            "obj_goal": {"slidedoor_joint": 0.44},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "slide_site",
        },
    )
    register(
        id="visual_kitchen_sdoor_close-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"slidedoor_joint": 0.44},
            "obj_goal": {"slidedoor_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "slide_site",
        },
    )

    # Lights
    register(
        id="visual_kitchen_light_on-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"lightswitch_joint": 0},
            "obj_goal": {"lightswitch_joint": -0.7},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "light_site",
        },
    )
    register(
        id="visual_kitchen_light_off-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"lightswitch_joint": -0.7},
            "obj_goal": {"lightswitch_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "light_site",
        },
    )

    # Knob4
    register(
        id="visual_kitchen_knob4_on-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob4_joint": 0},
            "obj_goal": {"knob4_joint": -1.57},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob4_site",
        },
    )
    register(
        id="visual_kitchen_knob4_off-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob4_joint": -1.57},
            "obj_goal": {"knob4_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob4_site",
        },
    )

    # Knob3
    register(
        id="visual_kitchen_knob3_on-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob3_joint": 0},
            "obj_goal": {"knob3_joint": -1.57},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob3_site",
        },
    )
    register(
        id="visual_kitchen_knob3_off-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob3_joint": -1.57},
            "obj_goal": {"knob3_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob3_site",
        },
    )

    # Knob2
    register(
        id="visual_kitchen_knob2_on-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob2_joint": 0},
            "obj_goal": {"knob2_joint": -1.57},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob2_site",
        },
    )
    register(
        id="visual_kitchen_knob2_off-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob2_joint": -1.57},
            "obj_goal": {"knob2_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob2_site",
        },
    )

    # Knob1
    register(
        id="visual_kitchen_knob1_on-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob1_joint": 0},
            "obj_goal": {"knob1_joint": -1.57},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob1_site",
        },
    )
    register(
        id="visual_kitchen_knob1_off-v3",
        entry_point=ENTRY_POINT,
        max_episode_steps=50,
        kwargs={
            "model_path": MODEL_PATH,
            "config_path": CONFIG_PATH,
            "obj_init": {"knob1_joint": -1.57},
            "obj_goal": {"knob1_joint": 0},
            "obs_keys_wt": obs_keys_wt,
            "interact_site": "knob1_site",
        },
    )


@set_directory(CURR_DIR)
def register_franka_envs():
    # Franka Appliance ======================================================================

    # MICROWAVE
    # obs_keys_wt = {
    #     "robot_jnt": 1.0,
    #     "end_effector": 1.0,
    # }
    register(
        id="visual_franka_micro_open-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=75,
        kwargs={
            "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
            "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
            "obj_init": {"micro0joint": 0},
            "obj_goal": {"micro0joint": -1.25},
            "obj_interaction_site": ("microhandle_site",),
            "obj_jnt_names": ("micro0joint",),
            "interact_site": "microhandle_site",
        },
    )
    register(
        id="visual_franka_micro_close-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=50,
        kwargs={
            "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
            "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
            "obj_init": {"micro0joint": -1.25},
            "obj_goal": {"micro0joint": 0},
            "obj_interaction_site": ("microhandle_site",),
            "obj_jnt_names": ("micro0joint",),
            "interact_site": "microhandle_site",
        },
    )
    register(
        id="visual_franka_micro_random-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=50,
        kwargs={
            "model_path": CURR_DIR + "/../common/microwave/franka_microwave.xml",
            "config_path": CURR_DIR + "/../common/microwave/franka_microwave.config",
            "obj_init": {"micro0joint": (-1.25, 0)},
            "obj_goal": {"micro0joint": (-1.25, 0)},
            "obj_interaction_site": ("microhandle_site",),
            "obj_jnt_names": ("micro0joint",),
            "obj_body_randomize": ("microwave",),
            "interact_site": "microhandle_site",
        },
    )

    # SLIDE-CABINET
    # obs_keys_wt = {
    #     "robot_jnt": 1.0,
    #     "end_effector": 1.0,
    # }
    register(
        id="visual_franka_slide_open-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=50,
        kwargs={
            "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
            "config_path": CURR_DIR
            + "/../common/slidecabinet/franka_slidecabinet.config",
            "obj_init": {"slidedoor_joint": 0},
            "obj_goal": {"slidedoor_joint": 0.44},
            "obj_interaction_site": ("slide_site",),
            "obj_jnt_names": ("slidedoor_joint",),
            "interact_site": "slide_site",
        },
    )
    register(
        id="visual_franka_slide_close-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=50,
        kwargs={
            "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
            "config_path": CURR_DIR
            + "/../common/slidecabinet/franka_slidecabinet.config",
            "obj_init": {"slidedoor_joint": 0.44},
            "obj_goal": {"slidedoor_joint": 0},
            "obj_interaction_site": ("slide_site",),
            "obj_jnt_names": ("slidedoor_joint",),
            "interact_site": "slide_site",
        },
    )
    register(
        id="visual_franka_slide_random-v3",
        entry_point="mj_envs.envs.multi_task.common.franka_appliance_v1:FrankaAppliance",
        max_episode_steps=50,
        kwargs={
            "model_path": CURR_DIR + "/../common/slidecabinet/franka_slidecabinet.xml",
            "config_path": CURR_DIR
            + "/../common/slidecabinet/franka_slidecabinet.config",
            "obj_init": {"slidedoor_joint": (0, 0.44)},
            "obj_goal": {"slidedoor_joint": (0, 0.44)},
            "obj_interaction_site": ("slide_site",),
            "obj_jnt_names": ("slidedoor_joint",),
            "obj_body_randomize": ("slidecabinet",),
            "interact_site": "slide_site",
        },
    )
