import argparse

import pytest
import torch
from rlhive.rl_envs import RoboHiveEnv
from torchrl.envs import (
    CatTensors,
    EnvCreator,
    ParallelEnv,
    R3MTransform,
    TransformedEnv,
)


def test_state_env():
    pass


def test_pixel_env():
    pass


@pytest.mark.parametrize(
    "env_name",
    [
        "visual_franka_slide_random-v3",
        "visual_franka_slide_close-v3",
        "visual_franka_slide_open-v3",
        "visual_franka_micro_random-v3",
        "visual_franka_micro_close-v3",
        "visual_franka_micro_open-v3",
        "visual_kitchen_knob1_off-v3",
        "visual_kitchen_knob1_on-v3",
        "visual_kitchen_knob2_off-v3",
        "visual_kitchen_knob2_on-v3",
        "visual_kitchen_knob3_off-v3",
        "visual_kitchen_knob3_on-v3",
        "visual_kitchen_knob4_off-v3",
        "visual_kitchen_knob4_on-v3",
        "visual_kitchen_light_off-v3",
        "visual_kitchen_light_on-v3",
        "visual_kitchen_sdoor_close-v3",
        "visual_kitchen_sdoor_open-v3",
        "visual_kitchen_ldoor_close-v3",
        "visual_kitchen_ldoor_open-v3",
        "visual_kitchen_rdoor_close-v3",
        "visual_kitchen_rdoor_open-v3",
        "visual_kitchen_micro_close-v3",
        "visual_kitchen_micro_open-v3",
        "visual_kitchen_close-v3",
    ],
)
def test_mixed_env(env_name):
    base_env = RoboHiveEnv(
        env_name,
        from_pixels=True,
    )
    env = TransformedEnv(
        base_env,
        CatTensors(
            [key for key in base_env.observation_spec.keys() if "pixels" not in key],
            "next_observation",
        ),
    )

    # reset
    tensordict = env.reset()
    assert {"done", "observation", "pixels"} == set(tensordict.keys())
    if "franka" in env_name:
        assert tensordict["pixels"].shape[0] == 3
    else:
        assert tensordict["pixels"].shape[0] == 2

    # step
    env.rand_step(tensordict)
    assert {
        "reward",
        "done",
        "observation",
        "pixels",
        "action",
        "next_observation",
        "next_pixels",
    } == set(tensordict.keys())

    # rollout
    tensordict = env.rollout(10)
    assert {
        "reward",
        "done",
        "observation",
        "pixels",
        "action",
        "next_observation",
        "next_pixels",
    } == set(tensordict.keys())
    assert tensordict.shape == torch.Size([10])
    env.close()


@pytest.mark.parametrize("parallel", [True, False])
def test_env_render_native(parallel):
    if not parallel:
        env = RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
    else:
        env = ParallelEnv(3, lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0"))
    td = env.reset()
    assert set(td.keys()) == {
        "done",
        "rgb:right_cam:240x424:2d",
        "qp",
        "qv",
        "rgb:left_cam:240x424:2d",
    }
    td = env.rand_step(td)
    assert set(td.keys()) == {
        "done",
        "next_rgb:right_cam:240x424:2d",
        "rgb:right_cam:240x424:2d",
        "qp",
        "next_qv",
        "qv",
        "rgb:left_cam:240x424:2d",
        "next_rgb:left_cam:240x424:2d",
        "reward",
        "action",
        "next_qp",
    }
    td = env.rollout(50)
    if not parallel:
        assert td.shape == torch.Size([50])
    else:
        assert td.shape == torch.Size([3, 50])

    assert set(td.keys()) == {
        "done",
        "next_rgb:right_cam:240x424:2d",
        "rgb:right_cam:240x424:2d",
        "qp",
        "next_qv",
        "qv",
        "rgb:left_cam:240x424:2d",
        "next_rgb:left_cam:240x424:2d",
        "reward",
        "action",
        "next_qp",
    }
    env.close()


@pytest.mark.parametrize(
    "parallel,env_creator", [[True, True], [True, False], [False, True]]
)
def test_env_r3m_native(parallel, env_creator):
    if not parallel:
        base_env = RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
    else:
        if env_creator:
            env_creator = EnvCreator(
                lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
            )
        else:
            env_creator = lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")

        base_env = ParallelEnv(3, env_creator)
    env = TransformedEnv(
        base_env,
        R3MTransform(
            "resnet18",
            ["next_rgb:right_cam:240x424:2d", "next_rgb:left_cam:240x424:2d"],
            ["pixels_embed"],
        ),
    )
    td = env.reset()
    td = env.rand_step(td)
    td = env.rollout(50)
    if parallel:
        assert td.shape == torch.Size([3, 50])
    else:
        assert td.shape == torch.Size([50])
    env.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
