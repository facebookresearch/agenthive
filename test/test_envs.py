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
from torchrl.envs.utils import check_env_specs


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
    )
    assert base_env.from_pixels
    env = TransformedEnv(
        base_env,
        CatTensors(
            [key for key in base_env.observation_spec.keys() if "pixels" not in key],
            "observation",
        ),
    )

    # reset
    tensordict = env.reset()
    assert {"done", "observation", "pixels"} == set(tensordict.keys())
    assert tensordict["pixels"].shape[0] == 2

    # step
    env.rand_step(tensordict)
    assert {
        "reward",
        "done",
        "observation",
        "pixels",
        "action",
        ("next", "observation"),
        ("next", "pixels"),
        "next",
    } == set(tensordict.keys(True))

    # rollout
    tensordict = env.rollout(10)
    assert {
        "reward",
        "done",
        "observation",
        "pixels",
        "action",
        ("next", "observation"),
        ("next", "pixels"),
        "next",
    } == set(tensordict.keys(True))
    assert tensordict.shape == torch.Size([10])
    env.close()

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
def test_specs(env_name):
    base_env = RoboHiveEnv(
        env_name,
    )
    check_env_specs(base_env)
    env = TransformedEnv(
        base_env,
        CatTensors(
            [key for key in base_env.observation_spec.keys() if "pixels" not in key],
            "observation",
        ),
    )
    check_env_specs(env)

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
def test_parallel(env_name):
    def make_env():
        base_env = RoboHiveEnv(
            env_name,
        )
        check_env_specs(base_env)
        env = TransformedEnv(
            base_env,
            CatTensors(
                [key for key in base_env.observation_spec.keys() if "pixels" not in key],
                "observation",
            ),
        )
        return env
    env = ParallelEnv(3, make_env)
    env.reset()
    env.rollout(3)

@pytest.mark.parametrize("parallel", [False, True])
def test_env_render_native(parallel):
    if not parallel:
        env = RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
    else:
        env = ParallelEnv(3, lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0"))
    td = env.reset()
    assert set(td.keys(True)) == {
        "done",
        "observation",
        "pixels",
    }
    td = env.rand_step(td)
    assert set(td.keys(True)) == {
        "done",
        "next",
        ("next", "pixels"),
        "pixels",
        "observation",
        ("next", "observation"),
        "reward",
        "action",
    }
    td = env.rollout(50)
    if not parallel:
        assert td.shape == torch.Size([50])
    else:
        assert td.shape == torch.Size([3, 50])

    assert set(td.keys(True)) == {
        "done",
        "next",
        ("next", "pixels"),
        "pixels",
        "observation",
        ("next", "observation"),
        "reward",
        "action",
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
            ["pixels"],
            ["pixels_embed"],
        ),
    )
    td = env.reset()
    _ = env.rand_step(td)
    td = env.rollout(50)
    if parallel:
        assert td.shape == torch.Size([3, 50])
    else:
        assert td.shape == torch.Size([50])
    env.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
