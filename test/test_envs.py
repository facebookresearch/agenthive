import argparse

import pytest
import torch
from rlhive.rl_envs import RoboHiveEnv


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
    env = RoboHiveEnv(
        env_name,
        from_pixels=True,
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
