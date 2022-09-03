import argparse

import pytest
import torch
from omegaconf import OmegaConf
from rlhive.sim_algos.helpers import EnvConfig
from rlhive.sim_algos.run import make_env_constructor
from utils import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
def test_make_r3menv(device):
    cfg = EnvConfig
    # hacky way of create a config that can be shared across processes
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    cfg.env_name = "FrankaReachRandom_v2d-v0"
    cfg.r3m = "resnet50"
    cfg.collector_devices = str(device)
    cfg.norm_stats = False
    cfg.env_per_collector = 2
    cfg.pin_memory = False
    cfg.batch_transform = True

    single_env_constructor, multi_env_constructor = make_env_constructor(cfg)
    env = single_env_constructor()
    print(env)
    td = env.reset()
    assert {"done", "observation_vector"} == set(td.keys())
    td = env.rollout(10)
    assert {
        "action",
        "done",
        "next_observation_vector",
        "observation_vector",
        "reward",
    } == set(td.keys())
    assert td.shape == torch.Size([10])

    env = multi_env_constructor
    print(env)
    td = env.reset()
    assert {"done", "observation_vector"} == set(td.keys())
    td = env.rollout(10)
    assert {
        "action",
        "done",
        "next_observation_vector",
        "observation_vector",
        "reward",
    } == set(td.keys())
    assert td.shape == torch.Size([2, 10])
    env.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
