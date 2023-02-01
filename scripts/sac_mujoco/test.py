import torch
from rlhive.rl_envs import RoboHiveEnv
from rlhive.sim_algos.helpers.rrl_transform import RRLTransform
from torchrl.envs.transforms import (
    Compose,
    FlattenObservation,
    RewardScaling,
    TransformedEnv,
)
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
    R3MTransform,
    SelectTransform,
    TransformedEnv,
)
from torchrl.envs.utils import set_exploration_mode


def make_env(task, visual_transform, reward_scaling, device):
    assert visual_transform in ("rrl", "r3m")
    base_env = RoboHiveEnv(task, device=device)
    env = make_transformed_env(
        env=base_env, reward_scaling=reward_scaling, visual_transform=visual_transform
    )
    print(env)
    # exit()

    return env


def make_transformed_env(
    env,
    reward_scaling=5.0,
    visual_transform="r3m",
    stats=None,
):
    """
    Apply transforms to the env (such as reward scaling and state normalization)
    """
    env = TransformedEnv(env, SelectTransform("solved", "pixels", "observation"))
    if visual_transform == "rrl":
        vec_keys = ["rrl_vec"]
        selected_keys = ["observation", "rrl_vec"]
        env.append_transform(
            Compose(
                RRLTransform("resnet50", in_keys=["pixels"], download=True),
                FlattenObservation(-2, -1, in_keys=vec_keys),
            )
        )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    elif visual_transform == "r3m":
        vec_keys = ["r3m_vec"]
        selected_keys = ["observation", "r3m_vec"]
        env.append_transform(
            Compose(
                R3MTransform("resnet50", in_keys=["pixels"], download=True),
                FlattenObservation(-2, -1, in_keys=vec_keys),
            )
        )  # Necessary to Compose R3MTransform with FlattenObservation; Track bug: https://github.com/pytorch/rl/issues/802
    else:
        raise NotImplementedError
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    #  we normalize the states
    if stats is None:
        _stats = {"loc": 0.0, "scale": 1.0}
    else:
        _stats = stats
    env.append_transform(
        ObservationNorm(**_stats, in_keys=[out_key], standard_normal=True)
    )
    env.append_transform(DoubleToFloat(in_keys=[out_key], in_keys_inv=[]))
    return env


env = make_env(
    task="visual_franka_slide_random-v3",
    reward_scaling=5.0,
    device=torch.device("cuda:0"),
    visual_transform="rrl",
)
with torch.no_grad(), set_exploration_mode("random"):
    td = env.reset()
    td = env.rand_step()
    print(td)
