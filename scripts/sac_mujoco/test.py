import torch
from rlhive.rl_envs import RoboHiveEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.envs.transforms import RewardScaling, TransformedEnv, FlattenObservation, Compose
from torchrl.envs import TransformedEnv, R3MTransform
from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    ObservationNorm,
)

def make_transformed_env(
    env,
    stats=None,
):
    env = TransformedEnv(env, Compose(R3MTransform('resnet50', in_keys=["pixels"], download=True), FlattenObservation(-2, -1, in_keys=["r3m_vec"])))
    return env

base_env = RoboHiveEnv("visual_franka_slide_random-v3", device=torch.device('cuda:0'))
env = base_env
env = make_transformed_env(base_env)
print(env)
with torch.no_grad(), set_exploration_mode("random"):
    td = env.reset()
    td = env.rand_step()
    print(td)
