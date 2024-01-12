# AgentHive

AgentHive is the agents module for [RoboHive](https://sites.google.com/view/robohive). It contains trained agents as well as the primitives and helper scripts to train new agents for RoboHive environments.

## Overview
AgentHive provides the tools and helper scripts for training agents as well as offline and online execution of [pre-packaged trained agents](agents). RoboHive can be used with any openAI-Gym compatible algorithmic baselines to train agents for its environment. RoboHive developers have used the following baseline frameworks during developments. The with a goal of expanding over time.
1. [TorchRL](https://pytorch.org/rl/)
2. [mjRL](https://github.com/aravindr93/mjrl)
3. [Stable Baselines](https://github.com/DLR-RM/stable-baselines3)

## Pretrained baselines
AgentHive comes prepackaged with as a set of pre-trained baselines. The goal of these baselines is to provide a mechanism for used to enjoy out-of-box capabilities wit RoboHive. We are continuously accepting contributions to grow our pre-trained collections, please send us a pull request.
- [Visual Imitation Learning Baselines](scripts)
- [Natural Policy Gradients Baselines](agents)
- [Trajectory Optimization](agents)


## Agent Utilities

### Environment wrappers

AgentHive provides environment wrappers specifically designed to work with RoboHive
gym environment.
Find examples in `test/test_envs.py`.

The basic usage is:
```python
import robohive
import rlhive.envs
from torchrl.envs import RoboHiveEnv
env = RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
```

The following `kitchen` and `franka` visual environments should be used (they will be executed without flattening/unflattening of
the images which is an expensive process):
```python
env_list = [
    "visual_motorFingerReachFixed-v0",
    "visual_door-v1",
    "visual_hammer-v1",
    "visual_pen-v1",
    "visual_relocate-v1",
    "visual_franka_slide_random-v3",
    "visual_franka_slide_close-v3",
    "visual_franka_slide_open-v3",
    "visual_franka_micro_random-v3",
    "visual_franka_micro_close-v3",
    "visual_franka_micro_open-v3",
    "visual_FK1_Knob1OffRandom-v4",
    "visual_FK1_Knob1OnRandom-v4",
    "visual_FK1_Knob2OffRandom-v4",
    "visual_FK1_Knob2OnRandom-v4",
    "visual_FK1_Knob3OffRandom-v4",
    "visual_FK1_Knob3OnRandom-v4",
    "visual_FK1_Knob4OffRandom-v4",
    "visual_FK1_Knob4OnRandom-v4",
    "visual_FK1_LightOffRandom-v4",
    "visual_FK1_LightOnRandom-v4",
    "visual_FK1_SdoorCloseRandom-v4",
    "visual_FK1_SdoorOpenRandom-v4",
    "visual_FK1_LdoorCloseRandom-v4",
    "visual_FK1_LdoorOpenRandom-v4",
    "visual_FK1_RdoorCloseRandom-v4",
    "visual_FK1_RdoorOpenRandom-v4",
    "visual_FK1_MicroOpenRandom-v4",
    "visual_FK1_MicroCloseRandom-v4",
    "visual_FK1_RelaxRandom-v4",
]
```

To use the environment in parallel, wrap it in a `ParallelEnv`:
```python
from torchrl.envs import EnvCreator, ParallelEnv
env = ParallelEnv(3, EnvCreator(lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")))
```

To use transforms (normalization, grayscale etc), use the env transforms:
```python
from torchrl.envs import EnvCreator, ParallelEnv, TransformedEnv, R3MTransform
env = ParallelEnv(3, EnvCreator(lambda: RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")))
env = TransformedEnv(
    base_env,
    R3MTransform(
        "resnet18",
        ["pixels"],
        ["pixels_embed"],
    ),
)
```
Make sure that the R3M or VIP transform is appended after the ParallelEnv, otherwise you will
pass as many images as there are processes through the ResNet module (and quickly run into an OOM
exception).

Finally, the script of a typical data collector (executed on 4 different GPUs in an asynchronous manner) reads
as follows:
```python
import tqdm
from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy
from agenthive.rl_envs import RoboHiveEnv
from torchrl.envs import ParallelEnv, TransformedEnv, GrayScale, ToTensorImage, Resize, ObservationNorm, EnvCreator, Compose, CatFrames

if __name__ == '__main__':
    # create a parallel env with 4 envs running independendly.
    # I put the 'cuda:0' device to show how to create an env on cuda (ie: the output tensors will be on cuda)
    # but this will be overwritten in the collector below
    penv = ParallelEnv(4, EnvCreator(lambda: RoboHiveEnv('FrankaReachRandom_v2d-v0', device='cuda:0', from_pixels=True)))
    # we append a series of standard transforms, all running on cuda
    tenv = TransformedEnv(penv, Compose(ToTensorImage(), Resize(84, 84), GrayScale(), CatFrames(4, in_keys=['pixels']), ObservationNorm(in_keys=['pixels'])))
    # this is how you initialize your observation norm transform (the API will be improved shortly)
    tenv.transform[-1].init_stats(reduce_dim=(0, 1), cat_dim=1, num_iter=1000)
    # we cheat a bit by using a totally random policy. A CNN will obviously slow down collection a bit
    policy = RandomPolicy(tenv.action_spec)  # some random policy

    # we create an async collector on 4 different devices. The "passing_devices"  indicate where the env is placed, and the "device" where the policy is executed.
    # For a maximum efficiency they should match. Also, you can either pass a string for those args (ie all devices match) or a list of strings/devices.
    collector = MultiaSyncDataCollector([tenv, tenv, tenv, tenv], policy=policy, frames_per_batch=400, max_frames_per_traj=1000, total_frames=1_000_000,
                                        passing_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                                        devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    # a simple collection loop to log the speed
    pbar = tqdm.tqdm(total=1_000_000)
    for data in collector:
        pbar.update(data.numel())
    del collector
    del tenv

```

### Model training

The safest way of coding up a model and training it is to refer to the official
torchrl examples:
- [torchrl](https://github.com/pytorch/rl/tree/main/examples)
- [torchrl_examples](https://github.com/compsciencelab/torchrl_examples)

## Execution

AgentHive is optimized for the `MUJOCO` backend. Make sure to set the `sim_backend` environment variable to `"MUJOCO"`
before running the code:
```
sim_backend=MUJOCO python script.py
```

## Installation
AgentHive has two core dependencies: torchrl and RoboHive. RoboHive relies on mujoco
and mujoco-py for physics simulation and rendering. As of now, RoboHive requires
you to use the old mujoco bindings as well as the v0.13 of gym.
TorchRL provides [detailed instructions](https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html#installing-mujoco).
on how to setup an environment with the old mujoco bindings.

See also the [Getting Started](GET_STARTED.md) markdown for more info on setting up your env.

For more complete instructions, check the installation pipeline in `.circleci/unittest/linux/script/install.sh`
