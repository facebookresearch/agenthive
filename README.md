# RLHive

RLHive provides the primitives and helpers for a seamless usage of robohive within TorchRL.

## Overview
RLHive provides the tools for offline and online execution

### Environment wrappers

RLHive provides environment wrappers specifically designed to work with RoboHive
gym environment.
Find examples in `test/test_envs.py`.

The basic usage is:
```python
env = RoboHiveEnv(env_name="FrankaReachRandom_v2d-v0")
```

The following `kitchen` and `franka` visual environments should be used (they will be executed without flattening/unflattening of
the images which is an expensive process):
```python
env_list = ["visual_franka_slide_random-v3",
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
   "visual_kitchen_close-v3"
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
from rlhive.rl_envs import RoboHiveEnv
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

*UNSTABLE*: One can train a model in state-only or with pixels.
When using pixels, the current API covers the R3M pipeline. Using a plain CNN is
an upcoming feature.
The entry point for all model trainings is `rlhive/sim_algos/run.py`.

#### State-only

As of now, one needs to specify the model and exploration method.
```bash
python run.py +model=sac +exploration=gaussian
```

## Exexution

RLHive is optimized for the `MUJOCO` backend. Make sure to set the `sim_backend` environment variable to `"MUJOCO"`
before running the code:
```
sim_backend=MUJOCO python script.py
```

## Installation
RLHive has two core dependencies: torchrl and RoboHive. RoboHive relies on mujoco 
and mujoco-py for physics simulation and rendering. As of now, RoboHive requires 
you to use the old mujoco bindings as well as the v0.13 of gym.
TorchRL provides [detailed instructions](https://github.com/facebookresearch/rl/pull/375) 
on how to setup an environment with the old mujoco bindings.

### Create conda env
```bash
$ conda create -n rlhive python=3.9
$ conda activate rlhive
```

### Installing TorchRL
```bash
$ # cuda 11.6
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
For other cuda versions or nightly builds, check the [torch installation instructions](https://pytorch.org/get-started/locally/).
To install torchrl, run
```
pip install torchrl-nightly
```
or install directly from github:
```
pip install git+https://github.com/pytorch/rl
```

### Installing RoboHive
First clone the repo and install it locally:
```bash
$ cd path/to/root
$ # follow the getting started instructions: https://github.com/vikashplus/mj_envs/tree/v0.3dev#getting-started
$ cd mj_envs
$ git checkout v0.3dev
$ git clone -c submodule.mj_envs/sims/neuromuscular_sim.update=none --branch v0.3dev --recursive https://github.com/vikashplus/mj_envs.git
$ cd mj_envs
$ pip install -e .
```

For more complete instructions, check the installation pipeline in `.circleci/unittest/linux/script/install.sh`
