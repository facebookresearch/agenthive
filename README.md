# RLHive

RLHive provides the primitives and helpers for a seamless usage of robohive within TorchRL.

## Overview
RLHive provides the tools for offline and online execution

### Environment wrappers

RLHive provides environment wrappers specifically designed to work with RoboHive
gym environment.
Find examples in `test/test_envs.py`.

### Model training

One can train a model in state-only or with pixels.
When using pixels, the current API covers the R3M pipeline. Using a plain CNN is
an upcoming feature.
The entry point for all model trainings is `rlhive/sim_algos/run.py`.

#### State-only

As of now, one needs to specify the model and exploration method.
```bash
python run.py +model=sac +exploration=gaussian
```

#### Mixed State-Pixel

For state-pixel training, the current API just consists in indicating the environment
with the `v2d` flag. This will ensure that the images are loaded. The helper functions
will ensure that the R3M transform is stacked onto the environment.

```bash
python run.py +model=sac +exploration=gaussian env_name=FrankaReachRandom_v2d-v0
```

### Executing model on hardware

#### Retrieving model and environment
We save the config in `path/to/save/cfg.pt` (yaml text file) and `path/to/save/trainer.pt`.
The trainer contains the state_dict from the loss module and running environment.
Indeed, environment have their own state dict as they may have normalizing constants that
are seed-dependent, hence it is important to retrieve them and load them when testing
a trained model.
The file `rlhive/sim_algos/load.py` contains the utils to do this.

WARNING: Loading the weights from the _loss_ onto the _policy_ is an ad-hoc operation that
may vary from architecture to architecture and from loss to loss. The current code may break
in specific cases.

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
You will also need to install functorch. As of now, the best way to get the matching functorch version is to run the following script:


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
