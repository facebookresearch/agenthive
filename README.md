# RLHive

RLHive provides the primitives and helpers for a seamless usage of robohive within TorchRL.

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