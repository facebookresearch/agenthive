
# Getting started

## Installing dependencies

The following code snippet installs the nightly versions of the libraries. For a faster installation, simply install `torchrl-nightly` and `tensordict-nightly`.

```shell
module load cuda/11.6 cudnn/v8.4.1.50-cuda.11.6

export MJENV_LIB_PATH="mj_envs"
conda create -n rlhive -y python=3.8
conda activate rlhive

python3 -mpip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

here=$(pwd)
git clone -c submodule.mj_envs/sims/neuromuscular_sim.update=none --branch v0.4dev --recursive https://github.com/vikashplus/mj_envs.git $MJENV_LIB_PATH
cd $MJENV_LIB_PATH
python3 -mpip install .  # one can also install it locally with the -e flag
cd $here

python3 -mpip install git+https://github.com/pytorch-labs/tensordict  # or stable or nightly with pip install tensordict(-nightly)
python3 -mpip install git+https://github.com/pytorch/rl.git  # or stable or nightly with pip install torchrl(-nightly)
python3 -mpip install git+https://github.com/facebookresearch/rlhive.git  # or stable or nightly with pip install torchrl(-nightly)

```

You can run these two commands to check that installation was successful:

```shell
python -c "import mj_envs"
MUJOCO_GL=egl sim_backend=MUJOCO python -c """
from rlhive.rl_envs import RoboHiveEnv
env_name = 'visual_franka_slide_random-v3'
base_env = RoboHiveEnv(env_name,)
print(base_env.rollout(3))

# check that the env specs are ok
from torchrl.envs.utils import check_env_specs
check_env_specs(base_env)
"""
```

## Build your environment (and collector)

Once you have installed the libraries and the sanity checks run, you can start using the envs.
Here's a step-by-step example of how to create an env, pass the output through R3M and create a data collector.
For more info, check the [torchrl environments doc](https://pytorch.org/rl/reference/envs.html).

```python
from rlhive.rl_envs import RoboHiveEnv
from torchrl.envs import ParallelEnv, TransformedEnv, R3MTransform
import torch

from torchrl.collectors.collectors import SyncDataCollector, MultiaSyncDataCollector, RandomPolicy
# make sure your ParallelEnv is inside the `if __name__ == "__main__":` condition, otherwise you'll
# be creating an infinite tree of subprocesses
if __name__ == "__main__":
    device = torch.device("cpu") # could be 'cuda:0'
    env_name = 'visual_franka_slide_random-v3'
    base_env = ParallelEnv(4, lambda: RoboHiveEnv(env_name, device=device))
    # build a transformed env with the R3M transform. The transform will be applied on a batch of data.
    # You can append other transforms by doing `env.append_transform(...)` if needed
    env = TransformedEnv(base_env, R3MTransform('resnet50', in_keys=["pixels"], download=True))
    assert env.device == device
    # example of a rollout
    print(env.rollout(3))

    # a simple, single-process data collector
    collector = SyncDataCollector(env, policy=RandomPolicy(env.action_spec), total_frames=1_000_000, frames_per_batch=200, init_random_frames=200, )
    for data in collector:
        print(data)

    # async multi-proc data collector
    collector = MultiaSyncDataCollector([env, env], policy=RandomPolicy(env.action_spec), total_frames=1_000_000, frames_per_batch=200, init_random_frames=200, )
    for data in collector:
        print(data)

```
