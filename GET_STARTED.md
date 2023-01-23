
# Getting started

## Installing dependencies

The following code snippet installs the nightly versions of the libraries. For a faster installation, simply install `torchrl-nightly` and `tensordict-nightly`.
However, we recommand using the `git` version as they will be more likely up-to-date with the latest features, and as we are
actively working on fine-tuning torchrl for RoboHive usage, keeping the latest version of the library may be beneficial.

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

## Designing experiments and logging values

TorchRL provides a series of wrappers around common loggers (tensorboard, mlflow, wandb etc).
We generally default to wandb.
Here are the details on how to set up your logger: wandb can work in one of two
modes: `online`, where you need an account and the machine you're running your experiment on must be
connected to the cloud, and `offline` where the logs are stored locally.
The latter is more general and easier to collect, hence we suggest you use this mode instead.
To configure and use your logger using TorchRL, procede as follows (notice that 
using the plain wandb API is very similar to this, TorchRL's conveniance just relies in the
interchangeability with other loggers):

```python
import argparse
import os

from torchrl.trainers.loggers import WandbLogger
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--total_frames", default=300, type=int)
parser.add_argument("--training_steps", default=3, type=int)
parser.add_argument("--wandb_exp_name", default="a2c")
parser.add_argument("--wandb_save_dir", default="./mylogs")
parser.add_argument("--wandb_project", default="rlhive")
parser.add_argument("--wandb_mode", default="offline",
                    choices=["online", "offline"])

if __name__ == "__main__":
    args = parser.parse_args()
    training_steps = args.training_steps
    if args.wandb_mode == "offline":
        # This will be integrated in torchrl
        dest_dir = args.wandb_save_dir
        os.makedirs(dest_dir, exist_ok=True)
    logger = WandbLogger(
        exp_name=args.wandb_exp_name,
        save_dir=dest_dir,
        project=args.wandb_project,
        mode=args.wandb_mode,
    )

    # we collect 3 frames in each batch
    collector = (torch.randn(3, 4, 0) for _ in range(args.total_frames // 3))
    total_frames = 0
    # main loop: collection of batches
    for batch in collector:
        for step in range(training_steps):
            pass
        total_frames += batch.shape[0]
        # We log according to the frames, which we believe is the less subject to experiment
        # hyperparameters
        logger.log_scalar("loss_value", torch.randn([]).item(),
                          step=total_frames)
        # one can log videos too! But custom steps do not work as expected :(
        video = torch.randint(255, (10, 11, 3, 64, 64))  # 10 videos of 11 frames, 64x64 pixels
        logger.log_video("demo", video)

```


This script will save your logs in `./mylogs`. Don't worry too much about `project` or `entity`, which can be [overwritten
at upload time](https://docs.wandb.ai/ref/cli/wandb-sync):

Once we'll have collected these logs, we will upload them to a wandb account using `wandb sync path/to/log --entity someone --project something`.

## What to log

In general, experiments should log the following items:
- dense reward (train and test)
- sparse reward (train and test)
- success perc (train and test)
- video: after every 1M runs or so, a test run should be performed. A video recorder should be appended
  to the test env to log the behaviour.
- number of training steps: since our "x"-axis will be the number of frames collected, keeping track of the
  training steps will help us interpolate one with the other.
- For behavioural cloning we should log the number of epochs instead.

## A more concrete example

TODO
