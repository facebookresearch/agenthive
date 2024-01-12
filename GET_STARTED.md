
# Getting started

## Installing dependencies

The following code snippet installs the nightly versions of the libraries. For a faster installation, simply install `torchrl` and `tensordict` using `pip`.

```shell
module load cuda/12.1 # if available

export MJENV_LIB_PATH="robohive"
conda create -n agenthive -y python=3.8
conda activate agenthive

pip3 install torch torchvision torchaudio
python3 -mpip install wandb 'robohive[mujoco, encoders]'  # installing robohive along with visual encoders
python3 -mpip install tensordict torchrl
python3 -mpip install git+https://github.com/facebookresearch/agenthive.git  # or stable or nightly with pip install torchrl(-nightly)

```

For more complete instructions, check the installation pipeline in `.circleci/unittest/linux/script/install.sh`

You can run these two commands to check that installation was successful:

```shell
python -c "import robohive"
MUJOCO_GL=egl sim_backend=MUJOCO python -c """
import robohive
from torchrl.envs import RoboHiveEnv
env_name = 'FrankaReachFixed-v0'
base_env = RoboHiveEnv(env_name)
print(base_env.rollout(3))

# check that the env specs are ok
from torchrl.envs.utils import check_env_specs
check_env_specs(base_env)
"""
```
