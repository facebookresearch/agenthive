## Installation 
```
git clone --branch=sac_dev https://github.com/facebookresearch/rlhive.git
conda create -n rlhive -y python=3.8
conda activate rlhive
bash rlhive/scripts/installation.sh
cd rlhive
pip install -e .
```

## Testing installation
```
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

## Launching experiments
[NOTE] Set ulimit for your shell (default 1024): `ulimit -n 4096`  
Set your slurm configs especially `partition` and `hydra.run.dir`   
Slurm files are located at `sac_mujoco/config/hydra/launcher/slurm.yaml` and `sac_mujoco/config/hydra/output/slurm.yaml`  
```
cd sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m hydra/launcher=slurm hydra/output=slurm
```

To run a small experiment for testing, run the following command:
```
cd sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m total_frames=2000 init_random_frames=25 buffer_size=2000 hydra/launcher=slurm hydra/output=slurm
```
