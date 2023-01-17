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
cd scripts/sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m hydra/launcher=slurm hydra/output=slurm
```

To run a small experiment for testing, run the following command:
```
cd scripts/sac_mujoco
sim_backend=MUJOCO MUJOCO_GL=egl python sac.py -m total_frames=2000 init_random_frames=25 buffer_size=2000 hydra/launcher=slurm hydra/output=slurm
```

## Parameter Sweep
1. R3M and RRL experiments: `visual_transform=r3m,rrl`  
2. Multiple seeds: `seed=42,43,44`  
3. List of environments: 
  ```
task=visual_franka_slide_random-v3,\  
     visual_franka_slide_close-v3,\  
     visual_franka_slide_open-v3,\  
     visual_franka_micro_random-v3,\  
     visual_franka_micro_close-v3,\  
     visual_franka_micro_open-v3,\  
     visual_kitchen_knob1_off-v3,\  
     visual_kitchen_knob1_on-v3,\  
     visual_kitchen_knob2_off-v3,\  
     visual_kitchen_knob2_on-v3,\  
     visual_kitchen_knob3_off-v3,\  
     visual_kitchen_knob3_on-v3,\  
     visual_kitchen_knob4_off-v3,\  
     visual_kitchen_knob4_on-v3,\  
     visual_kitchen_light_off-v3,\  
     visual_kitchen_light_on-v3,\  
     visual_kitchen_sdoor_close-v3,\  
     visual_kitchen_sdoor_open-v3,\  
     visual_kitchen_ldoor_close-v3,\  
     visual_kitchen_ldoor_open-v3,\  
     visual_kitchen_rdoor_close-v3,\  
     visual_kitchen_rdoor_open-v3,\  
     visual_kitchen_micro_close-v3,\  
     visual_kitchen_micro_open-v3,\  
     visual_kitchen_close-v3  
  ```
