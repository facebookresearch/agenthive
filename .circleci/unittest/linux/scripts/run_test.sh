#!/usr/bin/env bash

set -e

yum makecache
yum install -y glfw
yum install -y glew
yum install -y glew-devel
yum install -y mesa-libGL
yum install -y mesa-libGL-devel
yum install -y mesa-libOSMesa
yum install -y mesa-libOSMesa-devel
yum install -y glx-utils
yum -y install egl-utils
yum -y install freeglut

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch

MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_arms.py
MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_claws.py
MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_envs.py
MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_fm.py
MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_hand_manipulation_suite.py
MUJOCO_GL=egl sim_backend=MUJOCO python ./third_party/mj_envs/mj_envs/tests/test_multitask.py
MUJOCO_GL=egl sim_backend=MUJOCO python test/test_envs.py
