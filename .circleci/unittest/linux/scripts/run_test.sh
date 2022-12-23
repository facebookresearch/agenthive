#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

#yum makecache
#yum install -y glfw
#yum install -y glew
#yum install -y glew-devel
#yum install -y mesa-libGL
#yum install -y mesa-libGL-devel
#yum install -y mesa-libOSMesa
#yum install -y mesa-libOSMesa-devel
#yum install -y glx-utils
#yum -y install egl-utils
#yum -y install freeglut

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
PRIVATE_MUJOCO_GL=egl
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export DISPLAY=unix:0.0
export MUJOCO_GL=$PRIVATE_MUJOCO_GL
export PYOPENGL_PLATFORM=$PRIVATE_MUJOCO_GL
export sim_backend=MUJOCO

#python ./third_party/mj_envs/mj_envs/tests/test_arms.py
#python ./third_party/mj_envs/mj_envs/tests/test_claws.py
#python ./third_party/mj_envs/mj_envs/tests/test_envs.py
#python ./third_party/mj_envs/mj_envs/tests/test_fm.py
#python ./third_party/mj_envs/mj_envs/tests/test_hand_manipulation_suite.py
#python ./third_party/mj_envs/mj_envs/tests/test_multitask.py
python test/test_envs.py
