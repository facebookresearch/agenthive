#!/usr/bin/env bash

set -e

yum makecache
yum install -y glfw
yum install -y glew
yum install -y mesa-libGL
yum install -y mesa-libGL-devel
yum install -y mesa-libOSMesa-devel
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

MUJOCO_GL=egl python run ./third_party/mj_envs/mj_envs/tests/test_all.py
