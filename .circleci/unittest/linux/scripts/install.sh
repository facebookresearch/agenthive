#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.


yum makecache
yum install -y glfw
yum install -y glew
yum install -y mesa-libGL
yum install -y mesa-libGL-devel
yum install -y mesa-libOSMesa-devel
yum -y install egl-utils
yum -y install freeglut

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

if [ "${CU_VERSION:-}" == cpu ] ; then
    version="cpu"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
    version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    # conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    # use pip to install pytorch as conda can frequently pick older release
#    conda install -y pytorch cpuonly -c pytorch-nightly
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
else
    pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu113
fi

# install tensordict
pip install git+https://github.com/pytorch-labs/tensordict

# smoke test
python -c "import functorch"

# mj_envs
git clone -c submodule.mj_envs/sims/neuromuscular_sim.update=none --branch non-local-install --recursive https://github.com/vmoens/mj_envs.git third_party/mj_envs
cd third_party/mj_envs
pip install -e .
cd ../..

# torchrl
git clone --branch main https://github.com/pytorch/rl.git third_party/torchrl
cd third_party/torchrl
pip install -e .
cd ../..

printf "* Installing rlhive\n"
pip3 install -e .

# smoke test
printf "* Smoke test: torchrl\n"
python -c "import torchrl"
printf "* Smoke test: mj_envs\n"
python -c "import mj_envs"
