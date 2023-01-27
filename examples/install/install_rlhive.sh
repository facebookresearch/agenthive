#!/bin/zsh

set -e

conda_path=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source $conda_path/etc/profile.d/conda.sh

here=$(pwd)
module_path=$HOME/modules/

conda env remove -n rlhive -y

conda create -n rlhive -y python=3.8


conda activate rlhive

python3 -mpip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

mkdir $module_path
cd $module_path
git clone -c submodule.mj_envs/sims/neuromuscular_sim.update=none --branch v0.4dev --recursive https://github.com/vikashplus/mj_envs.git mj_envs
cd mj_envs
python3 -mpip install .  # one can also install it locally with the -e flag
cd $here

python3 -mpip install git+https://github.com/pytorch-labs/tensordict  # or stable or nightly with pip install tensordict(-nightly)
python3 -mpip install git+https://github.com/pytorch/rl.git  # or stable or nightly with pip install torchrl(-nightly)
python3 -mpip install git+https://github.com/facebookresearch/rlhive.git  # or stable or nightly with pip install torchrl(-nightly)
