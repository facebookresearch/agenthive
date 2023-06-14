#!/bin/zsh

# Instructions to install a fresh anaconda environment with RLHive

set -e

conda_path=$(conda info | grep -i 'base environment' | awk '{ print $4 }')
source $conda_path/etc/profile.d/conda.sh

here=$(pwd)
module_path=$HOME/modules/

module purge
module load cuda/11.6

conda env remove -n rlhive -y

conda create -n rlhive -y python=3.8

conda activate rlhive

python3 -mpip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

mkdir $module_path
cd $module_path
#git clone -c submodule.robohive/sims/neuromuscular_sim.update=none --branch v0.4dev --recursive https://github.com/vikashplus/robohive.git robohive
git clone --branch v0.5dev --recursive https://github.com/vikashplus/robohive.git robohive
cd robohive
python3 -mpip install .  # one can also install it locally with the -e flag
cd $here

python3 -mpip install git+https://github.com/pytorch-labs/tensordict  # or stable or nightly with pip install tensordict(-nightly)
python3 -mpip install git+https://github.com/pytorch/rl.git  # or stable or nightly with pip install torchrl(-nightly)

# this
# python3 -mpip install git+https://github.com/facebookresearch/rlhive.git  # or stable or nightly with pip install torchrl(-nightly)
# or this
cd ../..
pip install -e .
cd $here

pip install wandb tqdm hydra-core moviepy
