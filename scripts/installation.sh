export MJENV_LIB_PATH="robohive"

python3 -mpip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

here=$(pwd)
git clone -c submodule.robohive/simhive/myo_sim.update=none --branch v0.4dev  --recursive https://github.com/vikashplus/robohive.git $MJENV_LIB_PATH
cd $MJENV_LIB_PATH
python3 -mpip install .  # one can also install it locally with the -e flag
cd $here

python3 -mpip install git+https://github.com/pytorch-labs/tensordict  # or stable or nightly with pip install tensordict(-nightly)
python3 -mpip install git+https://github.com/pytorch/rl.git  # or stable or nightly with pip install torchrl(-nightly)
pip install wandb moviepy
pip install hydra-submitit-launcher --upgrade
