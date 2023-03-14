"""
Job script to learn policy using BC
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

import os
from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import pickle
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from logger import DataLog
from gaussian_mlp import MLP
from behavior_cloning import BC
from misc import control_seed, NpEncoder
from mj_envs.logger.grouped_datasets import Trace as Trace

@hydra.main(config_name="bc.yaml", config_path="config")
def main(job_data: DictConfig):
    OUT_DIR = os.getcwd()
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(OUT_DIR+'/iterations'): os.mkdir(OUT_DIR+'/iterations')
    if not os.path.exists(OUT_DIR+'/logs'): os.mkdir(OUT_DIR+'/logs')

    # Unpack args and make files for easy access
    logger = DataLog()
    ENV_NAME = job_data['env_name']
    EXP_FILE = OUT_DIR + '/job_data.yaml'
    SEED = job_data['seed']

    # base cases
    if 'device' not in job_data.keys(): job_data['device'] = 'cpu'
    assert 'data_file' in job_data.keys()

    yaml_config = OmegaConf.to_yaml(job_data)
    with open(EXP_FILE, 'w') as file: yaml.dump(yaml_config, file)

    # ===============================================================================
    # Setup functions and environment
    # ===============================================================================
    control_seed(SEED)
    paths_trace = Trace.load(job_data['data_file'])

    observation_dim = paths_trace['Rollout0']['observations'].shape[2]
    action_dim = paths_trace['Rollout0']['actions'].shape[2]
    print(f'Policy obs dim {observation_dim} act dim {action_dim}')
    policy = MLP(
                    None,
                    seed=SEED,
                    action_dim=action_dim,
                    observation_dim=observation_dim,
                    hidden_sizes=job_data['policy_size'],
                    init_log_std=job_data['init_log_std'],
                    min_log_std=job_data['min_log_std'],
    )

    # ===============================================================================
    # Model training
    # ===============================================================================
    print(f"{bcolors.OKBLUE}Training BC{bcolors.ENDC}")
    bc_paths = paths_trace
    policy.to(job_data['device'])
    bc_agent = BC(bc_paths, policy,
        epochs=job_data['bc_init_epoch'], batch_size=128, lr=1e-4, loss_type='MSE', save_logs=True, logger=logger,
        set_transforms=True)
    # bc_agent.train()
    bc_agent.train_h5()

    # pickle.dump(bc_agent, open(OUT_DIR + '/iterations/agent_final.pickle', 'wb'))
    pickle.dump(policy, open(OUT_DIR + '/iterations/policy_final.pickle', 'wb'))

if __name__ == '__main__':
    main()
