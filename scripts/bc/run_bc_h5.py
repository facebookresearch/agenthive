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

from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
import mjrl.envs
import time as timer
import argparse
import os
import json
from behavior_cloning import BC


from tqdm import tqdm
from tabulate import tabulate
from gaussian_mlp import MLP
# from mjrl.utils.gym_env import GymEnv
from logger import DataLog
from misc import parse_overrides

# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Model accelerated policy optimization.')
parser.add_argument('--output', '-o', type=str, required=True, help='location to store results')
parser.add_argument('--config', '-c', type=str, required=True, help='path to config file with exp params')
parser.add_argument('--include', '-i', type=str, required=False, help='package to import')
args, overrides = parser.parse_known_args()

OUT_DIR = args.output
if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
if not os.path.exists(OUT_DIR+'/iterations'): os.mkdir(OUT_DIR+'/iterations')
if not os.path.exists(OUT_DIR+'/logs'): os.mkdir(OUT_DIR+'/logs')
with open(args.config, 'r') as f:
    job_data = eval(f.read())
    print('Overriding parameters', overrides)
    job_data = parse_overrides(job_data, overrides)
if args.include: exec("import "+args.include)

# Unpack args and make files for easy access
logger = DataLog()
ENV_NAME = job_data['env_name']
EXP_FILE = OUT_DIR + '/job_data.json'
SEED = job_data['seed']

# base cases
if 'device' not in job_data.keys():         job_data['device'] = 'cpu'
assert 'data_file' in job_data.keys()
with open(EXP_FILE, 'w') as f:  json.dump(job_data, f, indent=4)
del(job_data['seed'])
job_data['base_seed'] = SEED

# ===============================================================================
# Setup functions and environment
# ===============================================================================

np.random.seed(SEED)
torch.random.manual_seed(SEED)

from mj_envs.logger.grouped_datasets import Trace as Trace
paths_trace = Trace.load(job_data['data_file'])

from franka import FrankaEnv
# e = FrankaEnv(obs_dim=paths_trace['Rollout0']['observations'].shape[2],\
#                 action_dim=paths_trace['Rollout0']['actions'].shape[2])

observation_dim = paths_trace['Rollout0']['observations'].shape[2]
action_dim = paths_trace['Rollout0']['actions'].shape[2]

# if ENV_NAME == '':
#     from franka import FrankaEnv
#     e = FrankaEnv(obs_dim=paths_trace['Rollout0']['observations'].shape[2],\
#                   action_dim=paths_trace['Rollout0']['actions'].shape[2])
# elif ENV_NAME == 'franka-vel':
#     from franka import FrankaVelEnv
#     e = FrankaVelEnv()
# elif ENV_NAME.split('_')[0] == 'dmc':
#     # import only if necessary (not part of package requirements)
#     import dmc2gym
#     backend, domain, task = ENV_NAME.split('_')
#     e = dmc2gym.make(domain_name=domain, task_name=task, seed=SEED)
#     e = GymEnv(e, act_repeat=job_data['act_repeat'])
# else:
#     e = GymEnv(ENV_NAME, act_repeat=job_data['act_repeat'])
#     e.set_seed(SEED)


# ===============================================================================
# Setup policy, model, and agent
# ===============================================================================

# Construct policy and set exploration level correctly for NPG
# print(f'Policy obs dim {e.observation_dim} act dim {e.action_dim}')
# policy = MLP(e.spec, seed=SEED, hidden_sizes=job_data['policy_size'],
#                 init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'],
#                 observation_dim=e.observation_dim,
#                     action_dim=e.action_dim)

print(f'Policy obs dim {observation_dim} act dim {action_dim}')
policy = MLP(None, seed=SEED, hidden_sizes=job_data['policy_size'],
                init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'],
                observation_dim=observation_dim,
                    action_dim=action_dim)

# baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=256, epochs=1,  learn_rate=1e-3,
#                        device=job_data['device'],
#                        observation_dim=e.observation_dim,
#                        action_dim=e.action_dim)
# agent = ModelBasedNPG(learned_model=models, env=e, policy=policy, baseline=baseline, seed=SEED,
#                       normalized_step_size=job_data['step_size'], save_logs=True,
#                       reward_function=reward_function, termination_function=termination_function,
#                       **job_data['npg_hp'])

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
exit(0)