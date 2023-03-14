"""
Job script to learn policy using BC
"""

import os
from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'
import pickle
import yaml
import hydra
import gym
import torch
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf

from gaussian_mlp import MLP
from behavior_cloning import BC
from misc import control_seed, NpEncoder, bcolors, stack_tensor_dict_list
from torchrl.record.loggers import WandbLogger
from mj_envs.logger.grouped_datasets import Trace as Trace

def evaluate_policy(
            policy,
            env,
            num_episodes,
            epoch,
            horizon=None,
            gamma=1,
            percentile=[],
            get_full_dist=False,
            eval_logger=None,
            device='cpu',
            seed=123,
            verbose=True,
    ):
    env.seed(seed)
    horizon = env.horizon if horizon is None else horizon
    mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
    ep_returns = np.zeros(num_episodes)
    policy.eval()
    paths = []

    for ep in range(num_episodes):
        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []
        o = env.reset()
        t, done = 0, False
        while t < horizon and (done == False):
            a = policy.get_action(o)[1]['evaluation']
            next_o, r, done, env_info = env.step(a)
            ep_returns[ep] += (gamma ** t) * r
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(None)
            env_infos.append(env_info)
            o = next_o
            t += 1
        if verbose:
            print("Episode: {}; Reward: {}".format(ep, ep_returns[ep]))
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            #agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list(env_infos),
            terminated=done
        )
        paths.append(path)

    mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
    min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
    base_stats = [mean_eval, std, min_eval, max_eval]

    percentile_stats = []
    for p in percentile:
        percentile_stats.append(np.percentile(ep_returns, p))

    full_dist = ep_returns if get_full_dist is True else None
    success = env.evaluate_success(paths, logger=None) ## Don't use the mj_envs logging function

    if not eval_logger is None:
        rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
        rwd_dense = np.mean([np.sum(p['env_infos']['rwd_dense'])/env.horizon for p in paths]) # return rwd/step
        eval_logger.log_scalar('eval/rwd_sparse', rwd_sparse, step=epoch)
        eval_logger.log_scalar('eval/rwd_dense', rwd_dense, step=epoch)
        eval_logger.log_scalar('eval/success', success, step=epoch)
    return [base_stats, percentile_stats, full_dist], success

@hydra.main(config_name="bc.yaml", config_path="config")
def main(job_data: DictConfig):
    OUT_DIR = os.getcwd()
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    if not os.path.exists(OUT_DIR+'/iterations'): os.mkdir(OUT_DIR+'/iterations')
    if not os.path.exists(OUT_DIR+'/logs'): os.mkdir(OUT_DIR+'/logs')

    exp_name = OUT_DIR.split('/')[-1] ## TODO: Customizer for logging
    # Unpack args and make files for easy access
    #logger = DataLog()
    logger = WandbLogger(
        exp_name=exp_name,
        config=job_data,
        name=job_data['env_name'],
        project=job_data['wandb_project'],
        entity=job_data['wandb_entity'],
        mode=job_data['wandb_mode'],
    )


    ENV_NAME = job_data['env_name']
    EXP_FILE = OUT_DIR + '/job_data.yaml'
    SEED = job_data['seed']

    # base cases
    if 'device' not in job_data.keys(): job_data['device'] = 'cpu'
    assert 'data_file' in job_data.keys()

    yaml_config = OmegaConf.to_yaml(job_data)
    with open(EXP_FILE, 'w') as file: yaml.dump(yaml_config, file)

    env = gym.make(job_data["env_name"])
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
    evaluate_policy(
                            policy=policy,
                            env=env,
                            epoch=0,
                            num_episodes=job_data['eval_traj'],
                            device='cpu', ## has to be one cpu??
                            eval_logger=logger,
                            seed=job_data['seed'] + 123,
                            verbose=False,
    )

    bc_agent = BC(
                    bc_paths,
                    policy,
                    epochs=job_data['bc_init_epoch'],
                    batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_lr'],
                    loss_type='MSE',
                    save_logs=True,
                    logger=logger,
                    set_transforms=True
    )
    # bc_agent.train()
    bc_agent.train_h5()

    _, success_rate =  evaluate_policy(
                            policy=policy,
                            env=env,
                            epoch=job_data['bc_init_epoch'],
                            num_episodes=job_data['eval_traj'],
                            device='cpu', ## has to be one cpu??
                            eval_logger=logger,
                            seed=job_data['seed'] + 123,
                            verbose=True,
    )

    print(f"{bcolors.BOLD}{bcolors.OKGREEN}Success Rate: {success_rate}{bcolors.ENDC}")

    # pickle.dump(bc_agent, open(OUT_DIR + '/iterations/agent_final.pickle', 'wb'))
    pickle.dump(policy, open(OUT_DIR + '/iterations/policy_final.pickle', 'wb'))

if __name__ == '__main__':
    main()
