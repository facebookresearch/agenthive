"""
Job script to learn policy using BC
"""

import os
import time
import copy
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
from misc import control_seed, NpEncoder, bcolors, \
        stack_tensor_dict_list
from torchrl.record.loggers import WandbLogger
from robohive.logger.grouped_datasets import Trace as Trace
from robohive.envs.env_base import MujocoEnv

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

class ObservationWrapper:
    def __init__(self, env_name, visual_keys, encoder):
        self.env = gym.make(env_name, visual_keys=visual_keys)
        self.horizon = self.env.horizon
        assert len(self.env.visual_keys) == 1 or len(self.env.visual_keys) == 0, f"Wrapper supports only envs with \
                        no visual keys or one. Current visual keys {self.env.visual_keys}"


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.get_obs(obs)

    def step(self, action):
        observation, reward, terminated, info = self.env.step(action)
        return self.get_obs(observation), reward, terminated, info

    def get_obs(self, observation=None):
        if len(self.env.visual_keys) > 0:
            visual_obs = self.env.get_exteroception()
            visual_obs = visual_obs[self.env.visual_keys[0]].reshape(-1)
            _, proprio_vec, _ = self.env.get_proprioception()
            observation = np.concatenate((visual_obs, proprio_vec))
        else:
            observation = self.env.get_obs() if observation is None else observation
        return observation

    def seed(self, seed):
        return self.env.seed(seed)

    def set_env_state(self, state_dict):
        return self.env.set_env_state(state_dict)

    def evaluate_success(self, paths, logger=None):
        return self.env.evaluate_success(paths, logger=logger)


def make_env(env_name, cam_name, encoder, from_pixels):
    if from_pixels:
        visual_keys = []
        assert encoder in ["vc1s", "vc1l", "r3m18", "rrl18", "2d", "1d"]
        if encoder == "1d" or encoder == "2d":
            visual_keys = [f'rgb:{cam_name}:84x84:{encoder}']
        else:
            visual_keys = [f'rgb:{cam_name}:224x224:{encoder}']
        env = ObservationWrapper(env_name, visual_keys=visual_keys, encoder=encoder)
    else:
        env = gym.make(env_name)
    return env

@hydra.main(config_name="bc.yaml", config_path="config")
def main(job_data: DictConfig):
    exp_start  = time.time()
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

    env = make_env(
            env_name=job_data["env_name"],
            cam_name=job_data["cam_name"],
            encoder=job_data["encoder"],
            from_pixels=job_data["from_pixels"]
    )
    # ===============================================================================
    # Setup functions and environment
    # ===============================================================================
    control_seed(SEED)
    env.seed(SEED)
    paths_trace = Trace.load(job_data['data_file'])
    #/home/rutavms/data/robohive/FK1-v4/FK1_Knob1OnRandom_v2d-v4/seed1/FK1_Knob1OnRandom_v2d-v4_trace.h5

    bc_paths = []
    for key, path in paths_trace.items():
        path_dict = {}
        traj_len = path['observations'].shape[0]
        obs_list = []
        ep_reward = 0.0
        env.reset()
        init_state_dict = {}
        t0 = time.time()
        for key, value in path['env_infos']['state'].items():
            init_state_dict[key] = value[0]
        env.set_env_state(init_state_dict)
        obs = env.get_obs()
        for step in range(traj_len):
            next_obs, reward, done, env_info = env.step(path["actions"][step])
            ep_reward += reward
            obs_list.append(obs)
            obs = next_obs
        t1 = time.time()
        obs_np = np.stack(obs_list, axis=0)
        path_dict['observations'] = obs_np[:-1]
        path_dict['actions'] = path['actions'][()][:-1]
        print(f"Time to convert one trajectory: {(t1-t0)/60:4.2f}")
        print("Converted episode reward:", ep_reward)
        print("Original episode reward:", np.sum(path["rewards"]))
        print(key, path_dict['observations'].shape, path_dict['actions'].shape)
        bc_paths.append(path_dict)

    observation_dim = bc_paths[0]['observations'].shape[-1]
    action_dim = bc_paths[0]['actions'].shape[-1]
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
    policy.to(job_data['device'])
    #evaluate_policy(
    #                        policy=policy,
    #                        env=env,
    #                        epoch=0,
    #                        num_episodes=job_data['eval_traj'],
    #                        device='cpu', ## has to be one cpu??
    #                        eval_logger=logger,
    #                        seed=job_data['seed'] + 123,
    #                        verbose=False,
    #)

    bc_agent = BC(
                    bc_paths,
                    policy,
                    epochs=job_data['bc_epochs'],
                    batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_lr'],
                    loss_type='MSE',
                    save_logs=True,
                    logger=logger,
                    set_transforms=True
    )
    bc_agent.train()
    # bc_agent.train_h5()

    _, success_rate =  evaluate_policy(
                            policy=policy,
                            env=env,
                            epoch=job_data['bc_epochs'],
                            num_episodes=job_data['eval_traj'],
                            device='cpu', ## has to be one cpu??
                            eval_logger=logger,
                            seed=job_data['seed'] + 123,
                            verbose=True,
    )

    exp_end = time.time()
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}Success Rate: {success_rate}. Time: {(exp_end - exp_start)/60:4.2f} minutes.{bcolors.ENDC}")

    # pickle.dump(bc_agent, open(OUT_DIR + '/iterations/agent_final.pickle', 'wb'))
    pickle.dump(policy, open(OUT_DIR + '/iterations/policy_final.pickle', 'wb'))

if __name__ == '__main__':
    main()
