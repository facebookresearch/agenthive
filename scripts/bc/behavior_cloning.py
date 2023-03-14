"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from logger import DataLog
from tqdm import tqdm

from misc import tensorize

class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 logger = None,
                 set_transforms = False,
                 *args, **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.device = self.policy.device
        assert (self.loss_type == 'MSE' or self.loss_type == 'MLE')

        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            #self.set_variance_with_data(out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

        # make logger
        if self.save_logs:
            self.logger = logger or DataLog()

    def compute_transformations(self):
        # get transformations
        if self.expert_paths == [] or self.expert_paths is None:
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            if type(self.expert_paths) is list:
                observations = np.concatenate([path["observations"] for path in self.expert_paths])
                actions = np.concatenate([path["actions"] for path in self.expert_paths])
            else: # 'h5py._hl.files.File'
                observations = np.concatenate([self.expert_paths[k]['observations'][0] for k in self.expert_paths.keys()])
                actions = np.concatenate([self.expert_paths[k]['actions'][0] for k in self.expert_paths.keys()])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        self.policy.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        out_scale = tensorize(out_scale, device=self.policy.device)
        data_log_std = torch.log(out_scale + 1e-3)
        self.policy.set_log_std(data_log_std)

    def loss(self, data, idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        mu, LL = self.policy.mean_LL(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None):
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx]
        act_expert = tensorize(act_expert, device=self.policy.device)
        act_pi = self.policy.forward(obs)
        return self.loss_criterion(act_pi, act_expert.detach())

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        num_samples = data["observations"].shape[0]

        # log stats before
        if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).to('cpu').data.numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)
            print('BC loss before', loss_val)

        # train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)

        # log stats after
        if self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            loss_val = self.loss(data, idx=range(num_samples)).to('cpu').data.numpy().ravel()[0]
            self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))
            print('BC val loss', loss_val)

    def train(self, **kwargs):
        if not hasattr(self, 'data'):
            observations = np.concatenate([path["observations"] for path in self.expert_paths])
            expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
            observations = tensorize(observations, device=self.policy.device)
            expert_actions = tensorize(expert_actions, self.policy.device)
            self.data = dict(observations=observations, expert_actions=expert_actions)
        self.fit(self.data, **kwargs)

    def train_h5(self, **kwargs):
        if not hasattr(self, 'data'):
            observations = np.concatenate([self.expert_paths[k]['observations'][0] for k in self.expert_paths.keys()])
            expert_actions = np.concatenate([self.expert_paths[k]['actions'][0] for k in self.expert_paths.keys()])
            observations = tensorize(observations, device=self.policy.device)
            expert_actions = tensorize(expert_actions, self.policy.device)
            self.data = dict(observations=observations, expert_actions=expert_actions)
        self.fit(self.data, **kwargs)



def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)

