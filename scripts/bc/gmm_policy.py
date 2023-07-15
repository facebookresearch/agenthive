import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class GMMPolicy(nn.Module):
    def __init__(self,
                 # network_kwargs
                 input_size,
                 output_size,
                 hidden_size=1024,
                 num_layers=2,
                 min_std=0.0001,
                 num_modes=5,
                 activation="softplus",
                 low_eval_noise=False,
                 # loss_kwargs
                 loss_coef=1.0):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = [nn.BatchNorm1d(num_features=input_size)]
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer   = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

        self.trainable_params = list(self.share.parameters()) + \
                list(self.mean_layer.parameters()) + \
                list(self.logstd_layer.parameters()) + \
                list(self.logits_layer.parameters())

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward_fn(self, x):
        # x: (B, input_size)
        share  = self.share(x)
        means  = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means  = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(-1, self.num_modes, self.output_size)
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        o = torch.from_numpy(o).to(self.device)
        means, stds, logits = self.forward_fn(o)

        compo = D.Normal(loc=means, scale=stds)
        compo = D.Independent(compo, 1)
        mix   = D.Categorical(logits=logits)
        gmm   = D.MixtureSameFamily(mixture_distribution=mix,
                                    component_distribution=compo)
        action = gmm.sample()
        mean = gmm.mean
        mean = mean.detach().cpu().numpy().ravel()
        return [action, {'mean': mean, 'std': stds, 'evaluation': mean}]

    def forward(self, x):
        means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix   = D.Categorical(logits=logits)
        gmm   = D.MixtureSameFamily(mixture_distribution=mix,
                                    component_distribution=compo)
        return gmm

    def mean_LL(self, x, target):
        gmm_dist = self.forward(x)
        # return mean, log_prob of the gmm
        return gmm_dist.mean, gmm_dist.log_prob(target)

    def loss_fn(self, gmm, target, reduction='mean'):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == 'mean':
            return loss.mean() * self.loss_coef
        elif reduction == 'none':
            return loss * self.loss_coef
        elif reduction == 'sum':
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError
