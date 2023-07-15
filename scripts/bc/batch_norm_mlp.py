import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

class FCNetworkWithBatchNorm(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='relu',   # either 'tanh' or 'relu'
                 dropout=0,           # probability to dropout activations (0 means no dropout)
                 *args, **kwargs,
                ):
        super(FCNetworkWithBatchNorm, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.device = 'cpu'

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.input_batchnorm = nn.BatchNorm1d(num_features=obs_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x.to(self.device)
        out = self.input_batchnorm(out)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.dropout(out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out

    def to(self, device):
        self.device = device
        return super().to(device)

    def set_transformations(self, *args, **kwargs):
        pass

class BatchNormMLP(nn.Module):
    def __init__(self, env_spec=None,
                 action_dim=None,
                 observation_dim=None,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None,
                 nonlinearity='relu',
                 dropout=0,
                 device='cpu',
                 *args, **kwargs,
        ):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        super(BatchNormMLP, self).__init__()

        self.device = device
        self.n = env_spec.observation_dim if observation_dim is None else observation_dim  # number of states
        self.m = env_spec.action_dim if action_dim is None else action_dim # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = FCNetworkWithBatchNorm(self.n, self.m, hidden_sizes, nonlinearity, dropout)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]
        self.model.eval()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def to(self, device):
        super().to(device)
        self.model = self.model.to(device)
        print(self.model)
        self.device = device
        return self

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).to('cpu').data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    # ============================================
    def forward(self, observations):
        if type(observations) == np.ndarray: observations = torch.from_numpy(observations).float()
        assert type(observations) == torch.Tensor
        observations = observations.to(self.device)
        out = self.model(observations)
        return out


