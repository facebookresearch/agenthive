import torch
import numpy as np

from misc import tensorize

class MLP(torch.nn.Module):
    def __init__(self, env_spec=None,
                 hidden_sizes=(64,64),
                 min_log_std=-3.0,
                 init_log_std=0.0,
                 seed=123,
                 device='cpu',
                 observation_dim=None,
                 action_dim=None,
                 max_log_std=1.0,
                 *args, **kwargs,
                 ):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        super(MLP, self).__init__()
        # check input specification
        if env_spec is None:
            assert observation_dim is not None
            assert action_dim is not None
        self.observation_dim = env_spec.observation_dim if env_spec is not None else observation_dim   # number of states
        self.action_dim = env_spec.action_dim if env_spec is not None else action_dim                  # number of actions
        self.device = device
        self.seed = seed

        self.min_log_std_val = min_log_std if type(min_log_std)==np.ndarray else min_log_std * np.ones(self.action_dim)
        self.max_log_std_val = max_log_std if type(max_log_std)==np.ndarray else max_log_std * np.ones(self.action_dim)
        self.min_log_std = tensorize(self.min_log_std_val)
        self.max_log_std = tensorize(self.max_log_std_val)

        # Set seed
        # ------------------------
        assert type(seed) == int
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Policy network
        # ------------------------
        self.layer_sizes = (self.observation_dim, ) + hidden_sizes + (self.action_dim, )
        self.nonlinearity = torch.tanh
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                             for i in range(len(self.layer_sizes)-1)])
        for param in list(self.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = torch.nn.Parameter(torch.ones(self.action_dim) * init_log_std, requires_grad=True)
        self.log_std.data = torch.max(self.log_std.data, self.min_log_std)
        self.log_std.data = torch.min(self.log_std.data, self.max_log_std)
        self.trainable_params = list(self.parameters())
        # transform variables
        self.in_shift, self.in_scale = torch.zeros(self.observation_dim), torch.ones(self.observation_dim)
        self.out_shift, self.out_scale = torch.zeros(self.action_dim), torch.ones(self.action_dim)

        # Easy access variables
        # -------------------------
        self.log_std_val = self.log_std.to('cpu').data.numpy().ravel()
        # clamp log_std to [min_log_std, max_log_std]
        self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = torch.zeros(self.observation_dim)

        # Move parameters to device
        # ------------------------
        self.to(device)


    # Network forward
    # ============================================
    def forward(self, observations):
        if type(observations) == np.ndarray: observations = torch.from_numpy(observations).float()
        assert type(observations) == torch.Tensor
        observations = observations.to(self.device)
        out = (observations - self.in_shift) / (self.in_scale + 1e-8)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out) # DROPOUT?
        out = self.fc_layers[-1](out) * (self.out_scale + 1e-8) + self.out_shift
        ### MANUAL CLIP !!!!!
        # out = torch.max(torch.min(out, observations[:,:8]+0.1), observations[:,:8]-0.1)
        return out


    # Utility functions
    # ============================================
    def to(self, device):
        super().to(device)
        self.min_log_std, self.max_log_std = self.min_log_std.to(device), self.max_log_std.to(device)
        self.in_shift, self.in_scale = self.in_shift.to(device), self.in_scale.to(device)
        self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
        self.trainable_params = list(self.parameters())
        self.device = device

    def get_param_values(self, *args, **kwargs):
        params = torch.cat([p.contiguous().view(-1).data for p in self.parameters()])
        return params.clone()

    def set_param_values(self, new_params, *args, **kwargs):
        current_idx = 0
        for idx, param in enumerate(self.parameters()):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            # clip std at minimum value
            vals = torch.max(vals, self.min_log_std) if idx == 0 else vals
            vals = torch.min(vals, self.max_log_std) if idx == 0 else vals
            param.data = vals.to(self.device).clone()
            current_idx += self.param_sizes[idx]
        # update log_std_val for sampling
        self.log_std_val = np.float64(self.log_std.to('cpu').data.numpy().ravel())
        self.log_std_val = np.clip(self.log_std_val, self.min_log_std_val, self.max_log_std_val)
        self.trainable_params = list(self.parameters())

    def set_transformations(self, in_shift=None, in_scale=None,
                            out_shift=None, out_scale=None, *args, **kwargs):
        in_shift = self.in_shift if in_shift is None else tensorize(in_shift)
        in_scale = self.in_scale if in_scale is None else tensorize(in_scale)
        out_shift = self.out_shift if out_shift is None else tensorize(out_shift)
        out_scale = self.out_scale if out_scale is None else tensorize(out_scale)
        self.in_shift, self.in_scale = in_shift.to(self.device), in_scale.to(self.device)
        self.out_shift, self.out_scale = out_shift.to(self.device), out_scale.to(self.device)

        print("Set transformations")
        print(f" in shift\n  {in_shift.cpu().numpy()}\n in scale\n  {in_scale.cpu().numpy()}")
        print(f" out shift\n   {out_shift.cpu().numpy()}\n out scale\n  {out_scale.cpu().numpy()}")


    # Main functions
    # ============================================
    def get_action(self, observation):
        assert type(observation) == np.ndarray, "Currently passed observation type {}".format(type(observation))
        if self.device != 'cpu':
            print("Warning: get_action function should be used only for simulation.")
            print("Requires policy on CPU. Changing policy device to CPU.")
            self.to('cpu')
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.forward(self.obs_var).to('cpu').data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.action_dim)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, log_std=None, *args, **kwargs):
        if type(observations) == np.ndarray: observations = torch.from_numpy(observations).float()
        if type(actions) == np.ndarray: actions = torch.from_numpy(actions).float()
        observations, actions = observations.to(self.device), actions.to(self.device)
        log_std = self.log_std if log_std is None else log_std
        mean = self.forward(observations)
        zs = (actions - mean) / torch.exp(self.log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.action_dim * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, *args, **kwargs):
        mean, LL = self.mean_LL(observations, actions)
        return LL.to('cpu').data.numpy()

    def mean_kl(self, observations, *args, **kwargs):
        new_log_std = self.log_std
        old_log_std = self.log_std.detach().clone()
        new_mean = self.forward(observations)
        old_mean = new_mean.detach()
        return self.kl_divergence(new_mean, old_mean, new_log_std, old_log_std, *args, **kwargs)

    def kl_divergence(self, new_mean, old_mean, new_log_std, old_log_std, *args, **kwargs):
        new_std, old_std = torch.exp(new_log_std), torch.exp(old_log_std)
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
