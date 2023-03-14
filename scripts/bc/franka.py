
from mjrl.utils.gym_env import GymEnv
import numpy as np

class FrankaSpec():
    observation_dim = 11
    action_dim = 8

    def __init__(self, obs_dim=11, action_dim=8):
        # pass
        self.observation_dim = obs_dim
        self.action_dim = action_dim


class FrankaEnv(GymEnv):
    observation_dim = 11
    action_dim = 8
    start_js = [-0.145, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029] + [0.08]
    spec = FrankaSpec()
    horizon = 500
    act_repeat = 1

    # observaion mask for scaling ???
    #obs_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    obs_mask = np.array([1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  1.0, 1.0])


    def __init__(self, obs_dim=11, action_dim=8, horizon=500):
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.spec = FrankaSpec(obs_dim=obs_dim, action_dim=action_dim)
        # pass

    def reset(self):
        sampled_jointstate = self.start_js + np.random.normal(scale=0.01, size=8)
        sampled_tag = np.random.uniform(low=[-0.3, -0.22, -0.3],high=[0.3, -0.18, 0.1])
        return np.concatenate([sampled_jointstate, sampled_jointstate, sampled_tag])



class FrankaVelSpec():
    observation_dim = 18
    action_dim = 8
    def __init__(self):
        pass

class FrankaVelEnv(GymEnv):
    observation_dim = 18
    action_dim = 8
    start_js = [-0.145, -0.67, -0.052, -2.3, 0.145, 1.13, 0.029] + [0.08]
    spec = FrankaVelSpec()
    horizon = 500
    act_repeat = 1

    # observaion mask for scaling ???
    obs_mask = np.ones(18)

    def __init__(self):
        pass

    def reset(self):
        raise NotImplemented
        #sampled_jointstate = self.start_js + np.random.normal(scale=0.01, size=8)
        #sampled_tag = np.random.uniform(low=[-0.3, -0.22, -0.3],high=[0.3, -0.18, 0.1])
        #return np.concatenate([sampled_jointstate, sampled_jointstate, sampled_tag])
