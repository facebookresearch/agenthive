# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from numbers import Number
from typing import Union

import numpy as np
import torch

from tensordict.nn import TensorDictSequential
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import Tensor

from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import SafeModule
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    distance_loss,
    next_state_value as get_next_state_value,
)

try:
    from functorch import vmap

    FUNCTORCH_ERR = ""
    _has_functorch = True
except ImportError as err:
    FUNCTORCH_ERR = str(err)
    _has_functorch = False


class SACLoss(LossModule):
    """SAC Loss module.
    Args:
        actor_network (SafeModule): the actor to be trained
        qvalue_network (SafeModule): a single Q-value network that will be multiplicated as many times as needed.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        gamma (Number, optional): gamma decay factor. Default is 0.99.
        priotity_key (str, optional): Key where to write the priority value for prioritized replay buffers. Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of  `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is 0.1.
        max_alpha (float, optional): max value of alpha.
            Default is 10.0.
        fixed_alpha (bool, optional): whether alpha should be trained to match a target entropy. Default is :obj:`False`.
        target_entropy (Union[str, Number], optional): Target entropy for the stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is :obj:`False`.
        gSDE (bool, optional): Knowing if gSDE is used is necessary to create random noise variables.
            Default is False
    """

    delay_actor: bool = False
    _explicit: bool = True

    def __init__(
        self,
        actor_network: SafeModule,
        qvalue_network: SafeModule,
        num_qvalue_nets: int = 2,
        gamma: Number = 0.99,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        gSDE: bool = False,
    ):
        if not _has_functorch:
            raise ImportError(
                f"Failed to import functorch with error message:\n{FUNCTORCH_ERR}"
            )

        super().__init__()
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist_params"],
        )

        # let's make sure that actor_network has `return_log_prob` set to True
        self.actor_network.return_log_prob = True

        self.delay_qvalue = delay_qvalue
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()),
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.register_buffer("gamma", torch.tensor(gamma))
        self.priority_key = priotity_key
        self.loss_function = loss_function

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        self.register_buffer(
            "min_log_alpha", torch.tensor(min_alpha, device=device).log()
        )
        self.register_buffer(
            "max_log_alpha", torch.tensor(max_alpha, device=device).log()
        )
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if target_entropy == "auto":
            if actor_network.spec["action"] is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        self.gSDE = gSDE

    @property
    def alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._explicit:
            # slow but explicit version
            return self._forward_explicit(tensordict)
        else:
            return self._forward_vectorized(tensordict)

    def _loss_alpha(self, log_pi: Tensor) -> Tensor:
        if torch.is_grad_enabled() and not log_pi.requires_grad:
            raise RuntimeError(
                "expected log_pi to require gradient for the alpha loss)"
            )
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    def _forward_vectorized(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.select(
            "reward", "done", "next", *obs_keys, "action"
        )

        actor_params = torch.stack(
            [self.actor_network_params, self.target_actor_network_params], 0
        )

        tensordict_actor_grad = tensordict_select.select(
            *obs_keys
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict_select).select(
            *self.actor_network.in_keys
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)
        tensordict_actor = tensordict_actor.contiguous()

        with set_exploration_mode("random"):
            if self.gSDE:
                tensordict_actor.set(
                    "_eps_gSDE",
                    torch.zeros(tensordict_actor.shape, device=tensordict_actor.device),
                )
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = vmap(self.actor_network.get_dist_params)(
                tensordict_actor,
                actor_params,
            )
            if isinstance(self.actor_network, TensorDictSequential):
                sample_key = self.actor_network[-1].out_keys[0]
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            else:
                sample_key = self.actor_network.out_keys[0]
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            tensordict_actor[sample_key] = self._rsample(tensordict_actor_dist)
            tensordict_actor["sample_log_prob"] = tensordict_actor_dist.log_prob(
                tensordict_actor[sample_key]
            )

        # repeat tensordict_actor to match the qvalue size
        _actor_loss_td = (
            tensordict_actor[0]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[0].batch_size)
        )  # for actor loss
        _qval_td = tensordict_select.select(*self.qvalue_network.in_keys).expand(
            self.num_qvalue_nets,
            *tensordict_select.select(*self.qvalue_network.in_keys).batch_size,
        )  # for qvalue loss
        _next_val_td = (
            tensordict_actor[1]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[1].batch_size)
        )  # for next value estimation
        tensordict_qval = torch.cat(
            [
                _actor_loss_td,
                _next_val_td,
                _qval_td,
            ],
            0,
        )

        # cat params
        q_params_detach = self.qvalue_network_params.detach()
        qvalue_params = torch.cat(
            [
                q_params_detach,
                self.target_qvalue_network_params,
                self.qvalue_network_params,
            ],
            0,
        )
        tensordict_qval = vmap(self.qvalue_network)(
            tensordict_qval,
            qvalue_params,
        )

        state_action_value = tensordict_qval.get("state_action_value").squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )
        sample_log_prob = tensordict_actor.get("sample_log_prob").squeeze(-1)
        (
            action_log_prob_actor,
            next_action_log_prob_qvalue,
        ) = sample_log_prob.unbind(0)

        # E[alpha * log_pi(a) - Q(s, a)] where a is reparameterized
        loss_actor = -(
            state_action_value_actor.min(0)[0] - self.alpha * action_log_prob_actor
        ).mean()

        next_state_value = (
            next_state_action_value_qvalue.min(0)[0]
            - self.alpha * next_action_log_prob_qvalue
        )

        target_value = get_next_state_value(
            tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_value,
        )
        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = (
            distance_loss(
                pred_val,
                target_value.expand_as(pred_val),
                loss_function=self.loss_function,
            )
            .mean(-1)
            .sum()
            * 0.5
        )

        tensordict.set("td_error", td_error.detach().max(0)[0])

        loss_alpha = self._loss_alpha(sample_log_prob)
        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self.alpha.detach(),
                "entropy": -sample_log_prob.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "action_log_prob_actor": action_log_prob_actor.mean().detach(),
                "next.state_value": next_state_value.mean().detach(),
                "target_value": target_value.mean().detach(),
            },
            [],
        )

        return td_out

    def _forward_explicit(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss_actor, sample_log_prob = self._loss_actor_explicit(tensordict.clone(False))
        loss_qval, td_error = self._loss_qval_explicit(tensordict.clone(False))
        tensordict.set("td_error", td_error.detach().max(0)[0])
        loss_alpha = self._loss_alpha(sample_log_prob)
        td_out = TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self.alpha.detach(),
                "entropy": -sample_log_prob.mean().detach(),
                # "state_action_value_actor": state_action_value_actor.mean().detach(),
                # "action_log_prob_actor": action_log_prob_actor.mean().detach(),
                # "next.state_value": next_state_value.mean().detach(),
                # "target_value": target_value.mean().detach(),
            },
            [],
        )
        return td_out

    def _rsample(self, dist, ):
        # separated only for the purpose of making the sampling
        # deterministic to compare methods
        return dist.rsample()


    def _sample_reparam(self, tensordict, params):
        """Given a policy param batch and input data in a tensordict, writes a reparam sample and log-prob key."""
        with set_exploration_mode("random"):
            if self.gSDE:
                raise NotImplementedError
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = self.actor_network.get_dist_params(tensordict, params,)
            if isinstance(self.actor_network, TensorDictSequential):
                sample_key = self.actor_network[-1].out_keys[0]
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            else:
                sample_key = self.actor_network.out_keys[0]
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            tensordict[sample_key] = self._rsample(tensordict_actor_dist)
            tensordict["sample_log_prob"] = tensordict_actor_dist.log_prob(
                tensordict[sample_key]
            )
        return tensordict

    def _loss_actor_explicit(self, tensordict):
        tensordict_actor = tensordict.clone(False)
        actor_params = self.actor_network_params
        tensordict_actor = self._sample_reparam(tensordict_actor, actor_params)
        action_log_prob_actor = tensordict_actor["sample_log_prob"]

        tensordict_qval = (
            tensordict_actor
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor.batch_size)
        )  # for actor loss
        qvalue_params = self.qvalue_network_params.detach()
        tensordict_qval = vmap(self.qvalue_network)(tensordict_qval, qvalue_params,)
        state_action_value_actor = tensordict_qval.get("state_action_value").squeeze(-1)
        state_action_value_actor = state_action_value_actor.min(0)[0]

        # E[alpha * log_pi(a) - Q(s, a)] where a is reparameterized
        loss_actor = (self.alpha * action_log_prob_actor - state_action_value_actor).mean()

        return loss_actor, action_log_prob_actor

    def _loss_qval_explicit(self, tensordict):
        next_tensordict = step_mdp(tensordict)
        next_tensordict = self._sample_reparam(next_tensordict, self.target_actor_network_params)
        next_action_log_prob_qvalue = next_tensordict["sample_log_prob"]
        next_state_action_value_qvalue = vmap(self.qvalue_network, (None, 0))(
            next_tensordict,
            self.target_qvalue_network_params,
        )["state_action_value"].squeeze(-1)

        next_state_value = (
            next_state_action_value_qvalue.min(0)[0]
            - self.alpha * next_action_log_prob_qvalue
        )

        pred_val = vmap(self.qvalue_network, (None, 0))(
            tensordict,
            self.qvalue_network_params,
        )["state_action_value"].squeeze(-1)

        target_value = get_next_state_value(
            tensordict,
            gamma=self.gamma,
            pred_next_val=next_state_value,
        )

        # 1/2 * E[Q(s,a) - (r + gamma * (Q(s,a)-alpha log pi(s, a)))
        loss_qval = (
            distance_loss(
                pred_val,
                target_value.expand_as(pred_val),
                loss_function=self.loss_function,
            )
            .mean(-1)
            .sum()
            * 0.5
        )
        td_error = (pred_val - target_value).pow(2)
        return loss_qval, td_error

if __name__ == "__main__":
    # Tests the vectorized version of SAC-v2 against plain implementation
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.data import BoundedTensorSpec
    from torch import nn
    from tensordict.nn import TensorDictModule
    from torchrl.modules.distributions import TanhNormal

    torch.manual_seed(0)

    action_spec = BoundedTensorSpec(-1, 1, shape=(3,))
    class Splitter(nn.Linear):
        def forward(self, x):
            loc, scale = super().forward(x).chunk(2, -1)
            return loc, scale.exp()
    actor_module = TensorDictModule(Splitter(6, 6), in_keys=["obs"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=TanhNormal,
        default_interaction_mode="random",
        return_log_prob=False,
    )
    class QVal(nn.Linear):
        def forward(self, s: Tensor, a: Tensor) -> Tensor:
            return super().forward(torch.cat([s, a], -1))

    qvalue = ValueOperator(QVal(9, 1), in_keys=["obs", "action"])
    _rsample_old = SACLoss._rsample
    def _rsample_new(self, dist):
        return torch.ones_like(_rsample_old(self, dist))
    SACLoss._rsample = _rsample_new
    loss = SACLoss(actor, qvalue)

    for batch in ((), (2, 3)):
        td_input = TensorDict({"obs": torch.rand(*batch, 6), "action": torch.rand(*batch, 3).clamp(-1, 1), "next": {"obs": torch.rand(*batch, 6)}, "reward": torch.rand(*batch, 1), "done": torch.zeros(*batch, 1, dtype=torch.bool)}, batch)
        loss._explicit = True
        loss0 = loss(td_input)
        loss._explicit = False
        loss1 = loss(td_input)
        print("a", loss0["loss_actor"]-loss1["loss_actor"])
        print("q", loss0["loss_qvalue"]-loss1["loss_qvalue"])
