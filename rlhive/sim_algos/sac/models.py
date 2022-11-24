# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import (
    MLP,
    NormalParamWrapper,
    ProbabilisticActor,
    TanhNormal,
    TensorDictModule,
    ValueOperator,
)
from torchrl.trainers.helpers.models import ACTIVATIONS


def make_sac_model(cfg, device, single_env_constructor):
    tanh_loc = cfg.model.tanh_loc
    default_policy_scale = cfg.model.default_policy_scale

    proof_environment = single_env_constructor()
    proof_environment.reset()
    action_spec = proof_environment.action_spec

    actor_net_kwargs = {}
    value_net_kwargs = {}
    qvalue_net_kwargs = {}

    in_keys = ["observation_vector"]

    actor_net_kwargs_default = {
        "num_cells": [cfg.model.actor_cells, cfg.model.actor_cells],
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": ACTIVATIONS[cfg.model.activation],
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = MLP(**actor_net_kwargs_default)

    qvalue_net_kwargs_default = {
        "num_cells": [cfg.model.qvalue_cells, cfg.model.qvalue_cells],
        "out_features": 1,
        "activation_class": ACTIVATIONS[cfg.model.activation],
    }
    qvalue_net_kwargs_default.update(qvalue_net_kwargs)
    qvalue_net = MLP(
        **qvalue_net_kwargs_default,
    )

    value_net_kwargs_default = {
        "num_cells": [cfg.model.value_cells, cfg.model.value_cells],
        "out_features": 1,
        "activation_class": ACTIVATIONS[cfg.model.activation],
    }
    value_net_kwargs_default.update(value_net_kwargs)
    value_net = MLP(
        **value_net_kwargs_default,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": tanh_loc,
    }

    actor_net = NormalParamWrapper(
        actor_net,
        scale_mapping=f"biased_softplus_{default_policy_scale}",
        scale_lb=cfg.model.scale_lb,
    )
    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )

    actor = ProbabilisticActor(
        spec=action_spec,
        dist_param_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )

    model = nn.ModuleList([actor, qvalue, value]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td

    return model
