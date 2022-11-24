# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from torchrl.objectives import LossModule, SACLoss
from torchrl.objectives.costs.utils import _TargetNetUpdate, HardUpdate, SoftUpdate


def make_sac_loss(cfg, model) -> Tuple[SACLoss, Optional[_TargetNetUpdate]]:
    """Builds the SAC loss module."""
    loss_kwargs = {}
    if hasattr(cfg, "distributional") and cfg.model.loss.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": cfg.model.loss.loss_function})
        loss_kwargs.update(
            {
                "target_entropy": cfg.model.loss.target_entropy
                if cfg.target_entropy is not None
                else "auto"
            }
        )
        loss_class = SACLoss
        if cfg.model.loss.loss == "double":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": True,
                    "delay_value": True,
                }
            )
        elif cfg.model.loss.loss == "single":
            loss_kwargs.update(
                {
                    "delay_actor": False,
                    "delay_qvalue": False,
                    "delay_value": False,
                }
            )
        else:
            raise NotImplementedError(
                f"cfg.loss {cfg.model.loss.loss} unsupported. Consider chosing from 'double' or 'single'"
            )

    actor_model, qvalue_model, value_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        value_network=value_model,
        num_qvalue_nets=cfg.model.loss.num_q_values,
        gamma=cfg.model.loss.gamma,
        **loss_kwargs,
    )
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_target_updater(
    cfg: "DictConfig", loss_module: LossModule
) -> Optional[_TargetNetUpdate]:
    """Builds a target network weight update object."""
    if cfg.model.loss.loss == "double":
        if not cfg.model.loss.hard_update:
            target_net_updater = SoftUpdate(
                loss_module, 1 - 1 / cfg.model.loss.value_network_update_interval
            )
        else:
            target_net_updater = HardUpdate(
                loss_module, cfg.model.loss.value_network_update_interval
            )
        # assert len(target_net_updater.net_pairs) == 3, "length of target_net_updater nets should be 3"
        target_net_updater.init_()
    else:
        if cfg.model.loss.hard_update:
            raise RuntimeError(
                "hard/soft-update are supposed to be used with double SAC loss. "
                "Consider using --loss=double or discarding the hard_update flag."
            )
        target_net_updater = None
    return target_net_updater
