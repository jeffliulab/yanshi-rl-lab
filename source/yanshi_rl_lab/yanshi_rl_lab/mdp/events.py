# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# NOTICE: ``randomize_joint_default_pos`` is a verbatim port of
# HybridRobotics/Berkeley-Humanoid-Lite @ 984741a3
# (``source/berkeley_humanoid_lite/berkeley_humanoid_lite/tasks/locomotion/
# velocity/mdp/events.py`` L15-47; repository licensed under BSD-3-Clause,
# see LICENCE in that repository). The vendor's humanoid env randomizes the
# joint default positions at startup to model calibration error
# (config/humanoid/env_cfg.py L256-264). The vendor file's companion
# ``randomize_actuator_torque_constant`` is NOT ported: the vendor config
# actually calls Isaac Lab's stock ``randomize_actuator_gains`` (a
# ManagerTermBase class, present in our pinned Isaac Lab), which we use
# directly.

"""Event terms shared by all velocity tasks (robot-independent: every robot
specific comes in through function parameters)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs.mdp.events import _randomize_prop_by_op  # private upstream helper, pinned by our IsaacLab version
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint default positions which may be different from URDF due to calibration errors.

    Verbatim port of the Berkeley Humanoid Lite vendor term (see NOTICE
    above); mutates ``asset.data.default_joint_pos`` in place. Note the
    vendor's private-API import of ``_randomize_prop_by_op`` is kept -- it is
    pinned by our IsaacLab version, same as for the vendor.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
