# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree_rl_lab (Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: ``energy``, ``feet_gait`` and ``air_time_variance_penalty`` are
# verbatim ports from unitree_rl_lab ``tasks/locomotion/mdp/rewards.py`` (the
# upstream file carries no per-file license header; the upstream repository is
# licensed under Apache-2.0). ``clearance_penalty`` is original work by this
# project (S2-Exp5 upstream-aligned package), written after the mjlab
# penalty-form foot-clearance term.

"""Reward terms shared by all velocity tasks (robot-independent: every robot
specific comes in through function parameters resolved from the RobotProfile)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """Reward feet contacts that match a fixed-period alternating gait phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def clearance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_height: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalty-form foot-clearance term (after mjlab; replaces the exp-form
    upstream reward whose ``exp(-0) = 1`` paid full score for feet that never
    move).

    Key difference from the exp form: no exponential shell. A foot at rest has
    zero velocity, hence zero penalty (neutral) instead of maximal reward, so
    "standing still" no longer collects free score from this term. The command
    gate uses ``|cmd_xy| + |cmd_yaw|`` -- the yaw term is required so the term
    stays active while turning in place.

    All robot/task specifics (foot bodies, command name, target height, gate
    threshold) are parameters supplied by the task layer from the RobotProfile
    and named constants -- nothing is hardcoded here.
    """
    asset = env.scene[asset_cfg.name]
    foot_h = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_v = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    cost = torch.sum(torch.abs(foot_h - target_height) * foot_v, dim=1)
    cmd = env.command_manager.get_command(command_name)
    gate = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    return cost * (gate > command_threshold).float()


def track_heading_exp(env: ManagerBasedRLEnv, command_name: str = "base_velocity", std: float = 0.5) -> torch.Tensor:
    """Reward matching the command's heading target with a Gaussian kernel.

    ``exp(-(wrap_to_pi(heading_target - heading_w))^2 / std^2)`` -- the direct
    heading signal that the mainstream stack lacks (it only P-controls wz from
    the heading error and rewards angular-velocity tracking; see S3-实验2 card
    section 1.1). For heading-mode envs the target is the sampled heading; for
    direct-wz envs it is pinned to the yaw at resample time (see
    ``mdp.commands.HeadingPinnedVelocityCommand``), which turns |wz|~0 samples
    into explicit straight-line training signal. The command term must expose
    ``heading_target`` (Isaac Lab ``UniformVelocityCommand`` lineage).
    """
    term = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[term.cfg.asset_name]
    from isaaclab.utils import math as math_utils

    heading_error = math_utils.wrap_to_pi(term.heading_target - asset.data.heading_w)
    return torch.exp(-torch.square(heading_error / std))
