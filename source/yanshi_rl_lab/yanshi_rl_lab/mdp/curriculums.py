# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree_rl_lab (Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: ``lin_vel_cmd_levels`` is a verbatim port from unitree_rl_lab
# ``tasks/locomotion/mdp/curriculums.py`` (the upstream file carries no
# per-file license header; the upstream repository is licensed under
# Apache-2.0). ``terrain_levels_survival`` is original work by this project,
# previously hosted as an overlay inside a working copy of that repository.

"""Curriculum terms shared by all velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Expand the linear-velocity command range towards ``limit_ranges`` once
    tracking reward exceeds 80% of its weight (upstream command curriculum)."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_survival(
    env, env_ids: Sequence[int], survive_frac: float = 0.9, demote_frac: float = 0.5
) -> torch.Tensor:
    """Survival-based terrain promotion: an episode surviving at least
    ``survive_frac`` of its length moves the env up one terrain level; falling
    before ``demote_frac`` moves it down one.

    Why not the displacement-based official ``terrain_levels_vel``: it promotes
    on "net displacement > 4 m", but command profiles that include sustained
    yaw commands make the robot walk arcs/circles, so net displacement stays
    small and the level pins at 0 even for policies that barely ever fall
    (measured on the predecessor stack). Survival promotion directly optimizes
    "how hard a terrain can this robot survive on" and is immune to the
    command profile.
    """
    episode_len = env.episode_length_buf[env_ids]
    terrain = env.scene.terrain
    move_up = episode_len >= survive_frac * env.max_episode_length
    move_down = episode_len < demote_frac * env.max_episode_length
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    # Diagnostic print roughly every 50 calls: confirms the term is being
    # called and shows episode lengths, promotions/demotions and level spread.
    cnt = getattr(terrain_levels_survival, "_dbg_cnt", 0) + 1
    terrain_levels_survival._dbg_cnt = cnt
    if cnt % 50 == 1:
        print(
            f"[survival-curr] call#{cnt} n={len(env_ids)} ep_len(min/mean/max)="
            f"{int(episode_len.min())}/{float(episode_len.float().mean()):.0f}/{int(episode_len.max())} "
            f"max_len={env.max_episode_length} up={int(move_up.sum())} down={int(move_down.sum())} "
            f"levels_mean={float(terrain.terrain_levels.float().mean()):.2f}",
            flush=True,
        )
    return torch.mean(terrain.terrain_levels.float())
