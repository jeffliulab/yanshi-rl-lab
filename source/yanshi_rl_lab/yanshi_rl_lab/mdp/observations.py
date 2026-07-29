# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree_rl_lab (Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: ``gait_phase`` is a verbatim port from unitree_rl_lab
# ``tasks/locomotion/mdp/observations.py`` (the upstream file carries no
# per-file license header; the upstream repository is licensed under
# Apache-2.0).

"""Observation terms shared by all velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """Sine/cosine encoding of a fixed-period global gait phase.

    Kept for parity with the upstream base config, where the term exists but is
    commented out (enabling it changes the observation width and therefore the
    deployed policy contract -- do not enable casually).
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
