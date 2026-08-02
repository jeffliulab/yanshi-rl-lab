# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree_rl_lab (Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: ``UniformLevelVelocityCommandCfg`` is a verbatim port of
# unitree_rl_lab ``tasks/locomotion/mdp/commands/velocity_command.py`` (the
# upstream file carries no per-file license header; the upstream repository is
# licensed under Apache-2.0). The deadband command below is original work by
# this project, previously hosted as an overlay inside a working copy of that
# repository (S2-Exp4, 2026-07-27).

"""Velocity command terms shared by all velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with a second ``limit_ranges`` envelope.

    ``ranges`` is the live sampling interval; ``limit_ranges`` is the ceiling
    that command curricula (see ``curriculums.lin_vel_cmd_levels``) may expand
    ``ranges`` towards.
    """

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


# Planar-speed magnitude below which a sampled linear-velocity command snaps to
# exactly zero. This is the legged_gym-family value, unchanged across
# legged_gym / unitree_rl_gym / humanoid-gym / HIMLoco / extreme-parkour for
# five years; the Isaac Lab port of UniformVelocityCommand dropped the line.
LIN_VEL_DEADBAND_MPS = 0.2


class DeadbandVelocityCommand(UniformVelocityCommand):
    """Uniform sampling plus a deadband snap: small linear-velocity commands
    become exactly zero while the yaw command is kept untouched.

    Why: with three independently-sampled command dimensions, "stand still and
    turn" samples (linear velocity exactly zero, large yaw rate) never occur,
    so turn-in-place is never trained. legged_gym's ``_resample_commands``
    solves this with a final snap-to-zero line; this class restores it on top
    of the Isaac Lab command term (S2-Exp4). Known cost, accepted upstream for
    years: commands slower than the deadband are never sampled.
    """

    cfg: DeadbandVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        lin = self.vel_command_b[env_ids, :2]
        hit = torch.norm(lin, dim=1) <= self.cfg.lin_vel_deadband
        self.vel_command_b[env_ids, :2] = lin * (~hit).unsqueeze(1)
        # One-shot self-check: report that the deadband is actually wired in.
        # Without this print the only "evidence" would be training curves that
        # look plausible, which is not evidence.
        if not getattr(self, "_deadband_reported", False) and len(env_ids) > 100:
            zeroed = hit.float().mean().item()
            print(
                f"[deadband] threshold {self.cfg.lin_vel_deadband} m/s | batch of "
                f"{len(env_ids)} envs: linear velocity snapped to zero for "
                f"{zeroed * 100:.1f}% of samples",
                flush=True,
            )
            self._deadband_reported = True


@configclass
class DeadbandVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = DeadbandVelocityCommand
    lin_vel_deadband: float = LIN_VEL_DEADBAND_MPS


def apply_lin_vel_deadband(cfg, deadband: float = LIN_VEL_DEADBAND_MPS) -> None:
    """Swap the env's ``base_velocity`` command term for the deadband subclass,
    carrying over **every** existing config field unchanged.

    Fields are copied via ``dataclasses.fields`` instead of being retyped by
    hand, so that if upstream ever adds a field we cannot silently drop it back
    to its default -- that kind of error is invisible in training and only
    shows up as "results look odd". (Verbatim port of the predecessor
    overlay's ``_apply_lin_vel_deadband``.)
    """
    import dataclasses

    old = cfg.commands.base_velocity
    carried = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "class_type"}
    new = DeadbandVelocityCommandCfg(**carried, lin_vel_deadband=deadband)
    assert new.class_type is DeadbandVelocityCommand, "deadband command term not wired in"
    cfg.commands.base_velocity = new


class HeadingPinnedVelocityCommand(UniformVelocityCommand):
    """Heading-mode velocity command whose ``heading_target`` is also defined
    for the *direct-wz* envs: at every command resample, non-heading envs get
    their target pinned to the current yaw ("hold the heading you had at
    resample time").

    Why (S3-实验2 发1): upstream only maintains ``heading_target`` for
    heading-mode envs, so a direct heading-tracking reward has no target for
    the direct-wz half of the mixed profile (``rel_heading_envs < 1``). The
    |wz| < deadband open-loop samples in that half are exactly the
    straight-line training signal, and pinning gives them a well-defined
    target without touching the P-control machinery.
    """

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        if not self.cfg.heading_command:
            return
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=self.device)
        non_heading = ids[~self.is_heading_env[ids]]
        if len(non_heading) > 0:
            self.heading_target[non_heading] = self.robot.data.heading_w[non_heading]
        # One-shot self-check (same discipline as the deadband report): prove
        # the pinning is wired in, not just plausible-looking.
        if not getattr(self, "_pinning_reported", False) and len(ids) > 100:
            pinned = (~self.is_heading_env[ids]).float().mean().item()
            print(
                f"[heading-pin] batch of {len(ids)} envs: heading_target pinned to current yaw "
                f"for {pinned * 100:.1f}% (direct-wz envs)",
                flush=True,
            )
            self._pinning_reported = True


@configclass
class HeadingPinnedVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = HeadingPinnedVelocityCommand


def apply_heading_pinning(cfg) -> None:
    """Swap the env's ``base_velocity`` command term for the heading-pinning
    subclass, carrying over **every** existing config field unchanged (same
    dataclass-fields carry-over discipline as ``apply_lin_vel_deadband``).
    """
    import dataclasses

    old = cfg.commands.base_velocity
    carried = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.name != "class_type"}
    new = HeadingPinnedVelocityCommandCfg(**carried)
    assert new.class_type is HeadingPinnedVelocityCommand, "heading-pinning command term not wired in"
    cfg.commands.base_velocity = new
