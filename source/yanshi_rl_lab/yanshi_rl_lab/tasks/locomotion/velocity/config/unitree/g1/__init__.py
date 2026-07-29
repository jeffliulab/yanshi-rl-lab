# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1 velocity-task registration.

The overlay stays empty of task code except the declared override slot below
(project acceptance rule: a robot with a complete profile trains with an
empty overlay; overrides exist only when measurement proves them necessary,
each with a documented reason).

M1 parity-gate override (decision D4, 2026-07-29): the parity run must
replicate the predecessor stack's battle-proven ``RobotTurnAlignedEnvCfg``
(S2-Exp5 aligned configuration, four gates passed 4/4) **verbatim**. The new
base already carries its reward side (the upstream-aligned package is the
default); what the base keeps at upstream-official values is the command
profile -- so exactly that is overridden here.
"""

from yanshi_rl_lab.mdp.commands import apply_lin_vel_deadband
from yanshi_rl_lab.robots.unitree.g1.profile import G1_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

# -- S2-Exp5 aligned command profile (source: old stack
# unitree_rl_lab .../g1/29dof/velocity_variants_env_cfg.py) -------------------
# ``_fix_command`` (L80-85) sets BOTH ``ranges`` and ``limit_ranges`` to the
# same dead values, bypassing the slow command curriculum;
# ``RobotTurnEnvCfg`` (L307) calls it with vx_hi=1.0, wz=0.8.
PARITY_LIN_VEL_X = (-0.5, 1.0)  # _fix_command default backward limit + Turn's forward cap
PARITY_LIN_VEL_Y = (-0.3, 0.3)  # _fix_command default, verbatim upstream limit_ranges value
# Turn-in-place needs a +-0.8 rad/s *training* range: at the official +-0.2 the
# tracking-reward gap between turning and not turning is ~0.15 and any energy
# penalty wins (old-stack experiment 14, RobotTurnEnvCfg docstring); the
# parity gate's turn-in-place exam point commands wz=0.6, unreachable at +-0.2.
PARITY_ANG_VEL_Z = (-0.8, 0.8)


def _parity_command_profile(cfg) -> None:
    """Verbatim replica of ``RobotTurnAlignedEnvCfg``'s command side.

    Old-stack lineage (velocity_variants_env_cfg.py): TurnAligned (L329-334)
    -> TurnDeadband (L311-321, adds the linear-velocity deadband snap)
    -> Turn (L294-307, wz +-0.8) -> ``_fix_command`` (L80-85, both range sets
    dead). Reward side (upstream-aligned package W/C/H/G) is already the new
    base's default, so commands are the only thing to replicate.
    """
    for r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        r.lin_vel_x = PARITY_LIN_VEL_X
        r.lin_vel_y = PARITY_LIN_VEL_Y
        r.ang_vel_z = PARITY_ANG_VEL_Z
    # Deadband snap (old stack L145-157 ``_apply_lin_vel_deadband``, threshold
    # LIN_VEL_DEADBAND = 0.2 at L109): small linear-velocity samples become
    # exactly zero so "stand and turn" is actually trained. Same helper,
    # ported to mdp.commands, same default threshold.
    apply_lin_vel_deadband(cfg)


register_velocity(G1_PROFILE, __name__, terrains=("flat",), overrides=_parity_command_profile)
