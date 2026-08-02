# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""AgiBot Lingxi X2 velocity-task registration.

The overlay carries exactly one measured-necessary override (project
acceptance rule: overrides only with evidence, each with a documented reason).

Attempts 1-2 verdict (2026-07-30, decisions D5/D8): both training runs died
at ~6 steps (100% bad_orientation) -- the robot never stood, so the command
side never even got tested. Root cause found by the 2026-07-31 probe battery
(decision D9): the vendor collision meshes interpenetrate, so self-collision
contact forces poisoned undesired_contacts and PPO converged to a suicide
equilibrium (the ~6-step death was LEARNED at iter ~1400, not physical). The
primary fix for attempt 3 is therefore in the robot profile
(self_collisions=False); the D8 kp1050 gain calibration never shipped (D8
appendix: Isaac zero-action probe refuted it). The dead-command override
below is the SECOND, independently-evidenced half of the attempt-3 double fix:

- Independent evidence (decision D7, BHL attempt 1): the shared S2-Exp5
  aligned reward table (notably the foot-clearance penalty gated on
  |cmd| >= 0.05) is only empirically validated under DEAD command ranges
  (G1 parity 4/4). Under the official slow curriculum (ranges grow from
  +-0.1), low-command episodes make "never lift a foot" a strong local
  optimum -- BHL converged to a standing statue exactly this way. X2 trains
  on the same reward table, so once it CAN stand (D9 fix), the same trap
  is waiting; commands are set dead pre-emptively rather than spending a
  fourth attempt (quota is 3, this is the last one) to rediscover D7.

Override values mirror the D7 BHL override verbatim: both range sets dead at
the upstream envelope vx (-0.5, 1.0) / vy (-0.3, 0.3), wz kept at the
official +-0.2 (X2's exam only commands wz = 0.2), plus the linear-velocity
deadband snap (0.2 m/s) so "stand when commanded ~zero, walk otherwise" is
actually trained.
"""

from yanshi_rl_lab.mdp.commands import apply_lin_vel_deadband
from yanshi_rl_lab.robots.agibot.x2.v1_4_0.profile import X2_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

# Dead command envelope = the regime in which the shared reward table is
# empirically validated (G1 parity 4/4). Values are the upstream envelope
# (base limit_ranges), NOT widened.
DEAD_LIN_VEL_X = (-0.5, 1.0)
DEAD_LIN_VEL_Y = (-0.3, 0.3)
OFFICIAL_ANG_VEL_Z = (-0.2, 0.2)


def _dead_command_profile(cfg) -> None:
    """Attempt-3 override half 2 (D8, mechanism D7): kill the slow command
    curriculum, keep the official command envelope, add the deadband snap."""
    for r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        r.lin_vel_x = DEAD_LIN_VEL_X
        r.lin_vel_y = DEAD_LIN_VEL_Y
        r.ang_vel_z = OFFICIAL_ANG_VEL_Z
    apply_lin_vel_deadband(cfg)


register_velocity(X2_PROFILE, __name__, terrains=("flat",), overrides=_dead_command_profile)
