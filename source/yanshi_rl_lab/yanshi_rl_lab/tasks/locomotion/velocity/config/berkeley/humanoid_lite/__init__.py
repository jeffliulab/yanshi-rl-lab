# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite velocity-task registration.

The overlay carries exactly one measured-necessary override (project
acceptance rule: overrides only with evidence, each with a documented reason).

Attempt-1 verdict (2026-07-30, decision D7): training run
``yanshi-bhl-flat-s42`` converged to a *standing statue* -- perfect stability
(episode length 983/1000, zero falls in sim2sim, 99% double-foot contact)
but no locomotion (Isaac-side ``error_vel_xy`` ~0.70 m/s at 10000 iterations;
sim2sim net displacement 0.03-0.06 m in 8 s, gates 1/4). Mechanism: the
S2-Exp5-aligned reward table (notably the foot-clearance PENALTY, gated on
|cmd| >= 0.05) was only ever validated with DEAD command ranges (the G1
parity recipe); under the official slow curriculum (ranges grow from +-0.1),
low-command episodes make "never lift a foot" a strong local optimum.

Override (single variable, mirrors the validated G1 parity command side minus
the turn-range widening): both range sets dead at the upstream envelope
vx (-0.5, 1.0) / vy (-0.3, 0.3), wz kept at the official +-0.2 (BHL's exam
only commands wz = 0.2), plus the linear-velocity deadband snap (0.2 m/s) so
"stand when commanded ~zero, walk otherwise" is actually trained.

The 25 Hz control-rate candidate recorded during onboarding stays NOT applied:
attempt 1 showed no instability (the robot is perfectly stable), so the
control rate is not the limiting factor; fleet-uniform 50 Hz is kept.

Registry naming rule (tasks/registry.py): vendor "berkeley" + model
"humanoid_lite" -> task-ID segments "Berkeley" + "Humanoid-Lite", so the
generated ID is ``Yanshi-Velocity-Flat-Berkeley-Humanoid-Lite-v0``.
"""

from yanshi_rl_lab.mdp.commands import apply_lin_vel_deadband
from yanshi_rl_lab.robots.berkeley.humanoid_lite.profile import BHL_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

# Dead command envelope = the regime in which the shared reward table is
# empirically validated (G1 parity 4/4). Values are the upstream envelope
# (base limit_ranges), NOT widened.
DEAD_LIN_VEL_X = (-0.5, 1.0)
DEAD_LIN_VEL_Y = (-0.3, 0.3)
OFFICIAL_ANG_VEL_Z = (-0.2, 0.2)


def _dead_command_profile(cfg) -> None:
    """Attempt-2 override (D7): kill the slow command curriculum, keep the
    official command envelope, add the deadband snap."""
    for r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        r.lin_vel_x = DEAD_LIN_VEL_X
        r.lin_vel_y = DEAD_LIN_VEL_Y
        r.ang_vel_z = OFFICIAL_ANG_VEL_Z
    apply_lin_vel_deadband(cfg)


register_velocity(BHL_PROFILE, __name__, terrains=("flat",), overrides=_dead_command_profile)
