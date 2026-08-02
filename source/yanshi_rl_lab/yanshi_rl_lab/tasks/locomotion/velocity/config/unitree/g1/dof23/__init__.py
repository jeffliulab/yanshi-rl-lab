# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1 (23-DoF) velocity-task registration.

An empty overlay, and deliberately so: the project's acceptance rule is that a
robot with a complete profile trains with nothing here but the registration
call, and overrides appear only when measurement proves them necessary.

The dof29 sibling carries two overrides because it replicates configurations
with a battle record (the S2-Exp5 parity command profile, the S3-实验2 rough
recipe). This configuration has no such record to reproduce, so copying those
overrides across would be assuming a result instead of measuring one. Flat
ground first; add terrains when there is a reason.
"""

from yanshi_rl_lab.robots.unitree.g1.dof23.profile import G1_DOF23_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

register_velocity(G1_DOF23_PROFILE, __name__, terrains=("flat",))
