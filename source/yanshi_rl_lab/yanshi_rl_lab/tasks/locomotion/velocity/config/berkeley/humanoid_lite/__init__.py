# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite velocity-task registration.

Deliberately an EMPTY overlay: registration only, no overrides. Like X2,
BHL has no battle-proven legacy configuration in this stack to replicate, so
it starts from the shared base's defaults (upstream-official command envelope
limit_ranges vx (-0.5, 1.0), vy +-0.3, wz +-0.2, no deadband; S2-Exp5-aligned
reward table; 50 Hz control). Overrides may be added only when measurement
proves them necessary, each with a documented reason -- project acceptance
rule.

Known deliberate deviations from the official BHL training setup (they are
base-level facts, not silent drops -- recorded here for the first smoke run):

- Control rate: official trains at 25 Hz (decimation 8, humanoid env_cfg.py
  L357); our shared base runs 50 Hz (decimation 4). The official 25 Hz choice
  serves their real-robot CAN bandwidth; for the cross-robot benchmark the
  fleet-uniform 50 Hz is kept. If the first smoke run fights instability,
  ``{"decimation": 8}`` here is the first candidate override (ledger entry
  required).
- Commands/observations/rewards follow this project's shared base (e.g.
  5-step history, no base-lin-vel in policy obs), not the official BHL MDP --
  same policy as for every robot in this stack: the benchmark examines robots
  under ONE common recipe.

Registry naming rule (tasks/registry.py): vendor "berkeley" + model
"humanoid_lite" -> task-ID segments "Berkeley" + "Humanoid-Lite", so the
generated ID is ``Yanshi-Velocity-Flat-Berkeley-Humanoid-Lite-v0`` (and the
config classes are ``BerkeleyHumanoidLiteVelocityFlatEnvCfg`` /
``...PlayEnvCfg``).
"""

from yanshi_rl_lab.robots.berkeley.humanoid_lite.profile import BHL_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

register_velocity(BHL_PROFILE, __name__, terrains=("flat",))
