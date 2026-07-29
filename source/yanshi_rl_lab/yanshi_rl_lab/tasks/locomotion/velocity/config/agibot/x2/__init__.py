# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""AgiBot Lingxi X2 velocity-task registration.

Deliberately an EMPTY overlay: registration only, no overrides. X2 has no
battle-proven legacy configuration to replicate (unlike G1, whose parity
override replays the predecessor stack's aligned command profile), so it
starts from the upstream-official command envelope of the shared base
(limit_ranges vx (-0.5, 1.0), vy +-0.3, wz +-0.2, no deadband). Overrides may
be added only when measurement proves them necessary, each with a documented
reason -- project acceptance rule.
"""

from yanshi_rl_lab.robots.agibot.x2.profile import X2_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity

register_velocity(X2_PROFILE, __name__, terrains=("flat",))
