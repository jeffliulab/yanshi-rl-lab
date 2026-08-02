# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite (humanoid configuration, 22 actuated joints; the
armless 12-DoF biped variant is deliberately not registered). This package
stays pure Python (no Isaac Lab imports).

Gains/limits are the OFFICIAL upstream training values (MIT-licensed
HybridRobotics stack), transcribed with file+line provenance in
``profile.py`` -- nothing is derived, unlike X2."""

from .profile import BHL_PROFILE  # noqa: F401
