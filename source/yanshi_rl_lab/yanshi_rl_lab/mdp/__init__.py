# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Robot-independent MDP terms shared by all tasks.

Planned contents (ported from the battle-tested predecessor stack in M1):
foot-clearance penalty, velocity-command deadband, air-time variance guard,
survival-based terrain curriculum. Every term in this package must be free of
concrete robot names -- robot specifics enter through function parameters
resolved from the RobotProfile.
"""
