# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for yanshi-rl-lab (Copyright (c) 2026 Jeff Liu, MIT): docstring, and
# removal of the template's ``ui_extension_example`` import (the module was
# dropped in M0 but the import had been left behind, breaking ``import
# yanshi_rl_lab``).

"""Yanshi RL Lab: vendor-neutral RL training framework for legged robots.

Importing this package registers all Yanshi-* gym tasks (and therefore pulls
in Isaac Lab). Pure-Python consumers that must not touch Isaac Lab (e.g. the
profile unit tests) load the leaf modules through a namespace stub instead of
this package root -- see ``tests/test_g1_profile.py``.
"""

# Register Gym environments.
from .tasks import *  # noqa: F401, F403
