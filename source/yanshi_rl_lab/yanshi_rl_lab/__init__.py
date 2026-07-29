# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for yanshi-rl-lab (Copyright (c) 2026 Jeff Liu, MIT): docstring,
# removal of the template's ``ui_extension_example`` import (M0), and the
# Isaac-optional import guard below (M2).

"""Yanshi RL Lab: vendor-neutral RL training framework for legged robots.

Importing this package registers all Yanshi-* gym tasks when Isaac Lab is
available. On machines WITHOUT the training stack (the exact machines
``yanshi doctor`` exists to diagnose, and any pure deploy/CI box) the
registration import is skipped with a notice instead of crashing -- the
pure-Python layers (robots.profile, deploy.contract/legacy, cli) keep
working. Training entry points import ``yanshi_rl_lab.tasks`` explicitly and
still fail loudly if Isaac Lab is broken.
"""

# Root modules of the training stack whose absence is tolerated at package
# import (Isaac Lab itself plus the Isaac Sim kit modules that only resolve
# inside a properly activated Isaac environment).
_TRAINING_STACK_ROOTS = ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaacsim", "pxr", "omni", "carb")

try:
    # Register Gym environments.
    from .tasks import *  # noqa: F401, F403
except ModuleNotFoundError as _exc:
    if _exc.name and _exc.name.split(".")[0] in _TRAINING_STACK_ROOTS:
        # Training stack absent: tolerated (see docstring). Not silent, though.
        import sys as _sys

        print(
            f"[yanshi_rl_lab] note: training stack not importable (missing {_exc.name}); "
            "Yanshi-* task registration skipped. Training needs the activated isaaclab "
            "environment; `yanshi doctor` can help diagnose.",
            file=_sys.stderr,
        )
    else:
        raise
