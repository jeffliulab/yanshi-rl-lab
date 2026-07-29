# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Robot-independent MDP terms shared by all tasks.

Namespace layout follows the upstream unitree_rl_lab pattern: the Isaac Lab
mdp namespaces form the base and local modules extend/override on top, so task
configs can reference every term uniformly as ``mdp.<name>``.

Every term defined in this package must be free of concrete robot names --
robot specifics enter through function parameters resolved from the
RobotProfile by the task layer.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
