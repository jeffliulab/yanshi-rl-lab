# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite, one sub-package per configuration upstream ships.

Upstream publishes two robots: ``biped`` (legs only, 12 DoF) and ``humanoid``
(legs + two 5-DoF arms, 22 DoF). Only ``humanoid`` is registered -- it is the
same morphology class as the other launch robots, which is what makes the
cross-robot benchmark comparable. Adding ``biped`` later means adding a
sibling directory, nothing else."""
