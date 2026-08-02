# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1, one sub-package per configuration Unitree ships.

Variant names are upstream's own: the vendor publishes ``g1_29dof_rev_1_0.urdf``
and ``g1_23dof_rev_1_0.urdf`` out of the same ``unitree_ros`` checkout, so the
two configurations share one fetched asset tree (``assets/unitree/g1/``) and
differ only in which files their profiles point at."""
