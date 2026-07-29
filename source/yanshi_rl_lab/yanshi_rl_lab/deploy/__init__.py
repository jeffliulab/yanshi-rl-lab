# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Deployment layer: the unified policy contract and the MuJoCo sim2sim runtime.

House rule: **training is not done until sim2sim passes its gates.**

Planned contents (milestone M2):

- ``contract.py`` -- the ``schema_version: 2`` policy contract (single source of
  truth for joints/gains/pd_mode/observation layout/action scaling/timing) and
  the single reflection-based exporter that dumps it from a live Isaac Lab env
  at training start. This replaces the three incompatible legacy formats.
- ``runtime.py``  -- contract-driven MuJoCo deployment loop (observation
  assembly, PD actuation in explicit or implicit mode, odometry, recording),
  parameterized by the RobotProfile (root joint name, foot bodies, scene).
"""
