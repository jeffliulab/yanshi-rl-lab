# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Velocity-tracking task family (the v0.1 flagship task).

``velocity_env_cfg.py`` (lands in M1) is the vendor-neutral base: every body or
joint reference goes through the RobotProfile's semantic slots, and every
parameter that can be derived from the profile is derived, so that a robot
with a complete profile trains with an empty per-robot overlay.
"""
