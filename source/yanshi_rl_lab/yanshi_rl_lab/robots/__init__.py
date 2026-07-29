# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Robot profile layer: one directory per vendor, one sub-directory per model.

Each ``robots/<vendor>/<model>/`` package holds everything the framework needs
to know about one robot ("robot facts", never "how to drive it"):

- ``profile.py``      -- RobotProfile: model file locations (URDF/USD/MJCF pair),
                         joint table + real-robot SDK ordering, actuator groups
                         (kp/kd/effort/armature) and ``pd_mode``, the semantic
                         body map (feet / hip-yaw joints / torso / illegal-contact
                         bodies / root joint), initial pose and spawn height,
                         action scale.
- ``articulation.py`` -- the Isaac Lab ``ArticulationCfg`` built from the profile.

Task recipes must reference robots ONLY through semantic slots supplied by the
profile (e.g. ``profile.feet_bodies``), never through concrete body names.

The RobotProfile dataclass itself lands here in M1 (see the project plan).
"""
