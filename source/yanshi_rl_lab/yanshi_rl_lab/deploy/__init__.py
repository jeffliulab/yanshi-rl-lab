# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Deployment layer: the unified policy contract and the MuJoCo sim2sim runtime.

House rule: **training is not done until sim2sim passes its gates.**

Modules (import cost matters -- consumers pick what they need):

- ``contract``    -- schema-v2 contract dataclasses (pure stdlib).
- ``legacy``      -- converters from the three predecessor formats to v2
  (stdlib + PyYAML for deploy.yaml).
- ``export``      -- reflection-based training-time exporter (duck-typed over
  a live Isaac Lab env; wired into scripts/rsl_rl/train.py).
- ``export_onnx`` -- pure-CPU rsl-rl checkpoint -> ONNX (needs torch).
- ``runtime``     -- MuJoCo closed-loop deployment (needs mujoco +
  onnxruntime; imported lazily on purpose so contract tooling stays light).

Only the pure-Python contract types are re-exported here; import
``yanshi_rl_lab.deploy.runtime`` explicitly for the MuJoCo side.
"""

from yanshi_rl_lab.deploy.contract import (  # noqa: F401
    SCHEMA_VERSION,
    ActionSpec,
    ActuatorGroupSpec,
    ContractV2,
    LayoutCarryover,
    ObsTermSpec,
    TimingSpec,
)
