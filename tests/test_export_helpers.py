# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Pure-function coverage of the training-time contract exporter.

The full exporter reflects over a live Isaac Lab env (cannot run in CI /
without a GPU), so every assembly helper is duck-typed and exercised here
with fake data; the ONNX exporter's architecture inference is covered the
same way. No Isaac Lab, no torch, numpy only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.deploy.export import (  # noqa: E402
    build_actuator_specs,
    build_command_ranges,
    build_obs_term_specs,
    resolve_body_regex,
    sdk_index_map,
)
from yanshi_rl_lab.deploy.export_onnx import infer_mlp_dims  # noqa: E402


# ------------------------------------------------------------ sdk_index_map
def test_sdk_index_map_matches_upstream_semantics():
    # Same shape as the real G1 case: policy (Isaac breadth-first) order vs
    # SDK grouped order; map[i_policy] = SDK slot.
    policy = ["left_hip", "right_hip", "waist", "left_knee"]
    sdk = ["left_hip", "left_knee", "right_hip", "waist"]
    assert sdk_index_map(policy, sdk) == [0, 2, 3, 1]


def test_sdk_index_map_rejects_mismatch():
    with pytest.raises(ValueError, match="not in joint_sdk_names"):
        sdk_index_map(["a", "b"], ["a", "c"])
    with pytest.raises(ValueError, match="profile mismatch"):
        sdk_index_map(["a"], ["a", "b"])


# -------------------------------------------------------- resolve_body_regex
def test_resolve_body_regex():
    bodies = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
    assert resolve_body_regex(".*ankle_roll.*", bodies) == [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    assert resolve_body_regex("torso_link", bodies) == ["torso_link"]
    with pytest.raises(ValueError, match="matches none"):
        resolve_body_regex(".*paw.*", bodies)


# ------------------------------------------------------- build_actuator_specs
class _FakeActuator:
    """Duck-typed stand-in for an Isaac Lab actuator (numpy instead of torch)."""

    def __init__(self, joint_names, **arrays):
        self.joint_names = joint_names
        for key, value in arrays.items():
            setattr(self, key, value)


def test_build_actuator_specs_takes_row0_of_batched_tensors():
    act = _FakeActuator(
        ["hip", "knee"],
        stiffness=np.array([[100.0, 150.0], [100.0, 150.0]]),  # (num_envs, joints)
        damping=np.array([[2.0, 4.0], [2.0, 4.0]]),
        effort_limit=np.array([[88.0, 139.0], [88.0, 139.0]]),
        armature=np.array([[0.01, 0.01], [0.01, 0.01]]),
    )
    specs = build_actuator_specs({"M1": act})
    assert specs["M1"].joint_names == ["hip", "knee"]
    assert specs["M1"].stiffness == [100.0, 150.0]
    assert specs["M1"].effort_limit == [88.0, 139.0]
    assert specs["M1"].armature == [0.01, 0.01]


def test_build_actuator_specs_missing_field_stays_none():
    specs = build_actuator_specs({"M1": _FakeActuator(["hip"], stiffness=np.array([100.0]))})
    assert specs["M1"].damping is None
    assert specs["M1"].armature is None


# ------------------------------------------------------- build_obs_term_specs
class _FakeObsCfg:
    def __init__(self, scale=None, history_length=0):
        self.scale = scale
        self.history_length = history_length


def test_build_obs_term_specs_real_robot_layout():
    # The exact G1 real-robot-format shape: 6 terms, group history 5.
    names = ["base_ang_vel", "projected_gravity", "velocity_commands",
             "joint_pos_rel", "joint_vel_rel", "last_action"]
    dims = [15, 15, 15, 145, 145, 145]
    cfgs = [
        _FakeObsCfg(scale=np.full(3, 0.2)),
        _FakeObsCfg(),
        _FakeObsCfg(),
        _FakeObsCfg(),
        _FakeObsCfg(scale=np.full(29, 0.05)),
        _FakeObsCfg(),
    ]
    specs = build_obs_term_specs(names, dims, cfgs, group_history=5)
    assert [t.name for t in specs] == names
    assert [t.dim for t in specs] == [3, 3, 3, 29, 29, 29]
    assert all(t.history_length == 5 for t in specs)
    assert specs[0].scale == 0.2  # constant array collapses to scalar
    assert specs[4].scale == 0.05
    assert sum(t.dim_with_history for t in specs) == 480


def test_build_obs_term_specs_rejects_indivisible_history():
    with pytest.raises(ValueError, match="not divisible"):
        build_obs_term_specs(["t"], [10], [_FakeObsCfg()], group_history=3)


# ------------------------------------------------------- build_command_ranges
class _Ranges:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Cmd:
    def __init__(self, ranges=None, limit_ranges=None):
        if ranges is not None:
            self.ranges = ranges
        if limit_ranges is not None:
            self.limit_ranges = limit_ranges


class _Commands:
    def __init__(self, base_velocity=None):
        if base_velocity is not None:
            self.base_velocity = base_velocity


def test_build_command_ranges_prefers_limit_ranges():
    cmd = _Cmd(
        ranges=_Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=None),
        limit_ranges=_Ranges(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8), heading=None),
    )
    assert build_command_ranges(_Commands(cmd)) == {
        "lin_vel_x": [-0.5, 1.0],
        "lin_vel_y": [-0.3, 0.3],
        "ang_vel_z": [-0.8, 0.8],
    }


def test_build_command_ranges_no_base_velocity():
    assert build_command_ranges(_Commands()) is None


# ------------------------------------------------------------ infer_mlp_dims
class _Shaped:
    def __init__(self, *shape):
        self.shape = shape


def test_infer_mlp_dims_from_state_dict_shapes():
    # rsl-rl 5.0 actor layout: mlp.0/2/4/6 Linear + distribution noise param.
    sd = {
        "mlp.0.weight": _Shaped(512, 480),
        "mlp.0.bias": _Shaped(512),
        "mlp.2.weight": _Shaped(256, 512),
        "mlp.2.bias": _Shaped(256),
        "mlp.4.weight": _Shaped(128, 256),
        "mlp.4.bias": _Shaped(128),
        "mlp.6.weight": _Shaped(29, 128),
        "mlp.6.bias": _Shaped(29),
        "distribution.std_param": _Shaped(29),
    }
    assert infer_mlp_dims(sd) == [480, 512, 256, 128, 29]


def test_infer_mlp_dims_rejects_broken_chain():
    sd = {"mlp.0.weight": _Shaped(512, 480), "mlp.2.weight": _Shaped(256, 999)}
    with pytest.raises(ValueError, match="chain broken"):
        infer_mlp_dims(sd)


def test_infer_mlp_dims_rejects_non_mlp():
    with pytest.raises(ValueError, match="mlp"):
        infer_mlp_dims({"lstm.weight": _Shaped(4, 4)})
