# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Legacy -> v2 converters, on synthetic fixtures shaped exactly like the
three historical formats (plus optional checks against the real predecessor
files when they exist on this machine; skipped elsewhere).

Pure Python (no Isaac Lab / MuJoCo).
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.deploy import legacy  # noqa: E402

# Optional cross-checks against real v1 contracts from the deployment repo.
# They live outside this repository, so they are opt-in: point these at your
# own copies to enable the two tests below; leave them unset and those tests
# skip while the rest of the module still runs on the synthetic fixtures.
V1_G1_CONTRACT_ENV_VAR = "YANSHI_V1_G1_CONTRACT"
V1_GO2_CONTRACT_ENV_VAR = "YANSHI_V1_GO2_CONTRACT"


def _contract_from_env(var: str) -> Path | None:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else None


_V1_G1_CONTRACT = _contract_from_env(V1_G1_CONTRACT_ENV_VAR)
_V1_GO2_CONTRACT = _contract_from_env(V1_GO2_CONTRACT_ENV_VAR)


# ------------------------------------------------------------ deploy.yaml
def _deploy_yaml_fixture() -> dict:
    # 3-joint toy in the exact deploy.yaml layout: gains in SDK order,
    # default pose in policy order, map[i_policy] = SDK slot.
    return {
        "joint_ids_map": [2, 0, 1],  # policy j order -> SDK slot
        "step_dt": 0.02,
        # SDK order is [knee, ankle, hip] (see SDK_NAMES below), so:
        "stiffness": [150.0, 40.0, 100.0],  # SDK order: knee, ankle, hip
        "damping": [4.0, 1.0, 2.0],  # SDK order: knee, ankle, hip
        "default_joint_pos": [-0.1, 0.3, 0.0],  # policy order
        "commands": {
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-0.3, 0.3],
                    "ang_vel_z": [-0.8, 0.8],
                    "heading": None,
                }
            }
        },
        "actions": {
            "JointPositionAction": {
                "clip": None,
                "scale": [0.25, 0.25, 0.25],
                "offset": [-0.1, 0.3, 0.0],
                "joint_ids": None,
            }
        },
        "observations": {
            "base_ang_vel": {"params": {}, "clip": None, "scale": [0.2, 0.2, 0.2], "history_length": 5},
            "joint_pos_rel": {"params": {}, "clip": None, "scale": [1.0, 1.0, 1.0], "history_length": 5},
        },
    }


SDK_NAMES = ["knee", "ankle", "hip"]  # SDK order; policy order = hip, knee, ankle


def test_from_deploy_yaml():
    contract = legacy.from_deploy_yaml(
        _deploy_yaml_fixture(),
        joint_sdk_names=SDK_NAMES,
        pd_mode="implicit",
        foot_bodies=["foot"],
        root_joint_name="root",
        base_link="torso",
    )
    # Names reconstructed from the SDK map: policy[i] = sdk[map[i]].
    assert contract.joint_names == ["hip", "knee", "ankle"]
    # Gains brought back from SDK order to policy order.
    grp = contract.actuators["deploy_yaml_all"]
    assert grp.stiffness == [100.0, 150.0, 40.0]
    assert grp.damping == [2.0, 4.0, 1.0]
    # deploy.yaml never records these -- must be None, not invented:
    assert grp.effort_limit is None
    assert grp.armature is None
    assert contract.default_root_pos_w is None
    assert contract.timing.physics_dt_s is None and contract.timing.decimation is None
    assert contract.timing.policy_dt_s == 0.02
    # Command ranges survive (heading=None dropped).
    assert contract.command_ranges == {
        "lin_vel_x": [-0.5, 1.0],
        "lin_vel_y": [-0.3, 0.3],
        "ang_vel_z": [-0.8, 0.8],
    }
    # Constant scale lists collapse to scalars; obs order follows the file.
    assert contract.action.scale == 0.25
    assert [t.name for t in contract.obs_terms] == ["base_ang_vel", "joint_pos_rel"]
    assert contract.obs_terms[0].scale == 0.2
    assert contract.history_length == 5
    # validate() flags exactly the honest gaps, raises nothing.
    warnings = contract.validate()
    assert any("timing" in w for w in warnings)
    assert any("effort_limit" in w for w in warnings)


def test_from_deploy_yaml_wrong_sdk_length_rejected():
    with pytest.raises(ValueError, match="joint_sdk_names"):
        legacy.from_deploy_yaml(_deploy_yaml_fixture(), joint_sdk_names=["a", "b"])


# --------------------------------------------------------- v1 G1 contract
def _v1_g1_fixture() -> dict:
    return {
        "generated_utc": "2026-07-20T06:37:14+00:00",
        "task": "Unitree-G1-29dof-Velocity-Normal",
        "policy": "/somewhere/policy.onnx",
        "vx": 0.6,
        "joint_names": ["hip", "knee"],
        "default_joint_pos": [-0.1, 0.3],
        "default_root_pos_w": [0.0, 0.0, 0.8],
        "obs_terms": [
            {"term": "base_ang_vel", "dim_with_history": 15},
            {"term": "joint_pos_rel", "dim_with_history": 10},
        ],
        "obs_dim": 25,
        "history_length": 5,
        "term_scales": {"base_ang_vel": 0.2, "joint_pos_rel": 1.0},
        "action": {"scale": 0.25, "dim": 2},
        "timing": {"policy_dt_s": 0.02, "physics_dt_s": 0.005, "decimation": 4},
        "actuators": {
            "M1": {
                "joint_names": ["hip", "knee"],
                "stiffness": [100.0, 150.0],
                "damping": [2.0, 4.0],
                "effort_limit": [88.0, 139.0],
                "armature": [0.01, 0.01],
            }
        },
        "layout_carryover": {
            "dumped_from_task": "Unitree-G1-29dof-Velocity-Turn",
            "dumped_utc": "2026-07-26T10:50:43+00:00",
            "why_valid": "layout untouched across overlay generations",
            "evidence": ["bitwise diff vs Normal contract"],
            "note": "re-dump if obs terms change",
        },
    }


def test_from_contract_v1_g1():
    contract = legacy.from_contract_v1_g1(
        _v1_g1_fixture(),
        pd_mode="implicit",
        foot_bodies=[".*ankle_roll.*"],
        root_joint_name="floating_base_joint",
        base_link="torso_link",
        joint_sdk_names=["knee", "hip"],
    )
    assert contract.joint_names == ["hip", "knee"]
    assert contract.joint_ids_map == [1, 0]
    assert contract.obs_terms[0].dim == 3  # 15 / history 5
    assert contract.obs_terms[1].dim == 2
    assert contract.obs_terms[0].scale == 0.2
    assert contract.actuators["M1"].armature == [0.01, 0.01]
    assert contract.pd_mode == "implicit"
    assert contract.layout_carryover.dumped_from_task == "Unitree-G1-29dof-Velocity-Turn"
    # The trace probe vx is provenance, NOT a command range.
    assert contract.command_ranges is None
    assert contract.extra["v1_trace_vx"] == 0.6
    warnings = contract.validate()
    assert any("command_ranges" in w for w in warnings)
    contract.require_runtime_fields()  # complete for sim2sim -- must not raise


# -------------------------------------------------------- v1 Go2 contract
def _v1_go2_fixture() -> dict:
    return {
        "task": "Unitree-Go2-Velocity",
        "joint_names": ["FL_hip_joint", "FL_calf_joint"],
        "default_joint_pos": [0.1, -1.5],
        "gains": {
            "FL_hip_joint": {"kp": 25.0, "kd": 0.5, "effort_limit": 23.7},
            "FL_calf_joint": {"kp": 25.0, "kd": 0.5, "effort_limit": 45.43},
        },
        "term_order": ["base_ang_vel", "joint_pos_rel"],
        "term_scales": {"base_ang_vel": 0.2, "joint_pos_rel": 1.0},
        "history_length": 1,
        "action": {"scale": 0.25, "use_default_offset": True, "dim": 2},
        "timing": {"policy_dt_s": 0.02, "physics_dt_s": 0.005, "decimation": 4},
        "obs_dim": 6,
    }


def test_from_contract_v1_go2():
    contract = legacy.from_contract_v1_go2(
        _v1_go2_fixture(),
        pd_mode="explicit",  # alice-house manifest fact for Go2
        foot_bodies=[".*_foot"],
        root_joint_name="root_joint",
        base_link="base",
    )
    assert contract.joint_names == ["FL_hip_joint", "FL_calf_joint"]
    grp = contract.actuators["go2_v1_all"]
    assert grp.stiffness == [25.0, 25.0]
    assert grp.effort_limit == [23.7, 45.43]
    assert grp.armature is None  # never recorded by hand -- honest None
    assert contract.default_root_pos_w is None
    assert contract.history_length == 1
    assert [t.name for t in contract.obs_terms] == ["base_ang_vel", "joint_pos_rel"]
    assert all(t.dim is None for t in contract.obs_terms)  # go2 v1 has no per-term dims
    warnings = contract.validate()
    assert any("armature" in w for w in warnings)
    with pytest.raises(ValueError, match="default_root_pos_w"):
        contract.require_runtime_fields()  # spawn height honestly missing


def test_go2_missing_gain_rejected():
    broken = _v1_go2_fixture()
    del broken["gains"]["FL_calf_joint"]
    with pytest.raises(ValueError, match="lacks gains"):
        legacy.from_contract_v1_go2(broken)


def test_go2_non_default_offset_rejected():
    fixture = _v1_go2_fixture()
    fixture["action"]["use_default_offset"] = False
    with pytest.raises(ValueError, match="use_default_offset"):
        legacy.from_contract_v1_go2(fixture)


# ----------------------------------------- optional: real predecessor files
@pytest.mark.skipif(
    _V1_G1_CONTRACT is None or not _V1_G1_CONTRACT.exists(),
    reason=f"set ${V1_G1_CONTRACT_ENV_VAR} to a v1 G1 contract.json to run this",
)
def test_real_g1_contract_converts():
    d = json.loads(_V1_G1_CONTRACT.read_text(encoding="utf-8"))
    contract = legacy.from_contract_v1_g1(
        d,
        pd_mode="implicit",
        foot_bodies=[".*ankle_roll.*"],
        root_joint_name="floating_base_joint",
        base_link="torso_link",
    )
    assert len(contract.joint_names) == 29
    assert contract.obs_dim == 480
    assert contract.history_length == 5
    assert contract.layout_carryover is not None
    contract.require_runtime_fields()
    contract.validate()


@pytest.mark.skipif(
    _V1_GO2_CONTRACT is None or not _V1_GO2_CONTRACT.exists(),
    reason=f"set ${V1_GO2_CONTRACT_ENV_VAR} to a v1 Go2 contract.json to run this",
)
def test_real_go2_contract_converts():
    d = json.loads(_V1_GO2_CONTRACT.read_text(encoding="utf-8"))
    contract = legacy.from_contract_v1_go2(d, pd_mode="explicit")
    assert len(contract.joint_names) == 12
    assert contract.obs_dim == 45
    contract.validate()
