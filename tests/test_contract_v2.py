# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Schema-v2 contract: round trip, validation, unknown-field tolerance.

Pure Python (no Isaac Lab / MuJoCo): loads the deploy modules through the
same namespace stub the profile tests use.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.deploy.contract import (  # noqa: E402
    SCHEMA_VERSION,
    ActionSpec,
    ActuatorGroupSpec,
    ContractV2,
    LayoutCarryover,
    ObsTermSpec,
    TimingSpec,
)


def _minimal_contract(**overrides) -> ContractV2:
    base = dict(
        task="Yanshi-Velocity-Flat-Test-Bot-v0",
        generated_utc="2026-07-29T00:00:00+00:00",
        joint_names=["j1", "j2"],
        default_joint_pos=[0.1, -0.2],
        obs_terms=[
            ObsTermSpec(name="base_ang_vel", scale=0.2, history_length=2, dim=3),
            ObsTermSpec(name="joint_pos_rel", scale=1.0, history_length=2, dim=2),
        ],
        obs_dim=10,
        history_length=2,
        action=ActionSpec(scale=0.25, dim=2),
        timing=TimingSpec(policy_dt_s=0.02, physics_dt_s=0.005, decimation=4),
        actuators={
            "grp": ActuatorGroupSpec(
                joint_names=["j1", "j2"],
                stiffness=[100.0, 40.0],
                damping=[2.0, 1.0],
                effort_limit=[88.0, 25.0],
                armature=[0.01, 0.01],
            )
        },
        pd_mode="implicit",
        default_root_pos_w=[0.0, 0.0, 0.8],
        foot_bodies=["foot_l", "foot_r"],
        root_joint_name="floating_base_joint",
        base_link="torso",
        joint_sdk_names=["j2", "j1"],
        joint_ids_map=[1, 0],
        command_ranges={"lin_vel_x": [-0.5, 1.0]},
    )
    base.update(overrides)
    return ContractV2(**base)


def test_round_trip(tmp_path):
    contract = _minimal_contract(
        layout_carryover=LayoutCarryover(
            dumped_from_task="Old-Task", dumped_utc="2026-07-26T00:00:00+00:00", why_valid="unchanged layout"
        )
    )
    path = tmp_path / "contract.json"
    contract.to_json(path)
    loaded = ContractV2.from_json(path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.joint_names == contract.joint_names
    assert loaded.actuators["grp"].effort_limit == [88.0, 25.0]
    assert loaded.layout_carryover.dumped_from_task == "Old-Task"
    assert [t.name for t in loaded.obs_terms] == ["base_ang_vel", "joint_pos_rel"]
    assert loaded.timing.decimation == 4
    assert loaded.to_json_dict() == contract.to_json_dict()


def test_fully_specified_contract_validates_clean():
    assert _minimal_contract().validate() == []


def test_wrong_schema_version_rejected(tmp_path):
    contract = _minimal_contract()
    d = contract.to_json_dict()
    d["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        ContractV2.from_json_dict(d)


def test_unknown_field_tolerated_but_warned(tmp_path):
    d = _minimal_contract().to_json_dict()
    d["from_the_future"] = {"x": 1}
    loaded = ContractV2.from_json_dict(d)
    assert loaded.extra == {"from_the_future": {"x": 1}}
    assert any("unknown fields" in w for w in loaded.validate())
    # ...and survives a save/load round trip.
    d2 = loaded.to_json_dict()
    assert d2["from_the_future"] == {"x": 1}


def test_missing_pd_mode_warns_and_blocks_runtime():
    contract = _minimal_contract(pd_mode=None)
    assert any("pd_mode" in w for w in contract.validate())
    with pytest.raises(ValueError, match="pd_mode"):
        contract.require_runtime_fields()


def test_invalid_pd_mode_raises():
    contract = _minimal_contract(pd_mode="magic")
    with pytest.raises(ValueError, match="pd_mode"):
        contract.validate()


def test_obs_dim_mismatch_raises():
    contract = _minimal_contract(obs_dim=11)
    with pytest.raises(ValueError, match="obs_dim"):
        contract.validate()


def test_bad_joint_ids_map_raises():
    with pytest.raises(ValueError, match="permutation"):
        _minimal_contract(joint_ids_map=[0, 0]).validate()
    with pytest.raises(ValueError, match="inconsistent"):
        _minimal_contract(joint_ids_map=[0, 1]).validate()  # sdk order is reversed


def test_actuator_coverage_enforced():
    contract = _minimal_contract(
        actuators={"grp": ActuatorGroupSpec(joint_names=["j1"], stiffness=[100.0], damping=[2.0])}
    )
    with pytest.raises(ValueError, match="missing from actuator groups"):
        contract.validate()


def test_gains_lookup():
    gains = _minimal_contract().gains()
    assert gains["j1"] == (100.0, 2.0, 88.0, 0.01)
    assert gains["j2"] == (40.0, 1.0, 25.0, 0.01)


def test_missing_effort_limit_warned_not_invented():
    contract = _minimal_contract(
        actuators={
            "grp": ActuatorGroupSpec(
                joint_names=["j1", "j2"], stiffness=[100.0, 40.0], damping=[2.0, 1.0]
            )
        }
    )
    warnings = contract.validate()
    assert any("effort_limit" in w for w in warnings)
    assert contract.gains()["j1"][2] is None  # None, never a made-up number
