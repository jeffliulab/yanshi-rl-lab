# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Pure-Python validation of the Unitree G1 profile (no Isaac Lab, no GPU).

The real ``yanshi_rl_lab/__init__.py`` imports ``.tasks`` (and therefore Isaac
Lab), so this test installs a namespace stub for the top-level package and
imports only the pure-Python profile modules underneath it. The ``robots``
sub-packages are docstring-only and safe to execute.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"


def _install_stub_package() -> None:
    if "yanshi_rl_lab" not in sys.modules:
        stub = types.ModuleType("yanshi_rl_lab")
        stub.__path__ = [str(_PKG_DIR)]
        sys.modules["yanshi_rl_lab"] = stub


_install_stub_package()

from yanshi_rl_lab.robots.profile import (  # noqa: E402
    ASSETS_ROOT_ENV_VAR,
    PD_MODES,
    assets_root,
)
from yanshi_rl_lab.robots.unitree.g1.dof29.profile import G1_PROFILE  # noqa: E402

# The G1 29-DoF deploy configuration has exactly 29 SDK joints:
# 2 legs x 6 + waist 3 + 2 arms x 7.
G1_EXPECTED_DOF = 29


def test_no_isaaclab_leak():
    """The profile import chain must not pull in Isaac Lab."""
    assert "isaaclab" not in sys.modules, "profile modules must stay pure Python"


def test_sdk_joint_names_count_and_unique():
    names = G1_PROFILE.joint_sdk_names
    assert names is not None
    assert len(names) == G1_EXPECTED_DOF
    assert len(set(names)) == G1_EXPECTED_DOF, "duplicate SDK joint names"


def test_sdk_joint_name_grouping():
    """Order contract: left leg 6 -> right leg 6 -> waist 3 -> left arm 7 -> right arm 7."""
    names = G1_PROFILE.joint_sdk_names
    assert all(n.startswith("left_") for n in names[0:6])
    assert all(n.startswith("right_") for n in names[6:12])
    assert all(n.startswith("waist_") for n in names[12:15])
    assert all(n.startswith("left_") for n in names[15:22])
    assert all(n.startswith("right_") for n in names[22:29])


def test_pd_mode_valid():
    assert G1_PROFILE.pd_mode == "implicit"
    assert G1_PROFILE.pd_mode in PD_MODES


def test_invalid_pd_mode_rejected():
    with pytest.raises(ValueError, match="pd_mode"):
        dataclasses.replace(G1_PROFILE, pd_mode="magic")


def test_asset_fields_are_relative():
    for field_name in ("urdf", "mjcf", "scene_mjcf"):
        value = getattr(G1_PROFILE, field_name)
        assert value is not None
        assert not Path(value).is_absolute(), f"{field_name} must be relative to the assets root"
    assert G1_PROFILE.usd is None, "G1 spawns from URDF (no USD asset)"


def test_asset_path_rejects_none_field():
    with pytest.raises(ValueError, match="usd"):
        G1_PROFILE.asset_path("usd")


def test_asset_path_rejects_unknown_field():
    with pytest.raises(ValueError):
        G1_PROFILE.asset_path("meshes")


def test_actuator_groups():
    groups = G1_PROFILE.actuator_groups
    assert set(groups) == {"N7520-14.3", "N7520-22.5", "N5020-16", "W4010-25"}
    for name, group in groups.items():
        assert group.joint_names_expr, name
        assert group.effort_limit > 0, name
        assert group.velocity_limit > 0, name
        assert group.armature > 0, name


def test_semantic_slots_nonempty():
    assert G1_PROFILE.base_link
    assert G1_PROFILE.feet_bodies
    assert G1_PROFILE.undesired_contact_bodies
    assert G1_PROFILE.leg_deviation_joints
    assert G1_PROFILE.arm_deviation_joints
    assert G1_PROFILE.waist_deviation_joints


def test_height_facts_consistent():
    assert 0.0 < G1_PROFILE.min_base_height_m < G1_PROFILE.target_base_height_m
    assert G1_PROFILE.target_base_height_m <= G1_PROFILE.spawn_height_m


def test_assets_root_default_and_override(monkeypatch):
    monkeypatch.delenv(ASSETS_ROOT_ENV_VAR, raising=False)
    default_root = assets_root()
    assert default_root == Path(__file__).resolve().parents[1] / "assets"
    monkeypatch.setenv(ASSETS_ROOT_ENV_VAR, "/somewhere/else")
    assert assets_root() == Path("/somewhere/else")
