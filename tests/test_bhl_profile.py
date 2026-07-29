# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Pure-Python validation of the Berkeley Humanoid Lite profile (no Isaac
Lab, no GPU; the one MuJoCo cross-check skips itself when mujoco is absent).

Same stub-package trick as ``test_g1_profile.py``: the real package
``__init__`` imports Isaac Lab, so only the pure-Python profile modules are
imported underneath a namespace stub.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"

if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.robots.berkeley.humanoid_lite.profile import BHL_PROFILE  # noqa: E402
from yanshi_rl_lab.robots.profile import PD_MODES  # noqa: E402

# The humanoid configuration has exactly 22 actuated joints: 2 arms x 5 +
# 2 legs x 6 (official HUMANOID_LITE_JOINTS; vendor MJCF actuator block).
BHL_EXPECTED_DOF = 22

# Official actuator-group values, transcribed from the pinned assets repo
# (berkeley_humanoid_lite_assets/robots/berkeley_humanoid_lite.py,
# HUMANOID_LITE_CFG L152-190). This test is a transcription checksum: the
# profile must carry EXACTLY these numbers -- they are official, not tunable.
OFFICIAL_GROUPS = {
    "arms": {"effort_limit": 4, "velocity_limit": 10.0, "stiffness": 10, "damping": 2, "armature": 0.002},
    "legs": {"effort_limit": 6, "velocity_limit": 10.0, "stiffness": 20, "damping": 2, "armature": 0.007},
    "ankles": {"effort_limit": 6, "velocity_limit": 10.0, "stiffness": 20, "damping": 2, "armature": 0.002},
}


def test_sdk_joint_names_count_and_unique():
    names = BHL_PROFILE.joint_sdk_names
    assert names is not None
    assert len(names) == BHL_EXPECTED_DOF
    assert len(set(names)) == BHL_EXPECTED_DOF, "duplicate SDK joint names"


def test_sdk_joint_name_grouping():
    """Order contract (official policy/deploy order, three witnesses: the
    assets HUMANOID_LITE_JOINTS list, the deploy yaml, the MJCF actuator
    block): left arm 5 -> right arm 5 -> left leg 6 -> right leg 6."""
    names = BHL_PROFILE.joint_sdk_names
    assert all(n.startswith("arm_left_") for n in names[0:5])
    assert all(n.startswith("arm_right_") for n in names[5:10])
    assert all(n.startswith("leg_left_") for n in names[10:16])
    assert all(n.startswith("leg_right_") for n in names[16:22])


def test_every_sdk_joint_covered_by_exactly_one_actuator_group():
    for name in BHL_PROFILE.joint_sdk_names:
        matches = [
            group_name
            for group_name, group in BHL_PROFILE.actuator_groups.items()
            if any(re.fullmatch(expr, name) for expr in group.joint_names_expr)
        ]
        assert len(matches) == 1, f"{name}: matched by {matches or 'no group'}"


def test_asset_paths_exist():
    """The pinned vendor assets are fetched; every declared path must resolve
    (BHL also ships a USD, which is the official training spawn source)."""
    for field_name in ("urdf", "mjcf", "scene_mjcf", "usd"):
        path = BHL_PROFILE.asset_path(field_name)
        assert path.is_file(), f"{field_name} missing at {path}"


def test_pd_mode_valid():
    # Official training uses ImplicitActuatorCfg for every group; the deploy
    # loop must reproduce the training-side physics (see profile.py).
    assert BHL_PROFILE.pd_mode == "implicit"
    assert BHL_PROFILE.pd_mode in PD_MODES


def test_gains_are_the_official_values():
    assert set(BHL_PROFILE.actuator_groups) == set(OFFICIAL_GROUPS)
    for group_name, expected in OFFICIAL_GROUPS.items():
        group = BHL_PROFILE.actuator_groups[group_name]
        for field_name, value in expected.items():
            actual = getattr(group, field_name)
            assert actual == value, f"{group_name}.{field_name}: {actual} != official {value}"
            assert value > 0  # and the official values are sane positives


def test_semantic_slots():
    assert BHL_PROFILE.base_link == "base"
    assert BHL_PROFILE.feet_bodies
    assert BHL_PROFILE.undesired_contact_bodies
    assert BHL_PROFILE.leg_deviation_joints
    assert BHL_PROFILE.arm_deviation_joints
    # No waist on BHL -- empty ON PURPOSE; the task base drops the term.
    assert BHL_PROFILE.waist_deviation_joints == []


def test_deviation_regexes_match_real_joints():
    """Every deviation regex must match at least one SDK joint name."""
    names = BHL_PROFILE.joint_sdk_names
    for slot in ("leg_deviation_joints", "arm_deviation_joints"):
        for expr in getattr(BHL_PROFILE, slot):
            assert any(re.fullmatch(expr, n) for n in names), f"{slot}: {expr} matches nothing"


def test_default_pose_regexes_match_real_joints():
    names = BHL_PROFILE.joint_sdk_names
    for expr in BHL_PROFILE.default_joint_pos:
        assert any(re.fullmatch(expr, n) for n in names), f"default pose regex {expr} matches nothing"


def test_height_facts_are_deliberately_absent():
    """BHL's root frame sits at ground level (CAD export quirk): root height
    is not a usable signal, official training carries no height reward or
    termination, and the profile records that as None (base drops the
    terms). Spawn height is the official 0.0."""
    assert BHL_PROFILE.target_base_height_m is None
    assert BHL_PROFILE.min_base_height_m is None
    assert BHL_PROFILE.spawn_height_m == 0.0


def test_self_collisions_official_off():
    assert BHL_PROFILE.self_collisions is False  # official enabled_self_collisions=False


def test_root_joint_registered():
    assert BHL_PROFILE.root_joint_name == "base_freejoint"


def test_mjcf_matches_profile():
    """Cross-check the vendor MJCF against the profile (skips without mujoco).

    Also exercises the assets/merged symlink that fetch.py must create --
    the vendor MJCF cannot compile without it."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(BHL_PROFILE.asset_path("mjcf")))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert model.nu == BHL_EXPECTED_DOF
    assert set(joint_names) == set(BHL_PROFILE.joint_sdk_names)
    free_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    ]
    assert free_joints == [BHL_PROFILE.root_joint_name]
    # actuator order == deploy order (one of the three order witnesses)
    actuator_order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert actuator_order == [n for n in BHL_PROFILE.joint_sdk_names]
