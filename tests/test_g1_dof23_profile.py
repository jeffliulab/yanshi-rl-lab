# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Validation of the Unitree G1 23-DoF profile, and of what makes it a peer.

Two jobs beyond the usual profile checks:

1. **Motor count is not DoF count on this robot.** The vendor MJCF declares 29
   motors over a 23-DoF body (Unitree keeps one DDS layout for both
   configurations; the six absent joints sit on massless bodies at z=20,
   detached from the tree). Reading 29 motors as 29 joints is precisely how
   the first published G1 leaderboard row got measured on the wrong model, so
   the MuJoCo cross-check below counts joints that actually reach the
   floating base -- not actuators.
2. **The derivation from dof29 must stay honest.** The actuator groups here
   are the dof29 groups restricted to joints this body has, justified by the
   two MJCFs carrying identical motor ctrlranges. The tests assert exactly
   that justification rather than trusting the copy.
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

from yanshi_rl_lab.robots.profile import PD_MODES  # noqa: E402
from yanshi_rl_lab.robots.unitree.g1.dof23.profile import G1_DOF23_PROFILE as P  # noqa: E402
from yanshi_rl_lab.robots.unitree.g1.dof29.profile import G1_PROFILE as P29  # noqa: E402

# 2 legs x 6 + waist_yaw 1 + 2 arms x 5 (shoulder pitch/roll/yaw, elbow,
# wrist_roll). Source: Unitree g1_joint_index_dds.md, "23DOF 版本".
EXPECTED_DOF = 23

# The six joints dof29 has and this configuration does not.
ABSENT_JOINTS = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def test_no_isaaclab_leak():
    assert "isaaclab" not in sys.modules, "profile modules must stay pure Python"


def test_identity_is_a_peer_of_dof29():
    assert P.key == "unitree/g1/dof23"
    assert P.vendor == P29.vendor and P.model == P29.model
    assert P.variant != P29.variant


def test_shares_one_asset_tree_with_dof29():
    """The whole point of keying assets by model: no second fetch, no drift."""
    assert P.asset_key == P29.asset_key == "unitree/g1"
    assert P.urdf != P29.urdf and P.mjcf != P29.mjcf and P.scene_mjcf != P29.scene_mjcf
    for field in ("urdf", "mjcf", "scene_mjcf"):
        assert getattr(P, field).startswith("unitree/g1/")


def test_sdk_joint_table():
    names = P.joint_sdk_names
    assert names is not None
    assert len(names) == EXPECTED_DOF
    assert len(set(names)) == EXPECTED_DOF, "duplicate SDK joint names"
    assert not (set(names) & ABSENT_JOINTS), "table names a joint this body lacks"


def test_sdk_order_is_legs_then_waist_then_arms():
    """Contract from g1_joint_index_dds.md: 6 + 6 + 1 + 5 + 5."""
    n = P.joint_sdk_names
    assert all(x.startswith("left_") for x in n[0:6])
    assert all(x.startswith("right_") for x in n[6:12])
    assert n[12] == "waist_yaw_joint"
    assert all(x.startswith("left_") for x in n[13:18])
    assert all(x.startswith("right_") for x in n[18:23])


def test_sdk_table_is_dof29_minus_the_absent_joints():
    """Same joints in the same relative order -- only the absent ones removed.

    Guards against a hand-reordered table: the two configurations must agree
    wherever they overlap, or a policy trained on one cannot be reasoned about
    against the other.
    """
    assert [x for x in P29.joint_sdk_names if x not in ABSENT_JOINTS] == P.joint_sdk_names


def test_pd_mode_valid():
    assert P.pd_mode in PD_MODES
    assert P.pd_mode == P29.pd_mode


def test_no_actuator_group_references_an_absent_joint():
    """Isaac Lab raises when a joint regex matches nothing; catch it on CPU first."""
    all_joints = set(P.joint_sdk_names)
    for group_name, group in P.actuator_groups.items():
        for expr in group.joint_names_expr:
            matched = {j for j in all_joints if re.fullmatch(expr, j)}
            assert matched, f"group {group_name!r} regex {expr!r} matches no joint on this body"


def test_every_joint_is_covered_by_exactly_one_actuator_group():
    counts = {j: 0 for j in P.joint_sdk_names}
    for group in P.actuator_groups.values():
        for expr in group.joint_names_expr:
            for j in counts:
                if re.fullmatch(expr, j):
                    counts[j] += 1
    uncovered = [j for j, c in counts.items() if c == 0]
    doubled = [j for j, c in counts.items() if c > 1]
    assert not uncovered, f"joints with no actuator group: {uncovered}"
    assert not doubled, f"joints in more than one actuator group: {doubled}"


def test_the_wrist_pitch_yaw_group_is_absent_not_empty():
    """dof29's W4010-25 drove joints this body lacks; an empty group would be
    a regex matching nothing, which fails loudly at articulation build time."""
    assert "W4010-25" in P29.actuator_groups
    assert "W4010-25" not in P.actuator_groups
    assert set(P.actuator_groups) == set(P29.actuator_groups) - {"W4010-25"}


def test_shared_groups_keep_dof29_gains():
    """The derivation is 'same motors, therefore same gains' -- so the numbers
    must actually be the same, not quietly re-tuned."""
    for name, group in P.actuator_groups.items():
        ref = P29.actuator_groups[name]
        assert group.effort_limit == ref.effort_limit
        assert group.velocity_limit == ref.velocity_limit
        assert group.armature == ref.armature


def test_height_facts_match_dof29():
    """Measured identical (torso_link at 0.8470 m in both models); see profile."""
    assert P.target_base_height_m == P29.target_base_height_m
    assert P.min_base_height_m == P29.min_base_height_m
    assert P.spawn_height_m == P29.spawn_height_m
    assert P.target_base_height_m > P.min_base_height_m


def test_default_pose_names_only_joints_this_body_has():
    for pattern in P.default_joint_pos:
        matched = [j for j in P.joint_sdk_names if re.fullmatch(pattern, j)]
        assert matched, f"default_joint_pos pattern {pattern!r} matches no joint"


# ---------------------------------------------------------------- MuJoCo side


def _model():
    mujoco = pytest.importorskip("mujoco")
    try:
        path = P.asset_path("mjcf")
    except FileNotFoundError:
        pytest.skip("vendor assets not fetched; run: python assets/fetch.py unitree/g1")
    return mujoco, mujoco.MjModel.from_xml_path(str(path))


def _attached_hinges(mujoco, m) -> list[str]:
    """Hinge joints whose body chain actually reaches the floating base."""
    attached = []
    for i in range(m.njnt):
        if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        body, chain = m.jnt_bodyid[i], []
        while body != 0:
            chain.append(body)
            body = m.body_parentid[body]
        has_root = any(
            m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for b in chain
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
        )
        if has_root:
            attached.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
    return attached


def test_mjcf_has_23_attached_joints_despite_29_motors():
    """The check that would have caught the wrong-scene bug.

    Counting actuators here gives 29 and tells you nothing; counting joints
    that reach the floating base gives 23, which is what the robot is.
    """
    mujoco, m = _model()
    attached = _attached_hinges(mujoco, m)
    assert m.nu == 29, "vendor keeps a 29-motor DDS layout on this body"
    assert len(attached) == EXPECTED_DOF
    detached = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(m.njnt)
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    } - set(attached)
    assert detached == ABSENT_JOINTS


def test_sdk_table_matches_the_attached_joints():
    mujoco, m = _model()
    assert set(P.joint_sdk_names) == set(_attached_hinges(mujoco, m))


def test_root_joint_and_semantic_bodies_exist():
    mujoco, m = _model()
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
    assert P.root_joint_name in names
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    assert P.base_link in bodies
    assert [b for b in bodies if b and re.fullmatch(P.feet_bodies, b)], "feet regex matches no body"


def test_motor_ctrlranges_are_identical_to_dof29():
    """The evidence the actuator-group derivation rests on.

    If a future vendor bump changes a shared joint's motor, this fails and the
    'same motors, same gains' argument has to be re-made rather than assumed.
    """
    pytest.importorskip("mujoco")
    try:
        paths = {"dof23": P.asset_path("mjcf"), "dof29": P29.asset_path("mjcf")}
    except FileNotFoundError:
        pytest.skip("vendor assets not fetched; run: python assets/fetch.py unitree/g1")

    def ctrlranges(path):
        text = Path(path).read_text(encoding="utf-8")
        out = {}
        for block in re.finditer(r"<motor\s+([^>]*?)/>", text):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', block.group(1)))
            out[attrs.get("joint", attrs.get("name"))] = attrs.get("ctrlrange")
        return out

    a, b = ctrlranges(paths["dof23"]), ctrlranges(paths["dof29"])
    shared = set(a) & set(b)
    assert len(shared) == 29
    assert {j: a[j] for j in shared} == {j: b[j] for j in shared}
