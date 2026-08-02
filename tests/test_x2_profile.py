# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Pure-Python validation of the AgiBot Lingxi X2 profile (no Isaac Lab, no GPU).

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

from yanshi_rl_lab.robots.agibot.x2.profile import X2_PROFILE  # noqa: E402
from yanshi_rl_lab.robots.profile import PD_MODES  # noqa: E402
from yanshi_rl_lab.robots.unitree.g1.profile import G1_PROFILE  # noqa: E402

# X2-Ultra has exactly 31 actuated joints: 2 legs x 6 + waist 3 + head 2
# + 2 arms x 7 (verified against the vendor MJCF actuator block).
X2_EXPECTED_DOF = 31

# Order-of-magnitude sanity bound for the gain cross-check against G1:
# same-size-class robots must not differ by 10x on same-role joints.
MAX_GAIN_RATIO = 10.0


def _gain_values(gain) -> list[float]:
    """Flatten a profile stiffness/damping entry (float or dict) to floats."""
    return list(gain.values()) if isinstance(gain, dict) else [gain]


def _g1_gain_for(joint_name: str, kind: str) -> float:
    """G1 gain (stiffness/damping) for the group+regex matching a joint name."""
    for group in G1_PROFILE.actuator_groups.values():
        if not any(re.fullmatch(expr, joint_name) for expr in group.joint_names_expr):
            continue
        gain = getattr(group, kind)
        if isinstance(gain, dict):
            for expr, value in gain.items():
                if re.fullmatch(expr, joint_name):
                    return value
            raise AssertionError(f"G1 group matches {joint_name} but no {kind} regex does")
        return gain
    raise AssertionError(f"no G1 actuator group matches joint {joint_name}")


def _x2_gain_for(joint_name: str, kind: str) -> float:
    for group in X2_PROFILE.actuator_groups.values():
        if any(re.fullmatch(expr, joint_name) for expr in group.joint_names_expr):
            gain = getattr(group, kind)
            assert isinstance(gain, float), "X2 groups use scalar gains"
            return gain
    raise AssertionError(f"no X2 actuator group matches joint {joint_name}")


def test_sdk_joint_names_count_and_unique():
    names = X2_PROFILE.joint_sdk_names
    assert names is not None
    assert len(names) == X2_EXPECTED_DOF
    assert len(set(names)) == X2_EXPECTED_DOF, "duplicate SDK joint names"


def test_sdk_joint_name_grouping():
    """Order contract (URDF document order = MJCF actuator order):
    left leg 6 -> right leg 6 -> waist 3 -> head 2 -> left arm 7 -> right arm 7."""
    names = X2_PROFILE.joint_sdk_names
    assert all(n.startswith("left_") for n in names[0:6])
    assert all(n.startswith("right_") for n in names[6:12])
    assert all(n.startswith("waist_") for n in names[12:15])
    assert all(n.startswith("head_") for n in names[15:17])
    assert all(n.startswith("left_") for n in names[17:24])
    assert all(n.startswith("right_") for n in names[24:31])


def test_every_sdk_joint_covered_by_exactly_one_actuator_group():
    for name in X2_PROFILE.joint_sdk_names:
        matches = [
            group_name
            for group_name, group in X2_PROFILE.actuator_groups.items()
            if any(re.fullmatch(expr, name) for expr in group.joint_names_expr)
        ]
        assert len(matches) == 1, f"{name}: matched by {matches or 'no group'}"


def test_asset_paths_exist():
    """Every declared vendor asset resolves once they have been fetched.

    Vendor models are not stored in this repository (licenses differ per
    vendor), so this skips where they are absent -- CI runs without them.
    ``asset_path`` itself raises when a file is missing, so the guard has to
    wrap the call, not just the ``is_file`` check afterwards.
    """
    try:
        paths = {f: X2_PROFILE.asset_path(f) for f in ("urdf", "mjcf", "scene_mjcf")}
    except FileNotFoundError:
        pytest.skip("vendor assets not fetched; run: python assets/fetch.py agibot/x2")
    for field_name, path in paths.items():
        assert path.is_file(), f"{field_name} missing at {path}"
    assert X2_PROFILE.usd is None, "X2 spawns from URDF (no USD asset)"


def test_pd_mode_valid():
    assert X2_PROFILE.pd_mode == "implicit"
    assert X2_PROFILE.pd_mode in PD_MODES


def test_gains_positive():
    for name, group in X2_PROFILE.actuator_groups.items():
        assert group.effort_limit > 0, name
        assert group.velocity_limit > 0, name
        assert group.armature > 0, name
        for value in _gain_values(group.stiffness) + _gain_values(group.damping):
            assert value > 0, name


def test_gain_magnitude_sanity_vs_g1():
    """Derived X2 gains must stay within 10x of G1's on same-role joints.

    Joint names below exist verbatim on both robots, so the comparison is a
    literal same-name lookup (hip/knee/ankle/waist-yaw/shoulder/elbow/wrist).
    Head joints have no G1 counterpart and share the small-motor tier with
    the wrists, which is covered via wrist_pitch.
    """
    same_role_joints = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "left_wrist_pitch_joint",
    ]
    for joint in same_role_joints:
        for kind in ("stiffness", "damping"):
            x2 = _x2_gain_for(joint, kind)
            g1 = _g1_gain_for(joint, kind)
            ratio = x2 / g1
            assert 1.0 / MAX_GAIN_RATIO < ratio < MAX_GAIN_RATIO, (
                f"{joint} {kind}: X2={x2:.2f} vs G1={g1:.2f} (ratio {ratio:.2f})"
            )


def test_semantic_slots_nonempty():
    assert X2_PROFILE.base_link
    assert X2_PROFILE.feet_bodies
    assert X2_PROFILE.undesired_contact_bodies
    assert X2_PROFILE.leg_deviation_joints
    assert X2_PROFILE.arm_deviation_joints
    assert X2_PROFILE.waist_deviation_joints


def test_deviation_regexes_match_real_joints():
    """Every deviation regex must match at least one SDK joint name."""
    names = X2_PROFILE.joint_sdk_names
    for slot in ("leg_deviation_joints", "arm_deviation_joints", "waist_deviation_joints"):
        for expr in getattr(X2_PROFILE, slot):
            assert any(re.fullmatch(expr, n) for n in names), f"{slot}: {expr} matches nothing"


def test_default_pose_regexes_match_real_joints():
    """Every default_joint_pos regex must match at least one SDK joint name."""
    names = X2_PROFILE.joint_sdk_names
    for expr in X2_PROFILE.default_joint_pos:
        assert any(re.fullmatch(expr, n) for n in names), f"{expr} matches nothing"


def test_default_pose_is_bent_knee_and_flat_foot():
    """D5 (2026-07-30): fix #1 after training run #1 never stood up from the
    vendor straight-knee zero pose. Guards the FK-derived pose's two load-
    bearing invariants: knees actually bent (positive = flexion on X2), and
    hip + knee + ankle = 0 (all three pitch axes are (0,1,0), so this sum
    being zero is exactly the flat-sole condition verified by FK)."""
    pose = X2_PROFILE.default_joint_pos
    hip = pose[".*_hip_pitch_joint"]
    knee = pose[".*_knee_joint"]
    ankle = pose[".*_ankle_pitch_joint"]
    assert knee > 0.0, "X2 knee flexes positive; default must be bent"
    assert abs(hip + knee + ankle) < 1e-9, "soles not flat: hip+knee+ankle != 0"


def test_height_facts_consistent():
    assert 0.0 < X2_PROFILE.min_base_height_m < X2_PROFILE.target_base_height_m
    assert X2_PROFILE.target_base_height_m <= X2_PROFILE.spawn_height_m


def test_root_joint_registered():
    assert X2_PROFILE.root_joint_name == "floating_base_joint"
