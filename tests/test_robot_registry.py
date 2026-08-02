# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""The robot identity scheme: ``<vendor>/<model>/<variant>``, enforced.

Why this module exists: before the scheme, ``unitree/g1`` silently MEANT the
29-DoF configuration and nothing in the repository said so. The published
leaderboard row, the task ID and the quickstart all inherited that unstated
assumption -- the same class of failure as the scene file whose name said
"23dof" while its contents said otherwise. These tests make the assumption
impossible to leave unstated.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"

# Stub-package trick shared by all profile tests: the real package __init__
# imports Isaac Lab, so only the pure-Python modules are imported underneath a
# namespace stub.
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.robots import registry  # noqa: E402
from yanshi_rl_lab.robots.profile import RobotProfile  # noqa: E402

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*$")


def test_registry_discovers_the_launch_robots():
    """Discovery is by directory; if a robot silently stops being found, the
    leaderboard and the gate runner lose the ability to resolve its assets."""
    assert set(registry.keys()) >= {
        "unitree/g1/dof29",
        "agibot/x2/v1_4_0",
        "berkeley/humanoid_lite/humanoid",
    }


@pytest.mark.parametrize("key", registry.keys())
def test_every_key_has_three_segments(key):
    assert _KEY_RE.match(key), (
        f"{key!r} is not <vendor>/<model>/<variant>. The variant segment is "
        "mandatory even for a model that currently ships one configuration."
    )


@pytest.mark.parametrize("key", registry.keys())
def test_key_matches_the_directory_it_lives_in(key):
    """A profile that disagrees with its own path would make the task ID and
    the leaderboard directory disagree with each other."""
    profile = registry.get(key)
    assert profile.key == key
    vendor, model, variant = key.split("/")
    assert (_PKG_DIR / "robots" / vendor / model / variant / "profile.py").is_file()


@pytest.mark.parametrize("key", registry.keys())
def test_variant_does_not_start_with_a_digit(key):
    """Every segment is also a Python package name.

    The predecessor stack named its package "29dof" and consequently could not
    ``import`` it at all -- every consumer reached it through importlib with a
    string. "dof29" costs nothing and avoids that entirely.
    """
    variant = key.split("/")[2]
    assert not variant[0].isdigit(), f"variant {variant!r} must start with a letter"


@pytest.mark.parametrize("key", registry.keys())
def test_asset_key_is_model_level_not_variant_level(key):
    """Variants of one model share one fetched asset tree.

    Keying assets by the full identity would clone the same upstream repo once
    per variant (147 MB for the two G1 configurations) and leave two copies
    free to drift apart.
    """
    profile = registry.get(key)
    assert profile.asset_key == "/".join(key.split("/")[:2])


def test_variants_of_one_model_share_an_asset_tree():
    by_model: dict[str, list[str]] = {}
    for key, profile in registry.all_profiles().items():
        by_model.setdefault(profile.asset_key, []).append(key)
    for asset_key, keys in by_model.items():
        if len(keys) < 2:
            continue
        roots = {registry.get(k).urdf.split("/")[:2][-1] for k in keys}
        assert len(roots) == 1, (
            f"{sorted(keys)} share asset key {asset_key!r} but their URDF paths "
            "point into different trees; variants must select files inside ONE tree"
        )


def test_unknown_key_names_the_registered_ones():
    with pytest.raises(KeyError) as excinfo:
        registry.get("unitree/g1")  # the old two-segment key
    message = str(excinfo.value)
    assert "unitree/g1/dof29" in message
    assert "variant segment is mandatory" in message


def test_profile_rejects_a_segment_that_is_not_a_package_name():
    """Constructing a profile with '29dof' must fail at definition time."""
    with pytest.raises(ValueError, match="variant"):
        RobotProfile(
            vendor="acme",
            model="biped",
            variant="29dof",
            urdf="acme/biped/robot.urdf",
            mjcf="acme/biped/robot.xml",
            scene_mjcf=None,
            usd=None,
            spawn_height_m=0.5,
            default_joint_pos={},
            actuator_groups={},
            joint_sdk_names=None,
            pd_mode="implicit",
            base_link="torso",
            feet_bodies=".*foot.*",
            undesired_contact_bodies=".*",
            leg_deviation_joints=[],
            arm_deviation_joints=[],
            waist_deviation_joints=[],
            target_base_height_m=None,
            min_base_height_m=None,
        )
