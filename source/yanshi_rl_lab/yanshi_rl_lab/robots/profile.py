# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Robot profile dataclasses -- the vendor-neutral "robot facts" layer.

This module is pure Python by design: it MUST NOT import Isaac Lab (or any
simulator). Tests, asset tooling and the MuJoCo deployment side all read
profiles without a GPU stack present. The Isaac-Lab-facing consumer lives in
``robots/articulation.py``.

A :class:`RobotProfile` records *what a robot is* (model files, joint table,
actuator groups, semantic body map, default pose), never *how to drive it*.
Task recipes reference robots only through the semantic slots defined here
(``feet_bodies``, ``base_link``, ...) -- concrete body-name regexes are banned
from shared task code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Environment variable that relocates the assets root (mirrors the
# HOUSENAV_ASSETS_ROOT pattern used by alice-house consumers).
ASSETS_ROOT_ENV_VAR = "YANSHI_ASSETS_ROOT"

# profile.py lives at <repo>/source/yanshi_rl_lab/yanshi_rl_lab/robots/profile.py,
# so the repository root is four directories up.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Asset-location fields of RobotProfile that asset_path() may resolve.
ASSET_FIELDS = ("urdf", "mjcf", "scene_mjcf", "usd")

# Allowed PD actuation modes. "implicit" = gains live inside the simulator's
# implicit joint drive; "explicit" = the deploy loop computes torques from
# kp/kd itself. Getting this wrong flips a robot over instantly (alice-house
# lesson), so consumers must validate explicitly and reject anything else.
PD_MODES = ("implicit", "explicit")


def assets_root() -> Path:
    """Root directory that all relative asset paths in profiles resolve against.

    Defaults to ``<repo>/assets``; override with the ``YANSHI_ASSETS_ROOT``
    environment variable (no absolute path is ever hardcoded).
    """
    env_value = os.environ.get(ASSETS_ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return _REPO_ROOT / "assets"


@dataclass(frozen=True)
class ActuatorGroup:
    """One actuator group (typically one motor model) shared by several joints.

    ``stiffness``/``damping`` accept either a single float for the whole group
    or a ``{joint-regex: value}`` dict, mirroring Isaac Lab actuator configs.
    """

    joint_names_expr: list[str]
    effort_limit: float
    velocity_limit: float
    stiffness: dict | float
    damping: dict | float
    armature: float


@dataclass(frozen=True)
class RobotProfile:
    """Everything the framework needs to know about one robot."""

    # -- identity --------------------------------------------------------
    vendor: str
    model: str

    # -- asset locations (paths relative to ``assets_root()``) ----------
    urdf: str
    mjcf: str
    scene_mjcf: str | None
    usd: str | None  # None = spawn from the URDF

    # -- embodiment ------------------------------------------------------
    spawn_height_m: float
    default_joint_pos: dict[str, float]  # keys are joint-name regexes
    actuator_groups: dict[str, ActuatorGroup]
    joint_sdk_names: list[str] | None  # real-robot SDK joint ordering
    pd_mode: str  # one of PD_MODES; consumers must validate explicitly

    # -- semantic body map (task recipes reference ONLY these slots) -----
    base_link: str
    feet_bodies: str  # body-name regex
    undesired_contact_bodies: str  # body-name regex
    leg_deviation_joints: list[str]
    arm_deviation_joints: list[str]
    waist_deviation_joints: list[str]

    # -- task-derivable facts --------------------------------------------
    # ``None`` means "this fact does not exist for this robot", and the task
    # base then DROPS the corresponding reward/termination term instead of
    # guessing a number. Concrete case: Berkeley Humanoid Lite's CAD export
    # puts the root frame at ground level (standing root height ~= -0.04 m,
    # a fallen pose can read HIGHER than standing), so a root-height reward
    # target or termination floor is meaningless there -- and the official
    # BHL training config indeed carries neither term. The fields stay
    # required (no default): a profile author must write None consciously.
    target_base_height_m: float | None
    min_base_height_m: float | None

    # -- defaulted fields -------------------------------------------------
    self_collisions: bool = True
    action_scale: float = 0.25
    # Name of the floating-base (free) joint in the robot's MJCF. A MuJoCo
    # model fact, needed by the sim2sim runtime to place/read the base; None
    # means "not registered yet" and the deploy side will refuse to run.
    root_joint_name: str | None = None

    def __post_init__(self) -> None:
        if self.pd_mode not in PD_MODES:
            raise ValueError(
                f"RobotProfile {self.vendor}/{self.model}: pd_mode must be one of "
                f"{PD_MODES}, got {self.pd_mode!r}. Refusing to guess -- a wrong "
                "PD mode makes the robot fall over on first contact."
            )

    def asset_path(self, field_name: str) -> Path:
        """Absolute path of one asset field, with an actionable error if missing."""
        if field_name not in ASSET_FIELDS:
            raise ValueError(
                f"asset_path() accepts one of {ASSET_FIELDS}, got {field_name!r}."
            )
        rel = getattr(self, field_name)
        if rel is None:
            raise ValueError(
                f"RobotProfile {self.vendor}/{self.model} does not define {field_name!r} "
                "(field is None)."
            )
        if Path(rel).is_absolute():
            raise ValueError(
                f"RobotProfile {self.vendor}/{self.model}: {field_name!r} must be a path "
                f"relative to the assets root, got absolute path {rel!r}."
            )
        path = assets_root() / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Asset {field_name!r} of {self.vendor}/{self.model} not found at {path}. "
                f"Fetch the vendor assets first: python assets/fetch.py "
                f"{self.vendor}/{self.model} (or point {ASSETS_ROOT_ENV_VAR} at an "
                "existing assets tree)."
            )
        return path
