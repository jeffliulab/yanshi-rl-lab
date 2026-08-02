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
import re
from dataclasses import dataclass
from pathlib import Path

# Environment variable that relocates the assets root (mirrors the
# HOUSENAV_ASSETS_ROOT pattern used by alice-house consumers).
ASSETS_ROOT_ENV_VAR = "YANSHI_ASSETS_ROOT"

# Every identity segment doubles as a Python package name (robots/<vendor>/
# <model>/<variant>/) and as a task-ID segment, so it must be a lowercase
# identifier. Note the leading letter: a segment like "29dof" is NOT
# importable, which is exactly the trap the predecessor stack fell into --
# its config lived at ``unitree_rl_lab...g1.29dof.velocity_variants_env_cfg``
# and every consumer had to reach it through importlib. Hence "dof29".
_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")

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
    # Three segments: vendor / model / variant. The variant is mandatory even
    # when a model currently ships only one configuration -- "there is only
    # one" is never permanent, and back-filling a segment later means editing
    # already-published leaderboard entries and task IDs.
    #
    # Naming rule: the variant name is UPSTREAM'S OWN name for that
    # configuration; never invent one. Registered examples:
    #   unitree/g1/dof29           upstream ships g1_29dof_*.urdf / g1_29dof.xml
    #   unitree/g1/dof23           upstream ships g1_23dof_*.urdf / g1_23dof.xml
    #   agibot/x2/v1_4_0           upstream directory is X2_URDF-v1.4.0
    #   berkeley/humanoid_lite/humanoid   upstream ships "biped" and "humanoid"
    #
    # What makes something a variant, in one line: swap it and a trained
    # policy no longer transfers. Terrain, reward tables and seeds are tasks
    # and experiments, not variants.
    vendor: str
    model: str
    variant: str

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
        for field_name in ("vendor", "model", "variant"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SEGMENT_RE.match(value):
                raise ValueError(
                    f"RobotProfile {field_name} must match {_SEGMENT_RE.pattern} "
                    f"(lowercase, starts with a letter -- it is a package name and a "
                    f"task-ID segment), got {value!r}."
                )
        if self.pd_mode not in PD_MODES:
            raise ValueError(
                f"RobotProfile {self.key}: pd_mode must be one of "
                f"{PD_MODES}, got {self.pd_mode!r}. Refusing to guess -- a wrong "
                "PD mode makes the robot fall over on first contact."
            )

    @property
    def key(self) -> str:
        """Canonical robot identity, ``<vendor>/<model>/<variant>``.

        This exact string is what leaderboard entries record and what
        ``yanshi robots`` prints; the leaderboard directory name is this with
        ``/`` replaced by ``-``.
        """
        return f"{self.vendor}/{self.model}/{self.variant}"

    @property
    def asset_key(self) -> str:
        """Asset-registry key, ``<vendor>/<model>`` -- deliberately NOT the
        full identity.

        Variants of one model share a single fetched asset tree: upstream ships
        every configuration in the same repository (both G1 DoF variants come
        out of one ``unitree_ros`` checkout), and ``assets/fetch.py`` creates
        one directory per key. Keying assets by variant would clone the same
        147 MB upstream repo twice and leave two copies free to drift apart.
        """
        return f"{self.vendor}/{self.model}"

    def asset_path(self, field_name: str) -> Path:
        """Absolute path of one asset field, with an actionable error if missing."""
        if field_name not in ASSET_FIELDS:
            raise ValueError(
                f"asset_path() accepts one of {ASSET_FIELDS}, got {field_name!r}."
            )
        rel = getattr(self, field_name)
        if rel is None:
            raise ValueError(
                f"RobotProfile {self.key} does not define {field_name!r} (field is None)."
            )
        if Path(rel).is_absolute():
            raise ValueError(
                f"RobotProfile {self.key}: {field_name!r} must be a path "
                f"relative to the assets root, got absolute path {rel!r}."
            )
        path = assets_root() / rel
        if not path.exists():
            raise FileNotFoundError(
                f"Asset {field_name!r} of {self.key} not found at {path}. "
                f"Fetch the vendor assets first: python assets/fetch.py "
                f"{self.asset_key} (or point {ASSETS_ROOT_ENV_VAR} at an "
                "existing assets tree)."
            )
        return path


# ── class-to-profile registry ────────────────────────────────────────────────
# Env-cfg classes must NOT carry their RobotProfile as a class or instance
# attribute: Isaac Lab's configclass copies plain class attributes into the
# instance __dict__, class_to_dict() then serializes them, and hydra's
# from_dict() round-trip tries to write values back into the frozen profile
# (FrozenInstanceError -- M1 smoke run r4, 2026-07-30). The association
# therefore lives here, completely outside the config-object namespace,
# keyed by class and resolved along the MRO so subclasses inherit it.

_CLASS_PROFILES: dict[type, RobotProfile] = {}


def attach_profile(cls: type, profile: RobotProfile) -> None:
    """Associate ``profile`` with an env-cfg class (called by the task registry)."""
    _CLASS_PROFILES[cls] = profile


def profile_of(obj: object) -> RobotProfile | None:
    """Return the profile attached to ``obj``'s class (or the class itself),
    searching base classes in MRO order; None if no ancestor has one."""
    klass = obj if isinstance(obj, type) else type(obj)
    for base in klass.__mro__:
        if base in _CLASS_PROFILES:
            return _CLASS_PROFILES[base]
    return None
