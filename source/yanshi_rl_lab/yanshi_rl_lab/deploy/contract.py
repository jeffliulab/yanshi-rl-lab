# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""The unified ``schema_version: 2`` policy deployment contract.

One schema for everything a trained policy needs outside Isaac Lab: joint
ordering (plus the real-robot SDK mapping), PD gains with effort limits and
armature, the PD actuation mode, default pose, the observation-term table
with scales and history, action scaling, command ranges, timing, and the
robot facts the MuJoCo runtime needs (foot bodies, root joint, base link).

Why version 2: the predecessor stack accumulated **three mutually
incompatible** contract formats (unitree_rl_lab ``deploy.yaml``, the G1
sim2sim ``contract.json``, and the hand-made Go2 ``contract.json`` used by
alice-house) because three consumers appeared at three different times and we
did not own the exporter. This schema is their field union; the converters in
:mod:`yanshi_rl_lab.deploy.legacy` lift each old format into it.

Design rules:

- Pure Python + stdlib only (no Isaac Lab, no MuJoCo, no numpy) so the
  contract can be read anywhere -- tests, CLI, real-robot tooling.
- Missing knowledge is ``None``, never an invented number. ``validate()``
  reports what is missing and what that blocks.
- Unknown JSON fields are tolerated on read (kept in ``extra``) so newer
  contracts still load, but ``validate()`` warns about them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The schema version this module implements. from_json_dict() refuses other
# major versions rather than mis-reading them.
SCHEMA_VERSION = 2

# PD actuation modes the deploy runtime knows how to reproduce. Must stay in
# sync with robots.profile.PD_MODES (duplicated on purpose: this module must
# not import the profile layer, contracts travel outside the repo).
CONTRACT_PD_MODES = ("implicit", "explicit")

# Recognized top-level JSON keys (everything else lands in ``extra``).
_KNOWN_KEYS = {
    "schema_version",
    "generated_utc",
    "task",
    "policy",
    "joint_names",
    "joint_sdk_names",
    "joint_ids_map",
    "default_joint_pos",
    "default_root_pos_w",
    "actuators",
    "pd_mode",
    "obs_terms",
    "obs_dim",
    "history_length",
    "action",
    "command_ranges",
    "timing",
    "foot_bodies",
    "root_joint_name",
    "base_link",
    "layout_carryover",
}


@dataclass
class ObsTermSpec:
    """One policy observation term, in concatenation order.

    ``scale`` is a scalar or a per-element list (matches Isaac Lab's ObsTerm
    scale semantics); ``history_length`` is the per-term history actually used
    when flattening; ``dim`` is the per-frame dimension WITHOUT history
    (None = unknown, tolerated for legacy formats that never recorded it).
    """

    name: str
    scale: float | list | None = None
    history_length: int = 1
    dim: int | None = None

    @property
    def dim_with_history(self) -> int | None:
        if self.dim is None:
            return None
        return self.dim * self.history_length


@dataclass
class ActuatorGroupSpec:
    """One actuator group: per-joint gain arrays aligned with ``joint_names``.

    All arrays are either None (unknown -- legacy formats) or lists with one
    entry per joint in this group.
    """

    joint_names: list
    stiffness: list | None = None
    damping: list | None = None
    effort_limit: list | None = None
    armature: list | None = None

    def _pick(self, arr: list | None, index: int) -> float | None:
        if arr is None or not arr:
            return None
        # A single value legitimately broadcasts over the group.
        if index >= len(arr):
            return float(arr[0]) if len(arr) == 1 else None
        return float(arr[index])

    def gains_for(self, joint_name: str) -> tuple:
        """(kp, kd, effort_limit, armature) for one joint; None where unknown."""
        i = self.joint_names.index(joint_name)
        return (
            self._pick(self.stiffness, i),
            self._pick(self.damping, i),
            self._pick(self.effort_limit, i),
            self._pick(self.armature, i),
        )


@dataclass
class ActionSpec:
    """Policy action head: joint-position offsets around the default pose."""

    scale: float | list
    dim: int
    clip: list | None = None  # [lo, hi] or per-joint [[lo, hi], ...]; None = unclipped


@dataclass
class TimingSpec:
    """Control-loop timing. ``physics_dt_s``/``decimation`` may be None for
    legacy formats that only recorded the policy step (deploy.yaml)."""

    policy_dt_s: float
    physics_dt_s: float | None = None
    decimation: int | None = None


@dataclass
class LayoutCarryover:
    """Provenance record for contracts whose layout was *carried over* from an
    earlier dump instead of re-exported (design copied from the alice-house
    g1-29dof-turn contract). Prevents a contract from silently lying about
    where its layout numbers came from."""

    dumped_from_task: str
    dumped_utc: str
    why_valid: str
    evidence: list = field(default_factory=list)
    note: str | None = None


@dataclass
class ContractV2:
    """Machine-readable deployment contract, schema version 2."""

    # -- identity / provenance -------------------------------------------
    task: str | None
    generated_utc: str | None
    joint_names: list  # policy (Isaac) order -- the canonical joint ordering
    default_joint_pos: list  # policy order, radians
    obs_terms: list  # list[ObsTermSpec], order == concatenation order
    action: ActionSpec
    timing: TimingSpec
    # -- defaulted / optional --------------------------------------------
    schema_version: int = SCHEMA_VERSION
    policy: str | None = None  # path of the policy this contract was dumped with
    joint_sdk_names: list | None = None  # real-robot SDK order
    joint_ids_map: list | None = None  # joint_ids_map[i_policy] = SDK index
    default_root_pos_w: list | None = None  # [x, y, z] spawn position
    actuators: dict = field(default_factory=dict)  # name -> ActuatorGroupSpec
    pd_mode: str | None = None  # one of CONTRACT_PD_MODES
    obs_dim: int | None = None  # flattened policy-input dimension
    history_length: int = 1  # group-level history (per-term specs may differ)
    command_ranges: dict | None = None  # e.g. {"lin_vel_x": [lo, hi], ...}
    foot_bodies: list | None = None  # MuJoCo body names or regexes, >= 1 per foot
    root_joint_name: str | None = None  # floating-base joint in the MJCF
    base_link: str | None = None
    layout_carryover: LayoutCarryover | None = None
    extra: dict = field(default_factory=dict)  # unknown fields kept on read

    # ------------------------------------------------------------------ IO
    def to_json_dict(self) -> dict:
        d = asdict(self)
        extra = d.pop("extra")
        if self.layout_carryover is None:
            d.pop("layout_carryover")
        # Unknown fields ride along so a load/save round trip loses nothing.
        for k, v in extra.items():
            d.setdefault(k, v)
        return d

    def to_json(self, path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json_dict(cls, d: dict) -> "ContractV2":
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Contract schema_version is {version!r}, this reader implements "
                f"{SCHEMA_VERSION}. Legacy formats must go through "
                "yanshi_rl_lab.deploy.legacy converters instead."
            )
        obs_terms = [
            t if isinstance(t, ObsTermSpec) else ObsTermSpec(**t) for t in d.get("obs_terms", [])
        ]
        actuators = {
            name: (g if isinstance(g, ActuatorGroupSpec) else ActuatorGroupSpec(**g))
            for name, g in (d.get("actuators") or {}).items()
        }
        action_raw = d.get("action") or {}
        action = action_raw if isinstance(action_raw, ActionSpec) else ActionSpec(**action_raw)
        timing_raw = d.get("timing") or {}
        timing = timing_raw if isinstance(timing_raw, TimingSpec) else TimingSpec(**timing_raw)
        carry = d.get("layout_carryover")
        if carry is not None and not isinstance(carry, LayoutCarryover):
            carry = LayoutCarryover(**carry)
        extra = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            schema_version=version,
            task=d.get("task"),
            generated_utc=d.get("generated_utc"),
            policy=d.get("policy"),
            joint_names=list(d.get("joint_names") or []),
            joint_sdk_names=(list(d["joint_sdk_names"]) if d.get("joint_sdk_names") else None),
            joint_ids_map=(list(d["joint_ids_map"]) if d.get("joint_ids_map") is not None else None),
            default_joint_pos=list(d.get("default_joint_pos") or []),
            default_root_pos_w=(
                list(d["default_root_pos_w"]) if d.get("default_root_pos_w") else None
            ),
            actuators=actuators,
            pd_mode=d.get("pd_mode"),
            obs_terms=obs_terms,
            obs_dim=d.get("obs_dim"),
            history_length=int(d.get("history_length") or 1),
            action=action,
            timing=timing,
            command_ranges=d.get("command_ranges"),
            foot_bodies=(list(d["foot_bodies"]) if d.get("foot_bodies") else None),
            root_joint_name=d.get("root_joint_name"),
            base_link=d.get("base_link"),
            layout_carryover=carry,
            extra=extra,
        )

    @classmethod
    def from_json(cls, path) -> "ContractV2":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------------------------------------------------------- validation
    def validate(self) -> list:
        """Hard-fail on inconsistencies, return a list of warning strings for
        missing-but-tolerated knowledge. Callers decide whether a warning is
        fatal for THEIR use (e.g. the MuJoCo runtime needs pd_mode; a pure
        obs-layout consumer does not)."""
        warnings: list = []

        if not self.joint_names:
            raise ValueError("Contract has no joint_names.")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Contract joint_names contain duplicates.")
        if len(self.default_joint_pos) != len(self.joint_names):
            raise ValueError(
                f"default_joint_pos has {len(self.default_joint_pos)} entries for "
                f"{len(self.joint_names)} joints."
            )
        if self.action.dim != len(self.joint_names):
            raise ValueError(
                f"action.dim={self.action.dim} != number of joints {len(self.joint_names)} "
                "(joint-position policies act on every contract joint)."
            )
        if not self.obs_terms:
            raise ValueError("Contract has no obs_terms (observation layout unknown).")

        if self.pd_mode is not None and self.pd_mode not in CONTRACT_PD_MODES:
            raise ValueError(
                f"pd_mode must be one of {CONTRACT_PD_MODES} or None, got {self.pd_mode!r}."
            )

        # SDK mapping consistency (when present).
        if self.joint_ids_map is not None:
            if len(self.joint_ids_map) != len(self.joint_names):
                raise ValueError("joint_ids_map length != joint_names length.")
            if sorted(self.joint_ids_map) != list(range(len(self.joint_names))):
                raise ValueError("joint_ids_map is not a permutation of joint indices.")
            if self.joint_sdk_names is not None:
                for i, name in enumerate(self.joint_names):
                    if self.joint_sdk_names[self.joint_ids_map[i]] != name:
                        raise ValueError(
                            f"joint_ids_map inconsistent: policy joint {name!r} maps to SDK slot "
                            f"{self.joint_ids_map[i]} = {self.joint_sdk_names[self.joint_ids_map[i]]!r}."
                        )

        # Actuator coverage: every joint should belong to exactly one group.
        if self.actuators:
            covered: list = []
            for g in self.actuators.values():
                covered.extend(g.joint_names)
            missing = [n for n in self.joint_names if n not in covered]
            if missing:
                raise ValueError(f"Joints missing from actuator groups: {missing}")
            dupes = {n for n in covered if covered.count(n) > 1}
            if dupes:
                raise ValueError(f"Joints listed in multiple actuator groups: {sorted(dupes)}")
        else:
            warnings.append("no actuator groups: PD gains unknown, sim2sim cannot run")

        # Obs dimension bookkeeping (only when every term knows its dim).
        dims = [t.dim_with_history for t in self.obs_terms]
        if all(d is not None for d in dims):
            total = sum(dims)
            if self.obs_dim is not None and total != self.obs_dim:
                raise ValueError(f"obs_dim={self.obs_dim} but obs_terms sum to {total}.")
        else:
            warnings.append("some obs_terms have unknown per-frame dim (legacy source)")

        # Timing consistency.
        t = self.timing
        if t.physics_dt_s is not None and t.decimation is not None:
            if abs(t.physics_dt_s * t.decimation - t.policy_dt_s) > 1e-9:
                raise ValueError(
                    f"timing inconsistent: physics_dt_s*decimation = "
                    f"{t.physics_dt_s * t.decimation} != policy_dt_s {t.policy_dt_s}."
                )
        else:
            warnings.append("timing missing physics_dt_s/decimation (deploy.yaml only has step_dt)")

        # Missing-knowledge warnings (None is honest, silence is not).
        for field_name, needed_for in (
            ("pd_mode", "sim2sim torque path"),
            ("foot_bodies", "foot-contact veto metrics"),
            ("root_joint_name", "MuJoCo base placement"),
            ("base_link", "documentation / real-robot tooling"),
            ("default_root_pos_w", "MuJoCo spawn height"),
            ("joint_sdk_names", "real-robot deployment"),
            ("joint_ids_map", "real-robot deployment"),
            ("command_ranges", "command clamping in teleop/deploy"),
            ("obs_dim", "policy input-size check"),
        ):
            if getattr(self, field_name) is None:
                warnings.append(f"{field_name} is not set (needed for: {needed_for})")

        if self.actuators:
            for name, g in self.actuators.items():
                for arr_name in ("stiffness", "damping", "effort_limit", "armature"):
                    if getattr(g, arr_name) is None:
                        warnings.append(
                            f"actuator group {name!r} has no {arr_name} (legacy source did not record it)"
                        )

        if self.extra:
            warnings.append(f"unknown fields tolerated on read: {sorted(self.extra)}")

        return warnings

    # ---------------------------------------------------------- accessors
    def gains(self) -> dict:
        """Per-joint gains in policy order:
        ``{joint_name: (kp, kd, effort_limit, armature)}`` (None where unknown).
        """
        out: dict = {}
        for g in self.actuators.values():
            for n in g.joint_names:
                out[n] = g.gains_for(n)
        missing = [n for n in self.joint_names if n not in out]
        if missing:
            raise ValueError(f"No actuator gains for joints: {missing}")
        return out

    def require_runtime_fields(self) -> None:
        """Raise with an actionable message if any field the MuJoCo sim2sim
        runtime depends on is missing. Legacy converters leave gaps as None on
        purpose; this is where those gaps become errors instead of guesses."""
        problems = []
        if self.pd_mode is None:
            problems.append("pd_mode (pass it from the RobotProfile when converting legacy formats)")
        if self.root_joint_name is None:
            problems.append("root_joint_name (from the RobotProfile / MJCF)")
        if not self.actuators:
            problems.append("actuators (PD gains)")
        t = self.timing
        if t.physics_dt_s is None or t.decimation is None:
            problems.append("timing.physics_dt_s + timing.decimation")
        if self.default_root_pos_w is None:
            problems.append("default_root_pos_w (spawn position)")
        if problems:
            raise ValueError(
                "Contract is missing fields the sim2sim runtime needs: "
                + "; ".join(problems)
                + ". Refusing to guess."
            )
