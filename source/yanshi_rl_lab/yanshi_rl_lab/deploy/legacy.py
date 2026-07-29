# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Converters lifting the three legacy contract formats into schema v2.

The predecessor stack left three mutually incompatible formats (see
:mod:`yanshi_rl_lab.deploy.contract` for the history):

1. ``deploy.yaml`` -- auto-dumped by unitree_rl_lab's train.py for real-robot
   deployment (upstream ``utils/export_deploy_cfg.py``, Apache-2.0 repo).
2. G1 ``contract.json`` -- dumped by unitree-g1-locomotion's
   ``sim2sim/dump_isaac_trace.py`` (MIT) for the MuJoCo sim2sim runtime.
3. Go2 ``contract.json`` -- hand-assembled earlier for alice-house (MIT).

Ground rules for every converter here:

- A field the old format never recorded stays ``None`` -- we do NOT invent
  numbers. ``ContractV2.validate()`` then lists exactly what is missing.
- Robot facts that no legacy format carries (``pd_mode``, ``foot_bodies``,
  ``root_joint_name``, ``base_link``) must be passed explicitly by the
  caller, normally from the robot's ``RobotProfile`` (e.g. ``G1_PROFILE``).
  There is no default: a wrong pd_mode flips the robot on first contact.
"""

from __future__ import annotations

import json
from pathlib import Path

from yanshi_rl_lab.deploy.contract import (
    ActionSpec,
    ActuatorGroupSpec,
    ContractV2,
    LayoutCarryover,
    ObsTermSpec,
    TimingSpec,
)


def _load(source, loader) -> dict:
    """Accept a dict, or a path to load with ``loader``."""
    if isinstance(source, dict):
        return source
    return loader(Path(source).read_text(encoding="utf-8"))


def _profile_runtime_fields(pd_mode, foot_bodies, root_joint_name, base_link) -> dict:
    return {
        "pd_mode": pd_mode,
        "foot_bodies": list(foot_bodies) if foot_bodies is not None else None,
        "root_joint_name": root_joint_name,
        "base_link": base_link,
    }


def from_deploy_yaml(
    source,
    *,
    joint_sdk_names,
    pd_mode=None,
    foot_bodies=None,
    root_joint_name=None,
    base_link=None,
    task=None,
) -> ContractV2:
    """Convert a unitree_rl_lab ``deploy.yaml`` into a v2 contract.

    What deploy.yaml HAS: joint_ids_map (policy index -> SDK slot), step_dt,
    per-joint stiffness/damping **in SDK order**, default_joint_pos and action
    offset/scale **in policy order**, command ranges, per-term observation
    scale/history (dict order = concatenation order; the upstream dumper
    writes with ``sort_keys=False`` so YAML order is authoritative).

    What deploy.yaml LACKS (left None here, flagged by ``validate()``):

    - the joint names themselves (only the index map!) -- reconstructed from
      the caller-provided ``joint_sdk_names`` via
      ``joint_names[i] = joint_sdk_names[joint_ids_map[i]]``;
    - ``effort_limit`` and ``armature`` (the real robot enforces its own
      limits, so the dumper never exported them);
    - ``physics_dt_s`` / ``decimation`` (only the combined step_dt);
    - ``default_root_pos_w`` (spawn height);
    - per-term observation dims (scale-list lengths are used where available);
    - pd_mode / foot_bodies / root_joint_name / base_link (never a training
      export concern) -- pass them from the RobotProfile.

    Args:
        source: dict or path to the deploy.yaml file.
        joint_sdk_names: real-robot SDK joint ordering (e.g. from
            ``G1_PROFILE.joint_sdk_names``); required because the file stores
            indices only.
    """
    import yaml  # local import: keep module import pure-stdlib

    d = _load(source, yaml.safe_load)

    ids_map = [int(i) for i in d["joint_ids_map"]]
    sdk = list(joint_sdk_names)
    if len(sdk) != len(ids_map):
        raise ValueError(
            f"joint_sdk_names has {len(sdk)} entries but deploy.yaml joint_ids_map has "
            f"{len(ids_map)} -- wrong robot?"
        )
    # Reconstruct policy-order joint names from the SDK map.
    joint_names = [sdk[ids_map[i]] for i in range(len(ids_map))]

    # stiffness/damping arrays are stored in SDK order; bring them back to
    # policy order (kp[i_policy] = file_kp[ids_map[i_policy]]).
    kp_sdk = [float(x) for x in d["stiffness"]]
    kd_sdk = [float(x) for x in d["damping"]]
    kp = [kp_sdk[ids_map[i]] for i in range(len(ids_map))]
    kd = [kd_sdk[ids_map[i]] for i in range(len(ids_map))]
    actuators = {
        # deploy.yaml flattens the actuator grouping; one umbrella group keeps
        # per-joint arrays without pretending we know the motor models.
        "deploy_yaml_all": ActuatorGroupSpec(
            joint_names=joint_names,
            stiffness=kp,
            damping=kd,
            effort_limit=None,  # not recorded by deploy.yaml
            armature=None,  # not recorded by deploy.yaml
        )
    }

    # Actions: exactly one JointPositionAction is what this stack trains.
    actions = d.get("actions") or {}
    if "JointPositionAction" not in actions:
        raise ValueError(
            f"deploy.yaml actions {sorted(actions)} do not include JointPositionAction; "
            "only joint-position policies are supported."
        )
    act = actions["JointPositionAction"]
    scale = act.get("scale")
    if isinstance(scale, list):
        # Collapse a constant per-joint list back to a scalar (round-trip of
        # the upstream dumper's broadcast); keep the list otherwise.
        scale = float(scale[0]) if len(set(scale)) == 1 else [float(s) for s in scale]
    action = ActionSpec(scale=scale, dim=len(joint_names), clip=act.get("clip"))

    # Observations: YAML mapping order is the concatenation order.
    obs_terms = []
    for name, term in (d.get("observations") or {}).items():
        t_scale = term.get("scale")
        dim = len(t_scale) if isinstance(t_scale, list) else None
        if isinstance(t_scale, list) and len(set(t_scale)) == 1:
            t_scale = float(t_scale[0])
        obs_terms.append(
            ObsTermSpec(
                name=name,
                scale=t_scale,
                history_length=int(term.get("history_length") or 1),
                dim=dim,
            )
        )
    histories = {t.history_length for t in obs_terms}
    history_length = histories.pop() if len(histories) == 1 else 1

    ranges = ((d.get("commands") or {}).get("base_velocity") or {}).get("ranges")
    command_ranges = None
    if ranges:
        command_ranges = {
            k: [float(v[0]), float(v[1])] for k, v in ranges.items() if v is not None
        }

    contract = ContractV2(
        task=task,
        generated_utc=None,  # deploy.yaml carries no timestamp
        joint_names=joint_names,
        joint_sdk_names=sdk,
        joint_ids_map=ids_map,
        default_joint_pos=[float(x) for x in d["default_joint_pos"]],
        default_root_pos_w=None,  # not recorded by deploy.yaml
        actuators=actuators,
        obs_terms=obs_terms,
        obs_dim=None,  # not recorded; per-term dims may be partial
        history_length=history_length,
        action=action,
        timing=TimingSpec(policy_dt_s=float(d["step_dt"]), physics_dt_s=None, decimation=None),
        command_ranges=command_ranges,
        **_profile_runtime_fields(pd_mode, foot_bodies, root_joint_name, base_link),
    )
    return contract


def from_contract_v1_g1(
    source,
    *,
    pd_mode=None,
    foot_bodies=None,
    root_joint_name=None,
    base_link=None,
    joint_sdk_names=None,
) -> ContractV2:
    """Convert a G1-style v1 ``contract.json`` (dump_isaac_trace format).

    What v1-G1 HAS: joint_names (policy order), default_joint_pos,
    default_root_pos_w, obs_terms with dim_with_history, obs_dim, group
    history_length, term_scales, action scale/dim, full timing, actuator
    groups with stiffness/damping/effort_limit/armature, and (in later dumps)
    a ``layout_carryover`` provenance block, which is preserved.

    What v1-G1 LACKS (left None, flagged by ``validate()``):

    - command ranges (the dumper recorded only the single vx used for the
      trace, which is a probe input, not a training range -- it is kept in
      ``extra`` but NOT promoted to command_ranges);
    - joint_ids_map / SDK names (sim2sim never needed them) -- derived here
      iff the caller passes ``joint_sdk_names``;
    - action clip (the training action term had none);
    - pd_mode / foot_bodies / root_joint_name / base_link -- pass them from
      the RobotProfile.
    """
    d = _load(source, json.loads)

    joint_names = list(d["joint_names"])
    history = int(d.get("history_length") or 1)
    scales = d.get("term_scales") or {}
    obs_terms = []
    for entry in d.get("obs_terms") or []:
        name = entry["term"]
        dwh = entry.get("dim_with_history")
        dim = int(dwh) // history if dwh is not None else None
        obs_terms.append(
            ObsTermSpec(name=name, scale=scales.get(name), history_length=history, dim=dim)
        )

    actuators = {
        name: ActuatorGroupSpec(
            joint_names=list(g["joint_names"]),
            stiffness=g.get("stiffness"),
            damping=g.get("damping"),
            effort_limit=g.get("effort_limit"),
            armature=g.get("armature"),
        )
        for name, g in (d.get("actuators") or {}).items()
    }

    ids_map = None
    sdk = None
    if joint_sdk_names is not None:
        sdk = list(joint_sdk_names)
        try:
            ids_map = [sdk.index(n) for n in joint_names]
        except ValueError as exc:
            raise ValueError(f"joint_sdk_names does not cover contract joints: {exc}") from exc

    carry = d.get("layout_carryover")
    if carry is not None:
        carry = LayoutCarryover(
            dumped_from_task=carry["dumped_from_task"],
            dumped_utc=carry["dumped_utc"],
            why_valid=carry["why_valid"],
            evidence=list(carry.get("evidence") or []),
            note=carry.get("note"),
        )

    timing = d["timing"]
    contract = ContractV2(
        task=d.get("task"),
        generated_utc=d.get("generated_utc"),
        policy=d.get("policy"),
        joint_names=joint_names,
        joint_sdk_names=sdk,
        joint_ids_map=ids_map,
        default_joint_pos=[float(x) for x in d["default_joint_pos"]],
        default_root_pos_w=(
            [float(x) for x in d["default_root_pos_w"]] if d.get("default_root_pos_w") else None
        ),
        actuators=actuators,
        obs_terms=obs_terms,
        obs_dim=(int(d["obs_dim"]) if d.get("obs_dim") is not None else None),
        history_length=history,
        action=ActionSpec(
            scale=d["action"]["scale"], dim=int(d["action"]["dim"]), clip=d["action"].get("clip")
        ),
        timing=TimingSpec(
            policy_dt_s=float(timing["policy_dt_s"]),
            physics_dt_s=float(timing["physics_dt_s"]),
            decimation=int(timing["decimation"]),
        ),
        command_ranges=None,  # v1-G1 never recorded training command ranges
        layout_carryover=carry,
        **_profile_runtime_fields(pd_mode, foot_bodies, root_joint_name, base_link),
    )
    # Keep the trace probe command in extra for the record (it is NOT a range).
    if "vx" in d:
        contract.extra["v1_trace_vx"] = d["vx"]
    return contract


def from_contract_v1_go2(
    source,
    *,
    pd_mode=None,
    foot_bodies=None,
    root_joint_name=None,
    base_link=None,
    joint_sdk_names=None,
) -> ContractV2:
    """Convert a Go2-style v1 ``contract.json`` (alice-house hand-made format).

    What v1-Go2 HAS: joint_names, default_joint_pos, per-joint ``gains``
    (kp/kd/effort_limit), ``term_order`` + ``term_scales``, history_length,
    action scale/dim/use_default_offset, timing, obs_dim.

    What v1-Go2 LACKS (left None, flagged by ``validate()``):

    - ``armature`` and ``default_root_pos_w`` (never recorded by hand);
    - per-term observation dims (term_order is names only; runtime frame
      builders know their own dims);
    - command ranges, generation timestamp, SDK map (derived iff the caller
      passes ``joint_sdk_names``);
    - pd_mode / foot_bodies / root_joint_name / base_link -- pass them from
      the RobotProfile (for Go2, alice-house's manifest says explicit PD).
    """
    d = _load(source, json.loads)

    joint_names = list(d["joint_names"])
    gains = d.get("gains") or {}
    missing = [n for n in joint_names if n not in gains]
    if missing:
        raise ValueError(f"Go2 v1 contract lacks gains for joints: {missing}")
    actuators = {
        "go2_v1_all": ActuatorGroupSpec(
            joint_names=joint_names,
            stiffness=[float(gains[n]["kp"]) for n in joint_names],
            damping=[float(gains[n]["kd"]) for n in joint_names],
            effort_limit=[float(gains[n]["effort_limit"]) for n in joint_names],
            armature=None,  # not recorded by the hand-made format
        )
    }

    history = int(d.get("history_length") or 1)
    scales = d.get("term_scales") or {}
    obs_terms = [
        ObsTermSpec(name=name, scale=scales.get(name), history_length=history, dim=None)
        for name in (d.get("term_order") or [])
    ]

    ids_map = None
    sdk = None
    if joint_sdk_names is not None:
        sdk = list(joint_sdk_names)
        ids_map = [sdk.index(n) for n in joint_names]

    if not d["action"].get("use_default_offset", True):
        # Every runtime here adds actions around the default pose; a contract
        # that says otherwise must not be silently reinterpreted.
        raise ValueError("Go2 v1 contract has use_default_offset=false; unsupported layout.")

    timing = d["timing"]
    return ContractV2(
        task=d.get("task"),
        generated_utc=None,  # hand-made format carries no timestamp
        joint_names=joint_names,
        joint_sdk_names=sdk,
        joint_ids_map=ids_map,
        default_joint_pos=[float(x) for x in d["default_joint_pos"]],
        default_root_pos_w=None,  # not recorded
        actuators=actuators,
        obs_terms=obs_terms,
        obs_dim=(int(d["obs_dim"]) if d.get("obs_dim") is not None else None),
        history_length=history,
        action=ActionSpec(
            scale=d["action"]["scale"], dim=int(d["action"]["dim"]), clip=d["action"].get("clip")
        ),
        timing=TimingSpec(
            policy_dt_s=float(timing["policy_dt_s"]),
            physics_dt_s=float(timing["physics_dt_s"]),
            decimation=int(timing["decimation"]),
        ),
        command_ranges=None,
        **_profile_runtime_fields(pd_mode, foot_bodies, root_joint_name, base_link),
    )
