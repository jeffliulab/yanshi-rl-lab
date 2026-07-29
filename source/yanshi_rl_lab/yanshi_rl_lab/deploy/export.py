# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Merges two predecessor exporters into ONE reflection-based dump:
# - unitree_rl_lab ``utils/export_deploy_cfg.py`` (upstream repo Apache-2.0):
#   joint_ids_map / SDK-order gain layout / command ranges / per-term action
#   and observation reflection;
# - unitree-g1-locomotion ``sim2sim/dump_isaac_trace.py`` (MIT, same author):
#   actuator groups with effort_limit + armature, default_root_pos_w,
#   obs_dim / per-term dims.
# Output is a single schema-v2 contract (JSON), the one file every consumer
# (sim2sim, real robot, alice-house) reads.

"""Training-time contract exporter: live Isaac Lab env -> v2 contract JSON.

Called by ``scripts/rsl_rl/train.py`` right after the params dump (wrapped in
try/except there: a failed export warns, never kills a training run).

The module is import-safe without Isaac Lab: everything reflects over the
live env via duck typing (attribute access only), so the pure assembly
helpers are unit-testable with fake objects. Robot facts that reflection
cannot see (pd_mode, foot bodies, root joint, base link) come from the env
cfg's RobotProfile (``env.cfg.profile``, a ClassVar installed by the task
registry).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from yanshi_rl_lab.deploy.contract import (
    ActionSpec,
    ActuatorGroupSpec,
    ContractV2,
    ObsTermSpec,
    TimingSpec,
)


# --------------------------------------------------------------- pure helpers
def _to_list(value) -> list | None:
    """Flatten a torch tensor / numpy array / list / scalar to a python list.

    For batched (2-D) per-env tensors, row 0 is taken -- every env shares the
    same nominal configuration at export time (same convention as both
    predecessor exporters). Duck-typed so tests run without torch.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    if hasattr(value, "ndim"):  # numpy array
        if value.ndim >= 2:
            value = value[0]
        return [float(x) for x in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


def sdk_index_map(joint_names, joint_sdk_names) -> list:
    """``map[i_policy] = SDK slot`` by exact name matching.

    Same result as upstream's resolve_matching_names(joint_names,
    joint_sdk_names, preserve_order=True) for exact (non-regex) SDK name
    lists, but pure Python. Raises on any mismatch: a policy joint absent
    from the SDK table means the profile and the trained robot disagree.
    """
    sdk = list(joint_sdk_names)
    if len(sdk) != len(joint_names):
        raise ValueError(
            f"{len(joint_names)} policy joints vs {len(sdk)} SDK names -- profile mismatch."
        )
    index_map = []
    for name in joint_names:
        try:
            index_map.append(sdk.index(name))
        except ValueError as exc:
            raise ValueError(f"Policy joint {name!r} not in joint_sdk_names.") from exc
    return index_map


def resolve_body_regex(pattern: str, body_names) -> list:
    """Concrete body names matching ``pattern`` (full match, Isaac Lab rule).

    Exact names pass through; zero matches raise -- a foot regex that matches
    nothing would silently disable the contact veto downstream.
    """
    names = list(body_names)
    if pattern in names:
        return [pattern]
    matched = sorted(n for n in names if re.fullmatch(pattern, n))
    if not matched:
        raise ValueError(f"Body pattern {pattern!r} matches none of {names}.")
    return matched


def build_actuator_specs(actuators: dict) -> dict:
    """ActuatorGroupSpec per group from live actuator objects (duck-typed:
    ``joint_names`` plus optional stiffness/damping/effort_limit/armature
    tensors)."""
    specs = {}
    for group_name, act in actuators.items():
        joint_names = list(act.joint_names)
        fields = {}
        for field in ("stiffness", "damping", "effort_limit", "armature"):
            fields[field] = _to_list(getattr(act, field, None))
        specs[group_name] = ActuatorGroupSpec(joint_names=joint_names, **fields)
    return specs


def build_obs_term_specs(term_names, term_dims, term_cfgs, group_history: int) -> list:
    """ObsTermSpec list in manager order.

    ``term_dims`` are the flattened per-term dims INCLUDING history (what
    ``group_obs_term_dim`` reports when the group flattens history);
    per-frame dim = dim_with_history / history. Per-term scale comes from the
    term cfg (scalar kept scalar; tensors flattened).
    """
    specs = []
    for name, dim_with_history, cfg in zip(term_names, term_dims, term_cfgs):
        history = int(getattr(cfg, "history_length", 0) or 0) or int(group_history or 1)
        scale = getattr(cfg, "scale", None)
        if scale is not None and not isinstance(scale, (int, float)):
            scale_list = _to_list(scale)
            scale = (
                float(scale_list[0])
                if len(set(scale_list)) == 1
                else scale_list
            )
        dim_total = int(dim_with_history)
        if dim_total % history != 0:
            raise ValueError(
                f"Obs term {name!r}: flattened dim {dim_total} not divisible by history {history}."
            )
        specs.append(
            ObsTermSpec(
                name=str(name),
                scale=scale,
                history_length=history,
                dim=dim_total // history,
            )
        )
    return specs


def build_command_ranges(commands_cfg) -> dict | None:
    """Command ranges from the env cfg (``limit_ranges`` preferred over
    ``ranges``, mirroring upstream export_deploy_cfg: with a command
    curriculum the limit ranges are the real envelope)."""
    base_velocity = getattr(commands_cfg, "base_velocity", None)
    if base_velocity is None:
        return None
    ranges = getattr(base_velocity, "limit_ranges", None) or getattr(base_velocity, "ranges", None)
    if ranges is None:
        return None
    out = {}
    for item in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        value = getattr(ranges, item, None)
        if value is not None:
            out[item] = [float(value[0]), float(value[1])]
    return out or None


def build_action_spec(action_manager) -> ActionSpec:
    """ActionSpec from the live action manager (single JointPositionAction --
    the only action layout this stack trains; anything else must fail
    loudly rather than export a contract the runtime would misread)."""
    terms = dict(zip(action_manager.active_terms, action_manager._terms.values()))
    if list(terms) != ["JointPositionAction"]:
        raise ValueError(
            f"Contract export supports exactly one JointPositionAction, got {list(terms)}."
        )
    term = terms["JointPositionAction"]
    cfg_scale = term.cfg.scale
    if isinstance(cfg_scale, (int, float)):
        scale = float(cfg_scale)
    else:
        scale_list = _to_list(term._scale)
        scale = float(scale_list[0]) if len(set(scale_list)) == 1 else scale_list
    clip = None
    if getattr(term.cfg, "clip", None) is not None:
        clip = _to_list(term._clip)
    return ActionSpec(scale=scale, dim=int(action_manager.total_action_dim), clip=clip)


# ------------------------------------------------------------- live exporter
def export_contract(env, task_name: str, policy_path: str | None = None) -> ContractV2:
    """Build a v2 contract from a live ManagerBasedRLEnv (reflection only).

    ``env`` is the unwrapped env; ``env.cfg.profile`` must be a RobotProfile
    (the task registry installs it). No file IO here -- the caller decides
    where the JSON goes (train.py writes ``<log_dir>/params/contract.json``).
    """
    robot = env.scene["robot"]
    profile = env.cfg.profile
    if profile is None:
        raise ValueError(
            "env.cfg.profile is None -- contract export needs the RobotProfile "
            "(register the task through yanshi_rl_lab.tasks.registry)."
        )

    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)

    ids_map = None
    sdk_names = None
    if profile.joint_sdk_names:
        sdk_names = list(profile.joint_sdk_names)
        ids_map = sdk_index_map(joint_names, sdk_names)

    om = env.observation_manager
    term_names = list(om.active_terms["policy"])
    term_dims = [int(_prod(d)) for d in om.group_obs_term_dim["policy"]]
    term_cfgs = list(om._group_obs_term_cfgs["policy"])
    group_history = int(getattr(env.cfg.observations.policy, "history_length", 1) or 1)
    obs_terms = build_obs_term_specs(term_names, term_dims, term_cfgs, group_history)

    contract = ContractV2(
        task=task_name,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        policy=policy_path,
        joint_names=joint_names,
        joint_sdk_names=sdk_names,
        joint_ids_map=ids_map,
        default_joint_pos=_to_list(robot.data.default_joint_pos),
        default_root_pos_w=_to_list(robot.data.default_root_state[0, :3]),
        actuators=build_actuator_specs(robot.actuators),
        pd_mode=profile.pd_mode,
        obs_terms=obs_terms,
        obs_dim=sum(t.dim_with_history for t in obs_terms),
        history_length=group_history,
        action=build_action_spec(env.action_manager),
        timing=TimingSpec(
            policy_dt_s=float(env.step_dt),
            physics_dt_s=float(env.cfg.sim.dt),
            decimation=int(env.cfg.decimation),
        ),
        command_ranges=build_command_ranges(env.cfg.commands),
        foot_bodies=resolve_body_regex(profile.feet_bodies, body_names),
        root_joint_name=profile.root_joint_name,
        base_link=profile.base_link,
    )
    # Fail the export (not the training -- train.py catches) on an
    # inconsistent dump; print what is honestly missing.
    for warning in contract.validate():
        print(f"[contract] note: {warning}")
    return contract


def _prod(shape) -> int:
    result = 1
    for s in shape if hasattr(shape, "__iter__") else (shape,):
        result *= int(s)
    return result
