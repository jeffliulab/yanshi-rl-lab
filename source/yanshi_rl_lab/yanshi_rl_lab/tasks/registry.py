# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Dynamic per-robot task registration.

One call per robot registers every requested terrain variant of the velocity
task, generating the env-config subclasses on the fly -- so a per-robot config
package contains nothing but the registration call (the "empty overlay"
acceptance rule) and adding a robot never touches any central file.

Naming rule (documented, fixed): a robot's identity is ``vendor/model/variant``,
all lowercase with underscores ("unitree", "g1", "dof29"; "berkeley",
"humanoid_lite", "humanoid"). ``vendor`` and ``model`` split on underscores and
capitalize each token ("humanoid_lite" -> ["Humanoid", "Lite"]); the ``variant``
stays one verbatim token (see ``_variant_segment``). Generated class names
concatenate the tokens (``UnitreeG1Dof29VelocityFlatEnvCfg``); task-ID segments
join them with hyphens::

    Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0
    Yanshi-Velocity-Flat-Agibot-X2-V1_4_0-v0
    Yanshi-Velocity-Rough-Nostairs-Berkeley-Humanoid-Lite-Humanoid-v0

The variant segment is what makes configurations of one model switchable: to
train the 23-DoF G1 instead of the 29-DoF one, change nothing but ``--task``.
"""

from __future__ import annotations

import sys

import gymnasium as gym

from isaaclab.utils import configclass

from yanshi_rl_lab.robots.profile import RobotProfile, attach_profile
from yanshi_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import VelocityEnvCfg, playify
from yanshi_rl_lab.terrains import presets as terrain_presets

# All velocity tasks share one PPO runner config (per-robot agent overrides are
# not needed until measurement proves otherwise).
RSL_RL_ENTRY_POINT = "yanshi_rl_lab.tasks.locomotion.agents:YanshiPPORunnerCfg"


def _camel(name: str) -> str:
    """"humanoid_lite" -> "HumanoidLite"; "g1" -> "G1" (class-name segment)."""
    return "".join(part.capitalize() for part in name.split("_"))


def _task_segment(name: str) -> str:
    """"humanoid_lite" -> "Humanoid-Lite"; "g1" -> "G1" (task-ID segment)."""
    return "-".join(part.capitalize() for part in name.split("_"))


def _variant_segment(name: str) -> str:
    """"dof29" -> "Dof29"; "v1_4_0" -> "V1_4_0" (variant segment, verbatim).

    Unlike vendor/model, a variant is kept as ONE segment with its underscores
    intact. Splitting "v1_4_0" on underscores the way _task_segment does would
    emit "V1-4-0", whose dashes are indistinguishable from the task-ID's own
    separators -- a reader (and any parser) could no longer tell where the
    variant begins and ends. Kept this way, the segment is exactly the variant
    key, so a task ID is reversible to a robot key by inspection.
    """
    return name[:1].upper() + name[1:]


def _apply_overrides(cfg, overrides) -> None:
    """Apply per-robot overrides to a config instance.

    ``overrides`` may be (a) a ``{"dotted.attr.path": value}`` dict, (b) a
    callable ``fn(cfg)`` for overrides a dotted path cannot express (e.g.
    swapping the command term class), or (c) a sequence mixing both, applied
    in order. Unknown dotted paths raise AttributeError immediately -- a typo
    must never pass silently.
    """
    if overrides is None:
        return
    entries = (overrides,) if (callable(overrides) or isinstance(overrides, dict)) else tuple(overrides)
    for entry in entries:
        if callable(entry):
            entry(cfg)
            continue
        if not isinstance(entry, dict):
            raise TypeError(f"Override entry must be a dict or callable, got {type(entry).__name__}")
        for path, value in entry.items():
            obj = cfg
            *parents, leaf = path.split(".")
            for name in parents:
                obj = getattr(obj, name)
            if not hasattr(obj, leaf):
                raise AttributeError(f"Override path {path!r} does not exist on {type(cfg).__name__}")
            setattr(obj, leaf, value)


def register_velocity(
    profile: RobotProfile,
    module_name: str,
    terrains: tuple[str, ...] = ("flat",),
    overrides=None,
    play_overrides=None,
    runner_entry_points: dict[str, str] | None = None,
) -> list[str]:
    """Register velocity tasks for one robot; returns the created task IDs.

    Args:
        profile: The robot's profile (single source of robot facts).
        module_name: ``__name__`` of the calling per-robot config package. The
            generated config classes are attached to that module so the gym
            entry-point strings resolve.
        terrains: Terrain preset names (see ``yanshi_rl_lab.terrains.presets``).
            "flat" disables the terrain-difficulty curriculum; any other
            preset keeps it (upstream base coupling logic).
        overrides: Optional overrides applied to the training cfg after
            profile resolution: a ``{"dotted.path": value}`` dict, a callable
            ``fn(cfg)``, or a sequence mixing both (see ``_apply_overrides``).
            This is the declared, documented tuning slot -- each entry must
            carry a reason at the call site.
        play_overrides: Same, applied to the play cfg after ``playify``.
        runner_entry_points: Optional per-terrain PPO runner entry points
            (``{"rough_nostairs": "...:SomeRunnerCfg"}``); terrains not listed
            keep the shared ``RSL_RL_ENTRY_POINT`` default. (S3-实验2: the
            rough recipe carries the official rough entropy delta.)
    """
    module = sys.modules[module_name]
    task_ids: list[str] = []

    for terrain in terrains:
        # fail fast on unknown preset names (before building any class)
        terrain_presets.get(terrain)

        base_name = (
            f"{_camel(profile.vendor)}{_camel(profile.model)}{_variant_segment(profile.variant)}"
            f"Velocity{_camel(terrain)}"
        )
        env_cls_name = f"{base_name}EnvCfg"
        play_cls_name = f"{base_name}PlayEnvCfg"

        def _make_env_post_init(terrain: str = terrain, overrides=overrides):
            def __post_init__(self):
                # terrain must be installed before the base __post_init__ runs
                # (it couples generator.curriculum to the terrain_levels term)
                self.scene.terrain.terrain_generator = terrain_presets.get(terrain)
                if terrain == "flat":
                    # no terrain-difficulty curriculum on flat ground (the
                    # upstream flat base kept the term as a no-op; dropping it
                    # is the explicit form of the same behavior)
                    self.curriculum.terrain_levels = None
                VelocityEnvCfg.__post_init__(self)
                _apply_overrides(self, overrides)

            return __post_init__

        env_cls = configclass(type(env_cls_name, (VelocityEnvCfg,), {"__post_init__": _make_env_post_init()}))
        env_cls.__module__ = module_name

        def _make_play_post_init(env_cls=env_cls, terrain: str = terrain, play_overrides=play_overrides):
            def __post_init__(self):
                env_cls.__post_init__(self)
                playify(self, hard_terrain=(terrain != "flat"))
                _apply_overrides(self, play_overrides)

            return __post_init__

        play_cls = configclass(type(play_cls_name, (env_cls,), {"__post_init__": _make_play_post_init()}))
        play_cls.__module__ = module_name

        # The profile is deliberately NOT stored on the class: configclass
        # copies plain class attributes into instances, class_to_dict()
        # serializes them, and hydra's from_dict() round-trip then writes into
        # the frozen RobotProfile (M1 smoke runs r3/r4, 2026-07-30). The
        # association lives in an external MRO-aware registry instead;
        # play_cls resolves through it via inheritance.
        attach_profile(env_cls, profile)

        # attach to the caller's module so entry-point strings resolve
        setattr(module, env_cls_name, env_cls)
        setattr(module, play_cls_name, play_cls)

        # The variant segment is what lets a user switch configurations of the
        # same model (G1 dof29 vs dof23) by changing only --task.
        task_id = (
            f"Yanshi-Velocity-{_task_segment(terrain)}-"
            f"{_task_segment(profile.vendor)}-{_task_segment(profile.model)}-"
            f"{_variant_segment(profile.variant)}-v0"
        )
        runner_entry_point = (runner_entry_points or {}).get(terrain, RSL_RL_ENTRY_POINT)
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{module_name}:{env_cls_name}",
                "play_env_cfg_entry_point": f"{module_name}:{play_cls_name}",
                "rsl_rl_cfg_entry_point": runner_entry_point,
            },
        )
        task_ids.append(task_id)

    return task_ids
