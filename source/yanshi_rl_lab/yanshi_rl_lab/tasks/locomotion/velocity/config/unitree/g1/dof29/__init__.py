# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1 velocity-task registration.

The overlay stays empty of task code except the declared override slot below
(project acceptance rule: a robot with a complete profile trains with an
empty overlay; overrides exist only when measurement proves them necessary,
each with a documented reason).

M1 parity-gate override (decision D4, 2026-07-29): the parity run must
replicate the predecessor stack's battle-proven ``RobotTurnAlignedEnvCfg``
(S2-Exp5 aligned configuration, four gates passed 4/4) **verbatim**. The new
base already carries its reward side (the upstream-aligned package is the
default); what the base keeps at upstream-official values is the command
profile -- so exactly that is overridden here.
"""

from yanshi_rl_lab.mdp.commands import apply_heading_pinning, apply_lin_vel_deadband
from yanshi_rl_lab.mdp import rewards as yanshi_mdp_rewards
from yanshi_rl_lab.robots.unitree.g1.dof29.profile import G1_PROFILE
from yanshi_rl_lab.tasks.registry import register_velocity
from yanshi_rl_lab.terrains import presets as terrain_presets

import math

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm

# -- S2-Exp5 aligned command profile (source: old stack
# unitree_rl_lab .../g1/29dof/velocity_variants_env_cfg.py) -------------------
# ``_fix_command`` (L80-85) sets BOTH ``ranges`` and ``limit_ranges`` to the
# same dead values, bypassing the slow command curriculum;
# ``RobotTurnEnvCfg`` (L307) calls it with vx_hi=1.0, wz=0.8.
PARITY_LIN_VEL_X = (-0.5, 1.0)  # _fix_command default backward limit + Turn's forward cap
PARITY_LIN_VEL_Y = (-0.3, 0.3)  # _fix_command default, verbatim upstream limit_ranges value
# Turn-in-place needs a +-0.8 rad/s *training* range: at the official +-0.2 the
# tracking-reward gap between turning and not turning is ~0.15 and any energy
# penalty wins (old-stack experiment 14, RobotTurnEnvCfg docstring); the
# parity gate's turn-in-place exam point commands wz=0.6, unreachable at +-0.2.
PARITY_ANG_VEL_Z = (-0.8, 0.8)


def _parity_command_profile(cfg) -> None:
    """Verbatim replica of ``RobotTurnAlignedEnvCfg``'s command side.

    Old-stack lineage (velocity_variants_env_cfg.py): TurnAligned (L329-334)
    -> TurnDeadband (L311-321, adds the linear-velocity deadband snap)
    -> Turn (L294-307, wz +-0.8) -> ``_fix_command`` (L80-85, both range sets
    dead). Reward side (upstream-aligned package W/C/H/G) is already the new
    base's default, so commands are the only thing to replicate.
    """
    for r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        r.lin_vel_x = PARITY_LIN_VEL_X
        r.lin_vel_y = PARITY_LIN_VEL_Y
        r.ang_vel_z = PARITY_ANG_VEL_Z
    # Deadband snap (old stack L145-157 ``_apply_lin_vel_deadband``, threshold
    # LIN_VEL_DEADBAND = 0.2 at L109): small linear-velocity samples become
    # exactly zero so "stand and turn" is actually trained. Same helper,
    # ported to mdp.commands, same default threshold.
    apply_lin_vel_deadband(cfg)


# terrains is append-only (T0 rule: adding a world must never drop an
# existing one): "flat" stays exactly as registered for M1; "rough_nostairs"
# (S3-实验2 §2.4) is the training carrier for blind rough walking -- the
# stair-free official rough mix, see terrains/presets.py. Non-flat presets
# keep the terrain-level curriculum (registry.py only disables it for
# "flat"), so the difficulty course survives unchanged.

# -- S3-实验2 发1 rough recipe (experiment card §2.5, frozen 2026-07-31) ------
# Base = E1 run-2 recipe verbatim (old stack RobotRoughAlignedV2EnvCfg):
# official rough profile (heading P-control mix, rel_heading_envs=0.5) +
# official rough reward deltas. THE single variable on top =
# track_heading_exp (direct heading signal the mainstream stack lacks).
# Provenance for every number: old stack
# velocity_variants_env_cfg.py S3E1 block + Isaac Lab official G1 rough.
ROUGH_HEADING_STIFFNESS = 0.5  # official G1 rough heading P-control stiffness
ROUGH_REL_HEADING_ENVS = 0.5   # E1 run-2 mix ratio (half heading, half direct-wz)
TRACK_HEADING_WEIGHT = 0.25    # S3E2-EXP-2 冻结值（EXP-1 = 0.5；直/脆权衡判别见卡 §七③）
TRACK_HEADING_STD = 0.5        # same magnitude as official track terms (sqrt(0.25))

_FLAT_SUB_TERRAIN_KEYS = set(terrain_presets.get("flat").sub_terrains)


def _is_rough_terrain(cfg) -> bool:
    """Terrain discriminator for overrides: the flat preset is a single
    mesh-plane sub-terrain; anything else (rough_nostairs today) is rough."""
    gen = cfg.scene.terrain.terrain_generator
    return gen is not None and set(gen.sub_terrains) != _FLAT_SUB_TERRAIN_KEYS


def _s3e2_rough_recipe(cfg) -> None:
    """E1 run-2 rough recipe + track_heading_exp (the single variable)."""
    cmd = cfg.commands.base_velocity
    # -- commands: official G1 rough heading-mode profile (E1 run-2 mix) ----
    cmd.heading_command = True
    cmd.heading_control_stiffness = ROUGH_HEADING_STIFFNESS
    cmd.rel_heading_envs = ROUGH_REL_HEADING_ENVS
    for r in (cmd.ranges, cmd.limit_ranges):
        r.lin_vel_x = (0.0, 1.0)   # official rough: forward only
        r.lin_vel_y = (0.0, 0.0)   # official rough: no lateral
        r.ang_vel_z = (-1.0, 1.0)  # clip of the heading P-control output
        r.heading = (-math.pi, math.pi)
    # heading_target for the direct-wz half pinned to yaw at resample, so
    # track_heading_exp has a target everywhere (see mdp.commands)
    apply_heading_pinning(cfg)
    # -- rewards: official rough deltas (E1 recipe, per-line provenance) ----
    cfg.rewards.termination_penalty = RewTerm(func=isaac_mdp.is_terminated, weight=-200.0)
    cfg.rewards.base_linear_velocity.weight = 0.0   # lin_vel_z: vertical speed is normal on rough
    cfg.rewards.flat_orientation_l2.weight = -1.0   # official rough value (flat base: -5.0)
    cfg.rewards.base_height.weight = 0.0            # world-frame height penalty = poison off flat
    # -- THE single variable: direct heading-tracking reward ----------------
    cfg.rewards.track_heading_exp = RewTerm(
        func=yanshi_mdp_rewards.track_heading_exp,
        weight=TRACK_HEADING_WEIGHT,
        params={"command_name": "base_velocity", "std": TRACK_HEADING_STD},
    )
    # -- terminations: drop world-frame absolute-height done ----------------
    cfg.terminations.base_height = None


def _g1_task_overrides(cfg) -> None:
    """Per-terrain dispatch: flat keeps the M1 parity profile verbatim
    (T0: flat behavior must not move); rough gets the S3E2 recipe."""
    if _is_rough_terrain(cfg):
        _s3e2_rough_recipe(cfg)
    else:
        _parity_command_profile(cfg)


def _g1_play_overrides(cfg) -> None:
    """Rough play only: heading P-control recomputes wz every step and would
    fight playify's locked commands -- revert to direct-wz (old-stack play
    precedent). Flat play is untouched."""
    if _is_rough_terrain(cfg):
        cfg.commands.base_velocity.heading_command = False
        cfg.commands.base_velocity.ranges.heading = None
        cfg.commands.base_velocity.limit_ranges.heading = None
        # 与统一尺崎岖门同工况：锁 vx=0.5（playify 默认锁 0.8，那是平地门的速度）
        for _r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
            _r.lin_vel_x = (0.5, 0.5)


register_velocity(
    G1_PROFILE,
    __name__,
    terrains=("flat", "rough_nostairs"),
    overrides=_g1_task_overrides,
    play_overrides=_g1_play_overrides,
    runner_entry_points={
        # S3E2 rough recipe carries the official rough entropy delta (0.008)
        "rough_nostairs": "yanshi_rl_lab.tasks.locomotion.agents:YanshiRoughPPORunnerCfg"
    },
)
