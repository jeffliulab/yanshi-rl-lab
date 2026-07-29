# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree_rl_lab (Apache-2.0) into yanshi-rl-lab; see NOTICE in file.
#
# NOTICE: The scene/command/observation/reward/termination/event/curriculum
# structure is ported from the unitree_rl_lab G1 29-DoF velocity base config
# (``tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py``; the upstream file
# carries no per-file license header; the upstream repository is licensed
# under Apache-2.0). Modifications by this project: all robot-specific body
# and joint references are re-routed through RobotProfile semantic slots, and
# four reward defaults follow the S2-Exp5 upstream-aligned package (marked
# inline). The play helper is ported from this project's own overlay
# previously hosted inside a working copy of that repository.

"""Vendor-neutral velocity-tracking base environment config.

Contract with the profile layer:

- The class variable ``profile`` must be set (by ``tasks.registry`` or a
  subclass) before instantiation; ``__post_init__`` asserts it.
- Every body/joint reference is resolved from the profile's semantic slots in
  ``__post_init__`` -- shared task code contains zero concrete robot names.
- Everything derivable from the profile IS derived (base-height reward target,
  termination height, action scale, feet/undesired-contact sensors), so a
  robot with a complete profile trains with an empty per-robot overlay.

Reward defaults = upstream unitree_rl_lab table + the S2-Exp5
upstream-aligned package (W/C/H/G, empirically validated on the predecessor
stack: four gates passed 4/4). Each of the four deviations from the raw
upstream table is marked with its provenance inline; everything else is
verbatim upstream.
"""

from __future__ import annotations

import math
import os
from typing import ClassVar

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from yanshi_rl_lab import mdp
from yanshi_rl_lab.robots.articulation import build_articulation_cfg
from yanshi_rl_lab.robots.profile import RobotProfile
from yanshi_rl_lab.terrains.presets import FLAT_TERRAINS_CFG

# -- S2-Exp5 upstream-aligned package, named constants ------------------------
# Swing-foot target clearance height (m); same value as the upstream exp-form
# term it replaces.
FEET_CLEARANCE_TARGET_HEIGHT_M = 0.1
# Command gate threshold for the clearance penalty (mjlab's value). The gate is
# |cmd_xy| + |cmd_yaw|, so the yaw term keeps the penalty active while turning
# in place.
FEET_CLEARANCE_CMD_THRESHOLD = 0.05


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain (generator swapped per terrain preset by the registry)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=FLAT_TERRAINS_CFG,
        max_init_terrain_level=FLAT_TERRAINS_CFG.num_rows - 1,  # recomputed in __post_init__
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robot: built from the RobotProfile in VelocityEnvCfg.__post_init__
    robot: ArticulationCfg = None
    # sensors (height-scanner prim path is completed from profile.base_link)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot",  # completed in __post_init__
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events. Base-link-specific entries are completed from
    the profile in ``VelocityEnvCfg.__post_init__``."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),  # profile.base_link
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),  # profile.base_link
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP (upstream ranges verbatim)."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP (scale overwritten from profile)."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (real-robot format: no base
    linear velocity in the policy group, 5-step history)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Values are the upstream unitree_rl_lab table except the four entries of the
    S2-Exp5 upstream-aligned package (marked). SceneEntityCfg entries with
    ``None`` names are completed from the profile in ``__post_init__``.
    """

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # S2-Exp5 upstream-aligned package "W" (empirically minimal-sufficient):
    # upstream kept the 2021 legged_gym default 0.5; Isaac Lab's official G1
    # config uses 2.0. Necessity proven by ablation (removing W fails the
    # turn-in-place gate).
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=None)},  # profile.arm_deviation_joints
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=None)},  # profile.waist_deviation_joints
    )
    # S2-Exp5 upstream-aligned package "H": upstream weight was -1.0, but the
    # joint set includes the hip-yaw joints that drive turning in place; Isaac
    # Lab's official G1 uses -0.1 for the identical joint set (10x difference).
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=None)},  # profile.leg_deviation_joints
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-10, params={"target_height": None}  # profile.target_base_height_m
    )

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=None),  # profile.feet_bodies
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),  # profile.feet_bodies
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=None),  # profile.feet_bodies
        },
    )
    # S2-Exp5 upstream-aligned package "C": penalty-form clearance term (after
    # mjlab) replacing the upstream exp-form reward whose exp(-0)=1 paid full
    # score to feet that never move (upstream PR #80 calls this the main cause
    # of slow learning). Gate includes |cmd_yaw|, see mdp.clearance_penalty.
    feet_clearance = RewTerm(
        func=mdp.clearance_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),  # profile.feet_bodies
            "command_name": "base_velocity",
            "target_height": FEET_CLEARANCE_TARGET_HEIGHT_M,
            "command_threshold": FEET_CLEARANCE_CMD_THRESHOLD,
        },
    )
    # S2-Exp5 upstream-aligned package "G" (guard, not a factor): penalizes
    # left/right air-time and contact-time variance (Isaac Lab Spot value
    # -1.0). Structural counter to the "one leg permanently airborne" hack
    # that quantitative gates and static frames failed to catch (v0.45).
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=None)},  # profile.feet_bodies
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=None),  # profile.undesired_contact_bodies
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": None}  # profile.min_base_height_m
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Vendor-neutral configuration for the locomotion velocity-tracking environment."""

    # The robot this env is built for. Set by tasks.registry (or a subclass)
    # before instantiation; never an instance field, so it stays out of the
    # serialized config dumps.
    profile: ClassVar[RobotProfile | None] = None

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization: resolve profile semantic slots, then apply the
        upstream general settings verbatim."""
        assert self.profile is not None, (
            "VelocityEnvCfg needs a RobotProfile: set the `profile` class attribute on a "
            "subclass (normally via yanshi_rl_lab.tasks.registry.register_velocity) before "
            "instantiating."
        )
        profile = self.profile

        # -- robot articulation and base-link wiring -----------------------
        self.scene.robot = build_articulation_cfg(profile).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + profile.base_link
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=profile.base_link)
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=profile.base_link
        )

        # -- actions --------------------------------------------------------
        self.actions.JointPositionAction.scale = profile.action_scale

        # -- rewards: semantic slots ---------------------------------------
        rew = self.rewards
        rew.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(profile.arm_deviation_joints)
        )
        rew.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(profile.waist_deviation_joints)
        )
        rew.joint_deviation_legs.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(profile.leg_deviation_joints)
        )
        rew.base_height.params["target_height"] = profile.target_base_height_m
        rew.gait.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=profile.feet_bodies)
        rew.feet_slide.params["asset_cfg"] = SceneEntityCfg("robot", body_names=profile.feet_bodies)
        rew.feet_slide.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=profile.feet_bodies)
        rew.feet_clearance.params["asset_cfg"] = SceneEntityCfg("robot", body_names=profile.feet_bodies)
        rew.air_time_variance.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=profile.feet_bodies)
        rew.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=profile.undesired_contact_bodies
        )

        # -- terminations ---------------------------------------------------
        self.terminations.base_height.params["minimum_height"] = profile.min_base_height_m

        # -- general settings (upstream verbatim) ---------------------------
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods: tick all sensors on the smallest
        # update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # keep the initial-spawn spread consistent with whatever generator the
        # registry installed (upstream hardcoded rows-1 for its one generator)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows - 1

        # couple terrain-generator curriculum to the terrain_levels term
        # (upstream verbatim)
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# -- play/recording variant ---------------------------------------------------

# Environment variable overriding the number of play envs (e.g. set 32 for
# batch visual checks).
PLAY_NUM_ENVS_ENV_VAR = "PLAY_NUM_ENVS"
# Default to a single robot on video: with 32 envs the subject drowns in the
# array (measured on the predecessor stack -- frame grabs barely showed it).
PLAY_DEFAULT_NUM_ENVS = 1
# Default locked forward speed (m/s) for recordings. Sampling the full command
# range often draws ~0 or negative speeds and the robot stands still on camera.
PLAY_DEFAULT_FORWARD_SPEED_MPS = 0.8
# Difficulty band used on non-flat terrain during play: play defaults would
# collapse the terrain to its easiest rows and record "stairs" that are
# actually flat ground (measured). 0.6-0.9 matches the levels policies
# actually reach.
PLAY_HARD_TERRAIN_DIFFICULTY = (0.6, 0.9)
# Follow camera: slightly above hip height, 45 degrees behind-left, so both
# gait and terrain are visible; locked to the robot so it cannot walk out of
# frame (play scripts are argparse-driven, so this must live in the cfg).
PLAY_CAMERA_EYE = (2.2, 2.2, 1.1)
PLAY_CAMERA_LOOKAT = (0.0, 0.0, 0.4)
PLAY_CAMERA_RESOLUTION = (1280, 720)


def playify(cfg, hard_terrain: bool = False, fwd_vx: float | None = None) -> None:
    """Turn a training env cfg into a play/recording cfg in place.

    Small env count, trimmed terrain grid, locked forward command, follow
    camera, command-arrow overlay off. Ported from the predecessor overlay's
    ``_playify`` (values and rationale unchanged).
    """
    cfg.scene.num_envs = int(os.environ.get(PLAY_NUM_ENVS_ENV_VAR, str(PLAY_DEFAULT_NUM_ENVS)))
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 10
        if hard_terrain:
            cfg.scene.terrain.terrain_generator.difficulty_range = PLAY_HARD_TERRAIN_DIFFICULTY
    # lock a definite forward speed instead of sampling the training range;
    # also disable "commanded standing" envs
    vx = fwd_vx if fwd_vx is not None else PLAY_DEFAULT_FORWARD_SPEED_MPS
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    for _r in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
        _r.lin_vel_x = (vx, vx)  # constant forward
        _r.lin_vel_y = (0.0, 0.0)
        _r.ang_vel_z = (0.0, 0.0)  # straight line, steady camera
    # follow camera
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = PLAY_CAMERA_EYE
    cfg.viewer.lookat = PLAY_CAMERA_LOOKAT
    cfg.viewer.resolution = PLAY_CAMERA_RESOLUTION
    # hide the command debug arrows (visual clutter on recordings)
    cfg.commands.base_velocity.debug_vis = False


@configclass
class VelocityPlayEnvCfg(VelocityEnvCfg):
    """Play/recording variant of the base config (single robot, locked forward
    command, follow camera). The registry builds terrain-specific play classes
    on top of its env classes; this base exists for direct subclassing."""

    def __post_init__(self):
        super().__post_init__()
        playify(self)
