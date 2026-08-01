# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Berkeley Humanoid Lite velocity-task registration (attempt 3).

Attempt-3 plan (Jeff-approved): ABANDON the in-house reward/command line and
transcribe the vendor's humanoid training configuration wholesale. Rationale:

- Attempt 1 (fleet base + official slow curriculum) and attempt 2 (D7
  dead-command override, formerly in this file) BOTH converged to a standing
  statue -- attempt 2's failure is triple-evidenced (2026-07-31 acceptance):
  training-side error_vel_xy 0.875 m/s and RISING at 10000 iterations,
  sim2sim gates 1/4 with 0.04 m displacement, and the trained policy driven
  inside Isaac itself moves at 0.03 m/s. The D7 hypothesis ("the aligned
  reward table only fails under the slow curriculum") is thereby refuted:
  the fleet reward table itself does not fit this robot, so the fallback is
  the vendor's own empirically-validated recipe.
- Every value below is a VENDOR FACT, transcribed with its source line
  (HybridRobotics/Berkeley-Humanoid-Lite @ 984741a3, paths abbreviated as
  ``official:`` = ``source/berkeley_humanoid_lite/berkeley_humanoid_lite/
  tasks/locomotion/velocity/``). This is not magic-number tuning: BHL has no
  battle record in this stack, and the vendor config is the correct prior.
- Documented deviations from the vendor config (each deliberate):
  1. PPO max_iterations 6000 -> 10000 (fleet-uniform budget for Yanshi Rank
     comparability; see agents.py in this package).
  2. Observation TERM ORDER keeps the fleet layout (policy group
     [ang_vel, gravity, commands, pos, vel, last_action]; official puts
     velocity_commands first, env_cfg.py L51-73). Order is a deploy-layout
     fact the contract carries, not a learning fact: a policy trained from
     scratch is indifferent. All per-term noise/scale/history values ARE
     transcribed.
  3. The command carrier stays our ``UniformLevelVelocityCommandCfg``
     subclass with ``limit_ranges`` pinned equal to ``ranges`` and the
     curriculum term removed -- behaviorally identical to the official
     stock ``UniformVelocityCommandCfg`` (official env_cfg.py L25-39 has no
     curriculum, CurriculumsCfg L319-323 is empty), while keeping the
     contract exporter's limit_ranges-first read and playify intact.
  4. Official splits hip/ankle_roll and shoulder/elbow joint-deviation into
     four terms (L195-214); we keep the profile's two semantic slots
     (legs = hip_yaw+hip_roll+ankle_roll, arms = shoulder*3+elbow*2) at the
     same weight -1.0 -- mathematically identical L1 sums.

Attempt-2 history (D7 dead-command override) lives in the decision ledger;
this file documents only the current recipe.

Registry naming rule (tasks/registry.py): vendor "berkeley" + model
"humanoid_lite" -> task-ID segments "Berkeley" + "Humanoid-Lite", so the
generated ID is ``Yanshi-Velocity-Flat-Berkeley-Humanoid-Lite-v0``.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from yanshi_rl_lab import mdp
from yanshi_rl_lab.robots.berkeley.humanoid_lite.profile import BHL_PROFILE
from yanshi_rl_lab.tasks.locomotion.velocity.velocity_env_cfg import EventCfg, RewardsCfg
from yanshi_rl_lab.tasks.registry import register_velocity

# ---------------------------------------------------------------------------
# Official command envelope (official: config/humanoid/env_cfg.py L33-38)
# ---------------------------------------------------------------------------
OFFICIAL_LIN_VEL_X = (-1.0, 1.0)  # L34
OFFICIAL_LIN_VEL_Y = (-0.5, 0.5)  # L35
OFFICIAL_ANG_VEL_Z = (-1.5, 1.5)  # L36
OFFICIAL_HEADING = (-math.pi, math.pi)  # L37
OFFICIAL_HEADING_STIFFNESS = 0.5  # L30
OFFICIAL_BAD_ORIENTATION_LIMIT_RAD = 0.78  # L227 (note: fleet default is 0.8)
OFFICIAL_DECIMATION = 8  # L354-356 (25 Hz policy)


@configclass
class BhlOfficialRewardsCfg(RewardsCfg):
    """Fleet reward base + the three official terms the fleet table lacks.

    Field names follow the official config (official: env_cfg.py L104-214)
    so training logs read like the vendor's; the disposition of every
    inherited term is set explicitly in ``_official_humanoid_recipe``.
    """

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)  # official L123-126
    dof_torques = RewTerm(
        func=mdp.joint_torques_l2,  # official L148-152 ("dof_torques_l2")
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,  # official L165-173; Isaac Lab stock,
        # identical to the vendor's vendored copy (official mdp/rewards.py L36-57) --
        # the ONLY anti-statue term in the official table.
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BHL_PROFILE.feet_bodies),
            "threshold": 0.5,  # official L170
        },
    )


@configclass
class BhlOfficialEventCfg(EventCfg):
    """Fleet event base + the two official startup randomizations the fleet
    table lacks (official: env_cfg.py L256-274)."""

    add_all_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,  # yanshi port, mdp/events.py
        mode="startup",  # official L256-264
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    scale_all_actuator_torque_constant = EventTerm(
        func=mdp.randomize_actuator_gains,  # Isaac Lab stock ManagerTermBase class,
        mode="startup",  # exactly what official L265-274 calls
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


def _official_humanoid_recipe(cfg) -> None:
    """Attempt-3 override: wholesale transcription of the vendor humanoid MDP.

    Section comments carry the official line refs (env_cfg.py @984741a3).
    """

    # -- commands (L22-39): heading-mode, official envelope, no curriculum ---
    cmd = cfg.commands.base_velocity
    cmd.heading_command = True  # L29
    cmd.heading_control_stiffness = OFFICIAL_HEADING_STIFFNESS  # L30
    cmd.rel_heading_envs = 1.0  # L32 (all envs heading-mode)
    # rel_standing_envs 0.02 (L31) and resampling_time_range (10, 10) (L26)
    # already match the fleet base; no deadband (official has none).
    for r in (cmd.ranges, cmd.limit_ranges):
        r.lin_vel_x = OFFICIAL_LIN_VEL_X
        r.lin_vel_y = OFFICIAL_LIN_VEL_Y
        r.ang_vel_z = OFFICIAL_ANG_VEL_Z
        r.heading = OFFICIAL_HEADING
    cfg.curriculum.lin_vel_cmd_levels = None  # official CurriculumsCfg is empty (L319-323)

    # -- rewards (L104-214): official table ---------------------------------
    rew = BhlOfficialRewardsCfg()
    # task terms: weights SWAPPED vs the fleet table (fleet 1.0/2.0, official 2.0/1.0)
    rew.track_lin_vel_xy.weight = 2.0  # L113; func yaw-frame exp + std 0.5 already match (L110-112)
    rew.track_ang_vel_z.func = mdp.track_ang_vel_z_world_exp  # L116 (world-frame variant)
    rew.track_ang_vel_z.weight = 1.0  # L118; std 0.5 already matches
    # terms the official table does not have -> removed
    rew.alive = None  # official uses termination_penalty instead (L123-126)
    rew.energy = None
    rew.joint_vel = None
    rew.gait = None
    rew.feet_clearance = None
    rew.air_time_variance = None
    rew.base_height = None  # already None for BHL (root frame at ground level)
    # weights transcribed
    rew.base_linear_velocity.weight = -0.1  # lin_vel_z_l2, L131
    rew.base_angular_velocity.weight = -0.05  # ang_vel_xy_l2, L135 (same as fleet)
    rew.action_rate.weight = -0.001  # L146
    rew.joint_acc.weight = -1.0e-7  # dof_acc_l2, L156
    rew.dof_pos_limits.weight = -1.0  # L160
    rew.flat_orientation_l2.weight = -1.0  # L140
    rew.feet_slide.weight = -0.1  # L181
    rew.feet_slide.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BHL_PROFILE.feet_bodies)
    rew.feet_slide.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=BHL_PROFILE.feet_bodies)
    # joint deviation: official four terms (L195-214) -> profile's two slots, same weight
    rew.joint_deviation_arms.weight = -1.0
    rew.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(BHL_PROFILE.arm_deviation_joints)
    )
    rew.joint_deviation_legs.weight = -1.0
    rew.joint_deviation_legs.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(BHL_PROFILE.leg_deviation_joints)
    )
    rew.joint_deviation_waists = None  # BHL has no waist (fleet base already drops it)
    # undesired contacts: weight -1.0 matches (L191); the profile's body list
    # already IS the official set (base/hip/knee/shoulder/elbow, L188) -- rewire explicitly
    rew.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=BHL_PROFILE.undesired_contact_bodies
    )
    cfg.rewards = rew

    # -- events (L231-316): official randomizations --------------------------
    ev = BhlOfficialEventCfg()
    ev.physics_material.params["static_friction_range"] = (0.4, 1.2)  # L240
    ev.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)  # L241
    ev.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)  # L251
    ev.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BHL_PROFILE.base_link)
    # reset root velocity: lin x/y +-0.5, lin z STAYS 0.0 (L284 -- the plan's
    # "all axes +-0.5" note is wrong here, the official file wins), ang +-0.5
    ev.reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),  # L282
        "y": (-0.5, 0.5),  # L283
        "z": (0.0, 0.0),  # L284
        "roll": (-0.5, 0.5),  # L285
        "pitch": (-0.5, 0.5),  # L286
        "yaw": (-0.5, 0.5),  # L287
    }
    ev.reset_robot_joints.params["position_range"] = (0.5, 1.5)  # L294
    ev.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)  # L295
    ev.base_external_force_torque.params["force_range"] = (-2.0, 2.0)  # L304
    ev.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)  # L305
    ev.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=BHL_PROFILE.base_link
    )
    ev.push_robot = None  # official comments it out (L311-316)
    cfg.events = ev

    # -- observations (L42-89): official noise/scale, no history -------------
    pol = cfg.observations.policy
    pol.history_length = 0  # official groups carry no history (L75-76, 83-84)
    pol.base_ang_vel.scale = None  # official has no scale (L55-58)
    pol.base_ang_vel.noise = Unoise(n_min=-0.3, n_max=0.3)  # L57
    pol.joint_pos_rel.noise = Unoise(n_min=-0.05, n_max=0.05)  # L66
    pol.joint_vel_rel.scale = None  # official has no scale (L68-72)
    pol.joint_vel_rel.noise = Unoise(n_min=-2.0, n_max=2.0)  # L71
    # projected_gravity noise +-0.05 (L61) and last_action/velocity_commands
    # plain already match the fleet terms; term ORDER intentionally kept (deviation 2).
    cri = cfg.observations.critic
    cri.history_length = 0
    cri.base_ang_vel.scale = None  # official critic inherits the scale-free policy terms
    cri.joint_vel_rel.scale = None

    # -- terminations (L217-229) ---------------------------------------------
    # official limit is 0.78 rad with a base-body filter (L225-228); the body
    # filter is a no-op in Isaac Lab's bad_orientation (it reads the root's
    # projected gravity), so only the limit is transcribed.
    cfg.terminations.bad_orientation.params["limit_angle"] = OFFICIAL_BAD_ORIENTATION_LIMIT_RAD

    # -- control rate (L354-356): 25 Hz policy --------------------------------
    cfg.decimation = OFFICIAL_DECIMATION
    cfg.sim.render_interval = OFFICIAL_DECIMATION  # base couples these (velocity_env_cfg.py)
    cfg.scene.height_scanner.update_period = OFFICIAL_DECIMATION * cfg.sim.dt


def _bhl_play_overrides(cfg) -> None:
    """Play only: heading P-control recomputes wz every step and would fight
    playify's locked commands -- revert to direct-wz (G1 rough play precedent,
    config/unitree/g1/__init__.py ``_g1_play_overrides``)."""
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.heading = None
    cfg.commands.base_velocity.limit_ranges.heading = None


register_velocity(
    BHL_PROFILE,
    __name__,
    terrains=("flat",),
    overrides=_official_humanoid_recipe,
    play_overrides=_bhl_play_overrides,
    runner_entry_points={
        # attempt-3 official PPO transcription (agents.py in this package)
        "flat": "yanshi_rl_lab.tasks.locomotion.velocity.config.berkeley.humanoid_lite.agents:BerkeleyHumanoidLitePPORunnerCfg"
    },
)
