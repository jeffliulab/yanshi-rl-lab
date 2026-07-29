# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1 profile (29-DoF deploy configuration).

Value provenance (transcribed 2026-07-29, read-only from the predecessor stack):

- Actuator groups, default pose, spawn height and SDK joint ordering:
  ``UNITREE_G1_29DOF_CFG`` in unitree_rl_lab ``assets/robots/unitree.py``
  (upstream repository Apache-2.0; group names are the motor model numbers).
- Semantic body map and task-derivable facts: the official G1 29-DoF velocity
  base config, unitree_rl_lab
  ``tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`` (torso link,
  ``.*ankle_roll.*`` feet, ``(?!.*ankle.*).*`` undesired contacts, the three
  joint-deviation groups, base-height target 0.78 m, termination height 0.2 m,
  action scale 0.25).
- Asset relative paths point into the vendor tree fetched by
  ``assets/fetch.py unitree/g1`` (pinned upstream commits in ``assets/registry.py``).
"""

from yanshi_rl_lab.robots.profile import ActuatorGroup, RobotProfile

G1_PROFILE = RobotProfile(
    vendor="unitree",
    model="g1",
    # -- asset locations (relative to the assets root) --------------------
    urdf="unitree/g1/urdf/g1_description/g1_29dof_rev_1_0.urdf",
    mjcf="unitree/g1/mjcf/g1/g1_29dof.xml",
    scene_mjcf="unitree/g1/mjcf/g1/scene_29dof.xml",
    usd=None,  # spawn from URDF, same as the predecessor stack
    # -- embodiment -------------------------------------------------------
    spawn_height_m=0.8,
    default_joint_pos={
        "left_hip_pitch_joint": -0.1,
        "right_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.3,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
        ".*_elbow_joint": 0.97,
        "left_wrist_roll_joint": 0.15,
        "right_wrist_roll_joint": -0.15,
    },
    # Group names are Unitree motor model numbers (upstream convention).
    actuator_groups={
        "N7520-14.3": ActuatorGroup(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit=88,
            velocity_limit=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ActuatorGroup(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit=139,
            velocity_limit=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16": ActuatorGroup(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit=25,
            velocity_limit=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ActuatorGroup(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit=5,
            velocity_limit=22,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
    # Real-robot SDK ordering (upstream unitree.py L482-512):
    # left leg 6 -> right leg 6 -> waist 3 -> left arm 7 -> right arm 7.
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    pd_mode="implicit",  # gains live in the simulator's implicit joint drive (Isaac ImplicitActuator)
    # -- semantic body map (verified against the upstream velocity base) ---
    base_link="torso_link",
    feet_bodies=".*ankle_roll.*",
    undesired_contact_bodies="(?!.*ankle.*).*",
    leg_deviation_joints=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
    arm_deviation_joints=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
    waist_deviation_joints=["waist.*"],
    # -- task-derivable facts ---------------------------------------------
    target_base_height_m=0.78,  # upstream base_height reward target
    min_base_height_m=0.2,  # upstream base_height termination minimum
    self_collisions=True,
    action_scale=0.25,  # upstream JointPositionAction scale
    # Free-joint name in the official unitree_mujoco G1 MJCF (verified in
    # assets/unitree/g1/mjcf/g1/g1_29dof.xml and in the predecessor stack's
    # sim2sim scene; both use the same name).
    root_joint_name="floating_base_joint",
)
