# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Unitree G1 profile (23-DoF configuration).

Peer of ``unitree/g1/dof29``. Same vendor, same legs, five arm joints per side
instead of seven, and one waist joint instead of three. Both configurations
come out of the SAME fetched asset tree (``assets/fetch.py unitree/g1``) and
differ only in which files this profile points at.

⚠️ The vendor MJCF ``g1_23dof.xml`` declares **29 motors over a 23-DoF body**.
Unitree does this on purpose: both configurations then share one DDS message
layout. The six extra joints (waist roll/pitch, both wrists' pitch/yaw) sit on
massless bodies parked at ``pos="0 0 20"``, detached from the kinematic tree --
verified with MuJoCo 3.10 by walking each hinge's body chain (23 of 29 hinges
reach the floating base). The file is not broken; motor count is simply not
DoF count here. Reading it as 29-DoF is how the first published G1 leaderboard
row got measured on the wrong robot (decision ledger D11).

Value provenance (established 2026-08-02, CPU-only, MuJoCo 3.10.0):

- **Asset files**: the vendor tree pinned in ``assets/registry.py``
  (``unitree_ros`` @ f3772ce5 for the URDF, ``unitree_mujoco`` @ ae6a8403 for
  the MJCF). Joint table read directly from
  ``g1_23dof_rev_1_0.urdf``: 23 non-fixed joints besides the floating base.
- **SDK joint ordering**: Unitree's own table, ``assets/unitree/g1/mjcf/g1/
  g1_joint_index_dds.md`` § "23DOF 版本" (indices 0-22). See the note on the
  ``joint_sdk_names`` field about the naming generation mismatch.
- **Actuator groups**: the ``dof29`` groups restricted to the joints this
  configuration has. This is a DERIVATION, and here is the evidence for it:
  the ``ctrlrange`` of every one of the 29 motors is **byte-identical between
  g1_23dof.xml and g1_29dof.xml** (checked programmatically, 0 differences),
  i.e. the shared joints are driven by the same motors, so the same kp/kd/
  effort/armature apply. Nothing was invented; two groups shrink and one
  disappears (see the comments at ``actuator_groups``).
- **Height facts**: measured, not assumed. Forward kinematics at the default
  pose puts ``pelvis`` at 0.7930 m and ``torso_link`` at 0.8470 m in BOTH
  models -- identical to four decimals -- so the dof29 height facts transfer
  exactly. (Total mass differs: 34.14 kg here vs 35.11 kg, the six absent
  actuators.)

⚠️ Known upstream inconsistency, recorded so nobody "fixes" it silently: the
vendor MJCF gives hip-roll a ``ctrlrange`` of +-88 N.m and the ankles +-50,
while the Isaac training config these groups come from puts hip-roll in the
139 N.m group and the ankles at 25 N.m. The training config is the authority
for training (it is what the dof29 policies were trained under and what its
gates were measured with); the discrepancy is upstream's, not ours, and it is
inherited here deliberately so the two configurations stay comparable.

⚠️ **No policy has been trained on this configuration.** It is registered so
that the 23-DoF G1 can be trained and ranked; it has no leaderboard entry, and
the old 4 m/s speed-ladder result from the predecessor stack is NOT a result
for this robot -- that experiment ran on Isaac Lab's stock G1 (a 37-joint model
with fingers and an older joint-naming generation, spawned from a remote
Nucleus USD that this repository cannot pin).
"""

from yanshi_rl_lab.robots.profile import ActuatorGroup, RobotProfile

G1_DOF23_PROFILE = RobotProfile(
    vendor="unitree",
    model="g1",
    # Upstream's own name for this configuration: unitree_ros ships
    # g1_23dof_rev_1_0.urdf and unitree_mujoco ships g1_23dof.xml.
    variant="dof23",
    # -- asset locations (relative to the assets root) --------------------
    # Same fetched tree as dof29; only the file names differ.
    urdf="unitree/g1/urdf/g1_description/g1_23dof_rev_1_0.urdf",
    mjcf="unitree/g1/mjcf/g1/g1_23dof.xml",
    scene_mjcf="unitree/g1/mjcf/g1/scene_23dof.xml",
    usd=None,  # spawn from URDF, same as dof29
    # -- embodiment -------------------------------------------------------
    # Same legs and the same vendor pelvis height (0.793 m in both MJCFs).
    spawn_height_m=0.8,
    # Every joint the dof29 default pose names also exists here (its arm
    # entries stop at wrist_roll), so the standing pose is the same pose --
    # not a trimmed copy.
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
    # Same three motor classes as dof29 minus one: every regex below must
    # match at least one real joint or Isaac Lab raises at articulation build
    # time ("Not all regular expressions are matched"), which is exactly the
    # check that keeps a copied-over group honest.
    actuator_groups={
        "N7520-14.3": ActuatorGroup(
            # waist_yaw is the ONLY waist joint here (dof29 has three).
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
            # dof29's version of this group also lists waist_roll_joint and
            # waist_pitch_joint; neither exists on this body.
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
            ],
            effort_limit=25,
            velocity_limit=37,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
            },
            armature=0.01,
        ),
        # dof29's fourth group, W4010-25, drove wrist pitch/yaw. This body has
        # neither joint, so the group is absent rather than empty.
    },
    # Real-robot SDK ordering, transcribed from Unitree's own
    # g1_joint_index_dds.md § "23DOF 版本" (IDL indices 0-22):
    # left leg 6 -> right leg 6 -> torso 1 -> left arm 5 -> right arm 5.
    #
    # ⚠️ Naming generations differ between that table and the rev_1_0 URDF,
    # and the mapping is by IDL index and kinematic position, not by string:
    #   index 12  TORSO           -> waist_yaw_joint      (the single waist joint)
    #   index 16  L_ELBOW_PITCH   -> left_elbow_joint     (4th arm joint)
    #   index 17  L_ELBOW_ROLL    -> left_wrist_roll_joint(5th arm joint, the
    #                                roll distal to the elbow)
    # and mirrored for the right arm. Both name the same five-joint arm chain
    # shoulder(pitch,roll,yaw) -> elbow -> roll; the table predates the
    # rev_1_0 rename. Anyone wiring a real 23-DoF G1 should re-confirm this
    # against the SDK build they are running before trusting it with torque.
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
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ],
    pd_mode="implicit",  # gains live in the simulator's implicit joint drive (Isaac ImplicitActuator)
    # -- semantic body map -------------------------------------------------
    # Every slot below resolves against this body's real joints/bodies:
    # torso_link exists, .*ankle_roll.* matches both feet, .*_wrist_.* now
    # matches only wrist_roll, and waist.* matches only waist_yaw.
    base_link="torso_link",
    feet_bodies=".*ankle_roll.*",
    undesired_contact_bodies="(?!.*ankle.*).*",
    leg_deviation_joints=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
    arm_deviation_joints=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
    # One joint instead of three. Note the consequence: the deviation penalty
    # that dof29 spreads over yaw/roll/pitch lands entirely on waist_yaw here,
    # the one waist joint that helps turning. If a turn gate underperforms on
    # this configuration, look here first.
    waist_deviation_joints=["waist.*"],
    # -- task-derivable facts ---------------------------------------------
    # Measured identical to dof29 (torso_link at 0.8470 m in both models at
    # the default pose), so the dof29 values transfer rather than being reused
    # on faith.
    target_base_height_m=0.78,
    min_base_height_m=0.2,
    self_collisions=True,
    action_scale=0.25,
    root_joint_name="floating_base_joint",  # verified in g1_23dof.xml
)
