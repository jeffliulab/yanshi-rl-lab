# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""AgiBot Lingxi X2 profile (X2-Ultra, 31 actuated joints).

Value provenance (model archaeology done 2026-07-29, CPU-only MuJoCo 3.10.0
loading of the vendor MJCF + ``xml.etree`` parsing of the vendor URDF; assets
pinned by ``assets/registry.py``, fetched to ``assets/agibot/x2/X2_URDF-v1.4.0/``):

- Joint table, torque limits, armature/damping/frictionloss, body tree, total
  mass (44.801 kg, both URDF and MJCF) and free-joint name all read from
  ``X2-Ultra.xml`` / ``X2-Ultra.urdf`` (v1.4.0). URDF and MJCF agree
  on all 31 joint names, ranges and effort limits (verified programmatically);
  they differ only in document order, see ``joint_sdk_names`` below.
- ``default_joint_pos`` / ``spawn_height_m`` / ``target_base_height_m`` /
  ``min_base_height_m`` are numerically derived by CPU MuJoCo forward
  kinematics (2026-07-30, decision D5): the vendor model ships only the
  straight-knee zero pose, and training run #1 with that pose never learned
  to stand (mean episode length ~6 steps, 100% bad_orientation). The
  bent-knee pose below is solved, not hand-picked -- constraints and solved
  numbers at each field.

⚠️ PD GAINS ARE DERIVED, NOT OFFICIAL. AgiBot publishes no kp/kd anywhere in
the v1.4.0 release. Gains below come from the natural-frequency method
(the derivation paradigm of Unitree's mimic config, unitree_rl_lab
``assets/robots/unitree.py`` ARMATURE_*/NATURAL_FREQ/DAMPING_RATIO block;
mjlab uses the same recipe):

    kp = I_reflected * w_n^2          w_n = 2*pi*10 rad/s (10 Hz)
    kd = 2 * zeta * I_reflected * w_n zeta = 2.0

with I_reflected = the reflected rotor inertia that the simulator itself uses
as the joint ``armature``. Details and per-group choices at the constants
below. First smoke training may revise these numbers; any revision must be
recorded in the maintainer decision ledger.
"""

import math

from yanshi_rl_lab.robots.profile import ActuatorGroup, RobotProfile

# ---------------------------------------------------------------------------
# PD gain derivation (derived, not official -- see module docstring).
#
# Source of the method + constants: unitree_rl_lab assets/robots/unitree.py
# (UNITREE_G1_29DOF_MIMIC_CFG block): NATURAL_FREQ = 10 Hz * 2*pi,
# DAMPING_RATIO = 2.0, STIFFNESS_<motor> = ARMATURE_<motor> * NATURAL_FREQ**2,
# DAMPING_<motor> = 2 * DAMPING_RATIO * ARMATURE_<motor> * NATURAL_FREQ.
# Unitree uses ONLY the per-motor reflected rotor inertia (their measured
# armature) as I_reflected -- no link-side inertia term -- because with high
# gear ratios the reflected rotor inertia dominates the inner tracking loop.
# We follow that paradigm exactly.
# ---------------------------------------------------------------------------
NATURAL_FREQ_RAD_S = 10.0 * 2.0 * math.pi  # 10 Hz target loop bandwidth
DAMPING_RATIO = 2.0  # over-damped, same as Unitree mimic

# I_reflected for the 120/60/36 N*m motor tiers: the vendor MJCF declares
# armature = 0.03 for every joint (default class "x2", X2-Ultra.xml line 9).
# 0.03 kg*m^2 is plausible for these large gearmotors (Unitree's 139 N*m
# motor measures 0.0251) and it is what MuJoCo will simulate, so using it as
# I_reflected keeps the in-sim closed loop at exactly 10 Hz / zeta 2.0.
MAIN_TIER_ARMATURE = 0.03  # vendor MJCF value (all joints)

# I_reflected for the small motors (wrist pitch/roll 6 N*m, head 2.3 N*m):
# the uniform 0.03 is clearly a class default, not a per-motor measurement
# (a 2.3 N*m head gimbal cannot have the same rotor inertia as a 120 N*m hip
# motor), and kp = 0.03 * w_n^2 = 118.4 would saturate a 6 N*m motor at
# 0.05 rad tracking error. We therefore rate these joints at Unitree's
# small-motor tier: ARMATURE_4010 = 0.00425 (W4010, 5 N*m -- the G1 motor
# closest in torque class to X2's 6 / 2.3 N*m joints).
SMALL_MOTOR_ARMATURE = 0.00425  # = Unitree ARMATURE_4010 (derived stand-in)

KP_MAIN = MAIN_TIER_ARMATURE * NATURAL_FREQ_RAD_S**2  # 118.44 N*m/rad
KD_MAIN = 2.0 * DAMPING_RATIO * MAIN_TIER_ARMATURE * NATURAL_FREQ_RAD_S  # 7.54
KP_SMALL = SMALL_MOTOR_ARMATURE * NATURAL_FREQ_RAD_S**2  # 16.78 N*m/rad
KD_SMALL = 2.0 * DAMPING_RATIO * SMALL_MOTOR_ARMATURE * NATURAL_FREQ_RAD_S  # 1.07
# NOTE: the small groups still carry armature = 0.03 below (the MJCF value,
# so Isaac and MuJoCo simulate the same joint inertia). With kp/kd from the
# 0.00425 tier their in-sim response is w_n = sqrt(16.78/0.03) = 23.6 rad/s
# (3.8 Hz), zeta = 0.75 -- softer and slightly under-damped, acceptable for
# joints that play no role in locomotion (head, wrist pitch/roll).

X2_PROFILE = RobotProfile(
    vendor="agibot",
    model="x2",
    # Upstream's own name for this revision: the pinned repo ships the model
    # under X2_URDF-v1.4.0 (see assets/registry.py).
    variant="v1_4_0",
    # -- asset locations (relative to the assets root) --------------------
    urdf="agibot/x2/X2_URDF-v1.4.0/X2-Ultra.urdf",
    mjcf="agibot/x2/X2_URDF-v1.4.0/X2-Ultra.xml",
    scene_mjcf="agibot/x2/X2_URDF-v1.4.0/scene.xml",
    usd=None,  # spawn from URDF, same flow as G1
    # -- embodiment -------------------------------------------------------
    # numerically derived (FK): bent-knee standing root height is 0.63751 m
    # (pelvis -> lowest sole point, MuJoCo FK over all collision geoms at the
    # default pose below); + the G1-implied drop clearance 0.8 - 0.7842 =
    # 0.0158 m -> 0.6533, rounded to 0.65. FK check at spawn z = 0.65: lowest
    # sole point is +12.5 mm above ground (inside the 5-15 mm drop band).
    spawn_height_m=0.65,
    # numerically derived (FK) bent-knee default pose (decision D5,
    # 2026-07-30). The vendor model has no keyframe: its official default is
    # the all-zeros straight-knee pose, which is kinematically singular, and
    # training run #1 spawned with it never learned to stand (mean episode
    # length ~6 steps = 0.12 s, 100% bad_orientation terminations; sim2sim
    # gates 0/4, FELL at 0.1 s). Pose solved under four constraints, all
    # FK-verified with CPU MuJoCo 3.10 on the vendor scene.xml:
    #   (1) soles flat on ground: hip/knee/ankle pitch axes are all (0,1,0)
    #       with no intermediate body rotations (verified in the MJCF chain),
    #       so hip + knee + ankle = 0 exactly; FK foot-body tilt = 0.0000 deg
    #       and all 20 sole spheres share the lowest z.
    #   (2) pelvis drops 3-6 cm vs the straight-knee stand (0.67495 m):
    #       knee = 0.70 rad gives 0.63751 m, a 3.74 cm drop.
    #   (3) whole-body CoM horizontal projection on the support-polygon
    #       center: hip solved on a 0.002 rad grid to center mj subtree_com;
    #       at hip = -0.38 the residual is dx = -0.25 mm, dy = +0.33 mm
    #       (constraint: < 2 cm).
    #   (4) every joint >= 0.1 rad from its limits: worst margin is
    #       shoulder_roll at 0.211 rad (the +/-0.15 roll below exists partly
    #       because 0.0 sits only 0.061 rad from the roll limit).
    # Sign conventions FK-probed, NOT copied from G1: knee bends positive
    # (range [0, 2.41], same as G1) but the elbow bends NEGATIVE (range
    # [-2.356, 0], opposite of G1), and the G1 forearm points forward at
    # zero while the X2 forearm hangs down -- raw G1 arm angles must not be
    # reused. Arms: natural hang, shoulder pitch +0.20 (slightly back),
    # elbow -0.40 (forearm slightly forward), roll +/-0.15 outward;
    # FK min hand/forearm-to-thigh clearance 9.6 cm per side (constraint:
    # > 5 cm).
    default_joint_pos={
        ".*_hip_pitch_joint": -0.38,
        ".*_knee_joint": 0.70,
        ".*_ankle_pitch_joint": -0.32,  # = -(hip + knee), keeps soles flat
        ".*_shoulder_pitch_joint": 0.20,
        "left_shoulder_roll_joint": 0.15,
        "right_shoulder_roll_joint": -0.15,
        ".*_elbow_joint": -0.40,
        # all other joints (hip roll/yaw, ankle roll, waist x3, head x2,
        # wrist x3) stay 0.0; their smallest limit margin is 0.236 rad.
    },
    # Groups are peak-torque tiers, which on X2 are exactly the motor classes
    # (README changelog names the 60 / 36 / 6 N*m motor upgrades). Effort
    # limits = MJCF ctrlrange = URDF <limit effort> (identical, verified);
    # velocity limits = URDF <limit velocity> (MJCF motors carry none).
    # waist_roll sits alone because its URDF velocity limit (13.088 rad/s)
    # differs from the rest of the 36 N*m tier (14.6608 rad/s).
    actuator_groups={
        "tau120_hip_knee_waist_yaw": ActuatorGroup(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", "waist_yaw_joint"],
            effort_limit=120.0,
            velocity_limit=11.93808,
            stiffness=KP_MAIN,
            damping=KD_MAIN,
            armature=MAIN_TIER_ARMATURE,
        ),
        "tau60_ankle_pitch_shoulder": ActuatorGroup(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit=60.0,
            velocity_limit=13.6136,
            stiffness=KP_MAIN,
            damping=KD_MAIN,
            armature=MAIN_TIER_ARMATURE,
        ),
        "tau36_ankle_roll_waist_pitch_arm": ActuatorGroup(
            joint_names_expr=[
                ".*_ankle_roll_joint",
                "waist_pitch_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit=36.0,
            velocity_limit=14.6608,
            stiffness=KP_MAIN,
            damping=KD_MAIN,
            armature=MAIN_TIER_ARMATURE,
        ),
        "tau36_waist_roll": ActuatorGroup(
            joint_names_expr=["waist_roll_joint"],
            effort_limit=36.0,
            velocity_limit=13.088,
            stiffness=KP_MAIN,
            damping=KD_MAIN,
            armature=MAIN_TIER_ARMATURE,
        ),
        "tau6_wrist": ActuatorGroup(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit=6.0,
            velocity_limit=15.708,
            stiffness=KP_SMALL,
            damping=KD_SMALL,
            armature=MAIN_TIER_ARMATURE,
        ),
        "tau2p3_head": ActuatorGroup(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit=2.3,
            velocity_limit=10.472,
            stiffness=KP_SMALL,
            damping=KD_SMALL,
            armature=MAIN_TIER_ARMATURE,
        ),
    },
    # Joint ordering: DERIVED, not from a vendor SDK document (the v1.4.0
    # release ships no SDK). This is the URDF document order, which equals
    # the MJCF <actuator> block order: left leg 6 -> right leg 6 -> waist 3
    # (yaw, pitch, roll) -> head 2 -> left arm 7 -> right arm 7. Note the
    # MJCF kinematic-tree joint order differs (head joints come last there);
    # our runtime maps joints by name, so only THIS list defines deploy order.
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
        "waist_pitch_joint",
        "waist_roll_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ],
    # Training drives the joints through Isaac's ImplicitActuatorCfg (see
    # robots/articulation.py), so the deploy loop must run implicit PD too --
    # same reasoning as G1; mixing the two modes flips the robot on first
    # contact (alice-house lesson).
    pd_mode="implicit",
    # -- semantic body map (verified against the MJCF body tree) -----------
    # torso_link exists on X2 (10.99 kg, heaviest body, carries the torso
    # IMU) -- same semantic role as G1's torso_link.
    base_link="torso_link",
    # Foot collision geoms (sole spheres + foot cylinders) all live on
    # left/right_ankle_roll_link -- same pattern as G1.
    feet_bodies=".*ankle_roll.*",
    undesired_contact_bodies="(?!.*ankle.*).*",
    # Same functional split as the G1 map; every regex checked against the
    # X2 joint table above.
    leg_deviation_joints=[".*_hip_roll_joint", ".*_hip_yaw_joint"],
    arm_deviation_joints=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"],
    # Head joints ride in the waist slot: like the waist they are posture
    # joints that should stay near default during locomotion, and the profile
    # schema has exactly three deviation slots (adding a fourth is a shared
    # base-config change, out of scope for the X2 CPU milestone).
    waist_deviation_joints=["waist_.*_joint", "head_.*_joint"],
    # -- task-derivable facts ---------------------------------------------
    # numerically derived (FK): standing root height at the bent-knee default
    # pose is 0.63751 m (sole on ground). G1's reward target /
    # standing-height ratio is 0.78 / 0.7842 = 0.9946;
    # 0.63751 * 0.9946 = 0.63406, rounded to 0.63.
    target_base_height_m=0.63,
    # Derivation: G1's termination floor / target ratio is 0.2 / 0.78 =
    # 0.2564; 0.63 * 0.2564 = 0.1615, rounded to 0.16.
    min_base_height_m=0.16,
    # False since attempt 3 (decision D9, 2026-07-31), THE attempt-3 fix.
    # The vendor URDF reuses VISUAL meshes as collision meshes and they
    # interpenetrate at the default pose in five places (torso<->head_pitch
    # -2.0 cm, torso<->waist_yaw -2.0 cm, wrist_yaw<->wrist_roll -2.9 cm,
    # knee<->ankle_roll -4.2 cm, pelvis<->hip -1.8 cm; CPU MuJoCo convex-hull
    # probe). With self-collisions on, Isaac reports 8-19 kN of permanent
    # internal contact force (robot weight is 440 N), which (a) shows up as
    # the ~10 rad/s head/wrist "oscillation" of the zero-action probe -- it
    # is contact-force spray, not under-damped self-excitation (kd x3/x5
    # changed nothing; with self-collisions off max|qvel| drops 13 -> 0.44),
    # and (b) keeps the undesired_contacts reward at -0.36~-0.44 per step
    # while merely standing (G1: -0.006), so PPO converges to a suicide
    # equilibrium: episode return -13.6 @ 47 steps vs -1.6 @ 6 steps, and
    # a2's mean episode length collapses 45 -> 6 at iter ~1400. With this
    # flag off the contact probe shows two clean 212 N feet and the reward
    # poison is gone. Precedent: BHL ships self_collisions=False officially;
    # Unitree ships hand-pruned collision geometry, X2 does not.
    self_collisions=False,
    action_scale=0.25,  # upstream velocity-base convention, same as G1
    # Free-joint name, read from X2-Ultra.xml (body "pelvis").
    root_joint_name="floating_base_joint",
)
