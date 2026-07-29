# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Numeric values below are transcribed from the official Berkeley Humanoid
# Lite training stack (MIT-licensed, HybridRobotics); each field cites its
# upstream file and line numbers.

"""Berkeley Humanoid Lite profile (humanoid configuration, 22 actuated joints).

Configuration choice: the upstream project ships TWO robots -- ``biped``
(legs only, 12 DoF) and ``humanoid`` (legs + two 5-DoF arms, 22 DoF). This
profile registers the **humanoid**: it is the same morphology class as the
other launch robots (G1, X2 -- full humanoids with arms), which is what makes
the cross-robot benchmark comparable. The armless biped is deliberately not
registered.

Value provenance (established 2026-07-29, CPU-only):

- Official Isaac Lab articulation config: ``berkeley_humanoid_lite_assets/
  robots/berkeley_humanoid_lite.py`` inside the pinned assets repo
  (HybridRobotics/Berkeley-Humanoid-Lite-Assets @ fc90fedd, pinned in
  ``assets/registry.py``; fetched copy at ``assets/berkeley/humanoid_lite/``).
  ``HUMANOID_LITE_CFG`` starts at L106. UNLIKE X2, nothing here is derived:
  gains/limits/armature are the exact values the upstream project trained
  with and ran on the physical robot.
- Official training environment: HybridRobotics/Berkeley-Humanoid-Lite
  @ 984741a3 (repo HEAD 2026-07-29, last commit 2026-03-10; read-only shallow
  clone, not vendored), files ``.../velocity/config/humanoid/env_cfg.py``
  ("env_cfg.py" below) and ``environments/mujoco.py``, plus the deploy config
  ``configs/policy_humanoid.yaml`` ("policy yaml" below).
- Model facts (joint table, free-joint name, foot bodies, FK heights): the
  pinned vendor MJCF ``mjcf/berkeley_humanoid_lite.xml`` loaded with MuJoCo
  3.10.0, ``CUDA_VISIBLE_DEVICES=""``. URDF cross-check: all 22 joint names
  and ranges match the MJCF exactly (programmatic comparison; the URDF's
  flat effort/velocity limits 20/15 are export-wide placeholders -- the
  authoritative per-group limits are the Isaac actuator values below, which
  the official deploy loop also enforces in software, policy yaml L88-110).

Known upstream quirks recorded here so nobody "fixes" them silently:

- The CAD export puts the model's root frame at GROUND level between the
  feet (base body inertial center sits at z=+0.675 in the root frame). Hence
  the official spawn position is exactly (0, 0, 0) and there is no usable
  root-height signal: see ``target_base_height_m`` below.
- The MJCF declares uniform armature 0.005 / frictionloss 0.1 (compiler
  class default), while the official Isaac training uses per-group armature
  0.007 / 0.002 and no frictionloss. This profile records the TRAINING
  values (they define the policy's world); the MJCF keeps its own numbers.
- The MJCF motors carry forcerange +-20 N*m, but training and the official
  deploy loop both clamp at 4 / 6 N*m (policy yaml L88-110); 4 / 6 are the
  real limits.
"""

from yanshi_rl_lab.robots.profile import ActuatorGroup, RobotProfile

# Root of the fetched assets repo snapshot, relative to the assets root.
_BHL_DATA = "berkeley/humanoid_lite/data/robots/berkeley_humanoid/berkeley_humanoid_lite"

BHL_PROFILE = RobotProfile(
    vendor="berkeley",
    model="humanoid_lite",
    # -- asset locations (relative to the assets root) --------------------
    urdf=f"{_BHL_DATA}/urdf/berkeley_humanoid_lite.urdf",
    # NOTE: the robot MJCF compiles only through the assets/merged symlink
    # that assets/fetch.py creates (declared in assets/registry.py) -- the
    # upstream repo references a mesh layout it does not ship.
    mjcf=f"{_BHL_DATA}/mjcf/berkeley_humanoid_lite.xml",
    scene_mjcf=f"{_BHL_DATA}/mjcf/bhl_scene.xml",
    # Spawn from the official USD -- the exact asset the upstream project
    # trained with (HUMANOID_LITE_CFG usd_path, berkeley_humanoid_lite.py
    # L108), not a URDF re-conversion.
    usd=f"{_BHL_DATA}/usd/berkeley_humanoid_lite.usd",
    # -- embodiment -------------------------------------------------------
    # Official init pos is exactly (0, 0, 0): berkeley_humanoid_lite.py L124
    # (and the deploy default_base_position, policy yaml L111-114). This is
    # NOT a mistake -- the root frame sits at ground level (module
    # docstring), and FK at the default pose puts the lowest sole point at
    # z = +0.0393 m ABOVE the root origin (MuJoCo 3.10, measured
    # 2026-07-29), so spawning at 0 already leaves a ~3.9 cm drop clearance
    # (same order as the 1.6 cm the G1 profile carries).
    spawn_height_m=0.0,
    # Official default pose, berkeley_humanoid_lite.py L123-148: bent knees
    # on the legs, arms at zero. Regexes cover left+right exactly like the
    # upstream per-side entries (values identical per side).
    default_joint_pos={
        "leg_.*_hip_pitch_joint": -0.2,
        "leg_.*_knee_pitch_joint": 0.4,
        "leg_.*_ankle_pitch_joint": -0.3,
    },
    # Actuator groups: verbatim HUMANOID_LITE_CFG "arms"/"legs"/"ankles"
    # (berkeley_humanoid_lite.py L152-190), including the upstream group
    # names and joint regexes. All three groups are ImplicitActuatorCfg
    # upstream -- see pd_mode below.
    actuator_groups={
        "arms": ActuatorGroup(
            # L154-160
            joint_names_expr=[
                "arm_.*_shoulder_pitch_joint",
                "arm_.*_shoulder_roll_joint",
                "arm_.*_shoulder_yaw_joint",
                "arm_.*_elbow_pitch_joint",
                "arm_.*_elbow_roll_joint",
            ],
            effort_limit=4,  # L161
            velocity_limit=10.0,  # L162
            stiffness=10,  # L163
            damping=2,  # L164
            armature=0.002,  # L165
        ),
        "legs": ActuatorGroup(
            # L168-173
            joint_names_expr=[
                "leg_.*_hip_yaw_joint",
                "leg_.*_hip_roll_joint",
                "leg_.*_hip_pitch_joint",
                "leg_.*_knee_pitch_joint",
            ],
            effort_limit=6,  # L174
            velocity_limit=10.0,  # L175
            stiffness=20,  # L176
            damping=2,  # L177
            armature=0.007,  # L178
        ),
        "ankles": ActuatorGroup(
            # L181-184
            joint_names_expr=[
                "leg_.*_ankle_pitch_joint",
                "leg_.*_ankle_roll_joint",
            ],
            effort_limit=6,  # L185
            velocity_limit=10.0,  # L186
            stiffness=20,  # L187
            damping=2,  # L188
            armature=0.002,  # L189
        ),
    },
    # Deploy/policy joint ordering: arms 10 -> legs 12. Three independent
    # official witnesses agree exactly: HUMANOID_LITE_JOINTS (= ARM + LEG,
    # berkeley_humanoid_lite.py L39, used with preserve_order=True for both
    # observations and actions in env_cfg.py L65/L70/L99), the deploy config
    # joint list (policy yaml L19-41), and the vendor MJCF <actuator> block
    # order (verified with MuJoCo 3.10). This IS the real-robot order -- the
    # official firmware consumes exactly this sequence.
    joint_sdk_names=[
        "arm_left_shoulder_pitch_joint",
        "arm_left_shoulder_roll_joint",
        "arm_left_shoulder_yaw_joint",
        "arm_left_elbow_pitch_joint",
        "arm_left_elbow_roll_joint",
        "arm_right_shoulder_pitch_joint",
        "arm_right_shoulder_roll_joint",
        "arm_right_shoulder_yaw_joint",
        "arm_right_elbow_pitch_joint",
        "arm_right_elbow_roll_joint",
        "leg_left_hip_roll_joint",
        "leg_left_hip_yaw_joint",
        "leg_left_hip_pitch_joint",
        "leg_left_knee_pitch_joint",
        "leg_left_ankle_pitch_joint",
        "leg_left_ankle_roll_joint",
        "leg_right_hip_roll_joint",
        "leg_right_hip_yaw_joint",
        "leg_right_hip_pitch_joint",
        "leg_right_knee_pitch_joint",
        "leg_right_ankle_pitch_joint",
        "leg_right_ankle_roll_joint",
    ],
    # PD mode evidence (both sides checked, they DISAGREE upstream):
    # - Training: all actuator groups are Isaac ImplicitActuatorCfg
    #   (berkeley_humanoid_lite.py L153/L167/L180) -> gains act inside the
    #   simulator's implicit joint drive. That defines the physics the
    #   policy was optimized against, so the profile records "implicit".
    # - Official deploy (environments/mujoco.py L183-201): the loop computes
    #   tau = kp*(q_target - q) + kd*(-qd) itself and writes torques -- an
    #   explicit-PD deploy of an implicitly-trained policy. It survives
    #   upstream because their MuJoCo runs at physics_dt = 0.0005 s (policy
    #   yaml L14, 2 kHz substeps), which keeps explicit damping stable; it is
    #   still the training/deploy mismatch our G1 sim2sim debugging
    #   identified as a bug class ("kd as external torque" velocity ringing).
    #   Our deploy runtime therefore reproduces the TRAINING-side physics
    #   (implicit posture: kd -> dof_damping, the mode the M2 regression
    #   validated bit-for-bit on G1) instead of copying the official
    #   deploy-side mismatch.
    pd_mode="implicit",
    # -- semantic body map (verified against the vendor MJCF body tree) ----
    # The torso body is literally named "base" (4.44 kg, carries the IMU
    # mount); official env terms target body_names="base" (env_cfg.py
    # L227/L250/L303).
    base_link="base",
    # Official foot sensor bodies ".*_ankle_roll" (env_cfg.py L169/L178-179);
    # MJCF confirms the sole boxes live on leg_left/right_ankle_roll.
    feet_bodies=".*_ankle_roll",
    # Official undesired-contact list (env_cfg.py L188) expressed as one
    # regex: base, hip, knee, shoulder and elbow bodies; hands/feet excluded.
    undesired_contact_bodies="(base|.*_hip_.*|.*_knee_.*|.*_shoulder_.*|.*_elbow_.*)",
    # Official deviation groups mapped onto the base's three slots:
    # - legs slot = official joint_deviation_hip (hip yaw+roll, env_cfg.py
    #   L195-199) PLUS official joint_deviation_ankle_roll (L200-204) -- the
    #   base has no separate ankle slot, and dropping the ankle-roll penalty
    #   would silently deviate from the official reward structure.
    leg_deviation_joints=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"],
    # - arms slot = official joint_deviation_shoulder (L205-209) +
    #   joint_deviation_elbow (L210-214).
    arm_deviation_joints=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_pitch_joint",
        ".*_elbow_roll_joint",
    ],
    # - BHL has no waist joints at all (22 = 2x5 arms + 2x6 legs); empty
    #   list -> the base drops the waist-deviation term (official config
    #   likewise has no waist term).
    waist_deviation_joints=[],
    # -- task-derivable facts ---------------------------------------------
    # None, deliberately: the root frame sits at ground level (module
    # docstring). Measured standing root height is -0.039 m (soles rest on
    # the ground 0.039 m ABOVE the root origin), and a fallen pose can read
    # HIGHER than standing -- root height carries no usable information.
    # The official BHL training config accordingly has NO base-height reward
    # (env_cfg.py RewardsCfg L104-214) and NO height termination
    # (TerminationsCfg L217-228: time_out + bad_orientation only). None
    # makes the shared base drop both terms, reproducing exactly that.
    target_base_height_m=None,
    min_base_height_m=None,
    # Official: enabled_self_collisions=False (berkeley_humanoid_lite.py
    # L120) -- unlike G1/X2. Transcribed, not judged.
    self_collisions=False,
    action_scale=0.25,  # official JointPositionAction scale, env_cfg.py L98
    # Free-joint name in the vendor MJCF (body "base").
    root_joint_name="base_freejoint",
)
