# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree-g1-locomotion ``sim2sim/runtime.py`` (MIT, same author)
# with every robot-specific constant parameterized through the v2 contract:
# joint ordering, gains, scales, timing, observation-term order, foot bodies
# and the root joint name are all read from the contract at runtime; the
# explicit-PD branch is ported from anima-zero ``world/sim-house-nav/sim.py``
# ``_setup_pd``/torque path (MIT, same author; empirically validated on Go2).

"""Contract-driven MuJoCo closed-loop deployment runtime (sim2sim).

House rule: training is not done until sim2sim passes its gates. This module
is the shared base for the deploy script, the acceptance gates and teleop --
one implementation of: read contract, assemble the real-robot-format
observation, map joints by name, set up PD actuation, drive torques, step
physics.

Rendering backend: this module imports ``mujoco`` at import time. Callers
that want offscreen video must set MUJOCO_GL themselves BEFORE importing;
callers that must stay off the GPU (the default in this repo's CPU-only
verification) must NOT set MUJOCO_GL=egl and must not construct a Recorder.

** PD actuation = faithful replica of the training-side actuator model **
(hard-won 2026-07-24 lesson from the predecessor stack; the mode is a
per-robot fact carried by the contract's ``pd_mode``):

- ``implicit`` (Isaac ImplicitActuatorCfg, e.g. Unitree G1): kp/kd are
  applied inside PhysX's implicit solver, unconditionally stable. To
  replicate: write kd into MuJoCo ``dof_damping`` (semi-implicit
  integration) and send ONLY the kp term as torque.
  Applying ``-kd*qd`` as an external torque instead makes joint velocities
  ring at high frequency from step 1 on kp=100-200 leg joints (measured
  jvel step-L2 ~38 vs ~3; positions track fine) -> the policy receives
  garbage jvel observations -> falls in ~0.8-1.4 s. That ringing was the
  true root cause of the predecessor's "falls ~1 s after start" sim2sim
  failure -- not policy robustness. Do not rediscover this.
- ``explicit`` (e.g. Unitree Go2 in alice-house): the training-side kd IS
  an explicit torque term, so ``dof_damping`` is zeroed and the loop
  computes ``tau = kp*(q_target - q) - kd*qd`` itself.

Both modes zero MuJoCo ``dof_frictionloss``: vendor MJCFs ship Coulomb
friction defaults for hand-written controllers, but the training environment
has none -- leaving it in is like walking through syrup.
"""

from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

from yanshi_rl_lab.deploy.contract import CONTRACT_PD_MODES, ContractV2

# Fall detection: tilt threshold in radians, aligned with the training-side
# ``bad_orientation`` termination (limit_angle=0.8 in the velocity base cfg,
# itself upstream-verbatim). Kept here as the deploy-side mirror of that
# training fact, same value as the predecessor stack's gate scripts.
FALL_TILT_RAD = 0.8

# Offscreen render size for the optional Recorder (predecessor stack values).
RENDER_H, RENDER_W = 720, 1280


class RealFmtObs:
    """Real-robot-format observation assembly.

    Per-term deques of length H; flattening order = the contract's obs_terms
    order; within each term, frames are ordered old -> new. ``reset`` fills
    the whole history with the first frame (NOT zeros -- matches Isaac's
    post-reset history semantics, locked against the golden trace on the
    predecessor stack).
    """

    def __init__(self, contract: ContractV2):
        self.c = contract
        self.term_names = [t.name for t in contract.obs_terms]
        self._hist = {
            t.name: deque(maxlen=int(t.history_length)) for t in contract.obs_terms
        }

    def reset(self, frame: dict) -> None:
        for t in self.c.obs_terms:
            buf = self._hist[t.name]
            buf.clear()
            for _ in range(int(t.history_length)):
                buf.append(frame[t.name])

    def push_build(self, frame: dict) -> np.ndarray:
        for name in self.term_names:
            self._hist[name].append(frame[name])
        vec = np.concatenate(
            [np.concatenate(list(self._hist[name])) for name in self.term_names]
        ).astype(np.float32)
        if self.c.obs_dim is not None and vec.shape[0] != self.c.obs_dim:
            raise ValueError(
                f"Assembled observation has {vec.shape[0]} dims, contract says "
                f"{self.c.obs_dim}. Layout mismatch -- refusing to run."
            )
        return vec


def _term_scale(spec) -> np.ndarray | float:
    """Scale factor of one obs term (scalar or per-element array)."""
    if spec.scale is None:
        return 1.0
    if isinstance(spec.scale, (int, float)):
        return float(spec.scale)
    return np.asarray(spec.scale, np.float64)


def make_frame(
    contract: ContractV2,
    ang_vel_b,
    grav_b,
    cmd,
    jpos_policy,
    jvel_policy,
    prev_action,
) -> dict:
    """Build one scaled observation frame keyed by term name.

    Only the six real-robot-format terms are implemented. An unknown term in
    the contract raises instead of silently zero-filling (predecessor-stack
    discipline: a wrong layout must fail loudly, not walk two steps and
    fall).
    """
    raw = {
        "base_ang_vel": np.asarray(ang_vel_b, np.float64),
        "projected_gravity": np.asarray(grav_b, np.float64),
        "velocity_commands": np.asarray(cmd, np.float64),
        "joint_pos_rel": np.asarray(jpos_policy, np.float64)
        - np.asarray(contract.default_joint_pos, np.float64),
        "joint_vel_rel": np.asarray(jvel_policy, np.float64),
        "last_action": np.asarray(prev_action, np.float64),
    }
    frame = {}
    for spec in contract.obs_terms:
        if spec.name not in raw:
            raise NotImplementedError(
                f"Observation term {spec.name!r} is in the contract but this runtime "
                f"has no builder for it (known terms: {sorted(raw)})."
            )
        frame[spec.name] = raw[spec.name] * _term_scale(spec)
    return frame


class Policy:
    """ONNX policy, deterministic CPU inference."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.path = path
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        # Static input dim (when the model declares one) must match the
        # contract before we waste a run on a mismatched pair.
        self.obs_dim = in_shape[-1] if isinstance(in_shape[-1], int) else None

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.in_name: vec[None, :]})[0].reshape(-1).astype(np.float64)


class MujocoRig:
    """MuJoCo model + by-name joint mapping + contract-driven PD setup."""

    def __init__(self, contract: ContractV2, scene_path: str):
        contract.require_runtime_fields()
        self.c = contract
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.model.opt.timestep = float(contract.timing.physics_dt_s)
        self.model.vis.global_.offwidth = RENDER_W
        self.model.vis.global_.offheight = RENDER_H
        self.data = mujoco.MjData(self.model)

        # By-name mapping: policy order i <-> MuJoCo qpos/qvel/actuator index.
        # Asserting on a missing name catches "wrong scene for this contract"
        # immediately instead of producing silently-permuted observations.
        qadr, dadr, actid = [], [], []
        for n in contract.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            assert jid >= 0, f"MuJoCo model has no joint {n!r} (scene: {scene_path})"
            qadr.append(int(self.model.jnt_qposadr[jid]))
            dadr.append(int(self.model.jnt_dofadr[jid]))
            act = next(
                (a for a in range(self.model.nu) if int(self.model.actuator_trnid[a][0]) == jid),
                None,
            )
            assert act is not None, f"MuJoCo model has no actuator on joint {n!r}"
            actid.append(act)
        self.qadr, self.dadr, self.actid = map(np.array, (qadr, dadr, actid))

        gains = contract.gains()
        missing_gain = [n for n in contract.joint_names if gains[n][0] is None or gains[n][1] is None]
        if missing_gain:
            raise ValueError(
                f"Contract has no kp/kd for joints {missing_gain}; the legacy source "
                "did not record them -- refusing to guess."
            )
        self.kp = np.array([gains[n][0] for n in contract.joint_names])
        self.kd = np.array([gains[n][1] for n in contract.joint_names])
        # Flat effort limit (training ImplicitActuator semantics). A missing
        # limit means the legacy source never recorded one: mirror the
        # training side's "no additional clamp" by using +inf, and say so.
        eff = [gains[n][2] for n in contract.joint_names]
        if any(e is None for e in eff):
            print(
                "[runtime] WARNING: effort_limit missing for "
                f"{[n for n, e in zip(contract.joint_names, eff) if e is None]}; "
                "torques will be unclamped (contract source did not record limits)."
            )
        self.eff = np.array([math.inf if e is None else float(e) for e in eff])

        # -- PD mode (see module docstring; wrong mode = instant fall) ------
        mode = contract.pd_mode
        if mode not in CONTRACT_PD_MODES:
            raise ValueError(
                f"contract.pd_mode is {mode!r}; must be one of {CONTRACT_PD_MODES}. "
                "This decides how torques are sent -- refusing to guess."
            )
        self.pd_mode = mode
        for i, dof in enumerate(self.dadr):
            # Both modes: clear Coulomb friction (absent in training).
            self.model.dof_frictionloss[dof] = 0.0
            # implicit: kd into dof_damping (semi-implicit, no ringing);
            # explicit: dof_damping zeroed, loop applies -kd*qd itself.
            self.model.dof_damping[dof] = float(self.kd[i]) if mode == "implicit" else 0.0

        # Root (floating base) joint: located by contract name, must be free.
        root_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, contract.root_joint_name
        )
        assert root_jid >= 0, (
            f"MuJoCo model has no joint {contract.root_joint_name!r} "
            f"(contract root_joint_name; scene: {scene_path})"
        )
        if self.model.jnt_type[root_jid] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"Joint {contract.root_joint_name!r} is not a free joint; "
                "root_joint_name must name the floating base."
            )
        self.root_qadr = int(self.model.jnt_qposadr[root_jid])  # 7 numbers: xyz + wxyz quat
        self.root_dadr = int(self.model.jnt_dofadr[root_jid])  # 6 numbers: lin + ang vel
        self.root_bid = int(self.model.jnt_bodyid[root_jid])

    # -------------------------------------------------------------- state
    def reset(self) -> None:
        """Place the robot in the Isaac default standing pose."""
        d = self.data
        d.qpos[:] = 0.0
        r = self.root_qadr
        d.qpos[r : r + 3] = np.asarray(self.c.default_root_pos_w, np.float64)
        d.qpos[r + 3 : r + 7] = (1.0, 0.0, 0.0, 0.0)  # identity orientation (w, x, y, z)
        d.qpos[self.qadr] = np.asarray(self.c.default_joint_pos, np.float64)
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def base_quants(self):
        """(body-frame angular velocity, body-frame gravity direction).

        MuJoCo free joints report qvel angular velocity ALREADY in the body
        frame -- do not rotate it again (golden-trace verified; rotating
        twice only shows up at non-zero yaw and flips the robot).
        """
        r = self.root_qadr
        quat = np.asarray(self.data.qpos[r + 3 : r + 7], np.float64)
        conj = np.empty(4)
        mujoco.mju_negQuat(conj, quat)
        grav = np.empty(3)
        mujoco.mju_rotVecQuat(grav, np.array([0.0, 0.0, -1.0]), conj)
        ang = np.asarray(self.data.qvel[self.root_dadr + 3 : self.root_dadr + 6], np.float64)
        return ang, grav

    @property
    def jpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr]

    @property
    def jvel(self) -> np.ndarray:
        return self.data.qvel[self.dadr]

    @property
    def xy(self):
        r = self.root_qadr
        return float(self.data.qpos[r]), float(self.data.qpos[r + 1])

    @property
    def height(self) -> float:
        return float(self.data.qpos[self.root_qadr + 2])

    @property
    def yaw(self) -> float:
        r = self.root_qadr
        q = self.data.qpos[r + 3 : r + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def drive(self, q_target: np.ndarray) -> None:
        """Torque inner loop for one policy step (contract decimation).

        implicit: send only kp*(q_target - q); kd acts via dof_damping.
        explicit: send kp*(q_target - q) - kd*qd (dof_damping is zero).
        Adding the kd torque in implicit mode would apply damping twice AND
        bring back the ringing -- see module docstring.
        """
        for _ in range(int(self.c.timing.decimation)):
            tau = self.kp * (q_target - self.data.qpos[self.qadr])
            if self.pd_mode == "explicit":
                tau = tau - self.kd * self.data.qvel[self.dadr]
            self.data.ctrl[self.actid] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.model, self.data)


class Odometer:
    """Net displacement / accumulated turn / path length / turning radius.

    Net turn uses per-step wrapped deltas, not last-minus-first yaw (which
    breaks past +/-180 deg). Radius = path_length / radians, NOT net
    displacement: circling shrinks net displacement and would flatter the
    radius.
    """

    # Below this accumulated |yaw| (rad) a radius estimate is numerically
    # meaningless; report infinity instead (predecessor-stack value).
    MIN_YAW_FOR_RADIUS_RAD = 0.2

    def __init__(self, rig: MujocoRig):
        self.rig = rig
        self.reset()

    def reset(self) -> None:
        self.acc_yaw = 0.0
        self.prev_yaw = self.rig.yaw
        self.path_len = 0.0
        self._px, self._py = self.rig.xy
        self._x0, self._y0 = self.rig.xy

    def update(self) -> None:
        yaw = self.rig.yaw
        self.acc_yaw += math.atan2(math.sin(yaw - self.prev_yaw), math.cos(yaw - self.prev_yaw))
        self.prev_yaw = yaw
        cx, cy = self.rig.xy
        self.path_len += math.hypot(cx - self._px, cy - self._py)
        self._px, self._py = cx, cy

    @property
    def net_displacement_m(self) -> float:
        cx, cy = self.rig.xy
        return float(math.hypot(cx - self._x0, cy - self._y0))

    @property
    def turn_deg(self) -> float:
        return math.degrees(self.acc_yaw)

    @property
    def radius_m(self) -> float:
        if abs(self.acc_yaw) <= self.MIN_YAW_FOR_RADIUS_RAD:
            return float("inf")
        return abs(self.path_len / self.acc_yaw)


class FootContactStats:
    """Per-foot ground-contact statistics -- catches "one leg permanently
    airborne" gait hacks that fooled training curves, batch evaluation and
    static frame grabs on the predecessor stack (v0.45 campaign; a human
    watching continuous video finally caught it). Necessary condition only:
    passing these numbers never replaces watching the video.

    Foot bodies come from the contract's ``foot_bodies``; each entry may be
    an exact MuJoCo body name or a regex (``re.fullmatch`` against all body
    names, same matching rule Isaac Lab uses). Supports N feet:
    ``asymmetry`` generalizes to max(frac) - min(frac), which equals the
    two-foot |left - right| definition used by the original gates.
    """

    def __init__(self, rig: MujocoRig):
        import re

        self.rig = rig
        m = rig.model
        patterns = rig.c.foot_bodies
        if not patterns:
            raise ValueError("Contract has no foot_bodies; cannot compute contact stats.")
        all_bodies = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
        ]
        names: list = []
        for pat in patterns:
            if pat in all_bodies:
                matched = [pat]
            else:
                matched = sorted(n for n in all_bodies if n and re.fullmatch(pat, n))
            if not matched:
                raise ValueError(
                    f"foot_bodies entry {pat!r} matches no body in the MuJoCo model."
                )
            names.extend(matched)
        self.names = names
        self.foot_geoms = []
        for n in self.names:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            adr, num = int(m.body_geomadr[bid]), int(m.body_geomnum[bid])
            self.foot_geoms.append(set(range(adr, adr + num)))
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.contact_steps = [0] * len(self.names)

    def update(self) -> None:
        """Call once per policy step. A foot touches iff any of its geoms
        appears in the current contact table."""
        d = self.rig.data
        touched = [False] * len(self.names)
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            for i, geoms in enumerate(self.foot_geoms):
                if g1 in geoms or g2 in geoms:
                    touched[i] = True
        for i, t in enumerate(touched):
            self.contact_steps[i] += int(t)
        self.steps += 1

    @property
    def contact_frac(self) -> list:
        return [c / self.steps if self.steps else 0.0 for c in self.contact_steps]

    @property
    def asymmetry(self) -> float:
        f = self.contact_frac
        if len(f) < 2:
            return 0.0
        return max(f) - min(f)


class DeployLoop:
    """Contract-driven closed loop: assemble obs -> policy inference -> PD
    torques -> step physics; one call to ``step()`` = one policy period.

    The velocity command ``cmd`` is mutable state: gate scripts set it once
    before running, teleop changes it on keypress.
    """

    def __init__(
        self,
        policy_path: str,
        contract: ContractV2,
        scene_path: str,
        physics_dt: float | None = None,
    ):
        self.c = contract
        if physics_dt:
            # Override the physics step; decimation is recomputed so the
            # policy rate stays fixed.
            self.c.timing.decimation = int(round(self.c.timing.policy_dt_s / physics_dt))
            self.c.timing.physics_dt_s = physics_dt
        self.rig = MujocoRig(self.c, scene_path)
        self.policy = Policy(policy_path)
        if self.policy.obs_dim is not None and self.c.obs_dim is not None:
            if self.policy.obs_dim != self.c.obs_dim:
                raise ValueError(
                    f"Policy input dim {self.policy.obs_dim} != contract obs_dim "
                    f"{self.c.obs_dim} -- wrong policy/contract pair."
                )
        self.obs = RealFmtObs(self.c)
        self.odo = Odometer(self.rig)
        self.cmd = np.zeros(3)
        self.prev_action = np.zeros(len(self.c.joint_names))
        self.step_idx = 0
        self.fell_at: float | None = None
        self.tilt = 0.0

    def _frame(self) -> dict:
        ang, grav = self.rig.base_quants()
        return make_frame(
            self.c, ang, grav, self.cmd, self.rig.jpos, self.rig.jvel, self.prev_action
        )

    def reset(self, cmd=None) -> None:
        if cmd is not None:
            self.cmd = np.asarray(cmd, np.float64)
        self.rig.reset()
        self.prev_action = np.zeros(len(self.c.joint_names))
        self.step_idx = 0
        self.fell_at = None
        self.obs.reset(self._frame())
        self.odo.reset()

    def step(self) -> None:
        """Advance one policy step. Fall detection uses the PRE-step gravity
        direction (same convention as the predecessor gate scripts)."""
        ang, grav = self.rig.base_quants()
        vec = self.obs.push_build(
            make_frame(self.c, ang, grav, self.cmd, self.rig.jpos, self.rig.jvel, self.prev_action)
        )
        action = self.policy(vec)
        self.prev_action = action
        # Scalar or per-joint action scale, per the contract.
        scale = (
            float(self.c.action.scale)
            if isinstance(self.c.action.scale, (int, float))
            else np.asarray(self.c.action.scale, np.float64)
        )
        target = np.asarray(self.c.default_joint_pos, np.float64) + scale * action
        self.rig.drive(target)
        self.odo.update()
        self.tilt = float(np.arctan2(np.hypot(grav[0], grav[1]), -grav[2]))
        if self.tilt > FALL_TILT_RAD and self.fell_at is None:
            self.fell_at = self.step_idx * self.c.timing.policy_dt_s
        self.step_idx += 1

    @property
    def elapsed_s(self) -> float:
        return self.step_idx * self.c.timing.policy_dt_s


class Recorder:
    """Follow-camera offscreen recording (mp4). Requires an offscreen GL
    backend (e.g. MUJOCO_GL=egl -- which uses the GPU); NOT used in this
    repo's CPU-only verification, kept for parity with the predecessor
    stack."""

    def __init__(
        self,
        rig: MujocoRig,
        path: str,
        fps: int | None = None,
        distance: float = 3.6,
        azimuth: float = 135.0,
        elevation: float = -12.0,
    ):
        import os

        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.rig = rig
        self.path = path
        self.renderer = mujoco.Renderer(rig.model, RENDER_H, RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.writer = imageio.get_writer(
            path, fps=fps or int(round(1 / rig.c.timing.policy_dt_s)), quality=8
        )

    def capture(self) -> None:
        self.cam.lookat[:] = self.rig.data.xpos[self.rig.root_bid]
        self.renderer.update_scene(self.rig.data, self.cam)
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        self.writer.close()
