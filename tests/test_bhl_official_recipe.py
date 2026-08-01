# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""BHL attempt-3 official-recipe transcription (Jeff-approved 2026-07-31).

Pins the wholesale transcription of the vendor humanoid config
(HybridRobotics/Berkeley-Humanoid-Lite @ 984741a3) into the BHL overlay:
command profile, reward table, events, observations, termination,
decimation, runner entry point -- plus the runtime semantics of the two MDP
functions the recipe relies on (``feet_air_time_positive_biped`` from the
Isaac Lab velocity task package, identical to the vendor's vendored copy;
``randomize_joint_default_pos`` ported to ``yanshi_rl_lab.mdp.events``).

Guard pattern mirrors test_s3e2_rough_recipe.py: on boxes without the Isaac
Lab stack the module SKIPS; in the training environment everything executes.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub


def _import_or_skip_clean():
    """Import the config stack, skipping cleanly where it is unavailable
    (evict half-initialized modules on failure -- see test_s3e2_rough_recipe.py
    for why. The except catches ModuleNotFoundError AND RuntimeError, so the
    import order inside the try does not affect the skip decision; ruff's
    isort keeps this block sorted.)"""
    before = set(sys.modules)
    try:
        import gymnasium as gym
        import torch

        import isaaclab.terrains as _tg  # noqa: F401  pxr-chain canary

        bhl_module = importlib.import_module(
            "yanshi_rl_lab.tasks.locomotion.velocity.config.berkeley.humanoid_lite"
        )
        bhl_agents = importlib.import_module(
            "yanshi_rl_lab.tasks.locomotion.velocity.config.berkeley.humanoid_lite.agents"
        )
        yanshi_events = importlib.import_module("yanshi_rl_lab.mdp.events")
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
            feet_air_time_positive_biped,
        )

        return gym, torch, bhl_module, bhl_agents, yanshi_events, feet_air_time_positive_biped
    except (ModuleNotFoundError, RuntimeError) as exc:
        for name in tuple(sys.modules):
            if name not in before:
                del sys.modules[name]
        detail = getattr(exc, "name", None) or str(exc).splitlines()[0]
        pytest.skip(
            f"BHL recipe tests need the Isaac Lab stack ({detail}); run in the training env",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def stack():
    return _import_or_skip_clean()


BHL_TASK = "Yanshi-Velocity-Flat-Berkeley-Humanoid-Lite-v0"


def _env_cfg_instance(gym, bhl_module, kind="env_cfg_entry_point"):
    entry = gym.registry[BHL_TASK].kwargs[kind]
    module_name, cls_name = entry.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()


# -- registration + runner entry point ----------------------------------------


def test_task_registered_with_official_runner(stack):
    gym, *_ = stack
    assert BHL_TASK in gym.registry
    ep = gym.registry[BHL_TASK].kwargs["rsl_rl_cfg_entry_point"]
    assert ep.endswith("humanoid_lite.agents:BerkeleyHumanoidLitePPORunnerCfg")


def test_runner_official_hyperparameters(stack):
    _, _, _, bhl_agents, *_ = stack
    cfg = bhl_agents.BerkeleyHumanoidLitePPORunnerCfg()
    # official rsl_rl_ppo_cfg.py @984741a3
    assert cfg.num_steps_per_env == 24
    assert cfg.algorithm.entropy_coef == 0.008
    assert cfg.policy.actor_hidden_dims == [256, 128, 128]
    assert cfg.policy.critic_hidden_dims == [256, 128, 128]
    assert cfg.policy.init_noise_std == 1.0
    assert cfg.empirical_normalization is False
    assert cfg.algorithm.learning_rate == 1.0e-3
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.desired_kl == 0.01
    # documented fleet deviation: 10000, not the official 6000
    assert cfg.max_iterations == 10000


# -- transcribed MDP (train cfg) ----------------------------------------------


def test_official_command_profile(stack):
    gym, _, bhl_module, *_ = stack
    cfg = _env_cfg_instance(gym, bhl_module)
    cmd = cfg.commands.base_velocity
    assert cmd.heading_command is True
    assert cmd.heading_control_stiffness == 0.5
    assert cmd.rel_heading_envs == 1.0
    assert cmd.rel_standing_envs == 0.02
    assert cmd.resampling_time_range == (10.0, 10.0)
    for r in (cmd.ranges, cmd.limit_ranges):
        assert r.lin_vel_x == (-1.0, 1.0)
        assert r.lin_vel_y == (-0.5, 0.5)
        assert r.ang_vel_z == (-1.5, 1.5)
        assert r.heading == (-3.141592653589793, 3.141592653589793)
    assert cfg.curriculum.lin_vel_cmd_levels is None


def test_official_reward_table(stack):
    gym, _, bhl_module, *_ = stack
    cfg = _env_cfg_instance(gym, bhl_module)
    rew = cfg.rewards
    # task terms: official weights (swapped vs the fleet table)
    assert rew.track_lin_vel_xy.weight == 2.0
    assert rew.track_ang_vel_z.weight == 1.0
    assert rew.track_ang_vel_z.func.__name__ == "track_ang_vel_z_world_exp"
    assert rew.track_lin_vel_xy.func.__name__ == "track_lin_vel_xy_yaw_frame_exp"
    assert rew.track_lin_vel_xy.params["std"] == 0.5
    assert rew.track_ang_vel_z.params["std"] == 0.5
    # removed terms (absent from the official table)
    for name in ("alive", "energy", "joint_vel", "gait", "feet_clearance",
                 "air_time_variance", "base_height", "joint_deviation_waists"):
        assert getattr(rew, name) is None, name
    # transcribed weights
    assert rew.termination_penalty.weight == -10.0
    assert rew.base_linear_velocity.weight == -0.1
    assert rew.base_angular_velocity.weight == -0.05
    assert rew.action_rate.weight == -0.001
    assert rew.dof_torques.weight == -2.0e-5
    assert rew.joint_acc.weight == -1.0e-7
    assert rew.dof_pos_limits.weight == -1.0
    assert rew.flat_orientation_l2.weight == -1.0
    assert rew.feet_slide.weight == -0.1
    assert rew.joint_deviation_arms.weight == -1.0
    assert rew.joint_deviation_legs.weight == -1.0
    assert rew.undesired_contacts.weight == -1.0
    # the anti-statue term
    assert rew.feet_air_time.weight == 2.0
    assert rew.feet_air_time.func.__name__ == "feet_air_time_positive_biped"
    assert rew.feet_air_time.params["threshold"] == 0.5
    assert rew.feet_air_time.params["sensor_cfg"].body_names == ".*_ankle_roll"


def test_official_events(stack):
    gym, _, bhl_module, *_ = stack
    cfg = _env_cfg_instance(gym, bhl_module)
    ev = cfg.events
    assert ev.physics_material.params["static_friction_range"] == (0.4, 1.2)
    assert ev.add_base_mass.params["mass_distribution_params"] == (-1.0, 2.0)
    assert ev.reset_base.params["velocity_range"] == {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0),
        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
    }
    assert ev.reset_robot_joints.params["position_range"] == (0.5, 1.5)
    assert ev.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert ev.base_external_force_torque.params["force_range"] == (-2.0, 2.0)
    assert ev.base_external_force_torque.params["torque_range"] == (-2.0, 2.0)
    assert ev.push_robot is None
    assert ev.add_all_joint_default_pos.params["pos_distribution_params"] == (-0.05, 0.05)
    assert ev.scale_all_actuator_torque_constant.params["stiffness_distribution_params"] == (0.8, 1.2)
    assert ev.scale_all_actuator_torque_constant.params["damping_distribution_params"] == (0.8, 1.2)


def test_official_observations_termination_decimation(stack):
    gym, _, bhl_module, *_ = stack
    cfg = _env_cfg_instance(gym, bhl_module)
    pol, cri = cfg.observations.policy, cfg.observations.critic
    assert pol.history_length == 0
    assert cri.history_length == 0
    assert pol.base_ang_vel.scale is None
    assert (pol.base_ang_vel.noise.n_min, pol.base_ang_vel.noise.n_max) == (-0.3, 0.3)
    assert (pol.joint_pos_rel.noise.n_min, pol.joint_pos_rel.noise.n_max) == (-0.05, 0.05)
    assert pol.joint_vel_rel.scale is None
    assert (pol.joint_vel_rel.noise.n_min, pol.joint_vel_rel.noise.n_max) == (-2.0, 2.0)
    assert cri.base_ang_vel.scale is None
    assert cri.joint_vel_rel.scale is None
    assert cfg.terminations.bad_orientation.params["limit_angle"] == 0.78
    assert cfg.decimation == 8
    assert cfg.sim.render_interval == 8
    assert cfg.scene.height_scanner.update_period == 8 * cfg.sim.dt


def test_play_cfg_reverts_to_direct_wz(stack):
    gym, _, bhl_module, *_ = stack
    cfg = _env_cfg_instance(gym, bhl_module, kind="play_env_cfg_entry_point")
    cmd = cfg.commands.base_velocity
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.limit_ranges.heading is None


# -- runtime semantics of the two MDP functions the recipe relies on ----------


def _fake_reward_env(torch, contact_time, air_time, cmd_xy):
    """Minimal fakes for feet_air_time_positive_biped (two feet)."""
    from isaaclab.managers import SceneEntityCfg

    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            current_air_time=torch.tensor([air_time], dtype=torch.float32),
            current_contact_time=torch.tensor([contact_time], dtype=torch.float32),
        )
    )
    env = types.SimpleNamespace(
        scene=types.SimpleNamespace(sensors={"contact_forces": sensor}),
        command_manager=types.SimpleNamespace(
            get_command=lambda name: torch.tensor([cmd_xy], dtype=torch.float32)
        ),
    )
    cfg = SceneEntityCfg("contact_forces")
    cfg.body_ids = [0, 1]
    return env, cfg


def test_feet_air_time_positive_biped_semantics(stack):
    _, torch, _, _, _, feet_air_time_positive_biped = stack
    # single stance (foot 2 airborne: contact_time 0), swing air time 0.3 s -> 0.3
    env, scfg = _fake_reward_env(torch, contact_time=[1.2, 0.0], air_time=[0.0, 0.3], cmd_xy=[0.5, 0.0, 0.0])
    got = feet_air_time_positive_biped(env, "base_velocity", 0.5, scfg)
    assert torch.allclose(got, torch.tensor([0.3]), atol=1e-6)
    # single stance, air time 0.9 s -> clamped to the 0.5 threshold
    env, scfg = _fake_reward_env(torch, contact_time=[1.2, 0.0], air_time=[0.0, 0.9], cmd_xy=[0.5, 0.0, 0.0])
    got = feet_air_time_positive_biped(env, "base_velocity", 0.5, scfg)
    assert torch.allclose(got, torch.tensor([0.5]), atol=1e-6)
    # double stance -> zero (the anti-statue property)
    env, scfg = _fake_reward_env(torch, contact_time=[1.2, 0.8], air_time=[0.0, 0.0], cmd_xy=[0.5, 0.0, 0.0])
    got = feet_air_time_positive_biped(env, "base_velocity", 0.5, scfg)
    assert torch.allclose(got, torch.tensor([0.0]), atol=1e-6)
    # near-zero command -> gated to zero even mid-step
    env, scfg = _fake_reward_env(torch, contact_time=[1.2, 0.0], air_time=[0.0, 0.3], cmd_xy=[0.05, 0.0, 0.0])
    got = feet_air_time_positive_biped(env, "base_velocity", 0.5, scfg)
    assert torch.allclose(got, torch.tensor([0.0]), atol=1e-6)


def test_randomize_joint_default_pos_semantics(stack):
    _, torch, _, _, yanshi_events, _ = stack
    from isaaclab.managers import SceneEntityCfg

    torch.manual_seed(0)
    defaults = torch.zeros(4, 3)
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(default_joint_pos=defaults.clone()), device="cpu"
    )
    env = types.SimpleNamespace(scene={"robot": asset})
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.joint_ids = slice(None)
    env_ids = torch.tensor([1, 2])
    yanshi_events.randomize_joint_default_pos(
        env, env_ids, asset_cfg, pos_distribution_params=(-0.05, 0.05), operation="add"
    )
    out = asset.data.default_joint_pos
    # untouched envs stay exactly at default
    assert torch.all(out[0] == 0.0) and torch.all(out[3] == 0.0)
    # touched envs moved, within the +-0.05 add band, and actually randomized
    delta = out - defaults
    assert torch.all(delta[1].abs() <= 0.05) and torch.all(delta[2].abs() <= 0.05)
    assert torch.any(delta[1] != 0.0) and torch.any(delta[2] != 0.0)
