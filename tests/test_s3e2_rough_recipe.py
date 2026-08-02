# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""S3-实验2 发1 rough recipe (experiment card §2.5, frozen 2026-07-31).

Covers the G1 rough-task wiring: E1 run-2 recipe (official rough heading-mix
command profile + official rough reward deltas) plus THE single variable
``track_heading_exp``, with the flat task kept byte-for-byte at M1 parity
behavior (T0 rule). Also covers the heading-target pinning command term and
the per-terrain runner entry point (rough entropy 0.008).

Everything here imports the Isaac Lab config stack (pxr chain), so on pure
deploy/CI boxes the module SKIPS, mirroring test_terrain_presets.py's guard;
in the training environment the tests execute for real.
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
    (evict half-initialized modules first -- the pre-app pollution trap in
    CLAUDE.md's pitfall log, guarded by test_*_profile.py).

    ``isaaclab.terrains`` goes in FIRST as the pxr-chain canary: elsewhere in
    the suite some guarded test may have evicted ``torch`` from sys.modules,
    and importing our stack via ``mdp.commands`` (plain ``import torch``)
    then dies with ``RuntimeError: ... already has a docstring`` instead of a
    clean ModuleNotFoundError. The canary fails fast with the expected error
    type; RuntimeError is caught defensively for the same trap.
    """
    before = set(sys.modules)
    try:
        import gymnasium as gym

        import isaaclab.terrains as _tg  # noqa: F401  pxr-chain canary, must stay first

        g1_module = importlib.import_module(
            "yanshi_rl_lab.tasks.locomotion.velocity.config.unitree.g1"
        )
        commands = importlib.import_module("yanshi_rl_lab.mdp.commands")
        rewards = importlib.import_module("yanshi_rl_lab.mdp.rewards")
        agents = importlib.import_module("yanshi_rl_lab.tasks.locomotion.agents")
        return gym, g1_module, commands, rewards, agents
    except (ModuleNotFoundError, RuntimeError) as exc:
        for name in tuple(sys.modules):
            if name not in before:
                del sys.modules[name]
        detail = getattr(exc, "name", None) or str(exc).splitlines()[0]
        pytest.skip(
            f"recipe tests need the Isaac Lab stack ({detail}); run in the training env",
            allow_module_level=True,
        )


@pytest.fixture(scope="module")
def stack():
    return _import_or_skip_clean()


def _env_cfg_instance(gym, g1_module, task_id: str, kind: str = "env_cfg_entry_point"):
    spec = gym.registry[task_id]
    entry = spec.kwargs[kind]
    module_name, cls_name = entry.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)()


FLAT_TASK = "Yanshi-Velocity-Flat-Unitree-G1-v0"
ROUGH_TASK = "Yanshi-Velocity-Rough-Nostairs-Unitree-G1-v0"


# -- registration + runner entry points --------------------------------------


def test_both_tasks_registered(stack):
    gym, *_ = stack
    assert FLAT_TASK in gym.registry
    assert ROUGH_TASK in gym.registry


def test_runner_entry_points_per_terrain(stack):
    gym, *_ = stack
    flat_ep = gym.registry[FLAT_TASK].kwargs["rsl_rl_cfg_entry_point"]
    rough_ep = gym.registry[ROUGH_TASK].kwargs["rsl_rl_cfg_entry_point"]
    assert flat_ep.endswith(":YanshiPPORunnerCfg")
    assert rough_ep.endswith(":YanshiRoughPPORunnerCfg")


def test_rough_runner_entropy_delta(stack):
    *_, agents = stack
    assert agents.YanshiPPORunnerCfg().algorithm.entropy_coef == 0.01
    assert agents.YanshiRoughPPORunnerCfg().algorithm.entropy_coef == 0.008


# -- rough recipe (train cfg) -------------------------------------------------


def test_rough_command_profile(stack):
    gym, g1_module, commands, *_ = stack
    cfg = _env_cfg_instance(gym, g1_module, ROUGH_TASK)
    cmd = cfg.commands.base_velocity
    assert cmd.heading_command is True
    assert cmd.heading_control_stiffness == 0.5
    assert cmd.rel_heading_envs == 0.5
    for r in (cmd.ranges, cmd.limit_ranges):
        assert r.lin_vel_x == (0.0, 1.0)
        assert r.lin_vel_y == (0.0, 0.0)
        assert r.ang_vel_z == (-1.0, 1.0)
        assert r.heading == (-3.141592653589793, 3.141592653589793)
    # heading-pinning term wired in (dataclass carry-over, class swapped)
    assert cmd.class_type is commands.HeadingPinnedVelocityCommand


def test_rough_reward_deltas_and_single_variable(stack):
    gym, g1_module, _, rewards, _ = stack
    cfg = _env_cfg_instance(gym, g1_module, ROUGH_TASK)
    assert cfg.rewards.termination_penalty.weight == -200.0
    assert cfg.rewards.base_linear_velocity.weight == 0.0
    assert cfg.rewards.flat_orientation_l2.weight == -1.0
    assert cfg.rewards.base_height.weight == 0.0
    th = cfg.rewards.track_heading_exp
    assert th.func is rewards.track_heading_exp
    assert th.weight == 0.25  # S3E2-EXP-2 冻结值（EXP-1 为 0.5）
    assert th.params == {"command_name": "base_velocity", "std": 0.5}
    assert cfg.terminations.base_height is None


def test_rough_play_reverts_to_direct_wz(stack):
    gym, g1_module, *_ = stack
    cfg = _env_cfg_instance(gym, g1_module, ROUGH_TASK, kind="play_env_cfg_entry_point")
    assert cfg.commands.base_velocity.heading_command is False
    assert cfg.commands.base_velocity.ranges.heading is None
    assert cfg.commands.base_velocity.limit_ranges.heading is None


# -- T0: flat task unchanged ---------------------------------------------------


def test_flat_task_keeps_parity_profile(stack):
    gym, g1_module, commands, *_ = stack
    cfg = _env_cfg_instance(gym, g1_module, FLAT_TASK)
    cmd = cfg.commands.base_velocity
    assert cmd.heading_command is False
    for r in (cmd.ranges, cmd.limit_ranges):
        assert r.lin_vel_x == (-0.5, 1.0)
        assert r.lin_vel_y == (-0.3, 0.3)
        assert r.ang_vel_z == (-0.8, 0.8)
    # deadband (parity) still wired, no heading pinning, no heading reward
    assert cmd.class_type is commands.DeadbandVelocityCommand
    assert not hasattr(cfg.rewards, "track_heading_exp")


# -- reward math ---------------------------------------------------------------


def test_track_heading_exp_math(stack):
    from types import SimpleNamespace

    import torch

    *_, rewards, _agents = stack

    heading_target = torch.tensor([0.0, 1.0, 3.0])
    heading_w = torch.tensor([0.0, 0.0, -3.0])
    term = SimpleNamespace(
        heading_target=heading_target, cfg=SimpleNamespace(asset_name="robot")
    )
    asset = SimpleNamespace(data=SimpleNamespace(heading_w=heading_w))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: term),
        scene={"robot": asset},
    )
    out = rewards.track_heading_exp(env, std=0.5)
    # err = 0 -> 1.0 ; err = 1 -> exp(-4) ; wrap_to_pi(-3 - 3 = -6 -> 0.283) -> exp(-(0.283/0.5)^2)
    import math

    assert out[0].item() == pytest.approx(1.0, abs=1e-6)
    assert out[1].item() == pytest.approx(math.exp(-4.0), abs=1e-6)
    wrapped = (-6.0 + math.pi) % (2 * math.pi) - math.pi
    assert out[2].item() == pytest.approx(math.exp(-((wrapped / 0.5) ** 2)), abs=1e-4)
