# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Terrain presets and per-robot terrain registration.

Covers the ``rough_nostairs`` preset (S3-实验2 §2.4): values transcribed
verbatim from the predecessor stack's S3E1_ROUGH_NOSTAIRS_CFG (the official
Isaac Lab ROUGH_TERRAINS_CFG minus the two stair sub-terrains, renormalized),
the shared generator head staying identical to "flat" (single-variable
discipline), and the G1 registration gaining the rough task WITHOUT losing
the flat one (T0 rule).

The terrain cfg classes come from isaaclab.terrains, whose import chain
needs pxr -- available only after the Isaac Sim app bootstrap. On pure
deploy/CI boxes (where `pytest -q` must stay green, see CLAUDE.md) the whole
module SKIPS with a reason, mirroring test_sim2sim_regression.py's guard
pattern; in the training environment the tests execute for real.
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
    """Import the terrain stack, skipping cleanly where it is unavailable.

    A FAILED import leaves the already-initialized parent packages in
    sys.modules (the pre-app pollution trap in CLAUDE.md's pitfall log,
    guarded by test_*_profile.py::test_no_isaaclab_leak), so on failure we
    evict every module this attempt pulled in before skipping.
    """
    before = set(sys.modules)
    try:
        from yanshi_rl_lab.terrains import presets

        import isaaclab.terrains as tg

        return presets, tg
    except ModuleNotFoundError as exc:
        for name in tuple(sys.modules):
            if name not in before:
                del sys.modules[name]
        pytest.skip(
            f"terrain presets need the Isaac Lab stack (missing {exc.name}); run in the training env",
            allow_module_level=True,
        )


presets, terrain_gen = _import_or_skip_clean()

# Expected values: S3E1_ROUGH_NOSTAIRS_CFG (unitree_rl_lab
# .../robots/g1/29dof/velocity_variants_env_cfg.py), itself a verbatim
# transcription of the official Isaac Lab ROUGH_TERRAINS_CFG sub-terrain
# parameters with the two stairs removed and proportions renormalized
# 0.2:0.2:0.1:0.1 -> 1/3:1/3:1/6:1/6.
_G1_CONFIG_MODULE = "yanshi_rl_lab.tasks.locomotion.velocity.config.unitree.g1"


def test_rough_nostairs_registered_alongside_existing():
    """T0: adding the preset must not drop the existing ones."""
    names = presets.names()
    assert "rough_nostairs" in names
    assert "flat" in names
    assert "rough" in names


def test_rough_nostairs_sub_terrains_verbatim():
    subs = presets.get("rough_nostairs").sub_terrains
    assert set(subs) == {"hf_random_rough", "boxes", "hf_pyramid_slope", "hf_pyramid_slope_inv"}

    rr = subs["hf_random_rough"]
    assert isinstance(rr, terrain_gen.HfRandomUniformTerrainCfg)
    assert rr.proportion == pytest.approx(1 / 3)
    assert rr.noise_range == (0.02, 0.10)
    assert rr.noise_step == 0.02
    assert rr.border_width == 0.25

    boxes = subs["boxes"]
    assert isinstance(boxes, terrain_gen.MeshRandomGridTerrainCfg)
    assert boxes.proportion == pytest.approx(1 / 3)
    assert boxes.grid_width == 0.45
    assert boxes.grid_height_range == (0.05, 0.2)
    assert boxes.platform_width == 2.0

    slope = subs["hf_pyramid_slope"]
    assert isinstance(slope, terrain_gen.HfPyramidSlopedTerrainCfg)
    assert slope.proportion == pytest.approx(1 / 6)
    assert slope.slope_range == (0.0, 0.4)
    assert slope.platform_width == 2.0
    assert slope.border_width == 0.25

    slope_inv = subs["hf_pyramid_slope_inv"]
    assert isinstance(slope_inv, terrain_gen.HfInvertedPyramidSlopedTerrainCfg)
    assert slope_inv.proportion == pytest.approx(1 / 6)
    assert slope_inv.slope_range == (0.0, 0.4)
    assert slope_inv.platform_width == 2.0
    assert slope_inv.border_width == 0.25


def test_rough_nostairs_shares_the_generator_head():
    """Only sub_terrains may differ between terrain names (presets.py rule)."""
    rough = presets.get("rough_nostairs")
    flat = presets.get("flat")
    for attr in (
        "size",
        "border_width",
        "num_rows",
        "num_cols",
        "horizontal_scale",
        "vertical_scale",
        "slope_threshold",
        "difficulty_range",
    ):
        assert getattr(rough, attr) == getattr(flat, attr), f"generator head drifted: {attr}"


def test_get_returns_independent_copies():
    cfg = presets.get("rough_nostairs")
    cfg.sub_terrains["boxes"].proportion = 0.99
    assert presets.get("rough_nostairs").sub_terrains["boxes"].proportion == pytest.approx(1 / 3)


def test_g1_registers_rough_without_losing_flat():
    """Registration smoke: both task IDs present in gym.registry (T0)."""
    gym = importlib.import_module("gymnasium")
    importlib.import_module(_G1_CONFIG_MODULE)
    ids = {spec.id for spec in gym.registry.values()}
    assert "Yanshi-Velocity-Flat-Unitree-G1-v0" in ids
    assert "Yanshi-Velocity-Rough-Nostairs-Unitree-G1-v0" in ids


def test_rough_nostairs_keeps_terrain_curriculum():
    """registry.py disables terrain_levels only for "flat"; rough keeps it."""
    mod = importlib.import_module(_G1_CONFIG_MODULE)
    rough_cfg = mod.UnitreeG1VelocityRoughNostairsEnvCfg()
    flat_cfg = mod.UnitreeG1VelocityFlatEnvCfg()
    assert rough_cfg.curriculum.terrain_levels is not None
    assert flat_cfg.curriculum.terrain_levels is None
