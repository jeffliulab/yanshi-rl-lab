# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Preset values are transcriptions:
# - FLAT: the flat generator of the unitree_rl_lab G1 velocity base config
#   ("COBBLESTONE_ROAD_CFG", upstream repository Apache-2.0).
# - ROUGH: generator head shared with FLAT, sub-terrains from the official
#   Isaac Lab ``ROUGH_TERRAINS_CFG`` (isaaclab/terrains/config/rough.py,
#   BSD-3-Clause).

"""Named terrain presets, robot-independent.

Tasks select a terrain by name at registration time
(``register_velocity(..., terrains=("flat", "rough"))``); no per-terrain
config files are ever created. A robot that needs a different difficulty
overrides terrain parameters in its own task config, never here.

``get()`` returns a deep copy so that per-task mutations (curriculum flag,
row/column trimming in play configs) never leak back into the shared presets.
"""

from __future__ import annotations

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


def _generator_head(sub_terrains: dict) -> TerrainGeneratorCfg:
    """Shared generator head (size/border/grid/scales), same for every preset.

    Values are the unitree_rl_lab G1 velocity base generator head, kept
    identical across presets so that only ``sub_terrains`` differ between
    terrain names (single-variable discipline).
    """
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=9,
        num_cols=21,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains=sub_terrains,
    )


# Flat ground, implemented as a generator with a single mesh-plane sub-terrain
# (verbatim port of the upstream flat base; functionally a plane, but keeps the
# terrain-importer code path identical to the predecessor stack for the M1
# parity gate).
FLAT_TERRAINS_CFG = _generator_head({
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
})

# Mixed rough terrain: official Isaac Lab ROUGH_TERRAINS_CFG sub-terrains
# (verbatim values) under the shared generator head above.
ROUGH_TERRAINS_CFG = _generator_head({
    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),
    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=0.2,
        step_height_range=(0.05, 0.23),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
    ),
})

_PRESETS: dict[str, TerrainGeneratorCfg] = {
    "flat": FLAT_TERRAINS_CFG,
    "rough": ROUGH_TERRAINS_CFG,
}


def names() -> tuple[str, ...]:
    """Registered preset names."""
    return tuple(_PRESETS)


def get(name: str) -> TerrainGeneratorCfg:
    """Return a fresh (deep) copy of the named preset."""
    try:
        preset = _PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown terrain preset {name!r}; available: {sorted(_PRESETS)}") from None
    return copy.deepcopy(preset)
