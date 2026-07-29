# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Named terrain presets (flat / rough / stairs ...), robot-independent.

Tasks select a terrain by name at registration time
(``register(..., terrains=["flat", "rough"])``); no per-terrain config files
are ever created. A robot that needs a different difficulty overrides terrain
parameters in its own task config, never here.

The actual ``TerrainGeneratorCfg`` presets are ported from the Isaac Lab
official terrain configs in milestone M1 (tracked in the maintainer ledger);
this module intentionally ships empty until then rather than carrying
untested placeholder values.
"""
