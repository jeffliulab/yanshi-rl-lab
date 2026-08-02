# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Parsing for the play/recording terrain difficulty band override.

Kept out of ``velocity_env_cfg`` on purpose: that module's import chain needs
``pxr`` (Isaac Sim app bootstrap), so anything living there cannot be unit
tested on a plain CI box. This parser is pure stdlib, so the review-video
code path is covered by tests that always run.
"""

from __future__ import annotations

# Name of the environment variable that overrides the play difficulty band.
# Used for gate-band review videos, e.g. "0.0,0.2" (low band) or "0.4,0.6"
# (mid band). Mirrors the predecessor stack's S3E1_PLAY_DIFF precedent.
PLAY_TERRAIN_DIFFICULTY_ENV_VAR = "PLAY_TERRAIN_DIFFICULTY"


def parse_difficulty_band(raw: str, *, var_name: str = PLAY_TERRAIN_DIFFICULTY_ENV_VAR) -> tuple[float, float]:
    """Parse a ``"lo,hi"`` difficulty band into a validated ``(lo, hi)`` tuple.

    Args:
        raw: the raw environment-variable value, e.g. ``"0.4,0.6"``.
        var_name: variable name to quote in error messages.

    Returns:
        ``(lo, hi)`` with ``0.0 <= lo < hi <= 1.0``.

    Raises:
        ValueError: on anything malformed or out of range. Deliberately a real
            exception rather than ``assert``: assertions vanish under ``python -O``,
            and a silently-ignored band would produce a review video of the wrong
            terrain -- a failure that looks exactly like a passing one.
    """
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f"{var_name} 必须是 'lo,hi' 两个数，收到: {raw!r}")
    try:
        lo, hi = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"{var_name} 的两项必须都是数字，收到: {raw!r}") from exc
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"{var_name} 需满足 0.0 <= lo < hi <= 1.0，收到: {raw!r}")
    return lo, hi
