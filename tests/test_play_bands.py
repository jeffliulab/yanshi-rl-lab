# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Play difficulty-band override (PLAY_TERRAIN_DIFFICULTY).

This is the knob the S3-实验2 review videos ride on: `record_videos_s3e2.sh`
sets it to record each gate's terrain band ("0.0,0.2" low, "0.4,0.6" mid).
It ships with the recording, so a malformed band that gets silently ignored
would produce a video of the WRONG terrain -- a failure indistinguishable
from success by eye.

The parser deliberately lives in `utils/play_bands.py` rather than in
`velocity_env_cfg.py`: the latter needs `pxr` (Isaac Sim bootstrap) and so
cannot be imported on a plain box, which is exactly how this code path went
untested when it was first written (2026-08-01 audit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable on its own. CI installs only pytest/pyyaml/numpy,
# so nothing puts `yanshi_rl_lab` on the path; relying on another test module
# to have done it first works only by collection order, which is not a contract.
_PKG_PARENT = Path(__file__).resolve().parents[1] / "source" / "yanshi_rl_lab"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from yanshi_rl_lab.utils.play_bands import (  # noqa: E402
    PLAY_TERRAIN_DIFFICULTY_ENV_VAR,
    parse_difficulty_band,
)


def test_env_var_name_is_the_one_the_recording_script_sets():
    """The script and the code must agree on the name, or the override is a no-op."""
    assert PLAY_TERRAIN_DIFFICULTY_ENV_VAR == "PLAY_TERRAIN_DIFFICULTY"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.0,0.2", (0.0, 0.2)),  # 缓带（S3E2 门②工况）
        ("0.4,0.6", (0.4, 0.6)),  # 中带（S3E2 门①工况）
        ("0.6,0.9", (0.6, 0.9)),  # 与 PLAY_HARD_TERRAIN_DIFFICULTY 默认值同值
        ("0,1", (0.0, 1.0)),  # 整数写法、闭区间两端
    ],
)
def test_valid_bands_parse(raw, expected):
    assert parse_difficulty_band(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "0.5",  # 只给一个数
        "0.1,0.2,0.3",  # 给了三个
        "",  # 空串
        "low,high",  # 非数字
        "0.5,0.5",  # lo == hi，空区间
        "0.6,0.4",  # lo > hi，写反了
        "-0.1,0.5",  # 越下界
        "0.5,1.5",  # 越上界
    ],
)
def test_malformed_bands_raise(raw):
    """Must raise, not silently fall through -- see module docstring."""
    with pytest.raises(ValueError):
        parse_difficulty_band(raw)


def test_error_message_names_the_variable():
    """The operator sees this in a tsp log; it has to say which knob is wrong."""
    with pytest.raises(ValueError, match=PLAY_TERRAIN_DIFFICULTY_ENV_VAR):
        parse_difficulty_band("0.9,0.1")


def test_validation_survives_optimized_mode():
    """Guard against regressing to a bare `assert`, which -O strips out.

    The original implementation used `assert 0.0 <= lo < hi <= 1.0`; under
    `python -O` that check disappears and a reversed band would sail through.
    """
    import os
    import subprocess

    code = (
        "from yanshi_rl_lab.utils.play_bands import parse_difficulty_band\n"
        "try:\n"
        "    parse_difficulty_band('0.9,0.1')\n"
        "except ValueError:\n"
        "    print('RAISED')\n"
    )
    # The child gets its own interpreter, so hand it the package path explicitly:
    # nothing installs this package in CI.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_PKG_PARENT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "RAISED"
