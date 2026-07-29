# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Smoke tests for the scripts/tools/new_robot.py scaffolder.

Everything runs against a temporary repo root (the scaffolder takes the root
as a parameter precisely so tests never touch the real tree).
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_new_robot_module():
    spec = importlib.util.spec_from_file_location("new_robot", _REPO / "scripts" / "tools" / "new_robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new_robot = _load_new_robot_module()


def test_scaffold_creates_all_files_and_they_compile(tmp_path):
    created = new_robot.scaffold(tmp_path, "acme", "biped_one")
    rel = sorted(str(p.relative_to(tmp_path)) for p in created)
    pkg = "source/yanshi_rl_lab/yanshi_rl_lab"
    assert rel == sorted(
        [
            f"{pkg}/robots/acme/__init__.py",
            f"{pkg}/robots/acme/biped_one/__init__.py",
            f"{pkg}/robots/acme/biped_one/profile.py",
            f"{pkg}/tasks/locomotion/velocity/config/acme/__init__.py",
            f"{pkg}/tasks/locomotion/velocity/config/acme/biped_one/__init__.py",
            "tests/test_biped_one_profile.py",
        ]
    )
    for path in created:
        py_compile.compile(str(path), doraise=True)


def test_unfilled_profile_scaffold_refuses_to_import(tmp_path):
    new_robot.scaffold(tmp_path, "acme", "biped_two")
    profile_py = tmp_path / "source/yanshi_rl_lab/yanshi_rl_lab/robots/acme/biped_two/profile.py"
    spec = importlib.util.spec_from_file_location("acme_profile_scaffold", profile_py)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(NotImplementedError, match="unfilled scaffold"):
        spec.loader.exec_module(module)


def test_scaffold_refuses_to_overwrite(tmp_path):
    new_robot.scaffold(tmp_path, "acme", "biped_three")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        new_robot.scaffold(tmp_path, "acme", "biped_three")


def test_scaffold_rejects_bad_names(tmp_path):
    with pytest.raises(SystemExit, match="invalid"):
        new_robot.scaffold(tmp_path, "Acme", "biped")
    with pytest.raises(SystemExit, match="invalid"):
        new_robot.scaffold(tmp_path, "acme", "Biped-1")
