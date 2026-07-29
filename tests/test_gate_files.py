# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Structural validation of every declarative gate file in benchmark/gates/.

Structure only, on purpose: gate THRESHOLDS live exclusively in the YAML
files (repo hard rule) -- no test may assert a specific line value, or the
number would exist in code again.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_GATES_DIR = _REPO / "benchmark" / "gates"

_PKG_DIR = _REPO / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub


def _load_run_gates_module():
    spec = importlib.util.spec_from_file_location(
        "run_gates", _REPO / "scripts" / "sim2sim" / "run_gates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_gates = _load_run_gates_module()

GATE_FILES = sorted(_GATES_DIR.glob("*.yaml"))


def test_gate_files_exist():
    assert GATE_FILES, f"no gate files in {_GATES_DIR}"


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_gate_file_loads_and_is_well_formed(path):
    spec = run_gates.load_gate_file(path)
    assert spec["task"]
    assert float(spec["protocol"]["seconds"]) > 0
    names = [g["name"] for g in spec["gates"]]
    assert len(names) == len(set(names)), "duplicate gate names"
    for gate in spec["gates"]:
        assert isinstance(float(gate["threshold"]), float)
        for axis, value in gate["command"].items():
            assert axis in ("vx", "vy", "wz")
            float(value)
    veto = spec["veto"]
    assert 0.0 <= float(veto["min_contact_frac"]) <= 1.0
    assert 0.0 <= float(veto["max_asymmetry"]) <= 1.0


def test_bad_schema_version_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\ntask: x\nprotocol: {seconds: 1}\ngates: []\nveto: {}\n")
    with pytest.raises(ValueError, match="schema_version"):
        run_gates.load_gate_file(bad)


def test_missing_gate_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\ntask: x\nprotocol: {seconds: 1}\n"
        "veto: {min_contact_frac: 0.2, max_asymmetry: 0.3}\n"
        "gates:\n  - name: g\n    command: {vx: 0}\n    metric: m\n    threshold: 1\n"
    )
    with pytest.raises(ValueError, match="direction"):
        run_gates.load_gate_file(bad)
