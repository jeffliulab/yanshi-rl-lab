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


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_gate_file_declares_its_full_identity(path):
    """robot / task / exam must all be present and mutually consistent.

    Schema v1 carried only ``task``, conflating "what is being compared" with
    "which paper was sat". Two robots examined at different commanded yaw
    rates then landed in the same leaderboard column looking comparable.
    """
    from yanshi_rl_lab.robots import registry as robot_registry

    spec = run_gates.load_gate_file(path)
    assert spec["exam"] == path.stem
    assert spec["task"] and "/" not in spec["task"]
    # The robot must actually exist -- a typo here would silently orphan the
    # exam from the profile that supplies its scene.
    robot_registry.get(spec["robot"])


@pytest.mark.parametrize("path", GATE_FILES, ids=lambda p: p.name)
def test_gate_file_needs_no_hand_written_scene(path):
    """Either every gate names its own scene, or the robot profile declares one.

    This is the property that lets a repro command omit --scene entirely.
    Checks the DECLARATION, not the fetched file: vendor assets are never
    committed, so CI has none, and "is the scene declared" is the question
    anyway -- whether it has been downloaded is a separate concern that
    ``asset_path()`` already reports on its own.
    """
    from yanshi_rl_lab.robots import registry as robot_registry

    spec = run_gates.load_gate_file(path)
    if all("scene" in gate for gate in spec["gates"]):
        return
    assert robot_registry.get(spec["robot"]).scene_mjcf, (
        f"{path.name} relies on the profile for its scene, but "
        f"{spec['robot']} declares scene_mjcf=None"
    )


def _minimal(**overrides) -> str:
    doc = {
        "schema_version": run_gates.GATES_SCHEMA_VERSION,
        "robot": "unitree/g1/dof29",
        "task": "velocity-flat",
        "exam": "bad",
        "protocol": "{seconds: 1}",
        "veto": "{min_contact_frac: 0.2, max_asymmetry: 0.3}",
        "gates": "\n  - name: g\n    command: {vx: 0}\n    metric: m\n    threshold: 1\n"
        "    direction: '>='\n    unit: m\n",
    }
    doc.update(overrides)
    lines = [f"{k}: {v}" if k != "gates" else f"gates:{v}" for k, v in doc.items()]
    return "\n".join(lines) + "\n"


def test_bad_schema_version_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(_minimal(schema_version=99))
    with pytest.raises(ValueError, match="schema_version"):
        run_gates.load_gate_file(bad)


def test_missing_identity_key_rejected(tmp_path):
    for key in ("robot", "task", "exam"):
        doc = _minimal()
        doc = "\n".join(ln for ln in doc.splitlines() if not ln.startswith(f"{key}:")) + "\n"
        bad = tmp_path / "bad.yaml"
        bad.write_text(doc)
        with pytest.raises(ValueError, match=key):
            run_gates.load_gate_file(bad)


def test_exam_name_must_match_filename(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(_minimal(exam="something-else"))
    with pytest.raises(ValueError, match="does not match its filename"):
        run_gates.load_gate_file(bad)


def test_missing_gate_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        _minimal(gates="\n  - name: g\n    command: {vx: 0}\n    metric: m\n    threshold: 1\n")
    )
    with pytest.raises(ValueError, match="direction"):
        run_gates.load_gate_file(bad)
