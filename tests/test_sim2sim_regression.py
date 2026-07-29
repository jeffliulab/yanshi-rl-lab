# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""M2 sim2sim regression: legacy contract -> v2 -> new runtime must reproduce
the predecessor stack's recorded gate results.

Chain under test: alice-house v1-G1 contract.json --(legacy.from_contract_v1_g1
+ G1_PROFILE runtime facts)--> v2 contract --> contract-driven MuJoCo runtime
--> the four declarative gates of benchmark/gates/velocity-flat-turn.yaml.
Expected: same pass/fail verdicts and per-gate values within the tolerance
recorded in tests/data/regression_reference_g1_turn.yaml (predecessor
measurements from S2-Exp5; see that file for provenance).

Requirements (test SKIPS with a reason when unmet -- this keeps the suite
green on machines without the archive, while the M2 acceptance run executes
it for real):

- mujoco + onnxruntime importable (conda isaaclab env; pure CPU -- run with
  CUDA_VISIBLE_DEVICES="" and WITHOUT MUJOCO_GL=egl);
- the read-only predecessor archives on this machine:
  - archived S2-Exp5 checkpoints (policy.onnx per cell),
  - the alice-house g1-29dof-turn contract.json,
  - the predecessor MuJoCo scene (scene_23dof.xml -- file name says 23dof,
    contents are the 29-DoF deploy G1; this is the scene every predecessor
    gate number was measured on).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("mujoco", reason="sim2sim regression needs MuJoCo (conda isaaclab env)")
pytest.importorskip("onnxruntime", reason="sim2sim regression needs onnxruntime")

_REPO = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO / "source" / "yanshi_rl_lab" / "yanshi_rl_lab"
if "yanshi_rl_lab" not in sys.modules:
    stub = types.ModuleType("yanshi_rl_lab")
    stub.__path__ = [str(_PKG_DIR)]
    sys.modules["yanshi_rl_lab"] = stub

from yanshi_rl_lab.deploy import legacy  # noqa: E402
from yanshi_rl_lab.robots.unitree.g1.profile import G1_PROFILE  # noqa: E402

# ---- read-only predecessor fixtures (machine-local; test skips if absent) ----
_PREDECESSOR = Path("/home/jeff/2026-summer-career-projects/unitree-g1-locomotion")
_ARCHIVE = (
    _PREDECESSOR
    / "个人用本地私人文档"
    / "ARCHIVES"
    / "29dof-媒体源与失败证据"
    / "s2e5-矩阵五格-最终checkpoint"
)
_V1_CONTRACT = Path(
    "/home/jeff/2026-summer-career-projects/alice-house/policies/g1-29dof-turn/contract.json"
)
_OLD_SCENE = _PREDECESSOR / "scenes" / "mujoco" / "assets" / "g1_unitree_mujoco" / "scene_23dof.xml"

_GATES_FILE = _REPO / "benchmark" / "gates" / "velocity-flat-turn.yaml"
_REFERENCE_FILE = Path(__file__).parent / "data" / "regression_reference_g1_turn.yaml"


def _missing() -> list:
    return [str(p) for p in (_ARCHIVE, _V1_CONTRACT, _OLD_SCENE) if not p.exists()]


pytestmark = pytest.mark.skipif(
    bool(_missing()), reason=f"predecessor archives not on this machine: {_missing()}"
)


def _load_run_gates_module():
    spec = importlib.util.spec_from_file_location(
        "run_gates_regression", _REPO / "scripts" / "sim2sim" / "run_gates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract_v2():
    """Legacy v1-G1 contract lifted to v2, runtime facts from G1_PROFILE."""
    d = json.loads(_V1_CONTRACT.read_text(encoding="utf-8"))
    contract = legacy.from_contract_v1_g1(
        d,
        pd_mode=G1_PROFILE.pd_mode,
        foot_bodies=[G1_PROFILE.feet_bodies],  # regex, resolved against the scene
        root_joint_name=G1_PROFILE.root_joint_name,
        base_link=G1_PROFILE.base_link,
        joint_sdk_names=G1_PROFILE.joint_sdk_names,
    )
    contract.validate()
    contract.require_runtime_fields()
    return contract


@pytest.fixture(scope="module")
def reference():
    return yaml.safe_load(_REFERENCE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_reports(contract_v2, reference):
    """Run the four gates once per archived policy cell (expensive: ~4x8 s of
    simulated time per cell, pure CPU)."""
    run_gates = _load_run_gates_module()
    spec = run_gates.load_gate_file(_GATES_FILE)
    reports = {}
    for cell in reference["policies"]:
        policy = _ARCHIVE / cell / "policy.onnx"
        assert policy.exists(), f"archived policy missing: {policy}"
        reports[cell] = run_gates.evaluate(spec, str(policy), contract_v2, str(_OLD_SCENE))
    return reports


def test_contract_layout_matches_predecessor(contract_v2):
    assert len(contract_v2.joint_names) == 29
    assert contract_v2.obs_dim == 480
    assert contract_v2.history_length == 5
    assert contract_v2.pd_mode == "implicit"
    assert contract_v2.layout_carryover is not None  # provenance survived the lift


@pytest.mark.parametrize("cell", ["aligned", "onlyw"])
def test_gate_verdicts_match_predecessor(gate_reports, reference, cell):
    report = gate_reports[cell]
    expected = reference["policies"][cell]
    assert bool(report["veto_hits"]) == expected["veto"], report["veto_hits"]
    all_passed = all(e["passed"] for e in report["results"])
    assert all_passed == expected["all_passed"], [
        (e["gate"]["name"], e["value"], e["passed"]) for e in report["results"]
    ]


@pytest.mark.parametrize("cell", ["aligned", "onlyw"])
def test_gate_values_within_tolerance(gate_reports, reference, cell):
    tol = float(reference["rel_tolerance"])
    report = gate_reports[cell]
    expected_gates = reference["policies"][cell]["gates"]
    measured = {e["gate"]["name"]: e for e in report["results"]}
    mismatches = []
    for gate_name, exp in expected_gates.items():
        entry = measured[gate_name]
        assert entry["gate"]["metric"] == exp["metric"], "gate/reference metric mismatch"
        got, want = float(entry["value"]), float(exp["value"])
        if abs(got - want) > tol * abs(want):
            mismatches.append(f"{gate_name}: measured {got:.3f} vs predecessor {want:.3f}")
    assert not mismatches, f"{cell}: outside +/-{tol:.0%}: {mismatches}"
