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

⚠️ This test deliberately runs on ``scene_23dof.xml``, NOT on the scene
``G1_PROFILE`` declares. That is not the wrong-scene bug (decision ledger
D11) reappearing -- it is the opposite. The predecessor measured every number
this test reproduces on that scene, so the scene is part of the acceptance
criterion; swapping in ``scene_29dof.xml`` would change what is being
reproduced and quietly redefine the M2 bar. (For the record, that swap moves
G1 gate values by 1.6-4.3%, comfortably inside this file's +-5% tolerance --
so it would have passed and told us nothing.) Everything that measures the
CURRENT stack resolves its scene from the profile; this one reproduces
history, so it pins history's scene.

The checkpoints live outside git (weights are never committed), so the module
is opt-in on those. Defaults point at the local archive; the environment
variables are overrides.

    YANSHI_S2E5_CHECKPOINT_ARCHIVE   directory holding one <cell>/policy.onnx
                                     per S2-Exp5 matrix cell. Defaults to
                                     LOCAL-PRIVATE/Archives/
                                     predecessor-checkpoints/s2e5-matrix/
    YANSHI_V1_G1_CONTRACT            the deployment repo's v1 G1 contract.json
                                     (alice-house; no default, it is another
                                     repository's file)

Also needs mujoco + onnxruntime importable (conda isaaclab env; pure CPU --
run with CUDA_VISIBLE_DEVICES="" and WITHOUT MUJOCO_GL=egl).
"""

from __future__ import annotations

import importlib.util
import json
import os
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
from yanshi_rl_lab.robots.unitree.g1.dof29.profile import G1_PROFILE  # noqa: E402

# ---- read-only predecessor fixtures (opt-in; test skips when unset) ----
# These live outside this repository: the predecessor stack's archived
# checkpoints and the deployment repo's v1 contract. Point the two variables
# below at your own copies to run this regression; leave them unset and the
# whole module skips with a reason.
ARCHIVE_ENV_VAR = "YANSHI_S2E5_CHECKPOINT_ARCHIVE"
V1_CONTRACT_ENV_VAR = "YANSHI_V1_G1_CONTRACT"

# Default archive location, inside this repository's gitignored private area.
# The checkpoints were rescued out of the predecessor stack before it was
# archived; before that this test read them across a repository boundary and
# would have died with it.
DEFAULT_ARCHIVE = _REPO / "LOCAL-PRIVATE" / "Archives" / "predecessor-checkpoints" / "s2e5-matrix"


def _from_env(var: str, default: Path | None = None) -> Path | None:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


_ARCHIVE = _from_env(ARCHIVE_ENV_VAR, DEFAULT_ARCHIVE)
_V1_CONTRACT = _from_env(V1_CONTRACT_ENV_VAR)
# The historical scene, from this repository's own pinned vendor tree. Verified
# byte-identical to the predecessor's copy before that copy was retired.
_OLD_SCENE = _REPO / "assets" / "unitree" / "g1" / "mjcf" / "g1" / "scene_23dof.xml"

_GATES_FILE = _REPO / "benchmark" / "gates" / "velocity-flat-turn.yaml"
_REFERENCE_FILE = Path(__file__).parent / "data" / "regression_reference_g1_turn.yaml"


def _missing() -> list[str]:
    """Which fixtures are unusable — either unset or pointing at nothing."""
    out = []
    for var, path in (
        ("(vendor assets; run: python assets/fetch.py unitree/g1)", _OLD_SCENE),
        (ARCHIVE_ENV_VAR, _ARCHIVE),
        (V1_CONTRACT_ENV_VAR, _V1_CONTRACT),
    ):
        if path is None:
            out.append(f"${var} unset")
        elif not path.exists():
            out.append(f"${var} -> {path} (not found)")
    return out


pytestmark = pytest.mark.skipif(
    bool(_missing()), reason=f"predecessor fixtures unavailable: {_missing()}"
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
