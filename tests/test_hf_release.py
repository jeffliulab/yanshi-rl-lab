# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Resolution of published Hugging Face releases in run_gates.py.

Published policy weights never enter git (benchmark/results/SCHEMA.md), so a
leaderboard row names a Hub release and ``--hf`` turns that name into the two
files the runner needs. These tests cover the naming and the failure modes
without touching the network: the download itself is stubbed, because the
property worth pinning is that a bad reference FAILS rather than quietly
evaluating something else.
"""

from __future__ import annotations

import importlib.util
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


def _load_run_gates_module():
    spec = importlib.util.spec_from_file_location(
        "run_gates_hf", _REPO / "scripts" / "sim2sim" / "run_gates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_gates = _load_run_gates_module()


# ---------------------------------------------------------------- parse_hf_ref


def test_parse_splits_repo_and_revision():
    assert run_gates.parse_hf_ref("jeffliulab/yanshi-unitree-g1-dof29@g1-flat-parity-s42") == (
        "jeffliulab/yanshi-unitree-g1-dof29",
        "g1-flat-parity-s42",
    )


@pytest.mark.parametrize(
    "ref",
    [
        "jeffliulab/yanshi-unitree-g1-dof29",  # no revision at all
        "jeffliulab/yanshi-unitree-g1-dof29@",  # empty revision
        "@g1-flat-parity-s42",  # no repository
        "yanshi-unitree-g1-dof29@v1",  # no owner
        "a/b/c@v1",  # too many path segments
    ],
)
def test_parse_rejects_malformed_references(ref):
    """A row that cannot say exactly which release it measured is not a row."""
    with pytest.raises(ValueError):
        run_gates.parse_hf_ref(ref)


# ------------------------------------------------------------ fetch_hf_release


def _fake_hub(monkeypatch, snapshot_root: Path, *, recorder: dict | None = None):
    """Install a huggingface_hub stub whose snapshot_download returns a path."""

    def snapshot_download(repo_id, revision, allow_patterns=None):
        if recorder is not None:
            recorder.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        return str(snapshot_root)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def _write_release(root: Path, subdir: str | None) -> Path:
    release = root if subdir is None else root / subdir
    release.mkdir(parents=True, exist_ok=True)
    (release / run_gates.HF_CONTRACT_NAME).write_text("{}", encoding="utf-8")
    (release / run_gates.HF_POLICY_NAME).write_bytes(b"onnx")
    return release


def test_fetch_returns_the_two_files_the_runner_needs(tmp_path, monkeypatch):
    release = _write_release(tmp_path, "velocity-flat")
    recorder: dict = {}
    _fake_hub(monkeypatch, tmp_path, recorder=recorder)

    contract, policy = run_gates.fetch_hf_release("owner/repo@v1", "velocity-flat")

    assert Path(contract) == release / run_gates.HF_CONTRACT_NAME
    assert Path(policy) == release / run_gates.HF_POLICY_NAME
    # The pinned revision must reach the Hub verbatim -- resolving it to
    # anything else (or to a default branch) is how a row silently drifts.
    assert recorder["repo_id"] == "owner/repo"
    assert recorder["revision"] == "v1"
    assert recorder["allow_patterns"] == "velocity-flat/*"


def test_fetch_supports_a_release_at_the_repository_root(tmp_path, monkeypatch):
    _write_release(tmp_path, None)
    recorder: dict = {}
    _fake_hub(monkeypatch, tmp_path, recorder=recorder)

    contract, policy = run_gates.fetch_hf_release("owner/repo@v1", None)

    assert Path(contract).parent == tmp_path
    assert Path(policy).parent == tmp_path
    assert recorder["allow_patterns"] is None


@pytest.mark.parametrize("absent", ["contract", "policy"])
def test_fetch_fails_loudly_when_the_release_is_incomplete(tmp_path, monkeypatch, absent):
    release = _write_release(tmp_path, "velocity-flat")
    name = run_gates.HF_CONTRACT_NAME if absent == "contract" else run_gates.HF_POLICY_NAME
    (release / name).unlink()
    _fake_hub(monkeypatch, tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        run_gates.fetch_hf_release("owner/repo@v1", "velocity-flat")
    assert name in str(excinfo.value)


def test_fetch_does_not_swallow_a_bad_revision(tmp_path, monkeypatch):
    """A missing tag must surface, never fall back to some other release.

    This is the fault-injection case: if --hf ever degraded to "download what
    you can", a leaderboard command could report numbers from weights it did
    not name.
    """

    def exploding_download(repo_id, revision, allow_patterns=None):
        raise RuntimeError(f"404: {repo_id}@{revision}")

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = exploding_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(RuntimeError, match="404"):
        run_gates.fetch_hf_release("owner/repo@no-such-tag", "velocity-flat")


def test_fetch_reports_the_missing_dependency_actionably(tmp_path, monkeypatch):
    """huggingface_hub is an optional import; the error must say how to fix it."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError

    with pytest.raises(SystemExit) as excinfo:
        run_gates.fetch_hf_release("owner/repo@v1", None)
    assert "pip install huggingface_hub" in str(excinfo.value)
