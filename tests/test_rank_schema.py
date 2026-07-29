# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Yanshi Rank result-schema validation and renderer smoke tests.

Fixture data lives in ``tests/data/rank/`` -- NEVER under
``benchmark/results/`` (fabricated numbers must not be mistakable for real
leaderboard entries; the fixture says so in its ``notes`` field too).
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO / "tests" / "data" / "rank" / "sample_result.json"


def _load_render_rank_module():
    spec = importlib.util.spec_from_file_location("render_rank", _REPO / "benchmark" / "render_rank.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_rank = _load_render_rank_module()


@pytest.fixture()
def sample() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _results_tree(tmp_path: Path, sample: dict) -> Path:
    """Materialize the fixture inside a schema-correct results tree."""
    root = tmp_path / "results" / "v1"
    dest = root / sample["robot"].replace("/", "-") / sample["task"]
    dest.mkdir(parents=True)
    (dest / "fixture-run.json").write_text(json.dumps(sample), encoding="utf-8")
    return root


# ---------------------------------------------------------------- validation


def test_sample_fixture_is_valid(sample):
    assert render_rank.validate_result(sample) == []


@pytest.mark.parametrize("field", render_rank.REQUIRED_FIELDS)
def test_missing_required_field_rejected(sample, field):
    del sample[field]
    assert any("missing required" in e for e in render_rank.validate_result(sample))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"trust_tier": "totally-legit"}, "trust_tier"),
        ({"protocol_version": 2}, "protocol_version"),
        ({"robot": "UnitreeG1"}, "robot"),
        ({"commit": "not-a-sha"}, "commit"),
        ({"seeds": []}, "seeds"),
        ({"seeds": [42, 42, 43]}, "duplicates"),
        ({"repro_command": "line one\nline two"}, "repro_command"),
        ({"checkpoint": {"hf_repo": "x/y"}}, "checkpoint"),
        ({"date": "29-07-2026"}, "date"),
        ({"surprise_field": 1}, "unknown field"),
    ],
)
def test_bad_values_rejected(sample, mutation, match):
    sample.update(mutation)
    errors = render_rank.validate_result(sample)
    assert errors, f"mutation {mutation} slipped through"
    assert any(match in e for e in errors)


def test_metric_seed_mismatch_rejected(sample):
    bad = copy.deepcopy(sample)
    bad["metrics"]["tracking_rmse"] = {"42": 0.1, "43": 0.1, "99": 0.1}
    assert any("do not match seeds" in e for e in render_rank.validate_result(bad))


def test_path_mismatch_rejected(tmp_path, sample):
    root = tmp_path / "results" / "v1"
    wrong = root / "agibot-x2" / sample["task"]  # file claims unitree/g1
    wrong.mkdir(parents=True)
    (wrong / "misplaced.json").write_text(json.dumps(sample), encoding="utf-8")
    entries, errors = render_rank.load_results(root)
    assert not entries
    assert any("does not match directory" in e for e in errors)


# ----------------------------------------------------------------- rendering


def test_render_from_results_tree(tmp_path, sample):
    root = _results_tree(tmp_path, sample)
    entries, errors = render_rank.load_results(root)
    assert errors == []
    assert len(entries) == 1

    md = render_rank.render_markdown(entries)
    assert "unitree/g1" in md
    assert "velocity-flat" in md
    page = render_rank.render_html(entries)
    assert "unitree/g1" in page
    assert "verified" in page
    # self-contained page: no external stylesheet/script/font/image requests
    for marker in ("<link", "<script", "src="):
        assert marker not in page


def test_empty_results_render_without_crashing(tmp_path):
    root = tmp_path / "results" / "v1"
    root.mkdir(parents=True)
    entries, errors = render_rank.load_results(root)
    assert entries == [] and errors == []
    md = render_rank.render_markdown(entries)
    assert "No results yet" in md
    page = render_rank.render_html(entries)
    assert "No results yet" in page


def test_main_check_mode_flags_broken_file(tmp_path, sample, capsys):
    sample["trust_tier"] = "bogus"
    root = _results_tree(tmp_path, sample)
    rc = render_rank.main(["--check", "--results-root", str(root)])
    assert rc == 1
    assert "SCHEMA ERROR" in capsys.readouterr().err


def test_main_renders_outputs(tmp_path, sample):
    root = _results_tree(tmp_path, sample)
    site = tmp_path / "site"
    rc = render_rank.main(["--results-root", str(root), "--site-dir", str(site)])
    assert rc == 0
    assert (site / "index.html").exists()
    assert (site / "rank.md").exists()


def test_aggregate_median_iqr(sample):
    stats = render_rank.aggregate({**sample, "_label": "x"})
    med, iqr = stats["net_displacement_m"]
    assert med == pytest.approx(4.42)
    # inclusive quantiles over [4.31, 4.42, 4.50]: p25=4.365, p75=4.46
    assert iqr == pytest.approx(0.095)
