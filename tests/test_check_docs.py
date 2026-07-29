# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Tests for scripts/tools/check_docs.py (bilingual README consistency)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _load_check_docs_module():
    spec = importlib.util.spec_from_file_location("check_docs", _REPO / "scripts" / "tools" / "check_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_docs = _load_check_docs_module()


def _write_readmes(root: Path, en: str, zh: str) -> None:
    (root / "README.md").write_text(en, encoding="utf-8")
    (root / "README_zh.md").write_text(zh, encoding="utf-8")


def test_real_repo_readmes_pass():
    assert check_docs.run_checks(_REPO) == []


def test_heading_count_drift_detected(tmp_path):
    _write_readmes(tmp_path, "## A\n## B\n", "## A\n")
    failures = check_docs.run_checks(tmp_path)
    assert any("section counts differ" in f for f in failures)


def test_missing_referenced_path_detected(tmp_path):
    _write_readmes(
        tmp_path,
        "## A\nSee [the plan](docs/missing_plan.md) and `scripts/gone.py`.\n",
        "## A\n",
    )
    failures = check_docs.run_checks(tmp_path)
    assert any("docs/missing_plan.md" in f for f in failures)
    assert any("scripts/gone.py" in f for f in failures)


def test_existing_paths_placeholders_and_urls_ignored(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "ok.py").write_text("", encoding="utf-8")
    body = (
        "## A\n"
        "Real: `scripts/ok.py` and [self](README.md).\n"
        "Ignored: [site](https://example.com/x.md), `robots/<vendor>/<model>/`,\n"
        "`schema_version: 2`, `results/v1/<vendor>-<model>/<task>/*.json`, `v0.1`.\n"
        "```\nfenced/fake/path.py\n## fenced heading\n```\n"
    )
    _write_readmes(tmp_path, body, "## A\n")
    assert check_docs.run_checks(tmp_path) == []


def test_missing_readme_reported(tmp_path):
    (tmp_path / "README.md").write_text("## A\n", encoding="utf-8")
    failures = check_docs.run_checks(tmp_path)
    assert any("README_zh.md" in f for f in failures)
