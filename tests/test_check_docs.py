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


LANGS = ("zh", "ja")

# Minimum a README needs to satisfy the language-switch check, so the other
# tests can vary one thing at a time.
_BADGE = 'https://img.shields.io/badge/Language-English-2f81f7'


def _switch(here_depth: int, self_lang: str | None) -> str:
    """Language badge block written from a file `here_depth` levels below root."""
    up = "../" * here_depth
    rows = [f'<a href="{up}README.md"><img src="{_BADGE}" alt="English"></a>']
    for lang in LANGS:
        target = "README.md" if lang == self_lang else (
            f"{up}docs/i18n/{lang}/README.md" if here_depth == 0 else f"../{lang}/README.md"
        )
        rows.append(f'<a href="{target}"><img src="{_BADGE}" alt="{lang}"></a>')
    return "\n".join(rows)


def _write_readmes(root: Path, en_body: str, *, per_lang: dict[str, str] | None = None) -> None:
    """Write an English README plus one translation per LANGS entry."""
    (root / "README.md").write_text(
        f"# T\n\n{_switch(0, None)}\n\n{en_body}", encoding="utf-8"
    )
    for lang in LANGS:
        d = root / "docs" / "i18n" / lang
        d.mkdir(parents=True, exist_ok=True)
        body = (per_lang or {}).get(lang, en_body)
        (d / "README.md").write_text(
            f"# T\n\n{_switch(3, lang)}\n\n{body}", encoding="utf-8"
        )


def test_real_repo_readmes_pass():
    assert check_docs.run_checks(_REPO) == []


def test_translations_are_discovered():
    names = check_docs.readme_files(_REPO)
    assert names[0] == "README.md"
    assert "docs/i18n/zh/README.md" in names


def test_heading_count_drift_detected(tmp_path):
    _write_readmes(tmp_path, "## A\n## B\n", per_lang={"zh": "## A\n"})
    failures = check_docs.run_checks(tmp_path)
    assert any("section counts differ" in f for f in failures)


def test_missing_referenced_path_detected(tmp_path):
    _write_readmes(tmp_path, "## A\nSee [the plan](docs/missing_plan.md) and `scripts/gone.py`.\n")
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
    _write_readmes(tmp_path, body)
    assert check_docs.run_checks(tmp_path) == []


def test_inline_code_path_is_root_relative_even_in_a_translation(tmp_path):
    """Prose naming `assets/registry.py` means the repo root, wherever it is written."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "registry.py").write_text("", encoding="utf-8")
    _write_readmes(tmp_path, "## A\nPin the source in `assets/registry.py`.\n")
    assert check_docs.run_checks(tmp_path) == []


def test_missing_readme_reported(tmp_path):
    (tmp_path / "docs" / "i18n" / "zh").mkdir(parents=True)
    failures = check_docs.run_checks(tmp_path)
    assert any("README.md" in f for f in failures)


def test_language_badges_above_the_title_rejected(tmp_path):
    _write_readmes(tmp_path, "## A\n")
    p = tmp_path / "README.md"
    lines = p.read_text(encoding="utf-8").split("\n")
    title = lines.pop(0)
    p.write_text("\n".join(lines + [title]), encoding="utf-8")
    failures = check_docs.run_checks(tmp_path)
    assert any("above the title" in f for f in failures)


def test_missing_language_link_detected(tmp_path):
    _write_readmes(tmp_path, "## A\n")
    p = tmp_path / "README.md"
    text = p.read_text(encoding="utf-8")
    text = "\n".join(ln for ln in text.split("\n") if "docs/i18n/ja/README.md" not in ln)
    p.write_text(text, encoding="utf-8")
    failures = check_docs.run_checks(tmp_path)
    assert any("missing language link" in f for f in failures)


def test_bold_inside_a_paragraph_rejected(tmp_path):
    _write_readmes(tmp_path, "## A\nThis sentence has **emphasis** in the middle.\n")
    failures = check_docs.run_checks(tmp_path)
    assert any("bold inside a paragraph" in f for f in failures)


def test_too_many_table_rows_rejected(tmp_path):
    rows = "\n".join("| a | b |" for _ in range(check_docs.MAX_TABLE_ROWS + 1))
    _write_readmes(tmp_path, f"## A\n{rows}\n")
    failures = check_docs.run_checks(tmp_path)
    assert any("table rows" in f for f in failures)
