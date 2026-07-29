#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Documentation consistency checks for the bilingual READMEs (CI + local).

Two checks, both over ``README.md`` and ``README_zh.md`` at the repo root:

1. **Bilingual parity**: the two files must contain the same number of
   second-level (``## ``) headings. The READMEs are maintained as mirrors;
   a diverging section count is the cheapest reliable drift signal a machine
   can check.
2. **Referenced paths exist**: every repo-relative path a README mentions
   (Markdown link targets plus path-looking inline-code spans) must exist on
   disk. Catches renames/moves that forget the docs.

Path-candidate rules for inline code spans (documented so false negatives are
explainable): a `` `span` `` is treated as a repo path only if it is composed
of ``[A-Za-z0-9_./-]`` (so placeholders like ``robots/<vendor>/<model>/`` and
shell snippets are ignored), contains no ``*`` wildcard, and either contains a
``/`` or ends in a known documentation/file suffix.

Usage::

    python scripts/tools/check_docs.py            # check repo READMEs
    python scripts/tools/check_docs.py --repo-root <dir>   # for tests

Exit code 0 = all good, 1 = at least one failure (each printed on its own
line). Pure stdlib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

README_FILES = ("README.md", "README_zh.md")

# A second-level heading at line start (Markdown ATX style).
_H2_RE = re.compile(r"^## ", re.MULTILINE)
# Fenced code blocks -- stripped before any scanning, headings or paths inside
# fences are examples, not document structure.
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
# [text](target) Markdown links (images included via the same syntax).
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# `inline code` spans.
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
# Charset a repo path may use; anything else (spaces, <>, :, *) disqualifies.
_PATHISH_RE = re.compile(r"^[A-Za-z0-9_.\-/]+$")
# Suffixes that make a slash-less code span still count as a file reference.
_FILE_SUFFIXES = (".md", ".py", ".yaml", ".yml", ".toml", ".json", ".txt", ".sbatch")

_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text)


def _link_candidates(text: str) -> set[str]:
    out = set()
    for target in _LINK_RE.findall(text):
        if target.startswith(_EXTERNAL_PREFIXES):
            continue
        target = target.split("#", 1)[0]  # drop in-page anchors
        if target:
            out.add(target)
    return out


def _code_span_candidates(text: str) -> set[str]:
    out = set()
    for span in _CODE_SPAN_RE.findall(text):
        span = span.strip()
        if not _PATHISH_RE.match(span) or "*" in span:
            continue
        if "/" not in span and not span.endswith(_FILE_SUFFIXES):
            continue
        # bare domain-ish or versioned tokens ("v0.1", "2.3.2.post1") are not paths
        if "/" not in span and span.count(".") > 1:
            continue
        out.add(span.rstrip("/"))
    return out


def check_readme_paths(repo_root: Path, readme: Path) -> list[str]:
    """Return one failure string per referenced-but-missing repo path."""
    text = _strip_fences(readme.read_text(encoding="utf-8"))
    failures = []
    for candidate in sorted(_link_candidates(text) | _code_span_candidates(text)):
        if not (repo_root / candidate).exists():
            failures.append(f"{readme.name}: referenced path does not exist: {candidate}")
    return failures


def check_heading_parity(repo_root: Path) -> list[str]:
    counts = {}
    for name in README_FILES:
        text = _strip_fences((repo_root / name).read_text(encoding="utf-8"))
        counts[name] = len(_H2_RE.findall(text))
    if len(set(counts.values())) != 1:
        detail = ", ".join(f"{name}: {n} '## ' headings" for name, n in counts.items())
        return [f"bilingual README drift: section counts differ ({detail})"]
    return []


def run_checks(repo_root: Path) -> list[str]:
    failures = []
    for name in README_FILES:
        path = repo_root / name
        if not path.is_file():
            failures.append(f"missing required file: {name}")
    if failures:
        return failures
    failures += check_heading_parity(repo_root)
    for name in README_FILES:
        failures += check_readme_paths(repo_root, repo_root / name)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)

    failures = run_checks(args.repo_root.resolve())
    for line in failures:
        print(f"DOCS ERROR  {line}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} documentation check(s) failed", file=sys.stderr)
        return 1
    print(f"docs OK: {', '.join(README_FILES)} (heading parity + referenced paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
