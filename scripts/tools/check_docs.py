#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Documentation consistency checks for the READMEs (CI + local).

English lives at ``README.md``; every translation lives at
``docs/i18n/<lang>/README.md`` and is discovered automatically, so adding a
language needs no change here.

Checks:

1. **Structure parity**: every translation must contain the same number of
   second-level (``## ``) headings as English. The READMEs are maintained as
   mirrors; a diverging section count is the cheapest reliable drift signal a
   machine can check, and it is the only one that scales past two languages.
2. **Referenced paths exist**: every repo-relative path a README mentions
   (Markdown link targets, HTML ``src``/``href`` attributes, and path-looking
   inline-code spans) must resolve. Paths are resolved **relative to the file
   that wrote them**, so a translation's ``../../../LICENSE`` is checked
   correctly, and against **what git tracks** rather than the filesystem --
   a file that exists locally but was never added renders as a broken link on
   GitHub.
3. **Language switcher**: each README links to every language, and the language
   badges sit *below* the title rather than above it.
4. **Prose thresholds**: line count, table rows and bold spans stay under the
   ceilings measured from two flagship robotics READMEs (Isaac Lab 141 lines,
   Unitree RL Lab 150; both under 10 table rows with single-digit bold counts).
   Bold is an item lead-in only -- inside a paragraph it stops meaning anything.

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
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

ENGLISH_README = "README.md"
I18N_DIR = "docs/i18n"

# Ceilings, not targets: a README that trips one has become a spec sheet.
MAX_LINES = 200
MAX_TABLE_ROWS = 10
MAX_BOLD = 9


def readme_files(repo_root: Path) -> tuple[str, ...]:
    """English first, then one per language directory that has a README."""
    out = [ENGLISH_README]
    i18n = repo_root / I18N_DIR
    if i18n.is_dir():
        out += [
            f"{I18N_DIR}/{d.name}/README.md"
            for d in sorted(i18n.iterdir())
            if (d / "README.md").is_file()
        ]
    return tuple(out)


def tracked_paths(repo_root: Path) -> set[str] | None:
    """What git knows about (index + HEAD), or None outside a repository."""
    out: set[str] = set()
    ok = False
    for args in (["git", "ls-files"], ["git", "ls-tree", "-r", "--name-only", "HEAD"]):
        try:
            r = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
            continue
        ok = True
        out.update(line for line in r.stdout.splitlines() if line)
    if not ok:
        return None  # tmp dirs in tests are not repositories; fall back to disk
    out.update(str(Path(p).parent) for p in list(out))
    return out

# A second-level heading at line start (Markdown ATX style).
_H2_RE = re.compile(r"^## ", re.MULTILINE)
# Fenced code blocks -- stripped before any scanning, headings or paths inside
# fences are examples, not document structure.
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
# [text](target) Markdown links (images included via the same syntax).
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# HTML src="" / href="" attributes (badges and <img> hero clips use these).
_HTML_ATTR_RE = re.compile(r'<(?:img[^>]+src|a[^>]+href)="([^"]+)"')
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


def _resolve(repo_root: Path, name: str, candidate: str) -> str | None:
    """Repo-relative form of a link written inside ``name``, or None if it escapes."""
    try:
        target = (repo_root / Path(name).parent / candidate).resolve()
        return str(target.relative_to(repo_root.resolve()))
    except ValueError:
        return None


def check_readme_paths(repo_root: Path, name: str, tracked: set[str] | None) -> list[str]:
    """One failure per referenced path that git does not have.

    Resolved relative to the referring file, so a translation's ``../../media``
    is checked where it actually points.
    """
    text = _strip_fences((repo_root / name).read_text(encoding="utf-8"))
    html = {t for t in _HTML_ATTR_RE.findall(text) if not t.startswith(_EXTERNAL_PREFIXES)}
    failures = []

    # Two kinds of reference, resolved differently on purpose:
    #   - links (Markdown / HTML) are relative to the file that wrote them, so a
    #     translation's ../../media/x.gif is what the reader's browser follows;
    #   - a path inside `inline code` is prose naming a file, and prose always
    #     names it from the repository root regardless of which file says it.
    checks = [(c, True) for c in _link_candidates(text) | html]
    checks += [(c, False) for c in _code_span_candidates(text)]

    for candidate, relative in sorted(checks):
        bare = candidate.split("#", 1)[0]
        resolved = _resolve(repo_root, name, bare) if relative else bare
        if resolved is None:
            failures.append(f"{name}: reference escapes the repository: {candidate}")
        elif tracked is None:
            if not (repo_root / resolved).exists():
                failures.append(f"{name}: referenced path does not exist: {candidate}")
        elif resolved not in tracked:
            failures.append(f"{name}: referenced path not tracked by git: {candidate}")
    return failures


def check_heading_parity(repo_root: Path, names: tuple[str, ...]) -> list[str]:
    counts = {}
    for name in names:
        text = _strip_fences((repo_root / name).read_text(encoding="utf-8"))
        counts[name] = len(_H2_RE.findall(text))
    if len(set(counts.values())) != 1:
        detail = ", ".join(f"{name}: {n} '## ' headings" for name, n in counts.items())
        return [f"README drift: section counts differ ({detail})"]
    return []


def check_language_switch(repo_root: Path, name: str, names: tuple[str, ...]) -> list[str]:
    """Every language linked, and the badges below the title rather than above."""
    text = (repo_root / name).read_text(encoding="utf-8")
    failures = []
    linked = set()
    for candidate in _link_candidates(text) | {
        t for t in _HTML_ATTR_RE.findall(text) if not t.startswith(_EXTERNAL_PREFIXES)
    }:
        resolved = _resolve(repo_root, name, candidate.split("#", 1)[0])
        if resolved:
            linked.add(resolved)
    for other in names:
        if other != name and other not in linked:
            failures.append(f"{name}: missing language link to {other}")

    lines = text.split("\n")
    title = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), None)
    badge = next((i for i, ln in enumerate(lines) if "img.shields.io/badge/Lang" in ln), None)
    if title is None:
        failures.append(f"{name}: no level-1 title")
    elif badge is None:
        failures.append(f"{name}: no language badges")
    elif badge < title:
        failures.append(f"{name}: language badges sit above the title; they belong below it")
    return failures


def check_prose_thresholds(repo_root: Path, name: str) -> list[str]:
    text = (repo_root / name).read_text(encoding="utf-8")
    lines = text.split("\n")
    failures = []
    if len(lines) > MAX_LINES:
        failures.append(f"{name}: {len(lines)} lines > {MAX_LINES}")
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if len(rows) > MAX_TABLE_ROWS:
        failures.append(f"{name}: {len(rows)} table rows > {MAX_TABLE_ROWS}")
    bold = re.findall(r"\*\*[^*]+\*\*", text)
    if len(bold) > MAX_BOLD:
        failures.append(f"{name}: {len(bold)} bold spans > {MAX_BOLD}")
    for i, ln in enumerate(lines, 1):
        stripped = ln.strip()
        if "**" in stripped and not stripped.startswith(("-", "*", "|", ">", "#")):
            failures.append(f"{name}: line {i}: bold inside a paragraph")
    return failures


def run_checks(repo_root: Path) -> list[str]:
    names = readme_files(repo_root)
    failures = [f"missing required file: {n}" for n in names if not (repo_root / n).is_file()]
    if not (repo_root / ENGLISH_README).is_file():
        return [f"missing required file: {ENGLISH_README}"]
    if failures:
        return failures
    tracked = tracked_paths(repo_root)
    failures += check_heading_parity(repo_root, names)
    for name in names:
        failures += check_readme_paths(repo_root, name, tracked)
        failures += check_language_switch(repo_root, name, names)
        failures += check_prose_thresholds(repo_root, name)
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
    names = readme_files(args.repo_root.resolve())
    print(f"docs OK: {len(names)} README(s) -- {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
