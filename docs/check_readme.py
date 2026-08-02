#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""README conformance checker (agent-rules workflows/github.md).

Run before every push that touches a README:

    python docs/check_readme.py

Checks, in the order the spec lists them:

1. Links and images are resolved against **what git has**, not the local
   filesystem. This is the one that bites: ``os.path.exists()`` goes green
   while the images were never ``git add``-ed, and GitHub shows broken images.
   Readers see the repository, not your disk.
2. Prose thresholds, benchmarked against two flagship robotics READMEs
   (Isaac Lab 141 lines / Unitree RL Lab 150 lines, both under 10 table rows
   and with single-digit bold counts).
3. Both READMEs carry the same number of level-2 headings, so the bilingual
   pair stays structurally aligned.
4. The language-switch badges are present in both files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
READMES = ("README.md", "README_zh.md")

# Thresholds from the benchmark pair. They are ceilings, not targets: a README
# that trips one has usually turned into a spec sheet.
MAX_LINES = 200
MAX_TABLE_ROWS = 10
MAX_BOLD = 9

# Local link targets that are intentionally not files in this repository.
LINK_ALLOWLIST = {"LICENSE"}


def tracked_paths() -> set[str]:
    """Everything git knows about — index plus HEAD, so a staged file counts."""
    out: set[str] = set()
    for args in (["git", "ls-files"], ["git", "ls-tree", "-r", "--name-only", "HEAD"]):
        try:
            r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            continue  # no HEAD yet (fresh repo) — the index alone is enough
        out.update(line for line in r.stdout.splitlines() if line)
    # A directory is tracked when anything under it is.
    out.update(str(Path(p).parent) for p in list(out))
    return out


def check(path: Path, tracked: set[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    problems: list[str] = []

    # ── 1. links and images, resolved against git ──────────────────────────
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    targets += re.findall(r'<img[^>]+src="([^"]+)"', text)
    for t in targets:
        if t.startswith(("http://", "https://", "#", "mailto:")):
            continue
        clean = t.split("#")[0].split("?")[0].lstrip("./")
        if not clean or clean in LINK_ALLOWLIST:
            continue
        if clean not in tracked:
            problems.append(f"link/image not tracked by git: {t}")

    # ── 2. prose thresholds ────────────────────────────────────────────────
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines > {MAX_LINES}")
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if len(rows) > MAX_TABLE_ROWS:
        problems.append(f"{len(rows)} table rows > {MAX_TABLE_ROWS}")
    bold = re.findall(r"\*\*[^*]+\*\*", text)
    if len(bold) > MAX_BOLD:
        problems.append(f"{len(bold)} bold spans > {MAX_BOLD}")

    # Bold is an item lead-in only; inside a paragraph it stops meaning anything.
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if "**" in s and not s.startswith(("-", "*", "|", ">", "#")):
            problems.append(f"line {i}: bold inside a paragraph")

    # ── 4. language switch present ─────────────────────────────────────────
    if "README_zh.md" not in text or "README.md)" not in text:
        problems.append("language-switch badges missing or incomplete")

    return problems


def main() -> int:
    tracked = tracked_paths()
    failed = False
    h2: dict[str, int] = {}

    for name in READMES:
        path = REPO / name
        if not path.is_file():
            print(f"✗ {name}: missing")
            failed = True
            continue
        h2[name] = len(re.findall(r"^## ", path.read_text(encoding="utf-8"), re.M))
        problems = check(path, tracked)
        if problems:
            failed = True
            print(f"✗ {name}")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"✓ {name}")

    # ── 3. bilingual structure alignment ───────────────────────────────────
    if len(h2) == len(READMES) and len(set(h2.values())) != 1:
        failed = True
        print(f"✗ level-2 heading counts differ between languages: {h2}")
    elif h2:
        print(f"✓ level-2 headings aligned ({next(iter(h2.values()))} each)")

    if failed:
        print("\nREADME check FAILED")
        return 1
    print("\nREADME check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
