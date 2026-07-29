#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Fetch pinned third-party robot assets listed in ``registry.py``.

Usage::

    python assets/fetch.py                     # fetch everything
    python assets/fetch.py agibot/x2           # fetch one robot
    python assets/fetch.py --force unitree/g1  # re-fetch even if up to date

For each source the script clones the upstream repo (blob-filtered, so only
needed files are downloaded), checks out the pinned commit, copies the listed
sub-paths into ``assets/<vendor>/<model>/<dest>/`` and the upstream LICENSE
text into ``assets/<vendor>/``. A ``.fetched`` marker records the pinned
commits so repeated runs are no-ops until a pin changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import ASSETS, get  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")


def _marker(dest_root: Path) -> Path:
    return dest_root / ".fetched"


def _apply_symlinks(entry: dict, dest_root: Path) -> None:
    """Create the relative symlinks a registry entry declares (idempotent).

    Runs on every invocation -- including the "already at pinned commits" skip
    path -- so adding a symlink to the registry never forces a re-download.
    Refuses to clobber anything that is not a symlink (a real file there means
    the upstream layout changed and the registry entry must be revisited).
    """
    for spec in entry.get("symlinks", ()):
        link = dest_root / spec["path"]
        target = spec["target"]
        if link.is_symlink():
            if os.readlink(link) == target:
                continue
            link.unlink()  # stale target from an older registry revision
        elif link.exists():
            raise FileExistsError(
                f"{link} exists and is not a symlink; upstream now ships this path "
                "itself -- drop the symlink entry from registry.py instead."
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        print(f"[link] {link} -> {target}")


def _pins(entry: dict) -> str:
    return "\n".join(f"{s['repo']}@{s['commit']}" for s in entry["sources"]) + "\n"


def fetch_one(key: str, force: bool = False) -> None:
    entry = get(key)
    dest_root = ASSETS_DIR / key
    if not force and _marker(dest_root).exists() and _marker(dest_root).read_text() == _pins(entry):
        _apply_symlinks(entry, dest_root)  # cheap; keeps links current without re-fetching
        print(f"[skip] {key} already at pinned commits (use --force to re-fetch)")
        return
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    for source in entry["sources"]:
        repo, commit = source["repo"], source["commit"]
        print(f"[fetch] {key}: {repo} @ {commit[:10]}")
        with tempfile.TemporaryDirectory(prefix="yanshi-assets-") as tmp:
            tmp_repo = Path(tmp) / "repo"
            # Blob-filtered clone downloads file contents lazily at checkout,
            # so we do not pull the whole mesh history of large vendor repos.
            _run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(tmp_repo)])
            # Sparse checkout: materialize only the sub-paths we need (plus
            # repo-root files, so the LICENSE text still lands). Vendor repos
            # like unitree_ros carry meshes for a dozen robots; without this
            # a checkout would download all of them.
            if source["paths"] != ["."]:
                _run(["git", "sparse-checkout", "set", *source["paths"]], cwd=tmp_repo)
            _run(["git", "checkout", commit], cwd=tmp_repo)

            for rel in source["paths"]:
                src = (tmp_repo / rel).resolve()
                if not src.exists():
                    raise FileNotFoundError(f"{key}: path {rel!r} not found in {repo} @ {commit[:10]}")
                dest = dest_root / source["dest"] if source["dest"] != "." else dest_root
                if rel == ".":
                    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
                else:
                    shutil.copytree(src, dest / src.name, dirs_exist_ok=True)

            # Upstream LICENSE text is the authoritative license for these files.
            vendor_dir = ASSETS_DIR / key.split("/")[0]
            found_license = False
            for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENCE", "COPYING"):
                lic = tmp_repo / name
                if lic.exists():
                    shutil.copyfile(lic, vendor_dir / f"LICENSE-{Path(repo).name}.txt")
                    found_license = True
                    break
            if not found_license:
                print(f"[warn] {key}: no LICENSE file found in {repo} -- check upstream manually")

    _apply_symlinks(entry, dest_root)
    _marker(dest_root).write_text(_pins(entry))
    print(f"[done] {key} -> {dest_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*", default=[], help="vendor/model keys (default: all registered)")
    parser.add_argument("--force", action="store_true", help="re-fetch even if the marker matches the pins")
    args = parser.parse_args()

    keys = args.keys or sorted(ASSETS)
    for key in keys:
        fetch_one(key, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
