# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""``yanshi`` command-line interface (thin wrapper, pure CPU).

Subcommands:

- ``yanshi doctor``          -- environment self-check. Every check prints
  PASS/FAIL/WARN plus a one-line fix suggestion; no check can crash the
  doctor (that would defeat its purpose on exactly the broken machines it
  exists for).
- ``yanshi assets fetch``    -- wraps ``assets/fetch.py`` (pinned upstream
  robot models).
- ``yanshi assets status``   -- fetch state of every registered robot.

``yanshi list`` (task listing) is NOT implemented here: enumerating gym
tasks requires booting Isaac Sim; use ``scripts/list_envs.py`` inside the
isaaclab environment instead.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from yanshi_rl_lab.robots.profile import ASSETS_ROOT_ENV_VAR, assets_root

# Minimum Python this project supports; mirrors python_requires in
# source/yanshi_rl_lab/setup.py (keep the two in sync).
MIN_PYTHON = (3, 10)

# Repository root when running from a source checkout (cli.py lives at
# <repo>/source/yanshi_rl_lab/yanshi_rl_lab/cli.py). Asset tooling lives in
# the repo, not in the installed package; on a non-editable install these
# paths may not exist and the commands say so instead of guessing.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_FETCH_SCRIPT = _REPO_ROOT / "assets" / "fetch.py"
_REGISTRY_FILE = _REPO_ROOT / "assets" / "registry.py"


# ----------------------------------------------------------------- reporting
_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status: str, name: str, detail: str, fix: str | None = None) -> None:
        self.rows.append((status, name, detail, fix))

    def print(self) -> int:
        worst = "PASS"
        for status, name, detail, fix in self.rows:
            print(f"[{status:4s}] {name}: {detail}")
            if fix and status != "PASS":
                print(f"       fix: {fix}")
            if _STATUS_ORDER[status] > _STATUS_ORDER[worst]:
                worst = status
        print()
        counts = {s: sum(1 for r in self.rows if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
        print(f"doctor summary: {counts['PASS']} pass / {counts['WARN']} warn / {counts['FAIL']} fail")
        return 1 if counts["FAIL"] else 0


def _guard(report: Report, name: str, fn) -> None:
    """Run one check; an unexpected exception becomes a FAIL row, never a crash."""
    try:
        fn(report)
    except Exception as exc:  # noqa: BLE001 -- doctor must never die on a check
        report.add("FAIL", name, f"check crashed: {exc}", "report this as a doctor bug")


# ------------------------------------------------------------------- checks
def _check_python(report: Report) -> None:
    version = sys.version_info
    detail = f"{version.major}.{version.minor}.{version.micro} ({sys.executable})"
    if (version.major, version.minor) >= MIN_PYTHON:
        report.add("PASS", "python", detail)
    else:
        report.add(
            "FAIL",
            "python",
            detail,
            f"use Python >= {'.'.join(map(str, MIN_PYTHON))} (conda env recommended)",
        )


def _check_glibc(report: Report) -> None:
    lib, version = platform.libc_ver()
    if lib:
        report.add("PASS", "glibc", f"{lib} {version}")
    else:
        report.add("WARN", "glibc", "not detected (non-glibc platform?)",
                   "Isaac Sim official builds target glibc-based Linux")


def _check_nvidia_driver(report: Report) -> None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        report.add(
            "FAIL",
            "nvidia driver",
            "nvidia-smi not found",
            "install the NVIDIA driver (needed for training; MuJoCo sim2sim runs CPU-only without it)",
        )
        return
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        report.add("FAIL", "nvidia driver", "nvidia-smi timed out", "driver/GPU in a bad state; reboot or reload the driver")
        return
    if out.returncode != 0:
        report.add("FAIL", "nvidia driver", f"nvidia-smi failed: {out.stderr.strip()}",
                   "driver not loaded? check `dmesg | grep -i nvidia`")
        return
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not lines:
        report.add("FAIL", "nvidia driver", "nvidia-smi reports no GPU", "check the GPU is visible on the PCI bus")
        return
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts else "?"
        driver = parts[1] if len(parts) > 1 else "?"
        memory = parts[2] if len(parts) > 2 else "?"
        report.add("PASS", "gpu", f"{name} | driver {driver} | {memory}")


def _check_vulkan(report: Report) -> None:
    if shutil.which("vulkaninfo"):
        report.add("PASS", "vulkan", "vulkaninfo present")
    else:
        report.add(
            "WARN",
            "vulkan",
            "vulkaninfo not found -- cannot verify the Vulkan loader",
            "install vulkan-tools; note: a working nvidia-smi with a broken Vulkan stack is "
            "the MOST COMMON false-negative when Isaac Sim fails to render",
        )


def _make_import_check(module: str, why: str):
    def check(report: Report) -> None:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- any import error is a finding, not a crash
            # WARN, not FAIL: a missing package is an installable gap, not a
            # broken machine (the fix column says which workflow needs it).
            report.add("WARN", module, f"not importable ({type(exc).__name__}: {exc})", why)
            return
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                import importlib.metadata as md

                version = md.version(module)
            except Exception:  # noqa: BLE001
                version = "unknown version"
        report.add("PASS", module, str(version))

    return check


def _check_assets(report: Report) -> None:
    root = assets_root()
    source = (
        f"${ASSETS_ROOT_ENV_VAR}" if os.environ.get(ASSETS_ROOT_ENV_VAR) else "default (<repo>/assets)"
    )
    if root.is_dir():
        report.add("PASS", "assets root", f"{root} [{source}]")
    else:
        report.add("WARN", "assets root", f"{root} does not exist [{source}]",
                   f"run `yanshi assets fetch` or point {ASSETS_ROOT_ENV_VAR} at an assets tree")
        return

    registry = _load_registry()
    if registry is None:
        report.add("WARN", "assets registry", f"{_REGISTRY_FILE} not found (installed outside a source checkout?)",
                   "run asset commands from a git checkout of yanshi-rl-lab")
        return
    for key in sorted(registry.ASSETS):
        state = _fetch_state(registry, root, key)
        if state == "fetched":
            report.add("PASS", f"assets {key}", "fetched at pinned commits")
        elif state == "stale":
            report.add("WARN", f"assets {key}", "fetched, but pins changed since",
                       f"re-fetch: python assets/fetch.py --force {key}")
        else:
            report.add("WARN", f"assets {key}", "not fetched",
                       f"fetch: python assets/fetch.py {key}")


def _load_registry():
    if not _REGISTRY_FILE.exists():
        return None
    spec = importlib.util.spec_from_file_location("yanshi_assets_registry", _REGISTRY_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins(entry: dict) -> str:
    # Same marker format assets/fetch.py writes.
    return "\n".join(f"{s['repo']}@{s['commit']}" for s in entry["sources"]) + "\n"


def _fetch_state(registry, root: Path, key: str) -> str:
    marker = root / key / ".fetched"
    if not marker.exists():
        return "missing"
    return "fetched" if marker.read_text() == _pins(registry.ASSETS[key]) else "stale"


def cmd_doctor(_args) -> int:
    report = Report()
    _guard(report, "python", _check_python)
    _guard(report, "glibc", _check_glibc)
    _guard(report, "nvidia driver", _check_nvidia_driver)
    _guard(report, "vulkan", _check_vulkan)
    _guard(report, "isaaclab", _make_import_check(
        "isaaclab", "training needs Isaac Lab (pip install isaaclab==2.3.2* inside the training env)"))
    _guard(report, "mujoco", _make_import_check(
        "mujoco", "sim2sim verification needs MuJoCo (pip install mujoco)"))
    _guard(report, "onnxruntime", _make_import_check(
        "onnxruntime", "policy inference needs onnxruntime (pip install onnxruntime)"))
    _guard(report, "assets", _check_assets)
    return report.print()


# -------------------------------------------------------------------- assets
def cmd_assets(args) -> int:
    if not _FETCH_SCRIPT.exists():
        print(
            f"assets tooling not found at {_FETCH_SCRIPT} -- the `yanshi assets` commands "
            "need a source checkout of yanshi-rl-lab (the fetcher and registry live in the "
            "repo, not in the installed package).",
            file=sys.stderr,
        )
        return 1
    if args.assets_command == "fetch":
        cmd = [sys.executable, str(_FETCH_SCRIPT), *args.keys]
        if args.force:
            cmd.append("--force")
        return subprocess.run(cmd).returncode
    # status
    registry = _load_registry()
    root = assets_root()
    print(f"assets root: {root}")
    for key in sorted(registry.ASSETS):
        state = _fetch_state(registry, root, key)
        license_name = registry.ASSETS[key]["license"]
        print(f"  {key:28s} {state:8s} (upstream license: {license_name})")
    return 0


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="yanshi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="environment self-check (pure CPU, never crashes)")
    doctor.set_defaults(func=cmd_doctor)

    assets = sub.add_parser("assets", help="fetch / inspect pinned robot model assets")
    assets_sub = assets.add_subparsers(dest="assets_command", required=True)
    fetch = assets_sub.add_parser("fetch", help="fetch pinned upstream assets (wraps assets/fetch.py)")
    fetch.add_argument("keys", nargs="*", default=[], help="vendor/model keys (default: all)")
    fetch.add_argument("--force", action="store_true", help="re-fetch even if up to date")
    fetch.set_defaults(func=cmd_assets)
    status = assets_sub.add_parser("status", help="show fetch state of every registered robot")
    status.set_defaults(func=cmd_assets)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
