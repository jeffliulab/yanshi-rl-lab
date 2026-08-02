# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Lookup of registered robots by their canonical ``<vendor>/<model>/<variant>`` key.

Discovery is by directory, not by a hand-maintained table: every
``robots/<vendor>/<model>/<variant>/profile.py`` that defines a module-level
:class:`RobotProfile` is registered automatically. Adding a robot therefore
touches no central file -- the same property the task layer already has
(``tasks/__init__.py`` auto-imports config packages), and the reason
``scripts/tools/new_robot.py`` can scaffold a robot without editing anything
shared.

Pure Python on purpose (like ``profile.py``): the MuJoCo gate runner resolves
a robot's scene through this module on machines with no Isaac Lab installed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from yanshi_rl_lab.robots.profile import RobotProfile

_PACKAGE = __name__.rsplit(".", 1)[0]  # yanshi_rl_lab.robots
_ROOT = Path(__file__).resolve().parent

# Directories under robots/ that are code, not vendors.
_NOT_VENDORS = {"__pycache__"}

_cache: dict[str, RobotProfile] | None = None


def _iter_profile_modules():
    """Yield ``(dotted_module, key)`` for every vendor/model/variant/profile.py."""
    for vendor_dir in sorted(p for p in _ROOT.iterdir() if p.is_dir()):
        if vendor_dir.name in _NOT_VENDORS:
            continue
        for model_dir in sorted(p for p in vendor_dir.iterdir() if p.is_dir()):
            if model_dir.name in _NOT_VENDORS:
                continue
            for variant_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                if variant_dir.name in _NOT_VENDORS:
                    continue
                if not (variant_dir / "profile.py").is_file():
                    continue
                key = f"{vendor_dir.name}/{model_dir.name}/{variant_dir.name}"
                dotted = f"{_PACKAGE}.{vendor_dir.name}.{model_dir.name}.{variant_dir.name}.profile"
                yield dotted, key


def _discover() -> dict[str, RobotProfile]:
    found: dict[str, RobotProfile] = {}
    for dotted, key in _iter_profile_modules():
        module = importlib.import_module(dotted)
        profiles = [v for v in vars(module).values() if isinstance(v, RobotProfile)]
        if not profiles:
            raise ImportError(
                f"{dotted} defines no RobotProfile. Every "
                "robots/<vendor>/<model>/<variant>/profile.py must expose exactly one."
            )
        if len(profiles) > 1:
            raise ImportError(
                f"{dotted} defines {len(profiles)} RobotProfile objects; expected exactly one."
            )
        profile = profiles[0]
        # The directory IS the identity. A mismatch means someone renamed a
        # folder without touching the profile (or vice versa), and every
        # downstream key -- task ID, leaderboard directory -- would silently
        # disagree with where the file lives.
        if profile.key != key:
            raise ImportError(
                f"{dotted} declares identity {profile.key!r} but lives at {key!r}. "
                "The directory path and the profile's vendor/model/variant must agree."
            )
        found[key] = profile
    return found


def all_profiles() -> dict[str, RobotProfile]:
    """Every registered robot, keyed by ``<vendor>/<model>/<variant>``."""
    global _cache
    if _cache is None:
        _cache = _discover()
    return dict(_cache)


def keys() -> list[str]:
    """Sorted list of registered robot keys."""
    return sorted(all_profiles())


def get(key: str) -> RobotProfile:
    """Return one robot's profile; fail loudly with the known keys listed."""
    profiles = all_profiles()
    if key not in profiles:
        known = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(
            f"Unknown robot {key!r}. Registered: {known}. "
            "Keys are <vendor>/<model>/<variant> -- the variant segment is mandatory."
        )
    return profiles[key]
