# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Shape checks for the asset registry (pure Python, no network, CI-safe)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
from registry import ASSETS, get  # noqa: E402

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_keys_are_vendor_slash_model():
    for key in ASSETS:
        vendor, _, model = key.partition("/")
        assert vendor and model and "/" not in model, f"bad key {key!r}"


def test_entries_have_pinned_sources_and_license():
    for key, entry in ASSETS.items():
        assert entry.get("license"), f"{key}: missing license label"
        assert entry.get("sources"), f"{key}: no sources"
        for source in entry["sources"]:
            assert source["repo"].startswith("https://"), f"{key}: non-https repo"
            assert FULL_SHA.match(source["commit"]), f"{key}: commit must be a full 40-char sha"
            assert source["paths"], f"{key}: source lists no paths"
            assert "dest" in source, f"{key}: source missing dest"


def test_get_fails_loudly_on_unknown_key():
    try:
        get("nonexistent/robot")
    except KeyError as err:
        assert "Registered" in str(err)
    else:
        raise AssertionError("get() must raise KeyError for unknown keys")
