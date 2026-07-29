# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Registry of third-party robot model assets, pinned to upstream commits.

Assets are NEVER committed to this repository: every vendor ships under its
own license, and meshes would bloat the git history (a known failure mode in
comparable projects). Instead, ``python assets/fetch.py`` downloads exactly
the pinned commit of each upstream repo, copies the needed sub-paths into
``assets/<vendor>/<model>/``, and drops the upstream LICENSE text next to
them. The LICENSE file fetched from upstream is always the authoritative
license text; the ``license`` field below is only a convenience label.

Pin update policy: bumping a ``commit`` is a foundation change -- record why
in the maintainer decision ledger before changing it.
"""

# Each entry: "<vendor>/<model>" -> {sources: [...], license: SPDX label}
# A source copies each of `paths` (relative to the upstream repo root, after
# checking out `commit`) into assets/<vendor>/<model>/<dest>/.
#
# Optional per-entry `symlinks`: relative symlinks fetch.py creates inside
# assets/<vendor>/<model>/ AFTER copying (and re-ensures on every run, even
# when the pin marker says "up to date"). Use this when an upstream file
# references a directory layout the upstream repo itself does not ship --
# bridging with a link keeps the vendor files byte-identical (project rule:
# never edit vendor assets).
ASSETS: dict[str, dict] = {
    "unitree/g1": {
        "license": "BSD-3-Clause",
        "sources": [
            {
                # URDF + meshes (29-DoF deploy variant lives in g1_description)
                "repo": "https://github.com/unitreerobotics/unitree_ros",
                "commit": "f3772ce54c56ef2d34c6aee8100bc768896c7d19",  # HEAD @ 2026-07-29
                "paths": ["robots/g1_description"],
                "dest": "urdf",
            },
            {
                # Paired MuJoCo MJCF from the official sim repo
                "repo": "https://github.com/unitreerobotics/unitree_mujoco",
                "commit": "ae6a8403e272733e9996ef59990880330496177f",  # HEAD @ 2026-07-29
                "paths": ["unitree_robots/g1"],
                "dest": "mjcf",
            },
        ],
    },
    "agibot/x2": {
        "license": "MulanPSL-2.0",
        "sources": [
            {
                # Official X2 release: URDF + MJCF + meshes in one repo
                # (published 2026-07-29). We pin the v1.4.0 model directory,
                # the newest revision (upgraded motors / reinforced frame).
                "repo": "https://github.com/AgibotTech/agibot_x2_urdf",
                "commit": "77f43eb0904dae4c48ccd9154fee824f8ffd4d38",  # HEAD @ 2026-07-29
                "paths": ["X2_URDF-v1.4.0"],
                "dest": ".",
            },
        ],
    },
    "berkeley/humanoid_lite": {
        "license": "CC-BY-SA-4.0",
        "sources": [
            {
                # Official assets repo: URDF + MJCF + USD from one CAD source
                "repo": "https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets",
                "commit": "fc90fedd008b1e56a22e3c5221548d6b24f49707",  # HEAD @ 2026-07-29
                "paths": ["."],
                "dest": ".",
            },
        ],
        "symlinks": [
            {
                # The vendor MJCF (mjcf/berkeley_humanoid_lite.xml) declares
                # meshdir="assets" and references "merged/<name>.stl", but the
                # repo ships those exact 26 STL files at ../meshes/ instead
                # (verified 2026-07-29: every referenced basename exists there,
                # and MuJoCo 3.10 fails to compile without the link). Upstream's
                # own onshape-export workspace recreates the assets/merged
                # layout; the published tree alone cannot compile.
                "path": "data/robots/berkeley_humanoid/berkeley_humanoid_lite/mjcf/assets/merged",
                "target": "../../meshes",
            },
        ],
    },
}


def get(key: str) -> dict:
    """Return the registry entry for ``<vendor>/<model>``; fail loudly otherwise."""
    if key not in ASSETS:
        known = ", ".join(sorted(ASSETS))
        raise KeyError(f"Unknown asset key {key!r}. Registered: {known}")
    return ASSETS[key]
