#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Render one thumbnail per registered robot configuration, for ROBOTS_INTRO.md.

    python scripts/tools/render_robot_thumbs.py            # all registered robots
    python scripts/tools/render_robot_thumbs.py unitree/g1/dof23

Every input comes from the robot profile: the scene file, the standing pose,
the spawn height. Nothing here names an asset path, and adding a robot needs
no edit to this script.

Framing is measured, not guessed. The launch robots differ by more than 2x in
height (Berkeley Humanoid Lite is a ~0.6 m machine, the G1 is ~1.3 m), so a
fixed camera distance would render one of them as a speck. The camera backs
off proportionally to each robot's own bounding box, which makes the thumbnails
comparable as portraits even though the robots are not the same size. Angle
follows the project recording standard: front three-quarter view, slight
downward tilt, whole body in frame.

Rendering uses MuJoCo offscreen with ``MUJOCO_GL=egl``, which does use the GPU
graphics context even though it never appears in ``nvidia-smi
--query-compute-apps`` (that lists CUDA contexts only). It is a few hundred MB
for a fraction of a second per robot.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "source" / "yanshi_rl_lab"))

# Thumbnail geometry. 4:3 portrait-ish framing: humanoids are tall, and a wide
# frame either crops the feet or leaves the robot a sliver in the middle (the
# hero-GIF lesson, recorded in the project recording standard).
WIDTH, HEIGHT = 480, 640

# Camera angle, shared with the hero recorder so a thumbnail and a clip of the
# same robot look like the same robot.
AZIMUTH_DEG = 135.0
ELEVATION_DEG = -15.0

# Fraction of the frame height the robot should occupy. Leaves ~13% margin
# above the head and below the feet -- the standard's ">= 10% margin" rule with
# a little slack for the perspective spread of a wide stance.
TARGET_FILL = 0.74

# MuJoCo's free camera frames roughly `distance` * model extent; this converts
# "how tall is the robot" into "how far back must the camera sit". Derived
# empirically once against the G1 and reused for every robot rather than
# per-robot tuning, so the robots stay comparable.
DISTANCE_PER_METRE = 1.55


def default_pose_qpos(mujoco, model, profile):
    """Standing pose from the profile, applied to whichever joints exist."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for pattern, value in profile.default_joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[i]] = value
                break
    mujoco.mj_forward(model, data)
    return data


def articulated_bodies(mujoco, model) -> list[int]:
    """Bodies that actually hang off the robot's floating base.

    Not merely "everything except the world". Unitree's g1_23dof.xml keeps a
    29-motor DDS layout over a 23-DoF body by parking the six absent joints on
    massless bodies at ``pos="0 0 20"``, detached from the tree. Measuring the
    frame against those puts the "robot" 20 m tall and renders it as a speck --
    which is what happened the first time this script ran. Anything that asks
    "how big is this robot" has to ask the kinematic tree, not the body list.
    """
    keep = []
    for body in range(1, model.nbody):
        node, chain = body, []
        while node != 0:
            chain.append(node)
            node = model.body_parentid[node]
        if any(
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for b in chain
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
        ):
            keep.append(body)
    return keep


def robot_extent(mujoco, model, data) -> tuple[float, float]:
    """(height, centre-z) of the robot itself -- no floor, no detached bodies."""
    bodies = articulated_bodies(mujoco, model)
    if not bodies:
        raise ValueError("no body is attached to a floating base; is this a fixed-base model?")
    zs = [data.xpos[i][2] for i in bodies]
    return max(zs) - min(zs), (max(zs) + min(zs)) / 2.0


def render_one(key: str, profile, out_dir: Path) -> Path:
    import mujoco

    scene = profile.asset_path("scene_mjcf" if profile.scene_mjcf else "mjcf")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = default_pose_qpos(mujoco, model, profile)
    height, centre_z = robot_extent(mujoco, model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth, camera.elevation = AZIMUTH_DEG, ELEVATION_DEG
    camera.lookat[:] = [data.xpos[1][0], data.xpos[1][1], centre_z]
    camera.distance = height / TARGET_FILL * DISTANCE_PER_METRE

    # MuJoCo's offscreen framebuffer defaults to 640x480 and refuses a taller
    # image. Raise it on the loaded model object -- editing the vendor XML to
    # add a <visual><global offheight=...> clause is banned (never touch
    # upstream assets), and this achieves the same thing per run.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)

    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    renderer.update_scene(data, camera)
    pixels = renderer.render()
    renderer.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key.replace('/', '-')}.png"
    _write_png(pixels, path)
    return path


def _write_png(pixels, path: Path) -> None:
    """Write a PNG without adding a dependency (Pillow is not required here)."""
    import struct
    import zlib

    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robots", nargs="*", help="robot keys (default: every registered one)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "docs" / "media" / "robots",
        help="where thumbnails are written (default: docs/media/robots/)",
    )
    args = ap.parse_args()

    from yanshi_rl_lab.robots import registry

    keys = args.robots or registry.keys()
    for key in keys:
        profile = registry.get(key)
        try:
            path = render_one(key, profile, args.out_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {key}: {exc}", file=sys.stderr)
            continue
        size_kb = path.stat().st_size / 1024
        print(f"[done] {key:36s} -> {path.relative_to(_REPO)}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
