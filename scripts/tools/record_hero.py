#!/usr/bin/env python3
# Continuous-deployment hero video recorder (acceptance tooling, not tracked).
# One reset, one uncut run; the velocity command switches live on a schedule.
# v2 (2026-07-31): walk-first ordering + on-frame HUD (Jeff's v1 review:
# "the first 10+ s look dead" -- v1 front-loaded 2 s stand + 16 s of wz=0.2
# turning, the slowest content X2 has; wz=0.2 rad/s IS the trained turn cap,
# so the HUD now labels every command to make slow segments read as intent).
#   0-1 s stand | 1-9 s walk (vx=0.6) | 9-15 s slow walk (vx=0.3)
#   | 15-16 s stand | 16-24 s turn left (wz=+0.2) | 24-32 s turn right (wz=-0.2)
#   | 32-35 s walk (vx=0.6)
# 35 s total, mirroring the G1 parity hero loop.
#
# Rendering: MuJoCo offscreen, MUJOCO_GL=egl (uses the GPU graphics context;
# invisible to nvidia-smi --query-compute-apps). 1080p, 50 fps = policy rate.

import argparse
import os
import sys
from pathlib import Path

# scripts/tools/record_hero.py -> repo root is three levels up.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "source" / "yanshi_rl_lab"))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from yanshi_rl_lab.deploy.contract import ContractV2  # noqa: E402
from yanshi_rl_lab.deploy.runtime import DeployLoop  # noqa: E402

SCHEDULE = [  # (segment_end_s, [vx, vy, wz], label)
    (1.0, [0.0, 0.0, 0.0], "stand"),
    (9.0, [0.6, 0.0, 0.0], "walk vx=0.6 m/s"),
    (15.0, [0.3, 0.0, 0.0], "slow walk vx=0.3 m/s"),
    (16.0, [0.0, 0.0, 0.0], "stand"),
    (24.0, [0.0, 0.0, 0.2], "turn left wz=+0.2 rad/s (trained cap)"),
    (32.0, [0.0, 0.0, -0.2], "turn right wz=-0.2 rad/s (trained cap)"),
    (35.0, [0.6, 0.0, 0.0], "walk vx=0.6 m/s"),
]
W, H = 1920, 1080

# HUD font. DejaVu ships with most Linux distros and with Pillow itself; the
# env var is the escape hatch for boxes that have neither. Falls back to
# Pillow's bitmap default rather than dying -- a HUD with ugly text is still a
# usable review video, a crash 30 s into recording is not.
HUD_FONT_ENV_VAR = "YANSHI_HUD_FONT"
_FONT_CANDIDATES = (
    os.environ.get(HUD_FONT_ENV_VAR),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",  # resolved via Pillow's bundled fonts / fontconfig
)


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if not path:
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    print(
        f"[record_hero] no TrueType font found (set ${HUD_FONT_ENV_VAR} to one); "
        "HUD falls back to Pillow's bitmap default",
        file=sys.stderr,
    )
    return ImageFont.load_default()


_FONT = _load_font(44)
_FONT_SMALL = _load_font(34)


def draw_hud(frame: np.ndarray, t: float, cmd: list[float], label: str) -> np.ndarray:
    """Burn the live command + timestamp into the top-left corner."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    lines = [
        (f"t = {t:4.1f} s", _FONT_SMALL),
        (f"cmd: {label}", _FONT),
        (f"vx={cmd[0]:+.1f}  vy={cmd[1]:+.1f}  wz={cmd[2]:+.1f}", _FONT_SMALL),
    ]
    y = 24
    for text, font in lines:
        d.text((26, y), text, font=font, fill=(0, 0, 0), stroke_width=0)
        d.text((24, y - 2), text, font=font, fill=(255, 255, 255), stroke_width=0)
        y += font.size + 14
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--distance", type=float, default=3.2)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-15.0)
    args = ap.parse_args()

    contract = ContractV2.from_json(args.contract)
    loop = DeployLoop(args.policy, contract, args.scene)
    loop.rig.model.vis.global_.offwidth = W
    loop.rig.model.vis.global_.offheight = H
    dt = loop.c.timing.policy_dt_s
    fps = int(round(1.0 / dt))
    total_s = SCHEDULE[-1][0]

    renderer = mujoco.Renderer(loop.rig.model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, quality=8)

    loop.reset(cmd=[0.0, 0.0, 0.0])
    seg_idx = 0
    n_steps = int(round(total_s / dt))
    # per-segment odometer snapshots for the log
    seg_log = []
    try:
        for k in range(n_steps):
            t = k * dt
            while seg_idx < len(SCHEDULE) - 1 and t >= SCHEDULE[seg_idx][0]:
                seg_log.append((SCHEDULE[seg_idx][0], loop.odo.net_displacement_m, loop.odo.turn_deg))
                seg_idx += 1
            _, cmd, label = SCHEDULE[seg_idx]
            loop.cmd[:] = cmd
            loop.step()
            cam.lookat[:] = loop.rig.data.xpos[loop.rig.root_bid]
            renderer.update_scene(loop.rig.data, cam)
            writer.append_data(draw_hud(renderer.render(), t, cmd, label))
            if loop.fell_at is not None:
                print(f"[record] FELL at {loop.fell_at:.2f}s -- stopping early")
                break
    finally:
        writer.close()
        renderer.close()
    print(f"[record] wrote {args.out}")
    print(f"[record] fell_at={loop.fell_at}  final disp={loop.odo.net_displacement_m:.2f} m  "
          f"turn={loop.odo.turn_deg:.1f} deg  path={loop.odo.path_len:.2f} m")
    for end_t, disp, turn in seg_log:
        print(f"[record]   t={end_t:5.1f}s  cum_disp={disp:6.2f} m  cum_turn={turn:7.1f} deg")


if __name__ == "__main__":
    main()
