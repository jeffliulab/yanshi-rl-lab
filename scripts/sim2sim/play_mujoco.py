#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Successor of the predecessor stack's deploy_mujoco.py (unitree-g1-locomotion,
# MIT, same author), rebuilt on the contract-v2 runtime. No recording in this
# variant: offscreen video needs MUJOCO_GL=egl, which uses the GPU -- this
# script is part of the pure-CPU verification path.

"""Single-command MuJoCo deployment: run one policy at one fixed command.

Usage (pure CPU)::

    CUDA_VISIBLE_DEVICES="" python scripts/sim2sim/play_mujoco.py \
        --policy <policy.onnx> --contract <contract.json> \
        --scene <scene.xml> --vx 0.6 [--vy 0] [--wz 0] [--seconds 12]

Prints a one-line odometry summary every simulated second and a final
summary (net displacement, path, turn, radius, fall time if any).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "source" / "yanshi_rl_lab"))

from yanshi_rl_lab.deploy.contract import ContractV2  # noqa: E402
from yanshi_rl_lab.deploy.runtime import DeployLoop  # noqa: E402

# Default run length: matches the predecessor stack's standard deploy check
# (12 s at vx=0.8 was its regression probe).
DEFAULT_SECONDS = 12.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="policy.onnx")
    ap.add_argument("--contract", required=True, help="schema-v2 contract JSON")
    ap.add_argument("--scene", required=True, help="MuJoCo scene XML matching the contract joints")
    ap.add_argument("--vx", type=float, default=0.0, help="forward velocity command (m/s)")
    ap.add_argument("--vy", type=float, default=0.0, help="lateral velocity command (m/s)")
    ap.add_argument("--wz", type=float, default=0.0, help="yaw-rate command (rad/s)")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="run length (s)")
    args = ap.parse_args()

    contract = ContractV2.from_json(args.contract)
    for warning in contract.validate():
        print(f"[contract] note: {warning}")

    loop = DeployLoop(args.policy, contract, args.scene)
    loop.reset(cmd=[args.vx, args.vy, args.wz])
    steps_per_second = int(round(1.0 / loop.c.timing.policy_dt_s))
    total_steps = int(args.seconds / loop.c.timing.policy_dt_s)
    print(
        f"cmd = (vx {args.vx}, vy {args.vy}, wz {args.wz}) for {args.seconds:.1f} s "
        f"({total_steps} policy steps @ {1.0 / loop.c.timing.policy_dt_s:.0f} Hz)"
    )
    for step in range(total_steps):
        loop.step()
        if (step + 1) % steps_per_second == 0:
            radius = "inf" if loop.odo.radius_m == float("inf") else f"{loop.odo.radius_m:.2f}"
            print(
                f"  t={loop.elapsed_s:5.1f}s  disp {loop.odo.net_displacement_m:6.2f} m  "
                f"turn {loop.odo.turn_deg:7.1f} deg  radius {radius} m  "
                f"h {loop.rig.height:.2f} m  tilt {loop.tilt:.2f} rad"
            )
        if loop.fell_at is not None:
            break

    print("-" * 64)
    fell = f"FELL at {loop.fell_at:.1f} s" if loop.fell_at is not None else "no fall"
    radius = "inf" if loop.odo.radius_m == float("inf") else f"{loop.odo.radius_m:.2f}"
    print(
        f"final: disp {loop.odo.net_displacement_m:.2f} m | path {loop.odo.path_len:.2f} m | "
        f"turn {loop.odo.turn_deg:.1f} deg | radius {radius} m | {fell}"
    )
    return 0 if loop.fell_at is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
