#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Declarative-gate runner, generalizing the predecessor stack's per-experiment
# gate scripts (unitree-g1-locomotion gate_turn_v2.py, MIT, same author): the
# gate table that used to be a frozen tuple in code now lives in
# benchmark/gates/*.yaml -- code carries ZERO thresholds.

"""Run a declarative gate file against one policy + contract in MuJoCo.

Usage (pure CPU; keep MuJoCo and onnxruntime off the GPU)::

    CUDA_VISIBLE_DEVICES="" python scripts/sim2sim/run_gates.py \
        --gates benchmark/gates/velocity-flat-turn.yaml \
        --contract <contract.json> --policy <policy.onnx> \
        --scene <scene.xml> [--out results.json]

Exit codes: 0 = all gates passed; 1 = at least one gate failed;
2 = the per-foot contact veto fired (policy void, gate numbers do not count).
Metrics are necessary conditions only: final acceptance still includes a
human watching continuous video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "source" / "yanshi_rl_lab"))

from yanshi_rl_lab.deploy.contract import ContractV2  # noqa: E402

# NOTE: yanshi_rl_lab.deploy.runtime (mujoco + onnxruntime) is imported inside
# run_one_gate() so that the gate-file loader stays importable for pure-Python
# schema tests on machines without the deploy stack.

# Gate-file schema version this runner understands.
GATES_SCHEMA_VERSION = 1

# direction string (from YAML) -> pass predicate
_DIRECTIONS = {
    ">=": lambda got, line: got >= line,
    "<=": lambda got, line: got <= line,
}


def load_gate_file(path) -> dict:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != GATES_SCHEMA_VERSION:
        raise ValueError(
            f"Gate file schema_version {spec.get('schema_version')!r} != {GATES_SCHEMA_VERSION}."
        )
    for required in ("protocol", "gates", "veto"):
        if required not in spec:
            raise ValueError(f"Gate file is missing the {required!r} section.")
    for gate in spec["gates"]:
        for key in ("name", "command", "metric", "threshold", "direction", "unit"):
            if key not in gate:
                raise ValueError(f"Gate {gate.get('name', '?')!r} is missing {key!r}.")
        if gate["direction"] not in _DIRECTIONS:
            raise ValueError(
                f"Gate {gate['name']!r} direction {gate['direction']!r} not in {sorted(_DIRECTIONS)}."
            )
    return spec


def run_one_gate(policy: str, contract: ContractV2, scene: str, command: dict, seconds: float) -> dict:
    """One deterministic closed-loop run; returns every metric the gate file
    may reference plus the veto inputs."""
    from yanshi_rl_lab.deploy.runtime import DeployLoop, FootContactStats

    loop = DeployLoop(policy, contract, scene)
    feet = FootContactStats(loop.rig)
    loop.reset(cmd=[command.get("vx", 0.0), command.get("vy", 0.0), command.get("wz", 0.0)])
    feet.reset()
    for _ in range(int(seconds / loop.c.timing.policy_dt_s)):
        loop.step()
        feet.update()
    return {
        "net_displacement_m": loop.odo.net_displacement_m,
        "turn_deg": abs(loop.odo.turn_deg),
        "radius_m": loop.odo.radius_m,
        "path_m": loop.odo.path_len,
        "fell_at": loop.fell_at,
        "final_height_m": loop.rig.height,
        "contact_frac": feet.contact_frac,
        "foot_names": feet.names,
        "asymmetry": feet.asymmetry,
    }


def evaluate(spec: dict, policy: str, contract: ContractV2, scene: str) -> dict:
    seconds = float(spec["protocol"]["seconds"])
    veto_cfg = spec["veto"]
    results = []
    veto_hits = []
    for gate in spec["gates"]:
        r = run_one_gate(policy, contract, scene, gate["command"], seconds)
        got = r[gate["metric"]]
        passed = _DIRECTIONS[gate["direction"]](got, float(gate["threshold"]))
        if r["fell_at"] is not None:
            passed = False
        veto = (
            min(r["contact_frac"]) < float(veto_cfg["min_contact_frac"])
            or r["asymmetry"] > float(veto_cfg["max_asymmetry"])
        )
        if veto:
            veto_hits.append(gate["name"])
        results.append({"gate": gate, "measured": r, "value": got, "passed": bool(passed), "veto": veto})
    return {"seconds": seconds, "results": results, "veto_hits": veto_hits}


def _fmt(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.2f}"


def print_report(report: dict, spec: dict) -> None:
    for entry in report["results"]:
        gate, r = entry["gate"], entry["measured"]
        cmd = gate["command"]
        fell = f"FELL at {r['fell_at']:.1f}s" if r["fell_at"] is not None else "no fall"
        contact = " / ".join(f"{n}={f:.2f}" for n, f in zip(r["foot_names"], r["contact_frac"]))
        print(f"[{gate['name']}] vx={cmd.get('vx', 0)} vy={cmd.get('vy', 0)} wz={cmd.get('wz', 0)}")
        print(
            f"    disp {r['net_displacement_m']:.2f} m | path {r['path_m']:.2f} m | "
            f"turn {r['turn_deg']:.1f} deg | radius {_fmt(r['radius_m'])} m | {fell}"
        )
        print(f"    contact {contact} | asymmetry {r['asymmetry']:.2f}" + ("   VETO" if entry["veto"] else ""))
        print(
            f"    gate: {gate['metric']} {gate['direction']} {gate['threshold']}{gate['unit']}"
            f"  ->  measured {_fmt(entry['value'])}{gate['unit']}   "
            + ("PASS" if entry["passed"] else "FAIL")
        )
        print()

    print("=" * 64)
    if report["veto_hits"]:
        veto_cfg = spec["veto"]
        print("VETO: suspicious per-foot contact (possible airborne leg / limp).")
        print(f"      Gates that tripped it: {report['veto_hits']}")
        print(
            f"      (criteria: any contact fraction < {veto_cfg['min_contact_frac']} "
            f"or asymmetry > {veto_cfg['max_asymmetry']})"
        )
        print("      Policy is VOID; the gate numbers above do not count.")
    else:
        passed = sum(e["passed"] for e in report["results"])
        total = len(report["results"])
        verdict = "ALL GATES PASSED" if passed == total else "NOT PASSED (report as-is; no 'almost')"
        print(f"Gates passed {passed}/{total} -- {verdict}")
    print("Metrics are necessary conditions only: watch the continuous video before shipping.")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", required=True, help="benchmark/gates/*.yaml gate file")
    ap.add_argument("--contract", required=True, help="schema-v2 contract JSON")
    ap.add_argument("--policy", required=True, help="policy.onnx")
    ap.add_argument("--scene", required=True, help="MuJoCo scene XML matching the contract joints")
    ap.add_argument("--out", default=None, help="write per-gate metrics to this JSON file")
    args = ap.parse_args()

    spec = load_gate_file(args.gates)
    contract = ContractV2.from_json(args.contract)
    for warning in contract.validate():
        print(f"[contract] note: {warning}")
    print(f"policy:   {args.policy}")
    print(f"contract: {args.contract}")
    print(f"scene:    {args.scene}")
    print(f"gates:    {args.gates} ({spec['task']}, {spec['protocol']['seconds']} s each, deterministic)\n")

    report = evaluate(spec, args.policy, contract, args.scene)
    print_report(report, spec)

    if args.out:
        payload = {
            "gate_file": str(args.gates),
            "gate_schema_version": spec["schema_version"],
            "task": spec["task"],
            "policy": str(args.policy),
            "contract": str(args.contract),
            "scene": str(args.scene),
            "results": [
                {
                    "name": e["gate"]["name"],
                    "command": e["gate"]["command"],
                    "metric": e["gate"]["metric"],
                    "threshold": e["gate"]["threshold"],
                    "direction": e["gate"]["direction"],
                    "value": (None if e["value"] == float("inf") else e["value"]),
                    "passed": e["passed"],
                    "veto": e["veto"],
                    "measured": {
                        # Infinity is not valid JSON; an unmeasurable radius
                        # (straight walk) is recorded as null.
                        k: (None if isinstance(v, float) and v == float("inf") else v)
                        for k, v in e["measured"].items()
                        if k not in ("foot_names",)
                    },
                    "foot_names": e["measured"]["foot_names"],
                }
                for e in report["results"]
            ],
            "veto_hits": report["veto_hits"],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"per-gate metrics -> {args.out}")

    if report["veto_hits"]:
        return 2
    return 0 if all(e["passed"] for e in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
