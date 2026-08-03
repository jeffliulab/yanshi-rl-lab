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
        --contract <contract.json> --policy <policy.onnx> [--out results.json]

Published weights live on the Hugging Face Hub (never in git), so a
leaderboard entry's reproduction command names the release instead of a local
path::

    CUDA_VISIBLE_DEVICES="" python scripts/sim2sim/run_gates.py \
        --gates benchmark/gates/velocity-flat-turn.yaml \
        --hf jeffliulab/yanshi-unitree-g1-dof29@g1-flat-parity-s42 \
        --subdir velocity-flat

``--hf`` and ``--contract/--policy`` are two ways of naming the same pair of
files and may not be combined: a command that names both has two sources of
truth for which policy produced the numbers.

Scene resolution, in order: a gate's own ``scene:`` key, then ``--scene``,
then **the robot profile named by the gate file's ``robot:`` key**. That last
step is why no command line here needs to name a scene file.

    Writing the scene path by hand is how the published G1 leaderboard row
    came to be measured on ``scene_23dof.xml`` while ``G1_PROFILE`` declared
    ``scene_29dof.xml`` -- and that file carries 29 motor names over a 23-DoF
    kinematic tree, six of whose joints hang off massless bodies parked at
    z=20. The numbers were produced by a policy driving thin air with six of
    its outputs, and nothing in the pipeline could notice. The profile is the
    single source of truth; a hand-typed path is a second one.

``--scene`` remains for deliberate off-profile experiments (comparing two
scenes, bisecting a discrepancy). Using it is a choice, no longer the norm,
and the run report says which source was used. Relative scene paths resolve
against the repository root.

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
#
# v2 (2026-08-02) split the old single ``task:`` string into three explicit
# identity keys, because v1 could not say who an exam was for or which paper
# it was:
#
#   robot: unitree/g1/dof29     which robot configuration this exam is for
#   task:  velocity-flat        the SUBJECT -- what is being compared
#   exam:  velocity-flat-turn   this specific paper
#
# v1 conflated subject and paper (``task: velocity-flat-turn`` for G1 vs
# ``task: velocity-flat-x2`` for X2) and then threw the value away at result-
# write time, so both robots landed under the leaderboard subject
# "velocity-flat" and looked comparable -- while G1 was examined at wz=0.6
# and X2 at wz=0.2. Same column, different question, 3x apparent advantage.
GATES_SCHEMA_VERSION = 2

# File names inside a published Hugging Face release directory. These are the
# layout this project publishes under (documented in each model card), not a
# Hub-wide convention -- a release that renames them must fail loudly here
# rather than silently evaluate some other file.
HF_POLICY_NAME = "policy.onnx"
HF_CONTRACT_NAME = "contract.json"

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
    for required in ("robot", "task", "exam", "protocol", "gates", "veto"):
        if required not in spec:
            raise ValueError(f"Gate file is missing the {required!r} section.")
    # The exam name IS the file name: a leaderboard row records the exam, and
    # a reader must be able to open exactly that file to see what was asked.
    stem = Path(path).stem
    if spec["exam"] != stem:
        raise ValueError(
            f"Gate file exam {spec['exam']!r} does not match its filename {stem!r}; "
            "results record the exam name and readers resolve it back to this file."
        )
    for gate in spec["gates"]:
        for key in ("name", "command", "metric", "threshold", "direction", "unit"):
            if key not in gate:
                raise ValueError(f"Gate {gate.get('name', '?')!r} is missing {key!r}.")
        if gate["direction"] not in _DIRECTIONS:
            raise ValueError(
                f"Gate {gate['name']!r} direction {gate['direction']!r} not in {sorted(_DIRECTIONS)}."
            )
        if "scene" in gate and not isinstance(gate["scene"], str):
            raise ValueError(f"Gate {gate['name']!r} scene must be a path string.")
    return spec


def _resolve_scene(scene: str) -> str:
    """Absolute paths pass through; relative ones anchor at the repo root."""
    path = Path(scene)
    return str(path if path.is_absolute() else _REPO / path)


def profile_scene(robot_key: str) -> str:
    """The scene declared by a robot's profile -- the default, and the truth.

    Raises with the robot key named if the profile declares no scene or the
    file is missing, rather than falling back to anything.
    """
    from yanshi_rl_lab.robots import registry as robot_registry

    return str(robot_registry.get(robot_key).asset_path("scene_mjcf"))


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


def evaluate(spec: dict, policy: str, contract: ContractV2, scene: str | None) -> dict:
    seconds = float(spec["protocol"]["seconds"])
    veto_cfg = spec["veto"]
    # Default scene comes from the robot's profile, not from the caller.
    default_scene = scene if scene is not None else profile_scene(spec["robot"])
    results = []
    veto_hits = []
    for gate in spec["gates"]:
        gate_scene = _resolve_scene(gate.get("scene", default_scene))
        r = run_one_gate(policy, contract, gate_scene, gate["command"], seconds)
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
        results.append(
            {
                "gate": gate,
                "scene": gate_scene,
                "measured": r,
                "value": got,
                "passed": bool(passed),
                "veto": veto,
            }
        )
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
        print(f"    scene: {entry['scene']}")
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


def parse_hf_ref(ref: str) -> tuple[str, str]:
    """``<owner>/<repo>@<revision>`` -> ``(repo_id, revision)``.

    Pure function, no network, so leaderboard reproduction strings are
    testable without the Hub. The revision is mandatory: branches move, and a
    published row must name the exact release its numbers were measured on.
    """
    repo_id, sep, revision = ref.rpartition("@")
    repo_id, revision = repo_id.strip(), revision.strip()
    if not sep or not repo_id or not revision:
        raise ValueError(
            f"--hf must look like '<owner>/<repo>@<revision>', got {ref!r}. The "
            "revision is required -- a tag or commit, never a moving branch."
        )
    if repo_id.count("/") != 1 or not all(part.strip() for part in repo_id.split("/")):
        raise ValueError(f"--hf repository must look like '<owner>/<repo>', got {repo_id!r}.")
    return repo_id, revision


def fetch_hf_release(ref: str, subdir: str | None) -> tuple[str, str]:
    """Download a published release; return ``(contract_path, policy_path)``.

    ``huggingface_hub`` is imported here rather than at module scope so the
    gate-file loader stays importable without it -- same reason the mujoco /
    onnxruntime runtime import lives inside run_one_gate().

    A missing repo or revision propagates as an error. It must never fall back
    to a local file: a reproduction command that quietly evaluates something
    other than the release it names is worse than one that fails.
    """
    repo_id, revision = parse_hf_ref(ref)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("--hf needs huggingface_hub: pip install huggingface_hub") from exc

    subdir = None if subdir is None else subdir.strip("/")
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=None if subdir is None else f"{subdir}/*",
        )
    )
    release = root if subdir is None else root / subdir
    contract, policy = release / HF_CONTRACT_NAME, release / HF_POLICY_NAME
    missing = [name for name, p in ((HF_CONTRACT_NAME, contract), (HF_POLICY_NAME, policy)) if not p.exists()]
    if missing:
        where = f" under {subdir!r}" if subdir else ""
        raise SystemExit(
            f"{repo_id}@{revision} has no {' and no '.join(missing)}{where} -- "
            "not a publishable release of this project."
        )
    return str(contract), str(policy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", required=True, help="benchmark/gates/*.yaml gate file")
    ap.add_argument("--contract", default=None, help="schema-v2 contract JSON (local; or use --hf)")
    ap.add_argument("--policy", default=None, help="policy.onnx (local; or use --hf)")
    ap.add_argument(
        "--hf",
        default=None,
        help="published release to evaluate, '<owner>/<repo>@<revision>'. Downloads "
        "the policy and its contract from the Hugging Face Hub instead of taking "
        "local paths; cannot be combined with --contract/--policy.",
    )
    ap.add_argument(
        "--subdir",
        default=None,
        help="directory inside the --hf repository holding this release "
        "(e.g. 'velocity-flat'); omit when the release sits at the repo root.",
    )
    ap.add_argument(
        "--scene",
        default=None,
        help="override the scene declared by the gate file's robot profile. "
        "Normally omit this: the profile is the single source of truth, and a "
        "hand-typed path is how a leaderboard row once got measured on the "
        "wrong robot. Per-gate 'scene:' keys still win over this.",
    )
    ap.add_argument("--out", default=None, help="write per-gate metrics to this JSON file")
    args = ap.parse_args()

    # --hf and --contract/--policy name the same pair of files two different
    # ways. Accepting both would leave the run report unable to say which one
    # actually produced the numbers, so refuse instead of picking a winner.
    if args.hf and (args.contract or args.policy):
        ap.error("--hf cannot be combined with --contract/--policy (two sources of truth)")
    if args.subdir and not args.hf:
        ap.error("--subdir only applies to --hf")
    if args.hf:
        try:
            contract_path, policy_path = fetch_hf_release(args.hf, args.subdir)
        except ValueError as exc:  # malformed reference: a usage error, not a crash
            ap.error(str(exc))
        policy_line = f"{policy_path}  (from {args.hf})"
    elif args.contract and args.policy:
        contract_path, policy_path = args.contract, args.policy
        policy_line = str(policy_path)
    else:
        ap.error("give either --hf, or both --contract and --policy")

    spec = load_gate_file(args.gates)
    contract = ContractV2.from_json(contract_path)
    for warning in contract.validate():
        print(f"[contract] note: {warning}")
    if args.scene is None:
        scene_line = f"{profile_scene(spec['robot'])}  (from {spec['robot']} profile)"
    else:
        scene_line = f"{args.scene}  (--scene OVERRIDE, not the profile's)"
    print(f"policy:   {policy_line}")
    print(f"contract: {contract_path}")
    print(f"robot:    {spec['robot']}")
    print(f"scene:    {scene_line}")
    print(
        f"gates:    {args.gates} (task {spec['task']}, exam {spec['exam']}, "
        f"{spec['protocol']['seconds']} s each, deterministic)\n"
    )

    report = evaluate(spec, policy_path, contract, args.scene)
    print_report(report, spec)

    if args.out:
        payload = {
            "gate_file": str(args.gates),
            "gate_schema_version": spec["schema_version"],
            # Full exam identity travels with the numbers: without it a
            # leaderboard cannot tell two robots examined on different papers
            # from two robots examined on the same one.
            "robot": spec["robot"],
            "task": spec["task"],
            "exam": spec["exam"],
            "policy": str(policy_path),
            "contract": str(contract_path),
            # Which published release these numbers came from, when they came
            # from one; None means local files were evaluated.
            "hf_release": args.hf,
            "scene_override": (None if args.scene is None else str(args.scene)),
            "results": [
                {
                    "name": e["gate"]["name"],
                    "command": e["gate"]["command"],
                    "scene": e["scene"],
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
