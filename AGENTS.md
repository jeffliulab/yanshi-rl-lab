# AGENTS.md — agent entry point for Yanshi RL Lab

Read this file first, then jump to the section your task needs. It is an index, not a manual:
the prose it points at is the authority.

> **This file is public.** Local working notes — dev logs, run ledgers, unreleased results,
> absolute paths — belong in `CLAUDE.md`, which is not in git.

## Inherited rules

This repository inherits shared rules from `agent-rules`.

- `agent-rules` version: `v0.4.0`
- upstream machine entry: [agent-rules/AGENTS.md](https://github.com/jeffliulab/agent-rules/blob/v0.4.0/AGENTS.md)
- upstream manifest: [reading-order.yaml](https://github.com/jeffliulab/agent-rules/blob/v0.4.0/manifests/reading-order.yaml)

Read this file first for project-local overrides, then the pinned upstream entry, then follow
exactly one matching upstream path — do not sweep the whole upstream repository. Never pin
`main`. Where a rule here conflicts with `agent-rules`, this file wins.

## What this is (and is not)

**A vendor-neutral reinforcement learning framework for legged robots.** Policies are trained in
Isaac Lab, validated in MuJoCo (sim2sim), and every robot is scored on the same exam paper —
**Yanshi Rank**. MIT licensed, Python 3.11, Isaac Lab 2.3.2. Launch robots: Unitree G1, AgiBot
Lingxi X2, Berkeley Humanoid Lite.

The name is the thesis. Yanshi (偃师) was the artificer who presented King Mu of Zhou with an
automaton that could walk — he was not the robot, he was the one who taught it to walk. This
repository is the workshop, not any particular body.

🚧 **Under construction (v0.1).** The README is finalized when the three launch robots pass
their training and sim2sim gates.

It is **not**:

- a robot-specific training repo — if you are about to write `g1` into shared code, you are in
  the wrong layer;
- a simulator or a physics engine — Isaac Lab and MuJoCo are dependencies, not things this
  repository reimplements;
- an asset host — third-party robot models are fetched from pinned upstream commits, never
  committed here.

## Robot identity: `<vendor>/<model>/<variant>`

Every robot is named by three segments, and the variant is **mandatory** even when a model
currently ships one configuration. `unitree/g1/dof29` and `unitree/g1/dof23` are peers; so are
`agibot/x2/v1_4_0` and whatever revision comes next.

- **What counts as a variant**: swap it and a trained policy no longer transfers. Terrain,
  reward tables and seeds are tasks and experiments, not variants.
- **Naming**: use *upstream's own name* for the configuration — `dof29` because Unitree ships
  `g1_29dof_rev_1_0.urdf`, `v1_4_0` because the X2 model directory is `X2_URDF-v1.4.0`,
  `humanoid` because that project ships `biped` and `humanoid`. Never invent one.
- **Must start with a letter**: every segment is also a Python package name. Write `dof29`,
  not `29dof`.
- **Assets are keyed `<vendor>/<model>`, not by variant.** Variants of one model share one
  fetched tree and differ only in which files their profiles point at; a second asset key
  would clone the same upstream repo twice and let the copies drift.

The identity appears in four places and must agree in all of them: the profile directory, the
task overlay directory, the task ID (`Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0`), and the
leaderboard entry (`"robot": "unitree/g1/dof29"`).

## Where a fact belongs (the layering rule)

Ask before writing a line: **would this still be true for a robot from a different vendor?**
If not, it belongs in that robot's profile, not in shared code.

| Layer | What may live here |
| --- | --- |
| `source/yanshi_rl_lab/yanshi_rl_lab/tasks/` | Task recipes. Reference robots only through **semantic slots** — never a concrete body name |
| `source/yanshi_rl_lab/yanshi_rl_lab/robots/<vendor>/<model>/<variant>/profile.py` | Everything true of exactly one robot configuration: joints, PD gains, semantic body-part mapping, **and which model files it uses**. **The single source of truth for that robot** — nothing downstream may name an asset path by hand |
| `source/yanshi_rl_lab/yanshi_rl_lab/mdp/`, `terrains/`, `utils/` | Reusable observation / reward / terrain building blocks, body-agnostic |
| `benchmark/gates/*.yaml` | Gate thresholds. **YAML only — never a number in code** |
| `assets/registry.py` | Which upstream commit each robot's assets come from |

A robot with a complete profile trains with an **empty** per-robot task overlay. An override
exists only when a measurement proves it necessary — and the measurement goes in the commit
message.

## Task → where to look

| If your task is… | Start at |
| --- | --- |
| Add a new robot | `python scripts/tools/new_robot.py <vendor> <model> <variant>`, then fill `robots/<vendor>/<model>/<variant>/profile.py`; mirror an existing profile test in `tests/` |
| See what robots exist / switch configuration | `yanshi robots` — identity is `<vendor>/<model>/<variant>`, and the variant rides in the task ID, so switching is a `--task` change |
| Change a task recipe | `tasks/locomotion/velocity/` — check the change holds for all three launch robots |
| Move a gate threshold | `benchmark/gates/*.yaml`, never the Python |
| Add or repin an asset | `assets/registry.py`, then `assets/fetch.py` |
| Deploy or export a policy | `yanshi_rl_lab/deploy/` — the policy contract is `schema_version: 2` |
| Understand the scoring | `benchmark/README.md` and `benchmark/results/SCHEMA.md` |

## Repo map

| Path | What is in it |
| --- | --- |
| `source/yanshi_rl_lab/` | The Isaac Lab extension: robots, tasks, mdp, terrains, deploy, CLI |
| `scripts/` | Entry points — `rsl_rl/train.py`, `rsl_rl/play.py`, `sim2sim/`, `tools/` |
| `benchmark/` | Yanshi Rank: gate YAMLs, results schema, the rendered site |
| `assets/` | Fetcher and pinned registry. The models themselves are gitignored |
| `cloud/` | Dockerfile and Slurm scripts for off-box training |
| `tests/` | Pure-CPU tests: one per robot profile, plus contract, gates, and scaffolder |

## How to run it

```bash
# environment self-check — pure CPU, never crashes
yanshi doctor

# fetch pinned robot assets (they are not in the repository)
yanshi assets status
yanshi assets fetch

# tests are CPU-only and need no simulator
pytest -q

# training runs through Isaac Lab's launcher, not bare python
./isaaclab.sh -p scripts/rsl_rl/train.py --task <Yanshi-...>
```

## Red lines

- **No concrete body names in shared code.** Vendor neutrality is the product; a single `if
  robot == "g1"` in a task recipe destroys it.
- **Gate thresholds live in YAML, never in code.** A threshold in Python is a number nobody can
  audit or diff.
- **No third-party assets in the repository.** Licenses differ per vendor; fetch from a pinned
  upstream commit.
- **One policy contract from training to deployment** (`schema_version: 2`). Training is not
  done until sim2sim passes its gates — a policy that only works in the trainer is not a result.
- **No hardcoding**: paths derived or from env, tunables in config, magic numbers named. A
  placeholder must be flagged to the maintainer, never buried.

Details: the pinned upstream rules above, and `README.md` § Design rules (frozen).
