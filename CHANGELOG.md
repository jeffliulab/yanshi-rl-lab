# Changelog

## v0.2 — 2026-08-02

**Idea.** A name has to say what it means. Two things in v0.1 were named by
convention rather than by record — which robot configuration a result measured,
and which exam it sat — and both silently produced wrong published numbers.

**Done.**

- **Robot identity is now `<vendor>/<model>/<variant>`** (`unitree/g1/dof29`),
  with the variant mandatory even for a model that ships one configuration.
  Variant names are upstream's own; assets stay keyed by model, so
  configurations of one robot share a single fetched tree. New `yanshi robots`
  lists them, and the variant rides in the task ID so switching configuration
  is a `--task` change.
- **Gate files declare `robot`, `task` and `exam`** (schema v2). The
  leaderboard groups by exam and links it, because rows are comparable only
  when they answered the same question.
- **The scene comes from the robot profile**, not from the command line.
  `run_gates.py` needs no `--scene`, and results no longer record one.
- **G1 flat numbers re-measured** and corrected — see below.

**Breaking.** Task IDs, leaderboard `robot` keys and result directory names all
gained a segment; result files now require `exam`; gate files require
`schema_version: 2`. No compatibility aliases: v0.1 was one day old with two
published results, and an alias would have outlived the mistake it hid.

**Corrections.**

- The published G1 flat entry was measured on `scene_23dof.xml`, whose 29 motor
  names sit over a 23-DoF kinematic tree — six joints (waist roll/pitch, both
  wrist pitch/yaw) hang off massless bodies parked at z=20, so six policy
  outputs drove nothing. The profile declared `scene_29dof.xml` all along; the
  hand-typed path in the repro command was a second source of truth. Re-measured
  on the profile's scene: verdict unchanged (4/4), every number moved — turn
  275.9°→280.4°, radius 0.61→0.60 m, walk 4.49→4.30 m, slow walk 2.29→2.20 m.
  X2 reproduces bit-identically; it had been run on its profile's scene.
- `benchmark/results/SCHEMA.md` showed a `--onnx` flag that never existed.

## v0.1 — 2026-08-01

First public release.

**Idea.** A robot should be a profile, not a rewrite. Everything true of exactly
one body lives in that body's profile; task recipes reach it only through
semantic slots. And a policy is not finished when the training curve looks good
— it has to survive a second physics engine, judged by thresholds frozen before
the run existed.

**Done.**

- Robot profile layer with a semantic body-part mapping; a complete profile
  trains with an empty per-robot task overlay.
- One policy contract (`schema_version: 2`) from training through MuJoCo replay,
  plus converters for three legacy formats.
- Declarative acceptance gates: thresholds in YAML, never in code.
- Three launch robots onboarded — Unitree G1, AgiBot Lingxi X2, Berkeley
  Humanoid Lite. G1 and X2 pass their flat-ground gates 4/4; BHL walks but
  scores 2/4, missing both slow-command gates because its training command
  envelope is far wider than the exam points.
- Yanshi Rank: first leaderboard entries, single seed.
- Assets fetched from pinned upstream commits, never committed.
- CPU-only test suite covering profiles, contract, gates and the scaffolder.
- `scripts/tools/new_robot.py` scaffolder, `yanshi doctor` self-check, Docker
  and Slurm templates for off-box training.

**Known limits.** Single seed throughout; multi-seed baselines are the v0.2
standard. Flat ground only. `benchmark/PROTOCOL_v1.md` is not written yet, so
the gate files plus each result's `repro_command` are the operative definition.
