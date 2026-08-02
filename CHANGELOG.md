# Changelog

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
