# Yanshi Rank

One exam paper, many robots: the same locomotion evaluation protocol applied
to every robot in the framework, rendered as a public leaderboard.

Layout (populated in milestone M5, after the three launch robots pass their
gates):

- `PROTOCOL_v1.md` — the frozen evaluation protocol (tasks, command points,
  seeds, metrics, statistics, numeric-precision settings). Frozen **before**
  baseline numbers are published; protocol version is stamped into every
  result file.
- `gates/<task>.yaml` — acceptance-gate lines, declarative. Gate thresholds
  live ONLY here, never in code.
- `results/v1/<vendor>-<model>/<task>/*.json` — one file per submission:
  code commit, Isaac Lab version, GPU, per-seed metrics, checkpoint location
  (HF Hub, never git), and a single copy-pasteable reproduction command.
  Trust tiers: `verified` / `reproduced` / `self-reported`.
- `render_rank.py` — renders the results into the README table and the
  GitHub Pages leaderboard.

External submissions open after the protocol has proven stable; until then
the board carries our own baselines only.
