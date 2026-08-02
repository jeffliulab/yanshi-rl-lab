# Yanshi Rank result-file schema (protocol v1)

Every leaderboard entry is one JSON file at

```
benchmark/results/v1/<vendor>-<model>/<task>/<label>.json
```

- `<vendor>-<model>` is the robot's profile identity with the `/` replaced by
  `-` (e.g. `unitree-g1`); it MUST match the `robot` field inside the file.
- `<task>` MUST match the `task` field inside the file.
- `<label>` is free (typically the run name, e.g. `yanshi-g1-flat-parity-s42`).

Files are validated by `benchmark/render_rank.py --check` (also wired into
CI). Validation is strict: unknown top-level fields are rejected so typos
cannot pass silently. Schema changes require bumping the results directory
version (`results/v2/...`) together with a new `protocol_version` — never a
silent field edit.

## Fields (all required unless marked optional)

| field | type | meaning |
|---|---|---|
| `protocol_version` | int, `1` | The frozen evaluation protocol this entry was measured under (`benchmark/PROTOCOL_v1.md` — **not written yet, due in M5**; until then the gate file plus this entry's `repro_command` are the operative definition). Must match the `results/v<N>/` directory. |
| `robot` | string `"<vendor>/<model>/<variant>"` | Profile identity, lowercase_with_underscores tokens starting with a letter (same rule as `robots/<vendor>/<model>/<variant>/`). The variant segment is mandatory: a row must say which configuration of a model it measured. |
| `task` | string | The **subject** being compared, e.g. `"velocity-flat"`. Rows sharing a task are asking the same kind of question. |
| `exam` | string | The **specific paper** sat, naming a file in `benchmark/gates/` (without `.yaml`). Its `robot:` and `task:` must agree with this entry's. The leaderboard groups by exam, because only same-exam rows are comparable: two robots under one task but different exams (e.g. turn gates commanded at wz=0.6 vs wz=0.2) are not a ranking. |
| `commit` | string, 7–40 hex chars | Code commit of this repository the numbers were produced with. |
| `isaaclab_version` | string | Isaac Lab version used for training (e.g. `"2.3.2"`). |
| `gpu` | string | GPU the training ran on (e.g. `"RTX 5070 Ti 16GB"`). |
| `seeds` | list of int, non-empty, unique | Seeds evaluated. PROTOCOL v1 requires >= 3 seeds for our own baselines; the schema does not hard-enforce it so partial third-party submissions remain representable (their tier tells the story). |
| `metrics` | object: metric name → { seed(string) → number } | Per-seed raw values, metric-major. The seed keys of every metric MUST be exactly `seeds` (as strings). Metric names refer to the protocol's metric definitions. |
| `checkpoint` | object `{hf_repo, revision}` | Where the evaluated policy weights live (Hugging Face Hub — weights are never committed to git). `hf_repo` like `"user/repo"`, `revision` a tag/branch/SHA. |
| `repro_command` | non-empty single-line string | One copy-pasteable command that reproduces the evaluation. It must have been RUN once before being written down, and it must not name a scene file: the gate file names the robot and the robot profile names the scene (a hand-typed `--scene` is a second source of truth, and that is exactly how the first published G1 row came to be measured on a scene the profile did not declare). |
| `trust_tier` | `"verified"` \| `"reproduced"` \| `"self-reported"` | `verified` = measured by the maintainers on stated hardware; `reproduced` = a third party reproduced a submitted result; `self-reported` = submitter's own numbers, not independently reproduced. |
| `date` | string `YYYY-MM-DD` (optional) | Measurement date. |
| `notes` | string (optional) | Free-form caveats (e.g. "gains derived, not official"). |

## Aggregation & precision (what the renderer shows)

`render_rank.py` displays, per metric, `median (IQR)` across seeds — the
statistics fixed by PROTOCOL v1 (IQR = p75 − p25, linear interpolation).
Displayed values are rounded to 3 decimal places; the JSON keeps full
precision. Renderer outputs:

- `benchmark/site/rank.md` — Markdown table for embedding into the README;
- `benchmark/site/index.html` — self-contained static leaderboard page
  (no external requests, light/dark aware) for GitHub Pages.

## Example

```json
{
  "protocol_version": 1,
  "robot": "unitree/g1/dof29",
  "task": "velocity-flat",
  "exam": "velocity-flat-turn",
  "commit": "fb9b73f",
  "isaaclab_version": "2.3.2",
  "gpu": "RTX 5070 Ti 16GB",
  "seeds": [42, 43, 44],
  "metrics": {
    "tracking_rmse": {"42": 0.041, "43": 0.043, "44": 0.040},
    "fall_rate": {"42": 0.0, "43": 0.0, "44": 0.0}
  },
  "checkpoint": {"hf_repo": "jeffliulab/yanshi-g1-velocity", "revision": "v0.1"},
  "repro_command": "python scripts/sim2sim/run_gates.py --gates benchmark/gates/velocity-flat-turn.yaml --contract ... --policy ...",
  "trust_tier": "verified",
  "date": "2026-07-29",
  "notes": ""
}
```
