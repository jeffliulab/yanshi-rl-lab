#!/usr/bin/env python3
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Render the Yanshi Rank leaderboard from result JSON files.

Reads every ``benchmark/results/v1/<vendor>-<model>/<task>/*.json`` (schema:
``benchmark/results/SCHEMA.md``), validates it strictly, and renders:

- ``benchmark/site/rank.md``    -- Markdown table for embedding in the README;
- ``benchmark/site/index.html`` -- self-contained static page for GitHub Pages
  (inline CSS only, no external resources, light/dark aware).

An empty results tree renders an empty (but valid) table -- the board goes
live before the first baseline lands.

Usage::

    python benchmark/render_rank.py            # validate + render both outputs
    python benchmark/render_rank.py --check    # validate only (CI schema gate)

Pure stdlib on purpose: CI and contributors need this without any project
dependency installed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = _REPO_ROOT / "benchmark" / "results" / "v1"
DEFAULT_SITE_DIR = _REPO_ROOT / "benchmark" / "site"

# Must match the results/v<N>/ directory this renderer reads (SCHEMA.md rule).
PROTOCOL_VERSION = 1

TRUST_TIERS = ("verified", "reproduced", "self-reported")

REQUIRED_FIELDS = (
    "protocol_version",
    "robot",
    "task",
    # Which exam paper produced these numbers. ``task`` says what is being
    # compared; ``exam`` says what was actually asked. They are not the same
    # thing, and conflating them let a G1 row examined at wz=0.6 rad/s sit in
    # the same column as an X2 row examined at wz=0.2 -- an apparent 3x
    # turning advantage that was really a different question. Must name a
    # file in benchmark/gates/.
    "exam",
    "commit",
    "isaaclab_version",
    "gpu",
    "seeds",
    "metrics",
    "checkpoint",
    "repro_command",
    "trust_tier",
)
OPTIONAL_FIELDS = ("date", "notes")

# Display rounding fixed by SCHEMA.md ("Aggregation & precision"); the JSON
# files keep full precision.
DISPLAY_DECIMALS = 3

# Robot identity is <vendor>/<model>/<variant>; the variant segment is
# mandatory (robots/profile.py explains why) so a leaderboard row always says
# which configuration of a model it measured.
_ROBOT_RE = re.compile(r"^[a-z][a-z0-9_]*/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*$")
_TASK_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


GATES_DIR = _REPO_ROOT / "benchmark" / "gates"


def _check_exam_agrees_with_gate_file(data: dict, source: str) -> list[str]:
    """The exam must exist, and claim the same robot and task as the result.

    Without this the three identity fields are just strings a submitter typed:
    a result could claim ``task: velocity-flat`` while pointing at a rough
    exam, and the leaderboard would put it in the wrong column. Deliberately
    a plain text scan, not a YAML parse -- this renderer is pure stdlib so CI
    and contributors can run it with nothing installed.
    """
    gate_file = GATES_DIR / f"{data['exam']}.yaml"
    if not gate_file.is_file():
        # An unfetched or partial checkout should not turn into schema noise;
        # only complain when the gates directory is actually present.
        if not GATES_DIR.is_dir():
            return []
        return [f"{source}: exam {data['exam']!r} names no file at benchmark/gates/{data['exam']}.yaml"]
    declared: dict[str, str] = {}
    for line in gate_file.read_text(encoding="utf-8").splitlines():
        for key in ("robot", "task"):
            prefix = f"{key}: "
            if line.startswith(prefix) and key not in declared:
                declared[key] = line[len(prefix) :].strip()
    errs = []
    for key in ("robot", "task"):
        if key in declared and declared[key] != data[key]:
            errs.append(
                f"{source}: {key} {data[key]!r} disagrees with exam "
                f"{data['exam']!r}, which declares {declared[key]!r}"
            )
    return errs


def validate_result(data: object, source: str = "<data>") -> list[str]:
    """Validate one result document; return a list of error strings (empty = valid)."""
    errs: list[str] = []

    def err(msg: str) -> None:
        errs.append(f"{source}: {msg}")

    if not isinstance(data, dict):
        return [f"{source}: top level must be a JSON object"]

    unknown = set(data) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        err(f"unknown field(s) {sorted(unknown)} (schema is strict; see SCHEMA.md)")
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        err(f"missing required field(s) {missing}")
        return errs  # further checks would just cascade

    if data["protocol_version"] != PROTOCOL_VERSION:
        err(f"protocol_version must be {PROTOCOL_VERSION}, got {data['protocol_version']!r}")
    if not (isinstance(data["robot"], str) and _ROBOT_RE.match(data["robot"])):
        err(
            "robot must look like 'vendor/model/variant' (lowercase_with_underscores, "
            f"each segment starting with a letter), got {data['robot']!r}"
        )
    if not (isinstance(data["task"], str) and _TASK_RE.match(data["task"])):
        err(f"task must be lowercase-with-hyphens, got {data['task']!r}")
    if not (isinstance(data["exam"], str) and _TASK_RE.match(data["exam"])):
        err(f"exam must be lowercase-with-hyphens, got {data['exam']!r}")
    else:
        errs.extend(_check_exam_agrees_with_gate_file(data, source))
    if not (isinstance(data["commit"], str) and _COMMIT_RE.match(data["commit"])):
        err(f"commit must be a 7-40 char hex SHA, got {data['commit']!r}")
    for field in ("isaaclab_version", "gpu"):
        if not (isinstance(data[field], str) and data[field].strip()):
            err(f"{field} must be a non-empty string")

    seeds = data["seeds"]
    if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) for s in seeds)):
        err(f"seeds must be a non-empty list of integers, got {seeds!r}")
        seeds = []
    elif len(set(seeds)) != len(seeds):
        err(f"seeds contains duplicates: {seeds}")

    metrics = data["metrics"]
    if not (isinstance(metrics, dict) and metrics):
        err("metrics must be a non-empty object of {metric: {seed: value}}")
    else:
        expected_keys = {str(s) for s in seeds}
        for name, per_seed in metrics.items():
            if not (
                isinstance(per_seed, dict)
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in per_seed.values())
            ):
                err(f"metric {name!r} must map seed strings to numbers")
                continue
            if seeds and set(per_seed) != expected_keys:
                err(f"metric {name!r} seed keys {sorted(per_seed)} do not match seeds {sorted(expected_keys)}")

    checkpoint = data["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"hf_repo", "revision"}:
        err("checkpoint must be an object with exactly {hf_repo, revision}")
    elif not all(isinstance(checkpoint[k], str) and checkpoint[k].strip() for k in ("hf_repo", "revision")):
        err("checkpoint.hf_repo and checkpoint.revision must be non-empty strings")

    repro = data["repro_command"]
    if not (isinstance(repro, str) and repro.strip()) or "\n" in repro:
        err("repro_command must be a non-empty single-line string")
    if data["trust_tier"] not in TRUST_TIERS:
        err(f"trust_tier must be one of {TRUST_TIERS}, got {data['trust_tier']!r}")
    if "date" in data and not (isinstance(data["date"], str) and _DATE_RE.match(data["date"])):
        err(f"date must be YYYY-MM-DD, got {data['date']!r}")
    if "notes" in data and not isinstance(data["notes"], str):
        err("notes must be a string")
    return errs


def load_results(results_root: Path) -> tuple[list[dict], list[str]]:
    """Load and validate every result file under the results root.

    Returns ``(entries, errors)``. Each entry is the parsed document plus a
    ``_label`` (file stem) used for display. Path layout is part of the schema:
    ``<vendor>-<model>/<task>/<label>.json`` must agree with the fields inside.
    """
    entries: list[dict] = []
    errors: list[str] = []
    for path in sorted(results_root.glob("*/*/*.json")):
        source = str(path.relative_to(results_root))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{source}: invalid JSON ({exc})")
            continue
        errs = validate_result(data, source)
        if not errs:
            robot_dir, task_dir = path.parent.parent.name, path.parent.name
            if data["robot"].replace("/", "-") != robot_dir:
                errs.append(f"{source}: robot {data['robot']!r} does not match directory {robot_dir!r}")
            if data["task"] != task_dir:
                errs.append(f"{source}: task {data['task']!r} does not match directory {task_dir!r}")
        if errs:
            errors.extend(errs)
            continue
        entries.append({**data, "_label": path.stem})
    # stray JSON at unexpected depths would silently escape the glob -- catch it
    for path in sorted(results_root.rglob("*.json")):
        rel = path.relative_to(results_root)
        if len(rel.parts) != 3:
            errors.append(f"{rel}: unexpected location (expected <vendor>-<model>/<task>/<label>.json)")
    entries.sort(key=lambda e: (e["task"], e["robot"], e["_label"]))
    return entries, errors


def aggregate(entry: dict) -> dict[str, tuple[float, float]]:
    """Per metric: (median, IQR) across seeds -- the PROTOCOL v1 statistics."""
    out: dict[str, tuple[float, float]] = {}
    for name, per_seed in entry["metrics"].items():
        values = sorted(float(v) for v in per_seed.values())
        med = statistics.median(values)
        if len(values) >= 2:
            q = statistics.quantiles(values, n=4, method="inclusive")
            iqr = q[2] - q[0]
        else:
            iqr = 0.0
        out[name] = (med, iqr)
    return out


def _fmt(value: float) -> str:
    return f"{value:.{DISPLAY_DECIMALS}f}"


def _metric_columns(entries: list[dict]) -> list[str]:
    names: set[str] = set()
    for entry in entries:
        names.update(entry["metrics"])
    return sorted(names)


def group_by_exam(entries: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Group entries into ``(task, exam, entries)``, ordered for display.

    The exam is the grouping key, not the task: rows are comparable exactly
    when they answered the same question. A task with several exams therefore
    renders as several tables under one heading, which is the honest shape --
    "same subject, different paper" must not look like one ranking.
    """
    buckets: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        buckets.setdefault((entry["task"], entry["exam"]), []).append(entry)
    return [(task, exam, buckets[(task, exam)]) for task, exam in sorted(buckets)]


def render_markdown(entries: list[dict]) -> str:
    """Markdown leaderboard, one table per exam (embedded into the README)."""
    lines = [
        "<!-- Generated by benchmark/render_rank.py -- do not edit by hand. -->",
        "",
    ]
    groups = group_by_exam(entries)
    if not groups:
        lines += ["_No results yet._", ""]
        return "\n".join(lines) + "\n"

    for task, exam, group in groups:
        # Metric columns are computed per exam, not globally: a single wide
        # table across every subject would be mostly em-dashes once rough,
        # stairs and speed-ladder land alongside flat.
        metric_cols = _metric_columns(group)
        header = [
            "Robot",
            *[f"{m} (median, IQR)" for m in metric_cols],
            "Seeds",
            "Trust",
            "Commit",
            "Checkpoint",
        ]
        lines += [
            f"### {task}",
            "",
            f"Exam: [`{exam}`](../gates/{exam}.yaml)",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        for entry in group:
            stats = aggregate(entry)
            cells = [entry["robot"]]
            for metric in metric_cols:
                if metric in stats:
                    med, iqr = stats[metric]
                    cells.append(f"{_fmt(med)} ({_fmt(iqr)})")
                else:
                    cells.append("—")
            ckpt = entry["checkpoint"]
            cells += [
                str(len(entry["seeds"])),
                entry["trust_tier"],
                f"`{entry['commit'][:7]}`",
                f"[{ckpt['hf_repo']}@{ckpt['revision']}]"
                f"(https://huggingface.co/{ckpt['hf_repo']}/tree/{ckpt['revision']})",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    by_task: dict[str, set[str]] = {}
    for task, exam, _ in groups:
        by_task.setdefault(task, set()).add(exam)
    split = {t: sorted(e) for t, e in by_task.items() if len(e) > 1}
    if split:
        lines += [
            "> Some subjects above are split across several exams. Numbers are "
            "comparable **within** an exam only:",
            "",
        ]
        lines += [f"> - `{t}`: {', '.join('`' + e + '`' for e in exams)}" for t, exams in sorted(split.items())]
        lines.append("")
    return "\n".join(lines) + "\n"


_HTML_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fdfcfa; --fg: #1c1c1c; --muted: #6a6a6a; --line: #e3e0da;
  --card: #ffffff; --accent: #8a5a2b;
  --tier-verified: #2e7d32; --tier-reproduced: #1565c0; --tier-self: #8a6d00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16181c; --fg: #e6e4e0; --muted: #9a9a9a; --line: #2c3038;
    --card: #1d2026; --accent: #d0a068;
    --tier-verified: #7bc67e; --tier-reproduced: #74a9e8; --tier-self: #d6b95c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 72rem; padding: 2rem 1.25rem 4rem;
  background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { margin: 0 0 .25rem; font-size: 1.8rem; }
.sub { color: var(--muted); margin: 0 0 2rem; }
h2 { font-size: 1.2rem; margin: 2.25rem 0 .75rem; }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; background: var(--card); }
th, td { padding: .55rem .8rem; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--line); }
tbody tr:last-child td { border-bottom: none; }
th { font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
td.num { font-variant-numeric: tabular-nums; }
.tier { font-size: .78rem; font-weight: 600; padding: .1rem .5rem; border-radius: 99px;
        border: 1px solid currentColor; }
.tier.verified { color: var(--tier-verified); }
.tier.reproduced { color: var(--tier-reproduced); }
.tier.self-reported { color: var(--tier-self); }
code { font: .85em/1 ui-monospace, "SF Mono", Menlo, monospace; }
a { color: var(--accent); }
.empty { color: var(--muted); font-style: italic; padding: 2rem 0; }
.exam { color: var(--muted); font-size: 0.9rem; margin: -0.4rem 0 0.8rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .82rem; }
"""


def render_html(entries: list[dict]) -> str:
    """Self-contained leaderboard page (inline CSS, no external resources)."""
    e = html.escape
    body: list[str] = []
    groups = group_by_exam(entries)
    if not groups:
        body.append('<p class="empty">No results yet — the first baselines land after protocol v1 freezes.</p>')
    for task, exam, task_entries in groups:
        metric_cols = _metric_columns(task_entries)
        columns = ["Robot", *[f"{m} — median (IQR)" for m in metric_cols], "Seeds", "Trust", "Commit", "Checkpoint"]
        head = "".join(f"<th>{e(col)}</th>" for col in columns)
        rows: list[str] = []
        for entry in task_entries:
            stats = aggregate(entry)
            cells = [f"<td>{e(entry['robot'])}</td>"]
            for metric in metric_cols:
                if metric in stats:
                    med, iqr = stats[metric]
                    cells.append(f'<td class="num">{_fmt(med)} ({_fmt(iqr)})</td>')
                else:
                    cells.append('<td class="num">—</td>')
            ckpt = entry["checkpoint"]
            ckpt_url = f"https://huggingface.co/{ckpt['hf_repo']}/tree/{ckpt['revision']}"
            tier = entry["trust_tier"]
            cells += [
                f'<td class="num">{len(entry["seeds"])}</td>',
                f'<td><span class="tier {e(tier)}">{e(tier)}</span></td>',
                f"<td><code>{e(entry['commit'][:7])}</code></td>",
                f'<td><a href="{e(ckpt_url)}">{e(ckpt["hf_repo"])}@{e(ckpt["revision"])}</a></td>',
            ]
            title = e(entry.get("notes") or entry["_label"])
            rows.append(f'<tr title="{title}">{"".join(cells)}</tr>')
        # The exam is named next to every table: rows are comparable only
        # within one, and a reader must be able to open the paper they sat.
        body.append(
            f"<h2>{e(task)}</h2>\n"
            f'<p class="exam">Exam: <a href="https://github.com/jeffliulab/yanshi-rl-lab/'
            f'blob/main/benchmark/gates/{e(exam)}.yaml"><code>{e(exam)}</code></a></p>\n'
            f'<div class="tablewrap"><table>\n'
            f"<thead><tr>{head}</tr></thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody></table></div>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Yanshi Rank</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>Yanshi Rank</h1>
<p class="sub">One exam paper, many robots — the same locomotion protocol applied to every
robot in <a href="https://github.com/jeffliulab/yanshi-rl-lab">yanshi-rl-lab</a>.
Values are median (IQR) across seeds, protocol v{PROTOCOL_VERSION}.</p>
{chr(10).join(body)}
<footer>Generated by <code>benchmark/render_rank.py</code> from
<code>benchmark/results/v{PROTOCOL_VERSION}/</code>; result-file schema in
<code>benchmark/results/SCHEMA.md</code>. Trust tiers: verified /
reproduced / self-reported (see SCHEMA.md).</footer>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate result files only, render nothing")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    args = parser.parse_args(argv)

    if not args.results_root.is_dir():
        print(f"error: results root {args.results_root} does not exist", file=sys.stderr)
        return 1
    entries, errors = load_results(args.results_root)
    if errors:
        for line in errors:
            print(f"SCHEMA ERROR  {line}", file=sys.stderr)
        print(f"{len(errors)} schema error(s); see benchmark/results/SCHEMA.md", file=sys.stderr)
        return 1
    print(f"validated {len(entries)} result file(s) under {args.results_root}")
    if args.check:
        return 0

    args.site_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.site_dir / "rank.md"
    html_path = args.site_dir / "index.html"
    md_path.write_text(render_markdown(entries), encoding="utf-8")
    html_path.write_text(render_html(entries), encoding="utf-8")
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
