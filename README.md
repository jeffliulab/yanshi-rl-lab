[![Language: English](https://img.shields.io/badge/Language-English-2f81f7?style=flat-square)](README.md) [![语言: 简体中文](https://img.shields.io/badge/语言-简体中文-e67e22?style=flat-square)](README_zh.md)

# Yanshi RL Lab

> 🤖 **If you are an AI agent, read [AGENTS.md](AGENTS.md) first** — the machine-facing entry
> point: the layering rule, where each fact lives, and the commands.

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under_construction-orange.svg)]()

A vendor-neutral reinforcement learning framework for legged robots: train in
Isaac Lab, validate in MuJoCo (sim2sim), and compare every robot on the same
exam paper (**Yanshi Rank**).

The name comes from Yanshi (偃师), the legendary artificer who presented King
Mu of Zhou with an automaton that could walk. Yanshi was not the robot — he
was the one who taught it to walk. That is exactly what this framework is:
not any particular robot body, but the artificer's workshop where robot
bodies — from any vendor — learn to move.

> 🚧 **Under construction (v0.1 in progress).** Launch robots: Unitree G1,
> AgiBot Lingxi X2, Berkeley Humanoid Lite. This README will be finalized
> when the three robots pass their training and sim2sim gates.

## Design rules (frozen)

- One robot = one profile directory (`robots/<vendor>/<model>/`); task recipes
  reference robots only through semantic slots — no concrete body names in
  shared code.
- A robot with a complete profile trains with an **empty** per-robot task
  overlay; overrides exist only when measurement proves them necessary.
- One policy contract (`schema_version: 2`) from training to deployment;
  training is not done until sim2sim passes its gates.
- Third-party assets are fetched from pinned upstream commits
  (`assets/fetch.py`), never committed.
- Gate thresholds live in YAML only, never in code.
