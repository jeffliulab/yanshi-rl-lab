# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Ported from unitree-g1-locomotion ``sim2sim/export_onnx_cpu.py`` (MIT, same
# author). The original hardcoded OBS_DIM=480 / ACT_DIM=29 / HIDDEN=[512,256,
# 128]; here the whole MLP architecture is inferred from the checkpoint's
# actor state dict, so any robot / any layer stack exports without edits.

"""Pure-CPU ONNX export for rsl-rl checkpoints -- no Isaac, no GPU.

An rsl-rl >= 5.0 actor is a plain MLP (``mlp.0/2/4/...`` Linear layers with
an activation between); deterministic inference = the MLP forward pass, so
the exported graph takes the flattened observation and returns raw actions
(exactly what the training-side policy computed -- no normalization layer).

Layer sizes (input dim, hidden dims, action dim) are read from the Linear
weight shapes in the checkpoint itself: the checkpoint IS the single source
of truth for the architecture, so nothing can drift. The activation cannot
be recovered from weights; it is read from the run's ``params/agent.yaml``
when present, else falls back to rsl-rl's default (elu) with a notice.

Usage::

    python -m yanshi_rl_lab.deploy.export_onnx <run_dir> [<run_dir> ...]
    python -m yanshi_rl_lab.deploy.export_onnx <run_dir> --ckpt 4000

``--ckpt`` exports one specific iteration (useful for capability-vs-training
curves); the default exports the newest checkpoint to
``<run_dir>/exported/policy.onnx``.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

# rsl-rl's default actor activation; used only when agent.yaml is absent.
DEFAULT_ACTIVATION = "elu"
# ONNX opset used by the predecessor stack's verified exports.
ONNX_OPSET = 17


def _activation_module(name: str):
    import torch.nn as nn

    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    if name not in table:
        raise ValueError(f"Unsupported activation {name!r} (known: {sorted(table)}).")
    return table[name]()


def infer_mlp_dims(actor_state_dict: dict) -> list:
    """[obs_dim, hidden..., act_dim] from ``mlp.<idx>.weight`` shapes.

    Pure function (works on anything with ``.shape``), unit-tested without
    torch. Raises if consecutive layer shapes do not chain -- a broken or
    non-MLP checkpoint must fail loudly.
    """
    layers = []
    for key, value in actor_state_dict.items():
        m = re.fullmatch(r"mlp\.(\d+)\.weight", key)
        if m:
            out_dim, in_dim = int(value.shape[0]), int(value.shape[1])
            layers.append((int(m.group(1)), in_dim, out_dim))
    if not layers:
        raise ValueError("actor_state_dict has no 'mlp.<i>.weight' entries; not an rsl-rl MLP actor.")
    layers.sort()
    dims = [layers[0][1]]
    for _, in_dim, out_dim in layers:
        if in_dim != dims[-1]:
            raise ValueError(
                f"MLP layer chain broken: layer expects input {in_dim}, previous output {dims[-1]}."
            )
        dims.append(out_dim)
    return dims


def read_activation(run_dir: str) -> str:
    """Actor activation from ``<run_dir>/params/agent.yaml``; DEFAULT_ACTIVATION
    with a notice when the file or key is absent."""
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(agent_yaml):
        import yaml

        with open(agent_yaml, encoding="utf-8") as f:
            agent = yaml.unsafe_load(f)  # Isaac Lab dumps python-tagged YAML
        policy = (agent or {}).get("policy") or {}
        act = policy.get("activation")
        if act:
            return str(act)
    print(
        f"[export] note: no activation found in {agent_yaml}; using rsl-rl default "
        f"{DEFAULT_ACTIVATION!r}"
    )
    return DEFAULT_ACTIVATION


def build_actor(dims: list, activation: str):
    import torch.nn as nn

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(_activation_module(activation))
    return nn.Sequential(*layers)


def latest_ckpt(run_dir: str) -> str:
    cks = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not cks:
        raise FileNotFoundError(f"{run_dir} has no model_*.pt")
    return max(cks, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def export(run_dir: str, ckpt: int | None = None) -> str:
    import torch

    if ckpt is None:
        ck_path = latest_ckpt(run_dir)
    else:
        ck_path = os.path.join(run_dir, f"model_{ckpt}.pt")
        if not os.path.exists(ck_path):
            raise FileNotFoundError(f"No such checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    # Keep only mlp.* (drops distribution.std_param -- deterministic deploy
    # has no noise); strip the "mlp." prefix to match Sequential key names.
    mlp_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}
    dims = infer_mlp_dims(sd)
    actor = build_actor(dims, read_activation(run_dir))
    actor.load_state_dict(mlp_sd, strict=True)
    actor.eval()

    out_dir = os.path.join(run_dir, "exported")
    os.makedirs(out_dir, exist_ok=True)
    # A specific --ckpt export gets its own file so policy.onnx always means
    # "this run's final policy".
    out = os.path.join(out_dir, "policy.onnx" if ckpt is None else f"policy-{ckpt}.onnx")
    torch.onnx.export(
        actor,
        torch.zeros(1, dims[0]),
        out,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=ONNX_OPSET,
    )
    print(f"[export] {os.path.basename(ck_path)} (obs {dims[0]} -> act {dims[-1]}, hidden {dims[1:-1]}) -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export rsl-rl checkpoints to CPU-inference ONNX")
    ap.add_argument("run_dirs", nargs="+", help="one or more run directories")
    ap.add_argument(
        "--ckpt", type=int, default=None, help="iteration number (default: newest checkpoint)"
    )
    args = ap.parse_args()
    for rd in args.run_dirs:
        export(rd, args.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
