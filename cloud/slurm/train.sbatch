#!/bin/bash
# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Slurm submission template for yanshi-rl-lab training on a university/HPC
# cluster, following the official Isaac Lab cluster route
# (isaac-sim/IsaacLab docker/cluster/*): build the Docker image from
# cloud/Dockerfile, convert it to an Apptainer image, run it with --nv.
#
# ⚠️ NOT end-to-end verified on a real cluster (see cloud/README.md): this
# project is developed on a single local workstation. The recipe follows the
# documented path; expect to adapt the EDIT-HERE parts to your site.
#
# One-time preparation (on a machine with Docker + Apptainer):
#   docker build -f cloud/Dockerfile --build-arg ACCEPT_EULA=Y -t yanshi-rl-lab .
#   apptainer build yanshi-rl-lab.sif docker-daemon://yanshi-rl-lab:latest
#   # copy yanshi-rl-lab.sif, the repo checkout and a pre-fetched assets tree
#   # (python assets/fetch.py) to cluster storage.
#
# Submission (site paths come from environment variables, nothing hardcoded):
#   export YANSHI_SIF=/cluster/storage/yanshi-rl-lab.sif
#   export YANSHI_REPO=/cluster/storage/yanshi-rl-lab
#   export YANSHI_ASSETS=/cluster/storage/yanshi-assets
#   sbatch cloud/slurm/train.sbatch Yanshi-Velocity-Flat-Unitree-G1-v0 --seed 42
#
# Logs land in $YANSHI_REPO/logs (the standard train.py output tree) and the
# Slurm stdout/stderr in the submission directory.

### EDIT HERE for your cluster's partitions/limits ###########################
#SBATCH --job-name=yanshi-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=yanshi-train-%j.out
##############################################################################

set -euo pipefail

# Site-specific module loads go here if your cluster needs them, e.g.:
# module load apptainer

# Required environment (fail with a clear message instead of guessing paths):
: "${YANSHI_SIF:?set YANSHI_SIF to the .sif built from cloud/Dockerfile}"
: "${YANSHI_REPO:?set YANSHI_REPO to the yanshi-rl-lab checkout on cluster storage}"
: "${YANSHI_ASSETS:?set YANSHI_ASSETS to a pre-fetched assets tree (python assets/fetch.py)}"

# First argument = task ID; everything after is passed through to train.py.
TASK="${1:?usage: sbatch train.sbatch <Yanshi-Task-ID> [extra train.py args]}"
shift

mkdir -p "${YANSHI_REPO}/logs"

# --nv exposes the host NVIDIA driver inside the container (Apptainer's
# equivalent of the docker --gpus flag used in the official cluster scripts).
apptainer exec --nv \
    --bind "${YANSHI_REPO}:/workspace/yanshi-rl-lab" \
    --bind "${YANSHI_ASSETS}:/workspace/assets" \
    --env YANSHI_ASSETS_ROOT=/workspace/assets \
    "${YANSHI_SIF}" \
    bash -c "cd /workspace/yanshi-rl-lab && \
        /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
        --task '${TASK}' --headless $*"
