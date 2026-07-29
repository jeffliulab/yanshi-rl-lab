# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT
#
# Headless training image for yanshi-rl-lab, following the official Isaac Lab
# container route (isaac-sim/IsaacLab docker/Dockerfile.base, simplified for
# a single-purpose training image).
#
# ⚠️ NOT end-to-end verified on a real cluster (see cloud/README.md): this
# project is developed on a single local workstation. Treat this file as a
# starting point that follows the documented recipe, not a certified build.
#
# We do NOT distribute prebuilt images: the NGC Isaac Sim base image is
# EULA-gated and Omniverse Kit components are not redistributable. Build it
# yourself, accepting the NVIDIA EULA explicitly:
#
#   docker build -f cloud/Dockerfile --build-arg ACCEPT_EULA=Y -t yanshi-rl-lab .
#
# (Run from the repository root so the build context contains the repo.
#  Pulling the base image requires `docker login nvcr.io` with an NGC key.)
#
# Train inside the container (headless):
#
#   docker run --gpus all --rm -v $(pwd)/logs:/workspace/yanshi-rl-lab/logs yanshi-rl-lab \
#     /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
#       --task Yanshi-Velocity-Flat-Unitree-G1-v0 --headless
#
# Version pins (foundation policy: Isaac Lab upgrades are the #1 killer of
# projects like this one -- bump pins only on a dedicated branch):
#   - Isaac Sim base image 5.1.0 (pairs with Isaac Lab 2.3.2)
#   - Isaac Lab tag v2.3.2 (the version this repo is developed against)

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0

FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION}

# The NVIDIA Isaac Sim EULA must be accepted by the person building the
# image; there is deliberately no default.
ARG ACCEPT_EULA=""
RUN test "${ACCEPT_EULA}" = "Y" || ( \
    echo "ERROR: you must accept the NVIDIA Isaac Sim EULA to build this image:" && \
    echo "  docker build --build-arg ACCEPT_EULA=Y ..." && exit 1)
ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
USER root

# Same minimal apt set as the official Dockerfile.base.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Isaac Lab at the pinned tag, wired to the Isaac Sim shipped in the base
# image via the official _isaac_sim symlink, then installed with rsl_rl only
# (the RL library this repo trains with).
ARG ISAACLAB_TAG=v2.3.2
RUN git clone --depth 1 --branch ${ISAACLAB_TAG} \
        https://github.com/isaac-sim/IsaacLab.git /workspace/isaaclab && \
    ln -sf /isaac-sim /workspace/isaaclab/_isaac_sim && \
    /workspace/isaaclab/isaaclab.sh --install rsl_rl

# This repository (build context = repo root; exclusions in .dockerignore --
# notably assets/, which are fetched at run time, and logs/).
COPY . /workspace/yanshi-rl-lab

# Install the extension into Isaac Sim's bundled Python.
RUN /workspace/isaaclab/isaaclab.sh -p -m pip install -e \
    /workspace/yanshi-rl-lab/source/yanshi_rl_lab

WORKDIR /workspace/yanshi-rl-lab

# Robot model assets are NOT baked into the image (vendor licenses differ and
# meshes are large). Either fetch them once inside the running container:
#     python3 assets/fetch.py <vendor>/<model>
# or bind-mount a pre-fetched tree and point the framework at it:
#     -v /cluster/assets:/workspace/assets -e YANSHI_ASSETS_ROOT=/workspace/assets

CMD ["bash"]
