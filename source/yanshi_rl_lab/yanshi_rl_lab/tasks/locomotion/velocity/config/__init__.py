# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""Per-robot task registrations, mirrored as ``config/<vendor>/<model>/``.

Task IDs follow ``Yanshi-<Task>-<Terrain>-<Vendor>-<Model>-v<N>``.
Each robot package contains only a registration call plus (when measurement
proves it necessary) documented parameter overrides -- adding a robot never
touches any central file.
"""
