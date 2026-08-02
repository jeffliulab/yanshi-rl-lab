# Copyright (c) 2026 Jeff Liu.
# SPDX-License-Identifier: MIT

"""AgiBot Lingxi X2 (X2-Ultra, 31 actuated joints). This package stays pure
Python (no Isaac Lab imports).

NOTE: upstream provides no PD gains -- ours are DERIVED (natural-frequency
method); see the provenance notes in ``profile.py``."""

from .profile import X2_PROFILE  # noqa: F401
