"""Where the real-model tests get their checkpoints.

One copy of this policy rather than six. Now that the five-seed ensemble ships
inside the package (``src/lamp_design/weights/``), these tests **run** on a
fresh clone and in CI instead of skipping -- which is the point. The code paths
that actually load the model and predict were the least-covered ones precisely
because they needed weights nobody checking out the repo had.

``LAMP_DESIGN_WEIGHTS`` still wins, so pointing the suite at a retrained
ensemble takes no edit here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Why a real-model test would skip. Reaching this now means a broken checkout
#: or a build that dropped package data, not an ordinary unconfigured machine.
WEIGHTS_MISSING_REASON = (
    "no checkpoints: bundled weights missing from the package and "
    "LAMP_DESIGN_WEIGHTS unset"
)


def weights_or_none() -> Optional[str]:
    """Resolved checkpoint directory, or ``None`` if even the bundled one is gone.

    A configured ``LAMP_DESIGN_WEIGHTS`` that does not exist is treated as
    absent here, so a stale shell variable degrades to the bundled ensemble
    rather than failing the whole suite. The library itself is stricter --
    ``resolve_weights_dir`` raises on a bad explicit path, because silently
    substituting a different model would change results without saying so.
    """
    configured = os.environ.get("LAMP_DESIGN_WEIGHTS")
    if configured and Path(configured).is_dir():
        return configured

    from lamp_design.models.inference import BUNDLED_WEIGHTS

    return str(BUNDLED_WEIGHTS) if BUNDLED_WEIGHTS.is_dir() else None
