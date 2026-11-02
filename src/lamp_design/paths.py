"""Filesystem locations that ship inside the package.

This module exists to stay **free of heavy imports**, and that is its whole
job. The bundled-weights path has to be resolvable where torch is not
installed:

* ``deploy/streamlit/perf/Dockerfile.thin`` and ``deploy/modal/portal.py``
  install ``lamp_design`` with ``--no-deps`` and no torch on purpose, because
  compute runs remotely. They still import the package to validate input,
  count candidate geometries and report where the weights are.
* ``models/inference.py`` does ``import torch`` at module scope.

So defining ``BUNDLED_WEIGHTS`` there and importing it from
``cli.resolve_weights`` made weight *resolution* depend on torch. In the thin
portal that surfaced as a sidebar reading "Weights: not configured" with
"No module named 'torch'" underneath -- the weights were present and fine, the
lookup just could not get far enough to see them.

Keep this module importable with nothing but the standard library.
"""

from __future__ import annotations

from pathlib import Path

#: The installed ``lamp_design`` package directory.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent

#: Training run whose five-seed ensemble ships with the package. Matches the
#: directory name used locally and in the Modal weights Volume, so one name
#: works everywhere.
WEIGHTS_RUN: str = "tp60_A_v0t9"

#: The bundled ensemble: ``seed_{0,1,2,3,42}/best_model.pt``, 2.8 MB in total.
#: Monkeypatch *this* attribute (``lamp_design.paths.BUNDLED_WEIGHTS``) to test
#: the missing-weights path; both resolvers read it at call time.
BUNDLED_WEIGHTS: Path = PACKAGE_ROOT / "weights" / WEIGHTS_RUN
