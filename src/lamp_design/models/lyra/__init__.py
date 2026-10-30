"""Vendored Lyra architecture — a verbatim copy, kept verbatim on purpose.

These modules are copied from ``~/LAMP/lyra/`` with exactly one change: the
``RMSNorm`` import in ``lyra_base.py`` is rewritten to be relative. Their value
is that they are byte-equivalent to the code that produced the checkpoints, so
``load_state_dict(..., strict=True)`` is a real check and parity is defined
against something fixed.

Do not optimize or refactor anything in this subpackage. Inference-only
speedups belong in :mod:`lamp_design.models.fastpath`, which patches them onto
module instances instead.

This file exists so the subpackage is found by ``setuptools.find_packages`` and
not only by ``find_namespace_packages``: without it a built wheel can omit the
whole architecture, and the failure surfaces as a ``ModuleNotFoundError`` inside
a container rather than at build time.
"""
