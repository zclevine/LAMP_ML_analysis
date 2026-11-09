"""``lamp_design``'s re-exported surface is a contract with the front ends.

The Streamlit app and the Modal harness resolve engine functions by dotted
name. A rename that only breaks them shows up nowhere else, so the public
surface is pinned here.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import lamp_design
from lamp_design import _EXPORTS


def test_every_export_resolves():
    """No entry in the table points at something that does not exist."""
    missing = []
    for name in sorted(_EXPORTS):
        try:
            getattr(lamp_design, name)
        except (AttributeError, ImportError) as exc:
            missing.append(f"{name} ({_EXPORTS[name]}): {exc}")
    assert not missing, "unresolvable exports:\n  " + "\n  ".join(missing)


def test_export_table_matches_dunder_all():
    assert set(lamp_design.__all__) == {"__version__", *_EXPORTS}
    assert set(dir(lamp_design)) == set(lamp_design.__all__)


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        lamp_design.nope


def test_each_export_comes_from_the_module_the_table_names():
    """Guards against the table drifting from where things actually live."""
    for name, module_name in sorted(_EXPORTS.items()):
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"{module_name} does not define {name}"


def test_names_the_front_ends_resolve_are_present():
    """The exact dotted paths deploy/streamlit/app.py looks up."""
    for name in (
        "generate_candidates",
        "default_geometries",
        "count_candidate_slots",
        "compute_thermo",
        "predict_tp",
        "build_design_result",
        "result_to_dict",
        "write_outputs",
    ):
        assert getattr(lamp_design, name, None) is not None, name


def test_building_the_cli_parser_does_not_import_torch():
    """`--help` and `prepare` must not pay torch's import cost.

    This regressed once: `cli.build_parser` imports every command module to
    register its subcommand, so a module-scope `from lamp_design.models...`
    import in commands/design.py pulled torch into `--help` (1.8s vs 0.24s).
    Those imports are deliberately function-local; this is the guard.
    """
    code = (
        "import sys; from lamp_design.cli import build_parser; build_parser(); "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", (
        "building the CLI parser imported torch; a command module has a "
        "module-scope import of lamp_design.models.*"
    )


def test_importing_the_package_does_not_import_torch():
    """`prepare` and `--help` must not pay for torch; a subprocess proves it."""
    code = (
        "import sys; import lamp_design; "
        "assert lamp_design.__version__; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", (
        "importing lamp_design pulled in torch; the lazy __getattr__ was bypassed"
    )
