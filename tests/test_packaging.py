"""The declared dependency set must match what the code actually imports.

This file exists because of a bug that reached a deployed container: `einops`
is imported at module level by the vendored S4D kernel, which is on the
production inference path, but it was never listed in ``pyproject.toml``. This
machine happened to have it installed by hand, so every local run and every
test passed, and the gap only surfaced when a clean Modal image ran `doctor`
and died with ``ModuleNotFoundError: No module named 'einops'``.

A dependency that is missing from the metadata but present in the dev
environment is invisible to ordinary tests -- by construction, since the tests
run in the dev environment. So the check has to be static: read the imports out
of the source, read the requirements out of the packaging metadata, and compare
the two without importing anything.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import subprocess
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "lamp_design"

#: Import name -> distribution name, where they differ.
_IMPORT_TO_DIST = {
    "Bio": "biopython",
    "RNA": "ViennaRNA",
    "primer3": "primer3-py",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}

#: Imports that are deliberately NOT declared, with the reason. Each one must
#: be reachable only from a function body, never at module level -- which
#: `test_optional_imports_are_function_local` enforces, so this list cannot be
#: used to wave away a real missing dependency.
_INTENTIONALLY_OPTIONAL = {
    "fla": "lyra_base._DeltaNetBackbone, only for ssm='deltanet'; production is s4d",
    "mamba_ssm": "lyra_base._build_ssm, only for ssm='mamba*'; production is s4d",
    "modal": "the deploy harness only; `import lamp_design` must work without it",
    "streamlit": "the front end only, never the engine",
    "pytest": "tests only",
}


def _module_level_imports(tree: ast.Module) -> set[str]:
    """Top-level import names, not descending into functions or classes."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks are not runtime imports.
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    continue
    return names


def _all_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _third_party(names: set[str]) -> set[str]:
    return {
        name
        for name in names
        if name not in sys.stdlib_module_names
        and name != "lamp_design"
        and not name.startswith("_")
    }


def _declared_distributions() -> set[str]:
    """Distribution names from ``pyproject.toml``'s dependencies, lowercased.

    Parsed with a regex rather than tomllib so this runs identically on 3.10,
    where tomllib does not exist.
    """
    text = (REPO / "pyproject.toml").read_text()
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no `dependencies = [...]` block"
    declared = set()
    for line in block.group(1).splitlines():
        line = line.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if not line:
            continue
        # "primer3-py>=2.0,<2.1" -> "primer3-py"
        declared.add(re.split(r"[<>=!~\[]", line)[0].strip().lower())
    return declared


def _source_files() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_source_tree_was_found():
    """Guard the guard: a bad path would make every check below vacuous."""
    files = _source_files()
    assert len(files) > 10, f"only found {len(files)} source files under {SRC}"


def test_every_module_level_import_is_declared():
    """The check that would have caught the missing `einops`."""
    declared = _declared_distributions()
    missing: dict[str, list[str]] = {}

    for path in _source_files():
        tree = ast.parse(path.read_text())
        for name in _third_party(_module_level_imports(tree)):
            if name in _INTENTIONALLY_OPTIONAL:
                continue
            dist = _IMPORT_TO_DIST.get(name, name).lower()
            if dist not in declared:
                missing.setdefault(name, []).append(str(path.relative_to(REPO)))

    assert not missing, (
        "module-level imports with no matching dependency in pyproject.toml: "
        + "; ".join(f"{name} ({', '.join(files)})" for name, files in sorted(missing.items()))
        + ". A dependency installed by hand on this machine still fails in a "
        "clean container -- add it to `dependencies`, or make the import "
        "function-local and record it in _INTENTIONALLY_OPTIONAL."
    )


@pytest.mark.parametrize("name", sorted(_INTENTIONALLY_OPTIONAL))
def test_optional_imports_are_function_local(name: str):
    """An "optional" dependency imported at module level is just a missing one.

    Without this, `_INTENTIONALLY_OPTIONAL` would be a way to silence the test
    above rather than a statement about where the import lives.
    """
    offenders = [
        str(path.relative_to(REPO))
        for path in _source_files()
        if name in _module_level_imports(ast.parse(path.read_text()))
    ]
    assert not offenders, (
        f"{name!r} is listed as optional but is imported at module level in "
        f"{', '.join(offenders)} -- so importing that module requires it. "
        f"Either declare it in pyproject.toml or move the import into the "
        f"function that needs it."
    )


def test_einops_specifically_is_declared():
    """Named explicitly, because this is the one that actually shipped broken.

    `models/lyra/s4d.py` imports einops at module level and is reached by
    `models.bridge.predict_tp` -> inference -> fastpath -> s4d, which is the
    production path. A parametrized sweep would cover it, but naming it keeps
    the regression legible.
    """
    assert "einops" in _declared_distributions()

    s4d = SRC / "models" / "lyra" / "s4d.py"
    assert "einops" in _module_level_imports(ast.parse(s4d.read_text())), (
        "s4d.py no longer imports einops at module level; if the vendored "
        "kernel changed, this test and the pyproject entry should be revisited"
    )


def test_the_optional_model_backends_are_still_optional():
    """`fla` / `mamba_ssm` are heavy CUDA builds we deliberately do not ship.

    The production config is ``ssm=s4d`` (export/New_Lyra_code/
    command_line_args.json), so these are dead branches. If one ever becomes
    reachable at module level, the engine stops installing on a plain CPU box.
    """
    lyra_base = SRC / "models" / "lyra" / "lyra_base.py"
    tree = ast.parse(lyra_base.read_text())
    top = _module_level_imports(tree)
    assert "mamba_ssm" not in top and "fla" not in top
    # They are, however, genuinely referenced somewhere -- otherwise this file
    # is asserting about imports that no longer exist.
    assert {"mamba_ssm", "fla"} & _all_imports(tree)


# ---------------------------------------------------------------------------
# dG reference
#
# Binding dG is a model input, so an environment that computes it differently
# produces different designs -- silently. scripts/check_dg_reference.py is the
# guard; these tests are the guard on the guard.
# ---------------------------------------------------------------------------


def test_the_dg_reference_check_passes_in_this_environment():
    """The dev environment must itself satisfy what the images are asked to."""
    script = REPO / "scripts" / "check_dg_reference.py"
    assert script.exists()
    done = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"


def test_the_dg_reference_check_fails_on_a_wrong_value():
    """A check that cannot fail is not a check.

    Verified by perturbing one reference value in a copy, rather than by
    trusting that the comparison works.
    """
    source = (REPO / "scripts" / "check_dg_reference.py").read_text()
    assert "DG_REFERENCE = [-26.4" in source, "reference literal moved"
    broken = source.replace("DG_REFERENCE = [-26.4", "DG_REFERENCE = [-99.9")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "broken_check.py"
        path.write_text(broken)
        done = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=300
        )
    assert done.returncode != 0, "a wrong reference value was accepted"
    assert "does not match" in done.stderr


def test_both_dockerfiles_run_the_dg_reference_check_or_pin_the_version():
    """Each image must guard dG one way or the other.

    The CLI image cannot pin ViennaRNA (bioconda has no 2.4.17 for py3.10), so
    it asserts the dG values instead; the Modal image pins the apt version.
    Neither may end up doing neither.
    """
    cli = (REPO / "Dockerfile").read_text()
    assert "check_dg_reference.py" in cli, "CLI image does not check dG"

    modal = (REPO / "deploy" / "modal" / "app.py").read_text()
    assert "RNADUPLEX_EXPECTED_VERSION" in modal
    assert 'RNAduplex --version | grep -q' in modal, "Modal image does not assert its pin"


def test_the_bundled_weights_are_present_and_complete():
    """All five seeds, where the library expects them.

    A missing seed would not crash: `discover_checkpoints` returns whatever it
    finds and the ensemble would quietly average over four models instead of
    five, shifting every prediction by a small amount nobody would notice.
    """
    from lamp_design.models.inference import (
        BUNDLED_WEIGHTS,
        discover_checkpoints,
        resolve_weights_dir,
    )

    assert BUNDLED_WEIGHTS.is_dir(), f"bundled weights missing from {BUNDLED_WEIGHTS}"
    assert sorted(discover_checkpoints(BUNDLED_WEIGHTS)) == [0, 1, 2, 3, 42]
    # Nothing configured must resolve to them -- this is what makes a clone run.
    assert resolve_weights_dir() == BUNDLED_WEIGHTS


def test_the_weights_are_declared_as_package_data():
    """Otherwise `pip install .` and every Docker build drop them silently.

    The failure mode is nasty: the source tree works, the wheel does not, and
    the error surfaces only when someone installs rather than clones. Checked
    statically against the packaging metadata, for the same reason as the
    dependency check above.
    """
    config = (REPO / "pyproject.toml").read_text()
    assert "[tool.setuptools.package-data]" in config, "no package-data section"
    assert "weights/*/*/*.pt" in config, (
        "the weights glob is not declared, so a built wheel will not contain them"
    )


def test_no_dangling_symlinks_among_the_bundled_weights():
    """They were symlinks into a training run before being committed.

    `cp -rL` dereferenced them; a future re-copy with plain `cp -r` would
    commit links that resolve on this machine and nowhere else.
    """
    from lamp_design.models.inference import BUNDLED_WEIGHTS

    for checkpoint in sorted(BUNDLED_WEIGHTS.rglob("*.pt")):
        assert not checkpoint.is_symlink(), f"{checkpoint} is a symlink, not data"
        assert checkpoint.stat().st_size > 100_000, f"{checkpoint} is suspiciously small"
