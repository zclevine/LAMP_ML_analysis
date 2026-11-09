"""Engine entry points, and the bounds the UI reads back from them.

Why a table of dotted names rather than plain imports: the front end integrates
against ``lamp_design``'s public surface, which loads lazily, and one entry
point (``evaluate``) landed only recently. Resolving at the call site means a
missing name fails with the exact name it wanted instead of an ImportError at
app start, so one page can degrade on its own while the rest of the app keeps
working. ``ENGINE_API`` in ``../modal/app.py`` is the same idea and documents
each signature; it carries more names because its fan-out drives the stages
individually, while this app makes one orchestration call per run.

Every widget bound in the UI is derived from a constant read back out of the
engine here, so the two cannot drift into disagreement about what is accepted.

Nothing in this module imports ``torch`` or ``streamlit`` at module scope: it is
imported by the subprocess runner and by ``tests/test_streamlit_app.py`` as
well as by the app.
"""

from __future__ import annotations

import importlib
import os
import shutil
from functools import lru_cache
from typing import Any

ENGINE_API: dict[str, tuple[str, str]] = {
    "read_fasta": ("lamp_design.read_fasta", "lamp_design.utils.sequences.read_fasta"),
    "prepare_panel": ("lamp_design.prepare_panel", "lamp_design.commands.prepare.prepare_panel"),
    "default_geometries": (
        "lamp_design.default_geometries",
        "lamp_design.generation.default_geometries",
    ),
    # Missing until 2026-11-09, which crashed EVERY local design run with
    # KeyError: 'geometry_plan' -- _backend.run_local calls it before
    # generation. It was registered in the Modal table and not this one, so the
    # hosted portal worked and the CPU path, the one we hand to collaborators,
    # did not. Found by a collaborator running it, not by the suite.
    "geometry_plan": ("lamp_design.geometry_plan", "lamp_design.generation.geometry_plan"),
    "count_candidate_slots": (
        "lamp_design.count_candidate_slots",
        "lamp_design.generation.count_candidate_slots",
    ),
    "result_to_dict": ("lamp_design.result_to_dict", "lamp_design.report.result_to_dict"),
    "write_outputs": ("lamp_design.write_outputs", "lamp_design.report.write_outputs"),
    # The orchestration entry point a design run goes through. One call, so the
    # app and the CLI cannot drift in what "a design run" means -- which is why
    # the stage-level names (generate_candidates, compute_thermo, predict_tp,
    # build_design_result) are NOT in this table. The Modal harness carries them
    # because its fan-out interleaves the stages; nothing here does.
    "design": ("lamp_design.design", "lamp_design.commands.design.design"),
    "resolve_device": ("lamp_design.resolve_device", "lamp_design.commands.design.resolve_device"),
    # Torch-free first: `cli.resolve_weights` is the one place the weights
    # policy lives, and preflight runs on every rerun, so it must not pay for
    # importing models.inference (and therefore torch).
    "resolve_weights": (
        "lamp_design.cli.resolve_weights",
        "lamp_design.models.inference.resolve_weights_dir",
    ),
    # `lamp-design evaluate` is a stub in cli.py as of this writing. Both names
    # are resolved the same way as everything else, so wiring it up is nothing
    # but that module landing. See `_backend.evaluate_local`.
    "evaluate_run": ("lamp_design.commands.evaluate.run", "lamp_design.evaluate"),
}

_MISSING_HINT = (
    "The LAMP design engine does not expose {key} yet. This app expects either "
    "{primary} or {fallback}. Implement one of them (and re-export it from "
    "lamp_design/__init__.py); this file deliberately does not stub it."
)


def engine(key: str) -> Any:
    """Resolve a public engine entry point, or raise naming exactly what it wanted."""
    primary, fallback = ENGINE_API[key]
    for dotted in (primary, fallback):
        module_name, _, attr = dotted.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        found = getattr(module, attr, None)
        if found is not None:
            return found
    raise NotImplementedError(_MISSING_HINT.format(key=key, primary=primary, fallback=fallback))


def has_engine(key: str) -> bool:
    """Whether one engine entry point exists. For page gating and diagnostics."""
    try:
        engine(key)
    except NotImplementedError:
        return False
    return True


def engine_default(module_name: str, attr: str, fallback: Any) -> Any:
    """Read a constant from the engine, falling back if that module is absent.

    Used only for UI bounds, so the widgets agree with the engine's own limits
    instead of hardcoding a second set that can drift.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return fallback
    return getattr(module, attr, fallback)


# ---------------------------------------------------------------------------
# Bounds and defaults, read from the engine
# ---------------------------------------------------------------------------

#: Sequences shorter than this are excluded by prepare (recorded, never dropped
#: silently); more than MAX_SEQUENCES is rejected outright; BUDGET is
#: `n_targets x alignment width`, the metric ARCHITECTURE.md caps on.
MIN_TARGET_LENGTH = int(engine_default("lamp_design.commands.prepare", "DEFAULT_MIN_LENGTH", 162))
MAX_SEQUENCES = int(engine_default("lamp_design.commands.prepare", "MAX_SEQUENCES", 20))
MAX_BUDGET = int(
    os.environ.get("LAMP_DESIGN_MAX_BUDGET")
    or engine_default("lamp_design.commands.prepare", "DEFAULT_MAX_BUDGET", 50_000)
)

DEFAULT_TP_THRESHOLD = float(
    engine_default("lamp_design.report", "DEFAULT_TP_THRESHOLD_MINUTES", 30.0)
)
DEFAULT_N_SETS = int(engine_default("lamp_design.report", "DEFAULT_N_SETS", 20))
DEFAULT_MIN_SEPARATION = int(engine_default("lamp_design.report", "DEFAULT_MIN_SEPARATION_BP", 30))

#: Evaluate's own bounds. ``censor_at`` is the Tp assigned to a reaction that
#: never amplified, matching the training data's ``tp60_censored`` convention;
#: ``max_window_mismatches`` rejects a located binding site worse than this,
#: because past it the placement is more likely to be the wrong site than a
#: diverged one.
DEFAULT_CENSOR_MINUTES = float(
    engine_default("lamp_design.commands.evaluate", "DEFAULT_CENSOR_MINUTES", 60.0)
)
DEFAULT_MAX_WINDOW_MISMATCHES = int(
    engine_default("lamp_design.commands.evaluate", "DEFAULT_MAX_WINDOW_MISMATCHES", 6)
)

#: Every anchor position, same as the engine and the CLI.
#:
#: This was 5, on my reasoning that stride 1 meant "minutes of wait" on an
#: interactive page. That was wrong twice over: stride 1 is about 40 s on the
#: GPU backend, and stride 1 was an explicit team decision I overrode without
#: saying so. The effect was a portal that quietly evaluated a fifth of the
#: positions while the run page advertised ~25,000 candidates. The UI does not
#: get its own default for a design decision.
ENGINE_DEFAULT_STRIDE = int(engine_default("lamp_design.generation", "DEFAULT_STRIDE", 1))
DEFAULT_STRIDE = ENGINE_DEFAULT_STRIDE

#: Local runs share the Modal path's candidate cap. `generate_candidates` thins
#: whole anchors at a uniform stride rather than truncating, so hitting it
#: degrades gracefully instead of returning a head-of-template prefix.
MAX_CANDIDATE_SETS = int(os.environ.get("LAMP_DESIGN_MAX_CANDIDATE_SETS", "200000"))

#: Master mixes come from the frozen contract (`types.MASTER_MIXES`). BstXT-dk
#: carries the canonical label BST in the training data.
MASTER_MIXES: tuple[str, ...] = tuple(
    engine_default("lamp_design.types", "MASTER_MIXES", ("SS", "BST", "WS"))
)
#: What the UI actually offers. WS stays in MASTER_MIXES above because it is
#: part of the model's frozen input contract -- the mix is a 3-way one-hot and
#: removing a channel would invalidate the checkpoints -- but it is not offered
#: as a choice. Team decision: SS and BST only. WS is also the thinnest slice
#: of the training data by a wide margin (12,564 rows against SS 91,451 and
#: BST 66,160), so predictions under it are the least supported.
SELECTABLE_MASTER_MIXES: tuple[str, ...] = ("SS", "BST")

#: Candidate sets laid down at each anchor: the canonical geometry plus
#: DEFAULT_N_SAMPLED drawn from the normal distribution over the permitted
#: length/spacing ranges. Read back from the engine so the UI cannot drift
#: from what `lamp-design design` does with no flags.
DEFAULT_N_GEOMETRIES: int = 1 + int(
    engine_default("lamp_design.generation", "DEFAULT_N_SAMPLED", 4)
)

#: Manuscript names, so the tool and the paper agree. WS is WarmStart, which
#: the manuscript does not name and the UI does not offer.
MASTER_MIX_LABELS = {
    "BST": "Bst-XT-WS",
    "SS": "Bst-SS",
    "WS": "WS — WarmStart",
}

#: The eight primer types in frozen layout order along the target.
PRIMER_TYPE_ORDER: tuple[str, ...] = tuple(
    engine_default(
        "lamp_design.types",
        "PRIMER_TYPE_ORDER",
        ("F3", "F2", "FL", "F1c", "B1c", "BL", "B2", "B3"),
    )
)

#: The engine's default sweep: 2 master mixes x 2 concentrations.
DEFAULT_MASTER_MIXES: tuple[str, ...] = tuple(
    engine_default("lamp_design.commands.design", "DEFAULT_MASTER_MIXES", ("BST",))
)
DEFAULT_CONCENTRATIONS: tuple[float, ...] = tuple(
    float(c)
    for c in engine_default(
        "lamp_design.commands.design", "DEFAULT_CONCENTRATIONS", (10_000.0, 100.0)
    )
)

#: Concentrations that actually appear in the training data (copies per
#: reaction). Offered as a fixed list rather than a free number box so the model
#: is not asked to extrapolate: `encoding.encode_target_concentration` feeds it
#: `log10(conc) - 3`, whose docstring confirms the scale ("100/1000/10000 ->
#: -1/0/+1").
TRAINED_CONCENTRATIONS: tuple[float, ...] = (
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    100_000_000.0,
)

#: Measured on one L4 at batch 512 (ARCHITECTURE.md, performance notes). Used
#: only to turn a candidate count into an expected wait, so the Run page is not
#: the first place a user learns a run takes eight minutes.
PREDICTIONS_PER_SECOND_GPU = 2_292.0
PREDICTIONS_PER_SECOND_CPU = 206.0


@lru_cache(maxsize=1)
def n_available_geometries() -> int:
    """How many deterministic geometries generation offers (canonical first)."""
    try:
        return len(engine("default_geometries")())
    except NotImplementedError:
        return 5


@lru_cache(maxsize=1)
def geometry_labels() -> tuple[str, ...]:
    """Geometry ids in the engine's canonical-first order, for UI captions."""
    try:
        return tuple(
            getattr(geometry, "geometry_id", str(geometry))
            for geometry in engine("default_geometries")()
        )
    except NotImplementedError:
        return ()


@lru_cache(maxsize=1)
def resolved_device() -> str:
    """``cuda`` when a GPU is usable here, else ``cpu``. Cached: it imports torch."""
    try:
        return str(engine("resolve_device")("auto"))
    except Exception:
        return "cpu"


def weights_status() -> tuple[bool, str]:
    """``(ok, message)`` for the configured checkpoint directory.

    Weights are bundled, so the common case is simply fine and this check is
    cheap reassurance. It still earns its place: a ``LAMP_DESIGN_WEIGHTS`` that
    is set but wrong, or a build that dropped package data, is worth reporting
    before a run rather than as a traceback afterwards. Resolution goes through
    the engine's own policy (``cli.resolve_weights``) so there is not a second
    copy of it here.
    """
    explicit = os.environ.get("LAMP_DESIGN_WEIGHTS") or None
    try:
        resolved = engine("resolve_weights")(explicit)
    except NotImplementedError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)
    return True, str(resolved)


def tool_status() -> dict[str, str | None]:
    """External binaries the engine shells out to, resolved the way it does.

    MAFFT is vestigial -- the engine never aligns now, since input must arrive
    pre-aligned and gapless -- but it is still reported so a stale image shows
    up here rather than confusing someone later. ``thermo`` prefers the
    in-process ``RNA`` bindings and only falls back to the ``RNAduplex``
    binary, so neither is unconditionally fatal.
    The Run page's preflight decides which of these matters for a given input.
    """
    status: dict[str, str | None] = {
        "mafft": shutil.which("mafft"),
        "RNAduplex": shutil.which("RNAduplex"),
    }
    try:
        import RNA

        status["RNA bindings"] = getattr(RNA, "__version__", "present")
    except Exception:
        status["RNA bindings"] = None
    return status
