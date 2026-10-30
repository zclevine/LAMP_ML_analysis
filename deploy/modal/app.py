"""Modal harness for the LAMP design engine.

STATUS: scaffolding. Nothing here has been executed -- this repo has no Modal
account yet, so no `modal deploy`, no container build, and no GPU run has been
verified. Every "cannot verify" point is called out in a comment where it
matters. See ``deploy/modal/README.md`` for the setup steps that remain.

Why the topology looks like this
-------------------------------
``docs/ARCHITECTURE.md`` is explicit that **the bottleneck is dG on CPU, not
the model on GPU**: the trained net is ~100K parameters in a 5-seed ensemble,
while the expensive stage is one ViennaRNA dG per unique (primer, target) pair.
The failure mode to avoid is the obvious one -- a single expensive GPU
container where the accelerator idles while one core grinds through RNAduplex.

So there are two paths over the *same* stage sequence (`_prepare_stage` ->
thermo -> `_infer_stage`); they differ only in how thermo is parallelised:

1. ``design_single_container``  -- V1 PATH, probably what you run first.
   One container: a GPU plus ``SINGLE_CONTAINER_CPUS`` cores, thermo spread
   over a multiprocessing pool. One cold start, nothing crosses the wire. The
   GPU is still idle during thermo; at design scale that waste is cents, and
   this path is far easier to reason about when something breaks.

2. ``design_fanout``            -- SCALE-UP PATH.
   ``thermo_chunk`` is CPU-only and fans out over primer-set chunks with
   ``.map()`` at high container concurrency; ``infer`` is the only function
   that ever asks for a GPU, and it runs once over the assembled features.
   Use this when the panel is large enough that thermo dominates wall clock.

Both return the same payload: ``report.result_to_dict`` -- the engine's own
"for the web UI" serialisation -- inside a thin envelope carrying the run id,
the request and a timing breakdown. Both can be invoked synchronously
(``.remote()``) or submit-and-poll (``.spawn()`` -> call id -> ``.get()``),
because design runs take tens of seconds to minutes and are not interactive
requests.

``import modal`` is top-level here on purpose: this module is only ever loaded
by the ``modal`` CLI, never by ``lamp_design``. The package itself must keep
importing without ``modal`` or ``streamlit`` installed (both are optional
extras in ``pyproject.toml``), and nothing under ``src/`` imports ``deploy/``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import modal

# ---------------------------------------------------------------------------
# Names and remote paths
# ---------------------------------------------------------------------------

APP_NAME = "lamp-design"

#: Volume holding the trained checkpoints. Weights are NOT in this repo and NOT
#: baked into the image (see .gitignore: `*.pt`, `weights/`). They are mounted
#: read-only at WEIGHTS_REMOTE and resolved through LAMP_DESIGN_WEIGHTS, which
#: is exactly what `models.inference.resolve_weights_dir` reads.
WEIGHTS_VOLUME = os.environ.get("LAMP_DESIGN_WEIGHTS_VOLUME", "lamp-design-weights")
WEIGHTS_REMOTE = "/weights"

#: Volume for run artifacts: the panel and candidates the thermo workers read,
#: and the final result. Modal function *outputs* expire (~24h), so the volume
#: is the durable copy of a result -- see README, "Fetching results later".
RESULTS_VOLUME = os.environ.get("LAMP_DESIGN_RESULTS_VOLUME", "lamp-design-runs")
RESULTS_REMOTE = "/runs"

REPO_REMOTE = "/root/lamp-design"

#: Resolved from this file's location rather than hardcoded, so the image
#: builds from whatever checkout you deploy from. deploy/modal/app.py -> repo.
LOCAL_REPO = Path(__file__).resolve().parents[2]

#: Where LAMP_DESIGN_WEIGHTS points inside the container. The layout
#: `models.inference.discover_checkpoints` expects is `<run>/seed_*/` with one
#: `best_model_*.pt` each, for seeds 0, 1, 2, 3, 42.
WEIGHTS_SUBDIR = os.environ.get(
    "LAMP_DESIGN_WEIGHTS_SUBDIR", "LAMP_experiment_20260905_021750_tp60_A_v0t9"
)

# ---------------------------------------------------------------------------
# Pinned tool versions
#
# IMPORTANT: the installed RNAduplex version changes the dG values the model is
# fed, and dg_ViennaRNA is a frozen model input (types.PHYSICS_FEATURES). A
# deployed run that uses a different RNAduplex than the local run will produce
# different predictions from identical inputs, silently. These pins mirror the
# development box exactly -- apt vienna-rna 2.4.17 for the `RNAduplex` binary,
# pip viennarna 2.7.2 for the in-process `RNA` bindings -- which is the pair
# ARCHITECTURE.md records and warns "may not agree" with each other. Whichever
# one thermo.py ends up using, both sides must agree, and the implementation is
# validated against the real `*_dg_ViennaRNA` columns in the training CSVs, not
# against another implementation. Do not bump these without re-validating.
# ---------------------------------------------------------------------------

#: jammy/multiverse. Ships /usr/bin/RNAduplex 2.4.17. If apt reports "version
#: not found" the archive rotated: re-pin to the then-current jammy version and
#: re-validate dG against the training CSVs before trusting any output. If apt
#: cannot find the package at all, the base image's sources.list is missing
#: `multiverse` (ubuntu:22.04 has it; a slimmer base may not).
VIENNARNA_APT_VERSION = "2.4.17+dfsg-2build2"
RNADUPLEX_EXPECTED_VERSION = "2.4.17"

#: pip wheel providing the `RNA` Python bindings (no CLI tools in this wheel).
VIENNARNA_PY_VERSION = "2.7.2"

#: jammy/universe. MAFFT is what `external/mafft.py` shells out to; prepare
#: needs it for any multi-sequence panel that is not already aligned.
MAFFT_APT_VERSION = "7.490-1"
MAFFT_EXPECTED_VERSION = "7.490"

#: Same base image the team's four scripts under
#: lamp-designer/experiments/modal/ use, so it is known to work on Modal.
#: UNVERIFIED: that image carries torch 2.4.1+cu121 while the dev venv has
#: 2.14.1+cpu. The checkpoints are plain state_dicts, which load across that
#: gap, but `torch.load` defaults changed in 2.6 -- run `doctor_gpu` once the
#: weights volume is populated to confirm the ensemble actually loads, and bump
#: this tag if it does not.
TORCH_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

#: Pinned to the development venv so numeric behaviour matches locally.
PIP_PINS = (
    "numpy==2.2.6",
    "pandas==2.3.3",
    "biopython==1.88",
    # Held below 2.1: 2.1.0 changed the santalucia salt correction and shifts
    # every Tm +2.17 C off the values the model was trained on. Must match the
    # bound in pyproject.toml.
    "primer3-py==2.0.3",
    "scipy==1.15.3",
)

# ---------------------------------------------------------------------------
# Tunables (deploy-time environment, so you can retune without editing code)
# ---------------------------------------------------------------------------

#: The GPU is what matters, and the measurement says so: a 5 kb x 10 target x
#: stride 1 x 4 condition sweep is 7.7 min on one L4 against ~76 min on 8 CPU
#: cores, a 15x gap, with inference 89% of the run and thermo 10.5%. The
#: ensemble is small (~100K parameters x 5 seeds, 3.4 GB peak, 18-29% L4
#: utilization), so a *bigger* GPU buys almost nothing -- more of them is the
#: lever, which is what `design_fanout` is for.
GPU = os.environ.get("LAMP_DESIGN_MODAL_GPU", "L4")

#: GPU containers the inference fan-out spreads shards across. Modal's free
#: Starter plan allows 10 concurrent GPUs and 100 containers; Team allows 50
#: and 5,000. Default to the free-plan ceiling so an unconfigured deploy does
#: not queue behind its own limit.
#:
#: Billing is per GPU-second, so N containers for 1/N of the wall clock costs
#: the same as one container -- minus a few seconds of extra cold start each.
#: That is the whole argument for fanning out: it is close to free.
INFER_MAX_CONTAINERS = int(os.environ.get("LAMP_DESIGN_INFER_CONTAINERS", "10"))

#: Cores per inference container. Encoding rows for the GPU is single-threaded
#: and runs at ~5,450 rows/s, about 42% of the GPU inference stage, so each
#: container wants a couple of cores rather than one -- but not many, because
#: nothing here is CPU-parallel yet. See the "pipelining" note in
#: docs/ARCHITECTURE.md.
INFER_CPUS = float(os.environ.get("LAMP_DESIGN_INFER_CPUS", "4"))

#: Primer sets per shard. Each shard does its own thermo and inference, so a
#: shard must do enough work to amortise a container cold start (seconds) plus
#: loading five checkpoints. At the measured ~2,300 predictions/s and 4
#: conditions, 2,000 sets is roughly 35s of GPU work per shard.
INFER_SHARD_SETS = int(os.environ.get("LAMP_DESIGN_INFER_SHARD_SETS", "2000"))

#: Fan-out width for the CPU thermo stage, used only by the legacy thermo-only
#: fan-out path. Retained because a very wide panel (many targets) shifts the
#: balance back toward thermo, but it is not the default: thermo is 10.5% of a
#: representative run, so fanning it out alone caps at a 1.1x speedup.
THERMO_MAX_CONTAINERS = int(os.environ.get("LAMP_DESIGN_THERMO_CONTAINERS", "50"))
THERMO_CPUS = float(os.environ.get("LAMP_DESIGN_THERMO_CPUS", "2"))

#: Primer sets per chunk, for both the container fan-out and the in-process
#: pool. A chunk must do enough work to amortise its own startup (seconds for a
#: container, ~nothing for a forked worker), so prefer chunks of >=10s of
#: thermo. 256 is a starting guess, NOT a measurement.
THERMO_CHUNK_SIZE = int(os.environ.get("LAMP_DESIGN_THERMO_CHUNK", "256"))

#: Single-container path: one GPU plus this many cores for the thermo pool.
SINGLE_CONTAINER_CPUS = float(os.environ.get("LAMP_DESIGN_SINGLE_CPUS", "16"))
SINGLE_CONTAINER_MEMORY_MB = int(os.environ.get("LAMP_DESIGN_SINGLE_MEMORY_MB", "32768"))

#: MAFFT threads inside prepare.
ALIGN_THREADS = int(os.environ.get("LAMP_DESIGN_ALIGN_THREADS", "4"))

# ---------------------------------------------------------------------------
# Input caps -- so a large upload cannot run away with the account's credits.
# Enforced in _validate_request, which BOTH entrypoints call before any work,
# and then again by the engine's own checks (prepare_panel's max_budget,
# generate_candidates' max_candidates) which are the authoritative ones.
# ---------------------------------------------------------------------------

#: Raw FASTA text. Generous next to the budget cap below; this is the crude
#: guard that stops a huge upload before it is even parsed.
MAX_FASTA_BYTES = int(os.environ.get("LAMP_DESIGN_MAX_FASTA_BYTES", str(5 * 1024 * 1024)))

#: n_targets x alignment width, i.e. TargetPanel.budget -- the metric
#: ARCHITECTURE.md caps on, because runtime scales with the (primer x target)
#: feature matrix and not with the sequence count. MUST TRACK
#: `commands.prepare.DEFAULT_MAX_BUDGET` (50_000): this value is passed into
#: `prepare_panel(max_budget=...)`, so prepare is the single enforcement point
#: and this is just how the deployment configures it.
MAX_BUDGET = int(os.environ.get("LAMP_DESIGN_MAX_BUDGET", "50000"))

#: Candidate sets. Passed to `generate_candidates(max_candidates=...)`, which
#: thins whole anchors at a uniform stride rather than truncating, so the cap
#: degrades gracefully instead of returning a head-of-template prefix.
MAX_CANDIDATE_SETS = int(os.environ.get("LAMP_DESIGN_MAX_CANDIDATE_SETS", "200000"))

MAX_N_SETS = 500
MAX_STRIDE = 500

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(TORCH_IMAGE)
    .apt_install(
        f"vienna-rna={VIENNARNA_APT_VERSION}",
        f"mafft={MAFFT_APT_VERSION}",
    )
    # Fail at BUILD time, not at design time, if the tools are not the ones we
    # pinned. A wrong RNAduplex does not crash -- it quietly returns different
    # dG, which is the worst possible failure for a frozen model input.
    .run_commands(
        f"RNAduplex --version | grep -q '{RNADUPLEX_EXPECTED_VERSION}'",
        f"mafft --version 2>&1 | grep -q '{MAFFT_EXPECTED_VERSION}'",
    )
    .pip_install(f"viennarna=={VIENNARNA_PY_VERSION}")
    .add_local_dir(
        str(LOCAL_REPO),
        remote_path=REPO_REMOTE,
        # .venv is multi-GB and weights/ must never reach the image.
        ignore=[
            "**/.git/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/*.egg-info/**",
            "**/.pytest_cache/**",
            "**/weights/**",
            "**/runs/**",
            "**/*.pt",
            "**/*.pth",
        ],
        copy=True,
    )
    .run_commands(
        f"pip install -e '{REPO_REMOTE}' " + " ".join(f"'{p}'" for p in PIP_PINS),
        # Build-time proof the editable install took, so no function needs
        # sys.path surgery at runtime.
        "python -c 'import lamp_design, RNA; print(lamp_design.__version__, RNA.__version__)'",
    )
    .env(
        {
            "LAMP_DESIGN_WEIGHTS": f"{WEIGHTS_REMOTE}/{WEIGHTS_SUBDIR}",
            "PYTHONUNBUFFERED": "1",
            # Keep BLAS from oversubscribing cores inside a thermo pool worker.
            "OMP_NUM_THREADS": "1",
        }
    )
)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

# A Secret is the way to override LAMP_DESIGN_WEIGHTS (a different checkpoint,
# or a download URL) without rebuilding the image. Modal has no concept of an
# optional Secret -- a missing one fails `modal deploy` -- so it is opt-in at
# deploy time. The default path needs no secret at all.
SECRETS: list = (
    [modal.Secret.from_name(os.environ.get("LAMP_DESIGN_SECRET", "lamp-design-weights"))]
    if os.environ.get("LAMP_DESIGN_USE_SECRET") == "1"
    else []
)

app = modal.App(APP_NAME, image=image)


def _read_only(volume: Any) -> Any:
    """Mount a Volume read-only when the installed client supports it.

    Weights should never be writable from a design container. `read_only()`
    arrived in a recent Modal client; degrade to a writable mount rather than
    failing to deploy on an older one.
    """
    getter = getattr(volume, "read_only", None)
    return getter() if callable(getter) else volume


# ---------------------------------------------------------------------------
# Engine API
#
# This harness calls the engine only through public `lamp_design.*` names --
# never into another module's internals -- and resolves each at the call site
# so a missing one fails with the exact name it wanted. Nothing here stubs or
# fakes a pipeline stage.
#
# Each entry is (preferred top-level re-export, stage-module path). Both now
# resolve: `lamp_design/__init__.py` re-exports every name below, lazily, so the
# first wins and this harness never reaches into a stage module's internals.
#
# Everything below resolves. The two entries that were missing when this file
# was written -- `compute_thermo` and the ThermoTable -> Prediction bridge --
# are implemented, in `thermo.py` and `models/bridge.py`.
# ---------------------------------------------------------------------------

ENGINE_API: dict[str, tuple[str, str]] = {
    # --- exists -----------------------------------------------------------
    # read_fasta(path) -> [(name, sequence)]
    "read_fasta": ("lamp_design.read_fasta", "lamp_design.utils.sequences.read_fasta"),
    # prepare_panel(records, *, max_budget, min_length, pre_aligned,
    #               force_align, threads, aligner, panel_name) -> PreparedPanel
    "prepare_panel": ("lamp_design.prepare_panel", "lamp_design.commands.prepare.prepare_panel"),
    # default_geometries() -> list[Geometry]  (canonical first)
    "default_geometries": (
        "lamp_design.default_geometries",
        "lamp_design.generation.default_geometries",
    ),
    # count_candidate_slots(template_length, *, geometries, stride) -> int
    "count_candidate_slots": (
        "lamp_design.count_candidate_slots",
        "lamp_design.generation.count_candidate_slots",
    ),
    # generate_candidates(template, *, geometries, stride, length_rules,
    #                     max_candidates) -> list[PrimerSet]
    "generate_candidates": (
        "lamp_design.generate_candidates",
        "lamp_design.generation.generate_candidates",
    ),
    # build_design_result(predictions, primer_sets, panel, *,
    #   ranking_condition, tp_threshold, n, min_separation, mismatches,
    #   funnel, excluded_targets, warnings) -> DesignResult
    "build_design_result": (
        "lamp_design.build_design_result",
        "lamp_design.report.build_design_result",
    ),
    # mismatch_map(thermo) -> {(set_id, target_name): total_mismatches}
    # Used by the shard fan-out to ship per-pair mismatch counts back to the
    # orchestrator without shipping the whole ThermoTable.
    "mismatch_map": (
        "lamp_design.mismatch_map",
        "lamp_design.models.bridge.mismatch_map",
    ),
    # result_to_dict(result, *, ranking_condition, tp_threshold) -> dict
    "result_to_dict": ("lamp_design.result_to_dict", "lamp_design.report.result_to_dict"),
    # write_outputs(result, out_prefix, *, ranking_condition, tp_threshold)
    #   -> {role: Path}; writes .sets.csv/.per_target.csv/.json/.summary.txt
    "write_outputs": ("lamp_design.write_outputs", "lamp_design.report.write_outputs"),
    # resolve_weights_dir(explicit=None) -> Path  (reads LAMP_DESIGN_WEIGHTS)
    "resolve_weights_dir": (
        "lamp_design.resolve_weights_dir",
        "lamp_design.models.inference.resolve_weights_dir",
    ),
    # --- MISSING ----------------------------------------------------------
    # compute_thermo(primer_sets, panel) -> ThermoTable
    "compute_thermo": ("lamp_design.compute_thermo", "lamp_design.thermo.compute_thermo"),
    # predict_tp(thermo, primer_sets, conditions, *, weights, device, batch_size)
    #   -> list[Prediction]
    "predict_tp": ("lamp_design.predict_tp", "lamp_design.models.bridge.predict_tp"),
}

_MISSING_HINT = (
    "The LAMP design engine does not expose {key} yet. This harness expects "
    "either {primary} or {fallback}. Implement one of them (and re-export it "
    "from lamp_design/__init__.py); this file deliberately does not stub it."
)


def _engine(key: str) -> Any:
    """Resolve one public engine entrypoint, or raise naming exactly what it wanted."""
    import importlib

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


def _has_engine(key: str) -> bool:
    """Whether one engine entrypoint exists. For diagnostics only."""
    try:
        _engine(key)
    except NotImplementedError:
        return False
    return True


# ---------------------------------------------------------------------------
# Wire format
#
# The result payload is `report.result_to_dict` -- the engine's own
# JSON-for-the-web-UI serialisation, which already carries the funnel, the
# excluded targets, the warnings and the metric directions -- wrapped in a thin
# envelope. Reusing it means there is no second copy of the result schema to
# drift. `deploy/README.md` writes the envelope down.
#
# Panel and candidates go through the volume as JSON: Target/TargetPanel/
# Primer/PrimerSet contain no numpy, so they round-trip exactly and a run's
# intermediate state can be read with `cat`. ThermoTable does hold arrays and
# is passed as the dataclass itself (both ends run the same image and the same
# lamp_design build, so cloudpickle round-trips it).
# ---------------------------------------------------------------------------

ENVELOPE_SCHEMA = "lamp-design/modal-result@1"
REQUEST_SCHEMA = "lamp-design/request@1"


def _panel_to_json(panel: Any) -> dict:
    return {
        "targets": [
            {"name": t.name, "sequence": t.sequence, "weight": float(t.weight)}
            for t in panel.targets
        ],
        "consensus": panel.consensus,
        "alignment": list(panel.alignment) if panel.alignment else None,
    }


def _panel_from_json(blob: dict) -> Any:
    from lamp_design.types import Target, TargetPanel

    return TargetPanel(
        targets=[
            Target(name=t["name"], sequence=t["sequence"], weight=float(t.get("weight", 1.0)))
            for t in blob["targets"]
        ],
        consensus=blob["consensus"],
        alignment=blob.get("alignment"),
    )


def _set_to_json(primer_set: Any) -> dict:
    return {
        "set_id": primer_set.set_id,
        "anchor": int(primer_set.anchor),
        "geometry_id": primer_set.geometry_id,
        "primers": {
            ptype: {
                "primer_type": p.primer_type,
                "sequence": p.sequence,
                "start": int(p.start),
                "end": int(p.end),
            }
            for ptype, p in primer_set.primers.items()
        },
    }


def _set_from_json(blob: dict) -> Any:
    from lamp_design.types import Primer, PrimerSet

    return PrimerSet(
        set_id=blob["set_id"],
        anchor=int(blob["anchor"]),
        geometry_id=blob["geometry_id"],
        primers={
            ptype: Primer(
                primer_type=p["primer_type"],
                sequence=p["sequence"],
                start=int(p["start"]),
                end=int(p["end"]),
            )
            for ptype, p in blob["primers"].items()
        },
    )


def _concat_thermo(tables: list) -> Any:
    """Assemble thermo chunk results into one ThermoTable.

    Order-independent on purpose: every row is labelled by ``set_ids`` /
    ``target_names``, so this does not rely on ``.map()`` preserving input
    order (it does, by default, but encoding that assumption here would be a
    trap). If ``thermo.py`` grows its own concat helper, use that instead.
    """
    import numpy as np

    from lamp_design.types import (
        GLOBAL_PHYSICS_FEATURES,
        PHYSICS_FEATURES,
        PRIMER_TYPE_ORDER,
        ThermoTable,
    )

    tables = [t for t in tables if len(t.set_ids)]
    if not tables:
        return ThermoTable(
            set_ids=[],
            target_names=[],
            values=np.empty((0, len(PRIMER_TYPE_ORDER), len(PHYSICS_FEATURES)), dtype=float),
            globals_=np.empty((0, len(GLOBAL_PHYSICS_FEATURES)), dtype=float),
        )

    set_ids: list[str] = []
    target_names: list[str] = []
    for table in tables:
        set_ids.extend(table.set_ids)
        target_names.extend(table.target_names)

    # target_windows defaults to an empty (0, 0) array in types.py. Only
    # concatenate it when every chunk actually populated it; otherwise leave
    # the default and let the encoder complain about the real problem.
    windows = [t.target_windows for t in tables]
    have_windows = all(w.shape[:1] == (len(t.set_ids),) for w, t in zip(windows, tables))

    merged = ThermoTable(
        set_ids=set_ids,
        target_names=target_names,
        values=np.concatenate([t.values for t in tables], axis=0),
        globals_=np.concatenate([t.globals_ for t in tables], axis=0),
    )
    if have_windows:
        merged.target_windows = np.concatenate(windows, axis=0)
    return merged


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _fasta_shape(text: str) -> tuple[int, int]:
    """(record count, longest sequence) from raw FASTA text, in one pass.

    Enough to bound the work before anything is parsed properly or aligned --
    no engine call, no allocation of the sequences.
    """
    n_records = 0
    longest = 0
    current = 0
    for line in text.splitlines():
        if line.startswith(">"):
            n_records += 1
            longest = max(longest, current)
            current = 0
        else:
            current += len(line.strip())
    return n_records, max(longest, current)


def _validate_request(request: dict) -> dict:
    """Normalise and bounds-check a request. Raises ValueError on rejection.

    Server-side because a client-side limit is not a control. Both entrypoints
    call this before doing anything, so the two paths are bounded identically.
    The engine's own caps (`prepare_panel(max_budget=)`,
    `generate_candidates(max_candidates=)`) are the authoritative ones and are
    configured from the same constants; this is the cheap pre-parse guard that
    stops an absurd upload before MAFFT or generation ever sees it.
    """
    from lamp_design.commands.prepare import DEFAULT_MIN_LENGTH, MAX_SEQUENCES
    from lamp_design.report import DEFAULT_MIN_SEPARATION_BP, DEFAULT_TP_THRESHOLD_MINUTES
    from lamp_design.types import MASTER_MIXES

    if not isinstance(request, dict):
        raise ValueError("request must be a dict")

    fasta = request.get("fasta") or ""
    if not isinstance(fasta, str) or not fasta.strip():
        raise ValueError("request['fasta'] must be non-empty FASTA text")
    n_bytes = len(fasta.encode("utf-8"))
    if n_bytes > MAX_FASTA_BYTES:
        raise ValueError(
            f"FASTA is {n_bytes} bytes, over the {MAX_FASTA_BYTES}-byte cap "
            f"(LAMP_DESIGN_MAX_FASTA_BYTES)"
        )

    master_mix = str(request.get("master_mix", "BST"))
    if master_mix not in MASTER_MIXES:
        raise ValueError(f"master_mix must be one of {list(MASTER_MIXES)}, got {master_mix!r}")

    try:
        target_conc = float(request.get("target_conc", 10000.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("target_conc must be a number") from exc
    if not target_conc > 0:
        # encoding.encode_target_concentration is log10(conc) - 3.
        raise ValueError("target_conc must be > 0 (encoded as log10(conc) - 3)")

    n_default_geometries = len(_engine("default_geometries")())

    def _bounded(name: str, default: Any, low: int, high: int) -> int:
        try:
            value = int(request.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")
        return value

    n_sets = _bounded("n_sets", 20, 1, MAX_N_SETS)
    stride = _bounded("stride", 5, 1, MAX_STRIDE)
    # n_geometries selects a prefix of generation.DEFAULT_GEOMETRIES, which is
    # ordered canonical-first -- so any prefix includes the canonical
    # (5,0,0,5,0,0,5,0) geometry the model is best calibrated on.
    n_geometries = _bounded("n_geometries", min(4, n_default_geometries), 1, n_default_geometries)
    min_separation = _bounded("min_separation", DEFAULT_MIN_SEPARATION_BP, 0, 100000)

    n_records, longest = _fasta_shape(fasta)
    if n_records > MAX_SEQUENCES:
        raise ValueError(
            f"{n_records} sequences given; prepare accepts 1-{MAX_SEQUENCES} "
            "(commands.prepare.MAX_SEQUENCES). Pick a representative subset first."
        )
    # n_records x longest is a lower bound on TargetPanel.budget, since MAFFT
    # only widens an alignment with gaps. prepare_panel re-checks it exactly.
    budget = n_records * longest
    if budget > MAX_BUDGET:
        raise ValueError(
            f"input is {n_records} sequences x {longest} nt = {budget}, over the "
            f"{MAX_BUDGET} budget cap (LAMP_DESIGN_MAX_BUDGET)"
        )
    if longest < DEFAULT_MIN_LENGTH:
        raise ValueError(
            f"the longest sequence is {longest} nt; prepare's floor is "
            f"{DEFAULT_MIN_LENGTH} nt (the tightest canonical geometry spans 167 nt)"
        )

    return {
        "schema": REQUEST_SCHEMA,
        "run_id": str(
            request.get("run_id") or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        ),
        "fasta": fasta,
        "master_mix": master_mix,
        "target_conc": target_conc,
        "n_sets": n_sets,
        "stride": stride,
        "n_geometries": n_geometries,
        "min_separation": min_separation,
        "tp_threshold_minutes": float(
            request.get("tp_threshold_minutes", DEFAULT_TP_THRESHOLD_MINUTES)
        ),
        "pre_aligned": bool(request.get("pre_aligned", False)),
    }


def _resolve_weights() -> Path:
    """The checkpoint directory, via the engine's own resolver where possible."""
    try:
        return _engine("resolve_weights_dir")()
    except NotImplementedError:
        pass
    raw = os.environ.get("LAMP_DESIGN_WEIGHTS", "").strip()
    if not raw or not Path(raw).exists():
        root = Path(WEIGHTS_REMOTE)
        present = sorted(p.name for p in root.iterdir()) if root.exists() else []
        raise RuntimeError(
            f"LAMP_DESIGN_WEIGHTS={raw!r} is unset or missing. {WEIGHTS_REMOTE} contains: "
            f"{present or '<empty>'}. Populate the '{WEIGHTS_VOLUME}' volume "
            "(see deploy/modal/README.md, 'Weights')."
        )
    return Path(raw)


def _run_dir(run_id: str) -> Path:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_.") or "run"
    path = Path(RESULTS_REMOTE) / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunked(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# Pipeline stages, shared by both paths
# ---------------------------------------------------------------------------


def _prepare_stage(req: dict, run_dir: Path) -> tuple[Any, list, dict]:
    """FASTA text -> (PreparedPanel, candidate sets, funnel so far).

    The FASTA is written into the run's own directory and read back with the
    engine's `read_fasta`, so the UI, the CLI and this harness all go through
    one parser.
    """
    read_fasta = _engine("read_fasta")
    prepare_panel = _engine("prepare_panel")
    default_geometries = _engine("default_geometries")
    count_candidate_slots = _engine("count_candidate_slots")
    generate_candidates = _engine("generate_candidates")

    input_fasta = run_dir / "input.fasta"
    input_fasta.write_text(req["fasta"])
    records = read_fasta(input_fasta)

    prepared = prepare_panel(
        records,
        max_budget=MAX_BUDGET,
        pre_aligned=req["pre_aligned"],
        threads=ALIGN_THREADS,
        panel_name=req["run_id"],
    )
    panel = prepared.panel

    geometries = list(default_geometries())[: req["n_geometries"]]
    # Recorded before generation so an empty result can say whether the
    # template was simply too short or the GC/ACGT screen emptied it.
    slots = count_candidate_slots(
        len(panel.consensus), geometries=geometries, stride=req["stride"]
    )
    primer_sets = generate_candidates(
        panel.consensus,
        geometries=geometries,
        stride=req["stride"],
        max_candidates=MAX_CANDIDATE_SETS,
    )

    funnel = dict(prepared.diagnostics.as_funnel()) if prepared.diagnostics else {}
    funnel["candidate_slots"] = int(slots)
    funnel["candidates_generated"] = len(primer_sets)
    return prepared, primer_sets, funnel


def _compute_chunk(primer_sets: list, panel: Any) -> Any:
    """Thermo features for one chunk of primer sets against the whole panel.

    Deduping dG on ``(primer_sequence, target_window)``, one RNAduplex process
    for all pairs, and Tm/GC once per unique primer are all ``thermo.py``'s job
    -- deliberately not reimplemented here. Note that thermo measures at ~3% of
    a run once batched, so this fan-out is a small win next to the GPU stage;
    see ARCHITECTURE.md's performance notes before scaling it up.
    """
    return _engine("compute_thermo")(primer_sets, panel)


def _thermo_worker(job: tuple) -> Any:
    """Pool worker. Module-level so a forked child can reach it."""
    primer_sets, panel = job
    return _compute_chunk(primer_sets, panel)


def _thermo_pool(primer_sets: list, panel: Any, n_workers: int) -> Any:
    """Thermo across a multiprocessing pool, for the single-container path.

    MUST RUN BEFORE TORCH/CUDA IS INITIALISED IN THIS PROCESS. The pool uses
    the `fork` start method -- cheap, and the children inherit this module so
    nothing is re-imported -- and forking a process that has already
    initialised CUDA is a classic hang. `design_single_container` therefore
    orders prepare -> thermo pool -> inference, and inference is the first
    thing that touches torch.
    """
    import multiprocessing

    chunks = list(_chunked(primer_sets, THERMO_CHUNK_SIZE))
    if not chunks:
        return _concat_thermo([])
    workers = max(1, min(int(n_workers), len(chunks)))
    if workers == 1:
        return _concat_thermo([_thermo_worker((chunk, panel)) for chunk in chunks])
    context = multiprocessing.get_context("fork")
    with context.Pool(processes=workers) as pool:
        tables = pool.map(_thermo_worker, [(chunk, panel) for chunk in chunks])
    return _concat_thermo(tables)


def _condition_from_request(req: dict) -> Any:
    """The single reaction condition a request is scored under.

    One place, because the orchestrator, the shards and the single-container
    path all have to agree on it: a shard that predicted under a different
    condition than the one the result is labelled with would be undetectable
    in the output.
    """
    from lamp_design.types import Condition

    return Condition(master_mix=req["master_mix"], target_conc=req["target_conc"])


def _predictions_to_json(predictions: list) -> list:
    """Predictions as plain rows, for returning from a shard."""
    return [
        [p.set_id, p.target_name, p.condition.master_mix,
         float(p.condition.target_conc), float(p.tp_minutes), float(p.tp_std)]
        for p in predictions
    ]


def _predictions_from_json(rows: list) -> list:
    """Inverse of :func:`_predictions_to_json`."""
    from lamp_design.types import Condition, Prediction

    return [
        Prediction(
            set_id=set_id,
            target_name=target,
            condition=Condition(master_mix=mix, target_conc=conc),
            tp_minutes=tp,
            tp_std=std,
        )
        for set_id, target, mix, conc, tp, std in rows
    ]


def _infer_stage(
    thermo: Any,
    panel: Any,
    primer_sets: list,
    req: dict,
    funnel: dict,
    excluded: dict,
    warnings: list,
) -> Any:
    """Assembled features -> DesignResult."""
    condition = _condition_from_request(req)
    weights = _resolve_weights()

    # `predict_tp` takes the primer sets, not the panel: ThermoTable holds the
    # resolved target windows but not the primer sequences, and both are needed
    # to encode a row. `conditions` is a sequence, so one call covers the whole
    # master-mix x concentration sweep.
    #
    # batch_size is left at the engine's default deliberately. Measured on an
    # L4, raising it from 512 to 16384 made things ~7% *slower* and the GPU
    # never passed 29% utilization -- the constraint is row encoding, not the
    # model. Do not tune this without re-measuring.
    predictions = _engine("predict_tp")(
        thermo,
        primer_sets,
        [condition],
        weights=weights,
        device=_device(),
    )

    return _engine("build_design_result")(
        predictions,
        primer_sets,
        panel,
        ranking_condition=condition,
        tp_threshold=req["tp_threshold_minutes"],
        n=req["n_sets"],
        min_separation=req["min_separation"],
        # ThermoTable is an accepted `mismatches` source, so the per-target
        # mismatch columns come out populated rather than blank.
        mismatches=thermo,
        funnel=funnel,
        excluded_targets=excluded,
        warnings=warnings,
    )


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _finish(result: Any, req: dict, run_dir: Path, timing: dict) -> dict:
    """Write the run's outputs and build the envelope that crosses the wire."""
    import json

    from lamp_design.types import Condition

    condition = Condition(master_mix=req["master_mix"], target_conc=req["target_conc"])
    payload = _engine("result_to_dict")(
        result, ranking_condition=condition, tp_threshold=req["tp_threshold_minutes"]
    )
    # The engine's own writer produces .sets.csv / .per_target.csv / .json /
    # .summary.txt, so a run's artifacts are identical to a CLI run's.
    try:
        _engine("write_outputs")(
            result,
            run_dir / "result",
            ranking_condition=condition,
            tp_threshold=req["tp_threshold_minutes"],
        )
    except NotImplementedError:
        pass

    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "run_id": req["run_id"],
        "request": {k: v for k, v in req.items() if k != "fasta"},
        "timing_seconds": {k: round(float(v), 3) for k, v in timing.items()},
        "result": payload,
    }
    # Durable copy: Modal function outputs expire, the volume does not.
    (run_dir / "envelope.json").write_text(json.dumps(envelope, indent=2))
    return envelope


# ---------------------------------------------------------------------------
# Diagnostics -- run these FIRST, before any design. Cheap, and they surface
# the things most likely to be misconfigured: tool versions, weights, and
# which engine stages have landed.
# ---------------------------------------------------------------------------


@app.function(timeout=300, volumes={WEIGHTS_REMOTE: _read_only(weights_volume)})
def doctor() -> dict:
    """Report the container's environment. No GPU, no weights load."""
    import shutil
    import subprocess

    def _version(cmd: list[str]) -> str:
        if shutil.which(cmd[0]) is None:
            return "NOT ON PATH"
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return (out.stdout or out.stderr).strip().splitlines()[0]
        except Exception as exc:  # diagnostics must report, not raise
            return f"error: {exc}"

    import lamp_design
    import numpy
    import RNA

    weights_root = Path(WEIGHTS_REMOTE)
    try:
        weights_status = str(_resolve_weights())
    except Exception as exc:
        weights_status = f"UNRESOLVED: {exc}"

    return {
        "rnaduplex": _version(["RNAduplex", "--version"]),
        "rnaduplex_expected": RNADUPLEX_EXPECTED_VERSION,
        "mafft": _version(["mafft", "--version"]),
        "rna_bindings": RNA.__version__,
        "numpy": numpy.__version__,
        "lamp_design": lamp_design.__version__,
        "weights_env": os.environ.get("LAMP_DESIGN_WEIGHTS", ""),
        "weights": weights_status,
        "weights_volume_contents": (
            sorted(p.name for p in weights_root.iterdir()) if weights_root.exists() else []
        ),
        "engine_api_present": sorted(k for k in ENGINE_API if _has_engine(k)),
        "engine_api_missing": sorted(k for k in ENGINE_API if not _has_engine(k)),
    }


@app.function(gpu=GPU, timeout=600, volumes={WEIGHTS_REMOTE: _read_only(weights_volume)})
def doctor_gpu() -> dict:
    """GPU facts, and a real load of the 5-seed ensemble from the volume.

    Reports failure; never fakes success. This is the step most likely to break
    on the torch-version gap noted at TORCH_IMAGE.
    """
    import torch

    facts: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_requested": GPU,
    }
    try:
        weights = _resolve_weights()
    except Exception as exc:
        facts["ensemble"] = f"weights unresolved: {exc}"
        return facts
    facts["weights"] = str(weights)
    facts["seed_dirs"] = sorted(p.name for p in weights.glob("seed_*"))
    try:
        from lamp_design.models.inference import LyraEnsemble

        ensemble = LyraEnsemble.load(weights, device=_device())
        facts["ensemble"] = "loaded"
        facts["ensemble_seeds"] = [m.seed for m in ensemble.members]
        facts["d_input"] = ensemble.d_input
    except Exception as exc:
        facts["ensemble"] = f"{type(exc).__name__}: {exc}"
    return facts


# ---------------------------------------------------------------------------
# Stage 1 (CPU, fan-out): thermodynamic features.
#
# No GPU here, by design -- this is the stage ARCHITECTURE.md identifies as the
# bottleneck. `retries=2` is safe because thermo is a pure function of
# (primer sets, panel): a retried chunk recomputes the same numbers.
# ---------------------------------------------------------------------------

#: Containers are reused across inputs, so cache the panel per run id rather
#: than re-reading it from the volume for every chunk.
_PANEL_CACHE: dict[str, Any] = {}


@app.function(
    cpu=THERMO_CPUS,
    timeout=1800,
    retries=2,
    max_containers=THERMO_MAX_CONTAINERS,
    volumes={RESULTS_REMOTE: results_volume},
)
def thermo_chunk(job: dict) -> Any:
    """Features for one chunk of primer sets against the whole panel.

    The panel is read from the results volume rather than passed in the input,
    so a wide fan-out does not re-serialise the panel once per chunk.
    """
    import json

    run_id = job["run_id"]
    panel = _PANEL_CACHE.get(run_id)
    if panel is None:
        results_volume.reload()
        panel_path = _run_dir(run_id) / "panel.json"
        if not panel_path.exists():
            raise RuntimeError(f"panel.json missing for run {run_id}; orchestrator did not commit")
        panel = _panel_from_json(json.loads(panel_path.read_text()))
        _PANEL_CACHE[run_id] = panel

    return _compute_chunk([_set_from_json(blob) for blob in job["sets"]], panel)


# ---------------------------------------------------------------------------
# Stage 2 (GPU, single container): inference + ranking.
#
# The only function in the fan-out path that asks for a GPU. Ranking runs here
# too because it is cheap -- a second container would cost more in cold start
# than it saves.
# ---------------------------------------------------------------------------


@app.function(
    gpu=GPU,
    timeout=1800,
    retries=1,
    secrets=SECRETS,
    volumes={
        WEIGHTS_REMOTE: _read_only(weights_volume),
        RESULTS_REMOTE: results_volume,
    },
)
def infer(run_id: str, thermo_tables: list, req: dict, timing: dict) -> dict:
    """Assembled features -> ranked result envelope."""
    import json

    started = time.time()
    results_volume.reload()
    run_dir = _run_dir(run_id)
    panel = _panel_from_json(json.loads((run_dir / "panel.json").read_text()))
    primer_sets = [
        _set_from_json(blob) for blob in json.loads((run_dir / "candidates.json").read_text())
    ]
    meta = json.loads((run_dir / "prepare_meta.json").read_text())

    result = _infer_stage(
        _concat_thermo(thermo_tables),
        panel,
        primer_sets,
        req,
        funnel=meta["funnel"],
        excluded=meta["excluded"],
        warnings=meta["warnings"],
    )
    timing = {**timing, "infer": time.time() - started}
    timing["total"] = sum(timing.values())
    envelope = _finish(result, req, run_dir, timing)
    results_volume.commit()
    return envelope


# ---------------------------------------------------------------------------
# Orchestrator for the fan-out path (CPU only; it never touches a GPU).
# ---------------------------------------------------------------------------


@app.function(
    cpu=2.0,
    timeout=3600,
    # retries=0 deliberately: retrying the orchestrator would re-run the whole
    # fan-out. The stages retry individually instead.
    retries=0,
    volumes={RESULTS_REMOTE: results_volume},
)
def design_fanout(request: dict) -> dict:
    """SCALE-UP PATH: shard the candidates across many GPU containers.

    Prepare and generation run inline here rather than in their own Modal
    functions: they are cheap (0.5s of a 7.7 min run), and the orchestrator
    already holds a CPU container, so separate functions would only add cold
    starts.

    ## Why shard candidates rather than stages

    Every candidate set's predicted Tp is independent of every other's --
    ranking is the only step that sees them together, and it happens here at
    the end. So the whole thermo + inference span is embarrassingly parallel
    over candidate sets, and a shard can do both stages locally instead of
    shipping a ThermoTable between containers.

    That matters because of where the time actually goes: inference is 89% of
    a representative run and thermo 10.5%. Fanning out thermo alone -- which
    is what this function used to do -- caps at about 1.1x no matter how many
    containers it gets. Sharding the pair makes both stages scale.

    ## Why this is nearly free

    Modal bills per GPU-second. The total GPU-seconds of work does not change
    when it is spread over more containers, so N shards cost the same as one
    and finish in roughly 1/N of the wall clock. The only extra is a cold
    start per container, which is why :data:`INFER_SHARD_SETS` is sized so a
    shard does tens of seconds of work.
    """
    import json

    req = _validate_request(request)
    run_id = req["run_id"]
    run_dir = _run_dir(run_id)
    timing: dict[str, float] = {}

    started = time.time()
    prepared, primer_sets, funnel = _prepare_stage(req, run_dir)
    timing["prepare"] = time.time() - started
    panel = prepared.panel

    # Publish the shared inputs once; every shard reads them from here rather
    # than having the panel serialized into each of its arguments.
    (run_dir / "panel.json").write_text(json.dumps(_panel_to_json(panel)))
    (run_dir / "candidates.json").write_text(
        json.dumps([_set_to_json(s) for s in primer_sets])
    )
    (run_dir / "prepare_meta.json").write_text(
        json.dumps(
            {
                "funnel": funnel,
                "excluded": dict(prepared.excluded),
                "warnings": list(prepared.warnings),
            }
        )
    )
    results_volume.commit()

    shards = [
        {"run_id": run_id, "sets": [_set_to_json(s) for s in shard], "req": req}
        for shard in _chunked(primer_sets, INFER_SHARD_SETS)
    ]
    print(
        f"run={run_id} targets={panel.n_targets} budget={panel.budget} "
        f"sets={len(primer_sets)} shards={len(shards)} "
        f"max_containers={INFER_MAX_CONTAINERS} gpu={GPU}",
        flush=True,
    )

    started = time.time()
    predictions: list = []
    mismatches: dict = {}
    n_pairs = 0
    for shard_result in design_shard.map(shards):
        predictions.extend(_predictions_from_json(shard_result["predictions"]))
        mismatches.update(
            {(s, t): m for s, t, m in shard_result["mismatches"]}
        )
        n_pairs += shard_result["n_pairs"]
    timing["shards"] = time.time() - started
    print(
        f"run={run_id} merged {len(predictions):,} predictions over "
        f"{n_pairs:,} pairs from {len(shards)} shard(s)",
        flush=True,
    )

    # Ranking is the one step that needs every candidate at once, so it runs
    # here, after the merge, on the orchestrator's CPU.
    started = time.time()
    result = _engine("build_design_result")(
        predictions,
        primer_sets,
        panel,
        ranking_condition=_condition_from_request(req),
        tp_threshold=req["tp_threshold_minutes"],
        n=req["n_sets"],
        min_separation=req["min_separation"],
        mismatches=mismatches,
        funnel=funnel,
        excluded_targets=dict(prepared.excluded),
        warnings=list(prepared.warnings),
    )
    timing["rank"] = time.time() - started
    timing["total"] = sum(timing.values())
    envelope = _finish(result, req, run_dir, timing)
    results_volume.commit()
    return envelope


@app.function(
    gpu=GPU,
    cpu=INFER_CPUS,
    max_containers=INFER_MAX_CONTAINERS,
    timeout=3600,
    retries=1,
    secrets=SECRETS,
    volumes={
        WEIGHTS_REMOTE: _read_only(weights_volume),
        RESULTS_REMOTE: results_volume,
    },
)
def design_shard(job: dict) -> dict:
    """Thermo + inference for one shard of candidate sets.

    Returns predictions and per-pair mismatch counts rather than a
    ``ThermoTable``: the orchestrator only needs those two things to rank, and
    they are a few hundred bytes per pair against a ThermoTable's arrays.
    """
    import json

    results_volume.reload()
    run_dir = _run_dir(job["run_id"])
    panel = _panel_from_json(json.loads((run_dir / "panel.json").read_text()))
    primer_sets = [_set_from_json(blob) for blob in job["sets"]]
    req = job["req"]

    thermo = _compute_chunk(primer_sets, panel)
    predictions = _engine("predict_tp")(
        thermo,
        primer_sets,
        [_condition_from_request(req)],
        weights=_resolve_weights(),
        device=_device(),
    )
    return {
        "predictions": _predictions_to_json(predictions),
        "mismatches": [
            [set_id, target, int(count)]
            for (set_id, target), count in _engine("mismatch_map")(thermo).items()
        ],
        "n_pairs": len(thermo.set_ids),
    }


# ---------------------------------------------------------------------------
# Single-container path.
# ---------------------------------------------------------------------------


@app.function(
    gpu=GPU,
    cpu=SINGLE_CONTAINER_CPUS,
    memory=SINGLE_CONTAINER_MEMORY_MB,
    timeout=3600,
    retries=0,
    secrets=SECRETS,
    volumes={
        WEIGHTS_REMOTE: _read_only(weights_volume),
        RESULTS_REMOTE: results_volume,
    },
)
def design_single_container(request: dict) -> dict:
    """V1 PATH: the whole pipeline in one container.

    One GPU plus SINGLE_CONTAINER_CPUS cores, with thermo spread over a
    multiprocessing pool so the cores stay busy even though the GPU is idle
    during that stage.

    TRADEOFF vs design_fanout: one cold start and no serialisation, against a
    GPU that sits partly idle and a hard ceiling of one machine's cores. At
    design scale that idle GPU costs cents; past the point where thermo
    dominates wall clock, switch to design_fanout. See deploy/README.md.

    Stage order matters: the thermo pool forks, so it must run before anything
    initialises CUDA. See _thermo_pool.
    """
    req = _validate_request(request)
    run_dir = _run_dir(req["run_id"])
    timing: dict[str, float] = {}

    started = time.time()
    prepared, primer_sets, funnel = _prepare_stage(req, run_dir)
    timing["prepare"] = time.time() - started

    started = time.time()
    thermo = _thermo_pool(primer_sets, prepared.panel, int(SINGLE_CONTAINER_CPUS))
    timing["thermo"] = time.time() - started

    started = time.time()
    result = _infer_stage(
        thermo,
        prepared.panel,
        primer_sets,
        req,
        funnel=funnel,
        excluded=dict(prepared.excluded),
        warnings=list(prepared.warnings),
    )
    timing["infer"] = time.time() - started
    timing["total"] = sum(timing.values())

    envelope = _finish(result, req, run_dir, timing)
    results_volume.commit()
    return envelope


# ---------------------------------------------------------------------------
# Local entrypoint.
#
# Submit-and-poll is Modal-native and needs nothing from this file: against a
# deployed app, `modal.Function.from_name(APP_NAME, "design_single_container")`
# then `.spawn(request)` returns a FunctionCall whose `.object_id` is the poll
# token, and `modal.FunctionCall.from_id(token).get(timeout=0)` either returns
# the envelope, raises TimeoutError while it is still running, or re-raises the
# remote exception. deploy/streamlit/app.py uses exactly that; the README
# spells it out. `--mode sync` below is the synchronous form for small jobs.
#
# An HTTP front door (@modal.fastapi_endpoint + fastapi in the image) is
# deliberately NOT built: every current caller is Python.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    mode: str = "doctor",
    fasta: str = "",
    master_mix: str = "BST",
    target_conc: float = 10000.0,
    n_sets: int = 20,
    stride: int = 5,
    n_geometries: int = 4,
    pre_aligned: bool = False,
) -> None:
    """`modal run deploy/modal/app.py [--mode doctor|sync|fanout|spawn] [--fasta FILE]`."""
    import json

    if mode == "doctor":
        print(json.dumps(doctor.remote(), indent=2), flush=True)
        print(json.dumps(doctor_gpu.remote(), indent=2), flush=True)
        return

    if not fasta:
        raise SystemExit("--fasta PATH is required for modes sync/fanout/spawn")
    path = Path(fasta)
    n_bytes = path.stat().st_size
    if n_bytes > MAX_FASTA_BYTES:
        # Fail before uploading anything.
        raise SystemExit(f"{path} is {n_bytes} bytes, over the {MAX_FASTA_BYTES}-byte cap")

    request = {
        "fasta": path.read_text(),
        "master_mix": master_mix,
        "target_conc": target_conc,
        "n_sets": n_sets,
        "stride": stride,
        "n_geometries": n_geometries,
        "pre_aligned": pre_aligned,
    }

    if mode == "sync":
        envelope = design_single_container.remote(request)
    elif mode == "fanout":
        envelope = design_fanout.remote(request)
    elif mode == "spawn":
        call = design_single_container.spawn(request)
        print(f"submitted call_id={call.object_id}", flush=True)
        print("poll with: modal.FunctionCall.from_id(call_id).get(timeout=0)", flush=True)
        envelope = call.get()
    else:
        raise SystemExit(f"unknown --mode {mode!r}; expected doctor, sync, fanout or spawn")

    result = envelope["result"]
    print(
        json.dumps(
            {
                "run_id": envelope["run_id"],
                "timing_seconds": envelope["timing_seconds"],
                "funnel": result["funnel"],
                "warnings": result["warnings"],
                "excluded_targets": result["excluded_targets"],
                "n_sets_returned": result["n_sets_returned"],
                "top_sets": result["sets"][:3],
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        f"full artifacts: modal volume get {RESULTS_VOLUME} {envelope['run_id']}/ ./",
        flush=True,
    )
