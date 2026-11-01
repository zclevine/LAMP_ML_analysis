"""Modal harness for the LAMP design engine.

STATUS: scaffolding. Nothing here has been executed -- this repo has no Modal
account yet, so no `modal deploy`, no container build, and no GPU run has been
verified. Every "cannot verify" point is called out in a comment where it
matters. See ``deploy/modal/README.md`` for the setup steps that remain.

Why the topology looks like this
-------------------------------
An earlier version of this file said the bottleneck was dG on CPU rather than
the model on GPU, and built a thermo-only fan-out around that claim. **It is
the other way around.** Measured on an L4 over a stride-1 sweep (461.18 s,
940,520 predictions, docs/ARCHITECTURE.md):

    generation   0.50 s   0.1%
    thermo      48.60 s  10.5%
    inference  410.35 s  89.0%     <- this is the stage that costs
    report       1.72 s   0.4%

The net really is ~100K parameters in a 5-seed ensemble, but it is run over
hundreds of thousands of (set, target, condition) rows, and batching dG
removed thermo as a rival (38x, plus a 16x dedup factor). So fanning out
thermo alone caps at 1/(1-0.105) = **1.1x** by Amdahl, which is why that path
is gone. Both surviving paths run the same stage sequence (`_prepare_stage` ->
thermo -> `_infer_stage`) and differ in how the *candidates* are split:

1. ``design_single_container``  -- V1 PATH, probably what you run first.
   One container: a GPU plus ``SINGLE_CONTAINER_CPUS`` cores, thermo spread
   over a multiprocessing pool. One cold start, nothing crosses the wire.

2. ``design_fanout``            -- SCALE-UP PATH.
   Shards the candidate sets across up to ``INFER_MAX_CONTAINERS`` GPU
   containers (``design_shard``), each doing its own thermo + inference, and
   ranks after the merge. Every set's predicted Tp is independent of every
   other's, so this is embarrassingly parallel over 99.5% of the run. Modal
   bills per GPU-second, so N containers cost what one does and finish in
   ~1/N the wall clock.

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
from typing import Any, Iterator, Optional

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
#:
#: Guarded, because this module is imported INSIDE the container too: there the
#: file lands at /root/app.py, `parents` is just (/root, /), and the bare
#: `parents[2]` raised IndexError at import and killed every function during
#: startup -- the first real Modal run failed on exactly this. Only the client
#: side (the image definition below) ever needs the true checkout path; in the
#: container the repo is already mounted at REPO_REMOTE.
_THIS_FILE = Path(__file__).resolve()
LOCAL_REPO = (
    _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else Path(REPO_REMOTE)
)

#: Where LAMP_DESIGN_WEIGHTS points inside the container. The layout
#: `models.inference.discover_checkpoints` expects is `<run>/seed_*/` with one
#: `best_model_*.pt` each, for seeds 0, 1, 2, 3, 42.
#: Checkpoint directory inside the weights volume. Named for the training run
#: suffix rather than the full experiment directory, to match the local layout
#: (~/lamp-weights/tp60_A_v0t9/seed_{0,1,2,3,42}/best_model.pt) so the same
#: name works locally and remotely. Override with LAMP_DESIGN_WEIGHTS_SUBDIR.
WEIGHTS_SUBDIR = os.environ.get("LAMP_DESIGN_WEIGHTS_SUBDIR", "tp60_A_v0t9")

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
    # models/lyra/s4d.py imports this at module level, so it is required on
    # the inference path, not optional. Pinned like the rest because this list
    # is the reproducibility record for a deployed run.
    "einops==0.8.2",
)

# ---------------------------------------------------------------------------
# Tunables (deploy-time environment, so you can retune without editing code)
# ---------------------------------------------------------------------------

#: The GPU is what matters: a 5 kb x 10 target x stride 1 sweep is minutes on
#: one GPU against ~76 min on 8 CPU cores.
#:
#: L40S is the default because it was measured, head to head, on the same
#: seeded 10 x 5 kb panel (22,773 candidate sets, 1x2 conditions), both via
#: `design_single_container`:
#:
#:     L4    thermo 11.70s  infer 207.80s  total 220.34 GPU-s  $0.0489
#:     L40S  thermo  6.83s  infer  76.74s  total  84.07 GPU-s  $0.0456
#:
#: So the L40S is 2.6x faster *and* marginally cheaper per run, because its
#: 2.44x rate ($0.000542/s vs $0.000222/s) is less than its 2.71x speedup on
#: the inference stage. Results are bit-identical: same 20 sets, same order,
#: max |diff| 0.0 across every reported score.
#:
#: This contradicts what this comment said before -- that the ensemble is
#: small (~100K parameters x 5 seeds, 3.4 GB peak, 18-29% L4 utilization) so a
#: bigger GPU "buys almost nothing, more of them is the lever". Low measured
#: utilization did not mean what it looked like: the stage is latency-bound on
#: many small sequential kernels, and the L40S has both higher clocks and
#: enough extra SMs to retire each one faster. Utilization percentage was the
#: wrong instrument; a stopwatch was the right one.
GPU = os.environ.get("LAMP_DESIGN_MODAL_GPU", "L40S")

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
#: loading five checkpoints.
#:
#: BEWARE: at the current maximum input, fanning out is probably not worth it.
#: A full stride-1 sweep is 22,773 sets = 84 GPU-s total on one L40S. Split 12
#: ways that is ~7s of work per shard, against a cold start plus five
#: checkpoint loads -- measured once at 26.6 GPU-s across 5 shards to do 7.3
#: GPU-s of work. Fan-out is for *throughput* (many concurrent users) or an
#: input well beyond the design notes' maximum, not for latency on one run.
#: One container at ~1.6 min is the better answer for a single design.
#:
#: 2,000 is also mis-sized for that input: 22,773 / 2,000 = 12 shards against
#: INFER_MAX_CONTAINERS = 10, so two shards run in a second wave while eight
#: GPUs idle. If you do fan out, size this as ceil(n_sets / n_containers).
INFER_SHARD_SETS = int(os.environ.get("LAMP_DESIGN_INFER_SHARD_SETS", "2000"))

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
# Spend circuit breaker
#
# Two jobs: keep an honest run from quietly eating the account's credits, and
# keep abusive or runaway traffic from doing it fast. The input caps further
# down bound what ONE request can cost; these bound what ALL requests can cost
# per unit time, which is the only thing that helps when the requests keep
# coming.
#
# Accounting is in GPU-seconds, recorded by this app into a `modal.Dict` after
# each run. Three thresholds, deliberately different in kind:
#
#   * hourly  -> DEGRADE. Stop fanning out; run single-container instead. The
#     job still completes, just on one GPU, so a busy hour slows down rather
#     than failing.
#   * daily   -> REFUSE, with a message saying when it resets.
#   * total   -> REFUSE. The cumulative ceiling the project agreed to.
#
# HONEST LIMITS OF THIS LEDGER, because a budget control that is trusted
# further than it works is worse than none:
#   1. It counts only what goes through these functions. `doctor`, other apps
#      in the workspace, and anything run by hand are invisible to it.
#   2. A container killed mid-run (timeout, OOM, preemption) may never reach
#      its record call, so usage is under-counted on exactly the failures that
#      waste the most.
#   3. It is advisory, enforced in our code. Modal's own workspace spend limit
#      is the only control that cannot be bypassed by a bug in this file, and
#      it should be set in the dashboard as well.
# ---------------------------------------------------------------------------

#: Fan-out is disabled above this many GPU-hours in the trailing hour.
#:
#: NOTE: on the free plan `INFER_MAX_CONTAINERS` = 10 makes 10 GPU-hours per
#: wall-clock hour the physical ceiling, so a 20 GPU-hour hourly limit cannot
#: currently be reached -- max_containers is the tighter control. This stays
#: because that ceiling moves (the Team plan allows 50 concurrent GPUs) and a
#: limit that only matters later is still worth writing down now.
BUDGET_HOURLY_GPU_HOURS = float(os.environ.get("LAMP_DESIGN_HOURLY_GPU_HOURS", "20"))

#: Runs are refused above this many GPU-hours in the trailing 24 hours.
BUDGET_DAILY_GPU_HOURS = float(os.environ.get("LAMP_DESIGN_DAILY_GPU_HOURS", "50"))

#: Runs are refused above this many GPU-hours in total, ever.
#:
#: 20 is the cumulative ceiling agreed for this project. It is also the one
#: threshold that binds in practice: at the L4 rate of $0.000222/GPU-s, the
#: free plan's $30 monthly credit is only ~37.5 GPU-hours, so the *credit*
#: runs out well before 50 GPU-hours/day could ever trigger. Keeping the total
#: under the credit is what actually prevents a surprise bill.
BUDGET_TOTAL_GPU_HOURS = float(os.environ.get("LAMP_DESIGN_TOTAL_GPU_HOURS", "20"))

#: Set to "0" to run with no ledger at all (for a deliberate benchmark).
BUDGET_ENABLED = os.environ.get("LAMP_DESIGN_BUDGET", "1") not in {"0", "false", "no"}

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

#: Cap on concentrations per request. Each one is another full inference pass
#: over every candidate, and inference is 89% of a run -- so this is the knob
#: that multiplies cost, and it is the one worth bounding server-side. 4 leaves
#: room for the engine's default 2 plus a couple, and refuses a sweep that
#: would quietly cost 10x.
MAX_CONCENTRATIONS = int(os.environ.get("LAMP_DESIGN_MAX_CONCENTRATIONS", "4"))

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
            # ---------------------------------------------------------------
            # Tunables that are read INSIDE a function body have to be baked
            # into the image, not just exported in the deploying shell.
            #
            # The README said "set them in the shell you run `modal deploy`
            # from", which is only true for values the *client* evaluates --
            # `gpu=`, `cpu=`, `max_containers=` on the decorators. Anything
            # read via os.environ at module scope and then used in a container
            # (shard size, chunk size, the budget ceilings) gets the container's
            # environment, which did not carry them: a run with
            # LAMP_DESIGN_INFER_SHARD_SETS=25 still sharded at the default
            # 2000 and silently produced one shard. Forwarding them here makes
            # the documented behaviour real.
            # ---------------------------------------------------------------
            "LAMP_DESIGN_INFER_SHARD_SETS": str(INFER_SHARD_SETS),
            "LAMP_DESIGN_THERMO_CHUNK": str(THERMO_CHUNK_SIZE),
            "LAMP_DESIGN_ALIGN_THREADS": str(ALIGN_THREADS),
            "LAMP_DESIGN_MAX_FASTA_BYTES": str(MAX_FASTA_BYTES),
            "LAMP_DESIGN_MAX_BUDGET": str(MAX_BUDGET),
            "LAMP_DESIGN_MAX_CANDIDATE_SETS": str(MAX_CANDIDATE_SETS),
            "LAMP_DESIGN_MAX_CONCENTRATIONS": str(MAX_CONCENTRATIONS),
            # The spend ceilings especially: a breaker configured only on the
            # client would not be a breaker at all.
            "LAMP_DESIGN_HOURLY_GPU_HOURS": str(BUDGET_HOURLY_GPU_HOURS),
            "LAMP_DESIGN_DAILY_GPU_HOURS": str(BUDGET_DAILY_GPU_HOURS),
            "LAMP_DESIGN_TOTAL_GPU_HOURS": str(BUDGET_TOTAL_GPU_HOURS),
            "LAMP_DESIGN_BUDGET": "1" if BUDGET_ENABLED else "0",
        }
    )
)

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

#: The spend ledger. A Dict rather than a file on the results volume because
#: several containers record into it concurrently and a volume gives no
#: read-modify-write ordering, while a Dict entry is a single atomic put.
budget_dict = modal.Dict.from_name(
    os.environ.get("LAMP_DESIGN_BUDGET_DICT", "lamp-design-budget"),
    create_if_missing=True,
)

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
    # geometry_plan(n_geometries, *, fixed) -> {"geometries": [...],
    #                                            "n_sampled": int}
    "geometry_plan": (
        "lamp_design.geometry_plan",
        "lamp_design.generation.geometry_plan",
    ),
    # count_candidate_slots(template_length, *, geometries, n_sampled,
    #                       stride) -> int
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

    # target_windows is REQUIRED, and it has to go in the constructor.
    #
    # This was previously assigned *after* construction, behind a `have_windows`
    # guard that left the field at its default when a chunk looked unpopulated.
    # That could never work: `ThermoTable.__post_init__` validates
    # target_windows.shape == (n, 8), so construction raised
    # "target_windows must have shape (344, 8), got (0, 8)" before the
    # assignment line was ever reached. The first real Modal design run died
    # on exactly that. (The old comment claimed the default was an unvalidated
    # (0, 0); types.py has since both reshaped and validated it.)
    #
    # Failing loudly on a wrong shape is also right on the merits: the
    # sequence encoder indexes target_windows[i, j] in lockstep with
    # values[i, j], so a silently-defaulted field is a mis-encoding, not a
    # missing nicety.
    windows = [t.target_windows for t in tables]
    wrong = [
        f"chunk {i}: {w.shape} != {(len(t.set_ids), len(PRIMER_TYPE_ORDER))}"
        for i, (w, t) in enumerate(zip(windows, tables))
        if w.shape != (len(t.set_ids), len(PRIMER_TYPE_ORDER))
    ]
    if wrong:
        raise ValueError(
            "thermo chunk(s) returned target_windows of the wrong shape, which "
            "cannot be defaulted away because the encoder reads it positionally: "
            + "; ".join(wrong)
        )

    return ThermoTable(
        set_ids=set_ids,
        target_names=target_names,
        values=np.concatenate([t.values for t in tables], axis=0),
        globals_=np.concatenate([t.globals_ for t in tables], axis=0),
        target_windows=np.concatenate(windows, axis=0),
    )


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

    # Concentrations are a LIST. The engine's default sweep is one master mix
    # at two concentrations (commands.design.DEFAULT_CONCENTRATIONS: one
    # comfortable, one near the limit of detection), and a remote run that
    # accepted only one would hand back a narrower table than the identical
    # local run -- silently, because nothing would error. `target_conc` is
    # still accepted as a one-element shorthand.
    from lamp_design.commands.design import DEFAULT_CONCENTRATIONS

    raw_concs = request.get("target_concs")
    if raw_concs is None:
        single = request.get("target_conc")
        raw_concs = [single] if single is not None else list(DEFAULT_CONCENTRATIONS)
    if not isinstance(raw_concs, (list, tuple)):
        raw_concs = [raw_concs]
    try:
        target_concs = [float(c) for c in raw_concs]
    except (TypeError, ValueError) as exc:
        raise ValueError("target_concs must be a list of numbers") from exc
    if not target_concs:
        raise ValueError("target_concs must not be empty")
    if any(c <= 0 for c in target_concs):
        # encoding.encode_target_concentration is log10(conc) - 3.
        raise ValueError("every target_conc must be > 0 (encoded as log10(conc) - 3)")
    if len(target_concs) > MAX_CONCENTRATIONS:
        raise ValueError(
            f"{len(target_concs)} concentrations given; the cap is "
            f"{MAX_CONCENTRATIONS} (LAMP_DESIGN_MAX_CONCENTRATIONS). Every "
            "extra concentration is another full inference pass over every "
            "candidate, and inference is 89% of a run."
        )
    # Dedupe, order-preserving: a duplicate would double the inference cost
    # for a column that already exists.
    deduped: list[float] = []
    for conc in target_concs:
        if conc not in deduped:
            deduped.append(conc)
    target_concs = deduped

    # The concentration sets are *ranked* under. Explicit, else the highest,
    # matching commands.design._ranking_condition -- so the low-copy
    # prediction is the robustness column and never silently drives ranking.
    raw_rank = request.get("rank_conc")
    try:
        target_conc = float(raw_rank) if raw_rank is not None else max(target_concs)
    except (TypeError, ValueError) as exc:
        raise ValueError("rank_conc must be a number") from exc
    if target_conc not in target_concs:
        raise ValueError(
            f"rank_conc {target_conc:g} is not among the concentrations being "
            f"scored ({', '.join(format(c, 'g') for c in target_concs)}); it has "
            "to be one of them or the ranking refers to a prediction that was "
            "never made"
        )

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
    # Layouts per position: the canonical (5,0,0,5,0,0,5,0) geometry the model
    # is best calibrated on, plus n_geometries - 1 sampled per anchor. Capped
    # at the fixed panel's size because the design notes ask for five per
    # position ("i.e. let's say 5"), not because of anything in the code.
    n_geometries = _bounded(
        "n_geometries", n_default_geometries, 1, n_default_geometries
    )
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
        #: Every concentration scored, in request order.
        "target_concs": target_concs,
        #: The one that ranks. Always a member of `target_concs`.
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
# Spend ledger (see "Spend circuit breaker" above for what it does NOT cover)
# ---------------------------------------------------------------------------

#: One key holding both the recent-run log and the running total, so a single
#: atomic put keeps the two consistent with each other.
_BUDGET_KEY = "usage"

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86400.0

#: USD per GPU-second, from modal.com/pricing (checked 2026-11-01). Used only
#: to turn the ledger's GPU-seconds into an indicative dollar figure; the
#: breaker itself gates on GPU-hours, so a stale rate here cannot affect
#: whether work is refused. The authoritative number is the Modal dashboard.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
}


def _gpu_rate(name: str) -> float:
    """Rate for a GPU name, falling back to the configured default's rate."""
    return GPU_USD_PER_SECOND.get(
        name, GPU_USD_PER_SECOND.get(GPU, GPU_USD_PER_SECOND["L4"])
    )


class BudgetExceeded(RuntimeError):
    """The spend circuit breaker is open.

    Deliberately not a ValueError: a caller needs to tell "your request is
    malformed, fix it" from "the budget is spent, come back later", and the
    two deserve different HTTP statuses when the endpoint exists (400 vs 429).
    """


def _budget_state() -> dict:
    """The ledger as it stands, or a fresh one.

    Never raises. A budget check that crashes the run it is meant to protect
    is strictly worse than no budget check, so an unreachable or corrupt Dict
    degrades to "no usage recorded" rather than to a failed design.
    """
    try:
        state = budget_dict.get(_BUDGET_KEY)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(f"[budget] ledger unreadable, treating as empty: {exc}", flush=True)
        state = None
    if not isinstance(state, dict):
        return {"runs": [], "total_gpu_seconds": 0.0}
    runs = state.get("runs")
    # Coerced defensively: a non-numeric total (a partial write, a hand-edited
    # Dict) would otherwise raise ValueError here and take down every run,
    # which is the opposite of what a spend control should do when its own
    # bookkeeping is damaged.
    try:
        total = float(state.get("total_gpu_seconds") or 0.0)
    except (TypeError, ValueError):
        print("[budget] total_gpu_seconds is not a number, treating as 0", flush=True)
        total = 0.0
    return {
        "runs": runs if isinstance(runs, list) else [],
        "total_gpu_seconds": total,
    }


def _budget_usage(now: Optional[float] = None) -> dict:
    """GPU-hours in the trailing hour, the trailing day, and in total."""
    moment = time.time() if now is None else now
    state = _budget_state()
    recent = [
        (float(entry[0]), float(entry[1]))
        for entry in state["runs"]
        if isinstance(entry, (list, tuple)) and len(entry) >= 2
    ]
    hour = sum(sec for stamp, sec in recent if moment - stamp <= _SECONDS_PER_HOUR)
    day = sum(sec for stamp, sec in recent if moment - stamp <= _SECONDS_PER_DAY)
    return {
        "last_hour_gpu_hours": hour / _SECONDS_PER_HOUR,
        "last_day_gpu_hours": day / _SECONDS_PER_HOUR,
        "total_gpu_hours": state["total_gpu_seconds"] / _SECONDS_PER_HOUR,
        "n_runs_recorded": len(recent),
    }


def _budget_usd_last_day(now: Optional[float] = None) -> float:
    """Trailing-day spend, priced per run from the GPU each one recorded."""
    moment = time.time() if now is None else now
    total = 0.0
    for entry in _budget_state()["runs"]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            stamp, seconds = float(entry[0]), float(entry[1])
        except (TypeError, ValueError):
            continue
        if moment - stamp > _SECONDS_PER_DAY:
            continue
        name = str(entry[3]) if len(entry) >= 4 else GPU
        total += seconds * _gpu_rate(name)
    return total


def _budget_record(gpu_seconds: float, run_id: str, now: Optional[float] = None) -> None:
    """Add a run's GPU-seconds to the ledger.

    Entries past 24 hours are pruned, since nothing reads further back than
    that -- but ``total_gpu_seconds`` is accumulated separately so pruning
    never forgets the cumulative figure.
    """
    if not BUDGET_ENABLED or gpu_seconds <= 0:
        return
    moment = time.time() if now is None else now
    try:
        state = _budget_state()
        kept = [
            entry
            for entry in state["runs"]
            if isinstance(entry, (list, tuple))
            and len(entry) >= 2
            and moment - float(entry[0]) <= _SECONDS_PER_DAY
        ]
        # The GPU name is the 4th element. Readers tolerate entries shorter
        # than this (`len(entry) >= 2`), so entries written before the field
        # existed still count toward every ceiling -- they just get priced at
        # the configured default's rate instead of their own.
        kept.append([moment, float(gpu_seconds), str(run_id), GPU])
        budget_dict[_BUDGET_KEY] = {
            # Bounded so a flood of tiny runs cannot grow the entry without
            # limit; the trailing-hour and trailing-day sums are what matter
            # and 5000 entries covers far more than a day of real traffic.
            "runs": kept[-5000:],
            "total_gpu_seconds": state["total_gpu_seconds"] + float(gpu_seconds),
        }
    except Exception as exc:  # noqa: BLE001 -- recording must not fail a run
        print(f"[budget] could not record {gpu_seconds:.1f} GPU-s: {exc}", flush=True)


def _budget_gate(run_id: str) -> bool:
    """Check the breaker before doing GPU work.

    Returns whether **fan-out** is still permitted. Raises
    :class:`BudgetExceeded` when a refusing threshold is crossed.
    """
    if not BUDGET_ENABLED:
        return True

    usage = _budget_usage()
    shared = (
        f"total {usage['total_gpu_hours']:.2f} GPU-h, "
        f"last 24h {usage['last_day_gpu_hours']:.2f} GPU-h, "
        f"last hour {usage['last_hour_gpu_hours']:.2f} GPU-h "
        f"over {usage['n_runs_recorded']} recorded run(s)"
    )

    if usage["total_gpu_hours"] >= BUDGET_TOTAL_GPU_HOURS:
        raise BudgetExceeded(
            f"cumulative GPU budget reached: {shared}. The ceiling is "
            f"{BUDGET_TOTAL_GPU_HOURS:g} GPU-h "
            "(LAMP_DESIGN_TOTAL_GPU_HOURS). This one does not reset on its "
            "own -- raise the ceiling deliberately, or clear the ledger with "
            "`modal dict clear lamp-design-budget`."
        )

    if usage["last_day_gpu_hours"] >= BUDGET_DAILY_GPU_HOURS:
        raise BudgetExceeded(
            f"24-hour GPU budget reached: {shared}. The ceiling is "
            f"{BUDGET_DAILY_GPU_HOURS:g} GPU-h (LAMP_DESIGN_DAILY_GPU_HOURS). "
            "It is a trailing window, so capacity returns gradually as the "
            "oldest runs age out rather than at a fixed reset time."
        )

    if usage["last_hour_gpu_hours"] >= BUDGET_HOURLY_GPU_HOURS:
        print(
            f"[budget] run {run_id}: fan-out disabled, {shared}; hourly "
            f"ceiling is {BUDGET_HOURLY_GPU_HOURS:g} GPU-h. Running on a "
            "single container instead -- slower, not refused.",
            flush=True,
        )
        return False
    return True


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
    geometry_plan = _engine("geometry_plan")
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

    # One knob -- "layouts per position" -- resolved in the engine, so the
    # Modal side cannot drift from the CLI on what n_geometries means.
    plan = geometry_plan(req["n_geometries"])
    # Recorded before generation so an empty result can say whether the
    # template was simply too short or the GC/ACGT screen emptied it.
    slots = count_candidate_slots(
        len(panel.consensus),
        geometries=plan["geometries"],
        n_sampled=plan["n_sampled"],
        stride=req["stride"],
    )
    primer_sets = generate_candidates(
        panel.consensus,
        **plan,
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


def _record_params(
    result: Any,
    req: dict,
    *,
    n_candidates: int,
    n_pairs: int,
    n_predictions: int,
    timings: dict,
) -> Any:
    """Populate ``DesignResult.params``, the way ``commands.design.design`` does.

    `build_design_result` does NOT fill this in -- the CLI's `design()` updates
    `result.params` itself after calling it. Both Modal paths call
    `build_design_result` directly, so without this a remote result carried an
    empty `params` and could not say what produced it: the front end's
    `run_facts` showed device, conditions, geometries and every count as blank,
    and `result.json` from a Modal run was not comparable to one from the CLI.
    """
    plan = _engine("geometry_plan")(req["n_geometries"])
    result.params.update(
        {
            "stride": req["stride"],
            "geometries": [
                getattr(g, "geometry_id", None) or str(g)
                for g in plan["geometries"]
            ]
            + [f"sampled-{i}" for i in range(int(plan["n_sampled"]))],
            "conditions": [str(c) for c in _conditions_from_request(req)],
            "ranking_condition": str(_ranking_condition_from_request(req)),
            "tp_threshold_minutes": req["tp_threshold_minutes"],
            "n_requested": req["n_sets"],
            "min_separation_bp": req["min_separation"],
            "device": _device(),
            "n_candidates": n_candidates,
            "n_pairs": n_pairs,
            "n_predictions": n_predictions,
            "timings_s": {k: round(float(v), 3) for k, v in timings.items()},
            # Only the remote paths set this, so an output states where it ran.
            "runtime": "modal",
            "gpu": GPU,
        }
    )
    return result


def _conditions_from_request(req: dict) -> list:
    """Every reaction condition a request is scored under, in request order.

    One place, because the orchestrator, the shards and the single-container
    path all have to agree: a shard that predicted under different conditions
    than the ones the result is labelled with would be undetectable in the
    output. `_validate_request` guarantees `target_concs` is a non-empty,
    deduped list, so this needs no defaulting of its own.
    """
    from lamp_design.types import Condition

    mix = req["master_mix"]
    return [Condition(master_mix=mix, target_conc=conc) for conc in req["target_concs"]]


def _ranking_condition_from_request(req: dict) -> Any:
    """The one condition the ranking is computed under.

    `_validate_request` guarantees it is a member of `target_concs`, so the
    ranking never refers to a prediction that was not made.
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
    timings: Optional[dict] = None,
) -> Any:
    """Assembled features -> DesignResult."""
    conditions = _conditions_from_request(req)
    condition = _ranking_condition_from_request(req)
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
        conditions,
        weights=weights,
        device=_device(),
    )

    result = _engine("build_design_result")(
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
    return _record_params(
        result,
        req,
        n_candidates=len(primer_sets),
        n_pairs=len(thermo.set_ids),
        n_predictions=len(predictions),
        timings=timings or {},
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


@app.function(timeout=120)
def budget_report() -> dict:
    """The spend ledger, on a CPU container.

    Separate from `doctor` so that checking spend costs nothing and needs no
    volume mount -- which matters because this is the thing you want to read
    when you suspect something is running away.
    """
    usage = _budget_usage()
    return {
        **usage,
        "enabled": BUDGET_ENABLED,
        "ceilings_gpu_hours": {
            "hourly_fanout_off": BUDGET_HOURLY_GPU_HOURS,
            "daily_refuse": BUDGET_DAILY_GPU_HOURS,
            "total_refuse": BUDGET_TOTAL_GPU_HOURS,
        },
        "remaining_gpu_hours": {
            "to_daily": max(0.0, BUDGET_DAILY_GPU_HOURS - usage["last_day_gpu_hours"]),
            "to_total": max(0.0, BUDGET_TOTAL_GPU_HOURS - usage["total_gpu_hours"]),
        },
        # Indicative only -- the authoritative figure is the Modal dashboard,
        # which also sees spend this ledger cannot (see "Spend circuit
        # breaker"). Two figures because only one of them can be exact:
        # trailing-day entries carry the GPU they ran on, so that sum is
        # priced per run; the cumulative total is a bare GPU-second scalar
        # that survives pruning, so it can only be priced at one rate.
        "approx_usd_last_day": round(_budget_usd_last_day(), 4),
        "approx_usd_total_at_default_rate": round(
            usage["total_gpu_hours"] * _gpu_rate(GPU) * _SECONDS_PER_HOUR, 4
        ),
        "default_gpu": GPU,
    }


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
        # The spend ledger, so `doctor` answers "how much have we used?"
        # without starting a GPU. Cheap: one Dict read.
        "budget": {
            **_budget_usage(),
            "enabled": BUDGET_ENABLED,
            "ceilings_gpu_hours": {
                "hourly_fanout_off": BUDGET_HOURLY_GPU_HOURS,
                "daily_refuse": BUDGET_DAILY_GPU_HOURS,
                "total_refuse": BUDGET_TOTAL_GPU_HOURS,
            },
        },
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
# Orchestrator for the fan-out path (CPU only; it never touches a GPU).
# ---------------------------------------------------------------------------


#: Timing keys that are NOT wall-clock stages, so they must never be summed
#: into ``total``. ``gpu_seconds`` is the sum across every shard -- 10 shards
#: of 7s each is 70 "seconds" of a run that took 33 -- so including it made
#: `design_fanout` report total=96.1 for a run measured at 33.1s wall. It did
#: not show up in `design_single_container` only because there `total` happens
#: to be computed before `gpu_seconds` is inserted, which is luck, not design.
_NON_STAGE_TIMINGS = frozenset({"gpu_seconds", "total"})


def _stage_total(timing: dict) -> float:
    """Wall-clock sum of the pipeline stages, excluding accounting entries."""
    return sum(
        float(v)
        for k, v in timing.items()
        if k not in _NON_STAGE_TIMINGS and isinstance(v, (int, float))
    )


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

    # Before any GPU work: the breaker may refuse the run outright, or allow
    # it while withdrawing fan-out. Prepare is CPU-cheap but there is no point
    # doing it for a run that is about to be refused.
    fanout_allowed = _budget_gate(run_id)

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

    # One shard holding everything when the breaker withdrew fan-out: the same
    # total GPU-seconds on one container instead of many, so the job completes
    # and only its wall clock suffers.
    shard_size = len(primer_sets) if not fanout_allowed else INFER_SHARD_SETS
    shards = [
        {"run_id": run_id, "sets": [_set_to_json(s) for s in shard], "req": req}
        for shard in _chunked(primer_sets, max(1, shard_size))
    ]
    print(
        f"run={run_id} targets={panel.n_targets} budget={panel.budget} "
        f"sets={len(primer_sets)} shards={len(shards)} "
        f"max_containers={INFER_MAX_CONTAINERS} gpu={GPU} "
        f"fanout={'on' if fanout_allowed else 'OFF (budget)'}",
        flush=True,
    )

    started = time.time()
    predictions: list = []
    mismatches: dict = {}
    n_pairs = 0
    gpu_seconds = 0.0
    for shard_result in design_shard.map(shards):
        predictions.extend(_predictions_from_json(shard_result["predictions"]))
        mismatches.update(
            {(s, t): m for s, t, m in shard_result["mismatches"]}
        )
        n_pairs += shard_result["n_pairs"]
        gpu_seconds += float(shard_result.get("gpu_seconds") or 0.0)
    timing["shards"] = time.time() - started
    # Recorded as soon as the GPU work is done, not at the end: ranking runs
    # on a CPU container, and a failure there must not lose the accounting for
    # GPU time already spent.
    _budget_record(gpu_seconds, run_id)
    timing["gpu_seconds"] = gpu_seconds
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
        ranking_condition=_ranking_condition_from_request(req),
        tp_threshold=req["tp_threshold_minutes"],
        n=req["n_sets"],
        min_separation=req["min_separation"],
        mismatches=mismatches,
        funnel=funnel,
        excluded_targets=dict(prepared.excluded),
        warnings=list(prepared.warnings),
    )
    _record_params(
        result,
        req,
        n_candidates=len(primer_sets),
        n_pairs=n_pairs,
        n_predictions=len(predictions),
        timings=dict(timing),
    )
    result.params["shards"] = len(shards)
    result.params["fanout"] = fanout_allowed
    timing["rank"] = time.time() - started
    timing["total"] = _stage_total(timing)
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

    shard_started = time.time()
    results_volume.reload()
    run_dir = _run_dir(job["run_id"])
    panel = _panel_from_json(json.loads((run_dir / "panel.json").read_text()))
    primer_sets = [_set_from_json(blob) for blob in job["sets"]]
    req = job["req"]

    thermo = _compute_chunk(primer_sets, panel)
    predictions = _engine("predict_tp")(
        thermo,
        primer_sets,
        _conditions_from_request(req),
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
        # This container held a GPU for its whole lifetime, which only it can
        # measure -- the orchestrator sees wall clock across all shards at
        # once, not the GPU-seconds each one consumed.
        "gpu_seconds": time.time() - shard_started,
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

    # This container has held a GPU since it started, so the clock begins
    # here, not at the inference stage. The breaker's return value is ignored
    # on purpose: there is no fan-out to withdraw on this path, so only its
    # refusing thresholds (daily, total) apply.
    container_started = time.time()
    _budget_gate(req["run_id"])

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
        timings=dict(timing),
    )
    timing["infer"] = time.time() - started
    timing["total"] = _stage_total(timing)

    # The whole container lifetime counts, including prepare and thermo: the
    # GPU was allocated and billed throughout, idle or not. Recording the
    # inference stage alone would under-count by ~11% and make the ledger
    # cheerfully wrong in the one direction that matters.
    gpu_seconds = time.time() - container_started
    _budget_record(gpu_seconds, req["run_id"])
    timing["gpu_seconds"] = gpu_seconds

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
# An HTTP front door IS now built, at the end of this file, behind
# LAMP_DESIGN_HTTP=1. It exists for one caller who is not Python-with-a-Modal-
# token: a collaborator deploying the portal in their own infrastructure. See
# the "HTTP front door" section and docs/HANDOFF.md.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    mode: str = "doctor",
    fasta: str = "",
    master_mix: str = "SS",
    # Comma-separated, e.g. "10000,100". Empty means the engine's own default
    # sweep (commands.design.DEFAULT_CONCENTRATIONS), so `modal run` and
    # `lamp-design design` agree out of the box.
    target_concs: str = "",
    rank_conc: float = 0.0,
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

    if mode == "budget":
        # CPU only, so checking the ledger never costs GPU time.
        print(json.dumps(budget_report.remote(), indent=2), flush=True)
        return

    if not fasta:
        raise SystemExit("--fasta PATH is required for modes sync/fanout/spawn")
    path = Path(fasta)
    n_bytes = path.stat().st_size
    if n_bytes > MAX_FASTA_BYTES:
        # Fail before uploading anything.
        raise SystemExit(f"{path} is {n_bytes} bytes, over the {MAX_FASTA_BYTES}-byte cap")

    request: dict = {
        "fasta": path.read_text(),
        "master_mix": master_mix,
        "n_sets": n_sets,
        "stride": stride,
        "n_geometries": n_geometries,
        "pre_aligned": pre_aligned,
    }
    if target_concs.strip():
        request["target_concs"] = [float(c) for c in target_concs.split(",") if c.strip()]
    if rank_conc:
        request["rank_conc"] = rank_conc

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
        raise SystemExit(
            f"unknown --mode {mode!r}; expected doctor, budget, sync, fanout or spawn"
        )

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


# ---------------------------------------------------------------------------
# HTTP front door
#
# Exists so a collaborator can run the portal without a Modal token. A Modal
# token is workspace-scoped: handing one to a third party hands over a key to
# the whole account -- every app, every Volume, every Secret, and the ability
# to spend. An API key checked here is scoped to this one function, is
# rotatable without touching their deployment, and keeps the weights inside
# this workspace.
#
# Their container then needs only Streamlit and `requests`, not the Modal SDK,
# torch, ViennaRNA or the checkpoints. See docs/HANDOFF.md, shape A.
#
# SETUP (once):
#   modal secret create lamp-design-api --from-literal LAMP_DESIGN_API_KEYS="$(openssl rand -hex 32)"
#   LAMP_DESIGN_HTTP=1 modal deploy deploy/modal/app.py
#
# Rotation is additive: put both keys in the secret, update callers, drop the
# old one. The variable is a comma-separated list for exactly that reason.
# ---------------------------------------------------------------------------

#: Opt-in, because an unauthenticated public URL for a GPU function is the one
#: mistake here that costs real money. Deploying without this simply does not
#: create the endpoint.
HTTP_ENABLED = os.environ.get("LAMP_DESIGN_HTTP") == "1"

#: Secret holding LAMP_DESIGN_API_KEYS. Separate from the weights secret so the
#: two rotate independently.
API_SECRET_NAME = os.environ.get("LAMP_DESIGN_API_SECRET", "lamp-design-api")

HTTP_SECRETS: list = (
    [modal.Secret.from_name(API_SECRET_NAME)] + SECRETS if HTTP_ENABLED else []
)


def _authorized(request_key: Optional[str]) -> bool:
    """Constant-time check of a presented key against the configured set.

    Fails CLOSED: no configured keys means no access. That is the opposite of
    the budget ledger's fail-open, and deliberately so -- a broken ledger
    should not stop legitimate work, but a missing key list must never open
    the door.
    """
    import hmac

    configured = [
        key.strip()
        for key in os.environ.get("LAMP_DESIGN_API_KEYS", "").split(",")
        if key.strip()
    ]
    if not configured or not request_key:
        return False
    # compare_digest on every candidate, and `any` over the results rather
    # than short-circuiting, so the time taken does not reveal which key
    # matched or how many are configured.
    return any([hmac.compare_digest(key, request_key) for key in configured])


def _bearer(headers: Any) -> Optional[str]:
    """The presented key, from `Authorization: Bearer ...` or `X-API-Key`."""
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or headers.get("X-API-Key")


if HTTP_ENABLED:

    @app.function(timeout=120, secrets=HTTP_SECRETS)
    @modal.fastapi_endpoint(method="POST", label="lamp-design-submit")
    def http_submit(request: dict, authorization: str = "", x_api_key: str = "") -> dict:
        """Submit a design run. Returns a call id immediately; does not block.

        Non-blocking on purpose: a full sweep is ~90 GPU-seconds, which is
        longer than most HTTP clients and proxies will wait, and far longer
        than Cloud Run's per-request budget allows a caller to hold.
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse  # noqa: F401  (documented shape)

        presented = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else (x_api_key or None)
        )
        if not _authorized(presented):
            # 401 with no detail: a precise message here is a probing oracle.
            raise HTTPException(status_code=401, detail="unauthorized")

        try:
            # Validate before spawning, so a bad request costs no GPU time and
            # the caller gets the reason rather than a failed job id.
            _validate_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        call = design_single_container.spawn(request)
        return {
            "schema": "lamp-design/http-submit@1",
            "call_id": call.object_id,
            "run_id": request.get("run_id"),
        }

    @app.function(timeout=120, secrets=HTTP_SECRETS)
    @modal.fastapi_endpoint(method="GET", label="lamp-design-status")
    def http_status(
        call_id: str = "", authorization: str = "", x_api_key: str = ""
    ) -> dict:
        """Poll a submitted run.

        ``state`` is one of queued / running / succeeded / failed, matching the
        vocabulary deploy/streamlit/_backend.py already polls on, so the
        Streamlit client speaks to this endpoint and to the Modal SDK path with
        the same code.
        """
        from fastapi import HTTPException

        presented = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else (x_api_key or None)
        )
        if not _authorized(presented):
            raise HTTPException(status_code=401, detail="unauthorized")
        if not call_id:
            raise HTTPException(status_code=400, detail="call_id is required")

        handle = modal.FunctionCall.from_id(call_id)
        try:
            # timeout=0 polls without waiting.
            envelope = handle.get(timeout=0)
        except TimeoutError:
            return {"schema": "lamp-design/http-status@1", "state": "running"}
        except BudgetExceeded as exc:
            # 429, not 500: the request was fine, the budget is spent. This is
            # why BudgetExceeded is deliberately not a ValueError.
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 -- surface the failure as data
            return {
                "schema": "lamp-design/http-status@1",
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "schema": "lamp-design/http-status@1",
            "state": "succeeded",
            "envelope": envelope,
        }
