"""The one backend seam: a design or evaluate run, wherever it executes.

Two implementations sit behind it and both return the **same envelope**, so
nothing in the UI branches on where a run happened:

``local``
    A subprocess on this host's GPU (``_runner.py``). Out of process rather
    than in process for three reasons that all bite in a long-lived Streamlit
    server: ``thermo.compute_dg`` forks a multiprocessing pool, and forking
    after CUDA has been initialised by an earlier run in the same process is
    the classic hang; a segfault in a native dependency would take the whole
    app down with it; and Stop cannot interrupt a Python thread but can signal
    a process group.
``modal``
    ``deploy/modal/app.py``'s ``design_single_container`` / ``design_fanout``,
    submitted with ``.spawn()`` and polled with ``.get(timeout=0)``.

Both are submit-and-poll. That is not a preference: ARCHITECTURE.md measures
the full sweep at 461 s on one L4, so no synchronous request survives a real
run, and the UI must be able to render a half-finished one.

The envelope is the contract, documented in ``deploy/README.md``::

    {schema, run_id, request, timing_seconds, result: report.result_to_dict(...)}

Reusing the engine's own ``result_to_dict`` means there is no second copy of the
result schema to drift.

Isolation: every run gets its own directory and this module NEVER calls
``os.chdir()``. ``/home/ssiddiqu/PrimerCast/gui/run_isolation.py`` documents the
bug class -- one process serving many sessions, two runs sharing a working
directory and clobbering each other's control files. Here the rule is stricter,
because ``os.chdir`` is process-global and would retarget every *other*
in-flight session's cwd, not merely collide on one directory. The engine
cooperates: ``external/mafft.py`` and ``report.write_outputs`` take explicit
paths and use ``tempfile`` for scratch, so no stage depends on the cwd.

No streamlit import: the runner subprocess and the tests both load this module.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from _engine_api import (
    MAX_BUDGET,
    MAX_CANDIDATE_SETS,
    engine,
    engine_default,
    has_engine,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent

#: Root for persisted run outputs. Local disk, which is **ephemeral on Cloud
#: Run** -- lost on redeploy and on scale events, and invisible to a second
#: instance. This variable is the seam for a mounted volume or a GCS-backed
#: path later, the same role ``PRIMERCAST_DATA_DIR`` plays in PrimerCast, where
#: it likewise defaults to the project root and runs land in ``<root>/runs``.
#: (An earlier revision defaulted it to ``<repo>/runs``, which put every run in
#: ``<repo>/runs/runs/<id>``.)
DATA_DIR = Path(os.environ.get("LAMP_DESIGN_DATA_DIR") or REPO_ROOT)
RUNS_DIR = DATA_DIR / "runs"

ENVELOPE_SCHEMA = "lamp-design/modal-result@1"
STATUS_SCHEMA = "lamp-design/run-status@1"

#: Must match the deployed Modal app. See ../modal/app.py.
MODAL_APP_NAME = os.environ.get("LAMP_DESIGN_MODAL_APP", "lamp-design")
MODAL_FN_SINGLE = "design_single_container"
MODAL_FN_FANOUT = "design_fanout"

BACKENDS = ("local", "modal-single", "modal-fanout")

#: Stage order for the Run page's checklist. `design()` calls back after
#: generation, thermo, inference and report; prepare happens in the runner
#: before it, so the UI can show panel QC against the same list.
DESIGN_STAGES: tuple[tuple[str, str], ...] = (
    ("prepare", "Preparing the target panel"),
    ("generation", "Generating candidate sets"),
    ("thermo", "Thermodynamic features"),
    ("inference", "Scoring with the Lyra ensemble"),
    ("report", "Ranking and diversity selection"),
)

#: Evaluate skips generation. Only ``prepare`` is this app's own; the rest are
#: the stages ``commands/evaluate.py``'s ``run`` prints, picked up out of the
#: run log by `stages_from_log` -- that CLI entry point owns its own progress
#: callback, and reimplementing its orchestration to get one of ours would be
#: the duplication `commands/design.py` warns against.
EVALUATE_STAGES: tuple[tuple[str, str], ...] = (
    ("prepare", "Reading the primer sets and target"),
    ("thermo", "Thermodynamic features"),
    ("inference", "Scoring with the Lyra ensemble"),
    ("report", "Predictions and agreement metrics"),
)

#: ``cli`` stage lines look like ``"  thermo     1,234 unique pairs"``: two
#: spaces, a stage name, whitespace, the message.
_LOG_STAGE = re.compile(r"^ {2}(\w+)\s{2,}(\S.*)$")

ProgressFn = Callable[[str, str], None]


class RunCancelled(RuntimeError):
    """Raised out of the progress callback when a Stop has been requested."""


def active_backend() -> str:
    """The backend this process runs with, from ``LAMP_DESIGN_BACKEND``.

    An environment variable rather than a widget: which compute a deployment
    may reach is an operator's decision, not a visitor's. The sidebar shows
    what is active. ``modal`` is accepted as a synonym for the V1 single-
    container path so the variable reads the way the deployment talks about it.
    """
    raw = (os.environ.get("LAMP_DESIGN_BACKEND") or "local").strip().lower()
    if raw in ("modal", "modal-single"):
        return "modal-single"
    if raw == "modal-fanout":
        return "modal-fanout"
    return "local"


def modal_function_name(backend: str) -> str:
    return MODAL_FN_FANOUT if backend == "modal-fanout" else MODAL_FN_SINGLE


# ---------------------------------------------------------------------------
# Per-run directories
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    """Where one run reads and writes."""

    run_id: str
    #: Persisted outputs. Survives the run; ephemeral on Cloud Run.
    out_dir: Path
    #: Private scratch for this run. Deleted afterwards.
    work_dir: Path


def new_run_id() -> str:
    """Timestamp for humans, uuid4 for unguessability. Never user-supplied."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def out_dir_for(run_id: str) -> Path:
    """This run's persisted output directory."""
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Explicit chmod: mkdir(mode=...) is masked by umask, and on a shared host
    # another local user should not be able to read a session's sequences.
    out_dir.chmod(0o700)
    return out_dir


def prepare_run(run_id: str) -> RunPaths:
    """Create this run's private output and scratch directories.

    Mirrors the split in PrimerCast's ``prepare_run_dir``: private scratch for
    anything a run writes while working, a separate persisted location for its
    outputs. No shared cwd and no symlinking are needed, because every engine
    stage takes explicit paths.
    """
    return RunPaths(
        run_id=run_id,
        out_dir=out_dir_for(run_id),
        work_dir=Path(tempfile.mkdtemp(prefix=f"lamp_design_{run_id}_")),
    )


def envelope_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "envelope.json"


def status_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "status.json"


def log_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "run.log"


def config_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "run_config.json"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def build_conditions(master_mixes: Iterable[str], concentrations: Iterable[float]) -> list[dict]:
    """Mixes-major cross product, matching ``design.build_conditions``' order.

    Kept as plain dicts because a request has to survive JSON on the way to a
    subprocess or to Modal; the engine's ``Condition`` is rebuilt at the far end.
    """
    return [
        {"master_mix": mix, "target_conc": float(conc)}
        for mix in master_mixes
        for conc in concentrations
    ]


def build_design_request(
    *,
    run_id: str,
    fasta: str,
    master_mixes: Iterable[str],
    concentrations: Iterable[float],
    rank_master_mix: str,
    rank_conc: float,
    n_sets: int,
    stride: int,
    n_geometries: int,
    tp_threshold_minutes: float,
    min_separation: int,
    pre_aligned: bool,
    device: str,
    threads: Optional[int],
) -> dict:
    """A design request, in the shape both backends accept.

    Conditions are sent three ways on purpose, because the two backends read
    different keys and silence is the dangerous failure here:

    * ``conditions`` -- the full (mix x concentration) sweep, which the local
      backend consumes directly.
    * ``target_concs`` + ``rank_conc`` -- the concentration sweep and which of
      them ranks, which ``deploy/modal/app.py``'s ``_validate_request`` reads.
    * ``master_mix`` / ``target_conc`` -- the ranking condition on its own, as
      the one-condition shorthand older Modal revisions understood.

    The remaining asymmetry is **master mix**: ``_validate_request`` takes a
    single ``master_mix``, so a multi-*mix* sweep still scores only the ranking
    mix remotely. ``warn_backend_limits`` says so on the Parameters page rather
    than letting the result quietly contain fewer conditions than were asked
    for. A multi-*concentration* sweep does now work remotely, which is the
    default (1 mix x 2 concentrations).
    """
    return {
        "run_id": run_id,
        "workflow": "design",
        "fasta": fasta,
        "conditions": build_conditions(master_mixes, concentrations),
        "master_mix": rank_master_mix,
        "target_conc": float(rank_conc),
        # Read by the Modal side, which scores every concentration and ranks
        # under `rank_conc`. Without these a remote run silently collapsed to
        # the single `target_conc` above.
        "target_concs": [float(c) for c in concentrations],
        "rank_conc": float(rank_conc),
        "n_sets": int(n_sets),
        "stride": int(stride),
        "n_geometries": int(n_geometries),
        "tp_threshold_minutes": float(tp_threshold_minutes),
        "min_separation": int(min_separation),
        "pre_aligned": bool(pre_aligned),
        "device": device,
        # None is the engine's documented "all cores", so it has to survive.
        "threads": None if threads is None else int(threads),
    }


def build_evaluate_request(
    *,
    run_id: str,
    fasta: str,
    primer_sets_text: str,
    primer_sets_filename: str,
    master_mixes: Iterable[str],
    concentrations: Iterable[float],
    rank_master_mix: str,
    rank_conc: float,
    tp_threshold_minutes: float,
    pre_aligned: bool,
    device: str,
    threads: Optional[int],
    use_input_conditions: bool = False,
    max_window_mismatches: int = 6,
    censor_at: float = 60.0,
    truth_column: str = "",
    windows: str = "auto",
) -> dict:
    """An evaluate request: the user's own primer sets, scored against a panel.

    ``pre_aligned`` is carried for symmetry and for the run config; evaluate
    locates each primer's window in each target directly and never builds a
    consensus, so it has no alignment step to switch.
    """
    return {
        "run_id": run_id,
        "workflow": "evaluate",
        "fasta": fasta,
        "primer_sets_text": primer_sets_text,
        "primer_sets_filename": primer_sets_filename,
        "conditions": build_conditions(master_mixes, concentrations),
        "master_mix": rank_master_mix,
        "target_conc": float(rank_conc),
        "tp_threshold_minutes": float(tp_threshold_minutes),
        "pre_aligned": bool(pre_aligned),
        "device": device,
        # None is the engine's documented "all cores", so it has to survive.
        "threads": None if threads is None else int(threads),
        "use_input_conditions": bool(use_input_conditions),
        "max_window_mismatches": int(max_window_mismatches),
        "censor_at": float(censor_at),
        "truth_column": truth_column,
        "windows": windows,
    }


def public_request(request: dict) -> dict:
    """The request without the user's sequences, for the envelope and the log."""
    return {k: v for k, v in request.items() if k not in ("fasta", "primer_sets_text")}


def save_run_config(request: dict, *, backend: str, status: str) -> Path:
    """Persist run metadata next to the outputs so a run can be revisited.

    Written when a run starts and again when it ends, so a failed run also has
    its settings on disk -- the role ``_save_run_config`` plays in PrimerCast.
    The sequences themselves are not copied here: they live in the run's own
    ``input.fasta`` and nowhere else.
    """
    path = config_path(request["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": request["run_id"],
                "workflow": request.get("workflow", "design"),
                "backend": backend,
                "status": status,
                "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request": public_request(request),
            },
            indent=2,
        )
    )
    return path


# ---------------------------------------------------------------------------
# Status, shared by the runner (writer) and the Run page (reader)
# ---------------------------------------------------------------------------


def write_status(run_id: str, status: dict) -> None:
    """Replace ``status.json`` atomically.

    The page polls this file while the runner writes it, so a half-written read
    is a real possibility; ``os.replace`` makes the swap atomic.
    """
    path = status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2))
    os.replace(temporary, path)


def read_status(run_id: str) -> dict | None:
    """The run's status, or None if it has not written one yet."""
    path = status_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def read_envelope(run_id: str) -> dict | None:
    """A finished run's envelope from disk."""
    path = envelope_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def stages_from_log(run_id: str, stage_keys: Iterable[str]) -> dict[str, str]:
    """Engine stage lines scraped out of the run log, ``{stage: message}``.

    PrimerCast drives its progress display by parsing Snakemake's stdout for
    ``rule`` lines; this is the same trick for the one entry point whose
    progress this app cannot receive directly. Only used to fill gaps the
    ``progress`` callback left, so a stage never appears twice.
    """
    wanted = set(stage_keys)
    found: dict[str, str] = {}
    path = log_path(run_id)
    if not path.exists():
        return found
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        match = _LOG_STAGE.match(line)
        if match and match.group(1) in wanted:
            found[match.group(1)] = match.group(2).strip()
    return found


def read_log(run_id: str, *, max_lines: int = 200) -> str:
    """The tail of the run log."""
    path = log_path(run_id)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# The local backend: the engine, in this process
#
# `run_local` and `evaluate_local` are called by `_runner.py` in a subprocess,
# and directly by tests and by anything that wants an envelope without a UI.
# ---------------------------------------------------------------------------


def _conditions(request: dict) -> tuple[list, Any]:
    """``(conditions, ranking_condition)`` as engine objects."""
    from lamp_design.types import Condition

    conditions = [
        Condition(master_mix=c["master_mix"], target_conc=float(c["target_conc"]))
        for c in request["conditions"]
    ]
    ranking = Condition(
        master_mix=request["master_mix"], target_conc=float(request["target_conc"])
    )
    if ranking not in conditions:
        # Ranking under a condition that was not scored has no meaning, and
        # `build_design_result` would have nothing to rank on.
        conditions.insert(0, ranking)
    return conditions, ranking


def _weights() -> str | None:
    return os.environ.get("LAMP_DESIGN_WEIGHTS") or None


def run_local(request: dict, paths: RunPaths, progress: ProgressFn | None = None) -> dict:
    """Run a design here and return the envelope. Blocks until it finishes.

    Prepare happens in this function so the panel's QC and the candidate-slot
    count are known before the expensive stages start; everything after it is
    one call to ``lamp_design.design``, the engine's orchestration entry point.
    Earlier revisions of this app re-implemented the stage sequence because no
    such entry point existed -- do not reintroduce that, or the app and the CLI
    drift in what a design run means.
    """
    say = progress or (lambda stage, message: None)
    conditions, ranking = _conditions(request)
    timing: dict[str, float] = {}

    started = time.time()
    input_fasta = paths.work_dir / "input.fasta"
    input_fasta.write_text(request["fasta"])
    records = engine("read_fasta")(input_fasta)
    prepared = engine("prepare_panel")(
        records,
        max_budget=MAX_BUDGET,
        pre_aligned=request["pre_aligned"],
        threads=int(os.environ.get("LAMP_DESIGN_ALIGN_THREADS", "2")),
        panel_name=request["run_id"],
    )
    panel = prepared.panel
    plan = engine("geometry_plan")(request["n_geometries"])
    slots = engine("count_candidate_slots")(
        len(panel.consensus),
        geometries=plan["geometries"],
        n_sampled=plan["n_sampled"],
        stride=request["stride"],
    )
    prepare_funnel = dict(prepared.diagnostics.as_funnel()) if prepared.diagnostics else {}
    timing["prepare_s"] = time.time() - started
    say(
        "prepare",
        f"{panel.n_targets} target(s), {len(panel.consensus):,} nt consensus, "
        f"{slots:,} candidate slots",
    )

    started = time.time()
    result = engine("design")(
        panel,
        weights=_weights(),
        conditions=conditions,
        ranking_condition=ranking,
        stride=request["stride"],
        geometries=plan["geometries"],
        n_sampled=plan["n_sampled"],
        max_candidates=MAX_CANDIDATE_SETS,
        n=request["n_sets"],
        min_separation=request["min_separation"],
        tp_threshold=request["tp_threshold_minutes"],
        device=request.get("device", "auto"),
        threads=request.get("threads") or None,
        progress=say,
    )
    timing["design_s"] = time.time() - started

    # prepare's funnel, exclusions and warnings are transferred onto the result
    # the way the CLI does it, then reordered so the table reads in pipeline
    # order: prepare's steps, the slot count, then the engine's own stages.
    prepared.apply_to(result)
    ordered = dict(prepare_funnel)
    ordered["candidate_slots"] = int(slots)
    ordered["candidates_generated"] = int(result.params.get("n_candidates", 0))
    ordered.update(result.funnel)
    result.funnel.clear()
    result.funnel.update(ordered)

    # The engine's own per-stage breakdown, so the UI shows where the time went
    # rather than one opaque number. Its keys are already suffixed `_s` and are
    # a subset of `design_s`, so they are reported but not re-summed.
    timing.update(result.params.get("timings_s", {}))
    timing["total"] = timing["prepare_s"] + timing["design_s"]

    payload = engine("result_to_dict")(
        result, ranking_condition=ranking, tp_threshold=request["tp_threshold_minutes"]
    )
    # The engine's own writer: .sets.csv / .per_target.csv / .json /
    # .summary.txt, identical to a CLI run's artifacts.
    engine("write_outputs")(
        result,
        paths.out_dir / "result",
        ranking_condition=ranking,
        tp_threshold=request["tp_threshold_minutes"],
    )
    return _envelope(request, payload, timing)


#: Payload schema for an evaluate run. Deliberately NOT
#: ``lamp-design/result@1``: ``EvaluationResult`` is per-row predictions plus
#: agreement metrics, not ranked sets with a panel, and dressing it up as the
#: design payload would make the Results page lie about what it is showing.
EVALUATE_PAYLOAD_SCHEMA = "lamp-design/evaluate-result@1"


def evaluate_local(request: dict, paths: RunPaths, progress: ProgressFn | None = None) -> dict:
    """Score user-supplied primer sets against a target FASTA. Returns the envelope.

    One call into the engine's own CLI entry point, built against the surface in
    ``commands/evaluate.py``: ``--in`` is a CSV with one column per primer,
    ``--targets`` is the FASTA its windows are located in, ``--out`` is a path
    prefix, and ``run(args)`` does the orchestration. Going through ``run``
    rather than ``evaluate()`` is deliberate -- row loading, window location and
    output writing are sequenced there once, and a second copy here would drift
    from the CLI exactly as ``commands/design.py`` warns.

    It is resolved by dotted name like everything else, so a build without that
    module fails with the name it wanted instead of an ImportError at app start.
    """
    say = progress or (lambda stage, message: None)
    run = engine("evaluate_run")

    targets_fasta = paths.work_dir / "targets.fasta"
    targets_fasta.write_text(request["fasta"])
    primers_in = paths.work_dir / (request.get("primer_sets_filename") or "primer_sets.csv")
    primers_in.write_text(request["primer_sets_text"])
    out_prefix = paths.out_dir / "result"

    # A single master mix / concentration overrides each row's own condition; two
    # or more make a cross product. Passing neither keeps whatever the input
    # file said, which is what `use_input_conditions` asks for.
    if request.get("use_input_conditions"):
        master_mixes: list[str] | None = None
        concentrations: list[float] | None = None
    else:
        master_mixes = sorted({c["master_mix"] for c in request["conditions"]})
        concentrations = sorted({float(c["target_conc"]) for c in request["conditions"]})
    say(
        "prepare",
        f"{primers_in.name} against {targets_fasta.name}, "
        + ("conditions from the input file" if master_mixes is None
           else f"{len(master_mixes) * len(concentrations)} condition(s)"),
    )

    import argparse

    args = argparse.Namespace(
        input=str(primers_in),
        targets=str(targets_fasta),
        output=str(out_prefix),
        weights=_weights(),
        windows=request.get("windows", "auto"),
        max_window_mismatches=int(request["max_window_mismatches"]),
        master_mixes=master_mixes,
        concentrations=concentrations,
        truth_column=request.get("truth_column") or None,
        censor_at=float(request["censor_at"]),
        tp_threshold=float(request["tp_threshold_minutes"]),
        device=request.get("device", "auto"),
        batch_size=None,
        pair_chunk=int(engine_default("lamp_design.types", "PAIR_CHUNK", 20_000)),
        threads=request.get("threads") or None,
    )
    started = time.time()
    result = run(args)
    timing = {"evaluate_s": time.time() - started, "total": time.time() - started}

    payload = _evaluate_payload(result, out_prefix)
    say("report", f"{payload['n_scored']:,} row(s) scored")
    return _envelope(request, payload, timing)


def _evaluate_payload(result: Any, out_prefix: Path) -> dict:
    """``EvaluationResult`` -> JSON-ready payload, via the files it just wrote.

    The per-row table is read back from ``<prefix>.predictions.csv`` rather than
    reaching into ``EvaluationResult.rows``: that CSV and its columns are the
    documented ``--out`` surface, so the UI depends on the contract rather than
    on the dataclass's internals.
    """
    predictions: list[dict[str, Any]] = []
    csv_path = Path(f"{out_prefix}.predictions.csv")
    if csv_path.exists():
        import csv as csv_module

        with open(csv_path, newline="") as handle:
            predictions = [dict(row) for row in csv_module.DictReader(handle)]

    return {
        "schema": EVALUATE_PAYLOAD_SCHEMA,
        "kind": "evaluate",
        "n_scored": int(getattr(result, "n_scored", len(predictions))),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "skipped": dict(getattr(result, "skipped", {}) or {}),
        "warnings": list(getattr(result, "warnings", []) or []),
        "params": dict(getattr(result, "params", {}) or {}),
        "predictions": predictions,
    }


def _envelope(request: dict, payload: dict, timing: dict[str, float]) -> dict:
    return {
        "schema": ENVELOPE_SCHEMA,
        "run_id": request["run_id"],
        "request": public_request(request),
        "timing_seconds": {k: round(float(v), 3) for k, v in timing.items()},
        "result": payload,
    }


def evaluate_available() -> bool:
    """Whether this build has an ``evaluate`` entry point at all."""
    return has_engine("evaluate_run")


# ---------------------------------------------------------------------------
# Submit and poll
#
# A job is a plain dict so it survives in `st.session_state` across reruns and
# is readable in `status.json`. The authoritative record of a run is always the
# run directory, never the session: a session that reconnects can pick a run
# back up, and a session that never saw a run can never see it (the app renders
# only run ids held in its own session state).
# ---------------------------------------------------------------------------

#: Live local runs, keyed by run id. Only used to reap exit codes: a child that
#: has exited but not been waited on is a zombie, and `os.kill(pid, 0)` on a
#: zombie succeeds, so liveness cannot be judged by signal alone. Process-wide
#: rather than per-session because the subprocess is a child of the process, not
#: of the session; a run missing from here (the app restarted) still reports
#: from its status file.
_LOCAL_JOBS: dict[str, subprocess.Popen] = {}


def start_run(request: dict, *, backend: str) -> dict:
    """Submit a run and return the job handle. Does not block.

    ``request["run_id"]`` must already be set, by :func:`new_run_id` -- it is
    never user-supplied, and it names the directory the run writes into. This
    is checked rather than defaulted: minting one here would silently give a
    caller that forgot a second, unrelated run directory, which is worse than
    being told.
    """
    if not request.get("run_id"):
        raise ValueError(
            "request['run_id'] is required; mint one with _backend.new_run_id() "
            "before calling start_run (it names the run's output directory and "
            "is never taken from user input)"
        )
    if backend == "local":
        return _start_local(request)
    return _start_modal(request, backend)


def _start_local(request: dict) -> dict:
    """Spawn ``_runner.py`` on this host and return immediately."""
    paths = prepare_run(request["run_id"])
    request_path = paths.work_dir / "request.json"
    request_path.write_text(json.dumps(request))
    write_status(
        request["run_id"],
        {
            "schema": STATUS_SCHEMA,
            "run_id": request["run_id"],
            "workflow": request.get("workflow", "design"),
            "backend": "local",
            "state": "queued",
            "started_at": time.time(),
            "updated_at": time.time(),
            "stages": [],
            "request": public_request(request),
        },
    )

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    log = open(log_path(request["run_id"]), "w", buffering=1)
    process = subprocess.Popen(
        [sys.executable, str(APP_DIR / "_runner.py"), str(request_path), str(paths.work_dir)],
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=str(APP_DIR),
        env=environment,
        # Its own process group, so Stop can signal the engine's own children
        # (thermo's multiprocessing pool) and not just the runner.
        start_new_session=True,
    )
    _LOCAL_JOBS[request["run_id"]] = process
    save_run_config(request, backend="local", status="running")
    return {
        "run_id": request["run_id"],
        "backend": "local",
        "workflow": request.get("workflow", "design"),
        "pid": process.pid,
        "work_dir": str(paths.work_dir),
        "submitted_at": time.time(),
    }


def _start_modal(request: dict, backend: str) -> dict:
    """Submit to the deployed Modal app; the call id is the poll token."""
    import modal  # lazy: only the Modal backend needs this extra

    function_name = modal_function_name(backend)
    call = modal.Function.from_name(MODAL_APP_NAME, function_name).spawn(request)
    out_dir_for(request["run_id"])
    write_status(
        request["run_id"],
        {
            "schema": STATUS_SCHEMA,
            "run_id": request["run_id"],
            "workflow": request.get("workflow", "design"),
            "backend": backend,
            "state": "running",
            "started_at": time.time(),
            "updated_at": time.time(),
            "stages": [],
            "call_id": call.object_id,
            "request": public_request(request),
        },
    )
    save_run_config(request, backend=backend, status="running")
    return {
        "run_id": request["run_id"],
        "backend": backend,
        "workflow": request.get("workflow", "design"),
        "call_id": call.object_id,
        "function": function_name,
        "submitted_at": time.time(),
    }


def poll_run(job: dict) -> dict:
    """Current state of a submitted run: ``{state, status, envelope, error}``.

    ``state`` is one of queued / running / succeeded / failed / cancelled /
    unknown. The envelope is present only on success.
    """
    run_id = job["run_id"]
    if job.get("backend", "local") != "local":
        return _poll_modal(job)

    status = read_status(run_id) or {}
    state = status.get("state", "unknown")
    if state in ("succeeded", "failed", "cancelled"):
        _LOCAL_JOBS.pop(run_id, None)
        return {
            "state": state,
            "status": status,
            "envelope": read_envelope(run_id) if state == "succeeded" else None,
            "error": status.get("error"),
        }

    process = _LOCAL_JOBS.get(run_id)
    if process is not None and process.poll() is not None:
        # The runner writes a terminal status in a `finally`, so reaching here
        # means it died without getting the chance: killed, OOM, or a native
        # crash. Say that, rather than showing "running" for ever.
        _LOCAL_JOBS.pop(run_id, None)
        failure = (
            f"The run process exited with code {process.returncode} without writing a "
            "result. A signal, the OOM killer or a crash in a native dependency all "
            "look like this; the log below is what it managed to print."
        )
        status = {**status, "state": "failed", "error": failure}
        write_status(run_id, status)
        return {"state": "failed", "status": status, "envelope": None, "error": failure}

    return {"state": state or "running", "status": status, "envelope": None, "error": None}


def _poll_modal(job: dict) -> dict:
    """Poll a Modal call: the envelope, still-running, or the remote failure."""
    import modal

    run_id = job["run_id"]
    status = read_status(run_id) or {}
    try:
        envelope = modal.FunctionCall.from_id(job["call_id"]).get(timeout=0)
    except TimeoutError:
        # UNVERIFIED (no Modal account here): that a not-yet-finished poll
        # surfaces as a builtin TimeoutError. If polling reports a hard failure
        # for a run that is merely still going, this one except clause is the
        # thing to widen.
        return {"state": "running", "status": status, "envelope": None, "error": None}
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = {**status, "state": "failed", "error": failure}
        write_status(run_id, status)
        return {"state": "failed", "status": status, "envelope": None, "error": failure}

    # No scratch directory to clean up here: the remote container owned its own.
    out_dir_for(run_id)
    envelope_path(run_id).write_text(json.dumps(envelope, indent=2))
    status = {**status, "state": "succeeded", "finished_at": time.time()}
    write_status(run_id, status)
    return {"state": "succeeded", "status": status, "envelope": envelope, "error": None}


def stop_run(job: dict) -> str:
    """Ask a run to stop. Returns a message describing what was actually done.

    For a local run this signals the runner's whole process group, which is why
    it was started with ``start_new_session=True``: thermo's multiprocessing
    pool is a child of the runner, and signalling only the runner would leave
    those workers holding the cores.
    """
    run_id = job["run_id"]
    if job.get("backend", "local") != "local":
        try:
            import modal

            modal.FunctionCall.from_id(job["call_id"]).cancel()
        except Exception as exc:
            return f"Could not cancel the Modal call: {type(exc).__name__}: {exc}"
        status = {**(read_status(run_id) or {}), "state": "cancelled"}
        write_status(run_id, status)
        return "Cancelled the Modal call."

    pid = job.get("pid")
    if not pid:
        return "No local process recorded for this run."
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return "The run process was already gone."
    status = {**(read_status(run_id) or {}), "state": "cancelled", "error": "Stopped by the user."}
    write_status(run_id, status)
    _LOCAL_JOBS.pop(run_id, None)
    return "Stopped the run."
