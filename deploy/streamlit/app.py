"""Streamlit front end for the LAMP design engine.

STATUS: scaffolding. `streamlit` is an optional extra and is not installed in
this repo's venv, so **no page here has ever been rendered**. The engine
underneath is complete, though: every name in ``ENGINE_API`` resolves, and the
``local`` backend runs a design through ``lamp_design.design``. What is
unverified is this file's own Streamlit usage -- widgets, session state,
reruns -- not the pipeline it calls. Nothing here stubs or fakes a pipeline
stage. See the README in this directory for what is and is not verified.

Run it (once streamlit exists)::

    .venv/bin/python -m pip install -e '.[streamlit]'
    .venv/bin/streamlit run deploy/streamlit/app.py \
        --server.maxUploadSize 5 --server.port 8080

Concurrency: one Streamlit process serves many sessions, and on Cloud Run it is
run with request concurrency > 1. ``/home/ssiddiqu/PrimerCast/gui/run_isolation.py``
documents the bug class that follows -- two runs sharing one working directory
clobber each other's control files and locks. The fix there was a private
scratch cwd per run. Here the rule is stricter and simpler:

    every run gets its own directory, and this app NEVER calls ``os.chdir()``.

``os.chdir`` is process-global, so in a multi-session Streamlit process it is
strictly worse than the Snakemake case: it would retarget the cwd of every
*other* in-flight session, not just collide on one directory. There is no
scenario in this file where changing the process cwd is correct. The engine
helps here -- ``external/mafft.py`` and ``report.write_outputs`` take explicit
paths and use ``tempfile`` for scratch, so no stage depends on the cwd.

Isolation of user data (the team raised this explicitly):

* uploaded sequences are written only inside that run's own private directory,
  never into a shared input directory. PrimerCast writes uploads into a shared
  ``target_seqs/original`` that every session can then list -- deliberately not
  copied here.
* run ids embed a uuid4, so a run directory name is not guessable, and the app
  only ever renders runs recorded in this session's ``st.session_state``. It
  never scans the runs directory.

Persistence: results are written under ``LAMP_DESIGN_DATA_DIR`` (defaults to
``<repo>/runs``). That is local disk, which is **ephemeral on Cloud Run** -- it
is lost on redeploy and on scale events, and a second instance cannot see the
first instance's runs. ``LAMP_DESIGN_DATA_DIR`` is the seam for pointing at a
mounted volume or a GCS-backed path later, the same role
``PRIMERCAST_DATA_DIR`` plays in PrimerCast.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# `streamlit` and `pandas` are top-level here because this module IS the app --
# it is only ever loaded by `streamlit run`, never by `lamp_design`, and nothing
# under src/ imports deploy/. `modal` is imported lazily, inside the remote
# backend only, so the app works with just the `[streamlit]` extra installed.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where run outputs are persisted. See the module docstring on ephemerality.
DATA_DIR = Path(os.environ.get("LAMP_DESIGN_DATA_DIR") or (REPO_ROOT / "runs"))
RUNS_DIR = DATA_DIR / "runs"

#: Upload cap. NOTE: Streamlit buffers an entire upload in memory *before* this
#: app sees a single byte, so `--server.maxUploadSize` (MB) is the control that
#: protects memory, and this check is the control that protects work. Set both;
#: the server default is 200 MB, which is far too generous for FASTA.
MAX_UPLOAD_MB = float(os.environ.get("LAMP_DESIGN_MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

#: Must match the deployed Modal app. See ../modal/app.py.
MODAL_APP_NAME = os.environ.get("LAMP_DESIGN_MODAL_APP", "lamp-design")
MODAL_FN_SINGLE = "design_single_container"
MODAL_FN_FANOUT = "design_fanout"

ENVELOPE_SCHEMA = "lamp-design/modal-result@1"

#: Concentrations that actually appear in the training data (copies per
#: reaction). Offered as a fixed list rather than a free number box so the
#: model is not asked to extrapolate: `encoding.encode_target_concentration`
#: feeds it `log10(conc) - 3`, whose docstring confirms the scale
#: ("100/1000/10000 -> -1/0/+1").
TRAINED_CONCENTRATIONS = [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 100000000.0]
DEFAULT_CONCENTRATION = 10000.0

#: Master mixes come from the frozen contract (`types.MASTER_MIXES`). BstXT-dk
#: carries the canonical label BST in the training data and is the default here.
MASTER_MIX_LABELS = {
    "BST": "BST - BstXT-dk (default)",
    "SS": "SS",
    "WS": "WS - WarmStart",
}

# ---------------------------------------------------------------------------
# Engine API
#
# Public `lamp_design.*` names only, resolved at the call site, failing with the
# exact name wanted. Keep this table in step with ENGINE_API in
# ../modal/app.py, which documents each signature.
# ---------------------------------------------------------------------------

ENGINE_API: dict[str, tuple[str, str]] = {
    "read_fasta": ("lamp_design.read_fasta", "lamp_design.utils.sequences.read_fasta"),
    "prepare_panel": ("lamp_design.prepare_panel", "lamp_design.commands.prepare.prepare_panel"),
    "default_geometries": (
        "lamp_design.default_geometries",
        "lamp_design.generation.default_geometries",
    ),
    "count_candidate_slots": (
        "lamp_design.count_candidate_slots",
        "lamp_design.generation.count_candidate_slots",
    ),
    "generate_candidates": (
        "lamp_design.generate_candidates",
        "lamp_design.generation.generate_candidates",
    ),
    "build_design_result": (
        "lamp_design.build_design_result",
        "lamp_design.report.build_design_result",
    ),
    "result_to_dict": ("lamp_design.result_to_dict", "lamp_design.report.result_to_dict"),
    "write_outputs": ("lamp_design.write_outputs", "lamp_design.report.write_outputs"),
    "compute_thermo": ("lamp_design.compute_thermo", "lamp_design.thermo.compute_thermo"),
    "predict_tp": ("lamp_design.predict_tp", "lamp_design.models.bridge.predict_tp"),
    # The orchestration entrypoint this app actually runs a design through.
    "design": ("lamp_design.design", "lamp_design.commands.design.design"),
}


def _engine(key: str) -> Any:
    """Resolve a public engine entrypoint, or raise naming exactly what it wanted."""
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
    raise NotImplementedError(
        f"The LAMP design engine does not expose {key} yet. This app expects "
        f"either {primary} or {fallback}. Implement one of them (and re-export "
        f"it from lamp_design/__init__.py); this file deliberately does not stub it."
    )


def _engine_default(key: str, attr: str, fallback: Any) -> Any:
    """Read a default from the engine, falling back if that module is absent.

    Used only for UI bounds, so that the widgets agree with the engine's own
    limits instead of hardcoding a second set that can drift.
    """
    import importlib

    try:
        module = importlib.import_module(key)
    except ModuleNotFoundError:
        return fallback
    return getattr(module, attr, fallback)


#: Mirrors `commands.prepare`. Sequences shorter than MIN_TARGET_LENGTH are
#: excluded by prepare (recorded, never dropped silently); more than
#: MAX_SEQUENCES is rejected outright; BUDGET is `n_targets x alignment width`,
#: the metric ARCHITECTURE.md caps on.
MIN_TARGET_LENGTH = int(_engine_default("lamp_design.commands.prepare", "DEFAULT_MIN_LENGTH", 160))
MAX_SEQUENCES = int(_engine_default("lamp_design.commands.prepare", "MAX_SEQUENCES", 20))
MAX_BUDGET = int(
    os.environ.get("LAMP_DESIGN_MAX_BUDGET")
    or _engine_default("lamp_design.commands.prepare", "DEFAULT_MAX_BUDGET", 50000)
)
DEFAULT_TP_THRESHOLD = float(
    _engine_default("lamp_design.report", "DEFAULT_TP_THRESHOLD_MINUTES", 30.0)
)
DEFAULT_N_SETS = int(_engine_default("lamp_design.report", "DEFAULT_N_SETS", 20))
DEFAULT_MIN_SEPARATION = int(
    _engine_default("lamp_design.report", "DEFAULT_MIN_SEPARATION_BP", 30)
)

#: Local runs share the Modal path's candidate cap. `generate_candidates` thins
#: whole anchors at a uniform stride rather than truncating, so hitting it
#: degrades gracefully instead of returning a head-of-template prefix.
MAX_CANDIDATE_SETS_LOCAL = int(os.environ.get("LAMP_DESIGN_MAX_CANDIDATE_SETS", "200000"))


def _n_available_geometries() -> int:
    """How many deterministic geometries generation offers (canonical first)."""
    try:
        return len(_engine("default_geometries")())
    except NotImplementedError:
        return 5


# ---------------------------------------------------------------------------
# FASTA pre-flight validation
#
# Runs before any work. A *gate*, not a replacement for prepare.py's QC: it
# checks size, shape and alphabet so a malformed or oversized upload is
# rejected without starting a design, and so the rules it enforces are the ones
# prepare will enforce anyway -- notably that only A/C/G/T are accepted, with
# no gaps and no IUPAC ambiguity codes, because RNAduplex and the model encoder
# take nothing else. Deliberately dependency-free.
# ---------------------------------------------------------------------------

ACGT = frozenset("ACGT")
GAP_CHARS = frozenset("-.")


@dataclass
class FastaCheck:
    records: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.records)

    @property
    def n_seqs(self) -> int:
        return len(self.records)

    @property
    def max_len(self) -> int:
        return max((len(s) for _, s in self.records), default=0)

    @property
    def total_bases(self) -> int:
        return sum(len(s) for _, s in self.records)

    @property
    def budget(self) -> int:
        """Lower bound on TargetPanel.budget: alignment gaps only widen it."""
        return self.n_seqs * self.max_len


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Minimal strict FASTA parse. Returns [(name, uppercased sequence)].

    The name is the header up to the first whitespace, matching
    `utils.sequences.read_fasta`, which is what actually parses the file later.
    """
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            header = line[1:].strip()
            name = header.split()[0] if header else ""
            chunks = []
        else:
            chunks.append(line.upper().replace(" ", ""))
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def validate_fasta(text: str, *, pre_aligned: bool = False, source: str = "input") -> FastaCheck:
    """Reject malformed, oversized or unusable input before any design work."""
    check = FastaCheck()
    allowed = ACGT | GAP_CHARS if pre_aligned else ACGT

    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        check.errors.append("No sequence provided.")
        return check
    if n_bytes > MAX_UPLOAD_BYTES:
        check.errors.append(
            f"{source} is {n_bytes / 1e6:.1f} MB, over the {MAX_UPLOAD_MB:g} MB limit."
        )
        return check
    if not text.lstrip().startswith(">"):
        check.errors.append(
            "Not FASTA: the first non-blank line must start with '>'. "
            "Paste a bare sequence into the text box instead and it will be wrapped for you."
        )
        return check

    records = _parse_fasta(text)
    if not records:
        check.errors.append("No sequences found.")
        return check
    if len(records) > MAX_SEQUENCES:
        check.errors.append(
            f"{len(records)} sequences given; the engine accepts 1-{MAX_SEQUENCES}. "
            "A panel this large is usually a whole database rather than a design "
            "target -- pick a representative subset first."
        )
        return check

    seen: dict[str, int] = {}
    for index, (name, seq) in enumerate(records, start=1):
        label = name or f"record {index}"
        if not name:
            check.errors.append(f"Record {index} has an empty header.")
            continue
        if not seq:
            check.errors.append(f"'{label}' has a header but no sequence.")
            continue
        seen[name] = seen.get(name, 0) + 1

        bad = sorted(set(seq) - allowed)
        if bad:
            shown = " ".join(repr(c) for c in bad[:6]) + (" ..." if len(bad) > 6 else "")
            if set(bad) & GAP_CHARS and not pre_aligned:
                check.errors.append(
                    f"'{label}' contains alignment gaps ({shown}). Tick "
                    "'input is already aligned' if you are supplying an alignment."
                )
            else:
                check.errors.append(
                    f"'{label}' contains characters other than A, C, G and T: {shown}. "
                    "Ambiguity codes (N, R, Y, ...) are rejected rather than cleaned -- "
                    "RNAduplex and the model encoder accept only A, C, G and T."
                )
            continue

        if len(seq) < MIN_TARGET_LENGTH:
            check.warnings.append(
                f"'{label}' is {len(seq)} nt, below the engine's {MIN_TARGET_LENGTH} nt floor. "
                "It will be excluded from the panel and reported as such."
            )
        check.records.append((name, seq))

    duplicates = sorted(n for n, count in seen.items() if count > 1)
    if duplicates:
        # Per-target results are keyed by name, so duplicates would collide in
        # the Tp matrix rather than merely look odd. prepare rejects them too.
        check.errors.append(
            "Duplicate sequence names: " + ", ".join(duplicates[:5]) + ". Names must be unique."
        )

    usable = [s for _, s in check.records if len(s) >= MIN_TARGET_LENGTH]
    if check.records and not usable:
        check.errors.append(
            f"Every sequence is shorter than {MIN_TARGET_LENGTH} nt, so no primer set can "
            "be placed. Provide at least one longer target."
        )
    if check.budget > MAX_BUDGET:
        check.errors.append(
            f"Input is too large: {check.n_seqs} sequences x {check.max_len} nt = "
            f"{check.budget:,}, over the {MAX_BUDGET:,} budget limit. Reduce the number "
            "of sequences or trim them to the region of interest."
        )
    return check


# ---------------------------------------------------------------------------
# Per-run isolation
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
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
    stage takes explicit paths -- so the lock/control-file collision that
    motivated that module cannot arise, as long as nobody adds ``os.chdir()``.
    """
    return RunPaths(
        run_id=run_id,
        out_dir=out_dir_for(run_id),
        work_dir=Path(tempfile.mkdtemp(prefix=f"lamp_design_{run_id}_")),
    )


def session_runs() -> list[str]:
    """Run ids this browser session owns. The only runs the app will render."""
    return st.session_state.setdefault("session_runs", [])


# ---------------------------------------------------------------------------
# Backends
#
# Both produce the envelope documented in deploy/README.md:
#   {schema, run_id, request, timing_seconds, result: report.result_to_dict(...)}
# Reusing the engine's own `result_to_dict` means there is no second copy of
# the result schema to drift.
# ---------------------------------------------------------------------------


def run_local(request: dict, paths: RunPaths) -> dict:
    """Run the design in this process. Blocks the session for the whole run.

    Prepare happens here so the UI can show the panel's QC before committing to
    a run; everything after it is one call to ``lamp_design.design``, which is
    the engine's orchestration entrypoint. Earlier revisions of this file
    re-implemented the stage sequence because no such entrypoint existed -- do
    not reintroduce that, or the app and the CLI drift.
    """
    from lamp_design.types import Condition

    timing: dict[str, float] = {}
    condition = Condition(
        master_mix=request["master_mix"], target_conc=request["target_conc"]
    )

    started = time.time()
    input_fasta = paths.work_dir / "input.fasta"
    input_fasta.write_text(request["fasta"])
    records = _engine("read_fasta")(input_fasta)
    prepared = _engine("prepare_panel")(
        records,
        max_budget=MAX_BUDGET,
        pre_aligned=request["pre_aligned"],
        threads=int(os.environ.get("LAMP_DESIGN_ALIGN_THREADS", "2")),
        panel_name=request["run_id"],
    )
    panel = prepared.panel
    geometries = list(_engine("default_geometries")())[: request["n_geometries"]]
    slots = _engine("count_candidate_slots")(
        len(panel.consensus), geometries=geometries, stride=request["stride"]
    )
    funnel = dict(prepared.diagnostics.as_funnel()) if prepared.diagnostics else {}
    funnel["candidate_slots"] = int(slots)
    timing["prepare"] = time.time() - started

    started = time.time()
    result = _engine("design")(
        panel,
        weights=os.environ.get("LAMP_DESIGN_WEIGHTS") or None,
        conditions=[condition],
        ranking_condition=condition,
        stride=request["stride"],
        geometries=geometries,
        max_candidates=MAX_CANDIDATE_SETS_LOCAL,
        n=request["n_sets"],
        min_separation=request["min_separation"],
        tp_threshold=request["tp_threshold_minutes"],
    )
    timing["design"] = time.time() - started

    # prepare's funnel/exclusions/warnings are transferred onto the result the
    # same way the CLI does it, then the slot counts this app computed are
    # merged on top.
    prepared.apply_to(result)
    funnel["candidates_generated"] = int(result.params.get("n_candidates", 0))
    result.funnel.update(funnel)

    # The engine's own per-stage breakdown, so the UI shows where time actually
    # went rather than one opaque number. Keys are suffixed `_s` by the engine
    # and are a subset of `design`, so they are reported but not re-summed.
    timing.update(result.params.get("timings_s", {}))
    timing["total"] = timing["prepare"] + timing["design"]

    payload = _engine("result_to_dict")(
        result, ranking_condition=condition, tp_threshold=request["tp_threshold_minutes"]
    )
    # The engine's own writer: .sets.csv / .per_target.csv / .json / .summary.txt,
    # identical to a CLI run's artifacts.
    _engine("write_outputs")(
        result,
        paths.out_dir / "result",
        ranking_condition=condition,
        tp_threshold=request["tp_threshold_minutes"],
    )
    return {
        "schema": ENVELOPE_SCHEMA,
        "run_id": request["run_id"],
        "request": {k: v for k, v in request.items() if k != "fasta"},
        "timing_seconds": {k: round(float(v), 3) for k, v in timing.items()},
        "result": payload,
    }


def modal_submit(request: dict, function_name: str) -> str:
    """Submit a design to the deployed Modal app; return the poll token.

    Design runs take tens of seconds to minutes, so the remote backend is
    submit-and-poll rather than a blocking call: Streamlit would otherwise hold
    the session (and a Cloud Run request) open for the whole run.
    """
    import modal  # lazy: only the Modal backend needs this extra

    call = modal.Function.from_name(MODAL_APP_NAME, function_name).spawn(request)
    return call.object_id


def modal_poll(call_id: str) -> dict | None:
    """Return the envelope, None if still running, or raise if the run failed."""
    import modal

    try:
        return modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError:
        # UNVERIFIED (no Modal account here): that a not-yet-finished poll
        # surfaces as a builtin TimeoutError. If polling reports a hard failure
        # for a run that is merely still going, this one except clause is the
        # thing to widen.
        return None


# ---------------------------------------------------------------------------
# Rendering
#
# Reads the payload shape that `report.result_to_dict` produces: `sets` with
# `primers` as a list in frozen PRIMER_TYPE_ORDER and `per_target` as a list,
# and the column names report.py uses, which spell out their direction
# ("..._lower_is_faster").
# ---------------------------------------------------------------------------

SPEED_FIELD = "mean_predicted_tp_minutes_lower_is_faster"
INCLUSIVITY_FIELD = "inclusivity_fraction_targets_below_tp_threshold"


def _sets_dataframe(result: dict) -> pd.DataFrame:
    rows = []
    for entry in result["sets"]:
        row = {
            "rank": entry["rank"],
            "set_id": entry["set_id"],
            "anchor": entry["anchor"],
            "geometry": entry["geometry_id"],
            "span_start": entry["span_start"],
            "span_end": entry["span_end"],
            "mean_tp_min": entry[SPEED_FIELD],
            "inclusivity": entry[INCLUSIVITY_FIELD],
            "slowest_target": entry.get("slowest_target_name"),
            "slowest_tp_min": entry.get("slowest_predicted_tp_minutes"),
            "total_mismatches": entry.get("total_primer_target_mismatches_all_targets"),
        }
        for primer in entry["primers"]:
            row[primer["primer_type"]] = primer["sequence"]
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix(result: dict, field_name: str) -> pd.DataFrame:
    """Rows = ranked sets, columns = targets."""
    targets = result["panel"]["target_names"]
    rows = []
    for entry in result["sets"]:
        by_target = {p["target_name"]: p.get(field_name) for p in entry["per_target"]}
        row = {"rank": entry["rank"], "set_id": entry["set_id"]}
        row.update({name: by_target.get(name) for name in targets})
        rows.append(row)
    return pd.DataFrame(rows)


def _primers_long(result: dict) -> pd.DataFrame:
    rows = []
    for entry in result["sets"]:
        for primer in entry["primers"]:
            rows.append(
                {
                    "rank": entry["rank"],
                    "set_id": entry["set_id"],
                    "primer_type": primer["primer_type"],
                    "strand": primer.get("strand"),
                    "sequence": primer["sequence"],
                    "start": primer["start"],
                    "end": primer["end"],
                    "length": primer["length"],
                }
            )
    return pd.DataFrame(rows)


def _download(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def render_funnel(result: dict, *, emphasise: bool) -> None:
    """Explain which stage emptied the result, instead of an empty table.

    Every stage records its surviving-candidate count into
    ``DesignResult.funnel`` precisely so that "no candidates" can name the
    filter that did it. The first stage whose count is zero is the answer.
    """
    funnel = result.get("funnel") or {}
    if not funnel:
        if emphasise:
            st.error(
                "No candidate sets were returned, and the run reported no funnel, so "
                "the stage that emptied it cannot be identified. Every stage is "
                "supposed to record its surviving-candidate count in DesignResult.funnel."
            )
        return

    frame = pd.DataFrame(
        [{"stage": stage, "candidates remaining": count} for stage, count in funnel.items()]
    )
    culprit = next((stage for stage, count in funnel.items() if count == 0), None)

    if emphasise:
        if culprit:
            st.error(
                f"**No candidate sets survived.** The funnel empties at **{culprit}** -- "
                "that is the stage to loosen."
            )
        else:
            st.error(
                "**No candidate sets were returned**, although no funnel stage reports "
                "zero. Check the warnings below."
            )
        st.dataframe(frame, hide_index=True, use_container_width=True)
    else:
        with st.expander(f"Candidate funnel ({len(funnel)} stages)"):
            st.dataframe(frame, hide_index=True, use_container_width=True)


def render_envelope(envelope: dict) -> None:
    run_id = envelope["run_id"]
    result = envelope["result"]
    ranking = result.get("ranking") or {}
    threshold = ranking.get("tp_threshold_minutes")

    for message in result.get("warnings") or []:
        st.warning(message)

    excluded = result.get("excluded_targets") or {}
    if excluded:
        with st.expander(f"{len(excluded)} target(s) excluded during preparation"):
            st.dataframe(
                pd.DataFrame(
                    [{"target": name, "reason": reason} for name, reason in excluded.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )

    if not result.get("sets"):
        render_funnel(result, emphasise=True)
        return

    timing = envelope.get("timing_seconds") or {}
    total = timing.get("total") or sum(timing.values())
    columns = st.columns(4)
    columns[0].metric("Sets returned", result["n_sets_returned"])
    columns[1].metric("Targets scored", result["panel"]["n_targets"])
    columns[2].metric("Best mean Tp", f"{result['sets'][0][SPEED_FIELD]:.1f} min")
    columns[3].metric("Runtime", f"{total:.0f} s")

    render_funnel(result, emphasise=False)

    sets_frame = _sets_dataframe(result)
    tp_frame = _matrix(result, "predicted_tp_minutes")
    mismatch_frame = _matrix(result, "primer_target_mismatches")
    primers_frame = _primers_long(result)

    st.subheader("Ranked primer sets")
    st.caption(
        "Spread across loci rather than ranked purely by score, so the list is not one "
        "design repeated. `mean_tp_min` is predicted minutes to amplification -- "
        "**lower is faster**. `inclusivity` is the fraction of targets predicted to "
        f"amplify in under {threshold} minutes -- higher is better."
    )
    st.dataframe(sets_frame, hide_index=True, use_container_width=True)

    st.subheader("Predicted Tp by target (minutes)")
    st.caption("Every returned set scored against every input sequence. Lower is faster.")
    st.dataframe(tp_frame, hide_index=True, use_container_width=True)

    with st.expander("Primer-target mismatches by target"):
        st.dataframe(mismatch_frame, hide_index=True, use_container_width=True)

    with st.expander("All primers, one row per primer"):
        st.caption(
            "Strand '-' means the sequence is the reverse complement of its target "
            "window, per the frozen convention."
        )
        st.dataframe(primers_frame, hide_index=True, use_container_width=True)

    st.subheader("Downloads")
    columns = st.columns(4)
    with columns[0]:
        _download("Ranked sets (CSV)", sets_frame, f"lamp_sets_{run_id}.csv", f"dl-sets-{run_id}")
    with columns[1]:
        _download("Tp matrix (CSV)", tp_frame, f"lamp_tp_{run_id}.csv", f"dl-tp-{run_id}")
    with columns[2]:
        _download(
            "Mismatches (CSV)", mismatch_frame, f"lamp_mismatches_{run_id}.csv", f"dl-mm-{run_id}"
        )
    with columns[3]:
        _download(
            "Primers (CSV)", primers_frame, f"lamp_primers_{run_id}.csv", f"dl-primers-{run_id}"
        )

    st.caption(
        f"Run `{run_id}` persisted to `{RUNS_DIR / run_id}`, alongside the engine's own "
        "`result.sets.csv`, `result.per_target.csv`, `result.json` and "
        "`result.summary.txt`. That is local disk: on Cloud Run it is lost on redeploy "
        "or scale events, and another instance cannot see it. Download anything you "
        "need to keep."
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LAMP primer design", layout="wide")
st.title("LAMP primer design")
st.caption(
    "ML-guided design of eight-primer LAMP sets. Candidates are tiled across the whole "
    "input and every set is scored against every input sequence."
)


def _sidebar() -> dict:
    st.sidebar.header("Reaction")
    master_mix = st.sidebar.selectbox(
        "Master mix",
        options=list(MASTER_MIX_LABELS),
        index=list(MASTER_MIX_LABELS).index("BST"),
        format_func=lambda key: MASTER_MIX_LABELS[key],
        help="The model is conditioned on the master mix as a 3-way one-hot.",
    )
    target_conc = st.sidebar.selectbox(
        "Target concentration (copies per reaction)",
        options=TRAINED_CONCENTRATIONS,
        index=TRAINED_CONCENTRATIONS.index(DEFAULT_CONCENTRATION),
        format_func=lambda value: f"{value:,.0f}",
        help=(
            "Only values present in the training data are offered; the encoder feeds "
            "the model log10(conc) - 3, and anything outside this range is extrapolation."
        ),
    )

    st.sidebar.header("Search")
    n_sets = st.sidebar.number_input(
        "Primer sets to return",
        min_value=1,
        max_value=200,
        value=DEFAULT_N_SETS,
        step=1,
        help="Returned spread across loci, not simply the top N by score.",
    )
    stride = st.sidebar.number_input(
        "Anchor stride (nt)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Step between candidate anchor positions. Smaller means more candidates "
        "and a longer run.",
    )
    n_available = _n_available_geometries()
    n_geometries = st.sidebar.number_input(
        "Geometries per anchor",
        min_value=1,
        max_value=n_available,
        value=min(4, n_available),
        step=1,
        help=(
            f"How many of the engine's {n_available} deterministic geometries to lay "
            "down at each anchor. They are ordered canonical-first, so any number "
            "includes the canonical (5,0,0,5,0,0,5,0) spacing at length 19-20, which "
            "is where the model is best calibrated."
        ),
    )

    with st.sidebar.expander("Advanced"):
        pre_aligned = st.checkbox(
            "Input is already aligned",
            help="Accept gap characters and use the rows as the alignment instead of "
            "running MAFFT.",
        )
        tp_threshold = st.number_input(
            "Tp threshold for inclusivity (minutes)",
            min_value=1.0,
            max_value=240.0,
            value=DEFAULT_TP_THRESHOLD,
            step=5.0,
            help="A set's inclusivity is the fraction of targets predicted to amplify "
            "strictly within this time.",
        )
        min_separation = st.number_input(
            "Minimum anchor separation (nt)",
            min_value=0,
            max_value=1000,
            value=DEFAULT_MIN_SEPARATION,
            step=5,
            help="Returned sets are kept at least this far apart, so the list is not "
            "one design repeated. 0 disables it and degrades to a plain top-N.",
        )
        backend = st.radio(
            "Backend",
            options=["local", "modal-single", "modal-fanout"],
            help=(
                "local: in this process, blocking. modal-single: one GPU container, "
                "submit-and-poll. modal-fanout: CPU fan-out for thermodynamics plus "
                "one GPU container -- the scale-up path."
            ),
        )

    return {
        "master_mix": master_mix,
        "target_conc": float(target_conc),
        "n_sets": int(n_sets),
        "stride": int(stride),
        "n_geometries": int(n_geometries),
        "tp_threshold_minutes": float(tp_threshold),
        "min_separation": int(min_separation),
        "pre_aligned": bool(pre_aligned),
        "backend": backend,
    }


def _input_panel() -> str:
    """Collect FASTA text from an upload or the paste box. No work is done here."""
    st.subheader("Target sequences")
    uploaded = st.file_uploader(
        f"FASTA file (max {MAX_UPLOAD_MB:g} MB, up to {MAX_SEQUENCES} sequences)",
        type=["fa", "fasta", "fna", "fas", "txt"],
        accept_multiple_files=False,
    )
    pasted = st.text_area(
        "...or paste sequences",
        height=160,
        placeholder=">my_target\nACGT...",
        help="A bare sequence with no '>' header is wrapped as a single record.",
    )

    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            st.error(
                f"'{uploaded.name}' is {len(data) / 1e6:.1f} MB, over the "
                f"{MAX_UPLOAD_MB:g} MB limit."
            )
            return ""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            st.error(
                f"'{uploaded.name}' is not UTF-8 text. Upload a plain FASTA file, not a "
                "compressed or binary one."
            )
            return ""

    text = pasted.strip()
    if text and not text.startswith(">"):
        return ">pasted_sequence\n" + text
    return text


params = _sidebar()
fasta_text = _input_panel()

check = (
    validate_fasta(fasta_text, pre_aligned=params["pre_aligned"]) if fasta_text else None
)
if check is not None:
    for message in check.errors:
        st.error(message)
    for message in check.warnings:
        st.warning(message)
    if check.ok:
        # Pure arithmetic from the chosen stride/geometries -- an estimate, not
        # a call into generation.py, so the cost is visible before committing.
        anchors = max(0, (check.max_len - MIN_TARGET_LENGTH) // params["stride"] + 1)
        st.success(
            f"{check.n_seqs} sequence(s), longest {check.max_len:,} nt, "
            f"{check.total_bases:,} nt total. Roughly "
            f"{anchors * params['n_geometries']:,} candidate sets at stride "
            f"{params['stride']} x {params['n_geometries']} geometries, each scored "
            f"against {check.n_seqs} target(s)."
        )

st.divider()
if st.button("Run design", type="primary", disabled=check is None or not check.ok):
    request = {
        "run_id": new_run_id(),
        "fasta": fasta_text,
        "master_mix": params["master_mix"],
        "target_conc": params["target_conc"],
        "n_sets": params["n_sets"],
        "stride": params["stride"],
        "n_geometries": params["n_geometries"],
        "tp_threshold_minutes": params["tp_threshold_minutes"],
        "min_separation": params["min_separation"],
        "pre_aligned": params["pre_aligned"],
    }
    session_runs().append(request["run_id"])

    if params["backend"] == "local":
        paths = prepare_run(request["run_id"])
        try:
            with st.spinner("Designing... this blocks this session until it finishes."):
                envelope = run_local(request, paths)
            (paths.out_dir / "envelope.json").write_text(json.dumps(envelope, indent=2))
            st.session_state["envelope"] = envelope
            st.session_state.pop("pending_call", None)
        except NotImplementedError as exc:
            st.error(str(exc))
        except Exception as exc:  # a failed design must not take the app down
            st.exception(exc)
        finally:
            # Always remove the scratch dir, success or failure. Only the
            # persisted out_dir survives.
            shutil.rmtree(paths.work_dir, ignore_errors=True)
    else:
        function_name = (
            MODAL_FN_SINGLE if params["backend"] == "modal-single" else MODAL_FN_FANOUT
        )
        try:
            call_id = modal_submit(request, function_name)
            st.session_state["pending_call"] = {
                "call_id": call_id,
                "run_id": request["run_id"],
                "function": function_name,
            }
            st.session_state.pop("envelope", None)
            st.info(f"Submitted to Modal as `{call_id}`. Poll for the result below.")
        except ModuleNotFoundError:
            st.error(
                "The `modal` extra is not installed. Either install it "
                "(`pip install -e '.[modal]'`) or use the local backend."
            )
        except Exception as exc:
            st.exception(exc)

pending = st.session_state.get("pending_call")
if pending:
    st.subheader("Remote run")
    st.write(
        f"Run `{pending['run_id']}` submitted to `{pending['function']}` "
        f"(call `{pending['call_id']}`)."
    )
    if st.button("Check for results"):
        # A manual poll rather than an auto-refresh loop: a design takes
        # minutes, and auto-rerunning would re-poll on every widget
        # interaction for no benefit.
        try:
            envelope = modal_poll(pending["call_id"])
        except Exception as exc:
            st.error("The remote run failed.")
            st.exception(exc)
            st.session_state.pop("pending_call", None)
        else:
            if envelope is None:
                st.info("Still running.")
            else:
                # No scratch dir here: the remote container owned its own.
                out_dir = out_dir_for(pending["run_id"])
                (out_dir / "envelope.json").write_text(json.dumps(envelope, indent=2))
                st.session_state["envelope"] = envelope
                st.session_state.pop("pending_call", None)

envelope = st.session_state.get("envelope")
if envelope:
    st.divider()
    # Only ever renders an envelope this session produced: nothing scans
    # RUNS_DIR, so one session cannot see another's sequences or results.
    if envelope["run_id"] not in session_runs():
        st.error("Result does not belong to this session; refusing to display it.")
    else:
        render_envelope(envelope)
