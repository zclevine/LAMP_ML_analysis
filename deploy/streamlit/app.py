"""Streamlit front end for the LAMP design engine.

Two workflows, each a wizard with a step indicator:

* **Design** — Target → Parameters → Run → Results. Tiles candidate
  eight-primer sets across the whole input, scores every candidate against
  every input sequence with the trained Simplified Lyra ensemble, and returns
  the top N positionally-diverse sets.
* **Evaluate** — Target → Primer Sets → Parameters → Run → Results. Scores
  primer sets the user already has. ``lamp-design evaluate`` is still being
  written, so this workflow resolves it lazily and says so when it is absent
  rather than crashing.

The structure mirrors PrimerCast's GUI (``/home/ssiddiqu/PrimerCast/gui/app.py``)
so the two portals navigate the same way: ``(page_key, label)`` step lists, a
sidebar that always shows the current selections, expander-grouped parameters,
a preflight gate, a run page with a live log and a candidate funnel, and
results tabs with per-table downloads.

Module layout, and why:

``_engine_api.py``  dotted-name table for every engine entry point, plus the
                    bounds the widgets are built from. No streamlit, no torch.
``_validation.py``  input gates (FASTA, primer sets) that run before any work.
``_backend.py``     the one backend seam — ``local`` subprocess or ``modal`` —
                    both returning the same envelope. No streamlit.
``_runner.py``      the out-of-process entry point the local backend spawns.
``_tables.py``      result payload → dataframes. Pandas only.
``_layout.py``      navigation, step indicator, sidebar, preflight.
``_results.py``     laying those frames out, with downloads.
``app.py``          the pages and the router.

The split is not cosmetic: everything above ``_layout`` is importable without
Streamlit, which is what lets the runner subprocess and ``tests/
test_streamlit_app.py`` use the same code the pages do.

Run it::

    .venv/bin/streamlit run deploy/streamlit/app.py \
        --server.headless true --server.port 8501 --server.maxUploadSize 5

``--server.maxUploadSize`` matters: Streamlit buffers an entire upload in
memory *before* this app sees a byte, so the flag is what protects memory and
``_validation.MAX_UPLOAD_MB`` is what protects work. Set both.
"""

from __future__ import annotations

import time

import streamlit as st

st.set_page_config(page_title="LAMP primer design", layout="wide")

import _backend  # noqa: E402  (must follow set_page_config)
import _layout  # noqa: E402
import _results  # noqa: E402
from _engine_api import (  # noqa: E402
    MASTER_MIXES,
    MAX_CANDIDATE_SETS,
    MIN_TARGET_LENGTH,
    PREDICTIONS_PER_SECOND_CPU,
    PREDICTIONS_PER_SECOND_GPU,
    PRIMER_TYPE_ORDER,
    TRAINED_CONCENTRATIONS,
    geometry_labels,
    n_available_geometries,
    resolved_device,
)
from _tables import funnel_frame  # noqa: E402
from _validation import (  # noqa: E402
    MAX_PRIMER_SETS,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    primer_sets_to_csv,
)

#: How often the Run page re-reads a live run's status file. Short enough that
#: the stage checklist feels live, long enough that a seven-minute run is not
#: hundreds of script reruns.
POLL_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

_ICON_DESIGN = """
<svg viewBox="0 0 120 80" width="120" height="80" xmlns="http://www.w3.org/2000/svg">
  <line x1="6" y1="22" x2="114" y2="22" stroke="#888" stroke-width="3" stroke-linecap="round"/>
  <rect x="10" y="30" width="12" height="5" rx="2" fill="#4CAF50" opacity="0.3"/>
  <rect x="26" y="30" width="12" height="5" rx="2" fill="#4CAF50" opacity="0.3"/>
  <rect x="42" y="30" width="12" height="5" rx="2" fill="#4CAF50" opacity="0.3"/>
  <rect x="66" y="30" width="12" height="5" rx="2" fill="#E8451E" opacity="0.3"/>
  <rect x="82" y="30" width="12" height="5" rx="2" fill="#E8451E" opacity="0.3"/>
  <rect x="98" y="30" width="12" height="5" rx="2" fill="#E8451E" opacity="0.3"/>
  <rect x="14" y="48" width="13" height="6" rx="2" fill="#4CAF50"/>
  <rect x="31" y="48" width="13" height="6" rx="2" fill="#4CAF50"/>
  <rect x="48" y="48" width="13" height="6" rx="2" fill="#4CAF50"/>
  <rect x="65" y="48" width="13" height="6" rx="2" fill="#E8451E"/>
  <rect x="82" y="48" width="13" height="6" rx="2" fill="#E8451E"/>
  <rect x="99" y="48" width="13" height="6" rx="2" fill="#E8451E"/>
  <rect x="9" y="43" width="108" height="16" rx="3" fill="none" stroke="#2E86AB" stroke-width="2"/>
  <rect x="22" y="68" width="12" height="5" rx="2" fill="#4CAF50" opacity="0.3"/>
  <rect x="38" y="68" width="12" height="5" rx="2" fill="#4CAF50" opacity="0.3"/>
  <rect x="70" y="68" width="12" height="5" rx="2" fill="#E8451E" opacity="0.3"/>
  <rect x="86" y="68" width="12" height="5" rx="2" fill="#E8451E" opacity="0.3"/>
</svg>
"""

_ICON_EVALUATE = """
<svg viewBox="0 0 120 80" width="120" height="80" xmlns="http://www.w3.org/2000/svg">
  <line x1="6" y1="22" x2="114" y2="22" stroke="#888" stroke-width="3" stroke-linecap="round"/>
  <rect x="14" y="30" width="16" height="6" rx="2" fill="#4CAF50"/>
  <rect x="36" y="30" width="16" height="6" rx="2" fill="#4CAF50"/>
  <rect x="68" y="30" width="16" height="6" rx="2" fill="#E8451E"/>
  <rect x="90" y="30" width="16" height="6" rx="2" fill="#E8451E"/>
  <line x1="18" y1="24" x2="18" y2="29" stroke="#4CAF50" stroke-width="1.5"/>
  <line x1="24" y1="24" x2="24" y2="29" stroke="#E8451E" stroke-width="1.5"/>
  <line x1="40" y1="24" x2="40" y2="29" stroke="#4CAF50" stroke-width="1.5"/>
  <line x1="74" y1="24" x2="74" y2="29" stroke="#4CAF50" stroke-width="1.5"/>
  <line x1="96" y1="24" x2="96" y2="29" stroke="#E8451E" stroke-width="1.5"/>
  <circle cx="60" cy="58" r="14" fill="none" stroke="#555" stroke-width="2.5"/>
  <line x1="70" y1="68" x2="80" y2="78" stroke="#555" stroke-width="3" stroke-linecap="round"/>
  <polyline points="53,58 58,64 68,52" fill="none" stroke="#4CAF50" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def _page_home() -> None:
    st.markdown(
        "<h1 style='text-align:center;margin-top:-2rem;margin-bottom:0;'>"
        "LAMP primer design</h1>"
        "<p style='text-align:center;font-size:1.2em;color:gray;margin-bottom:0;'>"
        "ML-guided design and evaluation of eight-primer LAMP sets</p>"
        "<hr style='margin-top:1rem;margin-bottom:1rem;'>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"<div style='text-align:center;padding:0.3em 0;'>{_ICON_DESIGN}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Design", width="stretch", type="primary", key="home_design"):
            _layout.reset_workflow_state()
            st.session_state.workflow = "design"
            _layout.navigate("target")
            st.rerun()
        st.caption(
            "Generate new primer sets for your target sequences. Candidates are tiled "
            "across the whole input and every set is scored against every sequence."
        )
    with right:
        st.markdown(
            f"<div style='text-align:center;padding:0.3em 0;'>{_ICON_EVALUATE}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Evaluate", width="stretch", type="primary", key="home_evaluate"):
            _layout.reset_workflow_state()
            st.session_state.workflow = "evaluate"
            _layout.navigate("target")
            st.rerun()
        st.caption(
            "Score primer sets you already have against a target panel: the same "
            "thermodynamics and inference as Design, without candidate generation."
        )
        if not _backend.evaluate_available():
            st.caption(":orange[Not available in this build — see the page for details.]")

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
Predictions come from the **Simplified Lyra** ensemble (five seeds, averaged),
which takes a primer set, a target sequence and a reaction condition and
predicts **Tp — time to positive, in minutes**. Lower is faster. A run scores
every candidate against every input sequence under each requested master mix
and template concentration, then returns sets spread across loci rather than
the top N by score, because the top N by score is usually one design repeated.

**What this does not do**, because each is easy to assume is present:

* **No specificity screen.** There is no off-target or host check of any kind.
  Every score is an *inclusivity* statement about the sequences you supplied.
* **No dimer or hairpin filter.** The model never saw non-specific
  amplification, so nothing here rejects a set whose primers prefer each other.
* **Absolute minutes are indicative.** What is validated is that this pipeline
  reproduces the published model's predictions. Treat the *ranking* as the
  product until it has been checked against known-good LAMP sets.
"""
    )


# ---------------------------------------------------------------------------
# Step 1: Target
# ---------------------------------------------------------------------------


def _decode_upload(uploaded) -> str | None:
    """Bytes from an upload as text, or None with the reason shown."""
    data = uploaded.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        st.error(
            f"'{uploaded.name}' is {len(data) / 1e6:.1f} MB, over the "
            f"{MAX_UPLOAD_MB:g} MB limit."
        )
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        st.error(
            f"'{uploaded.name}' is not UTF-8 text. Upload a plain FASTA file, not a "
            "compressed or binary one."
        )
        return None


def _page_target() -> None:
    _layout.render_workflow_progress("target")
    label = _layout.WORKFLOW_LABELS[_layout.workflow()]
    st.header(f"{label} — Target sequences")
    st.markdown(
        "Which sequences should the primers work against? One sequence designs "
        "directly against it; 2 to 20 are aligned, a consensus is derived, and every "
        "set is scored against **all** of them."
    )
    _layout.render_input_limits()
    st.divider()

    source = st.radio(
        "Input method",
        ["Upload a FASTA file", "Paste sequences"],
        key="target_source",
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Upload a FASTA file":
        uploaded = st.file_uploader(
            "Target FASTA",
            type=["fa", "fasta", "fna", "fas", "txt"],
            accept_multiple_files=False,
            key="target_upload",
        )
        if uploaded is not None:
            text = _decode_upload(uploaded)
            st.session_state["fasta_text"] = text or ""
            st.session_state["fasta_origin"] = uploaded.name
        elif st.session_state.get("fasta_origin") not in (None, "pasted"):
            st.info(
                f"Using `{st.session_state['fasta_origin']}` from earlier in this "
                "session. Upload another file to replace it."
            )
        else:
            st.session_state["fasta_text"] = ""
    else:
        if "target_paste" not in st.session_state and st.session_state.get(
            "fasta_origin"
        ) == "pasted":
            st.session_state["target_paste"] = st.session_state.get("fasta_text", "")
        pasted = st.text_area(
            "Sequences",
            height=200,
            key="target_paste",
            placeholder=">my_target\nACGT...",
            help="A bare sequence with no '>' header is wrapped as a single record.",
        )
        text = (pasted or "").strip()
        if text and not text.startswith(">"):
            text = ">pasted_sequence\n" + text
        st.session_state["fasta_text"] = text
        st.session_state["fasta_origin"] = "pasted"

    params = _layout.init_params()
    params["pre_aligned"] = st.checkbox(
        "Input is already aligned",
        value=bool(params["pre_aligned"]),
        key="p_pre_aligned",
        help=(
            "Accept gap characters and use the rows as the alignment instead of running "
            "MAFFT. Required when the sequences differ in length and MAFFT is not "
            "installed."
        ),
    )

    check = _layout.target_check()
    if check is not None:
        for message in check.errors:
            st.error(message)
        for message in check.warnings:
            st.warning(message)
        if check.ok:
            columns = st.columns(4)
            columns[0].metric("Sequences", check.n_seqs)
            columns[1].metric("Longest", f"{check.max_len:,} nt")
            columns[2].metric("Total", f"{check.total_bases:,} nt")
            columns[3].metric("Budget", f"{check.budget:,}")
            if check.ragged and not params["pre_aligned"]:
                st.info(
                    f"The sequences differ in length ({check.min_len:,}-{check.max_len:,} "
                    "nt), so MAFFT will be run to align them before a consensus is "
                    "derived."
                )
            elif check.n_seqs > 1 and not params["pre_aligned"]:
                st.info(
                    f"All {check.n_seqs} sequences are {check.max_len:,} nt with no gaps, "
                    "so the engine treats them as already aligned and does not run MAFFT."
                )

    st.divider()
    forward = ("primer_sets", "Primer sets") if _layout.workflow() == "evaluate" else (
        "parameters",
        "Parameters",
    )
    _layout.render_step_nav(
        back=("home", "Home"),
        forward=forward,
        forward_disabled=check is None or not check.ok,
        key="target",
    )
    if check is None or not check.ok:
        st.caption("Provide valid target sequences to continue.")


# ---------------------------------------------------------------------------
# Step 2 (Evaluate only): Primer sets
# ---------------------------------------------------------------------------


def _page_primer_sets() -> None:
    _layout.render_workflow_progress("primer_sets")
    st.header("Evaluate — Primer sets")

    if not _backend.evaluate_available():
        st.warning(
            "**`lamp-design evaluate` is not available in this build.** The command is "
            "registered in the CLI as a stub and the engine module behind it "
            "(`lamp_design.commands.evaluate`) has not landed yet, so a run here would "
            "have nothing to call. You can still enter primer sets and settings — the "
            "Run step will refuse and say why. The Design workflow is unaffected."
        )

    st.markdown(
        "Which sets should be scored? A LAMP set is all eight primers — "
        f"**{', '.join(PRIMER_TYPE_ORDER)}** — in the frozen layout order along the "
        "target. Forward-strand types are `F3, F2, B1c, BL`; the rest are supplied "
        "reverse-complemented, which is the orientation the model sees."
    )
    st.caption(
        f"Up to {MAX_PRIMER_SETS} sets, A/C/G/T only — degenerate primers are rejected "
        "rather than expanded. Each primer's binding site is located in the target "
        "sequences from the previous step by best Hamming match."
    )
    st.divider()

    source = st.radio(
        "Input method",
        ["Upload a table", "Paste primers"],
        key="primer_source",
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Upload a table":
        st.caption(
            "A CSV or TSV with one column per primer, named `F3_sequence` or `F3`. "
            "Optional columns are used when present: `primer_set` and `target_seq` for "
            "names, `master_mix` and `target_conc` for the condition each row is scored "
            "under, `<type>_target_sequence` for windows that are already known, and "
            "**`tp` for a measured Tp — which turns the run into a validation** and "
            "reports Spearman, MAE and ROC-AUC against it. The file is passed to the "
            "engine unchanged, so every one of those keeps working."
        )
        uploaded = st.file_uploader(
            "Primer sets",
            type=["csv", "tsv", "tab", "txt"],
            accept_multiple_files=False,
            key="primer_upload",
        )
        if uploaded is not None:
            text = _decode_upload(uploaded)
            st.session_state["primer_sets_text"] = text or ""
            st.session_state["primer_sets_filename"] = uploaded.name
        elif not st.session_state.get("primer_sets_text"):
            st.session_state["primer_sets_text"] = ""
    else:
        st.caption(
            "FASTA with one record per primer, headed `<set_name>_<type>`. This is "
            "converted to the engine's CSV input for you. For measured Tp or "
            "per-row conditions, upload a table instead."
        )
        if "primer_paste" not in st.session_state:
            st.session_state["primer_paste"] = st.session_state.get("primer_sets_text", "")
        pasted = st.text_area(
            "Primer sets",
            height=240,
            key="primer_paste",
            placeholder=">set1_F3\nACGT...\n>set1_F2\nACGT...",
        )
        st.session_state["primer_sets_text"] = (pasted or "").strip()
        st.session_state["primer_sets_filename"] = "primer_sets.csv"

    check = _layout.primer_set_check()
    if check is not None:
        for message in check.errors:
            st.error(message)
        for message in check.warnings[:10]:
            st.warning(message)
        if check.ok:
            st.success(
                f"{check.n_sets} row(s) read as {check.source_format}, "
                f"{check.n_sets * len(PRIMER_TYPE_ORDER)} primers in total."
            )
            if check.optional_columns:
                st.info(
                    "Optional columns found: "
                    + ", ".join(
                        f"**{role}** (`{column}`)"
                        for role, column in check.optional_columns.items()
                    )
                    + (
                        ". A measured-Tp column means this run also reports agreement "
                        "metrics."
                        if "measured Tp" in check.optional_columns
                        else "."
                    )
                )
            if check.has_window_columns:
                st.info(
                    "The table carries `<type>_target_sequence` columns, so the engine "
                    "uses those windows instead of locating them."
                )
            with st.expander("Primers as the engine will read them"):
                st.code(primer_sets_to_csv(check.sets[:5]), language="text")
                if check.n_sets > 5:
                    st.caption(f"First 5 of {check.n_sets} rows shown.")

    st.divider()
    _layout.render_step_nav(
        back=("target", "Target"),
        forward=("parameters", "Parameters"),
        forward_disabled=check is None or not check.ok,
        key="primer_sets",
    )


# ---------------------------------------------------------------------------
# Step 3: Parameters
# ---------------------------------------------------------------------------


def _estimate(params: dict) -> dict[str, float | int] | None:
    """Rough cost of the configured run, from the input and the parameters.

    Pure arithmetic on the chosen stride and geometries -- not a call into
    ``generation`` -- so the cost is visible before committing to it. The rate
    is the measured L4 figure from ARCHITECTURE.md, so the estimate is an
    extrapolation of one real measurement rather than a guess.
    """
    check = _layout.target_check()
    if check is None or not check.ok:
        return None
    conditions = max(1, len(params["master_mixes"]) * len(params["concentrations"]))
    if _layout.workflow() == "evaluate":
        primer_check = _layout.primer_set_check()
        candidates = primer_check.n_sets if primer_check and primer_check.ok else 0
        # Each input row is expanded to one row per target sequence, and either
        # keeps its own condition or is re-scored under every one in the sweep.
        if params["use_input_conditions"]:
            conditions = 1
    else:
        anchors = max(0, (check.max_len - MIN_TARGET_LENGTH) // params["stride"] + 1)
        candidates = min(anchors * params["n_geometries"], MAX_CANDIDATE_SETS)
    predictions = candidates * check.n_seqs * conditions
    rate = (
        PREDICTIONS_PER_SECOND_GPU
        if resolved_device() == "cuda" or _backend.active_backend() != "local"
        else PREDICTIONS_PER_SECOND_CPU
    )
    return {
        "candidates": candidates,
        "conditions": conditions,
        "predictions": predictions,
        "seconds": predictions / rate,
    }


def _render_estimate(params: dict) -> None:
    estimate = _estimate(params)
    if estimate is None:
        return
    minutes = estimate["seconds"] / 60.0
    columns = st.columns(3)
    columns[0].metric(
        "Candidate sets" if _layout.workflow() == "design" else "Supplied sets",
        f"{estimate['candidates']:,}",
    )
    columns[1].metric("Predictions", f"{estimate['predictions']:,}")
    columns[2].metric(
        "Inference, estimated",
        f"{estimate['seconds']:.0f} s" if minutes < 2 else f"{minutes:.1f} min",
    )
    st.caption(
        "One prediction is a (set, target, condition) triple through all five seeds. "
        "The estimate scales the measured 2,292 predictions/s on one L4; "
        "thermodynamics and reporting add roughly 11% on top."
    )
    if estimate["candidates"] >= MAX_CANDIDATE_SETS:
        st.warning(
            f"This hits the {MAX_CANDIDATE_SETS:,} candidate cap. Generation thins whole "
            "anchors at a uniform stride rather than truncating, so coverage stays even "
            "— but a larger stride is the honest way to ask for less."
        )


def _page_parameters() -> None:
    _layout.render_workflow_progress("parameters")
    label = _layout.WORKFLOW_LABELS[_layout.workflow()]
    st.header(f"{label} — Parameters")
    params = _layout.init_params()

    if st.button("Reset to defaults", key="params_reset"):
        _layout.reset_params()
        st.rerun()

    with st.expander("Reaction conditions", expanded=True):
        st.caption(
            "A LAMP reaction's speed depends on the master mix and the template "
            "concentration, so every candidate is scored under each combination. The "
            "default sweep is the engine's: two mixes x two concentrations."
        )
        if _layout.workflow() == "evaluate":
            check = _layout.primer_set_check()
            has_own = bool(check and set(check.optional_columns) & {"master mix", "concentration"})
            params["use_input_conditions"] = (
                st.radio(
                    "Which conditions?",
                    options=[False, True],
                    index=1 if params["use_input_conditions"] and has_own else 0,
                    format_func=lambda value: (
                        "Each row's own condition, from the input file"
                        if value
                        else "Override every row with the sweep below"
                    ),
                    key="p_use_input_conditions",
                    help=(
                        "The engine's rule: one master mix and one concentration replace "
                        "each row's own condition, two or more make a cross product, and "
                        "neither keeps what the file says."
                    ),
                )
                if has_own
                else False
            )
            if not has_own:
                st.caption(
                    "The input carries no master-mix or concentration column, so the "
                    "sweep below applies to every row."
                )
        left, right = st.columns(2)
        params["master_mixes"] = left.multiselect(
            "Master mixes",
            options=list(MASTER_MIXES),
            default=params["master_mixes"],
            format_func=_layout.master_mix_label,
            key="p_master_mixes",
            help="The model is conditioned on the master mix as a 3-way one-hot.",
        )
        params["concentrations"] = right.multiselect(
            "Template concentrations (copies/reaction)",
            options=list(TRAINED_CONCENTRATIONS),
            default=params["concentrations"],
            format_func=lambda value: f"{value:,.0f}",
            key="p_concentrations",
            help=(
                "Only values present in the training data are offered: the encoder feeds "
                "the model log10(conc) - 3, and anything outside this range is "
                "extrapolation."
            ),
        )

        st.markdown("**Ranking condition**")
        st.caption(
            "A ranking needs one number per set, so one condition is nominated and the "
            "rest are reported alongside it. The engine's default is the first mix at "
            "its highest concentration — \"how fast can this go\" — and the per-condition "
            "table shows how it degrades."
        )
        rank_left, rank_right = st.columns(2)
        mixes = params["master_mixes"] or list(MASTER_MIXES)
        concentrations = params["concentrations"] or list(TRAINED_CONCENTRATIONS)
        params["rank_master_mix"] = rank_left.selectbox(
            "Rank under master mix",
            options=mixes,
            index=mixes.index(params["rank_master_mix"]) if params["rank_master_mix"] in mixes else 0,
            format_func=_layout.master_mix_label,
            key="p_rank_mix",
        )
        params["rank_conc"] = rank_right.selectbox(
            "Rank under concentration",
            options=concentrations,
            index=(
                concentrations.index(params["rank_conc"])
                if params["rank_conc"] in concentrations
                else concentrations.index(max(concentrations))
            ),
            format_func=lambda value: f"{value:,.0f}",
            key="p_rank_conc",
        )

    if _layout.workflow() == "design":
        with st.expander("Candidate generation", expanded=True):
            st.caption(
                "Candidates are laid down at every anchor position along the consensus, "
                "with a small fixed set of spacing/length geometries at each. The whole "
                "input is tiled: there is no conserved-region pre-selection, because "
                "more conserved sites should score better on their own."
            )
            left, right = st.columns(2)
            params["stride"] = left.number_input(
                "Anchor stride (nt)",
                min_value=1,
                max_value=500,
                value=int(params["stride"]),
                step=1,
                key="p_stride",
                help=(
                    "Step between candidate anchor positions. 1 evaluates every position "
                    "(the engine's own default) and costs the most; larger strides trade "
                    "coverage for time."
                ),
            )
            available = n_available_geometries()
            params["n_geometries"] = right.number_input(
                "Geometries per anchor",
                min_value=1,
                max_value=available,
                value=min(int(params["n_geometries"]), available),
                step=1,
                key="p_n_geometries",
                help=(
                    f"How many of the engine's {available} deterministic geometries to lay "
                    "down at each anchor. They are ordered canonical-first, so any number "
                    "includes the canonical (5,0,0,5,0,0,5,0) spacing at length 19-20, "
                    "where the model is best calibrated."
                ),
            )
            labels = geometry_labels()
            if labels:
                chosen = ", ".join(f"`{name}`" for name in labels[: params["n_geometries"]])
                st.caption(f"In use: {chosen}")

        with st.expander("Ranking and output", expanded=True):
            left, right = st.columns(2)
            params["n_sets"] = left.number_input(
                "Primer sets to return",
                min_value=1,
                max_value=500,
                value=int(params["n_sets"]),
                step=1,
                key="p_n_sets",
                help="Returned spread across loci, not simply the top N by score.",
            )
            params["min_separation"] = right.number_input(
                "Minimum anchor separation (nt)",
                min_value=0,
                max_value=10_000,
                value=int(params["min_separation"]),
                step=5,
                key="p_min_separation",
                help=(
                    "Returned sets are kept at least this far apart, so the list is not "
                    "one design repeated. 0 disables it and degrades to a plain top-N."
                ),
            )

    with st.expander("Scoring", expanded=_layout.workflow() == "evaluate"):
        params["tp_threshold_minutes"] = st.number_input(
            "Tp threshold (minutes)",
            min_value=1.0,
            max_value=240.0,
            value=float(params["tp_threshold_minutes"]),
            step=5.0,
            key="p_tp_threshold",
            help=(
                "Design: a set's inclusivity is the fraction of targets predicted to "
                "amplify strictly within this time. Evaluate: the threshold below which "
                "a reaction counts as amplified, for the ROC-AUC. A reporting "
                "threshold either way, never a filter."
            ),
        )
        if _layout.workflow() == "evaluate":
            left, right = st.columns(2)
            params["max_window_mismatches"] = left.number_input(
                "Maximum window mismatches",
                min_value=0,
                max_value=25,
                value=int(params["max_window_mismatches"]),
                step=1,
                key="p_max_window_mismatches",
                help=(
                    "A primer whose best binding site in a target is worse than this is "
                    "dropped rather than scored: past roughly six mismatches on a 15-25 "
                    "nt primer the placement is more likely to be the wrong site than a "
                    "diverged one."
                ),
            )
            params["censor_at"] = right.number_input(
                "Censor non-amplifying rows at (minutes)",
                min_value=1.0,
                max_value=240.0,
                value=float(params["censor_at"]),
                step=5.0,
                key="p_censor_at",
                help=(
                    "Measured Tp assigned to a row that recorded no amplification, for "
                    "the censored metrics. Matches the training data's tp60_censored "
                    "convention, so those figures are comparable to the published ones. "
                    "Only used when the input carries a measured-Tp column."
                ),
            )

    with st.expander("Compute"):
        st.caption(
            f"Backend `{_backend.active_backend()}`, set by `LAMP_DESIGN_BACKEND` rather "
            "than here: which compute a deployment may reach is an operator's decision."
        )
        left, right = st.columns(2)
        params["device"] = left.selectbox(
            "Inference device",
            options=["auto", "cuda", "cpu"],
            index=["auto", "cuda", "cpu"].index(params["device"]),
            key="p_device",
            help=(
                f"`auto` resolves to `{resolved_device()}` here. The GPU is worth about "
                "15x on this pipeline, which is what makes it the primary path rather "
                "than an optimization."
            ),
        )
        params["threads"] = right.number_input(
            "Thermodynamics threads (0 = all cores)",
            min_value=0,
            max_value=64,
            value=int(params["threads"]),
            step=1,
            key="p_threads",
            help=(
                "Threads for ΔG and alignment. Thermodynamics is about 10% of a run, so "
                "this is a small lever next to the device."
            ),
        )

    st.divider()
    st.subheader("Estimated cost")
    _render_estimate(params)

    st.divider()
    back = ("primer_sets", "Primer sets") if _layout.workflow() == "evaluate" else (
        "target",
        "Target",
    )
    _layout.render_step_nav(
        back=back,
        forward=("run", "Run"),
        key="parameters",
        on_forward=_layout.clear_run_state,
    )


# ---------------------------------------------------------------------------
# Step 4: Run
# ---------------------------------------------------------------------------


def _stage_checklist(
    status: dict,
    stages: tuple[tuple[str, str], ...],
    from_log: dict[str, str] | None = None,
) -> str:
    """PrimerCast's rule list, over our pipeline stages.

    A stage is done once the engine's progress callback has reported it; the
    first unreported stage is the one in flight. The message the engine sent is
    kept, because "235,130 (set, target) pairs" is the useful part. ``from_log``
    fills in stages scraped out of the run log, which is how the evaluate
    workflow reports progress -- its CLI entry point owns its own callback.
    """
    reported = {entry["stage"]: entry for entry in status.get("stages") or []}
    logged = from_log or {}
    running = status.get("state") in ("queued", "running")
    lines = []
    current_found = False
    for key, label in stages:
        entry = reported.get(key)
        if entry is not None:
            detail = f" — {entry['message']}" if entry.get("message") else ""
            lines.append(f"&#9989; &nbsp; {label}{detail} _({entry['elapsed_s']:.1f}s)_")
        elif key in logged:
            lines.append(f"&#9989; &nbsp; {label} — {logged[key]}")
        elif running and not current_found:
            current_found = True
            lines.append(f"&#8987; &nbsp; **{label}** ...")
        else:
            lines.append(f"&#9898; &nbsp; {label}")
    return "<br>".join(lines)


def _page_run() -> None:
    _layout.render_workflow_progress("run")
    label = _layout.WORKFLOW_LABELS[_layout.workflow()]
    st.header(f"Run {label}")

    job = st.session_state.get("job")
    poll = _backend.poll_run(job) if job else None
    running = poll is not None and poll["state"] in ("queued", "running")

    if not running:
        _layout.render_step_nav(
            back=("parameters", "Parameters"), forward=None, key="run_nav"
        )

    blockers, cautions = _layout.preflight_checks()
    if blockers:
        st.subheader("Before this can run")
        for message in blockers:
            st.error(message)
    for message in cautions:
        st.warning(message)

    left, right = st.columns(2)
    with left:
        start = st.button(
            f"Run {label}",
            type="primary",
            width="stretch",
            disabled=running or bool(blockers),
            key="run_start",
        )
    with right:
        stop = st.button("Stop", width="stretch", disabled=not running, key="run_stop")

    if stop and job:
        st.session_state["stop_message"] = _backend.stop_run(job)
        st.rerun()
    if st.session_state.get("stop_message"):
        st.info(st.session_state.pop("stop_message"))

    if start:
        _start_run()
        st.rerun()

    if job is None:
        st.caption(
            "Nothing has been submitted yet. The run happens out of process and this "
            "page polls it, so you can leave it and come back."
        )
        return

    status = poll["status"] or {}
    is_design = job.get("workflow", "design") == "design"
    stages = _backend.DESIGN_STAGES if is_design else _backend.EVALUATE_STAGES
    from_log = (
        None
        if is_design
        else _backend.stages_from_log(job["run_id"], [key for key, _ in stages])
    )
    st.divider()
    st.markdown(f"Run `{job['run_id']}` on `{job.get('backend', 'local')}`")
    st.markdown(_stage_checklist(status, stages, from_log), unsafe_allow_html=True)

    started = status.get("started_at") or job.get("submitted_at") or time.time()
    finished = status.get("finished_at")
    elapsed = int((finished or time.time()) - started)
    minutes, seconds = divmod(elapsed, 60)
    st.caption(
        ("Completed in " if finished else "Elapsed: ") + f"{minutes}m {seconds}s"
    )

    log = _backend.read_log(job["run_id"])
    if log:
        with st.expander("Run log", expanded=poll["state"] == "failed"):
            st.code(log, language="text")

    if poll["state"] == "failed":
        st.error(poll["error"] or "The run failed.")
    elif poll["state"] == "cancelled":
        st.warning("The run was stopped. Partial outputs, if any, are in the run directory.")
    elif poll["state"] == "succeeded" and poll["envelope"]:
        st.session_state["envelope"] = poll["envelope"]
        result = poll["envelope"]["result"]
        if result.get("kind") == "evaluate":
            st.success(f"Finished: {result.get('n_scored', 0):,} row(s) scored.")
        elif result.get("sets"):
            st.success(f"Finished: {result['n_sets_returned']} set(s) returned.")
        else:
            st.error("Finished, but no primer sets survived the pipeline.")
        frame = funnel_frame(result)
        if not frame.empty:
            with st.expander("Candidate funnel", expanded=not result.get("sets")):
                st.caption(
                    "Surviving candidates after each stage. Every stage records its own "
                    "count, so an empty result can name the filter that emptied it."
                )
                st.dataframe(frame, hide_index=True)
        st.divider()
        if st.button("View results →", type="primary", width="stretch", key="run_to_results"):
            _layout.navigate("results")
            st.rerun()

    if running:
        # Poll by rerunning rather than holding the script open: the run is a
        # separate process, so this page can be left and returned to, and a
        # seven-minute run does not pin a session for seven minutes.
        time.sleep(POLL_SECONDS)
        st.rerun()


def _start_run() -> None:
    """Build the request for the current workflow and submit it."""
    params = _layout.init_params()
    run_id = _backend.new_run_id()
    common = {
        "run_id": run_id,
        "fasta": st.session_state["fasta_text"],
        "master_mixes": params["master_mixes"],
        "concentrations": params["concentrations"],
        "rank_master_mix": params["rank_master_mix"],
        "rank_conc": params["rank_conc"],
        "tp_threshold_minutes": params["tp_threshold_minutes"],
        "pre_aligned": params["pre_aligned"],
        "device": params["device"],
        "threads": params["threads"],
    }
    if _layout.workflow() == "design":
        request = _backend.build_design_request(
            **common,
            n_sets=params["n_sets"],
            stride=params["stride"],
            n_geometries=params["n_geometries"],
            min_separation=params["min_separation"],
        )
    else:
        check = _layout.primer_set_check()
        # A table goes to the engine verbatim, so its optional columns (measured
        # Tp, per-row conditions, known windows) keep working; pasted FASTA is
        # converted into the CSV `evaluate --in` documents.
        pasted_fasta = check is not None and check.source_format == "fasta"
        request = _backend.build_evaluate_request(
            **common,
            primer_sets_text=(
                primer_sets_to_csv(check.sets)
                if pasted_fasta
                else st.session_state["primer_sets_text"]
            ),
            primer_sets_filename=(
                "primer_sets.csv"
                if pasted_fasta
                else st.session_state.get("primer_sets_filename", "primer_sets.csv")
            ),
            use_input_conditions=params["use_input_conditions"],
            max_window_mismatches=params["max_window_mismatches"],
            censor_at=params["censor_at"],
        )

    _layout.clear_run_state()
    try:
        job = _backend.start_run(request, backend=_backend.active_backend())
    except ModuleNotFoundError:
        st.session_state["run_error"] = (
            "The `modal` extra is not installed, so the Modal backend cannot submit. "
            "Install it or set LAMP_DESIGN_BACKEND=local."
        )
        return
    st.session_state["job"] = job
    _layout.remember_run(job)


# ---------------------------------------------------------------------------
# Step 5: Results
# ---------------------------------------------------------------------------


def _page_results() -> None:
    _layout.render_workflow_progress("results")
    st.header("Results")
    _layout.render_step_nav(back=("run", "Run"), forward=None, key="results_nav")

    runs = _layout.session_runs()
    envelope = st.session_state.get("envelope")
    if runs:
        ids = [run["run_id"] for run in runs]
        current = envelope["run_id"] if envelope else ids[0]
        chosen = st.selectbox(
            "Run",
            options=ids,
            index=ids.index(current) if current in ids else 0,
            key="results_run",
            help="Runs submitted from this browser session.",
        )
        if not envelope or envelope["run_id"] != chosen:
            envelope = _backend.read_envelope(chosen)

    if envelope is None:
        st.info(
            "No finished run to show yet. Submit one on the Run step; this page reads "
            "the envelope the run wrote to disk."
        )
        return
    if not _layout.owns_run(envelope["run_id"]):
        # Nothing scans the runs directory, so this is belt and braces -- but
        # the rule is that a session renders only its own runs.
        st.error("That result does not belong to this session; refusing to display it.")
        return

    st.caption(
        f"Run `{envelope['run_id']}` — persisted to `{_backend.RUNS_DIR / envelope['run_id']}`. "
        "That is local disk, and on Cloud Run it is lost on redeploy or a scale event."
    )
    _results.render(envelope)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PAGES = {
    "home": _page_home,
    "target": _page_target,
    "primer_sets": _page_primer_sets,
    "parameters": _page_parameters,
    "run": _page_run,
    "results": _page_results,
}


def main() -> None:
    _layout.render_sidebar()
    if error := st.session_state.pop("run_error", None):
        st.error(error)
    page = _layout.current_page()
    if page == "primer_sets" and _layout.workflow() != "evaluate":
        page = "parameters"
    _PAGES.get(page, _page_home)()


main()
