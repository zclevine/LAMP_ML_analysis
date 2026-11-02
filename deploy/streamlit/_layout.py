"""Navigation, the step indicator, the sidebar, and the preflight gate.

The shape is PrimerCast's: a list of ``(page_key, label)`` steps per workflow, a
horizontal stepper at the top of every workflow page, ``navigate()`` writing
``st.session_state.page``, a sidebar that always shows what has been chosen so
far, and a ``preflight_checks()`` that returns human-readable blockers before a
run is allowed to start.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

import _backend

#: Mirrors ``MAX_CONCENTRATIONS`` in ``deploy/modal/app.py``, which is the
#: enforcement point. Duplicated rather than imported because that module
#: requires ``modal``, which the front end must not need; a test asserts the
#: two stay equal, the same way ``ENVELOPE_SCHEMA`` is kept in step.
_MODAL_MAX_CONCENTRATIONS = 4

from _engine_api import (
    DEFAULT_CENSOR_MINUTES,
    DEFAULT_CONCENTRATIONS,
    DEFAULT_MASTER_MIXES,
    DEFAULT_N_GEOMETRIES,
    DEFAULT_MAX_WINDOW_MISMATCHES,
    DEFAULT_MIN_SEPARATION,
    DEFAULT_N_SETS,
    DEFAULT_STRIDE,
    DEFAULT_TP_THRESHOLD,
    MASTER_MIX_LABELS,
    MAX_BUDGET,
    MAX_SEQUENCES,
    MIN_TARGET_LENGTH,
    n_available_geometries,
    resolved_device,
    tool_status,
    weights_status,
)
from _validation import FastaCheck, MAX_UPLOAD_MB, PrimerSetCheck

WORKFLOW_LABELS = {"design": "Design", "evaluate": "Evaluate"}

WORKFLOW_STEPS_DESIGN: list[tuple[str, str]] = [
    ("target", "Target"),
    ("parameters", "Parameters"),
    ("run", "Run"),
    ("results", "Results"),
]

WORKFLOW_STEPS_EVALUATE: list[tuple[str, str]] = [
    ("target", "Target"),
    ("primer_sets", "Primer Sets"),
    ("parameters", "Parameters"),
    ("run", "Run"),
    ("results", "Results"),
]

#: Parameter defaults, every one read back from the engine. Held in one dict so
#: "Reset to defaults" is a single assignment and so the whole set can be
#: persisted with a run.
DEFAULT_PARAMS: dict[str, Any] = {
    "master_mixes": list(DEFAULT_MASTER_MIXES),
    "concentrations": list(DEFAULT_CONCENTRATIONS),
    "rank_master_mix": DEFAULT_MASTER_MIXES[0],
    "rank_conc": max(DEFAULT_CONCENTRATIONS),
    "n_sets": DEFAULT_N_SETS,
    "stride": DEFAULT_STRIDE,
    # Five per position: the canonical geometry plus four sampled.
    # This was min(4, ...), an arbitrary cap that silently gave one
    # fewer set per anchor than the design calls for.
    "n_geometries": DEFAULT_N_GEOMETRIES,
    "tp_threshold_minutes": DEFAULT_TP_THRESHOLD,
    "min_separation": DEFAULT_MIN_SEPARATION,
    "pre_aligned": False,
    "device": "auto",
    "threads": 0,
    # Evaluate only.
    "use_input_conditions": False,
    "max_window_mismatches": DEFAULT_MAX_WINDOW_MISMATCHES,
    "censor_at": DEFAULT_CENSOR_MINUTES,
}

#: Widget keys bound to the parameter dict. Streamlit's widget state outlives a
#: plain reassignment of `params`, so "Reset to defaults" has to drop these too
#: or the boxes keep showing the old numbers.
_PARAM_WIDGET_PREFIX = "p_"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def current_page() -> str:
    return st.session_state.get("page", "home")


def navigate(page: str) -> None:
    """Set the current page. Paired with ``st.rerun()`` at the call site."""
    st.session_state.page = page


def workflow() -> str:
    return st.session_state.get("workflow", "design")


def workflow_steps() -> list[tuple[str, str]]:
    return WORKFLOW_STEPS_EVALUATE if workflow() == "evaluate" else WORKFLOW_STEPS_DESIGN


def init_params() -> dict[str, Any]:
    """The live parameter dict, seeded from the engine's defaults."""
    if "params" not in st.session_state:
        st.session_state["params"] = dict(DEFAULT_PARAMS)
    return st.session_state["params"]


def reset_params() -> None:
    st.session_state["params"] = dict(DEFAULT_PARAMS)
    for key in [k for k in st.session_state if str(k).startswith(_PARAM_WIDGET_PREFIX)]:
        st.session_state.pop(key, None)


def clear_run_state() -> None:
    """Drop the current run, keeping the inputs. For re-running after a change."""
    for key in ("job", "envelope", "run_error", "stop_message"):
        st.session_state.pop(key, None)


def reset_workflow_state() -> None:
    """Reset every selection. Used by Home, so a new workflow starts clean."""
    for key in (
        "workflow",
        "params",
        "fasta_text",
        "fasta_source",
        "primer_sets_text",
        "primer_sets_source",
        "primer_sets_filename",
        "target_upload",
        "primer_upload",
    ):
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if str(k).startswith(_PARAM_WIDGET_PREFIX)]:
        st.session_state.pop(key, None)
    clear_run_state()


def session_runs() -> list[dict]:
    """Runs this browser session owns -- the only runs the app will render.

    Nothing scans the runs directory, so one session cannot see another's
    sequences or results even though they share a process and a disk.
    """
    return st.session_state.setdefault("session_runs", [])


def remember_run(job: dict) -> None:
    runs = session_runs()
    if not any(r["run_id"] == job["run_id"] for r in runs):
        runs.insert(0, {"run_id": job["run_id"], "workflow": job.get("workflow", "design")})


def owns_run(run_id: str) -> bool:
    return any(r["run_id"] == run_id for r in session_runs())


def target_check() -> FastaCheck | None:
    """Validation of the target FASTA currently in session state."""
    from _validation import validate_fasta

    text = st.session_state.get("fasta_text", "")
    if not text:
        return None
    return validate_fasta(text, pre_aligned=bool(init_params()["pre_aligned"]))


def primer_set_check() -> PrimerSetCheck | None:
    """Validation of the primer sets currently in session state."""
    from _validation import validate_primer_sets

    text = st.session_state.get("primer_sets_text", "")
    if not text:
        return None
    return validate_primer_sets(text)


# ---------------------------------------------------------------------------
# Step indicator
# ---------------------------------------------------------------------------


def render_workflow_progress(current_step_key: str) -> None:
    """A horizontal progress stepper at the top of a workflow page.

    Static markup built from the step list rather than widgets: the steps are a
    position indicator, not navigation, and a row of buttons here would invite
    skipping a step whose output the next one needs. No user-derived string
    reaches ``unsafe_allow_html``.
    """
    steps = workflow_steps()
    current = next((i for i, (key, _) in enumerate(steps) if key == current_step_key), 0)

    rendered = []
    for index, (_, label) in enumerate(steps):
        number = index + 1
        if index < current:
            circle_style = "background:#4CAF50;color:#fff;"
            glyph = "&#10003;"
            label_style = "color:#4CAF50;font-weight:500;"
        elif index == current:
            circle_style = "background:#1976D2;color:#fff;"
            glyph = str(number)
            label_style = "color:#1976D2;font-weight:600;"
        else:
            circle_style = "background:transparent;border:2px solid #ccc;color:#ccc;"
            glyph = str(number)
            label_style = "color:#ccc;"
        circle = (
            "<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"width:28px;height:28px;border-radius:50%;font-size:14px;font-weight:600;"
            f"{circle_style}'>{glyph}</span>"
        )
        rendered.append(
            "<div style='display:flex;flex-direction:column;align-items:center;gap:4px;'>"
            f"{circle}<span style='{label_style}'>{label}</span></div>"
        )

    connector = (
        "<div style='flex:1;height:2px;background:#ddd;margin:0 4px;align-self:center;"
        "margin-bottom:20px;'></div>"
    )
    st.markdown(
        "<div style='display:flex;align-items:flex-start;justify-content:center;"
        f"margin:0.5em 2em 1em 2em;'>{connector.join(rendered)}</div>",
        unsafe_allow_html=True,
    )


def render_step_nav(
    *,
    back: tuple[str, str] | None,
    forward: tuple[str, str] | None,
    forward_disabled: bool = False,
    key: str,
    on_forward: Any = None,
) -> None:
    """The Back / Continue pair at the foot of a wizard page."""
    left, right = st.columns(2)
    with left:
        if back is not None and st.button(
            f"← {back[1]}", width="stretch", key=f"{key}_back"
        ):
            navigate(back[0])
            st.rerun()
    with right:
        if forward is not None and st.button(
            f"{forward[1]} →",
            type="primary",
            width="stretch",
            disabled=forward_disabled,
            key=f"{key}_forward",
        ):
            if on_forward is not None:
                on_forward()
            navigate(forward[0])
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Product title, environment, and a summary of the current selections."""
    with st.sidebar:
        st.title("LAMP design")
        st.caption("ML-guided eight-primer LAMP set design")

        st.divider()

        if st.button("Home", key="sidebar_home", type="tertiary", icon=":material/home:"):
            reset_workflow_state()
            navigate("home")
            st.rerun()

        page = current_page()
        if page != "home":
            st.markdown(f"**Workflow:** {WORKFLOW_LABELS.get(workflow(), '—')}")
            _sidebar_targets()
            if workflow() == "evaluate":
                _sidebar_primer_sets()
            _sidebar_parameters()
            _sidebar_runs()



def _sidebar_targets() -> None:
    check = target_check()
    if check is None:
        st.markdown("**Target:** —")
        return
    if not check.ok:
        st.markdown(f"**Target:** :red[{len(check.errors)} problem(s)]")
        return
    st.markdown(f"**Target:** {check.n_seqs} sequence(s), {check.total_bases:,} nt")
    for name, sequence in check.records:
        with st.expander(name):
            st.markdown(
                f"Length **{len(sequence):,} nt**  \n"
                f"GC **{check.gc_fraction(sequence):.1%}**  \n"
                + (
                    f":orange[below the {MIN_TARGET_LENGTH} nt floor — will be excluded]"
                    if len(sequence) < MIN_TARGET_LENGTH
                    else ""
                )
            )


def _sidebar_primer_sets() -> None:
    check = primer_set_check()
    if check is None:
        st.markdown("**Primer sets:** —")
        return
    if not check.ok:
        st.markdown(f"**Primer sets:** :red[{len(check.errors)} problem(s)]")
        return
    st.markdown(f"**Primer sets:** {check.n_sets} ({check.source_format})")
    with st.expander(f"{check.n_sets} set(s)"):
        st.markdown("  \n".join(f"`{set_id}`" for set_id, _ in check.sets[:50]))


def _sidebar_parameters() -> None:
    params = init_params()
    conditions = len(params["master_mixes"]) * len(params["concentrations"])
    st.markdown(f"**Conditions:** {conditions}")
    with st.expander("Parameters"):
        lines = [
            f"Master mixes: {', '.join(params['master_mixes']) or '—'}",
            "Concentrations: "
            + (", ".join(f"{c:,.0f}" for c in params["concentrations"]) or "—"),
            f"Ranking: {params['rank_master_mix']}@{params['rank_conc']:,.0f}",
            f"Tp threshold: {params['tp_threshold_minutes']:g} min",
        ]
        if workflow() == "design":
            lines[2:2] = [
                f"Stride: {params['stride']} nt",
                f"Geometries: {params['n_geometries']}",
                f"Sets returned: {params['n_sets']}",
                f"Min separation: {params['min_separation']} nt",
            ]
        st.markdown("  \n".join(lines))


def _sidebar_runs() -> None:
    runs = session_runs()
    if not runs:
        return
    with st.expander(f"This session's runs ({len(runs)})"):
        for run in runs[:20]:
            status = _backend.read_status(run["run_id"]) or {}
            st.markdown(f"`{run['run_id']}` — {status.get('state', 'unknown')}")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_checks() -> tuple[list[str], list[str]]:
    """``(blockers, cautions)``: what stops a run, and what merely costs.

    Blockers are the things that would otherwise surface as a traceback halfway
    through a run: no input, no weights, no aligner for ragged input, no ΔG
    backend. Cautions are real but survivable -- a CPU-only host, or a
    parameter combination that implies a long wait.
    """
    blockers: list[str] = []
    cautions: list[str] = []
    params = init_params()
    tools = tool_status()

    check = target_check()
    if check is None:
        blockers.append("No target sequences provided. Go back to the Target step.")
    elif not check.ok:
        blockers.extend(check.errors)

    if workflow() == "evaluate":
        if not _backend.evaluate_available():
            blockers.append(
                "`lamp-design evaluate` is not available in this build, so there is "
                "nothing to run. The Design workflow is unaffected."
            )
        primer_check = primer_set_check()
        if primer_check is None:
            blockers.append("No primer sets provided. Go back to the Primer Sets step.")
        elif not primer_check.ok:
            blockers.extend(primer_check.errors)

    # Weights and ViennaRNA are LOCAL capabilities, so they only block a LOCAL
    # run. With a remote backend the design happens in the Modal container,
    # which has torch and ViennaRNA; this container deliberately has neither
    # (deploy/modal/portal.py, perf/Dockerfile.thin drop them because compute
    # is elsewhere). Checking them unconditionally blocked every run on the
    # deployed portal with "No module named 'torch'" and "Neither the
    # in-process ViennaRNA bindings nor the RNAduplex binary is available" --
    # on a portal whose actual compute path was fine.
    if _backend.active_backend() == "local":
        weights_ok, weights_message = weights_status()
        if not weights_ok:
            blockers.append(weights_message)
        if not tools["RNA bindings"] and not tools["RNAduplex"]:
            blockers.append(
                "Neither the in-process ViennaRNA bindings nor the RNAduplex binary "
                "is available, so ΔG -- a frozen model input -- cannot be computed."
            )

    if not params["master_mixes"]:
        blockers.append("Select at least one master mix on the Parameters step.")
    if not params["concentrations"]:
        blockers.append("Select at least one target concentration on the Parameters step.")
    if (
        params["master_mixes"]
        and params["rank_master_mix"] not in params["master_mixes"]
    ):
        blockers.append(
            f"The ranking master mix ({params['rank_master_mix']}) is not among the "
            "mixes being scored. Ranking under a condition that was not scored has "
            "no meaning."
        )

    # Ragged input is refused outright, whether or not MAFFT is installed:
    # aligning here would introduce gap columns, and gapped panels are not
    # supported. This used to offer "install MAFFT or tick 'already aligned'",
    # neither of which does anything now.
    if check is not None and check.ok and check.ragged:
        blockers.append(
            f"The {check.n_seqs} input sequences differ in length "
            f"({check.min_len:,}-{check.max_len:,} nt), so they are not an "
            "alignment. Align them and trim to a gapless block, or submit one "
            "sequence."
        )

    if _backend.active_backend() == "local" and resolved_device() == "cpu":
        cautions.append(
            "No usable GPU here, so inference runs on CPU. Measured on this pipeline "
            "that is roughly 11x slower (206 vs 2,292 predictions/s)."
        )
    if _backend.active_backend() != "local" and len(params["master_mixes"]) > 1:
        # Concentrations fan out remotely now; master mixes still do not.
        # `_validate_request` in deploy/modal/app.py takes one `master_mix`
        # and a list of `target_concs`.
        cautions.append(
            f"The deployed Modal functions take one master mix per request, so this "
            f"run will score only {params['rank_master_mix']} remotely, not all "
            f"{len(params['master_mixes'])} selected. Concentrations do fan out. "
            "Use the local backend to sweep mixes."
        )
    if (
        _backend.active_backend() != "local"
        and len(params["concentrations"]) > _MODAL_MAX_CONCENTRATIONS
    ):
        cautions.append(
            f"{len(params['concentrations'])} concentrations exceeds the remote cap of "
            f"{_MODAL_MAX_CONCENTRATIONS} (LAMP_DESIGN_MAX_CONCENTRATIONS). Each one is "
            "another full inference pass over every candidate, and inference is 89% of "
            "a run, so the cap is a spend control rather than a technical limit."
        )
    if (
        check is not None
        and check.ok
        and check.n_seqs == 1
    ):
        cautions.append(
            "One target sequence: every score is a statement about that sequence "
            "alone, and inclusivity across a panel cannot be assessed from it."
        )
    return blockers, cautions


def render_input_limits() -> None:
    """The caps, stated where the user is about to run into them."""
    st.caption(
        f"Limits: up to {MAX_UPLOAD_MB:g} MB per upload, 1-{MAX_SEQUENCES} sequences, "
        f"n_sequences x longest ≤ {MAX_BUDGET:,}, each sequence ≥ {MIN_TARGET_LENGTH} nt, "
        "A/C/G/T only."
    )


def master_mix_label(key: str) -> str:
    return MASTER_MIX_LABELS.get(key, key)
