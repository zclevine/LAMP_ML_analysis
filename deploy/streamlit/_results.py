"""Rendering one run's envelope: metrics, funnel, result tabs, downloads.

Everything here reads the payload that ``report.result_to_dict`` produced, via
the frame builders in ``_tables``. The split is deliberate: the frames can be
built and checked without a Streamlit session, and this module only lays them
out.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import _backend
from _tables import (
    INCLUSIVITY_FIELD,
    SPEED_FIELD,
    conditions_frame,
    funnel_culprit,
    funnel_frame,
    primers_frame,
    run_facts,
    sets_frame,
    target_matrix,
    timings_frame,
)


def _download(label: str, frame: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
        width="stretch",
    )


def render_funnel(result: dict, *, emphasise: bool) -> None:
    """Explain which stage emptied the result, instead of an empty table.

    Every stage records its surviving-candidate count into
    ``DesignResult.funnel`` precisely so that "no candidates" can name the
    filter that did it. The first stage whose count is zero is the answer.
    """
    frame = funnel_frame(result)
    if frame.empty:
        if emphasise:
            st.error(
                "No candidate sets were returned, and the run reported no funnel, so "
                "the stage that emptied it cannot be identified. Every stage is "
                "supposed to record its surviving-candidate count in DesignResult.funnel."
            )
        return

    if not emphasise:
        with st.expander(f"Candidate funnel ({len(frame)} stages)"):
            st.dataframe(frame, hide_index=True)
        return

    culprit = funnel_culprit(result)
    if culprit:
        st.error(
            f"**No candidate sets survived.** The funnel empties at **{culprit}** — "
            "that is the stage to loosen."
        )
    else:
        st.error(
            "**No candidate sets were returned**, although no funnel stage reports "
            "zero. Check the warnings below."
        )
    st.dataframe(frame, hide_index=True)


def render_run_facts(envelope: dict) -> None:
    """What the run did, from ``DesignResult.params``, plus per-stage timings."""
    facts = run_facts(envelope)
    with st.expander("What this run did"):
        columns = st.columns(4)
        columns[0].metric("Candidate sets", f"{facts['n_candidates'] or 0:,}")
        columns[1].metric("(set, target) pairs", f"{facts['n_pairs'] or 0:,}")
        columns[2].metric("Predictions", f"{facts['n_predictions'] or 0:,}")
        columns[3].metric("Device", str(facts["device"] or "—"))

        if facts["predictions_per_second"]:
            st.caption(
                f"{facts['predictions_per_second']:,.0f} predictions/s during inference "
                f"at batch size {facts['batch_size']}. One prediction is a "
                "(set, target, condition) triple through all five seeds."
            )
        st.markdown(
            f"**Stride** {facts['stride']} nt  \n"
            f"**Geometries** {', '.join(facts['geometries'] or [])}  \n"
            f"**Conditions** {', '.join(facts['conditions'] or [])}  \n"
            f"**Ranked under** {facts['ranking_condition']}"
        )
        timings = timings_frame(envelope)
        if not timings.empty:
            st.dataframe(
                timings,
                hide_index=True,
                column_config={
                    "share": st.column_config.ProgressColumn(
                        "share of run", min_value=0.0, max_value=1.0, format="%.0f%%"
                    )
                },
            )


def render_envelope(envelope: dict) -> None:
    """The whole result: headline metrics, funnel, tables, downloads."""
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
            )

    if not result.get("sets"):
        render_funnel(result, emphasise=True)
        render_run_facts(envelope)
        return

    timing = envelope.get("timing_seconds") or {}
    total = timing.get("total") or sum(timing.values())
    best = result["sets"][0]
    columns = st.columns(4)
    columns[0].metric("Sets returned", result["n_sets_returned"])
    columns[1].metric("Targets scored", result["panel"]["n_targets"])
    columns[2].metric("Best mean Tp", f"{best[SPEED_FIELD]:.1f} min")
    columns[3].metric("Runtime", f"{total:.0f} s")
    st.caption(
        f"Best set `{best['set_id']}` covers {best[INCLUSIVITY_FIELD]:.0%} of the panel "
        f"under {threshold:g} minutes. Lower Tp is faster; higher inclusivity is better."
    )

    render_funnel(result, emphasise=False)
    render_run_facts(envelope)

    sets = sets_frame(result)
    tp = target_matrix(result, "predicted_tp_minutes")
    mismatches = target_matrix(result, "primer_target_mismatches")
    primers = primers_frame(result)
    conditions = conditions_frame(result)

    tab_labels = ["Ranked sets", "Tp by target", "Mismatches", "Primers"]
    if not conditions.empty and len(conditions.columns) > 3:
        tab_labels.append("Conditions")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.caption(
            "Spread across loci rather than ranked purely by score, so the list is not "
            "one design repeated. `mean_tp_min` is predicted minutes to amplification — "
            "**lower is faster**. `inclusivity` is the fraction of targets predicted to "
            f"amplify in under {threshold:g} minutes."
        )
        st.dataframe(sets, hide_index=True)
    with tabs[1]:
        st.caption("Every returned set scored against every input sequence. Lower is faster.")
        st.dataframe(tp, hide_index=True)
    with tabs[2]:
        st.caption(
            "Primer-target mismatches, summed over the eight primers, per target. The "
            "model takes these as an input rather than filtering on them."
        )
        st.dataframe(mismatches, hide_index=True)
    with tabs[3]:
        st.caption(
            "One row per primer. Strand `-` means the sequence is the reverse complement "
            "of its target window, per the frozen convention."
        )
        st.dataframe(primers, hide_index=True)
    if len(tabs) > 4:
        with tabs[4]:
            st.caption(
                "Mean predicted Tp per set under each scored condition. The ranking uses "
                "one condition; this is how a set degrades across the sweep."
            )
            st.dataframe(conditions, hide_index=True)

    st.subheader("Downloads")
    columns = st.columns(4)
    with columns[0]:
        _download("Ranked sets (CSV)", sets, f"lamp_sets_{run_id}.csv", f"dl-sets-{run_id}")
    with columns[1]:
        _download("Tp matrix (CSV)", tp, f"lamp_tp_{run_id}.csv", f"dl-tp-{run_id}")
    with columns[2]:
        _download(
            "Mismatches (CSV)", mismatches, f"lamp_mismatches_{run_id}.csv", f"dl-mm-{run_id}"
        )
    with columns[3]:
        _download("Primers (CSV)", primers, f"lamp_primers_{run_id}.csv", f"dl-primers-{run_id}")

    _render_engine_artifacts(run_id)


def _render_engine_artifacts(run_id: str) -> None:
    """Offer the engine's own output files, byte for byte as the CLI writes them."""
    out_dir = _backend.RUNS_DIR / run_id
    artifacts = sorted(
        path
        for path in out_dir.glob("result.*")
        if path.is_file() and path.suffix in (".csv", ".json", ".txt")
    )
    if not artifacts:
        return
    with st.expander(f"Engine output files ({len(artifacts)})"):
        st.caption(
            "Written by `report.write_outputs`, identical to a `lamp-design design` run's "
            "artifacts. The CSVs above are the same data reshaped for the browser."
        )
        columns = st.columns(min(4, len(artifacts)))
        for index, path in enumerate(artifacts):
            with columns[index % len(columns)]:
                st.download_button(
                    path.name,
                    data=path.read_bytes(),
                    file_name=f"{run_id}_{path.name}",
                    key=f"dl-art-{run_id}-{path.name}",
                    width="stretch",
                )


def render_unrenderable(envelope: dict) -> None:
    """Fallback for a payload that is not ``result_to_dict``-shaped.

    Only reachable from the Evaluate workflow, whose engine entry point is not
    written yet: better to show what came back than to claim it is wrong.
    """
    st.warning(
        "This result does not carry the `sets` / `panel` structure "
        "`report.result_to_dict` produces, so the tables above cannot be built from "
        "it. The raw payload is below, and the run directory holds whatever the engine "
        "wrote."
    )
    st.json(envelope.get("result") or {}, expanded=False)
    _render_engine_artifacts(envelope["run_id"])


def renderable(envelope: dict) -> bool:
    result = envelope.get("result") or {}
    return isinstance(result, dict) and "sets" in result and "panel" in result
