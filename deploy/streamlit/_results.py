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
    evaluate_predictions_display,
    evaluate_ranking_frame,
    evaluate_metrics_frame,
    is_evaluate,
    predictions_frame,
    funnel_culprit,
    primers_frame,
    sets_frame,
    target_matrix,
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


# render_funnel and render_run_facts are gone deliberately.
#
# The candidate funnel was a stage-by-stage survivor table, and "What this run
# did" reported candidate counts, (set, target) pairs, predictions/s, batch
# size, the resolved device, the stride and the geometry ids. All of it is
# instrumentation: it answers "is the pipeline behaving" -- my question while
# building it -- and not "which primers should I order", which is the user's.
# A device of `cuda` and a geometry id are not actionable, and showing them
# invited the reasonable question of why they were on screen at all.
#
# The diagnostics still exist in the result envelope and in the CLI's
# .summary.txt and .json, so nothing is lost for debugging; `funnel_culprit`
# is still used above to name the stage when a run returns nothing, which is
# the one moment that information helps whoever is looking at it.


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
        st.error("**No candidate primer sets survived.**")
        culprit = funnel_culprit(result)
        if culprit:
            st.caption(f"The last stage to retain anything was **{culprit}**.")
        return

    timing = envelope.get("timing_seconds") or {}
    total = timing.get("total") or sum(timing.values())
    best = result["sets"][0]
    columns = st.columns(4)
    columns[0].metric("Sets returned", result["n_sets_returned"])
    columns[1].metric("Targets scored", result["panel"]["n_targets"])
    columns[2].metric("Best set coverage", f"{best[INCLUSIVITY_FIELD]:.0%}")
    columns[3].metric("Runtime", f"{total:.0f} s")
    st.caption(
        f"Ranked best first. Rank 1 is `{best['set_id']}`, predicted to amplify "
        f"{best[INCLUSIVITY_FIELD]:.0%} of your sequences."
    )

    sets = sets_frame(result)
    mismatches = target_matrix(result, "primer_target_mismatches")
    primers = primers_frame(result)

    tabs = st.tabs(["Ranked sets", "Mismatches", "Primers"])

    with tabs[0]:
        st.caption(
            "Spread across the target rather than ranked purely by score, so the "
            "list is not one design repeated. `inclusivity` is the fraction of "
            "your sequences the set is predicted to amplify."
        )
        st.dataframe(sets, hide_index=True)
    with tabs[1]:
        st.caption(
            "Primer-target mismatches, summed over the eight primers, per target. The "
            "model takes these as an input rather than filtering on them."
        )
        st.dataframe(mismatches, hide_index=True)
    with tabs[2]:
        st.caption(
            "One row per primer. Strand `-` means the sequence is the reverse complement "
            "of its target window, per the frozen convention."
        )
        st.dataframe(primers, hide_index=True)

    st.subheader("Downloads")
    columns = st.columns(3)
    with columns[0]:
        _download("Ranked sets (CSV)", sets, f"lamp_sets_{run_id}.csv", f"dl-sets-{run_id}")
    with columns[1]:
        _download(
            "Mismatches (CSV)", mismatches, f"lamp_mismatches_{run_id}.csv", f"dl-mm-{run_id}"
        )
    with columns[2]:
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


def render_evaluate(envelope: dict) -> None:
    """An evaluate run: per-row predictions, a set x target matrix, metrics.

    Deliberately not the design layout. ``EvaluationResult`` is rows and
    agreement metrics, not ranked sets against a panel: there is nothing to rank
    and no funnel, and showing one would be inventing structure the run does not
    have.
    """
    run_id = envelope["run_id"]
    result = envelope["result"]
    timing = envelope.get("timing_seconds") or {}
    frame = predictions_frame(result)

    for message in result.get("warnings") or []:
        st.warning(message)
    skipped = {name: count for name, count in (result.get("skipped") or {}).items() if count}
    if skipped:
        st.warning(
            "Dropped rows: "
            + ", ".join(f"{count:,} ({reason.replace('_', ' ')})" for reason, count in skipped.items())
            + ". A primer whose binding site cannot be placed within the mismatch "
            "limit is dropped rather than scored against the wrong site."
        )

    columns = st.columns(4)
    columns[0].metric("Rows scored", f"{result.get('n_scored', 0):,}")
    columns[1].metric(
        "Primer sets",
        f"{frame['primer_set'].nunique():,}" if "primer_set" in frame.columns else "—",
    )
    columns[2].metric(
        "Targets",
        f"{frame['target'].nunique():,}" if "target" in frame.columns else "—",
    )
    columns[3].metric("Runtime", f"{timing.get('total', 0):.0f} s")

    params = result.get("params") or {}
    if params.get("conditions"):
        st.caption(
            "Scored under "
            + ", ".join(f"`{label}`" for label in params["conditions"])
            + (
                ". Those conditions replaced whatever the input file said, so a warning "
                "above about defaulting the master mix or concentration describes the "
                "file, not what was run."
                if len(params["conditions"]) > 1
                else "."
            )
        )

    metrics = evaluate_metrics_frame(result)
    if metrics.empty:
        st.caption(
            "The input carried no measured-Tp column, so these are predictions only. "
            "Add a `tp` column to turn the run into a validation: Spearman, MAE and "
            "ROC-AUC against the measured values are then reported here."
        )
    else:
        st.subheader("Agreement with measured Tp")
        st.caption(
            "`amplified` drops rows that recorded no amplification; `censored` keeps "
            "them at the censoring value, which is the population comparable to the "
            "published figures."
        )
        st.dataframe(metrics, hide_index=True)

    ranking = evaluate_ranking_frame(result)
    detail = evaluate_predictions_display(result)
    tabs = st.tabs(["Ranked sets", "Per target"])
    with tabs[0]:
        st.caption("Your primer sets, best first. Rank 1 is the fastest predicted.")
        st.dataframe(ranking, hide_index=True)
    with tabs[1]:
        st.caption("One row per (primer set, target, condition).")
        st.dataframe(detail, hide_index=True)

    st.subheader("Downloads")
    columns = st.columns(2)
    with columns[0]:
        _download(
            "Ranked sets (CSV)", ranking, f"lamp_eval_ranking_{run_id}.csv", f"dl-rank-{run_id}"
        )
    with columns[1]:
        _download(
            "Per target (CSV)", detail, f"lamp_predictions_{run_id}.csv", f"dl-pred-{run_id}"
        )
    _render_engine_artifacts(run_id)


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
    """Whether `render_envelope` can lay this payload out."""
    result = envelope.get("result") or {}
    return isinstance(result, dict) and "sets" in result and "panel" in result


def render(envelope: dict) -> None:
    """Dispatch on what the run produced, not on which page asked."""
    if is_evaluate(envelope):
        render_evaluate(envelope)
    elif renderable(envelope):
        render_envelope(envelope)
    else:
        render_unrenderable(envelope)
