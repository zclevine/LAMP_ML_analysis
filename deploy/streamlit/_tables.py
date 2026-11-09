"""Result payload -> dataframes. Pandas only, no streamlit.

Reads the shape ``report.result_to_dict`` produces: ``sets`` with ``primers`` as
a list in frozen ``PRIMER_TYPE_ORDER``, ``per_target`` and ``predictions`` as
lists, and the column names ``report.py`` uses, which spell out their direction
("..._lower_is_faster"). Kept apart from the rendering so the tables can be
built and checked without a Streamlit session.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

SPEED_FIELD = "mean_predicted_tp_minutes_lower_is_faster"
INCLUSIVITY_FIELD = "inclusivity_fraction_targets_below_tp_threshold"


def _format_speed(minutes: Any) -> str:
    """Predicted speed, banded. One definition, shared with the CLI report."""
    from lamp_design.report import format_tp

    return format_tp(None if minutes is None else float(minutes))


def sets_frame(result: dict) -> pd.DataFrame:
    """One row per returned set: rank, placement, scores, the eight primers."""
    rows = []
    for entry in result["sets"]:
        # A BANDED speed figure, and no geometry id.
        #
        # Speed came back by request: a rank alone cannot be compared between
        # two different runs, and an absolute measure can. It is banded rather
        # than raw -- anything under FAST_TP_FLOOR_MINUTES reads "<3 min" --
        # because the model does not resolve differences inside the fastest
        # band and the raw value can be negative (training floors the label at
        # -100; measured labels reach -4.611).
        #
        # Geometry: the id names an internal spacing/length layout -- since
        # cbfd454 most are drawn from a distribution, so the id is an index into
        # nothing a user can act on. The primer sequences and the span are the
        # information; the layout that produced them is not.
        row: dict[str, Any] = {
            "rank": entry["rank"],
            "set_id": entry["set_id"],
            "anchor": entry["anchor"],
            "span_start": entry["span_start"],
            "span_end": entry["span_end"],
            "speed": _format_speed(entry.get(SPEED_FIELD)),
            "inclusivity": entry[INCLUSIVITY_FIELD],
            "total_mismatches": entry.get("total_primer_target_mismatches_all_targets"),
        }
        for primer in entry["primers"]:
            row[primer["primer_type"]] = primer["sequence"]
        rows.append(row)
    return pd.DataFrame(rows)


def target_matrix(result: dict, field_name: str) -> pd.DataFrame:
    """Rows = ranked sets, columns = the panel's targets."""
    targets = result["panel"]["target_names"]
    rows = []
    for entry in result["sets"]:
        by_target = {p["target_name"]: p.get(field_name) for p in entry["per_target"]}
        row: dict[str, Any] = {"rank": entry["rank"], "set_id": entry["set_id"]}
        row.update({name: by_target.get(name) for name in targets})
        rows.append(row)
    return pd.DataFrame(rows)


def primers_frame(result: dict) -> pd.DataFrame:
    """One row per primer, for ordering and for pasting into a plate map."""
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


def condition_labels(result: dict) -> list[str]:
    """Condition labels in the order the engine scored them."""
    seen: list[str] = []
    for entry in result["sets"]:
        for prediction in entry.get("predictions") or []:
            label = _condition_label(prediction)
            if label not in seen:
                seen.append(label)
    return seen


def _condition_label(prediction: dict) -> str:
    return f"{prediction['master_mix']}@{prediction['target_conc']:,.0f}"


def conditions_frame(result: dict) -> pd.DataFrame:
    """Rows = ranked sets, columns = conditions, values = mean predicted Tp.

    The ranking collapses every condition into one number per set; this is how
    that number degrades across the sweep, which is the whole reason several
    conditions are scored.
    """
    labels = condition_labels(result)
    if not labels:
        return pd.DataFrame()
    rows = []
    for entry in result["sets"]:
        totals: dict[str, list[float]] = {label: [] for label in labels}
        for prediction in entry.get("predictions") or []:
            totals[_condition_label(prediction)].append(prediction["predicted_tp_minutes"])
        row: dict[str, Any] = {"rank": entry["rank"], "set_id": entry["set_id"]}
        for label in labels:
            values = totals[label]
            row[label] = sum(values) / len(values) if values else None
        rows.append(row)
    return pd.DataFrame(rows)


def funnel_frame(result: dict) -> pd.DataFrame:
    """The candidate funnel, in pipeline order."""
    funnel = result.get("funnel") or {}
    return pd.DataFrame(
        [{"stage": stage, "candidates remaining": count} for stage, count in funnel.items()]
    )


def funnel_culprit(result: dict) -> str | None:
    """The first funnel stage whose count is zero, if any.

    Every stage records its surviving-candidate count into
    ``DesignResult.funnel`` precisely so that "no candidates" can name the
    filter that did it.
    """
    funnel = result.get("funnel") or {}
    return next((stage for stage, count in funnel.items() if count == 0), None)


def timings_frame(envelope: dict) -> pd.DataFrame:
    """Per-stage timings with each stage's share of the total.

    Shares are computed over the engine's own ``timings_s`` keys only. The
    envelope also carries coarser totals (``design_s``, ``total``) which are
    sums of those, so including them would double-count.
    """
    timings = envelope.get("timing_seconds") or {}
    stage_keys = [
        key
        for key in ("prepare_s", "generation_s", "thermo_s", "model_load_s", "inference_s",
                    "report_s", "evaluate_s")
        if key in timings
    ]
    total = sum(float(timings[key]) for key in stage_keys) or 1.0
    rows = [
        {
            "stage": key[:-2] if key.endswith("_s") else key,
            "seconds": round(float(timings[key]), 2),
            "share": float(timings[key]) / total,
        }
        for key in stage_keys
    ]
    other = [key for key in timings if key not in stage_keys and key != "total"]
    for key in other:
        rows.append({"stage": f"{key} (total)", "seconds": round(float(timings[key]), 2),
                     "share": None})
    return pd.DataFrame(rows)


#: Columns in ``<prefix>.predictions.csv`` that are numbers, so the frame sorts
#: and formats as numbers rather than as the strings the CSV holds.
_EVALUATE_NUMERIC = (
    "target_conc",
    "predicted_tp_minutes",
    "predicted_tp_std",
    "measured_tp_minutes",
    "residual_minutes",
    "total_mismatches",
)


def is_evaluate(envelope: dict) -> bool:
    return (envelope.get("result") or {}).get("kind") == "evaluate"


def predictions_frame(result: dict) -> pd.DataFrame:
    """``<prefix>.predictions.csv`` as a frame, with numeric columns coerced."""
    frame = pd.DataFrame(result.get("predictions") or [])
    for column in _EVALUATE_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def evaluate_metrics_frame(result: dict) -> pd.DataFrame:
    """Agreement with measured Tp, one row per metric, one column per population.

    ``amplified`` drops rows with no measured Tp; ``censored`` keeps them at the
    censoring value. Both are reported because the censored figures are the ones
    comparable to the paper, while the amplified ones cannot be inflated by the
    model merely learning which reactions fail.
    """
    metrics = result.get("metrics") or {}
    if not metrics:
        return pd.DataFrame()
    populations = list(metrics)
    names: list[str] = []
    for population in populations:
        for name in metrics[population]:
            if name not in names:
                names.append(name)
    return pd.DataFrame(
        [
            {"metric": name, **{p: metrics[p].get(name) for p in populations}}
            for name in names
        ]
    )


def evaluate_ranking_frame(result: dict) -> pd.DataFrame:
    """The supplied primer sets, ranked fastest-first. No Tp column.

    Ranking is the product, in Evaluate exactly as in Design: the model's
    minute figure orders the sets and is not shown.
    """
    frame = predictions_frame(result)
    if frame.empty or not {"primer_set", "predicted_tp_minutes"} <= set(frame.columns):
        return pd.DataFrame()
    ordered = (
        frame.groupby("primer_set")["predicted_tp_minutes"]
        .mean()
        .sort_values()
        .reset_index()
        .drop(columns=["predicted_tp_minutes"])
    )
    ordered.insert(0, "rank", range(1, len(ordered) + 1))
    return ordered


def evaluate_predictions_display(result: dict) -> pd.DataFrame:
    """``predictions_frame`` with the predicted-Tp columns dropped."""
    frame = predictions_frame(result)
    drop = [c for c in ("predicted_tp_minutes", "predicted_tp_std") if c in frame.columns]
    return frame.drop(columns=drop)


def evaluate_matrix(result: dict) -> pd.DataFrame:
    """Rows = primer sets, columns = targets, values = mean predicted Tp.

    The same orientation as the design workflow's Tp matrix, so a set that was
    designed here and evaluated later reads the same way.
    """
    frame = predictions_frame(result)
    if frame.empty or not {"primer_set", "target", "predicted_tp_minutes"} <= set(frame.columns):
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index="primer_set", columns="target", values="predicted_tp_minutes", aggfunc="mean"
    )
    return pivot.reset_index()


_GEOMETRY_ID = re.compile(r"geometry_id='([^']+)'")


def geometry_names(params: dict) -> list[str]:
    """Readable geometry names out of ``DesignResult.params['geometries']``.

    ``design()`` now records ``geometry_id``, so this is usually a passthrough.
    It used to record ``getattr(g, "name", str(g))`` against a ``Geometry`` that
    has no ``name``, which wrote the whole dataclass repr -- so the id is still
    pulled back out of that form, to keep a ``result.json`` written before the
    fix readable.
    """
    names = []
    for entry in params.get("geometries") or []:
        match = _GEOMETRY_ID.search(str(entry))
        names.append(match.group(1) if match else str(entry))
    return names


def run_facts(envelope: dict) -> dict[str, Any]:
    """What the run actually did, from ``DesignResult.params``.

    ``design()`` records stride, geometries, conditions, device, batch size and
    the candidate / pair / prediction counts into ``params`` so an output says
    what produced it. Surfacing it here is the difference between "20 sets" and
    "20 sets out of 940,520 predictions on cuda".
    """
    params = (envelope.get("result") or {}).get("params") or {}
    timings = envelope.get("timing_seconds") or {}
    return {
        "device": params.get("device"),
        "batch_size": params.get("batch_size"),
        "stride": params.get("stride"),
        "geometries": geometry_names(params),
        "conditions": params.get("conditions"),
        "ranking_condition": params.get("ranking_condition"),
        "n_candidates": params.get("n_candidates"),
        "n_pairs": params.get("n_pairs"),
        "n_predictions": params.get("n_predictions"),
        "total_seconds": timings.get("total"),
        "predictions_per_second": (
            params["n_predictions"] / timings["inference_s"]
            if params.get("n_predictions") and timings.get("inference_s")
            else None
        ),
    }
