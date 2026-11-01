"""Ranking, positional diversity selection, and output for a design run.

This is the last stage of the pipeline: it takes the flat list of
:class:`~lamp_design.types.Prediction` rows that inference produced -- one per
``(primer set, target, condition)`` -- aggregates them into
:class:`~lamp_design.types.ScoredSet` objects, picks a positionally diverse
top-N, and writes the output files.

Direction, stated once and then repeated in every column name
-------------------------------------------------------------
``speed_score`` is **a time, not a quality score**: it is the mean predicted
time-to-positive (Tp) across the panel's targets, in **minutes**, at the
ranking condition. **Lower is faster, and faster is better**, so rank 1 has
the *lowest* mean Tp. Everything in this module sorts ascending on it, every
emitted column that carries it is named ``..._tp_minutes...`` with
``lower_is_faster`` spelled out, and :data:`SPEED_SCORE_DIRECTION` is available
for anything downstream that needs to assert the convention. Inverting this is
the single easiest silent bug in the engine.

``inclusivity`` is a plain fraction in ``[0, 1]`` where **higher is better**:
the share of the panel's targets whose predicted Tp is *strictly below*
``tp_threshold`` minutes. It exists because the team wants the top sets to be
"fastest across all target sequences", not "fastest on the consensus" -- a set
that is quick on eight sequences and never amplifies the ninth must be visibly
worse than a slightly slower set that covers all nine. The denominator is
therefore the whole panel, not the subset that happened to get a prediction.

Why diversity selection is not "top 20"
---------------------------------------
Candidate sets at adjacent anchors share most of their primers, so a naive
top-20 by score returns one design twenty times over. :func:`select_diverse`
walks the ranking best-first and skips any set whose anchor is within
``min_separation`` bp of an already-selected anchor -- the same trick as
PrimerCast's ``_select_top_probes(min_distance=...)``. If that leaves fewer
than ``n`` sets, the short list is returned *and a warning is recorded*; the
constraint is never quietly relaxed to pad the list.

Outputs (``write_outputs``)
---------------------------
``<prefix>.sets.csv``        one row per returned set, with all eight primers
``<prefix>.per_target.csv``  long format, one row per (set, target, condition)
``<prefix>.json``            the whole :class:`DesignResult`, for the web UI
``<prefix>.summary.txt``     short human-readable funnel + warnings report
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .types import (
    GLOBAL_PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    Condition,
    DesignResult,
    Prediction,
    PrimerSet,
    ScoredSet,
    TargetPanel,
    ThermoTable,
)

# ---------------------------------------------------------------------------
# Conventions and defaults
# ---------------------------------------------------------------------------

#: A target counts as "included" when its predicted Tp is strictly below this
#: many minutes. 30 min is the team's default call-it-positive cutoff.
DEFAULT_TP_THRESHOLD_MINUTES = 30.0

#: How many positionally distinct sets a design run returns.
DEFAULT_N_SETS = 20

#: Minimum distance, in bp, between the anchors of two returned sets.
DEFAULT_MIN_SEPARATION_BP = 30

#: ``ScoredSet.speed_score`` is a predicted *time*. Lower is faster is better.
SPEED_SCORE_DIRECTION = "lower_is_faster"
SPEED_SCORE_UNITS = "minutes"

#: ``ScoredSet.inclusivity`` is a fraction of the panel. Higher is better.
INCLUSIVITY_DIRECTION = "higher_is_better"

#: Funnel keys this stage contributes to ``DesignResult.funnel``.
FUNNEL_SCORED = "scored_sets"
FUNNEL_RETURNED = "returned_sets"

#: Mismatch counts may arrive as a ``ThermoTable`` or as a plain
#: ``{(set_id, target_name): count}`` mapping -- ``Prediction`` itself carries
#: no mismatch field, so the caller has to hand them over separately.
MismatchSource = Union[ThermoTable, Mapping[Tuple[str, str], float], None]

#: Sort key over ``ScoredSet``; ascending, so "smaller tuple" == "better set".
RankKey = Callable[[ScoredSet], Tuple[Any, ...]]


# ---------------------------------------------------------------------------
# Ranking functions -- named and swappable, because this is still being tuned
# ---------------------------------------------------------------------------


def rank_by_speed_then_inclusivity(scored: ScoredSet) -> Tuple[Any, ...]:
    """Default ranking: fastest mean Tp first, higher inclusivity breaks ties.

    Returned tuples sort **ascending**, so the best set is the smallest key:

    1. ``speed_score`` as-is -- it is mean predicted Tp in minutes, so
       ascending already means "fastest first".
    2. ``-inclusivity`` -- negated so that *more* covered targets sorts
       earlier.
    3. ``set_id`` -- deterministic final tie-break, no RNG anywhere.
    """
    return (scored.speed_score, -scored.inclusivity, scored.primer_set.set_id)


def rank_by_inclusivity_then_speed(scored: ScoredSet) -> Tuple[Any, ...]:
    """Coverage-first alternative: most targets under threshold, then fastest.

    Same ascending convention. Use this when a design has to work on every
    sequence even at the cost of a few minutes.
    """
    return (-scored.inclusivity, scored.speed_score, scored.primer_set.set_id)


#: Name -> ranking function, so a CLI flag can pick one without importing it.
RANKING_FUNCTIONS: Dict[str, RankKey] = {
    "speed_then_inclusivity": rank_by_speed_then_inclusivity,
    "inclusivity_then_speed": rank_by_inclusivity_then_speed,
}

DEFAULT_RANKING_FUNCTION: RankKey = rank_by_speed_then_inclusivity

#: One-line description of each ranking function, for reports.
RANKING_DESCRIPTIONS: Dict[str, str] = {
    "speed_then_inclusivity": (
        "mean predicted Tp (minutes) ascending -- lower is faster; "
        "ties broken by higher inclusivity"
    ),
    "inclusivity_then_speed": (
        "inclusivity descending -- higher is better; "
        "ties broken by lower (faster) mean predicted Tp in minutes"
    ),
}


def ranking_function_name(rank_key: Optional[RankKey]) -> str:
    """Registry name of ``rank_key``, or ``"custom"`` for anything else."""
    fn = DEFAULT_RANKING_FUNCTION if rank_key is None else rank_key
    for name, candidate in RANKING_FUNCTIONS.items():
        if candidate is fn:
            return name
    return getattr(fn, "__name__", "custom")


def rank_sets(
    scored: Iterable[ScoredSet], *, rank_key: Optional[RankKey] = None
) -> List[ScoredSet]:
    """Return ``scored`` best-first. Sorting an already-ranked list is a no-op."""
    fn = DEFAULT_RANKING_FUNCTION if rank_key is None else rank_key
    return sorted(scored, key=fn)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _mismatch_lookup(source: MismatchSource) -> Dict[Tuple[str, str], int]:
    """Normalize a mismatch source to ``{(set_id, target_name): int}``.

    ``ThermoTable.globals_`` stores ``total_mismatches`` as a float (wobble
    weighting can make it fractional), while ``ScoredSet.per_target_mismatches``
    is typed ``Dict[str, int]``; counts are rounded to the nearest integer here
    so the contract holds.
    """
    if source is None:
        return {}
    if isinstance(source, ThermoTable):
        col = GLOBAL_PHYSICS_FEATURES.index("total_mismatches")
        return {
            (set_id, target_name): int(round(float(source.globals_[row, col])))
            for row, (set_id, target_name) in enumerate(
                zip(source.set_ids, source.target_names)
            )
        }
    out: Dict[Tuple[str, str], int] = {}
    for key, value in dict(source).items():
        try:
            out[(str(key[0]), str(key[1]))] = int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return out


def _set_lookup(
    primer_sets: Union[Sequence[PrimerSet], Mapping[str, PrimerSet]]
) -> Dict[str, PrimerSet]:
    if isinstance(primer_sets, Mapping):
        return dict(primer_sets)
    return {ps.set_id: ps for ps in primer_sets}


def _panel_target_names(
    panel: Optional[TargetPanel], predictions: Sequence[Prediction]
) -> List[str]:
    """Ordered, de-duplicated target names that form the inclusivity denominator.

    The panel is authoritative: a target with no prediction for a given set is
    a target that set does not cover, and it has to drag inclusivity down. Only
    when there is no panel do we fall back to whatever the predictions mention.
    """
    names: List[str] = []
    seen = set()
    if panel is not None:
        for target in panel.targets:
            if target.name not in seen:
                seen.add(target.name)
                names.append(target.name)
    if names:
        return names
    for prediction in predictions:
        if prediction.target_name not in seen:
            seen.add(prediction.target_name)
            names.append(prediction.target_name)
    return names


def score_sets(
    predictions: Sequence[Prediction],
    primer_sets: Union[Sequence[PrimerSet], Mapping[str, PrimerSet]],
    panel: Optional[TargetPanel],
    *,
    ranking_condition: Condition,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    mismatches: MismatchSource = None,
    rank_key: Optional[RankKey] = None,
    warnings: Optional[List[str]] = None,
) -> List[ScoredSet]:
    """Aggregate per-(set, target, condition) predictions into ranked sets.

    Args:
        predictions: Flat prediction list across every set, target and
            condition. An empty list yields an empty result, never an error.
        primer_sets: The candidate sets, as a sequence or ``{set_id: set}``.
        panel: Panel the run was scored against. Supplies the authoritative
            target list used as the inclusivity denominator; may be ``None``,
            in which case the targets seen in ``predictions`` are used.
        ranking_condition: The one ``Condition`` whose predictions define
            ``speed_score``, ``inclusivity`` and ``per_target_tp``. Predictions
            at other conditions are still carried in ``ScoredSet.predictions``
            for the detailed report, they just do not drive the ranking.
        tp_threshold: Inclusivity cutoff in minutes. A target counts as
            included when ``predicted_tp_minutes < tp_threshold`` (strict, so a
            prediction sitting exactly on the threshold does *not* count).
        mismatches: Optional ``ThermoTable`` or ``{(set_id, target): count}``
            mapping for ``per_target_mismatches``. ``Prediction`` carries no
            mismatch field, so without this the per-target mismatch columns are
            left blank rather than filled with a misleading 0.
        rank_key: Ranking function; defaults to
            :func:`rank_by_speed_then_inclusivity`.
        warnings: Optional list that human-readable notes are appended to
            (unknown ``set_id``s, sets missing predictions at the ranking
            condition, sets missing some panel targets).

    Returns:
        ``ScoredSet`` objects ranked best-first by ``rank_key``.
    """
    sink: List[str] = warnings if warnings is not None else []
    sets_by_id = _set_lookup(primer_sets)
    target_names = _panel_target_names(panel, predictions)
    mismatch_by_pair = _mismatch_lookup(mismatches)
    target_rank = {name: i for i, name in enumerate(target_names)}

    by_set: Dict[str, List[Prediction]] = {}
    unknown_ids: List[str] = []
    for prediction in predictions:
        if prediction.set_id not in sets_by_id:
            if prediction.set_id not in unknown_ids:
                unknown_ids.append(prediction.set_id)
            continue
        by_set.setdefault(prediction.set_id, []).append(prediction)

    if unknown_ids:
        shown = ", ".join(sorted(unknown_ids)[:5])
        sink.append(
            f"{len(unknown_ids)} prediction set_id(s) had no matching primer set "
            f"and were dropped before ranking: {shown}"
            + (" ..." if len(unknown_ids) > 5 else "")
        )

    scored: List[ScoredSet] = []
    no_ranking_condition: List[str] = []
    incomplete: List[str] = []

    # Iterate the candidate sets, not the predictions, so the result order does
    # not depend on how inference happened to batch its output.
    for set_id, primer_set in sets_by_id.items():
        set_predictions = by_set.get(set_id)
        if not set_predictions:
            continue

        at_condition = {
            p.target_name: p
            for p in set_predictions
            if p.condition == ranking_condition
        }
        if not at_condition:
            no_ranking_condition.append(set_id)
            continue

        per_target_tp: Dict[str, float] = {}
        for name in target_names:
            if name in at_condition:
                per_target_tp[name] = float(at_condition[name].tp_minutes)
        # Targets predicted but absent from the panel still belong in the
        # detail view; they just cannot widen the denominator.
        for name, prediction in at_condition.items():
            per_target_tp.setdefault(name, float(prediction.tp_minutes))

        per_target_mismatches: Dict[str, int] = {}
        for name in per_target_tp:
            count = mismatch_by_pair.get((set_id, name))
            if count is not None:
                per_target_mismatches[name] = count

        # speed_score: mean predicted Tp in MINUTES across the targets that
        # have a prediction. Lower is faster is better.
        speed_score = sum(per_target_tp.values()) / len(per_target_tp)

        # inclusivity: share of the PANEL under the threshold, so a missing
        # target counts against the set instead of being averaged away.
        denominator = len(target_names) or len(per_target_tp)
        n_included = sum(1 for tp in per_target_tp.values() if tp < tp_threshold)
        inclusivity = n_included / denominator if denominator else 0.0

        missing = [name for name in target_names if name not in per_target_tp]
        if missing:
            incomplete.append(set_id)

        ordered_predictions = sorted(
            set_predictions,
            key=lambda p: (
                target_rank.get(p.target_name, len(target_rank)),
                p.target_name,
                p.condition.master_mix,
                p.condition.target_conc,
            ),
        )

        scored.append(
            ScoredSet(
                primer_set=primer_set,
                speed_score=speed_score,
                inclusivity=inclusivity,
                per_target_tp=per_target_tp,
                per_target_mismatches=per_target_mismatches,
                predictions=ordered_predictions,
            )
        )

    if no_ranking_condition:
        sink.append(
            f"{len(no_ranking_condition)} candidate set(s) had no prediction at the "
            f"ranking condition {ranking_condition} and could not be ranked."
        )
    if incomplete:
        shown = ", ".join(incomplete[:5])
        sink.append(
            f"{len(incomplete)} candidate set(s) are missing predictions for some "
            f"panel targets; the missing targets count against inclusivity: {shown}"
            + (" ..." if len(incomplete) > 5 else "")
        )

    return rank_sets(scored, rank_key=rank_key)


# ---------------------------------------------------------------------------
# Positional diversity selection
# ---------------------------------------------------------------------------


def _anchor(scored: ScoredSet) -> int:
    return int(scored.primer_set.anchor)


def select_diverse(
    scored: Sequence[ScoredSet],
    *,
    n: int = DEFAULT_N_SETS,
    min_separation: int = DEFAULT_MIN_SEPARATION_BP,
    rank_key: Optional[RankKey] = None,
) -> List[ScoredSet]:
    """Pick up to ``n`` sets that are at least ``min_separation`` bp apart.

    Greedy and best-first: walk the ranking from the top, keep a set when its
    anchor is at least ``min_separation`` bp from every anchor already kept,
    otherwise skip it as a positional near-duplicate of a better-scoring set.
    This is PrimerCast's ``_select_top_probes`` ``min_distance`` rule applied to
    primer-set anchors.

    When candidates are tightly packed the result is deliberately **shorter
    than** ``n``: the separation constraint is never relaxed to pad the list.
    Pair this with :func:`diversity_warnings` (or just use
    :func:`build_design_result`) so the short list is explained rather than
    silent.

    ``min_separation <= 0`` disables the constraint and degrades to a plain
    top-``n``.
    """
    if n <= 0:
        return []
    ordered = rank_sets(scored, rank_key=rank_key)
    selected: List[ScoredSet] = []
    for candidate in ordered:
        position = _anchor(candidate)
        if any(
            abs(position - _anchor(kept)) < min_separation for kept in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= n:
            break
    return selected


def diversity_warnings(
    scored: Sequence[ScoredSet],
    selected: Sequence[ScoredSet],
    *,
    n: int = DEFAULT_N_SETS,
    min_separation: int = DEFAULT_MIN_SEPARATION_BP,
) -> List[str]:
    """Warnings explaining a selection that came back shorter than requested."""
    if not scored:
        return [
            "No primer sets were scored, so no designs were returned. Check the "
            "funnel for the stage that emptied the candidate list."
        ]
    if len(selected) >= n:
        return []
    rejected = len(scored) - len(selected)
    return [
        f"Diversity selection returned {len(selected)} of {n} requested set(s): "
        f"{rejected} of {len(scored)} scored candidate(s) were within "
        f"{min_separation} bp of a better-scoring anchor and were dropped as "
        f"positional near-duplicates. The separation constraint was kept; lower "
        f"min_separation (currently {min_separation} bp) or widen the target "
        f"region to get more designs."
    ]


def build_design_result(
    predictions: Sequence[Prediction],
    primer_sets: Union[Sequence[PrimerSet], Mapping[str, PrimerSet]],
    panel: TargetPanel,
    *,
    ranking_condition: Condition,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    n: int = DEFAULT_N_SETS,
    min_separation: int = DEFAULT_MIN_SEPARATION_BP,
    mismatches: MismatchSource = None,
    rank_key: Optional[RankKey] = None,
    funnel: Optional[Mapping[str, int]] = None,
    excluded_targets: Optional[Mapping[str, str]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> DesignResult:
    """Score, rank, diversity-select and package a run into a ``DesignResult``.

    The integration point for the CLI: it is :func:`score_sets` plus
    :func:`select_diverse` plus the bookkeeping those two cannot do on their own
    (appending ``scored_sets`` / ``returned_sets`` to the funnel and recording
    the short-list warning). Earlier stages pass their own funnel counts and
    excluded targets straight through.
    """
    collected: List[str] = list(warnings or [])
    scored = score_sets(
        predictions,
        primer_sets,
        panel,
        ranking_condition=ranking_condition,
        tp_threshold=tp_threshold,
        mismatches=mismatches,
        rank_key=rank_key,
        warnings=collected,
    )
    selected = select_diverse(
        scored, n=n, min_separation=min_separation, rank_key=rank_key
    )
    collected.extend(
        diversity_warnings(scored, selected, n=n, min_separation=min_separation)
    )

    merged_funnel: Dict[str, int] = {k: int(v) for k, v in (funnel or {}).items()}
    merged_funnel[FUNNEL_SCORED] = len(scored)
    merged_funnel[FUNNEL_RETURNED] = len(selected)

    return DesignResult(
        scored_sets=selected,
        panel=panel,
        funnel=merged_funnel,
        excluded_targets=dict(excluded_targets or {}),
        warnings=collected,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

#: Leading columns of ``<prefix>.sets.csv``. The eight per-primer blocks and
#: the mismatch totals are appended by :func:`sets_csv_columns`.
SETS_LEAD_COLUMNS: Tuple[str, ...] = (
    "rank",
    "set_id",
    "anchor",
    "geometry_id",
    "span_start",
    "span_end",
    "mean_predicted_tp_minutes_lower_is_faster",
    "inclusivity_fraction_targets_below_tp_threshold",
    "tp_threshold_minutes",
    "n_targets_scored",
    "n_panel_targets",
    "slowest_target_name",
    "slowest_predicted_tp_minutes",
    "ranking_condition",
)

#: Trailing columns of ``<prefix>.sets.csv``, after the per-primer blocks.
SETS_TAIL_COLUMNS: Tuple[str, ...] = (
    "total_primer_target_mismatches_all_targets",
    "mean_primer_target_mismatches_per_target",
)

#: Columns of ``<prefix>.per_target.csv`` -- the long-format matrix a user
#: inspects to find *which* sequence is the problem.
PER_TARGET_COLUMNS: Tuple[str, ...] = (
    "rank",
    "set_id",
    "anchor",
    "target_name",
    "master_mix",
    "target_conc",
    "predicted_tp_minutes",
    "predicted_tp_std_minutes",
    "primer_target_mismatches",
    "is_ranking_condition",
    "tp_threshold_minutes",
    "predicted_tp_below_threshold",
)


def sets_csv_columns() -> List[str]:
    """Full ordered column list of ``<prefix>.sets.csv``.

    Per-primer blocks follow the frozen ``PRIMER_TYPE_ORDER`` layout, so the
    columns read left to right along the target: ``F3_sequence, F3_start,
    F3_end, F3_length, F2_sequence, ...``.
    """
    columns = list(SETS_LEAD_COLUMNS)
    for primer_type in PRIMER_TYPE_ORDER:
        columns += [
            f"{primer_type}_sequence",
            f"{primer_type}_start",
            f"{primer_type}_end",
            f"{primer_type}_length",
        ]
    columns += list(SETS_TAIL_COLUMNS)
    return columns


def _num(value: Any, digits: int = 3) -> str:
    """Fixed-point text for a CSV cell; blank for missing/NaN."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def _int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _conc(value: Any) -> str:
    """Concentration as written by ``Condition.__str__`` (``%g``).

    ``Condition.target_conc`` carries no unit in ``types.py``; the column is a
    passthrough of whatever the caller conditioned the model on.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:g}"


#: Below this many minutes a predicted Tp is reported as a band, not a number.
#:
#: The model is trained on a target floored at -100 (``regress_tp_cutoff``), so
#: its output legitimately goes negative -- a fitted amplification curve whose
#: threshold crossing extrapolates back past t=0. Printing "-5.46 min" as a
#: *time* invites the reader to call it impossible, when what it means is
#: "faster than anything in the training data".
#:
#: 5 minutes is where the data puts the boundary, not a round number: in the
#: curated corpus the 0.1st percentile of measured Tp is 1.76 min, the 5th is
#: 2.70, and the 25th is **5.06**. So "< 5 min" is almost exactly "in the
#: fastest quartile of everything we ever measured", and the model cannot
#: resolve within it in a way a user should act on.
FAST_TP_FLOOR_MINUTES: float = 5.0


def format_tp(minutes: Optional[float]) -> str:
    """A predicted Tp as a user-facing string.

    Banded below :data:`FAST_TP_FLOOR_MINUTES`, because the ordering inside
    that band is what the *rank* is for. The raw value stays in the CSV under
    its own column, so nothing is lost and a ranking stays auditable.
    """
    if minutes is None:
        return "n/a"
    if minutes < FAST_TP_FLOOR_MINUTES:
        return f"<{FAST_TP_FLOOR_MINUTES:g} min"
    return f"{minutes:.1f} min"


def _slowest(per_target_tp: Mapping[str, float]) -> Tuple[Optional[str], Optional[float]]:
    """The worst (highest) predicted Tp and its target -- the problem sequence."""
    if not per_target_tp:
        return None, None
    name = max(per_target_tp, key=lambda k: (per_target_tp[k], k))
    return name, per_target_tp[name]


def ranking_condition_from_params(result: DesignResult) -> Optional[Condition]:
    """The ranking condition as recorded in ``result.params``, if it is there.

    ``commands.design.design`` writes ``params["ranking_condition"]`` as the
    ``Condition.__str__`` label (``"BST@10000"``). Parsing it back is exact for
    anything this engine writes, and returning ``None`` on anything unexpected
    leaves :func:`infer_ranking_condition` to do the recovery instead of
    guessing.
    """
    label = result.params.get("ranking_condition")
    if not isinstance(label, str) or "@" not in label:
        return None
    master_mix, _, conc = label.rpartition("@")
    try:
        return Condition(master_mix, float(conc))
    except ValueError:
        return None


def infer_ranking_condition(result: DesignResult) -> Optional[Condition]:
    """Recover the ranking condition by matching predictions to ``per_target_tp``.

    The fallback for results that carry no ``params`` -- one built directly by
    :func:`build_design_result` rather than by ``commands.design.design``. Finds
    the condition whose per-target predictions reproduce ``per_target_tp``, and
    returns ``None`` when there is nothing to match against.
    """
    for scored in result.scored_sets:
        by_condition: Dict[Condition, Dict[str, float]] = {}
        for prediction in scored.predictions:
            by_condition.setdefault(prediction.condition, {})[
                prediction.target_name
            ] = float(prediction.tp_minutes)
        for condition, tps in by_condition.items():
            if set(tps) != set(scored.per_target_tp):
                continue
            if all(
                math.isclose(tps[name], scored.per_target_tp[name], rel_tol=1e-9, abs_tol=1e-9)
                for name in tps
            ):
                return condition
    return None


def _set_row(
    scored: ScoredSet,
    rank: int,
    *,
    tp_threshold: float,
    ranking_condition: Optional[Condition],
    n_panel_targets: Optional[int],
) -> Dict[str, str]:
    primer_set = scored.primer_set
    slowest_name, slowest_tp = _slowest(scored.per_target_tp)
    mismatches = list(scored.per_target_mismatches.values())

    row: Dict[str, str] = {
        "rank": str(rank),
        "set_id": primer_set.set_id,
        "anchor": _int(primer_set.anchor),
        "geometry_id": primer_set.geometry_id,
        "mean_predicted_tp_minutes_lower_is_faster": _num(scored.speed_score, 3),
        "inclusivity_fraction_targets_below_tp_threshold": _num(scored.inclusivity, 4),
        "tp_threshold_minutes": _num(tp_threshold, 1),
        "n_targets_scored": str(len(scored.per_target_tp)),
        "n_panel_targets": _int(n_panel_targets),
        "slowest_target_name": slowest_name or "",
        "slowest_predicted_tp_minutes": _num(slowest_tp, 3),
        "ranking_condition": str(ranking_condition) if ranking_condition else "",
    }

    try:
        span_start, span_end = primer_set.span
    except (KeyError, ValueError):
        span_start, span_end = None, None
    row["span_start"] = _int(span_start)
    row["span_end"] = _int(span_end)

    for primer_type in PRIMER_TYPE_ORDER:
        primer = primer_set.primers.get(primer_type)
        row[f"{primer_type}_sequence"] = primer.sequence if primer else ""
        row[f"{primer_type}_start"] = _int(primer.start) if primer else ""
        row[f"{primer_type}_end"] = _int(primer.end) if primer else ""
        row[f"{primer_type}_length"] = _int(primer.length) if primer else ""

    if mismatches:
        row["total_primer_target_mismatches_all_targets"] = str(int(sum(mismatches)))
        row["mean_primer_target_mismatches_per_target"] = _num(
            sum(mismatches) / len(mismatches), 2
        )
    else:
        # No mismatch source was supplied: leave blank rather than claim 0,
        # which would read as "perfect match against every target".
        row["total_primer_target_mismatches_all_targets"] = ""
        row["mean_primer_target_mismatches_per_target"] = ""

    return row


def _per_target_rows(
    scored: ScoredSet,
    rank: int,
    *,
    tp_threshold: float,
    ranking_condition: Optional[Condition],
) -> List[Dict[str, str]]:
    primer_set = scored.primer_set
    predictions = list(scored.predictions)

    if not predictions:
        # A ScoredSet built without its prediction grid still has the ranking
        # condition detail; emit that rather than an empty matrix.
        for name, tp in scored.per_target_tp.items():
            predictions.append(
                Prediction(
                    set_id=primer_set.set_id,
                    target_name=name,
                    condition=ranking_condition or Condition("", float("nan")),
                    tp_minutes=tp,
                    tp_std=float("nan"),
                )
            )

    rows: List[Dict[str, str]] = []
    for prediction in predictions:
        tp = float(prediction.tp_minutes)
        is_ranking = (
            ranking_condition is not None and prediction.condition == ranking_condition
        )
        rows.append(
            {
                "rank": str(rank),
                "set_id": prediction.set_id,
                "anchor": _int(primer_set.anchor),
                "target_name": prediction.target_name,
                "master_mix": prediction.condition.master_mix,
                "target_conc": _conc(prediction.condition.target_conc),
                "predicted_tp_minutes": _num(tp, 3),
                "predicted_tp_std_minutes": _num(prediction.tp_std, 3),
                "primer_target_mismatches": _int(
                    scored.per_target_mismatches.get(prediction.target_name)
                ),
                "is_ranking_condition": "True" if is_ranking else "False",
                "tp_threshold_minutes": _num(tp_threshold, 1),
                "predicted_tp_below_threshold": "True" if tp < tp_threshold else "False",
            }
        )
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def result_to_dict(
    result: DesignResult,
    *,
    ranking_condition: Optional[Condition] = None,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    ranking_function: Optional[RankKey] = None,
) -> Dict[str, Any]:
    """The whole ``DesignResult`` as JSON-ready plain data.

    Carries the funnel, the excluded targets and the warnings alongside the
    sets, plus the metric definitions and their directions, so a front end can
    label a column without hard-coding this module's conventions.
    """
    name = ranking_function_name(ranking_function)
    panel = result.panel
    sets: List[Dict[str, Any]] = []

    for rank, scored in enumerate(result.scored_sets, start=1):
        primer_set = scored.primer_set
        slowest_name, slowest_tp = _slowest(scored.per_target_tp)
        try:
            span_start, span_end = primer_set.span
        except (KeyError, ValueError):
            span_start, span_end = None, None

        primers = []
        for primer_type in PRIMER_TYPE_ORDER:
            primer = primer_set.primers.get(primer_type)
            if primer is None:
                continue
            primers.append(
                {
                    "primer_type": primer.primer_type,
                    "sequence": primer.sequence,
                    "start": int(primer.start),
                    "end": int(primer.end),
                    "length": int(primer.length),
                    "strand": "+" if primer.is_forward_strand else "-",
                }
            )

        mismatches = list(scored.per_target_mismatches.values())
        sets.append(
            {
                "rank": rank,
                "set_id": primer_set.set_id,
                "anchor": int(primer_set.anchor),
                "geometry_id": primer_set.geometry_id,
                "span_start": span_start,
                "span_end": span_end,
                "mean_predicted_tp_minutes_lower_is_faster": scored.speed_score,
                "inclusivity_fraction_targets_below_tp_threshold": scored.inclusivity,
                "n_targets_scored": len(scored.per_target_tp),
                "slowest_target_name": slowest_name,
                "slowest_predicted_tp_minutes": slowest_tp,
                "total_primer_target_mismatches_all_targets": (
                    int(sum(mismatches)) if mismatches else None
                ),
                "primers": primers,
                "per_target": [
                    {
                        "target_name": name,
                        "predicted_tp_minutes": tp,
                        "primer_target_mismatches": scored.per_target_mismatches.get(name),
                        "predicted_tp_below_threshold": tp < tp_threshold,
                    }
                    for name, tp in scored.per_target_tp.items()
                ],
                "predictions": [
                    {
                        "target_name": p.target_name,
                        "master_mix": p.condition.master_mix,
                        "target_conc": p.condition.target_conc,
                        "predicted_tp_minutes": p.tp_minutes,
                        "predicted_tp_std_minutes": p.tp_std,
                        "is_ranking_condition": (
                            ranking_condition is not None
                            and p.condition == ranking_condition
                        ),
                    }
                    for p in scored.predictions
                ],
            }
        )

    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ranking": {
            "function": name,
            "description": RANKING_DESCRIPTIONS.get(name, ""),
            "speed_score": {
                "field": "mean_predicted_tp_minutes_lower_is_faster",
                "units": SPEED_SCORE_UNITS,
                "direction": SPEED_SCORE_DIRECTION,
                "description": (
                    "mean predicted time-to-positive across the panel's targets "
                    "at the ranking condition"
                ),
            },
            "inclusivity": {
                "field": "inclusivity_fraction_targets_below_tp_threshold",
                "units": "fraction_of_panel_targets",
                "direction": INCLUSIVITY_DIRECTION,
                "description": (
                    "fraction of panel targets with predicted Tp strictly below "
                    "tp_threshold_minutes"
                ),
            },
            "tp_threshold_minutes": tp_threshold,
            "ranking_condition": (
                {
                    "master_mix": ranking_condition.master_mix,
                    "target_conc": ranking_condition.target_conc,
                    "label": str(ranking_condition),
                }
                if ranking_condition is not None
                else None
            ),
        },
        "panel": {
            "n_targets": panel.n_targets if panel is not None else 0,
            "target_names": [t.name for t in panel.targets] if panel is not None else [],
            "consensus_length": len(panel.consensus) if panel is not None else 0,
            "budget": panel.budget if panel is not None else 0,
        },
        "funnel": {k: int(v) for k, v in result.funnel.items()},
        # The run's own settings and stage timings, as recorded by
        # `commands.design.design`. Emitted verbatim so a result file says what
        # produced it -- stride, geometries, conditions, device -- without the
        # reader having to reconstruct any of it.
        "params": dict(result.params),
        "excluded_targets": dict(result.excluded_targets),
        "warnings": list(result.warnings),
        "n_sets_returned": len(result.scored_sets),
        "sets": sets,
    }


def format_summary(
    result: DesignResult,
    *,
    ranking_condition: Optional[Condition] = None,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    ranking_function: Optional[RankKey] = None,
    paths: Optional[Mapping[str, Path]] = None,
    max_sets_shown: int = 10,
) -> str:
    """Short human-readable run report: funnel, counts, warnings."""
    name = ranking_function_name(ranking_function)
    panel = result.panel
    n_panel = panel.n_targets if panel is not None else 0
    entered = result.funnel.get(FUNNEL_SCORED, len(result.scored_sets))

    lines: List[str] = []
    lines.append("LAMP primer design summary")
    lines.append("==========================")
    lines.append("")
    lines.append(f"Ranking condition : {ranking_condition if ranking_condition else '(unknown)'}")
    lines.append(
        "Output            : a RANKING. Rank 1 is the fastest predicted set; "
        "the rank is the result, not the Tp value."
    )
    lines.append(
        f"Predicted Tp      : shown as '<{FAST_TP_FLOOR_MINUTES:g} min' below "
        f"{FAST_TP_FLOOR_MINUTES:g} minutes -- that band is the fastest "
        f"quartile of all measured data (p25 = 5.06 min) and the model does "
        f"not resolve inside it. Raw scores are in the sets CSV."
    )
    lines.append(
        f"Inclusivity       : fraction of the {n_panel} panel target(s) with "
        f"predicted Tp < {tp_threshold:g} min (HIGHER is better)"
    )
    lines.append(f"Ranking function  : {name} -- {RANKING_DESCRIPTIONS.get(name, 'custom')}")
    lines.append("")

    lines.append("Panel")
    lines.append("-----")
    lines.append(f"  targets scored   : {n_panel}")
    lines.append(f"  targets excluded : {len(result.excluded_targets)}")
    for target_name, reason in result.excluded_targets.items():
        lines.append(f"    - {target_name}: {reason}")
    lines.append("")

    lines.append("Candidate funnel")
    lines.append("----------------")
    if result.funnel:
        width = max(len(k) for k in result.funnel)
        for stage, count in result.funnel.items():
            lines.append(f"  {stage.ljust(width)}  {count:>9,}")
    else:
        lines.append("  (no stage recorded a candidate count)")
    lines.append("")
    lines.append(f"Candidate sets entering ranking : {entered:,}")
    lines.append(f"Primer sets returned            : {len(result.scored_sets):,}")
    lines.append("")

    if result.scored_sets:
        lines.append("Ranked sets (rank: set_id @ anchor -- mean Tp / inclusivity / slowest target)")
        lines.append("-" * 78)
        for rank, scored in enumerate(result.scored_sets[:max_sets_shown], start=1):
            slowest_name, slowest_tp = _slowest(scored.per_target_tp)
            slowest = (
                f"{slowest_name} at {format_tp(slowest_tp)}" if slowest_name else "n/a"
            )
            lines.append(
                f"  {rank:>3}: {scored.primer_set.set_id} @ {scored.primer_set.anchor} "
                f"-- {format_tp(scored.speed_score)} mean Tp / "
                f"{scored.inclusivity:.0%} inclusivity / slowest {slowest}"
            )
        remaining = len(result.scored_sets) - max_sets_shown
        if remaining > 0:
            lines.append(f"  ... and {remaining} more (see the sets CSV)")
    else:
        lines.append("No primer sets were returned.")
    lines.append("")

    heading = f"Warnings ({len(result.warnings)})"
    lines.append(heading)
    lines.append("-" * len(heading))
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  none")
    lines.append("")

    if paths:
        lines.append("Files")
        lines.append("-----")
        width = max(len(k) for k in paths)
        for key, path in paths.items():
            lines.append(f"  {key.ljust(width)}  {path}")
        lines.append("")

    return "\n".join(lines)


def write_outputs(
    result: DesignResult,
    out_prefix: Union[str, Path],
    *,
    ranking_condition: Optional[Condition] = None,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    ranking_function: Optional[RankKey] = None,
) -> Dict[str, Path]:
    """Write every output file for a run and return the paths by role.

    Files written, all prefixed with ``out_prefix``:

    ``.sets.csv``
        One row per returned set: rank, identity, anchor/span, the aggregate
        scores, then the eight primers in frozen ``PRIMER_TYPE_ORDER`` with
        start/end/length, then the mismatch totals.
    ``.per_target.csv``
        Long format, one row per (set, target, condition), with predicted Tp,
        its ensemble std, and mismatches. This is the matrix to scan for the
        one sequence that is slow.
    ``.json``
        The full result including ``funnel``, ``excluded_targets`` and
        ``warnings``, plus the metric definitions and directions.
    ``.summary.txt``
        The short human-readable report.

    ``ranking_condition`` is only needed for labelling. When omitted it is
    taken from ``result.params`` (written by ``commands.design.design``), and
    failing that recovered from the predictions with
    :func:`infer_ranking_condition`.

    Returns:
        ``{"sets_csv": ..., "per_target_csv": ..., "json": ..., "summary_txt": ...}``
    """
    prefix = Path(out_prefix)
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)

    if ranking_condition is None:
        ranking_condition = ranking_condition_from_params(result)
    if ranking_condition is None:
        ranking_condition = infer_ranking_condition(result)
    if ranking_condition is not None and "tp_threshold_minutes" in result.params:
        # Keep the label consistent with the run that produced the result,
        # unless the caller explicitly overrode it.
        if tp_threshold == DEFAULT_TP_THRESHOLD_MINUTES:
            tp_threshold = float(result.params["tp_threshold_minutes"])

    paths: Dict[str, Path] = {
        "sets_csv": Path(f"{prefix}.sets.csv"),
        "per_target_csv": Path(f"{prefix}.per_target.csv"),
        "json": Path(f"{prefix}.json"),
        "summary_txt": Path(f"{prefix}.summary.txt"),
    }

    n_panel = result.panel.n_targets if result.panel is not None else None

    _write_csv(
        paths["sets_csv"],
        sets_csv_columns(),
        (
            _set_row(
                scored,
                rank,
                tp_threshold=tp_threshold,
                ranking_condition=ranking_condition,
                n_panel_targets=n_panel,
            )
            for rank, scored in enumerate(result.scored_sets, start=1)
        ),
    )

    per_target_rows: List[Dict[str, str]] = []
    for rank, scored in enumerate(result.scored_sets, start=1):
        per_target_rows.extend(
            _per_target_rows(
                scored,
                rank,
                tp_threshold=tp_threshold,
                ranking_condition=ranking_condition,
            )
        )
    _write_csv(paths["per_target_csv"], PER_TARGET_COLUMNS, per_target_rows)

    payload = result_to_dict(
        result,
        ranking_condition=ranking_condition,
        tp_threshold=tp_threshold,
        ranking_function=ranking_function,
    )
    paths["json"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    paths["summary_txt"].write_text(
        format_summary(
            result,
            ranking_condition=ranking_condition,
            tp_threshold=tp_threshold,
            ranking_function=ranking_function,
            paths=paths,
        ),
        encoding="utf-8",
    )

    return paths


__all__ = [
    "DEFAULT_TP_THRESHOLD_MINUTES",
    "DEFAULT_N_SETS",
    "DEFAULT_MIN_SEPARATION_BP",
    "SPEED_SCORE_DIRECTION",
    "SPEED_SCORE_UNITS",
    "INCLUSIVITY_DIRECTION",
    "FUNNEL_SCORED",
    "FUNNEL_RETURNED",
    "RANKING_FUNCTIONS",
    "RANKING_DESCRIPTIONS",
    "DEFAULT_RANKING_FUNCTION",
    "SETS_LEAD_COLUMNS",
    "SETS_TAIL_COLUMNS",
    "PER_TARGET_COLUMNS",
    "rank_by_speed_then_inclusivity",
    "rank_by_inclusivity_then_speed",
    "ranking_function_name",
    "rank_sets",
    "score_sets",
    "select_diverse",
    "diversity_warnings",
    "build_design_result",
    "sets_csv_columns",
    "infer_ranking_condition",
    "ranking_condition_from_params",
    "result_to_dict",
    "format_summary",
    "write_outputs",
]
