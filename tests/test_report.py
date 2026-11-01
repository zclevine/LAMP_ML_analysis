"""Tests for the ranking / diversity / output stage.

Every fixture here is synthetic: ``Prediction`` and ``PrimerSet`` objects are
built by hand so these tests never touch the model, the thermo stage, or any
other module's code. ``numpy`` is only used for the one ``ThermoTable``
fixture, because that type's contract requires arrays.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pytest

from lamp_design import report
from lamp_design.report import (
    FAST_TP_FLOOR_MINUTES,
    DEFAULT_MIN_SEPARATION_BP,
    DEFAULT_N_SETS,
    DEFAULT_TP_THRESHOLD_MINUTES,
    FUNNEL_RETURNED,
    FUNNEL_SCORED,
    RANKING_FUNCTIONS,
    build_design_result,
    diversity_warnings,
    format_summary,
    infer_ranking_condition,
    rank_by_inclusivity_then_speed,
    rank_by_speed_then_inclusivity,
    result_to_dict,
    score_sets,
    select_diverse,
    sets_csv_columns,
    write_outputs,
)
from lamp_design.types import (
    GLOBAL_PHYSICS_FEATURES,
    PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    Condition,
    DesignResult,
    Prediction,
    Primer,
    PrimerSet,
    ScoredSet,
    Target,
    TargetPanel,
    ThermoTable,
)

RANKING = Condition("BST", 1000.0)
OTHER = Condition("SS", 100.0)

_BASES = "ACGT"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_seq(seed: int, length: int = 20) -> str:
    """A deterministic DNA-ish sequence, distinct for every ``seed``.

    The seed is written out in base 4 first, then padded with a cycling tail --
    a plain rotation would repeat every 4 bases and give eight primers only
    four distinct sequences.
    """
    digits: List[str] = []
    n = abs(int(seed)) + 1
    while n:
        digits.append(_BASES[n % 4])
        n //= 4
    head = "".join(reversed(digits))[:length]
    tail = "".join(_BASES[(seed + i) % 4] for i in range(max(0, length - len(head))))
    return (head + tail)[:length]


def make_primer_set(
    set_id: str,
    anchor: int,
    *,
    geometry_id: str = "geo-5_0_0_5_0_0_5_0@20",
    primer_len: int = 20,
    spacing: int = 25,
) -> PrimerSet:
    """An eight-primer set laid out along the template from ``anchor``."""
    primers: Dict[str, Primer] = {}
    position = anchor
    for i, primer_type in enumerate(PRIMER_TYPE_ORDER):
        primers[primer_type] = Primer(
            primer_type=primer_type,
            sequence=make_seq(anchor + i, primer_len),
            start=position,
            end=position + primer_len,
        )
        position += spacing
    return PrimerSet(
        set_id=set_id, primers=primers, anchor=anchor, geometry_id=geometry_id
    )


def make_panel(n_targets: int = 3, *, consensus_len: int = 600) -> TargetPanel:
    targets = [
        Target(name=f"target_{i + 1}", sequence=make_seq(i, consensus_len))
        for i in range(n_targets)
    ]
    return TargetPanel(targets=targets, consensus=make_seq(0, consensus_len))


def make_predictions(
    set_id: str,
    tps: Dict[str, float],
    *,
    condition: Condition = RANKING,
    tp_std: float = 1.5,
) -> List[Prediction]:
    return [
        Prediction(
            set_id=set_id,
            target_name=name,
            condition=condition,
            tp_minutes=tp,
            tp_std=tp_std,
        )
        for name, tp in tps.items()
    ]


def even_predictions(
    set_id: str,
    tp: float,
    panel: TargetPanel,
    *,
    conditions: Sequence[Condition] = (RANKING,),
) -> List[Prediction]:
    """Same predicted Tp for every target, at each given condition."""
    out: List[Prediction] = []
    for condition in conditions:
        out += make_predictions(
            set_id,
            {t.name: tp for t in panel.targets},
            condition=condition,
        )
    return out


def make_cluster(
    anchors: Iterable[int], tps: Iterable[float], panel: TargetPanel
) -> tuple[List[PrimerSet], List[Prediction]]:
    """Primer sets at the given anchors, each with a flat predicted Tp."""
    sets: List[PrimerSet] = []
    predictions: List[Prediction] = []
    for anchor, tp in zip(anchors, tps):
        set_id = f"set_{anchor:05d}"
        sets.append(make_primer_set(set_id, anchor))
        predictions += even_predictions(set_id, tp, panel)
    return sets, predictions


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


# ---------------------------------------------------------------------------
# Scoring aggregation
# ---------------------------------------------------------------------------


def test_speed_score_is_mean_tp_in_minutes_at_the_ranking_condition():
    panel = make_panel(3)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions(
        "set_a", {"target_1": 10.0, "target_2": 20.0, "target_3": 30.0}
    )
    # A second condition with far slower predictions must not leak into the score.
    predictions += make_predictions(
        "set_a",
        {"target_1": 90.0, "target_2": 95.0, "target_3": 99.0},
        condition=OTHER,
    )

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING
    )

    assert len(scored) == 1
    assert scored[0].speed_score == pytest.approx(20.0)
    assert scored[0].per_target_tp == {
        "target_1": 10.0,
        "target_2": 20.0,
        "target_3": 30.0,
    }


def test_every_condition_is_kept_in_the_prediction_grid():
    panel = make_panel(3)
    primer_set = make_primer_set("set_a", 100)
    predictions = even_predictions(
        "set_a", 18.0, panel, conditions=(RANKING, OTHER)
    )

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING
    )

    # 3 targets x 2 conditions stay available for the detailed report, even
    # though only the ranking condition drives speed_score.
    assert len(scored[0].predictions) == 6
    assert {p.condition for p in scored[0].predictions} == {RANKING, OTHER}
    assert len(scored[0].per_target_tp) == 3


def test_per_target_detail_names_which_sequence_is_the_problem():
    panel = make_panel(3)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions(
        "set_a", {"target_1": 12.0, "target_2": 13.0, "target_3": 58.0}
    )

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING
    )[0]

    assert scored.per_target_tp["target_3"] == 58.0
    assert max(scored.per_target_tp, key=scored.per_target_tp.get) == "target_3"
    # Two of three targets amplify in time, so inclusivity is 2/3, not 1.0.
    assert scored.inclusivity == pytest.approx(2 / 3)


def test_scoring_handles_many_sets_targets_and_conditions():
    panel = make_panel(4)
    sets = [make_primer_set(f"set_{i}", 100 * (i + 1)) for i in range(3)]
    predictions: List[Prediction] = []
    for i, primer_set in enumerate(sets):
        predictions += even_predictions(
            primer_set.set_id, 15.0 + i, panel, conditions=(RANKING, OTHER)
        )

    scored = score_sets(predictions, sets, panel, ranking_condition=RANKING)

    assert [s.primer_set.set_id for s in scored] == ["set_0", "set_1", "set_2"]
    assert [s.speed_score for s in scored] == pytest.approx([15.0, 16.0, 17.0])
    assert all(len(s.predictions) == 8 for s in scored)


def test_mismatches_come_from_a_mapping_when_supplied():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions("set_a", {"target_1": 11.0, "target_2": 12.0})

    scored = score_sets(
        predictions,
        [primer_set],
        panel,
        ranking_condition=RANKING,
        mismatches={("set_a", "target_1"): 0, ("set_a", "target_2"): 3},
    )[0]

    assert scored.per_target_mismatches == {"target_1": 0, "target_2": 3}


def test_mismatches_can_come_from_a_thermo_table():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions("set_a", {"target_1": 11.0, "target_2": 12.0})

    n_pairs = 2
    table = ThermoTable(
        set_ids=["set_a", "set_a"],
        target_names=["target_1", "target_2"],
        values=np.zeros((n_pairs, len(PRIMER_TYPE_ORDER), len(PHYSICS_FEATURES))),
        globals_=np.array([[1.0, -8.0], [4.0, -7.5]], dtype=float),
        # Unused by the mismatch path, but ThermoTable requires it to be in
        # lockstep with `values` -- the encoder indexes the two together.
        target_windows=np.full((n_pairs, len(PRIMER_TYPE_ORDER)), "A" * 20, dtype=object),
    )
    assert GLOBAL_PHYSICS_FEATURES[0] == "total_mismatches"

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING, mismatches=table
    )[0]

    assert scored.per_target_mismatches == {"target_1": 1, "target_2": 4}


def test_missing_mismatch_source_leaves_counts_absent_not_zero():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions("set_a", {"target_1": 11.0, "target_2": 12.0})

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING
    )[0]

    # 0 mismatches means "perfect match"; an unknown count must not say that.
    assert scored.per_target_mismatches == {}


def test_predictions_for_unknown_set_ids_are_dropped_with_a_warning():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = even_predictions("set_a", 12.0, panel)
    predictions += even_predictions("set_ghost", 9.0, panel)

    warnings: List[str] = []
    scored = score_sets(
        predictions,
        [primer_set],
        panel,
        ranking_condition=RANKING,
        warnings=warnings,
    )

    assert [s.primer_set.set_id for s in scored] == ["set_a"]
    assert any("set_ghost" in w for w in warnings)


def test_set_without_a_prediction_at_the_ranking_condition_warns():
    panel = make_panel(2)
    sets = [make_primer_set("set_a", 100), make_primer_set("set_b", 300)]
    predictions = even_predictions("set_a", 12.0, panel)
    predictions += even_predictions("set_b", 9.0, panel, conditions=(OTHER,))

    warnings: List[str] = []
    scored = score_sets(
        predictions, sets, panel, ranking_condition=RANKING, warnings=warnings
    )

    assert [s.primer_set.set_id for s in scored] == ["set_a"]
    assert any("ranking condition" in w for w in warnings)


def test_inclusivity_denominator_is_the_whole_panel():
    panel = make_panel(4)
    primer_set = make_primer_set("set_a", 100)
    # Only two of four panel targets got a prediction.
    predictions = make_predictions("set_a", {"target_1": 10.0, "target_2": 11.0})

    warnings: List[str] = []
    scored = score_sets(
        predictions,
        [primer_set],
        panel,
        ranking_condition=RANKING,
        warnings=warnings,
    )[0]

    # Fastest "on the targets it covers" must not read as fully inclusive.
    assert scored.speed_score == pytest.approx(10.5)
    assert scored.inclusivity == pytest.approx(0.5)
    assert any("missing predictions" in w for w in warnings)


# ---------------------------------------------------------------------------
# Inclusivity threshold boundary
# ---------------------------------------------------------------------------


def test_inclusivity_threshold_is_strictly_below():
    panel = make_panel(3)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions(
        "set_a",
        {
            "target_1": 29.999,  # below -> counts
            "target_2": 30.0,  # exactly on the threshold -> does NOT count
            "target_3": 30.001,  # above -> does not count
        },
    )

    scored = score_sets(
        predictions,
        [primer_set],
        panel,
        ranking_condition=RANKING,
        tp_threshold=30.0,
    )[0]

    assert scored.inclusivity == pytest.approx(1 / 3)


def test_inclusivity_threshold_is_configurable():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions("set_a", {"target_1": 25.0, "target_2": 45.0})

    at_30 = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING, tp_threshold=30.0
    )[0]
    at_60 = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING, tp_threshold=60.0
    )[0]

    assert DEFAULT_TP_THRESHOLD_MINUTES == 30.0
    assert at_30.inclusivity == pytest.approx(0.5)
    assert at_60.inclusivity == pytest.approx(1.0)
    # The threshold must not move the speed score.
    assert at_30.speed_score == pytest.approx(at_60.speed_score)


def test_inclusivity_is_zero_when_nothing_amplifies_in_time():
    panel = make_panel(2)
    primer_set = make_primer_set("set_a", 100)
    predictions = make_predictions("set_a", {"target_1": 55.0, "target_2": 60.0})

    scored = score_sets(
        predictions, [primer_set], panel, ranking_condition=RANKING
    )[0]

    assert scored.inclusivity == 0.0
    assert scored.speed_score == pytest.approx(57.5)


# ---------------------------------------------------------------------------
# Ranking direction -- the most likely silent bug
# ---------------------------------------------------------------------------


def test_faster_predicted_tp_ranks_better():
    """Lower mean Tp == faster == better == earlier in the list."""
    panel = make_panel(3)
    sets, predictions = make_cluster(
        anchors=[100, 300, 500], tps=[30.0, 10.0, 20.0], panel=panel
    )

    scored = score_sets(predictions, sets, panel, ranking_condition=RANKING)

    assert [s.primer_set.anchor for s in scored] == [300, 500, 100]
    assert [s.speed_score for s in scored] == pytest.approx([10.0, 20.0, 30.0])
    # Stated the other way round, so an inversion cannot pass:
    assert scored[0].speed_score < scored[-1].speed_score
    assert scored[0].speed_score == min(s.speed_score for s in scored)


def test_rank_key_is_ascending_and_the_best_set_has_the_smallest_key():
    panel = make_panel(2)
    fast = ScoredSet(
        primer_set=make_primer_set("fast", 100),
        speed_score=12.0,
        inclusivity=1.0,
        per_target_tp={"target_1": 12.0, "target_2": 12.0},
        per_target_mismatches={},
    )
    slow = ScoredSet(
        primer_set=make_primer_set("slow", 400),
        speed_score=41.0,
        inclusivity=1.0,
        per_target_tp={"target_1": 41.0, "target_2": 41.0},
        per_target_mismatches={},
    )

    assert rank_by_speed_then_inclusivity(fast) < rank_by_speed_then_inclusivity(slow)
    assert report.rank_sets([slow, fast])[0] is fast
    assert report.SPEED_SCORE_DIRECTION == "lower_is_faster"
    assert report.SPEED_SCORE_UNITS == "minutes"


def test_inclusivity_breaks_ties_in_favour_of_more_covered_targets():
    panel = make_panel(4)
    sets = [make_primer_set("narrow", 100), make_primer_set("broad", 400)]
    # Same mean Tp (20 min), but "broad" gets every target under 30 min.
    predictions = make_predictions(
        "narrow",
        {"target_1": 5.0, "target_2": 5.0, "target_3": 5.0, "target_4": 65.0},
    )
    predictions += make_predictions(
        "broad",
        {"target_1": 20.0, "target_2": 20.0, "target_3": 20.0, "target_4": 20.0},
    )

    scored = score_sets(predictions, sets, panel, ranking_condition=RANKING)

    assert [s.speed_score for s in scored] == pytest.approx([20.0, 20.0])
    assert [s.primer_set.set_id for s in scored] == ["broad", "narrow"]
    assert scored[0].inclusivity == pytest.approx(1.0)
    assert scored[1].inclusivity == pytest.approx(0.75)


def test_ranking_function_is_swappable():
    panel = make_panel(4)
    sets = [make_primer_set("quick", 100), make_primer_set("inclusive", 400)]
    # "quick" is faster on average but misses one target entirely.
    predictions = make_predictions(
        "quick",
        {"target_1": 5.0, "target_2": 5.0, "target_3": 5.0, "target_4": 61.0},
    )
    predictions += make_predictions(
        "inclusive",
        {"target_1": 25.0, "target_2": 25.0, "target_3": 25.0, "target_4": 25.0},
    )

    by_speed = score_sets(predictions, sets, panel, ranking_condition=RANKING)
    by_inclusivity = score_sets(
        predictions,
        sets,
        panel,
        ranking_condition=RANKING,
        rank_key=rank_by_inclusivity_then_speed,
    )

    assert [s.primer_set.set_id for s in by_speed] == ["quick", "inclusive"]
    assert [s.primer_set.set_id for s in by_inclusivity] == ["inclusive", "quick"]
    assert RANKING_FUNCTIONS["speed_then_inclusivity"] is rank_by_speed_then_inclusivity
    assert RANKING_FUNCTIONS["inclusivity_then_speed"] is rank_by_inclusivity_then_speed
    assert report.ranking_function_name(None) == "speed_then_inclusivity"
    assert (
        report.ranking_function_name(rank_by_inclusivity_then_speed)
        == "inclusivity_then_speed"
    )


def test_ranking_is_deterministic_for_identical_scores():
    panel = make_panel(2)
    sets, predictions = make_cluster(
        anchors=[500, 100, 300], tps=[14.0, 14.0, 14.0], panel=panel
    )

    first = score_sets(predictions, sets, panel, ranking_condition=RANKING)
    second = score_sets(
        predictions, list(reversed(sets)), panel, ranking_condition=RANKING
    )

    ids = [s.primer_set.set_id for s in first]
    assert ids == sorted(ids)  # set_id is the final tie-break, no RNG
    assert ids == [s.primer_set.set_id for s in second]


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------


def _scored_at(anchor: int, speed: float, *, inclusivity: float = 1.0) -> ScoredSet:
    return ScoredSet(
        primer_set=make_primer_set(f"set_{anchor:05d}", anchor),
        speed_score=speed,
        inclusivity=inclusivity,
        per_target_tp={"target_1": speed},
        per_target_mismatches={},
    )


def test_select_diverse_rejects_near_duplicate_anchors():
    # A tight cluster: a naive top-N would return the same design five times.
    scored = [_scored_at(anchor, 10.0 + i) for i, anchor in enumerate(range(100, 125, 5))]

    selected = select_diverse(scored, n=5, min_separation=30)

    assert len(selected) == 1
    assert selected[0].primer_set.anchor == 100  # the best of the cluster


def test_select_diverse_enforces_the_separation_between_every_pair():
    anchors = list(range(0, 400, 7))
    scored = [_scored_at(anchor, 10.0 + 0.01 * anchor) for anchor in anchors]

    selected = select_diverse(scored, n=10, min_separation=30)

    chosen = [s.primer_set.anchor for s in selected]
    assert len(chosen) == 10
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            assert abs(a - b) >= 30


def test_select_diverse_spreads_positions_rather_than_taking_the_top_n():
    # The 20 best-scoring candidates all sit inside a 40 bp window; good
    # candidates are spread further out with slightly worse scores.
    scored = [_scored_at(anchor, 10.0 + 0.001 * anchor) for anchor in range(100, 140, 2)]
    scored += [_scored_at(anchor, 12.0) for anchor in range(200, 601, 50)]

    naive_top = [s.primer_set.anchor for s in report.rank_sets(scored)[:5]]
    selected = [s.primer_set.anchor for s in select_diverse(scored, n=5, min_separation=30)]

    assert max(naive_top) - min(naive_top) <= 40  # one design, five times over
    assert selected != naive_top
    assert len(set(selected)) == 5
    assert max(selected) - min(selected) >= 200  # reaches well past the cluster
    assert sum(1 for anchor in selected if anchor < 200) == 2  # cluster capped
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            assert abs(a - b) >= 30


def test_select_diverse_keeps_the_best_member_of_each_cluster():
    scored = [
        _scored_at(100, 25.0),
        _scored_at(110, 11.0),  # best in the first cluster
        _scored_at(120, 30.0),
        _scored_at(400, 40.0),
        _scored_at(405, 12.0),  # best in the second cluster
    ]

    selected = select_diverse(scored, n=5, min_separation=30)

    assert [s.primer_set.anchor for s in selected] == [110, 405]
    assert [s.speed_score for s in selected] == pytest.approx([11.0, 12.0])


def test_select_diverse_respects_n_even_when_everything_is_far_apart():
    scored = [_scored_at(anchor, 10.0) for anchor in range(0, 1000, 100)]

    assert len(select_diverse(scored, n=3, min_separation=30)) == 3
    assert select_diverse(scored, n=0, min_separation=30) == []
    assert select_diverse([], n=20, min_separation=30) == []


def test_zero_separation_degrades_to_a_plain_top_n():
    scored = [_scored_at(anchor, 10.0 + i) for i, anchor in enumerate(range(100, 125, 5))]

    selected = select_diverse(scored, n=3, min_separation=0)

    assert [s.primer_set.anchor for s in selected] == [100, 105, 110]


def test_tightly_packed_candidates_warn_instead_of_padding_the_list():
    panel = make_panel(2)
    # 25 candidates packed into 24 bp: only one can survive a 30 bp separation.
    sets, predictions = make_cluster(
        anchors=range(100, 125),
        tps=[10.0 + 0.1 * i for i in range(25)],
        panel=panel,
    )

    result = build_design_result(
        predictions,
        sets,
        panel,
        ranking_condition=RANKING,
        n=20,
        min_separation=30,
    )

    assert len(result.scored_sets) == 1
    assert result.funnel[FUNNEL_SCORED] == 25
    assert result.funnel[FUNNEL_RETURNED] == 1
    assert result.warnings, "a short list must be explained, not silent"
    message = " ".join(result.warnings)
    assert "1 of 20" in message
    assert "30 bp" in message
    # And the constraint really was kept rather than quietly relaxed.
    anchors = [s.primer_set.anchor for s in result.scored_sets]
    assert all(
        abs(a - b) >= 30 for i, a in enumerate(anchors) for b in anchors[i + 1 :]
    )


def test_no_warning_when_the_full_request_is_satisfied():
    panel = make_panel(2)
    sets, predictions = make_cluster(
        anchors=range(0, 1000, 100), tps=[10.0] * 10, panel=panel
    )

    result = build_design_result(
        predictions, sets, panel, ranking_condition=RANKING, n=5, min_separation=30
    )

    assert len(result.scored_sets) == 5
    assert result.warnings == []
    assert diversity_warnings(
        report.rank_sets([]), [], n=1, min_separation=30
    )  # empty input still explains itself


def test_default_selection_parameters_match_the_design_notes():
    assert DEFAULT_N_SETS == 20
    assert DEFAULT_MIN_SEPARATION_BP == 30


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def _three_set_result(panel: Optional[TargetPanel] = None) -> DesignResult:
    panel = panel or make_panel(3)
    sets = [
        make_primer_set("set_00100", 100),
        make_primer_set("set_00300", 300),
        make_primer_set("set_00500", 500),
    ]
    predictions: List[Prediction] = []
    predictions += make_predictions(
        "set_00100", {"target_1": 10.0, "target_2": 20.0, "target_3": 30.0}
    )
    predictions += make_predictions(
        "set_00300", {"target_1": 12.0, "target_2": 12.0, "target_3": 12.0}
    )
    predictions += make_predictions(
        "set_00500", {"target_1": 40.0, "target_2": 50.0, "target_3": 60.0}
    )
    # A second condition, so per_target.csv is genuinely (set x target x condition).
    for set_id, tp in (("set_00100", 25.0), ("set_00300", 26.0), ("set_00500", 70.0)):
        predictions += even_predictions(set_id, tp, panel, conditions=(OTHER,))

    return build_design_result(
        predictions,
        sets,
        panel,
        ranking_condition=RANKING,
        mismatches={
            ("set_00100", "target_1"): 0,
            ("set_00100", "target_2"): 1,
            ("set_00100", "target_3"): 2,
            ("set_00300", "target_1"): 0,
            ("set_00300", "target_2"): 0,
            ("set_00300", "target_3"): 1,
            ("set_00500", "target_1"): 4,
            ("set_00500", "target_2"): 5,
            ("set_00500", "target_3"): 6,
        },
        n=20,
        min_separation=30,
        funnel={"anchors": 420, "geometries": 1680, "dimer_filter": 900},
        excluded_targets={"ragged_seq": "more than 10% ambiguous bases"},
    )


def test_write_outputs_writes_every_file(tmp_path: Path):
    result = _three_set_result()

    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    assert set(paths) == {"sets_csv", "per_target_csv", "json", "summary_txt"}
    assert paths["sets_csv"] == tmp_path / "run.sets.csv"
    assert paths["per_target_csv"] == tmp_path / "run.per_target.csv"
    assert paths["json"] == tmp_path / "run.json"
    assert paths["summary_txt"] == tmp_path / "run.summary.txt"
    for path in paths.values():
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_write_outputs_creates_missing_directories(tmp_path: Path):
    result = _three_set_result()

    paths = write_outputs(result, tmp_path / "nested" / "deeper" / "run")

    assert all(path.exists() for path in paths.values())


def test_sets_csv_columns_are_self_documenting_and_round_trip(tmp_path: Path):
    result = _three_set_result()
    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    header = csv_header(paths["sets_csv"])
    rows = read_csv_rows(paths["sets_csv"])

    assert header == sets_csv_columns()
    assert header[:3] == ["rank", "set_id", "anchor"]
    assert "mean_predicted_tp_minutes_lower_is_faster" in header
    assert "inclusivity_fraction_targets_below_tp_threshold" in header
    assert "tp_threshold_minutes" in header
    assert header[-2:] == [
        "total_primer_target_mismatches_all_targets",
        "mean_primer_target_mismatches_per_target",
    ]
    # No bare "score" column anywhere; units/direction are always spelled out.
    assert "score" not in header
    assert "speed_score" not in header

    assert [row["rank"] for row in rows] == ["1", "2", "3"]
    assert [row["set_id"] for row in rows] == ["set_00300", "set_00100", "set_00500"]
    tps = [float(row["mean_predicted_tp_minutes_lower_is_faster"]) for row in rows]
    assert tps == pytest.approx([12.0, 20.0, 50.0])
    assert tps == sorted(tps), "rank 1 must be the fastest (lowest Tp)"

    best = rows[0]
    assert best["anchor"] == "300"
    assert float(best["inclusivity_fraction_targets_below_tp_threshold"]) == 1.0
    assert float(best["tp_threshold_minutes"]) == 30.0
    assert best["n_targets_scored"] == "3"
    assert best["n_panel_targets"] == "3"
    assert best["ranking_condition"] == "BST@1000"
    assert best["total_primer_target_mismatches_all_targets"] == "1"

    slow = rows[-1]
    assert slow["slowest_target_name"] == "target_3"
    assert float(slow["slowest_predicted_tp_minutes"]) == 60.0
    assert float(slow["inclusivity_fraction_targets_below_tp_threshold"]) == 0.0


def test_sets_csv_carries_all_eight_primers_with_coordinates(tmp_path: Path):
    result = _three_set_result()
    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)
    row = read_csv_rows(paths["sets_csv"])[0]

    primer_set = next(
        s.primer_set for s in result.scored_sets if s.primer_set.set_id == row["set_id"]
    )
    for primer_type in PRIMER_TYPE_ORDER:
        primer = primer_set.primers[primer_type]
        assert row[f"{primer_type}_sequence"] == primer.sequence
        assert row[f"{primer_type}_start"] == str(primer.start)
        assert row[f"{primer_type}_end"] == str(primer.end)
        assert row[f"{primer_type}_length"] == str(primer.length)
        assert int(row[f"{primer_type}_length"]) == len(primer.sequence)

    assert int(row["span_start"]) == primer_set.span[0]
    assert int(row["span_end"]) == primer_set.span[1]
    # Eight distinct sequences, i.e. the columns are not all the same primer.
    assert len({row[f"{t}_sequence"] for t in PRIMER_TYPE_ORDER}) == 8


def test_per_target_csv_is_long_format_and_round_trips(tmp_path: Path):
    result = _three_set_result()
    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    header = csv_header(paths["per_target_csv"])
    rows = read_csv_rows(paths["per_target_csv"])

    assert header == list(report.PER_TARGET_COLUMNS)
    assert "predicted_tp_minutes" in header
    assert "predicted_tp_std_minutes" in header
    # 3 sets x 3 targets x 2 conditions
    assert len(rows) == 18
    assert {row["master_mix"] for row in rows} == {"BST", "SS"}

    ranking_rows = [row for row in rows if row["is_ranking_condition"] == "True"]
    assert len(ranking_rows) == 9
    assert all(row["master_mix"] == "BST" for row in ranking_rows)
    assert all(row["target_conc"] == "1000" for row in ranking_rows)

    # The matrix must point at the problem sequence for the slowest set.
    slow = [
        row
        for row in ranking_rows
        if row["set_id"] == "set_00500" and row["target_name"] == "target_3"
    ]
    assert len(slow) == 1
    assert float(slow[0]["predicted_tp_minutes"]) == 60.0
    assert float(slow[0]["predicted_tp_std_minutes"]) == 1.5
    assert slow[0]["predicted_tp_below_threshold"] == "False"
    assert slow[0]["primer_target_mismatches"] == "6"
    assert slow[0]["rank"] == "3"

    fast = [
        row
        for row in ranking_rows
        if row["set_id"] == "set_00300" and row["target_name"] == "target_1"
    ]
    assert fast[0]["predicted_tp_below_threshold"] == "True"
    assert fast[0]["rank"] == "1"

    # Ranks agree with the sets CSV.
    sets_rank = {row["set_id"]: row["rank"] for row in read_csv_rows(paths["sets_csv"])}
    assert all(row["rank"] == sets_rank[row["set_id"]] for row in rows)


def test_csvs_are_readable_by_pandas(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    result = _three_set_result()
    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    sets_df = pd.read_csv(paths["sets_csv"])
    per_target_df = pd.read_csv(paths["per_target_csv"])

    assert list(sets_df["rank"]) == [1, 2, 3]
    assert sets_df["mean_predicted_tp_minutes_lower_is_faster"].is_monotonic_increasing
    assert len(per_target_df) == 18
    assert per_target_df["predicted_tp_minutes"].max() == 70.0
    assert per_target_df.groupby("set_id")["target_name"].nunique().eq(3).all()


def test_json_round_trips_with_funnel_excluded_targets_and_warnings(tmp_path: Path):
    panel = make_panel(3)
    result = _three_set_result(panel)
    result.warnings.append("a hand-written note")

    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)
    payload = json.loads(paths["json"].read_text())

    assert payload == result_to_dict(
        result, ranking_condition=RANKING
    ) | {"generated_utc": payload["generated_utc"]}

    assert payload["funnel"] == {
        "anchors": 420,
        "geometries": 1680,
        "dimer_filter": 900,
        FUNNEL_SCORED: 3,
        FUNNEL_RETURNED: 3,
    }
    assert payload["excluded_targets"] == {
        "ragged_seq": "more than 10% ambiguous bases"
    }
    assert "a hand-written note" in payload["warnings"]
    assert payload["n_sets_returned"] == 3
    assert payload["panel"]["n_targets"] == 3
    assert payload["panel"]["target_names"] == ["target_1", "target_2", "target_3"]

    ranking = payload["ranking"]
    assert ranking["function"] == "speed_then_inclusivity"
    assert ranking["tp_threshold_minutes"] == 30.0
    assert ranking["ranking_condition"]["master_mix"] == "BST"
    assert ranking["ranking_condition"]["label"] == "BST@1000"
    assert ranking["speed_score"]["units"] == "minutes"
    assert ranking["speed_score"]["direction"] == "lower_is_faster"
    assert ranking["inclusivity"]["direction"] == "higher_is_better"

    best = payload["sets"][0]
    assert best["rank"] == 1
    assert best["set_id"] == "set_00300"
    assert best["mean_predicted_tp_minutes_lower_is_faster"] == pytest.approx(12.0)
    assert [p["primer_type"] for p in best["primers"]] == list(PRIMER_TYPE_ORDER)
    assert len(best["per_target"]) == 3
    assert len(best["predictions"]) == 6
    assert sum(p["is_ranking_condition"] for p in best["predictions"]) == 3
    assert [s["rank"] for s in payload["sets"]] == [1, 2, 3]
    assert [
        s["mean_predicted_tp_minutes_lower_is_faster"] for s in payload["sets"]
    ] == sorted(s["mean_predicted_tp_minutes_lower_is_faster"] for s in payload["sets"])


def test_summary_text_reports_the_funnel_counts_and_warnings(tmp_path: Path):
    result = _three_set_result()
    result.warnings.append("ViennaRNA 2.4.17 vs 2.7.2 may disagree on dG")

    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)
    text = paths["summary_txt"].read_text()

    assert "LAMP primer design summary" in text
    assert "BST@1000" in text
    # The summary now leads with the ranking rather than a Tp value, because
    # predicted Tp legitimately goes negative (the model is trained with
    # regress_tp_cutoff = -100) and a portal printing "-5.46 min" as a
    # duration reads as a bug. The rank is the output; Tp appears banded.
    assert "a RANKING" in text
    assert "Rank 1 is the fastest" in text
    assert f"<{FAST_TP_FLOOR_MINUTES:g} min" in text
    assert "Candidate funnel" in text
    for stage in ("anchors", "geometries", "dimer_filter", FUNNEL_SCORED, FUNNEL_RETURNED):
        assert stage in text
    assert "420" in text and "1,680" in text
    assert "Candidate sets entering ranking : 3" in text
    assert "Primer sets returned            : 3" in text
    assert "ragged_seq" in text
    assert "ViennaRNA 2.4.17 vs 2.7.2 may disagree on dG" in text
    assert "set_00300" in text  # the top set is named
    assert str(paths["sets_csv"]) in text


def test_ranking_condition_is_inferred_when_not_passed(tmp_path: Path):
    result = _three_set_result()

    assert infer_ranking_condition(result) == RANKING

    paths = write_outputs(result, tmp_path / "run")
    payload = json.loads(paths["json"].read_text())

    assert payload["ranking"]["ranking_condition"]["label"] == "BST@1000"
    rows = read_csv_rows(paths["per_target_csv"])
    assert sum(row["is_ranking_condition"] == "True" for row in rows) == 9


def test_write_outputs_overwrites_cleanly(tmp_path: Path):
    result = _three_set_result()
    write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    smaller = DesignResult(
        scored_sets=result.scored_sets[:1], panel=result.panel, funnel=result.funnel
    )
    paths = write_outputs(smaller, tmp_path / "run", ranking_condition=RANKING)

    assert len(read_csv_rows(paths["sets_csv"])) == 1


def test_scored_set_without_a_prediction_grid_still_writes_per_target_rows(
    tmp_path: Path,
):
    panel = make_panel(2)
    scored = ScoredSet(
        primer_set=make_primer_set("set_00100", 100),
        speed_score=15.0,
        inclusivity=1.0,
        per_target_tp={"target_1": 14.0, "target_2": 16.0},
        per_target_mismatches={"target_1": 0, "target_2": 2},
        predictions=[],
    )
    result = DesignResult(scored_sets=[scored], panel=panel)

    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)
    rows = read_csv_rows(paths["per_target_csv"])

    assert len(rows) == 2
    assert {row["target_name"] for row in rows} == {"target_1", "target_2"}
    assert all(row["predicted_tp_std_minutes"] == "" for row in rows)


def test_unknown_mismatch_counts_are_blank_in_the_csvs(tmp_path: Path):
    panel = make_panel(2)
    sets, predictions = make_cluster(anchors=[100], tps=[12.0], panel=panel)
    result = build_design_result(
        predictions, sets, panel, ranking_condition=RANKING, n=1
    )

    paths = write_outputs(result, tmp_path / "run", ranking_condition=RANKING)

    sets_row = read_csv_rows(paths["sets_csv"])[0]
    assert sets_row["total_primer_target_mismatches_all_targets"] == ""
    assert sets_row["mean_primer_target_mismatches_per_target"] == ""
    assert all(
        row["primer_target_mismatches"] == ""
        for row in read_csv_rows(paths["per_target_csv"])
    )


# ---------------------------------------------------------------------------
# Empty run
# ---------------------------------------------------------------------------


def test_empty_prediction_list_scores_to_nothing_without_raising():
    panel = make_panel(3)
    sets = [make_primer_set("set_00100", 100)]

    assert score_sets([], sets, panel, ranking_condition=RANKING) == []
    assert score_sets([], [], panel, ranking_condition=RANKING) == []
    assert score_sets([], [], None, ranking_condition=RANKING) == []
    assert select_diverse([], n=20, min_separation=30) == []


def test_empty_run_produces_a_valid_empty_report(tmp_path: Path):
    panel = make_panel(3)

    result = build_design_result(
        [],
        [],
        panel,
        ranking_condition=RANKING,
        funnel={"anchors": 0},
    )

    assert result.scored_sets == []
    assert result.funnel == {"anchors": 0, FUNNEL_SCORED: 0, FUNNEL_RETURNED: 0}
    assert result.warnings, "an empty run must say so"
    assert "no designs" in " ".join(result.warnings).lower()

    paths = write_outputs(result, tmp_path / "empty", ranking_condition=RANKING)

    # Header-only CSVs, valid JSON, readable summary -- nothing raises.
    assert csv_header(paths["sets_csv"]) == sets_csv_columns()
    assert read_csv_rows(paths["sets_csv"]) == []
    assert csv_header(paths["per_target_csv"]) == list(report.PER_TARGET_COLUMNS)
    assert read_csv_rows(paths["per_target_csv"]) == []

    payload = json.loads(paths["json"].read_text())
    assert payload["sets"] == []
    assert payload["n_sets_returned"] == 0
    assert payload["funnel"][FUNNEL_SCORED] == 0
    assert payload["warnings"]

    text = paths["summary_txt"].read_text()
    assert "No primer sets were returned." in text
    assert "Primer sets returned            : 0" in text


def test_empty_panel_and_empty_result_still_render(tmp_path: Path):
    panel = TargetPanel(targets=[], consensus="")
    result = DesignResult(scored_sets=[], panel=panel)

    paths = write_outputs(result, tmp_path / "bare")
    payload = json.loads(paths["json"].read_text())

    assert payload["panel"] == {
        "n_targets": 0,
        "target_names": [],
        "consensus_length": 0,
        "budget": 0,
    }
    assert payload["ranking"]["ranking_condition"] is None
    assert "(unknown)" in paths["summary_txt"].read_text()
    assert format_summary(result).startswith("LAMP primer design summary")


# ---------------------------------------------------------------------------
# format_tp
#
# Predicted Tp goes negative by design (the model is trained with
# regress_tp_cutoff = -100), so the user-facing string is banded. The rank is
# the output; the number is context.
# ---------------------------------------------------------------------------


def test_fast_predictions_are_banded_not_printed():
    from lamp_design.report import format_tp

    assert format_tp(-5.46) == "<5 min"
    assert format_tp(0.0) == "<5 min"
    assert format_tp(4.99) == "<5 min"


def test_predictions_above_the_floor_keep_their_value():
    from lamp_design.report import format_tp

    assert format_tp(5.0) == "5.0 min"
    assert format_tp(12.74) == "12.7 min"
    assert format_tp(59.9) == "59.9 min"


def test_a_missing_prediction_is_not_rendered_as_a_number():
    from lamp_design.report import format_tp

    assert format_tp(None) == "n/a"


def test_no_negative_duration_can_reach_a_user_facing_string():
    """The specific thing this exists to prevent."""
    from lamp_design.report import format_tp

    for value in (-100.0, -42.0, -5.46, -1e-9):
        rendered = format_tp(value)
        assert "-" not in rendered, f"{value} rendered as {rendered!r}"


def test_the_band_matches_the_measured_quartile():
    """5 minutes is where the data puts it, not a round number.

    p25 of measured Tp in the curated corpus is 5.06 min (p0.1 = 1.76,
    p5 = 2.70), so the band is almost exactly "the fastest quartile of
    everything ever measured".
    """
    from lamp_design.report import FAST_TP_FLOOR_MINUTES

    assert FAST_TP_FLOOR_MINUTES == 5.0
