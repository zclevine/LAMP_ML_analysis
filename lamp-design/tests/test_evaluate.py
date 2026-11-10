"""``lamp-design evaluate`` scores primer sets that already exist.

The risk this file covers is not arithmetic -- ``evaluate`` reuses ``design``'s
thermo and inference path verbatim -- but *input handling*. It reads whatever
CSV a lab actually has, which means guessing column names, tolerating blank
measurements, and placing primers in a target by approximate match. Each of
those is a place where being silently wrong is easy and costly: a mis-read
master mix scores against no mix at all, and a mis-placed window turns a
mismatch feature into noise.
"""

from __future__ import annotations

import pathlib
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lamp_design.commands.evaluate import (
    DEFAULT_MAX_WINDOW_MISMATCHES,
    EvalRow,
    EvaluationResult,
    attach_located_windows,
    format_summary,
    load_rows,
    locate_window,
    metrics_table,
    predictions_rows,
    roc_auc,
    write_outputs,
)
from lamp_design.commands.prepare import InputError
from lamp_design.thermo import revcomp
from lamp_design.types import FORWARD_STRAND_TYPES, PRIMER_TYPE_ORDER, Condition

# A fixed set of eight distinguishable 20-mers, one per primer type.
_PRIMERS = {
    "F3": "ACGTACGTACGTACGTACGT",
    "F2": "TTGGCCAATTGGCCAATTGG",
    "FL": "AAGGTTCCAAGGTTCCAAGG",
    "F1c": "GGCCTTAAGGCCTTAAGGCC",
    "B1c": "CCAATTGGCCAATTGGCCAA",
    "BL": "TACGATCGTACGATCGTACG",
    "B2": "GATTACAGATTACAGATTAC",
    "B3": "CCCGGGAAATTTCCCGGGAA",
}


def _row(**overrides) -> dict:
    """One input record with perfect-match windows."""
    record = {
        "primer_set": "PS01",
        "target_seq": "T01",
        "master_mix": "BST",
        "target_conc": "10000",
        "tp": "12.5",
    }
    for ptype in PRIMER_TYPE_ORDER:
        record[f"{ptype}_sequence"] = _PRIMERS[ptype]
        record[f"{ptype}_target_sequence"] = _PRIMERS[ptype]
    record.update(overrides)
    return record


def _write_csv(path: Path, records: list[dict]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def test_reads_a_results_csv_with_windows(tmp_path):
    rows, warnings = load_rows(_write_csv(tmp_path / "in.csv", [_row()]))
    assert len(rows) == 1
    row = rows[0]
    assert row.set_id == "PS01"
    assert row.target_name == "T01"
    assert row.condition == Condition("BST", 10_000.0)
    assert row.measured_tp == pytest.approx(12.5)
    assert row.windows is not None and len(row.windows) == len(PRIMER_TYPE_ORDER)
    assert not warnings


def test_primer_columns_may_be_bare_type_names(tmp_path):
    """`F3` is as acceptable a column name as `F3_sequence`."""
    record = {"primer_set": "PS01"}
    for ptype in PRIMER_TYPE_ORDER:
        record[ptype] = _PRIMERS[ptype]
    rows, _ = load_rows(_write_csv(tmp_path / "bare.csv", [record]))
    assert rows[0].primers == [_PRIMERS[p] for p in PRIMER_TYPE_ORDER]
    # No window columns, so they have to be located later.
    assert rows[0].windows is None


def test_a_missing_primer_column_names_what_is_missing(tmp_path):
    record = _row()
    del record["B3_sequence"]
    with pytest.raises(InputError, match=r"missing a sequence column for \['B3'\]"):
        load_rows(_write_csv(tmp_path / "short.csv", [record]))


def test_primers_are_read_in_layout_order_not_column_order(tmp_path):
    """The eight sequences must come back in PRIMER_TYPE_ORDER.

    The encoder re-permutes into the model's alphabetical order from this
    layout order, so a row assembled in column order would be silently
    scrambled -- the single most consequential mistake available here.
    """
    record = {"primer_set": "PS01"}
    # Deliberately write the columns in reverse layout order.
    for ptype in reversed(PRIMER_TYPE_ORDER):
        record[f"{ptype}_sequence"] = _PRIMERS[ptype]
    rows, _ = load_rows(_write_csv(tmp_path / "rev.csv", [record]))
    assert rows[0].primers == [_PRIMERS[p] for p in PRIMER_TYPE_ORDER]


def test_blank_tp_becomes_nan_not_zero(tmp_path):
    """A reaction that never amplified is censored, never "amplified at 0"."""
    rows, _ = load_rows(
        _write_csv(tmp_path / "blank.csv", [_row(tp=""), _row(tp="NA"), _row(tp="12")])
    )
    assert [np.isnan(r.measured_tp) for r in rows] == [True, True, False]


def test_master_mix_aliases_are_canonicalized(tmp_path):
    """`BstXT` and `WS-SYTO9` are spellings of trained mixes, not new ones."""
    rows, warnings = load_rows(
        _write_csv(
            tmp_path / "alias.csv",
            [_row(master_mix="BstXT"), _row(master_mix="WS-SYTO9")],
        )
    )
    assert [r.condition.master_mix for r in rows] == ["BST", "WS"]
    assert not warnings


def test_an_untrained_master_mix_is_dropped_and_reported(tmp_path):
    """Silently encoding an unknown mix as an all-zero one-hot is the bug."""
    rows, warnings = load_rows(
        _write_csv(
            tmp_path / "unknown.csv",
            [_row(master_mix="Bst 2.0"), _row(master_mix="SS")],
        )
    )
    assert len(rows) == 1 and rows[0].condition.master_mix == "SS"
    assert any("not trained on" in w and "Bst 2.0" in w for w in warnings)


def test_all_rows_untrained_is_an_input_error(tmp_path):
    with pytest.raises(InputError, match="untrained master mix"):
        load_rows(_write_csv(tmp_path / "all_bad.csv", [_row(master_mix="Bst 2.0")]))


def test_a_window_of_the_wrong_length_is_dropped(tmp_path):
    """Mismatch counting is elementwise, so a short window is meaningless."""
    record = _row()
    record["F3_target_sequence"] = _PRIMERS["F3"][:-3]
    rows, warnings = load_rows(
        _write_csv(tmp_path / "ragged.csv", [record, _row()])
    )
    assert len(rows) == 1
    assert any("wrong-length" in w for w in warnings)


def test_overriding_the_condition_replaces_what_the_file_says(tmp_path):
    rows, _ = load_rows(
        _write_csv(tmp_path / "ov.csv", [_row(master_mix="SS", target_conc="100")]),
        master_mix="WS",
        target_conc=1_000_000.0,
    )
    assert rows[0].condition == Condition("WS", 1_000_000.0)


def test_a_bad_truth_column_name_is_rejected(tmp_path):
    with pytest.raises(InputError, match="--truth-column 'nope' is not a column"):
        load_rows(_write_csv(tmp_path / "t.csv", [_row()]), truth_column="nope")


def test_require_windows_explains_the_alternative(tmp_path):
    record = {"primer_set": "PS01"}
    for ptype in PRIMER_TYPE_ORDER:
        record[f"{ptype}_sequence"] = _PRIMERS[ptype]
    with pytest.raises(InputError, match="--windows locate"):
        load_rows(_write_csv(tmp_path / "nw.csv", [record]), require_windows=True)


def test_empty_and_headerless_inputs_are_distinguished(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(InputError, match="no header row"):
        load_rows(tmp_path / "empty.csv")
    (tmp_path / "head.csv").write_text("primer_set,F3_sequence\n")
    with pytest.raises(InputError, match="header but no data rows"):
        load_rows(tmp_path / "head.csv")


def test_tsv_is_read_with_tab_delimiters(tmp_path):
    record = _row()
    path = tmp_path / "in.tsv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record), delimiter="\t")
        writer.writeheader()
        writer.writerow(record)
    rows, _ = load_rows(path)
    assert len(rows) == 1 and rows[0].set_id == "PS01"


# ---------------------------------------------------------------------------
# Window location
# ---------------------------------------------------------------------------


def test_locate_window_finds_an_exact_match():
    target = "TTTT" + _PRIMERS["F3"] + "GGGG"
    start, mm = locate_window(_PRIMERS["F3"], target)
    assert (start, mm) == (4, 0)


def test_locate_window_finds_the_best_of_several_imperfect_sites():
    primer = _PRIMERS["F2"]
    two_off = "AA" + primer[2:]
    one_off = "A" + primer[1:]
    target = "CCCC" + two_off + "CCCC" + one_off + "CCCC"
    start, mm = locate_window(primer, target)
    assert mm == 1
    assert target[start : start + len(primer)] == one_off


def test_locate_window_counts_ambiguity_codes_as_mismatches():
    """An N is absence of evidence, not evidence of a match."""
    primer = _PRIMERS["F3"]
    target = primer[:5] + "NNN" + primer[8:]
    _, mm = locate_window(primer, target)
    assert mm == 3


def test_locate_window_rejects_a_target_shorter_than_the_primer():
    start, mm = locate_window(_PRIMERS["F3"], "ACGT")
    assert start is None and mm == len(_PRIMERS["F3"])


def _synthetic_target() -> str:
    """A target that every one of the eight primers binds perfectly.

    Forward-strand primers appear as themselves; the rest appear as their
    reverse complement, which is how they sit on the forward strand.
    """
    parts = []
    for ptype in PRIMER_TYPE_ORDER:
        seq = _PRIMERS[ptype]
        parts.append(seq if ptype in FORWARD_STRAND_TYPES else revcomp(seq))
    return "GGGGG".join(parts)


def test_located_windows_respect_strand_convention():
    """A located window must come back in the orientation the model expects."""
    row = EvalRow(
        set_id="PS01",
        target_name="T",
        primers=[_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
        windows=None,
        condition=Condition("BST", 10_000.0),
    )
    out, skipped = attach_located_windows([row], [("T", _synthetic_target())])
    assert len(out) == 1 and not skipped["no_placement"]
    # Perfect placement, so each window should equal its primer exactly.
    assert out[0].windows == row.primers
    assert out[0].max_located_mismatches == 0


def test_a_set_that_does_not_bind_is_dropped_not_scored():
    """Scoring a primer against its least-bad coincidence would be nonsense."""
    row = EvalRow(
        set_id="PS01",
        target_name="T",
        primers=[_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
        windows=None,
        condition=Condition("BST", 10_000.0),
    )
    unrelated = "AT" * 500
    out, skipped = attach_located_windows([row], [("T", unrelated)])
    assert out == []
    assert skipped["no_placement"] == 1


def test_locating_expands_one_row_per_target():
    row = EvalRow(
        set_id="PS01",
        target_name="ignored",
        primers=[_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
        windows=None,
        condition=Condition("BST", 10_000.0),
    )
    target = _synthetic_target()
    out, _ = attach_located_windows([row], [("A", target), ("B", target)])
    assert [r.target_name for r in out] == ["A", "B"]


def test_max_window_mismatches_is_enforced():
    primers = [_PRIMERS[p] for p in PRIMER_TYPE_ORDER]
    target = list(_synthetic_target())
    # Corrupt one base inside the F3 window -- F3 is first and forward-strand.
    target[3] = "G" if target[3] != "G" else "C"
    row = EvalRow("PS01", "T", primers, None, Condition("BST", 10_000.0))
    kept, _ = attach_located_windows([row], [("T", "".join(target))], max_mismatches=1)
    assert len(kept) == 1
    dropped, skipped = attach_located_windows(
        [row], [("T", "".join(target))], max_mismatches=0
    )
    assert dropped == [] and skipped["no_placement"] == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_roc_auc_matches_hand_computed_values():
    assert roc_auc([True, False], [1.0, 0.0]) == pytest.approx(1.0)
    assert roc_auc([True, False], [0.0, 1.0]) == pytest.approx(0.0)
    # A tie between one positive and one negative is worth exactly 0.5.
    assert roc_auc([True, False], [1.0, 1.0]) == pytest.approx(0.5)
    assert roc_auc([True, True, False, False], [3, 2, 1, 0]) == pytest.approx(1.0)
    assert roc_auc([True, False, True, False], [3, 2, 1, 0]) == pytest.approx(0.75)


def test_roc_auc_is_nan_without_both_classes():
    assert np.isnan(roc_auc([True, True], [1.0, 2.0]))
    assert np.isnan(roc_auc([False, False], [1.0, 2.0]))


def test_metrics_report_amplified_and_censored_separately():
    measured = [5.0, 10.0, 15.0, float("nan")]
    predicted = [6.0, 11.0, 16.0, 55.0]
    out = metrics_table(measured, predicted, censor_at=60.0)
    assert out["amplified"]["n"] == 3
    assert out["censored"]["n"] == 4
    # Amplified rows are over-predicted by exactly a minute.
    assert out["amplified"]["bias_minutes"] == pytest.approx(1.0)
    assert out["amplified"]["mae_minutes"] == pytest.approx(1.0)
    # The censored row predicted 55 against a censored truth of 60.
    assert out["censored"]["bias_minutes"] == pytest.approx((1 + 1 + 1 - 5) / 4)


def test_a_measurement_above_the_censor_point_is_clipped():
    out = metrics_table([90.0], [60.0], censor_at=60.0)
    assert out["censored"]["bias_minutes"] == pytest.approx(0.0)


def test_classification_counts_only_amplified_within_threshold():
    measured = [5.0, 40.0, float("nan")]
    out = metrics_table(measured, [5.0, 40.0, 60.0], tp_threshold=30.0)
    assert out["classification"]["n_positive"] == 1
    assert out["classification"]["n"] == 3


def test_metrics_survive_degenerate_input():
    """Constant or tiny inputs give nan, not an exception."""
    out = metrics_table([5.0], [6.0])
    assert out["amplified"]["n"] == 1
    assert np.isnan(out["amplified"]["spearman_rho"])
    empty = metrics_table([float("nan")], [10.0])
    assert empty["amplified"] == {"n": 0.0}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _result() -> EvaluationResult:
    rows = [
        EvalRow("PS01", "T01", [_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
                [_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
                Condition("BST", 10_000.0), measured_tp=12.5),
        EvalRow("PS02", "T01", [_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
                [_PRIMERS[p] for p in PRIMER_TYPE_ORDER],
                Condition("BST", 10_000.0), measured_tp=float("nan")),
    ]
    return EvaluationResult(
        rows=rows,
        predicted_tp=[13.0, 58.0],
        predicted_std=[1.0, 2.0],
        metrics=metrics_table([12.5, float("nan")], [13.0, 58.0]),
        params={"n_rows": 2, "device": "cpu", "censor_at_minutes": 60.0},
    )


def test_prediction_rows_distinguish_censored_from_missing():
    rows = predictions_rows(_result())
    assert rows[0]["amplified"] == "1"
    assert rows[0]["measured_tp_minutes"] == "12.500"
    assert rows[0]["residual_minutes"] == "+0.500"
    # Measured but non-amplifying: flagged, with no residual to report.
    assert rows[1]["amplified"] == "0"
    assert rows[1]["measured_tp_minutes"] == "NA"
    assert rows[1]["residual_minutes"] == ""


def test_write_outputs_writes_three_files(tmp_path):
    paths = write_outputs(_result(), tmp_path / "run")
    assert set(paths) == {"predictions_csv", "metrics_json", "summary_txt"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    blob = json.loads(paths["metrics_json"].read_text())
    assert blob["metrics"]["amplified"]["n"] == 1
    assert "lamp-design evaluate" in paths["summary_txt"].read_text()


def test_a_prefix_containing_a_dot_keeps_it(tmp_path):
    """`with_suffix` would turn `run.v2` into `run.predictions.csv`."""
    paths = write_outputs(_result(), tmp_path / "run.v2")
    assert paths["predictions_csv"].name == "run.v2.predictions.csv"


def test_summary_says_so_when_there_is_no_truth_column():
    result = _result()
    result.metrics = {}
    assert "No measured Tp column" in format_summary(result)


# ---------------------------------------------------------------------------
# End to end, through the real CLI and the real checkpoints
# ---------------------------------------------------------------------------


def _weights_or_skip() -> str:
    from _weights import WEIGHTS_MISSING_REASON, weights_or_none

    weights = weights_or_none()
    if weights is None:
        pytest.skip(WEIGHTS_MISSING_REASON)
    return weights


def test_cli_evaluate_scores_a_csv_with_windows(tmp_path):
    _weights_or_skip()
    path = _write_csv(
        tmp_path / "in.csv",
        [_row(), _row(primer_set="PS02", tp=""), _row(master_mix="SS", tp="30.1")],
    )
    out = subprocess.run(
        [sys.executable, "-m", "lamp_design.cli", "evaluate",
         "--in", str(path), "--out", str(tmp_path / "run"), "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rows = list(csv.DictReader((tmp_path / "run.predictions.csv").open()))
    assert len(rows) == 3
    assert all(r["predicted_tp_minutes"] for r in rows)
    # Two of the three carry a measurement, so metrics must be present.
    blob = json.loads((tmp_path / "run.metrics.json").read_text())
    assert blob["metrics"]["amplified"]["n"] == 2
    assert blob["params"]["has_measured_tp"] is True


def test_cli_evaluate_locates_windows_in_a_target_fasta(tmp_path):
    _weights_or_skip()
    record = {"primer_set": "PS01"}
    for ptype in PRIMER_TYPE_ORDER:
        record[f"{ptype}_sequence"] = _PRIMERS[ptype]
    path = _write_csv(tmp_path / "primers.csv", [record])
    fasta = tmp_path / "targets.fa"
    fasta.write_text(f">T1\n{_synthetic_target()}\n>T2\n{_synthetic_target()}\n")

    out = subprocess.run(
        [sys.executable, "-m", "lamp_design.cli", "evaluate",
         "--in", str(path), "--targets", str(fasta),
         "--out", str(tmp_path / "run"), "--device", "cpu"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rows = list(csv.DictReader((tmp_path / "run.predictions.csv").open()))
    assert [r["target"] for r in rows] == ["T1", "T2"]
    # Perfect placements, so no mismatches anywhere.
    assert all(r["total_mismatches"] == "0" for r in rows)
    # No truth column in this input, so no metrics.
    assert json.loads((tmp_path / "run.metrics.json").read_text())["metrics"] == {}


def test_cli_evaluate_needs_targets_when_the_csv_has_no_windows(tmp_path):
    _weights_or_skip()
    record = {"primer_set": "PS01"}
    for ptype in PRIMER_TYPE_ORDER:
        record[f"{ptype}_sequence"] = _PRIMERS[ptype]
    path = _write_csv(tmp_path / "primers.csv", [record])
    out = subprocess.run(
        [sys.executable, "-m", "lamp_design.cli", "evaluate",
         "--in", str(path), "--out", str(tmp_path / "run")],
        capture_output=True, text=True,
    )
    assert out.returncode == 2
    assert "--targets is required" in out.stderr


def test_evaluate_is_deterministic_across_two_runs(tmp_path):
    """Two identical invocations must agree exactly, or ranking is unstable."""
    _weights_or_skip()
    path = _write_csv(
        tmp_path / "in.csv", [_row(), _row(primer_set="PS02", master_mix="SS")]
    )

    def predictions(tag: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "lamp_design.cli", "evaluate",
             "--in", str(path), "--out", str(tmp_path / tag), "--device", "cpu"],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
        return [
            r["predicted_tp_minutes"]
            for r in csv.DictReader((tmp_path / f"{tag}.predictions.csv").open())
        ]

    assert predictions("a") == predictions("b")


def test_cross_product_mode_scores_every_requested_condition(tmp_path):
    _weights_or_skip()
    path = _write_csv(tmp_path / "in.csv", [_row()])
    out = subprocess.run(
        [sys.executable, "-m", "lamp_design.cli", "evaluate",
         "--in", str(path), "--out", str(tmp_path / "run"), "--device", "cpu",
         "--master-mix", "BST", "--master-mix", "SS", "--conc", "10000"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rows = list(csv.DictReader((tmp_path / "run.predictions.csv").open()))
    assert {r["master_mix"] for r in rows} == {"BST", "SS"}
    # The measurement belongs to one condition only; it must not be
    # duplicated onto the others as if it had been measured there too.
    assert sum(1 for r in rows if r["measured_tp_minutes"] not in ("", "NA")) == 1


def test_evaluate_and_design_agree_on_the_same_primer_set(tmp_path):
    """A set scored via `evaluate` must get `design`'s number, not a near miss.

    This is the contract that makes `evaluate` trustworthy: it shares the
    thermo and inference path, so the only way the two can disagree is if the
    window or ordering conventions have drifted apart.
    """
    weights = _weights_or_skip()
    from lamp_design.commands.design import design
    from lamp_design.commands.evaluate import evaluate as evaluate_rows
    from lamp_design.models.inference import LyraEnsemble
    from lamp_design.types import Target, TargetPanel

    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACGT"), size=400))
    panel = TargetPanel(targets=[Target("T1", sequence)], consensus=sequence)
    ensemble = LyraEnsemble.load(weights, device="cpu")
    condition = Condition("BST", 10_000.0)

    designed = design(
        panel,
        ensemble=ensemble,
        conditions=[condition],
        ranking_condition=condition,
        stride=37,
        n=3,
        device="cpu",
    )
    if not designed.scored_sets:
        pytest.skip("no candidates survived on this random sequence")

    best = designed.scored_sets[0]
    # Rebuild that set as an evaluate row, windows taken from the target the
    # design run scored it against.
    from lamp_design.thermo import WindowResolver

    resolver = WindowResolver.from_panel(panel)
    primers, windows = [], []
    for ptype in PRIMER_TYPE_ORDER:
        primer = best.primer_set.primers[ptype]
        window = resolver.window(0, primer.start, primer.end)
        assert window is not None
        primers.append(primer.sequence)
        windows.append(window if primer.is_forward_strand else revcomp(window))

    scored = evaluate_rows(
        [EvalRow(best.primer_set.set_id, "T1", primers, windows, condition)],
        ensemble=ensemble,
        device="cpu",
    )
    assert scored.predicted_tp[0] == pytest.approx(
        best.per_target_tp["T1"], abs=1e-4
    )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _subparser(name):
    """The real CLI's subparser for `name`, not a hand-built stand-in."""
    from lamp_design.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if getattr(action, "choices", None) and name in action.choices:
            return action.choices[name]
    raise AssertionError(f"no subcommand {name!r}")


def _flags(subparser):
    return {flag for action in subparser._actions for flag in action.option_strings}


def test_design_has_no_model_flag():
    """Fusion must be unreachable from `design`, not merely discouraged.

    Measured: Fusion is ~29x slower per prediction, and a stride-1 design run
    projects to ~1.9 h on GPU and ~14 days on CPU. The option does not exist
    there -- not a cap, not a warning -- so nobody can start that by accident.
    """
    assert "--model" not in _flags(_subparser("design"))


def test_evaluate_offers_both_models_and_defaults_to_simplified():
    action = next(
        a for a in _subparser("evaluate")._actions if "--model" in a.option_strings
    )
    assert set(action.choices) == {"simplified", "fusion"}
    assert action.default == "simplified"


def test_an_unknown_model_is_rejected_before_any_work():
    """Cheap to check, and the alternative is a confusing late failure."""
    from lamp_design.commands.evaluate import evaluate
    from lamp_design.commands.prepare import InputError

    with pytest.raises(InputError, match="unknown model"):
        evaluate([], model="tabpfn-only")


def test_fusion_on_cpu_warns_before_the_wait():
    """Five hours of silence is indistinguishable from a hang.

    TabPFN's cost is the 34,283-row context it re-encodes on every call, not
    the number of rows scored: one CPU forward pass measured 4,587 s, and
    there are four estimators. So a CPU run takes roughly five hours whether
    the user supplies three primer sets or three hundred -- the usual
    intuition that a small input means a quick run is wrong here, which is
    exactly why it needs saying.
    """
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "lamp_design" / "commands" / "evaluate.py"
    ).read_text()
    block = source[source.index('if model == "fusion":'):]
    block = block[: block.index("fusion = load_fusion")]
    assert 'not device.startswith("cuda")' in block
    assert "FIVE HOURS" in block
    assert "not recommended" in block
    # Both escape hatches must be named, or the warning is just bad news.
    assert "--device cuda" in block and "--model simplified" in block
    # Printed, not logged: a single log line scrolls past.
    assert "print(line" in block
