"""Tests for the Streamlit front end's importable modules.

``deploy/streamlit/`` is an application, not a package: ``streamlit run`` puts
the script's directory on ``sys.path``, which is what makes ``import _backend``
work inside it. These tests reproduce that one condition and then import the
modules directly. The module names are distinctive enough (``_engine_api``,
``_validation``, ``_backend``, ``_tables``) that putting that directory on the
path cannot shadow anything else in the suite.

Only the layers that do not import Streamlit are covered here, plus one
``AppTest`` smoke test that is skipped when the optional ``streamlit`` extra is
absent. Nothing here runs the model: the pipeline itself is covered by
``test_design_command.py`` and ``test_evaluate.py``.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "deploy" / "streamlit"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import _backend  # noqa: E402
import _engine_api  # noqa: E402
import _tables  # noqa: E402
import _validation  # noqa: E402

PRIMER_TYPES = _engine_api.PRIMER_TYPE_ORDER


def target_fasta(n_seqs: int = 2, length: int = 400) -> str:
    """Equal-length ungapped records, which prepare accepts without MAFFT."""
    return "".join(
        f">seq{i}\n" + ("ACGT" * (length // 4))[:length] + f"{'A' * 0}\n"
        for i in range(n_seqs)
    )


# ---------------------------------------------------------------------------
# Target FASTA validation
# ---------------------------------------------------------------------------


def test_a_well_formed_fasta_is_accepted_with_its_shape_reported():
    check = _validation.validate_fasta(">a\n" + "ACGT" * 100 + "\n>b\n" + "ACGA" * 100 + "\n")
    assert check.ok
    assert check.n_seqs == 2
    assert check.max_len == 400
    assert check.budget == 800
    assert not check.ragged


def test_a_bare_sequence_is_rejected_as_not_fasta():
    check = _validation.validate_fasta("ACGTACGT")
    assert not check.ok
    assert "Not FASTA" in check.errors[0]


def test_ambiguity_codes_are_rejected_rather_than_cleaned():
    """RNAduplex and the model encoder accept only A/C/G/T, so N is an error."""
    check = _validation.validate_fasta(">a\n" + "ACGN" * 100 + "\n")
    assert not check.ok
    assert "other than A, C, G and T" in check.errors[0]


def test_gaps_are_rejected_unless_the_input_is_declared_aligned():
    gapped = ">a\n" + "ACG-" * 100 + "\n>b\n" + "ACGT" * 100 + "\n"
    assert "alignment gaps" in _validation.validate_fasta(gapped).errors[0]
    assert _validation.validate_fasta(gapped, pre_aligned=True).ok


def test_duplicate_sequence_names_are_rejected():
    """Per-target results are keyed by name, so duplicates would collide."""
    check = _validation.validate_fasta(">a\n" + "ACGT" * 100 + "\n>a\n" + "ACGA" * 100 + "\n")
    assert not check.ok
    assert "Duplicate sequence names" in check.errors[-1]


def test_more_sequences_than_prepare_accepts_is_rejected_with_the_engine_limit():
    text = "".join(f">s{i}\n" + "ACGT" * 100 + "\n" for i in range(_engine_api.MAX_SEQUENCES + 1))
    check = _validation.validate_fasta(text)
    assert not check.ok
    assert str(_engine_api.MAX_SEQUENCES) in check.errors[0]


def test_input_over_the_budget_cap_is_rejected():
    length = _engine_api.MAX_BUDGET  # one sequence already over the cap
    check = _validation.validate_fasta(">a\n" + "ACGT" * (length // 4 + 1) + "\n")
    assert not check.ok
    assert "too large" in check.errors[0]


def test_a_short_sequence_is_a_warning_because_prepare_excludes_it():
    short = "ACGT" * 10
    long = "ACGT" * 100
    check = _validation.validate_fasta(f">short\n{short}\n>long\n{long}\n")
    assert check.ok
    assert any("below the engine's" in w for w in check.warnings)


def test_every_sequence_too_short_is_an_error_not_a_warning():
    check = _validation.validate_fasta(">a\n" + "ACGT" * 10 + "\n")
    assert not check.ok
    assert "no primer set can be placed" in check.errors[-1]


def test_sequences_of_different_lengths_are_reported_as_needing_an_aligner():
    check = _validation.validate_fasta(">a\n" + "ACGT" * 100 + "\n>b\n" + "ACGT" * 120 + "\n")
    assert check.ok
    assert check.ragged


def test_an_upload_over_the_byte_cap_is_rejected_before_parsing():
    oversized = ">a\n" + "A" * (_validation.MAX_UPLOAD_BYTES + 1) + "\n"
    check = _validation.validate_fasta(oversized)
    assert not check.ok
    assert "over the" in check.errors[0]


# ---------------------------------------------------------------------------
# Primer-set validation, against `commands/evaluate.py`'s input contract
# ---------------------------------------------------------------------------


def primer_row(set_id: str = "set1") -> dict[str, str]:
    return {t: "ACGT" * 5 for t in PRIMER_TYPES} | {"primer_set": set_id}


def primer_table(*, suffix: str = "_sequence", extra: str = "") -> str:
    header = ["primer_set"] + [f"{t}{suffix}" for t in PRIMER_TYPES]
    row = ["set1"] + ["ACGT" * 5 for _ in PRIMER_TYPES]
    if extra:
        header.append(extra)
        row.append("12.5")
    return ",".join(header) + "\n" + ",".join(row) + "\n"


@pytest.mark.parametrize("suffix", ["_sequence", ""])
def test_both_primer_column_spellings_the_engine_accepts_are_read(suffix):
    check = _validation.validate_primer_sets(primer_table(suffix=suffix))
    assert check.ok, check.errors
    assert check.source_format == "table"
    assert check.n_sets == 1
    assert set(check.sets[0][1]) == set(PRIMER_TYPES)


def test_a_table_missing_a_primer_column_names_the_missing_type():
    table = primer_table()
    table = table.replace("B3_sequence,", "").replace(",ACGT" * 1, "", 1)
    check = _validation.validate_primer_sets(table)
    assert not check.ok
    assert "B3" in check.errors[0]


def test_optional_columns_are_detected_so_the_page_can_report_them():
    check = _validation.validate_primer_sets(primer_table(extra="tp"))
    assert check.optional_columns["set name"] == "primer_set"
    assert check.optional_columns["measured Tp"] == "tp"
    assert not check.has_window_columns


def test_known_target_windows_are_detected():
    header = (
        ["primer_set"]
        + [f"{t}_sequence" for t in PRIMER_TYPES]
        + [f"{t}_target_sequence" for t in PRIMER_TYPES]
    )
    row = ["set1"] + ["ACGT" * 5 for _ in PRIMER_TYPES] * 2
    check = _validation.validate_primer_sets(",".join(header) + "\n" + ",".join(row) + "\n")
    assert check.ok, check.errors
    assert check.has_window_columns


def test_primer_sets_can_be_given_as_fasta_and_convert_to_the_engines_csv():
    fasta = "".join(f">set1_{t}\n{'ACGT' * 5}\n" for t in PRIMER_TYPES)
    check = _validation.validate_primer_sets(fasta)
    assert check.ok, check.errors
    assert check.source_format == "fasta"

    csv_text = _validation.primer_sets_to_csv(check.sets)
    header = csv_text.splitlines()[0].split(",")
    assert header == ["primer_set", *PRIMER_TYPES]
    # Round-trips: the CSV it emits is readable as the table shape.
    assert _validation.validate_primer_sets(csv_text).ok


def test_a_fasta_record_without_a_primer_type_suffix_is_rejected():
    check = _validation.validate_primer_sets(">set1_F9\nACGTACGT\n")
    assert not check.ok
    assert "does not end in a primer type" in check.errors[0]


def test_a_set_missing_primers_names_them():
    fasta = "".join(f">set1_{t}\n{'ACGT' * 5}\n" for t in PRIMER_TYPES[:6])
    check = _validation.validate_primer_sets(fasta)
    assert not check.ok
    assert "missing 2 of the eight primers" in check.errors[0]


def test_degenerate_primers_are_rejected():
    fasta = "".join(f">set1_{t}\n{'ACGN' * 5}\n" for t in PRIMER_TYPES)
    check = _validation.validate_primer_sets(fasta)
    assert not check.ok
    assert "other than A, C, G and T" in check.errors[0]


def test_an_unusual_primer_length_is_a_warning_not_a_rejection():
    fasta = "".join(f">set1_{t}\n{'ACGT' * 15}\n" for t in PRIMER_TYPES)
    check = _validation.validate_primer_sets(fasta)
    assert check.ok
    assert any("outside the" in w for w in check.warnings)


# ---------------------------------------------------------------------------
# Requests and the envelope contract
# ---------------------------------------------------------------------------


def test_conditions_are_the_mixes_major_cross_product_the_engine_builds():
    from lamp_design.commands.design import build_conditions as engine_build

    mixes, concentrations = ["BST", "SS"], [10_000.0, 100.0]
    ours = _backend.build_conditions(mixes, concentrations)
    theirs = engine_build(mixes, concentrations)
    assert [(c["master_mix"], c["target_conc"]) for c in ours] == [
        (c.master_mix, c.target_conc) for c in theirs
    ]


def test_a_design_request_also_carries_the_ranking_condition_on_its_own():
    """deploy/modal/app.py's _validate_request takes one master_mix/target_conc."""
    request = _backend.build_design_request(
        run_id="r",
        fasta=">a\nACGT\n",
        master_mixes=["BST", "SS"],
        concentrations=[10_000.0, 100.0],
        rank_master_mix="SS",
        rank_conc=100.0,
        n_sets=20,
        stride=5,
        n_geometries=4,
        tp_threshold_minutes=30.0,
        min_separation=30,
        pre_aligned=False,
        device="auto",
        threads=0,
    )
    assert len(request["conditions"]) == 4
    assert request["master_mix"] == "SS"
    assert request["target_conc"] == 100.0


def test_the_public_request_never_carries_the_users_sequences():
    request = _backend.build_evaluate_request(
        run_id="r",
        fasta=">a\nACGT\n",
        primer_sets_text="primer_set,F3\nx,ACGT\n",
        primer_sets_filename="p.csv",
        master_mixes=["BST"],
        concentrations=[10_000.0],
        rank_master_mix="BST",
        rank_conc=10_000.0,
        tp_threshold_minutes=30.0,
        pre_aligned=False,
        device="auto",
        threads=0,
    )
    public = _backend.public_request(request)
    assert "fasta" not in public
    assert "primer_sets_text" not in public
    assert public["workflow"] == "evaluate"


def test_the_envelope_schema_matches_the_modal_harness():
    """One wire format for both backends, so the UI code is identical."""
    modal_app = (APP_DIR.parent / "modal" / "app.py").read_text()
    assert f'ENVELOPE_SCHEMA = "{_backend.ENVELOPE_SCHEMA}"' in modal_app


def test_the_remote_concentration_cap_matches_the_modal_side():
    """``_layout`` warns using its own copy of the cap; it must not drift.

    The front end cannot import ``deploy/modal/app.py`` (that would make
    ``modal`` a dependency of the UI), so the number is duplicated. If the
    enforcement point changes and the warning does not, the UI promises a
    sweep the server will refuse.
    """
    import re

    import _layout

    modal_app = (APP_DIR.parent / "modal" / "app.py").read_text()
    match = re.search(
        r'MAX_CONCENTRATIONS = int\(os\.environ\.get\(\s*"LAMP_DESIGN_MAX_CONCENTRATIONS",\s*"(\d+)"',
        modal_app,
    )
    assert match, "could not find MAX_CONCENTRATIONS in deploy/modal/app.py"
    assert _layout._MODAL_MAX_CONCENTRATIONS == int(match.group(1))


def test_a_design_request_carries_the_whole_concentration_sweep():
    """A remote run must score every concentration, not just the ranking one.

    This regressed in the other direction once: the Modal side was taught to
    accept ``target_concs`` while the request builder still sent only
    ``target_conc``, so a Modal run silently returned a one-condition table
    where local returned two.
    """
    request = _backend.build_design_request(
        run_id="20260101-000000-abcdef12",
        fasta=">a\n" + "ACGT" * 50 + "\n",
        master_mixes=["SS"],
        concentrations=[10_000.0, 100.0],
        rank_master_mix="SS",
        rank_conc=10_000.0,
        n_sets=20,
        stride=20,
        n_geometries=1,
        tp_threshold_minutes=60.0,
        min_separation=30,
        pre_aligned=False,
        device="auto",
        threads=None,
    )
    assert request["target_concs"] == [10_000.0, 100.0]
    assert request["rank_conc"] == 10_000.0
    # The ranking condition is still carried alone, for the older shorthand.
    assert request["target_conc"] == 10_000.0
    assert request["rank_conc"] in request["target_concs"], (
        "the ranking concentration has to be one of the scored ones, or the "
        "ranking refers to a prediction that was never made"
    )


def test_the_backend_is_chosen_by_environment_and_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LAMP_DESIGN_BACKEND", raising=False)
    assert _backend.active_backend() == "local"
    monkeypatch.setenv("LAMP_DESIGN_BACKEND", "modal")
    assert _backend.active_backend() == "modal-single"
    assert _backend.modal_function_name("modal-single") == _backend.MODAL_FN_SINGLE
    monkeypatch.setenv("LAMP_DESIGN_BACKEND", "modal-fanout")
    assert _backend.modal_function_name(_backend.active_backend()) == _backend.MODAL_FN_FANOUT
    monkeypatch.setenv("LAMP_DESIGN_BACKEND", "nonsense")
    assert _backend.active_backend() == "local"


def test_run_ids_are_unguessable_and_unique(monkeypatch, tmp_path):
    ids = {_backend.new_run_id() for _ in range(200)}
    assert len(ids) == 200
    for run_id in ids:
        stamp, _, suffix = run_id.rpartition("-")
        assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)
        assert len(stamp) == len("20260101-000000")


def test_each_run_gets_its_own_private_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(_backend, "RUNS_DIR", tmp_path / "runs")
    first = _backend.prepare_run("a")
    second = _backend.prepare_run("b")
    try:
        assert first.out_dir != second.out_dir
        assert first.work_dir != second.work_dir
        # 0700: mkdir(mode=...) is masked by umask, and a shared host may have
        # other local users who should not read a session's sequences.
        assert oct(first.out_dir.stat().st_mode)[-3:] == "700"
    finally:
        for paths in (first, second):
            shutil.rmtree(paths.work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Status and log, the two things the Run page polls
# ---------------------------------------------------------------------------


def test_status_round_trips_and_a_missing_status_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(_backend, "RUNS_DIR", tmp_path / "runs")
    assert _backend.read_status("absent") is None
    _backend.write_status("r", {"state": "running", "stages": []})
    assert _backend.read_status("r")["state"] == "running"


def test_a_truncated_status_file_reads_as_none_rather_than_raising(monkeypatch, tmp_path):
    """The page polls this file while the runner writes it."""
    monkeypatch.setattr(_backend, "RUNS_DIR", tmp_path / "runs")
    _backend.write_status("r", {"state": "running"})
    _backend.status_path("r").write_text('{"state": "run')
    assert _backend.read_status("r") is None


def test_engine_stage_lines_are_scraped_out_of_the_run_log(monkeypatch, tmp_path):
    """How the evaluate workflow reports progress: its CLI owns the callback."""
    monkeypatch.setattr(_backend, "RUNS_DIR", tmp_path / "runs")
    _backend.out_dir_for("r")
    _backend.log_path("r").write_text(
        "Input: 2 row(s) from x.csv\n"
        "  thermo     6 unique (set, target) pairs\n"
        "  inference  12 predictions on cuda\n"
        "  report     12 rows scored\n"
    )
    stages = _backend.stages_from_log("r", ["thermo", "inference", "report"])
    assert stages == {
        "thermo": "6 unique (set, target) pairs",
        "inference": "12 predictions on cuda",
        "report": "12 rows scored",
    }


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------


def design_payload() -> dict:
    """A minimal `report.result_to_dict`-shaped payload."""
    return {
        "panel": {"n_targets": 2, "target_names": ["t1", "t2"]},
        "n_sets_returned": 1,
        "funnel": {"candidates_generated": 10, "scored_sets": 0},
        "warnings": [],
        "excluded_targets": {},
        "ranking": {"tp_threshold_minutes": 30.0},
        "params": {
            "device": "cuda",
            "n_predictions": 100,
            "geometries": [
                "Geometry(geometry_id='canonical', lengths=(20,), gaps=(5,))"
            ],
            "timings_s": {"inference_s": 2.0},
        },
        "sets": [
            {
                "rank": 1,
                "set_id": "canonical@0",
                "anchor": 0,
                "geometry_id": "canonical",
                "span_start": 0,
                "span_end": 175,
                _tables.SPEED_FIELD: 5.0,
                _tables.INCLUSIVITY_FIELD: 1.0,
                "slowest_target_name": "t2",
                "slowest_predicted_tp_minutes": 7.0,
                "total_primer_target_mismatches_all_targets": 3,
                "primers": [
                    {
                        "primer_type": t,
                        "sequence": "ACGT",
                        "start": 0,
                        "end": 4,
                        "length": 4,
                        "strand": "+",
                    }
                    for t in PRIMER_TYPES
                ],
                "per_target": [
                    {"target_name": "t1", "predicted_tp_minutes": 3.0,
                     "primer_target_mismatches": 0},
                    {"target_name": "t2", "predicted_tp_minutes": 7.0,
                     "primer_target_mismatches": 3},
                ],
                "predictions": [
                    {"target_name": "t1", "master_mix": "BST", "target_conc": 10_000.0,
                     "predicted_tp_minutes": 3.0},
                    {"target_name": "t2", "master_mix": "SS", "target_conc": 100.0,
                     "predicted_tp_minutes": 7.0},
                ],
            }
        ],
    }


def test_the_ranked_sets_table_carries_the_eight_primers_as_columns():
    frame = _tables.sets_frame(design_payload())
    assert list(frame.columns)[-8:] == list(PRIMER_TYPES)
    assert frame.loc[0, "mean_tp_min"] == 5.0


def test_the_target_matrices_are_keyed_by_the_panels_target_names():
    payload = design_payload()
    tp = _tables.target_matrix(payload, "predicted_tp_minutes")
    assert list(tp.columns) == ["rank", "set_id", "t1", "t2"]
    assert tp.loc[0, "t2"] == 7.0
    mismatches = _tables.target_matrix(payload, "primer_target_mismatches")
    assert mismatches.loc[0, "t1"] == 0


def test_the_primer_table_has_one_row_per_primer():
    assert len(_tables.primers_frame(design_payload())) == len(PRIMER_TYPES)


def test_the_conditions_table_has_one_column_per_scored_condition():
    frame = _tables.conditions_frame(design_payload())
    assert list(frame.columns) == ["rank", "set_id", "BST@10,000", "SS@100"]


def test_the_funnel_names_the_stage_that_emptied_the_result():
    assert _tables.funnel_culprit(design_payload()) == "scored_sets"
    assert _tables.funnel_culprit({"funnel": {"a": 1}}) is None


def test_geometry_names_are_recovered_from_the_params_repr():
    """`design()` records str(geometry) because Geometry has no `.name`."""
    facts = _tables.run_facts({"result": design_payload(), "timing_seconds": {}})
    assert facts["geometries"] == ["canonical"]


def test_run_facts_reports_the_measured_throughput():
    facts = _tables.run_facts(
        {"result": design_payload(), "timing_seconds": {"inference_s": 2.0, "total": 3.0}}
    )
    assert facts["predictions_per_second"] == 50.0
    assert facts["device"] == "cuda"


def test_an_evaluate_payload_is_recognised_and_tabulated():
    envelope = {
        "run_id": "r",
        "result": {
            "kind": "evaluate",
            "n_scored": 2,
            "metrics": {"amplified": {"n": 2.0, "spearman_rho": 0.5}},
            "predictions": [
                {"primer_set": "s1", "target": "t1", "predicted_tp_minutes": "3.0"},
                {"primer_set": "s1", "target": "t2", "predicted_tp_minutes": "7.0"},
            ],
        },
    }
    assert _tables.is_evaluate(envelope)
    frame = _tables.predictions_frame(envelope["result"])
    assert frame["predicted_tp_minutes"].tolist() == [3.0, 7.0]
    metrics = _tables.evaluate_metrics_frame(envelope["result"])
    assert list(metrics["metric"]) == ["n", "spearman_rho"]
    matrix = _tables.evaluate_matrix(envelope["result"])
    assert list(matrix.columns) == ["primer_set", "t1", "t2"]


# ---------------------------------------------------------------------------
# Engine resolution
# ---------------------------------------------------------------------------


def test_every_engine_entry_point_the_app_calls_resolves():
    missing = [key for key in _engine_api.ENGINE_API if not _engine_api.has_engine(key)]
    assert missing == []


def test_an_unknown_engine_name_fails_naming_what_it_wanted():
    _engine_api.ENGINE_API["nonexistent"] = ("lamp_design.nope", "lamp_design.also_nope")
    try:
        with pytest.raises(NotImplementedError, match="lamp_design.nope"):
            _engine_api.engine("nonexistent")
    finally:
        del _engine_api.ENGINE_API["nonexistent"]


def test_ui_bounds_are_read_from_the_engine_not_hardcoded():
    from lamp_design.commands import prepare
    from lamp_design import report

    assert _engine_api.MIN_TARGET_LENGTH == prepare.DEFAULT_MIN_LENGTH
    assert _engine_api.MAX_SEQUENCES == prepare.MAX_SEQUENCES
    assert _engine_api.DEFAULT_N_SETS == report.DEFAULT_N_SETS
    assert _engine_api.DEFAULT_TP_THRESHOLD == report.DEFAULT_TP_THRESHOLD_MINUTES
    assert _engine_api.DEFAULT_MIN_SEPARATION == report.DEFAULT_MIN_SEPARATION_BP


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_no_deploy_module_ever_changes_the_process_working_directory():
    """`os.chdir` is process-global: in a multi-session Streamlit process it
    would retarget every other in-flight session's cwd, not merely collide on
    one directory. The engine takes explicit paths everywhere, so there is no
    correct use of it here."""
    offenders = []
    for path in sorted((APP_DIR.parent).rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "chdir":
                offenders.append(str(path))
            if isinstance(node, ast.Name) and node.id == "chdir":
                offenders.append(str(path))
    assert offenders == []


def test_no_user_supplied_string_reaches_unsafe_allow_html():
    """Every `unsafe_allow_html=True` call site must be a literal or an
    f-string built only from this app's own constants, never from input."""
    suspicious = []
    for path in sorted(APP_DIR.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not any(
                isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in node.keywords
                if kw.arg == "unsafe_allow_html"
            ):
                continue
            for argument in node.args:
                for inner in ast.walk(argument):
                    if isinstance(inner, ast.Name) and inner.id in (
                        "fasta",
                        "text",
                        "sequence",
                        "name",
                        "message",
                    ):
                        suspicious.append(f"{path.name}:{node.lineno}")
    assert suspicious == []


# ---------------------------------------------------------------------------
# The pages themselves
# ---------------------------------------------------------------------------


def test_the_home_page_renders_and_offers_both_workflows():
    pytest.importorskip("streamlit", reason="the [streamlit] extra is not installed")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert [button.label for button in app.button][:2] == ["Design", "Evaluate"]


def test_choosing_a_workflow_enters_its_first_step():
    pytest.importorskip("streamlit", reason="the [streamlit] extra is not installed")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60).run()
    app.button(key="home_evaluate").click().run()
    assert not app.exception
    assert app.session_state["page"] == "target"
    assert app.session_state["workflow"] == "evaluate"
    assert app.header[0].value == "Evaluate — Target sequences"


def test_the_run_page_blocks_a_run_with_no_input_and_says_why():
    pytest.importorskip("streamlit", reason="the [streamlit] extra is not installed")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60).run()
    app.button(key="home_design").click().run()
    app.session_state["page"] = "run"
    app.run()
    assert not app.exception
    assert any("No target sequences provided" in error.value for error in app.error)
    start = next(button for button in app.button if button.label == "Run Design")
    assert start.disabled


def test_a_result_from_another_session_is_refused():
    pytest.importorskip("streamlit", reason="the [streamlit] extra is not installed")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60).run()
    app.button(key="home_design").click().run()
    app.session_state["page"] = "results"
    app.session_state["envelope"] = {"run_id": "someone-elses", "result": {}}
    app.run()
    assert not app.exception
    assert any("does not belong to this session" in error.value for error in app.error)


def test_the_runner_is_invoked_as_a_subprocess_with_a_json_request(monkeypatch, tmp_path):
    """The local backend submits and returns; it never blocks the session."""
    monkeypatch.setattr(_backend, "RUNS_DIR", tmp_path / "runs")
    captured: dict = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(_backend.subprocess, "Popen", FakePopen)
    request = _backend.build_design_request(
        run_id="r1",
        fasta=">a\nACGT\n",
        master_mixes=["BST"],
        concentrations=[10_000.0],
        rank_master_mix="BST",
        rank_conc=10_000.0,
        n_sets=1,
        stride=5,
        n_geometries=1,
        tp_threshold_minutes=30.0,
        min_separation=0,
        pre_aligned=False,
        device="auto",
        threads=1,
    )
    job = _backend.start_run(request, backend="local")

    assert job["pid"] == 4242
    assert Path(captured["argv"][1]).name == "_runner.py"
    # argv is (python, _runner.py, request.json, work_dir).
    assert json.loads(Path(captured["argv"][2]).read_text())["run_id"] == "r1"
    assert Path(captured["argv"][3]) == Path(job["work_dir"])
    shutil.rmtree(job["work_dir"], ignore_errors=True)
    # Its own process group, so Stop can signal the engine's children too.
    assert captured["kwargs"]["start_new_session"] is True
    assert _backend.read_status("r1")["state"] == "queued"
    assert _backend.poll_run(job)["state"] == "queued"
    # Run metadata lands on disk when the run starts, not only when it ends.
    assert json.loads(_backend.config_path("r1").read_text())["status"] == "running"


# ---------------------------------------------------------------------------
# The shipped demo panel
#
# A first run must not require the user to have a FASTA of their own -- the
# fastest way to lose someone is to open on an empty file picker.
# ---------------------------------------------------------------------------


def test_the_demo_panel_ships_and_parses():
    """The file is a deliverable, so its absence is a packaging failure."""
    import sys

    sys.path.insert(0, str(APP_DIR))
    import app as portal  # noqa: PLC0415

    assert portal.DEMO_PANELS, "at least one demo panel must be offered"
    for label, (relative, blurb) in portal.DEMO_PANELS.items():
        path = portal._demo_root() / relative
        assert path.exists(), f"{label} -> missing {relative}"
        text = path.read_text()
        assert text.startswith(">"), f"{relative} is not FASTA"
        assert blurb.strip(), f"{label} has no description"
        # The engine rejects ambiguity codes, so a demo containing them would
        # fail on the user's first click -- the worst possible place.
        bases = set(
            "".join(
                line.strip().upper()
                for line in text.splitlines()
                if not line.startswith(">")
            )
        )
        assert bases <= set("ACGT"), f"{relative} has non-ACGT: {sorted(bases - set('ACGT'))}"


def test_the_demo_panel_is_within_the_engines_input_limits():
    import sys

    sys.path.insert(0, str(APP_DIR))
    import app as portal  # noqa: PLC0415
    from lamp_design.utils.sequences import read_fasta  # noqa: PLC0415

    for relative, _ in portal.DEMO_PANELS.values():
        path = portal._demo_root() / relative
        records = read_fasta(str(path))
        longest = max(len(seq) for _, seq in records)
        assert len(records) * longest <= _engine_api.MAX_BUDGET, (
            f"{relative} is {len(records)} x {longest} = "
            f"{len(records) * longest:,}, over the {_engine_api.MAX_BUDGET:,} budget"
        )


# ---------------------------------------------------------------------------
# The run-page scope sentence
# ---------------------------------------------------------------------------


def _portal():
    import sys

    sys.path.insert(0, str(APP_DIR))
    import app as portal  # noqa: PLC0415

    return portal


def test_the_scope_sentence_names_candidates_targets_and_concentrations():
    scope = _portal().describe_scope(
        {
            "candidates": 24033,
            "conditions": 2,
            "n_targets": 10,
            "n_mixes": 1,
            "n_concs": 2,
            "predictions": 480660,
            "seconds": 210.0,
        }
    )
    # Rounded on purpose, and counted as the full positions x layouts grid:
    # sets that would overrun the template end are candidates generation
    # rejects, exactly like the GC screen's. Counting them keeps the number
    # predictable (length x layouts) instead of inviting "why isn't it
    # 25,000?". The exact surviving count is in the funnel.
    assert "~24,000" in scope and "candidate primer sets" in scope
    assert "10" in scope and "targets" in scope
    assert "2" in scope and "concentrations" in scope
    assert "~480,000" in scope
    # One master mix must not be announced -- it would read as a parameter the
    # user chose rather than the default it is.
    assert "master mix" not in scope


def test_the_scope_sentence_mentions_mixes_only_when_there_is_more_than_one():
    portal = _portal()
    many = portal.describe_scope(
        {
            "candidates": 100,
            "conditions": 6,
            "n_targets": 3,
            "n_mixes": 3,
            "n_concs": 2,
            "predictions": 1800,
            "seconds": 1.0,
        }
    )
    assert "3** master mixes" in many or "master mixes" in many


def test_the_scope_sentence_singularises():
    scope = _portal().describe_scope(
        {
            "candidates": 5,
            "conditions": 1,
            "n_targets": 1,
            "n_mixes": 1,
            "n_concs": 1,
            "predictions": 5,
            "seconds": 0.1,
        }
    )
    assert "1 target " in scope or "1** target " in scope
    assert "targets" not in scope
    assert "concentrations" not in scope


def test_the_pre_run_candidate_count_is_rounded():
    """Exact-looking estimates invite a question the number cannot answer."""
    round_candidates = _portal()._round_candidates
    assert round_candidates(24033) == "~24,000"
    assert round_candidates(24145) == "~24,000"
    assert round_candidates(480660) == "~480,000"
    # Small counts are exact: there is nothing misleading about "5".
    assert round_candidates(5) == "5"
    assert round_candidates(999) == "999"


def test_the_candidate_estimate_is_length_times_layouts():
    """The number a user can predict, with no edge-case deduction.

    5 kb at stride 1 with 5 layouts per position must read 25,000, not 24,033.
    Sets overrunning the template end are candidates generation rejects, and
    counting them is what makes the figure intuitive.
    """
    portal = _portal()
    assert portal._round_candidates(5000 * 5) == "~25,000"
    assert portal._round_candidates(1751 * 5) == "~9,000"


def test_the_thin_portal_image_excludes_the_model_stack():
    """The thin image's value is what it does NOT contain.

    Asserted on the Dockerfile rather than by building it, so the contract is
    checked in CI without a Docker daemon.
    """
    text = (APP_DIR.parents[1] / "deploy" / "streamlit" / "perf" / "Dockerfile.thin").read_text()
    # Instructions only. The header comments name torch/ViennaRNA/MAFFT in
    # order to explain that they are deliberately absent, so scanning the
    # whole file finds them and proves nothing -- which is how this test
    # failed first time round.
    instructions = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).lower()
    install = instructions[: instructions.index("copy pyproject.toml")]
    for heavy in ("torch", "viennarna", "mafft"):
        assert heavy not in install, f"thin image installs {heavy}"
    # ...must use --no-deps so pip cannot drag torch in behind our back...
    assert "--no-deps" in text
    # ...must assert torch is really absent at build time...
    assert "find_spec('torch') is None" in text
    assert "'torch' not in sys.modules" in text
    # ...and must declare the backend it is the only usable one for.
    assert "LAMP_DESIGN_BACKEND=modal" in text


def test_every_portal_image_honours_the_cloud_run_port_contract():
    """$PORT, 0.0.0.0 and headless, or it will not serve on Cloud Run."""
    for name in ("Dockerfile.portal", "deploy/streamlit/perf/Dockerfile.thin"):
        text = (APP_DIR.parents[1] / name).read_text()
        assert "${PORT:-8080}" in text, f"{name} ignores $PORT"
        assert "--server.address=0.0.0.0" in text, f"{name} does not bind 0.0.0.0"
        assert "STREAMLIT_SERVER_HEADLESS=true" in text, f"{name} is not headless"
