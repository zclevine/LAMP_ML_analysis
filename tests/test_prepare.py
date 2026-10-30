"""Tests for input preparation, sequence utilities and the stage funnel.

No test here requires MAFFT, RNAduplex, a GPU or the network: the aligner is
injected as a callable, and the only external-tool code path exercised is the
"MAFFT is missing" error.
"""

from __future__ import annotations

import json

import pytest

from lamp_design.commands.prepare import (
    DEFAULT_MAX_BUDGET,
    DEFAULT_MIN_LENGTH,
    MAX_SEQUENCES,
    InputError,
    load_panel,
    panel_to_dict,
    prepare_panel,
    write_panel,
)
from lamp_design.types import DesignResult, TargetPanel
from lamp_design.utils.diagnostics import (
    NO_CANDIDATES_EXIT_CODE,
    NoCandidatesError,
    StageDiagnostics,
    diagnostics_path,
    explain_run,
    first_blocking,
    merge_funnels,
)
from lamp_design.utils.sequences import (
    SequenceValidationError,
    consensus_columns,
    find_degenerate,
    find_gaps,
    find_non_acgt,
    gap_fractions,
    plurality_consensus,
    read_fasta,
    reverse_complement,
    strip_gaps,
    validate_sequence,
    write_fasta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATTERN = "ACGTTGCAGGATCCTAGCATGCAAT"


def make_seq(length: int, shift: int = 0) -> str:
    """Deterministic ACGT sequence; ``shift`` gives a different one, same length."""
    offset = shift % len(_PATTERN)
    pattern = _PATTERN[offset:] + _PATTERN[:offset]
    return (pattern * (length // len(pattern) + 1))[:length]


def substitute(seq: str, position: int, char: str) -> str:
    """Replace the 1-based ``position`` with ``char``."""
    return seq[: position - 1] + char + seq[position:]


class RecordingAligner:
    """Stand-in for MAFFT: right-pads with gaps and records every call."""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, records, threads=1):
        records = list(records)
        self.calls.append((records, threads))
        width = max(len(seq) for _, seq in records)
        return [(name, seq + "-" * (width - len(seq))) for name, seq in records]


# ---------------------------------------------------------------------------
# Sequence count and shape
# ---------------------------------------------------------------------------


class TestPanelShape:
    def test_single_sequence_designs_directly(self):
        seq = make_seq(300)
        prepared = prepare_panel([("solo", seq)])
        panel = prepared.panel

        assert panel.n_targets == 1
        assert panel.targets[0].sequence == seq
        # types.py: alignment is present only for multi-sequence input.
        assert panel.alignment is None
        assert panel.consensus == seq
        assert panel.budget == 300
        assert any("single sequence" in w for w in prepared.warnings)

    def test_multi_sequence_equal_length_is_treated_as_aligned(self):
        records = [(f"s{i}", make_seq(300, shift=i)) for i in range(4)]
        aligner = RecordingAligner()

        prepared = prepare_panel(records, aligner=aligner)
        panel = prepared.panel

        assert aligner.calls == [], "equal-length ungapped input needs no aligner"
        assert panel.n_targets == 4
        assert panel.alignment == [seq for _, seq in records]
        assert panel.consensus == plurality_consensus([s for _, s in records])
        assert panel.budget == 4 * 300
        assert any("treated as already aligned" in w for w in prepared.warnings)

    def test_multi_sequence_unequal_length_is_aligned(self):
        records = [("long", make_seq(300)), ("short", make_seq(250, shift=3))]
        aligner = RecordingAligner()

        prepared = prepare_panel(records, aligner=aligner, threads=4)
        panel = prepared.panel

        assert len(aligner.calls) == 1
        assert aligner.calls[0][1] == 4, "threads are passed through to the aligner"
        assert panel.alignment is not None
        assert len({len(row) for row in panel.alignment}) == 1
        # Targets keep their own ungapped sequences; the gaps live only in the
        # alignment the consensus came from.
        assert [t.sequence for t in panel.targets] == [seq for _, seq in records]
        assert "-" not in panel.consensus
        assert prepared.alignment_names == ["long", "short"]

    def test_force_align_overrides_the_equal_length_shortcut(self):
        records = [(f"s{i}", make_seq(300, shift=i)) for i in range(3)]
        aligner = RecordingAligner()

        prepare_panel(records, aligner=aligner, force_align=True)

        assert len(aligner.calls) == 1

    def test_equal_weighting_for_v1(self):
        records = [(f"s{i}", make_seq(300, shift=i)) for i in range(3)]
        prepared = prepare_panel(records)
        assert [t.weight for t in prepared.panel.targets] == [1.0, 1.0, 1.0]

    def test_empty_input_rejected(self):
        with pytest.raises(InputError, match="no sequences found"):
            prepare_panel([])

    def test_too_many_sequences_rejected(self):
        records = [(f"s{i}", make_seq(300, shift=i)) for i in range(MAX_SEQUENCES + 1)]
        with pytest.raises(InputError) as exc:
            prepare_panel(records)
        assert f"1-{MAX_SEQUENCES}" in str(exc.value)
        assert "--max-budget" in str(exc.value)

    def test_duplicate_names_rejected(self):
        # Per-target results are dicts keyed by name (types.py), so duplicate
        # names would silently overwrite each other's scores.
        records = [("dup", make_seq(300)), ("dup", make_seq(300, shift=1))]
        with pytest.raises(InputError) as exc:
            prepare_panel(records)
        assert "duplicate sequence name" in str(exc.value)
        assert "'dup'" in str(exc.value)


# ---------------------------------------------------------------------------
# Rejections: gaps and ambiguity
# ---------------------------------------------------------------------------


class TestRejections:
    def test_gapped_input_rejected_with_actionable_message(self):
        seq = make_seq(300)
        gapped = seq[:100] + "---" + seq[100:]

        with pytest.raises(SequenceValidationError) as exc:
            prepare_panel([("HBV_AY123456", gapped)])

        message = str(exc.value)
        assert "HBV_AY123456" in message, "names the offending sequence"
        assert "position 101" in message, "names the position (1-based)"
        assert "3 gap character" in message
        assert "--pre-aligned" in message, "says how to proceed deliberately"
        assert exc.value.name == "HBV_AY123456"

    def test_gapped_input_rejected_even_when_other_sequences_are_clean(self):
        clean = make_seq(300)
        with pytest.raises(SequenceValidationError, match="'bad'"):
            prepare_panel([("good", clean), ("bad", clean[:50] + "-" + clean[50:])])

    @pytest.mark.parametrize(
        "code,meaning",
        [("R", "A/G"), ("Y", "C/T"), ("N", "any base"), ("W", "A/T")],
    )
    def test_degenerate_bases_rejected(self, code, meaning):
        seq = substitute(make_seq(300), 88, code)

        with pytest.raises(SequenceValidationError) as exc:
            prepare_panel([("flu_seg4", seq)])

        message = str(exc.value)
        assert "flu_seg4" in message
        assert f"'{code}' ({meaning}) at position 88" in message
        assert "degenerate" in message
        # The reason this is not negotiable, in the message itself.
        assert "A, C, G and T" in message

    def test_degenerate_message_counts_all_and_lists_the_first_few(self):
        seq = make_seq(400)
        for pos in (10, 20, 30, 40, 50):
            seq = substitute(seq, pos, "N")

        with pytest.raises(SequenceValidationError) as exc:
            prepare_panel([("many_ns", seq)])

        message = str(exc.value)
        assert "5 degenerate base(s)" in message
        assert "and 2 more" in message

    def test_degenerate_bases_rejected_even_when_pre_aligned(self):
        seq = substitute(make_seq(300), 12, "R")
        with pytest.raises(SequenceValidationError, match="degenerate"):
            prepare_panel(
                [("a", seq), ("b", make_seq(300, shift=1))], pre_aligned=True
            )

    def test_rna_input_gets_a_u_to_t_hint(self):
        seq = substitute(make_seq(300), 51, "U")
        with pytest.raises(SequenceValidationError) as exc:
            prepare_panel([("rna", seq)])
        assert "This looks like RNA" in str(exc.value)
        assert "position(s) 51" in str(exc.value)

    def test_empty_sequence_rejected(self):
        with pytest.raises(SequenceValidationError, match="is empty"):
            prepare_panel([("blank", "")])


# ---------------------------------------------------------------------------
# Exclusions: recorded, never silent
# ---------------------------------------------------------------------------


class TestExclusions:
    def test_short_sequence_is_excluded_with_a_reason(self):
        records = [("full", make_seq(300)), ("partial", make_seq(97, shift=2))]

        prepared = prepare_panel(records)

        assert [t.name for t in prepared.panel.targets] == ["full"]
        assert "partial" in prepared.excluded
        reason = prepared.excluded["partial"]
        assert "97 nt" in reason and f"{DEFAULT_MIN_LENGTH} nt minimum" in reason
        # Also surfaced as a warning, so a caller watching only warnings sees it.
        assert any("excluded" in w for w in prepared.warnings)

    def test_every_sequence_excluded_is_an_error_not_an_empty_panel(self):
        with pytest.raises(InputError) as exc:
            prepare_panel([("tiny", make_seq(50))])
        assert "every input sequence was excluded" in str(exc.value)
        assert "'tiny'" in str(exc.value), "the reason names the sequence"

    def test_min_length_is_configurable(self):
        prepared = prepare_panel([("tiny", make_seq(50))], min_length=40)
        assert prepared.panel.n_targets == 1
        assert prepared.excluded == {}

    def test_identical_sequences_are_scored_once_and_recorded(self):
        seq = make_seq(300)
        records = [("accession_1", seq), ("accession_2", seq), ("other", make_seq(300, shift=5))]

        prepared = prepare_panel(records)

        assert [t.name for t in prepared.panel.targets] == ["accession_1", "other"]
        assert "accession_2" in prepared.excluded
        assert "identical" in prepared.excluded["accession_2"]
        assert "accession_1" in prepared.excluded["accession_2"]
        # The budget follows the surviving targets, not the input count.
        assert prepared.panel.budget == 2 * 300

    def test_exclusions_reach_the_design_result(self):
        records = [("full", make_seq(300)), ("partial", make_seq(97, shift=2))]
        prepared = prepare_panel(records)

        result = prepared.to_result()

        assert isinstance(result, DesignResult)
        assert "partial" in result.excluded_targets
        assert result.funnel, "the prepare funnel is carried onto the result"
        assert result.warnings


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------


class TestBudgetCap:
    def test_budget_equal_to_the_cap_is_accepted(self):
        records = [(f"s{i}", make_seq(250, shift=i)) for i in range(4)]  # 4 x 250 = 1000

        prepared = prepare_panel(records, max_budget=1000)

        assert prepared.panel.budget == 1000
        assert prepared.diagnostics.details["budget"] == 1000

    def test_budget_one_over_the_cap_is_rejected(self):
        records = [(f"s{i}", make_seq(250, shift=i)) for i in range(4)]

        with pytest.raises(InputError) as exc:
            prepare_panel(records, max_budget=999)

        message = str(exc.value)
        assert "1,000" in message, "states the budget"
        assert "999" in message, "states the cap"
        assert "4 x 250" in message, "shows how the budget is computed"
        # Three concrete ways out.
        assert "keep at most 3 sequence(s)" in message
        assert "at most 249 columns" in message
        assert "--max-budget 1000" in message

    def test_cap_is_on_the_product_not_the_sequence_count(self):
        # 20 short sequences: fine. 3 long ones: over the same cap. That is the
        # whole point of capping n_targets x width.
        short = [(f"s{i}", make_seq(500, shift=i)) for i in range(20)]
        assert prepare_panel(short, max_budget=10_000).panel.budget == 10_000

        long = [(f"g{i}", make_seq(4_000, shift=i)) for i in range(3)]
        with pytest.raises(InputError, match="over the cap"):
            prepare_panel(long, max_budget=10_000)

    def test_budget_is_checked_before_paying_for_an_alignment(self):
        records = [("a", make_seq(4_000)), ("b", make_seq(3_000, shift=1))]
        aligner = RecordingAligner()

        with pytest.raises(InputError) as exc:
            prepare_panel(records, max_budget=5_000, aligner=aligner)

        assert aligner.calls == [], "the cap must be enforced before MAFFT runs"
        assert "estimated from the longest input sequence" in str(exc.value)

    def test_alignment_width_not_input_length_decides_the_final_budget(self):
        # Two 250 nt sequences that align to 300 columns: the budget is 600,
        # over a 550 cap, even though the raw input is only 500 nt.
        records = [("a", make_seq(250)), ("b", make_seq(249, shift=1))]

        def wide_aligner(recs, threads=1):
            return [(name, seq + "-" * (300 - len(seq))) for name, seq in recs]

        with pytest.raises(InputError) as exc:
            prepare_panel(records, max_budget=550, aligner=wide_aligner)

        message = str(exc.value)
        assert "2 x 300" in message
        assert "estimated" not in message, "this is the authoritative post-align check"

    def test_single_oversized_sequence_says_the_count_is_not_the_problem(self):
        with pytest.raises(InputError) as exc:
            prepare_panel([("genome", make_seq(60_000))], max_budget=DEFAULT_MAX_BUDGET)
        assert "even a single sequence of this length" in str(exc.value)

    def test_default_cap_value(self):
        assert DEFAULT_MAX_BUDGET == 50_000


# ---------------------------------------------------------------------------
# Alignment handling
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_missing_mafft_tells_the_user_to_pre_align(self, monkeypatch):
        import lamp_design.external.mafft as mafft

        monkeypatch.setattr(mafft, "mafft_available", lambda: False)
        records = [("a", make_seq(300)), ("b", make_seq(250, shift=1))]

        with pytest.raises(InputError) as exc:
            prepare_panel(records)

        message = str(exc.value)
        assert "MAFFT was not found" in message
        assert "will not guess" in message
        assert "--pre-aligned" in message
        assert "conda install" in message

    def test_single_sequence_never_needs_mafft(self, monkeypatch):
        import lamp_design.external.mafft as mafft

        monkeypatch.setattr(mafft, "mafft_available", lambda: False)
        assert prepare_panel([("solo", make_seq(300))]).panel.n_targets == 1

    def test_pre_aligned_accepts_gaps_and_degaps_the_targets(self):
        row_a = make_seq(200) + "-" * 10 + make_seq(100, shift=1)
        row_b = make_seq(200) + make_seq(10, shift=7) + make_seq(100, shift=1)

        prepared = prepare_panel([("a", row_a), ("b", row_b)], pre_aligned=True)
        panel = prepared.panel

        assert panel.alignment == [row_a, row_b]
        assert len(panel.targets[0].sequence) == 300, "target sequences are ungapped"
        assert len(panel.targets[1].sequence) == 310
        assert "-" not in panel.consensus
        assert panel.budget == 2 * 310, "budget uses the alignment width"

    def test_pre_aligned_requires_a_rectangular_alignment(self):
        with pytest.raises(InputError) as exc:
            prepare_panel(
                [("a", make_seq(300)), ("b", make_seq(299, shift=1))], pre_aligned=True
            )
        assert "--pre-aligned expects an alignment" in str(exc.value)
        assert "300 columns" in str(exc.value)

    def test_gap_rich_columns_are_warned_about(self):
        records = [("a", make_seq(300)), ("b", make_seq(200, shift=1))]
        prepared = prepare_panel(records, aligner=RecordingAligner())
        assert any("gaps" in w for w in prepared.warnings)


# ---------------------------------------------------------------------------
# MAFFT wrapper (mocked: the tests never invoke mafft)
# ---------------------------------------------------------------------------


class TestMafftWrapper:
    def test_find_mafft_reports_how_to_install_it(self, monkeypatch):
        import lamp_design.external.mafft as mafft

        monkeypatch.setattr(mafft.shutil, "which", lambda name: None)
        assert mafft.mafft_available() is False
        with pytest.raises(FileNotFoundError, match="conda install -c bioconda mafft"):
            mafft.find_mafft()

    def test_mafft_available_when_on_path(self, monkeypatch):
        import lamp_design.external.mafft as mafft

        monkeypatch.setattr(mafft.shutil, "which", lambda name: "/opt/conda/bin/mafft")
        assert mafft.mafft_available() is True
        assert mafft.find_mafft() == "/opt/conda/bin/mafft"

    def test_align_records_maps_rows_back_by_id_not_position(self, monkeypatch):
        """MAFFT output is matched by generated id, upper-cased, input-ordered."""
        import lamp_design.external.mafft as mafft

        def fake_align_sequences(input_fasta, output_fasta, threads=1, **kwargs):
            rows = read_fasta(input_fasta)
            width = max(len(seq) for _, seq in rows)
            padded = [(name, (seq + "-" * (width - len(seq))).lower()) for name, seq in rows]
            # MAFFT lower-cases its output; also return the rows out of order
            # to prove the mapping does not rely on position.
            write_fasta(reversed(padded), output_fasta)

        monkeypatch.setattr(mafft, "find_mafft", lambda: "/opt/conda/bin/mafft")
        monkeypatch.setattr(mafft, "align_sequences", fake_align_sequences)

        records = [("beta gamma", make_seq(40)), ("beta delta", make_seq(30, shift=1))]
        aligned = mafft.align_records(records, threads=2)

        assert [name for name, _ in aligned] == ["beta gamma", "beta delta"]
        assert aligned[0][1] == make_seq(40), "upper-cased again"
        assert aligned[1][1] == make_seq(30, shift=1) + "-" * 10
        assert len({len(seq) for _, seq in aligned}) == 1

    def test_align_records_needs_two_sequences(self):
        import lamp_design.external.mafft as mafft

        with pytest.raises(ValueError, match="at least two sequences"):
            mafft.align_records([("solo", make_seq(40))])

    def test_align_records_notices_a_dropped_row(self, monkeypatch):
        import lamp_design.external.mafft as mafft

        def drops_a_row(input_fasta, output_fasta, threads=1, **kwargs):
            write_fasta(read_fasta(input_fasta)[:1], output_fasta)

        monkeypatch.setattr(mafft, "find_mafft", lambda: "/opt/conda/bin/mafft")
        monkeypatch.setattr(mafft, "align_sequences", drops_a_row)

        with pytest.raises(ValueError, match="did not return a row for sequence 'b'"):
            mafft.align_records([("a", make_seq(40)), ("b", make_seq(40, shift=1))])


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


class TestPluralityConsensus:
    def test_plurality_per_column(self):
        assert plurality_consensus(["ACGT", "ACGT", "AGGT"]) == "ACGT"

    def test_gaps_never_vote(self):
        # Column 2 is nine gaps and one C: the C wins, because a gap is not a
        # base. This is the "mostly gaps" case.
        aligned = ["AC" + "GT", "A-" + "GT", "A-" + "GT", "A-" + "GT"]
        assert plurality_consensus(aligned) == "ACGT"

    def test_column_that_is_entirely_gaps_is_dropped(self):
        aligned = ["AC-T", "AC-T", "AC-T"]
        assert plurality_consensus(aligned) == "ACT"
        assert consensus_columns(aligned) == [0, 1, 3]

    def test_ties_break_alphabetically_and_deterministically(self):
        # A vs G tie in column 0, C vs T tie in column 1.
        aligned = ["AC", "GT"]
        assert plurality_consensus(aligned) == "AC"
        assert plurality_consensus(["GT", "AC"]) == "AC", "order of rows is irrelevant"
        assert plurality_consensus(["GC", "TC"]) == "GC", "G beats T"
        assert plurality_consensus(["TC", "GC"]) == "GC"

    def test_never_emits_n_or_an_iupac_code(self):
        # Even a column with four different bases, or with Ns in it, commits to
        # one base: RNAduplex and the encoder cannot take ambiguity.
        aligned = ["ANGT", "CNGT", "GNGT", "TNGT"]
        consensus = plurality_consensus(aligned)
        assert set(consensus) <= set("ACGT")
        assert consensus == "AGT", "the all-N column carries no vote and is dropped"

    def test_single_sequence_consensus_is_itself(self):
        seq = make_seq(50)
        assert plurality_consensus([seq]) == seq

    def test_ragged_alignment_is_an_error(self):
        with pytest.raises(ValueError, match="needs an alignment"):
            plurality_consensus(["ACGT", "ACG"])

    def test_no_sequences_is_an_error(self):
        with pytest.raises(ValueError, match="at least one sequence"):
            plurality_consensus([])

    def test_gap_fractions(self):
        assert gap_fractions(["AC-T", "A--T"]) == [0.0, 0.5, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------


class TestSequenceUtils:
    def test_reverse_complement(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAC") == "GTTT"
        assert reverse_complement("ACGTTGCA") == "TGCAACGT"

    def test_reverse_complement_handles_iupac_and_gaps(self):
        assert reverse_complement("ACGTRY-") == "-RYACGT"
        assert reverse_complement(reverse_complement("ACGTRYN-")) == "ACGTRYN-"

    def test_strip_gaps(self):
        assert strip_gaps("AC-G.T~") == "ACGT"

    def test_fasta_round_trip(self, tmp_path):
        records = [("one", make_seq(120)), ("two", make_seq(80, shift=4))]
        path = tmp_path / "out.fa"

        write_fasta(records, path)

        assert read_fasta(path) == records
        assert path.read_text().startswith(">one\n")

    def test_read_fasta_uppercases_and_keeps_order(self, tmp_path):
        path = tmp_path / "in.fa"
        path.write_text(">b desc here\nacgt\n>a\nTTTT\n")
        assert read_fasta(path) == [("b", "ACGT"), ("a", "TTTT")]

    def test_read_fasta_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_fasta(tmp_path / "nope.fa")

    def test_validation_finders_report_1_based_positions(self):
        assert find_gaps("AC-GT") == [3]
        assert find_degenerate("ACRGT") == [(3, "R")]
        assert find_non_acgt("ACRG-T") == [(3, "R"), (5, "-")]
        assert find_non_acgt("ACRG-T", allow_gaps=True) == [(3, "R")]

    def test_validate_sequence_length_check(self):
        with pytest.raises(SequenceValidationError) as exc:
            validate_sequence("x", make_seq(100), min_length=160)
        assert "100 nt" in str(exc.value) and "160 nt" in str(exc.value)

    def test_validate_sequence_accepts_clean_input(self):
        validate_sequence("x", make_seq(300), min_length=160)


# ---------------------------------------------------------------------------
# Panel serialization
# ---------------------------------------------------------------------------


class TestPanelSerialization:
    def test_round_trip(self, tmp_path):
        records = [("a", make_seq(300)), ("b", make_seq(300, shift=2))]
        prepared = prepare_panel(records)
        path = tmp_path / "panel.json"

        write_panel(prepared, path)
        panel = load_panel(path)

        assert isinstance(panel, TargetPanel)
        assert [t.name for t in panel.targets] == ["a", "b"]
        assert panel.consensus == prepared.panel.consensus
        assert panel.alignment == prepared.panel.alignment
        assert panel.budget == prepared.panel.budget

    def test_dict_carries_exclusions_warnings_and_funnel(self):
        records = [("full", make_seq(300)), ("partial", make_seq(97, shift=1))]
        data = panel_to_dict(prepare_panel(records))

        assert data["schema"] == "lamp-design/panel/1"
        assert "partial" in data["excluded_targets"]
        assert data["warnings"]
        assert data["funnel"]
        assert data["budget"] == 300

    def test_load_rejects_a_foreign_json(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"targets": []}))
        with pytest.raises(InputError, match="not a lamp-design panel file"):
            load_panel(path)


# ---------------------------------------------------------------------------
# The stage funnel
# ---------------------------------------------------------------------------


def _panel() -> TargetPanel:
    seq = make_seq(300)
    return prepare_panel([("solo", seq)]).panel


class TestFunnel:
    def test_prepare_records_a_count_for_every_stage_step(self):
        records = [("full", make_seq(300)), ("partial", make_seq(97, shift=1))]
        funnel = prepare_panel(records).diagnostics.as_funnel()

        assert funnel["prepare: Sequences in input"] == 2
        assert funnel["prepare: Minimum length"] == 1
        assert funnel["prepare: Unique sequences"] == 1
        assert funnel["prepare: Within the input budget"] == 1
        assert all(key.startswith("prepare: ") for key in funnel)

    def test_funnel_identifies_which_filter_emptied_the_stage(self):
        diag = StageDiagnostics("generation", target="panel", unit="primer sets")
        diag.record("Tiling", 4820)
        diag.record(
            "GC fraction",
            0,
            params={"gc_min": 0.2, "gc_max": 0.8},
            hint="Widen the GC window.",
        )
        diag.record("Dimer dG", 0)

        assert diag.empty
        index, step = diag.blocking_step()
        assert index == 1, "the FIRST zero is the culprit, not the last"
        assert step["filter"] == "GC fraction"

        message = diag.message()
        assert "Candidate generation (panel)" in message
        assert "'GC fraction'" in message
        assert "4,820 -> 0" in message
        assert "gc_min=0.2" in message
        assert "Widen the GC window." in message

    def test_a_stage_with_nothing_to_start_with_says_so(self):
        diag = StageDiagnostics("thermo", unit="primer sets")
        diag.record("Candidates received", 0)
        assert "no primer sets to start with at 'Candidates received'" in diag.message()

    def test_non_empty_stage_message(self):
        diag = StageDiagnostics("report", unit="primer sets")
        diag.record("Diversity filter", 20)
        assert not diag.empty
        assert "remained after every filter" in diag.message()

    def test_breakdown_appears_in_the_message(self):
        diag = StageDiagnostics("generation")
        diag.record("Tiling", 10)
        diag.record("Placement", 0, breakdown={"F3": 0, "B3": 5})
        assert "Remaining by type: F3 0, B3 5." in diag.message()

    def test_funnels_merge_across_stages_without_colliding(self):
        prepare_diag = StageDiagnostics("prepare", unit="target sequences")
        prepare_diag.record("Sequences in input", 3)
        gen_diag = StageDiagnostics("generation", unit="primer sets")
        gen_diag.record("Sequences in input", 900)  # same step name, other stage

        funnel = merge_funnels(prepare_diag, gen_diag)

        assert funnel == {
            "prepare: Sequences in input": 3,
            "generation: Sequences in input": 900,
        }

    def test_first_blocking_follows_pipeline_order(self):
        report = StageDiagnostics("report")
        report.record("Diversity", 0)
        generation = StageDiagnostics("generation")
        generation.record("Tiling", 0)
        ok = StageDiagnostics("thermo")
        ok.record("dG", 5)

        assert first_blocking([report, generation, ok]) is generation
        assert first_blocking([ok]) is None

    def test_non_fatal_stage_is_not_blocking(self):
        diag = StageDiagnostics("thermo", fatal=False)
        diag.record("dG", 0)
        assert diag.empty
        assert first_blocking([diag]) is None
        diag.raise_if_empty()  # does not raise

    def test_apply_to_populates_the_design_result_funnel(self):
        diag = StageDiagnostics("generation")
        diag.record("Tiling", 42)
        result = DesignResult(scored_sets=[], panel=_panel())

        diag.apply_to(result)

        assert result.funnel == {"generation: Tiling": 42}

    def test_raise_if_empty_is_the_library_safe_sibling_of_finish(self):
        diag = StageDiagnostics("generation")
        diag.record("Tiling", 100)
        diag.record("GC fraction", 0)

        with pytest.raises(NoCandidatesError) as exc:
            diag.raise_if_empty()

        assert "GC fraction" in str(exc.value)
        assert exc.value.diagnostics is diag

    def test_finish_writes_the_file_and_exits_3_when_empty(self, tmp_path):
        out = tmp_path / "sets.tsv"
        diag = StageDiagnostics("generation")
        diag.record("Tiling", 0)

        with pytest.raises(SystemExit) as exc:
            diag.finish(out)

        assert exc.value.code == NO_CANDIDATES_EXIT_CODE
        data = json.loads(diagnostics_path(out).read_text())
        assert data["empty"] is True
        assert data["stage_label"] == "Candidate generation"

    def test_explain_run_finds_the_earliest_empty_stage(self, tmp_path):
        late = StageDiagnostics("report")
        late.record("Diversity", 0)
        late.write(tmp_path / "report.tsv")
        early = StageDiagnostics("generation")
        early.record("Tiling", 0)
        early.write(tmp_path / "candidates.json")

        found = explain_run(tmp_path)

        assert [d.stage for d in found] == ["generation", "report"]
        assert "Candidate generation" in found[0].message
        assert explain_run(tmp_path / "missing") == []

    def test_prepare_writes_diagnostics_next_to_its_output(self, tmp_path):
        prepared = prepare_panel([("solo", make_seq(300))])
        out = tmp_path / "panel.json"
        write_panel(prepared, out)
        prepared.diagnostics.finish(out)  # not empty: must not exit

        data = json.loads(diagnostics_path(out).read_text())
        assert data["stage"] == "prepare"
        assert data["empty"] is False
        assert data["details"]["budget"] == 300
