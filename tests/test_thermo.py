"""Physics features, checked against the values the model was trained on.

``thermo.py`` had no tests of its own: it was covered only through
``test_end_to_end`` and ``test_parity``, which assert on predictions rather
than on features. That is the wrong place to catch a feature regression,
because the model will happily consume a wrong Tm and return a plausible
number -- as it did. The Tm values here are transcribed from the training
data, so a dependency bump that shifts them fails here, loudly, instead of
silently biasing every prediction the portal makes.
"""

from __future__ import annotations

import numpy as np
import pytest

from lamp_design.thermo import (
    WindowResolver,
    compute_dg,
    compute_thermo,
    compute_tm,
    count_mismatches_vectorized,
    gc_fraction,
    revcomp,
    thermo_from_windows,
)
from lamp_design.types import (
    GLOBAL_PHYSICS_FEATURES,
    PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    Primer,
    PrimerSet,
    Target,
    TargetPanel,
)

#: Known-good Tm values, primer -> degrees C, transcribed from Borrelia.csv in
#: the training data -- the exact column the checkpoints were fit on.
#: Reproduced exactly by primer3-py 2.0.x; primer3-py >= 2.1.0 returns these
#: +2.17 C because it changed the santalucia salt correction. pyproject.toml
#: pins <2.1 for precisely this reason.
_TRAINING_TM = {
    "TGCTAATCTTAGCAAAACTC": 53.83061523880797,
}


class TestMeltingTemperature:
    def test_tm_matches_the_training_data_exactly(self):
        """The guard on the primer3 pin. See the module docstring."""
        primers = list(_TRAINING_TM)
        got = compute_tm(primers)
        for primer, value in zip(primers, got):
            expected = _TRAINING_TM[primer]
            assert value == pytest.approx(expected, abs=1e-3), (
                f"Tm for {primer} is {value}, training data says {expected}. "
                "A difference near 2.17 C means primer3-py >= 2.1.0 is "
                "installed; pyproject.toml pins <2.1 because 2.1.0 changed the "
                "santalucia salt correction and the model was fit on the old "
                "values."
            )

    def test_tm_rises_with_gc_content(self):
        low, high = compute_tm(["ATATATATATATATATATAT", "GCGCGCGCGCGCGCGCGCGC"])
        assert high > low

    def test_tm_is_returned_once_per_input_in_order(self):
        primers = ["ACGTACGTACGTACGTACGT", "TTTTTTTTTTTTTTTTTTTT"]
        got = compute_tm(primers)
        assert got.shape == (2,)
        assert got[0] != got[1]
        assert compute_tm(primers[::-1])[::-1] == pytest.approx(got)


class TestGcFraction:
    @pytest.mark.parametrize(
        "seq,expected",
        [
            ("GCGC", 1.0),
            ("ATAT", 0.0),
            ("ACGT", 0.5),
            ("TGCTAATCTTAGCAAAACTC", 0.35),  # the training row above
        ],
    )
    def test_known_values(self, seq, expected):
        assert gc_fraction(seq) == pytest.approx(expected)

    def test_empty_sequence_does_not_divide_by_zero(self):
        assert gc_fraction("") == 0.0


class TestMismatches:
    def test_counts_elementwise_against_the_window(self):
        got = count_mismatches_vectorized(
            ["ACGTACGT", "ACGTACGT", "ACGTACGT"],
            ["ACGTACGT", "ACGTACGA", "TGCATGCA"],
        )
        assert list(got) == [0, 1, 8]

    def test_ambiguity_in_the_window_counts_as_a_mismatch(self):
        """An N is not evidence that the primer binds."""
        got = count_mismatches_vectorized(["ACGTACGT"], ["ACGTNNNN"])
        assert got[0] == 4

    def test_empty_input_returns_an_empty_array(self):
        assert len(count_mismatches_vectorized([], [])) == 0


class TestDeltaG:
    def test_perfect_duplex_is_more_stable_than_a_mismatched_one(self):
        perfect, mismatched = compute_dg(
            [
                ("TGCTAATCTTAGCAAAACTC", "TGCTAATCTTAGCAAAACTC"),
                ("TGCTAATCTTAGCAAAACTC", "TGCTAATCTTAGCAAAAAAA"),
            ]
        )
        assert perfect < mismatched

    def test_matches_the_training_value(self):
        """One known ΔG from the training CSV, to pin the ViennaRNA call."""
        (dg,) = compute_dg([("TGCTAATCTTAGCAAAACTC", "TGCTAATCTTAGCAAAACTC")])
        assert dg == pytest.approx(-22.7, abs=0.05)

    def test_empty_input_returns_an_empty_result(self):
        assert list(compute_dg([])) == []

    def test_duplicate_pairs_each_get_an_answer(self):
        pair = ("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT")
        got = compute_dg([pair, pair, pair])
        assert len(got) == 3 and len(set(got)) == 1

    def test_backends_agree(self):
        """The in-process bindings and the subprocess must not disagree.

        Both are shipped -- `compute_dg` picks by availability -- so a
        divergence would make results depend on how the container was built.
        """
        pairs = [
            ("TGCTAATCTTAGCAAAACTC", "TGCTAATCTTAGCAAAACTC"),
            ("ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGA"),
            ("GGCCTTAAGGCCTTAAGGCC", "GGCCTTAAGGCCTTAAGGCC"),
        ]
        try:
            bindings = compute_dg(pairs, backend="bindings")
        except Exception:  # pragma: no cover - bindings not built
            pytest.skip("ViennaRNA python bindings unavailable")
        assert bindings == pytest.approx(compute_dg(pairs, backend="subprocess"))


class TestRevcomp:
    def test_known_value(self):
        assert revcomp("ACGTACGT") == "ACGTACGT"
        assert revcomp("AAAA") == "TTTT"
        assert revcomp("TGCTAATCTTAGCAAAACTC") == "GAGTTTTGCTAAGATTAGCA"

    def test_is_its_own_inverse(self):
        seq = "TGCTAATCTTAGCAAAACTC"
        assert revcomp(revcomp(seq)) == seq


class TestThermoFromWindows:
    """The path ``evaluate`` takes, where windows arrive from outside."""

    def _row(self):
        primers = [
            "ACGTACGTACGTACGTACGT", "TTGGCCAATTGGCCAATTGG",
            "AAGGTTCCAAGGTTCCAAGG", "GGCCTTAAGGCCTTAAGGCC",
            "CCAATTGGCCAATTGGCCAA", "TACGATCGTACGATCGTACG",
            "GATTACAGATTACAGATTAC", "CCCGGGAAATTTCCCGGGAA",
        ]
        return primers

    def test_shapes_and_ordering(self):
        primers = self._row()
        table = thermo_from_windows(["s1"], ["t1"], [primers], [primers])
        assert table.set_ids == ["s1"] and table.target_names == ["t1"]
        assert table.values.shape == (1, len(PRIMER_TYPE_ORDER), len(PHYSICS_FEATURES))
        assert table.globals_.shape == (1, len(GLOBAL_PHYSICS_FEATURES))
        # Perfect matches, so every per-primer mismatch count is zero.
        i_mm = PHYSICS_FEATURES.index("mismatches")
        assert table.values[0, :, i_mm].tolist() == [0.0] * len(PRIMER_TYPE_ORDER)

    def test_globals_are_the_sum_and_mean_of_the_per_primer_block(self):
        primers = self._row()
        windows = list(primers)
        windows[0] = "T" + primers[0][1:]  # one mismatch in F3
        table = thermo_from_windows(["s1"], ["t1"], [primers], [windows])
        i_mm = PHYSICS_FEATURES.index("mismatches")
        i_dg = PHYSICS_FEATURES.index("dg_ViennaRNA")
        g_mm = GLOBAL_PHYSICS_FEATURES.index("total_mismatches")
        g_dg = GLOBAL_PHYSICS_FEATURES.index("mean_dg_ViennaRNA")
        assert table.globals_[0, g_mm] == pytest.approx(table.values[0, :, i_mm].sum())
        assert table.globals_[0, g_mm] == pytest.approx(1.0)
        assert table.globals_[0, g_dg] == pytest.approx(
            table.values[0, :, i_dg].mean(), abs=1e-4
        )

    def test_empty_input_returns_a_valid_empty_table(self):
        table = thermo_from_windows([], [], [], [])
        assert table.set_ids == [] and table.values.shape[0] == 0

    def test_a_row_with_the_wrong_number_of_primers_is_rejected(self):
        with pytest.raises(ValueError, match="exactly 8 primers"):
            thermo_from_windows(["s1"], ["t1"], [["ACGT"] * 7], [["ACGT"] * 7])

    def test_dedup_does_not_change_results(self):
        """The (primer, window) dedup must be invisible in the output."""
        primers = self._row()
        one = thermo_from_windows(["s1"], ["t1"], [primers], [primers])
        three = thermo_from_windows(
            ["s1", "s1", "s1"], ["t1", "t2", "t3"], [primers] * 3, [primers] * 3
        )
        for i in range(3):
            assert three.values[i] == pytest.approx(one.values[0])


class TestComputeThermo:
    """The path ``design`` takes, where windows come from the panel."""

    def _panel_and_set(self):
        # A target built so the set's windows are exactly the primers.
        sequence = "ACGT" * 100
        panel = TargetPanel(
            targets=[Target("t1", sequence), Target("t2", sequence)],
            consensus=sequence,
        )
        primers = {}
        cursor = 0
        for ptype in PRIMER_TYPE_ORDER:
            window = sequence[cursor : cursor + 20]
            seq = window if ptype in {"F3", "F2", "B1c", "BL"} else revcomp(window)
            primers[ptype] = Primer(ptype, seq, cursor, cursor + 20)
            cursor += 25
        return panel, PrimerSet("s1", primers, anchor=0, geometry_id="g")

    def test_one_row_per_set_target_pair(self):
        panel, pset = self._panel_and_set()
        table = compute_thermo([pset], panel)
        assert table.set_ids == ["s1", "s1"]
        assert table.target_names == ["t1", "t2"]

    def test_perfect_placement_gives_zero_mismatches(self):
        panel, pset = self._panel_and_set()
        table = compute_thermo([pset], panel)
        i_mm = PHYSICS_FEATURES.index("mismatches")
        assert table.values[:, :, i_mm].sum() == 0.0

    def test_agrees_with_thermo_from_windows_on_the_same_pair(self):
        """Both entry points must produce identical features.

        This is what makes an `evaluate` score comparable to a `design` score;
        without it the two commands could silently diverge.
        """
        panel, pset = self._panel_and_set()
        table = compute_thermo([pset], panel)
        direct = thermo_from_windows(
            ["s1"],
            ["t1"],
            [[pset.primers[p].sequence for p in PRIMER_TYPE_ORDER]],
            [[str(w) for w in table.target_windows[0]]],
        )
        assert direct.values[0] == pytest.approx(table.values[0])
        assert direct.globals_[0] == pytest.approx(table.globals_[0])

    def test_a_gapped_target_drops_the_pair_rather_than_shortening_it(self):
        sequence = "ACGT" * 100
        gapped = sequence[:50] + "-" * 10 + sequence[60:]
        panel = TargetPanel(
            targets=[Target("good", sequence), Target("gapped", gapped)],
            consensus=sequence,
            alignment=[sequence, gapped],
        )
        _, pset = self._panel_and_set()
        table = compute_thermo([pset], panel)
        assert table.target_names == ["good"]

    def test_empty_candidate_list(self):
        panel, _ = self._panel_and_set()
        table = compute_thermo([], panel)
        assert table.set_ids == []


class TestWindowResolver:
    def test_ungapped_panel_degenerates_to_a_slice(self):
        sequence = "ACGTACGTACGTACGT"
        panel = TargetPanel(targets=[Target("t", sequence)], consensus=sequence)
        resolver = WindowResolver.from_panel(panel)
        assert resolver.window(0, 4, 8) == sequence[4:8]

    def test_a_gap_in_the_window_returns_none(self):
        sequence = "ACGTACGTACGTACGT"
        gapped = "ACGT--GTACGTACGT"
        panel = TargetPanel(
            targets=[Target("a", sequence), Target("b", gapped)],
            consensus=sequence,
            alignment=[sequence, gapped],
        )
        resolver = WindowResolver.from_panel(panel)
        assert resolver.window(0, 4, 8) == "ACGT"
        assert resolver.window(1, 4, 8) is None

    def test_running_off_the_end_returns_none(self):
        sequence = "ACGTACGT"
        panel = TargetPanel(targets=[Target("t", sequence)], consensus=sequence)
        resolver = WindowResolver.from_panel(panel)
        assert resolver.window(0, 4, 20) is None

    def test_windows_are_upper_cased(self):
        sequence = "acgtacgtacgtacgt"
        panel = TargetPanel(targets=[Target("t", sequence)], consensus=sequence)
        resolver = WindowResolver.from_panel(panel)
        assert resolver.window(0, 0, 4) == "ACGT"
