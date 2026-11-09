"""Tests for candidate generation.

The two things that silently corrupt every downstream prediction are strand
assignment and primer ordering, so those are checked against hand-computed
values and against an independent reverse-complement implementation
(Biopython), not against the code under test.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pytest
from Bio.Seq import Seq

from lamp_design.generation import (
    CANONICAL_GAPS,
    CANONICAL_GEOMETRY_ID,
    CANONICAL_LENGTHS,
    DEFAULT_GEOMETRIES,
    DEFAULT_N_SAMPLED,
    GAP_BOUNDS,
    GAP_LABELS,
    GC_BOUNDS,
    LENGTH_BOUNDS,
    Geometry,
    canonical_geometry,
    count_candidate_slots,
    default_geometries,
    derive_geometry_seed,
    generate_candidates,
    random_geometries,
)
from lamp_design.generation import _normal_draws
from lamp_design.types import FORWARD_STRAND_TYPES, PRIMER_TYPE_ORDER

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

#: A geometry made only of 15-mers with canonical spacing, so a whole set fits
#: in a template short enough to write out by hand. Span 135.
MIN15 = Geometry(geometry_id="min15", lengths=(15,) * 8, gaps=CANONICAL_GAPS)

#: Hand-laid template for MIN15 at anchor 0: eight distinct 15-mer windows
#: separated by the canonical 5/0/0/5/0/0/5 spacing, all with GC 7/15 = 0.467.
HAND_WINDOWS = {
    "F3": "AAGCTGTAACGGTCA",
    "F2": "CCGATTGCAAGTCTA",
    "FL": "GGTACCATTGCAAGT",
    "F1c": "CATGGCTAAGTCCAT",
    "B1c": "GGATCCAATGCTAGA",
    "BL": "ACGTTGCAATCCGAT",
    "B2": "TTGCCAAGGTTACCA",
    "B3": "CGTAAGCTTGGACAT",
}
_SPACER = "TTTAA"
HAND_TEMPLATE = (
    HAND_WINDOWS["F3"]
    + _SPACER  # F3-F2 gap, 15:20
    + HAND_WINDOWS["F2"]
    + HAND_WINDOWS["FL"]
    + HAND_WINDOWS["F1c"]
    + _SPACER  # F1c-B1c gap, 65:70
    + HAND_WINDOWS["B1c"]
    + HAND_WINDOWS["BL"]
    + HAND_WINDOWS["B2"]
    + _SPACER  # B2-B3 gap, 115:120
    + HAND_WINDOWS["B3"]
    + "ACGT"  # tail, so anchor 0 is not the only anchor that fits
)


def fixed(template, **kwargs):
    """``generate_candidates`` over the deterministic fixed panel.

    The shipped default is ``canonical`` plus four layouts sampled per anchor
    (the design notes' scheme), so the closed-form counts and
    "every geometry appears" invariants below are about the *fixed* panel,
    which is still a supported mode and still has an exact closed form. Tests
    for the sampled default live in their own section at the end of the file.
    """
    kwargs.setdefault("geometries", DEFAULT_GEOMETRIES)
    kwargs.setdefault("n_sampled", 0)
    return generate_candidates(template, **kwargs)


#: Hand-computed windows at anchor 0 for MIN15.
HAND_COORDS = {
    "F3": (0, 15),
    "F2": (20, 35),
    "FL": (35, 50),
    "F1c": (50, 65),
    "B1c": (70, 85),
    "BL": (85, 100),
    "B2": (100, 115),
    "B3": (120, 135),
}


def all_pass(length: int) -> str:
    """``ACGT`` repeat: pure ACGT and every window's GC fraction near 0.5, so
    no candidate is ever dropped by the cheap screen."""
    return ("ACGT" * (length // 4 + 1))[:length]


def random_template(length: int, seed: int) -> str:
    """Seeded random ACGT, for when two *different* templates are needed."""
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


def _layout_of(primer_set) -> tuple:
    """``(lengths, gaps)`` recovered from a set's primer coordinates.

    Deliberately reconstructed from the output rather than read off a
    Geometry: a sampled set has no Geometry object, and this is exactly the
    route a user has to take to recover one from a result file.
    """
    primers = primer_set.ordered()
    lengths = tuple(p.length for p in primers)
    gaps = tuple(primers[i + 1].start - primers[i].end for i in range(len(primers) - 1))
    return lengths + gaps


def expected_count(template_length: int, geometries, stride: int) -> int:
    """Independent closed form for the unfiltered (anchor x geometry) grid."""
    return sum(
        len(range(0, template_length - geometry.span + 1, stride))
        for geometry in geometries
        if template_length >= geometry.span
    )


def rng_state_fingerprint():
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


def one_set(template: str, geometry: Geometry, anchor: int = 0):
    """The single set produced by ``geometry`` at ``anchor``."""
    sets = [
        candidate
        for candidate in fixed(template, geometries=[geometry])
        if candidate.anchor == anchor
    ]
    assert len(sets) == 1
    return sets[0]


# ---------------------------------------------------------------------------
# Geometry panel
# ---------------------------------------------------------------------------


def test_default_panel_is_five_geometries_with_unique_stable_ids():
    assert len(DEFAULT_GEOMETRIES) == 5
    ids = [geometry.geometry_id for geometry in DEFAULT_GEOMETRIES]
    assert len(set(ids)) == 5
    assert all(isinstance(geometry_id, str) and geometry_id for geometry_id in ids)
    # default_geometries() hands out a copy, not the module constant
    panel = default_geometries()
    assert panel == list(DEFAULT_GEOMETRIES)
    panel.clear()
    assert len(DEFAULT_GEOMETRIES) == 5


def test_default_geometries_stay_inside_the_documented_ranges():
    for geometry in DEFAULT_GEOMETRIES:
        for primer_type, length in zip(PRIMER_TYPE_ORDER, geometry.lengths):
            assert LENGTH_BOUNDS[0] <= length <= LENGTH_BOUNDS[1], (
                geometry.geometry_id,
                primer_type,
            )
        for label, gap, (low, high) in zip(GAP_LABELS, geometry.gaps, GAP_BOUNDS):
            assert low <= gap <= high, (geometry.geometry_id, label, gap)
        assert geometry.gaps[-1] == 0  # unused trailing gap after B3


def test_canonical_geometry_is_present_and_correct():
    assert CANONICAL_GEOMETRY_ID in {g.geometry_id for g in DEFAULT_GEOMETRIES}
    canonical = canonical_geometry()
    assert canonical.geometry_id == CANONICAL_GEOMETRY_ID
    assert canonical.gaps == (5, 0, 0, 5, 0, 0, 5, 0)
    assert set(canonical.lengths) <= {19, 20}
    assert len(canonical.lengths) == len(PRIMER_TYPE_ORDER)
    # Hand-computed from CANONICAL_LENGTHS (the LML per-type medians,
    # 19/20/20/20/19/19/20/20) and CANONICAL_GAPS (5,0,0,5,0,0,5):
    #   F3 19 +5 | F2 20 +0 | FL 20 +0 | F1c 20 +5 | B1c 19 +0 | BL 19 +0
    #   | B2 20 +5 | B3 20
    assert canonical.lengths == CANONICAL_LENGTHS
    assert canonical.windows(0) == [
        ("F3", 0, 19),
        ("F2", 24, 44),
        ("FL", 44, 64),
        ("F1c", 64, 84),
        ("B1c", 89, 108),
        ("BL", 108, 127),
        ("B2", 127, 147),
        ("B3", 152, 172),
    ]
    assert canonical.span == 172


def test_panel_spreads_over_the_design_space():
    """The other four geometries must not all be clones of the canonical one."""
    assert len({geometry.lengths for geometry in DEFAULT_GEOMETRIES}) == 5
    assert len({geometry.gaps for geometry in DEFAULT_GEOMETRIES}) >= 4
    all_lengths = [
        length for geometry in DEFAULT_GEOMETRIES for length in geometry.lengths
    ]
    assert min(all_lengths) <= 17 and max(all_lengths) >= 23
    f1c_b1c_gaps = {geometry.gap_after("F1c") for geometry in DEFAULT_GEOMETRIES}
    assert len(f1c_b1c_gaps) >= 3


def test_geometry_offsets_are_anchor_relative():
    for geometry in DEFAULT_GEOMETRIES:
        base = geometry.windows(0)
        shifted = geometry.windows(137)
        assert [(t, s + 137, e + 137) for t, s, e in base] == shifted
        # layout order, exact gaps, no overlaps
        for index, (primer_type, start, end) in enumerate(base):
            assert primer_type == PRIMER_TYPE_ORDER[index]
            assert end - start == geometry.lengths[index]
            if index:
                previous_end = base[index - 1][2]
                assert start - previous_end == geometry.gaps[index - 1]
                assert start >= previous_end
        assert base[0][1] == 0  # anchor is the F3 start
        assert base[-1][2] == geometry.span


def test_geometry_accepts_seven_gaps_and_normalises_to_eight():
    seven = Geometry(geometry_id="g7", lengths=(19,) * 8, gaps=(5, 0, 0, 5, 0, 0, 5))
    assert seven.gaps == CANONICAL_GAPS
    # 8 x 19 nt plus the three 5 nt canonical gaps. Computed here rather than
    # as an offset from canonical_geometry(), which no longer has uniform
    # lengths -- that coupling is what made this test fail when the canonical
    # geometry moved onto the LML per-type medians.
    assert seven.span == 8 * 19 + 15 == 167


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lengths": (14,) + (19,) * 7, "gaps": CANONICAL_GAPS},  # length below 15
        {"lengths": (26,) + (19,) * 7, "gaps": CANONICAL_GAPS},  # length above 25
        {"lengths": (19,) * 7, "gaps": CANONICAL_GAPS},  # not 8 lengths
        {"lengths": (19,) * 8, "gaps": (4, 0, 0, 5, 0, 0, 5, 0)},  # F3-F2 not 5
        {"lengths": (19,) * 8, "gaps": (5, 0, 0, 5, 0, 0, 4, 0)},  # B2-B3 not 5
        {"lengths": (19,) * 8, "gaps": (5, 11, 0, 5, 0, 0, 5, 0)},  # F2-FL > 10
        {"lengths": (19,) * 8, "gaps": (5, 0, 0, 21, 0, 0, 5, 0)},  # F1c-B1c > 20
        {"lengths": (19,) * 8, "gaps": (5, 0, 0, 5, 0, 0, 5, 3)},  # trailing gap != 0
        {"lengths": (19,) * 8, "gaps": (5, 0, 0, 5, 0, 5)},  # not 7 or 8 gaps
        {"lengths": (19.5,) * 8, "gaps": CANONICAL_GAPS},  # non-integer length
    ],
)
def test_geometry_rejects_out_of_range_layouts(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Geometry(geometry_id="bad", **kwargs)


def test_geometry_rejects_empty_id():
    with pytest.raises(ValueError):
        Geometry(geometry_id="", lengths=(19,) * 8)


# ---------------------------------------------------------------------------
# Frozen conventions: order, strand, reverse complement
# ---------------------------------------------------------------------------


def test_hand_computed_set_matches_frozen_strand_convention():
    """Literal expected sequences for every primer type, no helper in sight."""
    primer_set = one_set(HAND_TEMPLATE, MIN15)

    # forward-strand types store the window verbatim
    assert primer_set.primers["F3"].sequence == "AAGCTGTAACGGTCA"
    assert primer_set.primers["F2"].sequence == "CCGATTGCAAGTCTA"
    assert primer_set.primers["B1c"].sequence == "GGATCCAATGCTAGA"
    assert primer_set.primers["BL"].sequence == "ACGTTGCAATCCGAT"

    # the rest store the reverse complement of their window
    assert primer_set.primers["FL"].sequence == "ACTTGCAATGGTACC"
    assert primer_set.primers["F1c"].sequence == "ATGGACTTAGCCATG"
    assert primer_set.primers["B2"].sequence == "TGGTAACCTTGGCAA"
    assert primer_set.primers["B3"].sequence == "ATGTCCAAGCTTACG"

    # coordinates are forward-strand template coordinates in both cases
    for primer_type, (start, end) in HAND_COORDS.items():
        primer = primer_set.primers[primer_type]
        assert (primer.start, primer.end) == (start, end), primer_type
        assert primer.length == 15
        assert HAND_TEMPLATE[start:end] == HAND_WINDOWS[primer_type]

    assert primer_set.anchor == 0
    assert primer_set.geometry_id == "min15"
    assert primer_set.span == (0, 135)


def test_strand_assignment_matches_forward_strand_types_exactly():
    """Independent oracle (Biopython) over every default geometry."""
    template = all_pass(400)
    candidates = fixed(template, stride=7)
    assert candidates
    seen_types = set()
    for primer_set in candidates:
        for primer in primer_set.ordered():
            window = template[primer.start : primer.end]
            reverse_complement = str(Seq(window).reverse_complement())
            if primer.primer_type in FORWARD_STRAND_TYPES:
                assert primer.sequence == window, primer.primer_type
                assert primer.is_forward_strand
            else:
                assert primer.sequence == reverse_complement, primer.primer_type
                assert not primer.is_forward_strand
            seen_types.add(primer.primer_type)
    assert seen_types == set(PRIMER_TYPE_ORDER)
    assert {"F3", "F2", "B1c", "BL"} == set(FORWARD_STRAND_TYPES)


def test_reverse_complement_is_not_merely_complement_or_reverse():
    """Guards the classic revcomp bugs on an asymmetric window."""
    primer_set = one_set(HAND_TEMPLATE, MIN15)
    window = HAND_WINDOWS["F1c"]
    stored = primer_set.primers["F1c"].sequence
    assert stored != window
    assert stored != window[::-1]
    assert stored != window.translate(str.maketrans("ACGT", "TGCA"))
    assert stored == window.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def test_every_set_is_complete_and_ordered():
    for primer_set in fixed(all_pass(300), stride=11):
        assert set(primer_set.primers) == set(PRIMER_TYPE_ORDER)
        ordered = primer_set.ordered()
        assert [p.primer_type for p in ordered] == list(PRIMER_TYPE_ORDER)
        starts = [p.start for p in ordered]
        assert starts == sorted(starts)
        for previous, current in zip(ordered, ordered[1:]):
            assert current.start >= previous.end  # no overlaps
        assert len(primer_set.sequences()) == len(PRIMER_TYPE_ORDER)
        for primer in ordered:
            assert len(primer.sequence) == primer.length


def test_anchor_is_the_f3_start_and_geometry_does_not_move_it():
    template = all_pass(600)
    by_anchor = {}
    for primer_set in fixed(template, stride=100):
        by_anchor.setdefault(primer_set.anchor, []).append(primer_set)
    for anchor, sets in by_anchor.items():
        assert len({frozenset([s.geometry_id]) for s in sets}) == len(sets)
        for primer_set in sets:
            assert primer_set.primers["F3"].start == anchor
            assert primer_set.span[0] == anchor


# ---------------------------------------------------------------------------
# Determinism and RNG hygiene
# ---------------------------------------------------------------------------


def test_two_calls_give_identical_output():
    template = all_pass(500)
    first = fixed(template, stride=3)
    second = fixed(template, stride=3)
    assert [s.set_id for s in first] == [s.set_id for s in second]
    assert [s.anchor for s in first] == [s.anchor for s in second]
    assert [s.geometry_id for s in first] == [s.geometry_id for s in second]
    assert [s.sequences() for s in first] == [s.sequences() for s in second]
    assert [
        [(p.start, p.end) for p in s.ordered()] for s in first
    ] == [[(p.start, p.end) for p in s.ordered()] for s in second]


def test_generation_does_not_touch_the_global_numpy_rng():
    np.random.seed(123)
    np.random.random(5)  # move off the freshly-seeded state
    before = rng_state_fingerprint()
    candidates = fixed(all_pass(1200), stride=2, max_candidates=100)
    assert candidates
    assert rng_state_fingerprint() == before


def test_global_rng_state_does_not_change_the_result():
    template = all_pass(400)
    np.random.seed(0)
    first = [s.set_id for s in fixed(template)]
    np.random.seed(999)
    second = [s.set_id for s in fixed(template)]
    assert first == second


def test_random_geometries_uses_a_local_rng_only():
    np.random.seed(7)
    np.random.random(3)
    before = rng_state_fingerprint()
    first = random_geometries(4, seed=11)
    assert rng_state_fingerprint() == before
    second = random_geometries(4, seed=11)
    assert first == second
    assert random_geometries(4, seed=12) != first
    assert len({geometry.geometry_id for geometry in first}) == 4
    for geometry in first:
        assert all(
            LENGTH_BOUNDS[0] <= length <= LENGTH_BOUNDS[1]
            for length in geometry.lengths
        )
        assert geometry.gap_after("F3") == 5 and geometry.gap_after("B2") == 5


def test_random_geometries_honours_length_rules():
    geometries = random_geometries(
        6, seed=3, length_rules={primer_type: (19, 20) for primer_type in PRIMER_TYPE_ORDER}
    )
    for geometry in geometries:
        assert set(geometry.lengths) <= {19, 20}


def test_set_ids_are_unique_and_identify_geometry_and_anchor():
    candidates = fixed(all_pass(700), stride=5)
    assert len({s.set_id for s in candidates}) == len(candidates)
    for primer_set in candidates:
        assert primer_set.set_id == f"{primer_set.geometry_id}@{primer_set.anchor}"


# ---------------------------------------------------------------------------
# Anchors, stride, counts, bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2, 3, 7, 50])
def test_anchor_spacing_follows_stride(stride):
    template = all_pass(500)
    geometry = canonical_geometry()
    anchors = [
        s.anchor for s in fixed(template, geometries=[geometry], stride=stride)
    ]
    assert anchors == list(range(0, len(template) - geometry.span + 1, stride))


@pytest.mark.parametrize("stride", [1, 4, 13])
@pytest.mark.parametrize("length", [200, 500, 1000])
def test_candidate_count_matches_the_closed_form(length, stride):
    template = all_pass(length)
    candidates = fixed(template, stride=stride)
    assert len(candidates) == expected_count(length, DEFAULT_GEOMETRIES, stride)
    assert len(candidates) == count_candidate_slots(length, stride=stride)


def test_count_scales_with_template_length_and_inversely_with_stride():
    short = len(fixed(all_pass(1000)))
    long = len(fixed(all_pass(2000)))
    # every geometry gains exactly 1000 anchors
    assert long - short == 1000 * len(DEFAULT_GEOMETRIES)
    strided = len(fixed(all_pass(1000), stride=10))
    assert short / 11 < strided < short / 9


def test_every_default_geometry_is_represented():
    counts = Counter(s.geometry_id for s in fixed(all_pass(1000)))
    assert set(counts) == {g.geometry_id for g in DEFAULT_GEOMETRIES}
    assert all(count > 0 for count in counts.values())


def test_output_is_ordered_by_anchor():
    anchors = [s.anchor for s in fixed(all_pass(400), stride=3)]
    assert anchors == sorted(anchors)


def test_coordinates_stay_in_bounds():
    for length in (170, 171, 260, 261, 999):
        template = all_pass(length)
        for primer_set in fixed(template, stride=1):
            for primer in primer_set.ordered():
                assert 0 <= primer.start < primer.end <= length
                assert primer.end - primer.start == len(primer.sequence)
            assert primer_set.span[1] <= length


def test_template_exactly_one_span_long_yields_one_set():
    geometry = canonical_geometry()
    template = all_pass(geometry.span)
    candidates = fixed(template, geometries=[geometry])
    assert len(candidates) == 1
    assert candidates[0].anchor == 0
    assert candidates[0].span == (0, geometry.span)


@pytest.mark.parametrize("length", [0, 1, 10, 100, 161])
def test_short_templates_return_zero_candidates_without_raising(length):
    assert fixed(all_pass(length)) == []


def test_template_shorter_than_one_span_still_uses_the_others():
    """A 170 bp template fits the three short geometries, not the two long ones."""
    candidates = fixed(all_pass(170))
    used = {s.geometry_id for s in candidates}
    assert used == {"canonical-19", "short-compact"}  # canonical span is 175


# ---------------------------------------------------------------------------
# Cheap screens: ACGT and GC
# ---------------------------------------------------------------------------


def test_gc_poor_template_yields_nothing():
    assert fixed("AT" * 500) == []


def test_gc_rich_template_yields_nothing():
    assert fixed("GC" * 500) == []


def test_balanced_template_yields_candidates():
    assert fixed(all_pass(1000))


def test_gc_bounds_are_inclusive_at_both_ends():
    """Period-5 repeats put every 20-mer at exactly 20% / 80% GC.

    These must be kept: the bounds are inclusive, and an integer GC count
    compared against ``0.2 * 20`` in IEEE-754 floating point is the exact case
    a naive comparison gets wrong.
    """
    # An all-20 geometry, not the canonical one: the exact-boundary arithmetic
    # this test exists for needs every primer length divisible by the repeat
    # period, and canonical is 19/20 mixed (the LML medians).
    geometry = Geometry(geometry_id="all20", lengths=(20,) * 8, gaps=CANONICAL_GAPS)
    assert set(geometry.lengths) == {20}
    low = "AATTG" * 40
    high = "GGCCA" * 40
    assert (low.count("G") + low.count("C")) / len(low) == pytest.approx(GC_BOUNDS[0])
    assert (high.count("G") + high.count("C")) / len(high) == pytest.approx(GC_BOUNDS[1])
    for template in (low, high):
        candidates = fixed(template, geometries=[geometry])
        assert len(candidates) == len(template) - geometry.span + 1
        for primer in candidates[0].ordered():
            gc = sum(base in "GC" for base in primer.sequence) / primer.length
            assert GC_BOUNDS[0] - 1e-9 <= gc <= GC_BOUNDS[1] + 1e-9


def test_gc_outside_the_window_drops_only_the_offending_sets():
    """One AT-only stretch inside F1c's window kills that anchor, not all."""
    start, end = HAND_COORDS["F1c"]
    template = HAND_TEMPLATE[:start] + "A" * (end - start) + HAND_TEMPLATE[end:]
    assert len(template) == len(HAND_TEMPLATE)
    anchors = {s.anchor for s in fixed(template, geometries=[MIN15])}
    assert 0 not in anchors
    all_anchors = {s.anchor for s in fixed(HAND_TEMPLATE, geometries=[MIN15])}
    assert 0 in all_anchors
    assert anchors < all_anchors


def test_non_acgt_inside_a_primer_window_drops_the_set():
    template = "N" + HAND_TEMPLATE[1:]  # position 0 is inside F3 at anchor 0
    anchors = {s.anchor for s in fixed(template, geometries=[MIN15])}
    assert 0 not in anchors


def test_non_acgt_inside_a_gap_is_tolerated():
    """Only primer windows are screened; the inter-primer gaps are not."""
    template = HAND_TEMPLATE[:17] + "N" + HAND_TEMPLATE[18:]  # 15:20 is the F3-F2 gap
    assert len(template) == len(HAND_TEMPLATE)
    anchors = {s.anchor for s in fixed(template, geometries=[MIN15])}
    assert 0 in anchors


@pytest.mark.parametrize("bad", ["N", "-", "n", " ", "X"])
def test_every_non_acgt_character_is_rejected(bad):
    template = all_pass(200)
    position = 10  # inside F3 for the canonical geometry at anchor 0
    poisoned = template[:position] + bad + template[position + 1 :]
    anchors = {
        s.anchor
        for s in fixed(poisoned, geometries=[canonical_geometry()])
    }
    assert 0 not in anchors


def test_vectorized_screen_matches_a_pure_python_screen():
    """The GC/ACGT screen is vectorized for speed; pin it to the naive version.

    A template with sparse ``N`` and a GC-skewed tail, so the comparison is
    non-trivial across all five geometries.
    """
    rng = np.random.default_rng(9)
    template = "".join(
        rng.choice(list("ACGTN"), size=900, p=[0.245, 0.245, 0.245, 0.245, 0.02])
    ) + "GCGCGCGC" * 20 + "ATATATAT" * 20

    expected = set()
    for geometry in DEFAULT_GEOMETRIES:
        for anchor in range(0, len(template) - geometry.span + 1):
            windows = [template[s:e] for _, s, e in geometry.windows(anchor)]
            if all(
                set(window) <= set("ACGT")
                and GC_BOUNDS[0]
                <= (window.count("G") + window.count("C")) / len(window)
                <= GC_BOUNDS[1]
                for window in windows
            ):
                expected.add((geometry.geometry_id, anchor))

    actual = {(s.geometry_id, s.anchor) for s in fixed(template)}
    assert 0 < len(expected) < count_candidate_slots(len(template))  # non-trivial
    assert actual == expected


def test_lowercase_template_is_upper_cased_without_moving_coordinates():
    lower = HAND_TEMPLATE.lower()
    primer_set = one_set(lower, MIN15)
    reference = one_set(HAND_TEMPLATE, MIN15)
    assert primer_set.sequences() == reference.sequences()
    assert [(p.start, p.end) for p in primer_set.ordered()] == [
        (p.start, p.end) for p in reference.ordered()
    ]
    assert all(seq == seq.upper() for seq in primer_set.sequences())


@pytest.mark.parametrize("template", ["AATTG" * 40, "GGCCA" * 40])
def test_no_tm_filter_is_applied(template):
    """Sets far outside any textbook Tm window must survive.

    Tm is a model *input*, not a filter: the PrimerExplorer 59-61 / 64-66 C
    windows exclude essentially all of the training distribution. ``AATTG``
    repeats melt near 40 C and ``GGCCA`` repeats well above 70 C; both are
    inside the permissive GC window, so both must come through.
    """
    candidates = fixed(template, geometries=[canonical_geometry()])
    assert candidates

    primer3 = pytest.importorskip("primer3")
    tms = [
        primer3.calc_tm(primer.sequence, mv_conc=50, dv_conc=8, dna_conc=100)
        for primer in candidates[0].ordered()
    ]
    assert all(not 55.0 <= tm <= 70.0 for tm in tms), tms  # would fail any Tm filter


# ---------------------------------------------------------------------------
# length_rules and max_candidates
# ---------------------------------------------------------------------------


def test_length_rules_filter_the_geometry_panel():
    template = all_pass(1000)
    rules = {primer_type: (19, 20) for primer_type in PRIMER_TYPE_ORDER}
    used = {s.geometry_id for s in fixed(template, length_rules=rules)}
    assert used == {"canonical", "canonical-19"}


def test_partial_length_rules_only_constrain_the_named_types():
    template = all_pass(1000)
    used = {
        s.geometry_id
        for s in fixed(template, length_rules={"F3": (16, 16)})
    }
    assert used == {"short-compact"}  # the only geometry with a 16 nt F3


def test_length_rules_that_exclude_everything_raise():
    with pytest.raises(ValueError, match="excluded every geometry"):
        fixed(all_pass(1000), length_rules={"F3": (15, 15)})


def test_length_rules_reject_unknown_primer_types():
    with pytest.raises(ValueError, match="unknown primer types"):
        fixed(all_pass(400), length_rules={"LF": (19, 20)})


def test_length_rules_reject_inverted_or_impossible_bounds():
    with pytest.raises(ValueError):
        fixed(all_pass(400), length_rules={"F3": (22, 18)})
    with pytest.raises(ValueError, match="permitted length range"):
        fixed(all_pass(400), length_rules={"F3": (30, 40)})


def test_max_candidates_caps_the_result_and_keeps_coverage():
    template = all_pass(2000)
    total = len(fixed(template))
    capped = fixed(template, max_candidates=50)
    assert total > 50
    assert len(capped) == 50
    # still spread across the template and across geometries, not a prefix
    assert max(s.anchor for s in capped) > 0.5 * len(template)
    assert len({s.geometry_id for s in capped}) == len(DEFAULT_GEOMETRIES)
    assert len({s.anchor for s in capped}) > 5


def test_max_candidates_above_the_total_changes_nothing():
    template = all_pass(400)
    full = fixed(template)
    assert [s.set_id for s in fixed(template, max_candidates=10**6)] == [
        s.set_id for s in full
    ]


@pytest.mark.parametrize("cap", [1, 2, 7, 13])
def test_max_candidates_is_exact(cap):
    assert len(fixed(all_pass(1500), max_candidates=cap)) == cap


def test_max_candidates_zero_returns_empty():
    assert fixed(all_pass(1000), max_candidates=0) == []


def test_max_candidates_is_deterministic():
    template = all_pass(3000)
    first = [s.set_id for s in fixed(template, max_candidates=77)]
    second = [s.set_id for s in fixed(template, max_candidates=77)]
    assert first == second


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [0, -1])
def test_stride_must_be_positive(stride):
    with pytest.raises(ValueError, match="stride"):
        fixed(all_pass(400), stride=stride)


def test_empty_geometry_list_is_an_error_not_an_empty_result():
    with pytest.raises(ValueError, match="geometries must not be empty"):
        fixed(all_pass(400), geometries=[])


def test_duplicate_geometry_ids_are_rejected():
    geometry = canonical_geometry()
    with pytest.raises(ValueError, match="duplicate geometry_id"):
        fixed(all_pass(400), geometries=[geometry, geometry])


def test_non_geometry_entries_are_rejected():
    with pytest.raises(TypeError):
        fixed(all_pass(400), geometries=[{"lengths": (19,) * 8}])


def test_negative_max_candidates_is_rejected():
    with pytest.raises(ValueError, match="max_candidates"):
        fixed(all_pass(400), max_candidates=-1)


def test_count_candidate_slots_ignores_the_screens():
    """The funnel's pre-screen count: every grid slot, filtered or not."""
    length = 600
    assert count_candidate_slots(length) == expected_count(
        length, DEFAULT_GEOMETRIES, 1
    )
    # a template that passes nothing still has the same number of slots
    assert len(fixed("AT" * (length // 2))) == 0
    assert count_candidate_slots(length) > 0
    assert count_candidate_slots(10) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_five_kb_template_at_stride_one_is_fast():
    rng = np.random.default_rng(0)
    template = "".join(rng.choice(list("ACGT"), size=5000))
    started = time.perf_counter()
    candidates = fixed(template, stride=1)
    elapsed = time.perf_counter() - started
    assert 20_000 < len(candidates) <= count_candidate_slots(5000)
    assert count_candidate_slots(5000) == 24_036
    assert elapsed < 10.0, f"5 kb stride-1 generation took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# The sampled default (the design notes' scheme)
#
# "for every sampled position, I think we have the compute to sample multiple
# primersets for that position with different spacings and lengths (i.e. let's
# say 5). Because the LML dataset had gaps of (5, 0, 0, 5, 0, 0, 5, 0), I think
# it would be helpful to as a rule have one of those 5 primerset spacings" /
# "for our 'default' primerset at that position ... I'd nominate our median
# primerlengths in the LML datasets, which is 19-20."
# ---------------------------------------------------------------------------


def test_the_default_is_canonical_plus_four_sampled():
    assert DEFAULT_N_SAMPLED == 4
    candidates = generate_candidates(all_pass(1000), stride=1)
    ids = Counter(s.geometry_id for s in candidates)
    assert set(ids) == {
        CANONICAL_GEOMETRY_ID,
        "sampled-0",
        "sampled-1",
        "sampled-2",
        "sampled-3",
    }, "five layouts per position, one of them canonical"


def test_the_canonical_layout_is_always_present_at_every_surviving_anchor():
    """The notes make this a rule, not a tendency."""
    candidates = generate_candidates(all_pass(800), stride=1)
    canonical_anchors = {
        s.anchor for s in candidates if s.geometry_id == CANONICAL_GEOMETRY_ID
    }
    expected = set(range(0, 800 - canonical_geometry().span + 1))
    assert canonical_anchors == expected


def test_the_same_input_reproduces_the_same_sets():
    """Sampling must not cost reproducibility -- the seed comes from the input."""
    template = all_pass(1200)
    first = generate_candidates(template, stride=3)
    second = generate_candidates(template, stride=3)
    assert [s.set_id for s in first] == [s.set_id for s in second]
    assert [p.sequence for s in first for p in s.ordered()] == [
        p.sequence for s in second for p in s.ordered()
    ]


def test_a_different_template_gets_different_layouts():
    """Otherwise the seed is a constant in disguise."""
    a = generate_candidates(all_pass(600), stride=1)
    b = generate_candidates(random_template(600, seed=99), stride=1)
    layouts_a = {_layout_of(s) for s in a if s.geometry_id.startswith("sampled")}
    layouts_b = {_layout_of(s) for s in b if s.geometry_id.startswith("sampled")}
    assert layouts_a != layouts_b


def test_an_explicit_seed_overrides_the_derived_one():
    template = all_pass(600)
    derived = generate_candidates(template, stride=1)
    pinned = generate_candidates(template, stride=1, geometry_seed=12345)
    assert [_layout_of(s) for s in derived] != [_layout_of(s) for s in pinned]
    again = generate_candidates(template, stride=1, geometry_seed=12345)
    assert [s.set_id for s in pinned] == [s.set_id for s in again]


def test_the_global_numpy_rng_is_never_touched():
    """The reference implementation's bug: concurrent runs perturbing each other."""
    np.random.seed(1234)
    before = np.random.get_state()[1][:8].copy()
    generate_candidates(all_pass(900), stride=1)
    after = np.random.get_state()[1][:8]
    assert np.array_equal(before, after)


def test_sampling_explores_far_more_layouts_than_the_fixed_panel():
    """The whole point: coverage of the design space, not five points in it."""
    template = all_pass(3000)
    sampled = generate_candidates(template, stride=1)
    panel = fixed(template, stride=1)
    n_sampled_layouts = len({_layout_of(s) for s in sampled})
    n_panel_layouts = len({_layout_of(s) for s in panel})
    assert n_panel_layouts == 5
    assert n_sampled_layouts > 1000, n_sampled_layouts
    # Comparable candidate counts -- the diversity is free, not bought with
    # extra inference.
    assert 0.8 < len(sampled) / len(panel) < 1.25


def test_every_sampled_layout_obeys_the_declared_bounds():
    """Draws are clipped, not rejected, so the bounds are hard."""
    lo, hi = LENGTH_BOUNDS
    for primer_set in generate_candidates(all_pass(1500), stride=1):
        primers = primer_set.ordered()
        for primer in primers:
            assert lo <= primer.length <= hi, (primer_set.set_id, primer)
        gaps = [
            primers[i + 1].start - primers[i].end for i in range(len(primers) - 1)
        ]
        for label, gap, (gap_lo, gap_hi) in zip(GAP_LABELS, gaps, GAP_BOUNDS):
            assert gap_lo <= gap <= gap_hi, (primer_set.set_id, label, gap)


def test_sampled_sets_stay_inside_the_template():
    """A drawn span is not known until drawn, so bounds are checked after."""
    length = 400
    for primer_set in generate_candidates(all_pass(length), stride=1):
        for primer in primer_set.ordered():
            assert 0 <= primer.start < primer.end <= length


def test_n_sampled_zero_is_purely_deterministic():
    template = all_pass(700)
    one = generate_candidates(template, n_sampled=0, geometries=DEFAULT_GEOMETRIES)
    two = generate_candidates(template, n_sampled=0, geometries=DEFAULT_GEOMETRIES)
    assert [s.set_id for s in one] == [s.set_id for s in two]
    assert {s.geometry_id for s in one} == {
        g.geometry_id for g in DEFAULT_GEOMETRIES
    }


def test_negative_n_sampled_is_an_error():
    with pytest.raises(ValueError, match="n_sampled"):
        generate_candidates(all_pass(400), n_sampled=-1)


def test_no_fixed_geometries_is_allowed_when_sampling():
    """Sampled layouts can carry a run on their own."""
    candidates = generate_candidates(all_pass(700), geometries=[], n_sampled=2)
    assert candidates
    assert {s.geometry_id for s in candidates} == {"sampled-0", "sampled-1"}


def test_no_geometries_at_all_is_still_an_error():
    with pytest.raises(ValueError, match="must not be empty"):
        generate_candidates(all_pass(700), geometries=[], n_sampled=0)


def test_max_candidates_thins_sampled_runs_by_whole_anchors():
    candidates = generate_candidates(all_pass(2000), stride=1, max_candidates=100)
    assert len(candidates) <= 100
    per_anchor = Counter(s.anchor for s in candidates)
    # Every retained anchor keeps its full complement of five layouts, except
    # possibly the last, which the truncation may cut.
    full = [n for anchor, n in sorted(per_anchor.items())][:-1]
    assert full and set(full) == {5}


def test_canonical_lengths_are_the_measured_lml_medians():
    """Recorded so a future edit cannot quietly drift off the data.

    Measured over LMLv1 + LMLv2 (119,928 sets): per-type medians are
    F3 19, F2 20, FL 20, F1c 20, B1c 19, BL 19, B2 20, B3 20.
    """
    assert CANONICAL_LENGTHS == (19, 20, 20, 20, 19, 19, 20, 20)
    assert dict(zip(PRIMER_TYPE_ORDER, CANONICAL_LENGTHS)) == {
        "F3": 19, "F2": 20, "FL": 20, "F1c": 20,
        "B1c": 19, "BL": 19, "B2": 20, "B3": 20,
    }
    assert set(CANONICAL_LENGTHS) <= {19, 20}, "the notes say 19-20"


def test_normal_draws_centre_on_the_midpoint_and_respect_the_range():
    rng = np.random.default_rng(0)
    drawn = _normal_draws(rng, 15, 25, 20_000)
    assert drawn.min() >= 15 and drawn.max() <= 25
    assert abs(drawn.mean() - 20.0) < 0.1
    # ~99.7% within range before clipping means sigma = width / 6.
    assert 1.5 < drawn.std() < 1.9


def test_a_pinned_range_draws_that_constant():
    """GAP_BOUNDS pins F3-F2 and B2-B3 at 5; sigma 0 must not produce NaN."""
    rng = np.random.default_rng(0)
    drawn = _normal_draws(rng, 5, 5, 500)
    assert set(drawn.tolist()) == {5}


def test_derive_geometry_seed_is_stable_and_input_sensitive():
    assert derive_geometry_seed("ACGT" * 50, 1) == derive_geometry_seed("ACGT" * 50, 1)
    assert derive_geometry_seed("ACGT" * 50, 1) != derive_geometry_seed("ACGT" * 50, 2)
    assert derive_geometry_seed("ACGT" * 50, 1) != derive_geometry_seed("ACGA" * 50, 1)
    assert 0 <= derive_geometry_seed("ACGT" * 50, 1) < 2**63
