"""Candidate generation: enumerate eight-primer LAMP sets across a template.

The generator tiles the template with :class:`PrimerSet` candidates. Every
candidate is the product of two independent choices:

* an **anchor** -- the forward-strand coordinate of the F3 5' end. Anchors are
  taken at ``0, stride, 2*stride, ...``, so position is a single scalar knob.
* a **geometry** -- a fixed ``(lengths, gaps)`` pair covering all eight primer
  types, see :class:`Geometry`.

Because a geometry's eight windows are fixed integer offsets from the anchor,
position and geometry are fully decoupled: swapping geometries never moves the
anchor, and walking the anchor never changes a primer's length. That is the
deliberate break from the reference implementation
(``lamp_pipeline/primer_design/tiling.py``), which drew every primer length
from ``np.random.normal`` inside one cumulative left-to-right walk. There, a
single long draw shifted the seven downstream primers, so candidate coverage
depended on the seed, and -- because the reference seeded the *global* numpy
RNG in ``__init__`` -- two concurrent design runs in one process perturbed each
other. Nothing in this module consumes randomness; the only RNG offered,
:func:`random_geometries`, owns a local ``np.random.default_rng(seed)``.

Screens applied here are cheap and purely sequence-local (requirement: no
thermodynamics in generation -- Tm/ΔG live in ``thermo.py``):

* every primer window must be in bounds (guaranteed by construction),
* every primer window must be pure ACGT,
* every primer must have GC fraction within :data:`GC_BOUNDS` (20-80%).

There is deliberately **no Tm window filter**: Tm is a model input, not a
filter (see ``docs/ARCHITECTURE.md``).

Coordinates and strand follow the frozen conventions in :mod:`lamp_design.types`:
``Primer.start``/``Primer.end`` are always forward-strand template coordinates,
primers in ``FORWARD_STRAND_TYPES`` (F3, F2, B1c, BL) store the forward window
verbatim and the rest store its reverse complement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import FORWARD_STRAND_TYPES, PRIMER_TYPE_ORDER, Primer, PrimerSet

__all__ = [
    "Geometry",
    "CANONICAL_GAPS",
    "CANONICAL_GEOMETRY_ID",
    "DEFAULT_GEOMETRIES",
    "DEFAULT_LENGTH_BOUNDS",
    "DEFAULT_LENGTH_RULES",
    "DEFAULT_STRIDE",
    "GAP_BOUNDS",
    "MIN_DESIGNABLE_LENGTH",
    "GAP_LABELS",
    "GC_BOUNDS",
    "LENGTH_BOUNDS",
    "canonical_geometry",
    "count_candidate_slots",
    "default_geometries",
    "generate_candidates",
    "random_geometries",
]


# ---------------------------------------------------------------------------
# Design-space ranges (team design notes; wider than textbook LAMP on purpose
# -- they match the training set, so do not tighten them here)
# ---------------------------------------------------------------------------

N_PRIMER_TYPES: int = len(PRIMER_TYPE_ORDER)

#: Hard per-primer length bounds, inclusive.
LENGTH_BOUNDS: Tuple[int, int] = (15, 25)

#: Default (canonical-region) per-primer length bounds, inclusive.
DEFAULT_LENGTH_BOUNDS: Tuple[int, int] = (19, 20)

#: Human-readable name of each of the seven inter-primer gaps, in layout order.
GAP_LABELS: Tuple[str, ...] = (
    "F3-F2", "F2-FL", "FL-F1c", "F1c-B1c", "B1c-BL", "BL-B2", "B2-B3",
)

#: Inclusive (min, max) for each gap in :data:`GAP_LABELS`. F3-F2 and B2-B3 are
#: pinned at 5.
GAP_BOUNDS: Tuple[Tuple[int, int], ...] = (
    (5, 5),    # F3-F2
    (0, 10),   # F2-FL
    (0, 10),   # FL-F1c
    (0, 20),   # F1c-B1c
    (0, 10),   # B1c-BL
    (0, 10),   # BL-B2
    (5, 5),    # B2-B3
)

#: Inclusive GC-fraction window every primer must fall inside. Permissive on
#: purpose: it matches the LAMP-ML design space.
GC_BOUNDS: Tuple[float, float] = (0.2, 0.8)

#: The spacing the training data occupies, in :data:`GAP_LABELS` order plus an
#: unused trailing entry so it lines up with ``PRIMER_TYPE_ORDER``.
CANONICAL_GAPS: Tuple[int, ...] = (5, 0, 0, 5, 0, 0, 5, 0)

#: ``geometry_id`` of the canonical geometry.
CANONICAL_GEOMETRY_ID: str = "canonical"

#: Default per-type length rules: the full permitted range. Treat as read-only.
DEFAULT_LENGTH_RULES: Dict[str, Tuple[int, int]] = {
    primer_type: LENGTH_BOUNDS for primer_type in PRIMER_TYPE_ORDER
}

# Integer GC counts are compared against float bounds scaled by the primer
# length; 0.2 * 20 is 4.000000000000001 in IEEE-754, so an exact-boundary
# primer needs a tolerance to be kept.
_GC_TOL: float = 1e-9

_COMPLEMENT = str.maketrans("ACGTUNRYKMSWBVDH", "TGCAANYRMKSWVBHD")

_ACGT_ORDS: Tuple[int, ...] = tuple(ord(base) for base in "ACGT")


def _reverse_complement(sequence: str) -> str:
    """Reverse complement an uppercase IUPAC DNA string.

    Local copy rather than a ``utils.sequences`` import: generation must stay
    dependency-light, and that module belongs to another stage.
    """
    return sequence.translate(_COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """One deterministic spacing/length layout for all eight primers.

    Parameters
    ----------
    geometry_id:
        Stable identifier recorded on every :class:`PrimerSet` this geometry
        produces, so a result can be reproduced without any RNG state.
    lengths:
        Eight primer lengths in ``PRIMER_TYPE_ORDER``.
    gaps:
        Gap *following* each primer, i.e. ``gaps[i]`` separates primer ``i``
        from primer ``i + 1``. Seven values (one per entry of
        :data:`GAP_LABELS`) or eight, where the trailing entry is the unused
        gap after B3 and must be 0. Stored as eight for shape-compatibility
        with ``PRIMER_TYPE_ORDER`` and with the reference implementation.

    The derived ``start_offsets``/``end_offsets`` are offsets from the set's
    anchor (the F3 start), so a set at anchor ``a`` occupies
    ``[a + start_offsets[i], a + end_offsets[i])`` for primer ``i``.
    """

    geometry_id: str
    lengths: Tuple[int, ...]
    gaps: Tuple[int, ...] = CANONICAL_GAPS
    start_offsets: Tuple[int, ...] = field(init=False, repr=False, compare=False)
    end_offsets: Tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_id, str) or not self.geometry_id:
            raise ValueError("geometry_id must be a non-empty string")

        lengths = _as_int_tuple(self.lengths, "lengths")
        gaps = _as_int_tuple(self.gaps, "gaps")

        if len(lengths) != N_PRIMER_TYPES:
            raise ValueError(
                f"{self.geometry_id}: lengths must have {N_PRIMER_TYPES} entries "
                f"(one per primer type), got {len(lengths)}"
            )
        if len(gaps) == N_PRIMER_TYPES - 1:
            gaps = gaps + (0,)
        elif len(gaps) == N_PRIMER_TYPES:
            if gaps[-1] != 0:
                raise ValueError(
                    f"{self.geometry_id}: gaps[{N_PRIMER_TYPES - 1}] is the unused gap "
                    f"after {PRIMER_TYPE_ORDER[-1]} and must be 0, got {gaps[-1]}"
                )
        else:
            raise ValueError(
                f"{self.geometry_id}: gaps must have {N_PRIMER_TYPES - 1} or "
                f"{N_PRIMER_TYPES} entries, got {len(gaps)}"
            )

        lo, hi = LENGTH_BOUNDS
        for primer_type, length in zip(PRIMER_TYPE_ORDER, lengths):
            if not lo <= length <= hi:
                raise ValueError(
                    f"{self.geometry_id}: {primer_type} length {length} outside "
                    f"the permitted range {lo}-{hi}"
                )
        for label, gap, (gap_lo, gap_hi) in zip(GAP_LABELS, gaps, GAP_BOUNDS):
            if not gap_lo <= gap <= gap_hi:
                raise ValueError(
                    f"{self.geometry_id}: {label} gap {gap} outside the permitted "
                    f"range {gap_lo}-{gap_hi}"
                )

        starts: List[int] = []
        ends: List[int] = []
        cursor = 0
        for length, gap in zip(lengths, gaps):
            starts.append(cursor)
            cursor += length
            ends.append(cursor)
            cursor += gap

        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "gaps", gaps)
        object.__setattr__(self, "start_offsets", tuple(starts))
        object.__setattr__(self, "end_offsets", tuple(ends))

    @property
    def span(self) -> int:
        """Template bases a set of this geometry occupies, F3 start to B3 end."""
        return self.end_offsets[-1]

    def length_of(self, primer_type: str) -> int:
        return self.lengths[PRIMER_TYPE_ORDER.index(primer_type)]

    def gap_after(self, primer_type: str) -> int:
        return self.gaps[PRIMER_TYPE_ORDER.index(primer_type)]

    def windows(self, anchor: int = 0) -> List[Tuple[str, int, int]]:
        """``(primer_type, start, end)`` forward-strand windows at ``anchor``."""
        return [
            (primer_type, anchor + start, anchor + end)
            for primer_type, start, end in zip(
                PRIMER_TYPE_ORDER, self.start_offsets, self.end_offsets
            )
        ]

    def fits(self, length_rules: Mapping[str, Tuple[int, int]]) -> bool:
        """True when every primer length satisfies ``length_rules``."""
        for primer_type, length in zip(PRIMER_TYPE_ORDER, self.lengths):
            lo, hi = length_rules.get(primer_type, LENGTH_BOUNDS)
            if not lo <= length <= hi:
                return False
        return True


def _as_int_tuple(values: Sequence[int], name: str) -> Tuple[int, ...]:
    originals = tuple(values)  # materialize first: `values` may be an iterator
    try:
        out = tuple(int(v) for v in originals)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of ints") from exc
    for value, original in zip(out, originals):
        if value != original:
            raise ValueError(f"{name} must be whole numbers, got {original!r}")
    return out


# ---------------------------------------------------------------------------
# The default geometry panel
# ---------------------------------------------------------------------------

#: Five deterministic geometries. ``canonical`` is the spacing/length the
#: training data occupies and where the model is best calibrated; the other
#: four spread over the permitted length and gap ranges so a design run still
#: sees compact and loose amplicons. Order is stable and so are the ids.
DEFAULT_GEOMETRIES: Tuple[Geometry, ...] = (
    # Canonical: (5,0,0,5,0,0,5,0) spacing, every primer 20 nt. Span 175.
    Geometry(
        geometry_id=CANONICAL_GEOMETRY_ID,
        lengths=(20, 20, 20, 20, 20, 20, 20, 20),
        gaps=CANONICAL_GAPS,
    ),
    # Canonical spacing at the short end of the default 19-20 nt window. Span 167.
    Geometry(
        geometry_id="canonical-19",
        lengths=(19, 19, 19, 19, 19, 19, 19, 19),
        gaps=CANONICAL_GAPS,
    ),
    # Short primers, small gaps: the compact corner of the design space. Span 162.
    Geometry(
        geometry_id="short-compact",
        lengths=(16, 17, 17, 18, 18, 17, 17, 16),
        gaps=(5, 2, 2, 8, 2, 2, 5, 0),
    ),
    # Mid lengths, mid gaps, longer F1c-B1c stem spacing. Span 208.
    Geometry(
        geometry_id="mid-wide",
        lengths=(20, 21, 20, 22, 22, 20, 21, 20),
        gaps=(5, 5, 5, 12, 5, 5, 5, 0),
    ),
    # Long primers at the top of every gap range: the loose corner. Span 260.
    Geometry(
        geometry_id="long-wide",
        lengths=(23, 24, 23, 25, 25, 23, 24, 23),
        gaps=(5, 10, 10, 20, 10, 10, 5, 0),
    ),
)


#: Anchor spacing when none is given. 1 evaluates every template position,
#: which is what the design notes specify; larger strides trade coverage for
#: time.
DEFAULT_STRIDE = 1

#: Shortest template that can yield any candidate at all: the narrowest default
#: geometry's span. Derived, so it cannot drift from the geometry table -- a
#: hardcoded floor below this silently accepts inputs that generate nothing.
MIN_DESIGNABLE_LENGTH: int = min(g.span for g in DEFAULT_GEOMETRIES)


def default_geometries() -> List[Geometry]:
    """The five default geometries, as a fresh list."""
    return list(DEFAULT_GEOMETRIES)


def canonical_geometry() -> Geometry:
    """The canonical geometry: canonical spacing, every primer 20 nt."""
    for geometry in DEFAULT_GEOMETRIES:
        if geometry.geometry_id == CANONICAL_GEOMETRY_ID:
            return geometry
    raise AssertionError("DEFAULT_GEOMETRIES must contain the canonical geometry")


def random_geometries(
    n_geometries: int,
    *,
    seed: int = 0,
    length_rules: Optional[Mapping[str, Tuple[int, int]]] = None,
    id_prefix: str = "rng",
) -> List[Geometry]:
    """Sample ``n_geometries`` geometries from a **local** RNG.

    Optional broadening of the default panel. Uses
    ``np.random.default_rng(seed)``, never the global numpy RNG, so calling it
    cannot perturb another design run in the same process and the same ``seed``
    always yields the same geometries. Lengths are drawn uniformly inside
    ``length_rules`` (default :data:`LENGTH_BOUNDS`) and gaps uniformly inside
    :data:`GAP_BOUNDS`, which keeps the two pinned gaps at 5.
    """
    if n_geometries < 0:
        raise ValueError("n_geometries must be >= 0")
    rules = _normalise_length_rules(length_rules)
    rng = np.random.default_rng(seed)

    geometries: List[Geometry] = []
    for index in range(n_geometries):
        lengths = tuple(
            int(rng.integers(rules[primer_type][0], rules[primer_type][1] + 1))
            for primer_type in PRIMER_TYPE_ORDER
        )
        gaps = tuple(int(rng.integers(lo, hi + 1)) for lo, hi in GAP_BOUNDS) + (0,)
        geometries.append(
            Geometry(
                geometry_id=f"{id_prefix}{seed}-{index}",
                lengths=lengths,
                gaps=gaps,
            )
        )
    return geometries


# ---------------------------------------------------------------------------
# Cheap vectorized screen
# ---------------------------------------------------------------------------


def _prefix_tables(template: str) -> Tuple[np.ndarray, np.ndarray]:
    """Inclusive-exclusive prefix sums of (GC count, ACGT count).

    ``errors="replace"`` keeps the byte array the same length as the string so
    coordinates still index the caller's template; any non-ASCII character
    becomes ``?`` and so fails the ACGT screen, which is what we want.
    """
    data = np.frombuffer(template.encode("ascii", errors="replace"), dtype=np.uint8)
    a, c, g, t = _ACGT_ORDS
    is_gc = (data == g) | (data == c)
    is_acgt = is_gc | (data == a) | (data == t)
    cum_gc = np.zeros(data.size + 1, dtype=np.int64)
    cum_acgt = np.zeros(data.size + 1, dtype=np.int64)
    np.cumsum(is_gc, dtype=np.int64, out=cum_gc[1:])
    np.cumsum(is_acgt, dtype=np.int64, out=cum_acgt[1:])
    return cum_gc, cum_acgt


def _surviving_anchors(
    geometry: Geometry,
    template_length: int,
    stride: int,
    cum_gc: np.ndarray,
    cum_acgt: np.ndarray,
) -> np.ndarray:
    """Anchors whose eight windows are all in bounds, pure ACGT and GC-ok.

    Fully vectorized: one pass per primer slot over all anchors at once, using
    the prefix-sum tables, so no per-base Python work.
    """
    anchors = _anchors(geometry, template_length, stride)
    if anchors.size == 0:
        return anchors

    gc_lo, gc_hi = GC_BOUNDS
    keep = np.ones(anchors.size, dtype=bool)
    for length, start_offset, end_offset in zip(
        geometry.lengths, geometry.start_offsets, geometry.end_offsets
    ):
        starts = anchors + start_offset
        ends = anchors + end_offset
        n_gc = cum_gc[ends] - cum_gc[starts]
        n_acgt = cum_acgt[ends] - cum_acgt[starts]
        keep &= n_acgt == length
        keep &= n_gc >= gc_lo * length - _GC_TOL
        keep &= n_gc <= gc_hi * length + _GC_TOL
        if not keep.any():
            return anchors[:0]
    return anchors[keep]


def _anchors(geometry: Geometry, template_length: int, stride: int) -> np.ndarray:
    """``0, stride, 2*stride, ...`` for every anchor whose set fits in bounds."""
    last = template_length - geometry.span
    if last < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last + 1, stride, dtype=np.int64)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def count_candidate_slots(
    template_length: int,
    *,
    geometries: Optional[Sequence[Geometry]] = None,
    stride: int = 1,
) -> int:
    """Candidates an (anchor x geometry) grid holds *before* any screening.

    Cheap closed form for the ``DesignResult.funnel`` entry that precedes
    generation, so an empty result can say whether the template was simply too
    short or the GC/ACGT screen emptied it.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    geoms = default_geometries() if geometries is None else list(geometries)
    return sum(
        max(0, (template_length - geometry.span) // stride + 1)
        for geometry in geoms
        if template_length >= geometry.span
    )


def generate_candidates(
    template: str,
    *,
    geometries: Optional[Sequence[Geometry]] = None,
    stride: int = 1,
    length_rules: Optional[Mapping[str, Tuple[int, int]]] = None,
    max_candidates: Optional[int] = None,
) -> List[PrimerSet]:
    """Enumerate candidate primer sets across ``template``.

    Parameters
    ----------
    template:
        Forward-strand sequence to design against (usually
        ``TargetPanel.consensus``). Upper-cased internally; no other
        transformation is applied, so ``Primer.start``/``end`` index the
        caller's string directly. Characters outside ACGT (``N``, gaps,
        whitespace) are never silently accepted -- any set with such a
        character inside a primer window is dropped, though one inside an
        inter-primer *gap* is harmless and does not drop the set.
    geometries:
        Geometries to lay down at each anchor. Defaults to
        :data:`DEFAULT_GEOMETRIES` (five, canonical included). ``geometry_id``
        values must be unique. Passing an empty sequence is an error rather
        than a silent empty result.
    stride:
        Anchor step in bases. Sets are generated at anchor ``0, stride,
        2*stride, ...`` up to the last anchor whose B3 end is in bounds.
    length_rules:
        Optional per-type inclusive ``(min_length, max_length)``; geometries
        with any primer length outside its type's rule are dropped. Types left
        out keep :data:`LENGTH_BOUNDS`, which rules can narrow but not widen.
        ``ValueError`` if the rules exclude every geometry -- that is a
        misconfiguration, and reporting it as "no candidates" would hide it.
    max_candidates:
        Optional cap. When the grid is larger, whole anchors are thinned at a
        uniform stride before truncation, so the retained sets stay spread
        across the template and every geometry stays represented (never a
        head-of-template prefix, and never one geometry).

    Returns
    -------
    Candidate sets ordered by anchor, then by position in ``geometries``.
    ``set_id`` is ``f"{geometry_id}@{anchor}"``. A template shorter than the
    shortest geometry's span yields ``[]``; it is not an error.

    Notes
    -----
    Deterministic: no RNG, global or local, is consulted. Two calls with equal
    arguments return equal output, and the global numpy RNG state is untouched.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be >= 0 or None")

    geoms = default_geometries() if geometries is None else list(geometries)
    if not geoms:
        raise ValueError("geometries must not be empty")
    seen: Dict[str, int] = {}
    for index, geometry in enumerate(geoms):
        if not isinstance(geometry, Geometry):
            raise TypeError(f"geometries[{index}] is not a Geometry: {geometry!r}")
        if geometry.geometry_id in seen:
            raise ValueError(
                f"duplicate geometry_id {geometry.geometry_id!r} at positions "
                f"{seen[geometry.geometry_id]} and {index}"
            )
        seen[geometry.geometry_id] = index

    if length_rules is not None:
        rules = _normalise_length_rules(length_rules)
        kept = [geometry for geometry in geoms if geometry.fits(rules)]
        if not kept:
            raise ValueError(
                "length_rules excluded every geometry: "
                + ", ".join(
                    f"{geometry.geometry_id} lengths={geometry.lengths}"
                    for geometry in geoms
                )
            )
        geoms = kept

    if max_candidates == 0:
        return []

    sequence = template.upper()
    template_length = len(sequence)
    if template_length == 0 or all(
        template_length < geometry.span for geometry in geoms
    ):
        return []

    cum_gc, cum_acgt = _prefix_tables(sequence)

    anchor_chunks: List[np.ndarray] = []
    index_chunks: List[np.ndarray] = []
    for index, geometry in enumerate(geoms):
        anchors = _surviving_anchors(
            geometry, template_length, stride, cum_gc, cum_acgt
        )
        if anchors.size:
            anchor_chunks.append(anchors)
            index_chunks.append(np.full(anchors.size, index, dtype=np.int64))

    if not anchor_chunks:
        return []

    all_anchors = np.concatenate(anchor_chunks)
    all_indices = np.concatenate(index_chunks)
    # Anchor-major, geometry-minor: keeps nearby sets adjacent in the output and
    # makes the max_candidates thinning below a positional subsample.
    order = np.lexsort((all_indices, all_anchors))
    all_anchors = all_anchors[order]
    all_indices = all_indices[order]

    if max_candidates is not None and all_anchors.size > max_candidates:
        all_anchors, all_indices = _thin_by_anchor(
            all_anchors, all_indices, max_candidates
        )

    # Per-geometry slot plans, hoisted out of the hot loop.
    slot_plans = [
        tuple(
            (primer_type, start, end, primer_type in FORWARD_STRAND_TYPES)
            for primer_type, start, end in zip(
                PRIMER_TYPE_ORDER, geometry.start_offsets, geometry.end_offsets
            )
        )
        for geometry in geoms
    ]
    geometry_ids = [geometry.geometry_id for geometry in geoms]
    # revcomp(sequence[s:e]) == rc_sequence[n - e : n - s], so the reverse-
    # complemented primer types cost one slice instead of a translate per set.
    rc_sequence = _reverse_complement(sequence)

    candidates: List[PrimerSet] = []
    for anchor_value, geometry_index in zip(
        all_anchors.tolist(), all_indices.tolist()
    ):
        primers: Dict[str, Primer] = {}
        for primer_type, start_offset, end_offset, is_forward in slot_plans[
            geometry_index
        ]:
            start = anchor_value + start_offset
            end = anchor_value + end_offset
            primers[primer_type] = Primer(
                primer_type=primer_type,
                sequence=(
                    sequence[start:end]
                    if is_forward
                    else rc_sequence[template_length - end : template_length - start]
                ),
                start=start,
                end=end,
            )
        geometry_id = geometry_ids[geometry_index]
        candidates.append(
            PrimerSet(
                set_id=f"{geometry_id}@{anchor_value}",
                primers=primers,
                anchor=anchor_value,
                geometry_id=geometry_id,
            )
        )
    return candidates


def _thin_by_anchor(
    anchors: np.ndarray, indices: np.ndarray, max_candidates: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep every ``step``-th anchor whole, then truncate to ``max_candidates``.

    Thinning whole anchors rather than individual (anchor, geometry) pairs
    avoids aliasing onto a single geometry when the pair stride happens to be a
    multiple of the number of geometries.
    """
    unique_anchors, first, counts = np.unique(
        anchors, return_index=True, return_counts=True
    )
    step = max(1, math.ceil(anchors.size / max_candidates))
    selected = np.arange(0, unique_anchors.size, step)
    keep = np.concatenate(
        [
            np.arange(first[position], first[position] + counts[position])
            for position in selected
        ]
    )[:max_candidates]
    return anchors[keep], indices[keep]


def _normalise_length_rules(
    length_rules: Optional[Mapping[str, Tuple[int, int]]],
) -> Dict[str, Tuple[int, int]]:
    """Validate partial per-type length rules and fill in the defaults."""
    rules = dict(DEFAULT_LENGTH_RULES)
    if length_rules is None:
        return rules

    hard_lo, hard_hi = LENGTH_BOUNDS
    unknown = set(length_rules) - set(PRIMER_TYPE_ORDER)
    if unknown:
        raise ValueError(
            f"length_rules has unknown primer types: {sorted(unknown)}; "
            f"expected a subset of {list(PRIMER_TYPE_ORDER)}"
        )
    for primer_type, bounds in length_rules.items():
        try:
            lo, hi = (int(bounds[0]), int(bounds[1]))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise TypeError(
                f"length_rules[{primer_type!r}] must be a (min, max) pair, "
                f"got {bounds!r}"
            ) from exc
        if lo > hi:
            raise ValueError(
                f"length_rules[{primer_type!r}] has min {lo} > max {hi}"
            )
        lo, hi = max(lo, hard_lo), min(hi, hard_hi)
        if lo > hi:
            raise ValueError(
                f"length_rules[{primer_type!r}]={bounds!r} does not overlap the "
                f"permitted length range {hard_lo}-{hard_hi}"
            )
        rules[primer_type] = (lo, hi)
    return rules
