"""Candidate generation: enumerate eight-primer LAMP sets across a template.

The generator tiles the template with :class:`PrimerSet` candidates. Every
candidate is the product of two independent choices:

* an **anchor** -- the forward-strand coordinate of the F3 5' end. Anchors are
  taken at ``0, stride, 2*stride, ...``, so position is a single scalar knob.
* a **layout** -- a ``(lengths, gaps)`` pair covering all eight primer types.

At every anchor the generator lays down the **canonical** layout
(:data:`CANONICAL_GAPS` spacing, :data:`CANONICAL_LENGTHS` lengths) plus
:data:`DEFAULT_N_SAMPLED` layouts drawn per anchor, which is what the design
notes specify: five primer sets per position, one of them the LML spacing, the
rest sampled from a normal distribution over the permitted length and gap
ranges.

So this module *does* consume randomness, and the two failure modes of the
reference implementation (``lamp_pipeline/primer_design/tiling.py``) are worth
naming because they are avoided deliberately rather than by accident:

* The reference seeded the **global** numpy RNG in ``__init__``, so two
  concurrent design runs in one process perturbed each other. Everything here
  draws from a local ``np.random.default_rng``, passed down explicitly.
* Sampling made "candidate coverage depend partly on random seed, not just the
  length bounds" (the design notes' own caveat). The seed is therefore derived
  from the input by :func:`derive_geometry_seed`, so the same FASTA always
  returns the same primer sets while each anchor still gets its own draw.

Anchors stay uniformly spaced regardless: the anchor *is* the F3 start, so a
long draw lengthens the set to the right rather than displacing it, and
position coverage is unaffected by the layout draws. Only the span varies.

:data:`DEFAULT_GEOMETRIES` remains as an opt-in fixed panel for controlled
comparisons, where a reproducible candidate count matters more than coverage.

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

import hashlib
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

#: Per-primer **median** length in the LML datasets, in ``PRIMER_TYPE_ORDER``.
#:
#: The design notes specify exactly this for the default geometry: "for our
#: 'default' primerset at that position, (i.e. with gaps of (5, 0, 0, 5, 0, 0,
#: 5, 0)), I'd nominate our median primerlengths in the LML datasets, which is
#: 19-20."
#:
#: Measured, not assumed, over LMLv1 + LMLv2 (119,928 sets, 8 primers each):
#: every type's median is 19 or 20, and the pooled mean is 19.49 --
#: F3 19, F2 20, FL 20, F1c 20, B1c 19, BL 19, B2 20, B3 20. Spelling out the
#: per-type medians rather than flattening to a single 20 keeps the default
#: geometry on the data's actual centre.
CANONICAL_LENGTHS: Tuple[int, ...] = (19, 20, 20, 20, 19, 19, 20, 20)

#: Sampled geometries laid down at each anchor *in addition to* the canonical
#: one. Five total per position is what the design notes ask for: "for every
#: sampled position, I think we have the compute to sample multiple primersets
#: for that position with different spacings and lengths (i.e. let's say 5).
#: Because the LML dataset had gaps of (5, 0, 0, 5, 0, 0, 5, 0), I think it
#: would be helpful to as a rule have one of those 5 primerset spacings."
DEFAULT_N_SAMPLED: int = 4

#: Divisor turning a bound's width into a standard deviation. The notes
#: specify "mean at the midpoint of min/max, ~99.7% of draws within range",
#: i.e. the range spans +/- 3 sigma.
_SIGMA_DIVISOR: float = 6.0

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

#: Five deterministic geometries: ``canonical`` plus a fixed spread over the
#: permitted length and gap ranges.
#:
#: NOT the default design panel any more -- see :func:`generate_candidates`,
#: which lays down ``canonical`` plus :data:`DEFAULT_N_SAMPLED` geometries
#: drawn per anchor, because that is what the design notes specify. These four
#: non-canonical entries were a choice of mine, not a requirement, and they
#: explore exactly five spacings no matter how many positions are tiled.
#:
#: They are kept, and selectable with ``--fixed-geometries``, because a fixed
#: panel is the right instrument for a controlled comparison: the L4-vs-L40S
#: benchmark in docs/ARCHITECTURE.md used it, and its candidate count has to
#: stay reproducible for that number to keep meaning anything.
DEFAULT_GEOMETRIES: Tuple[Geometry, ...] = (
    # Canonical: (5,0,0,5,0,0,5,0) spacing, LML median lengths. Span 172.
    Geometry(
        geometry_id=CANONICAL_GEOMETRY_ID,
        lengths=CANONICAL_LENGTHS,
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
    """The five fixed geometries, as a fresh list.

    The *fixed panel*, not the shipped default -- see
    :func:`generate_candidates` and :func:`geometry_plan`.
    """
    return list(DEFAULT_GEOMETRIES)


def geometry_plan(n_geometries: int, *, fixed: bool = False) -> Dict[str, object]:
    """``generate_candidates`` kwargs for "``n_geometries`` layouts per anchor".

    One knob for every caller -- the CLI, the Modal harness and the Streamlit
    backend all speak in "how many geometries per position", and this is the
    single place that decides what that means.

    ``fixed=False`` (default) gives the design notes' scheme: the canonical
    layout plus ``n_geometries - 1`` sampled per anchor. ``fixed=True`` gives a
    prefix of :data:`DEFAULT_GEOMETRIES` and no sampling, for reproducible
    benchmarking.
    """
    if n_geometries < 1:
        raise ValueError("n_geometries must be >= 1")
    if fixed:
        if n_geometries > len(DEFAULT_GEOMETRIES):
            raise ValueError(
                f"fixed panel has {len(DEFAULT_GEOMETRIES)} geometries, "
                f"asked for {n_geometries}"
            )
        return {"geometries": list(DEFAULT_GEOMETRIES[:n_geometries]), "n_sampled": 0}
    return {"geometries": [canonical_geometry()], "n_sampled": n_geometries - 1}


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
# Per-anchor sampling (the design notes' scheme)
# ---------------------------------------------------------------------------


def derive_geometry_seed(template: str, stride: int) -> int:
    """A seed determined by the input, so a rerun reproduces its own output.

    The design notes ask for sampled lengths and gaps, and the notes also flag
    the cost of that: "candidate coverage depends partly on random seed, not
    just the length bounds". Deriving the seed from the template means both
    hold at once -- each anchor gets its own draw, and the same FASTA always
    returns the same primer sets. ``stride`` is mixed in because it changes
    which anchors exist, and a user who changes it should not be told the run
    is "the same".
    """
    digest = hashlib.sha256(f"{stride}:{template}".encode()).digest()
    # 63 bits: comfortably inside numpy's accepted seed range on all platforms.
    return int.from_bytes(digest[:8], "big") >> 1


def _normal_draws(
    rng: "np.random.Generator", lo: int, hi: int, size: int
) -> np.ndarray:
    """``size`` integer draws from a normal centred in ``[lo, hi]``.

    Mean is the midpoint and sigma is the width over :data:`_SIGMA_DIVISOR`,
    which is the notes' "mean at the midpoint of min/max, ~99.7% of draws
    within range". Draws are rounded then clipped, so the ~0.3% that fall
    outside pile up on the two endpoints rather than being rejected -- a
    rejection loop would make the number of RNG draws data-dependent and cost
    the reproducibility that :func:`derive_geometry_seed` buys.

    A pinned range (``lo == hi``, as :data:`GAP_BOUNDS` does for F3-F2 and
    B2-B3) gives sigma 0 and returns that constant, while still consuming the
    stream, so the draw sequence does not depend on the bounds.
    """
    mean = (lo + hi) / 2.0
    sigma = (hi - lo) / _SIGMA_DIVISOR
    drawn = np.rint(rng.normal(mean, sigma, size))
    return np.clip(drawn, lo, hi).astype(np.int64)


def _sampled_layouts(
    rng: "np.random.Generator",
    n_anchors: int,
    length_rules: Mapping[str, Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one layout per anchor; return ``(start_offsets, end_offsets)``.

    Both are ``(n_anchors, 8)`` in ``PRIMER_TYPE_ORDER``, offsets from the
    anchor, exactly like :attr:`Geometry.start_offsets` but varying per row.
    """
    lengths = np.empty((n_anchors, N_PRIMER_TYPES), dtype=np.int64)
    for index, primer_type in enumerate(PRIMER_TYPE_ORDER):
        lo, hi = length_rules[primer_type]
        lengths[:, index] = _normal_draws(rng, lo, hi, n_anchors)

    # The trailing column stays 0: the unused gap after the last primer.
    gaps = np.zeros((n_anchors, N_PRIMER_TYPES), dtype=np.int64)
    for index, (lo, hi) in enumerate(GAP_BOUNDS):
        gaps[:, index] = _normal_draws(rng, lo, hi, n_anchors)

    starts = np.empty_like(lengths)
    ends = np.empty_like(lengths)
    cursor = np.zeros(n_anchors, dtype=np.int64)
    for index in range(N_PRIMER_TYPES):
        starts[:, index] = cursor
        cursor = cursor + lengths[:, index]
        ends[:, index] = cursor
        cursor = cursor + gaps[:, index]
    return starts, ends


def _surviving_sampled(
    anchors: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    template_length: int,
    cum_gc: np.ndarray,
    cum_acgt: np.ndarray,
) -> np.ndarray:
    """Rows whose eight windows are in bounds, pure ACGT and GC-ok.

    Same screen as :func:`_surviving_anchors`, but every row has its own
    offsets, so the per-slot arithmetic is over arrays instead of scalars.
    Bounds are applied *first*: a row whose span overruns the template would
    otherwise index the prefix tables out of range.
    """
    in_bounds = np.nonzero(anchors + ends[:, -1] <= template_length)[0]
    if in_bounds.size == 0:
        return in_bounds

    anchor = anchors[in_bounds]
    start = starts[in_bounds]
    end = ends[in_bounds]

    gc_lo, gc_hi = GC_BOUNDS
    keep = np.ones(in_bounds.size, dtype=bool)
    for index in range(N_PRIMER_TYPES):
        window_start = anchor + start[:, index]
        window_end = anchor + end[:, index]
        length = end[:, index] - start[:, index]
        n_gc = cum_gc[window_end] - cum_gc[window_start]
        n_acgt = cum_acgt[window_end] - cum_acgt[window_start]
        keep &= n_acgt == length
        keep &= n_gc >= gc_lo * length - _GC_TOL
        keep &= n_gc <= gc_hi * length + _GC_TOL
        if not keep.any():
            return in_bounds[:0]
    return in_bounds[keep]


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
    n_sampled: int = 0,
    stride: int = 1,
) -> int:
    """Candidates an (anchor x layout) grid holds *before* any screening.

    Cheap closed form for the ``DesignResult.funnel`` entry that precedes
    generation, so an empty result can say whether the template was simply too
    short or the GC/ACGT screen emptied it.

    Exact for fixed geometries, whose spans are known up front. For
    ``n_sampled`` slots it is an **estimate**: a sampled layout's span is not
    known until it is drawn, so each is counted at the canonical span. Sampled
    spans average wider than canonical (the gap midpoints exceed the LML
    spacing), so this runs slightly high -- which is the right direction for a
    pre-flight figure that exists to be compared against what came back.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if n_sampled < 0:
        raise ValueError("n_sampled must be >= 0")
    geoms = default_geometries() if geometries is None else list(geometries)
    spans = [geometry.span for geometry in geoms]
    spans.extend([canonical_geometry().span] * n_sampled)
    return sum(
        max(0, (template_length - span) // stride + 1)
        for span in spans
        if template_length >= span
    )


def generate_candidates(
    template: str,
    *,
    geometries: Optional[Sequence[Geometry]] = None,
    n_sampled: int = DEFAULT_N_SAMPLED,
    geometry_seed: Optional[int] = None,
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
        Fixed geometries to lay down at *every* anchor. Defaults to the
        canonical one alone, which with ``n_sampled=4`` gives the design
        notes' five-per-position. Pass :data:`DEFAULT_GEOMETRIES` for the
        fixed five-geometry panel instead. ``geometry_id`` values must be
        unique. Passing an empty sequence is an error rather than a silent
        empty result -- but it is permitted when ``n_sampled > 0``, since the
        sampled layouts then carry the run.
    n_sampled:
        Layouts drawn *per anchor*, in addition to ``geometries``. Lengths are
        drawn inside ``length_rules`` and gaps inside :data:`GAP_BOUNDS`, both
        from a normal centred on the midpoint (see :func:`_normal_draws`).
        0 reproduces the old purely deterministic behaviour.
    geometry_seed:
        Seed for the sampled layouts. ``None`` derives one from ``template``
        and ``stride`` via :func:`derive_geometry_seed`, so the same input
        reproduces the same sets; pass an int to override. Ignored when
        ``n_sampled`` is 0.
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
    Candidate sets ordered by anchor, then by slot (fixed geometries first, in
    the order given, then the sampled layouts). ``set_id`` is
    ``f"{geometry_id}@{anchor}"``, where a sampled layout's id is
    ``f"sampled-{slot}"``. A template shorter than the shortest span yields
    ``[]``; it is not an error.

    Notes
    -----
    Reproducible, not RNG-free. With ``n_sampled > 0`` a local
    ``np.random.default_rng`` is consulted; the global numpy RNG state is never
    touched or read, so concurrent design runs cannot perturb each other. Two
    calls with equal arguments return equal output, because the default seed is
    a function of the input.

    A sampled set's exact lengths and gaps are not recoverable from its
    ``geometry_id`` alone -- but they are always recoverable from the set
    itself, since every ``Primer`` carries its ``start`` and ``end``.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates must be >= 0 or None")
    if n_sampled < 0:
        raise ValueError("n_sampled must be >= 0")

    geoms = [canonical_geometry()] if geometries is None else list(geometries)
    if not geoms and n_sampled == 0:
        raise ValueError("geometries must not be empty when n_sampled is 0")
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

    rules = _normalise_length_rules(length_rules)
    if length_rules is not None and geoms:
        kept = [geometry for geometry in geoms if geometry.fits(rules)]
        if not kept and n_sampled == 0:
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
    if template_length == 0:
        return []
    if n_sampled == 0 and all(
        template_length < geometry.span for geometry in geoms
    ):
        return []

    cum_gc, cum_acgt = _prefix_tables(sequence)

    # Every slot -- fixed or sampled -- contributes (anchors, starts, ends),
    # where starts/ends are (n, 8) offsets from the anchor. A fixed geometry's
    # rows are all identical; a sampled slot's differ per row. Unifying them
    # here is what lets one build loop serve both.
    slot_ids: List[str] = []
    anchor_chunks: List[np.ndarray] = []
    start_chunks: List[np.ndarray] = []
    end_chunks: List[np.ndarray] = []
    slot_chunks: List[np.ndarray] = []

    def _add_slot(
        slot_id: str, anchors: np.ndarray, starts: np.ndarray, ends: np.ndarray
    ) -> None:
        if not anchors.size:
            return
        slot_chunks.append(np.full(anchors.size, len(slot_ids), dtype=np.int64))
        slot_ids.append(slot_id)
        anchor_chunks.append(anchors)
        start_chunks.append(starts)
        end_chunks.append(ends)

    for geometry in geoms:
        anchors = _surviving_anchors(
            geometry, template_length, stride, cum_gc, cum_acgt
        )
        _add_slot(
            geometry.geometry_id,
            anchors,
            np.broadcast_to(
                np.asarray(geometry.start_offsets, dtype=np.int64),
                (anchors.size, N_PRIMER_TYPES),
            ),
            np.broadcast_to(
                np.asarray(geometry.end_offsets, dtype=np.int64),
                (anchors.size, N_PRIMER_TYPES),
            ),
        )

    if n_sampled:
        seed = (
            derive_geometry_seed(sequence, stride)
            if geometry_seed is None
            else int(geometry_seed)
        )
        rng = np.random.default_rng(seed)
        # Every anchor at this stride is offered to each sampled slot; the
        # bounds check lives in `_surviving_sampled`, because a sampled span is
        # not known until it is drawn.
        offered = np.arange(0, template_length, stride, dtype=np.int64)
        for slot in range(n_sampled):
            starts, ends = _sampled_layouts(rng, offered.size, rules)
            rows = _surviving_sampled(
                offered, starts, ends, template_length, cum_gc, cum_acgt
            )
            _add_slot(
                f"sampled-{slot}", offered[rows], starts[rows], ends[rows]
            )

    if not anchor_chunks:
        return []

    all_anchors = np.concatenate(anchor_chunks)
    all_slots = np.concatenate(slot_chunks)
    all_starts = np.concatenate(start_chunks)
    all_ends = np.concatenate(end_chunks)
    # Anchor-major, slot-minor: keeps nearby sets adjacent in the output and
    # makes the max_candidates thinning below a positional subsample.
    order = np.lexsort((all_slots, all_anchors))

    if max_candidates is not None and order.size > max_candidates:
        order = order[
            _thin_by_anchor(all_anchors[order], max_candidates)
        ]

    all_anchors = all_anchors[order]
    all_slots = all_slots[order]
    all_starts = all_starts[order]
    all_ends = all_ends[order]

    forward = tuple(
        primer_type in FORWARD_STRAND_TYPES for primer_type in PRIMER_TYPE_ORDER
    )
    # revcomp(sequence[s:e]) == rc_sequence[n - e : n - s], so the reverse-
    # complemented primer types cost one slice instead of a translate per set.
    rc_sequence = _reverse_complement(sequence)

    candidates: List[PrimerSet] = []
    for row, (anchor_value, slot_index) in enumerate(
        zip(all_anchors.tolist(), all_slots.tolist())
    ):
        start_row = all_starts[row]
        end_row = all_ends[row]
        primers: Dict[str, Primer] = {}
        for index, primer_type in enumerate(PRIMER_TYPE_ORDER):
            start = anchor_value + int(start_row[index])
            end = anchor_value + int(end_row[index])
            primers[primer_type] = Primer(
                primer_type=primer_type,
                sequence=(
                    sequence[start:end]
                    if forward[index]
                    else rc_sequence[template_length - end : template_length - start]
                ),
                start=start,
                end=end,
            )
        geometry_id = slot_ids[slot_index]
        candidates.append(
            PrimerSet(
                set_id=f"{geometry_id}@{anchor_value}",
                primers=primers,
                anchor=anchor_value,
                geometry_id=geometry_id,
            )
        )
    return candidates


def _thin_by_anchor(anchors: np.ndarray, max_candidates: int) -> np.ndarray:
    """Positions to keep: every ``step``-th anchor whole, then truncated.

    Thinning whole anchors rather than individual (anchor, slot) pairs avoids
    aliasing onto a single slot when the pair stride happens to be a multiple
    of the number of slots.

    ``anchors`` must already be sorted anchor-major. Returns indices into it,
    so the caller can apply the same selection to the parallel offset arrays.
    """
    unique_anchors, first, counts = np.unique(
        anchors, return_index=True, return_counts=True
    )
    step = max(1, math.ceil(anchors.size / max_candidates))
    selected = np.arange(0, unique_anchors.size, step)
    return np.concatenate(
        [
            np.arange(first[position], first[position] + counts[position])
            for position in selected
        ]
    )[:max_candidates]


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
