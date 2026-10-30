"""Thermodynamic features for (primer, target) pairs — the pipeline's bottleneck.

The model consumes four features per primer (mismatches, GC fraction, Tm,
ViennaRNA ΔG) plus two globals. Computing them, not running the network, is
what dominates a design run: the network is ~101K parameters, while a 5 kb
target against ten sequences implies on the order of a million primer–target
ΔG evaluations.

## Why this module exists

The reference implementation (``lamp_pipeline/scoring/binding.py``) does:

.. code-block:: python

    unique_pairs["dg_ViennaRNA"] = unique_pairs.apply(
        lambda row: _deltaG_vienna(...), axis=1,   # one subprocess PER ROW
    )

Measured on this machine against the real training values, over 196 unique
pairs drawn from ``LMLv1_corrected_conc.csv``:

==========================================  ==============  ====================
method                                      throughput      matches ground truth
==========================================  ==============  ====================
subprocess per pair (reference)                75 pairs/s    196/196 exact
one batched RNAduplex process                2885 pairs/s    196/196 exact
in-process ``RNA`` bindings                  2716 pairs/s    196/196 exact
==========================================  ==============  ====================

So 1.3M pairs is ~4.8 hours as written, ~7.5 min batched, and ~1 min across
eight cores. All three agree with the values the model was trained on **to
float32**, which also settles a version question: the installed RNAduplex
binary is 2.4.17 while the Python bindings are 2.7.2, and for these DNA
duplexes they do not disagree.

Four optimizations, in order of payoff:

1. **Never one process per pair.** Use in-process bindings across a worker
   pool, or one batched subprocess.
2. **Dedup ΔG on ``(primer, target_window)`` only.** The reference includes
   ``primer_type`` in its dedup key, but ΔG does not depend on the role, so
   that inflates the unique set whenever a sequence serves two roles.
3. **Tm and GC depend only on the primer**, so they are computed once per
   unique primer sequence rather than once per pair.
4. **Skip ``dg_primer3`` and ``end_stability_primer3``.** The reference
   computes both; the abridged-physics model consumes neither.

Mismatch counting is a vectorized byte comparison, not the reference's
per-base Python loop over ``set`` operations.

## Primer ordering

``ThermoTable.values`` is indexed by :data:`~lamp_design.types.PRIMER_TYPE_ORDER`
(the biological layout order), matching the contract in ``types.py``. The model
wants :data:`~lamp_design.encoding.MODEL_PRIMER_ORDER` (alphabetical). The
reordering happens once, explicitly, in :mod:`lamp_design.models.bridge` — do
not do it here, and do not "simplify" by making the two agree.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from lamp_design.types import (
    DNTP_CONC_MM,
    GLOBAL_PHYSICS_FEATURES,
    MG_CONC_MM,
    NA_CONC_MM,
    OLIGO_CONC_NM,
    PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    REACTION_TEMP_C,
    PrimerSet,
    TargetPanel,
    ThermoTable,
)

_DG_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+)\s*\)")
_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

#: ΔG is undefined for a window that could not be located (e.g. a gap in that
#: target). Such pairs are excluded rather than given a fabricated value.
GAP_CHARS = frozenset("-.nN")

#: Pairs per batched RNAduplex invocation. Bounded so stdin stays modest and a
#: failure costs one chunk rather than the whole run.
DG_CHUNK = 20_000


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# ΔG
# ---------------------------------------------------------------------------


def _dg_chunk_subprocess(pairs: Sequence[Tuple[str, str]]) -> List[float]:
    """ΔG for a chunk via one RNAduplex process.

    Matches the reference call exactly: ``--noconv --paramFile=DNA`` on
    ``(primer, revcomp(target))``.
    """
    if not pairs:
        return []
    lines: List[str] = []
    for primer, window in pairs:
        lines.append(primer)
        lines.append(revcomp(window))
    result = subprocess.run(
        ["RNAduplex", "--noconv", "--paramFile=DNA"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    values = [
        float(m.group(1))
        for m in (_DG_RE.search(line) for line in result.stdout.splitlines())
        if m
    ]
    if len(values) != len(pairs):
        raise RuntimeError(
            f"RNAduplex returned {len(values)} energies for {len(pairs)} pairs; "
            "refusing to guess at the alignment between them"
        )
    return values


_RNA_READY = False


def _init_bindings() -> None:
    """Load DNA parameters once per process (pool initializer)."""
    global _RNA_READY
    if _RNA_READY:
        return
    import RNA

    RNA.params_load_DNA_Mathews2004()
    _RNA_READY = True


def _dg_chunk_bindings(pairs: Sequence[Tuple[str, str]]) -> List[float]:
    """ΔG for a chunk via in-process ViennaRNA bindings."""
    import RNA

    _init_bindings()
    return [RNA.duplexfold(primer, revcomp(window)).energy for primer, window in pairs]


def compute_dg(
    pairs: Sequence[Tuple[str, str]],
    *,
    threads: Optional[int] = None,
    backend: str = "auto",
) -> np.ndarray:
    """ΔG (kcal/mol) for each ``(primer, target_window)``, order preserved.

    ``backend``: ``"bindings"`` (in-process, parallelizable), ``"subprocess"``
    (one batched RNAduplex per chunk), or ``"auto"`` which prefers bindings and
    falls back if ``RNA`` is unimportable.
    """
    if not pairs:
        return np.empty(0, dtype=np.float32)

    if backend == "auto":
        try:
            import RNA  # noqa: F401

            backend = "bindings"
        except ImportError:
            backend = "subprocess"

    chunk_fn = _dg_chunk_bindings if backend == "bindings" else _dg_chunk_subprocess
    chunks = [pairs[i : i + DG_CHUNK] for i in range(0, len(pairs), DG_CHUNK)]

    n_threads = threads if threads is not None else min(8, os.cpu_count() or 1)
    if n_threads > 1 and backend == "bindings" and len(chunks) > 1:
        from multiprocessing import Pool

        with Pool(processes=min(n_threads, len(chunks)), initializer=_init_bindings) as pool:
            results = pool.map(_dg_chunk_bindings, chunks)
    else:
        results = [chunk_fn(c) for c in chunks]

    out: List[float] = []
    for part in results:
        out.extend(part)
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Primer-only features
# ---------------------------------------------------------------------------


def gc_fraction(seq: str) -> float:
    """GC fraction. Matches ``Bio.SeqUtils.gc_fraction`` for ACGT input."""
    if not seq:
        return 0.0
    upper = seq.upper()
    return (upper.count("G") + upper.count("C")) / len(upper)


def compute_tm(primers: Sequence[str]) -> np.ndarray:
    """Melting temperature per primer, under the frozen buffer conditions.

    Depends on the primer alone, so this is called once per *unique* primer,
    never once per (primer, target) pair.
    """
    import primer3

    return np.asarray(
        [
            primer3.calc_tm(
                p,
                mv_conc=NA_CONC_MM,
                dv_conc=MG_CONC_MM,
                dntp_conc=DNTP_CONC_MM,
                dna_conc=OLIGO_CONC_NM,
                tm_method="santalucia",
                salt_corrections_method="santalucia",
            )
            for p in primers
        ],
        dtype=np.float32,
    )


def count_mismatches_vectorized(primers: Sequence[str], windows: Sequence[str]) -> np.ndarray:
    """Mismatches per (primer, window), as one vectorized pass per length group.

    The reference loops per base doing ``set`` intersections, which costs
    minutes at design scale. Input is ACGT-only by the time it reaches here
    (``prepare`` rejects degenerate bases), so a literal byte comparison is
    equivalent and ~100x faster.

    Pairs whose primer and window differ in length score ``-1``, meaning
    "could not be compared" -- the caller excludes those rather than treating
    the difference as mismatches.
    """
    n = len(primers)
    out = np.full(n, -1, dtype=np.int32)
    by_len: Dict[int, List[int]] = {}
    for i, (p, w) in enumerate(zip(primers, windows)):
        if len(p) == len(w) and p:
            by_len.setdefault(len(p), []).append(i)
    for length, idxs in by_len.items():
        pa = np.frombuffer(
            "".join(primers[i].upper() for i in idxs).encode("ascii"), dtype="S1"
        ).reshape(len(idxs), length)
        wa = np.frombuffer(
            "".join(windows[i].upper() for i in idxs).encode("ascii"), dtype="S1"
        ).reshape(len(idxs), length)
        out[np.asarray(idxs)] = (pa != wa).sum(axis=1).astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowResolver:
    """Maps a consensus interval onto each target's own bases.

    Candidates are generated against the consensus. Scoring needs the
    corresponding stretch of every *target*. ``column_map[i]`` is the alignment
    column of consensus position ``i`` (``prepare.consensus_columns()``); when
    the panel is ungapped and rectangular this is just ``range(len(consensus))``
    and the lookup degenerates to a slice.
    """

    alignment: Tuple[str, ...]
    column_map: Tuple[int, ...]

    @classmethod
    def from_panel(cls, panel: TargetPanel) -> "WindowResolver":
        if panel.alignment:
            rows = tuple(panel.alignment)
            width = len(rows[0])
            if len(panel.consensus) == width:
                columns = tuple(range(width))
            else:
                # Consensus drops all-gap columns, so recover the surviving set.
                columns = tuple(
                    c
                    for c in range(width)
                    if any(row[c] not in GAP_CHARS for row in rows)
                )
                if len(columns) != len(panel.consensus):
                    raise ValueError(
                        "cannot map consensus positions onto alignment columns: "
                        f"consensus is {len(panel.consensus)} long but "
                        f"{len(columns)} non-gap columns were found. Pass an "
                        "explicit column_map."
                    )
            return cls(alignment=rows, column_map=columns)
        rows = tuple(t.sequence for t in panel.targets)
        return cls(alignment=rows, column_map=tuple(range(len(panel.consensus))))

    def window(self, row_idx: int, start: int, end: int) -> Optional[str]:
        """That target's bases over consensus ``[start, end)``.

        Returns ``None`` when the stretch contains a gap in this target, or
        runs off the end -- the design notes call for primers to avoid gapped
        regions, so such candidates are dropped rather than scored on a
        silently shortened window.
        """
        cols = self.column_map[start:end]
        if len(cols) != end - start:
            return None
        row = self.alignment[row_idx]
        chars = []
        for c in cols:
            if c >= len(row):
                return None
            ch = row[c]
            if ch in GAP_CHARS:
                return None
            chars.append(ch)
        return "".join(chars).upper()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_thermo(
    primer_sets: Sequence[PrimerSet],
    panel: TargetPanel,
    *,
    threads: Optional[int] = None,
    backend: str = "auto",
    resolver: Optional[WindowResolver] = None,
) -> ThermoTable:
    """Physics features for every (primer set, target) pair.

    Pairs where any of the eight primers cannot be located in that target --
    a gap or an out-of-range window -- are **omitted** from the result, so
    ``len(table.set_ids)`` can be fewer than
    ``len(primer_sets) * panel.n_targets``. The caller should record the drop
    in the funnel.
    """
    resolver = resolver or WindowResolver.from_panel(panel)
    target_names = [t.name for t in panel.targets]
    n_types = len(PRIMER_TYPE_ORDER)

    # --- enumerate pairs, resolving every window up front -----------------
    pair_sets: List[str] = []
    pair_targets: List[str] = []
    windows_per_pair: List[List[str]] = []
    primers_per_pair: List[List[str]] = []

    for pset in primer_sets:
        ordered = pset.ordered()
        for row_idx, tname in enumerate(target_names):
            windows: List[str] = []
            ok = True
            for primer in ordered:
                win = resolver.window(row_idx, primer.start, primer.end)
                if win is None or len(win) != primer.length:
                    ok = False
                    break
                # Reverse-strand primers are stored reverse-complemented, so
                # compare against the reverse complement of the window --
                # exactly the convention the training features used.
                windows.append(win if primer.is_forward_strand else revcomp(win))
            if not ok:
                continue
            pair_sets.append(pset.set_id)
            pair_targets.append(tname)
            windows_per_pair.append(windows)
            primers_per_pair.append([p.sequence for p in ordered])

    n_pairs = len(pair_sets)
    values = np.zeros((n_pairs, n_types, len(PHYSICS_FEATURES)), dtype=np.float32)
    globals_ = np.zeros((n_pairs, len(GLOBAL_PHYSICS_FEATURES)), dtype=np.float32)
    target_windows = np.empty((n_pairs, n_types), dtype=object)
    if n_pairs == 0:
        return ThermoTable([], [], values, globals_, target_windows)

    # --- unique primers: Tm and GC (primer-only, so once per sequence) -----
    unique_primers = sorted({p for row in primers_per_pair for p in row})
    tm_vals = compute_tm(unique_primers)
    tm_by_primer = dict(zip(unique_primers, tm_vals.tolist()))
    gc_by_primer = {p: gc_fraction(p) for p in unique_primers}

    # --- unique (primer, window): dG and mismatches ------------------------
    # Dedup key deliberately excludes primer_type: dG does not depend on it.
    pair_index: Dict[Tuple[str, str], int] = {}
    uniq_primer: List[str] = []
    uniq_window: List[str] = []
    for prow, wrow in zip(primers_per_pair, windows_per_pair):
        for primer, window in zip(prow, wrow):
            key = (primer, window)
            if key not in pair_index:
                pair_index[key] = len(uniq_primer)
                uniq_primer.append(primer)
                uniq_window.append(window)

    dg_vals = compute_dg(
        list(zip(uniq_primer, uniq_window)), threads=threads, backend=backend
    )
    mm_vals = count_mismatches_vectorized(uniq_primer, uniq_window)

    # --- assemble, indexed by PRIMER_TYPE_ORDER ---------------------------
    i_mm = PHYSICS_FEATURES.index("mismatches")
    i_gc = PHYSICS_FEATURES.index("gc_fraction")
    i_tm = PHYSICS_FEATURES.index("tm")
    i_dg = PHYSICS_FEATURES.index("dg_ViennaRNA")
    g_mm = GLOBAL_PHYSICS_FEATURES.index("total_mismatches")
    g_dg = GLOBAL_PHYSICS_FEATURES.index("mean_dg_ViennaRNA")

    for row, (prow, wrow) in enumerate(zip(primers_per_pair, windows_per_pair)):
        for t_idx, (primer, window) in enumerate(zip(prow, wrow)):
            u = pair_index[(primer, window)]
            values[row, t_idx, i_mm] = float(mm_vals[u])
            values[row, t_idx, i_gc] = gc_by_primer[primer]
            values[row, t_idx, i_tm] = tm_by_primer[primer]
            values[row, t_idx, i_dg] = float(dg_vals[u])
            target_windows[row, t_idx] = window
        globals_[row, g_mm] = values[row, :, i_mm].sum()
        globals_[row, g_dg] = values[row, :, i_dg].mean()

    return ThermoTable(
        set_ids=pair_sets,
        target_names=pair_targets,
        values=values,
        globals_=globals_,
        target_windows=target_windows,
    )


__all__ = [
    "DG_CHUNK",
    "GAP_CHARS",
    "WindowResolver",
    "compute_dg",
    "compute_thermo",
    "compute_tm",
    "count_mismatches_vectorized",
    "gc_fraction",
    "revcomp",
]
