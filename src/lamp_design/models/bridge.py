"""ThermoTable -> encoded tensors -> Prediction objects.

This is the one place where the two primer orderings meet, and the only place
the reordering is allowed to happen:

* :data:`lamp_design.types.PRIMER_TYPE_ORDER` -- ``F3, F2, FL, F1c, B1c, BL,
  B2, B3`` -- the biological layout order. Candidate generation and
  ``ThermoTable.values`` use it.
* :data:`lamp_design.encoding.MODEL_PRIMER_ORDER` -- ``B1c, B2, B3, BL, F1c,
  F2, F3, FL`` -- alphabetical, and what the trained model expects for both the
  ``'|'``-joined sequences and the 34-element physics vector.

Encoding in layout order instead of model order produces predictions that look
entirely reasonable and are wrong. The permutation below is therefore derived
once from the two constants rather than written out, so it cannot drift if
either changes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from lamp_design.encoding import (
    MODEL_PRIMER_ORDER,
    PHYSICS_GLOBAL_FEATURES,
    PHYSICS_ROLE_FEATURES,
    encode_row,
    transform_physics,
)
from lamp_design.models.inference import LyraEnsemble
from lamp_design.types import (
    GLOBAL_PHYSICS_FEATURES,
    PAIR_CHUNK,
    PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    Condition,
    Prediction,
    PrimerSet,
    ThermoTable,
)

#: layout index -> model index. Derived, never hand-written.
MODEL_PERMUTATION: Tuple[int, ...] = tuple(
    PRIMER_TYPE_ORDER.index(p) for p in MODEL_PRIMER_ORDER
)

# PAIR_CHUNK is imported from lamp_design.types above, which owns it so the CLI
# can print it as a --pair-chunk default without importing torch. It stays in
# this module's namespace (and __all__) because predict_tp is its only consumer.

# The two feature-name tuples are declared in separate modules (one is part of
# the public data contract, one mirrors the training script). They must agree,
# so assert it at import rather than discovering a silent mismatch later.
if tuple(PHYSICS_FEATURES) != tuple(PHYSICS_ROLE_FEATURES):
    raise AssertionError(
        f"PHYSICS_FEATURES {PHYSICS_FEATURES} != "
        f"PHYSICS_ROLE_FEATURES {PHYSICS_ROLE_FEATURES}"
    )
if tuple(GLOBAL_PHYSICS_FEATURES) != tuple(PHYSICS_GLOBAL_FEATURES):
    raise AssertionError("global physics feature names disagree between modules")


def physics_matrix(
    thermo: ThermoTable, indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """``(n_pairs, 34)`` raw physics, permuted into model order.

    Vectorized across pairs: the flat layout is primer-major, matching
    ``compute_physics_standardizer``'s ``role.reshape(n, -1)``. ``indices``
    restricts the rows, for chunked encoding.
    """
    rows: slice | np.ndarray = (
        slice(None) if indices is None else np.asarray(indices, dtype=np.intp)
    )
    values = thermo.values[rows]
    globals_ = thermo.globals_[rows]
    n = values.shape[0]
    n_role = len(PHYSICS_ROLE_FEATURES)
    block = len(MODEL_PRIMER_ORDER) * n_role
    raw = np.zeros((n, block + len(PHYSICS_GLOBAL_FEATURES)), dtype=np.float32)
    for k, layout_idx in enumerate(MODEL_PERMUTATION):
        raw[:, k * n_role : (k + 1) * n_role] = values[:, layout_idx, :]
    raw[:, block:] = globals_
    return np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)


def _set_lookup(primer_sets: Iterable[PrimerSet] | Mapping[str, PrimerSet]) -> Dict[str, PrimerSet]:
    if isinstance(primer_sets, Mapping):
        return dict(primer_sets)
    return {ps.set_id: ps for ps in primer_sets}


def encode_pairs(
    thermo: ThermoTable,
    primer_sets: Iterable[PrimerSet] | Mapping[str, PrimerSet],
    condition: Condition,
    standardizer: Mapping[str, np.ndarray],
    *,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Encode every pair in ``thermo`` for one reaction condition.

    Returns ``(rows, kept_indices)``; a pair is skipped only if its primer set
    is absent from ``primer_sets`` or its target windows were never populated.
    ``indices`` restricts which pairs are encoded -- the returned
    ``kept_indices`` are still indices into ``thermo``, not into ``indices``.
    """
    lookup = _set_lookup(primer_sets)
    selected = list(range(len(thermo.set_ids))) if indices is None else list(indices)
    raw = physics_matrix(thermo, indices=selected)
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    z_all = (raw - mean) / std
    block = len(MODEL_PRIMER_ORDER) * len(PHYSICS_ROLE_FEATURES)

    rows: List[np.ndarray] = []
    kept: List[int] = []
    for slot, i in enumerate(selected):
        set_id = thermo.set_ids[i]
        pset = lookup.get(set_id)
        if pset is None:
            continue
        primer_seqs: List[str] = []
        window_seqs: List[str] = []
        ok = True
        for layout_idx in MODEL_PERMUTATION:
            primer_type = PRIMER_TYPE_ORDER[layout_idx]
            primer = pset.primers.get(primer_type)
            window = thermo.target_windows[i, layout_idx]
            if primer is None or not window:
                ok = False
                break
            primer_seqs.append(primer.sequence)
            window_seqs.append(str(window))
        if not ok:
            continue
        role_z = z_all[slot, :block].reshape(
            len(MODEL_PRIMER_ORDER), len(PHYSICS_ROLE_FEATURES)
        )
        glob_z = z_all[slot, block:]
        rows.append(
            encode_row(
                primer_seqs,
                window_seqs,
                list(MODEL_PRIMER_ORDER),
                master_mix=condition.master_mix,
                target_conc=condition.target_conc,
                role_z=role_z,
                glob_z=glob_z,
            )
        )
        kept.append(i)
    return rows, kept


def _dedup_key(thermo: ThermoTable, pair_idx: int) -> tuple:
    """Everything :func:`encode_row` reads for one pair, as a hashable key.

    Provably sufficient: ``encode_pairs`` builds row ``i`` from exactly four
    things -- the primer sequences of ``thermo.set_ids[i]``, the eight strings
    in ``thermo.target_windows[i]``, and the standardized
    ``thermo.values[i]`` / ``thermo.globals_[i]`` -- plus the condition and the
    standardizer, which are fixed across the rows being grouped. Two pairs
    agreeing on all four therefore encode to the same tensor and predict the
    same number.

    ``set_id`` stands in for the primer sequences, since the set determines
    them. That makes the key cheap but confines deduplication to one candidate
    set. Measured cost of that choice: none. On the H5N1 demo panel, grouping
    globally and grouping within a set both give 26,495 distinct rows out of
    75,114, because two *different* sets essentially never agree on windows and
    physics at once.
    """
    return (
        thermo.set_ids[pair_idx],
        tuple(thermo.target_windows[pair_idx].tolist()),
        thermo.values[pair_idx].tobytes(),
        thermo.globals_[pair_idx].tobytes(),
    )


def predict_tp(
    thermo: ThermoTable,
    primer_sets: Iterable[PrimerSet] | Mapping[str, PrimerSet],
    conditions: Sequence[Condition],
    *,
    ensemble: Optional[LyraEnsemble] = None,
    weights: Optional[str] = None,
    device: str = "cpu",
    pad_mode: str = "group_by_length",
    clip: bool = True,
    batch_size: int = 512,
    pair_chunk: int = PAIR_CHUNK,
    dedup: bool = True,
) -> List[Prediction]:
    """Predicted Tp for every (pair, condition).

    Pass either a loaded ``ensemble`` or a ``weights`` directory. ``pad_mode``
    defaults to the batch-independent path; see :mod:`lamp_design.models.inference`
    for why that differs from the published pipeline.

    Pairs are encoded and predicted ``pair_chunk`` at a time so peak memory
    stays flat as the sweep grows -- see :data:`PAIR_CHUNK`. Chunking cannot
    change a prediction under the default ``pad_mode``, because
    ``group_by_length`` never pads and so never lets one row influence another.
    Under ``"batch_max"`` it *can*, since that mode is batch-composition
    dependent by design; set ``pair_chunk`` larger than the pair count to
    reproduce a specific published batching.

    ``batch_size`` is rows per forward pass. 512 measured fastest on both CPU
    and an L4 GPU; 4096 and 16384 were ~7% slower, and the GPU never exceeded
    29% utilization, so the constraint is encoding throughput rather than batch
    launch overhead. See ``commands.design.BATCH_SIZE_CPU``.

    ``dedup`` groups pairs whose model input is identical and runs each group
    once -- see :func:`_dedup_key` for why the key is sufficient. It pays
    because a panel is usually closely related sequences: one candidate set
    reads only ~170 nt of each target, and two genomes agreeing over that
    stretch give a byte-identical tensor, since the master mix and
    concentration are the *same condition* for every target. Measured on the
    H5N1 demo panel, 75,114 pairs collapse to 26,495 rows (**2.84x**) and the
    inference stage goes 63.0s -> 23.4s on an L4, a **2.69x** speedup. The
    factor is panel-dependent by nature: near-clonal input gains a lot, a
    diverse panel approaches 1x and loses only a hash per row.

    **Exact in arithmetic, not bit-exact in floating point.** Grouping does not
    change what any row computes, but it changes how many rows there are, hence
    the batches, hence the order of float32 reductions inside them. Measured
    against ``dedup=False`` on the demo panel: max |dTp| **7.4e-04 min** on an
    L4 and **3.8e-06 min** on CPU, on predictions spanning 0-60 min. For scale,
    the GPU-vs-CPU difference this pipeline already accepts is 5.8e-04 min, so
    this is the same order as existing device noise. The returned top 20 was
    identical in id and order. Set ``dedup=False`` to force every pair through.
    """
    if ensemble is None:
        ensemble = LyraEnsemble.load(weights, device=device)
    if not thermo.set_ids or not conditions:
        return []
    if pair_chunk < 1:
        raise ValueError(f"pair_chunk must be >= 1, got {pair_chunk}")

    standardizer = ensemble.physics_standardizer
    n_pairs = len(thermo.set_ids)
    out: List[Prediction] = []
    for condition in conditions:
        for start in range(0, n_pairs, pair_chunk):
            chunk = list(range(start, min(start + pair_chunk, n_pairs)))

            if dedup:
                # Group first, encode and predict one member per group. Groups
                # are built per chunk, not globally, so peak memory stays the
                # property `pair_chunk` exists to bound.
                groups: Dict[tuple, List[int]] = {}
                for pair_idx in chunk:
                    groups.setdefault(_dedup_key(thermo, pair_idx), []).append(pair_idx)
                encode_indices = [members[0] for members in groups.values()]
            else:
                groups = {}
                encode_indices = chunk

            rows, kept = encode_pairs(
                thermo, primer_sets, condition, standardizer, indices=encode_indices
            )
            if not rows:
                continue
            mean, std = ensemble.predict(
                rows, pad_mode=pad_mode, clip=clip, batch_size=batch_size
            )

            # pair index -> (tp, std). Built first, then drained in chunk order,
            # so deduplication cannot reorder the output: callers compare these
            # lists against non-deduplicated runs.
            scored: Dict[int, Tuple[float, float]] = {}
            slot_of = {pair_idx: slot for slot, pair_idx in enumerate(kept)}
            if dedup:
                for members in groups.values():
                    slot = slot_of.get(members[0])
                    # A representative that failed to encode means the whole
                    # group fails: the members are identical by construction,
                    # so none of them could have encoded either.
                    if slot is None:
                        continue
                    value = (float(mean[slot]), float(std[slot]))
                    for pair_idx in members:
                        scored[pair_idx] = value
            else:
                for pair_idx, slot in slot_of.items():
                    scored[pair_idx] = (float(mean[slot]), float(std[slot]))

            for pair_idx in chunk:
                hit = scored.get(pair_idx)
                if hit is None:
                    continue
                out.append(
                    Prediction(
                        set_id=thermo.set_ids[pair_idx],
                        target_name=thermo.target_names[pair_idx],
                        condition=condition,
                        tp_minutes=hit[0],
                        tp_std=hit[1],
                    )
                )
    return out


def mismatch_map(thermo: ThermoTable) -> Dict[Tuple[str, str], int]:
    """``(set_id, target_name) -> total mismatches``, for ``report.score_sets``.

    ``Prediction`` carries no mismatch field, so the report stage takes this
    mapping separately rather than having mismatch columns silently blank.
    """
    idx = GLOBAL_PHYSICS_FEATURES.index("total_mismatches")
    return {
        (sid, tname): int(round(float(thermo.globals_[i, idx])))
        for i, (sid, tname) in enumerate(zip(thermo.set_ids, thermo.target_names))
    }


__all__ = [
    "MODEL_PERMUTATION",
    "PAIR_CHUNK",
    "encode_pairs",
    "mismatch_map",
    "physics_matrix",
    "predict_tp",
]
