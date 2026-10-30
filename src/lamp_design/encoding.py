"""Model input encoding — byte-matched to the training pipeline.

Every function here mirrors a specific function in the training script
``~/LAMP/lyra/examples/LAMP_wave1_block9_ablate.py``. Line references are given
per function. **Do not "clean up" anything in this module without re-running the
parity test** (``tests/test_parity.py``): the model was trained on exactly these
channel orders and conventions, and a mismatch degrades predictions silently
rather than raising.

Two traps worth naming explicitly, because both are invisible at runtime:

1. **The model's primer order is alphabetical, not the biological layout
   order.** ``MODEL_PRIMER_ORDER`` is ``B1c, B2, B3, BL, F1c, F2, F3, FL``,
   while primers are *laid out along the target* as ``F3, F2, FL, F1c, B1c, BL,
   B2, B3`` (``types.PRIMER_TYPE_ORDER``). The encoder concatenates sequences
   and orders physics features in the former; generation uses the latter.

2. **Training zero-padded to the batch maximum with no mask**, and the model
   mean-pools over the full padded length (``Lyra.forward``: ``x.mean(dim=1)``).
   Padded positions are zeros at the *input*, but biases make them non-zero in
   the pooled hidden state, so a prediction depends on what else was in its
   batch. See ``models.inference`` for how we handle that.

Channel layout (19 = ``model_config['d_input']``)::

      0:4   primer one-hot    A C G T
      4:8   target one-hot    A C G T
      8     '|' spacer flag
      9:12  master mix one-hot (SS, BST, WS)
     12     log10(target_conc) - 3
    13:17   per-token physics  (mismatches, gc_fraction, tm, dg_ViennaRNA),
            standardized, broadcast across that primer's tokens; spacers stay 0
    17:19   global physics     (total_mismatches, mean_dg_ViennaRNA), standardized
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from lamp_design.types import MASTER_MIXES

# ---------------------------------------------------------------------------
# Frozen constants -- these match the trained checkpoint
# ---------------------------------------------------------------------------

#: Training script line 692. Alphabetical. Used for BOTH the '|'-joined
#: sequence concatenation order AND the flat physics feature order.
MODEL_PRIMER_ORDER: Tuple[str, ...] = ("B1c", "B2", "B3", "BL", "F1c", "F2", "F3", "FL")

#: Training script line 688. Note unknown characters map to the spacer index.
NUC_TO_IDX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3, "|": 4}

#: Per-primer physics features, in the order `physics_raw_role_and_global`
#: writes them into `role[i, :]` (training script ~line 777).
PHYSICS_ROLE_FEATURES: Tuple[str, ...] = ("mismatches", "gc_fraction", "tm", "dg_ViennaRNA")

#: Globals appended after the flattened role block (training script ~line 790).
PHYSICS_GLOBAL_FEATURES: Tuple[str, ...] = ("total_mismatches", "mean_dg_ViennaRNA")

N_ROLE = len(PHYSICS_ROLE_FEATURES)
N_GLOBAL = len(PHYSICS_GLOBAL_FEATURES)
D_INPUT = 19


def physics_flat_names() -> list[str]:
    """The 34 physics features in checkpoint `physics_standardizer` order.

    ``compute_physics_standardizer`` builds ``role`` with shape
    ``(n, 8, 4)`` indexed by ``MODEL_PRIMER_ORDER`` then does
    ``role.reshape(n, -1)`` and concatenates the 2 globals, so the flat order is
    primer-major: ``B1c_mismatches, B1c_gc_fraction, B1c_tm, B1c_dg_ViennaRNA,
    B2_mismatches, ...`` then the globals.
    """
    names = [f"{p}_{f}" for p in MODEL_PRIMER_ORDER for f in PHYSICS_ROLE_FEATURES]
    return names + list(PHYSICS_GLOBAL_FEATURES)


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------


def one_hot_encode_nucleotide(seq: str) -> np.ndarray:
    """(len(seq), 5) one-hot over A,C,G,T,'|'.

    Mirrors the training script's ``one_hot_encode_nucleotide`` (line 694).
    Anything not in ``NUC_TO_IDX`` -- including ``N`` and gaps -- falls through
    to index 4, the spacer channel. v1 rejects degenerate input upstream so this
    path should never fire; it is kept for exact behavioural parity.
    """
    arr = np.zeros((len(seq), 5), dtype=np.float32)
    for i, nuc in enumerate(seq):
        arr[i, NUC_TO_IDX.get(nuc.upper(), 4)] = 1.0
    return arr


def encode_primer_target_pair(
    primer_seqs: Sequence[str], target_seqs: Sequence[str]
) -> np.ndarray:
    """(L, 9) primer/target one-hot with '|' spacers.

    Mirrors ``encode_primer_target_pair`` with ``pair_onehot=False``. The
    spacer channel is taken from the *primer* encoding, and the target block is
    truncated to the primer length if they disagree -- both are faithful to the
    original, which guards with ``n = min(...)``.
    """
    combined_primer = "|".join(primer_seqs)
    combined_target = "|".join(target_seqs)
    primer_oh = one_hot_encode_nucleotide(combined_primer)
    target_oh = one_hot_encode_nucleotide(combined_target)
    n = min(primer_oh.shape[0], target_oh.shape[0])
    combined = np.zeros((primer_oh.shape[0], 9), dtype=np.float32)
    combined[:n, :4] = primer_oh[:n, :4]
    combined[:n, 4:8] = target_oh[:n, :4]
    combined[:, 8] = primer_oh[:, 4]
    return combined


def canonical_master_mix(value: str) -> Optional[str]:
    """Map a master-mix label onto one of :data:`MASTER_MIXES`, or ``None``.

    The training data spells the same three mixes several ways, so the aliases
    are collapsed here: the ``BstXT`` variants onto ``BST`` and the SYTO9 /
    no-UDG variants onto ``WS``, which is the pooling the training labels used.

    ``None`` means "not a mix this model was trained on". Callers should treat
    that as something to report, not something to pass on quietly:
    :func:`encode_master_mix` turns it into an all-zero one-hot, which the
    model will happily consume and score against no mix at all.
    """
    mm = str(value).strip().upper()
    if mm == "SS":
        return "SS"
    if mm in {"BST", "BSTXT", "BSTXT-DK", "BSTXT-MM", "BSTXT_MM"}:
        return "BST"
    if mm in {"WS", "WS-NOUDG", "WS-SYTO9", "WS_SYTO9"}:
        return "WS"
    return None


def encode_master_mix(value: str) -> np.ndarray:
    """3-way one-hot over (SS, BST, WS); unknown -> all zeros."""
    canon = canonical_master_mix(value)
    return np.array([1.0 if canon == m else 0.0 for m in MASTER_MIXES], dtype=np.float32)


def encode_target_concentration(value: float) -> float:
    """log10(conc) - 3, so 100/1000/10000 -> -1/0/+1."""
    try:
        conc = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conc <= 0:
        return 0.0
    return float(np.log10(conc) - 3.0)


# ---------------------------------------------------------------------------
# Physics encoding
# ---------------------------------------------------------------------------


def physics_raw(
    per_primer: Mapping[str, Mapping[str, float]],
    total_mismatches: float,
    mean_dg: float,
) -> np.ndarray:
    """(34,) raw physics vector in checkpoint standardizer order.

    ``per_primer`` maps primer type -> {mismatches, gc_fraction, tm,
    dg_ViennaRNA}. Missing primers or features become 0.0, matching
    ``_row_float``'s NaN/absent handling in the training script.
    """
    role = np.zeros((len(MODEL_PRIMER_ORDER), N_ROLE), dtype=np.float32)
    for i, primer in enumerate(MODEL_PRIMER_ORDER):
        feats = per_primer.get(primer, {})
        for j, name in enumerate(PHYSICS_ROLE_FEATURES):
            val = feats.get(name, 0.0)
            role[i, j] = 0.0 if val is None or not np.isfinite(val) else float(val)
    glob = np.array([total_mismatches, mean_dg], dtype=np.float32)
    glob = np.where(np.isfinite(glob), glob, 0.0).astype(np.float32)
    return np.concatenate([role.reshape(-1), glob])


def transform_physics(
    raw: np.ndarray, standardizer: Mapping[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize a raw (34,) vector -> (role_z (8,4), glob_z (2,)).

    Mirrors ``transform_physics``. The standardizer comes from the checkpoint's
    ``physics_standardizer`` key and is fit on the *training* set -- never
    recompute it from inference data.
    """
    mean = np.asarray(standardizer["mean"], dtype=np.float32)
    std = np.asarray(standardizer["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    z = (raw - mean) / std
    block = len(MODEL_PRIMER_ORDER) * N_ROLE
    return z[:block].reshape(len(MODEL_PRIMER_ORDER), N_ROLE), z[block:]


# ---------------------------------------------------------------------------
# Full row encoder
# ---------------------------------------------------------------------------


def encode_row(
    primer_seqs: Sequence[str],
    target_seqs: Sequence[str],
    used_primers: Sequence[str],
    *,
    master_mix: str,
    target_conc: float,
    role_z: np.ndarray,
    glob_z: np.ndarray,
) -> np.ndarray:
    """Build the (L, 19) model input for one (primer set, target, condition).

    ``used_primers`` must be in ``MODEL_PRIMER_ORDER`` and parallel to
    ``primer_seqs`` / ``target_seqs``.

    The per-token physics walk reproduces the training script's offset
    arithmetic exactly, including that the ``'|'`` spacer between primers is
    stepped over and so keeps zeros in the physics channels.
    """
    if not (len(primer_seqs) == len(target_seqs) == len(used_primers)):
        raise ValueError("primer_seqs, target_seqs and used_primers must align")

    encoded = encode_primer_target_pair(primer_seqs, target_seqs)
    length = encoded.shape[0]

    mm_cols = np.tile(encode_master_mix(master_mix), (length, 1))
    conc_col = np.full((length, 1), encode_target_concentration(target_conc), dtype=np.float32)

    per_token = np.zeros((length, N_ROLE), dtype=np.float32)
    offset = 0
    for idx, (primer, pseq) in enumerate(zip(used_primers, primer_seqs)):
        end = min(offset + len(pseq), length)
        if end > offset:
            per_token[offset:end] = role_z[MODEL_PRIMER_ORDER.index(primer)]
        offset = end
        if idx != len(used_primers) - 1 and offset < length:
            offset += 1  # step over the '|' spacer

    glob_tok = np.broadcast_to(glob_z.astype(np.float32), (length, N_GLOBAL)).copy()

    out = np.concatenate([encoded, mm_cols, conc_col, per_token, glob_tok], axis=1)
    if out.shape[1] != D_INPUT:
        raise AssertionError(f"expected {D_INPUT} channels, built {out.shape[1]}")
    return out.astype(np.float32)


__all__ = [
    "MODEL_PRIMER_ORDER",
    "NUC_TO_IDX",
    "PHYSICS_ROLE_FEATURES",
    "PHYSICS_GLOBAL_FEATURES",
    "D_INPUT",
    "physics_flat_names",
    "one_hot_encode_nucleotide",
    "encode_primer_target_pair",
    "canonical_master_mix",
    "encode_master_mix",
    "encode_target_concentration",
    "physics_raw",
    "transform_physics",
    "encode_row",
]
