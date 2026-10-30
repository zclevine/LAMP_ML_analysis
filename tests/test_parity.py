"""End-to-end parity against the published model's own predictions.

The shipped checkpoint (`tp60_A_v0t9`) was used to produce
`~/LAMP/lyra/export/A_tp60c/*.csv`, which carry an `observed,predicted` pair per
(primer_set, target_seq, target_conc, master_mix). Those `predicted` values are
ground truth for this engine's inference path: if our encoder, standardizer and
ensemble are wired up correctly we must reproduce them.

This is the test that catches silent feature-parity breakage -- a wrong channel
order or a recomputed standardizer still yields plausible numbers, so only a
comparison against the real model's output detects it.

Skipped (not failed) when the weights or the reference datasets are absent, so
the suite stays runnable without them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lamp_design.encoding import (
    MODEL_PRIMER_ORDER,
    PHYSICS_ROLE_FEATURES,
    encode_row,
    physics_raw,
    transform_physics,
)

pd = pytest.importorskip("pandas")

LYRA_ROOT = Path.home() / "LAMP" / "lyra"
FEATURES_CSV = LYRA_ROOT / "collaborator_exports" / "mpox.csv"
PUBLISHED_CSV = LYRA_ROOT / "export" / "A_tp60c" / "mpox.csv"

JOIN_KEYS = ["primer_set", "target_seq", "target_conc", "master_mix"]


def _weights_dir() -> str | None:
    return os.environ.get("LAMP_DESIGN_WEIGHTS")


requires_reference = pytest.mark.skipif(
    not (FEATURES_CSV.exists() and PUBLISHED_CSV.exists()),
    reason="reference mpox datasets not present",
)
requires_weights = pytest.mark.skipif(
    _weights_dir() is None, reason="LAMP_DESIGN_WEIGHTS not set",
)


def load_joined(max_rows: int | None = None) -> "pd.DataFrame":
    """Join published predictions onto their feature rows."""
    feats = pd.read_csv(FEATURES_CSV, low_memory=False)
    pub = pd.read_csv(PUBLISHED_CSV)
    pub = pub.rename(columns={"validation_mm": "master_mix"})

    # Feature rows repeat per technical replicate; the features themselves are
    # identical within a join key, so one row per key is enough.
    feats = feats.drop_duplicates(subset=JOIN_KEYS, keep="first")

    merged = pub.merge(feats, on=JOIN_KEYS, how="inner", validate="one_to_one")
    if max_rows is not None:
        merged = merged.head(max_rows)
    return merged


def encode_dataframe(df: "pd.DataFrame", standardizer) -> list[np.ndarray]:
    """Encode each joined row exactly as the training dataset builder did."""
    rows: list[np.ndarray] = []
    for _, row in df.iterrows():
        primer_seqs, target_seqs, used = [], [], []
        per_primer: dict[str, dict[str, float]] = {}
        for primer in MODEL_PRIMER_ORDER:
            pseq = row.get(f"{primer}_sequence")
            tseq = row.get(f"{primer}_target_sequence")
            if pd.notna(pseq) and pd.notna(tseq):
                primer_seqs.append(str(pseq))
                target_seqs.append(str(tseq))
                used.append(primer)
            per_primer[primer] = {
                feat: row.get(f"{primer}_{feat}", 0.0) for feat in PHYSICS_ROLE_FEATURES
            }
        raw = physics_raw(
            per_primer,
            row.get("total_mismatches", 0.0),
            row.get("mean_dg_ViennaRNA", 0.0),
        )
        role_z, glob_z = transform_physics(raw, standardizer)
        rows.append(
            encode_row(
                primer_seqs,
                target_seqs,
                used,
                master_mix=row["master_mix"],
                target_conc=row["target_conc"],
                role_z=role_z,
                glob_z=glob_z,
            )
        )
    return rows


@requires_reference
def test_join_is_non_empty():
    merged = load_joined()
    assert len(merged) > 50, f"expected a substantial join, got {len(merged)} rows"
    assert merged["predicted"].notna().all()


@requires_reference
@requires_weights
def test_reproduces_published_predictions():
    """The decisive check: our pipeline must reproduce `predicted`.

    An exact match on every row is **not achievable in principle**, and that is
    a property of the published pipeline rather than of this code. Training and
    evaluation zero-padded each batch to its longest member with no mask, and
    the model mean-pools over the padded length, so a published prediction
    depends on which other rows happened to share its batch -- information that
    is not recorded anywhere.

    Measured against `export/A_tp60c/mpox.csv` with per-row (unpadded)
    inference: 126/232 rows land within 1e-3 min immediately, and 186/232
    (80%) are reproduced to 1e-3 at *some* pad length, with a median best
    deviation of 2.6e-5 min. Sweeping pad length per row shows the best-fitting
    values cluster a few positions above each row's own length -- the signature
    of similar-length rows batched together. The 46 rows that never reach 1e-3
    sit within 0.11 min at best.

    Reproducing a prediction to float32 precision is impossible with a wrong
    channel order, a wrong primer order or a recomputed standardizer, so the
    exact-match population is the real parity evidence. These assertions are
    therefore on correlation and the median, which collapse immediately under
    any genuine parity break, plus a floor on the exactly-reproduced fraction.
    """
    from lamp_design.models.inference import LyraEnsemble

    merged = load_joined()
    ensemble = LyraEnsemble.load(_weights_dir())
    rows = encode_dataframe(merged, ensemble.physics_standardizer)

    got, _ = ensemble.predict(rows, pad_mode="group_by_length", clip=False)
    want = merged["predicted"].to_numpy(dtype=float)
    dev = np.abs(got - want)

    corr = np.corrcoef(got, want)[0, 1]
    assert corr > 0.999, (
        f"correlation with published predictions only {corr:.5f}; "
        "feature parity is broken"
    )
    assert np.median(dev) < 0.01, (
        f"median deviation {np.median(dev):.5f} min is too large; "
        "feature parity is broken"
    )
    exact = int((dev < 1e-3).sum())
    assert exact >= len(dev) // 2, (
        f"only {exact}/{len(dev)} rows reproduced to 1e-3; a wrong channel or "
        "primer order cannot reproduce any row to float32 precision, so a drop "
        "here means the encoding changed"
    )


@requires_reference
@requires_weights
def test_padding_changes_predictions_materially():
    """Guard rail: padding is not free, so nobody should "optimize" by padding.

    Padding every row to the global maximum instead of grouping by length
    moves predictions by minutes, which can flip the clinically relevant
    "amplifies within 30 min" call. This test exists so that a future change
    to batching cannot quietly reintroduce it.
    """
    from lamp_design.models.inference import LyraEnsemble

    merged = load_joined(max_rows=150)
    ensemble = LyraEnsemble.load(_weights_dir())
    rows = encode_dataframe(merged, ensemble.physics_standardizer)

    grouped, _ = ensemble.predict(rows, pad_mode="group_by_length", clip=False)
    # One batch big enough to hold everything => pads to the global max.
    padded, _ = ensemble.predict(
        rows, pad_mode="batch_max", batch_size=len(rows), clip=False
    )

    spread = np.abs(grouped - padded)
    assert spread.max() > 0.1, (
        "padding to the global max made no difference, which contradicts the "
        "unmasked mean-pool in Lyra.forward -- has the architecture changed?"
    )


@requires_reference
@requires_weights
def test_group_by_length_is_close_to_batch_max():
    """Quantify the cost of our batch-independence divergence.

    `group_by_length` is the production default and is *not* expected to match
    `batch_max` exactly, because training pooled over padded positions. This
    test pins how large that difference actually is so a future change cannot
    widen it unnoticed.
    """
    from lamp_design.models.inference import LyraEnsemble

    merged = load_joined(max_rows=200)
    ensemble = LyraEnsemble.load(_weights_dir())
    rows = encode_dataframe(merged, ensemble.physics_standardizer)

    faithful, _ = ensemble.predict(rows, pad_mode="batch_max", clip=False)
    grouped, _ = ensemble.predict(rows, pad_mode="group_by_length", clip=False)

    dev = np.abs(faithful - grouped)
    assert dev.max() < 5.0, (
        f"group_by_length deviates up to {dev.max():.3f} min from the "
        "training-faithful path; that is larger than expected"
    )


@requires_reference
@requires_weights
def test_group_by_length_is_batch_independent():
    """A candidate's prediction must not depend on the rest of the batch."""
    from lamp_design.models.inference import LyraEnsemble

    merged = load_joined(max_rows=120)
    ensemble = LyraEnsemble.load(_weights_dir())
    rows = encode_dataframe(merged, ensemble.physics_standardizer)

    full, _ = ensemble.predict(rows, pad_mode="group_by_length", clip=False)
    # Same rows, shuffled, then mapped back.
    order = np.random.default_rng(0).permutation(len(rows))
    shuffled, _ = ensemble.predict(
        [rows[i] for i in order], pad_mode="group_by_length", clip=False
    )
    restored = np.empty_like(shuffled)
    restored[order] = shuffled

    assert np.allclose(full, restored, atol=1e-4), (
        "group_by_length predictions changed under reordering, so it is not "
        "batch-independent after all"
    )
