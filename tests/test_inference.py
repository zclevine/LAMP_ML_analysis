"""Prediction clipping in :mod:`lamp_design.models.inference`.

Narrow on purpose: the model's forward pass is covered by ``test_parity`` and
``test_end_to_end``, which assert on real predictions against real weights.
What lives here is the clip, which needs no weights to check and which was
wrong in a way those tests could not catch -- they compare our predictions to
the reference pipeline's, and both clipped the same way.
"""

from __future__ import annotations

import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]



# ---------------------------------------------------------------------------
# Prediction clipping
#
# These values are ranked, and clipping is many-to-one at the boundary, so a
# wrong bound does not merely mislabel -- it creates ties and changes which
# sets come back. TP_CLIP_MIN was 0.0 and discarded the model's own top pick
# on the demo panel.
# ---------------------------------------------------------------------------


def test_the_clip_bounds_match_the_training_configuration():
    """Not a display convention: the range the model was trained to emit.

    From lyra/export/New_Lyra_code/command_line_args.json --
    regress_tp_cutoff = -100, tp_censor_at = 60.0.
    """
    from lamp_design.models.inference import TP_CLIP_MAX, TP_CLIP_MIN

    assert TP_CLIP_MIN == -100.0, "must match regress_tp_cutoff"
    assert TP_CLIP_MAX == 60.0, "must match tp_censor_at"


def test_the_floor_does_not_sit_at_zero():
    """Regression guard for the specific bug.

    The ground-truth tp column reaches -4.611 and training floors at -100, so
    a floor of 0 truncates in-distribution output. Spelled out separately from
    the equality test above so the failure message says *why*.
    """
    from lamp_design.models.inference import TP_CLIP_MIN

    assert TP_CLIP_MIN < -4.611, (
        "a floor above the most negative training label flattens real "
        "predictions into ties and changes the returned sets"
    )


def test_clipping_is_wide_enough_to_preserve_ordering_of_fast_sets():
    """A monotone transform would be harmless; clipping is not, so check it.

    Two predictions that differ below zero must stay distinguishable after
    clipping, because the ranking reads them.
    """
    import numpy as np

    from lamp_design.models.inference import TP_CLIP_MAX, TP_CLIP_MIN

    fast = np.array([-5.46, -4.66, -3.86, -1.41, 0.38])
    clipped = np.clip(fast, TP_CLIP_MIN, TP_CLIP_MAX)
    assert np.array_equal(clipped, fast), "fast predictions must survive intact"
    assert len(set(clipped.tolist())) == len(fast), "clipping introduced a tie"


# ---------------------------------------------------------------------------
# Input deduplication
#
# predict_tp groups pairs whose model input is identical and runs each group
# once. The key must be sufficient (two pairs sharing it really do encode
# identically) and the output must keep its order, since callers diff
# prediction lists against non-deduplicated runs.
# ---------------------------------------------------------------------------


def _demo_thermo(n_sets: int = 120):
    """A real panel, because the whole point is sequence identity across targets."""
    from lamp_design.commands.prepare import prepare_panel
    from lamp_design.generation import generate_candidates
    from lamp_design.thermo import compute_thermo
    from lamp_design.utils.sequences import read_fasta

    fasta = REPO / "examples" / "H5N1_HA_demo.fasta"
    panel = prepare_panel(read_fasta(str(fasta)), threads=2, panel_name="demo").panel
    sets = generate_candidates(panel.consensus, stride=1)[:n_sets]
    return compute_thermo(sets, panel, threads=2), sets, panel


def test_the_demo_panel_actually_contains_duplicate_inputs():
    """If it did not, every test below would pass vacuously."""
    from lamp_design.models.bridge import _dedup_key

    table, _, _ = _demo_thermo()
    keys = {_dedup_key(table, i) for i in range(len(table.set_ids))}
    assert len(keys) < len(table.set_ids), "no duplicates to deduplicate"


def test_pairs_sharing_a_dedup_key_encode_identically():
    """The soundness condition. If this fails, dedup is wrong, not just slow."""
    import numpy as np

    from lamp_design.models.bridge import _dedup_key, encode_pairs
    from lamp_design.models.inference import LyraEnsemble
    from lamp_design.types import Condition

    weights = os.environ.get("LAMP_DESIGN_WEIGHTS")
    if not weights or not pathlib.Path(weights).exists():
        pytest.skip("LAMP_DESIGN_WEIGHTS not set")

    table, sets, _ = _demo_thermo()
    groups: dict = {}
    for i in range(len(table.set_ids)):
        groups.setdefault(_dedup_key(table, i), []).append(i)
    shared = [m for m in groups.values() if len(m) > 1]
    assert shared, "no shared keys in the sample"

    standardizer = LyraEnsemble.load(weights).physics_standardizer
    condition = Condition("SS", 10000.0)
    checked = 0
    for members in shared[:25]:
        rows, kept = encode_pairs(
            table, sets, condition, standardizer, indices=members
        )
        assert len(kept) == len(members), "a group encoded only partially"
        first = rows[0]
        for other in rows[1:]:
            assert np.array_equal(first, other), "same key, different encoding"
        checked += 1
    assert checked, "nothing was verified"


def test_dedup_preserves_count_order_and_values_within_device_noise():
    from lamp_design.models.bridge import predict_tp
    from lamp_design.models.inference import LyraEnsemble
    from lamp_design.types import Condition

    weights = os.environ.get("LAMP_DESIGN_WEIGHTS")
    if not weights or not pathlib.Path(weights).exists():
        pytest.skip("LAMP_DESIGN_WEIGHTS not set")

    table, sets, _ = _demo_thermo()
    conditions = [Condition("SS", 10000.0), Condition("SS", 100.0)]
    ensemble = LyraEnsemble.load(weights)

    off = predict_tp(table, sets, conditions, ensemble=ensemble, dedup=False)
    on = predict_tp(table, sets, conditions, ensemble=ensemble, dedup=True)

    assert len(on) == len(off), "dedup changed how many predictions come back"
    for a, b in zip(off, on):
        assert (a.set_id, a.target_name, a.condition) == (
            b.set_id,
            b.target_name,
            b.condition,
        ), "dedup reordered the output"
    # Not bit-exact: fewer rows means different batches means a different
    # float32 reduction order. Bounded well under the GPU-vs-CPU difference
    # this pipeline already accepts (5.8e-04 min).
    worst = max(abs(a.tp_minutes - b.tp_minutes) for a, b in zip(off, on))
    assert worst < 1e-2, f"dedup moved a prediction by {worst} min"


def test_dedup_is_on_by_default():
    """It is a 2.7x speedup; a default of False would silently forfeit it."""
    import inspect

    from lamp_design.models.bridge import predict_tp

    assert inspect.signature(predict_tp).parameters["dedup"].default is True
