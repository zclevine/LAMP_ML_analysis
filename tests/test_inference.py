"""Prediction clipping in :mod:`lamp_design.models.inference`.

Narrow on purpose: the model's forward pass is covered by ``test_parity`` and
``test_end_to_end``, which assert on real predictions against real weights.
What lives here is the clip, which needs no weights to check and which was
wrong in a way those tests could not catch -- they compare our predictions to
the reference pipeline's, and both clipped the same way.
"""

from __future__ import annotations



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
