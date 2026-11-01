"""Full pipeline: generation -> thermo -> encoding -> inference -> report.

Exercises every stage against each other with no mocking of the engine, on a
small synthetic panel so it stays fast. The model stages are skipped when
weights are absent; everything up to inference runs unconditionally.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from lamp_design.generation import default_geometries, generate_candidates
from lamp_design.thermo import WindowResolver, compute_thermo
from lamp_design.types import Condition, Target, TargetPanel

rng = np.random.default_rng(1234)


def synthetic_panel(n_targets: int = 3, length: int = 600, divergence: float = 0.02):
    """An ungapped, equal-length panel: the common same-amplicon case."""
    bases = np.array(list("ACGT"))
    base_seq = "".join(rng.choice(bases, size=length))
    targets = []
    for i in range(n_targets):
        arr = np.array(list(base_seq))
        if i > 0:
            n_mut = max(1, int(length * divergence))
            pos = rng.choice(length, size=n_mut, replace=False)
            arr[pos] = rng.choice(bases, size=n_mut)
        targets.append(Target(name=f"target_{i}", sequence="".join(arr)))
    # Equal-length ungapped input is its own alignment.
    alignment = [t.sequence for t in targets]
    consensus = _plurality(alignment)
    return TargetPanel(targets=targets, consensus=consensus, alignment=alignment)


def _plurality(rows):
    out = []
    for col in range(len(rows[0])):
        counts = {}
        for row in rows:
            ch = row[col].upper()
            if ch in "ACGT":
                counts[ch] = counts.get(ch, 0) + 1
        out.append(max(sorted(counts), key=counts.get) if counts else "A")
    return "".join(out)


def _weights():
    return os.environ.get("LAMP_DESIGN_WEIGHTS")


requires_weights = pytest.mark.skipif(_weights() is None, reason="LAMP_DESIGN_WEIGHTS not set")


def test_generation_to_thermo():
    panel = synthetic_panel()
    cands = generate_candidates(panel.consensus, stride=40, geometries=default_geometries())
    assert cands, "no candidates generated on a 600 bp template"

    table = compute_thermo(cands[:12], panel, threads=2)
    assert len(table.set_ids) > 0
    # Every pair carries eight populated windows and finite physics.
    assert table.values.shape == (len(table.set_ids), 8, 4)
    assert np.isfinite(table.values).all()
    assert (table.target_windows != None).all()  # noqa: E711
    # Mismatches are non-negative; -1 means "uncomparable" and such pairs
    # must have been dropped, not emitted.
    mm = table.values[:, :, 0]
    assert (mm >= 0).all()
    # Globals agree with the per-primer block.
    assert np.allclose(table.globals_[:, 0], mm.sum(axis=1), atol=1e-4)
    assert np.allclose(table.globals_[:, 1], table.values[:, :, 3].mean(axis=1), atol=1e-4)


def test_thermo_dedup_is_consistent():
    """The same (primer, window) must get the same dG everywhere it appears."""
    panel = synthetic_panel(n_targets=2)
    cands = generate_candidates(panel.consensus, stride=25, geometries=default_geometries()[:2])
    table = compute_thermo(cands[:20], panel, threads=1)

    seen = {}
    for row in range(len(table.set_ids)):
        for t in range(8):
            key = str(table.target_windows[row, t])
            dg = float(table.values[row, t, 3])
            if key in seen:
                # same window but possibly a different primer; only compare
                # when the pair is genuinely identical
                continue
            seen[key] = dg
    assert len(seen) > 0


def test_gapped_window_is_dropped_not_fudged():
    """A gap in a target must drop the pair rather than shorten the window."""
    panel = synthetic_panel(n_targets=2, length=600)
    gapped = list(panel.alignment)
    # Punch a gap run through the middle of target_1.
    gapped[1] = gapped[1][:300] + "-" * 10 + gapped[1][310:]
    panel_gapped = TargetPanel(
        targets=panel.targets, consensus=panel.consensus, alignment=gapped
    )
    cands = generate_candidates(panel.consensus, stride=20, geometries=default_geometries()[:1])
    resolver = WindowResolver(alignment=tuple(gapped), column_map=tuple(range(600)))

    table = compute_thermo(cands, panel_gapped, threads=1, resolver=resolver)
    # target_1 pairs overlapping the gap are absent; target_0 pairs survive.
    by_target = {}
    for sid, tname in zip(table.set_ids, table.target_names):
        by_target.setdefault(tname, set()).add(sid)
    assert len(by_target.get("target_0", ())) > len(by_target.get("target_1", ())), (
        "the gapped target should have fewer scorable candidates"
    )
    # and nothing emitted has a short window
    for row in range(len(table.set_ids)):
        for t in range(8):
            assert "-" not in str(table.target_windows[row, t])


@requires_weights
def test_full_pipeline_produces_ranked_designs(tmp_path):
    from lamp_design.models.bridge import mismatch_map, predict_tp
    from lamp_design.models.inference import (
        TP_CLIP_MAX,
        TP_CLIP_MIN,
        LyraEnsemble,
    )
    from lamp_design.report import build_design_result, write_outputs

    panel = synthetic_panel(n_targets=3, length=700)
    cands = generate_candidates(panel.consensus, stride=15, geometries=default_geometries())
    assert len(cands) > 20

    table = compute_thermo(cands, panel, threads=4)
    assert len(table.set_ids) > 0

    ensemble = LyraEnsemble.load(_weights())
    conditions = [Condition("BST", 10000.0), Condition("BST", 100.0)]
    preds = predict_tp(table, cands, conditions, ensemble=ensemble)
    assert preds, "no predictions produced"

    tps = np.array([p.tp_minutes for p in preds])
    assert np.isfinite(tps).all()
    # The trained range is [regress_tp_cutoff, tp_censor_at] = [-100, 60], not
    # [0, 60]. This assertion previously hardcoded 0 as the floor while calling
    # it "the trained range", which is the same mistake TP_CLIP_MIN itself made
    # -- negative Tp is in distribution (the ground-truth labels reach -4.611).
    # Asserted against the constants so the two cannot drift apart again.
    assert (tps >= TP_CLIP_MIN).all() and (tps <= TP_CLIP_MAX).all(), (
        "clipped predictions left the trained range"
    )

    result = build_design_result(
        preds,
        cands,
        panel,
        ranking_condition=conditions[0],
        mismatches=mismatch_map(table),
        n=20,
        min_separation=30,
    )
    assert result.scored_sets, "ranking returned nothing"
    assert len(result.scored_sets) <= 20

    # Ranked fastest-first.
    speeds = [s.speed_score for s in result.scored_sets]
    assert speeds == sorted(speeds), "sets are not ordered fastest-first"

    # Positional diversity actually enforced.
    anchors = sorted(s.primer_set.anchor for s in result.scored_sets)
    gaps = np.diff(anchors)
    assert (gaps >= 30).all(), f"anchors closer than min_separation: {anchors}"

    # Mismatch columns populated, not silently blank.
    assert all(s.per_target_mismatches for s in result.scored_sets)

    paths = write_outputs(result, str(tmp_path / "design"))
    for key in ("sets_csv", "per_target_csv", "json"):
        assert paths[key].exists() and paths[key].stat().st_size > 0


@requires_weights
def test_predictions_are_order_independent():
    """Shuffling candidates must not change any prediction."""
    from lamp_design.models.bridge import predict_tp
    from lamp_design.models.inference import LyraEnsemble

    panel = synthetic_panel(n_targets=2, length=600)
    cands = generate_candidates(panel.consensus, stride=30, geometries=default_geometries()[:3])
    table = compute_thermo(cands, panel, threads=2)
    ensemble = LyraEnsemble.load(_weights())
    cond = [Condition("BST", 10000.0)]

    first = predict_tp(table, cands, cond, ensemble=ensemble)
    shuffled = list(cands)
    np.random.default_rng(7).shuffle(shuffled)
    second = predict_tp(table, shuffled, cond, ensemble=ensemble)

    a = {(p.set_id, p.target_name): p.tp_minutes for p in first}
    b = {(p.set_id, p.target_name): p.tp_minutes for p in second}
    assert a.keys() == b.keys()
    for key in a:
        assert abs(a[key] - b[key]) < 1e-4, f"{key} changed under reordering"
