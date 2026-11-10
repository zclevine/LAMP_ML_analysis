from __future__ import annotations

import hashlib
import math

import pandas as pd


def deterministic_group_fraction(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Return a nested, group-preserving fraction of a training frame.

    Sampling happens at the assay-unit ``group_id`` level.  Groups are ranked by
    a stable hash within source/block/amplification strata, so 10% is a subset
    of 20%, 20% is a subset of 30%, and so on for a fixed seed.  The held-out
    evaluation frame is never passed to this function.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if frame.empty:
        raise ValueError("cannot subsample an empty training frame")
    if "group_id" not in frame:
        raise ValueError("training frame is missing group_id")

    data = frame.copy()
    if fraction == 1:
        selected = data
    else:
        strata_cols = [c for c in ("source", "block") if c in data]
        group_cols = ["group_id", *strata_cols]
        groups = data[group_cols].drop_duplicates("group_id").copy()
        if "amp_fraction" in data:
            amp = data.groupby("group_id", sort=False)["amp_fraction"].mean()
            groups["amplification_stratum"] = (
                groups["group_id"].map(amp).fillna(0).ge(0.5).astype(int)
            )
            strata_cols.append("amplification_stratum")
        if not strata_cols:
            groups["all"] = "all"
            strata_cols = ["all"]

        def score(group_id: object) -> str:
            payload = f"lampbench-data-efficiency-v1|{seed}|{group_id}".encode()
            return hashlib.sha256(payload).hexdigest()

        groups["_score"] = groups["group_id"].map(score)
        keep: list[object] = []
        grouper = strata_cols[0] if len(strata_cols) == 1 else strata_cols
        for _, stratum in groups.groupby(grouper, sort=True, dropna=False):
            ordered = stratum.sort_values(["_score", "group_id"], kind="stable")
            n_keep = max(1, math.ceil(fraction * len(ordered)))
            keep.extend(ordered.iloc[:n_keep]["group_id"].tolist())
        selected = data[data["group_id"].isin(set(keep))].copy()

    selected_ids = sorted(selected["group_id"].astype(str).unique().tolist())
    signature = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
    manifest = {
        "fraction": float(fraction),
        "sampling_seed": int(seed),
        "sampling_unit": "group_id",
        "strata": [c for c in ("source", "block", "amplification_stratum")
                   if c in (list(frame.columns) + ["amplification_stratum"])],
        "n_rows_available": int(len(frame)),
        "n_rows_selected": int(len(selected)),
        "n_groups_available": int(frame["group_id"].nunique()),
        "n_groups_selected": int(len(selected_ids)),
        "selected_group_ids_sha256": signature,
    }
    return selected, manifest
