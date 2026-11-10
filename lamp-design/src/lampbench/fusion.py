from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_oof
from .metrics import cluster_bootstrap, selection_value, stratified_metrics, task_metrics


def _aligned_predictions(sequence_root: Path, tabular_root: Path, task: str,
                         sequence_model: str, tabular_model: str):
    sf, sy, sp = load_oof(sequence_root, task, sequence_model)
    tf, ty, tp = load_oof(tabular_root, task, tabular_model)
    s_pos = pd.Series(np.arange(len(sf)), index=sf.group_id.astype(str))
    t_pos = pd.Series(np.arange(len(tf)), index=tf.group_id.astype(str))
    if set(s_pos.index) != set(t_pos.index):
        raise ValueError("Fusion candidates do not have identical group coverage")
    order = s_pos.index.sort_values()
    si, ti = s_pos.loc[order].to_numpy(), t_pos.loc[order].to_numpy()
    if not np.allclose(sy[si], ty[ti], equal_nan=True):
        raise ValueError("Fusion candidate targets differ")
    frame = sf.iloc[si].reset_index(drop=True)
    return frame, sy[si], sp[si], tp[ti]


def probe_fusion(sequence_root: Path, tabular_root: Path, task: str,
                 sequence_models: list[str], tabular_model: str,
                 weights: list[float] | None = None) -> dict:
    weights = weights or [round(x, 2) for x in np.linspace(0, 1, 21)]
    rows = []
    for sequence_model in sequence_models:
        frame, y, sequence_pred, tabular_pred = _aligned_predictions(
            sequence_root, tabular_root, task, sequence_model, tabular_model)
        for sequence_weight in weights:
            pred = sequence_weight * sequence_pred + (1 - sequence_weight) * tabular_pred
            pooled = selection_value(task, task_metrics(task, y, pred))
            block_values = []
            for _, part in frame.groupby("block"):
                idx = part.index.to_numpy()
                block_values.append(selection_value(task, task_metrics(task, y[idx], pred[idx])))
            rows.append({
                "sequence_model": sequence_model,
                "tabular_model": tabular_model,
                "sequence_weight": sequence_weight,
                "tabular_weight": round(1 - sequence_weight, 2),
                "pooled_selection": float(pooled),
                "block_macro_mean": float(np.nanmean(block_values)),
                "block_macro_sd": float(np.nanstd(block_values, ddof=1)),
                "n_units": len(frame),
            })
    ranked = sorted(rows, key=lambda row: row["block_macro_mean"], reverse=True)
    return {
        "task": task,
        "selection_policy": "screen-only weight grid; rank by block-macro mean",
        "weights": weights,
        "best": ranked[0],
        "rows": ranked,
    }


def materialize_fusion(root: Path, task: str, spec: dict) -> dict:
    """Write a pre-selected value blend as ordinary OOF prediction artifacts."""
    sequence_model = spec["sequence_model"]
    tabular_model = spec["tabular_model"]
    name = spec["name"]
    sequence_weight = float(spec["sequence_weight"])
    tabular_weight = float(spec["tabular_weight"])
    if not np.isclose(sequence_weight + tabular_weight, 1):
        raise ValueError("Fusion weights must sum to one")
    records = []
    for sequence_path in sorted(
            (root / task / sequence_model).glob("seed_*/*/predictions.npz")):
        seed_dir, block = sequence_path.parents[1].name, sequence_path.parent.name
        tabular_path = root / task / tabular_model / seed_dir / block / "predictions.npz"
        if not tabular_path.exists():
            raise FileNotFoundError(f"Missing fusion base prediction: {tabular_path}")
        sequence = np.load(sequence_path, allow_pickle=False)
        tabular = np.load(tabular_path, allow_pickle=False)
        sequence_ids = sequence["group_id"].astype(str)
        tabular_ids = tabular["group_id"].astype(str)
        if set(sequence_ids) != set(tabular_ids):
            raise ValueError(f"Fusion group coverage differs for {seed_dir}/{block}")
        tabular_index = {gid: i for i, gid in enumerate(tabular_ids)}
        order = np.asarray([tabular_index[gid] for gid in sequence_ids])
        y = np.asarray(sequence["y"])
        if not np.allclose(y, np.asarray(tabular["y"])[order], equal_nan=True):
            raise ValueError(f"Fusion targets differ for {seed_dir}/{block}")
        pred = (sequence_weight * np.asarray(sequence["pred"]) +
                tabular_weight * np.asarray(tabular["pred"])[order])
        frame = pd.DataFrame({
            "group_id": sequence_ids,
            "primer_set": sequence["primer_set"].astype(str),
            "canonical_mm": sequence["canonical_mm"].astype(str),
            "target_conc": sequence["target_conc"].astype(float),
            "n_replicates": (sequence["n_replicates"].astype(int)
                             if "n_replicates" in sequence.files else 1),
            "amp_fraction": (sequence["amp_fraction"].astype(float)
                             if "amp_fraction" in sequence.files else 1.0),
        })
        dest = root / task / name / seed_dir / block
        dest.mkdir(parents=True, exist_ok=True)
        result = {
            "model": name, "task": task, "seed": int(seed_dir[5:]),
            "outer_block": block, "elapsed_sec": 0.0,
            "fusion": spec, "metrics": stratified_metrics(task, frame, y, pred),
            "bootstrap": cluster_bootstrap(task, frame, y, pred, n_boot=200),
        }
        (dest / "result.json").write_text(json.dumps(result, indent=2))
        arrays = {key: sequence[key] for key in sequence.files if key != "pred"}
        arrays["pred"] = pred
        np.savez_compressed(dest / "predictions.npz", **arrays)
        records.append({"seed": int(seed_dir[5:]), "block": block,
                        "selection": selection_value(task, result["metrics"]["overall"])})
    if not records:
        raise FileNotFoundError(f"No OOF predictions for fusion base {sequence_model}")
    return {"fusion": spec, "n_results": len(records), "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--tabular-root", type=Path, required=True)
    parser.add_argument("--sequence-models", nargs="+")
    parser.add_argument("--tabular-model", default="catboost")
    parser.add_argument("--task", default="positive")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-file", type=Path)
    args = parser.parse_args()
    if args.selection_file:
        selection = json.loads(args.selection_file.read_text())
        spec = selection.get("selected_fusion")
        if not spec:
            raise SystemExit("Selection file does not contain selected_fusion")
        payload = materialize_fusion(args.sequence_root, args.task, spec)
        print(json.dumps(payload, indent=2))
        return
    if not args.sequence_models or args.output is None:
        raise SystemExit("Probe mode requires --sequence-models and --output")
    payload = probe_fusion(args.sequence_root, args.tabular_root, args.task,
                           args.sequence_models, args.tabular_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["best"], indent=2))


if __name__ == "__main__":
    main()
