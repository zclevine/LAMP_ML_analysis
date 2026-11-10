from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import (TABULAR_FEATURES, isolate_outer_fold, load_units,
                   restrict_train_chemistry, split_confirmatory,
                   tabular_frame_for_ablation)
from .metrics import cluster_bootstrap, stratified_metrics


TRAIN_FILES = ["LMLv1.csv", "LMLv2.csv", "Systematic.csv"]
MODELS = {
    "aft_normal": ("normal", 1.0),
    "aft_normal_s5": ("normal", 5.0),
    "aft_logistic": ("logistic", 1.0),
    "aft_extreme": ("extreme", 1.0),
}


def _label_and_ceiling(task: str) -> tuple[str, float]:
    if task == "tp60_censored":
        return "tp60_censored", 60.0
    if task == "tp120":
        return "tp120", 120.0
    raise ValueError(f"AFT task must be tp120 or tp60_censored, got {task!r}")


def fit_predict(train, test, model_name: str, seed: int, ablation: str = "none",
                task: str = "tp120"):
    distribution, scale = MODELS[model_name]
    label, ceiling = _label_and_ceiling(task)
    fit_train = tabular_frame_for_ablation(train, ablation)
    fit_test = tabular_frame_for_ablation(test, ablation)
    x_train = fit_train[TABULAR_FEATURES].to_numpy(np.float32)
    x_test = fit_test[TABULAR_FEATURES].to_numpy(np.float32)
    y = train[label].to_numpy(np.float32)
    plateau = y >= (ceiling - 1e-3)

    dtrain = xgb.DMatrix(x_train)
    dtrain.set_float_info("label_lower_bound", y)
    dtrain.set_float_info(
        "label_upper_bound", np.where(plateau, np.inf, y).astype(np.float32))
    dtest = xgb.DMatrix(x_test)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": distribution,
        "aft_loss_distribution_scale": scale,
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": seed,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=800)
    pred = np.clip(booster.predict(dtest), 0.0, ceiling)
    return booster, pred


def run_fold(units, outer_block: str, model_name: str, seed: int, output: Path,
             ablation: str = "none", task: str = "tp120",
             train_chemistry: str | None = None):
    label, ceiling = _label_and_ceiling(task)
    train, outer, isolation = isolate_outer_fold(units, outer_block)
    train = restrict_train_chemistry(train, train_chemistry)
    train = train.reset_index(drop=True)
    outer = outer.reset_index(drop=True)
    y = outer[label].to_numpy(np.float32)
    started = time.time()
    model, pred = fit_predict(train, outer, model_name, seed, ablation, task=task)
    result = {
        "model": model_name,
        "task": task,
        "seed": seed,
        "outer_block": outer_block,
        "ablation": ablation,
        "method": "XGBoost accelerated-failure-time survival regression",
        "censoring": (
            f"exact TP below {ceiling:.0f}; right-censored at {ceiling:.0f} "
            "for non-amplification"
        ),
        "elapsed_sec": time.time() - started,
        "n_train": len(train),
        "n_outer": len(outer),
        "train_chemistry": train_chemistry,
        "n_train_right_censored": int((train[label] >= (ceiling - 1e-3)).sum()),
        "sequence_identity_isolation": {"outer": isolation},
        "metrics": stratified_metrics(task, outer, y, pred),
        "bootstrap": cluster_bootstrap(task, outer, y, pred, n_boot=200),
    }
    fold_dir = output / task / model_name / f"seed_{seed}" / outer_block
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(fold_dir / "model.ubj")
    np.savez_compressed(
        fold_dir / "predictions.npz",
        y=y,
        pred=pred,
        group_id=outer.group_id.astype(str).to_numpy(dtype=str),
        primer_set=outer.primer_set.astype(str).to_numpy(dtype=str),
        canonical_mm=outer.canonical_mm.astype(str).to_numpy(dtype=str),
        target_conc=outer.target_conc.to_numpy(),
        n_replicates=outer.n_replicates.to_numpy(),
        amp_fraction=outer.amp_fraction.to_numpy(),
    )
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def summarize(output: Path):
    rows = []
    for path in output.glob("tp120/*/seed_*/*/result.json"):
        result = json.loads(path.read_text())
        rows.append({
            "task": "tp120",
            "model": result["model"],
            "block": result["outer_block"],
            "selection": result["metrics"]["overall"]["spearman"],
            "mae": result["metrics"]["overall"]["mae"],
            "elapsed_sec": result["elapsed_sec"],
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    summary = frame.groupby(["task", "model"], as_index=False).agg(
        mean=("selection", "mean"),
        std=("selection", "std"),
        mean_mae=("mae", "mean"),
        n=("selection", "size"),
        elapsed_sec=("elapsed_sec", "sum"),
    ).sort_values("mean", ascending=False)
    frame.to_csv(output / "all_fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(summary.to_json(orient="records", indent=2))
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument("--blocks", nargs="+", default=["block0", "block4", "block9"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--task", default="tp120", choices=["tp120", "tp60_censored"])
    parser.add_argument("--train-chemistry", default=None)
    args = parser.parse_args()
    unknown = set(args.models) - MODELS.keys()
    if unknown:
        raise SystemExit(f"Unknown censored models: {sorted(unknown)}")

    args.output.mkdir(parents=True, exist_ok=True)
    units, _ = split_confirmatory(load_units(args.data_dir, TRAIN_FILES))
    failures = []
    for model_name in args.models:
        for block in args.blocks:
            result_path = (
                args.output / args.task / model_name / f"seed_{args.seed}" /
                block / "result.json"
            )
            if result_path.exists():
                continue
            try:
                result = run_fold(
                    units, block, model_name, args.seed, args.output, args.ablation,
                    task=args.task, train_chemistry=args.train_chemistry)
                print(json.dumps({
                    "status": "complete",
                    "model": model_name,
                    "block": block,
                    "selection": result["metrics"]["overall"]["spearman"],
                }), flush=True)
            except Exception as exc:
                failures.append((model_name, block, repr(exc)))
                print(json.dumps({
                    "status": "error", "model": model_name,
                    "block": block, "error": repr(exc),
                }), flush=True)
    summarize(args.output)
    if failures:
        raise RuntimeError(f"{len(failures)} censored folds failed: {failures[:5]}")


if __name__ == "__main__":
    main()
