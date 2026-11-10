from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import TABULAR_FEATURES, THERMO_FEATURES, load_units, split_confirmatory
from .extended import (
    TRAIN_FILES,
    _catboost_regressor,
    _lyra_config,
    _metadata,
    _predict_lyra,
)
from .extended_analysis import _cluster_bootstrap_difference
from .metrics import regression_metrics
from .tabular import build_tabular
from .train import fit_epochs


VIRUS_FILES = {
    "mpox_wt": "mpox.csv",
    "mpox_mutation_panel": "mpox_validation.csv",
    "borrelia": "Borrelia.csv",
    "lasv": "LASV.csv",
}
AFT_MODELS = {"aft_normal": ("normal", 1.0)}


def _save_predictions(path: Path, frame: pd.DataFrame, y, **predictions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y=np.asarray(y, np.float32),
        **{name: np.asarray(pred, np.float32) for name, pred in predictions.items()},
        **_metadata(frame.reset_index(drop=True)),
    )


def _external_sets(data_dir: Path, development: pd.DataFrame):
    development_ids = set(development.sequence_group_id.astype(str))
    sets = {}
    for name, filename in VIRUS_FILES.items():
        frame = load_units(data_dir, [filename]).reset_index(drop=True)
        overlap = development_ids & set(frame.sequence_group_id.astype(str))
        if overlap:
            raise RuntimeError(f"External sequence-identity overlap for {name}: {len(overlap)}")
        sets[name] = frame
    return sets


def _fit_aft(train: pd.DataFrame, seed: int):
    import xgboost as xgb

    distribution, scale = AFT_MODELS["aft_normal"]
    x_train = train[TABULAR_FEATURES].to_numpy(np.float32)
    y = train.tp120.to_numpy(np.float32)
    plateau = y >= 119.999
    dtrain = xgb.DMatrix(x_train)
    dtrain.set_float_info("label_lower_bound", y)
    dtrain.set_float_info(
        "label_upper_bound", np.where(plateau, np.inf, y).astype(np.float32)
    )
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
    return xgb.train(params, dtrain, num_boost_round=800)


def run_tabular(data_dir: Path, output: Path, seed: int) -> None:
    import xgboost as xgb

    development, _ = split_confirmatory(load_units(data_dir, TRAIN_FILES))
    external = _external_sets(data_dir, development)
    started = time.time()

    sameed = build_tabular("sameed_rf", seed)
    sameed.fit(
        development[THERMO_FEATURES].to_numpy(np.float32),
        development.tp120.to_numpy(np.float32),
    )
    catboost = _catboost_regressor(seed, "gpu")
    raw_y = development.tp120.to_numpy(np.float32)
    catboost.fit(
        development[TABULAR_FEATURES].to_numpy(np.float32),
        raw_y,
        sample_weight=np.where(raw_y >= 119.999, 0.25, 1.0),
    )
    tabpfn = build_tabular("tabpfn", seed)
    tabpfn.fit(development[TABULAR_FEATURES].to_numpy(np.float32), raw_y)
    aft = _fit_aft(development, seed)

    results = []
    for virus, frame in external.items():
        x = frame[TABULAR_FEATURES].to_numpy(np.float32)
        y = frame.tp120.to_numpy(np.float32)
        predictions = {
            "sameed": sameed.predict(frame[THERMO_FEATURES].to_numpy(np.float32)),
            "tabpfn": tabpfn.predict(x),
            "weighted_catboost": catboost.predict(x),
            "aft": np.clip(aft.predict(xgb.DMatrix(x)), 0.0, 120.0),
        }
        _save_predictions(output / virus / "predictions.npz", frame, y, **predictions)
        metrics = {name: regression_metrics(y, pred) for name, pred in predictions.items()}
        result = {
            "stage": "virus_tabular",
            "virus": virus,
            "source_file": VIRUS_FILES[virus],
            "n_train": len(development),
            "n_test": len(frame),
            "sequence_identity_overlap": 0,
            "metrics": metrics,
        }
        (output / virus).mkdir(parents=True, exist_ok=True)
        (output / virus / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (output / "summary.json").write_text(json.dumps({
        "seed": seed,
        "elapsed_sec": time.time() - started,
        "results": results,
    }, indent=2))


def run_lyra(data_dir: Path, output: Path, seed: int, epochs: int) -> None:
    import torch

    development, _ = split_confirmatory(load_units(data_dir, TRAIN_FILES))
    external = _external_sets(data_dir, development)
    cfg = _lyra_config("none", seed)
    train_y = development.tp120.to_numpy(np.float32)
    started = time.time()
    model, standardizer, mean, target_std, _, _ = fit_epochs(
        development, train_y, None, None, cfg, fixed_epochs=epochs
    )
    results = []
    for virus, frame in external.items():
        y = frame.tp120.to_numpy(np.float32)
        _, pred = _predict_lyra(model, frame, y, standardizer, cfg, mean, target_std)
        pred = np.asarray(pred, np.float32)
        _save_predictions(output / virus / "predictions.npz", frame, y, lyra=pred)
        result = {
            "stage": "virus_lyra",
            "virus": virus,
            "source_file": VIRUS_FILES[virus],
            "n_train": len(development),
            "n_test": len(frame),
            "sequence_identity_overlap": 0,
            "fixed_epochs": epochs,
            "config": asdict(cfg),
            "metrics": regression_metrics(y, pred),
        }
        (output / virus).mkdir(parents=True, exist_ok=True)
        (output / virus / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    (output / "summary.json").write_text(json.dumps({
        "seed": seed,
        "fixed_epochs": epochs,
        "elapsed_sec": time.time() - started,
        "results": results,
    }, indent=2))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_spearman(y, pred):
    if len(y) < 3 or np.unique(y).size < 2 or np.unique(pred).size < 2:
        return np.nan
    return float(spearmanr(y, pred).statistic)


def _breakdown(frame: pd.DataFrame, y, pred) -> dict:
    result = {"overall": regression_metrics(y, pred)}
    amplified = y < 119.999
    result["amplified_only"] = regression_metrics(y[amplified], pred[amplified])
    concentration_rows = []
    for concentration, indices in frame.groupby("target_conc").groups.items():
        idx = np.asarray(list(indices), int)
        metrics = regression_metrics(y[idx], pred[idx])
        concentration_rows.append({"target_conc": float(concentration), **metrics})
    result["by_concentration"] = concentration_rows
    valid = [row["spearman"] for row in concentration_rows if np.isfinite(row["spearman"])]
    result["macro_within_concentration_spearman"] = (
        float(np.mean(valid)) if valid else np.nan
    )
    chemistry_rows = []
    for chemistry, indices in frame.groupby("canonical_mm").groups.items():
        idx = np.asarray(list(indices), int)
        chemistry_rows.append({"canonical_mm": str(chemistry), **regression_metrics(y[idx], pred[idx])})
    result["by_chemistry"] = chemistry_rows
    cell_correlations = []
    for _, indices in frame.groupby(["target_conc", "canonical_mm"]).groups.items():
        idx = np.asarray(list(indices), int)
        rho = _safe_spearman(y[idx], pred[idx])
        if np.isfinite(rho):
            cell_correlations.append(rho)
    result["macro_within_concentration_chemistry_spearman"] = (
        float(np.mean(cell_correlations)) if cell_correlations else np.nan
    )
    result["n_concentration_chemistry_cells"] = len(cell_correlations)
    return result


def analyze(
    tabular_root: Path,
    lyra_root: Path,
    output: Path,
    bootstrap: int = 2000,
    seed: int = 0,
) -> None:
    model_keys = ["sameed", "lyra", "tabpfn", "weighted_catboost", "aft", "fixed_fusion"]
    rows = []
    details = []
    comparisons = []
    for virus in VIRUS_FILES:
        tabular = np.load(tabular_root / virus / "predictions.npz", allow_pickle=True)
        lyra = np.load(lyra_root / virus / "predictions.npz", allow_pickle=True)
        ids = tabular["group_id"].astype(str)
        if not np.array_equal(ids, lyra["group_id"].astype(str)):
            raise RuntimeError(f"Virus prediction identity mismatch: {virus}")
        y = tabular["y"].astype(float)
        predictions = {
            "sameed": tabular["sameed"].astype(float),
            "lyra": lyra["lyra"].astype(float),
            "tabpfn": tabular["tabpfn"].astype(float),
            "weighted_catboost": tabular["weighted_catboost"].astype(float),
            "aft": tabular["aft"].astype(float),
        }
        predictions["fixed_fusion"] = (
            0.30 * predictions["lyra"] + 0.20 * predictions["tabpfn"]
            + 0.30 * predictions["weighted_catboost"] + 0.20 * predictions["aft"]
        )
        frame = pd.DataFrame({
            "target_conc": tabular["target_conc"],
            "canonical_mm": tabular["canonical_mm"].astype(str),
        })
        for model in model_keys:
            breakdown = _breakdown(frame, y, predictions[model])
            overall = breakdown["overall"]
            amplified = breakdown["amplified_only"]
            rows.append({
                "virus": virus,
                "model": model,
                "n": len(y),
                "n_amplified": int((y < 119.999).sum()),
                "spearman": overall["spearman"],
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "amplified_only_spearman": amplified["spearman"],
                "amplified_only_mae": amplified["mae"],
                "macro_within_concentration_spearman": breakdown[
                    "macro_within_concentration_spearman"
                ],
                "macro_within_concentration_chemistry_spearman": breakdown[
                    "macro_within_concentration_chemistry_spearman"
                ],
            })
            details.append({"virus": virus, "model": model, **breakdown})
        clusters = tabular["primer_set"].astype(str)
        amplified = y < 119.999
        for reference in ("sameed", "lyra"):
            for subset, mask in (("overall", np.ones(len(y), bool)), ("amplified_only", amplified)):
                for metric in ("spearman", "mae"):
                    comparisons.append({
                        "virus": virus,
                        "candidate": "fixed_fusion",
                        "reference": reference,
                        "subset": subset,
                        "metric": metric,
                        **_cluster_bootstrap_difference(
                            y[mask],
                            predictions["fixed_fusion"][mask],
                            predictions[reference][mask],
                            clusters[mask],
                            metric,
                            bootstrap,
                            seed + len(comparisons),
                        ),
                    })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "virus_performance.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "virus_bootstrap_comparisons.csv", index=False)
    (output / "virus_performance_details.json").write_text(json.dumps(details, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("tabular", "lyra"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=0)
        if name == "lyra":
            command.add_argument("--epochs", type=int, default=5)
    command = sub.add_parser("analyze")
    command.add_argument("--tabular-root", type=Path, required=True)
    command.add_argument("--lyra-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--bootstrap", type=int, default=2000)
    command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "tabular":
        run_tabular(args.data_dir, args.output, args.seed)
    elif args.stage == "lyra":
        run_lyra(args.data_dir, args.output, args.seed, args.epochs)
    else:
        analyze(args.tabular_root, args.lyra_root, args.output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
