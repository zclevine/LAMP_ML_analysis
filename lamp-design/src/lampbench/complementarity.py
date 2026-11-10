from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analysis import load_oof
from .data import (TABULAR_FEATURES, TABULAR_FEATURE_GROUPS, load_units,
                   split_confirmatory)
from .metrics import regression_metrics


TRAIN_FILES = ["LMLv1.csv", "LMLv2.csv", "Systematic.csv"]
DEFAULT_MODELS = [
    "lyra_seqonly", "lyra_corrected", "lyra_deep", "lyra_crossgate", "catboost",
]

# These groups are mutually exclusive and exhaust the corrected 70-feature
# tabular representation. Keeping them disjoint makes permutation losses and
# attribution totals interpretable.
FEATURE_GROUPS = TABULAR_FEATURE_GROUPS


def _validate_feature_partition() -> None:
    flattened = list(itertools.chain.from_iterable(FEATURE_GROUPS.values()))
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Complementarity feature groups overlap")
    if set(flattened) != set(TABULAR_FEATURES):
        missing = sorted(set(TABULAR_FEATURES) - set(flattened))
        extra = sorted(set(flattened) - set(TABULAR_FEATURES))
        raise RuntimeError(f"Feature groups do not partition TABULAR_FEATURES: {missing=}, {extra=}")


def _rho(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _align_oof(prediction_root: Path, models: list[str]):
    loaded = {model: load_oof(prediction_root, "positive", model) for model in models}
    inventories = {model: set(frame.group_id.astype(str))
                   for model, (frame, _, _) in loaded.items()}
    reference = inventories[models[0]]
    if any(inventory != reference for inventory in inventories.values()):
        sizes = {model: len(value) for model, value in inventories.items()}
        raise ValueError(f"Models lack identical OOF group coverage: {sizes}")

    ordered_ids = sorted(reference)
    frame = None
    target = None
    predictions = {}
    for model, (model_frame, y, pred) in loaded.items():
        positions = pd.Series(np.arange(len(model_frame)),
                              index=model_frame.group_id.astype(str)).loc[ordered_ids].to_numpy()
        aligned_frame = model_frame.iloc[positions].reset_index(drop=True)
        aligned_y = np.asarray(y)[positions]
        if frame is None:
            frame, target = aligned_frame, aligned_y
        elif not np.allclose(target, aligned_y, equal_nan=True):
            raise ValueError(f"Target mismatch while aligning {model}")
        predictions[model] = np.asarray(pred)[positions]
    return frame, np.asarray(target), predictions


def _catboost_attribution(units: pd.DataFrame, prediction_root: Path, repeats: int,
                          seed: int, threads: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute held-out SHAP and group-permutation importance fold by fold."""
    from catboost import Pool

    unit_index = units.set_index("group_id", drop=False)
    shap_totals = np.zeros(len(TABULAR_FEATURES), dtype=np.float64)
    shap_n = 0
    permutation_rows = []
    model_root = prediction_root / "positive" / "catboost"
    for model_path in sorted(model_root.glob("seed_*/*/model.pkl")):
        fold_dir = model_path.parent
        prediction_path = fold_dir / "predictions.npz"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        data = np.load(prediction_path, allow_pickle=False)
        group_ids = data["group_id"].astype(str)
        missing = sorted(set(group_ids) - set(unit_index.index.astype(str)))
        if missing:
            raise ValueError(f"OOF groups missing from unit table: {missing[:5]}")
        outer = unit_index.loc[group_ids]
        y = np.asarray(data["y"], dtype=float)
        baseline_pred = np.asarray(data["pred"], dtype=float)
        baseline = _rho(y, baseline_pred)
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        estimator = artifact["estimator"]
        feature_cols = artifact["feature_cols"]
        if list(feature_cols) != list(TABULAR_FEATURES):
            raise ValueError(f"Unexpected CatBoost feature inventory in {model_path}")
        x = outer[feature_cols].to_numpy(np.float32)

        # The fold estimator was fit from NumPy and therefore carries ordinal
        # feature identifiers; keep the evaluation Pool ordinal as well and map
        # columns back to the audited feature_cols list ourselves.
        shap = estimator.get_feature_importance(
            Pool(x), type="ShapValues", thread_count=threads)
        shap_totals += np.abs(np.asarray(shap)[:, :-1]).sum(axis=0)
        shap_n += len(x)

        fold_seed = seed + int(fold_dir.parent.name.removeprefix("seed_")) * 100
        fold_seed += int(fold_dir.name.removeprefix("block"))
        rng = np.random.default_rng(fold_seed)
        for group, names in FEATURE_GROUPS.items():
            columns = [feature_cols.index(name) for name in names]
            losses = []
            for _ in range(repeats):
                permuted = x.copy()
                order = rng.permutation(len(x))
                # One shared row permutation preserves dependence within a
                # biological feature group while breaking its link to outcome.
                permuted[:, columns] = permuted[order][:, columns]
                losses.append(baseline - _rho(
                    y, estimator.predict(permuted, thread_count=threads)))
            permutation_rows.append({
                "seed": int(fold_dir.parent.name.removeprefix("seed_")),
                "block": fold_dir.name,
                "feature_group": group,
                "baseline_spearman": baseline,
                "mean_spearman_loss": float(np.nanmean(losses)),
                "sd_spearman_loss": float(np.nanstd(losses, ddof=1)),
                "n_outer": int(len(x)),
                "n_repeats": int(repeats),
            })

    if shap_n == 0:
        raise FileNotFoundError(f"No CatBoost fold artifacts under {model_root}")
    shap_frame = pd.DataFrame({
        "feature": TABULAR_FEATURES,
        "mean_abs_oof_shap": shap_totals / shap_n,
    })
    feature_to_group = {feature: group for group, features in FEATURE_GROUPS.items()
                        for feature in features}
    shap_frame["feature_group"] = shap_frame.feature.map(feature_to_group)
    shap_frame = shap_frame.sort_values("mean_abs_oof_shap", ascending=False)
    return shap_frame, pd.DataFrame(permutation_rows)


def analyze(data_dir: Path, prediction_root: Path, output: Path,
            models: list[str] | None = None, repeats: int = 20,
            seed: int = 20260813, threads: int = 8) -> dict:
    _validate_feature_partition()
    models = models or list(DEFAULT_MODELS)
    if "lyra_seqonly" not in models or "catboost" not in models:
        raise ValueError("Complementarity analysis requires lyra_seqonly and catboost")
    frame, y, predictions = _align_oof(prediction_root, models)
    units, _ = split_confirmatory(load_units(data_dir, TRAIN_FILES))
    unit_index = units.set_index("group_id", drop=False)
    missing = sorted(set(frame.group_id.astype(str)) - set(unit_index.index.astype(str)))
    if missing:
        raise ValueError(f"OOF groups missing from unit table: {missing[:5]}")
    joined = unit_index.loc[frame.group_id.astype(str)].reset_index(drop=True).copy()

    metrics = {model: regression_metrics(y, pred) for model, pred in predictions.items()}
    errors = {model: y - pred for model, pred in predictions.items()}
    candidate_lyras = [model for model in models
                       if model.startswith("lyra_") and model != "lyra_seqonly"]
    best_lyra = max(candidate_lyras, key=lambda model: metrics[model]["spearman"])
    pair_rows = []
    for a, b in itertools.combinations(models, 2):
        oracle = np.where(np.abs(errors[a]) <= np.abs(errors[b]), predictions[a], predictions[b])
        pair_rows.append({
            "model_a": a,
            "model_b": b,
            "prediction_spearman": _rho(predictions[a], predictions[b]),
            "residual_spearman": _rho(errors[a], errors[b]),
            "absolute_error_spearman": _rho(np.abs(errors[a]), np.abs(errors[b])),
            "fraction_a_lower_absolute_error": float(np.mean(
                np.abs(errors[a]) < np.abs(errors[b]))),
            "oracle_spearman_upper_bound": _rho(y, oracle),
        })

    seq_error = np.abs(errors["lyra_seqonly"])
    cat_error = np.abs(errors["catboost"])
    cat_advantage = seq_error - cat_error
    corrected_advantage = (seq_error - np.abs(errors["lyra_corrected"])
                           if "lyra_corrected" in errors else np.full(len(y), np.nan))
    best_lyra_advantage = seq_error - np.abs(errors[best_lyra])
    association_rows = []
    for feature in TABULAR_FEATURES:
        values = joined[feature].to_numpy(float)
        association_rows.append({
            "feature": feature,
            "feature_group": next(group for group, names in FEATURE_GROUPS.items()
                                  if feature in names),
            "target_spearman": _rho(values, y),
            "catboost_advantage_spearman": _rho(values, cat_advantage),
            "corrected_lyra_advantage_spearman": _rho(values, corrected_advantage),
            "best_lyra": best_lyra,
            "best_lyra_advantage_spearman": _rho(values, best_lyra_advantage),
        })
    associations = pd.DataFrame(association_rows)
    associations["abs_catboost_advantage_spearman"] = associations[
        "catboost_advantage_spearman"].abs()
    associations = associations.sort_values("abs_catboost_advantage_spearman", ascending=False)

    stratum_rows = []
    strata = {
        "source": joined.source.astype(str),
        "chemistry": joined.canonical_mm.astype(str),
        "concentration": joined.target_conc.astype(str),
        "replicate_count": joined.n_replicates.astype(str),
        "amplification_consistency": pd.cut(
            joined.amp_fraction, [-0.01, 0.0, 0.999999, 1.0],
            labels=["none", "partial", "all"]).astype(str),
        "total_mismatch_quartile": pd.qcut(
            joined.total_mismatches, 4, duplicates="drop").astype(str),
    }
    for stratum, labels in strata.items():
        for level in sorted(labels.unique()):
            idx = np.flatnonzero(labels.to_numpy() == level)
            for model, pred in predictions.items():
                result = regression_metrics(y[idx], pred[idx])
                stratum_rows.append({
                    "stratum": stratum, "level": level, "model": model,
                    **result,
                })

    shap_frame, permutation = _catboost_attribution(
        units, prediction_root, repeats=repeats, seed=seed, threads=threads)
    permutation_summary = permutation.groupby("feature_group", as_index=False).agg(
        mean_spearman_loss=("mean_spearman_loss", "mean"),
        sd_across_folds=("mean_spearman_loss", "std"),
        n_folds=("block", "size"),
    ).sort_values("mean_spearman_loss", ascending=False)
    shap_group = shap_frame.groupby("feature_group", as_index=False).agg(
        total_mean_abs_oof_shap=("mean_abs_oof_shap", "sum"),
        mean_feature_abs_oof_shap=("mean_abs_oof_shap", "mean"),
        n_features=("feature", "size"),
    ).sort_values("total_mean_abs_oof_shap", ascending=False)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(output / "model_error_complementarity.csv", index=False)
    associations.to_csv(output / "feature_residual_associations.csv", index=False)
    pd.DataFrame(stratum_rows).to_csv(output / "conditional_performance.csv", index=False)
    shap_frame.to_csv(output / "catboost_oof_shap_features.csv", index=False)
    shap_group.to_csv(output / "catboost_oof_shap_groups.csv", index=False)
    permutation.to_csv(output / "catboost_group_permutation_folds.csv", index=False)
    permutation_summary.to_csv(output / "catboost_group_permutation_summary.csv", index=False)

    payload = {
        "scope": "exploratory internal OOF diagnostics; frozen data not used",
        "prediction_root": str(prediction_root),
        "models": models,
        "n_oof_units": int(len(y)),
        "feature_groups": FEATURE_GROUPS,
        "metrics": metrics,
        "best_lyra": best_lyra,
        "model_error_complementarity": pair_rows,
        "top_catboost_advantage_associations": associations.head(15).to_dict("records"),
        "catboost_group_permutation": permutation_summary.to_dict("records"),
        "catboost_shap_groups": shap_group.to_dict("records"),
        "random_seed": seed,
        "permutation_repeats_per_fold": repeats,
        "catboost_threads": threads,
    }
    (output / "complementarity_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    payload = analyze(args.data_dir, args.prediction_root, args.output, args.models,
                      args.permutation_repeats, threads=args.threads)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
