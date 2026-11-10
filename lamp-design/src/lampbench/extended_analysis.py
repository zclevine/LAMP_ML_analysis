from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .metrics import regression_metrics


BLOCKS = [f"block{i}" for i in range(10)]
BASE_MODELS = {
    "Sameed RF": "sameed_pred",
    "Lyra physics-token": "lyra_pred",
    "TabPFN v2": "tabpfn_pred",
    "Weighted CatBoost": "weighted_catboost_pred",
    "AFT censored": "aft_pred",
    "Prior Lyra + TabPFN": "prior_lyra_tabpfn_pred",
    "Fixed fusion": "pred",
}
SOURCE_MODELS = {
    "Sameed RF": "sameed",
    "Lyra physics-token": "lyra",
    "TabPFN v2": "tabpfn",
    "Weighted CatBoost": "weighted_catboost",
    "AFT censored": "aft",
    "Fixed fusion": "fixed_fusion",
}
ABLATIONS = {
    "mismatch_count": "drop_mismatch_count",
    "mismatch_position": "drop_mismatch_position",
    "all_mismatch": "drop_all_mismatch",
}


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fdr_bh(values) -> np.ndarray:
    """Benjamini-Hochberg adjustment, preserving NaNs and input order."""
    values = np.asarray(values, float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_index = np.flatnonzero(np.isfinite(values))
    if not len(finite_index):
        return adjusted
    finite = values[finite_index]
    order = np.argsort(finite)
    ranked = finite[order]
    scaled = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    restored = np.empty_like(scaled)
    restored[order] = np.clip(scaled, 0.0, 1.0)
    adjusted[finite_index] = restored
    return adjusted


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _collect_oof(root: Path, variant: str) -> dict[str, np.ndarray]:
    archives = []
    for block in BLOCKS:
        path = root / variant / block / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        archives.append(_load(path))
    keys = set.intersection(*(set(archive) for archive in archives))
    return {key: np.concatenate([archive[key] for archive in archives]) for key in keys}


def _collect_classifier_oof(
    root: Path, variant: str, legacy_lyra_logits: bool = False
) -> dict[str, np.ndarray]:
    archives = []
    for block in BLOCKS:
        path = root / variant / block / "predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        archive = _load(path)
        prediction = archive["pred"].astype(float)
        if legacy_lyra_logits and (
            np.any(prediction < 0.0) or np.any(prediction > 1.0)
        ):
            archive["pred"] = expit(prediction)
        archives.append(archive)
    keys = set.intersection(*(set(archive) for archive in archives))
    return {key: np.concatenate([archive[key] for archive in archives]) for key in keys}


def _align(reference_ids, archive: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = archive["group_id"].astype(str)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Prediction archive contains duplicate group_id values")
    lookup = {value: index for index, value in enumerate(ids)}
    missing = [value for value in reference_ids if str(value) not in lookup]
    if missing:
        raise ValueError(f"Prediction archive is missing {len(missing)} reference units")
    order = np.asarray([lookup[str(value)] for value in reference_ids], dtype=int)
    return {
        key: value[order] if np.ndim(value) > 0 and len(value) == len(ids) else value
        for key, value in archive.items()
    }


def _regression_row(model: str, y, prediction, **extra) -> dict:
    metrics = regression_metrics(np.asarray(y, float), np.asarray(prediction, float))
    y_log = np.log1p(np.clip(np.asarray(y, float), 0, None))
    pred_log = np.log1p(np.clip(np.asarray(prediction, float), 0, None))
    return {
        **extra,
        "model": model,
        **metrics,
        "log_mae": float(np.mean(np.abs(y_log - pred_log))),
        "log_rmse": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
        "n": int(len(y_log)),
    }


def _classification_row(model: str, y, score, binary=None, **extra) -> dict:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    binary = np.asarray(score >= 0.5 if binary is None else binary, bool)
    negative = y == 0
    return {
        **extra,
        "model": model,
        "accuracy": float(accuracy_score(y, binary)),
        "balanced_accuracy": float(balanced_accuracy_score(y, binary)),
        "sensitivity": float(recall_score(y, binary, zero_division=0)),
        "specificity": float(np.mean(~binary[negative])) if negative.any() else float("nan"),
        "precision": float(precision_score(y, binary, zero_division=0)),
        "f1": float(f1_score(y, binary, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, binary)),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }


def _mismatch_bin(values) -> np.ndarray:
    values = np.rint(np.asarray(values, float)).astype(int)
    return np.select(
        [values == 0, values == 1, values == 2, (values >= 3) & (values <= 5)],
        ["0", "1", "2", "3-5"],
        default="6+",
    )


def _terminal_bin(values) -> np.ndarray:
    values = np.rint(np.asarray(values, float)).astype(int)
    return np.select([values == 0, values == 1], ["0", "1"], default="2+")


def _cluster_index(clusters) -> tuple[np.ndarray, list[np.ndarray]]:
    clusters = np.asarray(clusters).astype(str)
    unique = np.unique(clusters)
    return unique, [np.flatnonzero(clusters == name) for name in unique]


def _metric(name: str, y, prediction) -> float:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    if name == "spearman":
        return float(spearmanr(y, prediction).statistic)
    if name == "mae":
        return float(np.mean(np.abs(y - prediction)))
    if name == "log_mae":
        return float(np.mean(np.abs(
            np.log1p(np.clip(y, 0, None)) -
            np.log1p(np.clip(prediction, 0, None))
        )))
    raise ValueError(name)


def _cluster_bootstrap_difference(
    y, candidate, reference, clusters, metric: str, n_boot: int, seed: int
) -> dict:
    y = np.asarray(y)
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    unique, members = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    observed = _metric(metric, y, candidate) - _metric(metric, y, reference)
    draws = np.empty(n_boot, float)
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[position] for position in sampled])
        draws[iteration] = (
            _metric(metric, y[index], candidate[index]) -
            _metric(metric, y[index], reference[index])
        )
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        return {"difference": observed, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    tail_low = (np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1)
    tail_high = (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)
    return {
        "difference": observed,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_value": float(min(1.0, 2 * min(tail_low, tail_high))),
        "n_boot": int(len(finite)),
        "n_clusters": int(len(unique)),
    }


def _accuracy_bootstrap(y, candidate, reference, clusters, n_boot: int, seed: int) -> dict:
    y = np.asarray(y, bool)
    candidate = np.asarray(candidate, bool)
    reference = np.asarray(reference, bool)
    unique, members = _cluster_index(clusters)
    candidate_correct = candidate == y
    reference_correct = reference == y
    observed = float(np.mean(candidate_correct) - np.mean(reference_correct))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, float)
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[position] for position in sampled])
        draws[iteration] = float(
            np.mean(candidate_correct[index]) - np.mean(reference_correct[index])
        )
    tail_low = (np.count_nonzero(draws <= 0) + 1) / (n_boot + 1)
    tail_high = (np.count_nonzero(draws >= 0) + 1) / (n_boot + 1)
    return {
        "difference": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_value": float(min(1.0, 2 * min(tail_low, tail_high))),
        "n_boot": n_boot,
        "n_clusters": int(len(unique)),
    }


def _mcnemar(y, candidate, reference) -> dict:
    y = np.asarray(y, bool)
    candidate_correct = np.asarray(candidate, bool) == y
    reference_correct = np.asarray(reference, bool) == y
    candidate_only = int(np.count_nonzero(candidate_correct & ~reference_correct))
    reference_only = int(np.count_nonzero(~candidate_correct & reference_correct))
    discordant = candidate_only + reference_only
    p_value = float(binomtest(candidate_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def _primer_wilcoxon(y, candidate, reference, primer_sets, log_scale: bool) -> dict:
    y = np.asarray(y, float)
    candidate = np.asarray(candidate, float)
    reference = np.asarray(reference, float)
    if log_scale:
        y = np.log1p(np.clip(y, 0, None))
        candidate = np.log1p(np.clip(candidate, 0, None))
        reference = np.log1p(np.clip(reference, 0, None))
    frame = pd.DataFrame({
        "primer_set": np.asarray(primer_sets).astype(str),
        "candidate": np.abs(y - candidate),
        "reference": np.abs(y - reference),
    }).groupby("primer_set", as_index=False).mean(numeric_only=True)
    difference = frame.candidate.to_numpy() - frame.reference.to_numpy()
    if np.allclose(difference, 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(difference, zero_method="pratt", alternative="two-sided")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "difference": float(np.mean(difference)),
        "statistic": statistic,
        "p_value": p_value,
        "n_primer_sets": int(len(frame)),
    }


def _dataset_performance(source_tabular: Path, source_lyra: Path) -> pd.DataFrame:
    rows = []
    jobs = [
        "source_LMLv1", "source_LMLv2", "source_Systematic",
        "confirmatory_LMLv2_block19",
    ]
    for job in jobs:
        tabular = _load(source_tabular / job / "predictions.npz")
        lyra = _align(tabular["group_id"], _load(source_lyra / job / "predictions.npz"))
        if not np.allclose(tabular["y"], lyra["y"]):
            raise ValueError(f"Source prediction targets do not align: {job}")
        predictions = {
            "sameed": tabular["sameed"],
            "lyra": lyra["pred"],
            "tabpfn": tabular["tabpfn"],
            "weighted_catboost": tabular["weighted_catboost"],
            "aft": tabular["aft"],
        }
        predictions["fixed_fusion"] = (
            0.30 * predictions["lyra"] + 0.20 * predictions["tabpfn"] +
            0.30 * predictions["weighted_catboost"] + 0.20 * predictions["aft"]
        )
        dataset = {
            "source_LMLv1": "LMLv1 held out",
            "source_LMLv2": "LMLv2 development held out",
            "source_Systematic": "Systematic held out",
            "confirmatory_LMLv2_block19": "LMLv2 block19 confirmatory",
        }[job]
        for model, key in SOURCE_MODELS.items():
            rows.append(_regression_row(
                model, tabular["y"], predictions[key],
                dataset=dataset,
                evaluation="confirmatory" if job.startswith("confirmatory") else "source-held-out",
                exact_120_fraction=float(np.mean(tabular["y"] >= 119.999)),
                tp30_prevalence=float(np.mean(tabular["y"] <= 30.0)),
            ))
    return pd.DataFrame(rows)


def _format(value, digits=3):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _report(
    dataset: pd.DataFrame,
    mismatch: pd.DataFrame,
    ablations: pd.DataFrame,
    classification: pd.DataFrame,
    log_performance: pd.DataFrame,
    log_tests: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Extended TP120 analysis: dataset transfer, mismatch, TP30, and log targets",
        "",
        "All development comparisons use the original sequence-isolated LMLv1 block OOF rows. "
        "Dataset transfer holds out an entire source and removes matching sequence identities; "
        "LMLv2 block19 remains the untouched confirmatory set.",
        "",
        "## Performance by dataset",
        "",
        "| Dataset | Model | Spearman | MAE | RMSE | R² | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in dataset.iterrows():
        lines.append(
            f"| {row.dataset} | {row.model} | {_format(row.spearman)} | "
            f"{_format(row.mae, 2)} | {_format(row.rmse, 2)} | {_format(row.r2)} | {int(row.n)} |"
        )
    lines += [
        "",
        "## Performance by mismatch burden",
        "",
        "| Mismatch grouping | Level | Model | Spearman | MAE | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    display = mismatch[mismatch.model.isin(["Sameed RF", "Lyra physics-token", "Fixed fusion"])]
    for _, row in display.iterrows():
        lines.append(
            f"| {row.grouping} | {row.level} | {row.model} | {_format(row.spearman)} | "
            f"{_format(row.mae, 2)} | {int(row.n)} |"
        )
    lines += [
        "",
        "## Engineered mismatch-feature ablations",
        "",
        "| Family | Transform | Variant | Spearman | Δρ vs full | MAE | ΔMAE vs full |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in ablations.iterrows():
        lines.append(
            f"| {row['family']} | {row['transform']} | {row['ablation']} | {_format(row.spearman)} | "
            f"{_format(row.delta_spearman)} | {_format(row.mae, 2)} | {_format(row.delta_mae, 2)} |"
        )
    lines += [
        "",
        "## Classification: amplification by 30 minutes",
        "",
        "The positive class is `TP120 ≤ 30 minutes`. Regression models are thresholded at 30; "
        "dedicated classifiers use a 0.5 probability threshold. Legacy Lyra classifier folds "
        "that stored logits are detected per fold and sigmoid-transformed before scoring.",
        "",
        "| Model | Method | Accuracy | Balanced accuracy | Sensitivity | Specificity | AUROC | AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in classification.iterrows():
        if row.get("ablation", "full") != "full":
            continue
        lines.append(
            f"| {row.model} | {row.method} | {_format(row.accuracy)} | "
            f"{_format(row.balanced_accuracy)} | {_format(row.sensitivity)} | "
            f"{_format(row.specificity)} | {_format(row.roc_auc)} | "
            f"{_format(row.average_precision)} |"
        )
    lines += [
        "",
        "### TP30 mismatch ablations for dedicated classifiers",
        "",
        "| Model | Variant | Accuracy | Δ accuracy vs full | Balanced accuracy | AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    dedicated = classification[classification.method.eq("dedicated classifier")]
    for model in dedicated.model.unique():
        model_rows = dedicated[dedicated.model.eq(model)]
        full_accuracy = float(model_rows.loc[model_rows.ablation.eq("full"), "accuracy"].iloc[0])
        for _, row in model_rows.iterrows():
            lines.append(
                f"| {row.model} | {row.ablation} | {_format(row.accuracy)} | "
                f"{_format(row.accuracy - full_accuracy)} | "
                f"{_format(row.balanced_accuracy)} | {_format(row.roc_auc)} |"
            )
    lines += [
        "",
        "## Raw versus log1p target training",
        "",
        "A monotonic post-hoc log transform cannot change Spearman. These experiments instead "
        "retrain each model on `log1p(TP120)` and inverse-transform predictions before raw-scale scoring.",
        "",
        "| Family | Training target | Spearman | MAE | Log-MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in log_performance.iterrows():
        lines.append(
            f"| {row['family']} | {row['transform']} | {_format(row.spearman)} | "
            f"{_format(row.mae, 2)} | {_format(row.log_mae, 4)} |"
        )
    lines += [
        "",
        "| Family | Metric | Log − raw difference | 95% cluster-bootstrap CI | p | BH q | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in log_tests.iterrows():
        lines.append(
            f"| {row.family} | {row.metric} | {_format(row.difference, 4)} | "
            f"[{_format(row.ci_low, 4)}, {_format(row.ci_high, 4)}] | "
            f"{_format(row.p_value, 4)} | {_format(row.p_value_fdr_bh, 4)} | "
            f"{_format(row.wilcoxon_p_value, 4)} |"
        )
    lines += [
        "",
        "Statistical intervals and p-values use paired primer-set cluster bootstrap resampling; "
        "the Wilcoxon test is a secondary paired test on primer-set mean absolute errors.",
        "",
    ]
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--tabular-oof", type=Path, required=True)
    parser.add_argument("--lyra-oof", type=Path, required=True)
    parser.add_argument("--source-tabular", type=Path, required=True)
    parser.add_argument("--source-lyra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = _load(args.baseline_npz)
    reference_ids = baseline["group_id"].astype(str)
    y = baseline["y"].astype(float)
    primer_sets = baseline["primer_set"].astype(str)
    base_predictions = {
        model: baseline[key].astype(float) for model, key in BASE_MODELS.items()
    }

    tabular_variants = {}
    for transform in ("raw", "log1p"):
        for suffix in ("drop_mismatch_count", "drop_mismatch_position", "drop_all_mismatch"):
            variant = f"{transform}_{suffix}"
            tabular_variants[variant] = _align(reference_ids, _collect_oof(args.tabular_oof, variant))
    lyra_variants = {}
    for transform in ("raw", "log1p"):
        suffixes = ("drop_mismatch_count", "drop_mismatch_position", "drop_all_mismatch")
        for suffix in suffixes:
            variant = f"{transform}_{suffix}"
            lyra_variants[variant] = _align(reference_ids, _collect_oof(args.lyra_oof, variant))
    for family, root, destination in (
        ("CatBoost", args.tabular_oof, tabular_variants),
        ("Lyra", args.lyra_oof, lyra_variants),
    ):
        destination["log1p_full"] = _align(reference_ids, _collect_oof(root, "log1p_full"))

    metadata_source = next(iter(tabular_variants.values()))
    total_mismatches = metadata_source["total_mismatches"]
    terminal_mismatches = metadata_source["terminal_3p3_mismatches"]
    if not np.allclose(metadata_source["y"], y):
        raise ValueError("Extended OOF targets do not align with the baseline")

    mismatch_rows = []
    grouping_values = {
        "total mismatch count": _mismatch_bin(total_mismatches),
        "terminal 3-prime mismatch count": _terminal_bin(terminal_mismatches),
        "chemistry/mismatch class": baseline["canonical_mm"].astype(str),
    }
    for grouping, values in grouping_values.items():
        for level in np.unique(values):
            keep = values == level
            for model, prediction in base_predictions.items():
                mismatch_rows.append(_regression_row(
                    model, y[keep], prediction[keep], grouping=grouping, level=level,
                ))
    mismatch_frame = pd.DataFrame(mismatch_rows)

    ablation_rows = []
    ablation_tests = []
    full_predictions = {
        ("CatBoost", "raw"): base_predictions["Weighted CatBoost"],
        ("Lyra", "raw"): base_predictions["Lyra physics-token"],
        ("CatBoost", "log1p"): tabular_variants["log1p_full"]["pred"],
        ("Lyra", "log1p"): lyra_variants["log1p_full"]["pred"],
    }
    for (family, transform), full in full_predictions.items():
        full_row = _regression_row(family, y, full)
        ablation_rows.append({
            "family": family, "transform": transform, "ablation": "full",
            **{key: full_row[key] for key in ("spearman", "mae", "rmse", "r2", "log_mae", "log_rmse", "n")},
            "delta_spearman": 0.0, "delta_mae": 0.0,
        })
        registry = tabular_variants if family == "CatBoost" else lyra_variants
        for ablation, suffix in ABLATIONS.items():
            prediction = registry[f"{transform}_{suffix}"]["pred"]
            row = _regression_row(family, y, prediction)
            ablation_rows.append({
                "family": family, "transform": transform, "ablation": ablation,
                **{key: row[key] for key in ("spearman", "mae", "rmse", "r2", "log_mae", "log_rmse", "n")},
                "delta_spearman": row["spearman"] - full_row["spearman"],
                "delta_mae": row["mae"] - full_row["mae"],
            })
            for metric in ("spearman", "mae"):
                test = _cluster_bootstrap_difference(
                    y, prediction, full, primer_sets, metric,
                    args.bootstrap, args.seed + len(ablation_tests),
                )
                ablation_tests.append({
                    "family": family, "transform": transform,
                    "ablation": ablation, "metric": metric,
                    "comparison": "ablated - full", **test,
                })
    ablation_frame = pd.DataFrame(ablation_rows)
    ablation_test_frame = pd.DataFrame(ablation_tests)
    ablation_test_frame["p_value_fdr_bh"] = _fdr_bh(ablation_test_frame.p_value)

    classification_rows = []
    classifier_predictions = {}
    y30 = (y <= 30.0).astype(int)
    for model, prediction in base_predictions.items():
        classification_rows.append(_classification_row(
            model, y30, -prediction, binary=prediction <= 30.0,
            method="regression threshold", ablation="full",
        ))
        classifier_predictions[(model, "regression threshold", "full")] = prediction <= 30.0
    for family, root in (("CatBoost", args.tabular_oof), ("Lyra", args.lyra_oof)):
        for ablation, variant in {
            "full": "tp30_full",
            "mismatch_count": "tp30_drop_mismatch_count",
            "mismatch_position": "tp30_drop_mismatch_position",
            "all_mismatch": "tp30_drop_all_mismatch",
        }.items():
            archive = _align(
                reference_ids,
                _collect_classifier_oof(
                    root, variant, legacy_lyra_logits=family == "Lyra"
                ),
            )
            probability = archive["pred"].astype(float)
            classification_rows.append(_classification_row(
                f"{family} TP30 classifier", y30, probability,
                method="dedicated classifier", ablation=ablation,
            ))
            classifier_predictions[(family, "dedicated classifier", ablation)] = probability >= 0.5
    classification_frame = pd.DataFrame(classification_rows)

    classification_tests = []
    comparisons = [
        (("CatBoost", "dedicated classifier", "full"), ("Sameed RF", "regression threshold", "full")),
        (("CatBoost", "dedicated classifier", "full"), ("Weighted CatBoost", "regression threshold", "full")),
        (("Lyra", "dedicated classifier", "full"), ("Lyra physics-token", "regression threshold", "full")),
        (("Lyra", "dedicated classifier", "full"), ("Fixed fusion", "regression threshold", "full")),
    ]
    for candidate_key, reference_key in comparisons:
        candidate = classifier_predictions[candidate_key]
        reference = classifier_predictions[reference_key]
        classification_tests.append({
            "candidate": " / ".join(candidate_key),
            "reference": " / ".join(reference_key),
            "test": "primer-set cluster bootstrap accuracy difference",
            **_accuracy_bootstrap(
                y30, candidate, reference, primer_sets,
                args.bootstrap, args.seed + len(classification_tests),
            ),
            **{f"mcnemar_{key}": value for key, value in _mcnemar(y30, candidate, reference).items()},
        })
    for family in ("CatBoost", "Lyra"):
        full = classifier_predictions[(family, "dedicated classifier", "full")]
        for ablation in ABLATIONS:
            candidate = classifier_predictions[(family, "dedicated classifier", ablation)]
            classification_tests.append({
                "candidate": f"{family} / dedicated classifier / {ablation}",
                "reference": f"{family} / dedicated classifier / full",
                "test": "primer-set cluster bootstrap accuracy difference",
                **_accuracy_bootstrap(
                    y30, candidate, full, primer_sets,
                    args.bootstrap, args.seed + len(classification_tests),
                ),
                **{f"mcnemar_{key}": value for key, value in _mcnemar(y30, candidate, full).items()},
            })
    classification_test_frame = pd.DataFrame(classification_tests)
    classification_test_frame["p_value_fdr_bh"] = _fdr_bh(
        classification_test_frame.p_value
    )
    classification_test_frame["mcnemar_p_value_fdr_bh"] = _fdr_bh(
        classification_test_frame.mcnemar_p_value
    )

    log_rows = []
    log_tests = []
    for family in ("CatBoost", "Lyra"):
        raw = full_predictions[(family, "raw")]
        logged = full_predictions[(family, "log1p")]
        for transform, prediction in (("raw", raw), ("log1p", logged)):
            row = _regression_row(family, y, prediction, family=family, transform=transform)
            log_rows.append(row)
        for metric, log_scale in (("spearman", False), ("mae", False), ("log_mae", True)):
            test = _cluster_bootstrap_difference(
                y, logged, raw, primer_sets, metric,
                args.bootstrap, args.seed + 100 + len(log_tests),
            )
            wilcoxon_result = (
                _primer_wilcoxon(y, logged, raw, primer_sets, log_scale)
                if metric in {"mae", "log_mae"} else {}
            )
            log_tests.append({
                "family": family, "metric": metric,
                "comparison": "log1p-trained - raw-trained",
                **test,
                "wilcoxon_statistic": wilcoxon_result.get("statistic", np.nan),
                "wilcoxon_p_value": wilcoxon_result.get("p_value", np.nan),
                "wilcoxon_n_primer_sets": wilcoxon_result.get("n_primer_sets", np.nan),
            })
    log_frame = pd.DataFrame(log_rows)
    log_test_frame = pd.DataFrame(log_tests)
    log_test_frame["p_value_fdr_bh"] = _fdr_bh(log_test_frame.p_value)
    log_test_frame["wilcoxon_p_value_fdr_bh"] = _fdr_bh(
        log_test_frame.wilcoxon_p_value
    )

    dataset_frame = _dataset_performance(args.source_tabular, args.source_lyra)

    outputs = {
        "dataset_performance.csv": dataset_frame,
        "mismatch_stratified_performance.csv": mismatch_frame,
        "mismatch_ablation_performance.csv": ablation_frame,
        "mismatch_ablation_statistical_tests.csv": ablation_test_frame,
        "tp30_classification_performance.csv": classification_frame,
        "tp30_classification_statistical_tests.csv": classification_test_frame,
        "log_transform_performance.csv": log_frame,
        "log_transform_statistical_tests.csv": log_test_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output / filename, index=False)
    summary = {
        "endpoint": "TP120 in minutes; TP30 positive class is TP120 <= 30 minutes",
        "development_units": int(len(y)),
        "development_primer_sets": int(len(np.unique(primer_sets))),
        "bootstrap_replicates": args.bootstrap,
        "classifier_prediction_handling": (
            "Per-fold backward-compatible detection converts legacy Lyra logits with sigmoid; "
            "new probability-valued folds are not transformed."
        ),
        "fixed_fusion_weights": {
            "Lyra": 0.30, "TabPFN v2": 0.20,
            "weighted CatBoost": 0.30, "AFT": 0.20,
        },
        "files": list(outputs),
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(_safe(summary), indent=2))
    _report(
        dataset_frame, mismatch_frame, ablation_frame,
        classification_frame, log_frame, log_test_frame,
        args.output / "EXTENDED_ANALYSIS_REPORT.md",
    )
    print(json.dumps(_safe(summary), indent=2))


if __name__ == "__main__":
    main()
