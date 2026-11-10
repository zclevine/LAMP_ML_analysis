from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, r2_score, roc_auc_score


REGRESSION_TASKS = {"positive", "tp30", "tp60", "tp60_censored", "tp120"}


def _finite_mean(values):
    """Mean of estimable components, or NaN when none can be estimated."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _corr(fn, a, b):
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    value = fn(a, b).statistic
    return float(value) if np.isfinite(value) else float("nan")


def regression_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if y.size == 0:
        return {
            "spearman": float("nan"), "pearson": float("nan"),
            "mae": float("nan"), "median_ae": float("nan"),
            "rmse": float("nan"), "r2": float("nan"),
            "within_5min": float("nan"), "within_10min": float("nan"), "n": 0,
        }
    absolute_error = np.abs(y - pred)
    return {
        "spearman": _corr(spearmanr, y, pred),
        "pearson": _corr(pearsonr, y, pred),
        "mae": float(np.mean(absolute_error)),
        "median_ae": float(np.median(absolute_error)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "within_5min": float(np.mean(absolute_error <= 5.0)),
        "within_10min": float(np.mean(absolute_error <= 10.0)),
        "n": int(len(y)),
    }


def classification_metrics(fraction, prob):
    fraction, prob = np.asarray(fraction), np.asarray(prob)
    y = (fraction >= 0.5).astype(int)
    auc = float(roc_auc_score(y, prob)) if np.unique(y).size > 1 else float("nan")
    ap = float(average_precision_score(y, prob)) if np.unique(y).size > 1 else float("nan")
    bins = np.linspace(0, 1, 11)
    bucket = np.clip(np.digitize(prob, bins) - 1, 0, 9)
    ece = sum(
        np.mean(bucket == i) * abs(float(np.mean(fraction[bucket == i])) - float(np.mean(prob[bucket == i])))
        for i in range(10) if np.any(bucket == i)
    )
    return {
        "auc": auc,
        "average_precision": ap,
        "accuracy": float(accuracy_score(y, prob >= 0.5)),
        "brier_soft": float(np.mean((fraction - prob) ** 2)),
        "ece_soft_10bin": float(ece),
        "n": int(len(y)),
    }


def task_metrics(task, y, pred):
    if task in REGRESSION_TASKS:
        return regression_metrics(y, pred)
    if task == "classification":
        return classification_metrics(y, pred)
    if task == "hurdle":
        amp, tp, tp_mask = y[:, 0], y[:, 1], y[:, 2].astype(bool)
        prob, cond_tp = pred[:, 0], pred[:, 1]
        expected_tp = prob * cond_tp + (1 - prob) * 120.0
        true_tp120 = amp * tp + (1 - amp) * 120.0
        out = {
            "amp": classification_metrics(amp, prob),
            "conditional_tp": regression_metrics(tp[tp_mask], cond_tp[tp_mask]),
            "expected_tp120": regression_metrics(true_tp120, expected_tp),
        }
        out["selection"] = _finite_mean([
            out["amp"]["auc"], out["conditional_tp"]["spearman"]
        ])
        return out
    raise ValueError(task)


def selection_value(task, metrics):
    if task in REGRESSION_TASKS:
        return metrics["spearman"]
    if task == "classification":
        return metrics["auc"]
    return metrics["selection"]


def stratified_metrics(task, frame: pd.DataFrame, y, pred):
    result = {"overall": task_metrics(task, y, pred), "by_mm": {}, "by_concentration": {},
              "by_replicate_count": {}, "by_amplification_consistency": {}}
    for value, part in frame.groupby("canonical_mm"):
        idx = part.index.to_numpy()
        result["by_mm"][str(value)] = task_metrics(task, np.asarray(y)[idx], np.asarray(pred)[idx])
    for value, part in frame.groupby("target_conc"):
        idx = part.index.to_numpy()
        result["by_concentration"][str(value)] = task_metrics(
            task, np.asarray(y)[idx], np.asarray(pred)[idx]
        )
    if "n_replicates" in frame and frame.n_replicates.notna().any():
        for value, part in frame.groupby("n_replicates"):
            idx = part.index.to_numpy()
            result["by_replicate_count"][str(value)] = task_metrics(
                task, np.asarray(y)[idx], np.asarray(pred)[idx])
    if "amp_fraction" in frame and frame.amp_fraction.notna().any():
        consistency = pd.cut(frame.amp_fraction, [-0.01, 0.0, 0.999999, 1.0],
                             labels=["none", "partial", "all"])
        for value in consistency.dropna().unique():
            idx = np.flatnonzero(consistency.astype(str).to_numpy() == str(value))
            result["by_amplification_consistency"][str(value)] = task_metrics(
                task, np.asarray(y)[idx], np.asarray(pred)[idx])
    return result


def cluster_bootstrap(task, frame, y, pred, n_boot=500, seed=20260813):
    rng = np.random.default_rng(seed)
    clusters = frame["primer_set"].astype(str).unique()
    cluster_rows = {c: np.flatnonzero(frame["primer_set"].astype(str).to_numpy() == c) for c in clusters}
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([cluster_rows[c] for c in chosen])
        m = task_metrics(task, np.asarray(y)[idx], np.asarray(pred)[idx])
        values.append(selection_value(task, m))
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return {
        "n_boot": n_boot,
        "cluster": "primer_set",
        "n_estimable": int(finite.size),
        "mean": _finite_mean(values),
        "ci95": ([float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
                 if finite.size else [float("nan"), float("nan")]),
    }
