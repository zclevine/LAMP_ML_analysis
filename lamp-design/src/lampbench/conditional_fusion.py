from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from .analysis import load_oof
from .data import PRIMER_ORDER, load_units, split_confirmatory
from .metrics import regression_metrics
from .runner import TRAIN_FILES


SCREEN_BLOCKS = {"block0", "block4", "block9"}
HOLDOUT_BLOCKS = {f"block{i}" for i in (1, 2, 3, 5, 6, 7, 8)}


def _align(root: Path, sequence_model: str, tabular_model: str):
    sf, sy, sp = load_oof(root, "positive", sequence_model)
    tf, ty, tp = load_oof(root, "positive", tabular_model)
    ids = sorted(set(sf.group_id.astype(str)) & set(tf.group_id.astype(str)))
    if len(ids) != len(sf) or len(ids) != len(tf):
        raise ValueError("Conditional-fusion bases lack identical OOF coverage")
    si = pd.Series(np.arange(len(sf)), index=sf.group_id.astype(str)).loc[ids].to_numpy()
    ti = pd.Series(np.arange(len(tf)), index=tf.group_id.astype(str)).loc[ids].to_numpy()
    if not np.allclose(np.asarray(sy)[si], np.asarray(ty)[ti], equal_nan=True):
        raise ValueError("Conditional-fusion targets disagree")
    return sf.iloc[si].reset_index(drop=True), np.asarray(sy)[si], np.asarray(sp)[si], np.asarray(tp)[ti]


def _raw_gate_features(frame: pd.DataFrame, units: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    indexed = units.set_index("group_id", drop=False)
    joined = indexed.loc[frame.group_id.astype(str)]
    three_prime = joined[[f"{p}_mm_3p3" for p in PRIMER_ORDER]].sum(axis=1).to_numpy(float)
    chemistry = np.stack([
        (joined.canonical_mm.astype(str).to_numpy() == name).astype(float)
        for name in ("SS", "BST", "WS")
    ], axis=1)
    continuous = np.stack([
        joined.total_mismatches.to_numpy(float),
        three_prime,
        joined.mean_dg_ViennaRNA.to_numpy(float),
    ], axis=1)
    return np.concatenate([continuous, chemistry], axis=1), [
        "total_mismatches", "total_3p3_mismatches", "mean_dg_ViennaRNA",
        "chemistry_SS", "chemistry_BST", "chemistry_WS",
    ]


def _design(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    continuous = (raw[:, :3] - mean) / std
    return np.concatenate([np.ones((len(raw), 1)), continuous, raw[:, 3:]], axis=1)


def _huber(residual: np.ndarray, delta: float = 5.0) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual ** 2,
                    delta * (absolute - 0.5 * delta))


def _fit_gate(x: np.ndarray, y: np.ndarray, sequence_pred: np.ndarray,
              tabular_pred: np.ndarray, l2: float) -> dict:
    def objective(beta):
        weight = expit(x @ beta)
        pred = weight * sequence_pred + (1.0 - weight) * tabular_pred
        return float(np.mean(_huber(y - pred)) + l2 * np.mean(beta[1:] ** 2))

    initial = np.zeros(x.shape[1], dtype=float)
    initial[0] = logit(0.2)
    result = minimize(objective, initial, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Conditional fusion optimization failed: {result.message}")
    return {"beta": result.x, "objective": float(result.fun),
            "iterations": int(result.nit), "optimizer": "L-BFGS-B"}


def _block_macro(frame: pd.DataFrame, y, pred, blocks: set[str]) -> tuple[float, dict]:
    values = {}
    for block in sorted(blocks):
        idx = np.flatnonzero(frame.block.astype(str).to_numpy() == block)
        values[block] = regression_metrics(np.asarray(y)[idx], np.asarray(pred)[idx])["spearman"]
    return float(np.nanmean(list(values.values()))), values


def analyze(data_dir: Path, screen_root: Path, full_root: Path, selection_file: Path,
            output: Path, sequence_model: str | None = None,
            tabular_model: str = "catboost", l2: float = 0.1) -> dict:
    selection = json.loads(selection_file.read_text())
    locked = selection["selected_fusion"]
    sequence_model = sequence_model or locked["sequence_model"]
    if tabular_model != locked["tabular_model"]:
        raise ValueError("Tabular model must match the internally locked fusion")
    fixed_weight = float(locked["sequence_weight"])

    units, _ = split_confirmatory(load_units(data_dir, TRAIN_FILES))
    screen_frame, screen_y, screen_seq, screen_tab = _align(
        screen_root, sequence_model, tabular_model)
    full_frame, full_y, full_seq, full_tab = _align(full_root, sequence_model, tabular_model)
    if set(screen_frame.block.astype(str)) != SCREEN_BLOCKS:
        raise ValueError("Gate fitting data are not exactly the three declared screen blocks")
    if not HOLDOUT_BLOCKS.issubset(set(full_frame.block.astype(str))):
        raise ValueError("Seven-block internal holdout is incomplete")

    screen_raw, feature_names = _raw_gate_features(screen_frame, units)
    full_raw, _ = _raw_gate_features(full_frame, units)
    mean = screen_raw[:, :3].mean(axis=0)
    std = screen_raw[:, :3].std(axis=0)
    std[std == 0] = 1.0
    screen_x = _design(screen_raw, mean, std)
    full_x = _design(full_raw, mean, std)
    fitted = _fit_gate(screen_x, screen_y, screen_seq, screen_tab, l2)
    beta = fitted["beta"]
    screen_weight = expit(screen_x @ beta)
    full_weight = expit(full_x @ beta)
    screen_conditional = screen_weight * screen_seq + (1 - screen_weight) * screen_tab
    full_conditional = full_weight * full_seq + (1 - full_weight) * full_tab
    full_fixed = fixed_weight * full_seq + (1 - fixed_weight) * full_tab

    holdout = full_frame.block.astype(str).isin(HOLDOUT_BLOCKS).to_numpy()
    holdout_metrics = {
        "conditional": regression_metrics(full_y[holdout], full_conditional[holdout]),
        "fixed_locked": regression_metrics(full_y[holdout], full_fixed[holdout]),
        "catboost": regression_metrics(full_y[holdout], full_tab[holdout]),
        "sequence": regression_metrics(full_y[holdout], full_seq[holdout]),
    }
    block_macro = {}
    block_values = {}
    for name, pred in {
        "conditional": full_conditional,
        "fixed_locked": full_fixed,
        "catboost": full_tab,
        "sequence": full_seq,
    }.items():
        block_macro[name], block_values[name] = _block_macro(
            full_frame, full_y, pred, HOLDOUT_BLOCKS)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "internal_holdout_predictions.npz",
        group_id=full_frame.group_id.astype(str).to_numpy(dtype=str),
        block=full_frame.block.astype(str).to_numpy(dtype=str), y=full_y,
        sequence_pred=full_seq, tabular_pred=full_tab,
        fixed_pred=full_fixed, conditional_pred=full_conditional,
        sequence_weight=full_weight, holdout=holdout,
    )
    payload = {
        "status": "internal_holdout_evaluated",
        "frozen_data_status": "unopened",
        "selection_policy": (
            "Gate form and features fixed from internal OOF attribution; fit on blocks 0/4/9 "
            "and evaluated once on disjoint blocks 1/2/3/5/6/7/8."
        ),
        "sequence_model": sequence_model,
        "tabular_model": tabular_model,
        "feature_names": ["intercept"] + feature_names,
        "continuous_mean": mean.tolist(), "continuous_std": std.tolist(),
        "beta": beta.tolist(), "l2": l2,
        "optimization": {k: v for k, v in fitted.items() if k != "beta"},
        "screen_n": int(len(screen_y)),
        "screen_metrics": {
            "conditional": regression_metrics(screen_y, screen_conditional),
            "fixed_locked": regression_metrics(
                screen_y, fixed_weight * screen_seq + (1 - fixed_weight) * screen_tab),
        },
        "holdout_blocks": sorted(HOLDOUT_BLOCKS),
        "holdout_n": int(holdout.sum()),
        "holdout_metrics": holdout_metrics,
        "holdout_block_macro_spearman": block_macro,
        "holdout_block_spearman": block_values,
        "weight_summary": {
            "mean": float(full_weight[holdout].mean()),
            "sd": float(full_weight[holdout].std(ddof=1)),
            "min": float(full_weight[holdout].min()),
            "max": float(full_weight[holdout].max()),
        },
        "locked_fixed_sequence_weight": fixed_weight,
    }
    (output / "conditional_fusion_summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-model")
    parser.add_argument("--l2", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(analyze(
        args.data_dir, args.screen_root, args.full_root, args.selection_file,
        args.output, args.sequence_model, l2=args.l2), indent=2))


if __name__ == "__main__":
    main()
