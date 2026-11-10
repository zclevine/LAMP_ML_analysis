from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data import (HISTORICAL_THERMO_FEATURES, META_FEATURES, POSITION_FEATURES,
                   TABULAR_FEATURES, TABULAR_FEATURE_GROUPS, THERMO_FEATURES, isolate_outer_fold,
                   restrict_train_chemistry,
                   tabular_frame_for_ablation, target_for_task)
from .metrics import cluster_bootstrap, stratified_metrics
from .sampling import deterministic_group_fraction


def build_tabular(name: str, seed: int, classification=False):
    base_name = name.removesuffix("_rank")
    if base_name == "tabpfn":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        # Pin the openly documented v2 checkpoint. Newer defaults require a
        # separate Prior Labs license-token flow, which is inappropriate for an
        # unattended private-data benchmark and would weaken reproducibility.
        return TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device="cuda",
            n_estimators=4,
            random_state=seed,
            ignore_pretraining_limits=True,
            memory_saving_mode="auto",
        )
    if base_name == "tabpfn_v3":
        from tabpfn import TabPFNRegressor
        # tabpfn>=8 defaults to the TabPFN-3 checkpoint. Pin the estimator
        # count so the comparison is reproducible across library point releases.
        return TabPFNRegressor(
            device="cuda",
            n_estimators=8,
            random_state=seed,
            ignore_pretraining_limits=True,
            memory_saving_mode="auto",
        )
    if base_name == "sameed_rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1,
                                     random_state=seed)
    if base_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0)
    if base_name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=5000, random_state=seed)
    if base_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=400, max_depth=14, min_samples_leaf=2,
                                     max_features=0.8, n_jobs=-1, random_state=seed)
    if base_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=1.0,
                                   n_jobs=-1, random_state=seed)
    if base_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=seed)
    if base_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=400, learning_rate=0.04,
                                         max_depth=4, min_samples_leaf=4,
                                         loss="huber", random_state=seed)
    if base_name in {"catboost", "catboost_historical", "catboost_source_balanced"} \
            or base_name.startswith("catboost_tp120_w") \
            or base_name.startswith("catboost_tp60_w"):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=800, depth=8, learning_rate=0.04,
                                 loss_function="RMSE", verbose=False, random_seed=seed,
                                 thread_count=-1)
    if base_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                             max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, n_jobs=-1, random_state=seed, verbosity=-1)
    if base_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=8,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            n_jobs=-1, random_state=seed, tree_method="hist")
    raise ValueError(name)


def feature_columns(name, ablation="none"):
    base_name = name.removesuffix("_rank")
    cols = list(THERMO_FEATURES if base_name == "sameed_rf" else TABULAR_FEATURES)
    if base_name == "catboost_historical":
        cols += HISTORICAL_THERMO_FEATURES
    if ablation == "no_position":
        cols = [x for x in cols if x not in POSITION_FEATURES]
    elif ablation == "no_context":
        cols = [x for x in cols if x not in META_FEATURES]
    elif ablation == "no_thermo":
        cols = [x for x in cols if x not in THERMO_FEATURES + POSITION_FEATURES]
    elif ablation == "historical_thermo":
        if name == "sameed_rf":
            raise ValueError("historical_thermo is a corrected-feature analysis, not Sameed RF")
        cols += [x for x in HISTORICAL_THERMO_FEATURES if x not in cols]
    elif ablation.startswith("drop_group_"):
        group = ablation.removeprefix("drop_group_")
        if group not in TABULAR_FEATURE_GROUPS:
            raise ValueError(f"Unknown tabular feature group: {group}")
        cols = [x for x in cols if x not in TABULAR_FEATURE_GROUPS[group]]
    elif ablation.startswith("drop_primer_"):
        # Primer-role ablations retain a fixed-width schema: role-local columns
        # are zeroed and aggregate summaries are recomputed in
        # tabular_frame_for_ablation(). Keeping the columns makes the same
        # transformation usable by estimators that require a stable schema.
        pass
    elif ablation != "none":
        raise ValueError(ablation)
    return cols


def _fit_predict(name, task, train, train_y, test, seed, ablation="none"):
    feature_cols = feature_columns(name, ablation)
    fit_train = tabular_frame_for_ablation(train, ablation)
    fit_test = tabular_frame_for_ablation(test, ablation)
    missing = [x for x in feature_cols if x not in fit_train.columns or x not in fit_test.columns]
    if missing:
        raise ValueError(f"Requested features are unavailable: {missing}")
    if (fit_train[feature_cols].isna().any().any()
            or fit_test[feature_cols].isna().any().any()):
        raise ValueError(
            "Requested features contain missing values; historical thermodynamics must be "
            "evaluated on the matched LMLv1 source only"
        )
    x_train = fit_train[feature_cols].to_numpy(np.float32)
    x_test = fit_test[feature_cols].to_numpy(np.float32)
    if task != "hurdle":
        model = build_tabular(name, seed)
        fit_y = train_y
        if name.endswith("_rank") and task in {
            "positive", "tp30", "tp60", "tp60_censored", "tp120"
        }:
            fit_y = rankdata(train_y, method="average") / (len(train_y) + 1.0)
        fit_kwargs = {}
        base_name = name.removesuffix("_rank")
        if task == "tp120" and base_name.startswith("catboost_tp120_w"):
            plateau_weight = int(base_name.removeprefix("catboost_tp120_w")) / 100.0
            fit_kwargs["sample_weight"] = np.where(train_y >= 119.999, plateau_weight, 1.0)
        elif task == "tp60_censored" and base_name.startswith("catboost_tp60_w"):
            plateau_weight = int(base_name.removeprefix("catboost_tp60_w")) / 100.0
            fit_kwargs["sample_weight"] = np.where(train_y >= 59.999, plateau_weight, 1.0)
        elif base_name == "catboost_source_balanced":
            counts = train.source.value_counts()
            fit_kwargs["sample_weight"] = train.source.map(
                {source: len(train) / (len(counts) * count)
                 for source, count in counts.items()}
            ).to_numpy(float)
        model.fit(x_train, fit_y, **fit_kwargs)
        pred = np.clip(model.predict(x_test), 0, 1) if task == "classification" else model.predict(x_test)
        return model, pred
    amp_model = build_tabular(name, seed)
    tp_model = build_tabular(name, seed + 1000)
    amp_model.fit(x_train, train_y[:, 0])
    positive = train_y[:, 2].astype(bool)
    tp_model.fit(x_train[positive], train_y[positive, 1])
    pred = np.stack([
        np.clip(amp_model.predict(x_test), 0, 1),
        tp_model.predict(x_test),
    ], axis=1)
    return (amp_model, tp_model), pred


def tabular_fold(units, outer_block: str, model_name: str, task: str, seed: int,
                 output_dir: Path, ablation: str = "none",
                 train_fraction: float = 1.0, sampling_seed: int = 0,
                 train_chemistry: str | None = None):
    train, outer, isolation = isolate_outer_fold(units, outer_block)
    train = restrict_train_chemistry(train, train_chemistry)
    train, train_y = target_for_task(train, task)
    outer, outer_y = target_for_task(outer, task)
    train, sampling = deterministic_group_fraction(train, train_fraction, sampling_seed)
    train, train_y = target_for_task(train, task)
    started = time.time()
    model, pred = _fit_predict(model_name, task, train, train_y, outer, seed, ablation)
    outer = outer.reset_index(drop=True)
    result = {
        "model": model_name, "task": task, "seed": seed, "outer_block": outer_block,
        "ablation": ablation, "feature_cols": feature_columns(model_name, ablation),
        "elapsed_sec": time.time() - started, "n_train": len(train), "n_outer": len(outer),
        "training_fraction": float(train_fraction),
        "sampling_seed": int(sampling_seed),
        "training_subsample": sampling,
        "train_chemistry": train_chemistry,
        "sequence_identity_isolation": {"outer": isolation},
        "metrics": stratified_metrics(task, outer, outer_y, pred),
        "bootstrap": cluster_bootstrap(task, outer, outer_y, pred, n_boot=200),
    }
    fold_dir = output_dir / task / model_name / f"seed_{seed}" / outer_block
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / "model.pkl"
    if model_name.removesuffix("_rank") in {"tabpfn", "tabpfn_v3"}:
        # A fitted TabPFN object embeds a roughly 200 MB pretrained checkpoint.
        # OOF predictions plus the pinned constructor/config are sufficient to
        # reproduce benchmark scoring; avoid duplicating that checkpoint once
        # per fold and variant.
        result["model_bytes"] = 0
        result["model_persistence"] = "omitted_pretrained_tabpfn_checkpoint"
    else:
        with model_path.open("wb") as f:
            pickle.dump({"estimator": model,
                         "feature_cols": feature_columns(model_name, ablation)}, f)
        result["model_bytes"] = model_path.stat().st_size
    np.savez_compressed(fold_dir / "predictions.npz", y=outer_y, pred=pred,
                        group_id=outer.group_id.astype(str).to_numpy(dtype=str),
                        primer_set=outer.primer_set.astype(str).to_numpy(dtype=str),
                        canonical_mm=outer.canonical_mm.astype(str).to_numpy(dtype=str),
                        target_conc=outer.target_conc.to_numpy(),
                        n_replicates=outer.n_replicates.to_numpy(),
                        amp_fraction=outer.amp_fraction.to_numpy())
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
