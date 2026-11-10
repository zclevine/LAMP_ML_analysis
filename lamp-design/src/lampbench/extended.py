from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import DataLoader

from .data import (
    POSITION_FEATURES,
    PRIMER_ORDER,
    TABULAR_FEATURES,
    TABULAR_FEATURE_GROUPS,
    LampDataset,
    LengthBucketBatchSampler,
    collate,
    isolate_inner_fold,
    isolate_outer_fold,
    load_units,
    split_confirmatory,
    tabular_frame_for_ablation,
)
from .metrics import regression_metrics
from .tabular import build_tabular
from .train import TrainConfig, _device, fit_epochs, predict


TRAIN_FILES = ["LMLv1.csv", "LMLv2.csv", "Systematic.csv"]
BLOCKS = [f"block{i}" for i in range(10)]
ALL_DEVELOPMENT_BLOCKS = [f"block{i}" for i in range(19)]
SOURCE_HOLDOUTS = ["LMLv1", "LMLv2", "Systematic"]
LYRA_VARIANTS = {
    "raw_full": ("identity", "none"),
    "raw_drop_mismatch_count": ("identity", "token_drop_mismatch_count"),
    "raw_drop_mismatch_position": ("identity", "token_drop_mismatch_position"),
    "raw_drop_all_mismatch": ("identity", "token_drop_all_mismatch"),
    "log1p_full": ("log1p", "none"),
    "log1p_drop_mismatch_count": ("log1p", "token_drop_mismatch_count"),
    "log1p_drop_mismatch_position": ("log1p", "token_drop_mismatch_position"),
    "log1p_drop_all_mismatch": ("log1p", "token_drop_all_mismatch"),
}
LYRA_VARIANTS.update({
    f"drop_{primer}": ("identity", f"drop_primer_{primer}")
    for primer in PRIMER_ORDER
})
LYRA_CLASSIFIER_VARIANTS = {
    "tp30_full": "none",
    "tp30_drop_mismatch_count": "token_drop_mismatch_count",
    "tp30_drop_mismatch_position": "token_drop_mismatch_position",
    "tp30_drop_all_mismatch": "token_drop_all_mismatch",
}
REGRESSION_VARIANTS = {
    "raw_full": ("identity", "none"),
    "raw_drop_mismatch_count": ("identity", "drop_group_mismatch_count"),
    "raw_drop_mismatch_position": ("identity", "drop_group_mismatch_position"),
    "raw_drop_all_mismatch": ("identity", "drop_all_mismatch"),
    "log1p_full": ("log1p", "none"),
    "log1p_drop_mismatch_count": ("log1p", "drop_group_mismatch_count"),
    "log1p_drop_mismatch_position": ("log1p", "drop_group_mismatch_position"),
    "log1p_drop_all_mismatch": ("log1p", "drop_all_mismatch"),
}
CLASSIFIER_VARIANTS = {
    "tp30_full": "none",
    "tp30_drop_mismatch_count": "drop_group_mismatch_count",
    "tp30_drop_mismatch_position": "drop_group_mismatch_position",
    "tp30_drop_all_mismatch": "drop_all_mismatch",
}


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _append_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _mismatch_position_columns() -> list[str]:
    return list(POSITION_FEATURES)


def _feature_columns(ablation: str) -> list[str]:
    columns = list(TABULAR_FEATURES)
    if ablation == "none":
        return columns
    if ablation.startswith("drop_primer_"):
        # Role-local columns become constants after transformation, and the
        # aggregate summaries are recomputed from the seven retained roles.
        return columns
    if ablation == "drop_group_mismatch_count":
        removed = set(TABULAR_FEATURE_GROUPS["mismatch_count"])
    elif ablation == "drop_group_mismatch_position":
        removed = set(_mismatch_position_columns())
    elif ablation == "drop_all_mismatch":
        removed = set(TABULAR_FEATURE_GROUPS["mismatch_count"] + _mismatch_position_columns())
    else:
        raise ValueError(f"Unknown extended ablation: {ablation}")
    return [column for column in columns if column not in removed]


def _metadata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    terminal_columns = [f"{primer}_mm_3p3" for primer in PRIMER_ORDER]
    return {
        "group_id": frame.group_id.astype(str).to_numpy(dtype=str),
        "primer_set": frame.primer_set.astype(str).to_numpy(dtype=str),
        "source": frame.source.astype(str).to_numpy(dtype=str),
        "block": frame.block.astype(str).to_numpy(dtype=str),
        "canonical_mm": frame.canonical_mm.astype(str).to_numpy(dtype=str),
        "target_conc": frame.target_conc.to_numpy(),
        "n_replicates": frame.n_replicates.to_numpy(),
        "amp_fraction": frame.amp_fraction.to_numpy(),
        "total_mismatches": frame.total_mismatches.to_numpy(np.float32),
        "terminal_3p3_mismatches": frame[terminal_columns].sum(axis=1).to_numpy(np.float32),
    }


def _save_predictions(path: Path, frame: pd.DataFrame, y, pred, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        y=np.asarray(y),
        pred=np.asarray(pred),
        **_metadata(frame.reset_index(drop=True)),
        **extra,
    )


def _catboost_regressor(seed: int, device: str):
    from catboost import CatBoostRegressor

    kwargs = {}
    if device == "gpu":
        kwargs.update(task_type="GPU", devices="0")
    return CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.04,
        loss_function="RMSE",
        verbose=False,
        random_seed=seed,
        thread_count=-1,
        **kwargs,
    )


def _catboost_classifier(seed: int, device: str):
    from catboost import CatBoostClassifier

    kwargs = {}
    if device == "gpu":
        kwargs.update(task_type="GPU", devices="0")
    return CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=False,
        random_seed=seed,
        thread_count=-1,
        **kwargs,
    )


def _classification_metrics(y, prob) -> dict:
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = prob >= 0.5
    negative = y == 0
    specificity = float(np.mean(~pred[negative])) if negative.any() else float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }


def _fit_regression(train, test, transform: str, ablation: str, seed: int, device: str):
    columns = _feature_columns(ablation)
    fit_train = tabular_frame_for_ablation(train, ablation)
    fit_test = tabular_frame_for_ablation(test, ablation)
    x_train = fit_train[columns].to_numpy(np.float32)
    x_test = fit_test[columns].to_numpy(np.float32)
    raw_y = train.tp120.to_numpy(np.float32)
    fit_y = np.log1p(raw_y) if transform == "log1p" else raw_y
    weights = np.where(raw_y >= 119.999, 0.25, 1.0)
    model = _catboost_regressor(seed, device)
    model.fit(x_train, fit_y, sample_weight=weights)
    pred = model.predict(x_test)
    if transform == "log1p":
        pred = np.expm1(pred)
    return np.asarray(pred, np.float32), columns


def _fit_classifier(train, test, ablation: str, seed: int, device: str):
    columns = _feature_columns(ablation)
    x_train = train[columns].to_numpy(np.float32)
    x_test = test[columns].to_numpy(np.float32)
    y = (train.tp120.to_numpy(np.float32) <= 30.0).astype(np.int8)
    model = _catboost_classifier(seed, device)
    model.fit(x_train, y)
    return model.predict_proba(x_test)[:, 1].astype(np.float32), columns


def run_tabular_oof(units, output: Path, variants, blocks, seed: int, device: str):
    log = output / "events.jsonl"
    failures = []
    for variant in variants:
        is_classifier = variant in CLASSIFIER_VARIANTS
        if not is_classifier and variant not in REGRESSION_VARIANTS:
            raise ValueError(f"Unknown tabular variant: {variant}")
        for block in blocks:
            fold = output / variant / block
            if (fold / "result.json").exists():
                continue
            try:
                _append_event(log, {"status": "start", "variant": variant, "block": block})
                train, test, isolation = isolate_outer_fold(units, block)
                train, test = train.reset_index(drop=True), test.reset_index(drop=True)
                started = time.time()
                if is_classifier:
                    probability, columns = _fit_classifier(
                        train, test, CLASSIFIER_VARIANTS[variant], seed, device)
                    y = (test.tp120.to_numpy(np.float32) <= 30.0).astype(np.int8)
                    metrics = _classification_metrics(y, probability)
                    _save_predictions(
                        fold / "predictions.npz", test, y, probability,
                        tp120=test.tp120.to_numpy(np.float32),
                    )
                else:
                    transform, ablation = REGRESSION_VARIANTS[variant]
                    prediction, columns = _fit_regression(
                        train, test, transform, ablation, seed, device)
                    y = test.tp120.to_numpy(np.float32)
                    metrics = regression_metrics(y, prediction)
                    metrics["log_mae"] = float(
                        np.mean(np.abs(np.log1p(y) - np.log1p(np.clip(prediction, 0, None))))
                    )
                    _save_predictions(fold / "predictions.npz", test, y, prediction)
                result = {
                    "stage": "tabular_oof",
                    "variant": variant,
                    "outer_block": block,
                    "seed": seed,
                    "device": device,
                    "elapsed_sec": time.time() - started,
                    "n_train": len(train),
                    "n_test": len(test),
                    "feature_columns": columns,
                    "sequence_identity_isolation": isolation,
                    "metrics": metrics,
                }
                fold.mkdir(parents=True, exist_ok=True)
                (fold / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
                _append_event(log, {"status": "complete", "variant": variant, "block": block, "metrics": metrics})
            except Exception as exc:
                failures.append((variant, block, repr(exc)))
                _append_event(log, {"status": "error", "variant": variant, "block": block, "error": repr(exc), "traceback": traceback.format_exc()})
    if failures:
        raise RuntimeError(f"{len(failures)} tabular folds failed: {failures[:5]}")


def _lyra_config(ablation: str, seed: int, task: str = "tp120") -> TrainConfig:
    return TrainConfig(
        model="lyra_physics_tokens",
        task=task,
        d_model=64,
        dropout=0.2,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-3,
        max_epochs=80,
        patience=15,
        seed=seed,
        loss="rank_huber",
        ablation=ablation,
        num_workers=0,
    )


def _predict_lyra(model, frame, target_values, standardizer, cfg, mean, target_std):
    dataset = LampDataset(frame, target_values, standardizer, cfg.ablation, token_physics=True)
    loader = DataLoader(
        dataset,
        batch_sampler=LengthBucketBatchSampler(dataset, cfg.batch_size, False),
        collate_fn=collate,
        num_workers=0,
    )
    return predict(model, loader, cfg.task, _device(), mean, target_std, cfg.model)


def run_lyra_oof(units, output: Path, variants, blocks, seed: int):
    log = output / "events.jsonl"
    failures = []
    for variant in variants:
        is_classifier = variant in LYRA_CLASSIFIER_VARIANTS
        if not is_classifier and variant not in LYRA_VARIANTS:
            raise ValueError(f"Unknown Lyra variant: {variant}")
        transform, ablation = (
            ("classification", LYRA_CLASSIFIER_VARIANTS[variant])
            if is_classifier else LYRA_VARIANTS[variant]
        )
        cfg = _lyra_config(ablation, seed, "classification" if is_classifier else "tp120")
        for block in blocks:
            fold = output / variant / block
            if (fold / "result.json").exists():
                continue
            try:
                _append_event(log, {"status": "start", "variant": variant, "block": block})
                outer_index = int(block.replace("block", ""))
                if outer_index <= 9:
                    inner_block = f"block{(outer_index + 1) % 10}"
                else:
                    inner_block = f"block{10 + ((outer_index - 10 + 1) % 9)}"
                candidate, outer, outer_isolation = isolate_outer_fold(units, block)
                fit, inner, inner_isolation = isolate_inner_fold(candidate, inner_block)
                if is_classifier:
                    transform_y = lambda values: (values <= 30.0).astype(np.float32)
                else:
                    transform_y = np.log1p if transform == "log1p" else lambda values: values
                fit_y = transform_y(fit.tp120.to_numpy(np.float32))
                inner_y = transform_y(inner.tp120.to_numpy(np.float32))
                started = time.time()
                _, _, _, _, best_epoch, history = fit_epochs(fit, fit_y, inner, inner_y, cfg)
                full_y = transform_y(candidate.tp120.to_numpy(np.float32))
                final, standardizer, mean, target_std, _, _ = fit_epochs(
                    candidate, full_y, None, None, cfg, fixed_epochs=best_epoch
                )
                outer_target = transform_y(outer.tp120.to_numpy(np.float32))
                _, transformed_prediction = _predict_lyra(
                    final, outer, outer_target, standardizer, cfg, mean, target_std
                )
                if is_classifier:
                    prediction = np.asarray(transformed_prediction, np.float32)
                    y = transform_y(outer.tp120.to_numpy(np.float32)).astype(np.int8)
                    metrics = _classification_metrics(y, prediction)
                    _save_predictions(
                        fold / "predictions.npz", outer, y, prediction,
                        tp120=outer.tp120.to_numpy(np.float32),
                    )
                else:
                    prediction = (
                        np.expm1(transformed_prediction)
                        if transform == "log1p" else transformed_prediction
                    ).astype(np.float32)
                    y = outer.tp120.to_numpy(np.float32)
                    metrics = regression_metrics(y, prediction)
                    metrics["log_mae"] = float(
                        np.mean(np.abs(np.log1p(y) - np.log1p(np.clip(prediction, 0, None))))
                    )
                    _save_predictions(fold / "predictions.npz", outer, y, prediction)
                result = {
                    "stage": "lyra_oof",
                    "variant": variant,
                    "target_transform": transform,
                    "endpoint": "TP120 <= 30 minutes" if is_classifier else "TP120 regression",
                    "outer_block": block,
                    "inner_block": inner_block,
                    "seed": seed,
                    "config": asdict(cfg),
                    "best_epoch": best_epoch,
                    "elapsed_sec": time.time() - started,
                    "n_fit": len(fit),
                    "n_inner": len(inner),
                    "n_outer": len(outer),
                    "sequence_identity_isolation": {
                        "outer": outer_isolation,
                        "inner": inner_isolation,
                    },
                    "metrics": metrics,
                    "history": history,
                }
                fold.mkdir(parents=True, exist_ok=True)
                (fold / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
                _append_event(log, {"status": "complete", "variant": variant, "block": block, "metrics": metrics})
                del final
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                failures.append((variant, block, repr(exc)))
                _append_event(log, {"status": "error", "variant": variant, "block": block, "error": repr(exc), "traceback": traceback.format_exc()})
    if failures:
        raise RuntimeError(f"{len(failures)} Lyra folds failed: {failures[:5]}")


def _isolated_holdout(development: pd.DataFrame, source: str):
    test = development.loc[development.source.eq(source)].copy()
    nominal_train = development.loc[~development.source.eq(source)].copy()
    test_ids = set(test.sequence_group_id.astype(str))
    overlap = nominal_train.sequence_group_id.astype(str).isin(test_ids)
    train = nominal_train.loc[~overlap].copy()
    audit = {
        "test_source": source,
        "test_units": len(test),
        "test_sequence_groups": len(test_ids),
        "nominal_train_units": len(nominal_train),
        "excluded_identity_units": int(overlap.sum()),
        "train_units": len(train),
        "post_isolation_overlap": 0,
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), audit


def _holdout_jobs(development, confirmatory):
    jobs = {
        f"source_{source}": _isolated_holdout(development, source)
        for source in SOURCE_HOLDOUTS
    }
    jobs["confirmatory_LMLv2_block19"] = (
        development.reset_index(drop=True),
        confirmatory.reset_index(drop=True),
        {
            "test_source": "LMLv2",
            "test_block": "block19",
            "test_units": len(confirmatory),
            "train_units": len(development),
            "post_isolation_overlap": 0,
        },
    )
    # Expose every principal development block through the same multi-model
    # source-tabular path. This gives the YAML runner a uniform block0--18
    # interface for Sameed, TabPFN, weighted CatBoost, and AFT predictions.
    for block in ALL_DEVELOPMENT_BLOCKS:
        train, test, isolation = isolate_outer_fold(development, block)
        jobs[f"outer_{block}"] = (train, test, isolation)
    return jobs


def run_source_tabular(development, confirmatory, output: Path, jobs, seed: int,
                       device: str, ablation: str = "none"):
    from .censored import fit_predict as fit_aft

    log = output / "events.jsonl"
    failures = []
    available = _holdout_jobs(development, confirmatory)
    for job_name in jobs:
        if job_name not in available:
            raise ValueError(f"Unknown source job: {job_name}")
        destination = output / job_name
        if (destination / "result.json").exists():
            continue
        train, test, isolation = available[job_name]
        try:
            _append_event(log, {"status": "start", "job": job_name})
            started = time.time()
            fit_train = tabular_frame_for_ablation(train, ablation)
            fit_test = tabular_frame_for_ablation(test, ablation)
            sameed_model = build_tabular("sameed_rf", seed)
            # Sameed's feature schema is exactly the original thermodynamic subset.
            from .data import THERMO_FEATURES
            sameed_columns = list(THERMO_FEATURES)
            sameed_model.fit(fit_train[sameed_columns].to_numpy(np.float32), train.tp120.to_numpy(np.float32))
            sameed = sameed_model.predict(fit_test[sameed_columns].to_numpy(np.float32)).astype(np.float32)
            weighted, _ = _fit_regression(
                train, test, "identity", ablation, seed, device)
            tabpfn = build_tabular("tabpfn", seed)
            tabpfn.fit(fit_train[TABULAR_FEATURES].to_numpy(np.float32), train.tp120.to_numpy(np.float32))
            tabpfn_prediction = tabpfn.predict(fit_test[TABULAR_FEATURES].to_numpy(np.float32)).astype(np.float32)
            _, aft = fit_aft(train, test, "aft_normal", seed, ablation)
            y = test.tp120.to_numpy(np.float32)
            predictions = {
                "sameed": sameed,
                "tabpfn": tabpfn_prediction,
                "weighted_catboost": weighted,
                "aft": np.asarray(aft, np.float32),
            }
            metrics = {name: regression_metrics(y, pred) for name, pred in predictions.items()}
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination / "predictions.npz",
                y=y,
                **predictions,
                **_metadata(test),
            )
            result = {
                "stage": "source_tabular",
                "job": job_name,
                "seed": seed,
                "device": device,
                "ablation": ablation,
                "elapsed_sec": time.time() - started,
                "n_train": len(train),
                "n_test": len(test),
                "sequence_identity_isolation": isolation,
                "metrics": metrics,
            }
            (destination / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
            _append_event(log, {"status": "complete", "job": job_name, "metrics": metrics})
        except Exception as exc:
            failures.append((job_name, repr(exc)))
            _append_event(log, {"status": "error", "job": job_name, "error": repr(exc), "traceback": traceback.format_exc()})
    if failures:
        raise RuntimeError(f"{len(failures)} source-tabular jobs failed: {failures[:5]}")


def run_source_lyra(development, confirmatory, output: Path, jobs, seed: int, epochs: int):
    log = output / "events.jsonl"
    failures = []
    available = _holdout_jobs(development, confirmatory)
    cfg = _lyra_config("none", seed)
    for job_name in jobs:
        if job_name not in available:
            raise ValueError(f"Unknown source job: {job_name}")
        destination = output / job_name
        if (destination / "result.json").exists():
            continue
        train, test, isolation = available[job_name]
        try:
            _append_event(log, {"status": "start", "job": job_name})
            started = time.time()
            train_y = train.tp120.to_numpy(np.float32)
            model, standardizer, mean, target_std, _, _ = fit_epochs(
                train, train_y, None, None, cfg, fixed_epochs=epochs
            )
            test_y = test.tp120.to_numpy(np.float32)
            _, prediction = _predict_lyra(
                model, test, test_y, standardizer, cfg, mean, target_std
            )
            prediction = np.asarray(prediction, np.float32)
            metrics = regression_metrics(test_y, prediction)
            _save_predictions(destination / "predictions.npz", test, test_y, prediction)
            result = {
                "stage": "source_lyra",
                "job": job_name,
                "seed": seed,
                "fixed_epochs": epochs,
                "config": asdict(cfg),
                "elapsed_sec": time.time() - started,
                "n_train": len(train),
                "n_test": len(test),
                "sequence_identity_isolation": isolation,
                "metrics": metrics,
            }
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
            _append_event(log, {"status": "complete", "job": job_name, "metrics": metrics})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            failures.append((job_name, repr(exc)))
            _append_event(log, {"status": "error", "job": job_name, "error": repr(exc), "traceback": traceback.format_exc()})
    if failures:
        raise RuntimeError(f"{len(failures)} source-Lyra jobs failed: {failures[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tabular-oof", "lyra-oof", "source-tabular", "source-lyra"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--blocks", nargs="+", default=BLOCKS)
    parser.add_argument("--jobs", nargs="+", default=[
        "source_LMLv1", "source_LMLv2", "source_Systematic", "confirmatory_LMLv2_block19"
    ])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--catboost-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--fixed-epochs", type=int, default=5)
    parser.add_argument("--ablation", default="none")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    units = load_units(args.data_dir, TRAIN_FILES)
    development, confirmatory = split_confirmatory(units)
    if args.stage == "tabular-oof":
        variants = args.variants or list(REGRESSION_VARIANTS) + list(CLASSIFIER_VARIANTS)
        run_tabular_oof(development, args.output, variants, args.blocks, args.seed, args.catboost_device)
    elif args.stage == "lyra-oof":
        run_lyra_oof(
            development, args.output,
            args.variants or list(LYRA_VARIANTS) + list(LYRA_CLASSIFIER_VARIANTS),
            args.blocks, args.seed,
        )
    elif args.stage == "source-tabular":
        run_source_tabular(
            development, confirmatory, args.output, args.jobs, args.seed,
            args.catboost_device, args.ablation
        )
    else:
        run_source_lyra(
            development, confirmatory, args.output, args.jobs, args.seed, args.fixed_epochs
        )


if __name__ == "__main__":
    main()
