from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.random_projection import GaussianRandomProjection

from .data import (PRIMER_ORDER, TABULAR_FEATURES, isolate_outer_fold, load_units,
                   split_confirmatory, target_for_task)
from .metrics import cluster_bootstrap, selection_value, stratified_metrics


MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
TRAIN_FILES = ["LMLv1.csv", "LMLv2.csv", "Systematic.csv"]
FOUNDATION_MODELS = {
    "nt50_ridge", "nt50_ridge_seqonly",
    "nt50_catboost", "nt50_catboost_seqonly",
}


def assay_strings(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    separator = "NNNNNN"
    primers = frame[[f"{p}_sequence" for p in PRIMER_ORDER]].astype(str).apply(
        lambda row: separator.join(x.strip().upper() for x in row), axis=1)
    targets = frame[[f"{p}_target_sequence" for p in PRIMER_ORDER]].astype(str).apply(
        lambda row: separator.join(x.strip().upper() for x in row), axis=1)
    return primers.tolist(), targets.tolist()


def _mean_embeddings(sequences: list[str], batch_size: int) -> np.ndarray:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    # The checkpoint declares a custom EsmForMaskedLM implementation. Using
    # generic AutoModel selects the library's built-in ESM encoder instead and
    # can silently construct a different feed-forward width before failing to
    # load the published weights.
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        # The checkpoint's custom rotary-attention path mixes an fp32
        # attention matrix with value states, so bf16 causes a dtype error.
        # At 50M parameters fp32 comfortably fits the target GPU.
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = tokenizer(
                sequences[start:start + batch_size], padding=True, truncation=True,
                max_length=min(512, tokenizer.model_max_length), return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True)
            hidden = output.hidden_states[-1].float()
            mask = batch["attention_mask"].bool()
            for token_id in tokenizer.all_special_ids:
                mask &= batch["input_ids"] != token_id
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            rows.append(pooled.cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def build_embedding_cache(units: pd.DataFrame, cache_path: Path, batch_size: int) -> np.ndarray:
    ids = units.group_id.astype(str).to_numpy()
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["group_id"].astype(str), ids):
            return cached["features"].astype(np.float32)
    primer_strings, target_strings = assay_strings(units)
    all_strings = primer_strings + target_strings
    unique, inverse = np.unique(np.asarray(all_strings, dtype=str), return_inverse=True)
    encoded = _mean_embeddings(unique.tolist(), batch_size)
    primer = encoded[inverse[:len(units)]]
    target = encoded[inverse[len(units):]]
    raw = np.concatenate([primer, target, np.abs(primer - target), primer * target], axis=1)
    # A fixed, data-independent Johnson-Lindenstrauss projection avoids fitting
    # any transform on held-out sequences while keeping downstream OOF fits fast.
    projector = GaussianRandomProjection(n_components=256, random_state=20260818)
    projector.fit(np.zeros((1, raw.shape[1]), dtype=np.float32))
    features = projector.transform(raw).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, group_id=ids.astype(str), features=features,
        model_id=np.asarray([MODEL_ID]), model_revision=np.asarray([MODEL_REVISION]),
        projection_seed=np.asarray([20260818]),
    )
    return features


def _estimator(model_name: str, seed: int):
    if "ridge" in model_name:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if "catboost" in model_name:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=500, depth=8, learning_rate=0.04, loss_function="RMSE",
            random_seed=seed, verbose=False,
            task_type="GPU" if torch.cuda.is_available() else "CPU",
            devices="0" if torch.cuda.is_available() else None,
        )
    raise ValueError(model_name)


def _features(frame: pd.DataFrame, feature_by_id: dict[str, np.ndarray], seqonly: bool):
    sequence = np.stack([feature_by_id[str(gid)] for gid in frame.group_id]).astype(np.float32)
    if seqonly:
        return sequence
    tabular = frame[TABULAR_FEATURES].to_numpy(np.float32)
    return np.concatenate([sequence, tabular], axis=1)


def _fit_predict(model_name, task, train, train_y, test, seed, feature_by_id):
    seqonly = model_name.endswith("_seqonly")
    x_train = _features(train, feature_by_id, seqonly)
    x_test = _features(test, feature_by_id, seqonly)
    if task != "hurdle":
        model = _estimator(model_name, seed)
        model.fit(x_train, train_y)
        pred = model.predict(x_test)
        if task == "classification":
            pred = np.clip(pred, 0, 1)
        return model, pred
    amp_model = _estimator(model_name, seed)
    tp_model = _estimator(model_name, seed + 1000)
    amp_model.fit(x_train, train_y[:, 0])
    positive = train_y[:, 2].astype(bool)
    tp_model.fit(x_train[positive], train_y[positive, 1])
    pred = np.stack([
        np.clip(amp_model.predict(x_test), 0, 1),
        tp_model.predict(x_test),
    ], axis=1)
    return (amp_model, tp_model), pred


def run_fold(units, feature_by_id, outer_block, model_name, task, seed, output):
    train, outer, isolation = isolate_outer_fold(units, outer_block)
    train, train_y = target_for_task(train, task)
    outer, outer_y = target_for_task(outer, task)
    started = time.time()
    _, pred = _fit_predict(model_name, task, train, train_y, outer, seed, feature_by_id)
    result = {
        "model": model_name, "task": task, "seed": seed, "outer_block": outer_block,
        "pretrained_model": MODEL_ID,
        "pretrained_model_revision": MODEL_REVISION,
        "embedding_pooling": "mean non-special-token embedding",
        "assay_representation": (
            "ordered primer concat and aligned-target concat; frozen primer, target, "
            "absolute-difference, and product embeddings; fixed 256D Gaussian projection"
        ),
        "sequence_only": model_name.endswith("_seqonly"),
        "elapsed_sec": time.time() - started,
        "n_train": len(train), "n_outer": len(outer),
        "sequence_identity_isolation": {"outer": isolation},
        "metrics": stratified_metrics(task, outer, outer_y, pred),
        "bootstrap": cluster_bootstrap(task, outer, outer_y, pred, n_boot=200),
    }
    fold_dir = output / task / model_name / f"seed_{seed}" / outer_block
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fold_dir / "predictions.npz", y=outer_y, pred=pred,
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
    for path in output.glob("*/*/seed_*/*/result.json"):
        result = json.loads(path.read_text())
        rows.append({
            "task": result["task"], "model": result["model"],
            "block": result["outer_block"],
            "selection": selection_value(result["task"], result["metrics"]["overall"]),
            "elapsed_sec": result["elapsed_sec"],
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    summary = frame.groupby(["task", "model"], as_index=False).agg(
        mean=("selection", "mean"), std=("selection", "std"),
        n=("selection", "size"), elapsed_sec=("elapsed_sec", "sum"),
    ).sort_values(["task", "mean"], ascending=[True, False])
    frame.to_csv(output / "all_fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(summary.to_json(orient="records", indent=2))
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=sorted(FOUNDATION_MODELS))
    parser.add_argument("--tasks", nargs="+", default=["tp120", "positive", "classification", "hurdle"])
    parser.add_argument("--blocks", nargs="+", default=["block0", "block4", "block9"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()
    unknown = set(args.models) - FOUNDATION_MODELS
    if unknown:
        raise SystemExit(f"Unknown pretrained models: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    units, _ = split_confirmatory(load_units(args.data_dir, TRAIN_FILES))
    cache_path = args.output / "nt50_embeddings.npz"
    features = build_embedding_cache(units, cache_path, args.embedding_batch_size)
    feature_by_id = dict(zip(units.group_id.astype(str), features))
    failures = []
    for task in args.tasks:
        for model in args.models:
            for block in args.blocks:
                result_path = args.output / task / model / f"seed_{args.seed}" / block / "result.json"
                if result_path.exists():
                    continue
                try:
                    result = run_fold(
                        units, feature_by_id, block, model, task, args.seed, args.output)
                    print(json.dumps({
                        "status": "complete", "task": task, "model": model,
                        "block": block,
                        "selection": selection_value(task, result["metrics"]["overall"]),
                    }), flush=True)
                except Exception as exc:
                    failures.append((task, model, block, repr(exc)))
                    print(json.dumps({"status": "error", "task": task, "model": model,
                                      "block": block, "error": repr(exc)}), flush=True)
    summarize(args.output)
    if failures:
        raise RuntimeError(f"{len(failures)} pretrained-sequence folds failed: {failures[:5]}")


if __name__ == "__main__":
    main()
