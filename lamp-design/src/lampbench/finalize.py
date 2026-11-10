from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import sha256_file, target_for_task
from .models import build_model
from .tabular import TABULAR_FEATURES, build_tabular, feature_columns
from .train import TrainConfig, fit_epochs


def _existing_record(output: Path, dest: Path, expected: dict) -> dict | None:
    """Return a complete matching checkpoint record or reject partial/corrupt state."""
    record_path = dest / "record.json"
    checkpoint_paths = [dest / "model.pkl", dest / "model.pt"]
    present = [path for path in checkpoint_paths if path.exists()]
    if not record_path.exists() and not present:
        return None
    if not record_path.exists() or len(present) != 1:
        raise RuntimeError(f"Partial or ambiguous final checkpoint state: {dest}")
    record = json.loads(record_path.read_text())
    mismatched = {
        key: (record.get(key), value)
        for key, value in expected.items() if record.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"Final checkpoint record mismatch at {dest}: {mismatched}")
    checkpoint = output / record.get("checkpoint", "")
    if checkpoint != present[0] or not checkpoint.exists():
        raise RuntimeError(f"Final checkpoint path mismatch at {dest}")
    if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
        raise RuntimeError(f"Final checkpoint hash mismatch at {dest}")
    return record


def _write_record(dest: Path, record: dict) -> None:
    temporary = dest / "record.json.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, dest / "record.json")


def _selected(selection_file: Path, models: list[str]):
    selection = json.loads(selection_file.read_text())
    unauthorized = set(models) - set(selection["selected_models"])
    if unauthorized:
        raise ValueError(f"Models not selected by internal CV: {sorted(unauthorized)}")
    return selection


def _median_epoch(cv_root: Path, task: str, model: str) -> int:
    epochs = []
    for path in cv_root.glob(f"{task}/{model}/seed_*/*/result.json"):
        result = json.loads(path.read_text())
        if "best_epoch" in result:
            epochs.append(int(result["best_epoch"]))
    if len(epochs) < 10:
        raise ValueError(f"Need >=10 internal-CV epoch choices to finalize {model}; found {len(epochs)}")
    return max(1, int(np.median(epochs)))


def finalize_models(units, cv_root: Path, selection_file: Path, output: Path,
                    models: list[str], task="positive", seeds=(0, 1, 2)):
    selection = _selected(selection_file, models)
    frame, y = target_for_task(units, task)
    records = []
    for model_name in models:
        if model_name in {
            "sameed_rf", "ridge", "elastic_net", "random_forest", "extra_trees",
            "hist_gradient_boosting", "gradient_boosting", "catboost", "lightgbm", "xgboost",
        }:
            for seed in seeds:
                cols = feature_columns(model_name)
                dest = output / task / model_name / f"seed_{seed}" / "final"
                expected = {
                    "model": model_name, "seed": seed, "task": task,
                    "n_train": len(frame), "type": "tabular",
                    "feature_cols": cols,
                }
                existing = _existing_record(output, dest, expected)
                if existing is not None:
                    records.append(existing)
                    continue
                estimator = build_tabular(model_name, seed)
                if task == "hurdle":
                    amp = estimator
                    tp = build_tabular(model_name, seed + 1000)
                    amp.fit(frame[cols].to_numpy(np.float32), y[:, 0])
                    keep = y[:, 2].astype(bool)
                    tp.fit(frame.loc[keep, cols].to_numpy(np.float32), y[keep, 1])
                    estimator = (amp, tp)
                else:
                    estimator.fit(frame[cols].to_numpy(np.float32), y)
                dest.mkdir(parents=True, exist_ok=True)
                checkpoint = dest / "model.pkl"
                temporary = dest / "model.pkl.tmp"
                with temporary.open("wb") as f:
                    pickle.dump({"estimator": estimator, "feature_cols": cols}, f)
                os.replace(temporary, checkpoint)
                record = {
                    **expected,
                    "checkpoint": str(checkpoint.relative_to(output)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
                _write_record(dest, record)
                records.append(record)
            continue
        epoch = _median_epoch(cv_root, task, model_name)
        tuned = selection.get("model_configs", {}).get(model_name, {})
        for seed in seeds:
            cfg = TrainConfig(model=model_name, task=task, seed=seed, max_epochs=epoch,
                              d_model=int(tuned.get("d_model", 64)),
                              dropout=float(tuned.get("dropout", 0.2)),
                              lr=float(tuned.get("lr", 1e-3)),
                              loss=tuned.get("loss", "huber"))
            dest = output / task / model_name / f"seed_{seed}" / "final"
            expected = {
                "model": model_name, "seed": seed, "task": task,
                "n_train": len(frame), "type": "neural", "fixed_epoch": epoch,
                "config": asdict(cfg),
            }
            existing = _existing_record(output, dest, expected)
            if existing is not None:
                records.append(existing)
                continue
            model, std, mean, target_std, _, _ = fit_epochs(
                frame, y, None, None, cfg, fixed_epochs=epoch)
            dest.mkdir(parents=True, exist_ok=True)
            checkpoint = dest / "model.pt"
            temporary = dest / "model.pt.tmp"
            torch.save({"state_dict": model.state_dict(), "standardizer": std.to_json(),
                        "target_mean": mean, "target_std": target_std,
                        "config": asdict(cfg), "fixed_epoch": epoch}, temporary)
            os.replace(temporary, checkpoint)
            record = {
                **expected,
                "checkpoint": str(checkpoint.relative_to(output)),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            _write_record(dest, record)
            records.append(record)
    payload = {"selection_file": str(selection_file), "selection": selection,
               "cv_root": str(cv_root), "records": records}
    output.mkdir(parents=True, exist_ok=True)
    (output / "finalization.json").write_text(json.dumps(payload, indent=2))
    return payload
