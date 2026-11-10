from __future__ import annotations

import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (LampDataset, LengthBucketBatchSampler, Standardizer,
                   TABULAR_FEATURES, aggregate_replicates, collate, load_rows,
                   sha256_file, target_for_task)
from .metrics import cluster_bootstrap, stratified_metrics
from .models import build_model
from .train import _device, predict


INTERNAL_COMPLETION_FILES = (
    "tasks/summary.json",
    "tasks/summary.csv",
    "ablations/summary.json",
    "ablations/summary.csv",
    "property_attribution/property_attribution.json",
    "analysis/full/positive_paper_summary.json",
    "analysis/tasks/tp60/tp60_paper_summary.json",
    "analysis/tasks/tp120/tp120_paper_summary.json",
    "analysis/tasks/classification/classification_paper_summary.json",
    "analysis/tasks/hurdle/hurdle_paper_summary.json",
    "improvement_gate/selection.json",
    "tuned_selection/selection.json",
    "protocol_deviations.json",
    "shard_receipts/worker2.json",
    "shard_receipts/worker3.json",
)


def _internal_completion_hashes(results_root: Path) -> dict[str, str]:
    """Lock every internal artifact that must predate frozen-data opening."""
    paths = [results_root / relative for relative in INTERNAL_COMPLETION_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Frozen evaluation requires completed internal artifacts: {missing}")
    return {str(path.relative_to(results_root)): sha256_file(path) for path in paths}


def _selected_fusion_prediction(selection: dict, predictions: dict[str, np.ndarray]):
    fusion = selection.get("selected_fusion")
    if not fusion:
        return None
    sequence_model = fusion["sequence_model"]
    tabular_model = fusion["tabular_model"]
    if sequence_model not in predictions or tabular_model not in predictions:
        raise ValueError("Selected fusion base models are absent from the prediction inventory")
    sequence_weight = float(fusion["sequence_weight"])
    tabular_weight = float(fusion["tabular_weight"])
    if not np.isclose(sequence_weight + tabular_weight, 1):
        raise ValueError("Selected fusion weights do not sum to one")
    pred = (sequence_weight * predictions[sequence_model] +
            tabular_weight * predictions[tabular_model])
    return fusion, pred


def _final_checkpoint_inventory(checkpoint_root: Path, task: str,
                                models: list[str]) -> dict[str, list[Path]]:
    finalization_path = checkpoint_root / "finalization.json"
    if not finalization_path.exists():
        raise FileNotFoundError(f"Missing finalization record: {finalization_path}")
    finalization = json.loads(finalization_path.read_text())
    locked = {(x["model"], int(x["seed"])): x for x in finalization.get("records", [])}
    inventory = {}
    for model in sorted(models):
        base = checkpoint_root / task / model
        paths = sorted(base.glob("seed_*/final/model.pt")) or sorted(
            base.glob("seed_*/final/model.pkl"))
        if len(paths) != 3:
            raise ValueError(f"Expected 3 finalized checkpoints for {model}; found {len(paths)}")
        for path in paths:
            seed = int(path.parents[1].name.replace("seed_", ""))
            record = locked.get((model, seed))
            per_checkpoint = path.parent / "record.json"
            if (record is None or record.get("task") != task or
                    record.get("checkpoint") != str(path.relative_to(checkpoint_root)) or
                    record.get("checkpoint_sha256") != sha256_file(path) or
                    not per_checkpoint.exists() or
                    json.loads(per_checkpoint.read_text()) != record):
                raise RuntimeError(f"Finalized checkpoint does not match its lock record: {path}")
        inventory[model] = paths
    return inventory


def _evaluation_receipt(output: Path, kind: str, task: str, models: list[str],
                        selection_file: Path, checkpoint_root: Path,
                        inventory: dict[str, list[Path]], data_hashes: dict[str, str],
                        internal_hashes: dict[str, str]) -> dict:
    finalization = checkpoint_root / "finalization.json"
    if not finalization.exists():
        raise FileNotFoundError(f"Missing finalization record: {finalization}")
    selection = json.loads(selection_file.read_text())
    selected_fusion = selection.get("selected_fusion")
    payload = {
        "kind": kind, "task": task, "models": sorted(models),
        "derived_models": ([selected_fusion] if selected_fusion else []),
        "selection_sha256": sha256_file(selection_file),
        "finalization_sha256": sha256_file(finalization),
        "data_sha256": data_hashes,
        "internal_completion_sha256": internal_hashes,
        "checkpoint_sha256": {
            model: {str(path.relative_to(checkpoint_root)): sha256_file(path)
                    for path in paths}
            for model, paths in inventory.items()
        },
        "policy": "immutable receipt; no fitting or model selection after holdout opening",
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "EVALUATION_OPENED.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous != payload:
            raise RuntimeError(
                "Frozen evaluation was already opened with different data, internal "
                "artifacts, selection, models, or checkpoints"
            )
    else:
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _completed_results(output: Path, aggregate_name: str, receipt: dict):
    aggregate = output / aggregate_name
    completion = output / "EVALUATION_COMPLETE.json"
    if not aggregate.exists() or not completion.exists():
        return None
    record = json.loads(completion.read_text())
    expected = {
        "receipt_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()).hexdigest(),
        "results_file": aggregate_name,
        "results_sha256": sha256_file(aggregate),
    }
    if record != expected:
        raise RuntimeError("Frozen evaluation completion checksum mismatch")
    return json.loads(aggregate.read_text())


def _mark_complete(output: Path, aggregate_name: str, receipt: dict) -> None:
    aggregate = output / aggregate_name
    record = {
        "receipt_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True).encode()).hexdigest(),
        "results_file": aggregate_name,
        "results_sha256": sha256_file(aggregate),
    }
    (output / "EVALUATION_COMPLETE.json").write_text(
        json.dumps(record, indent=2, sort_keys=True))


def _neural_prediction(checkpoint: Path, frame, y, batch_size=256):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    std = Standardizer(np.asarray(payload["standardizer"]["mean"], np.float32),
                       np.asarray(payload["standardizer"]["std"], np.float32))
    model = build_model(cfg["model"], cfg["task"], cfg["d_model"], cfg["dropout"])
    model.load_state_dict(payload["state_dict"])
    device = _device()
    model.to(device)
    model_name = str(cfg.get("model", ""))
    dataset = LampDataset(
        frame, y, std, cfg.get("ablation", "none"),
        model_name.startswith(("lyra_physics_", "transformer_physics_", "mamba2_physics_")),
    )
    loader = DataLoader(dataset, batch_sampler=LengthBucketBatchSampler(
        dataset, batch_size, False), collate_fn=collate, num_workers=2)
    _, pred = predict(model, loader, cfg["task"], device, payload["target_mean"],
                      payload["target_std"], cfg["model"])
    return pred


def _tabular_prediction(checkpoint: Path, frame, task):
    with checkpoint.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "estimator" in payload:
        model, feature_cols = payload["estimator"], payload["feature_cols"]
    else:
        model, feature_cols = payload, TABULAR_FEATURES
    x = frame[feature_cols].to_numpy(np.float32)
    if task == "hurdle":
        return np.stack([np.clip(model[0].predict(x), 0, 1), model[1].predict(x)], axis=1)
    pred = model.predict(x)
    return np.clip(pred, 0, 1) if task == "classification" else pred


def evaluate_external(data_dir: Path, checkpoint_root: Path, output: Path,
                      models: list[str], task: str = "positive", selection_file: Path | None = None):
    """Evaluate only pre-selected internal-CV finalists; no fitting occurs here."""
    if selection_file is None or not selection_file.exists():
        raise ValueError("A pre-existing internal-CV selection file is required before external evaluation")
    selection = json.loads(selection_file.read_text())
    unauthorized = set(models) - set(selection["selected_models"])
    if unauthorized:
        raise ValueError(f"Models were not pre-selected by internal CV: {sorted(unauthorized)}")
    datasets = ["mpox.csv", "mpox_validation.csv", "Borrelia.csv", "LASV.csv"]
    inventory = _final_checkpoint_inventory(checkpoint_root, task, models)
    receipt = _evaluation_receipt(
        output, "external", task, models, selection_file, checkpoint_root, inventory,
        {name: sha256_file(data_dir / name) for name in datasets},
        _internal_completion_hashes(output.parent))
    completed = _completed_results(output, "external_results.json", receipt)
    if completed is not None:
        return completed
    all_results = []
    for filename in datasets:
        frame = aggregate_replicates(load_rows(data_dir, [filename]))
        frame, y = target_for_task(frame, task)
        frame = frame.reset_index(drop=True)
        dataset_predictions = {}
        for model_name in models:
            checkpoints = inventory[model_name]
            dest = output / task / model_name / Path(filename).stem
            existing = dest / "result.json"
            if existing.exists():
                all_results.append(json.loads(existing.read_text()))
                dataset_predictions[model_name] = np.load(
                    dest / "predictions.npz", allow_pickle=False)["pred"]
                continue
            preds = [
                _neural_prediction(p, frame, y) if p.suffix == ".pt"
                else _tabular_prediction(p, frame, task)
                for p in checkpoints
            ]
            pred = np.mean(np.stack(preds), axis=0)
            dataset_predictions[model_name] = pred
            result = {
                "dataset": Path(filename).stem, "model": model_name, "task": task,
                "n_checkpoints": len(checkpoints), "n_units": len(frame),
                "metrics": stratified_metrics(task, frame, y, pred),
                "bootstrap": cluster_bootstrap(task, frame, y, pred, n_boot=1000),
                "selection_policy": "models supplied after internal CV; no external fitting",
                "selection_file": str(selection_file),
            }
            all_results.append(result)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "result.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(dest / "predictions.npz", y=y, pred=pred,
                                group_id=frame.group_id.astype(str).to_numpy(dtype=str))
        selected_fusion = _selected_fusion_prediction(selection, dataset_predictions)
        if selected_fusion:
            fusion, pred = selected_fusion
            model_name = fusion["name"]
            dest = output / task / model_name / Path(filename).stem
            existing = dest / "result.json"
            if existing.exists():
                all_results.append(json.loads(existing.read_text()))
            else:
                result = {
                    "dataset": Path(filename).stem, "model": model_name, "task": task,
                    "n_checkpoints": 6, "n_units": len(frame),
                    "metrics": stratified_metrics(task, frame, y, pred),
                    "bootstrap": cluster_bootstrap(task, frame, y, pred, n_boot=1000),
                    "selection_policy": "fusion architecture and weights locked by internal screen",
                    "selection_file": str(selection_file), "fusion": fusion,
                }
                all_results.append(result)
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "result.json").write_text(json.dumps(result, indent=2))
                np.savez_compressed(dest / "predictions.npz", y=y, pred=pred,
                                    group_id=frame.group_id.astype(str).to_numpy(dtype=str))
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_results.json").write_text(json.dumps(all_results, indent=2))
    _mark_complete(output, "external_results.json", receipt)
    return all_results


def evaluate_confirmatory(units, checkpoint_root: Path, output: Path, models: list[str],
                          task: str, selection_file: Path):
    selection = json.loads(selection_file.read_text())
    unauthorized = set(models) - set(selection["selected_models"])
    if unauthorized:
        raise ValueError(f"Models were not pre-selected: {sorted(unauthorized)}")
    inventory = _final_checkpoint_inventory(checkpoint_root, task, models)
    identity_cols = ["group_id", "tp_positive", "amp_fraction", "n_replicates"]
    identity = units[identity_cols].sort_values("group_id").to_csv(index=False).encode()
    receipt = _evaluation_receipt(
        output, "confirmatory", task, models, selection_file, checkpoint_root, inventory,
        {"LMLv2_block19_aggregates": hashlib.sha256(identity).hexdigest()},
        _internal_completion_hashes(output.parent))
    completed = _completed_results(output, "confirmatory_results.json", receipt)
    if completed is not None:
        return completed
    frame, y = target_for_task(units, task)
    frame = frame.reset_index(drop=True)
    results = []
    model_predictions = {}
    for model_name in models:
        checkpoints = inventory[model_name]
        dest = output / task / model_name
        existing = dest / "result.json"
        if existing.exists():
            results.append(json.loads(existing.read_text()))
            model_predictions[model_name] = np.load(
                dest / "predictions.npz", allow_pickle=False)["pred"]
            continue
        pred = np.mean(np.stack([
            _neural_prediction(p, frame, y) if p.suffix == ".pt"
            else _tabular_prediction(p, frame, task) for p in checkpoints
        ]), axis=0)
        model_predictions[model_name] = pred
        result = {"dataset": "LMLv2_block19_confirmatory", "model": model_name,
                  "task": task, "n_checkpoints": len(checkpoints), "n_units": len(frame),
                  "metrics": stratified_metrics(task, frame, y, pred),
                  "bootstrap": cluster_bootstrap(task, frame, y, pred, n_boot=5000),
                  "selection_file": str(selection_file)}
        results.append(result)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(dest / "predictions.npz", y=y, pred=pred,
                            group_id=frame.group_id.astype(str).to_numpy(dtype=str))
    selected_fusion = _selected_fusion_prediction(selection, model_predictions)
    if selected_fusion:
        fusion, pred = selected_fusion
        model_name = fusion["name"]
        dest = output / task / model_name
        existing = dest / "result.json"
        if existing.exists():
            results.append(json.loads(existing.read_text()))
        else:
            result = {
                "dataset": "LMLv2_block19_confirmatory", "model": model_name,
                "task": task, "n_checkpoints": 6, "n_units": len(frame),
                "metrics": stratified_metrics(task, frame, y, pred),
                "bootstrap": cluster_bootstrap(task, frame, y, pred, n_boot=5000),
                "selection_policy": "fusion architecture and weights locked by internal screen",
                "selection_file": str(selection_file), "fusion": fusion,
            }
            results.append(result)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "result.json").write_text(json.dumps(result, indent=2))
            np.savez_compressed(dest / "predictions.npz", y=y, pred=pred,
                                group_id=frame.group_id.astype(str).to_numpy(dtype=str))
    output.mkdir(parents=True, exist_ok=True)
    (output / "confirmatory_results.json").write_text(json.dumps(results, indent=2))
    _mark_complete(output, "confirmatory_results.json", receipt)
    return results
