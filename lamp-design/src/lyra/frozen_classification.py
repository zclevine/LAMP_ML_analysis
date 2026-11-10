"""Five-fold frozen-backbone linear heads for sequence classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .classification import METRICS, classification_metrics
from .frozen_probe import _discover, _load_features
from .proteingym import _append_row, _resolve_device, _scheduled_learning_rate, _write_split_manifest, load_assay


LOGGER = logging.getLogger("lyra.frozen_classification")
EPOCH_FIELDS = ("task", "fold", "training_step", "n_train", "n_test", "train_cross_entropy", *METRICS, "elapsed_seconds")
FOLD_FIELDS = ("task", "fold", "n_train", "n_test", *METRICS, "elapsed_seconds")
TASK_FIELDS = ("task", "source_file", "n_samples", "sequence_length", "n_classes", "n_folds", *(f"{metric}_mean" for metric in METRICS), *(f"{metric}_std" for metric in METRICS), *(f"pooled_{metric}" for metric in METRICS), "feature_checkpoint", "linear_head_parameters")


@dataclass(frozen=True)
class FrozenClassificationConfig:
    input_path: Path
    feature_dir: Path
    output_dir: Path
    backbone: str
    sequence_column: str = "sequence"
    target_column: str = "target"
    fold_column: str | None = None
    pattern: str = "*.csv"
    n_folds: int = 5
    seed: int = 42
    training_steps: int = 10000
    eval_every_steps: int = 2000
    batch_size: int = 2048
    learning_rate: float = 1e-3
    min_learning_rate: float | None = 4e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    device: str = "auto"
    resume: bool = True


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _evaluate(model, features, labels, indices, device, batch_size, n_classes):
    model.eval()
    logits = torch.cat([model(features[batch].to(device)).cpu() for batch in indices.split(batch_size)])
    predictions = logits.argmax(dim=-1)
    return classification_metrics(labels[indices], predictions, n_classes), predictions, logits.softmax(dim=-1)


def run_frozen_classification(config: FrozenClassificationConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config.output_dir / "run_config.json"
    metadata = {**asdict(config), "input_path": str(config.input_path), "feature_dir": str(config.feature_dir), "output_dir": str(config.output_dir), "backbone_frozen": True, "head": "direct_linear_classifier", "weights_saved": True}
    if not run_config_path.exists(): run_config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    device = _resolve_device(config.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    completed = set()
    task_path = config.output_dir / "task_metrics.csv"
    if task_path.exists():
        with task_path.open(newline="", encoding="utf-8") as handle: completed = {row["task"] for row in csv.DictReader(handle)}
    task_rows = []
    for task_index, path in enumerate(_discover(config.input_path, config.pattern), 1):
        if path.stem in completed:
            LOGGER.info("[%d] %s already complete", task_index, path.stem)
            continue
        assay = load_assay(path, sequence_column=config.sequence_column, target_column=config.target_column, fold_column=config.fold_column, n_folds=config.n_folds, seed=config.seed)
        labels = torch.tensor([int(record.target) for record in assay.records], dtype=torch.long)
        classes = sorted(labels.unique().tolist())
        if classes != list(range(len(classes))) or len(classes) < 2: raise ValueError(f"{assay.name}: invalid classes {classes}")
        n_classes = len(classes)
        features, feature_metadata = _load_features(config.feature_dir / f"{assay.name}.pt", len(assay.records))
        _write_split_manifest(config.output_dir / "split_manifest.csv", assay, config.seed, config.fold_column)
        fold_rows, pooled_targets, pooled_predictions = [], [], []
        for fold in range(config.n_folds):
            started = time.perf_counter()
            fold_seed = config.seed + fold + zlib.crc32(assay.name.encode())
            _seed(fold_seed)
            train_indices = torch.tensor([i for i, value in enumerate(assay.folds) if value != fold])
            test_indices = torch.tensor([i for i, value in enumerate(assay.folds) if value == fold])
            model = nn.Linear(features.size(1), n_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            device_features, device_labels, device_train = features.to(device), labels.to(device), train_indices.to(device)
            generator = torch.Generator(device=device).manual_seed(fold_seed)
            interval_loss, interval_samples, final = 0.0, 0, None
            for step in range(1, config.training_steps + 1):
                lr = _scheduled_learning_rate(config, step)
                for group in optimizer.param_groups: group["lr"] = lr
                sampled = device_train[torch.randint(len(train_indices), (min(config.batch_size, len(train_indices)),), generator=generator, device=device)]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(device_features[sampled]), device_labels[sampled])
                loss.backward(); optimizer.step()
                interval_loss += float(loss.detach()) * len(sampled); interval_samples += len(sampled)
                if step % config.eval_every_steps and step != config.training_steps: continue
                final = _evaluate(model, features, labels, test_indices, device, config.batch_size, n_classes)
                metrics = final[0]
                _append_row(config.output_dir / "epoch_metrics.csv", EPOCH_FIELDS, {"task": assay.name, "fold": fold, "training_step": step, "n_train": len(train_indices), "n_test": len(test_indices), "train_cross_entropy": interval_loss / interval_samples, **metrics, "elapsed_seconds": time.perf_counter() - started})
                LOGGER.info("%s fold=%d step=%d accuracy=%.4f macro_f1=%.4f mcc=%.4f", assay.name, fold, step, metrics["accuracy"], metrics["macro_f1"], metrics["mcc"])
                interval_loss, interval_samples = 0.0, 0
            assert final is not None
            metrics, predictions, probabilities = final
            pooled_targets.append(labels[test_indices]); pooled_predictions.append(predictions)
            fold_row = {"task": assay.name, "fold": fold, "n_train": len(train_indices), "n_test": len(test_indices), **metrics, "elapsed_seconds": time.perf_counter() - started}
            fold_rows.append(fold_row); _append_row(config.output_dir / "fold_metrics.csv", FOLD_FIELDS, fold_row)
            prediction_dir = config.output_dir / "predictions" / assay.name; prediction_dir.mkdir(parents=True, exist_ok=True)
            with (prediction_dir / f"fold_{fold}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle); writer.writerow(("row_index", "target", "prediction", "probabilities"))
                for index, target, prediction, probability in zip(test_indices.tolist(), labels[test_indices].tolist(), predictions.tolist(), probabilities.tolist(), strict=True): writer.writerow((index, target, prediction, json.dumps(probability, separators=(",", ":"))))
            checkpoint_dir = config.output_dir / "models" / assay.name; checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"head_state_dict": model.state_dict(), "backbone": config.backbone, "n_classes": n_classes, "class_ids": classes, "fold": fold, "seed": fold_seed}, checkpoint_dir / f"fold_{fold}.pt")
        pooled = classification_metrics(torch.cat(pooled_targets), torch.cat(pooled_predictions), n_classes)
        task_row = {"task": assay.name, "source_file": str(assay.source), "n_samples": len(assay.records), "sequence_length": assay.sequence_length, "n_classes": n_classes, "n_folds": len(fold_rows), "feature_checkpoint": feature_metadata.get("checkpoint", config.backbone), "linear_head_parameters": features.size(1) * n_classes + n_classes}
        for metric in METRICS:
            values = torch.tensor([float(row[metric]) for row in fold_rows], dtype=torch.float64)
            task_row[f"{metric}_mean"] = float(values.mean()); task_row[f"{metric}_std"] = float(values.std(unbiased=False)); task_row[f"pooled_{metric}"] = pooled[metric]
        task_rows.append(task_row); _append_row(task_path, TASK_FIELDS, task_row)
    if task_rows:
        aggregate = {metric: float(torch.tensor([float(row[f"pooled_{metric}"]) for row in task_rows]).mean()) for metric in METRICS}
        (config.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return task_rows


def load_config(path: Path) -> FrozenClassificationConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(values) - {field.name for field in fields(FrozenClassificationConfig)})
    if unknown: raise ValueError(f"unknown configuration keys: {unknown}")
    base = path.resolve().parent
    for key in ("input_path", "feature_dir", "output_dir"):
        value = Path(values[key]); values[key] = value if value.is_absolute() else base / value
    return FrozenClassificationConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("config", type=Path); args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_frozen_classification(load_config(args.config))


if __name__ == "__main__": main()
