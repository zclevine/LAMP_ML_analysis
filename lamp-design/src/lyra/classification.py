"""Five-fold sequence classification for FLIP and FLIP2 tasks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

from .proteingym import (
    PROTEIN_ALPHABET,
    ProteinAlphabet,
    _append_row,
    _build_model,
    _loader,
    _resolve_device,
    _scheduled_learning_rate,
    _write_split_manifest,
    discover_assays,
    load_assay,
)


LOGGER = logging.getLogger("lyra.classification")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "mcc")
EPOCH_FIELDS = (
    "task", "fold", "training_step", "n_train", "n_test", "train_cross_entropy",
    *METRICS, "elapsed_seconds",
)
FOLD_FIELDS = ("task", "fold", "n_train", "n_test", *METRICS, "elapsed_seconds")
TASK_FIELDS = (
    "task", "source_file", "n_samples", "sequence_length", "n_classes", "n_folds",
    *(f"{metric}_mean" for metric in METRICS),
    *(f"{metric}_std" for metric in METRICS),
    *(f"pooled_{metric}" for metric in METRICS),
)


@dataclass(frozen=True)
class ClassificationConfig:
    input_path: Path
    output_dir: Path
    sequence_column: str = "sequence"
    target_column: str = "target"
    fold_column: str | None = None
    pattern: str = "*.csv"
    alphabet: str = PROTEIN_ALPHABET
    n_folds: int = 5
    seed: int = 42
    model_type: str = "lyra"
    d_model: int = 128
    d_state: int = 128
    num_blocks: int = 1
    mamba_expand: int = 3
    hyena_max_length: int = 4096
    hyena_order: int | None = None
    dropout: float = 0.1
    causal_pgc: bool = False
    lyra_pgc_residual: bool = False
    featurewise_dropout: bool = False
    pooling: str = "mean"
    bidirectional: bool = False
    class_weighting: str = "none"
    class_weight_beta: float = 0.999
    class_weight_max: float = 10.0
    training_steps: int = 10000
    eval_every_steps: int = 2000
    batch_size: int = 32
    max_tokens_per_batch: int | None = 24000
    learning_rate: float = 1e-3
    min_learning_rate: float | None = 4e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    resume: bool = True


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    targets: torch.Tensor, predictions: torch.Tensor, n_classes: int
) -> dict[str, float]:
    targets = targets.long().cpu()
    predictions = predictions.long().cpu()
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.float64)
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        confusion[target, prediction] += 1
    total = confusion.sum()
    true_counts = confusion.sum(dim=1)
    predicted_counts = confusion.sum(dim=0)
    diagonal = confusion.diag()
    recalls = diagonal / true_counts.clamp_min(1)
    precision = diagonal / predicted_counts.clamp_min(1)
    f1 = 2 * precision * recalls / (precision + recalls).clamp_min(1e-12)
    correct = diagonal.sum()
    numerator = correct * total - torch.dot(true_counts, predicted_counts)
    denominator = torch.sqrt(
        (total.square() - true_counts.square().sum())
        * (total.square() - predicted_counts.square().sum())
    )
    return {
        "accuracy": float(correct / total),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "mcc": float(numerator / denominator) if denominator > 0 else float("nan"),
    }


@torch.inference_mode()
def _evaluate(model, loader, device: torch.device, n_classes: int):
    model.eval()
    targets, predictions, probabilities, records = [], [], [], []
    for token_ids, padding_mask, batch_targets, batch_records in loader:
        logits = model(
            token_ids.to(device).long(), padding_mask=padding_mask.to(device)
        )
        targets.append(batch_targets.long().cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
        probabilities.append(logits.softmax(dim=-1).cpu())
        records.extend(batch_records)
    target_tensor = torch.cat(targets)
    prediction_tensor = torch.cat(predictions)
    return (
        classification_metrics(target_tensor, prediction_tensor, n_classes),
        target_tensor,
        prediction_tensor,
        torch.cat(probabilities),
        tuple(records),
    )


def _write_predictions(path, task, fold, records, targets, predictions, probabilities):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "fold", "row_index", "target", "prediction", "probabilities"),
        )
        writer.writeheader()
        for record, target, prediction, probability in zip(
            records, targets.tolist(), predictions.tolist(), probabilities.tolist(), strict=True
        ):
            writer.writerow({
                "task": task,
                "fold": fold,
                "row_index": record.row_index,
                "target": target,
                "prediction": prediction,
                "probabilities": json.dumps(probability, separators=(",", ":")),
            })


def _summarize(task, source, n_samples, sequence_length, n_classes, fold_rows, pooled):
    row = {
        "task": task,
        "source_file": str(source),
        "n_samples": n_samples,
        "sequence_length": sequence_length,
        "n_classes": n_classes,
        "n_folds": len(fold_rows),
    }
    for metric in METRICS:
        values = torch.tensor([float(item[metric]) for item in fold_rows], dtype=torch.float64)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(unbiased=False))
        row[f"pooled_{metric}"] = pooled[metric]
    return row


def _class_weights(
    labels: torch.Tensor,
    train_indices: Sequence[int],
    n_classes: int,
    scheme: str,
    beta: float,
    maximum: float,
) -> torch.Tensor | None:
    if scheme == "none":
        return None
    counts = torch.bincount(labels[list(train_indices)], minlength=n_classes).float()
    if (counts == 0).any():
        raise ValueError("every class must occur in each training fold")
    if scheme == "inverse_sqrt":
        weights = counts.rsqrt()
    elif scheme == "effective_number":
        if not 0.0 < beta < 1.0:
            raise ValueError("class_weight_beta must be between zero and one")
        weights = (1.0 - beta) / (1.0 - beta**counts)
    else:
        raise ValueError("class_weighting must be none, inverse_sqrt, or effective_number")
    weights = weights / weights.mean()
    return weights.clamp(max=maximum)


def run_classification(config: ClassificationConfig) -> list[dict[str, object]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**asdict(config), "input_path": str(config.input_path), "output_dir": str(config.output_dir), "loss": "cross_entropy", "weights_saved": True}
    if config.model_type == "hyena":
        run_config["parameter_count_policy"] = "exclude_hyena_positional_embeddings"
        run_config["hyena_effective_order"] = (
            config.hyena_order
            if config.hyena_order is not None
            else (2 if config.d_model <= 64 else 4)
        )
    run_config_path = config.output_dir / "run_config.json"
    if not run_config_path.exists():
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    alphabet = ProteinAlphabet(config.alphabet)
    completed = set()
    task_path = config.output_dir / "task_metrics.csv"
    if task_path.exists():
        with task_path.open(newline="", encoding="utf-8") as handle:
            completed = {row["task"] for row in csv.DictReader(handle)}
    task_rows = []
    for task_index, path in enumerate(discover_assays(config.input_path, config.pattern), 1):
        if path.stem in completed:
            LOGGER.info("[%d] %s already complete", task_index, path.stem)
            continue
        assay = load_assay(path, sequence_column=config.sequence_column, target_column=config.target_column, fold_column=config.fold_column, n_folds=config.n_folds, seed=config.seed)
        labels = torch.tensor([int(record.target) for record in assay.records], dtype=torch.long)
        classes = sorted(labels.unique().tolist())
        if classes != list(range(len(classes))) or len(classes) < 2:
            raise ValueError(f"{assay.name}: labels must be contiguous integers from zero; got {classes}")
        n_classes = len(classes)
        _write_split_manifest(config.output_dir / "split_manifest.csv", assay, config.seed, config.fold_column)
        token_ids = [alphabet.encode_token_ids([record.sequence]).squeeze(0) for record in assay.records]
        fold_rows, pooled_targets, pooled_predictions = [], [], []
        for fold in range(config.n_folds):
            started = time.perf_counter()
            fold_seed = config.seed + fold + zlib.crc32(assay.name.encode())
            _seed(fold_seed)
            train_indices = [i for i, value in enumerate(assay.folds) if value != fold]
            test_indices = [i for i, value in enumerate(assay.folds) if value == fold]
            train_loader = _loader(assay, train_indices, token_ids, labels, config, alphabet.unknown_index, shuffle=True, seed=fold_seed)
            test_loader = _loader(assay, test_indices, token_ids, labels, config, alphabet.unknown_index, shuffle=False, seed=fold_seed)
            model = _build_model(config, alphabet.size)
            head_width = model.output_projection.in_features
            model.output_projection = nn.Linear(head_width, n_classes)
            model = model.to(device)
            class_weights = _class_weights(
                labels, train_indices, n_classes, config.class_weighting,
                config.class_weight_beta, config.class_weight_max,
            )
            if class_weights is not None:
                class_weights = class_weights.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            iterator = iter(train_loader)
            interval_loss = 0.0
            interval_samples = 0
            final = None
            for step in range(1, config.training_steps + 1):
                lr = _scheduled_learning_rate(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                ids, mask, targets, _ = batch
                ids, mask, targets = ids.to(device).long(), mask.to(device), targets.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids, padding_mask=mask)
                loss = F.cross_entropy(logits, targets, weight=class_weights)
                loss.backward()
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                interval_loss += float(loss.detach()) * len(targets)
                interval_samples += len(targets)
                if step % config.eval_every_steps and step != config.training_steps:
                    continue
                final = _evaluate(model, test_loader, device, n_classes)
                metrics = final[0]
                _append_row(config.output_dir / "epoch_metrics.csv", EPOCH_FIELDS, {
                    "task": assay.name, "fold": fold, "training_step": step,
                    "n_train": len(train_indices), "n_test": len(test_indices),
                    "train_cross_entropy": interval_loss / interval_samples,
                    **metrics, "elapsed_seconds": time.perf_counter() - started,
                })
                LOGGER.info("%s fold=%d step=%d accuracy=%.4f macro_f1=%.4f mcc=%.4f", assay.name, fold, step, metrics["accuracy"], metrics["macro_f1"], metrics["mcc"])
                interval_loss = 0.0
                interval_samples = 0
            assert final is not None
            metrics, targets, predictions, probabilities, records = final
            pooled_targets.append(targets)
            pooled_predictions.append(predictions)
            fold_row = {"task": assay.name, "fold": fold, "n_train": len(train_indices), "n_test": len(test_indices), **metrics, "elapsed_seconds": time.perf_counter() - started}
            fold_rows.append(fold_row)
            _append_row(config.output_dir / "fold_metrics.csv", FOLD_FIELDS, fold_row)
            _write_predictions(config.output_dir / "predictions" / assay.name / f"fold_{fold}.csv", assay.name, fold, records, targets, predictions, probabilities)
            checkpoint_dir = config.output_dir / "models" / assay.name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "model_type": config.model_type, "d_model": config.d_model, "d_state": config.d_state, "n_classes": n_classes, "class_ids": classes, "fold": fold, "seed": fold_seed}, checkpoint_dir / f"fold_{fold}.pt")
        pooled = classification_metrics(torch.cat(pooled_targets), torch.cat(pooled_predictions), n_classes)
        task_row = _summarize(assay.name, assay.source, len(assay.records), assay.sequence_length, n_classes, fold_rows, pooled)
        task_rows.append(task_row)
        _append_row(task_path, TASK_FIELDS, task_row)
    if task_rows:
        aggregate = {metric: float(torch.tensor([float(row[f"pooled_{metric}"]) for row in task_rows]).mean()) for metric in METRICS}
        (config.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return task_rows


def load_config(path: Path) -> ClassificationConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(ClassificationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    base = path.resolve().parent
    for key in ("input_path", "output_dir"):
        value = Path(values[key])
        values[key] = value if value.is_absolute() else base / value
    return ClassificationConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_classification(load_config(args.config))


if __name__ == "__main__":
    main()
