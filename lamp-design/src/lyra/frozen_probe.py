"""Five-fold direct linear probes over cached frozen-backbone embeddings."""

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

from .proteingym import (
    FOLD_FIELDS,
    METRIC_NAMES,
    TASK_FIELDS,
    _append_row,
    _read_rows,
    _resolve_device,
    _scheduled_learning_rate,
    _summarize_task,
    _write_aggregate,
    _write_split_manifest,
    load_assay,
    regression_metrics,
)
from .extract_embeddings import _sequence_sha256


LOGGER = logging.getLogger("lyra.frozen_probe")
PROBE_TASK_FIELDS = (*TASK_FIELDS, "feature_checkpoint", "linear_head_parameters")
PROBE_EPOCH_FIELDS = (
    "task",
    "fold",
    "training_step",
    "n_train",
    "n_test",
    "train_mse",
    *METRIC_NAMES,
    "elapsed_seconds",
)


@dataclass(frozen=True)
class FrozenProbeConfig:
    input_path: Path
    feature_dir: Path
    output_dir: Path
    backbone: str
    sequence_column: str = "mutated_sequence"
    target_column: str = "DMS_score"
    fold_column: str | None = None
    pattern: str = "*.csv"
    n_folds: int = 5
    seed: int = 42
    training_steps: int = 10000
    eval_every_steps: int = 1000
    batch_size: int = 2048
    learning_rate: float = 1e-3
    min_learning_rate: float | None = 4e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    device: str = "auto"
    normalize_targets: bool = True
    resume: bool = True


def _discover(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {path}")
    return files


def _load_features(path: Path, expected_rows: int) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"{path}: expected a dictionary containing 'features'")
    features = payload["features"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"{path}: features must have shape (samples, hidden_dim)")
    if features.size(0) != expected_rows:
        raise ValueError(
            f"{path}: feature rows {features.size(0)} do not match assay rows "
            f"{expected_rows}"
        )
    metadata = payload.get("metadata", {})
    return features.float(), metadata if isinstance(metadata, dict) else {}


@torch.inference_mode()
def _evaluate(
    model: nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    predictions = []
    for batch in indices.split(batch_size):
        predictions.append(model(features[batch].to(device)).squeeze(-1).cpu())
    prediction_tensor = torch.cat(predictions)
    return regression_metrics(targets[indices], prediction_tensor), prediction_tensor


def _write_predictions(
    path: Path,
    assay,
    fold: int,
    indices: torch.Tensor,
    normalized_targets: torch.Tensor,
    predictions: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "task",
            "fold",
            "row_index",
            "mutant",
            "DMS_score",
            "DMS_score_normalized",
            "prediction",
            "prediction_raw_scale",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, target, prediction in zip(
            indices.tolist(),
            normalized_targets[indices].tolist(),
            predictions.tolist(),
            strict=True,
        ):
            record = assay.records[index]
            writer.writerow(
                {
                    "task": assay.name,
                    "fold": fold,
                    "row_index": record.row_index,
                    "mutant": record.mutant,
                    "DMS_score": record.target,
                    "DMS_score_normalized": target,
                    "prediction": prediction,
                    "prediction_raw_scale": prediction * target_std + target_mean,
                }
            )


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_frozen_probes(config: FrozenProbeConfig) -> list[dict[str, object]]:
    """Fit only one ``Linear(hidden_dim, 1)`` head for each assay and fold."""

    if (
        config.output_dir.exists()
        and any(config.output_dir.iterdir())
        and not config.resume
    ):
        raise FileExistsError(
            f"output directory is not empty: {config.output_dir}; choose a new "
            "directory or set resume: true"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    paths = _discover(config.input_path, config.pattern)
    task_rows: list[dict[str, object]] = []
    run_metadata: dict[str, object] = asdict(config)
    run_metadata.update(
        input_path=str(config.input_path),
        feature_dir=str(config.feature_dir),
        output_dir=str(config.output_dir),
        head="direct_linear",
        backbone_frozen=True,
    )
    run_config_path = config.output_dir / "run_config.json"
    if run_config_path.exists():
        with run_config_path.open(encoding="utf-8") as handle:
            previous_config = json.load(handle)
        comparable_previous = dict(previous_config)
        comparable_previous["resume"] = config.resume
        if comparable_previous != run_metadata:
            raise ValueError(
                f"cannot resume {config.output_dir}: run_config.json does not "
                "match the requested configuration"
            )
    else:
        with run_config_path.open("w", encoding="utf-8") as handle:
            json.dump(run_metadata, handle, indent=2)

    existing_task_rows = {
        row["task"]: row for row in _read_rows(config.output_dir / "task_metrics.csv")
    }
    existing_fold_rows = {
        (row["task"], int(row["fold"])): row
        for row in _read_rows(config.output_dir / "fold_metrics.csv")
    }
    existing_split_tasks = {
        row["task"] for row in _read_rows(config.output_dir / "split_manifest.csv")
    }

    for task_index, path in enumerate(paths, start=1):
        if path.stem in existing_task_rows:
            LOGGER.info(
                "[%d/%d] %s: already complete; skipping",
                task_index,
                len(paths),
                path.stem,
            )
            task_rows.append(existing_task_rows[path.stem])
            continue
        assay = load_assay(
            path,
            sequence_column=config.sequence_column,
            target_column=config.target_column,
            fold_column=config.fold_column,
            n_folds=config.n_folds,
            seed=config.seed,
        )
        if assay.name not in existing_split_tasks:
            _write_split_manifest(
                config.output_dir / "split_manifest.csv",
                assay,
                config.seed,
                config.fold_column,
            )
        features, metadata = _load_features(
            config.feature_dir / f"{assay.name}.pt", len(assay.records)
        )
        expected_sequence_hash = _sequence_sha256(
            [record.sequence for record in assay.records]
        )
        cached_sequence_hash = metadata.get("sequence_sha256")
        if (
            cached_sequence_hash is not None
            and cached_sequence_hash != expected_sequence_hash
        ):
            raise ValueError(
                f"{assay.name}: cached features do not match the current sequence "
                "order or contents"
            )
        LOGGER.info(
            "[%d/%d] %s: %d x %d frozen features",
            task_index,
            len(paths),
            assay.name,
            features.size(0),
            features.size(1),
        )
        raw_targets = torch.tensor([record.target for record in assay.records])
        fold_rows: list[dict[str, object]] = []
        pooled_targets: list[torch.Tensor] = []
        pooled_predictions: list[torch.Tensor] = []
        for fold in range(config.n_folds):
            prediction_path = (
                config.output_dir / "predictions" / assay.name / f"fold_{fold}.csv"
            )
            existing_fold = existing_fold_rows.get((assay.name, fold))
            if existing_fold is not None and prediction_path.exists():
                predictions = _read_rows(prediction_path)
                pooled_targets.append(
                    torch.tensor(
                        [float(row["DMS_score_normalized"]) for row in predictions]
                    )
                )
                pooled_predictions.append(
                    torch.tensor([float(row["prediction"]) for row in predictions])
                )
                fold_rows.append(existing_fold)
                LOGGER.info("%s fold=%d already complete; skipping", assay.name, fold)
                continue
            started = time.perf_counter()
            fold_seed = config.seed + fold + zlib.crc32(assay.name.encode())
            _seed(fold_seed)
            train_indices = torch.tensor(
                [index for index, value in enumerate(assay.folds) if value != fold]
            )
            test_indices = torch.tensor(
                [index for index, value in enumerate(assay.folds) if value == fold]
            )
            target_mean = float(raw_targets[train_indices].mean())
            target_std = float(raw_targets[train_indices].std(unbiased=False))
            if not config.normalize_targets:
                target_mean, target_std = 0.0, 1.0
            if target_std <= 0:
                raise ValueError(f"{assay.name} fold {fold}: zero target variance")
            targets = (raw_targets - target_mean) / target_std
            device_features = features.to(device)
            device_targets = targets.to(device)
            device_train_indices = train_indices.to(device)

            model = nn.Linear(features.size(1), 1).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            generator = torch.Generator(device=device).manual_seed(fold_seed)
            interval_error = 0.0
            interval_samples = 0
            final_metrics: dict[str, float] | None = None
            final_predictions: torch.Tensor | None = None
            for step in range(1, config.training_steps + 1):
                learning_rate = _scheduled_learning_rate(config, step)  # type: ignore[arg-type]
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                sampled = device_train_indices[
                    torch.randint(
                        len(train_indices),
                        (min(config.batch_size, len(train_indices)),),
                        generator=generator,
                        device=device,
                    )
                ]
                batch_features = device_features[sampled]
                batch_targets = device_targets[sampled]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_features).squeeze(-1)
                loss = F.mse_loss(prediction, batch_targets)
                loss.backward()
                optimizer.step()
                interval_error += float(loss.detach()) * len(sampled)
                interval_samples += len(sampled)

                if step % config.eval_every_steps != 0 and step != config.training_steps:
                    continue
                final_metrics, final_predictions = _evaluate(
                    model,
                    features,
                    targets,
                    test_indices,
                    device,
                    config.batch_size,
                )
                _append_row(
                    config.output_dir / "epoch_metrics.csv",
                    PROBE_EPOCH_FIELDS,
                    {
                        "task": assay.name,
                        "fold": fold,
                        "training_step": step,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "train_mse": interval_error / interval_samples,
                        **final_metrics,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                interval_error = 0.0
                interval_samples = 0

            assert final_metrics is not None and final_predictions is not None
            test_targets = targets[test_indices]
            pooled_targets.append(test_targets)
            pooled_predictions.append(final_predictions)
            fold_row = {
                "task": assay.name,
                "fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                **final_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            fold_rows.append(fold_row)
            _append_row(config.output_dir / "fold_metrics.csv", FOLD_FIELDS, fold_row)
            _write_predictions(
                prediction_path,
                assay,
                fold,
                test_indices,
                targets,
                final_predictions,
                target_mean,
                target_std,
            )

        pooled = regression_metrics(torch.cat(pooled_targets), torch.cat(pooled_predictions))
        task_row = _summarize_task(assay, fold_rows, pooled)
        task_row["feature_checkpoint"] = metadata.get("checkpoint", config.backbone)
        task_row["linear_head_parameters"] = features.size(1) + 1
        task_rows.append(task_row)
        _append_row(
            config.output_dir / "task_metrics.csv", PROBE_TASK_FIELDS, task_row
        )

    _write_aggregate(config.output_dir / "aggregate_metrics.csv", task_rows)
    return task_rows


def load_config(path: Path) -> FrozenProbeConfig:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    allowed = {field.name for field in fields(FrozenProbeConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown configuration keys: {unknown}")
    required = {"input_path", "feature_dir", "output_dir", "backbone"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    base = path.resolve().parent
    for key in ("input_path", "feature_dir", "output_dir"):
        value = Path(values[key])
        values[key] = value if value.is_absolute() else base / value
    return FrozenProbeConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit frozen-backbone linear probes")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_frozen_probes(load_config(args.config))


if __name__ == "__main__":
    main()
