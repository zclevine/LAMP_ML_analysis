"""Train and evaluate reproducible Lyra synthetic experiment grids."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .synthetic_data import SyntheticDataset, make_frequency_dataset, make_walsh_epistasis
from .synthetic_metrics import (
    fit_affine_calibration,
    frequency_metrics,
    per_channel_metrics,
    regression_metrics,
    walsh_recovery_metrics,
)
from .synthetic_models import ContinuousSyntheticModel, count_parameters


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def load_synthetic_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("synthetic config must be a mapping")
    for required in ("tasks", "models", "training", "output_dir"):
        if required not in config:
            raise ValueError(f"synthetic config is missing {required}")
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    config["output_dir"] = str(output)
    config["config_path"] = str(path)
    config.setdefault("seeds", [1, 2, 3, 4, 5])
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(task: dict, seed: int) -> SyntheticDataset:
    data = copy.deepcopy(task.get("data", {}))
    data["seed"] = int(data.get("seed", 0)) + int(seed)
    if task["kind"] == "walsh_epistasis":
        return make_walsh_epistasis(**data)
    if task["kind"] == "frequency":
        for name in (
            "amplitude_range",
            "test_amplitude_range",
            "phase_range",
            "test_phase_range",
            "offset_range",
        ):
            if name in data and data[name] is not None:
                data[name] = tuple(data[name])
        return make_frequency_dataset(**data)
    raise ValueError(f"unknown task kind: {task['kind']}")


def _target_statistics(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dimensions = tuple(range(target.ndim - 1))
    mean = target.mean(dim=dimensions, keepdim=True)
    std = target.std(dim=dimensions, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def _optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float):
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        override = getattr(parameter, "_optim", {})
        key = (
            float(override.get("lr", learning_rate)),
            float(override.get("weight_decay", weight_decay)),
        )
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": parameters, "lr": key[0], "weight_decay": key[1]}
            for key, parameters in groups.items()
        ]
    )


def _loader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _loss_slice(
    output: torch.Tensor, target: torch.Tensor, observed_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 3 or observed_fraction >= 1.0:
        return output, target
    start = max(0, min(output.size(1) - 1, int(round(output.size(1) * observed_fraction))))
    return output[:, start:], target[:, start:]


def _predict(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    model.eval()
    output = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(inputs), batch_size=batch_size):
            with _autocast(device, precision):
                prediction = model(batch.to(device))
            output.append(prediction.float().cpu())
    return torch.cat(output)


def _run_identifier(task: dict, model: dict, seed: int) -> str:
    payload = json.dumps(
        _jsonable({"task": task, "model": model, "seed": seed}), sort_keys=True
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
    safe_task = "".join(character if character.isalnum() else "-" for character in task["name"])
    safe_model = "".join(character if character.isalnum() else "-" for character in model["name"])
    return f"{safe_task}__{safe_model}__seed-{seed}__{digest}"


def train_synthetic_run(
    task: dict,
    model_config: dict,
    training: dict,
    seed: int,
    device: torch.device,
) -> dict:
    _set_seed(seed)
    dataset = _make_dataset(task, seed)
    target = dataset.train.targets
    output_mode = "sequence" if target.ndim == 3 else "scalar"
    model_kwargs = copy.deepcopy(model_config)
    model_name = model_kwargs.pop("name")
    kind = model_kwargs.pop("kind")
    model_kwargs.pop("training", None)
    model = ContinuousSyntheticModel(
        kind=kind,
        input_dim=dataset.train.inputs.size(-1),
        output_dim=target.size(-1),
        sequence_length=dataset.train.inputs.size(1),
        output_mode=output_mode,
        **model_kwargs,
    ).to(device)
    parameter_count = count_parameters(model)

    epochs = int(training.get("epochs", 50))
    batch_size = int(training.get("batch_size", 128))
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-4))
    patience = int(training.get("patience", epochs))
    workers = int(training.get("workers", 0))
    precision = str(training.get("precision", "fp32"))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    observed_fraction = float(dataset.metadata.get("observed_fraction", 1.0))
    pin_memory = device.type == "cuda"

    target_mean, target_std = _target_statistics(dataset.train.targets)
    train_targets = (dataset.train.targets - target_mean) / target_std
    validation_targets = (dataset.validation.targets - target_mean) / target_std
    train_loader = _loader(
        dataset.train.inputs,
        train_targets,
        batch_size,
        True,
        seed,
        workers,
        pin_memory,
    )
    validation_loader = _loader(
        dataset.validation.inputs,
        validation_targets,
        batch_size,
        False,
        seed,
        workers,
        pin_memory,
    )
    optimizer = _optimizer(model, learning_rate, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                prediction = model(inputs)
                prediction_loss, target_loss = _loss_slice(
                    prediction, targets, observed_fraction
                )
                loss = torch.nn.functional.mse_loss(prediction_loss, target_loss)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_sum += float(loss.detach().item()) * target_loss.numel()
            train_count += target_loss.numel()

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with _autocast(device, precision):
                    prediction = model(inputs)
                    prediction_loss, target_loss = _loss_slice(
                        prediction, targets, observed_fraction
                    )
                    loss = torch.nn.functional.mse_loss(
                        prediction_loss, target_loss, reduction="sum"
                    )
                validation_sum += float(loss.item())
                validation_count += target_loss.numel()
        validation_loss = validation_sum / max(1, validation_count)
        train_loss = train_sum / max(1, train_count)
        history.append(
            {"epoch": epoch + 1, "train_mse": train_loss, "validation_mse": validation_loss}
        )
        if validation_loss < best_loss - 1e-9:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = _predict(
        model, dataset.validation.inputs, batch_size, device, precision
    )
    test_prediction = _predict(model, dataset.test.inputs, batch_size, device, precision)
    validation_prediction = validation_prediction * target_std + target_mean
    test_prediction = test_prediction * target_std + target_mean

    reconstructions = []
    reconstruction_count = int(training.get("save_reconstructions", 0))
    if output_mode == "sequence" and reconstruction_count > 0:
        for index in range(min(reconstruction_count, dataset.test.inputs.size(0))):
            reconstructions.append(
                {
                    "index": index,
                    "inputs": dataset.test.inputs[index].float().tolist(),
                    "targets": dataset.test.targets[index].float().tolist(),
                    "predictions": test_prediction[index].float().tolist(),
                }
            )

    evaluation_target = dataset.test.targets
    evaluation_validation_target = dataset.validation.targets
    if output_mode == "sequence" and observed_fraction < 1.0:
        start = int(round(evaluation_target.size(1) * observed_fraction))
        evaluation_target = evaluation_target[:, start:]
        evaluation_validation_target = evaluation_validation_target[:, start:]
        test_prediction = test_prediction[:, start:]
        validation_prediction = validation_prediction[:, start:]

    if task["kind"] == "frequency" and dataset.metadata["target"] == "components":
        metrics = frequency_metrics(
            evaluation_target,
            test_prediction,
            evaluation_validation_target,
            validation_prediction,
            dataset.metadata["frequencies"],
            spectral_eligible=observed_fraction >= 1.0,
        )
    elif evaluation_target.size(-1) > 1:
        channels = per_channel_metrics(
            evaluation_target,
            test_prediction,
            evaluation_validation_target,
            validation_prediction,
        )
        metrics = {
            "overall": regression_metrics(
                evaluation_target,
                test_prediction,
                calibration=fit_affine_calibration(
                    validation_prediction, evaluation_validation_target
                ),
            ),
            "channels": channels,
        }
    else:
        calibration = fit_affine_calibration(
            validation_prediction, evaluation_validation_target
        )
        metrics = regression_metrics(
            evaluation_target, test_prediction, calibration=calibration
        )

    if task["kind"] == "walsh_epistasis":
        recovery_prediction = test_prediction
        if dataset.metadata.get("target") == "terms":
            recovery_prediction = test_prediction.sum(dim=-1, keepdim=True)
        metrics["walsh_recovery"] = walsh_recovery_metrics(
            recovery_prediction,
            dataset.test.components,
            dataset.metadata["weights"],
            dataset.metadata["term_degrees"],
        )

    return {
        "status": "complete",
        "task": task["name"],
        "task_kind": task["kind"],
        "model": model_name,
        "model_kind": kind,
        "seed": seed,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_standardized_mse": best_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "precision": precision,
        "dataset": dataset.metadata,
        "model_config": model_config,
        "training_config": training,
        "metrics": metrics,
        "history": history,
        "reconstructions": reconstructions,
    }


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_flat_csv(output_dir: Path) -> None:
    rows = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        with path.open() as handle:
            row = json.load(handle)
        if row.get("status") != "complete":
            continue
        metrics = row["metrics"].get("composite", row["metrics"].get("overall", row["metrics"]))
        rows.append(
            {
                "task": row["task"],
                "model": row["model"],
                "seed": row["seed"],
                "parameters": row["parameters"],
                "r2": metrics.get("r2"),
                "pearson": metrics.get("pearson"),
                "spearman": metrics.get("spearman"),
                "validation_calibrated_r2": metrics.get("validation_calibrated_r2"),
                "oracle_calibration_gap": metrics.get("oracle_calibration_gap"),
                "rmse": metrics.get("rmse"),
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    if not rows:
        return
    with (output_dir / "all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_grid(
    config: dict,
    *,
    device: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    selected_models: set[str] | None = None,
    selected_tasks: set[str] | None = None,
) -> list[dict]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    output_dir = Path(config["output_dir"])
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w") as handle:
        json.dump(_jsonable(config), handle, indent=2, sort_keys=True)

    work = []
    for task in config["tasks"]:
        if selected_tasks and task["name"] not in selected_tasks:
            continue
        for model in config["models"]:
            if selected_models and model["name"] not in selected_models:
                continue
            if task.get("models") and model["name"] not in set(task["models"]):
                continue
            for seed in config["seeds"]:
                work.append((task, model, int(seed)))
    work.sort(key=lambda values: _run_identifier(*values))
    work = [item for index, item in enumerate(work) if index % num_shards == shard_index]

    results = []
    for task, model, seed in work:
        identifier = _run_identifier(task, model, seed)
        result_path = runs_dir / f"{identifier}.json"
        if result_path.exists() and config.get("resume", True):
            with result_path.open() as handle:
                existing = json.load(handle)
            if existing.get("status") == "complete":
                results.append(existing)
                continue
        try:
            run_training = {
                **config["training"],
                **task.get("training", {}),
                **model.get("training", {}),
            }
            row = train_synthetic_run(
                task, model, run_training, seed, torch_device
            )
        except Exception as error:
            row = {
                "status": "failed",
                "task": task["name"],
                "model": model["name"],
                "seed": seed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            if not config.get("continue_on_error", True):
                raise
        with result_path.open("w") as handle:
            json.dump(_jsonable(row), handle, indent=2, sort_keys=True)
        _append_jsonl(output_dir / "events.jsonl", row)
        results.append(row)
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    _write_flat_csv(output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lyra synthetic experiment grids")
    parser.add_argument("config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()
    config = load_synthetic_config(args.config)
    results = run_grid(
        config,
        device=args.device,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        selected_models=set(args.models) if args.models else None,
        selected_tasks=set(args.tasks) if args.tasks else None,
    )
    complete = sum(result.get("status") == "complete" for result in results)
    failed = sum(result.get("status") == "failed" for result in results)
    print(json.dumps({"complete": complete, "failed": failed, "total": len(results)}))


if __name__ == "__main__":
    main()
