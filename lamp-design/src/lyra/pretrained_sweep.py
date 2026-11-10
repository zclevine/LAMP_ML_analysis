"""Run frozen-backbone embedding extraction and direct-linear probe sweeps."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from .extract_embeddings import EmbeddingConfig, extract_embeddings
from .frozen_probe import FrozenProbeConfig, run_frozen_probes


LOGGER = logging.getLogger("lyra.pretrained_sweep")


@dataclass(frozen=True)
class PretrainedModelConfig:
    name: str
    checkpoint: str
    backend: str = "huggingface"
    batch_size: int = 8
    max_residues: int = 1022
    dtype: str = "bfloat16"
    layer_name: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class PretrainedSweepConfig:
    input_path: Path
    output_dir: Path
    devices: tuple[str, ...]
    models: tuple[PretrainedModelConfig, ...]
    sequence_column: str = "mutated_sequence"
    target_column: str = "DMS_score"
    pattern: str = "*.csv"
    fold_column: str | None = None
    n_folds: int = 5
    seed: int = 42
    training_steps: int = 10000
    eval_every_steps: int = 10000
    probe_batch_size: int = 2048
    learning_rates: tuple[float, ...] = (1e-3, 4e-4)
    min_learning_rate: float | None = 4e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-2
    normalize_targets: bool = True


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _validate_name(name: str) -> None:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name):
        raise ValueError(
            f"model name {name!r} may contain only letters, digits, '-' and '_'"
        )


def load_pretrained_sweep(path: Path) -> PretrainedSweepConfig:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    allowed = {
        "input_path",
        "output_dir",
        "devices",
        "models",
        "sequence_column",
        "target_column",
        "pattern",
        "fold_column",
        "n_folds",
        "seed",
        "training_steps",
        "eval_every_steps",
        "probe_batch_size",
        "learning_rates",
        "min_learning_rate",
        "warmup_steps",
        "weight_decay",
        "normalize_targets",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown keys: {unknown}")
    required = {"input_path", "output_dir", "models"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    model_values = document.pop("models")
    if not isinstance(model_values, list) or not model_values:
        raise ValueError(f"{path}: models must be a non-empty list")
    models = []
    model_fields = set(PretrainedModelConfig.__dataclass_fields__)
    for values in model_values:
        if not isinstance(values, dict):
            raise ValueError(f"{path}: every model must be a YAML mapping")
        unknown_model = sorted(set(values) - model_fields)
        if unknown_model:
            raise ValueError(f"{path}: unknown model keys: {unknown_model}")
        model = PretrainedModelConfig(**values)
        _validate_name(model.name)
        models.append(model)
    if len({model.name for model in models}) != len(models):
        raise ValueError(f"{path}: model names must be unique")

    base = path.resolve().parent
    document["input_path"] = _resolve_path(document["input_path"], base)
    document["output_dir"] = _resolve_path(document["output_dir"], base)
    document["devices"] = tuple(str(value) for value in document.get("devices", ["auto"]))
    document["learning_rates"] = tuple(
        float(value) for value in document.get("learning_rates", [1e-3, 4e-4])
    )
    if not document["devices"] or not document["learning_rates"]:
        raise ValueError("devices and learning_rates must be non-empty")
    return PretrainedSweepConfig(models=tuple(models), **document)


def _lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:g}".replace(".", "p").replace("-", "m")


def _run_models(
    config: PretrainedSweepConfig,
    models: Sequence[PretrainedModelConfig],
    device: str,
) -> None:
    for model in models:
        feature_dir = config.output_dir / "features" / model.name
        extract_embeddings(
            EmbeddingConfig(
                input_path=config.input_path,
                output_dir=feature_dir,
                checkpoint=model.checkpoint,
                backend=model.backend,
                sequence_column=config.sequence_column,
                pattern=config.pattern,
                batch_size=model.batch_size,
                max_residues=model.max_residues,
                device=device,
                dtype=model.dtype,
                resume=True,
                layer_name=model.layer_name,
                trust_remote_code=model.trust_remote_code,
            )
        )
        for learning_rate in config.learning_rates:
            output_dir = (
                config.output_dir
                / "probes"
                / model.name
                / f"lr-{_lr_slug(learning_rate)}"
            )
            if (output_dir / "aggregate_metrics.csv").exists():
                LOGGER.info("%s lr=%g already complete; skipping", model.name, learning_rate)
                continue
            run_frozen_probes(
                FrozenProbeConfig(
                    input_path=config.input_path,
                    feature_dir=feature_dir,
                    output_dir=output_dir,
                    backbone=model.name,
                    sequence_column=config.sequence_column,
                    target_column=config.target_column,
                    fold_column=config.fold_column,
                    pattern=config.pattern,
                    n_folds=config.n_folds,
                    seed=config.seed,
                    training_steps=config.training_steps,
                    eval_every_steps=config.eval_every_steps,
                    batch_size=config.probe_batch_size,
                    learning_rate=learning_rate,
                    min_learning_rate=config.min_learning_rate,
                    warmup_steps=config.warmup_steps,
                    weight_decay=config.weight_decay,
                    device=device,
                    normalize_targets=config.normalize_targets,
                )
            )


def run_pretrained_sweep(config: PretrainedSweepConfig) -> None:
    assignments = [config.models[index :: len(config.devices)] for index in range(len(config.devices))]
    if len(config.devices) == 1:
        _run_models(config, assignments[0], config.devices[0])
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(config.devices), mp_context=context) as executor:
        futures = [
            executor.submit(_run_models, config, models, device)
            for models, device in zip(assignments, config.devices, strict=True)
            if models
        ]
        for future in futures:
            future.result()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen embeddings and run direct-linear probe sweeps"
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pretrained_sweep(load_pretrained_sweep(args.config))


if __name__ == "__main__":
    main()
