"""Five-fold ProteinGym regression with incremental CSV logging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import time
import zlib
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .hyena import HyenaRegressor, parameter_counts
from .mamba import MambaRegressor
from .model import LyraRegressor


LOGGER = logging.getLogger("lyra.proteingym")
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX-"

METRIC_NAMES = ("mse", "rmse", "mae", "r2", "pearson", "spearman")
EPOCH_FIELDS = (
    "task",
    "fold",
    "epoch",
    "training_step",
    "n_train",
    "n_test",
    "train_mse",
    *METRIC_NAMES,
    "elapsed_seconds",
)
FOLD_FIELDS = (
    "task",
    "fold",
    "n_train",
    "n_test",
    *METRIC_NAMES,
    "elapsed_seconds",
)
TASK_FIELDS = (
    "task",
    "source_file",
    "n_samples",
    "sequence_length",
    "n_folds",
    *(f"{metric}_mean" for metric in METRIC_NAMES),
    *(f"{metric}_std" for metric in METRIC_NAMES),
    *(f"pooled_{metric}" for metric in METRIC_NAMES),
)
SPLIT_MANIFEST_FIELDS = (
    "task",
    "source_file",
    "n_samples",
    "sequence_length",
    "fold_column",
    "split_seed",
    "fold_counts",
    "assignment_sha256",
)


@dataclass(frozen=True)
class AssayRecord:
    row_index: int
    sequence: str
    target: float
    mutant: str


@dataclass(frozen=True)
class Assay:
    name: str
    source: Path
    records: tuple[AssayRecord, ...]
    folds: tuple[int, ...]
    sequence_length: int


@dataclass(frozen=True)
class BenchmarkConfig:
    input_path: Path
    output_dir: Path
    sequence_column: str = "mutated_sequence"
    target_column: str = "DMS_score"
    # ``None`` deliberately means an independently generated random split.
    # Set this to ``fold_random_5`` only when reproducing ProteinGym's supplied
    # random folds.
    fold_column: str | None = None
    pattern: str = "*.csv"
    alphabet: str = PROTEIN_ALPHABET
    n_folds: int = 5
    seed: int = 42
    model_type: str = "lyra"
    d_model: int = 128
    d_state: int = 64
    num_blocks: int = 1
    mamba_expand: int = 3
    hyena_max_length: int = 4096
    # ``None`` selects the closest parameter match: order 2 at d64 and order 4
    # at d128, with positional tables excluded from the comparison.
    hyena_order: int | None = None
    dropout: float = 0.1
    causal_pgc: bool = False
    lyra_pgc_residual: bool = False
    featurewise_dropout: bool = False
    pooling: str = "mean"
    bidirectional: bool = False
    epochs: int = 20
    training_steps: int | None = None
    eval_every_steps: int = 100
    batch_size: int = 32
    max_tokens_per_batch: int | None = None
    learning_rate: float = 1e-4
    min_learning_rate: float | None = None
    warmup_steps: int = 0
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    save_models: bool = False
    normalize_targets: bool = False
    resume: bool = False


def _build_model(config: BenchmarkConfig, vocab_size: int) -> nn.Module:
    if config.model_type == "lyra":
        return LyraRegressor(
            d_model=config.d_model,
            vocab_size=vocab_size,
            d_state=config.d_state,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            causal_pgc=config.causal_pgc,
            pgc_residual=config.lyra_pgc_residual,
            featurewise_dropout=config.featurewise_dropout,
            pooling=config.pooling,
            bidirectional=config.bidirectional,
        )
    if config.model_type in {"mamba2", "mamba3"}:
        if config.num_blocks != 1:
            raise ValueError("Mamba baselines currently require num_blocks: 1")
        return MambaRegressor(
            version=config.model_type,
            d_model=config.d_model,
            vocab_size=vocab_size,
            d_state=config.d_state,
            dropout=config.dropout,
            expand=config.mamba_expand,
            pooling=config.pooling,
            bidirectional=config.bidirectional,
        )
    if config.model_type == "hyena":
        hyena_order = config.hyena_order
        if hyena_order is None:
            hyena_order = 2 if config.d_model <= 64 else 4
        return HyenaRegressor(
            d_model=config.d_model,
            vocab_size=vocab_size,
            d_state=config.d_state,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            max_length=config.hyena_max_length,
            order=hyena_order,
            pooling=config.pooling,
            bidirectional=config.bidirectional,
        )
    raise ValueError("model_type must be one of: lyra, mamba2, mamba3, hyena")


class ProteinAlphabet:
    """Encode biological sequences into a fixed number of one-hot channels."""

    def __init__(self, symbols: str = PROTEIN_ALPHABET) -> None:
        if len(symbols) != len(set(symbols)):
            raise ValueError("alphabet symbols must be unique")
        self.symbols = symbols
        if "X" in symbols:
            self.unknown_index = symbols.index("X")
        elif "N" in symbols:
            self.unknown_index = symbols.index("N")
        else:
            raise ValueError("alphabet must include X or N for unknown symbols")
        self._indices = {symbol: index for index, symbol in enumerate(symbols)}
        translation = bytearray([self.unknown_index] * 256)
        for symbol, index in self._indices.items():
            translation[ord(symbol)] = index
        self._translation = bytes(translation)

    @property
    def size(self) -> int:
        return len(self.symbols)

    def encode_token_ids(self, sequences: Sequence[str]) -> torch.Tensor:
        if not sequences:
            raise ValueError("cannot encode an empty batch")
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError("all sequences in a batch must have the same length")
        sequence_length = lengths.pop()
        encoded = bytearray().join(
            sequence.encode("ascii").translate(self._translation)
            for sequence in sequences
        )
        return torch.frombuffer(encoded, dtype=torch.uint8).reshape(
            len(sequences), sequence_length
        )

    def encode_batch(self, sequences: Sequence[str]) -> torch.Tensor:
        token_ids = self.encode_token_ids(sequences).long()
        return F.one_hot(token_ids, num_classes=self.size).float()


class EncodedAssayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, AssayRecord]]):
    """Index pre-tokenized assay tensors without rebuilding OHE on the CPU."""

    def __init__(
        self,
        assay: Assay,
        indices: Sequence[int],
        token_ids: Sequence[torch.Tensor],
        targets: torch.Tensor,
    ) -> None:
        self.assay = assay
        self.indices = tuple(indices)
        self.token_ids = token_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, AssayRecord]:
        assay_index = self.indices[index]
        return (
            self.token_ids[assay_index],
            self.targets[assay_index],
            self.assay.records[assay_index],
        )


def _assign_random_folds(n_samples: int, n_folds: int, seed: int) -> tuple[int, ...]:
    if n_samples < n_folds:
        raise ValueError(
            f"need at least {n_folds} samples for {n_folds}-fold CV; got {n_samples}"
        )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_samples, generator=generator).tolist()
    folds = [0] * n_samples
    for position, sample_index in enumerate(permutation):
        folds[sample_index] = position % n_folds
    return tuple(folds)


def load_assay(
    path: Path,
    *,
    sequence_column: str = "mutated_sequence",
    target_column: str = "DMS_score",
    fold_column: str | None = "fold_random_5",
    n_folds: int = 5,
    seed: int = 42,
) -> Assay:
    """Load one ProteinGym assay CSV.

    An existing fold column is respected. If it is absent, deterministic and
    balanced random folds are generated without rewriting the source CSV.
    """

    records: list[AssayRecord] = []
    supplied_folds: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = {sequence_column, target_column} - fields
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        has_fold_column = fold_column is not None and fold_column in fields

        for row_index, row in enumerate(reader):
            sequence = row[sequence_column].strip().upper()
            if not sequence:
                raise ValueError(f"{path}: empty sequence at data row {row_index + 2}")
            try:
                target = float(row[target_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: invalid target at data row {row_index + 2}"
                ) from error
            if not math.isfinite(target):
                raise ValueError(f"{path}: non-finite target at data row {row_index + 2}")

            records.append(
                AssayRecord(
                    row_index=row_index,
                    sequence=sequence,
                    target=target,
                    mutant=row.get("mutant", ""),
                )
            )
            if has_fold_column:
                try:
                    supplied_folds.append(int(row[fold_column]))  # type: ignore[index]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}: invalid {fold_column} at data row {row_index + 2}"
                    ) from error

    if len(records) < n_folds:
        raise ValueError(f"{path}: need at least {n_folds} rows; got {len(records)}")
    lengths = {len(record.sequence) for record in records}

    if supplied_folds:
        invalid_folds = sorted(set(supplied_folds) - set(range(n_folds)))
        if invalid_folds:
            raise ValueError(
                f"{path}: fold values must be in [0, {n_folds - 1}]; "
                f"found {invalid_folds}"
            )
        empty_folds = sorted(set(range(n_folds)) - set(supplied_folds))
        if empty_folds:
            raise ValueError(f"{path}: fold column has empty folds {empty_folds}")
        folds = tuple(supplied_folds)
    else:
        task_seed = seed + zlib.crc32(path.stem.encode())
        folds = _assign_random_folds(len(records), n_folds, task_seed)

    return Assay(
        name=path.stem,
        source=path,
        records=tuple(records),
        folds=folds,
        # Retain the historical scalar column. For variable-length FLIP tasks it
        # records the maximum length; batches are padded dynamically below.
        sequence_length=max(lengths),
    )


def discover_assays(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    paths = sorted(path for path in input_path.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {input_path}")
    return paths


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().double().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ranks = torch.empty_like(values)
    start = 0
    for count in counts.tolist():
        stop = start + count
        average_rank = (start + stop - 1) / 2.0 + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().cpu()
    right = right.detach().double().cpu()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator == 0:
        return float("nan")
    return float((left * right).sum() / denominator)


def regression_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    targets = targets.detach().float().cpu()
    predictions = predictions.detach().float().cpu()
    errors = predictions - targets
    mse = float(errors.square().mean())
    total_variance = (targets - targets.mean()).square().sum()
    r2 = (
        float(1.0 - errors.square().sum() / total_variance)
        if total_variance > 0
        else float("nan")
    )
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(errors.abs().mean()),
        "r2": r2,
        "pearson": _correlation(targets, predictions),
        "spearman": _correlation(_rankdata(targets), _rankdata(predictions)),
    }


def _append_row(path: Path, fieldnames: Sequence[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_split_manifest(
    path: Path, assay: Assay, seed: int, fold_column: str | None
) -> None:
    """Persist a compact, comparable fingerprint of the exact split."""

    split_seed = seed + zlib.crc32(assay.name.encode())
    assignment = bytes(assay.folds)
    fold_counts = [assay.folds.count(fold) for fold in range(max(assay.folds) + 1)]
    _append_row(
        path,
        SPLIT_MANIFEST_FIELDS,
        {
            "task": assay.name,
            "source_file": str(assay.source),
            "n_samples": len(assay.records),
            "sequence_length": assay.sequence_length,
            "fold_column": fold_column or "",
            "split_seed": split_seed,
            "fold_counts": json.dumps(fold_counts, separators=(",", ":")),
            "assignment_sha256": hashlib.sha256(assignment).hexdigest(),
        },
    )


def _write_predictions(
    path: Path,
    task: str,
    fold: int,
    records: Sequence[AssayRecord],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "task",
            "fold",
            "row_index",
            "mutant",
            "DMS_score",
            "DMS_score_normalized",
            "prediction",
            "prediction_raw_scale",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, target, prediction in zip(
            records, targets.tolist(), predictions.tolist(), strict=True
        ):
            writer.writerow(
                {
                    "task": task,
                    "fold": fold,
                    "row_index": record.row_index,
                    "mutant": record.mutant,
                    "DMS_score": record.target,
                    "DMS_score_normalized": target,
                    "prediction": prediction,
                    "prediction_raw_scale": prediction * target_std + target_mean,
                }
            )


def _collate_encoded(
    items: Sequence[tuple[torch.Tensor, torch.Tensor, AssayRecord]],
    *,
    padding_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[AssayRecord, ...]]:
    token_ids, targets, records = zip(*items, strict=True)
    lengths = torch.tensor([tokens.numel() for tokens in token_ids])
    padded = pad_sequence(token_ids, batch_first=True, padding_value=padding_value)
    positions = torch.arange(padded.size(1)).unsqueeze(0)
    padding_mask = positions < lengths.unsqueeze(1)
    return padded, padding_mask, torch.stack(targets), tuple(records)


def _loader(
    assay: Assay,
    indices: Sequence[int],
    token_ids: Sequence[torch.Tensor],
    targets: torch.Tensor,
    config: BenchmarkConfig,
    padding_value: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "dataset": EncodedAssayDataset(assay, indices, token_ids, targets),
        "num_workers": config.num_workers,
        "pin_memory": _resolve_device(config.device).type == "cuda",
        "persistent_workers": config.num_workers > 0,
        "collate_fn": partial(_collate_encoded, padding_value=padding_value),
    }
    if config.max_tokens_per_batch is not None:
        if config.max_tokens_per_batch < 1:
            raise ValueError("max_tokens_per_batch must be positive when supplied")
        ordered = sorted(
            range(len(indices)), key=lambda position: token_ids[indices[position]].numel()
        )
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for position in ordered:
            length = token_ids[indices[position]].numel()
            next_max = max(current_max, length)
            if current and (
                len(current) >= config.batch_size
                or next_max * (len(current) + 1) > config.max_tokens_per_batch
            ):
                batches.append(current)
                current = []
                current_max = 0
            current.append(position)
            current_max = max(current_max, length)
        if current:
            batches.append(current)
        if shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return DataLoader(batch_sampler=batches, **common)
    return DataLoader(
        common.pop("dataset"),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        **common,
    )


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scheduled_learning_rate(config: BenchmarkConfig, training_step: int) -> float:
    if config.warmup_steps > 0 and training_step <= config.warmup_steps:
        return config.learning_rate * training_step / config.warmup_steps
    if config.training_steps is None or config.min_learning_rate is None:
        return config.learning_rate
    decay_steps = max(1, config.training_steps - config.warmup_steps)
    progress = min(1.0, (training_step - config.warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def _train_epoch(
    model: LyraRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    squared_error = 0.0
    n_samples = 0
    for token_ids, padding_mask, targets, _ in loader:
        loss, batch_size = _train_batch(
            model,
            token_ids,
            padding_mask,
            targets,
            optimizer,
            device,
            gradient_clip,
        )
        squared_error += loss * batch_size
        n_samples += batch_size
    return squared_error / n_samples


def _train_batch(
    model: LyraRegressor,
    token_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, int]:
    model.train()
    token_ids = token_ids.to(device, non_blocking=True).long()
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    predictions = model(
        token_ids,
        padding_mask=padding_mask.to(device, non_blocking=True),
    )
    loss = F.mse_loss(predictions, targets)
    loss.backward()
    if gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    return float(loss.detach()), targets.numel()


@torch.inference_mode()
def _evaluate(
    model: LyraRegressor, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, tuple[AssayRecord, ...]]:
    model.eval()
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_records: list[AssayRecord] = []
    for token_ids, padding_mask, targets, records in loader:
        predictions = model(
            token_ids.to(device, non_blocking=True).long(),
            padding_mask=padding_mask.to(device, non_blocking=True),
        ).cpu()
        all_targets.append(targets)
        all_predictions.append(predictions)
        all_records.extend(records)
    target_tensor = torch.cat(all_targets)
    prediction_tensor = torch.cat(all_predictions)
    return (
        regression_metrics(target_tensor, prediction_tensor),
        target_tensor,
        prediction_tensor,
        tuple(all_records),
    )


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    finite = torch.tensor(
        [value for value in values if math.isfinite(value)], dtype=torch.float64
    )
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std(unbiased=False))


def _summarize_task(
    assay: Assay,
    fold_rows: Sequence[dict[str, object]],
    pooled_metrics: dict[str, float],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "task": assay.name,
        "source_file": str(assay.source),
        "n_samples": len(assay.records),
        "sequence_length": assay.sequence_length,
        "n_folds": len(fold_rows),
    }
    for metric in METRIC_NAMES:
        mean, std = _mean_std(float(row[metric]) for row in fold_rows)
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"pooled_{metric}"] = pooled_metrics[metric]
    return summary


def _write_aggregate(path: Path, task_rows: Sequence[dict[str, object]]) -> None:
    row: dict[str, object] = {
        "n_tasks": len(task_rows),
        "total_samples": sum(int(task["n_samples"]) for task in task_rows),
    }
    fields = ["n_tasks", "total_samples"]
    for metric in METRIC_NAMES:
        field = f"{metric}_mean"
        mean, std = _mean_std(float(task[field]) for task in task_rows)
        row[f"macro_{metric}_mean"] = mean
        row[f"macro_{metric}_std"] = std
        fields.extend((f"macro_{metric}_mean", f"macro_{metric}_std"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, object]]:
    """Train and evaluate Lyra independently on every discovered assay."""

    if config.n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.training_steps is not None and config.training_steps < 1:
        raise ValueError("training_steps must be positive when supplied")
    if config.eval_every_steps < 1:
        raise ValueError("eval_every_steps must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if config.training_steps is not None and config.warmup_steps > config.training_steps:
        raise ValueError("warmup_steps cannot exceed training_steps")
    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.resume:
        raise FileExistsError(
            f"output directory is not empty: {config.output_dir}; choose a new "
            "directory or set resume: true"
        )
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    alphabet = ProteinAlphabet(config.alphabet)
    parameter_probe = _build_model(config, alphabet.size)
    counts = parameter_counts(parameter_probe)
    del parameter_probe

    config.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = asdict(config)
    serializable["input_path"] = str(config.input_path)
    serializable["output_dir"] = str(config.output_dir)
    serializable["total_parameters"] = counts["total_parameters"]
    serializable["trainable_parameters"] = counts["trainable_parameters"]
    if config.model_type == "hyena":
        serializable.update(
            {
                key: value
                for key, value in counts.items()
                if key not in {"total_parameters", "trainable_parameters"}
            }
        )
        serializable["parameter_count_policy"] = (
            "exclude_hyena_positional_embeddings"
        )
        serializable["hyena_effective_order"] = (
            config.hyena_order
            if config.hyena_order is not None
            else (2 if config.d_model <= 64 else 4)
        )
    serializable["split_source"] = (
        f"column:{config.fold_column}"
        if config.fold_column is not None
        else "deterministic_balanced_random"
    )
    run_config_path = config.output_dir / "run_config.json"
    if run_config_path.exists():
        with run_config_path.open(encoding="utf-8") as handle:
            previous_config = json.load(handle)
        comparable_previous = dict(previous_config)
        comparable_previous["resume"] = config.resume
        if comparable_previous != serializable:
            raise ValueError(
                f"cannot resume {config.output_dir}: run_config.json does not "
                "match the requested configuration"
            )
    else:
        with run_config_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)

    assay_paths = discover_assays(config.input_path, config.pattern)
    assay_names = [path.stem for path in assay_paths]
    if len(assay_names) != len(set(assay_names)):
        raise ValueError(
            "assay filenames must have unique stems so CSV output paths do not collide"
        )
    LOGGER.info("Found %d assays; training on %s", len(assay_paths), device)

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

    task_rows: list[dict[str, object]] = []
    for task_number, path in enumerate(assay_paths, start=1):
        if path.stem in existing_task_rows:
            LOGGER.info(
                "[%d/%d] %s: already complete; skipping",
                task_number,
                len(assay_paths),
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
        LOGGER.info(
            "[%d/%d] %s: %d sequences of length %d",
            task_number,
            len(assay_paths),
            assay.name,
            len(assay.records),
            assay.sequence_length,
        )
        assay_token_ids = tuple(
            alphabet.encode_token_ids([record.sequence]).squeeze(0)
            for record in assay.records
        )
        padding_value = (
            alphabet.symbols.index("-")
            if "-" in alphabet.symbols
            else alphabet.unknown_index
        )
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
            fold_start = time.perf_counter()
            fold_seed = config.seed + fold + zlib.crc32(assay.name.encode())
            _seed_everything(fold_seed)
            train_indices = [index for index, value in enumerate(assay.folds) if value != fold]
            test_indices = [index for index, value in enumerate(assay.folds) if value == fold]
            raw_train_targets = torch.tensor(
                [assay.records[index].target for index in train_indices],
                dtype=torch.float64,
            )
            if config.normalize_targets:
                target_mean = float(raw_train_targets.mean())
                target_std = float(raw_train_targets.std(unbiased=False))
                if target_std <= 0:
                    raise ValueError(
                        f"{assay.name} fold {fold}: training targets have zero variance"
                    )
            else:
                target_mean = 0.0
                target_std = 1.0
            normalized_targets = torch.tensor(
                [
                    (record.target - target_mean) / target_std
                    for record in assay.records
                ],
                dtype=torch.float32,
            )
            train_loader = _loader(
                assay,
                train_indices,
                assay_token_ids,
                normalized_targets,
                config,
                padding_value,
                shuffle=True,
                seed=fold_seed,
            )
            test_loader = _loader(
                assay,
                test_indices,
                assay_token_ids,
                normalized_targets,
                config,
                padding_value,
                shuffle=False,
                seed=fold_seed,
            )

            model = _build_model(config, alphabet.size).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            final_metrics: dict[str, float] | None = None
            final_targets: torch.Tensor | None = None
            final_predictions: torch.Tensor | None = None
            final_records: tuple[AssayRecord, ...] = ()
            if config.training_steps is None:
                training_step = 0
                for epoch in range(1, config.epochs + 1):
                    train_mse = _train_epoch(
                        model, train_loader, optimizer, device, config.gradient_clip
                    )
                    training_step += len(train_loader)
                    final_metrics, final_targets, final_predictions, final_records = (
                        _evaluate(model, test_loader, device)
                    )
                    epoch_row: dict[str, object] = {
                        "task": assay.name,
                        "fold": fold,
                        "epoch": epoch,
                        "training_step": training_step,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "train_mse": train_mse,
                        **final_metrics,
                        "elapsed_seconds": time.perf_counter() - fold_start,
                    }
                    _append_row(
                        config.output_dir / "epoch_metrics.csv", EPOCH_FIELDS, epoch_row
                    )
                    LOGGER.info(
                        "%s fold=%d epoch=%d step=%d train_mse=%.5g "
                        "test_mse=%.5g spearman=%.4f",
                        assay.name,
                        fold,
                        epoch,
                        training_step,
                        train_mse,
                        final_metrics["mse"],
                        final_metrics["spearman"],
                    )
            else:
                train_iterator = iter(train_loader)
                interval_squared_error = 0.0
                interval_samples = 0
                for training_step in range(1, config.training_steps + 1):
                    learning_rate = _scheduled_learning_rate(config, training_step)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate
                    try:
                        token_ids, padding_mask, targets, _ = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        token_ids, padding_mask, targets, _ = next(train_iterator)
                    loss, batch_samples = _train_batch(
                        model,
                        token_ids,
                        padding_mask,
                        targets,
                        optimizer,
                        device,
                        config.gradient_clip,
                    )
                    interval_squared_error += loss * batch_samples
                    interval_samples += batch_samples
                    should_evaluate = (
                        training_step % config.eval_every_steps == 0
                        or training_step == config.training_steps
                    )
                    if not should_evaluate:
                        continue
                    train_mse = interval_squared_error / interval_samples
                    interval_squared_error = 0.0
                    interval_samples = 0
                    final_metrics, final_targets, final_predictions, final_records = (
                        _evaluate(model, test_loader, device)
                    )
                    epoch = math.ceil(training_step / len(train_loader))
                    epoch_row = {
                        "task": assay.name,
                        "fold": fold,
                        "epoch": epoch,
                        "training_step": training_step,
                        "n_train": len(train_indices),
                        "n_test": len(test_indices),
                        "train_mse": train_mse,
                        **final_metrics,
                        "elapsed_seconds": time.perf_counter() - fold_start,
                    }
                    _append_row(
                        config.output_dir / "epoch_metrics.csv", EPOCH_FIELDS, epoch_row
                    )
                    LOGGER.info(
                        "%s fold=%d step=%d train_mse=%.5g test_mse=%.5g "
                        "spearman=%.4f",
                        assay.name,
                        fold,
                        training_step,
                        train_mse,
                        final_metrics["mse"],
                        final_metrics["spearman"],
                    )

            assert (
                final_metrics is not None
                and final_targets is not None
                and final_predictions is not None
            )
            pooled_targets.append(final_targets)
            pooled_predictions.append(final_predictions)
            elapsed = time.perf_counter() - fold_start
            fold_row = {
                "task": assay.name,
                "fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                **final_metrics,
                "elapsed_seconds": elapsed,
            }
            fold_rows.append(fold_row)
            _append_row(config.output_dir / "fold_metrics.csv", FOLD_FIELDS, fold_row)
            _write_predictions(
                prediction_path,
                assay.name,
                fold,
                final_records,
                final_targets,
                final_predictions,
                target_mean,
                target_std,
            )
            if config.save_models:
                checkpoint_dir = config.output_dir / "models" / assay.name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / f"fold_{fold}.pt")

        pooled_metrics = regression_metrics(
            torch.cat(pooled_targets), torch.cat(pooled_predictions)
        )
        task_row = _summarize_task(assay, fold_rows, pooled_metrics)
        task_rows.append(task_row)
        _append_row(config.output_dir / "task_metrics.csv", TASK_FIELDS, task_row)

    _write_aggregate(config.output_dir / "aggregate_metrics.csv", task_rows)
    return task_rows


def _default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"lyra-proteingym-{timestamp}"


def load_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark configuration from YAML.

    Only ``input_path`` is required. Relative input and output paths are
    resolved relative to the YAML file so a run is independent of the current
    working directory.
    """

    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    allowed = {field.name for field in fields(BenchmarkConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown configuration keys: {unknown}")
    if "input_path" not in values:
        raise ValueError(f"{path}: input_path is required")

    base_dir = path.resolve().parent
    input_path = Path(values.pop("input_path"))
    output_value = values.pop("output_dir", None)
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    if output_value is None:
        output_dir = base_dir / _default_output_dir()
    else:
        output_dir = Path(output_value)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir

    return BenchmarkConfig(
        input_path=input_path,
        output_dir=output_dir,
        **values,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YAML-configured Lyra five-fold ProteinGym regression"
    )
    parser.add_argument("config", type=Path, help="YAML run configuration")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    run_benchmark(load_config(args.config))


if __name__ == "__main__":
    main()
