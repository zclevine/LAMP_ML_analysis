"""Prepare and run faithful ProteinNPT cross-validation sweeps.

ProteinNPT is an assay-trained conditional model, not a generic frozen encoder.
This bridge preserves the official implementation while giving it the exact
deterministic balanced random folds used by the Lyra runners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from .proteingym import _assign_random_folds, discover_assays


LOGGER = logging.getLogger("lyra.proteinnpt_sweep")
FOLD_COLUMN = "lyra_random_5"


@dataclass(frozen=True)
class ProteinNPTSweepConfig:
    input_path: Path
    output_dir: Path
    reference_file: Path
    repository_path: Path
    data_path: Path
    devices: tuple[str, ...] = ("cuda:0",)
    pattern: str = "*.csv"
    sequence_column: str = "mutated_sequence"
    target_column: str = "DMS_score"
    n_folds: int = 5
    seed: int = 42
    learning_rates: tuple[float, ...] = (1e-3, 4e-4)
    min_learning_rate: float = 4e-5
    training_steps: int = 10000
    warmup_steps: int = 100
    embedding_batch_size: int = 1
    msa_sequences: int = 384
    python_executable: str = "python"
    execute: bool = False
    resume: bool = True
    training_fp16: bool = True
    cleanup_embeddings_after_training: bool = False


@dataclass(frozen=True)
class ProteinNPTAssayJob:
    task: str
    device: str
    n_folds: int
    prepared_assay: Path
    embedding_file: Path
    embedding_command: tuple[str, ...]
    training_commands: tuple[tuple[str, ...], ...]
    learning_rates: tuple[float, ...]
    result_files: tuple[Path, ...]
    log_file: Path


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_proteinnpt_sweep(path: Path) -> ProteinNPTSweepConfig:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    fields = set(ProteinNPTSweepConfig.__dataclass_fields__)
    unknown = sorted(set(document) - fields)
    if unknown:
        raise ValueError(f"{path}: unknown keys: {unknown}")
    required = {
        "input_path",
        "output_dir",
        "reference_file",
        "repository_path",
        "data_path",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing}")
    base = path.resolve().parent
    for field in (
        "input_path",
        "output_dir",
        "reference_file",
        "repository_path",
        "data_path",
    ):
        document[field] = _resolve_path(document[field], base)
    document["devices"] = tuple(str(value) for value in document.get("devices", ["cuda:0"]))
    document["learning_rates"] = tuple(
        float(value) for value in document.get("learning_rates", [1e-3, 4e-4])
    )
    if not document["devices"] or not document["learning_rates"]:
        raise ValueError("devices and learning_rates must be non-empty")
    return ProteinNPTSweepConfig(**document)


def _read_reference(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "DMS_id",
        "target_seq",
        "MSA_filename",
        "MSA_start",
        "MSA_end",
        "weight_file_name",
    }
    fields = set(rows[0]) if rows else set()
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}: missing ProteinNPT reference columns: {missing}")
    for index, row in enumerate(rows):
        row["_lyra_reference_index"] = str(index)
    mapping = {row["DMS_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{path}: duplicate DMS_id values")
    return mapping


def _prepare_assay(
    source: Path,
    destination: Path,
    *,
    n_folds: int,
    seed: int,
    sequence_column: str,
    target_column: str,
) -> tuple[int, str]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"{source}: empty assay")
    missing = {sequence_column, target_column} - set(fieldnames)
    if missing:
        raise ValueError(f"{source}: missing required columns: {sorted(missing)}")
    if FOLD_COLUMN not in fieldnames:
        fieldnames.append(FOLD_COLUMN)
    task_seed = seed + zlib.crc32(source.stem.encode())
    folds = _assign_random_folds(len(rows), n_folds, task_seed)
    for row, fold in zip(rows, folds, strict=True):
        row[FOLD_COLUMN] = str(fold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    fingerprint = hashlib.sha256(bytes(folds)).hexdigest()
    return len(rows), fingerprint


def _lr_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _build_job(
    config: ProteinNPTSweepConfig,
    source: Path,
    reference: dict[str, str],
    device: str,
) -> tuple[ProteinNPTAssayJob, dict[str, object]]:
    scripts = config.repository_path / "scripts"
    prepared = config.output_dir / "prepared_assays" / source.name
    n_samples, fingerprint = _prepare_assay(
        source,
        prepared,
        n_folds=config.n_folds,
        seed=config.seed,
        sequence_column=config.sequence_column,
        target_column=config.target_column,
    )
    embedding_root = config.output_dir / "embeddings" / "MSA_Transformer"
    embedding_file = embedding_root / f"{source.stem}.h5"
    msa_root = config.data_path / "data" / "MSA" / "MSA_files"
    weight_root = config.data_path / "data" / "MSA" / "MSA_weights"
    msa_file = msa_root / reference["MSA_filename"]
    checkpoint = (
        config.data_path
        / "ESM"
        / "MSA_Transformer"
        / "esm_msa1b_t12_100M_UR50S.pt"
    )
    embedding_command = [
        config.python_executable,
        str(scripts / "embeddings.py"),
        "--assay_reference_file_location",
        str(config.reference_file),
        "--assay_index",
        reference["_lyra_reference_index"],
        "--input_data_location",
        str(prepared.parent),
        "--output_data_location",
        str(embedding_root),
        "--model_type",
        "MSA_Transformer",
        "--model_location",
        str(checkpoint),
        "--max_positions",
        "1024",
        "--batch_size",
        str(config.embedding_batch_size),
        "--half_precision",
        "--num_MSA_sequences",
        str(config.msa_sequences),
        "--MSA_data_folder",
        str(msa_root),
        "--MSA_weight_data_folder",
        str(weight_root),
        "--path_to_hhfilter",
        str(config.data_path / "utils" / "hhfilter"),
    ]

    training_commands: list[tuple[str, ...]] = []
    result_files: list[Path] = []
    for learning_rate in config.learning_rates:
        slug = _lr_slug(learning_rate)
        run_name = f"lyra-{source.stem}-lr{slug}"
        metrics_dir = config.output_dir / "metrics" / run_name
        result_file = metrics_dir / f"test_performance_overall_{run_name}.csv"
        command = [
            config.python_executable,
            str(scripts / "train.py"),
            "--data_location",
            str(config.data_path),
            "--assay_data_location",
            str(prepared),
            "--model_config_location",
            str(config.repository_path / "proteinnpt" / "proteinnpt" / "model_configs" / "PNPT_final.json"),
            "--fold_variable_name",
            FOLD_COLUMN,
            "--target_config_location",
            str(config.repository_path / "proteinnpt" / "utils" / "target_configs" / "fitness.json"),
            "--zero_shot_fitness_predictions_location",
            str(config.data_path / "data" / "zero_shot_fitness_predictions" / "substitutions"),
            "--sequence_embeddings_folder",
            str(embedding_root),
            "--MSA_data_folder",
            str(msa_root),
            "--MSA_location",
            str(msa_file),
            "--MSA_weight_data_folder",
            str(weight_root),
            "--MSA_sequence_weights_filename",
            reference["weight_file_name"],
            "--target_seq",
            reference["target_seq"],
            "--test_fold_index",
            "-1",
            "--augmentation",
            "zero_shot_fitness_predictions_auxiliary_labels",
            "--model_name_suffix",
            run_name,
            "--output_results_location",
            str(metrics_dir),
            "--num_total_training_steps",
            str(config.training_steps),
            "--num_warmup_steps",
            str(config.warmup_steps),
            "--max_learning_rate",
            str(learning_rate),
            "--min_learning_rate",
            str(config.min_learning_rate),
            "--seed",
            str(config.seed),
            "--do_not_save_model_checkpoint",
        ]
        if config.training_fp16:
            command.append("--training_fp16")
        if reference["MSA_start"]:
            command.extend(("--MSA_start", reference["MSA_start"]))
        if reference["MSA_end"]:
            command.extend(("--MSA_end", reference["MSA_end"]))
        training_commands.append(tuple(command))
        result_files.append(result_file)
    job = ProteinNPTAssayJob(
        task=source.stem,
        device=device,
        n_folds=config.n_folds,
        prepared_assay=prepared,
        embedding_file=embedding_file,
        embedding_command=tuple(embedding_command),
        training_commands=tuple(training_commands),
        learning_rates=config.learning_rates,
        result_files=tuple(result_files),
        log_file=config.output_dir / "logs" / f"{source.stem}.log",
    )
    manifest = {
        "task": source.stem,
        "source_file": str(source),
        "prepared_assay": str(prepared),
        "n_samples": n_samples,
        "fold_column": FOLD_COLUMN,
        "split_seed": config.seed + zlib.crc32(source.stem.encode()),
        "assignment_sha256": fingerprint,
        "device": device,
        "embedding_file": str(embedding_file),
        "learning_rates": json.dumps(config.learning_rates),
        "embedding_command": json.dumps(embedding_command),
        "training_commands": json.dumps(training_commands),
    }
    return job, manifest


def _run_job(
    job: ProteinNPTAssayJob,
    repository_path: Path,
    resume: bool,
    cleanup_embeddings_after_training: bool,
) -> str:
    env = os.environ.copy()
    # PyTorch 2.6 changed torch.load's default to weights_only=True. The
    # official ESM checkpoint contains argparse.Namespace metadata and is
    # loaded by ProteinNPT without an explicit flag. Restrict this compatibility
    # override to the official subprocess rather than patching global behavior.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if job.device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = job.device.split(":", 1)[1]
    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    with job.log_file.open("a", encoding="utf-8") as log:
        if not (resume and job.embedding_file.exists()):
            subprocess.run(
                job.embedding_command,
                cwd=repository_path / "scripts",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        for command, result_file in zip(
            job.training_commands, job.result_files, strict=True
        ):
            if resume and result_file.exists():
                continue
            result_file.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                command,
                cwd=repository_path / "scripts",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    if (
        cleanup_embeddings_after_training
        and job.embedding_file.exists()
        and all(path.exists() for path in job.result_files)
    ):
        job.embedding_file.unlink()
    return job.task


def _run_job_chunk(
    jobs: Sequence[ProteinNPTAssayJob],
    repository_path: Path,
    resume: bool,
    cleanup_embeddings_after_training: bool,
) -> tuple[str, ...]:
    return tuple(
        _run_job(
            job,
            repository_path,
            resume,
            cleanup_embeddings_after_training,
        )
        for job in jobs
    )


def _write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_proteinnpt_metrics(
    jobs: Sequence[ProteinNPTAssayJob], output_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Normalize official ProteinNPT outputs into the shared benchmark schema."""

    task_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for job in jobs:
        for learning_rate, result_file in zip(
            job.learning_rates, job.result_files, strict=True
        ):
            if not result_file.exists():
                continue
            with result_file.open(newline="", encoding="utf-8-sig") as handle:
                overall_rows = list(csv.DictReader(handle))
            if len(overall_rows) != 1:
                raise ValueError(f"{result_file}: expected exactly one overall row")
            row = overall_rows[0]
            task_rows.append(
                {
                    "task": job.task,
                    "model": "ProteinNPT",
                    "learning_rate": learning_rate,
                    "n_folds": job.n_folds,
                    "mse": row["loss_fitness"],
                    "r2_mean": row["mean_R2_fitness"],
                    "r2_std": row["std_dev_R2_fitness"],
                    "pooled_r2": row["overall_R2_fitness"],
                    "spearman_mean": row["mean_Spearman_fitness"],
                    "spearman_std": row["std_dev_Spearman_fitness"],
                    "pooled_spearman": row["overall_Spearman_fitness"],
                    "n_samples": row["num_obs_fitness"],
                    "split_source": FOLD_COLUMN,
                }
            )
            run_name = result_file.stem.removeprefix("test_performance_overall_")
            fold_file = result_file.parent / f"test_performance_by_fold_{run_name}.csv"
            if not fold_file.exists():
                raise FileNotFoundError(fold_file)
            with fold_file.open(newline="", encoding="utf-8-sig") as handle:
                for fold in csv.DictReader(handle):
                    fold_rows.append(
                        {
                            "task": job.task,
                            "model": "ProteinNPT",
                            "learning_rate": learning_rate,
                            "fold": fold["fold_index"],
                            "mse": fold["loss_fitness"],
                            "r2": fold["R2_fitness"],
                            "spearman": fold["Spearman_fitness"],
                            "n_samples": fold["num_obs_Spearman_fitness"],
                            "split_source": FOLD_COLUMN,
                        }
                    )
    if task_rows:
        _write_manifest(output_dir / "task_metrics.csv", task_rows)
    if fold_rows:
        _write_manifest(output_dir / "fold_metrics.csv", fold_rows)
    return task_rows, fold_rows


def run_proteinnpt_sweep(config: ProteinNPTSweepConfig) -> list[ProteinNPTAssayJob]:
    reference = _read_reference(config.reference_file)
    sources = discover_assays(config.input_path, config.pattern)
    missing = sorted(source.stem for source in sources if source.stem not in reference)
    if missing:
        raise ValueError(f"assays missing from ProteinNPT reference: {missing}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[ProteinNPTAssayJob] = []
    manifests: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        job, manifest = _build_job(
            config,
            source,
            reference[source.stem],
            config.devices[index % len(config.devices)],
        )
        jobs.append(job)
        manifests.append(manifest)
    _write_manifest(config.output_dir / "job_manifest.csv", manifests)
    if not config.execute:
        LOGGER.info("Prepared %d ProteinNPT jobs; execute=false", len(jobs))
        return jobs

    for required in (
        config.repository_path / "scripts" / "embeddings.py",
        config.repository_path / "scripts" / "train.py",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    chunks = [jobs[index :: len(config.devices)] for index in range(len(config.devices))]
    with ProcessPoolExecutor(max_workers=len(config.devices)) as executor:
        futures = [
            executor.submit(
                _run_job_chunk,
                chunk,
                config.repository_path,
                config.resume,
                config.cleanup_embeddings_after_training,
            )
            for chunk in chunks
            if chunk
        ]
        for future in futures:
            LOGGER.info("Completed ProteinNPT assays %s", ", ".join(future.result()))
    collect_proteinnpt_metrics(jobs, config.output_dir)
    return jobs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_proteinnpt_sweep(load_proteinnpt_sweep(args.config))


if __name__ == "__main__":
    main()
