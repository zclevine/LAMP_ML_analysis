"""Expand and execute compact YAML benchmark sweeps."""

from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from .proteingym import BenchmarkConfig, run_benchmark


LOGGER = logging.getLogger("lyra.sweep")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _slug(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value).replace("/", "-").replace(" ", "-")


def _run_name(values: Mapping[str, object]) -> str:
    preferred = ("model_type", "d_model", "d_state", "learning_rate")
    labels = {
        "model_type": "",
        "d_model": "d",
        "d_state": "s",
        "learning_rate": "lr",
    }
    parts = []
    for key in preferred:
        if key in values:
            parts.append(f"{labels[key]}{_slug(values[key])}")
    for key in sorted(set(values) - set(preferred)):
        parts.append(f"{key}-{_slug(values[key])}")
    return "-".join(parts)


def expand_sweep(path: Path) -> tuple[list[BenchmarkConfig], tuple[str, ...]]:
    """Load a compact sweep YAML and expand its Cartesian parameter grid."""

    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    allowed_top_level = {"input_path", "output_dir", "devices", "base", "matrix"}
    unknown_top_level = sorted(set(document) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"{path}: unknown sweep keys: {unknown_top_level}")
    if "input_path" not in document:
        raise ValueError(f"{path}: input_path is required")

    base = document.get("base", {})
    matrix = document.get("matrix", {})
    if not isinstance(base, dict) or not isinstance(matrix, dict):
        raise ValueError(f"{path}: base and matrix must be YAML mappings")
    if set(base) & set(matrix):
        overlap = sorted(set(base) & set(matrix))
        raise ValueError(f"{path}: keys cannot appear in both base and matrix: {overlap}")

    configurable = {field.name for field in fields(BenchmarkConfig)} - {
        "input_path",
        "output_dir",
        "device",
    }
    unknown_parameters = sorted((set(base) | set(matrix)) - configurable)
    if unknown_parameters:
        raise ValueError(f"{path}: unknown benchmark keys: {unknown_parameters}")
    for key, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path}: matrix.{key} must be a non-empty list")

    base_dir = path.resolve().parent
    input_path = _resolve_path(document["input_path"], base_dir)
    output_root = _resolve_path(document.get("output_dir", "../results/sweep"), base_dir)
    devices_value = document.get("devices", ["auto"])
    if not isinstance(devices_value, list) or not devices_value:
        raise ValueError(f"{path}: devices must be a non-empty list")
    devices = tuple(str(device) for device in devices_value)

    keys = list(matrix)
    combinations = itertools.product(*(matrix[key] for key in keys))
    configs: list[BenchmarkConfig] = []
    seen_names: set[str] = set()
    for combination in combinations:
        overrides = dict(zip(keys, combination, strict=True))
        values = {**base, **overrides}
        name = _run_name(overrides)
        if name in seen_names:
            raise ValueError(f"{path}: parameter matrix produced duplicate run name {name}")
        seen_names.add(name)
        configs.append(
            BenchmarkConfig(
                input_path=input_path,
                output_dir=output_root / "runs" / name,
                **values,
            )
        )
    if not configs:
        raise ValueError(f"{path}: matrix did not produce any runs")
    return configs, devices


def _write_expanded_config(path: Path, config: BenchmarkConfig) -> None:
    values = asdict(config)
    values["input_path"] = str(config.input_path)
    values["output_dir"] = str(config.output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)


def _run_worker(jobs: Sequence[BenchmarkConfig], device: str) -> None:
    for config in jobs:
        run_benchmark(BenchmarkConfig(**{**config.__dict__, "device": device}))


def run_sweep(path: Path, *, dry_run: bool = False) -> list[BenchmarkConfig]:
    configs, devices = expand_sweep(path)
    output_root = configs[0].output_dir.parent.parent
    for config in configs:
        _write_expanded_config(
            output_root / "expanded_configs" / f"{config.output_dir.name}.yaml",
            config,
        )
    LOGGER.info("Expanded %d runs across %d device(s)", len(configs), len(devices))
    if dry_run:
        return configs

    assignments = [configs[index :: len(devices)] for index in range(len(devices))]
    if len(devices) == 1:
        _run_worker(assignments[0], devices[0])
        return configs

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
        futures = [
            executor.submit(_run_worker, jobs, device)
            for jobs, device in zip(assignments, devices, strict=True)
            if jobs
        ]
        for future in futures:
            future.result()
    return configs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact YAML Lyra sweep")
    parser.add_argument("config", type=Path, help="YAML sweep configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="expand and validate configs without training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    run_sweep(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
