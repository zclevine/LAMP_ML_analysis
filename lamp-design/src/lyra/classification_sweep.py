"""Expand compact FLIP classification sweep YAML files."""

from __future__ import annotations

import argparse
import itertools
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Sequence

import yaml

from .classification import ClassificationConfig, run_classification
from .sweep import _run_name


LOGGER = logging.getLogger("lyra.classification_sweep")


def _worker(configs, device):
    for config in configs:
        run_classification(ClassificationConfig(**{**config.__dict__, "device": device}))


def run_sweep(path: Path, dry_run: bool = False):
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base, matrix = document.get("base", {}), document.get("matrix", {})
    runs = document.get("runs")
    configurable = {field.name for field in fields(ClassificationConfig)} - {"input_path", "output_dir", "device"}
    explicit_keys = set().union(*(set(run) - {"name"} for run in (runs or [])))
    unknown = sorted((set(base) | set(matrix) | explicit_keys) - configurable)
    if unknown:
        raise ValueError(f"unknown classification keys: {unknown}")
    root = path.resolve().parent
    input_path = Path(document["input_path"])
    output_dir = Path(document["output_dir"])
    if not input_path.is_absolute(): input_path = root / input_path
    if not output_dir.is_absolute(): output_dir = root / output_dir
    configs = []
    if runs is not None:
        if matrix:
            raise ValueError("classification sweep cannot define both matrix and runs")
        for run in runs:
            overrides = {key: value for key, value in run.items() if key != "name"}
            name = run.get("name") or _run_name(overrides)
            configs.append(ClassificationConfig(input_path=input_path, output_dir=output_dir / "runs" / name, **base, **overrides))
    else:
        keys = list(matrix)
        for values in itertools.product(*(matrix[key] for key in keys)):
            overrides = dict(zip(keys, values, strict=True))
            configs.append(ClassificationConfig(input_path=input_path, output_dir=output_dir / "runs" / _run_name(overrides), **base, **overrides))
    devices = tuple(document.get("devices", ["auto"]))
    LOGGER.info("Expanded %d runs across %d devices", len(configs), len(devices))
    if dry_run: return configs
    assignments = [configs[index::len(devices)] for index in range(len(devices))]
    if len(devices) == 1:
        _worker(assignments[0], devices[0])
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as executor:
            futures = [executor.submit(_worker, jobs, device) for jobs, device in zip(assignments, devices, strict=True) if jobs]
            for future in futures: future.result()
    return configs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_sweep(args.config, args.dry_run)


if __name__ == "__main__": main()
