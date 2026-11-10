"""Extract frozen embeddings and run direct-linear classification heads."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from .extract_embeddings import EmbeddingConfig, extract_embeddings
from .frozen_classification import FrozenClassificationConfig, run_frozen_classification
from .pretrained_sweep import PretrainedModelConfig, PretrainedSweepConfig, _lr_slug, load_pretrained_sweep


LOGGER = logging.getLogger("lyra.pretrained_classification_sweep")


def _run_models(config: PretrainedSweepConfig, models: Sequence[PretrainedModelConfig], device: str) -> None:
    for model in models:
        feature_dir = config.output_dir / "features" / model.name
        extract_embeddings(EmbeddingConfig(input_path=config.input_path, output_dir=feature_dir, checkpoint=model.checkpoint, backend=model.backend, sequence_column=config.sequence_column, pattern=config.pattern, batch_size=model.batch_size, max_residues=model.max_residues, device=device, dtype=model.dtype, resume=True, layer_name=model.layer_name, trust_remote_code=model.trust_remote_code))
        for learning_rate in config.learning_rates:
            output_dir = config.output_dir / "probes" / model.name / f"lr-{_lr_slug(learning_rate)}"
            if (output_dir / "aggregate_metrics.json").exists():
                LOGGER.info("%s lr=%g complete", model.name, learning_rate); continue
            run_frozen_classification(FrozenClassificationConfig(input_path=config.input_path, feature_dir=feature_dir, output_dir=output_dir, backbone=model.name, sequence_column=config.sequence_column, target_column=config.target_column, fold_column=config.fold_column, pattern=config.pattern, n_folds=config.n_folds, seed=config.seed, training_steps=config.training_steps, eval_every_steps=config.eval_every_steps, batch_size=config.probe_batch_size, learning_rate=learning_rate, min_learning_rate=config.min_learning_rate, warmup_steps=config.warmup_steps, weight_decay=config.weight_decay, device=device))


def run(config: PretrainedSweepConfig) -> None:
    assignments = [config.models[index::len(config.devices)] for index in range(len(config.devices))]
    if len(config.devices) == 1: _run_models(config, assignments[0], config.devices[0]); return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(config.devices), mp_context=context) as executor:
        futures = [executor.submit(_run_models, config, models, device) for models, device in zip(assignments, config.devices, strict=True) if models]
        for future in futures: future.result()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("config", type=Path); args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_pretrained_sweep(args.config))


if __name__ == "__main__": main()
