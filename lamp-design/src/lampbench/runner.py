from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import paper_summary, select_finalists
from .data import TABULAR_FEATURE_GROUPS, load_units, split_confirmatory, write_manifest
from .external import (_internal_completion_hashes, evaluate_confirmatory,
                       evaluate_external)
from .finalize import finalize_models
from .metrics import selection_value
from .tabular import tabular_fold
from .train import TrainConfig, nested_fold

TRAIN_FILES = ["LMLv1.csv", "LMLv2.csv", "Systematic.csv"]
EXTERNAL_FILES = ["mpox.csv", "mpox_validation.csv", "Borrelia.csv", "LASV.csv"]


def require_improvement_gate(results_root: Path) -> dict:
    """Prevent frozen evaluation before the added model-development round is locked."""
    path = results_root / "improvement_gate" / "selection.json"
    if not path.exists():
        raise SystemExit(
            "Evidence-driven Lyra improvement gate is not locked; complete internal "
            "complementarity diagnostics and candidate screening before finalization")
    payload = json.loads(path.read_text())
    required = {"status", "frozen_data_status", "evaluated_candidates",
                "reference_models", "decision"}
    if not required.issubset(payload):
        raise SystemExit(f"Malformed improvement gate: missing {sorted(required - set(payload))}")
    if payload["status"] != "locked" or payload["frozen_data_status"] != "unopened":
        raise SystemExit("Improvement gate must be locked while frozen data are unopened")
    if not payload["evaluated_candidates"] or not payload["reference_models"]:
        raise SystemExit("Improvement gate does not document any evaluated candidate")
    return payload
NEURAL_MODELS = [
    "lyra_original", "lyra_original_tabular", "lyra_corrected",
    "lyra_corrected_crossgate", "lyra_corrected_tabdeep", "lyra_corrected_film",
    "lyra_paper_mean",
    "lyra_paper_terminal", "lyra_paper_resfix", "lyra_paper_resfix_bidir",
    "lyra_seqonly", "lyra_no_role",
    "lyra_unidirectional", "lyra_deep", "lyra_multiscale", "tab_mlp",
    "lyra_attnpool", "lyra_crossgate", "s4d", "cnn", "bigru", "transformer",
    "mamba1", "mamba2", "mamba2_terminal", "mamba2_terminal_bidir",
    "mamba3_siso", "mamba3_terminal", "mamba3_terminal_bidir",
]
# The pinned upstream Mamba-3 MIMO TileLang backward kernel fails to compile on
# both RTX 4090 and H100 hardware. Keep the constructor available for explicit
# compatibility probes, but do not include a non-trainable model in the paper
# matrix. The trainable Mamba-3 SISO, terminal, and bidirectional variants remain.
EXPERIMENTAL_MODELS = ["mamba3_mimo"]
IMPROVEMENT_MODELS = [
    "lyra_physics_tokens", "lyra_physics_crossgate", "lyra_physics_nolate",
]
IMPROVEMENT_REFERENCES = [
    "lyra_corrected", "lyra_corrected_nolate", "lyra_corrected_film",
    "lyra_corrected_crossgate",
]
TABULAR_MODELS = [
    "sameed_rf", "ridge", "elastic_net", "random_forest", "extra_trees",
    "hist_gradient_boosting", "gradient_boosting",
    "catboost", "lightgbm", "xgboost",
]
TP120_EXPERIMENTAL_TABULAR_MODELS = [
    "extra_trees_rank", "hist_gradient_boosting_rank", "catboost_rank",
    "lightgbm_rank", "xgboost_rank", "tabpfn", "tabpfn_rank",
    "tabpfn_v3", "tabpfn_v3_rank",
    "catboost_tp120_w025", "catboost_tp120_w050", "catboost_tp120_w075",
    "catboost_tp120_w150", "catboost_tp60_w025", "catboost_source_balanced",
]
SPECIAL_TABULAR_MODELS = ["catboost_historical"]


def ablation_experiment_models(include_physics_tokens: bool = True) -> dict[str, list[str]]:
    """Return the declared ablation inventory in execution order.

    The names are also stable shard keys: separate workers may execute disjoint
    subsets with ``--experiments`` and merge their complete experiment folders.
    """
    experiments = {
        "lmlv1": ["lyra_corrected"],
        "lmlv1_lmlv2": ["lyra_corrected"],
        "all_sources": ["lyra_corrected"],
        "chem_BST": ["lyra_corrected"],
        "chem_SS": ["lyra_corrected"],
    }
    for model in (
        "lyra_original", "lyra_original_tabular", "lyra_seqonly", "lyra_no_role",
        "lyra_unidirectional", "lyra_corrected", "lyra_corrected_nolate", "tab_mlp",
    ):
        experiments[f"architecture_{model}"] = [model]
    for model in (
        "lyra_corrected", "lyra_corrected_crossgate",
        "lyra_corrected_tabdeep", "lyra_corrected_film",
    ):
        experiments[f"fusion_{model}"] = [model]
    experiments["strict_nucleotide_only"] = ["lyra_corrected_nolate"]
    for name in ("no_context", "no_position", "no_thermo"):
        experiments[f"features_{name}"] = ["lyra_corrected"]
        experiments[f"tabular_features_{name}"] = ["hist_gradient_boosting"]
    for group in TABULAR_FEATURE_GROUPS:
        experiments[f"catboost_drop_{group}"] = ["catboost"]
    if include_physics_tokens:
        for group in (
            "mismatch_count", "mismatch_position", "gc_fraction",
            "melting_temperature", "vienna_dg",
        ):
            experiments[f"physics_tokens_drop_{group}"] = ["lyra_physics_tokens"]
    experiments["historical_thermo"] = ["catboost", "catboost_historical"]
    for width in (32, 64, 128):
        experiments[f"width_{width}"] = ["lyra_corrected"]
    for loss in ("mse", "huber", "rank_huber"):
        experiments[f"loss_{loss}"] = ["lyra_corrected"]
    return experiments


def result_exists(root, task, model, seed, block):
    return (root / task / model / f"seed_{seed}" / block / "result.json").exists()


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def run_matrix(units, output, stage, models, tasks, blocks, seeds, max_epochs,
               patience, d_model, batch_size, sources=None, chemistry=None,
               loss="huber", ablation="none", config_overrides=None,
               train_fraction=1.0, sampling_seed=0, train_chemistry=None):
    selected = units.copy()
    if sources:
        selected = selected[selected.source.isin(sources)].copy()
    if chemistry:
        selected = selected[selected.canonical_mm.eq(chemistry)].copy()
    log = output / "events.jsonl"
    failures = []
    for task in tasks:
        for model in models:
            for seed in seeds:
                for block in blocks:
                    if result_exists(output, task, model, seed, block):
                        append_event(log, {"status": "skip", "stage": stage, "task": task,
                                           "model": model, "seed": seed, "block": block})
                        continue
                    try:
                        append_event(log, {"status": "start", "stage": stage, "task": task,
                                           "model": model, "seed": seed, "block": block})
                        if model in (TABULAR_MODELS + SPECIAL_TABULAR_MODELS +
                                     TP120_EXPERIMENTAL_TABULAR_MODELS):
                            result = tabular_fold(
                                selected, block, model, task, seed, output, ablation,
                                train_fraction=train_fraction,
                                sampling_seed=sampling_seed,
                                train_chemistry=train_chemistry,
                            )
                        else:
                            override = (config_overrides or {}).get(model, {})
                            cfg = TrainConfig(model=model, task=task, seed=seed,
                                              max_epochs=override.get("max_epochs", max_epochs),
                                              patience=override.get("patience", patience),
                                              d_model=override.get("d_model", d_model),
                                              batch_size=override.get("batch_size", batch_size),
                                              dropout=override.get("dropout", 0.2),
                                              lr=override.get("lr", 1e-3),
                                              weight_decay=override.get("weight_decay", 1e-3),
                                              loss=override.get("loss", loss),
                                              ablation=override.get("ablation", ablation),
                                              train_fraction=train_fraction,
                                              sampling_seed=sampling_seed,
                                              train_chemistry=train_chemistry)
                            result = nested_fold(selected, block, cfg, output)
                        append_event(log, {"status": "complete", "stage": stage, "task": task,
                                           "model": model, "seed": seed, "block": block,
                                           "selection": selection_value(task, result["metrics"]["overall"])})
                    except Exception as exc:
                        failures.append((task, model, seed, block, repr(exc)))
                        append_event(log, {"status": "error", "stage": stage, "task": task,
                                           "model": model, "seed": seed, "block": block,
                                           "error": repr(exc), "traceback": traceback.format_exc()})
                        print(traceback.format_exc(), file=sys.stderr, flush=True)
    if failures:
        preview = "; ".join(
            f"{task}/{model}/seed_{seed}/{block}: {error}"
            for task, model, seed, block, error in failures[:10])
        raise RuntimeError(
            f"{len(failures)} declared folds failed in stage {stage}; "
            f"completed folds were checkpointed. {preview}")


def summarize(output: Path):
    rows = []
    for path in output.rglob("result.json"):
        r = json.loads(path.read_text())
        task = r.get("task", r.get("config", {}).get("task"))
        model = r.get("model", r.get("config", {}).get("model"))
        seed = r.get("seed", r.get("config", {}).get("seed"))
        metrics = r["metrics"]["overall"]
        relative = path.relative_to(output).parts
        experiment = relative[0] if len(relative) > 5 else "overall"
        rows.append({"experiment": experiment, "task": task, "model": model, "seed": seed,
                     "block": r["outer_block"], "selection": selection_value(task, metrics),
                     "elapsed_sec": r["elapsed_sec"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = frame.groupby(["experiment", "task", "model"], as_index=False).agg(
        mean=("selection", "mean"), std=("selection", "std"),
        min=("selection", "min"), max=("selection", "max"),
        n=("selection", "size"), elapsed_sec=("elapsed_sec", "sum"),
    ).sort_values(["experiment", "task", "mean"], ascending=[True, True, False])
    frame.to_csv(output / "all_fold_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(summary.to_json(orient="records", indent=2))
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["manifest", "smoke", "screen", "select", "tune", "select_tuning", "full", "tasks", "ablations", "improvements", "data_efficiency", "finalize", "confirmatory", "external", "analyze", "summarize"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--blocks", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--loss")
    parser.add_argument(
        "--ablation", default="none",
        help="Feature ablation applied consistently to fitting and evaluation")
    parser.add_argument(
        "--experiments", nargs="*",
        help="Named ablation experiments to run; omitted means the full inventory")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--expected-seeds", type=int)
    parser.add_argument("--train-fractions", nargs="*", type=float)
    parser.add_argument("--sampling-seed", type=int, default=20260821)
    parser.add_argument(
        "--train-chemistry",
        default=None,
        help="Train only on this canonical master mix; evaluate all mixes",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"finalize", "confirmatory", "external"}:
        require_improvement_gate(args.output.parent)
    # Fail before loading LMLv2 or hashing any external file if the declared
    # internal task/ablation/attribution artifacts are not yet complete.
    if args.stage in {"confirmatory", "external"}:
        _internal_completion_hashes(args.output.parent)
    if args.stage == "manifest":
        write_manifest(args.output / "manifest.json", args.data_dir, TRAIN_FILES, EXTERNAL_FILES)
        return
    if args.stage == "summarize":
        summarize(args.output)
        return
    if args.stage == "select":
        if not args.checkpoint_root:
            raise SystemExit("select requires --checkpoint-root pointing to screen results")
        select_finalists(args.checkpoint_root, args.output / "selection.json",
                         (args.tasks or ["positive"])[0])
        return
    if args.stage == "select_tuning":
        if not args.selection_file or not args.checkpoint_root:
            raise SystemExit("select_tuning requires --selection-file and --checkpoint-root")
        selection = json.loads(args.selection_file.read_text())
        candidates = []
        for path in args.checkpoint_root.glob("*/*/*/seed_*/*/result.json"):
            result = json.loads(path.read_text())
            cfg = result["config"]
            candidates.append({"model": cfg["model"], "config_id": path.parts[-6],
                               "block": result["outer_block"],
                               "selection": selection_value(cfg["task"], result["metrics"]["overall"]),
                               "config": {k: cfg[k] for k in ("d_model", "dropout", "lr", "loss")}})
        frame = pd.DataFrame(candidates)
        if frame.empty:
            raise SystemExit("No tuning results found")
        agg = frame.groupby(["model", "config_id"], as_index=False).agg(
            mean=("selection", "mean"), n=("selection", "size"),
            n_blocks=("block", "nunique"))
        valid = frame.groupby(["model", "config_id"])["block"].agg(
            lambda x: set(x) == {"block0", "block4", "block9"}).rename("valid")
        agg = agg.merge(valid.reset_index(), on=["model", "config_id"])
        best = agg[agg.valid].sort_values("mean", ascending=False).groupby("model").head(1)
        configs = {}
        for _, row in best.iterrows():
            match = next(x for x in candidates if x["model"] == row.model and x["config_id"] == row.config_id)
            configs[row.model] = {**match["config"], "tuning_mean": float(row["mean"]),
                                  "config_id": row.config_id}
        selection["model_configs"] = configs
        selection["tuning_root"] = str(args.checkpoint_root)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
        return
    if args.stage == "analyze":
        if not args.checkpoint_root:
            raise SystemExit("analyze requires --checkpoint-root")
        paper_summary(
            args.checkpoint_root, args.output, (args.tasks or ["positive"])[0],
            set(args.blocks) if args.blocks else None, args.expected_seeds)
        return
    if args.stage == "external":
        if not args.models or not args.checkpoint_root:
            raise SystemExit("external requires --models and --checkpoint-root")
        evaluate_external(args.data_dir, args.checkpoint_root, args.output, args.models,
                          (args.tasks or ["positive"])[0], args.selection_file)
        return
    units, confirmatory = split_confirmatory(load_units(args.data_dir, TRAIN_FILES))
    selected_configs = {}
    if args.selection_file and args.selection_file.exists():
        selected_configs = json.loads(args.selection_file.read_text()).get("model_configs", {})
    command_line_config = {
        key: value for key, value in {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
        }.items() if value is not None
    }
    if command_line_config and args.models:
        selected_configs = {
            **selected_configs,
            **{
                model: {**selected_configs.get(model, {}), **command_line_config}
                for model in args.models
                if model not in (TABULAR_MODELS + SPECIAL_TABULAR_MODELS
                                 + TP120_EXPERIMENTAL_TABULAR_MODELS)
            },
        }
    if args.stage == "finalize":
        if not args.models or not args.checkpoint_root or not args.selection_file:
            raise SystemExit("finalize requires --models, --checkpoint-root, --selection-file")
        finalize_models(units, args.checkpoint_root, args.selection_file, args.output,
                        args.models, (args.tasks or ["positive"])[0])
        return
    if args.stage == "confirmatory":
        if not args.models or not args.checkpoint_root or not args.selection_file:
            raise SystemExit("confirmatory requires --models, --checkpoint-root, --selection-file")
        evaluate_confirmatory(confirmatory, args.checkpoint_root, args.output, args.models,
                              (args.tasks or ["positive"])[0], args.selection_file)
        return
    if args.stage == "smoke":
        run_matrix(units, args.output, "smoke", args.models or ["lyra_corrected", "cnn"],
                   args.tasks or ["positive"], args.blocks or ["block9"], [0], 2, 2, 32, 64)
    elif args.stage == "screen":
        run_matrix(units, args.output, "screen", args.models or NEURAL_MODELS + TABULAR_MODELS,
                   args.tasks or ["positive"], args.blocks or ["block0", "block4", "block9"],
                   [0], 35, 8, 64, 256)
    elif args.stage == "tune":
        blocks = args.blocks or ["block0", "block4", "block9"]
        tune_models = args.models or ["lyra_corrected"]
        grid = [
            ("d32_base", 32, 0.2, 1e-3, "huber"),
            ("d64_lowdrop", 64, 0.1, 1e-3, "huber"),
            ("d64_base", 64, 0.2, 1e-3, "huber"),
            ("d64_slow", 64, 0.2, 3e-4, "huber"),
            ("d64_highdrop", 64, 0.4, 1e-3, "huber"),
            ("d64_rank", 64, 0.2, 1e-3, "rank_huber"),
            ("d128_slow", 128, 0.1, 3e-4, "huber"),
            ("d128_base", 128, 0.2, 1e-3, "huber"),
        ]
        for config_id, width, dropout, lr, tune_loss in grid:
            overrides = {m: {"d_model": width, "dropout": dropout, "lr": lr,
                             "loss": tune_loss} for m in tune_models}
            run_matrix(units, args.output / config_id, "tune", tune_models, ["positive"],
                       blocks, [0], 50, 10, width, 256, config_overrides=overrides)
    elif args.stage == "full":
        if not args.models:
            raise SystemExit("--models is required for the full stage after screening")
        run_matrix(units, args.output, "full", args.models, args.tasks or ["positive"],
                   args.blocks or [f"block{i}" for i in range(10)], args.seeds or [0, 1, 2],
                   args.max_epochs or 100, args.patience or 18,
                   args.d_model or 64, args.batch_size or 256,
                   ablation=args.ablation,
                   config_overrides=selected_configs,
                   train_chemistry=args.train_chemistry)
    elif args.stage == "tasks":
        if not args.models:
            raise SystemExit("--models is required for task formulation runs")
        run_matrix(units, args.output, "tasks", args.models,
                   args.tasks or ["tp60", "tp120", "classification", "hurdle"],
                   args.blocks or [f"block{i}" for i in range(10)], [0], 80, 15, 64, 256,
                   config_overrides=selected_configs)
    elif args.stage == "improvements":
        if "lyra_corrected" not in selected_configs:
            raise SystemExit(
                "improvements require --selection-file with the internally tuned "
                "lyra_corrected configuration")
        tuned = {k: v for k, v in selected_configs["lyra_corrected"].items()
                 if k in {"d_model", "dropout", "lr", "weight_decay", "loss"}}
        models = args.models or (IMPROVEMENT_MODELS + IMPROVEMENT_REFERENCES)
        run_matrix(units, args.output, "improvements", models, ["positive"],
                   args.blocks or ["block0", "block4", "block9"], [0], 50, 10, 64, 256,
                   config_overrides={model: dict(tuned) for model in models})
    elif args.stage == "data_efficiency":
        if not args.models:
            raise SystemExit("--models is required for data-efficiency runs")
        fractions = args.train_fractions or [i / 10 for i in range(1, 11)]
        if any(not 0 < fraction <= 1 for fraction in fractions):
            raise SystemExit("--train-fractions must all be in (0, 1]")
        blocks = args.blocks or [f"block{i}" for i in range(19)]
        for fraction in sorted(set(fractions)):
            fraction_root = args.output / f"fraction_{round(100 * fraction):03d}"
            run_matrix(
                units, fraction_root, "data_efficiency", args.models,
                args.tasks or ["tp120"], blocks, args.seeds or [0],
                args.max_epochs or 100, args.patience or 18,
                args.d_model or 64, args.batch_size or 256,
                config_overrides=selected_configs,
                train_fraction=fraction,
                sampling_seed=args.sampling_seed,
            )
    elif args.stage == "ablations":
        if "lyra_corrected" not in selected_configs:
            raise SystemExit(
                "ablations require --selection-file with the internally tuned "
                "lyra_corrected configuration")
        blocks = args.blocks or ["block0", "block4", "block9"]
        tuned_lyra = {k: v for k, v in selected_configs["lyra_corrected"].items()
                      if k in {"d_model", "dropout", "lr", "weight_decay", "loss"}}
        available = ablation_experiment_models(
            include_physics_tokens="lyra_physics_tokens" in selected_configs)
        requested = set(args.experiments) if args.experiments else set(available)
        unknown = requested - set(available)
        if unknown:
            raise SystemExit(
                f"Unknown ablation experiments: {sorted(unknown)}; "
                f"available={list(available)}")

        def run_ablation(label, stage, models, **kwargs):
            if label not in requested:
                return
            run_matrix(
                units, args.output / label, stage, models, ["positive"],
                blocks, [0], kwargs.pop("max_epochs", 50),
                kwargs.pop("patience", 10), 64, 256, **kwargs)

        def controlled(models, **changes):
            config = {**tuned_lyra, **changes}
            return {model: dict(config) for model in models}

        # Dataset-source ablations are isolated in named output subdirectories.
        for label, sources in [
            ("lmlv1", ["LMLv1"]),
            ("lmlv1_lmlv2", ["LMLv1", "LMLv2"]),
            ("all_sources", ["LMLv1", "LMLv2", "Systematic"]),
        ]:
            run_ablation(label, "ablation_sources", ["lyra_corrected"],
                         sources=sources,
                         config_overrides=controlled(["lyra_corrected"]))
        for chemistry in ["BST", "SS"]:
            label = f"chem_{chemistry}"
            run_ablation(label, "ablation_chemistry", ["lyra_corrected"],
                         chemistry=chemistry,
                         config_overrides=controlled(["lyra_corrected"]))
        for model in ["lyra_original", "lyra_original_tabular", "lyra_seqonly", "lyra_no_role",
                      "lyra_unidirectional", "lyra_corrected", "lyra_corrected_nolate",
                      "tab_mlp"]:
            run_ablation(f"architecture_{model}", "ablation_architecture", [model],
                         config_overrides=controlled([model]))
        for model in ["lyra_corrected", "lyra_corrected_crossgate",
                      "lyra_corrected_tabdeep", "lyra_corrected_film"]:
            run_ablation(f"fusion_{model}", "ablation_fusion", [model],
                         config_overrides=controlled([model]))
        run_ablation(
            "strict_nucleotide_only", "ablation_strict_nucleotide_only",
            ["lyra_corrected_nolate"], ablation="no_context",
            config_overrides=controlled(["lyra_corrected_nolate"],
                                        ablation="no_context"))
        for feature_ablation in ["no_context", "no_position", "no_thermo"]:
            run_ablation(
                f"features_{feature_ablation}", "ablation_features",
                ["lyra_corrected"], ablation=feature_ablation,
                config_overrides=controlled(["lyra_corrected"],
                                            ablation=feature_ablation))
            run_ablation(
                f"tabular_features_{feature_ablation}",
                "ablation_tabular_features", ["hist_gradient_boosting"],
                max_epochs=1, patience=1, ablation=feature_ablation)
        # Refit the strongest tabular model after dropping each mutually
        # exclusive feature group. Unlike impurity importance, these folds
        # directly measure which engineered signals carry unique OOF value.
        for group in TABULAR_FEATURE_GROUPS:
            run_ablation(
                f"catboost_drop_{group}", "ablation_catboost_feature_group",
                ["catboost"], max_epochs=1, patience=1,
                ablation=f"drop_group_{group}")
        # If the pre-frozen improvement gate retained physics-token Lyra,
        # isolate which engineered property actually benefits from entering
        # the sequence path. The late tabular branch is deliberately left
        # untouched in these group drops.
        if "lyra_physics_tokens" in selected_configs:
            physics_config = {
                k: v for k, v in selected_configs["lyra_physics_tokens"].items()
                if k in {"d_model", "dropout", "lr", "weight_decay", "loss"}
            }
            for group in (
                "mismatch_count", "mismatch_position", "gc_fraction",
                "melting_temperature", "vienna_dg",
            ):
                ablation = f"token_drop_{group}"
                run_ablation(
                    f"physics_tokens_drop_{group}",
                    "ablation_physics_token_group", ["lyra_physics_tokens"],
                    ablation=ablation,
                    config_overrides={"lyra_physics_tokens": {
                        **physics_config, "ablation": ablation,
                    }})
        # The September-2025 energy calculations exist only for the exact
        # LMLv1 export. Compare on identical LMLv1-only folds so availability
        # cannot reveal source identity or contaminate the primary benchmark.
        run_ablation(
            "historical_thermo", "ablation_historical_thermo",
            ["catboost", "catboost_historical"], max_epochs=1, patience=1,
            sources=["LMLv1"])
        for width in [32, 64, 128]:
            run_ablation(
                f"width_{width}", "ablation_width", ["lyra_corrected"],
                config_overrides=controlled(["lyra_corrected"], d_model=width))
        for loss in ["mse", "huber", "rank_huber"]:
            run_ablation(
                f"loss_{loss}", "ablation_loss", ["lyra_corrected"], loss=loss,
                config_overrides=controlled(["lyra_corrected"], loss=loss))
    summarize(args.output)


if __name__ == "__main__":
    main()
