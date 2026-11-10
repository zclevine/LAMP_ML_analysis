from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import sha256_file
from .external import _completed_results, _internal_completion_hashes
from .metrics import selection_value
from .runner import (IMPROVEMENT_MODELS, IMPROVEMENT_REFERENCES, NEURAL_MODELS,
                     TABULAR_MODELS, ablation_experiment_models)


SCREEN_BLOCKS = ("block0", "block4", "block9")
FULL_BLOCKS = tuple(f"block{i}" for i in range(10))
TASKS = ("tp60", "tp120", "classification", "hurdle")
EXTERNAL_DATASETS = ("mpox", "mpox_validation", "Borrelia", "LASV")
NEURAL_AUDIT_MODELS = set(NEURAL_MODELS + IMPROVEMENT_MODELS + IMPROVEMENT_REFERENCES)


def _expected_inventory(models, tasks, seeds, blocks):
    return {(task, model, f"seed_{seed}", block)
            for task in tasks for model in models for seed in seeds for block in blocks}


def _verify_matrix(root: Path, models, tasks, seeds, blocks, label: str) -> dict:
    expected = _expected_inventory(models, tasks, seeds, blocks)
    observed = set()
    malformed = []
    missing_predictions = []
    missing_identity_isolation = []
    missing_inner_identity_isolation = []
    for path in root.glob("*/*/seed_*/*/result.json"):
        task, model, seed, block = path.relative_to(root).parts[:4]
        observed.add((task, model, seed, block))
        result = json.loads(path.read_text())
        config = result.get("config", {})
        if (result.get("outer_block") != block or
                result.get("model", config.get("model")) != model or
                result.get("task", config.get("task")) != task or
                int(result.get("seed", config.get("seed", -1))) != int(seed[5:])):
            malformed.append(str(path))
        if not (path.parent / "predictions.npz").exists():
            missing_predictions.append(str(path.parent / "predictions.npz"))
        if not model.startswith("fusion_"):
            isolation = result.get("sequence_identity_isolation", {})
            outer_isolation = isolation.get("outer", {})
            if outer_isolation.get("post_isolation_overlap") != 0:
                missing_identity_isolation.append(str(path))
            # Neural models use an inner block for epoch selection. Requiring
            # this receipt prevents a nominally clean outer split from hiding
            # sequence aliases between the fitting and early-stopping sets.
            if model in NEURAL_AUDIT_MODELS:
                inner_isolation = isolation.get("inner", {})
                if inner_isolation.get("post_isolation_overlap") != 0:
                    missing_inner_identity_isolation.append(str(path))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if (missing or unexpected or malformed or missing_predictions
            or missing_identity_isolation or missing_inner_identity_isolation):
        raise RuntimeError(
            f"{label} matrix audit failed: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}, malformed={malformed[:10]}, "
            f"missing_predictions={missing_predictions[:10]}, "
            f"missing_identity_isolation={missing_identity_isolation[:10]}, "
            f"missing_inner_identity_isolation="
            f"{missing_inner_identity_isolation[:10]}")
    error_events = []
    event_path = root / "events.jsonl"
    if event_path.exists():
        for line in event_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("status") == "error":
                error_events.append(event)
    if error_events:
        raise RuntimeError(f"{label} contains {len(error_events)} error events")
    return {"label": label, "n_results": len(observed),
            "n_models": len(set(models)), "n_tasks": len(set(tasks)),
            "n_seeds": len(set(seeds)), "n_blocks": len(set(blocks))}


def _verify_finalization(root: Path, selection: dict) -> dict:
    payload = json.loads((root / "finalization.json").read_text())
    expected = {(model, seed) for model in selection["selected_models"] for seed in range(3)}
    observed = set()
    for record in payload.get("records", []):
        key = (record["model"], int(record["seed"]))
        observed.add(key)
        checkpoint = root / record["checkpoint"]
        if not checkpoint.exists() or sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Final checkpoint is absent or changed: {checkpoint}")
        record_path = checkpoint.parent / "record.json"
        if not record_path.exists() or json.loads(record_path.read_text()) != record:
            raise RuntimeError(f"Per-checkpoint finalization record is absent or changed: {record_path}")
    if observed != expected:
        raise RuntimeError(
            f"Final checkpoint inventory mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}")
    return {"n_checkpoints": len(observed), "models": selection["selected_models"]}


def _verify_shard_receipt(path: Path, ablations: dict[str, list[str]]) -> dict:
    """Validate provenance for experiment-level results produced on another worker."""
    payload = json.loads(path.read_text())
    if (payload.get("status") != "complete" or
            payload.get("frozen_data_status_at_execution") != "unopened"):
        raise RuntimeError("Multi-GPU shard receipt is not complete and pre-frozen")
    experiments = payload.get("experiments", [])
    if not experiments or len(experiments) != len(set(experiments)):
        raise RuntimeError("Shard receipt has no experiments or contains duplicates")
    unknown = set(experiments) - set(ablations)
    if unknown:
        raise RuntimeError(f"Shard receipt contains undeclared experiments: {sorted(unknown)}")
    expected_folds = sum(3 * len(ablations[name]) for name in experiments)
    if int(payload.get("expected_folds", -1)) != expected_folds:
        raise RuntimeError(
            f"Shard receipt fold count mismatch: {payload.get('expected_folds')} "
            f"!= {expected_folds}")
    archives = payload.get("archives", [])
    archived_experiments = []
    archived_results = 0
    for archive in archives:
        digest = archive.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError("Shard archive is missing a valid SHA-256 digest")
        if archive.get("strict_audit") != "passed":
            raise RuntimeError("Shard archive was not strict-audited after merge")
        archived_experiments.extend(archive.get("experiments", []))
        archived_results += int(archive.get("n_results", 0))
    if (len(archived_experiments) != len(set(archived_experiments)) or
            set(archived_experiments) != set(experiments)):
        raise RuntimeError("Shard archives do not partition the declared experiments")
    if archived_results != expected_folds:
        raise RuntimeError(
            f"Shard archive result count mismatch: {archived_results} != {expected_folds}")
    validation = payload.get("validation", {})
    for key in ("pip_freeze_sha256", "hardware_validation_sha256"):
        digest = validation.get(key, "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RuntimeError(f"Shard validation is missing {key}")
    if (int(validation.get("unit_tests_passed", 0)) < 1 or
            int(validation.get("gpu_models_validated", 0)) < 1):
        raise RuntimeError("Shard validation did not record tests and GPU models")
    final_shard = payload.get("final_checkpoint_shard")
    final_summary = None
    if final_shard is not None:
        models = final_shard.get("models", [])
        seeds = final_shard.get("seeds", [])
        if not models or len(models) != len(set(models)):
            raise RuntimeError("Final-checkpoint shard has no models or contains duplicates")
        if seeds != [0, 1, 2]:
            raise RuntimeError("Final-checkpoint shard must contain exactly seeds 0, 1, and 2")
        for key in ("selection_sha256", "cv_metadata_sha256"):
            digest = final_shard.get(key, "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RuntimeError(f"Final-checkpoint shard is missing {key}")
        checkpoint_archives = final_shard.get("archives", [])
        archived_models = []
        archived_checkpoints = 0
        for archive in checkpoint_archives:
            digest = archive.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RuntimeError("Final-checkpoint archive is missing a valid SHA-256 digest")
            if archive.get("strict_audit") != "passed":
                raise RuntimeError("Final-checkpoint archive was not strict-audited after merge")
            archived_models.extend(archive.get("models", []))
            archived_checkpoints += int(archive.get("n_checkpoints", 0))
        if (len(archived_models) != len(set(archived_models)) or
                set(archived_models) != set(models)):
            raise RuntimeError("Final-checkpoint archives do not partition declared models")
        expected_checkpoints = len(models) * len(seeds)
        if archived_checkpoints != expected_checkpoints:
            raise RuntimeError(
                "Final-checkpoint archive count mismatch: "
                f"{archived_checkpoints} != {expected_checkpoints}")
        final_summary = {
            "models": models,
            "seeds": seeds,
            "n_checkpoints": expected_checkpoints,
            "selection_sha256": final_shard["selection_sha256"],
            "cv_metadata_sha256": final_shard["cv_metadata_sha256"],
        }
    return {
        "worker_id": payload.get("worker_id"),
        "experiments": experiments,
        "n_experiments": len(experiments),
        "n_results": expected_folds,
        "n_archives": len(archives),
        "final_checkpoint_shard": final_summary,
        "receipt_sha256": sha256_file(path),
    }


def _verify_frozen(root: Path, aggregate_name: str, expected_pairs: set[tuple[str, str]],
                   label: str) -> dict:
    receipt = json.loads((root / "EVALUATION_OPENED.json").read_text())
    locked_internal = receipt.get("internal_completion_sha256")
    current_internal = _internal_completion_hashes(root.parent)
    if locked_internal != current_internal:
        raise RuntimeError(
            f"{label} internal artifacts are missing, changed, or were not locked "
            "before frozen-data opening")
    results = _completed_results(root, aggregate_name, receipt)
    if results is None:
        raise RuntimeError(f"{label} is not checksum-verified as complete")
    observed = {(row["dataset"], row["model"]) for row in results}
    if observed != expected_pairs or len(results) != len(expected_pairs):
        raise RuntimeError(
            f"{label} inventory mismatch: missing={sorted(expected_pairs-observed)}, "
            f"unexpected={sorted(observed-expected_pairs)}")
    prediction_hashes = {}
    result_hashes = {}
    for row in results:
        base = root / row["task"] / row["model"]
        prediction = (base / "predictions.npz" if label == "confirmatory"
                      else base / row["dataset"] / "predictions.npz")
        result_path = (base / "result.json" if label == "confirmatory"
                       else base / row["dataset"] / "result.json")
        if not prediction.exists() or not result_path.exists():
            raise FileNotFoundError(
                f"{label} result or prediction artifact is missing for "
                f"{row['dataset']}/{row['model']}")
        stored = json.loads(result_path.read_text())
        for key in ("dataset", "model", "task", "n_units"):
            if stored.get(key) != row.get(key):
                raise RuntimeError(
                    f"{label} aggregate/result mismatch for {row['dataset']}/"
                    f"{row['model']}: {key}")
        arrays = np.load(prediction, allow_pickle=False)
        required_arrays = {"y", "pred", "group_id"}
        if not required_arrays.issubset(arrays.files):
            raise RuntimeError(
                f"{label} prediction arrays are incomplete for "
                f"{row['dataset']}/{row['model']}")
        lengths = {len(arrays[key]) for key in arrays.files}
        if lengths != {int(row["n_units"])}:
            raise RuntimeError(
                f"{label} prediction length mismatch for "
                f"{row['dataset']}/{row['model']}: {sorted(lengths)}")
        if (not np.isfinite(arrays["y"]).all()
                or not np.isfinite(arrays["pred"]).all()):
            raise RuntimeError(
                f"{label} contains non-finite targets or predictions for "
                f"{row['dataset']}/{row['model']}")
        if len(np.unique(arrays["group_id"].astype(str))) != int(row["n_units"]):
            raise RuntimeError(
                f"{label} contains duplicate unit identities for "
                f"{row['dataset']}/{row['model']}")
        score = selection_value(row["task"], stored["metrics"]["overall"])
        if not np.isfinite(score):
            raise RuntimeError(
                f"{label} primary metric is not estimable for "
                f"{row['dataset']}/{row['model']}")
        prediction_hashes[str(prediction.relative_to(root))] = sha256_file(prediction)
        result_hashes[str(result_path.relative_to(root))] = sha256_file(result_path)
    return {"n_results": len(results),
            "receipt_sha256": sha256_file(root / "EVALUATION_OPENED.json"),
            "completion_sha256": sha256_file(root / "EVALUATION_COMPLETE.json"),
            "prediction_sha256": prediction_hashes,
            "result_sha256": result_hashes}


def audit_pipeline(results_root: Path, output: Path | None = None) -> dict:
    required = [
        results_root / "manifest" / "manifest.json",
        results_root / "selection" / "selection.json",
        results_root / "tuned_selection" / "selection.json",
        results_root / "analysis" / "full" / "positive_paper_summary.json",
        results_root / "protocol_deviations.json",
        results_root / "hardware_validation" / "mamba3_mimo_compatibility.json",
        results_root / "fusion_screen_probe.json",
        results_root / "complementarity" / "complementarity_summary.json",
        results_root / "conditional_fusion" / "conditional_fusion_summary.json",
        results_root / "improvement_gate" / "selection.json",
        results_root / "property_attribution" / "property_attribution.json",
    ]
    secondary_analysis_paths = {
        task: results_root / "analysis" / "tasks" / task / f"{task}_paper_summary.json"
        for task in TASKS
    }
    required.extend(secondary_analysis_paths.values())
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise FileNotFoundError(f"Required audit artifacts are missing: {missing_required}")
    deviations = json.loads((results_root / "protocol_deviations.json").read_text())
    deviation_ids = {row.get("id") for row in deviations.get("deviations", [])}
    shard_receipt_paths = [
        results_root / "shard_receipts" / name
        for name in ("worker2.json", "worker3.json")
    ]
    if "disjoint-multigpu-ablation-sharding" in deviation_ids:
        missing_receipts = [str(path) for path in shard_receipt_paths if not path.exists()]
        if missing_receipts:
            raise FileNotFoundError(
                f"Required multi-GPU shard receipts are missing: {missing_receipts}")
        required.extend(shard_receipt_paths)
    selection = json.loads((results_root / "tuned_selection" / "selection.json").read_text())
    improvement_gate = json.loads(
        (results_root / "improvement_gate" / "selection.json").read_text())
    if (improvement_gate.get("status") != "locked" or
            improvement_gate.get("frozen_data_status") != "unopened"):
        raise RuntimeError("Improvement gate was not locked before frozen evaluation")
    finalists = selection.get("selected_models", [])
    if not finalists or "lyra_corrected" not in finalists or "sameed_rf" not in finalists:
        raise RuntimeError("Finalist lock must retain corrected Lyra and Sameed RF")
    if "lyra_corrected" not in selection.get("model_configs", {}):
        raise RuntimeError("The corrected Lyra tuning winner is absent from the finalist lock")
    fusion = json.loads((results_root / "fusion_screen_probe.json").read_text())
    best_fusion = fusion.get("best", {})
    if (best_fusion.get("sequence_model") not in NEURAL_AUDIT_MODELS or
            best_fusion.get("tabular_model") != "catboost" or
            not 0 <= float(best_fusion.get("sequence_weight", -1)) <= 1):
        raise RuntimeError("The internal fusion probe is missing or malformed")
    selected_fusion = selection.get("selected_fusion")
    gate_fusion = improvement_gate.get("selected_fusion", selected_fusion)
    if not selected_fusion or any(
            selected_fusion.get(key) != gate_fusion.get(key)
            for key in ("sequence_model", "tabular_model", "sequence_weight", "tabular_weight")):
        raise RuntimeError("The finalist fusion does not match the pre-frozen improvement lock")
    if (selected_fusion["sequence_model"] not in finalists or
            selected_fusion["tabular_model"] not in finalists):
        raise RuntimeError("The locked fusion base models are absent from the finalist set")
    fusion_name = selected_fusion["name"]

    # The analysis can predate a late, pre-frozen improvement decision.  Its
    # mere presence is therefore insufficient: require exact coverage of the
    # final locked model set so a retained candidate or replacement fusion
    # cannot silently be omitted from the paper tables and paired tests.
    analysis = json.loads(
        (results_root / "analysis" / "full" / "positive_paper_summary.json").read_text())
    expected_analysis_models = set(finalists + [fusion_name])
    observed_analysis_models = {
        row.get("model") for row in analysis.get("models", []) if row.get("model")}
    inventory_models = set(analysis.get("fold_inventory", {}))
    if (observed_analysis_models != expected_analysis_models or
            inventory_models != expected_analysis_models):
        raise RuntimeError(
            "Full paper analysis does not match the locked finalist set: "
            f"missing={sorted(expected_analysis_models - observed_analysis_models)}, "
            f"unexpected={sorted(observed_analysis_models - expected_analysis_models)}, "
            f"inventory_missing={sorted(expected_analysis_models - inventory_models)}, "
            f"inventory_unexpected={sorted(inventory_models - expected_analysis_models)}")

    secondary_analysis = {}
    for task, path in secondary_analysis_paths.items():
        payload = json.loads(path.read_text())
        observed = {row.get("model") for row in payload.get("models", [])
                    if row.get("model")}
        inventory = set(payload.get("fold_inventory", {}))
        expected = set(finalists)
        if observed != expected or inventory != expected:
            raise RuntimeError(
                f"{task} paper analysis does not match the locked finalist set: "
                f"missing={sorted(expected-observed)}, "
                f"unexpected={sorted(observed-expected)}, "
                f"inventory_missing={sorted(expected-inventory)}, "
                f"inventory_unexpected={sorted(inventory-expected)}")
        secondary_analysis[task] = {
            "n_models": len(observed),
            "n_paired_comparisons": len(payload.get("paired_comparisons", [])),
            "sha256": sha256_file(path),
        }

    checks = [_verify_matrix(
        results_root / "screen", NEURAL_MODELS + TABULAR_MODELS,
        ["positive"], [0], SCREEN_BLOCKS, "screen")]
    for config_id in (
        "d32_base", "d64_lowdrop", "d64_base", "d64_slow", "d64_highdrop",
        "d64_rank", "d128_slow", "d128_base",
    ):
        checks.append(_verify_matrix(
            results_root / "tuning" / config_id, ["lyra_corrected"],
            ["positive"], [0], SCREEN_BLOCKS, f"tuning/{config_id}"))
    checks.append(_verify_matrix(
        results_root / "full", finalists + [fusion_name], ["positive"], [0, 1, 2],
        FULL_BLOCKS, "full"))
    checks.append(_verify_matrix(
        results_root / "tasks", finalists, TASKS, [0], FULL_BLOCKS, "tasks"))
    checks.append(_verify_matrix(
        results_root / "improvements",
        improvement_gate["evaluated_candidates"] + improvement_gate["reference_models"],
        ["positive"], [0], SCREEN_BLOCKS, "improvements"))
    checks.append(_verify_matrix(
        results_root / "improvement_full", [improvement_gate["selected_candidate"]],
        ["positive"], [0, 1, 2], FULL_BLOCKS, "improvement_full"))

    ablations = ablation_experiment_models(
        include_physics_tokens="lyra_physics_tokens" in finalists)
    for experiment, models in ablations.items():
        checks.append(_verify_matrix(
            results_root / "ablations" / experiment, models,
            ["positive"], [0], SCREEN_BLOCKS, f"ablations/{experiment}"))
    shard_execution = None
    if "disjoint-multigpu-ablation-sharding" in deviation_ids:
        shard_execution = [
            _verify_shard_receipt(path, ablations) for path in shard_receipt_paths]
        assigned = [
            experiment for receipt in shard_execution
            for experiment in receipt["experiments"]
        ]
        if len(assigned) != len(set(assigned)):
            raise RuntimeError("Multi-GPU shard receipts assign an experiment more than once")
        if "disjoint-final-checkpoint-sharding" in deviation_ids:
            final_shards = [receipt["final_checkpoint_shard"]
                            for receipt in shard_execution]
            if any(shard is None for shard in final_shards):
                raise RuntimeError(
                    "Final-checkpoint sharding deviation requires a final shard in every receipt")
            assigned_models = [model for shard in final_shards for model in shard["models"]]
            expected_auxiliary = {
                model for model in finalists
                if model in NEURAL_AUDIT_MODELS and not model.startswith("mamba")
            }
            if (len(assigned_models) != len(set(assigned_models)) or
                    set(assigned_models) != expected_auxiliary):
                raise RuntimeError(
                    "Final-checkpoint shard model inventory mismatch: "
                    f"missing={sorted(expected_auxiliary - set(assigned_models))}, "
                    f"unexpected={sorted(set(assigned_models) - expected_auxiliary)}")
            selection_digest = sha256_file(
                results_root / "tuned_selection" / "selection.json")
            if any(shard["selection_sha256"] != selection_digest
                   for shard in final_shards):
                raise RuntimeError("Final-checkpoint shard selection hash mismatch")
            metadata_digests = {shard["cv_metadata_sha256"] for shard in final_shards}
            if len(metadata_digests) != 1:
                raise RuntimeError(
                    "Final-checkpoint workers did not use identical CV metadata")

    finalization = _verify_finalization(results_root / "final", selection)
    evaluation_models = finalists + [fusion_name]
    confirmatory_pairs = {
        ("LMLv2_block19_confirmatory", model) for model in evaluation_models}
    external_pairs = {
        (dataset, model) for dataset in EXTERNAL_DATASETS for model in evaluation_models}
    frozen = {
        "confirmatory": _verify_frozen(
            results_root / "confirmatory", "confirmatory_results.json",
            confirmatory_pairs, "confirmatory"),
        "external": _verify_frozen(
            results_root / "external", "external_results.json",
            external_pairs, "external"),
    }
    payload = {
        "status": "complete",
        "results_root": str(results_root),
        "checks": checks,
        "secondary_task_analyses": secondary_analysis,
        "shard_execution": shard_execution,
        "finalization": finalization,
        "frozen_evaluations": frozen,
        "required_artifact_sha256": {
            str(path.relative_to(results_root)): sha256_file(path) for path in required
        },
    }
    destination = output or (results_root / "audit" / "completion_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit_pipeline(args.results_root, args.output)


if __name__ == "__main__":
    main()
