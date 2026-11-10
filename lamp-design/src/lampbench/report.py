from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .data import sha256_file
from .external import _completed_results
from .metrics import selection_value


def _json_safe(value):
    """Convert non-finite statistics to JSON null without hiding finite failures."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _source_hashes(project_root: Path | None = None) -> dict[str, str]:
    """Fingerprint the executable benchmark source included with the report."""
    root = project_root or Path(__file__).resolve().parents[2]
    paths = []
    for package in ("lampbench", "lyra", "lamp_designer"):
        paths.extend(sorted((root / "src" / package).glob("*.py")))
    paths.extend(sorted((root / "experiments" / "yaml").rglob("*.yaml")))
    paths += [
        root / "README.md",
        root / "docs" / "PROTOCOL.md",
        root / "tests" / "test_benchmark.py",
        root / "requirements-gpu.txt",
        root / "pyproject.toml",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Benchmark source fingerprint is incomplete: {missing}")
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def _required_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required paper artifact is missing: {path}")
    return json.loads(path.read_text())


def _metric_row(result: dict) -> dict:
    overall = result["metrics"]["overall"]
    return {
        "dataset": result["dataset"], "model": result["model"],
        "task": result["task"], "n_units": result["n_units"],
        "selection": selection_value(result["task"], overall),
        **{key: value for key, value in overall.items()
           if isinstance(value, (int, float))},
        "bootstrap_ci95": result["bootstrap"]["ci95"],
    }


def generate_report(results_root: Path, output: Path) -> dict:
    manifest_path = results_root / "manifest" / "manifest.json"
    selection_path = results_root / "tuned_selection" / "selection.json"
    internal_path = results_root / "analysis" / "full" / "positive_paper_summary.json"
    confirmatory_path = results_root / "confirmatory" / "confirmatory_results.json"
    external_path = results_root / "external" / "external_results.json"
    deviations_path = results_root / "protocol_deviations.json"
    compatibility_path = (results_root / "hardware_validation" /
                          "mamba3_mimo_compatibility.json")
    audit_path = results_root / "audit" / "completion_audit.json"
    fusion_path = results_root / "fusion_screen_probe.json"
    complementarity_path = (results_root / "complementarity" /
                            "complementarity_summary.json")
    conditional_fusion_path = (results_root / "conditional_fusion" /
                               "conditional_fusion_summary.json")
    improvement_gate_path = results_root / "improvement_gate" / "selection.json"
    property_attribution_path = (results_root / "property_attribution" /
                                 "property_attribution.json")
    shard_receipt_paths = sorted((results_root / "shard_receipts").glob("*.json"))
    screen_path = results_root / "screen" / "summary.csv"
    task_analysis_paths = {
        task: results_root / "analysis" / "tasks" / task / f"{task}_paper_summary.json"
        for task in ("tp60", "tp120", "classification", "hurdle")
    }
    required = [manifest_path, selection_path, internal_path,
                confirmatory_path, external_path, deviations_path,
                compatibility_path, audit_path, fusion_path,
                complementarity_path, conditional_fusion_path,
                improvement_gate_path, property_attribution_path]
    required.extend(shard_receipt_paths)
    required.extend(task_analysis_paths.values())
    manifest, selection, internal = [_required_json(path) for path in required[:3]]
    deviations = _required_json(deviations_path)
    compatibility = _required_json(compatibility_path)
    audit = _required_json(audit_path)
    fusion = _required_json(fusion_path)
    complementarity = _required_json(complementarity_path)
    conditional_fusion = _required_json(conditional_fusion_path)
    improvement_gate = _required_json(improvement_gate_path)
    property_attribution = _required_json(property_attribution_path)
    task_analyses = {
        task: _required_json(path) for task, path in task_analysis_paths.items()
    }
    if audit.get("status") != "complete":
        raise RuntimeError("Pipeline completion audit did not pass")
    confirmatory_root = results_root / "confirmatory"
    external_root = results_root / "external"
    confirmatory_receipt = _required_json(confirmatory_root / "EVALUATION_OPENED.json")
    external_receipt = _required_json(external_root / "EVALUATION_OPENED.json")
    confirmatory = _completed_results(
        confirmatory_root, "confirmatory_results.json", confirmatory_receipt)
    external = _completed_results(external_root, "external_results.json", external_receipt)
    if confirmatory is None or external is None:
        raise RuntimeError("Frozen evaluation is not checksum-verified as complete")

    tasks_path = results_root / "tasks" / "summary.csv"
    ablations_path = results_root / "ablations" / "summary.csv"
    if not screen_path.exists() or not tasks_path.exists() or not ablations_path.exists():
        raise FileNotFoundError("Screen, task, or ablation summary is missing")
    screen = pd.read_csv(screen_path).to_dict(orient="records")
    tasks = pd.read_csv(tasks_path).to_dict(orient="records")
    ablations = pd.read_csv(ablations_path).to_dict(orient="records")
    internal_rows = []
    for item in internal["models"]:
        overall = item["pooled_metrics"]["overall"]
        internal_rows.append({
            "model": item["model"], "task": item["task"],
            "n_units": item["n_units"],
            "pooled_selection": selection_value(item["task"], overall),
            "block_macro_mean": item["block_macro_mean"],
            "block_macro_sd": item["block_macro_sd"],
            **{key: value for key, value in overall.items()
               if isinstance(value, (int, float))},
        })
    confirmatory_rows = [_metric_row(x) for x in confirmatory]
    external_rows = [_metric_row(x) for x in external]
    weakness_controls = [
        {
            "weakness": "primer-set and preprocessing leakage",
            "control": ("Outer LMLv1 blocks are primer-set disjoint and all sources are also "
                        "grouped by an exact full primer/target sequence identity; outer-equivalent "
                        "units are removed from all fitting and inner-equivalent units are removed "
                        "during epoch selection. Standardization is fit after isolation."),
            "evidence": ("manifest.protocol, manifest.outer_fold_identity_isolation, "
                         "per-fold sequence_identity_isolation records"),
        },
        {
            "weakness": "padding and invalid reverse-sequence computation",
            "control": ("Corrected Lyra batches/groups exact lengths, reverses only valid tokens, "
                        "masks every block, and pools only valid positions."),
            "evidence": "padding-invariance test and corrected Lyra implementation",
        },
        {
            "weakness": "inconsistent chemistry labels and hidden source effects",
            "control": ("Master-mix labels are canonicalized to BST/SS/WS; results are stratified "
                        "by chemistry and concentration; chemistry- and source-specific ablations "
                        "are required by the completion audit."),
            "evidence": "manifest provenance, stratified metrics, chemistry/source ablations",
        },
        {
            "weakness": "technical-replicate pseudoreplication",
            "control": ("Technical replicates are aggregated before splitting, fitting, or scoring; "
                        "replicate count and amplification consistency remain explicit strata."),
            "evidence": (f"{manifest['counts']['rows']} rows -> "
                         f"{manifest['counts']['units']} replicate-aggregated units"),
        },
        {
            "weakness": "validation reuse and holdout adaptation",
            "control": ("LMLv2 block19 is removed before all development runs; finalist models and "
                        "three-checkpoint hashes are frozen before confirmatory and external receipts "
                        "are created; receipts reject changed data, models, or checkpoints."),
            "evidence": "finalization lock plus EVALUATION_OPENED/EVALUATION_COMPLETE checksums",
        },
        {
            "weakness": "single pooled metric and correlated uncertainty",
            "control": ("Primary Spearman is accompanied by Pearson, MAE, RMSE, R2, block-macro "
                        "mean/SD, chemistry/concentration/replicate strata, and paired primer-set "
                        "cluster bootstrap intervals."),
            "evidence": "internal paper summary, paired comparisons, frozen result bootstraps",
        },
    ]

    artifact_hashes = {str(path.relative_to(results_root)): sha256_file(path)
                       for path in required + [screen_path, tasks_path, ablations_path,
                                               results_root / "confirmatory" / "EVALUATION_OPENED.json",
                                               results_root / "confirmatory" / "EVALUATION_COMPLETE.json",
                                               results_root / "external" / "EVALUATION_OPENED.json",
                                               results_root / "external" / "EVALUATION_COMPLETE.json"]}
    payload = {
        "protocol": manifest["protocol"],
        "canonical_lyra_provenance": manifest["canonical_lyra_provenance"],
        "external_novelty": manifest["external_novelty"],
        "selected_models": selection["selected_models"],
        "selected_fusion": selection.get("selected_fusion"),
        "selected_model_configs": selection.get("model_configs", {}),
        "protocol_deviations": deviations,
        "model_compatibility": {"mamba3_mimo": compatibility},
        "completion_audit": audit,
        "internal_fusion_probe": fusion,
        "sequence_tabular_complementarity": complementarity,
        "conditional_fusion_holdout": conditional_fusion,
        "physics_lyra_improvement_gate": improvement_gate,
        "property_attribution": property_attribution,
        "weakness_controls": weakness_controls,
        "model_family_screen": {
            "scope": "three prespecified LMLv1 outer blocks, seed 0; screening only",
            "blocks": ["block0", "block4", "block9"],
            "seeds": [0],
            "rows": screen,
        },
        "internal_cv": internal_rows,
        "internal_paired_comparisons": internal["paired_comparisons"],
        "secondary_tasks": tasks,
        "secondary_task_paired_analyses": task_analyses,
        "ablations": ablations,
        "confirmatory": confirmatory_rows,
        "external": external_rows,
        "artifact_sha256": artifact_hashes,
        "source_sha256": _source_hashes(),
        "interpretation_guards": [
            "Primary model selection used internal development folds only.",
            "Confirmatory and external evaluations used immutable finalized checkpoints.",
            "Earlier tabular external scores were visible during audit; external evaluation is algorithmically locked, not analyst-blind.",
            "External datasets have zero primer-set, target-id, primer-tuple, and full sequence-tuple overlap with development.",
            "Mamba-3 MIMO was excluded before selection because its pinned upstream backward kernel failed reproducible compatibility probes; trainable Mamba-3 SISO variants remain.",
            "Auxiliary GPU work was disjoint by named ablation or final model/seed and required checksum, per-checkpoint, and strict matrix receipts before merge.",
            "The pipeline completion marker is emitted only after exact matrix-coverage, checkpoint-hash, and frozen-evaluation receipt audits pass.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(payload)
    json_path = output / "paper_results.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))

    def table(rows, columns):
        if not rows:
            return "_No results._"
        use = [column for column in columns if any(column in row for row in rows)]
        def cell(value):
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return "NA"
            if isinstance(value, float):
                return f"{value:.4f}"
            if isinstance(value, (list, dict)):
                return json.dumps(value, separators=(",", ":"))
            return str(value).replace("|", "\\|")
        lines = ["| " + " | ".join(use) + " |",
                 "| " + " | ".join("---" for _ in use) + " |"]
        lines.extend("| " + " | ".join(cell(row.get(column, "")) for column in use) + " |"
                     for row in rows)
        return "\n".join(lines)

    lines = [
        "# LAMP paper benchmark results", "",
        "## Selected models", "",
        ", ".join(selection["selected_models"]), "",
        f"Locked fusion: `{selection.get('selected_fusion', {}).get('name', 'none')}`.", "",
        "## Audited model-family screen", "",
        "These are three-block, one-seed screening results, not final full-CV estimates. "
        "They document every evaluated Lyra, Mamba-1/2/3, generic neural, and tabular baseline.", "",
        table(screen, ["model", "task", "mean", "std", "min", "max", "n"]), "",
        "## Internal 10-block × 3-seed cross-validation", "",
        table(internal_rows, ["model", "n_units", "pooled_selection",
                              "block_macro_mean", "block_macro_sd", "mae", "rmse", "r2"]), "",
        "## Controls for identified benchmark weaknesses", "",
        table(weakness_controls, ["weakness", "control", "evidence"]), "",
        "## Frozen LMLv2 block19 confirmatory evaluation", "",
        table(confirmatory_rows, ["model", "n_units", "selection", "mae", "rmse", "r2",
                                  "bootstrap_ci95"]), "",
        "## Frozen external-domain evaluation", "",
        table(external_rows, ["dataset", "model", "n_units", "selection", "mae", "rmse", "r2",
                              "bootstrap_ci95"]), "",
        "## Secondary tasks", "",
        table(tasks, ["task", "model", "mean", "std", "min", "max", "n"]), "",
        "Paired primer-set-cluster bootstrap comparisons and exact fold inventories for "
        "each secondary task are included in `secondary_task_paired_analyses` in the "
        "machine-readable report.", "",
        "## Ablations", "",
        table(ablations, ["experiment", "task", "model", "mean", "std", "n"]), "",
        "## Protocol deviations and compatibility", "",
        f"Recorded deviations: {len(deviations['deviations'])}.", "",
        f"Mamba-3 MIMO status: `{compatibility['status']}`.", "",
        f"Executable source files fingerprinted: {len(payload['source_sha256'])}.", "",
        "## Exploratory tabular/sequence fusion probe", "",
        table([fusion["best"]], ["sequence_model", "tabular_model", "sequence_weight",
                                  "tabular_weight", "pooled_selection",
                                  "block_macro_mean", "block_macro_sd", "n_units"]), "",
        "## What the tabular model contributes", "",
        table(complementarity["catboost_group_permutation"],
              ["feature_group", "mean_spearman_loss", "sd_across_folds", "n_folds"]), "",
        "The table reports the loss from permuting each feature group in held-out folds; "
        "larger values identify information the sequence representation does not recover reliably.", "",
        "## Property-level refit and Lyra token-path attribution", "",
        table(property_attribution["rows"], [
            "feature_group", "catboost_refit_spearman_loss",
            "catboost_permutation_spearman_loss",
            "physics_token_path_spearman_loss",
        ]), "",
        "## Sequence-only information gaps", "",
        table([property_attribution["context_controls"]], [
            "corrected_multimodal_spearman", "strict_nucleotide_only_spearman",
            "strict_nucleotide_gap", "concentration_and_chemistry_gain",
            "mismatch_position_gain", "thermodynamics_and_position_gain",
        ]), "",
        "These matched internal refits show how much predictive ranking is lost when "
        "assay context or engineered physical statistics are withheld. The strict "
        "nucleotide-only gap is an interaction control because it removes both late "
        "tabular fusion and sequence-token context.", "",
        "## Conditional versus fixed fusion", "",
        table([
            {"method": name,
             "pooled_spearman": metrics["spearman"],
             "block_macro_spearman": conditional_fusion[
                 "holdout_block_macro_spearman"][name]}
            for name, metrics in conditional_fusion["holdout_metrics"].items()
        ], ["method", "pooled_spearman", "block_macro_spearman"]), "",
        "## Physics-conditioned Lyra gate", "",
        table([{
            "selected_candidate": improvement_gate["selected_candidate"],
            "decision": improvement_gate["decision"],
            "candidate_block_macro_spearman": improvement_gate[
                "candidate_block_macro_spearman"],
            "standalone_improves": improvement_gate["standalone_improves"],
            "fusion_improves": improvement_gate["fusion_improves"],
        }], ["selected_candidate", "decision", "candidate_block_macro_spearman",
             "standalone_improves", "fusion_improves"]), "",
        "## Interpretation guards", "",
        *[f"- {guard}" for guard in payload["interpretation_guards"]], "",
        f"Machine-readable results: `{json_path.name}`", "",
    ]
    (output / "paper_results.md").write_text("\n".join(lines))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.results_root, args.output)


if __name__ == "__main__":
    main()
