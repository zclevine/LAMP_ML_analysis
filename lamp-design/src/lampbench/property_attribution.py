from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import TABULAR_FEATURE_GROUPS
from .metrics import selection_value


SCREEN_BLOCKS = {"block0", "block4", "block9"}
TOKEN_GROUPS = {
    "mismatch_count", "mismatch_position", "gc_fraction",
    "melting_temperature", "vienna_dg",
}


def _screen_mean(root: Path, model: str) -> float:
    values = []
    blocks = set()
    for path in root.glob(f"positive/{model}/seed_0/*/result.json"):
        result = json.loads(path.read_text())
        block = result["outer_block"]
        blocks.add(block)
        values.append(selection_value("positive", result["metrics"]["overall"]))
    if blocks != SCREEN_BLOCKS or len(values) != len(SCREEN_BLOCKS):
        raise ValueError(
            f"Expected exactly the three internal screen blocks for {root}/{model}; "
            f"found {sorted(blocks)}")
    return float(sum(values) / len(values))


def build_property_attribution(results_root: Path, output: Path) -> dict:
    complementarity_path = (
        results_root / "complementarity" / "complementarity_summary.json")
    gate_path = results_root / "improvement_gate" / "selection.json"
    if not complementarity_path.exists() or not gate_path.exists():
        raise FileNotFoundError("Complementarity and improvement-lock artifacts are required")
    complementarity = json.loads(complementarity_path.read_text())
    gate = json.loads(gate_path.read_text())
    if (gate.get("status") != "locked" or
            gate.get("selected_candidate") != "lyra_physics_tokens"):
        raise RuntimeError("Physics-token Lyra was not locked before attribution")

    catboost_baseline = _screen_mean(results_root / "screen", "catboost")
    physics_baseline = _screen_mean(
        results_root / "improvements", "lyra_physics_tokens")
    permutation = {
        row["feature_group"]: row
        for row in complementarity["catboost_group_permutation"]
    }
    shap = {
        row["feature_group"]: row
        for row in complementarity["catboost_shap_groups"]
    }
    rows = []
    for group in TABULAR_FEATURE_GROUPS:
        cat_drop = _screen_mean(
            results_root / "ablations" / f"catboost_drop_{group}", "catboost")
        row = {
            "feature_group": group,
            "catboost_screen_spearman": catboost_baseline,
            "catboost_drop_spearman": cat_drop,
            "catboost_refit_spearman_loss": catboost_baseline - cat_drop,
            "catboost_permutation_spearman_loss": permutation[group][
                "mean_spearman_loss"],
            "catboost_total_mean_abs_oof_shap": shap[group][
                "total_mean_abs_oof_shap"],
            "physics_token_screen_spearman": physics_baseline,
            "physics_token_drop_spearman": None,
            "physics_token_path_spearman_loss": None,
        }
        if group in TOKEN_GROUPS:
            token_drop = _screen_mean(
                results_root / "ablations" / f"physics_tokens_drop_{group}",
                "lyra_physics_tokens")
            row["physics_token_drop_spearman"] = token_drop
            row["physics_token_path_spearman_loss"] = physics_baseline - token_drop
        rows.append(row)

    rows.sort(key=lambda row: row["catboost_refit_spearman_loss"], reverse=True)
    ablations = results_root / "ablations"
    corrected = _screen_mean(
        ablations / "architecture_lyra_corrected", "lyra_corrected")
    no_late = _screen_mean(
        ablations / "architecture_lyra_corrected_nolate", "lyra_corrected_nolate")
    sequence_only = _screen_mean(
        ablations / "architecture_lyra_seqonly", "lyra_seqonly")
    tabular_mlp = _screen_mean(
        ablations / "architecture_tab_mlp", "tab_mlp")
    improvement_corrected = _screen_mean(
        results_root / "improvements", "lyra_corrected")
    historical_standard = _screen_mean(
        ablations / "historical_thermo", "catboost")
    historical_augmented = _screen_mean(
        ablations / "historical_thermo", "catboost_historical")
    modality_decomposition = {
        "corrected_multimodal_spearman": corrected,
        "matched_no_late_tabular_spearman": no_late,
        "sequence_only_alternate_pooling_spearman": sequence_only,
        "tabular_mlp_spearman": tabular_mlp,
        "late_tabular_gain_over_matched_no_late": corrected - no_late,
        "sequence_gain_over_tabular_mlp": corrected - tabular_mlp,
        "physics_token_matched_corrected_spearman": improvement_corrected,
        "physics_token_spearman": physics_baseline,
        "physics_token_gain_over_matched_corrected": (
            physics_baseline - improvement_corrected),
    }
    architecture_controls = {
        "corrected": corrected,
        "no_role": _screen_mean(
            ablations / "architecture_lyra_no_role", "lyra_no_role"),
        "unidirectional": _screen_mean(
            ablations / "architecture_lyra_unidirectional", "lyra_unidirectional"),
    }
    architecture_controls["role_embedding_gain"] = (
        corrected - architecture_controls["no_role"])
    architecture_controls["bidirectional_gain"] = (
        corrected - architecture_controls["unidirectional"])
    fusion_controls = {
        name: _screen_mean(ablations / f"fusion_{name}", name)
        for name in (
            "lyra_corrected", "lyra_corrected_crossgate",
            "lyra_corrected_tabdeep", "lyra_corrected_film",
        )
    }
    width_controls = {
        str(width): _screen_mean(ablations / f"width_{width}", "lyra_corrected")
        for width in (32, 64, 128)
    }
    strict_nucleotide = _screen_mean(
        ablations / "strict_nucleotide_only", "lyra_corrected_nolate")
    context_drops = {
        name: _screen_mean(
            ablations / f"features_{name}", "lyra_corrected")
        for name in ("no_context", "no_position", "no_thermo")
    }
    context_controls = {
        "corrected_multimodal_spearman": corrected,
        "strict_nucleotide_only_spearman": strict_nucleotide,
        "strict_nucleotide_gap": corrected - strict_nucleotide,
        "no_concentration_or_chemistry_spearman": context_drops["no_context"],
        "concentration_and_chemistry_gain": corrected - context_drops["no_context"],
        "no_mismatch_position_spearman": context_drops["no_position"],
        "mismatch_position_gain": corrected - context_drops["no_position"],
        "no_thermodynamics_or_position_spearman": context_drops["no_thermo"],
        "thermodynamics_and_position_gain": corrected - context_drops["no_thermo"],
    }
    payload = {
        "status": "complete",
        "scope": ("exploratory internal blocks 0/4/9 only; all group drops refit from "
                  "scratch and cannot alter the locked model or frozen evaluation"),
        "selected_candidate": gate["selected_candidate"],
        "catboost_baseline_spearman": catboost_baseline,
        "physics_token_baseline_spearman": physics_baseline,
        "modality_decomposition": modality_decomposition,
        "architecture_controls": architecture_controls,
        "fusion_controls": fusion_controls,
        "width_controls": width_controls,
        "context_controls": context_controls,
        "historical_thermodynamics": {
            "standard_catboost_spearman": historical_standard,
            "historical_augmented_catboost_spearman": historical_augmented,
            "historical_feature_gain": historical_augmented - historical_standard,
        },
        "rows": rows,
        "interpretation": {
            "catboost_refit_spearman_loss": (
                "Unique predictive value after all remaining feature groups can compensate."),
            "catboost_permutation_spearman_loss": (
                "Dependence of the fitted CatBoost predictor on the group; correlated groups "
                "may share credit."),
            "physics_token_path_spearman_loss": (
                "Value of exposing the group inside Lyra's sequence-token path while its late "
                "tabular branch remains unchanged."),
            "modality_decomposition": (
                "Matched three-block internal controls distinguish the dominant late-tabular "
                "contribution, the residual sequence contribution over a neural tabular model, "
                "and the incremental value of physics-aware token conditioning. These are "
                "exploratory effect sizes, not paired confidence intervals."),
            "context_controls": (
                "Matched refits isolate concentration/chemistry context, mismatch-position "
                "statistics, and the combined thermodynamic-plus-position feature family. "
                "The strict nucleotide-only gap additionally removes both the late tabular "
                "branch and concentration/chemistry token channels, so it is an interaction "
                "control rather than an additive single-group effect."),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "property_attribution.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(output / "property_attribution.csv", index=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_property_attribution(args.results_root, args.output), indent=2))


if __name__ == "__main__":
    main()
