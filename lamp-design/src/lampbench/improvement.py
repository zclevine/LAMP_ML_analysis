from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_oof
from .fusion import _aligned_predictions, materialize_fusion, probe_fusion
from .metrics import selection_value
from .data import load_units, split_confirmatory
from .runner import (IMPROVEMENT_MODELS, IMPROVEMENT_REFERENCES, TRAIN_FILES,
                     run_matrix, summarize)


SCREEN_BLOCKS = {"block0", "block4", "block9"}


def select_screen(improvements_root: Path, base_selection_file: Path,
                  output: Path) -> dict:
    """Lock one physics candidate using only the declared internal screen folds."""
    expected_models = IMPROVEMENT_MODELS + IMPROVEMENT_REFERENCES
    rows = []
    for path in improvements_root.glob("positive/*/seed_0/*/result.json"):
        result = json.loads(path.read_text())
        model = result.get("model", result.get("config", {}).get("model"))
        rows.append({
            "model": model,
            "block": result["outer_block"],
            "spearman": selection_value("positive", result["metrics"]["overall"]),
        })
    frame = pd.DataFrame(rows)
    inventory = {(row.model, row.block) for row in frame.itertuples()}
    expected = {(model, block) for model in expected_models for block in SCREEN_BLOCKS}
    if inventory != expected or len(frame) != len(expected):
        raise ValueError(
            f"Incomplete improvement screen: missing={sorted(expected - inventory)}, "
            f"unexpected={sorted(inventory - expected)}, rows={len(frame)}")
    ranking = frame.groupby("model", as_index=False).agg(
        mean_spearman=("spearman", "mean"),
        sd_spearman=("spearman", "std"),
        min_spearman=("spearman", "min"),
        n_blocks=("block", "nunique"),
    ).sort_values("mean_spearman", ascending=False)
    candidate_ranking = ranking[ranking.model.isin(IMPROVEMENT_MODELS)]
    reference_ranking = ranking[ranking.model.isin(IMPROVEMENT_REFERENCES)]
    winner = candidate_ranking.iloc[0]
    reference = reference_ranking.iloc[0]

    base_selection = json.loads(base_selection_file.read_text())
    if "lyra_corrected" not in base_selection.get("model_configs", {}):
        raise ValueError("Base selection lacks the tuned corrected-Lyra configuration")
    candidate_selection = copy.deepcopy(base_selection)
    candidate_selection["selected_models"] = [str(winner.model)]
    candidate_selection.setdefault("model_configs", {})[str(winner.model)] = copy.deepcopy(
        base_selection["model_configs"]["lyra_corrected"])
    candidate_selection["improvement_screen"] = str(output / "screen_selection.json")

    output.mkdir(parents=True, exist_ok=True)
    candidate_selection_path = output / "candidate_selection.json"
    candidate_selection_path.write_text(json.dumps(candidate_selection, indent=2))
    payload = {
        "status": "screen_locked",
        "frozen_data_status": "unopened",
        "scope": "internal blocks 0, 4, and 9 only",
        "rule": (
            "Select the physics-conditioned candidate with the highest three-block mean "
            "Spearman; evaluate it once on all ten internal blocks, using the seven "
            "non-screen blocks as the confirmatory internal comparison."
        ),
        "evaluated_candidates": IMPROVEMENT_MODELS,
        "reference_models": IMPROVEMENT_REFERENCES,
        "selected_candidate": str(winner.model),
        "selected_candidate_mean": float(winner.mean_spearman),
        "best_reference": str(reference.model),
        "best_reference_mean": float(reference.mean_spearman),
        "screen_margin": float(winner.mean_spearman - reference.mean_spearman),
        "ranking": ranking.to_dict("records"),
        "candidate_selection_file": str(candidate_selection_path),
    }
    (output / "screen_selection.json").write_text(json.dumps(payload, indent=2))
    ranking.to_csv(output / "screen_leaderboard.csv", index=False)
    return payload


def run_selected_full(data_dir: Path, candidate_selection_file: Path,
                      output: Path) -> None:
    """Evaluate the single screen-locked candidate on the full internal matrix."""
    selection = json.loads(candidate_selection_file.read_text())
    models = selection.get("selected_models", [])
    if len(models) != 1 or models[0] not in IMPROVEMENT_MODELS:
        raise ValueError(f"Expected one locked improvement candidate; found {models}")
    model = models[0]
    config = selection.get("model_configs", {}).get(model)
    if not config:
        raise ValueError(f"Candidate selection lacks a configuration for {model}")
    units, _ = split_confirmatory(load_units(data_dir, TRAIN_FILES))
    run_matrix(
        units, output, "improvement_full", [model], ["positive"],
        [f"block{i}" for i in range(10)], [0, 1, 2], 100, 18, 64, 256,
        config_overrides={model: config},
    )
    summarize(output)


def _block_macro(frame: pd.DataFrame, y, pred, blocks: set[str]) -> tuple[float, dict]:
    values = {}
    for block in sorted(blocks):
        idx = np.flatnonzero(frame.block.astype(str).to_numpy() == block)
        values[block] = float(selection_value(
            "positive", {"spearman": float(pd.Series(y[idx]).corr(
                pd.Series(np.asarray(pred)[idx]), method="spearman"))}))
    return float(np.nanmean(list(values.values()))), values


def lock_full(candidate_full_root: Path, existing_full_root: Path,
              improvements_root: Path, screen_root: Path,
              base_selection_file: Path, gate_root: Path) -> dict:
    """Make the one predeclared seven-block decision and update the finalist lock."""
    locked_path = gate_root / "selection.json"
    if locked_path.exists():
        return json.loads(locked_path.read_text())
    results_root = existing_full_root.parent
    for frozen in (results_root / "confirmatory" / "EVALUATION_OPENED.json",
                   results_root / "external" / "EVALUATION_OPENED.json"):
        if frozen.exists():
            raise RuntimeError(f"Refusing post-frozen improvement lock: {frozen} exists")
    screen = json.loads((gate_root / "screen_selection.json").read_text())
    candidate = screen["selected_candidate"]
    inventory = {
        (path.parts[-3], path.parts[-2])
        for path in (candidate_full_root / "positive" / candidate).glob(
            "seed_*/*/predictions.npz")
    }
    expected = {(f"seed_{seed}", f"block{block}")
                for seed in range(3) for block in range(10)}
    if inventory != expected:
        raise ValueError(
            f"Candidate full matrix is incomplete: missing={sorted(expected - inventory)}, "
            f"unexpected={sorted(inventory - expected)}")

    candidate_frame, candidate_y, candidate_pred = load_oof(
        candidate_full_root, "positive", candidate)
    holdout = {f"block{i}" for i in (1, 2, 3, 5, 6, 7, 8)}
    candidate_macro, candidate_blocks = _block_macro(
        candidate_frame, candidate_y, candidate_pred, holdout)
    selection = json.loads(base_selection_file.read_text())
    lyra_references = [
        model for model in selection["selected_models"]
        if model.startswith("lyra_")
        and (existing_full_root / "positive" / model).exists()
    ]
    reference_rows = []
    for model in lyra_references:
        frame, y, pred = load_oof(existing_full_root, "positive", model)
        macro, values = _block_macro(frame, y, pred, holdout)
        reference_rows.append({"model": model, "block_macro_spearman": macro,
                               "block_spearman": values})
    best_reference = max(reference_rows, key=lambda row: row["block_macro_spearman"])

    fusion_probe = probe_fusion(
        improvements_root, screen_root, "positive", [candidate], "catboost")
    best_candidate_fusion = fusion_probe["best"]
    _, fusion_y, fusion_seq, fusion_tab = _aligned_predictions(
        candidate_full_root, existing_full_root, "positive", candidate, "catboost")
    fusion_frame, _, _, _ = _aligned_predictions(
        candidate_full_root, existing_full_root, "positive", candidate, "catboost")
    weight = float(best_candidate_fusion["sequence_weight"])
    candidate_fusion_pred = weight * fusion_seq + (1 - weight) * fusion_tab
    candidate_fusion_macro, candidate_fusion_blocks = _block_macro(
        fusion_frame, fusion_y, candidate_fusion_pred, holdout)
    old_fusion = selection["selected_fusion"]["name"]
    old_frame, old_y, old_pred = load_oof(existing_full_root, "positive", old_fusion)
    old_fusion_macro, old_fusion_blocks = _block_macro(
        old_frame, old_y, old_pred, holdout)

    standalone_improves = candidate_macro > best_reference["block_macro_spearman"]
    fusion_improves = candidate_fusion_macro > old_fusion_macro
    retain_candidate = standalone_improves or fusion_improves
    replace_fusion = fusion_improves and retain_candidate
    candidate_selection = json.loads(
        (gate_root / "candidate_selection.json").read_text())
    pre_selection = copy.deepcopy(selection)
    gate_root.mkdir(parents=True, exist_ok=True)
    (gate_root / "pre_improvement_selection.json").write_text(
        json.dumps(pre_selection, indent=2))

    selected_fusion = selection["selected_fusion"]
    if retain_candidate:
        source = candidate_full_root / "positive" / candidate
        destination = existing_full_root / "positive" / candidate
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination)
        selection["selected_models"] = list(dict.fromkeys(
            selection["selected_models"] + [candidate]))
        selection.setdefault("model_configs", {})[candidate] = copy.deepcopy(
            candidate_selection["model_configs"][candidate])
        if replace_fusion:
            archive = results_root / "archive" / "pre_improvement_fusion" / old_fusion
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(existing_full_root / "positive" / old_fusion), str(archive))
            selected_fusion = {
                "name": f"fusion_{candidate}_catboost",
                "sequence_model": candidate,
                "tabular_model": "catboost",
                "sequence_weight": weight,
                "tabular_weight": float(1 - weight),
                "selection_metric": "improvement-screen block-macro Spearman",
                "screen_probe": str(gate_root / "candidate_fusion_probe.json"),
            }
            materialize_fusion(existing_full_root, "positive", selected_fusion)
            selection["selected_fusion"] = selected_fusion
        selection["improvement_gate"] = str(locked_path)
        base_selection_file.write_text(json.dumps(selection, indent=2))

    (gate_root / "candidate_fusion_probe.json").write_text(
        json.dumps(fusion_probe, indent=2))
    payload = {
        "status": "locked",
        "frozen_data_status": "unopened",
        "evaluated_candidates": screen["evaluated_candidates"],
        "reference_models": screen["reference_models"],
        "selected_candidate": candidate,
        "decision": ("retain_and_replace_fusion" if replace_fusion
                     else "retain_standalone" if retain_candidate
                     else "reject"),
        "rule": (
            "Retain the screen-selected candidate iff its seven-block macro Spearman "
            "beats the strongest existing Lyra or its screen-locked CatBoost fusion "
            "beats the existing fixed fusion; replace the fusion only for the latter."
        ),
        "holdout_blocks": sorted(holdout),
        "candidate_block_macro_spearman": candidate_macro,
        "candidate_block_spearman": candidate_blocks,
        "lyra_references": reference_rows,
        "best_lyra_reference": best_reference,
        "standalone_improves": standalone_improves,
        "candidate_fusion": {
            **best_candidate_fusion,
            "holdout_block_macro_spearman": candidate_fusion_macro,
            "holdout_block_spearman": candidate_fusion_blocks,
        },
        "existing_fusion": {
            "name": old_fusion,
            "holdout_block_macro_spearman": old_fusion_macro,
            "holdout_block_spearman": old_fusion_blocks,
        },
        "fusion_improves": fusion_improves,
        "selected_fusion": selected_fusion,
        "updated_finalists": selection["selected_models"] if retain_candidate
                              else pre_selection["selected_models"],
    }
    locked_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "full", "lock"])
    parser.add_argument("--improvements-root", type=Path)
    parser.add_argument("--base-selection-file", type=Path)
    parser.add_argument("--candidate-selection-file", type=Path)
    parser.add_argument("--candidate-full-root", type=Path)
    parser.add_argument("--existing-full-root", type=Path)
    parser.add_argument("--screen-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "select":
        if not args.improvements_root or not args.base_selection_file:
            parser.error("select requires --improvements-root and --base-selection-file")
        print(json.dumps(select_screen(
            args.improvements_root, args.base_selection_file, args.output), indent=2))
        return
    if args.mode == "lock":
        required = [args.candidate_full_root, args.existing_full_root,
                    args.improvements_root, args.screen_root, args.base_selection_file]
        if any(value is None for value in required):
            parser.error(
                "lock requires --candidate-full-root, --existing-full-root, "
                "--improvements-root, --screen-root, and --base-selection-file")
        print(json.dumps(lock_full(
            args.candidate_full_root, args.existing_full_root,
            args.improvements_root, args.screen_root,
            args.base_selection_file, args.output), indent=2))
        return
    if not args.data_dir or not args.candidate_selection_file:
        parser.error("full requires --data-dir and --candidate-selection-file")
    run_selected_full(args.data_dir, args.candidate_selection_file, args.output)


if __name__ == "__main__":
    main()
