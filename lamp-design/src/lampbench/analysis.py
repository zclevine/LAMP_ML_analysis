from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import selection_value, stratified_metrics, task_metrics

TABULAR = {"sameed_rf", "ridge", "elastic_net", "random_forest", "extra_trees",
           "hist_gradient_boosting", "gradient_boosting", "catboost", "lightgbm", "xgboost",
           "extra_trees_rank", "hist_gradient_boosting_rank", "catboost_rank",
           "lightgbm_rank", "xgboost_rank", "tabpfn", "tabpfn_rank"}
GENERIC_NEURAL = {"s4d", "cnn", "bigru", "transformer", "tab_mlp"}
LYRA_SELECTION_ALIASES = {
    # These names intentionally expose the last step of the paper-ablation
    # ladder, but currently instantiate the exact same architecture. Prefer the
    # corrected-model label so the top-two Lyra rule selects two distinct models.
    "lyra_paper_resfix_bidir": "lyra_corrected",
}


def select_finalists(screen_root: Path, output: Path, task="positive"):
    """Apply the predeclared architecture-family selection rule to internal folds."""
    rows = []
    for path in screen_root.glob(f"{task}/*/seed_*/*/result.json"):
        result = json.loads(path.read_text())
        model = result.get("model", result.get("config", {}).get("model"))
        rows.append({"model": model, "block": result["outer_block"],
                     "selection": selection_value(task, result["metrics"]["overall"])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No screen results under {screen_root}")
    ranked = frame.groupby("model", as_index=False).agg(
        mean=("selection", "mean"), sd=("selection", "std"), n=("selection", "size"),
        n_blocks=("block", "nunique")
    ).sort_values("mean", ascending=False)
    required_blocks = {"block0", "block4", "block9"}
    complete_models = {
        model for model, part in frame.groupby("model")
        if set(part.block) == required_blocks and len(part) == len(required_blocks)
    }
    eligible = ranked[ranked.model.isin(complete_models)].copy()
    chosen = []
    def take(names, count):
        part = eligible[eligible.model.isin(names)]
        seen_architectures = set()
        for model in part.model:
            architecture = LYRA_SELECTION_ALIASES.get(model, model)
            if architecture in seen_architectures:
                continue
            seen_architectures.add(architecture)
            chosen.append(architecture)
            if len(seen_architectures) == count:
                break
    take(set(eligible.model[eligible.model.str.startswith("lyra_")]), 2)
    take(set(eligible.model[eligible.model.str.startswith("mamba")]), 1)
    take(GENERIC_NEURAL, 1)
    take(TABULAR, 1)
    if "lyra_corrected" in set(eligible.model):
        chosen.append("lyra_corrected")
    if "sameed_rf" in set(eligible.model):
        chosen.append("sameed_rf")
    selected_fusion = None
    fusion_path = screen_root.parent / "fusion_screen_probe.json"
    if fusion_path.exists():
        fusion = json.loads(fusion_path.read_text())
        best = fusion["best"]
        sequence_model = best["sequence_model"]
        tabular_model = best["tabular_model"]
        if sequence_model not in complete_models or tabular_model not in complete_models:
            raise ValueError("Fusion probe selected a base model with incomplete screen coverage")
        chosen.extend([sequence_model, tabular_model])
        selected_fusion = {
            "name": f"fusion_{sequence_model}_{tabular_model}",
            "sequence_model": sequence_model,
            "tabular_model": tabular_model,
            "sequence_weight": float(best["sequence_weight"]),
            "tabular_weight": float(best["tabular_weight"]),
            "selection_metric": "screen block-macro Spearman",
            "screen_probe": str(fusion_path),
        }
    chosen = list(dict.fromkeys(chosen))
    event_path = screen_root / "events.jsonl"
    payload = {
        "task": task, "selected_models": chosen,
        "rule": ("top 2 Lyra; top Mamba; top generic neural; top tabular; "
                 "always corrected Lyra and the exact Sameed RF reference baseline; "
                 "retain both base models required by the internally selected fusion"),
        "eligibility": ("exactly block0, block4, and block9 completed once each; "
                        "architecture aliases are de-duplicated before family selection"),
        "architecture_aliases": LYRA_SELECTION_ALIASES,
        "selected_fusion": selected_fusion,
        "screen_root": str(screen_root),
        "events_sha256": (hashlib.sha256(event_path.read_bytes()).hexdigest()
                          if event_path.exists() else None),
        "ranking": ranked.to_dict(orient="records"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def load_oof(root: Path, task: str, model: str):
    by_group: dict[str, dict] = {}
    for path in sorted((root / task / model).glob("seed_*/*/predictions.npz")):
        block = path.parent.name
        try:
            data = np.load(path, allow_pickle=False)
            # Access now so legacy object arrays fail inside this try block.
            group_ids = data["group_id"].astype(str)
        except ValueError:
            # Trusted locally-generated pre-v1 artifacts used object-string
            # arrays. New outputs use Unicode and never take this path.
            data = np.load(path, allow_pickle=True)
            group_ids = data["group_id"].astype(str)
        required = {"group_id", "primer_set", "canonical_mm", "target_conc", "y", "pred"}
        if not required.issubset(data.files):
            raise ValueError(f"{path} predates paper metadata schema; rerun this fold")
        for i, gid in enumerate(group_ids):
            item = by_group.setdefault(gid, {
                "group_id": gid, "primer_set": str(data["primer_set"][i]),
                "canonical_mm": str(data["canonical_mm"][i]),
                "target_conc": float(data["target_conc"][i]), "block": block,
                "n_replicates": (int(data["n_replicates"][i])
                                 if "n_replicates" in data.files else None),
                "amp_fraction": (float(data["amp_fraction"][i])
                                 if "amp_fraction" in data.files else None),
                "y": np.asarray(data["y"][i]).tolist(), "preds": [],
            })
            if not np.allclose(np.asarray(item["y"]), np.asarray(data["y"][i]), equal_nan=True):
                raise ValueError(f"Target mismatch across seeds for {gid}")
            item["preds"].append(np.asarray(data["pred"][i]))
    if not by_group:
        raise FileNotFoundError(f"No OOF predictions for {task}/{model} in {root}")
    rows, ys, preds = [], [], []
    for item in by_group.values():
        rows.append({k: item[k] for k in ("group_id", "primer_set", "canonical_mm",
                                          "target_conc", "block", "n_replicates",
                                          "amp_fraction")})
        ys.append(item["y"])
        preds.append(np.mean(np.stack(item["preds"]), axis=0))
    return pd.DataFrame(rows), np.asarray(ys), np.asarray(preds)


def paired_cluster_bootstrap(task, frame, y, pred_a, pred_b, n_boot=5000, seed=20260813):
    rng = np.random.default_rng(seed)
    clusters = frame.primer_set.astype(str).unique()
    rows = {c: np.flatnonzero(frame.primer_set.astype(str).to_numpy() == c) for c in clusters}
    diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([rows[c] for c in chosen])
        a = selection_value(task, task_metrics(task, y[idx], pred_a[idx]))
        b = selection_value(task, task_metrics(task, y[idx], pred_b[idx]))
        diffs.append(a - b)
    diffs = np.asarray(diffs, float)
    return {
        "difference_a_minus_b": float(selection_value(task, task_metrics(task, y, pred_a))
                                      - selection_value(task, task_metrics(task, y, pred_b))),
        "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
        "p_superiority": float(np.nanmean(diffs > 0)), "n_boot": n_boot,
        "cluster": "primer_set",
    }


def paper_summary(root: Path, output: Path, task="positive",
                  expected_blocks: set[str] | None = None,
                  expected_seeds: int | None = None):
    models = sorted(p.name for p in (root / task).iterdir() if p.is_dir())
    inventories = {}
    for model in models:
        inventory = {(path.parts[-3], path.parts[-2])
                     for path in (root / task / model).glob("seed_*/*/predictions.npz")}
        inventories[model] = inventory
        blocks = {block for _, block in inventory}
        seeds = {seed for seed, _ in inventory}
        if expected_blocks is not None and blocks != expected_blocks:
            raise ValueError(f"Incomplete block coverage for {model}: {sorted(blocks)}")
        if expected_seeds is not None and len(seeds) != expected_seeds:
            raise ValueError(f"Expected {expected_seeds} seeds for {model}; found {len(seeds)}")
        if len(inventory) != len(blocks) * len(seeds):
            raise ValueError(f"Unbalanced block/seed coverage for {model}")
    if inventories and any(value != next(iter(inventories.values()))
                           for value in inventories.values()):
        raise ValueError("Models do not have identical block/seed prediction coverage")
    loaded = {m: load_oof(root, task, m) for m in models}
    summary = []
    for model, (frame, y, pred) in loaded.items():
        metrics = stratified_metrics(task, frame.reset_index(drop=True), y, pred)
        block_values = []
        for _, part in frame.groupby("block"):
            idx = part.index.to_numpy()
            block_values.append(selection_value(task, task_metrics(task, y[idx], pred[idx])))
        summary.append({
            "model": model, "task": task, "n_units": len(frame),
            "n_seeds": len([p for p in (root / task / model).glob("seed_*") if p.is_dir()]),
            "pooled_metrics": metrics, "block_macro_mean": float(np.nanmean(block_values)),
            "block_macro_sd": float(np.nanstd(block_values, ddof=1)),
        })
    comparisons = []
    for a, b in itertools.combinations(models, 2):
        fa, ya, pa = loaded[a]
        fb, yb, pb = loaded[b]
        ia = fa.set_index("group_id")
        ib = fb.set_index("group_id")
        common = ia.index.intersection(ib.index)
        if len(common) != len(ia) or len(common) != len(ib):
            continue
        posa = pd.Series(np.arange(len(fa)), index=fa.group_id).loc[common].to_numpy()
        posb = pd.Series(np.arange(len(fb)), index=fb.group_id).loc[common].to_numpy()
        if not np.allclose(ya[posa], yb[posb], equal_nan=True):
            raise ValueError(f"Target mismatch between {a} and {b}")
        comparisons.append({"model_a": a, "model_b": b,
                            **paired_cluster_bootstrap(task, fa.iloc[posa].reset_index(drop=True),
                                                       ya[posa], pa[posa], pb[posb])})
    output.mkdir(parents=True, exist_ok=True)
    payload = {"root": str(root), "task": task, "models": summary,
               "fold_inventory": {m: sorted([list(x) for x in values])
                                  for m, values in inventories.items()},
               "paired_comparisons": comparisons}
    (output / f"{task}_paper_summary.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame([{
        "model": x["model"], "n_units": x["n_units"],
        "pooled_selection": selection_value(task, x["pooled_metrics"]["overall"]),
        "block_macro_mean": x["block_macro_mean"], "block_macro_sd": x["block_macro_sd"],
    } for x in summary]).sort_values("pooled_selection", ascending=False).to_csv(
        output / f"{task}_leaderboard.csv", index=False)
    return payload
