"""The four-model Fusion, for ``lamp-design evaluate``.

Fusion is a frozen weighted blend of four predictors:

    0.30 Lyra  +  0.20 TabPFN  +  0.30 CatBoost  +  0.20 AFT

trained on all three chemistries (n=34,283; blocks 0-18 plus Systematic,
block 19 stripped), and shipped in
``lamp_design/weights/fusion_tp60c_allmix/``.

WHY THIS IS EVALUATE-ONLY
-------------------------
Measured on one L4, same workload (147,620 predictions), dedup off:

    Simplified Lyra     72.7 s
    Fusion            2116.9 s        29x slower

TabPFN is 91% of that. It is *prior-fitted*: calling ``.fit()`` stores the
34,283-row training matrix rather than learning weights, and every prediction
attends over the whole thing. It also cannot be handed a large batch --
147,620 rows CUDA-OOMs a 23 GB card -- so it must be chunked, and there is no
context caching between calls (78.4 / 79.6 / 79.3 s for three identical calls
in one process). The fixed cost is therefore paid per chunk, which is why a
design run projects to ~1.9 h on GPU and ~14 days on CPU.

Evaluate scores a handful of supplied sets: one call, a few hundred rows,
roughly a minute and a half. That is the workload Fusion was built for.

So there is no ``--model`` flag on ``design`` at all -- not a cap, not a
warning. The option does not exist, so nobody can start a two-hour job by
accident.

ACCURACY, BOTH WAYS
-------------------
Fusion is better in-distribution and worse out of it. On block 19 (held out
from training, same corpus) it beats every one of its own branches; on
held-out *pathogens* the Lyra branch alone wins two of three:

    panel        lyra     fusion
    block 19     0.6729   0.7016     <- fusion
    mpox         0.6769   0.7017     <- fusion
    Borrelia     0.7317   0.7255     <- lyra
    LASV         0.5361   0.5132     <- lyra

Designing primers for a new target is the out-of-distribution case, which is
the other half of why Simplified Lyra remains the default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from lamp_design import paths
from lamp_design.types import (
    GLOBAL_PHYSICS_FEATURES,
    PHYSICS_FEATURES,
    PRIMER_TYPE_ORDER,
    Condition,
    ThermoTable,
)

#: Where the bundle lives. Sibling of the Simplified Lyra checkpoints.
BUNDLED_FUSION: Path = paths.PACKAGE_ROOT / "weights" / "fusion_tp60c_allmix"

#: lampbench's own primer order, which is NOT ours. Ours is layout order
#: (F3, F2, FL, F1c, B1c, BL, B2, B3); lampbench pairs each forward primer
#: with its reverse counterpart. Same eight names, different sequence -- so
#: everything below keys by NAME. Indexing by position would silently swap
#: FL with F1c and B1c with BL, and nothing would raise.
LAMPBENCH_PRIMER_ORDER: tuple = ("F3", "F2", "F1c", "FL", "B3", "B2", "B1c", "BL")

__all__ = [
    "BUNDLED_FUSION",
    "FusionModel",
    "frame_from_thermo",
    "load_fusion",
]


def frame_from_thermo(
    thermo: ThermoTable,
    conditions: Sequence[Condition],
    primer_sequences: Dict[str, Dict[str, str]],
):
    """Build the raw frame lampbench's feature pipeline expects.

    One row per (pair, condition). ``primer_sequences`` maps a set_id to a
    ``{primer_type: sequence}`` dict -- keyed by name, never by position,
    because lampbench orders its eight primers differently from us.

    Only the *raw* columns are produced here. lampbench derives the rest --
    the 32 mismatch-position features, the mix one-hots and log10_conc -- in
    ``lampbench.data``, from the primer and target sequences we supply. We do
    not reimplement that: a second implementation of a feature the model was
    trained on is a silent-wrongness risk with no upside.
    """
    import pandas as pd

    our_index = {name: i for i, name in enumerate(PRIMER_TYPE_ORDER)}
    phys_index = {name: i for i, name in enumerate(PHYSICS_FEATURES)}
    glob_index = {name: i for i, name in enumerate(GLOBAL_PHYSICS_FEATURES)}

    rows = []
    for pair_idx, (set_id, target_name) in enumerate(
        zip(thermo.set_ids, thermo.target_names)
    ):
        primers = primer_sequences[set_id]
        for condition in conditions:
            row = {
                "set_id": set_id,
                "target_name": target_name,
                "master_mix": condition.master_mix,
                "target_conc": float(condition.target_conc),
                # lampbench's loader expects these; they are not features.
                "block": "0",
                "source": "lamp_design",
            }
            for primer_type in LAMPBENCH_PRIMER_ORDER:
                j = our_index[primer_type]
                row[f"{primer_type}_sequence"] = primers[primer_type]
                row[f"{primer_type}_target_sequence"] = str(
                    thermo.target_windows[pair_idx, j]
                )
                for feature in PHYSICS_FEATURES:
                    row[f"{primer_type}_{feature}"] = float(
                        thermo.values[pair_idx, j, phys_index[feature]]
                    )
            for feature in GLOBAL_PHYSICS_FEATURES:
                row[feature] = float(thermo.globals_[pair_idx, glob_index[feature]])
            rows.append(row)
    return pd.DataFrame(rows)


class FusionModel:
    """The four branches plus the frozen blend.

    Load once and reuse: TabPFN's ``fit`` stores a 34,283-row matrix, and
    every ``predict`` re-encodes it, so constructing this per batch would pay
    that cost repeatedly.
    """

    def __init__(self, root: Path = BUNDLED_FUSION, device: str = "cpu") -> None:
        self.root = Path(root)
        self.device = device
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.weights: Dict[str, float] = self.manifest["blend_weights"]
        self.feature_names: List[str] = self.manifest["feature_names"]
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import xgboost as xgb
        from catboost import CatBoostRegressor
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self._catboost = CatBoostRegressor()
        self._catboost.load_model(str(self.root / "catboost.cbm"))

        self._aft = xgb.Booster()
        self._aft.load_model(str(self.root / "aft.json"))
        self._aft.set_param({"device": self.device})

        # ModelVersion.V2 is not a style choice. TabPFN 2.5 and newer are
        # released under NON-COMMERCIAL licences; V2 is Apache 2.0 plus an
        # attribution clause, which is what makes this redistributable.
        context = np.load(self.root / "tabpfn_context.npz")
        self._tabpfn = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=self.device,
            n_estimators=4,
            random_state=self.manifest["seed"],
            ignore_pretraining_limits=True,
            memory_saving_mode="auto",
        )
        self._tabpfn.fit(context["x_train"], context["y_train"])
        self._loaded = True

    def predict(self, frame) -> np.ndarray:
        """Blended predicted Tp, one value per row of ``frame``.

        ``frame`` is the output of :func:`frame_from_thermo` after lampbench's
        feature derivation. TabPFN is called **once** for the whole batch:
        each call costs ~78 s regardless of size, so splitting the work would
        multiply that, not amortise it.
        """
        import xgboost as xgb
        from lampbench.data import TABULAR_FEATURES, tabular_frame_for_ablation

        if list(TABULAR_FEATURES) != self.feature_names:
            raise RuntimeError(
                "lampbench's TABULAR_FEATURES no longer matches the order the "
                "shipped Fusion weights were fitted with. These models take a "
                "bare float matrix, so a reordered frame predicts nonsense "
                "rather than raising. Refusing to run."
            )
        self._load()

        x = frame[self.feature_names].to_numpy(np.float32)
        low, high = self.manifest["aft_clip"]
        branches = {
            "catboost": np.asarray(self._catboost.predict(x), np.float64),
            "aft": np.clip(
                self._aft.predict(
                    xgb.DMatrix(
                        tabular_frame_for_ablation(frame, "none")[self.feature_names]
                        .to_numpy(np.float32)
                    )
                ),
                low,
                high,
            ).astype(np.float64),
            "tabpfn": np.asarray(self._tabpfn.predict(x), np.float64),
            "lyra": self._predict_lyra(frame),
        }
        blended = np.zeros(len(frame), dtype=np.float64)
        for name, weight in self.weights.items():
            blended += weight * branches[name]
        return blended

    def _predict_lyra(self, frame) -> np.ndarray:
        """Mean over the three Fusion Lyra seeds.

        This is not the Simplified Lyra we ship as the default: it is
        ``lyra_physics_tokens``, a different architecture with a 23-channel
        input (13 nucleotide/context + 10 physics) against our 19. Its
        encoder comes from lampbench, which is the reason lampbench is
        vendored rather than reimplemented.
        """
        import torch
        from lampbench.extended import _lyra_config, _predict_lyra
        from lampbench.models import build_model
        from lampbench.train import Standardizer

        target = np.zeros(len(frame), dtype=np.float32)
        preds = []
        for seed_dir in sorted((self.root / "lyra").glob("seed_*")):
            checkpoint = torch.load(
                seed_dir / "model.pt", map_location="cpu", weights_only=False
            )
            config = _lyra_config("none", checkpoint["seed"], self.manifest["task"])
            standardizer = Standardizer(
                np.asarray(checkpoint["standardizer"]["mean"], np.float32),
                np.asarray(checkpoint["standardizer"]["std"], np.float32),
            )
            model = build_model(
                config.model, config.task, config.d_model, config.dropout
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(self.device)
            _, pred = _predict_lyra(
                model,
                frame,
                target,
                standardizer,
                config,
                checkpoint["target_mean"],
                checkpoint["target_std"],
            )
            preds.append(np.asarray(pred, np.float64))
        return np.mean(preds, axis=0)


def load_fusion(root: Optional[Path] = None, device: str = "cpu") -> FusionModel:
    """Construct a :class:`FusionModel`, with a clear error if deps are absent."""
    try:
        import catboost  # noqa: F401
        import tabpfn  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"the Fusion model needs catboost, xgboost and tabpfn ({exc}). "
            'Install them with: pip install -e ".[fusion]"'
        ) from exc
    return FusionModel(root or BUNDLED_FUSION, device=device)
