"""Checkpoint loading and ensemble inference for Simplified Lyra.

The shipped model is a 5-seed ensemble (seeds 0, 1, 2, 3, 42) of a
bidirectional PGC + S4D Lyra, ``d_model=64``, ``dropout=0.2``, 101,121
parameters each. Predictions are the arithmetic mean over seeds.

Weights are **not** bundled with this package. Point ``LAMP_DESIGN_WEIGHTS`` (or
``--weights``) at a directory containing ``seed_*/`` subtrees, each holding one
``best_model_*.pt``.

Each checkpoint carries everything needed for inference:

``model_state_dict``
    weights; loads with ``strict=True`` against the vendored architecture.
``model_config``
    ``d_input``/``d_output``/``d_model``/``dropout``/``ssm``/``bidirectional``.
``physics_standardizer``
    ``mean``/``std`` over the 34 physics features, **fit on the training set**.
    Never recompute these from inference data -- see ``encoding.transform_physics``.
``label_mean`` / ``label_std`` / ``unscale_preds``
    the model regresses z-scored Tp, so raw output is multiplied back up.

## Padding, and why we diverge from training

Training's ``collate_fn`` zero-pads each batch to its longest member with **no
mask**, and ``Lyra.forward`` ends in ``x.mean(dim=1)`` -- an unmasked mean over
the padded length. Padded positions are zeros at the *input*, but the network's
biases make them non-zero in the pooled hidden state, so **a prediction depends
on what else was in its batch**.

That is tolerable when scoring a fixed evaluation set and unacceptable when
ranking candidates against each other, which is this engine's whole job. Two
modes, therefore:

``"group_by_length"`` (default)
    batch only equal-length rows, so no padding ever happens and a candidate's
    prediction is independent of the rest of the run. Deterministic, and faster
    -- deterministic geometries mean few distinct lengths.
``"batch_max"``
    reproduce training exactly: pad to the batch maximum in the given row
    order. Use this to reproduce published predictions.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from lamp_design.models.fastpath import enable_kernel_cache
from lamp_design.models.lyra.lyra_base import Lyra

#: Seeds of the shipped ensemble, in the order the training config lists them.
ENSEMBLE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 42)

#: Training censored Tp at 60 min, so predictions outside [0, 60] are
#: extrapolation past anything the model ever saw.
TP_CLIP_MIN = 0.0
TP_CLIP_MAX = 60.0

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")

PadMode = str  # "group_by_length" | "batch_max"


class WeightsNotFound(RuntimeError):
    """Raised when the checkpoint directory cannot be located or is incomplete."""


def resolve_weights_dir(explicit: Optional[str | os.PathLike] = None) -> Path:
    """Find the checkpoint root: explicit argument, else ``LAMP_DESIGN_WEIGHTS``."""
    candidate = explicit or os.environ.get("LAMP_DESIGN_WEIGHTS")
    if not candidate:
        raise WeightsNotFound(
            "No model weights location given. Pass --weights <dir> or set "
            "LAMP_DESIGN_WEIGHTS to a directory containing seed_*/ subdirectories, "
            "each with one best_model_*.pt. Weights are intentionally not "
            "distributed with this package."
        )
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise WeightsNotFound(f"Weights directory does not exist: {path}")
    return path


def discover_checkpoints(weights_dir: Path) -> Dict[int, Path]:
    """Map seed -> checkpoint file for every ``seed_*/`` under ``weights_dir``."""
    found: Dict[int, Path] = {}
    for child in sorted(weights_dir.iterdir()):
        if not child.is_dir():
            continue
        match = _SEED_DIR_RE.match(child.name)
        if not match:
            continue
        pts = sorted(child.rglob("*.pt")) + sorted(child.rglob("*.pth"))
        if not pts:
            continue
        if len(pts) > 1:
            best = [p for p in pts if "best_model" in p.name]
            pts = best or pts
        found[int(match.group(1))] = pts[0]
    if not found:
        raise WeightsNotFound(
            f"No seed_*/ checkpoints under {weights_dir}. Expected a layout like "
            "<dir>/seed_0/.../best_model_*.pt"
        )
    return found


@dataclass
class LoadedModel:
    """One seed: the network plus the metadata needed to interpret its output."""

    seed: int
    model: Lyra
    physics_standardizer: Dict[str, np.ndarray]
    label_mean: float
    label_std: float
    unscale: bool
    config: Dict[str, object]
    path: Path


def load_checkpoint(
    path: Path, device: str = "cpu", *, cache_kernels: bool = True
) -> LoadedModel:
    """Load one checkpoint, asserting a strict state_dict match.

    ``cache_kernels`` memoizes the S4D convolution kernels, which are constant
    once the parameters are frozen -- a large speedup at identical output. See
    :mod:`lamp_design.models.fastpath`. Pass ``False`` to run the stock
    architecture unmodified.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model_config", "physics_standardizer"):
        if key not in ck:
            raise WeightsNotFound(f"{path} is missing required key {key!r}")

    cfg = ck["model_config"]
    model = Lyra(
        d_input=cfg["d_input"],
        d_output=cfg["d_output"],
        d_model=cfg["d_model"],
        dropout=cfg.get("dropout", 0.2),
        ssm=cfg.get("ssm", "s4d"),
        bidirectional=cfg.get("bidirectional", True),
    )
    # strict=True on purpose: a silently-partial load is the exact failure mode
    # that produces plausible-looking wrong predictions.
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    if cache_kernels and not enable_kernel_cache(model):
        raise WeightsNotFound(
            f"{path}: kernel caching found no S4D modules to patch. The "
            "architecture is not what this code expects; refusing to proceed "
            "silently. Pass cache_kernels=False to bypass."
        )

    ps = ck["physics_standardizer"]
    standardizer = {
        "mean": np.asarray(ps["mean"], dtype=np.float32),
        "std": np.asarray(ps["std"], dtype=np.float32),
    }
    return LoadedModel(
        seed=int(ck.get("seed", -1)),
        model=model,
        physics_standardizer=standardizer,
        label_mean=float(ck.get("label_mean", 0.0)),
        label_std=float(ck.get("label_std", 1.0)),
        unscale=bool(ck.get("unscale_preds", True)),
        config=dict(cfg),
        path=path,
    )


class LyraEnsemble:
    """The 5-seed ensemble. Averages per-seed predictions in minutes."""

    def __init__(self, members: Sequence[LoadedModel], *, device: str = "cpu"):
        if not members:
            raise WeightsNotFound("Cannot build an ensemble with no members")
        self.members = list(members)
        self.device = device
        self._check_consistency()

    @classmethod
    def load(
        cls,
        weights_dir: Optional[str | os.PathLike] = None,
        *,
        device: str = "cpu",
        require_seeds: Optional[Sequence[int]] = ENSEMBLE_SEEDS,
        cache_kernels: bool = True,
    ) -> "LyraEnsemble":
        root = resolve_weights_dir(weights_dir)
        found = discover_checkpoints(root)
        if require_seeds is not None:
            missing = [s for s in require_seeds if s not in found]
            if missing:
                raise WeightsNotFound(
                    f"Expected seeds {list(require_seeds)} under {root} but "
                    f"{missing} are missing (found {sorted(found)}). The published "
                    "model is a 5-seed ensemble; running a subset changes its "
                    "predictions. Pass require_seeds=None to override deliberately."
                )
            order = list(require_seeds)
        else:
            order = sorted(found)
        return cls(
            [
                load_checkpoint(found[s], device=device, cache_kernels=cache_kernels)
                for s in order
            ],
            device=device,
        )

    def _check_consistency(self) -> None:
        """All seeds must share the standardizer and label scaling."""
        first = self.members[0]
        for other in self.members[1:]:
            for key in ("mean", "std"):
                if not np.allclose(
                    first.physics_standardizer[key], other.physics_standardizer[key], atol=1e-6
                ):
                    raise WeightsNotFound(
                        f"physics_standardizer['{key}'] differs between seed "
                        f"{first.seed} and seed {other.seed}. These checkpoints were "
                        "fit on different training splits and must not be ensembled."
                    )
            if not np.isclose(first.label_mean, other.label_mean, atol=1e-6) or not np.isclose(
                first.label_std, other.label_std, atol=1e-6
            ):
                raise WeightsNotFound(
                    f"label scaling differs between seed {first.seed} and "
                    f"seed {other.seed}; refusing to ensemble."
                )

    @property
    def physics_standardizer(self) -> Dict[str, np.ndarray]:
        return self.members[0].physics_standardizer

    @property
    def d_input(self) -> int:
        return int(self.members[0].config["d_input"])

    def _to_minutes(self, raw: np.ndarray, member: LoadedModel) -> np.ndarray:
        if not member.unscale:
            return raw
        return raw * member.label_std + member.label_mean

    def predict(
        self,
        rows: Sequence[np.ndarray],
        *,
        pad_mode: PadMode = "group_by_length",
        batch_size: int = 512,
        clip: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict Tp in minutes for each encoded row.

        ``rows`` is a sequence of ``(L_i, 19)`` float arrays; ``L_i`` may vary.
        Returns ``(mean_minutes, std_minutes)``, both shape ``(len(rows),)``,
        where the std is the spread *across ensemble seeds* and is a measure of
        model disagreement, not of experimental uncertainty.
        """
        if len(rows) == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        for i, r in enumerate(rows):
            if r.ndim != 2 or r.shape[1] != self.d_input:
                raise ValueError(
                    f"rows[{i}] has shape {r.shape}; expected (L, {self.d_input})"
                )

        per_seed = np.zeros((len(self.members), len(rows)), dtype=np.float64)
        for m_idx, member in enumerate(self.members):
            raw = self._run_one(member, rows, pad_mode=pad_mode, batch_size=batch_size)
            per_seed[m_idx] = self._to_minutes(raw, member)

        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0, ddof=0)
        if clip:
            mean = np.clip(mean, TP_CLIP_MIN, TP_CLIP_MAX)
        return mean.astype(np.float32), std.astype(np.float32)

    def _run_one(
        self,
        member: LoadedModel,
        rows: Sequence[np.ndarray],
        *,
        pad_mode: PadMode,
        batch_size: int,
    ) -> np.ndarray:
        out = np.zeros(len(rows), dtype=np.float64)
        for idx_batch in self._batches(rows, pad_mode=pad_mode, batch_size=batch_size):
            lengths = [rows[i].shape[0] for i in idx_batch]
            max_len = max(lengths)
            stacked = np.zeros((len(idx_batch), max_len, self.d_input), dtype=np.float32)
            for slot, row_idx in enumerate(idx_batch):
                row = rows[row_idx]
                stacked[slot, : row.shape[0]] = row
            tensor = torch.from_numpy(stacked).to(self.device)
            with torch.no_grad():
                pred = member.model(tensor).squeeze(-1).detach().cpu().numpy()
            pred = np.atleast_1d(pred)
            for slot, row_idx in enumerate(idx_batch):
                out[row_idx] = float(pred[slot])
        return out

    @staticmethod
    def _batches(
        rows: Sequence[np.ndarray], *, pad_mode: PadMode, batch_size: int
    ) -> List[List[int]]:
        """Index batches. See the module docstring on why the default groups."""
        if pad_mode == "batch_max":
            # Training-faithful: consecutive slices of the given order, padded
            # to each batch's own maximum.
            return [
                list(range(i, min(i + batch_size, len(rows))))
                for i in range(0, len(rows), batch_size)
            ]
        if pad_mode != "group_by_length":
            raise ValueError(
                f"unknown pad_mode {pad_mode!r}; expected 'group_by_length' or 'batch_max'"
            )
        by_len: Dict[int, List[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_len[row.shape[0]].append(i)
        batches: List[List[int]] = []
        for length in sorted(by_len):
            idxs = by_len[length]
            batches.extend(
                idxs[i : i + batch_size] for i in range(0, len(idxs), batch_size)
            )
        return batches


__all__ = [
    "ENSEMBLE_SEEDS",
    "TP_CLIP_MAX",
    "TP_CLIP_MIN",
    "WeightsNotFound",
    "LoadedModel",
    "LyraEnsemble",
    "discover_checkpoints",
    "load_checkpoint",
    "resolve_weights_dir",
]
