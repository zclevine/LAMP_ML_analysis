from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

PRIMER_TYPES = ["B1c", "B2", "B3", "BL", "F1c", "F2", "F3", "FL"]
PRIMER_ORDER = ["F3", "F2", "F1c", "FL", "B3", "B2", "B1c", "BL"]
ASSAY_SEQUENCE_COLUMNS = (
    [f"{p}_sequence" for p in PRIMER_ORDER]
    + [f"{p}_target_sequence" for p in PRIMER_ORDER]
)
MM_CLASSES = ["SS", "BST", "WS"]
EXCLUDED_MASTER_MIX_FORMULATIONS = frozenset({"BSTXT-MM", "BSTXT_MM"})
THERMO_FEATURES = [
    f"{p}_{f}" for p in PRIMER_ORDER
    for f in ["mismatches", "gc_fraction", "tm", "dg_ViennaRNA"]
] + ["total_mismatches", "mean_dg_ViennaRNA"]
META_FEATURES = ["log10_conc", "mm_SS", "mm_BST", "mm_WS"]
POSITION_FEATURES = [
    f"{p}_{name}" for p in PRIMER_ORDER
    for name in ("mm_3p3", "mm_3p5", "mm_5p5", "mm_pos_weighted")
]
TABULAR_FEATURES = THERMO_FEATURES + POSITION_FEATURES + META_FEATURES
TABULAR_FEATURE_GROUPS = {
    "mismatch_count": [f"{p}_mismatches" for p in PRIMER_ORDER] + ["total_mismatches"],
    "gc_fraction": [f"{p}_gc_fraction" for p in PRIMER_ORDER],
    "melting_temperature": [f"{p}_tm" for p in PRIMER_ORDER],
    "vienna_dg": [f"{p}_dg_ViennaRNA" for p in PRIMER_ORDER] + ["mean_dg_ViennaRNA"],
    "mismatch_position": list(POSITION_FEATURES),
    "concentration": ["log10_conc"],
    "chemistry": [x for x in META_FEATURES if x != "log10_conc"],
}
TOKEN_PHYSICS_GROUP_CHANNELS = {
    # Indices are relative to the ten physics channels appended after the
    # thirteen nucleotide/context channels in ``LampDataset``.
    "mismatch_count": (0, 8),
    "gc_fraction": (1,),
    "melting_temperature": (2,),
    "vienna_dg": (3, 9),
    "mismatch_position": (4, 5, 6, 7),
    "all_mismatch": (0, 4, 5, 6, 7, 8),
}
HISTORICAL_THERMO_FILENAME = (
    "LML_chamber_with_thermo_all_primer_types_updated_2025-09-11.csv"
)
HISTORICAL_THERMO_FEATURES = [
    f"{p}_{kind}_cal_mol"
    for p in PRIMER_ORDER
    for kind in ("binding_free_energy", "end_stability")
] + [
    "mean_binding_free_energy_cal_mol",
    "mean_end_stability_cal_mol",
]
UNIT_COLUMNS = [
    "source", "primer_set", "target_seq", "target_conc", "canonical_mm", "block"
]
CONFIRMATORY_SOURCE = "LMLv2"
CONFIRMATORY_BLOCK = "block19"
UNIT_CACHE_SCHEMA = (
    "replicate-units-v5-tp60-censored-exclude-bstxt-mm-source-independent-assay-id-20260824"
)


def assay_sequence_identity(frame: pd.DataFrame) -> pd.Series:
    """Hash the exact model-input sequence tuple independently of source labels."""
    missing = [column for column in ASSAY_SEQUENCE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Assay identity columns are missing: {missing}")
    canonical = frame[ASSAY_SEQUENCE_COLUMNS].astype(str).apply(
        lambda column: column.str.strip().str.upper()
    )
    if canonical.eq("").any().any() or canonical.eq("NAN").any().any():
        raise ValueError("Assay identity contains an empty sequence")
    joined = canonical.agg("\x1f".join, axis=1)
    return joined.map(lambda value: hashlib.sha256(value.encode()).hexdigest())


def isolate_outer_fold(units: pd.DataFrame, outer_block: str):
    """Return a sequence-isolated genomic-block outer fold.

    LMLv1 contains development blocks 0--9 and LMLv2 contains development
    blocks 10--18. LMLv2 block19 is reserved by ``split_confirmatory`` and is
    intentionally not a valid development fold here.
    """
    if "sequence_group_id" not in units:
        raise ValueError("Units lack source-independent sequence_group_id")
    try:
        block_index = int(str(outer_block).replace("block", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid genomic block: {outer_block}") from exc
    if 0 <= block_index <= 9:
        source = "LMLv1"
    elif 10 <= block_index <= 18:
        source = "LMLv2"
    else:
        raise ValueError(f"Outer development block must be block0--block18: {outer_block}")
    outer_mask = (units.source == source) & (units.block == outer_block)
    outer = units.loc[outer_mask].copy()
    if outer.empty:
        raise ValueError(f"Outer fold is empty: {outer_block}")
    nominal = units.loc[~outer_mask].copy()
    outer_ids = set(outer.sequence_group_id.astype(str))
    overlap = nominal.sequence_group_id.astype(str).isin(outer_ids)
    candidate = nominal.loc[~overlap].copy()
    if set(candidate.sequence_group_id.astype(str)) & outer_ids:
        raise RuntimeError(f"Outer sequence isolation failed: {outer_block}")
    audit = {
        "identity": "sha256 of ordered primer and aligned target-sequence tuple",
        "outer_source": source,
        "outer_sequence_groups": len(outer_ids),
        "nominal_candidate_units": len(nominal),
        "excluded_outer_identity_units": int(overlap.sum()),
        "excluded_outer_identity_groups": int(
            nominal.loc[overlap, "sequence_group_id"].nunique()
        ),
        "candidate_units": len(candidate),
        "post_isolation_overlap": 0,
    }
    return candidate.reset_index(drop=True), outer.reset_index(drop=True), audit


def restrict_train_chemistry(frame: pd.DataFrame, chemistry: str | None):
    """Keep every evaluation row; restrict only the training chemistry."""
    if not chemistry:
        return frame
    kept = frame.loc[frame.canonical_mm.eq(chemistry)].copy()
    if kept.empty:
        raise ValueError(f"No {chemistry} units left after fold isolation")
    return kept.reset_index(drop=True)


def isolate_inner_fold(candidate: pd.DataFrame, inner_block: str):
    """Keep inner identities entirely out of the epoch-selection fitting set."""
    try:
        block_index = int(str(inner_block).replace("block", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid genomic block: {inner_block}") from exc
    if 0 <= block_index <= 9:
        source = "LMLv1"
    elif 10 <= block_index <= 18:
        source = "LMLv2"
    else:
        raise ValueError(f"Inner development block must be block0--block18: {inner_block}")
    inner_mask = (candidate.source == source) & (candidate.block == inner_block)
    inner = candidate.loc[inner_mask].copy()
    if inner.empty:
        raise ValueError(f"Inner fold is empty: {inner_block}")
    nominal_fit = candidate.loc[~inner_mask].copy()
    inner_ids = set(inner.sequence_group_id.astype(str))
    overlap = nominal_fit.sequence_group_id.astype(str).isin(inner_ids)
    fit = nominal_fit.loc[~overlap].copy()
    if set(fit.sequence_group_id.astype(str)) & inner_ids:
        raise RuntimeError(f"Inner sequence isolation failed: {inner_block}")
    audit = {
        "inner_source": source,
        "inner_sequence_groups": len(inner_ids),
        "nominal_fit_units": len(nominal_fit),
        "excluded_inner_identity_units": int(overlap.sum()),
        "excluded_inner_identity_groups": int(
            nominal_fit.loc[overlap, "sequence_group_id"].nunique()
        ),
        "fit_units": len(fit),
        "post_isolation_overlap": 0,
    }
    return fit.reset_index(drop=True), inner.reset_index(drop=True), audit


def split_confirmatory(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = (units.source == CONFIRMATORY_SOURCE) & (units.block == CONFIRMATORY_BLOCK)
    development, confirmatory = units.loc[~holdout].copy(), units.loc[holdout].copy()
    overlap = set(development.primer_set.astype(str)) & set(confirmatory.primer_set.astype(str))
    if overlap:
        raise ValueError(f"Confirmatory primer-set overlap: {sorted(overlap)[:5]}")
    sequence_overlap = (
        set(development.sequence_group_id.astype(str))
        & set(confirmatory.sequence_group_id.astype(str))
    )
    if sequence_overlap:
        raise ValueError(
            f"Confirmatory sequence-identity overlap: {sorted(sequence_overlap)[:5]}"
        )
    return development.reset_index(drop=True), confirmatory.reset_index(drop=True)


def canonical_master_mix(value: object) -> str:
    mm = str(value).strip().upper()
    if mm == "SS":
        return "SS"
    if mm in {"BST", "BSTXT", "BSTXT-DK"}:
        return "BST"
    if mm in {"WS", "WS-NOUDG", "WS-SYTO9", "WS_SYTO9"}:
        return "WS"
    return "UNK"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_provenance(data_dir: Path, files: Iterable[str]) -> dict:
    provenance_files = {
            name: {"sha256": sha256_file(data_dir / name), "bytes": (data_dir / name).stat().st_size}
            for name in files
    }
    historical = data_dir / HISTORICAL_THERMO_FILENAME
    if historical.exists():
        provenance_files[HISTORICAL_THERMO_FILENAME] = {
            "sha256": sha256_file(historical), "bytes": historical.stat().st_size
        }
    return {
        "feature_schema": "July-2026 export plus deterministic terminal-mismatch features v2",
        "excluded_master_mix_formulations": sorted(EXCLUDED_MASTER_MIX_FORMULATIONS),
        "exclusion_reason": (
            "BstXT-MM is a distinct pre-release formulation and must not be "
            "pooled with BstXT-dk technical replicates"
        ),
        "historical_thermodynamics": {
            "status": "available for a matched LMLv1-only sensitivity analysis"
                      if historical.exists() else "not present",
            "features": HISTORICAL_THERMO_FEATURES,
            "units": "cal/mol as represented in the September-2025 source export",
            "merge_policy": (
                "row-position join allowed only after exact row-count, primer-set, "
                "master-mix, target-concentration, and Tp identity checks"
            ),
            "scope_limitation": (
                "not included in the primary multi-source models because corresponding "
                "values are unavailable for LMLv2, Systematic, and viral datasets"
            ),
        },
        "files": provenance_files,
    }


def _attach_historical_thermodynamics(frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach version-matched September-2025 LMLv1 energy features.

    The historical export has the same reaction rows in the same order but lacks
    a portable reaction identifier. Refuse the positional join unless independent
    experimental fields prove that the exact known export pair is being used.
    """
    path = data_dir / HISTORICAL_THERMO_FILENAME
    if not path.exists():
        return frame
    source_cols = [
        "primer_set", "master_mix", "target_conc", "tp",
        *[f"{p}_{kind}" for p in PRIMER_ORDER
          for kind in ("free_energy", "end_stability")],
        "mean_target_bind_dg", "mean_end_stability_dg",
    ]
    historical = pd.read_csv(path, usecols=source_cols, low_memory=False)
    if len(historical) != len(frame):
        raise ValueError(
            f"Historical thermodynamics row-count mismatch: {len(historical)} != {len(frame)}"
        )
    string_checks = {
        "primer_set": historical["primer_set"].astype(str).to_numpy()
                      == frame["primer_set"].astype(str).to_numpy(),
        "master_mix": historical["master_mix"].map(canonical_master_mix).to_numpy()
                      == frame["master_mix"].map(canonical_master_mix).to_numpy(),
    }
    numeric_checks = {
        "target_conc": np.isclose(
            pd.to_numeric(historical["target_conc"], errors="coerce"),
            pd.to_numeric(frame["target_conc"], errors="coerce"), equal_nan=True,
        ),
        "tp": np.isclose(
            pd.to_numeric(historical["tp"], errors="coerce"),
            pd.to_numeric(frame["tp"], errors="coerce"), equal_nan=True,
        ),
    }
    failed = [name for name, values in {**string_checks, **numeric_checks}.items()
              if not np.asarray(values).all()]
    if failed:
        raise ValueError(f"Historical thermodynamics identity check failed: {failed}")
    rename = {
        **{f"{p}_free_energy": f"{p}_binding_free_energy_cal_mol" for p in PRIMER_ORDER},
        **{f"{p}_end_stability": f"{p}_end_stability_cal_mol" for p in PRIMER_ORDER},
        "mean_target_bind_dg": "mean_binding_free_energy_cal_mol",
        "mean_end_stability_dg": "mean_end_stability_cal_mol",
    }
    values = historical[list(rename)].rename(columns=rename).reset_index(drop=True)
    if values.isna().any().any():
        raise ValueError("Historical thermodynamics contains missing energy values")
    enriched = frame.reset_index(drop=True).copy()
    for col in HISTORICAL_THERMO_FEATURES:
        enriched[col] = values[col].to_numpy(np.float32)
    return enriched


def _valid_sequences(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for p in PRIMER_TYPES:
        for suffix in ("sequence", "target_sequence"):
            col = f"{p}_{suffix}"
            mask &= df[col].notna() & df[col].astype(str).str.strip().ne("")
    return mask


def load_rows(data_dir: Path, files: Iterable[str]) -> pd.DataFrame:
    frames = []
    for name in files:
        frame = pd.read_csv(data_dir / name)
        if Path(name).name == "LMLv1.csv":
            frame = _attach_historical_thermodynamics(frame, data_dir)
        frame["source"] = Path(name).stem
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return derive_features(df)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features from a frame of raw per-reaction columns.

    Split out of ``load_rows`` so callers that build rows in memory -- the
    lamp-design Fusion path scores primer sets that have never been in a CSV
    -- can reuse this instead of reimplementing it. Reimplementing would mean
    a second version of features the model was trained on, which fails
    silently rather than loudly.

    Needs, per row: the eight ``{primer}_sequence`` and
    ``{primer}_target_sequence`` columns, the four ``{primer}_{feature}``
    thermodynamic columns, ``total_mismatches``, ``mean_dg_ViennaRNA``,
    ``master_mix``, ``target_conc`` and ``block``.
    """
    df = df.copy()
    df["block"] = df["block"].astype(str)
    normalized_master_mix = df["master_mix"].astype(str).str.strip().str.upper()
    df = df.loc[
        ~normalized_master_mix.isin(EXCLUDED_MASTER_MIX_FORMULATIONS)
    ].copy()
    df["canonical_mm"] = df["master_mix"].map(canonical_master_mix)
    df = df[df["canonical_mm"].isin(MM_CLASSES)].copy()
    df = df[_valid_sequences(df)].copy()
    df = df[df["target_conc"].notna() & (df["target_conc"] > 0)].copy()
    df = df.dropna(subset=THERMO_FEATURES).copy()
    df["log10_conc"] = np.log10(df["target_conc"].astype(float)) - 3.0
    for mm in MM_CLASSES:
        df[f"mm_{mm}"] = (df["canonical_mm"] == mm).astype(np.float32)
    # Polymerase extension is especially sensitive to the primer 3' end,
    # which a total mismatch count cannot distinguish. Inputs are 5' -> 3'.
    for p in PRIMER_ORDER:
        primer = df[f"{p}_sequence"].astype(str).str.upper()
        target = df[f"{p}_target_sequence"].astype(str).str.upper()
        mismatch = [np.fromiter((a != b for a, b in zip(x, y)), dtype=np.float32)
                    for x, y in zip(primer, target)]
        df[f"{p}_mm_3p3"] = [float(v[-3:].sum()) for v in mismatch]
        df[f"{p}_mm_3p5"] = [float(v[-5:].sum()) for v in mismatch]
        df[f"{p}_mm_5p5"] = [float(v[:5].sum()) for v in mismatch]
        df[f"{p}_mm_pos_weighted"] = [
            float(np.dot(v, np.linspace(0.25, 1.0, len(v), dtype=np.float32)))
            for v in mismatch
        ]
    return df.reset_index(drop=True)


def aggregate_replicates(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical replicates before splitting or fitting.

    Static sequence/tabular columns are checked for consistency. Targets retain
    amplification fraction, positive-only mean Tp, and all-reaction mean Tp with
    failures represented at both the 60-minute and 120-minute censoring limits.
    """
    static_cols = []
    for p in PRIMER_TYPES:
        static_cols.extend([f"{p}_sequence", f"{p}_target_sequence"])
    static_cols += TABULAR_FEATURES
    static_cols += [x for x in HISTORICAL_THERMO_FEATURES if x in rows.columns]
    grouped = rows.groupby(UNIT_COLUMNS, dropna=False, sort=False)
    nunique = grouped[static_cols].nunique(dropna=False)
    if (nunique > 1).any().any():
        bad = nunique.index[(nunique > 1).any(axis=1)][:5].tolist()
        raise ValueError(f"Static features vary within replicate groups: {bad}")
    static = grouped[static_cols].first()
    targets = grouped["tp"].agg(
        n_replicates="size",
        n_amplified=lambda x: int(x.notna().sum()),
        amp_fraction=lambda x: float(x.notna().mean()),
        tp_positive=lambda x: float(x.mean()) if x.notna().any() else np.nan,
        tp_positive_sd=lambda x: float(x.std(ddof=1)) if x.notna().sum() > 1 else 0.0,
        tp60_censored=lambda x: float(x.fillna(60.0).mean()),
        tp120=lambda x: float(x.fillna(120.0).mean()),
    )
    out = static.join(targets).reset_index()
    out["group_id"] = (
        out["source"].astype(str) + "|" + out["primer_set"].astype(str)
        + "|" + out["target_seq"].astype(str) + "|" + out["target_conc"].astype(str)
        + "|" + out["canonical_mm"].astype(str)
    )
    out["sequence_group_id"] = assay_sequence_identity(out)
    return out


def _unit_cache_path(data_dir: Path, files: list[str]) -> Path:
    fingerprint = hashlib.sha256(UNIT_CACHE_SCHEMA.encode())
    for name in files:
        fingerprint.update(name.encode())
        fingerprint.update(sha256_file(data_dir / name).encode())
    historical = data_dir / HISTORICAL_THERMO_FILENAME
    if historical.exists() and "LMLv1.csv" in files:
        fingerprint.update(HISTORICAL_THERMO_FILENAME.encode())
        fingerprint.update(sha256_file(historical).encode())
    return data_dir / ".lampbench_cache" / f"units_{fingerprint.hexdigest()}.pkl.gz"


def load_units(data_dir: Path, files: Iterable[str], use_cache: bool = True) -> pd.DataFrame:
    """Load and aggregate one source at a time to bound peak memory.

    `source` is part of UNIT_COLUMNS, so no replicate group can span files and
    this is exactly equivalent to concatenating raw rows before aggregation.
    """
    files = list(files)
    cache_path = _unit_cache_path(data_dir, files)
    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path, compression="gzip")
    units = pd.concat(
        [aggregate_replicates(load_rows(data_dir, [name])) for name in files],
        ignore_index=True,
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Separate workers may materialize the same deterministic cache at the
        # same time. A process-unique temporary name preserves atomic replace
        # semantics without letting one writer remove another writer's temp.
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        units.to_pickle(temporary, compression="gzip")
        os.replace(temporary, cache_path)
    return units


def target_for_task(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task == "tp30":
        keep = df["tp_positive"].notna() & (df["tp_positive"] < 30)
        return df.loc[keep].reset_index(drop=True), df.loc[keep, "tp_positive"].to_numpy(np.float32)
    if task == "tp60":
        keep = df["tp_positive"].notna() & (df["tp_positive"] < 60)
        return df.loc[keep].reset_index(drop=True), df.loc[keep, "tp_positive"].to_numpy(np.float32)
    if task == "tp60_censored":
        return df.reset_index(drop=True), df["tp60_censored"].to_numpy(np.float32)
    if task == "positive":
        keep = df["tp_positive"].notna()
        return df.loc[keep].reset_index(drop=True), df.loc[keep, "tp_positive"].to_numpy(np.float32)
    if task == "tp120":
        return df.reset_index(drop=True), df["tp120"].to_numpy(np.float32)
    if task in {"classification", "hurdle"}:
        if task == "classification":
            return df.reset_index(drop=True), df["amp_fraction"].to_numpy(np.float32)
        targets = np.stack([
            df["amp_fraction"].to_numpy(np.float32),
            df["tp_positive"].fillna(0.0).to_numpy(np.float32),
            df["tp_positive"].notna().to_numpy(np.float32),
        ], axis=1)
        return df.reset_index(drop=True), targets
    raise ValueError(task)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str] = TABULAR_FEATURES) -> "Standardizer":
        x = df[cols].to_numpy(np.float32)
        mean = x.mean(0)
        std = x.std(0)
        std[std == 0] = 1.0
        return cls(mean, std)

    def transform(self, df: pd.DataFrame, cols: list[str] = TABULAR_FEATURES) -> np.ndarray:
        return (df[cols].to_numpy(np.float32) - self.mean) / self.std

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def encode_pair(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Legacy-compatible 13-channel sequence plus an explicit primer-role id."""
    nuc = {"A": 0, "C": 1, "G": 2, "T": 3}
    chunks, roles = [], []
    mm = np.array([row[f"mm_{x}"] for x in MM_CLASSES], dtype=np.float32)
    conc = np.float32(row["log10_conc"])
    for role, p in enumerate(PRIMER_TYPES):
        primer = str(row[f"{p}_sequence"]).upper()
        target = str(row[f"{p}_target_sequence"]).upper()
        if len(primer) != len(target):
            raise ValueError(f"Unaligned primer/target for {row['group_id']} {p}")
        x = np.zeros((len(primer), 13), np.float32)
        for i, (a, b) in enumerate(zip(primer, target)):
            if a in nuc:
                x[i, nuc[a]] = 1
            if b in nuc:
                x[i, 4 + nuc[b]] = 1
        x[:, 9:12] = mm
        x[:, 12] = conc
        chunks.append(x)
        roles.extend([role] * len(primer))
        if role < len(PRIMER_TYPES) - 1:
            sep = np.zeros((1, 13), np.float32)
            sep[0, 8] = 1
            sep[0, 9:12] = mm
            sep[0, 12] = conc
            chunks.append(sep)
            roles.append(8)
    return np.concatenate(chunks), np.asarray(roles, np.int64)


def primer_role_from_ablation(ablation: str) -> str | None:
    """Return the primer role removed by a ``drop_primer_*`` ablation."""
    prefix = "drop_primer_"
    if not ablation.startswith(prefix):
        return None
    primer = ablation.removeprefix(prefix)
    if primer not in PRIMER_TYPES:
        raise ValueError(
            f"Unknown primer-role ablation: {ablation}; "
            f"expected one of {[f'{prefix}{p}' for p in PRIMER_ORDER]}"
        )
    return primer


def tabular_frame_for_ablation(df: pd.DataFrame, ablation: str) -> pd.DataFrame:
    """Remove a primer role from engineered features without aggregate leakage.

    Role-prefixed values become constants and therefore standardize to zero. The
    two global summaries are recalculated from the seven retained roles so the
    dropped role cannot be recovered by subtraction from a global total/mean.
    """
    primer = primer_role_from_ablation(ablation)
    if primer is None:
        return df
    out = df.copy()
    out["total_mismatches"] = (
        out["total_mismatches"].astype(np.float32)
        - out[f"{primer}_mismatches"].astype(np.float32)
    )
    out["mean_dg_ViennaRNA"] = (
        8.0 * out["mean_dg_ViennaRNA"].astype(np.float32)
        - out[f"{primer}_dg_ViennaRNA"].astype(np.float32)
    ) / 7.0
    primer_columns = [
        column for column in TABULAR_FEATURES if column.startswith(f"{primer}_")
    ]
    out.loc[:, primer_columns] = 0.0
    return out


class LampDataset(Dataset):
    def __init__(self, df: pd.DataFrame, targets: np.ndarray, standardizer: Standardizer,
                 ablation: str = "none", token_physics: bool = False):
        self.df = df.reset_index(drop=True)
        self.targets = np.asarray(targets, np.float32)
        self.tab = standardizer.transform(
            tabular_frame_for_ablation(self.df, ablation)
        )
        self.encoded = [encode_pair(row) for _, row in self.df.iterrows()]
        dropped_primer = primer_role_from_ablation(ablation)
        token_drop_group = None
        if ablation.startswith("token_drop_"):
            token_drop_group = ablation.removeprefix("token_drop_")
            if not token_physics or token_drop_group not in TOKEN_PHYSICS_GROUP_CHANNELS:
                raise ValueError(f"Unknown token-physics ablation: {ablation}")
        elif ablation == "no_context":
            context_idx = [TABULAR_FEATURES.index(x) for x in META_FEATURES]
            self.tab[:, context_idx] = 0
            self.encoded = [(np.concatenate([x[:, :9], np.zeros_like(x[:, 9:])], axis=1), r)
                            for x, r in self.encoded]
        elif ablation == "no_position":
            position_idx = [TABULAR_FEATURES.index(x) for x in POSITION_FEATURES]
            self.tab[:, position_idx] = 0
        elif ablation == "no_thermo":
            thermo_idx = [TABULAR_FEATURES.index(x) for x in THERMO_FEATURES + POSITION_FEATURES]
            self.tab[:, thermo_idx] = 0
        elif dropped_primer is not None:
            # Remove the entire aligned primer/target span. Zeroing tokens would
            # retain primer length, so physical removal is the stricter test of
            # whether the model needs information from this role.
            dropped_role = PRIMER_TYPES.index(dropped_primer)
            self.encoded = [
                (seq[role != dropped_role], role[role != dropped_role])
                for seq, role in self.encoded
            ]
        elif ablation != "none":
            raise ValueError(f"Unknown feature ablation: {ablation}")
        if token_physics:
            role_indices = np.asarray([
                [TABULAR_FEATURES.index(f"{primer}_{feature}")
                 for feature in ("mismatches", "gc_fraction", "tm", "dg_ViennaRNA",
                                 "mm_3p3", "mm_3p5", "mm_5p5", "mm_pos_weighted")]
                for primer in PRIMER_TYPES
            ], dtype=np.int64)
            global_indices = np.asarray([
                TABULAR_FEATURES.index("total_mismatches"),
                TABULAR_FEATURES.index("mean_dg_ViennaRNA"),
            ], dtype=np.int64)
            augmented = []
            for i, (seq, role) in enumerate(self.encoded):
                role_table = self.tab[i, role_indices]
                role_table = np.concatenate(
                    [role_table, np.zeros((1, role_table.shape[1]), np.float32)], axis=0)
                per_token = role_table[role]
                global_values = np.broadcast_to(
                    self.tab[i, global_indices], (len(seq), len(global_indices)))
                augmented.append((np.concatenate(
                    [seq, per_token, global_values], axis=1).astype(np.float32), role))
            self.encoded = augmented
            if token_drop_group is not None:
                channels = np.asarray(
                    TOKEN_PHYSICS_GROUP_CHANNELS[token_drop_group], dtype=np.int64) + 13
                self.encoded = [
                    (np.array(seq, copy=True), role) for seq, role in self.encoded
                ]
                for seq, _ in self.encoded:
                    seq[:, channels] = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        seq, role = self.encoded[idx]
        return {
            "seq": torch.from_numpy(seq),
            "role": torch.from_numpy(role),
            "tab": torch.from_numpy(self.tab[idx]),
            "target": torch.tensor(self.targets[idx]),
            "index": idx,
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Batch equal-length examples so FFT models never compute through padding."""
    def __init__(self, dataset: LampDataset, batch_size: int, shuffle: bool, seed: int = 0):
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        buckets: dict[int, list[int]] = {}
        for idx, (seq, _) in enumerate(dataset.encoded):
            buckets.setdefault(len(seq), []).append(idx)
        self.buckets = buckets
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for indices in self.buckets.values():
            values = np.asarray(indices, dtype=int).copy()
            if self.shuffle:
                rng.shuffle(values)
            batches.extend(values[i:i + self.batch_size].tolist()
                           for i in range(0, len(values), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return sum((len(x) + self.batch_size - 1) // self.batch_size
                   for x in self.buckets.values())


class MutationOrderedBatchSampler(Sampler[list[int]]):
    """Batch in ascending ``total_mismatches`` order. Does not shuffle.

    Batches can mix sequence lengths; ``collate`` still pads. That is the
    trade vs ``LengthBucketBatchSampler`` (FFT-friendly equal-length buckets).
    """

    def __init__(self, dataset: LampDataset, batch_size: int):
        if "total_mismatches" not in dataset.df.columns:
            raise ValueError("mutation-ordered sampler requires total_mismatches")
        mismatches = dataset.df["total_mismatches"].to_numpy(np.float64)
        self.indices = np.argsort(mismatches, kind="mergesort").astype(int)
        self.batch_size = int(batch_size)

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            yield self.indices[start:start + self.batch_size].tolist()

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(x["seq"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    bsz, channels = len(batch), batch[0]["seq"].shape[1]
    seq = torch.zeros((bsz, max_len, channels), dtype=torch.float32)
    role = torch.full((bsz, max_len), 8, dtype=torch.long)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    for i, item in enumerate(batch):
        n = len(item["seq"])
        seq[i, :n] = item["seq"]
        role[i, :n] = item["role"]
    return {
        "seq": seq,
        "role": role,
        "mask": mask,
        "lengths": lengths,
        "tab": torch.stack([x["tab"] for x in batch]),
        "target": torch.stack([x["target"] for x in batch]),
        "index": torch.tensor([x["index"] for x in batch]),
    }


def write_manifest(path: Path, data_dir: Path, train_files: list[str], external_files: list[str]) -> None:
    row_counts = {name: sum(1 for _ in (data_dir / name).open()) - 1 for name in train_files}
    units = load_units(data_dir, train_files)
    development, confirmatory = split_confirmatory(units)
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent
    code_paths = (
        sorted(package_dir.glob("*.py"))
        + sorted((project_dir / "tests").glob("*.py"))
        + [project_dir / name for name in (
            "experiment_plan.json", "pyproject.toml", "requirements-gpu.txt",
            "bootstrap_gpu.sh", "gpu_validate.py", "run_gpu_pipeline.sh",
            "prepare_runpod_bundle.sh", "continue_after_screen.sh",
            "protocol_deviations.json",
        )]
    )
    import lyra
    lyra_dir = Path(lyra.__file__).resolve().parent
    lyra_paths = [lyra_dir / name for name in ("__init__.py", "pgc.py", "s4d.py", "model.py")]
    packages = {}
    for package in ["scipy", "scikit-learn", "catboost", "lightgbm", "xgboost",
                    "lyra-sequence",
                    "mamba-ssm", "causal-conv1d"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    identity_cols = (["primer_set", "target_seq"]
                     + [f"{p}_sequence" for p in PRIMER_ORDER]
                     + [f"{p}_target_sequence" for p in PRIMER_ORDER])
    development_rows = pd.concat(
        [pd.read_csv(data_dir / name, usecols=identity_cols, dtype=str)
         for name in train_files], ignore_index=True)
    primer_cols = [f"{p}_sequence" for p in PRIMER_ORDER]
    full_cols = primer_cols + [f"{p}_target_sequence" for p in PRIMER_ORDER]
    development_identity = {
        "primer_set": set(development_rows.primer_set.dropna()),
        "target_seq": set(development_rows.target_seq.dropna()),
        "primer_tuple": set(map(tuple, development_rows[primer_cols].fillna("").drop_duplicates().to_numpy())),
        "full_tuple": set(map(tuple, development_rows[full_cols].fillna("").drop_duplicates().to_numpy())),
    }
    external_novelty = {}
    external_primer_sets = {}
    for name in external_files:
        frame = pd.read_csv(data_dir / name, usecols=identity_cols, dtype=str)
        identities = {
            "primer_set": set(frame.primer_set.dropna()),
            "target_seq": set(frame.target_seq.dropna()),
            "primer_tuple": set(map(tuple, frame[primer_cols].fillna("").drop_duplicates().to_numpy())),
            "full_tuple": set(map(tuple, frame[full_cols].fillna("").drop_duplicates().to_numpy())),
        }
        external_primer_sets[name] = identities["primer_set"]
        external_novelty[name] = {
            "rows": len(frame),
            "unique_primer_sets": len(identities["primer_set"]),
            "unique_targets": len(identities["target_seq"]),
            "development_overlap": {
                key: len(value & development_identity[key]) for key, value in identities.items()
            },
        }
    external_pairwise_primer_set_overlap = {
        f"{a}|{b}": len(external_primer_sets[a] & external_primer_sets[b])
        for i, a in enumerate(external_files) for b in external_files[i + 1:]
    }
    payload = {
        "protocol": {
            "outer_folds": [f"block{i}" for i in range(10)],
            "inner_fold_rule": "block((outer_index + 1) mod 10)",
            "replicates": "aggregated before fitting and scoring",
            "sequence_identity_isolation": (
                "outer-equivalent full primer/target sequence tuples are excluded across "
                "all sources; inner-equivalent tuples are excluded during epoch selection"
            ),
            "external_policy": "algorithmically locked until finalist selection",
            "external_blinding_caveat": "earlier tabular external scores were visible during audit",
            "confirmatory_holdout": "LMLv2 block19; excluded before every model/tuning run",
        },
        "counts": {"rows": sum(row_counts.values()), "rows_by_file": row_counts,
                   "units": len(units), "primer_sets": int(units.primer_set.nunique())},
        "confirmatory_counts": {"units": len(confirmatory),
                                "primer_sets": int(confirmatory.primer_set.nunique()),
                                "development_units": len(development)},
        "external_novelty": external_novelty,
        "external_pairwise_primer_set_overlap": external_pairwise_primer_set_overlap,
        "block_counts": units.block.value_counts().sort_index().to_dict(),
        "outer_fold_identity_isolation": {
            f"block{i}": isolate_outer_fold(development, f"block{i}")[2]
            for i in range(10)
        },
        "provenance": feature_provenance(data_dir, train_files + external_files),
        "code_provenance": {
            str(p.relative_to(project_dir)): sha256_file(p) for p in code_paths
        },
        "canonical_lyra_provenance": {
            "package": packages["lyra-sequence"],
            "files": {f"lyra/{p.name}": sha256_file(p) for p in lyra_paths},
            "primitive_policy": (
                "corrected and paper Lyra variants directly instantiate "
                "lyra.pgc.PGC and lyra.s4d.S4D"
            ),
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
            "packages": packages,
        },
    }
    path.write_text(json.dumps(payload, indent=2))
