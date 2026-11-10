"""``lamp-design evaluate`` -- score primer sets that already exist.

    CSV of primer sets -> thermo -> encoding -> inference -> predictions (+ metrics)

Where ``design`` searches for new sets, ``evaluate`` scores ones you already
have: a published set you want a second opinion on, a panel a collaborator
sent, or -- the case this was built against -- a plate of sets whose Tp was
actually *measured*, so the model's prediction can be checked against reality.

## Two ways the target windows arrive

The model does not see a primer in isolation; it sees the primer paired with
the stretch of target it binds, because mismatches are a feature. So every
(set, target) pair needs eight target windows, and they can come from either:

``--windows csv``
    The input already carries them, as ``<type>_target_sequence`` columns.
    This is the lab's own results format, and it is the honest choice when
    validating: the windows are the ones the experiment actually used, with no
    re-derivation that could silently disagree.

``--windows locate``
    Search each target in ``--targets`` for each primer's best match, by
    minimum Hamming distance over every offset. Needed when the input is a
    bare list of primers. ``--max-window-mismatches`` rejects a placement that
    is too poor to be a real binding site, which is what distinguishes "this
    primer binds here with 3 mismatches" from "this primer is not in this
    target at all".

Default is ``auto``: use the columns when they are present, locate otherwise.

## Conditions come from the data, not from a sweep

``design`` scores a cross product of master mixes and concentrations, because
it does not know which one you care about. ``evaluate`` does: each input row
names its own ``master_mix`` and ``target_conc``, and that row is scored under
exactly that condition. Overriding with ``--master-mix``/``--conc`` switches
to a cross product, which is what you want for "how would these sets behave
under a mix I have not tried", and is wrong for validation.

## Censoring

A reaction that never amplified has no Tp. The training data encoded this by
censoring at 60 minutes (the ``tp60_censored`` flag in the production config),
so a blank or non-numeric measurement here is treated the same way, at
``--censor-at``. Metrics are reported twice: over the amplified rows only, and
over all rows with censoring applied. The second is comparable to the paper's
numbers; the first answers "given that it worked, did we predict how fast".
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from lamp_design.commands.design import (
    BATCH_SIZE_CPU,
    BATCH_SIZE_GPU,
    build_conditions,
    resolve_device,
)
from lamp_design.commands.prepare import InputError
from lamp_design.encoding import canonical_master_mix
from lamp_design.types import (
    FORWARD_STRAND_TYPES,
    MASTER_MIXES,
    PAIR_CHUNK,
    PRIMER_TYPE_ORDER,
    Condition,
    Primer,
    PrimerSet,
    ThermoTable,
)
from lamp_design.utils.sequences import read_fasta

if TYPE_CHECKING:  # pragma: no cover
    from lamp_design.models.inference import LyraEnsemble

#: Measured Tp assigned to a reaction that never amplified. Matches the
#: training data's ``tp60_censored`` convention, so censored metrics here are
#: on the same footing as the published ones.
DEFAULT_CENSOR_MINUTES = 60.0

#: Reject a located window worse than this many mismatches. A LAMP primer is
#: 15-25 nt, so 6 mismatches is already past the point where the placement is
#: more likely to be the wrong site than a diverged one.
DEFAULT_MAX_WINDOW_MISMATCHES = 6

#: Column names accepted for each field, lower-cased, first match wins.
_SET_NAME_COLUMNS = ("primer_set", "set_id", "set", "name", "primer_set_id")
_TARGET_NAME_COLUMNS = ("target_seq", "target", "target_name", "target_id")
_MASTER_MIX_COLUMNS = ("master_mix", "mastermix", "mix", "enzyme")
_CONC_COLUMNS = ("target_conc", "conc", "concentration", "copies")
_TRUTH_COLUMNS = ("tp", "tp_minutes", "measured_tp", "observed_tp")

__all__ = [
    "DEFAULT_CENSOR_MINUTES",
    "DEFAULT_MAX_WINDOW_MISMATCHES",
    "EvalRow",
    "EvaluationResult",
    "evaluate",
    "load_rows",
    "locate_window",
    "metrics_table",
    "register",
    "roc_auc",
    "run",
    "spearman",
]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    """One (primer set, target, condition) to score, as read from the input."""

    set_id: str
    target_name: str
    #: Eight primer sequences in :data:`PRIMER_TYPE_ORDER`, reverse-strand
    #: types already reverse-complemented -- the orientation the model sees.
    primers: List[str]
    #: Eight target windows in the same order and orientation, or ``None``
    #: when they still have to be located in a target sequence.
    windows: Optional[List[str]]
    condition: Condition
    #: Measured Tp in minutes; ``None`` when the input has no truth column,
    #: ``nan`` when the row recorded no amplification.
    measured_tp: Optional[float] = None
    #: Worst per-primer mismatch count from window location, for diagnostics.
    max_located_mismatches: Optional[int] = None


@dataclass
class EvaluationResult:
    """Everything one ``evaluate`` run produces."""

    rows: List[EvalRow]
    predicted_tp: List[float]
    predicted_std: List[float]
    #: Metric name -> value, for each of the ``amplified`` and ``censored``
    #: populations. Empty when the input carried no measured Tp.
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_scored(self) -> int:
        return len(self.predicted_tp)


def _pick(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """First column in ``candidates`` that ``fieldnames`` actually has."""
    lowered = {name.strip().lower(): name for name in fieldnames}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    return None


def _primer_columns(fieldnames: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Locate the per-primer sequence and target-window columns.

    Returns ``(sequence_columns, window_columns)``, each keyed by primer type.
    ``window_columns`` is empty when the input does not carry windows.
    """
    lowered = {name.strip().lower(): name for name in fieldnames}
    seq_cols: Dict[str, str] = {}
    win_cols: Dict[str, str] = {}
    for ptype in PRIMER_TYPE_ORDER:
        low = ptype.lower()
        for suffix in ("_sequence", "_seq", ""):
            key = f"{low}{suffix}"
            if key in lowered:
                seq_cols[ptype] = lowered[key]
                break
        for suffix in ("_target_sequence", "_target_seq", "_window"):
            key = f"{low}{suffix}"
            if key in lowered:
                win_cols[ptype] = lowered[key]
                break
    return seq_cols, win_cols


def _parse_float(raw: Optional[str]) -> float:
    """Parse a measurement, mapping blanks and ``nan`` to ``nan``."""
    if raw is None:
        return math.nan
    text = raw.strip()
    if not text or text.lower() in {"na", "nan", "none", "null", "-", "inf"}:
        return math.nan
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_rows(
    path: str | Path,
    *,
    master_mix: Optional[str] = None,
    target_conc: Optional[float] = None,
    truth_column: Optional[str] = None,
    require_windows: bool = False,
) -> Tuple[List[EvalRow], List[str]]:
    """Read primer sets to score from a CSV or TSV.

    Returns ``(rows, warnings)``. ``master_mix`` and ``target_conc`` override
    whatever the file says, which is how a single-condition re-score is
    requested; without them each row keeps its own condition.

    Raises:
        InputError: the file is missing, empty, or lacks the eight primer
            sequence columns.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"input does not exist: {path}")

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        records = list(reader)

    if not fieldnames:
        raise InputError(f"{path} has no header row")
    if not records:
        raise InputError(f"{path} has a header but no data rows")

    seq_cols, win_cols = _primer_columns(fieldnames)
    missing = [p for p in PRIMER_TYPE_ORDER if p not in seq_cols]
    if missing:
        raise InputError(
            f"{path} is missing a sequence column for {missing}. Expected one "
            f"column per primer named e.g. 'F3_sequence' or 'F3', for all of "
            f"{list(PRIMER_TYPE_ORDER)}; found columns {fieldnames[:12]}"
            + (" ..." if len(fieldnames) > 12 else "")
        )
    have_windows = len(win_cols) == len(PRIMER_TYPE_ORDER)
    if require_windows and not have_windows:
        absent = [p for p in PRIMER_TYPE_ORDER if p not in win_cols]
        raise InputError(
            f"--windows csv needs a target-window column per primer; {path} is "
            f"missing {absent}. Expected e.g. 'F3_target_sequence'. Use "
            "--windows locate with --targets to derive them instead."
        )

    set_col = _pick(fieldnames, _SET_NAME_COLUMNS)
    target_col = _pick(fieldnames, _TARGET_NAME_COLUMNS)
    mix_col = _pick(fieldnames, _MASTER_MIX_COLUMNS)
    conc_col = _pick(fieldnames, _CONC_COLUMNS)
    truth_col = truth_column or _pick(fieldnames, _TRUTH_COLUMNS)
    if truth_column and truth_column not in fieldnames:
        raise InputError(
            f"--truth-column '{truth_column}' is not a column in {path}; "
            f"columns are {fieldnames}"
        )

    warnings: List[str] = []
    if mix_col is None and master_mix is None:
        warnings.append(
            f"no master-mix column found (looked for {list(_MASTER_MIX_COLUMNS)}); "
            f"defaulting every row to {MASTER_MIXES[1]}. Pass --master-mix to choose."
        )
    if conc_col is None and target_conc is None:
        warnings.append(
            "no concentration column found; defaulting every row to 10,000 "
            "copies/reaction. Pass --conc to choose."
        )

    rows: List[EvalRow] = []
    bad_rows = 0
    unknown_mixes: Dict[str, int] = {}
    for i, record in enumerate(records):
        primers = [(record.get(seq_cols[p]) or "").strip().upper() for p in PRIMER_TYPE_ORDER]
        if any(not seq for seq in primers):
            bad_rows += 1
            continue
        if any(set(seq) - set("ACGT") for seq in primers):
            bad_rows += 1
            continue

        windows: Optional[List[str]] = None
        if have_windows:
            candidate = [
                (record.get(win_cols[p]) or "").strip().upper() for p in PRIMER_TYPE_ORDER
            ]
            # A window has to be the same length as its primer, or the
            # elementwise mismatch count is meaningless.
            if all(
                w and len(w) == len(s) and not (set(w) - set("ACGT"))
                for w, s in zip(candidate, primers)
            ):
                windows = candidate
            else:
                bad_rows += 1
                continue

        raw_mix = master_mix or (record.get(mix_col) if mix_col else None) or MASTER_MIXES[1]
        # Canonicalized here so the output labels a row 'BST' rather than
        # whichever alias the input spelled, and so an unrecognized mix is
        # reported instead of silently encoding as an all-zero one-hot.
        mix = canonical_master_mix(raw_mix)
        if mix is None:
            unknown_mixes[str(raw_mix).strip()] = unknown_mixes.get(
                str(raw_mix).strip(), 0
            ) + 1
            continue
        if target_conc is not None:
            conc = target_conc
        else:
            conc = _parse_float(record.get(conc_col) if conc_col else None)
            if not math.isfinite(conc) or conc <= 0:
                conc = 10_000.0
        measured = _parse_float(record.get(truth_col)) if truth_col else None

        rows.append(
            EvalRow(
                set_id=(record.get(set_col) or f"set{i + 1:04d}").strip() if set_col
                else f"set{i + 1:04d}",
                target_name=(record.get(target_col) or "target").strip() if target_col
                else "target",
                primers=primers,
                windows=windows,
                condition=Condition(str(mix).strip(), float(conc)),
                measured_tp=measured,
            )
        )

    if unknown_mixes:
        listed = ", ".join(
            f"{name!r} ({count:,} row(s))" for name, count in sorted(unknown_mixes.items())
        )
        warnings.append(
            f"dropped rows with a master mix the model was not trained on: {listed}. "
            f"Trained mixes are {list(MASTER_MIXES)} (plus the BstXT/SYTO9 aliases). "
            "Pass --master-mix to score these rows under a trained mix instead."
        )
    if not rows:
        raise InputError(
            f"no usable rows in {path}: all {len(records)} were dropped for a "
            "missing, non-ACGT, or wrong-length primer or window sequence, or an "
            "untrained master mix"
            + (f" ({listed})" if unknown_mixes else "")
        )
    if bad_rows:
        warnings.append(
            f"dropped {bad_rows} of {len(records)} input rows for a missing, "
            "non-ACGT, or wrong-length primer or target-window sequence"
        )
    return rows, warnings


# ---------------------------------------------------------------------------
# Window location
# ---------------------------------------------------------------------------

_ACGT_CODE = np.full(128, -1, dtype=np.int8)
for _i, _base in enumerate("ACGT"):
    _ACGT_CODE[ord(_base)] = _i


def _encode_bases(seq: str) -> np.ndarray:
    return _ACGT_CODE[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]


def locate_window(primer: str, target: str) -> Tuple[Optional[int], int]:
    """Best binding site for ``primer`` in ``target``, by Hamming distance.

    Both arguments are in the same orientation: the caller reverse-complements
    the primer for reverse-strand types before calling. Returns
    ``(start, mismatches)``, or ``(None, len(primer))`` when the primer does
    not fit. Ambiguity codes in the target count as mismatches, which is the
    conservative reading -- an ``N`` is not evidence of a match.

    A full sliding-window scan rather than an index: a LAMP primer is 15-25 nt
    against a target of at most a few hundred kb, and the exact-match
    shortcuts do not survive the mismatches that are the whole point of the
    feature.
    """
    n, m = len(target), len(primer)
    if m == 0 or n < m:
        return None, m

    t = _encode_bases(target)
    p = _encode_bases(primer)
    n_offsets = n - m + 1
    # One (n_offsets, m) view over the target, no copy, then compare against
    # the primer broadcast across rows.
    windows = np.lib.stride_tricks.sliding_window_view(t, m)
    mismatches = (windows != p[None, :]).sum(axis=1)
    # An ambiguity code decoded to -1 never equals a primer base, so it is
    # already counted as a mismatch; nothing more to do.
    best = int(np.argmin(mismatches))
    return best, int(mismatches[best])


def _revcomp(seq: str) -> str:
    from lamp_design.thermo import revcomp

    return revcomp(seq)


def attach_located_windows(
    rows: Sequence[EvalRow],
    targets: Sequence[Tuple[str, str]],
    *,
    max_mismatches: int = DEFAULT_MAX_WINDOW_MISMATCHES,
) -> Tuple[List[EvalRow], Dict[str, int]]:
    """Expand each row to one row per target, with located windows.

    Returns ``(rows, skipped)``. A (set, target) pair is dropped when any of
    the eight primers cannot be placed within ``max_mismatches``: a set that
    does not bind a target at all should be absent from the output rather than
    scored against its least-bad coincidence.
    """
    out: List[EvalRow] = []
    skipped = {"no_placement": 0}
    # Located windows depend only on (primer, target), not on the row, and a
    # panel re-uses primers heavily across conditions.
    cache: Dict[Tuple[str, str], Optional[Tuple[str, int]]] = {}

    for row in rows:
        for tname, tseq in targets:
            windows: List[str] = []
            worst = 0
            ok = True
            for ptype, primer in zip(PRIMER_TYPE_ORDER, row.primers):
                key = (primer, tname)
                if key not in cache:
                    forward = ptype in FORWARD_STRAND_TYPES
                    probe = primer if forward else _revcomp(primer)
                    start, mm = locate_window(probe, tseq)
                    if start is None or mm > max_mismatches:
                        cache[key] = None
                    else:
                        found = tseq[start : start + len(probe)]
                        cache[key] = (found if forward else _revcomp(found), mm)
                hit = cache[key]
                if hit is None:
                    ok = False
                    break
                windows.append(hit[0])
                worst = max(worst, hit[1])
            if not ok:
                skipped["no_placement"] += 1
                continue
            out.append(
                EvalRow(
                    set_id=row.set_id,
                    target_name=tname,
                    primers=row.primers,
                    windows=windows,
                    condition=row.condition,
                    measured_tp=row.measured_tp,
                    max_located_mismatches=worst,
                )
            )
    return out, skipped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``nan`` when it is undefined."""
    if len(x) < 3:
        return math.nan
    from scipy.stats import spearmanr

    rho = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).statistic
    return float(rho)


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Tie-aware ROC-AUC from the Mann-Whitney U statistic.

    ``scores`` is oriented so that higher means more likely positive. Computed
    from ranks rather than a sweep over thresholds so ties contribute 0.5,
    which is what ``sklearn.metrics.roc_auc_score`` does -- avoiding a
    scikit-learn dependency for one number.
    """
    from scipy.stats import rankdata

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_table(
    measured: Sequence[float],
    predicted: Sequence[float],
    *,
    censor_at: float = DEFAULT_CENSOR_MINUTES,
    tp_threshold: float = 30.0,
) -> Dict[str, Dict[str, float]]:
    """Agreement between measured and predicted Tp, two populations.

    ``amplified`` drops rows with no measured Tp; ``censored`` keeps them at
    ``censor_at``. Reporting both is deliberate: the censored figures are the
    ones comparable to the paper, while the amplified ones cannot be inflated
    by the model merely learning which reactions fail.
    """
    m = np.asarray(measured, dtype=float)
    p = np.asarray(predicted, dtype=float)
    amplified = np.isfinite(m)

    out: Dict[str, Dict[str, float]] = {}
    for name, (mm, pp) in {
        "amplified": (m[amplified], p[amplified]),
        "censored": (np.where(amplified, np.minimum(m, censor_at), censor_at), p),
    }.items():
        if len(mm) == 0:
            out[name] = {"n": 0.0}
            continue
        residual = pp - mm
        block = {
            "n": float(len(mm)),
            "spearman_rho": spearman(mm, pp),
            "pearson_r": (
                float(np.corrcoef(mm, pp)[0, 1])
                if len(mm) > 2 and mm.std() > 0 and pp.std() > 0
                else math.nan
            ),
            "mae_minutes": float(np.abs(residual).mean()),
            "rmse_minutes": float(np.sqrt((residual**2).mean())),
            "bias_minutes": float(residual.mean()),
        }
        out[name] = block

    # Classification: did this reaction amplify inside the threshold at all?
    # Scored on -Tp because a faster prediction is a more confident yes.
    labels = amplified & (m <= tp_threshold)
    out["classification"] = {
        "n": float(len(m)),
        "n_positive": float(labels.sum()),
        "tp_threshold_minutes": float(tp_threshold),
        "roc_auc": roc_auc(labels, -p),
    }
    return out


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _primer_set(set_id: str, primers: Sequence[str]) -> PrimerSet:
    """A :class:`PrimerSet` for scoring, with synthetic coordinates.

    ``evaluate`` has no template to place these on -- and the encoder only
    reads ``primers[type].sequence`` -- so positions are laid out end to end
    purely to keep each primer's ``length`` correct and the dataclass honest.
    """
    laid_out: Dict[str, Primer] = {}
    cursor = 0
    for ptype, seq in zip(PRIMER_TYPE_ORDER, primers):
        laid_out[ptype] = Primer(ptype, seq, cursor, cursor + len(seq))
        cursor += len(seq)
    return PrimerSet(set_id=set_id, primers=laid_out, anchor=0, geometry_id="evaluate")


def _subset(thermo: ThermoTable, indices: Sequence[int]) -> ThermoTable:
    """A ThermoTable holding only ``indices``, preserving their order."""
    idx = list(indices)
    return ThermoTable(
        set_ids=[thermo.set_ids[i] for i in idx],
        target_names=[thermo.target_names[i] for i in idx],
        values=thermo.values[idx],
        globals_=thermo.globals_[idx],
        target_windows=thermo.target_windows[idx],
    )


def evaluate(
    rows: Sequence[EvalRow],
    *,
    weights: Optional[str | Path] = None,
    ensemble: Optional["LyraEnsemble"] = None,
    model: str = "simplified",
    conditions: Optional[Sequence[Condition]] = None,
    censor_at: float = DEFAULT_CENSOR_MINUTES,
    tp_threshold: float = 30.0,
    device: str = "auto",
    batch_size: Optional[int] = None,
    pair_chunk: int = PAIR_CHUNK,
    threads: Optional[int] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> EvaluationResult:
    """Score ``rows`` and, where they carry a measured Tp, report agreement.

    Every row must already have its ``windows`` populated -- by the input
    columns or by :func:`attach_located_windows`. When ``conditions`` is given
    each row is scored under all of them (a cross product, as in ``design``);
    otherwise each row is scored under its own :attr:`EvalRow.condition`.
    """
    from lamp_design.models.bridge import predict_tp
    from lamp_design.models.inference import LyraEnsemble
    from lamp_design.thermo import thermo_from_windows

    say = progress or (lambda stage, message: None)
    device = resolve_device(device)
    if batch_size is None:
        batch_size = BATCH_SIZE_GPU if device.startswith("cuda") else BATCH_SIZE_CPU

    missing = [i for i, r in enumerate(rows) if not r.windows]
    if missing:
        raise InputError(
            f"{len(missing)} row(s) have no target windows (first: index "
            f"{missing[0]}, set {rows[missing[0]].set_id}). Supply "
            "<type>_target_sequence columns, or pass --targets to locate them."
        )

    # Cross-product mode re-scores each input row under each requested
    # condition; otherwise the row's own condition is the only one.
    if conditions:
        scored_rows = [
            EvalRow(
                set_id=r.set_id,
                target_name=r.target_name,
                primers=r.primers,
                windows=r.windows,
                condition=c,
                measured_tp=r.measured_tp if c == r.condition else None,
                max_located_mismatches=r.max_located_mismatches,
            )
            for r in rows
            for c in conditions
        ]
    else:
        scored_rows = list(rows)

    timings: Dict[str, float] = {}

    # Thermo is per (set, target, windows) and independent of condition, so it
    # is computed over the deduplicated pairs and then fanned back out.
    t = time.perf_counter()
    pair_key: Dict[Tuple[str, ...], int] = {}
    pair_of_row: List[int] = []
    uniq_sets: List[str] = []
    uniq_targets: List[str] = []
    uniq_primers: List[List[str]] = []
    uniq_windows: List[List[str]] = []
    for row in scored_rows:
        assert row.windows is not None  # checked above
        key = (row.set_id, row.target_name, *row.primers, *row.windows)
        if key not in pair_key:
            pair_key[key] = len(uniq_sets)
            uniq_sets.append(row.set_id)
            uniq_targets.append(row.target_name)
            uniq_primers.append(list(row.primers))
            uniq_windows.append(list(row.windows))
        pair_of_row.append(pair_key[key])

    thermo = thermo_from_windows(
        uniq_sets, uniq_targets, uniq_primers, uniq_windows, threads=threads
    )
    timings["thermo_s"] = time.perf_counter() - t
    say("thermo", f"{len(uniq_sets):,} unique (set, target) pairs")

    if model not in {"simplified", "fusion"}:
        raise InputError(
            f"unknown model {model!r}; choose 'simplified' or 'fusion'."
        )

    t = time.perf_counter()
    fusion = None
    if model == "fusion":
        from lamp_design.models.fusion import load_fusion

        if not device.startswith("cuda"):
            # Measured, and worth shouting about: TabPFN's cost is the
            # 34,283-row context it re-encodes on every call, NOT the number
            # of rows being scored. One CPU forward pass took 4,587 s and
            # there are four estimators, so a CPU run is roughly five hours
            # whether you supply three primer sets or three hundred. The
            # usual intuition -- small input, quick run -- is wrong here, and
            # five hours of silence is indistinguishable from a hang.
            #
            # Printed as a block rather than a log line because a single line
            # scrolls past; this is the last moment before the wait starts.
            # Not a prompt: evaluate has to stay scriptable.
            for line in (
                "",
                "=" * 72,
                "  WARNING: Fusion on CPU is not recommended.",
                "",
                "  This run will take roughly FIVE HOURS, and that does not",
                "  depend on how many primer sets you gave it -- TabPFN",
                "  re-encodes its 34,283-row training context on every call,",
                "  so three sets cost the same as three hundred.",
                "",
                "    --device cuda        about 90 seconds",
                "    --model simplified   under a second",
                "",
                "  Continuing anyway. Press Ctrl-C to stop.",
                "=" * 72,
                "",
            ):
                print(line, flush=True)
        fusion = load_fusion(device=device)
    elif ensemble is None:
        ensemble = LyraEnsemble.load(weights, device=device)
    timings["model_load_s"] = time.perf_counter() - t

    # A set_id can appear with different primer sequences only if the input is
    # inconsistent; the lookup takes the first, and the dedup key above means
    # a genuine collision would already have produced separate pairs.
    sets = {s: _primer_set(s, p) for s, p in zip(uniq_sets, uniq_primers)}

    t = time.perf_counter()
    by_condition: Dict[Condition, List[int]] = {}
    for row_idx, row in enumerate(scored_rows):
        by_condition.setdefault(row.condition, []).append(row_idx)

    predicted = [math.nan] * len(scored_rows)
    std = [math.nan] * len(scored_rows)
    unscored = 0

    if fusion is not None:
        # ONE call for every row, across every condition. TabPFN re-encodes
        # its 34,283-row context on each predict() -- ~78 s regardless of how
        # many rows it is given, with no caching between calls -- so looping
        # per condition would multiply that cost rather than amortise it.
        from lamp_design.models.fusion import frame_from_thermo
        from lampbench.data import derive_features

        primer_sequences = {
            set_id: {ptype: pr.sequence for ptype, pr in ps.primers.items()}
            for set_id, ps in sets.items()
        }
        conditions_in_order = list(by_condition)
        raw = frame_from_thermo(thermo, conditions_in_order, primer_sequences)
        blended = fusion.predict(derive_features(raw))

        # frame_from_thermo emits rows pair-major, condition-minor.
        n_conditions = len(conditions_in_order)
        slot_of_condition = {c: i for i, c in enumerate(conditions_in_order)}
        for row_idx, row in enumerate(scored_rows):
            flat = pair_of_row[row_idx] * n_conditions + slot_of_condition[row.condition]
            predicted[row_idx] = float(blended[flat])
            # No std: the four branches are different model families, so the
            # spread between them is not an uncertainty the way the Lyra
            # ensemble's seed spread is.
        timings["inference_s"] = time.perf_counter() - t
        say("inference", f"{len(scored_rows):,} fusion predictions on {device}")
    else:
        for condition, row_indices in by_condition.items():
            # One sub-table per condition, holding just the pairs that condition
            # needs, so predict_tp's single-condition path does the right thing.
            pair_indices = sorted({pair_of_row[i] for i in row_indices})
            slot_of_pair = {p: s for s, p in enumerate(pair_indices)}
            preds = predict_tp(
                _subset(thermo, pair_indices),
                sets,
                [condition],
                ensemble=ensemble,
                batch_size=batch_size,
                pair_chunk=pair_chunk,
            )
            # predict_tp returns one Prediction per surviving pair, in pair order.
            by_slot = {i: pred for i, pred in enumerate(preds)}
            for row_idx in row_indices:
                slot = slot_of_pair[pair_of_row[row_idx]]
                pred = by_slot.get(slot)
                if pred is None:
                    unscored += 1
                    continue
                predicted[row_idx] = pred.tp_minutes
                std[row_idx] = pred.tp_std
        timings["inference_s"] = time.perf_counter() - t
        say("inference", f"{len(scored_rows) - unscored:,} predictions on {device}")

    measured = [
        r.measured_tp if r.measured_tp is not None else math.nan for r in scored_rows
    ]
    have_truth = any(r.measured_tp is not None for r in scored_rows)
    metrics = (
        metrics_table(
            measured, predicted, censor_at=censor_at, tp_threshold=tp_threshold
        )
        if have_truth
        else {}
    )

    result = EvaluationResult(
        rows=scored_rows,
        predicted_tp=predicted,
        predicted_std=std,
        metrics=metrics,
        skipped={"unscored_pairs": unscored} if unscored else {},
        params={
            "n_rows": len(scored_rows),
            "n_unique_pairs": len(uniq_sets),
            "n_sets": len(set(uniq_sets)),
            "n_targets": len(set(uniq_targets)),
            "conditions": sorted({str(r.condition) for r in scored_rows}),
            "censor_at_minutes": censor_at,
            "tp_threshold_minutes": tp_threshold,
            "device": device,
            "batch_size": batch_size,
            "has_measured_tp": have_truth,
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
        },
    )
    say("report", f"{result.n_scored:,} rows scored")
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_PREDICTION_COLUMNS = (
    "primer_set",
    "target",
    "master_mix",
    "target_conc",
    "predicted_tp_minutes",
    "predicted_tp_std",
    "measured_tp_minutes",
    "amplified",
    "residual_minutes",
    "total_mismatches",
)


def predictions_rows(result: EvaluationResult) -> List[Dict[str, str]]:
    """The per-row prediction table, as strings ready for CSV."""
    out: List[Dict[str, str]] = []
    for row, pred, sd in zip(result.rows, result.predicted_tp, result.predicted_std):
        measured = row.measured_tp
        amplified = measured is not None and math.isfinite(measured)
        residual = pred - measured if amplified and math.isfinite(pred) else math.nan
        out.append(
            {
                "primer_set": row.set_id,
                "target": row.target_name,
                "master_mix": row.condition.master_mix,
                "target_conc": f"{row.condition.target_conc:g}",
                "predicted_tp_minutes": "" if math.isnan(pred) else f"{pred:.3f}",
                "predicted_tp_std": "" if math.isnan(sd) else f"{sd:.3f}",
                "measured_tp_minutes": (
                    f"{measured:.3f}" if amplified else ("" if measured is None else "NA")
                ),
                "amplified": "" if measured is None else ("1" if amplified else "0"),
                "residual_minutes": "" if math.isnan(residual) else f"{residual:+.3f}",
                "total_mismatches": (
                    "" if row.max_located_mismatches is None
                    else str(row.max_located_mismatches)
                ),
            }
        )
    return out


def format_summary(result: EvaluationResult) -> str:
    """Human-readable run summary."""
    p = result.params
    lines = [
        "lamp-design evaluate",
        "=" * 60,
        f"rows scored        {p.get('n_rows', 0):,}",
        f"unique pairs       {p.get('n_unique_pairs', 0):,} "
        f"({p.get('n_sets', 0)} set(s) x {p.get('n_targets', 0)} target(s))",
        f"conditions         {', '.join(p.get('conditions', []))}",
        f"device             {p.get('device', '?')}",
    ]
    timings = p.get("timings_s", {})
    if timings:
        total = sum(timings.values())
        lines.append(
            "timings            "
            + ", ".join(f"{k.removesuffix('_s')}={v:.2f}s" for k, v in timings.items())
            + f" (total {total:.2f}s)"
        )

    finite = [t for t in result.predicted_tp if math.isfinite(t)]
    if finite:
        arr = np.asarray(finite)
        lines += [
            "",
            "Predicted Tp (minutes)",
            "-" * 60,
            f"  min {arr.min():6.2f}   median {np.median(arr):6.2f}   "
            f"max {arr.max():6.2f}   mean {arr.mean():6.2f}",
        ]

    if result.metrics:
        lines += ["", "Agreement with measured Tp", "-" * 60]
        for name in ("amplified", "censored"):
            block = result.metrics.get(name, {})
            if not block or not block.get("n"):
                continue
            label = (
                "amplified only" if name == "amplified"
                else f"all rows, censored at {p.get('censor_at_minutes', 60):g} min"
            )
            lines.append(f"  {label} (n={int(block['n']):,})")
            lines.append(
                f"    Spearman rho {block.get('spearman_rho', float('nan')):+.3f}   "
                f"Pearson r {block.get('pearson_r', float('nan')):+.3f}"
            )
            lines.append(
                f"    MAE {block.get('mae_minutes', float('nan')):.2f} min   "
                f"RMSE {block.get('rmse_minutes', float('nan')):.2f} min   "
                f"bias {block.get('bias_minutes', float('nan')):+.2f} min"
            )
        clf = result.metrics.get("classification", {})
        if clf:
            lines.append(
                f"  amplified within {clf.get('tp_threshold_minutes', 30):g} min: "
                f"ROC-AUC {clf.get('roc_auc', float('nan')):.3f} "
                f"({int(clf.get('n_positive', 0)):,} positive "
                f"of {int(clf.get('n', 0)):,})"
            )
    else:
        lines += [
            "",
            "No measured Tp column in the input, so no agreement metrics.",
        ]

    if result.skipped:
        lines += ["", "Skipped", "-" * 60]
        lines += [f"  {k:<20} {v:,}" for k, v in result.skipped.items()]
    if result.warnings:
        lines += ["", "Warnings", "-" * 60]
        lines += [f"  {w}" for w in result.warnings]
    return "\n".join(lines) + "\n"


def write_outputs(result: EvaluationResult, prefix: str | Path) -> Dict[str, Path]:
    """Write ``<prefix>.predictions.csv``, ``.metrics.json`` and ``.summary.txt``."""
    prefix = Path(prefix)
    if prefix.parent != Path(""):
        prefix.parent.mkdir(parents=True, exist_ok=True)

    # Concatenated, not `with_suffix`: a prefix like `results.v2` would
    # otherwise lose its `.v2` -- same convention as `report.write_outputs`.
    paths = {
        "predictions_csv": Path(f"{prefix}.predictions.csv"),
        "metrics_json": Path(f"{prefix}.metrics.json"),
        "summary_txt": Path(f"{prefix}.summary.txt"),
    }
    with open(paths["predictions_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_PREDICTION_COLUMNS))
        writer.writeheader()
        writer.writerows(predictions_rows(result))
    with open(paths["metrics_json"], "w") as handle:
        json.dump(
            {
                "params": result.params,
                "metrics": result.metrics,
                "skipped": result.skipped,
                "warnings": result.warnings,
            },
            handle,
            indent=2,
            default=str,
        )
    paths["summary_txt"].write_text(format_summary(result))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Register the evaluate subcommand."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Score primer sets you already have",
        description="""
Score existing eight-primer LAMP sets with the trained Simplified Lyra
ensemble, instead of designing new ones: the same thermo + encoding +
inference path as `design`, without candidate generation.

--in is a CSV (or TSV) with one column per primer -- 'F3_sequence' or 'F3',
for all of F3 F2 FL F1c B1c BL B2 B3. The target windows each primer binds
either come from matching '<type>_target_sequence' columns, or are located in
--targets by best Hamming match.

Optional columns are used when present: 'primer_set' and 'target_seq' for
names, 'master_mix' and 'target_conc' for the condition each row is scored
under, and 'tp' for a measured Tp -- which turns the run into a validation and
prints Spearman, MAE and ROC-AUC against it.

--out is a path PREFIX: the run writes <prefix>.predictions.csv,
<prefix>.metrics.json and <prefix>.summary.txt.
""",
    )
    parser.add_argument(
        "--in", dest="input", required=True,
        help="Primer sets to score: a CSV/TSV with eight primer columns",
    )
    parser.add_argument(
        "--targets", default=None,
        help="Target FASTA, for locating target windows. Not needed when the "
             "input carries <type>_target_sequence columns",
    )
    parser.add_argument(
        "--out", dest="output", required=True,
        help="Output path prefix (writes .predictions.csv, .metrics.json, .summary.txt)",
    )
    parser.add_argument(
        "--windows", choices=["auto", "csv", "locate"], default="auto",
        help="Where target windows come from (default: %(default)s -- the input "
             "columns when present, otherwise located in --targets)",
    )
    parser.add_argument(
        "--max-window-mismatches", dest="max_window_mismatches", type=int,
        default=DEFAULT_MAX_WINDOW_MISMATCHES,
        help="With --windows locate, reject a placement worse than this and drop "
             "the (set, target) pair (default: %(default)s)",
    )
    parser.add_argument(
        "--master-mix", dest="master_mixes", action="append", choices=list(MASTER_MIXES),
        help="Score every row under this mix instead of its own; repeatable, and "
             "repeating it makes a cross product",
    )
    parser.add_argument(
        "--conc", dest="concentrations", action="append", type=float,
        help="Score every row at this concentration in copies/reaction instead of "
             "its own; repeatable",
    )
    parser.add_argument(
        "--truth-column", dest="truth_column", default=None,
        help=f"Column holding measured Tp in minutes (default: the first of "
             f"{list(_TRUTH_COLUMNS)} present, if any)",
    )
    parser.add_argument(
        "--censor-at", dest="censor_at", type=float, default=DEFAULT_CENSOR_MINUTES,
        help="Measured Tp assigned to a row that did not amplify, for the censored "
             "metrics (default: %(default)s)",
    )
    parser.add_argument(
        "--tp-threshold", dest="tp_threshold", type=float, default=30.0,
        help="Tp in minutes below which a reaction counts as amplified, for the "
             "ROC-AUC (default: %(default)s)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: %(default)s -- cuda when available)",
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, default=None,
        help=f"Rows per forward pass (default: {BATCH_SIZE_CPU} on CPU, "
             f"{BATCH_SIZE_GPU} on GPU)",
    )
    parser.add_argument(
        "--pair-chunk", dest="pair_chunk", type=int, default=PAIR_CHUNK,
        help="Pairs encoded at a time; caps peak memory (default: %(default)s)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Threads for thermodynamics (default: all cores)",
    )
    # Deliberately absent from `design`. Fusion is ~29x slower per prediction
    # and a design run projects to ~1.9 h on GPU, so the option does not
    # exist there -- not a cap, not a warning. Scoring a handful of supplied
    # sets is the workload Fusion was built for.
    parser.add_argument(
        "--model", choices=("simplified", "fusion"), default="simplified",
        help="Which model to score with. 'simplified' is the 5-seed Lyra "
             "ensemble (default). 'fusion' is the four-model blend -- more "
             "accurate in-distribution, far slower, and needs the [fusion] "
             "extra installed (default: %(default)s)",
    )
    from lamp_design.cli import add_weights_argument

    add_weights_argument(parser)
    parser.set_defaults(func=run)


def run(args) -> EvaluationResult:
    """Run the evaluate command."""
    from lamp_design.cli import resolve_weights

    weights = resolve_weights(getattr(args, "weights", None))

    override_mix = args.master_mixes[0] if args.master_mixes else None
    override_conc = args.concentrations[0] if args.concentrations else None
    # A single --master-mix/--conc overrides each row's own condition; several
    # mean a cross product, which build_conditions assembles below.
    cross = bool(args.master_mixes and args.concentrations) and (
        len(args.master_mixes) > 1 or len(args.concentrations) > 1
    )

    rows, warnings = load_rows(
        args.input,
        master_mix=None if cross else override_mix,
        target_conc=None if cross else override_conc,
        truth_column=args.truth_column,
        require_windows=args.windows == "csv",
    )
    print(f"Input: {len(rows):,} row(s) from {args.input}")

    need_locate = args.windows == "locate" or (
        args.windows == "auto" and any(r.windows is None for r in rows)
    )
    skipped: Dict[str, int] = {}
    if need_locate:
        if not args.targets:
            raise InputError(
                "the input does not carry <type>_target_sequence columns, so "
                "--targets is required to locate each primer's binding site "
                "(or pass --windows csv to insist the columns be present)"
            )
        targets = read_fasta(args.targets)
        if not targets:
            raise InputError(f"{args.targets} contains no sequences")
        rows, skipped = attach_located_windows(
            rows, targets, max_mismatches=args.max_window_mismatches
        )
        print(
            f"Located windows in {len(targets)} target(s): "
            f"{len(rows):,} (set, target) pair(s)"
            + (
                f", {skipped['no_placement']:,} dropped for no placement within "
                f"{args.max_window_mismatches} mismatches"
                if skipped.get("no_placement")
                else ""
            )
        )
        if not rows:
            raise InputError(
                "no (set, target) pair had all eight primers placeable within "
                f"{args.max_window_mismatches} mismatches. Raise "
                "--max-window-mismatches, or check that --targets holds the "
                "right sequences."
            )

    conditions = (
        build_conditions(args.master_mixes, args.concentrations) if cross else None
    )

    started = time.perf_counter()
    result = evaluate(
        rows,
        weights=weights,
        model=args.model,
        conditions=conditions,
        censor_at=args.censor_at,
        tp_threshold=args.tp_threshold,
        device=args.device,
        batch_size=args.batch_size,
        pair_chunk=args.pair_chunk,
        threads=args.threads,
        progress=lambda stage, message: print(f"  {stage:<10} {message}"),
    )
    result.warnings.extend(warnings)
    result.skipped.update(skipped)

    paths = write_outputs(result, args.output)
    print()
    print(format_summary(result), end="")
    print(f"\nDone in {time.perf_counter() - started:.1f}s.")
    for key, path in paths.items():
        print(f"  {key:<16} {path}")
    return result
