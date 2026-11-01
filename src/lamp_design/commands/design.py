"""``lamp-design design`` -- the full pipeline, end to end.

    FASTA/panel -> generation -> thermo -> encoding -> inference -> report

This is the module the CLI's self-replacing stub mechanism picks up: its
presence, with a ``register(subparsers)``, is what makes the ``design`` stub in
``cli.py`` disappear. It is also the single entry point the Streamlit and Modal
front ends call, so the orchestration lives here once rather than three times --
see :func:`design` for the library form that takes plain values instead of an
argparse namespace.

## Conditions, and which one ranks

A LAMP reaction's speed depends on the master mix and the template
concentration, so every candidate is scored under each requested
``(master_mix, concentration)`` combination -- that is the ``x2 x2`` in the
design notes' compute budget. But a *ranking* needs one number per candidate,
so one condition is nominated as the ranking condition (by default the first
master mix at the highest concentration) and the rest are reported alongside.
Ranking under the easiest condition is deliberate: it answers "how fast can
this set go", and the per-condition columns show how it degrades.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from lamp_design.commands.prepare import (
    DEFAULT_MAX_BUDGET,
    InputError,
    load_panel,
    prepare_panel,
)
from lamp_design.generation import (
    DEFAULT_STRIDE,
    default_geometries,
    random_geometries,
)
from lamp_design.report import (
    DEFAULT_MIN_SEPARATION_BP,
    DEFAULT_N_SETS,
    DEFAULT_TP_THRESHOLD_MINUTES,
    build_design_result,
    write_outputs,
)
from lamp_design.thermo import compute_thermo
from lamp_design.types import (
    MASTER_MIXES,
    PAIR_CHUNK,
    Condition,
    DesignResult,
    TargetPanel,
)
from lamp_design.utils.sequences import read_fasta

if TYPE_CHECKING:  # pragma: no cover
    from lamp_design.models.inference import LyraEnsemble

#: Master mix scored by default. One mix, not two: ranking only ever uses a
#: single condition (see `score_sets`), so a second mix quadrupled inference --
#: 89% of runtime -- without changing which sets come back.
#:
#: ``SS`` is the one, on training coverage rather than preference. Across the
#: curated corpus (170,175 rows in datasets/LAMP/final) the split is
#: SS 53.7% / BST 38.9% / WS 7.4%, so SS and BST are both well represented.
#: The tiebreaker is *where*: BST appears in no LASV, Systematic or Borrelia
#: row, so the held-out pathogen validation in docs/ARCHITECTURE.md (LASV
#: ROC-AUC 0.875, Borrelia 0.984) contains no BST measurements at all. Ranking
#: under SS is ranking under the only mix those numbers actually cover.
#:
#: ``BST`` and ``WS`` remain selectable with ``--master-mix``.
DEFAULT_MASTER_MIXES: Tuple[str, ...] = ("SS",)

#: Template concentrations scored by default, in copies per reaction: one
#: comfortable and one near the limit of detection. Both are scored for every
#: candidate; the higher one ranks (`_ranking_condition`) and the lower one is
#: the robustness column, which is the question a diagnostic actually turns on.
DEFAULT_CONCENTRATIONS: Tuple[float, ...] = (10_000.0, 100.0)

#: Rows per forward pass, same on CPU and GPU.
#:
#: Measured on an L4 (2026-10-30, 74,640 predictions): 512 -> 2,480 pred/s,
#: 4096 -> 2,323, 16384 -> 2,308. Bigger batches are slightly *worse*, and the
#: GPU sat at 18-29% utilization throughout, because the pipeline is bound by
#: single-threaded row encoding rather than by the model. So there is nothing to
#: gain from a GPU-specific batch size until `encode_pairs` is parallelized --
#: at which point re-measure rather than assuming bigger is better.
BATCH_SIZE_CPU = 512
BATCH_SIZE_GPU = 512

__all__ = [
    "BATCH_SIZE_CPU",
    "BATCH_SIZE_GPU",
    "DEFAULT_CONCENTRATIONS",
    "DEFAULT_MASTER_MIXES",
    "build_conditions",
    "design",
    "load_input_panel",
    "register",
    "resolve_device",
    "run",
]


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_input_panel(
    path: str | Path,
    *,
    max_budget: int = DEFAULT_MAX_BUDGET,
    pre_aligned: bool = False,
    threads: int = 1,
) -> Tuple[TargetPanel, Optional[object]]:
    """Load either a ``prepare`` panel JSON or a raw FASTA.

    Returns ``(panel, prepared_or_None)``; the second element is the
    :class:`~lamp_design.commands.prepare.PreparedPanel` when this call did the
    preparing, so its warnings and exclusions can be carried onto the result.
    A panel JSON has already been QC'd, so it is used as-is.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"input does not exist: {path}")

    # Sniff rather than trust the extension: panels are JSON objects, FASTA
    # starts with '>'. A mislabelled file is a common enough mistake that
    # guessing from the suffix alone produces a confusing error.
    with open(path, "r") as handle:
        first = handle.read(1)
    if first == "{":
        try:
            return load_panel(path), None
        except (json.JSONDecodeError, KeyError) as exc:
            raise InputError(
                f"{path} looks like JSON but is not a panel written by "
                f"`lamp-design prepare`: {exc}"
            ) from exc
    if first != ">":
        raise InputError(
            f"{path} is neither FASTA (expected '>' first) nor a panel JSON "
            f"from `lamp-design prepare` (expected '{{'); first character is {first!r}"
        )

    prepared = prepare_panel(
        read_fasta(path),
        max_budget=max_budget,
        pre_aligned=pre_aligned,
        threads=threads,
        panel_name=path.stem,
    )
    return prepared.panel, prepared


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def build_conditions(
    master_mixes: Sequence[str] = DEFAULT_MASTER_MIXES,
    concentrations: Sequence[float] = DEFAULT_CONCENTRATIONS,
) -> List[Condition]:
    """The cross product of mixes and concentrations, mixes-major."""
    unknown = [m for m in master_mixes if m not in MASTER_MIXES]
    if unknown:
        raise InputError(
            f"unknown master mix(es) {unknown}; the model was trained on "
            f"{list(MASTER_MIXES)}"
        )
    if not master_mixes or not concentrations:
        raise InputError("at least one master mix and one concentration are required")
    bad = [c for c in concentrations if c <= 0]
    if bad:
        raise InputError(f"concentrations must be positive, got {bad}")
    return [Condition(m, float(c)) for m in master_mixes for c in concentrations]


def _ranking_condition(
    conditions: Sequence[Condition],
    master_mix: Optional[str],
    conc: Optional[float],
) -> Condition:
    """Pick the condition sets are ranked under; must be one being scored."""
    if master_mix is None and conc is None:
        # First mix at its highest concentration: "how fast can this go".
        first_mix = conditions[0].master_mix
        return max(
            (c for c in conditions if c.master_mix == first_mix),
            key=lambda c: c.target_conc,
        )
    mix = master_mix if master_mix is not None else conditions[0].master_mix
    target = conc if conc is not None else max(c.target_conc for c in conditions)
    for cond in conditions:
        if cond.master_mix == mix and cond.target_conc == target:
            return cond
    raise InputError(
        f"ranking condition {mix}@{target:g} is not among the conditions being "
        f"scored ({[str(c) for c in conditions]}). Add it with --master-mix / "
        "--conc, or pick one of those."
    )


def resolve_device(requested: str = "auto") -> str:
    """``auto`` -> ``cuda`` when a GPU is actually usable, else ``cpu``."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken: CPU is the honest answer
        return "cpu"


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def design(
    panel: TargetPanel,
    *,
    weights: Optional[str | Path] = None,
    ensemble: Optional["LyraEnsemble"] = None,
    conditions: Optional[Sequence[Condition]] = None,
    ranking_condition: Optional[Condition] = None,
    stride: int = DEFAULT_STRIDE,
    geometries: Optional[Sequence] = None,
    max_candidates: Optional[int] = None,
    n: int = DEFAULT_N_SETS,
    min_separation: int = DEFAULT_MIN_SEPARATION_BP,
    tp_threshold: float = DEFAULT_TP_THRESHOLD_MINUTES,
    device: str = "auto",
    batch_size: Optional[int] = None,
    pair_chunk: int = PAIR_CHUNK,
    threads: Optional[int] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> DesignResult:
    """Design primer sets against ``panel``. The library entry point.

    Everything the CLI, Streamlit and Modal front ends do funnels through here,
    so there is one definition of what "a design run" means. ``progress`` is
    called as ``progress(stage, message)`` after each stage.
    """
    # Imported here, not at module scope: `cli.build_parser` imports this
    # module to register the subcommand, and these two pull in torch. Keeping
    # them local is what makes `lamp-design --help` and `prepare` torch-free.
    from lamp_design.generation import generate_candidates
    from lamp_design.models.bridge import mismatch_map, predict_tp
    from lamp_design.models.inference import LyraEnsemble

    say = progress or (lambda stage, message: None)
    conditions = list(conditions) if conditions else build_conditions()
    ranking_condition = ranking_condition or _ranking_condition(conditions, None, None)
    device = resolve_device(device)
    if batch_size is None:
        batch_size = BATCH_SIZE_GPU if device.startswith("cuda") else BATCH_SIZE_CPU
    geometries = list(geometries) if geometries else default_geometries()

    timings: Dict[str, float] = {}

    t = time.perf_counter()
    candidates = generate_candidates(
        panel.consensus,
        geometries=geometries,
        stride=stride,
        max_candidates=max_candidates,
    )
    timings["generation_s"] = time.perf_counter() - t
    say("generation", f"{len(candidates):,} candidate sets")

    t = time.perf_counter()
    thermo = compute_thermo(candidates, panel, threads=threads)
    timings["thermo_s"] = time.perf_counter() - t
    say(
        "thermo",
        f"{len(thermo.set_ids):,} (set, target) pairs "
        f"[{len(thermo.set_ids) * 8:,} primer-target windows]",
    )

    t = time.perf_counter()
    if ensemble is None:
        ensemble = LyraEnsemble.load(weights, device=device)
    timings["model_load_s"] = time.perf_counter() - t

    t = time.perf_counter()
    predictions = predict_tp(
        thermo,
        candidates,
        conditions,
        ensemble=ensemble,
        batch_size=batch_size,
        pair_chunk=pair_chunk,
    )
    timings["inference_s"] = time.perf_counter() - t
    say(
        "inference",
        f"{len(predictions):,} predictions over {len(conditions)} condition(s) "
        f"on {device}",
    )

    t = time.perf_counter()
    result = build_design_result(
        predictions,
        candidates,
        panel,
        ranking_condition=ranking_condition,
        tp_threshold=tp_threshold,
        n=n,
        min_separation=min_separation,
        mismatches=mismatch_map(thermo),
    )
    timings["report_s"] = time.perf_counter() - t
    say("report", f"{len(result.scored_sets)} sets returned")

    result.params.update(
        {
            "stride": stride,
            # `geometry_id`, not `name`: Geometry has no `name`, so a
            # `getattr(g, "name", str(g))` here silently recorded the whole
            # dataclass repr instead of 'canonical'.
            "geometries": [
                getattr(g, "geometry_id", None) or str(g) for g in geometries
            ],
            "conditions": [str(c) for c in conditions],
            "ranking_condition": str(ranking_condition),
            "tp_threshold_minutes": tp_threshold,
            "n_requested": n,
            "min_separation_bp": min_separation,
            "device": device,
            "batch_size": batch_size,
            "n_candidates": len(candidates),
            "n_pairs": len(thermo.set_ids),
            "n_predictions": len(predictions),
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
        }
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Register the design subcommand."""
    parser = subparsers.add_parser(
        "design",
        help="Run the full design pipeline",
        description="""
Generate candidate eight-primer LAMP sets across the target, score every
candidate against every target sequence with the trained Simplified Lyra
ensemble, and return the top N positionally-diverse sets.

Input is either a FASTA (prepared on the fly) or a panel JSON from
`lamp-design prepare`. --out is a path PREFIX: the run writes
<prefix>.sets.csv, <prefix>.per_target.csv, <prefix>.json and
<prefix>.summary.txt.

Runtime is dominated by model inference, not by thermodynamics, so --device
cuda is worth far more than --threads. Use --stride to trade candidate
coverage for time: stride 1 evaluates every anchor position.
""",
    )
    parser.add_argument(
        "--in", dest="input", required=True,
        help="Input FASTA, or a panel JSON from `lamp-design prepare`",
    )
    parser.add_argument(
        "--out", dest="output", required=True,
        help="Output path prefix (writes .sets.csv, .per_target.csv, .json, .summary.txt)",
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_N_SETS,
        help="Number of positionally-diverse sets to return (default: %(default)s)",
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help="Anchor spacing in nt; 1 evaluates every position (default: %(default)s)",
    )
    parser.add_argument(
        "--min-separation", dest="min_separation", type=int,
        default=DEFAULT_MIN_SEPARATION_BP,
        help="Minimum nt between the anchors of two returned sets (default: %(default)s)",
    )
    parser.add_argument(
        "--master-mix", dest="master_mixes", action="append", choices=list(MASTER_MIXES),
        help=f"Master mix to score; repeatable (default: {' '.join(DEFAULT_MASTER_MIXES)})",
    )
    parser.add_argument(
        "--conc", dest="concentrations", action="append", type=float,
        help="Template concentration in copies/reaction; repeatable "
             f"(default: {' '.join(f'{c:g}' for c in DEFAULT_CONCENTRATIONS)})",
    )
    parser.add_argument(
        "--rank-master-mix", dest="rank_master_mix", choices=list(MASTER_MIXES),
        default=None,
        help="Master mix to rank under (default: the first --master-mix)",
    )
    parser.add_argument(
        "--rank-conc", dest="rank_conc", type=float, default=None,
        help="Concentration to rank under (default: the highest --conc)",
    )
    parser.add_argument(
        "--tp-threshold", dest="tp_threshold", type=float,
        default=DEFAULT_TP_THRESHOLD_MINUTES,
        help="Tp in minutes below which a target counts as amplified, for the "
             "inclusivity column (default: %(default)s)",
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
        "--max-candidates", dest="max_candidates", type=int, default=None,
        help="Hard cap on candidate sets, applied during generation",
    )
    parser.add_argument(
        "--random-geometries", dest="random_geometries", type=int, default=None,
        help="Use N randomly-sampled geometries instead of the deterministic set",
    )
    parser.add_argument(
        "--geometry-seed", dest="geometry_seed", type=int, default=0,
        help="Seed for --random-geometries (default: %(default)s)",
    )
    parser.add_argument(
        "--max-budget", dest="max_budget", type=int, default=DEFAULT_MAX_BUDGET,
        help="Cap on n_targets x alignment width, when preparing a FASTA "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--pre-aligned", dest="pre_aligned", action="store_true",
        help="A FASTA input is already an alignment: accept gap characters",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Threads for thermodynamics and alignment (default: all cores)",
    )
    from lamp_design.cli import add_weights_argument

    add_weights_argument(parser)
    parser.set_defaults(func=run)


def run(args) -> DesignResult:
    """Run the design command."""
    from lamp_design.cli import resolve_weights

    weights = resolve_weights(getattr(args, "weights", None))

    panel, prepared = load_input_panel(
        args.input,
        max_budget=args.max_budget,
        pre_aligned=args.pre_aligned,
        threads=args.threads or 1,
    )
    print(
        f"Target panel: {panel.n_targets} sequence(s), "
        f"{len(panel.consensus):,} nt consensus"
    )
    if prepared is not None:
        for name, reason in prepared.excluded.items():
            print(f"  EXCLUDED {name}: {reason}")
        for warning in prepared.warnings:
            print(f"  WARNING: {warning}")

    conditions = build_conditions(
        args.master_mixes or DEFAULT_MASTER_MIXES,
        args.concentrations or DEFAULT_CONCENTRATIONS,
    )
    ranking_condition = _ranking_condition(
        conditions, args.rank_master_mix, args.rank_conc
    )
    geometries = (
        random_geometries(args.random_geometries, seed=args.geometry_seed)
        if args.random_geometries
        else default_geometries()
    )

    started = time.perf_counter()
    result = design(
        panel,
        weights=weights,
        conditions=conditions,
        ranking_condition=ranking_condition,
        stride=args.stride,
        geometries=geometries,
        max_candidates=args.max_candidates,
        n=args.top,
        min_separation=args.min_separation,
        tp_threshold=args.tp_threshold,
        device=args.device,
        batch_size=args.batch_size,
        pair_chunk=args.pair_chunk,
        threads=args.threads,
        progress=lambda stage, message: print(f"  {stage:<10} {message}"),
    )
    if prepared is not None:
        prepared.apply_to(result)

    paths = write_outputs(
        result,
        args.output,
        ranking_condition=ranking_condition,
        tp_threshold=args.tp_threshold,
    )
    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s. Ranked under {ranking_condition}.")
    for key, path in paths.items():
        print(f"  {key:<16} {path}")

    from lamp_design.utils.diagnostics import NoCandidatesError

    if not result.scored_sets:
        raise NoCandidatesError(
            "no primer sets survived the pipeline. "
            + "; ".join(f"{k}={v}" for k, v in result.funnel.items())
        )
    return result
