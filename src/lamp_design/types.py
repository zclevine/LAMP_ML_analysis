"""Core data contracts for the LAMP design engine.

Every module in the pipeline consumes and produces the types defined here.
They are deliberately plain (dataclasses + numpy arrays) so that candidate
generation, thermodynamic scoring, and model inference can be developed and
tested independently.

Pipeline shape::

    FASTA ──> prepare ──> generation ──> thermo ──> encoding ──> inference ──> report
              (targets)   (candidates)  (features)  (tensors)   (predictions) (top-N)

Primer conventions follow the training code exactly (see
``LAMP_ML-main/lamp-primer-pipeline/src/lamp_pipeline/primer_design/tiling.py``)
and must not be changed without retraining: the model was trained on sequences
laid out in ``PRIMER_TYPE_ORDER`` with the strand assignment in
``FORWARD_STRAND_TYPES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Primer conventions -- FROZEN, these match the trained model
# ---------------------------------------------------------------------------

#: Order the eight primers are laid out along the target, 5'->3'.
#: This is also the order their sequences are joined with '|' for the encoder.
PRIMER_TYPE_ORDER: Tuple[str, ...] = (
    "F3", "F2", "FL", "F1c", "B1c", "BL", "B2", "B3",
)

#: Primer types taken directly off the forward strand. The rest are stored as
#: the reverse complement of their target window.
FORWARD_STRAND_TYPES = frozenset({"F3", "F2", "B1c", "BL"})

#: The four per-primer physics features, in the order the model expects them.
PHYSICS_FEATURES: Tuple[str, ...] = ("mismatches", "gc_fraction", "tm", "dg_ViennaRNA")

#: Global physics features appended after the per-primer block.
GLOBAL_PHYSICS_FEATURES: Tuple[str, ...] = ("total_mismatches", "mean_dg_ViennaRNA")

#: Master mixes the model is conditioned on (3-way one-hot).
MASTER_MIXES: Tuple[str, ...] = ("SS", "BST", "WS")

#: Buffer conditions used for all thermodynamic calculations. These were fixed
#: across the training data, so they are constants rather than parameters.
NA_CONC_MM = 50.0
MG_CONC_MM = 8.0
DNTP_CONC_MM = 1.4
OLIGO_CONC_NM = 100.0
REACTION_TEMP_C = 65.0

#: Pairs encoded per chunk during inference. One encoded row is roughly
#: ``L x 19`` float32 -- about 14 kB at a typical LAMP span -- so a 5 kb
#: stride-1 sweep against 10 targets is ~235k rows, or 3.3 GB per condition if
#: held at once. Chunking keeps peak encoding memory near 270 MB regardless of
#: sweep size, which is what lets a full run fit in a container.
#:
#: It lives in this module, rather than next to its use in ``models/bridge.py``,
#: because the CLI prints it as a ``--pair-chunk`` default and ``types.py`` is
#: import-cheap: reaching into ``models/`` for it would make ``--help`` import
#: torch.
PAIR_CHUNK = 20_000


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One input sequence to design against."""

    name: str
    sequence: str
    #: Fraction of the input panel this target represents (equal weighting by
    #: default; reserved for the Tp-weighted mode in the design notes).
    weight: float = 1.0

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class TargetPanel:
    """The full set of sequences a design run is evaluated against.

    ``consensus`` is the plurality consensus used to *generate* candidates;
    ``targets`` are the sequences candidates are *scored* against. The
    consensus is never scored against itself unless it also appears in
    ``targets``.
    """

    targets: List[Target]
    consensus: str
    #: Present only when the input was a multi-sequence FASTA.
    alignment: Optional[List[str]] = None

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def budget(self) -> int:
        """Input size metric the engine caps on: n_seqs x alignment length.

        Chosen over a bare sequence count because runtime scales with the
        size of the (primer x target) feature matrix, not the number of
        sequences on its own.
        """
        width = len(self.alignment[0]) if self.alignment else len(self.consensus)
        return self.n_targets * width


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Primer:
    """A single primer placed on the generation template."""

    primer_type: str
    sequence: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def is_forward_strand(self) -> bool:
        return self.primer_type in FORWARD_STRAND_TYPES


@dataclass(frozen=True)
class PrimerSet:
    """A complete eight-primer LAMP set.

    ``primers`` is keyed by primer type and always contains every entry in
    ``PRIMER_TYPE_ORDER``. ``geometry_id`` identifies which spacing/length
    geometry produced this set, so results stay reproducible without relying
    on global RNG state.
    """

    set_id: str
    primers: Dict[str, Primer]
    #: The set's position on the template, as the **F3 start** -- the leftmost
    #: base of the leftmost primer. Generation assigns it and the report stage
    #: keys positional diversity on it, so the two agree by this definition
    #: rather than by coincidence. It is a consensus/template coordinate, not a
    #: coordinate in any individual target.
    anchor: int
    geometry_id: str

    def ordered(self) -> List[Primer]:
        return [self.primers[t] for t in PRIMER_TYPE_ORDER]

    def sequences(self) -> List[str]:
        return [p.sequence for p in self.ordered()]

    @property
    def span(self) -> Tuple[int, int]:
        ps = self.ordered()
        return min(p.start for p in ps), max(p.end for p in ps)


# ---------------------------------------------------------------------------
# Thermodynamic features
# ---------------------------------------------------------------------------


@dataclass
class ThermoTable:
    """Per-(primer, target) physics features for a batch of primer sets.

    Stored column-wise so the encoder can slice without per-row Python work.
    ``values`` has shape ``(n_pairs, n_primer_types, len(PHYSICS_FEATURES))``
    and ``globals_`` has shape ``(n_pairs, len(GLOBAL_PHYSICS_FEATURES))``,
    where a "pair" is one (primer set, target) combination indexed by
    ``set_ids`` and ``target_names``.
    """

    set_ids: List[str]
    target_names: List[str]
    values: np.ndarray
    globals_: np.ndarray
    #: Target-window sequence each primer was scored against, same layout as
    #: ``values`` minus the feature axis. Needed by the sequence encoder.
    target_windows: np.ndarray = field(
        default_factory=lambda: np.empty((0, len(PRIMER_TYPE_ORDER)), dtype=object)
    )

    def __post_init__(self) -> None:
        n = len(self.set_ids)
        if len(self.target_names) != n:
            raise ValueError("set_ids and target_names must be the same length")
        expected = (n, len(PRIMER_TYPE_ORDER), len(PHYSICS_FEATURES))
        if self.values.shape != expected:
            raise ValueError(f"values must have shape {expected}, got {self.values.shape}")
        if self.globals_.shape != (n, len(GLOBAL_PHYSICS_FEATURES)):
            raise ValueError(
                f"globals_ must have shape {(n, len(GLOBAL_PHYSICS_FEATURES))}, "
                f"got {self.globals_.shape}"
            )
        # Checked like the others: the encoder indexes target_windows[i, j] in
        # lockstep with values[i, j], so a wrong second axis is a silent
        # mis-encoding rather than an error at the point of the mistake.
        if self.target_windows.shape != (n, len(PRIMER_TYPE_ORDER)):
            raise ValueError(
                f"target_windows must have shape {(n, len(PRIMER_TYPE_ORDER))}, "
                f"got {self.target_windows.shape}"
            )


# ---------------------------------------------------------------------------
# Predictions and output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One reaction context the model is conditioned on."""

    master_mix: str
    #: Template concentration in **copies per reaction**, matching the training
    #: data's ``target_conc`` column. The encoder passes ``log10(conc) - 3``, so
    #: 1,000 copies encodes as 0 -- see ``encoding.encode_target_concentration``.
    target_conc: float

    def __str__(self) -> str:
        return f"{self.master_mix}@{self.target_conc:g}"


@dataclass
class Prediction:
    """Model output for one (primer set, target, condition).

    ``tp_minutes`` is the ensemble mean prediction in minutes, already
    un-standardized with the checkpoint's ``label_mean`` / ``label_std``.
    """

    set_id: str
    target_name: str
    condition: Condition
    tp_minutes: float
    tp_std: float


@dataclass
class ScoredSet:
    """A primer set with its aggregate scores and per-target detail."""

    primer_set: PrimerSet
    #: Mean predicted Tp across targets at the ranking condition.
    speed_score: float
    #: Fraction of targets predicted to amplify within the Tp threshold.
    inclusivity: float
    #: Per-target predicted Tp at the ranking condition.
    per_target_tp: Dict[str, float]
    #: Per-target total primer-target mismatches.
    per_target_mismatches: Dict[str, int]
    #: Full prediction grid, for the detailed report.
    predictions: List[Prediction] = field(default_factory=list)


@dataclass
class DesignResult:
    """Everything a single design run produces."""

    scored_sets: List[ScoredSet]
    panel: TargetPanel
    #: Candidate counts after each pipeline stage, for the funnel report.
    funnel: Dict[str, int] = field(default_factory=dict)
    #: Targets excluded during preparation, with the reason.
    excluded_targets: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: The settings this run used -- stride, geometries, conditions, ranking
    #: condition, device, stage timings. Written by
    #: :func:`lamp_design.commands.design.design` and consumed by the report
    #: writer, so output files can label their own columns from the run that
    #: produced them instead of re-deriving the ranking condition from the
    #: predictions. JSON-serializable by construction.
    params: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PRIMER_TYPE_ORDER",
    "FORWARD_STRAND_TYPES",
    "PHYSICS_FEATURES",
    "GLOBAL_PHYSICS_FEATURES",
    "MASTER_MIXES",
    "NA_CONC_MM",
    "MG_CONC_MM",
    "DNTP_CONC_MM",
    "OLIGO_CONC_NM",
    "REACTION_TEMP_C",
    "Target",
    "TargetPanel",
    "Primer",
    "PrimerSet",
    "ThermoTable",
    "Condition",
    "Prediction",
    "ScoredSet",
    "DesignResult",
]
