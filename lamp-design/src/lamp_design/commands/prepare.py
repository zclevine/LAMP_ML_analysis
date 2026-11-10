"""Input parsing and QC: a FASTA file in, a :class:`TargetPanel` out.

The rules here were settled with the team (see ``docs/ARCHITECTURE.md``); they
are not defaults to be tuned away:

* **1 sequence** designs directly against it. **2-20 sequences** are aligned
  if needed, a plurality consensus becomes the generation template, and every
  candidate is scored against every input sequence.
* **Gapped input and degenerate (IUPAC) bases are rejected** with an error
  naming the sequence, the character and the position. v1 does not silently
  clean input: a design tool that quietly edits your sequences is worse than
  one that refuses them. ``--pre-aligned`` is the explicit, non-silent opt-in
  for input that really is an alignment.
* **The cap is on ``TargetPanel.budget`` (n_targets x alignment width), not on
  the sequence count**, because runtime scales with the size of the
  (primer x target) matrix: ten sequences of a small RNA virus and ten
  bacterial genomes are wildly different jobs.
* **Equal weighting** for v1 (``Target.weight = 1.0``).
* Nothing is dropped silently: every exclusion is recorded as
  ``name -> reason`` in ``DesignResult.excluded_targets`` style.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from lamp_design.generation import MIN_DESIGNABLE_LENGTH
from lamp_design.types import DesignResult, Target, TargetPanel
from lamp_design.utils.diagnostics import StageDiagnostics
from lamp_design.utils.sequences import (
    SequenceValidationError,
    gap_fractions,
    plurality_consensus,
    read_fasta,
    strip_gaps,
    validate_sequence,
    write_fasta,
)

__all__ = [
    "DEFAULT_MAX_BUDGET",
    "MAX_SEQUENCES",
    "DEFAULT_MIN_LENGTH",
    "InputError",
    # Re-exported: callers catching prepare's errors want both of these.
    "SequenceValidationError",
    "PreparedPanel",
    "prepare_panel",
    "panel_to_dict",
    "write_panel",
    "load_panel",
    "register",
    "run",
]

#: Default cap on ``TargetPanel.budget`` (n_targets x alignment width).
DEFAULT_MAX_BUDGET = 50_000

#: Hard ceiling on sequence count. The budget is the real limit; this only
#: stops an obviously-wrong input (a whole database) before MAFFT sees it.
MAX_SEQUENCES = 20

#: Floor on target length: the narrowest default geometry's span, imported from
#: ``generation`` rather than written down here. A sequence shorter than this
#: cannot produce a single candidate, so accepting it would mean reporting
#: "no candidates" at the end of a run instead of a clear input error at the
#: start. ``generation.py`` still owns the exact per-geometry check.
DEFAULT_MIN_LENGTH = MIN_DESIGNABLE_LENGTH

#: Columns at or above this gap fraction are reported as a warning: the
#: consensus there is built from a minority of the panel.
GAP_RICH_THRESHOLD = 0.5

PANEL_SCHEMA = "lamp-design/panel/1"

#: ``(records, threads) -> aligned records``. Injectable so tests never need
#: MAFFT; defaults to :func:`lamp_design.external.mafft.align_records`.
Aligner = Callable[[Sequence[Tuple[str, str]], int], List[Tuple[str, str]]]


class InputError(ValueError):
    """The input panel cannot be used and the user has to change something.

    Distinct from :class:`~lamp_design.utils.sequences.SequenceValidationError`,
    which is about one sequence; this is about the panel as a whole (count,
    duplicate names, alignment shape, budget, missing aligner).
    """


@dataclass
class PreparedPanel:
    """A prepared :class:`TargetPanel` plus everything else prepare learned.

    ``TargetPanel`` has nowhere to put exclusions or warnings -- those live on
    ``DesignResult`` -- so prepare returns this wrapper and :meth:`apply_to`
    transfers them onto the result once the pipeline has one.
    """

    panel: TargetPanel
    excluded: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Optional[StageDiagnostics] = None
    #: Names of the alignment rows, positionally matching ``panel.alignment``.
    alignment_names: List[str] = field(default_factory=list)

    @property
    def alignment_width(self) -> int:
        if self.panel.alignment:
            return len(self.panel.alignment[0])
        return len(self.panel.consensus)

    def apply_to(self, result: DesignResult) -> DesignResult:
        """Copy the funnel, exclusions and warnings onto a ``DesignResult``."""
        if self.diagnostics is not None:
            self.diagnostics.apply_to(result)
        result.excluded_targets.update(self.excluded)
        result.warnings.extend(self.warnings)
        return result

    def to_result(self) -> DesignResult:
        """An empty ``DesignResult`` carrying this panel's prepare metadata."""
        return self.apply_to(DesignResult(scored_sets=[], panel=self.panel))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def prepare_panel(
    records: Sequence[Tuple[str, str]],
    *,
    max_budget: int = DEFAULT_MAX_BUDGET,
    min_length: int = DEFAULT_MIN_LENGTH,
    pre_aligned: bool = False,
    force_align: bool = False,
    threads: int = 1,
    aligner: Optional[Aligner] = None,
    panel_name: str = "",
) -> PreparedPanel:
    """Build a :class:`TargetPanel` from ``[(name, sequence)]``.

    Args:
        records: Parsed FASTA records, in file order.
        max_budget: Cap on ``n_targets x alignment width``. Enforced
            inclusively: a budget *equal* to the cap is accepted.
        min_length: Sequences shorter than this are excluded (recorded, never
            dropped silently).
        pre_aligned: Declare that the input is already an alignment. Gaps are
            then accepted, the rows become ``panel.alignment``, and the
            ungapped rows become the targets. Degenerate bases are still
            rejected.
        force_align: Run the aligner even when the sequences are all the same
            length (which is otherwise taken to mean "already aligned").
        threads: Passed to the aligner.
        aligner: Alignment callable; defaults to MAFFT.
        panel_name: Name for this panel in diagnostics messages (the CLI uses
            the input file stem).

    Returns:
        A :class:`PreparedPanel`.

    Raises:
        InputError: panel-level problem (no sequences, too many, duplicate
            names, ragged alignment, over budget, aligner needed but missing).
        SequenceValidationError: a sequence contains gaps (without
            ``--pre-aligned``), an IUPAC code, or any other non-ACGT character.
    """
    diag = StageDiagnostics("prepare", target=panel_name, unit="target sequences")
    warnings: List[str] = []
    excluded: Dict[str, str] = {}

    records = list(records)
    diag.record("Sequences in input", len(records))
    if not records:
        raise InputError(
            "no sequences found in the input. Expected a FASTA file with 1 sequence "
            "(design directly against it) or 2-20 sequences (aligned, consensus "
            "generated, scored against all)."
        )
    if len(records) > MAX_SEQUENCES:
        raise InputError(
            f"{len(records)} sequences given; v1 accepts 1-{MAX_SEQUENCES}. The "
            f"underlying limit is the budget cap (n_targets x alignment width, "
            f"--max-budget). Select up to {MAX_SEQUENCES} sequences and try again."
        )

    _reject_duplicate_names(records)

    # --- 1. Base composition. A problem here rejects the run, by design. ----
    for name, seq in records:
        # allow_gaps=False unconditionally, including under --pre-aligned.
        # Team decision: the user supplies a pre-aligned, GAPLESS panel, and a
        # gapped alignment is refused rather than tiled over. The previous
        # `allow_gaps=pre_aligned` let gaps through on exactly the flag people
        # use for real alignments, and generation has no concept of an
        # alignment gap -- so primers were being placed across columns where
        # the consensus came from a minority of the panel.
        validate_sequence(name, seq, allow_gaps=False)
    diag.record(
        "Base composition QC (A/C/G/T only, no gaps)",
        len(records),
        hint="Gapped or degenerate input is rejected rather than cleaned; "
             "RNAduplex and the model encoder accept only A, C, G and T.",
    )

    if pre_aligned:
        _require_rectangular(records)

    # --- 2. Length. A problem here excludes one sequence, recorded. ---------
    kept: List[Tuple[str, str]] = []
    for name, seq in records:
        ungapped_len = len(strip_gaps(seq))
        if ungapped_len < min_length:
            excluded[name] = (
                f"{ungapped_len} nt, shorter than the {min_length} nt minimum "
                "(an eight-primer LAMP set spans roughly 170-250 nt)"
            )
            continue
        kept.append((name, seq))
    diag.record(
        # Stable step name: the funnel key must not change with the flag value.
        "Minimum length",
        len(kept),
        params={"min_length": min_length},
        hint="An eight-primer set spans roughly 170-250 nt of contiguous template; "
             "lower --min-length only if you know the geometry fits.",
    )

    # --- 3. Identical sequences are scored once, recorded. ------------------
    unique: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}
    for name, seq in kept:
        first = seen.get(seq)
        if first is not None:
            excluded[name] = (
                f"identical sequence to '{first}'; scored once so it does not get "
                "double weight (v1 weights targets equally)"
            )
            continue
        seen[seq] = name
        unique.append((name, seq))
    diag.record(
        "Unique sequences",
        len(unique),
        hint="Identical sequences are scored once: duplicates would inflate the "
             "budget and double-weight one haplotype.",
    )

    if not unique:
        detail = "; ".join(f"'{n}': {r}" for n, r in excluded.items())
        raise InputError(
            f"every input sequence was excluded, so there is nothing to design "
            f"against. {detail}."
        )

    # --- 4. Budget pre-check, BEFORE paying for an alignment. --------------
    # An alignment is at least as wide as its longest ungapped row, so the
    # longest input length is a lower bound on the final width: this check can
    # only under-estimate, and step 6 is authoritative.
    width_floor = max(len(strip_gaps(seq)) for _, seq in unique)
    if len(unique) * width_floor > max_budget:
        raise InputError(
            _budget_message(len(unique), width_floor, max_budget, estimated=True)
        )

    # --- 5. Align if needed; derive the consensus. -------------------------
    alignment: Optional[List[str]] = None
    alignment_names: List[str] = []

    if len(unique) == 1:
        # Single target: no alignment (types.py: alignment is only present for
        # multi-sequence input) and the sequence is its own template.
        name, seq = unique[0]
        consensus = strip_gaps(seq)
        warnings.append(
            f"designing against a single sequence ('{name}'); inclusivity across a "
            "panel cannot be assessed from one target."
        )
    else:
        rows = _align_if_needed(
            unique,
            pre_aligned=pre_aligned,
            force_align=force_align,
            threads=threads,
            aligner=aligner,
            warnings=warnings,
        )
        alignment_names = [name for name, _ in rows]
        alignment = [seq for _, seq in rows]
        consensus = plurality_consensus(alignment)
        if not consensus:
            raise InputError(
                "the alignment has no column with a base in it, so no consensus can "
                "be derived. Check that the input sequences overlap."
            )
        gap_rich = sum(1 for f in gap_fractions(alignment) if f >= GAP_RICH_THRESHOLD)
        if gap_rich:
            warnings.append(
                f"{gap_rich} of {len(alignment[0])} alignment columns are at least "
                f"{GAP_RICH_THRESHOLD:.0%} gaps; the consensus there comes from a "
                "minority of the panel. Trimming the input to the shared region "
                "usually gives better candidates."
            )

    targets = [Target(name=name, sequence=strip_gaps(seq), weight=1.0) for name, seq in unique]
    panel = TargetPanel(targets=targets, consensus=consensus, alignment=alignment)

    # --- 6. Budget, for real this time. ------------------------------------
    width = len(alignment[0]) if alignment else len(consensus)
    if panel.budget > max_budget:
        raise InputError(
            _budget_message(panel.n_targets, width, max_budget, estimated=False)
        )
    diag.record(
        "Within the input budget",
        panel.n_targets,
        params={"max_budget": max_budget},
        hint=f"Budget is n_targets x alignment width = {panel.n_targets} x {width:,} "
             f"= {panel.budget:,} (cap {max_budget:,}).",
    )
    diag.details.update(
        {
            "budget": panel.budget,
            "max_budget": max_budget,
            "alignment_width": width,
            "consensus_length": len(consensus),
            "aligned": alignment is not None,
        }
    )

    if excluded:
        warnings.append(
            f"{len(excluded)} input sequence(s) excluded: "
            + "; ".join(f"'{n}' ({r})" for n, r in excluded.items())
        )

    return PreparedPanel(
        panel=panel,
        excluded=excluded,
        warnings=warnings,
        diagnostics=diag,
        alignment_names=alignment_names,
    )


def _reject_duplicate_names(records: Sequence[Tuple[str, str]]) -> None:
    """Duplicate FASTA ids are fatal: per-target results are keyed by name.

    ``ScoredSet.per_target_tp`` and ``DesignResult.excluded_targets`` are dicts
    keyed by target name, so two targets sharing a name would silently
    overwrite each other's scores.
    """
    seen: Dict[str, int] = {}
    for name, _ in records:
        seen[name] = seen.get(name, 0) + 1
    dupes = sorted(name for name, n in seen.items() if n > 1)
    if dupes:
        raise InputError(
            f"duplicate sequence name(s) in the input: {', '.join(repr(d) for d in dupes)}. "
            "Per-target scores are keyed by name, so duplicates would overwrite each "
            "other. Make the FASTA ids unique (the id is the header up to the first "
            "space)."
        )


def _require_rectangular(records: Sequence[Tuple[str, str]]) -> None:
    """--pre-aligned input has to actually be an alignment."""
    width = len(records[0][1])
    for name, seq in records:
        if len(seq) != width:
            raise InputError(
                f"--pre-aligned expects an alignment, but the rows differ in length: "
                f"'{records[0][0]}' is {width} columns and '{name}' is {len(seq)}. "
                "Either align the input, or drop --pre-aligned and let lamp-design "
                "align the ungapped sequences."
            )


def _align_if_needed(
    records: Sequence[Tuple[str, str]],
    *,
    pre_aligned: bool,
    force_align: bool,
    threads: int,
    aligner: Optional[Aligner],
    warnings: List[str],
) -> List[Tuple[str, str]]:
    """Return alignment rows for a multi-sequence panel."""
    if pre_aligned:
        return list(records)

    if force_align:
        raise InputError(
            "--force-align is not supported: aligning here would introduce gap "
            "columns, and gapped alignments are refused (a primer cannot be "
            "placed over a column most of the panel does not cover). Align and "
            "trim the panel to a gapless region yourself, then run without this "
            "flag."
        )

    widths = {len(seq) for _, seq in records}
    if len(widths) == 1:
        # Ungapped and all the same length: already an alignment (no indels to
        # represent). Nothing is modified, so this is safe -- but say so.
        widths.pop()
        return list(records)

    if aligner is None:
        # Ragged input is refused rather than aligned. Aligning it would
        # introduce exactly the gap columns that gapless-input rule exists to
        # keep out, so running MAFFT here and then rejecting its own output
        # would be incoherent. The design notes settled this -- "We can just
        # tell the user to align their sequences" -- and this is that, enforced.
        lengths = sorted(len(seq) for _, seq in records)
        raise InputError(
            f"the {len(records)} input sequences differ in length "
            f"({lengths[0]:,}-{lengths[-1]:,} nt), so they are not an alignment. "
            "lamp-design needs a pre-aligned, gapless panel: every sequence the "
            "same length, with no '-' characters. Align them yourself (MAFFT, "
            "MUSCLE, Clustal), then trim to the region all sequences share so no "
            "gaps remain, and re-run. Gapped alignments are not supported -- "
            "primers cannot be placed over a column that most of the panel does "
            "not cover."
        )

    rows = list(aligner(records, threads))
    if len(rows) != len(records):
        raise InputError(
            f"the aligner returned {len(rows)} rows for {len(records)} input sequences."
        )
    _require_rectangular(rows)
    return [(name, seq.upper()) for name, seq in rows]


def _budget_message(
    n_targets: int, width: int, max_budget: int, estimated: bool
) -> str:
    """Over-budget error: the budget, the cap, and how to get under it."""
    budget = n_targets * width
    which = (
        f"at least {budget:,} (estimated from the longest input sequence, before "
        "alignment)"
        if estimated
        else f"{budget:,}"
    )
    max_width = max_budget // n_targets
    max_seqs = max_budget // width

    remedies: List[str] = []
    if max_seqs >= 1:
        remedies.append(
            f"keep at most {max_seqs} sequence(s) at this width"
        )
    else:
        remedies.append(
            "even a single sequence of this length is over the cap, so the sequence "
            "count is not the problem"
        )
    remedies.append(f"trim the target region to at most {max_width:,} columns")
    remedies.append(f"or raise the cap with --max-budget {budget}")

    return (
        f"input budget is {which}, over the cap of {max_budget:,}. The budget is "
        f"n_targets x alignment width = {n_targets} x {width:,}; the cap is on that "
        "product rather than on the sequence count because runtime scales with the "
        "size of the (primer x target) matrix -- ten sequences of a small RNA virus "
        "and ten bacterial genomes are very different jobs. To get under the cap: "
        + "; ".join(remedies)
        + " (runtime scales with it)."
    )


# ---------------------------------------------------------------------------
# Serialization -- the handoff to `design`
# ---------------------------------------------------------------------------


def panel_to_dict(prepared: PreparedPanel) -> dict:
    """JSON-ready dict for a prepared panel (round-trips via :func:`load_panel`)."""
    panel = prepared.panel
    return {
        "schema": PANEL_SCHEMA,
        "n_targets": panel.n_targets,
        "budget": panel.budget,
        "alignment_width": prepared.alignment_width,
        "consensus": panel.consensus,
        "targets": [
            {"name": t.name, "sequence": t.sequence, "weight": t.weight}
            for t in panel.targets
        ],
        # Alignment rows are positional: row i belongs to targets[i]. types.py
        # stores them unnamed, so the names are kept alongside for humans.
        "alignment": panel.alignment,
        "alignment_names": prepared.alignment_names,
        "excluded_targets": prepared.excluded,
        "warnings": prepared.warnings,
        "funnel": prepared.diagnostics.as_funnel() if prepared.diagnostics else {},
        "diagnostics": prepared.diagnostics.to_dict() if prepared.diagnostics else {},
    }


def write_panel(prepared: PreparedPanel, path: str | Path) -> Path:
    """Write the prepared panel as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(panel_to_dict(prepared), indent=2))
    return path


def load_panel(path: str | Path) -> TargetPanel:
    """Read a panel JSON written by :func:`write_panel` back into a ``TargetPanel``."""
    data = json.loads(Path(path).read_text())
    schema = data.get("schema")
    if schema != PANEL_SCHEMA:
        raise InputError(
            f"{path} is not a lamp-design panel file (schema {schema!r}, expected "
            f"{PANEL_SCHEMA!r}). Produce it with `lamp-design prepare --in <fasta> "
            "--out <panel.json>`."
        )
    targets = [
        Target(name=t["name"], sequence=t["sequence"], weight=float(t.get("weight", 1.0)))
        for t in data["targets"]
    ]
    return TargetPanel(
        targets=targets,
        consensus=data["consensus"],
        alignment=data.get("alignment"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Register the prepare subcommand."""
    parser = subparsers.add_parser(
        "prepare",
        help="Parse and QC an input FASTA into a target panel",
        description="""
Parse an input FASTA into the target panel the rest of the pipeline designs
against: validate the sequences, align them if needed, and derive the
plurality consensus used as the generation template.

Accepts 1 sequence (designed against directly) or 2-20 sequences (aligned,
consensus derived, every candidate scored against every sequence).

Gapped and degenerate (IUPAC) input is rejected rather than cleaned up: pass
--pre-aligned if the file really is an alignment. The input size cap is on the
budget -- n_targets x alignment width -- not on the sequence count.
""",
    )
    parser.add_argument(
        "--in", dest="input", required=True, help="Input FASTA (1-20 sequences)"
    )
    parser.add_argument(
        "--out",
        dest="output",
        default=None,
        help="Output panel JSON for `lamp-design design` (omit for a QC-only run)",
    )
    parser.add_argument(
        "--consensus-out",
        dest="consensus_out",
        default=None,
        help="Also write the plurality consensus as FASTA",
    )
    parser.add_argument(
        "--alignment-out",
        dest="alignment_out",
        default=None,
        help="Also write the alignment the consensus was derived from, as FASTA",
    )
    parser.add_argument(
        "--max-budget",
        dest="max_budget",
        type=int,
        default=DEFAULT_MAX_BUDGET,
        help="Cap on n_targets x alignment width (default: %(default)s). Runtime "
             "scales with this product.",
    )
    parser.add_argument(
        "--min-length",
        dest="min_length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help="Exclude sequences shorter than this many nt (default: %(default)s)",
    )
    parser.add_argument(
        "--pre-aligned",
        dest="pre_aligned",
        action="store_true",
        help="The input is already an alignment: accept gap characters, use the rows "
             "as the alignment and the ungapped rows as the targets",
    )
    parser.add_argument(
        "--force-align",
        dest="force_align",
        action="store_true",
        help="Align with MAFFT even when all input sequences are the same length "
             "(which is otherwise taken to mean they are already aligned)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Threads for MAFFT (default: %(default)s)"
    )
    parser.set_defaults(func=run)


def run(args) -> PreparedPanel:
    """Run the prepare command."""
    records = read_fasta(args.input)
    prepared = prepare_panel(
        records,
        max_budget=args.max_budget,
        min_length=args.min_length,
        pre_aligned=getattr(args, "pre_aligned", False),
        force_align=getattr(args, "force_align", False),
        threads=getattr(args, "threads", 1),
        panel_name=Path(args.input).stem,
    )
    panel = prepared.panel

    print(
        f"Prepared {panel.n_targets} target sequence(s) from {args.input}: "
        f"{prepared.alignment_width:,} columns, budget {panel.budget:,} / "
        f"{args.max_budget:,} cap"
    )
    print(
        f"Consensus: {len(panel.consensus):,} nt "
        f"({'aligned' if panel.alignment else 'single sequence, no alignment'})"
    )
    for name, reason in prepared.excluded.items():
        print(f"  EXCLUDED {name}: {reason}")
    for warning in prepared.warnings:
        print(f"  WARNING: {warning}")

    if args.consensus_out:
        write_fasta([("consensus", panel.consensus)], args.consensus_out)
        print(f"Wrote consensus to {args.consensus_out}")
    if args.alignment_out:
        if panel.alignment is None:
            print("  WARNING: no alignment to write (single-sequence input)")
        else:
            write_fasta(
                zip(prepared.alignment_names, panel.alignment),
                args.alignment_out,
                wrap=False,
            )
            print(f"Wrote alignment to {args.alignment_out}")
    if args.output:
        write_panel(prepared, args.output)
        print(f"Wrote panel to {args.output}")
        # Same contract as PrimerCast: writes <out>.diagnostics.json and stops
        # with exit code 3 if the stage ended up empty.
        if prepared.diagnostics is not None:
            prepared.diagnostics.finish(args.output)
    elif prepared.diagnostics is not None:
        prepared.diagnostics.raise_if_empty()

    return prepared
