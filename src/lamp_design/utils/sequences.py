"""FASTA I/O, reverse complement, plurality consensus, and input QC.

Everything downstream of ``prepare`` assumes unambiguous ACGT. The thermo
stage shells out to ``RNAduplex``, which has no notion of an ambiguity code,
and the encoder one-hot encodes exactly four bases (see
``docs/ARCHITECTURE.md``). Two consequences shape this module:

* :func:`plurality_consensus` never emits ``N`` or an IUPAC code. A consensus
  with ambiguity in it would be unusable as a generation template.
* The validation helpers *reject* degenerate and gapped input rather than
  cleaning it up. A design tool that quietly edits your sequences is worse
  than one that refuses them, so the errors name the sequence, the offending
  character and its position.

Positions in messages are **1-based from the 5' end**, because they are read
by somebody looking at a FASTA file in an editor.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

__all__ = [
    "ACGT",
    "GAP_CHARS",
    "IUPAC_AMBIGUITY",
    "SequenceValidationError",
    "read_fasta",
    "write_fasta",
    "reverse_complement",
    "plurality_consensus",
    "consensus_columns",
    "gap_fractions",
    "strip_gaps",
    "find_gaps",
    "find_degenerate",
    "find_non_acgt",
    "validate_sequence",
    "format_positions",
]


#: The only bases the model and RNAduplex accept. The order doubles as the
#: consensus tie-break priority (see :func:`plurality_consensus`).
ACGT: Tuple[str, ...] = ("A", "C", "G", "T")

#: Characters treated as alignment gaps on input.
GAP_CHARS: Tuple[str, ...] = ("-", ".", "~")

#: IUPAC ambiguity codes mapped to what they stand for, for error messages.
#: ``N`` is included: it is just as unusable as ``R`` downstream.
IUPAC_AMBIGUITY: Dict[str, str] = {
    "R": "A/G",
    "Y": "C/T",
    "S": "G/C",
    "W": "A/T",
    "K": "G/T",
    "M": "A/C",
    "B": "C/G/T",
    "D": "A/G/T",
    "H": "A/C/T",
    "V": "A/C/G",
    "N": "any base",
}

# Reverse complement covers ACGT, the IUPAC codes and gaps so that it is safe
# to call on anything that passed validation. Characters outside the table are
# passed through unchanged by str.translate -- validate first if that matters.
_COMPLEMENT = str.maketrans(
    "ACGTacgtRYSWKMBDHVNryswkmbdhvn-.~",
    "TGCAtgcaYRSWMKVHDBNyrswmkvhdbn-.~",
)

_ACGT_SET = frozenset(ACGT)
_GAP_SET = frozenset(GAP_CHARS)


class SequenceValidationError(ValueError):
    """An input sequence failed QC and the user has to fix it.

    Carries ``name`` so a caller can record the exclusion against the right
    sequence without re-parsing the message.
    """

    def __init__(self, message: str, name: str = "") -> None:
        super().__init__(message)
        self.name = name


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------


def read_fasta(path) -> List[Tuple[str, str]]:
    """Read a FASTA file as ``[(name, sequence)]`` in file order.

    Sequences are upper-cased; nothing else is changed (gaps and ambiguity
    codes survive so that validation can complain about them by position).
    ``name`` is the FASTA id, i.e. the header up to the first whitespace --
    the same convention as the rest of the toolchain.

    Raises:
        FileNotFoundError: the path does not exist.
    """
    with open(path) as handle:
        return [(rec.id, str(rec.seq).upper()) for rec in SeqIO.parse(handle, "fasta")]


def write_fasta(records: Iterable[Tuple[str, str]], path, wrap: bool = True) -> None:
    """Write ``[(name, sequence)]`` to a FASTA file.

    ``wrap=False`` writes two lines per record, which is easier to diff for
    alignments.
    """
    seq_records = [
        SeqRecord(Seq(sequence), id=name, description="") for name, sequence in records
    ]
    with open(path, "w") as handle:
        SeqIO.write(seq_records, handle, "fasta" if wrap else "fasta-2line")


# ---------------------------------------------------------------------------
# Sequence operations
# ---------------------------------------------------------------------------


def reverse_complement(seq: str) -> str:
    """Reverse complement of a DNA sequence (fast translation table).

    Handles ACGT, IUPAC codes and gap characters; anything else is passed
    through unchanged.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def strip_gaps(seq: str) -> str:
    """Drop every gap character, leaving the ungapped sequence."""
    return "".join(c for c in seq if c not in _GAP_SET)


def _column_votes(aligned_seqs: Sequence[str]) -> np.ndarray:
    """Per-column ACGT counts, shape ``(4, width)``.

    Only A, C, G and T vote. Gaps -- and any other character -- are ignored
    rather than counted, which is what makes the consensus unambiguous.
    """
    if not aligned_seqs:
        raise ValueError("need at least one sequence to build a consensus")
    width = len(aligned_seqs[0])
    for i, seq in enumerate(aligned_seqs):
        if len(seq) != width:
            raise ValueError(
                "consensus needs an alignment: row 0 is "
                f"{width} columns but row {i} is {len(seq)}"
            )
    if width == 0:
        return np.zeros((len(ACGT), 0), dtype=np.int64)
    try:
        flat = "".join(aligned_seqs).upper().encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"non-ASCII character in alignment: {exc}") from exc
    arr = np.frombuffer(flat, dtype=np.uint8).reshape(len(aligned_seqs), width)
    return np.stack([(arr == ord(base)).sum(axis=0) for base in ACGT])


def plurality_consensus(aligned_seqs: Sequence[str]) -> str:
    """Plurality consensus of an alignment: the most common base per column.

    Rules, all deliberate:

    * **Only A/C/G/T vote.** Gaps never vote, so a column that is mostly gaps
      still contributes the most common base actually present -- one real base
      among nine gaps yields that base.
    * **Ties break by the fixed priority A > C > G > T** (the order of
      :data:`ACGT`). The alternative -- emitting an ambiguity code -- is not
      available to us: RNAduplex and the encoder accept only four bases, so
      the consensus has to commit. Deterministic and documented beats random:
      the same alignment always gives the same template, and candidates are
      scored against every real target anyway, so a coin-flip column costs at
      most one mismatch in the score, never a wrong answer.
    * **All-gap columns are dropped.** They carry no sequence, so the result
      can be shorter than the alignment is wide (``len(consensus) <= width``).
      :func:`consensus_columns` gives the surviving column indices.
    * The result therefore contains **only** A, C, G and T -- never ``N``,
      never an IUPAC code.

    Raises:
        ValueError: no sequences, or rows of differing length.
    """
    counts = _column_votes(aligned_seqs)
    # argmax returns the FIRST maximal index, which is exactly the documented
    # A > C > G > T tie-break.
    best = counts.argmax(axis=0)
    voted = counts.sum(axis=0) > 0
    return "".join(ACGT[int(b)] for b, keep in zip(best, voted) if keep)


def consensus_columns(aligned_seqs: Sequence[str]) -> List[int]:
    """Alignment column index for each position of the plurality consensus.

    Only differs from ``range(width)`` when the alignment has all-gap columns,
    which :func:`plurality_consensus` drops.
    """
    counts = _column_votes(aligned_seqs)
    return [i for i, total in enumerate(counts.sum(axis=0)) if total > 0]


def gap_fractions(aligned_seqs: Sequence[str]) -> List[float]:
    """Fraction of rows that are *not* a voting base, per alignment column.

    Used to warn about gap-rich columns, where the consensus is built from a
    small minority of the panel.
    """
    counts = _column_votes(aligned_seqs)
    n = len(aligned_seqs)
    return [1.0 - float(total) / n for total in counts.sum(axis=0)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def format_positions(positions: Sequence[int], limit: int = 3) -> str:
    """``"88, 512, 790 (and 38 more)"`` -- 1-based positions for a message."""
    shown = ", ".join(str(p) for p in positions[:limit])
    extra = len(positions) - limit
    return f"{shown} (and {extra} more)" if extra > 0 else shown


def find_gaps(seq: str) -> List[int]:
    """1-based positions of gap characters."""
    return [i for i, c in enumerate(seq, start=1) if c in _GAP_SET]


def find_degenerate(seq: str) -> List[Tuple[int, str]]:
    """1-based ``(position, code)`` for every IUPAC ambiguity code, ``N`` included."""
    upper = seq.upper()
    return [
        (i, c) for i, c in enumerate(upper, start=1) if c in IUPAC_AMBIGUITY
    ]


def find_non_acgt(seq: str, allow_gaps: bool = False) -> List[Tuple[int, str]]:
    """1-based ``(position, character)`` for everything that is not A/C/G/T.

    Includes IUPAC codes. Gap characters are included unless ``allow_gaps``.
    """
    upper = seq.upper()
    out: List[Tuple[int, str]] = []
    for i, c in enumerate(upper, start=1):
        if c in _ACGT_SET:
            continue
        if allow_gaps and c in _GAP_SET:
            continue
        out.append((i, c))
    return out


def validate_sequence(
    name: str,
    seq: str,
    allow_gaps: bool = False,
    min_length: int | None = None,
) -> None:
    """Raise :class:`SequenceValidationError` unless ``seq`` is usable as a target.

    Checks, in the order the user most likely needs to hear about them:
    empty, gapped (unless ``allow_gaps``), degenerate/IUPAC, any other
    non-ACGT character, then length. ``min_length`` is measured on the
    ungapped sequence.
    """
    if not seq:
        raise SequenceValidationError(
            f"sequence '{name}' is empty. Remove the record or give it a sequence.",
            name=name,
        )

    if not allow_gaps:
        gaps = find_gaps(seq)
        if gaps:
            raise SequenceValidationError(
                f"sequence '{name}' contains {len(gaps)} gap character(s), first at "
                f"position {gaps[0]} ('{seq[gaps[0] - 1]}'). Gapped input is rejected: "
                "lamp-design designs against ungapped sequences and builds its own "
                "alignment, so a gapped FASTA is either an alignment or a corrupted "
                "sequence and guessing which would be wrong. Either supply the "
                "sequences without gaps (they will be aligned with MAFFT), or pass "
                "--pre-aligned to declare that this file *is* the alignment.",
                name=name,
            )

    degenerate = find_degenerate(seq)
    if degenerate:
        detail = ", ".join(
            f"'{code}' ({IUPAC_AMBIGUITY[code]}) at position {pos}"
            for pos, code in degenerate[:3]
        )
        extra = len(degenerate) - 3
        if extra > 0:
            detail += f", and {extra} more"
        raise SequenceValidationError(
            f"sequence '{name}' contains {len(degenerate)} degenerate base(s): {detail}. "
            "Ambiguity codes are rejected: RNAduplex and the model encoder accept only "
            "A, C, G and T, and silently substituting a base would change the design "
            "without telling you. Resolve each position to a single base, or drop this "
            "sequence from the input.",
            name=name,
        )

    others = find_non_acgt(seq, allow_gaps=allow_gaps)
    if others:
        positions = [pos for pos, _ in others]
        chars = sorted({char for _, char in others})
        hint = ""
        if "U" in chars:
            hint = " This looks like RNA: convert U to T before designing."
        raise SequenceValidationError(
            f"sequence '{name}' contains {len(others)} character(s) that are not A, C, G "
            f"or T: {', '.join(repr(c) for c in chars)} at position(s) "
            f"{format_positions(positions)}.{hint}",
            name=name,
        )

    if min_length is not None:
        ungapped = len(strip_gaps(seq))
        if ungapped < min_length:
            raise SequenceValidationError(
                f"sequence '{name}' is {ungapped} nt, shorter than the {min_length} nt "
                "minimum. An eight-primer LAMP set spans roughly 170-250 nt of "
                "contiguous template, so a shorter sequence cannot host one. Supply a "
                "longer region, or lower --min-length if you know what you are doing.",
                name=name,
            )
