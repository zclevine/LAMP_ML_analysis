"""Input gates that run before any work starts.

A *gate*, not a replacement for ``commands/prepare.py``'s QC: it checks size,
shape and alphabet so a malformed or oversized upload is rejected without
starting a design, and the rules it enforces are the ones prepare will enforce
anyway -- notably that only A/C/G/T are accepted, with no gaps and no IUPAC
ambiguity codes, because RNAduplex and the model encoder take nothing else.

Deliberately dependency-free (no streamlit, no pandas, no engine import beyond
the bounds in ``_engine_api``), so it is cheap to call on every rerun and
testable on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from _engine_api import MAX_BUDGET, MAX_SEQUENCES, MIN_TARGET_LENGTH, PRIMER_TYPE_ORDER

#: Upload cap. NOTE: Streamlit buffers an entire upload in memory *before* this
#: app sees a single byte, so ``--server.maxUploadSize`` (MB) is the control
#: that protects memory and this check is the control that protects work. Set
#: both; the server default is 200 MB, far too generous for FASTA.
MAX_UPLOAD_MB = float(os.environ.get("LAMP_DESIGN_MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

ACGT = frozenset("ACGT")
GAP_CHARS = frozenset("-.")

#: Primer length window used only for a warning. The deterministic geometries
#: lay down 16-25 nt primers, so anything far outside that is being scored by a
#: model that never saw its like, which is worth saying out loud.
TYPICAL_PRIMER_LENGTHS = (15, 30)

#: Cap on sets submitted to Evaluate. Scoring is (sets x targets x conditions)
#: predictions, the same arithmetic as a design run, so an unbounded paste is
#: an unbounded run.
MAX_PRIMER_SETS = int(os.environ.get("LAMP_DESIGN_MAX_PRIMER_SETS", "200"))


# ---------------------------------------------------------------------------
# Target FASTA
# ---------------------------------------------------------------------------


@dataclass
class FastaCheck:
    """Outcome of validating target FASTA text."""

    records: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.records)

    @property
    def n_seqs(self) -> int:
        return len(self.records)

    @property
    def max_len(self) -> int:
        return max((len(s) for _, s in self.records), default=0)

    @property
    def min_len(self) -> int:
        return min((len(s) for _, s in self.records), default=0)

    @property
    def total_bases(self) -> int:
        return sum(len(s) for _, s in self.records)

    @property
    def budget(self) -> int:
        """Lower bound on TargetPanel.budget: alignment gaps only widen it."""
        return self.n_seqs * self.max_len

    @property
    def ragged(self) -> bool:
        """Whether a multi-sequence panel would need an aligner.

        prepare treats equal-length ungapped rows as the alignment; ragged
        input is refused outright rather than aligned, because aligning would
        introduce the gap columns the engine does not support.
        """
        return self.n_seqs > 1 and self.min_len != self.max_len

    def gc_fraction(self, sequence: str) -> float:
        return (sequence.count("G") + sequence.count("C")) / len(sequence) if sequence else 0.0


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Minimal strict FASTA parse. Returns ``[(name, uppercased sequence)]``.

    The name is the header up to the first whitespace, matching
    ``utils.sequences.read_fasta``, which is what actually parses the file later.
    """
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            header = line[1:].strip()
            name = header.split()[0] if header else ""
            chunks = []
        else:
            chunks.append(line.upper().replace(" ", ""))
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def validate_fasta(text: str, *, pre_aligned: bool = False, source: str = "input") -> FastaCheck:
    """Reject malformed, oversized or unusable target input before any work."""
    check = FastaCheck()
    # Gaps are never allowed, whatever `pre_aligned` says. The parameter used
    # to widen this to ACGT | GAP_CHARS, which let the UI accept input the
    # engine then refused -- the validator's job is to predict the engine, not
    # to be more permissive than it.
    allowed = ACGT

    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        check.errors.append("No sequence provided.")
        return check
    if n_bytes > MAX_UPLOAD_BYTES:
        check.errors.append(
            f"{source} is {n_bytes / 1e6:.1f} MB, over the {MAX_UPLOAD_MB:g} MB limit."
        )
        return check
    if not text.lstrip().startswith(">"):
        check.errors.append(
            "Not FASTA: the first non-blank line must start with '>'. "
            "Paste a bare sequence into the text box instead and it will be wrapped for you."
        )
        return check

    records = parse_fasta(text)
    if not records:
        check.errors.append("No sequences found.")
        return check
    if len(records) > MAX_SEQUENCES:
        check.errors.append(
            f"{len(records)} sequences given; the engine accepts 1-{MAX_SEQUENCES}. "
            f"Select up to {MAX_SEQUENCES} sequences and try again."
        )
        return check

    seen: dict[str, int] = {}
    for index, (name, seq) in enumerate(records, start=1):
        label = name or f"record {index}"
        if not name:
            check.errors.append(f"Record {index} has an empty header.")
            continue
        if not seq:
            check.errors.append(f"'{label}' has a header but no sequence.")
            continue
        seen[name] = seen.get(name, 0) + 1

        bad = sorted(set(seq) - allowed)
        if bad:
            shown = " ".join(repr(c) for c in bad[:6]) + (" ..." if len(bad) > 6 else "")
            if set(bad) & GAP_CHARS:
                check.errors.append(
                    f"'{label}' contains alignment gaps ({shown}). Gapped "
                    "alignments are not supported: trim the panel to a "
                    "contiguous block of columns where every sequence has a "
                    "base, and upload that."
                )
            else:
                check.errors.append(
                    f"'{label}' contains characters other than A, C, G and T: {shown}. "
                    "Ambiguity codes (N, R, Y, ...) are rejected rather than cleaned -- "
                    "RNAduplex and the model encoder accept only A, C, G and T."
                )
            continue

        if len(seq) < MIN_TARGET_LENGTH:
            check.warnings.append(
                f"'{label}' is {len(seq)} nt, below the engine's {MIN_TARGET_LENGTH} nt floor. "
                "It will be excluded from the panel and reported as such."
            )
        check.records.append((name, seq))

    duplicates = sorted(n for n, count in seen.items() if count > 1)
    if duplicates:
        # Per-target results are keyed by name, so duplicates would collide in
        # the Tp matrix rather than merely look odd. prepare rejects them too.
        check.errors.append(
            "Duplicate sequence names: " + ", ".join(duplicates[:5]) + ". Names must be unique."
        )

    usable = [s for _, s in check.records if len(s) >= MIN_TARGET_LENGTH]
    if check.records and not usable:
        check.errors.append(
            f"Every sequence is shorter than {MIN_TARGET_LENGTH} nt, so no primer set can "
            "be placed. Provide at least one longer target."
        )
    if check.budget > MAX_BUDGET:
        check.errors.append(
            f"Input is too large: {check.n_seqs} sequences x {check.max_len} nt = "
            f"{check.budget:,}, over the {MAX_BUDGET:,} budget limit. Reduce the number "
            "of sequences or trim them to the region of interest."
        )
    return check


# ---------------------------------------------------------------------------
# Primer sets, for the Evaluate workflow
# ---------------------------------------------------------------------------


#: Column aliases ``commands.evaluate.load_rows`` accepts, repeated here so the
#: page can say what it found before the engine reads the file. Lower-cased,
#: first match wins, exactly as there.
SET_NAME_COLUMNS = ("primer_set", "set_id", "set", "name", "primer_set_id")
TARGET_NAME_COLUMNS = ("target_seq", "target", "target_name", "target_id")
MASTER_MIX_COLUMNS = ("master_mix", "mastermix", "mix", "enzyme")
CONC_COLUMNS = ("target_conc", "conc", "concentration", "copies")
TRUTH_COLUMNS = ("tp", "tp_minutes", "measured_tp", "observed_tp")


@dataclass
class PrimerSetCheck:
    """Outcome of validating user-supplied primer sets."""

    sets: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Which shape the text was read as: ``fasta`` or ``table``.
    source_format: str = ""
    #: Optional columns a table carried, by role, so the page can report that a
    #: measured-Tp column turns the run into a validation.
    optional_columns: dict[str, str] = field(default_factory=dict)
    #: Whether the table already carries ``<type>_target_sequence`` columns, in
    #: which case the engine does not have to locate the windows.
    has_window_columns: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.sets)

    @property
    def n_sets(self) -> int:
        return len(self.sets)


def _split_table_row(line: str) -> list[str]:
    """Split a table row on tabs, or on commas when there is no tab."""
    return [cell.strip() for cell in (line.split("\t") if "\t" in line else line.split(","))]


def _check_one_set(
    label: str, primers: dict[str, str], check: PrimerSetCheck
) -> dict[str, str] | None:
    """Validate one set's eight primers; returns the normalised set or None."""
    missing = [t for t in PRIMER_TYPE_ORDER if not primers.get(t)]
    if missing:
        check.errors.append(
            f"'{label}' is missing {len(missing)} of the eight primers: "
            f"{', '.join(missing)}. A LAMP set needs all of "
            f"{', '.join(PRIMER_TYPE_ORDER)}."
        )
        return None

    normalised: dict[str, str] = {}
    for primer_type in PRIMER_TYPE_ORDER:
        sequence = primers[primer_type].upper().replace(" ", "")
        bad = sorted(set(sequence) - ACGT)
        if bad:
            shown = " ".join(repr(c) for c in bad[:6])
            check.errors.append(
                f"'{label}' {primer_type} contains characters other than A, C, G and T: "
                f"{shown}. Degenerate primers are rejected rather than expanded -- the "
                "model encoder accepts only A, C, G and T."
            )
            return None
        low, high = TYPICAL_PRIMER_LENGTHS
        if not low <= len(sequence) <= high:
            check.warnings.append(
                f"'{label}' {primer_type} is {len(sequence)} nt, outside the "
                f"{low}-{high} nt range the trained geometries cover. It will be "
                "scored, but that is extrapolation."
            )
        normalised[primer_type] = sequence
    return normalised


def validate_primer_sets(text: str, *, source: str = "input") -> PrimerSetCheck:
    """Parse and check primer sets given as a table or as FASTA.

    A table is what ``lamp-design evaluate --in`` actually reads: one column per
    primer, named ``F3`` or ``F3_sequence``, optionally with
    ``<type>_target_sequence`` windows and the ``primer_set`` / ``target_seq`` /
    ``master_mix`` / ``target_conc`` / ``tp`` columns that module recognises. A
    table is passed through to the engine **verbatim**, so those optional
    columns keep working -- in particular ``tp``, which turns the run into a
    validation with Spearman, MAE and ROC-AUC against measured values.

    FASTA with ``<set_id>_<primer_type>`` headers is also accepted, because that
    is what someone with a handful of primers has on hand; it is converted to
    the table shape before the engine sees it.
    """
    check = PrimerSetCheck()
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        check.errors.append("No primer sets provided.")
        return check
    if n_bytes > MAX_UPLOAD_BYTES:
        check.errors.append(
            f"{source} is {n_bytes / 1e6:.1f} MB, over the {MAX_UPLOAD_MB:g} MB limit."
        )
        return check

    if text.lstrip().startswith(">"):
        check.source_format = "fasta"
        _read_primer_fasta(text, check)
    else:
        check.source_format = "table"
        _read_primer_table(text, check)

    if check.n_sets > MAX_PRIMER_SETS:
        check.errors.append(
            f"{check.n_sets} sets given; this page accepts up to {MAX_PRIMER_SETS}. "
            "Scoring cost is sets x targets x conditions, the same arithmetic as a "
            "design run."
        )
    return check


def _read_primer_fasta(text: str, check: PrimerSetCheck) -> None:
    """``>set_id_F3`` style FASTA -> grouped sets, in first-seen order."""
    by_type = {t.upper(): t for t in PRIMER_TYPE_ORDER}
    grouped: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for index, (header, sequence) in enumerate(parse_fasta(text), start=1):
        if not header:
            check.errors.append(f"Record {index} has an empty header.")
            continue
        set_id, _, suffix = header.rpartition("_")
        primer_type = by_type.get(suffix.upper())
        if primer_type is None or not set_id:
            check.errors.append(
                f"Header '{header}' does not end in a primer type. Name each record "
                f"'<set_name>_<type>' with type one of {', '.join(PRIMER_TYPE_ORDER)}."
            )
            continue
        if primer_type in grouped.get(set_id, {}):
            check.errors.append(f"'{set_id}' has two {primer_type} records.")
            continue
        if set_id not in grouped:
            grouped[set_id] = {}
            order.append(set_id)
        grouped[set_id][primer_type] = sequence

    if not grouped and not check.errors:
        check.errors.append("No primer records found.")
    for set_id in order:
        normalised = _check_one_set(set_id, grouped[set_id], check)
        if normalised is not None:
            check.sets.append((set_id, normalised))


def _find_column(columns: dict[str, int], aliases: tuple[str, ...]) -> str | None:
    """First alias the header actually has, mirroring ``evaluate._pick``."""
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def _read_primer_table(text: str, check: PrimerSetCheck) -> None:
    """CSV/TSV with one column per primer, as ``evaluate.load_rows`` reads it."""
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        check.errors.append(
            "A primer table needs a header row and at least one data row. Expected "
            f"one column per primer named 'F3' or 'F3_sequence', for all of "
            f"{', '.join(PRIMER_TYPE_ORDER)}."
        )
        return

    header = _split_table_row(rows[0])
    columns = {cell.strip().lower(): position for position, cell in enumerate(header)}
    type_columns: dict[str, int | None] = {}
    for primer_type in PRIMER_TYPE_ORDER:
        lowered = primer_type.lower()
        type_columns[primer_type] = columns.get(f"{lowered}_sequence", columns.get(lowered))
    absent = [t for t, position in type_columns.items() if position is None]
    if absent:
        check.errors.append(
            f"The header is missing a sequence column for {', '.join(absent)}. "
            "Expected one column per primer named e.g. 'F3_sequence' or 'F3', for all "
            f"of {', '.join(PRIMER_TYPE_ORDER)}. Found: {', '.join(header[:12])}"
            + (" ..." if len(header) > 12 else "")
        )
        return

    for role, aliases in (
        ("set name", SET_NAME_COLUMNS),
        ("target name", TARGET_NAME_COLUMNS),
        ("master mix", MASTER_MIX_COLUMNS),
        ("concentration", CONC_COLUMNS),
        ("measured Tp", TRUTH_COLUMNS),
    ):
        found = _find_column(columns, aliases)
        if found:
            check.optional_columns[role] = found
    check.has_window_columns = all(
        f"{t.lower()}_target_sequence" in columns for t in PRIMER_TYPE_ORDER
    )

    name_column = columns.get(check.optional_columns.get("set name", ""))
    for index, line in enumerate(rows[1:], start=1):
        cells = _split_table_row(line)
        label = (
            cells[name_column].strip()
            if name_column is not None and name_column < len(cells) and cells[name_column].strip()
            else f"row_{index}"
        )
        primers = {
            primer_type: (cells[position] if position is not None and position < len(cells) else "")
            for primer_type, position in type_columns.items()
        }
        normalised = _check_one_set(label, primers, check)
        if normalised is not None:
            check.sets.append((label, normalised))


def primer_sets_to_csv(sets: list[tuple[str, dict[str, str]]]) -> str:
    """Render parsed sets as the CSV ``lamp-design evaluate --in`` reads.

    Used when the sets came in as FASTA, so the engine is always handed its own
    documented input shape rather than a second format it would have to learn.
    The target windows are deliberately not included: with none present the
    engine locates them in ``--targets`` by best Hamming match, which is what
    the Target step supplied.
    """
    header = ["primer_set", *PRIMER_TYPE_ORDER]
    lines = [",".join(header)]
    for set_id, primers in sets:
        lines.append(",".join([set_id, *(primers[t] for t in PRIMER_TYPE_ORDER)]))
    return "\n".join(lines) + "\n"
