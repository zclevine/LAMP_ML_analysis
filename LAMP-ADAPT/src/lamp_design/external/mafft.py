"""MAFFT wrapper -- optional, used only to align unaligned multi-FASTA input.

MAFFT is assumed to be on PATH (conda / the Docker image). It is *optional*:
a single-sequence run never needs it, and neither does input that is already
an alignment. When it is needed and missing, the caller fails with an
instruction to pre-align rather than guessing at an alignment.

Two details that bite if you call ``mafft`` naively, both handled here:

* MAFFT lower-cases its output, so sequences come back upper-cased.
* MAFFT writes the input names back out, but FASTA ids are only the header up
  to the first whitespace, so two records whose names differ after a space
  would become indistinguishable. :func:`align_records` therefore aligns under
  generated ``s0``, ``s1`` ... ids and maps the rows back to the caller's
  names, in the caller's input order (MAFFT preserves input order unless
  ``--reorder`` is passed, but mapping by id does not rely on that).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

from lamp_design.utils.sequences import read_fasta, write_fasta

__all__ = ["find_mafft", "mafft_available", "align_sequences", "align_records"]

MAFFT_INSTALL_HINT = "conda install -c bioconda mafft"


def find_mafft() -> str:
    """Path to the ``mafft`` executable.

    Raises:
        FileNotFoundError: mafft is not on PATH.
    """
    path = shutil.which("mafft")
    if path is None:
        raise FileNotFoundError(
            f"mafft not found in PATH. Install: {MAFFT_INSTALL_HINT}"
        )
    return path


def mafft_available() -> bool:
    """True if ``mafft`` is on PATH."""
    return shutil.which("mafft") is not None


def align_sequences(
    input_fasta: str | Path,
    output_fasta: str | Path,
    threads: int = 1,
    auto: bool = True,
    quiet: bool = True,
) -> None:
    """Align a FASTA file with MAFFT, writing aligned FASTA to ``output_fasta``.

    Raises:
        FileNotFoundError: mafft is missing, or the input file does not exist.
        ValueError: the input file is empty or is not FASTA.
        subprocess.CalledProcessError: mafft failed.
    """
    find_mafft()  # Verify it exists before touching the filesystem.

    input_fasta = Path(input_fasta)
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {input_fasta}")
    if input_fasta.stat().st_size == 0:
        raise ValueError(f"Input FASTA file is empty: {input_fasta}")
    if not input_fasta.read_text().strip().startswith(">"):
        raise ValueError(
            f"Input file does not appear to be in FASTA format: {input_fasta}"
        )

    cmd = ["mafft", "--thread", str(threads)]
    if auto:
        cmd.append("--auto")
    if quiet:
        cmd.append("--quiet")
    cmd.append(str(input_fasta))

    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", dir=output_fasta.parent, delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            subprocess.run(cmd, stdout=tmp_file, stderr=subprocess.PIPE, check=True)
            tmp_file.flush()
            shutil.move(str(tmp_path), str(output_fasta))
        except subprocess.CalledProcessError as exc:
            tmp_path.unlink(missing_ok=True)
            if exc.stderr:
                raise subprocess.CalledProcessError(
                    exc.returncode, cmd, stderr=exc.stderr
                ) from exc
            raise


def align_records(
    records: Sequence[Tuple[str, str]], threads: int = 1
) -> List[Tuple[str, str]]:
    """Align ``[(name, sequence)]`` in memory, preserving names and input order.

    Returns the aligned rows, upper-cased and all the same length. This is the
    entry point ``commands/prepare.py`` uses, and the seam its tests replace.

    Raises:
        FileNotFoundError: mafft is not on PATH.
        ValueError: fewer than two records, or MAFFT dropped/garbled a row.
        subprocess.CalledProcessError: mafft failed.
    """
    if len(records) < 2:
        raise ValueError("align_records() needs at least two sequences")
    find_mafft()

    with tempfile.TemporaryDirectory(prefix="lamp-design-mafft-") as tmpdir:
        tmp = Path(tmpdir)
        unaligned, aligned = tmp / "input.fasta", tmp / "aligned.fasta"
        # Align under generated ids so names with spaces, duplicate prefixes or
        # characters MAFFT dislikes cannot confuse the mapping back.
        write_fasta(((f"s{i}", seq) for i, (_, seq) in enumerate(records)), unaligned)
        align_sequences(unaligned, aligned, threads=threads)

        rows = dict(read_fasta(aligned))

    out: List[Tuple[str, str]] = []
    for i, (name, _) in enumerate(records):
        row = rows.get(f"s{i}")
        if row is None:
            raise ValueError(
                f"MAFFT did not return a row for sequence '{name}'; "
                f"got {len(rows)} of {len(records)} rows back"
            )
        out.append((name, row.upper()))

    width = len(out[0][1])
    if any(len(seq) != width for _, seq in out):
        raise ValueError("MAFFT returned rows of differing length")
    return out
