#!/usr/bin/env python3

import argparse
import sys
from math import nan

import pandas as pd
from skbio import DNA, TabularMSA
from Bio.Seq import Seq


# -----------------------------
# Primer orientation definitions
# -----------------------------

F_TYPES = {'F1c', 'F2', 'F3', 'LF', 'FL'}
R_TYPES = {'B1c', 'B2', 'B3', 'LB', 'BL'}
ALLOWED_TYPES = F_TYPES | R_TYPES


# -----------------------------
# Core helper functions
# -----------------------------

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def find_primer_with_mismatches(consensus_seq,
                                primer_seq,
                                max_mismatches=2,
                                expected_orientation=None):
    """
    Find primer matches on a consensus sequence with up to max_mismatches.

    Parameters
    ----------
    consensus_seq : str
        DNA sequence to search in.
    primer_seq : str
        Primer sequence (5'->3').
    max_mismatches : int
        Maximum allowed mismatches.
    expected_orientation : {"F", "RC", None}
        - "F"  : only report forward-orientation hits.
        - "RC" : only report reverse-complement hits.
        - None : consider both orientations, but only return
                 hit(s) with the fewest mismatches overall.

    Returns
    -------
    list of tuples
        (start, end, orientation, mismatches)
        start is 0-based, end is exclusive.
    """
    matches = []
    seq = consensus_seq.upper()
    primer = primer_seq.upper()
    k = len(primer)

    def scan(primer_string, orientation):
        for i in range(0, len(seq) - k + 1):
            window = seq[i:i + k]
            mismatches = hamming_distance(window, primer_string)
            if mismatches <= max_mismatches:
                matches.append((i, i + k, orientation, mismatches))

    # Decide which orientations to scan based on expected_orientation
    if expected_orientation in (None, "F"):
        scan(primer, "F")

    if expected_orientation in (None, "RC"):
        rc_primer = str(Seq(primer).reverse_complement())
        scan(rc_primer, "RC")

    if not matches:
        return []

    # If user specified an orientation, just return all matches in that direction
    if expected_orientation in ("F", "RC"):
        return matches

    # expected_orientation is None:
    # return only the hit(s) with the fewest mismatches across both F and RC
    min_mismatches = min(m[3] for m in matches)
    best_matches = [m for m in matches if m[3] == min_mismatches]
    return best_matches


def type_to_orientation(primer_type):
    if primer_type in F_TYPES:
        return "F"
    if primer_type in R_TYPES:
        return "RC"
    return None


def find_best_hit_row(row,
                      consensus_seq,
                      max_mismatches=2,
                      use_primer_type_orientation=True):
    """
    Find best hit for a single primer row.

    Parameters
    ----------
    row : pandas.Series
        Must have 'sequence' and 'primer_type'.
    consensus_seq : str
        Consensus sequence to search in.
    max_mismatches : int
        Maximum allowed mismatches.
    use_primer_type_orientation : bool
        If True, use primer_type to set expected_orientation (F or RC).
        If False, expected_orientation=None (search both F and RC and
        choose the best match).
    """
    primer_seq = row['sequence']
    primer_type = row['primer_type']

    if use_primer_type_orientation:
        expected_orientation = type_to_orientation(primer_type)
    else:
        expected_orientation = None

    hits = find_primer_with_mismatches(
        consensus_seq=consensus_seq,
        primer_seq=primer_seq,
        max_mismatches=max_mismatches,
        expected_orientation=expected_orientation
    )

    # No hit found: return NaNs
    if not hits:
        return pd.Series({
            'start': nan,
            'end': nan,
            'orientation': None,
            'mismatches': nan
        })

    # Take the first best hit (if there are ties, first in sequence)
    start, end, orientation, mismatches = hits[0]

    return pd.Series({
        'start': start,       # 0-based
        'end': end,           # end is exclusive
        'orientation': orientation,
        'mismatches': mismatches
    })


def validate_primer_df(df):
    required_cols = {"primer_name", "sequence", "primer_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Primer DataFrame is missing required column(s): {', '.join(sorted(missing))}"
        )

    unknown_types = set(df['primer_type']) - ALLOWED_TYPES
    if unknown_types:
        raise ValueError(
            "Primer DataFrame has unknown primer_type values. "
            f"Expected only {sorted(ALLOWED_TYPES)}, but found: {sorted(unknown_types)}"
        )


# -----------------------------
# High-level function (import-friendly)
# -----------------------------

def run_primer_search(fasta_path,
                      primer_csv_path,
                      max_mismatches=2,
                      use_primer_type_orientation=True):
    """
    Run primer search given an alignment FASTA and a primer CSV.

    Parameters
    ----------
    fasta_path : str
        Path to FASTA alignment file.
    primer_csv_path : str
        Path to primer CSV file with columns:
        primer_name, sequence, primer_type.
    max_mismatches : int, default 2
        Maximum allowed mismatches.
    use_primer_type_orientation : bool, default True
        - True  : use primer_type to set expected_orientation (F or RC).
        - False : set expected_orientation=None, search both F and RC and
                  keep best match.

    Returns
    -------
    pandas.DataFrame
        Input primer table with extra columns:
        start, end, orientation, mismatches.
    """
    # Read alignment and build consensus
    msa = TabularMSA.read(fasta_path, constructor=DNA)

    # For newer skbio:
    try:
        cons = msa.consensus_sequence()
    except AttributeError:
        # for older skbio that used .consensus()
        cons = msa.consensus()

    consensus_seq = str(cons)

    # Read primers
    primers = pd.read_csv(primer_csv_path)

    # Validate primer file
    validate_primer_df(primers)

    # Find best hit for each primer
    hits_df = primers.apply(
        find_best_hit_row,
        axis=1,
        consensus_seq=consensus_seq,
        max_mismatches=max_mismatches,
        use_primer_type_orientation=use_primer_type_orientation
    )

    out_df = pd.concat([primers, hits_df], axis=1)
    return out_df


# -----------------------------
# CLI wrapper
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find best primer matches on a consensus sequence from an alignment."
    )
    parser.add_argument(
        "fasta",
        help="Input FASTA alignment file."
    )
    parser.add_argument(
        "primers",
        help="Input primer CSV file with columns: primer_name, sequence, primer_type."
    )
    parser.add_argument(
        "--max-mismatches", "-m",
        type=int,
        default=2,
        help="Maximum allowed mismatches (default: 2)."
    )
    # Mutually exclusive flags to toggle orientation behavior
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-primer-type-orientation",
        dest="use_primer_type_orientation",
        action="store_true",
        help="Use primer_type to set expected_orientation (default)."
    )
    group.add_argument(
        "--no-primer-type-orientation",
        dest="use_primer_type_orientation",
        action="store_false",
        help="Ignore primer_type; search both F and RC and choose best match."
    )
    parser.set_defaults(use_primer_type_orientation=True)

    parser.add_argument(
        "--output", "-o",
        help="Output CSV file (default: print to stdout)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        out_df = run_primer_search(
            fasta_path=args.fasta,
            primer_csv_path=args.primers,
            max_mismatches=args.max_mismatches,
            use_primer_type_orientation=args.use_primer_type_orientation
        )
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)

    if args.output:
        try:
            out_df.to_csv(args.output, index=False)
        except Exception as e:
            sys.stderr.write(f"ERROR: Failed to write output CSV '{args.output}': {e}\n")
            sys.exit(1)
    else:
        out_df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()

from skbio import DNA, TabularMSA
import pandas as pd
import numpy as np

def extract_target_regions_from_alignment(
    fasta_path: str,
    primer_hits_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Given an alignment FASTA and the primer hits DataFrame from run_primer_search,
    extract the target region from each sequence in the alignment for each primer.

    Parameters
    ----------
    fasta_path : str
        Path to FASTA alignment file.
    primer_hits_df : pandas.DataFrame
        Output from run_primer_search, must contain:
        - primer_name
        - sequence        (primer sequence)
        - start, end      (0-based, end exclusive)
        - orientation     ("F" or "RC")

    Returns
    -------
    pandas.DataFrame
        Columns:
        - primer_name
        - primer_sequence
        - target_name     (sequence ID from FASTA)
        - target_sequence (subsequence, RC if orientation == "RC")
    """
    # Read alignment
    msa = TabularMSA.read(fasta_path, constructor=DNA)

    records = []
    # Iterate over primers with valid start/end and orientation
    for _, row in primer_hits_df.iterrows():
        start = row.get('start')
        end = row.get('end')
        orientation = row.get('orientation')

        # Skip rows without a valid hit
        if pd.isna(start) or pd.isna(end) or orientation not in ("F", "RC"):
            continue

        start = int(start)
        end = int(end)

        primer_name = row['primer_name']
        primer_sequence = row['sequence']
        primer_set = row['primer_set']
        primer_type = row['primer_type']

        # For each sequence in the MSA, extract the region
        for i, seq in enumerate(msa):
            # seq is a skbio.DNA
            # Get a nice name for the target sequence
            target_name = seq.metadata.get('id', f"seq_{i}")

            subseq = seq[start:end]
            if orientation == "RC":
                subseq = subseq.reverse_complement()

            target_sequence = str(subseq)

            records.append({
                "primer_set":primer_set,
                "primer_type":primer_type,
                "primer_name": primer_name,
                "primer_sequence": primer_sequence,
                "target_name": target_name,
                "target_sequence": target_sequence,
            })

    return pd.DataFrame.from_records(records)

PRIMER_TYPE_ORDER = ["F3", "F2", "F1c", "LF", "FL",
                     "B3", "B2", "B1c", "LB", "BL"]

def reshape_targets_by_set_and_target(targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape targets_df so that index is (primer_set, target_name) and
    columns are per-primer-type blocks:

        F3, F3_sequence, F3_target_bind_sequence,
        F2, F2_sequence, F2_target_bind_sequence,
        ...

    Parameters
    ----------
    targets_df : pandas.DataFrame
        Output from extract_target_regions_from_alignment, must contain:
        - primer_set
        - primer_name
        - primer_sequence
        - primer_type
        - target_name
        - target_sequence

    Returns
    -------
    pandas.DataFrame
        Index: MultiIndex (primer_set, target_name)
        Columns: for each primer_type in PRIMER_TYPE_ORDER,
                 [type, f"{type}_sequence", f"{type}_target_bind_sequence"]
    """
    # Ensure primer_set exists; if not, create NA
    if "primer_set" not in targets_df.columns:
        targets_df = targets_df.copy()
        targets_df["primer_set"] = np.nan

    # This will be our base index
    base_index_cols = ["primer_set", "target_name"]
    targets_df = targets_df.set_index(base_index_cols + ["primer_type"], drop=False)

    # Start with an empty DataFrame indexed by (primer_set, target_name)
    all_index = targets_df.index.droplevel("primer_type").drop_duplicates()
    wide = pd.DataFrame(index=all_index)

    # For each primer_type, pull out rows of that type and join
    for pt in PRIMER_TYPE_ORDER:
        df_pt = targets_df[targets_df["primer_type"] == pt]

        if df_pt.empty:
            # No such primer type; just create empty columns when we join
            cols = {
                pt: pd.Series(index=wide.index, dtype=object),
                f"{pt}_sequence": pd.Series(index=wide.index, dtype=object),
                f"{pt}_target_bind_sequence": pd.Series(index=wide.index, dtype=object),
            }
            df_pt_wide = pd.DataFrame(cols)
        else:
            # Collapse to one row per (primer_set, target_name, primer_type)
            # If duplicates exist, take the first
            df_pt = df_pt.reset_index(drop=True)
            df_pt = df_pt.drop_duplicates(
                subset=["primer_set", "target_name", "primer_type"]
            )
            df_pt = df_pt.set_index(base_index_cols, drop=True)

            df_pt_wide = pd.DataFrame({
                pt: df_pt["primer_name"],
                f"{pt}_sequence": df_pt["primer_sequence"],
                f"{pt}_target_bind_sequence": df_pt["target_sequence"],
            })

        # Outer join to accumulate columns
        wide = wide.join(df_pt_wide, how="outer")

    # Ensure MultiIndex in the right order
    wide = wide.sort_index()
    return wide