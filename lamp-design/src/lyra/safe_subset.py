"""Build an auditable assay directory from benchmark metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SubsetSummary:
    included: int
    excluded: int
    missing: int


def build_safe_subset(
    *,
    input_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    id_column: str,
    exclude_column: str,
    exclude_values: Sequence[str],
    filename_column: str | None = None,
) -> SubsetSummary:
    """Link included assay CSVs and record every metadata decision."""

    input_dir = input_dir.resolve()
    metadata_path = metadata_path.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if output_dir == input_dir:
        raise ValueError("output_dir must differ from input_dir")

    excluded_normalized = {value.strip().casefold() for value in exclude_values}
    if not excluded_normalized:
        raise ValueError("exclude_values must be non-empty")

    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, exclude_column}
        if filename_column is not None:
            required.add(filename_column)
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(
                f"{metadata_path}: missing required columns: {missing_columns}"
            )
        rows = list(reader)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "subset_manifest.json"
    for entry in output_dir.iterdir():
        if entry.name in {manifest_path.name, "subset_manifest.csv"} or entry.is_symlink():
            continue
        raise ValueError(f"refusing to modify unexpected output entry: {entry}")
    for entry in output_dir.iterdir():
        if entry.is_symlink():
            entry.unlink()

    manifest_rows: list[dict[str, str]] = []
    included = excluded = missing = 0
    seen_filenames: set[str] = set()
    for row in rows:
        assay_id = row[id_column].strip()
        if not assay_id:
            raise ValueError(f"{metadata_path}: empty assay identifier")
        filename = (
            row[filename_column].strip()
            if filename_column is not None
            else f"{assay_id}.csv"
        )
        if not filename or Path(filename).name != filename:
            raise ValueError(f"unsafe assay filename in metadata: {filename!r}")
        if filename in seen_filenames:
            raise ValueError(f"duplicate assay filename in metadata: {filename}")
        seen_filenames.add(filename)

        exclusion_value = row[exclude_column].strip()
        source = input_dir / filename
        if exclusion_value.casefold() in excluded_normalized:
            status = "excluded"
            reason = f"{exclude_column}={exclusion_value}"
            excluded += 1
        elif not source.is_file():
            status = "missing"
            reason = "source file not found"
            missing += 1
        else:
            status = "included"
            reason = ""
            os.symlink(source, output_dir / filename)
            included += 1
        manifest_rows.append(
            {
                "assay_id": assay_id,
                "source_file": str(source),
                "status": status,
                "reason": reason,
            }
        )

    legacy_manifest = output_dir / "subset_manifest.csv"
    if legacy_manifest.exists():
        legacy_manifest.unlink()
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, indent=2)
        handle.write("\n")
    return SubsetSummary(included=included, excluded=excluded, missing=missing)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a symlinked benchmark subset with an audit manifest"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--filename-column")
    parser.add_argument("--exclude-column", required=True)
    parser.add_argument("--exclude-value", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = build_safe_subset(
        input_dir=args.input_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        id_column=args.id_column,
        filename_column=args.filename_column,
        exclude_column=args.exclude_column,
        exclude_values=args.exclude_value,
    )
    print(
        f"included={summary.included} excluded={summary.excluded} "
        f"missing={summary.missing}"
    )


if __name__ == "__main__":
    main()
