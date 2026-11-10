"""Prepare the regression/ordinal FLIP and FLIP2 tasks for shared random CV."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Sequence


VALID_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYX-")


@dataclass(frozen=True)
class TaskSource:
    name: str
    path: Path
    member: str | None = None


def _legacy_sources(root: Path) -> list[TaskSource]:
    tasks: list[TaskSource] = []
    specifications = {
        "aav": (
            root / "aav" / "splits.zip",
            ("des_mut", "mut_des", "one_vs_many", "two_vs_many", "seven_vs_many", "low_vs_high"),
        ),
        "gb1": (
            root / "gb1" / "splits.zip",
            ("one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high"),
        ),
        "meltome": (
            root / "meltome" / "splits.zip",
            ("mixed_split", "human", "human_cell"),
        ),
    }
    for dataset, (archive, split_names) in specifications.items():
        for split_name in split_names:
            tasks.append(
                TaskSource(
                    name=f"flip_{dataset}_{split_name}",
                    path=archive,
                    member=f"splits/{split_name}.csv",
                )
            )
    return tasks


def _flip2_sources(root: Path) -> list[TaskSource]:
    return [
        TaskSource(name=f"flip2_{path.name.removesuffix('.csv.gz')}", path=path)
        for path in sorted(root.glob("*.csv.gz"))
    ]


def _normalize_sequence(value: str) -> str:
    sequence = value.strip().upper()
    if not sequence:
        raise ValueError("empty sequence")
    return "".join(residue if residue in VALID_RESIDUES else "X" for residue in sequence)


def _open_source(source: TaskSource) -> tuple[IO[str], zipfile.ZipFile | None]:
    if source.member is not None:
        archive = zipfile.ZipFile(source.path)
        stream = archive.open(source.member)
        return io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""), archive
    return gzip.open(source.path, mode="rt", encoding="utf-8-sig", newline=""), None


def prepare_tasks(sources: Iterable[TaskSource], output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for source in sources:
        output_path = output_dir / f"{source.name}.csv"
        digest = hashlib.sha256()
        lengths: list[int] = []
        rows = 0
        handle, archive = _open_source(source)
        try:
            reader = csv.DictReader(handle)
            missing = {"sequence", "target"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{source.path}: missing columns {sorted(missing)}")
            with output_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=("sequence", "target"))
                writer.writeheader()
                for row_index, row in enumerate(reader, start=2):
                    sequence = _normalize_sequence(row["sequence"])
                    try:
                        target = float(row["target"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{source.path}: invalid target on row {row_index}"
                        ) from error
                    if not math.isfinite(target):
                        raise ValueError(
                            f"{source.path}: non-finite target on row {row_index}"
                        )
                    writer.writerow({"sequence": sequence, "target": repr(target)})
                    encoded = sequence.encode("ascii")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                    digest.update(repr(target).encode("ascii"))
                    lengths.append(len(sequence))
                    rows += 1
        finally:
            handle.close()
            if archive is not None:
                archive.close()
        if rows < 5:
            raise ValueError(f"{source.name}: need at least five rows; got {rows}")
        manifest.append(
            {
                "task": source.name,
                "source": str(source.path),
                "archive_member": source.member,
                "rows": rows,
                "min_sequence_length": min(lengths),
                "max_sequence_length": max(lengths),
                "sequence_target_sha256": digest.hexdigest(),
            }
        )
    with (output_dir / "task_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "deterministic_balanced_random_5_fold",
                "supplied_split_columns_used": False,
                "metrics": ["spearman", "r2"],
                "excluded": ["kermut"],
                "tasks": manifest,
            },
            handle,
            indent=2,
        )
    return manifest


def prepare_landscape_view(
    task_dir: Path, output_dir: Path, task_manifest: Sequence[dict[str, object]]
) -> None:
    """Create the non-redundant random-CV view and record published-task aliases."""

    canonical_to_aliases = {
        "flip_aav_des_mut": [
            "flip_aav_mut_des",
            "flip_aav_one_vs_many",
            "flip_aav_two_vs_many",
            "flip_aav_seven_vs_many",
            "flip_aav_low_vs_high",
        ],
        "flip_gb1_one_vs_rest": [
            "flip_gb1_two_vs_rest",
            "flip_gb1_three_vs_rest",
            "flip_gb1_low_vs_high",
        ],
        "flip_meltome_mixed_split": [],
        "flip_meltome_human": [],
        "flip_meltome_human_cell": [],
        "flip2_amylase_one_to_many": [
            "flip2_amylase_by_mutation",
            "flip2_amylase_close_to_far",
            "flip2_amylase_far_to_close",
        ],
        "flip2_hydro_three_to_many": [
            "flip2_hydro_low_to_high",
            "flip2_hydro_to_P01053",
            "flip2_hydro_to_P06241",
            "flip2_hydro_to_P0A9X9",
        ],
        "flip2_ired_two_to_many": [],
        "flip2_nucb_two_to_many": [],
        "flip2_pdz3_single_to_double": [],
        "flip2_rhomax_by_wild_type": [],
        "flip2_trpb_one_to_many": [
            "flip2_trpb_by_position",
            "flip2_trpb_two_to_many",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task = {str(row["task"]): row for row in task_manifest}
    for canonical in canonical_to_aliases:
        source = task_dir / f"{canonical}.csv"
        destination = output_dir / source.name
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
    with (output_dir / "landscape_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "deterministic_balanced_random_5_fold",
                "supplied_split_columns_used": False,
                "canonical_landscapes": [
                    {
                        **by_task[canonical],
                        "published_task_aliases": aliases,
                    }
                    for canonical, aliases in canonical_to_aliases.items()
                ],
                "excluded": ["kermut"],
            },
            handle,
            indent=2,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--flip2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landscape-output-dir", type=Path)
    args = parser.parse_args(argv)
    sources = [*_legacy_sources(args.legacy_root), *_flip2_sources(args.flip2_root)]
    manifest = prepare_tasks(sources, args.output_dir)
    if args.landscape_output_dir is not None:
        prepare_landscape_view(args.output_dir, args.landscape_output_dir, manifest)
    print(f"prepared {len(manifest)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
