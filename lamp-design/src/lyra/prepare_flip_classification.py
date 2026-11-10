"""Prepare sequence-level FLIP/FLIP2 classification CSVs for random CV."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


VALID = frozenset("ACDEFGHIKLMNPQRSTVWYX-")
SOURCES = {
    "scl": ("mixed_soft", "mixed_hard", "human_soft", "human_hard", "balanced"),
    "sav": ("mixed", "human", "only_savs"),
}


def normalize(sequence: str) -> str:
    return "".join(character if character in VALID else "X" for character in sequence.strip().upper())


def prepare(legacy_root: Path, nucb_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family, splits in SOURCES.items():
        archive_path = legacy_root / family / "splits.zip"
        with zipfile.ZipFile(archive_path) as archive:
            for split in splits:
                task = f"flip_{family}_{split}"
                with archive.open(f"splits/{split}.csv") as raw:
                    rows = list(csv.DictReader(line.decode("utf-8-sig") for line in raw))
                class_names = sorted({row["target"].strip() for row in rows})
                class_ids = {name: index for index, name in enumerate(class_names)}
                destination = output_dir / f"{task}.csv"
                with destination.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("sequence", "target"))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({"sequence": normalize(row["sequence"]), "target": class_ids[row["target"].strip()]})
                manifest.append({"task": task, "family": family, "split": split, "rows": len(rows), "classes": class_names, "source": str(archive_path), "published_split_ignored_for_random_cv": True})
    task = "flip2_nucb_two_to_many_classification"
    with nucb_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({int(float(row["target"])) for row in rows})
    if labels != list(range(len(labels))):
        raise ValueError(f"NucB labels are not contiguous: {labels}")
    with (output_dir / f"{task}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sequence", "target"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"sequence": normalize(row["sequence"]), "target": int(float(row["target"]))})
    manifest.append({"task": task, "family": "nucb", "split": "two_to_many", "rows": len(rows), "classes": [str(label) for label in labels], "source": str(nucb_csv), "published_split_ignored_for_random_cv": True})
    (output_dir / "classification_manifest.json").write_text(json.dumps({"protocol": "deterministic_balanced_random_5_fold", "metrics": ["accuracy", "balanced_accuracy", "macro_f1", "mcc"], "weights_saved": True, "excluded": ["kermut"], "deferred_tasks": ["flip_scl_mixed_vs_human_2 (multilabel)", "flip_bind_* (residue-level)", "flip_secondary_structure_sampled (residue-level)", "flip_conservation_sampled (residue-level)"], "tasks": manifest}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--nucb-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.legacy_root, args.nucb_csv, args.output_dir)


if __name__ == "__main__":
    main()
