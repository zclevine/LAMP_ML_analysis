"""Aggregate synthetic runs into rebuttal-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    critical_95 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(values) - 1, 1.96)
    half = critical_95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _primary_metrics(run: dict) -> dict:
    metrics = run["metrics"]
    return metrics.get("composite", metrics.get("overall", metrics))


def load_runs(output_dir: str | Path) -> list[dict]:
    runs = []
    for path in sorted((Path(output_dir) / "runs").glob("*.json")):
        with path.open() as handle:
            run = json.load(handle)
        if run.get("status") == "complete":
            runs.append(run)
    return runs


def build_report(output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(output_dir)
    if not runs:
        raise RuntimeError(f"no complete runs found under {output_dir}")

    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["task"], run["model"])].append(run)
    summary = []
    for (task, model), values in sorted(grouped.items()):
        metrics = [_primary_metrics(value) for value in values]
        row = {
            "task": task,
            "model": model,
            "seeds": len(values),
            "parameters_mean": statistics.fmean(value["parameters"] for value in values),
        }
        for metric in (
            "r2",
            "spearman",
            "pearson",
            "validation_calibrated_r2",
            "oracle_calibration_gap",
            "rmse",
        ):
            mean, low, high = _mean_ci([value.get(metric) for value in metrics])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        if values[0]["task_kind"] == "walsh_epistasis":
            recovery = [value["metrics"]["walsh_recovery"] for value in values]
            for metric in (
                "coefficient_pearson",
                "coefficient_relative_l2",
                "coefficient_sign_accuracy",
                "spectrum_l1",
            ):
                mean, low, high = _mean_ci([value[metric] for value in recovery])
                row[f"walsh_{metric}_mean"] = mean
                row[f"walsh_{metric}_ci_low"] = low
                row[f"walsh_{metric}_ci_high"] = high
        summary.append(row)

    fields = sorted({key for row in summary for key in row})
    with (report_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    band_rows = []
    for run in runs:
        for band in run["metrics"].get("spectral", []):
            band_rows.append(
                {
                    "task": run["task"],
                    "model": run["model"],
                    "seed": run["seed"],
                    **band,
                }
            )
    if band_rows:
        with (report_dir / "frequency_band_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(band_rows[0]))
            writer.writeheader()
            writer.writerows(band_rows)

    _make_figures(runs, report_dir)
    manifest = {
        "complete_runs": len(runs),
        "tasks": sorted({run["task"] for run in runs}),
        "models": sorted({run["model"] for run in runs}),
        "summary": str(report_dir / "summary.csv"),
        "frequency_bands": str(report_dir / "frequency_band_metrics.csv") if band_rows else None,
    }
    with (report_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _make_figures(runs: list[dict], report_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epistasis = [
        run
        for run in runs
        if run["task_kind"] == "walsh_epistasis"
        and run["dataset"].get("pure_order") is not None
        and run["task"] == f"epistasis-order-{run['dataset']['pure_order']}"
    ]
    if epistasis:
        values = defaultdict(lambda: defaultdict(list))
        for run in epistasis:
            order = int(run["dataset"]["pure_order"])
            metric = _primary_metrics(run)
            values[run["model"]][order].append(metric)
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
        specs = (
            ("r2", "Test $R^2$"),
            ("spearman", "Test Spearman"),
            ("oracle_calibration_gap", "Calibration gap"),
        )
        for model, orders in sorted(values.items()):
            x = sorted(orders)
            for axis, (metric, label) in zip(axes, specs):
                y = [statistics.fmean(item[metric] for item in orders[order]) for order in x]
                axis.plot(x, y, marker="o", linewidth=1.8, label=model)
                axis.set_ylabel(label)
                axis.set_xlabel("Planted epistatic order")
                axis.grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(report_dir / "epistasis_order.png", dpi=220)
        figure.savefig(report_dir / "epistasis_order.pdf")
        plt.close(figure)

    frequency_tasks = sorted(
        {
            run["task"]
            for run in runs
            if run["task_kind"] == "frequency"
            and run["dataset"].get("target") == "components"
            and run["metrics"].get("channels")
        }
    )
    for task in frequency_tasks:
        frequency = [
            run
            for run in runs
            if run["task"] == task and run["metrics"].get("channels")
        ]
        by_model = defaultdict(lambda: defaultdict(list))
        for run in frequency:
            for frequency_value, channel in zip(
                run["dataset"]["frequencies"], run["metrics"]["channels"]
            ):
                by_model[run["model"]][float(frequency_value)].append(
                    float(channel["r2"])
                )
        figure, axis = plt.subplots(figsize=(6.8, 4.2))
        for model, bands in sorted(by_model.items()):
            x = sorted(bands)
            y = [statistics.fmean(bands[frequency]) for frequency in x]
            axis.semilogx(x, y, marker="o", linewidth=1.8, label=model)
        axis.set_xlabel("Frequency (cycles / sequence)")
        axis.set_ylabel("Whole-component sequence $R^2$")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        safe_task = "".join(
            character if character.isalnum() else "-" for character in task
        ).strip("-")
        figure.savefig(report_dir / f"frequency_component_r2__{safe_task}.png", dpi=220)
        figure.savefig(report_dir / f"frequency_component_r2__{safe_task}.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Lyra synthetic runs")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(build_report(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
