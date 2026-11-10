"""Publication figures for the corrected frequency-forecast benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


MODEL_ORDER = ["Lyra", "Mamba-2", "Mamba-3-SISO", "Hyena", "Transformer"]
MODEL_LABEL = {
    "Lyra": "Lyra",
    "Mamba-2": "Mamba-2",
    "Mamba-3-SISO": "Mamba-3",
    "Hyena": "Hyena",
    "Transformer": "Transformer",
}
MODEL_COLOR = {
    "Lyra": "#D55E00",
    "Mamba-2": "#0072B2",
    "Mamba-3-SISO": "#009E73",
    "Hyena": "#CC79A7",
    "Transformer": "#6F6F6F",
}
MODEL_MARKER = {
    "Lyra": "o",
    "Mamba-2": "s",
    "Mamba-3-SISO": "^",
    "Hyena": "D",
    "Transformer": "P",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _load_runs(directory: Path) -> list[dict]:
    rows = []
    for path in sorted((directory / "runs").glob("*.json")):
        with path.open() as handle:
            row = json.load(handle)
        if row.get("status") == "complete":
            rows.append(row)
    return rows


def _component_mean(row: dict, metric: str = "r2") -> float:
    return float(np.mean([channel[metric] for channel in row["metrics"]["channels"]]))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.nanmean(array))
    if array.size <= 1:
        return mean, 0.0
    return mean, float(1.96 * np.nanstd(array, ddof=1) / math.sqrt(array.size))


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=600)
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def write_prefix_figure_data(
    legacy_rows: list[dict], corrected_rows: list[dict], output: Path
) -> None:
    protocols = [
        ("zero", legacy_rows, "frequency-prefix-forecast"),
        ("query", corrected_rows, "frequency-prefix-query"),
        ("query_time", corrected_rows, "frequency-prefix-query-time"),
    ]
    records = []
    for protocol, rows, task in protocols:
        for model in MODEL_ORDER:
            model_rows = [row for row in rows if row["task"] == task and row["model"] == model]
            if not model_rows:
                continue
            aggregate, aggregate_ci = _mean_ci([_component_mean(row) for row in model_rows])
            for channel, frequency in enumerate(model_rows[0]["dataset"]["frequencies"]):
                mean, ci = _mean_ci(
                    [row["metrics"]["channels"][channel]["r2"] for row in model_rows]
                )
                records.append(
                    {
                        "protocol": protocol,
                        "task": task,
                        "model": MODEL_LABEL[model],
                        "frequency": frequency,
                        "n_seeds": len(model_rows),
                        "component_r2_mean": mean,
                        "component_r2_95ci": ci,
                        "mean_across_frequencies_r2": aggregate,
                        "mean_across_frequencies_95ci": aggregate_ci,
                    }
                )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_prefix_protocol_ablation(
    legacy_rows: list[dict], corrected_rows: list[dict], output: Path
) -> None:
    protocols = [
        ("All-zero suffix\n(legacy)", legacy_rows, "frequency-prefix-forecast"),
        ("Query token", corrected_rows, "frequency-prefix-query"),
        ("Query + time", corrected_rows, "frequency-prefix-query-time"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 3.8))
    figure.subplots_adjust(wspace=0.36, top=0.77, left=0.055, right=0.99, bottom=0.2)

    axis = axes[0]
    x = np.arange(len(protocols))
    for model in MODEL_ORDER:
        means, cis = [], []
        for _, rows, task in protocols:
            values = [
                _component_mean(row)
                for row in rows
                if row["task"] == task and row["model"] == model
            ]
            mean, ci = _mean_ci(values)
            means.append(mean)
            cis.append(ci)
        axis.errorbar(
            x,
            means,
            yerr=cis,
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            markersize=5,
            linewidth=1.5,
            capsize=2,
            label=MODEL_LABEL[model],
        )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axis.set_xticks(x, [value[0] for value in protocols])
    axis.set_ylabel("Mean component $R^2$")
    axis.set_title("a  Future-token protocol", loc="left", fontweight="bold")
    axis.set_ylim(-0.08, 1.03)

    axis = axes[1]
    query_time = [
        row for row in corrected_rows if row["task"] == "frequency-prefix-query-time"
    ]
    for model in MODEL_ORDER:
        model_rows = [row for row in query_time if row["model"] == model]
        frequencies = model_rows[0]["dataset"]["frequencies"]
        means, cis = [], []
        for channel in range(len(frequencies)):
            values = [row["metrics"]["channels"][channel]["r2"] for row in model_rows]
            mean, ci = _mean_ci(values)
            means.append(mean)
            cis.append(ci)
        axis.errorbar(
            frequencies,
            means,
            yerr=cis,
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            markersize=5,
            linewidth=1.5,
            capsize=2,
        )
    axis.axhline(0.0, color="#B8B8B8", linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 16, 256], ["1", "16", "256"])
    axis.set_xlabel("Frequency (cycles / sequence)")
    axis.set_ylabel("Component $R^2$")
    axis.set_title("b  Query + time by frequency", loc="left", fontweight="bold")
    axis.set_ylim(-0.08, 1.03)

    axis = axes[2]
    y = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        model_rows = [row for row in query_time if row["model"] == model]
        raw, rank = _mean_ci([row["metrics"]["composite"]["r2"] for row in model_rows])[0], _mean_ci(
            [row["metrics"]["composite"]["spearman"] for row in model_rows]
        )[0]
        axis.plot([raw, rank], [index, index], color="#B8B8B8", linewidth=1.2, zorder=1)
        axis.scatter(
            raw,
            index,
            s=38,
            color=MODEL_COLOR[model],
            marker="o",
            zorder=2,
        )
        axis.scatter(
            rank,
            index,
            s=42,
            facecolors="none",
            edgecolors=MODEL_COLOR[model],
            marker="D",
            linewidths=1.2,
            zorder=2,
        )
    axis.axvline(0.0, color="#B8B8B8", linewidth=0.8)
    axis.set_yticks(y, [MODEL_LABEL[model] for model in MODEL_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Composite score")
    axis.set_title("c  Calibration versus rank", loc="left", fontweight="bold")
    axis.set_xlim(-0.08, 1.03)
    axis.scatter([], [], s=38, color="#555555", marker="o", label="Raw $R^2$")
    axis.scatter(
        [], [], s=42, facecolors="none", edgecolors="#555555", marker="D", label="Spearman"
    )
    axis.legend(frameon=False, loc="lower right")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    _save_figure(figure, output)


def _representative_seed(rows: list[dict]) -> int:
    relevant = [
        row
        for row in rows
        if row["task"] == "frequency-prefix-query-time"
        and row["model"] in {"Lyra", "Mamba-2", "Mamba-3-SISO"}
    ]
    by_model = {
        model: {int(row["seed"]): _component_mean(row) for row in relevant if row["model"] == model}
        for model in ("Lyra", "Mamba-2", "Mamba-3-SISO")
    }
    shared_seeds = set.intersection(*(set(values) for values in by_model.values()))
    if not shared_seeds:
        raise ValueError("no shared seed across Lyra, Mamba-2, and Mamba-3")
    centers = {model: float(np.mean(list(values.values()))) for model, values in by_model.items()}
    scales = {
        model: max(float(np.std(list(values.values()), ddof=1)), 1e-6)
        for model, values in by_model.items()
    }
    distance = {
        seed: sum(
            abs(by_model[model][seed] - centers[model]) / scales[model]
            for model in by_model
        )
        for seed in shared_seeds
    }
    return min(distance, key=distance.get)


def plot_sequence_reconstruction_compact(rows: list[dict], output: Path) -> tuple[int, int]:
    models = ["Lyra", "Mamba-2", "Mamba-3-SISO"]
    task_rows = [row for row in rows if row["task"] == "frequency-prefix-query-time"]
    seed = _representative_seed(task_rows)
    selected = {
        row["model"]: row
        for row in task_rows
        if int(row["seed"]) == seed and row["model"] in models
    }
    example_index = 0
    target = np.asarray(selected["Lyra"]["reconstructions"][example_index]["targets"])
    predictions = {
        model: np.asarray(selected[model]["reconstructions"][example_index]["predictions"])
        for model in models
    }
    length = target.shape[0]
    boundary = int(round(length * selected["Lyra"]["dataset"]["observed_fraction"]))
    panels = [
        (0, length, None, "a  Composite forecast"),
        (0, length, 0, "b  Low frequency ($f=1$)"),
        (boundary - 32, boundary + 128, 1, "c  Medium frequency ($f=16$)"),
        (boundary - 12, boundary + 36, 2, "d  High frequency ($f=256$)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.2))
    figure.subplots_adjust(wspace=0.22, hspace=0.34, top=0.89, left=0.075, right=0.985, bottom=0.09)
    for axis, (start, stop, component, title) in zip(axes.flat, panels):
        truth = target.sum(axis=-1) if component is None else target[:, component]
        forecast_values = []
        for model in models:
            prediction = predictions[model]
            prediction = prediction.sum(axis=-1) if component is None else prediction[:, component]
            forecast_values.append(prediction[max(boundary, start) : stop])
        finite = np.concatenate([truth[start:stop], *[value for value in forecast_values if value.size]])
        low, high = np.nanpercentile(finite, [0.5, 99.5])
        pad = max(0.15, 0.08 * (high - low))
        axis.axvspan(max(boundary, start), stop, color="#888888", alpha=0.06, linewidth=0)
        if start < boundary < stop:
            axis.axvline(boundary, color="#777777", linewidth=0.8, linestyle="--")
        axis.plot(np.arange(start, stop), truth[start:stop], color="#202020", linewidth=1.05, label="Target")
        for model in models:
            prediction = predictions[model]
            prediction = prediction.sum(axis=-1) if component is None else prediction[:, component]
            forecast_start = max(boundary, start)
            axis.plot(
                np.arange(forecast_start, stop),
                prediction[forecast_start:stop],
                color=MODEL_COLOR[model],
                linewidth=1.25,
                label=MODEL_LABEL[model],
            )
        axis.set_xlim(start, stop - 1)
        axis.set_ylim(low - pad, high + pad)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Sequence position")
        axis.set_ylabel("Signal")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#EAEAEA", linewidth=0.55)
        for index, model in enumerate(models):
            metric = selected[model]["metrics"]["composite" if component is None else "channels"]
            if component is not None:
                metric = metric[component]
            axis.text(
                0.02,
                0.06 + 0.07 * index,
                f"{MODEL_LABEL[model]}  $R^2={metric['r2']:.3f}$",
                transform=axis.transAxes,
                color=MODEL_COLOR[model],
                fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99))
    _save_figure(figure, output)
    return seed, example_index


def plot_sequence_reconstruction(
    rows: list[dict], output: Path, models: list[str]
) -> tuple[int, int]:
    task_rows = [row for row in rows if row["task"] == "frequency-prefix-query-time"]
    seed = _representative_seed(task_rows)
    selected = {
        row["model"]: row
        for row in task_rows
        if int(row["seed"]) == seed and row["model"] in models
    }
    missing = set(models) - set(selected)
    if missing:
        raise ValueError(f"missing reconstruction runs for {sorted(missing)}")
    example_index = 0
    target = np.asarray(selected[models[0]]["reconstructions"][example_index]["targets"])
    for model in models[1:]:
        other = np.asarray(selected[model]["reconstructions"][example_index]["targets"])
        if not np.allclose(target, other, atol=1e-6):
            raise ValueError("models do not share the same deterministic test example")

    length = target.shape[0]
    boundary = int(round(length * selected[models[0]]["dataset"]["observed_fraction"]))
    windows = [
        (0, length, None, "Composite — full sequence"),
        (max(0, boundary - 64), min(length, boundary + 128), None, "Composite — boundary zoom"),
        (0, length, 0, "$f=1$ component"),
        (max(0, boundary - 32), min(length, boundary + 128), 1, "$f=16$ component"),
        (max(0, boundary - 12), min(length, boundary + 36), 2, "$f=256$ component"),
    ]
    figure, axes = plt.subplots(
        len(windows),
        len(models),
        figsize=(3.6 * len(models), 9.1),
        squeeze=False,
    )
    figure.subplots_adjust(wspace=0.15, hspace=0.48, top=0.93, left=0.075, right=0.99, bottom=0.065)
    all_predictions = {
        model: np.asarray(selected[model]["reconstructions"][example_index]["predictions"])
        for model in models
    }

    for row_index, (start, stop, component, label) in enumerate(windows):
        truth = target.sum(axis=-1) if component is None else target[:, component]
        prediction_series = [
            values.sum(axis=-1) if component is None else values[:, component]
            for values in all_predictions.values()
        ]
        visible = [truth[start:stop]] + [value[max(boundary, start) : stop] for value in prediction_series]
        finite = np.concatenate([value for value in visible if value.size])
        low, high = np.nanpercentile(finite, [0.5, 99.5])
        pad = max(0.15, 0.08 * (high - low))
        for column, model in enumerate(models):
            axis = axes[row_index, column]
            prediction = all_predictions[model]
            prediction = prediction.sum(axis=-1) if component is None else prediction[:, component]
            steps = np.arange(start, stop)
            axis.axvspan(max(boundary, start), stop, color=MODEL_COLOR[model], alpha=0.055, linewidth=0)
            if start < boundary < stop:
                axis.axvline(boundary, color="#777777", linewidth=0.8, linestyle="--")
            axis.plot(steps, truth[start:stop], color="#202020", linewidth=1.0, label="Target")
            forecast_start = max(boundary, start)
            axis.plot(
                np.arange(forecast_start, stop),
                prediction[forecast_start:stop],
                color=MODEL_COLOR[model],
                linewidth=1.25,
                label="Forecast",
            )
            axis.set_xlim(start, stop - 1)
            axis.set_ylim(low - pad, high + pad)
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#EAEAEA", linewidth=0.55)
            if column == 0:
                axis.set_ylabel(label)
            else:
                axis.tick_params(labelleft=False)
            if row_index == 0:
                axis.set_title(MODEL_LABEL[model], color=MODEL_COLOR[model], fontweight="bold")
            if row_index == len(windows) - 1:
                axis.set_xlabel("Sequence position")
            metric = selected[model]["metrics"]["composite" if component is None else "channels"]
            if component is not None:
                metric = metric[component]
            axis.text(
                0.98,
                0.92,
                f"$R^2={metric['r2']:.3f}$",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.997))
    axes[0, 0].text(
        0.25,
        1.06,
        "observed prefix",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    axes[0, 0].text(
        0.75,
        1.06,
        "forecast suffix",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=MODEL_COLOR[models[0]],
    )
    _save_figure(figure, output)
    return seed, example_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _style()
    legacy_rows = _load_runs(args.legacy_dir)
    corrected_rows = _load_runs(args.corrected_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix_protocol_ablation(
        legacy_rows, corrected_rows, args.output_dir / "frequency-prefix-protocol-ablation"
    )
    write_prefix_figure_data(
        legacy_rows, corrected_rows, args.output_dir / "frequency-prefix-figure-data.csv"
    )
    seed, example = plot_sequence_reconstruction(
        corrected_rows,
        args.output_dir / "sequence-reconstruction-main",
        ["Lyra", "Mamba-2", "Mamba-3-SISO"],
    )
    plot_sequence_reconstruction_compact(
        corrected_rows, args.output_dir / "sequence-reconstruction-compact"
    )
    plot_sequence_reconstruction(
        corrected_rows,
        args.output_dir / "sequence-reconstruction-all-models",
        MODEL_ORDER,
    )
    with (args.output_dir / "figure-manifest.json").open("w") as handle:
        json.dump(
            {
                "representative_seed": seed,
                "representative_example": example,
                "legacy_runs": len(legacy_rows),
                "corrected_runs": len(corrected_rows),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
