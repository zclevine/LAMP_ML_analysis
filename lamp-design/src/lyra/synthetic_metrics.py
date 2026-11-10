"""Metrics that separate ranking quality from calibration and spectral fidelity."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def _flatten(values: torch.Tensor) -> torch.Tensor:
    return values.detach().to(torch.float64).reshape(-1)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = _flatten(left)
    right = _flatten(right)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return float("nan")
    return float((left @ right / denominator).item())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = _flatten(values)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    stops = counts.cumsum(dim=0)
    starts = stops - counts
    average = 0.5 * (starts + stops - 1).to(values.dtype)
    ranks[order] = torch.repeat_interleave(average, counts)
    return ranks


def spearman(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return _correlation(_average_ranks(target), _average_ranks(prediction))


def r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = _flatten(target)
    prediction = _flatten(prediction)
    denominator = ((target - target.mean()) ** 2).sum()
    if denominator <= 0:
        return float("nan")
    return float((1.0 - ((target - prediction) ** 2).sum() / denominator).item())


def fit_affine_calibration(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Fit y = intercept + slope * prediction by least squares."""
    prediction = _flatten(prediction)
    target = _flatten(target)
    centered = prediction - prediction.mean()
    variance = centered.square().sum()
    if variance <= 0:
        return float(target.mean().item()), 0.0
    slope = float((centered @ (target - target.mean()) / variance).item())
    intercept = float((target.mean() - slope * prediction.mean()).item())
    return intercept, slope


def apply_affine_calibration(
    prediction: torch.Tensor, calibration: tuple[float, float]
) -> torch.Tensor:
    intercept, slope = calibration
    return prediction * slope + intercept


def regression_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    calibration: tuple[float, float] | None = None,
) -> dict[str, float]:
    target_flat = _flatten(target)
    prediction_flat = _flatten(prediction)
    residual = target_flat - prediction_flat
    pearson = _correlation(target_flat, prediction_flat)
    raw_r2 = r2(target_flat, prediction_flat)
    oracle_intercept, oracle_slope = fit_affine_calibration(prediction_flat, target_flat)
    oracle_prediction = apply_affine_calibration(
        prediction_flat, (oracle_intercept, oracle_slope)
    )
    result = {
        "mse": float(residual.square().mean().item()),
        "rmse": float(residual.square().mean().sqrt().item()),
        "mae": float(residual.abs().mean().item()),
        "r2": raw_r2,
        "pearson": pearson,
        "spearman": spearman(target_flat, prediction_flat),
        "mean_bias": float((prediction_flat.mean() - target_flat.mean()).item()),
        "dispersion_ratio": float(
            (prediction_flat.std(unbiased=False) / target_flat.std(unbiased=False).clamp_min(1e-12)).item()
        ),
        "oracle_calibration_intercept": oracle_intercept,
        "oracle_calibration_slope": oracle_slope,
        "oracle_calibrated_r2": r2(target_flat, oracle_prediction),
        "oracle_calibration_gap": r2(target_flat, oracle_prediction) - raw_r2,
    }
    if calibration is not None:
        calibrated = apply_affine_calibration(prediction_flat, calibration)
        result.update(
            {
                "validation_calibration_intercept": float(calibration[0]),
                "validation_calibration_slope": float(calibration[1]),
                "validation_calibrated_r2": r2(target_flat, calibrated),
            }
        )
    return result


def per_channel_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    validation_target: torch.Tensor,
    validation_prediction: torch.Tensor,
) -> list[dict[str, float]]:
    if target.shape != prediction.shape or target.size(-1) != validation_target.size(-1):
        raise ValueError("target and prediction channel shapes must match")
    output = []
    for channel in range(target.size(-1)):
        calibration = fit_affine_calibration(
            validation_prediction[..., channel], validation_target[..., channel]
        )
        output.append(
            regression_metrics(
                target[..., channel], prediction[..., channel], calibration=calibration
            )
        )
    return output


def frequency_metrics(
    target_components: torch.Tensor,
    predicted_components: torch.Tensor,
    validation_components: torch.Tensor,
    validation_prediction: torch.Tensor,
    frequencies: Sequence[float],
    *,
    spectral_eligible: bool = True,
) -> dict:
    """Report component, composite, and Fourier amplitude/phase fidelity."""
    channel = per_channel_metrics(
        target_components,
        predicted_components,
        validation_components,
        validation_prediction,
    )
    composite_target = target_components.sum(dim=-1)
    composite_prediction = predicted_components.sum(dim=-1)
    composite_validation = validation_components.sum(dim=-1)
    composite_validation_prediction = validation_prediction.sum(dim=-1)
    composite_calibration = fit_affine_calibration(
        composite_validation_prediction, composite_validation
    )
    composite = regression_metrics(
        composite_target,
        composite_prediction,
        calibration=composite_calibration,
    )

    spectral = []
    if spectral_eligible:
        length = target_components.size(1)
        steps = torch.arange(length, dtype=torch.float64)
        for index, frequency in enumerate(frequencies):
            angle = 2.0 * math.pi * float(frequency) * steps / float(length)
            design = torch.stack((angle.cos(), angle.sin(), torch.ones_like(angle)), dim=1)
            projection = torch.linalg.pinv(design).T
            truth_coefficient = target_components[:, :, index].to(torch.float64) @ projection
            estimate_coefficient = (
                predicted_components[:, :, index].to(torch.float64) @ projection
            )
            truth_amplitude = torch.linalg.vector_norm(truth_coefficient[:, :2], dim=1)
            estimate_amplitude = torch.linalg.vector_norm(
                estimate_coefficient[:, :2], dim=1
            )
            truth_phase = torch.atan2(-truth_coefficient[:, 1], truth_coefficient[:, 0])
            estimate_phase = torch.atan2(
                -estimate_coefficient[:, 1], estimate_coefficient[:, 0]
            )
            phase_delta = estimate_phase - truth_phase
            phase_delta = torch.atan2(phase_delta.sin(), phase_delta.cos())
            spectral.append(
                {
                    "frequency": float(frequency),
                    "amplitude_mae": float(
                        (estimate_amplitude - truth_amplitude).abs().mean().item()
                    ),
                    "amplitude_relative_mae": float(
                        (
                            (estimate_amplitude - truth_amplitude).abs()
                            / truth_amplitude.clamp_min(1e-8)
                        ).mean().item()
                    ),
                    "phase_mae_radians": float(phase_delta.abs().mean().item()),
                    "component_r2": channel[index]["r2"],
                    "component_spearman": channel[index]["spearman"],
                    "component_calibration_gap": channel[index]["oracle_calibration_gap"],
                }
            )
    return {
        "composite": composite,
        "channels": channel,
        "spectral": spectral,
        "spectral_eligible": spectral_eligible,
    }


def walsh_recovery_metrics(
    prediction: torch.Tensor,
    features: torch.Tensor,
    true_weights: Sequence[float],
    term_degrees: Sequence[int],
    ridge: float = 1e-5,
) -> dict:
    """Project predictions onto planted Walsh terms and recover degree energy."""
    prediction = prediction.detach().to(torch.float64).reshape(-1, 1)
    features = features.detach().to(torch.float64)
    design = torch.cat((features, torch.ones(features.size(0), 1)), dim=1)
    identity = torch.eye(design.size(1), dtype=design.dtype)
    identity[-1, -1] = 0.0
    coefficient = torch.linalg.solve(
        design.T @ design + float(ridge) * identity, design.T @ prediction
    ).squeeze(-1)
    recovered = coefficient[:-1]
    projected = features @ recovered + coefficient[-1]
    truth = torch.tensor(true_weights, dtype=torch.float64)
    relative_error = (recovered - truth).norm() / truth.norm().clamp_min(1e-12)
    degrees = sorted(set(int(value) for value in term_degrees))
    true_energy = {}
    recovered_energy = {}
    for degree in degrees:
        mask = torch.tensor([int(value) == degree for value in term_degrees])
        true_energy[str(degree)] = float(truth[mask].square().sum().item())
        recovered_energy[str(degree)] = float(recovered[mask].square().sum().item())
    total_true = sum(true_energy.values()) or 1.0
    total_recovered = sum(recovered_energy.values()) or 1.0
    true_fraction = {key: value / total_true for key, value in true_energy.items()}
    recovered_fraction = {
        key: value / total_recovered for key, value in recovered_energy.items()
    }
    spectrum_l1 = sum(
        abs(true_fraction[key] - recovered_fraction[key]) for key in true_fraction
    )
    return {
        "prediction_projection_r2": r2(prediction, projected),
        "coefficient_pearson": _correlation(truth, recovered),
        "coefficient_relative_l2": float(relative_error.item()),
        "coefficient_sign_accuracy": float(
            (truth.sign() == recovered.sign()).to(torch.float64).mean().item()
        ),
        "true_energy_fraction": true_fraction,
        "recovered_energy_fraction": recovered_fraction,
        "spectrum_l1": float(spectrum_l1),
    }


def finite_metrics(metrics: dict) -> bool:
    for value in metrics.values():
        if isinstance(value, dict) and not finite_metrics(value):
            return False
        if isinstance(value, list) and any(
            isinstance(item, dict) and not finite_metrics(item) for item in value
        ):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True
