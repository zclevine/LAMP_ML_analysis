"""Deterministic synthetic datasets for epistasis and frequency experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TensorSplit:
    inputs: torch.Tensor
    targets: torch.Tensor
    components: torch.Tensor | None = None


@dataclass(frozen=True)
class SyntheticDataset:
    train: TensorSplit
    validation: TensorSplit
    test: TensorSplit
    metadata: dict


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _sample_unique_subsets(
    length: int,
    order: int,
    count: int,
    generator: torch.Generator,
    span: str = "mixed",
    span_window: int | None = None,
    factor_window: int = 5,
) -> list[tuple[int, ...]]:
    if not 1 <= order <= length:
        raise ValueError("order must be between 1 and sequence length")
    maximum = math.comb(length, order)
    if count > maximum:
        raise ValueError(f"requested {count} unique terms but only {maximum} exist")
    if span not in {"local", "global", "mixed", "factorized"}:
        raise ValueError("span must be local, global, mixed, or factorized")
    if span != "mixed" and span_window is None:
        raise ValueError("span_window is required for local/global subsets")
    if span == "factorized" and (order < 2 or factor_window < math.ceil(order / 2)):
        raise ValueError("factorized terms require order >= 2 and sufficiently large factor_window")

    subsets: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = max(10_000, count * 2_000)
    while len(subsets) < count and attempts < max_attempts:
        attempts += 1
        if span == "factorized":
            left_order = order // 2
            right_order = order - left_order
            minimum_gap = int(span_window)
            maximum_left = length - 2 * factor_window - minimum_gap
            if maximum_left < 0:
                raise ValueError("sequence is too short for the requested factorization")
            left_start = int(
                torch.randint(0, maximum_left + 1, (1,), generator=generator).item()
            )
            right_minimum = left_start + factor_window + minimum_gap
            right_start = int(
                torch.randint(
                    right_minimum,
                    length - factor_window + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            left = (
                torch.randperm(factor_window, generator=generator)[:left_order]
                + left_start
            )
            right = (
                torch.randperm(factor_window, generator=generator)[:right_order]
                + right_start
            )
            candidate = tuple(sorted(torch.cat((left, right)).tolist()))
        else:
            candidate = tuple(
                sorted(torch.randperm(length, generator=generator)[:order].tolist())
            )
        width = candidate[-1] - candidate[0] if order > 1 else 0
        if span == "local" and width >= int(span_window):
            continue
        if span == "global" and width < int(span_window):
            continue
        subsets.add(candidate)
    if len(subsets) != count:
        raise RuntimeError(
            f"could not sample {count} {span} order-{order} terms; "
            "relax span_window or reduce terms_per_order"
        )
    return sorted(subsets)


def walsh_features(
    sequences: torch.Tensor, subsets: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Evaluate Walsh characters chi_S(x)=prod(i in S) x_i."""
    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (examples, length)")
    if not subsets:
        return sequences.new_zeros((sequences.size(0), 0), dtype=torch.float32)
    columns = [sequences[:, list(subset)].prod(dim=1) for subset in subsets]
    return torch.stack(columns, dim=1).to(torch.float32)


def _unit_weights(count: int, energy: float, generator: torch.Generator) -> torch.Tensor:
    values = torch.randn(count, generator=generator)
    return values * (math.sqrt(float(energy)) / values.norm().clamp_min(1e-12))


def make_walsh_epistasis(
    *,
    sequence_length: int = 48,
    max_order: int = 8,
    terms_per_order: int = 12,
    n_train: int = 20_000,
    n_validation: int = 2_500,
    n_test: int = 2_500,
    noise_std: float = 0.1,
    seed: int = 1,
    pure_order: int | None = None,
    energy_decay: float = 1.0,
    weight_scale: float = 1.0,
    target: str = "scalar",
    span: str = "mixed",
    span_window: int | None = 24,
    factor_window: int = 5,
) -> SyntheticDataset:
    """Create a pure-order or mixed-order Walsh/ANOVA regression problem.

    Inputs are Rademacher sequences represented as one continuous channel.  In
    mixed mode each degree receives equal energy when ``energy_decay=1``;
    otherwise degree-k energy is proportional to ``energy_decay ** (k - 1)``.
    """
    if max_order <= 0:
        raise ValueError("max_order must be positive")
    if pure_order is not None and not 1 <= pure_order <= max_order:
        raise ValueError("pure_order must be in [1, max_order]")
    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("split sizes must be positive")
    if target not in {"scalar", "terms"}:
        raise ValueError("target must be scalar or terms")

    generator = _generator(seed)
    degrees = [pure_order] if pure_order is not None else list(range(1, max_order + 1))
    subsets: list[tuple[int, ...]] = []
    weights: list[torch.Tensor] = []
    term_degrees: list[int] = []
    for degree in degrees:
        degree_subsets = _sample_unique_subsets(
            sequence_length,
            degree,
            terms_per_order,
            generator,
            span=span,
            span_window=span_window,
            factor_window=factor_window,
        )
        degree_norm = float(weight_scale) * float(energy_decay) ** (degree - 1)
        energy = degree_norm**2
        subsets.extend(degree_subsets)
        weights.append(_unit_weights(len(degree_subsets), energy, generator))
        term_degrees.extend([degree] * len(degree_subsets))
    coefficient = torch.cat(weights)

    def split(size: int) -> TensorSplit:
        sequence = torch.randint(
            0, 2, (size, sequence_length), generator=generator, dtype=torch.int8
        ).to(torch.float32)
        sequence = sequence.mul_(2).sub_(1)
        feature = walsh_features(sequence, subsets)
        clean = feature @ coefficient
        noise = torch.randn(size, generator=generator) * float(noise_std)
        if target == "scalar":
            split_target = (clean + noise).unsqueeze(-1)
        else:
            split_target = feature * coefficient.unsqueeze(0)
            if noise_std:
                split_target = split_target + noise.unsqueeze(-1) / math.sqrt(
                    split_target.size(-1)
                )
        return TensorSplit(sequence.unsqueeze(-1), split_target, feature)

    return SyntheticDataset(
        train=split(n_train),
        validation=split(n_validation),
        test=split(n_test),
        metadata={
            "kind": "walsh_epistasis",
            "sequence_length": sequence_length,
            "max_order": max_order,
            "pure_order": pure_order,
            "terms_per_order": terms_per_order,
            "subsets": [list(value) for value in subsets],
            "weights": coefficient.tolist(),
            "term_degrees": term_degrees,
            "noise_std": noise_std,
            "target": target,
            "weight_scale": weight_scale,
            "seed": seed,
            "span": span,
            "span_window": span_window,
            "factor_window": factor_window,
        },
    )


def _frequency_split(
    *,
    size: int,
    length: int,
    frequencies: Sequence[float],
    generator: torch.Generator,
    amplitude_range: tuple[float, float],
    phase_range: tuple[float, float],
    offset_range: tuple[float, float],
    input_noise_std: float,
    observed_fraction: float,
    forecast_features: str,
    target: str,
) -> TensorSplit:
    n_components = len(frequencies)
    amplitudes = torch.empty(size, n_components).uniform_(
        amplitude_range[0], amplitude_range[1], generator=generator
    )
    phases = torch.empty(size, n_components).uniform_(
        phase_range[0], phase_range[1], generator=generator
    )
    offsets = torch.empty(size, n_components).uniform_(
        offset_range[0], offset_range[1], generator=generator
    )
    steps = torch.arange(length, dtype=torch.float32)
    angular = 2.0 * math.pi * steps / float(length)
    frequency = torch.tensor(frequencies, dtype=torch.float32)
    component = amplitudes[:, None, :] * torch.cos(
        angular[None, :, None] * frequency[None, None, :] + phases[:, None, :]
    ) + offsets[:, None, :]
    composite = component.sum(dim=-1, keepdim=True)
    noisy = composite + torch.randn(composite.shape, generator=generator) * input_noise_std

    observed = max(1, min(length, int(round(length * observed_fraction))))
    mask = torch.zeros(size, length, 1)
    mask[:, :observed] = 1.0
    input_channels = [noisy * mask, mask]
    if forecast_features in {"query", "query_time"}:
        # A target-independent query token keeps the held-out suffix nonzero.
        # This matters for selective SSMs whose readout and gate are generated
        # from the current token: an all-zero token can make the state
        # unobservable even when the recurrent state contains useful history.
        input_channels.append(torch.ones(size, length, 1))
    if forecast_features == "query_time":
        normalized_time = torch.linspace(-1.0, 1.0, length).view(1, length, 1)
        input_channels.append(normalized_time.expand(size, -1, -1))
    inputs = torch.cat(input_channels, dim=-1)
    if target == "components":
        targets = component
    elif target == "composite":
        targets = composite
    elif target == "coefficients":
        targets = amplitudes
    else:
        raise ValueError("target must be components, composite, or coefficients")
    return TensorSplit(inputs, targets, component)


def make_frequency_dataset(
    *,
    frequencies: Iterable[float] = (1.0, 16.0, 256.0),
    sequence_length: int = 1024,
    n_train: int = 8_000,
    n_validation: int = 1_000,
    n_test: int = 1_000,
    amplitude_range: tuple[float, float] = (0.2, 1.5),
    test_amplitude_range: tuple[float, float] | None = None,
    phase_range: tuple[float, float] = (-math.pi, math.pi),
    test_phase_range: tuple[float, float] | None = None,
    offset_range: tuple[float, float] = (-0.25, 0.25),
    input_noise_std: float = 0.0,
    observed_fraction: float = 1.0,
    forecast_features: str = "zero",
    target: str = "components",
    seed: int = 1,
) -> SyntheticDataset:
    """Create coefficient, component-separation, or forecasting wave tasks.

    The default frequencies reproduce the low/moderate/high triplet from the
    frequency-bias SSM paper.  ``target='components'`` upgrades coefficient
    regression to full-sequence component prediction.  Setting
    ``observed_fraction < 1`` creates a prefix-to-future forecasting task; the
    second input channel is the observation mask, preventing target leakage.
    ``forecast_features='query'`` appends a constant, target-independent future
    query channel. ``'query_time'`` also appends normalized absolute time. The
    default ``'zero'`` preserves the original all-zero suffix protocol.
    """
    frequencies = tuple(float(value) for value in frequencies)
    if not frequencies:
        raise ValueError("at least one frequency is required")
    if max(frequencies) > sequence_length / 2:
        raise ValueError("frequencies must not exceed the Nyquist frequency")
    if not 0.0 < observed_fraction <= 1.0:
        raise ValueError("observed_fraction must be in (0, 1]")
    if forecast_features not in {"zero", "query", "query_time"}:
        raise ValueError("forecast_features must be zero, query, or query_time")
    generator = _generator(seed)
    test_amplitude_range = test_amplitude_range or amplitude_range
    test_phase_range = test_phase_range or phase_range

    shared = dict(
        length=sequence_length,
        frequencies=frequencies,
        generator=generator,
        offset_range=offset_range,
        input_noise_std=input_noise_std,
        observed_fraction=observed_fraction,
        forecast_features=forecast_features,
        target=target,
    )
    train = _frequency_split(
        size=n_train,
        amplitude_range=amplitude_range,
        phase_range=phase_range,
        **shared,
    )
    validation = _frequency_split(
        size=n_validation,
        amplitude_range=amplitude_range,
        phase_range=phase_range,
        **shared,
    )
    test = _frequency_split(
        size=n_test,
        amplitude_range=test_amplitude_range,
        phase_range=test_phase_range,
        **shared,
    )
    return SyntheticDataset(
        train=train,
        validation=validation,
        test=test,
        metadata={
            "kind": "frequency",
            "frequencies": list(frequencies),
            "sequence_length": sequence_length,
            "amplitude_range": list(amplitude_range),
            "test_amplitude_range": list(test_amplitude_range),
            "phase_range": list(phase_range),
            "test_phase_range": list(test_phase_range),
            "offset_range": list(offset_range),
            "input_noise_std": input_noise_std,
            "observed_fraction": observed_fraction,
            "forecast_features": forecast_features,
            "target": target,
            "seed": seed,
        },
    )
