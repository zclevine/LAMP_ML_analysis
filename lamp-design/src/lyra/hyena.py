"""Standalone Hyena baseline for sequence-level regression/classification.

This is a compact adaptation of HazyResearch Safari's Apache-2.0 licensed
``standalone_hyena.py`` implementation:
https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py

The implementation was changed to avoid an einops dependency, integrate with
Lyra's one-hot inputs and pooling heads, support stacked residual blocks, and
mark the trainable positional table so it can be excluded from parameter
matching without being frozen during optimization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fft_convolution(
    inputs: torch.Tensor, kernel: torch.Tensor, diagonal: torch.Tensor
) -> torch.Tensor:
    """Apply a causal FFT convolution over the final dimension."""

    length = inputs.size(-1)
    fft_size = 2 * length
    fft_dtype = (
        torch.float32
        if inputs.dtype in {torch.float16, torch.bfloat16}
        else inputs.dtype
    )
    kernel_fft = torch.fft.rfft(kernel.to(fft_dtype), n=fft_size) / fft_size
    input_fft = torch.fft.rfft(inputs.to(fft_dtype), n=fft_size)
    output = torch.fft.irfft(
        input_fft * kernel_fft, n=fft_size, norm="forward"
    )[..., :length]
    return (output + inputs * diagonal.unsqueeze(-1)).to(inputs.dtype)


class _Sine(nn.Module):
    def __init__(self, width: int, frequency: float = 1.0) -> None:
        super().__init__()
        self.frequency = nn.Parameter(torch.full((1, width), frequency))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.frequency * inputs)


class HyenaPositionalEmbedding(nn.Module):
    """Trainable complex-exponential features used only to synthesize filters."""

    def __init__(self, embedding_dim: int, max_length: int) -> None:
        super().__init__()
        if embedding_dim < 3 or embedding_dim % 2 == 0:
            raise ValueError("embedding_dim must be odd and at least 3")
        time = torch.linspace(0, 1, max_length)[None, :, None]
        bands = (embedding_dim - 1) // 2
        positions = torch.arange(max_length, dtype=torch.float32)[None, :, None]
        frequencies = torch.linspace(1e-4, max(bands - 1, 1), bands)[None, None]
        phase = 2 * math.pi * positions * frequencies / max_length
        embedding = torch.cat((time, torch.cos(phase), -torch.sin(phase)), dim=-1)
        self.embedding = nn.Parameter(embedding)
        # Parameter matching deliberately ignores this table. The marker keeps
        # that policy local and robust to module nesting/name changes.
        self.embedding._exclude_from_parameter_count = True
        self.register_buffer("time", time, persistent=False)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.embedding.size(1):
            raise ValueError(
                f"sequence length {length} exceeds Hyena max_length "
                f"{self.embedding.size(1)}"
            )
        return self.embedding[:, :length], self.time[:, :length]


class HyenaFilter(nn.Module):
    """Implicit long filter with exponential modulation."""

    def __init__(
        self,
        channels: int,
        filter_width: int,
        max_length: int,
        embedding_dim: int = 3,
        inner_layers: int = 2,
        frequency: float = 1.0,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(channels))
        self.positional_embedding = HyenaPositionalEmbedding(
            embedding_dim, max_length
        )
        layers: list[nn.Module] = [
            nn.Linear(embedding_dim, filter_width),
            _Sine(filter_width, frequency),
        ]
        for _ in range(inner_layers):
            layers.extend((nn.Linear(filter_width, filter_width), _Sine(filter_width, frequency)))
        layers.append(nn.Linear(filter_width, channels, bias=False))
        self.implicit_filter = nn.Sequential(*layers)
        maximum_decay = math.log(1e-2) / 0.3
        minimum_decay = math.log(1e-2) / 1.5
        decay = torch.linspace(minimum_decay, maximum_decay, channels)[None, None]
        self.register_buffer("decay", decay, persistent=True)

    def filter(self, length: int) -> torch.Tensor:
        embedding, time = self.positional_embedding(length)
        kernel = self.implicit_filter(embedding)
        return kernel * torch.exp(-time * self.decay.abs())


class HyenaOperator(nn.Module):
    """One Safari-style Hyena recurrence operating on ``(B, L, D)`` tensors."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        max_length: int = 4096,
        order: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("Hyena order must be at least 2")
        self.d_model = d_model
        self.order = order
        inner_width = d_model * (order + 1)
        self.in_projection = nn.Linear(d_model, inner_width)
        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            kernel_size=3,
            padding=2,
            groups=inner_width,
        )
        self.long_filter = HyenaFilter(
            channels=d_model * (order - 1),
            filter_width=d_state,
            max_length=max_length,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(1)
        projected = self.in_projection(inputs).transpose(1, 2)
        projected = self.short_filter(projected)[..., :length]
        *gates, values = projected.split(self.d_model, dim=1)
        kernels = self.long_filter.filter(length).squeeze(0).transpose(0, 1)
        kernels = kernels.reshape(self.order - 1, self.d_model, length)
        diagonal = self.long_filter.bias.reshape(self.order - 1, self.d_model)
        for index, gate in enumerate(reversed(gates[1:])):
            values = self.dropout(values * gate)
            values = _fft_convolution(values, kernels[index], diagonal[index])
        output = (values * gates[0]).transpose(1, 2)
        return self.out_projection(output)


class HyenaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        max_length: int = 4096,
        order: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = HyenaOperator(
            d_model=d_model,
            d_state=d_state,
            max_length=max_length,
            order=order,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.mixer(self.norm(inputs)))


class HyenaRegressor(nn.Module):
    """Hyena encoder with the same input, pooling, and linear head as baselines."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 22,
        d_state: int = 128,
        num_blocks: int = 1,
        dropout: float = 0.1,
        max_length: int = 4096,
        order: int = 2,
        pooling: str = "mean",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if pooling not in {"mean", "mean_max", "mean_max_terminal"}:
            raise ValueError("pooling must be mean, mean_max, or mean_max_terminal")
        self.vocab_size = vocab_size
        self.input_projection = nn.Linear(vocab_size, d_model, bias=False)
        self.blocks = nn.ModuleList(
            HyenaBlock(d_model, d_state, max_length, order, dropout)
            for _ in range(num_blocks)
        )
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.output_projection = nn.Linear(
            d_model * {"mean": 1, "mean_max": 2, "mean_max_terminal": 4}[pooling],
            1,
        )

    def _one_hot(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            if inputs.is_floating_point() or inputs.is_complex():
                raise TypeError("token IDs must use an integer dtype")
            return F.one_hot(inputs, num_classes=self.vocab_size).to(
                self.input_projection.weight.dtype
            )
        if inputs.ndim == 3 and inputs.size(-1) == self.vocab_size:
            return inputs.to(self.input_projection.weight.dtype)
        raise ValueError(
            "expected token IDs shaped (batch, length) or one-hot inputs shaped "
            f"(batch, length, {self.vocab_size}); got {tuple(inputs.shape)}"
        )

    @staticmethod
    def _reverse_valid(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1)
        positions = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0)
        indices = torch.where(
            positions < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - positions, positions
        )
        if inputs.ndim == 3:
            indices = indices.unsqueeze(-1).expand_as(inputs)
        return inputs.gather(1, indices)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self._one_hot(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self._encode(inputs)
        if self.bidirectional:
            if padding_mask is None:
                reverse = self._encode(inputs.flip(1)).flip(1)
            else:
                reverse_inputs = self._reverse_valid(inputs, padding_mask)
                reverse = self._reverse_valid(self._encode(reverse_inputs), padding_mask)
            hidden = 0.5 * (hidden + reverse)
        if padding_mask is None:
            mean = hidden.mean(dim=1)
            maximum = hidden.amax(dim=1)
            first, last = hidden[:, 0], hidden[:, -1]
        else:
            if padding_mask.shape != hidden.shape[:2]:
                raise ValueError("padding_mask must have shape (batch, length)")
            weights = padding_mask.to(hidden.dtype).unsqueeze(-1)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            maximum = hidden.masked_fill(~padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
            lengths = padding_mask.sum(dim=1).clamp_min(1)
            first = hidden[:, 0]
            last = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths - 1]
        pooled = mean
        if self.pooling == "mean_max":
            pooled = torch.cat((mean, maximum), dim=-1)
        elif self.pooling == "mean_max_terminal":
            pooled = torch.cat((mean, maximum, first, last), dim=-1)
        return self.output_projection(pooled).squeeze(-1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return matching counts, explicitly excluding Hyena positional tables."""

    parameters = tuple(model.parameters())
    positional = tuple(
        parameter
        for parameter in parameters
        if getattr(parameter, "_exclude_from_parameter_count", False)
    )
    total_raw = sum(parameter.numel() for parameter in parameters)
    trainable_raw = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    positional_total = sum(parameter.numel() for parameter in positional)
    positional_trainable = sum(
        parameter.numel() for parameter in positional if parameter.requires_grad
    )
    return {
        "total_parameters": total_raw - positional_total,
        "trainable_parameters": trainable_raw - positional_trainable,
        "positional_embedding_parameters_excluded": positional_total,
        "total_parameters_including_positional_embeddings": total_raw,
        "trainable_parameters_including_positional_embeddings": trainable_raw,
    }
