"""Parameter-transparent models for the Lyra synthetic benchmark suite."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LyraBlock
from .pgc import PGC
from .s4d import S4D


class S4DResidualBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = S4D(d_model, d_state=d_state, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.mixer(self.norm(inputs)))


class PGCBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, residual: bool = False) -> None:
        super().__init__()
        self.narrow = PGC(d_model, expansion_factor=0.25, dropout=dropout)
        self.wide = PGC(d_model, expansion_factor=2.0, dropout=dropout)
        self.residual = residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.wide(self.narrow(inputs))
        return inputs + output if self.residual else output


class S4DThenPGCBlock(nn.Module):
    """Ordering ablation that reverses Lyra's PGC -> S4D composition."""

    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.s4d = S4DResidualBlock(d_model, d_state, dropout)
        self.pgc = PGCBlock(d_model, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pgc(self.s4d(inputs))


class TransformerResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, heads: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by transformer heads")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layer = layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


class MambaResidualBlock(nn.Module):
    """One official Mamba-2 or Mamba-3 SISO mixer with a residual path."""

    def __init__(
        self,
        version: str,
        d_model: int,
        d_state: int,
        dropout: float,
        expand: int,
        headdim: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2, Mamba3
        except ImportError as error:
            raise ImportError(
                "Mamba synthetic baselines require the official mamba-ssm package"
            ) from error
        if version == "mamba2":
            self.mixer = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand,
                headdim=headdim,
            )
        elif version == "mamba3":
            self.mixer = Mamba3(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                is_mimo=False,
                chunk_size=chunk_size,
                is_outproj_norm=False,
            )
        else:
            raise ValueError("version must be mamba2 or mamba3")
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.dropout(self.mixer(self.norm(inputs)))


class ImplicitHyenaFilter(nn.Module):
    """Small implicit long filter following Safari's standalone Hyena design."""

    def __init__(
        self,
        channels: int,
        length: int,
        hidden: int = 64,
        frequency: float = 1.0,
    ) -> None:
        super().__init__()
        self.length = length
        steps = torch.linspace(0.0, 1.0, length).view(1, length, 1)
        phase = 2.0 * math.pi * torch.arange(length).view(1, length, 1) / length
        embedding = torch.cat((steps, phase.sin(), phase.cos()), dim=-1)
        self.register_buffer("embedding", embedding)
        self.frequency = nn.Parameter(torch.full((1, 1, hidden), frequency))
        self.input_projection = nn.Linear(3, hidden)
        self.hidden_projection = nn.Linear(hidden, hidden)
        self.output_projection = nn.Linear(hidden, channels, bias=False)
        decay = torch.linspace(math.log(1e-2) / 1.5, math.log(1e-2) / 0.3, channels)
        self.register_buffer("decay", decay.view(1, 1, channels))

    def forward(self, length: int) -> torch.Tensor:
        values = self.input_projection(self.embedding[:, :length])
        values = torch.sin(self.frequency * values)
        values = torch.sin(self.hidden_projection(values))
        values = self.output_projection(values)
        time = self.embedding[:, :length, :1]
        return values * torch.exp(-time * self.decay.abs())


def _fft_convolution(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    length = values.size(-1)
    fft_dtype = (
        torch.float32
        if values.dtype in {torch.float16, torch.bfloat16}
        else values.dtype
    )
    kernel_fft = torch.fft.rfft(kernel.to(fft_dtype), n=2 * length) / (2 * length)
    values_fft = torch.fft.rfft(values.to(fft_dtype), n=2 * length)
    return torch.fft.irfft(
        values_fft * kernel_fft, n=2 * length, norm="forward"
    )[..., :length].to(values.dtype)


class HyenaResidualBlock(nn.Module):
    """Order-2 Hyena operator wrapped as a pre-normalized residual block."""

    def __init__(self, d_model: int, length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.RMSNorm(d_model)
        self.in_projection = nn.Linear(d_model, 3 * d_model)
        self.short_filter = nn.Conv1d(
            3 * d_model,
            3 * d_model,
            kernel_size=3,
            padding=2,
            groups=3 * d_model,
        )
        self.filter = ImplicitHyenaFilter(d_model, length)
        self.bias = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(1)
        projected = self.in_projection(self.norm(inputs)).transpose(1, 2)
        projected = self.short_filter(projected)[..., :length]
        gate_in, gate_out, values = projected.chunk(3, dim=1)
        values = self.dropout(values * gate_out)
        kernel = self.filter(length).squeeze(0).transpose(0, 1)
        values = _fft_convolution(values, kernel) + values * self.bias[:, None]
        mixed = (values * gate_in).transpose(1, 2)
        return inputs + self.dropout(self.out_projection(mixed))


class ContinuousSyntheticModel(nn.Module):
    """Shared continuous-input wrapper for scalar and full-sequence targets."""

    def __init__(
        self,
        *,
        kind: str,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        output_mode: str,
        d_model: int = 64,
        d_state: int = 64,
        depth: int = 1,
        dropout: float = 0.1,
        transformer_heads: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 32,
        mamba_chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if output_mode not in {"scalar", "sequence"}:
            raise ValueError("output_mode must be scalar or sequence")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.kind = kind
        self.output_mode = output_mode
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        self.position = (
            nn.Parameter(torch.randn(1, sequence_length, d_model) / math.sqrt(d_model))
            if kind == "transformer"
            else None
        )

        def block() -> nn.Module:
            if kind == "lyra":
                return LyraBlock(d_model, d_state=d_state, dropout=dropout)
            if kind == "s4d":
                return S4DResidualBlock(d_model, d_state, dropout)
            if kind == "pgc":
                return PGCBlock(d_model, dropout)
            if kind == "s4d_pgc":
                return S4DThenPGCBlock(d_model, d_state, dropout)
            if kind == "transformer":
                return TransformerResidualBlock(d_model, dropout, transformer_heads)
            if kind == "hyena":
                return HyenaResidualBlock(d_model, sequence_length, dropout)
            if kind in {"mamba2", "mamba3"}:
                return MambaResidualBlock(
                    kind,
                    d_model,
                    d_state,
                    dropout,
                    mamba_expand,
                    mamba_headdim,
                    mamba_chunk_size,
                )
            raise ValueError(f"unknown model kind: {kind}")

        self.blocks = nn.ModuleList(block() for _ in range(depth))
        if output_mode == "scalar":
            self.spatial_pool = nn.Linear(sequence_length, 1, bias=False)
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.spatial_pool = None
            self.output_projection = nn.Linear(d_model, output_dim)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.size(1) != self.sequence_length:
            raise ValueError(
                f"expected (batch, {self.sequence_length}, input_dim), got "
                f"{tuple(inputs.shape)}"
            )
        hidden = self.input_projection(inputs)
        if self.position is not None:
            hidden = hidden + self.position
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_features(inputs)
        if self.output_mode == "scalar":
            hidden = self.spatial_pool(hidden.transpose(1, 2)).squeeze(-1)
        return self.output_projection(hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
