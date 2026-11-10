"""Parallel Gated Convolution for local sequence mixing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGC(nn.Module):
    """Depthwise local convolution modulated by a learned gate.

    Inputs and outputs use shape ``(batch, length, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.0,
        dropout: float = 0.0,
        causal: bool = False,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        if kernel_size <= 0 or (not causal and kernel_size % 2 == 0):
            raise ValueError("kernel_size must be positive and odd when non-causal")

        hidden_dim = int(d_model * expansion_factor)
        if hidden_dim < 1:
            raise ValueError("d_model * expansion_factor must be at least 1")

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.kernel_size = kernel_size

        self.in_proj = nn.Linear(d_model, 2 * hidden_dim)
        self.in_norm = nn.RMSNorm(2 * hidden_dim, eps=1e-8)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            groups=hidden_dim,
            padding=0,
        )
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.out_norm = nn.RMSNorm(d_model, eps=1e-8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.size(-1) != self.d_model:
            raise ValueError(
                f"expected (batch, length, {self.d_model}), got {tuple(inputs.shape)}"
            )

        projected = self.in_norm(self.in_proj(inputs))
        values, gates = projected.chunk(2, dim=-1)

        values = values.transpose(1, 2)
        if self.causal:
            values = F.pad(values, (self.kernel_size - 1, 0))
        else:
            half_width = self.kernel_size // 2
            values = F.pad(values, (half_width, half_width))
        values = self.conv(values).transpose(1, 2)

        mixed = values * gates
        return self.dropout(self.out_norm(self.out_proj(mixed)))
