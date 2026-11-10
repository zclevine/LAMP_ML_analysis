"""Thin regression wrappers around the official Mamba-2 and Mamba-3 layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaRegressor(nn.Module):
    """One Mamba layer followed by mean pooling and one linear projection."""

    def __init__(
        self,
        version: str,
        d_model: int,
        vocab_size: int = 22,
        d_state: int = 64,
        dropout: float = 0.1,
        expand: int = 2,
        headdim: int = 64,
        pooling: str = "mean",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if version not in {"mamba2", "mamba3"}:
            raise ValueError("version must be 'mamba2' or 'mamba3'")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")

        try:
            from mamba_ssm import Mamba2, Mamba3
        except ImportError as error:
            raise ImportError(
                "Mamba baselines require the official mamba-ssm source package; "
                "install the 'mamba' optional dependency"
            ) from error

        mixer_class = Mamba2 if version == "mamba2" else Mamba3
        self.version = version
        self.vocab_size = vocab_size
        self.input_projection = nn.Linear(vocab_size, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)
        self.mixer = mixer_class(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)
        if pooling not in {"mean", "mean_max", "mean_max_terminal"}:
            raise ValueError("pooling must be mean, mean_max, or mean_max_terminal")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.output_projection = nn.Linear(
            d_model * {"mean": 1, "mean_max": 2, "mean_max_terminal": 4}[pooling], 1
        )

    @staticmethod
    def _reverse_valid(inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        lengths = padding_mask.sum(dim=1)
        positions = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0)
        indices = torch.where(
            positions < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - positions, positions
        )
        if inputs.ndim == 3:
            indices = indices.unsqueeze(-1).expand_as(inputs)
        return inputs.gather(1, indices)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self._one_hot(inputs))
        return hidden + self.dropout(self.mixer(self.norm(hidden)))

    def _one_hot(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            if inputs.is_floating_point() or inputs.is_complex():
                raise TypeError("token IDs must use an integer dtype")
            return F.one_hot(inputs, num_classes=self.vocab_size).to(
                dtype=self.input_projection.weight.dtype
            )
        if inputs.ndim == 3 and inputs.size(-1) == self.vocab_size:
            return inputs.to(dtype=self.input_projection.weight.dtype)
        raise ValueError(
            "expected token IDs shaped (batch, length) or one-hot inputs shaped "
            f"(batch, length, {self.vocab_size}); got {tuple(inputs.shape)}"
        )

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
                raise ValueError(
                    "padding_mask must have shape (batch, length); got "
                    f"{tuple(padding_mask.shape)}"
                )
            weights = padding_mask.to(hidden.dtype).unsqueeze(-1)
            mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
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
