"""Minimal Lyra model with explicit one-hot vocabulary inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pgc import PGC
from .s4d import S4D


class LyraBlock(nn.Module):
    """Two local PGC mixers followed by a residual S4D mixer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        causal_pgc: bool = False,
        pgc_residual: bool = False,
        featurewise_dropout: bool = False,
        **kernel_kwargs: float,
    ) -> None:
        super().__init__()
        self.narrow_pgc = PGC(
            d_model, expansion_factor=0.25, dropout=dropout, causal=causal_pgc
        )
        self.wide_pgc = PGC(
            d_model, expansion_factor=2.0, dropout=dropout, causal=causal_pgc
        )
        self.norm = nn.RMSNorm(d_model)
        self.s4d = S4D(
            d_model,
            d_state=d_state,
            dropout=dropout,
            featurewise_dropout=featurewise_dropout,
            **kernel_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.pgc_residual = pgc_residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed = self.wide_pgc(self.narrow_pgc(inputs))
        if self.pgc_residual:
            mixed = inputs + mixed
        return mixed + self.dropout(self.s4d(self.norm(mixed)))


class LyraEncoder(nn.Module):
    """Lyra sequence encoder using explicit one-hot inputs.

    Args:
        vocab_size: Number of input vocabulary symbols. Protein models default
            to 22 channels.
        d_model: Hidden feature dimension.
        d_state: S4D state dimension; must be even.
        num_blocks: Number of stacked Lyra blocks.
        dropout: Dropout probability throughout the mixer.
        causal_pgc: Use left-padded causal PGC convolutions when true.

    Input:
        Integer IDs of shape ``(batch, length)`` or one-hot values of shape
        ``(batch, length, vocab_size)``.

    Output:
        Per-position features of shape ``(batch, length, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 22,
        d_state: int = 64,
        num_blocks: int = 1,
        dropout: float = 0.1,
        causal_pgc: bool = False,
        pgc_residual: bool = False,
        featurewise_dropout: bool = False,
        **kernel_kwargs: float,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.vocab_size = vocab_size
        self.input_projection = nn.Linear(vocab_size, d_model, bias=False)
        self.blocks = nn.ModuleList(
            LyraBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                causal_pgc=causal_pgc,
                pgc_residual=pgc_residual,
                featurewise_dropout=featurewise_dropout,
                **kernel_kwargs,
            )
            for _ in range(num_blocks)
        )

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self._one_hot(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class Lyra(nn.Module):
    """Lyra token model with a per-position vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_state: int = 64,
        num_blocks: int = 1,
        dropout: float = 0.1,
        causal_pgc: bool = False,
        pgc_residual: bool = False,
        featurewise_dropout: bool = False,
        **kernel_kwargs: float,
    ) -> None:
        super().__init__()
        self.encoder = LyraEncoder(
            d_model=d_model,
            vocab_size=vocab_size,
            d_state=d_state,
            num_blocks=num_blocks,
            dropout=dropout,
            causal_pgc=causal_pgc,
            pgc_residual=pgc_residual,
            featurewise_dropout=featurewise_dropout,
            **kernel_kwargs,
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.forward_features(inputs))


class LyraRegressor(nn.Module):
    """Mean-pooled Lyra with a single linear regression projection.

    The computation is exactly ``encoder(inputs).mean(dim=1)`` followed by one
    ``nn.Linear(d_model, 1)`` when no padding mask is supplied. A mask is only
    needed for variable-length padded batches.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 22,
        d_state: int = 64,
        num_blocks: int = 1,
        dropout: float = 0.1,
        causal_pgc: bool = False,
        pgc_residual: bool = False,
        featurewise_dropout: bool = False,
        pooling: str = "mean",
        bidirectional: bool = False,
        **kernel_kwargs: float,
    ) -> None:
        super().__init__()
        self.encoder = LyraEncoder(
            d_model=d_model,
            vocab_size=vocab_size,
            d_state=d_state,
            num_blocks=num_blocks,
            dropout=dropout,
            causal_pgc=causal_pgc,
            pgc_residual=pgc_residual,
            featurewise_dropout=featurewise_dropout,
            **kernel_kwargs,
        )
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
        reverse_positions = torch.where(
            positions < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - positions, positions
        )
        if inputs.ndim == 3:
            reverse_positions = reverse_positions.unsqueeze(-1).expand_as(inputs)
        return inputs.gather(1, reverse_positions)

    def _features(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        forward = self.encoder(inputs)
        if not self.bidirectional:
            return forward
        if padding_mask is None:
            reverse_inputs = inputs.flip(1)
            reverse = self.encoder(reverse_inputs).flip(1)
        else:
            reverse_inputs = self._reverse_valid(inputs, padding_mask)
            reverse = self.encoder(reverse_inputs)
            reverse = self._reverse_valid(reverse, padding_mask)
        return 0.5 * (forward + reverse)

    def _pool(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if padding_mask is None:
            mean = features.mean(dim=1)
            maximum = features.amax(dim=1)
            first, last = features[:, 0], features[:, -1]
        else:
            weights = padding_mask.to(features.dtype).unsqueeze(-1)
            mean = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            maximum = features.masked_fill(~padding_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
            lengths = padding_mask.sum(dim=1).clamp_min(1)
            first = features[:, 0]
            last = features[torch.arange(features.size(0), device=features.device), lengths - 1]
        if self.pooling == "mean":
            return mean
        if self.pooling == "mean_max":
            return torch.cat((mean, maximum), dim=-1)
        return torch.cat((mean, maximum, first, last), dim=-1)

    def forward(
        self, inputs: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if padding_mask is not None:
            if padding_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape (batch, length); got "
                    f"{tuple(padding_mask.shape)}"
                )
        sequence_features = self._features(inputs, padding_mask)
        pooled = self._pool(sequence_features, padding_mask)
        return self.output_projection(pooled).squeeze(-1)
