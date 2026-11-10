"""Diagonal structured state-space (S4D) sequence mixer."""

import math

import torch
import torch.nn as nn


class SequenceDropout(nn.Module):
    """Drop whole features with a mask shared across sequence positions."""

    def __init__(self, probability: float = 0.0, featurewise: bool = False) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = probability
        self.featurewise = featurewise

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (
            (inputs.size(0), 1, inputs.size(2))
            if self.featurewise
            else (inputs.size(0), inputs.size(1), 1)
        )
        mask = torch.rand(shape, device=inputs.device) < keep_probability
        return inputs * mask / keep_probability


class S4DKernel(nn.Module):
    """Generate a convolution kernel from diagonal SSM parameters."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        learning_rate: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state <= 0 or d_state % 2:
            raise ValueError("d_state must be a positive even integer")
        if not 0.0 < dt_min < dt_max:
            raise ValueError("expected 0 < dt_min < dt_max")

        log_dt = torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max))
        coefficients = torch.randn(d_model, d_state // 2, dtype=torch.cfloat)
        log_a_real = torch.full((d_model, d_state // 2), math.log(0.5))
        a_imag = math.pi * torch.arange(d_state // 2).repeat(d_model, 1)

        self.coefficients = nn.Parameter(torch.view_as_real(coefficients))
        self._register_ssm_parameter("log_dt", log_dt, learning_rate)
        self._register_ssm_parameter("log_a_real", log_a_real, learning_rate)
        self._register_ssm_parameter("a_imag", a_imag, learning_rate)

    def _register_ssm_parameter(
        self, name: str, value: torch.Tensor, learning_rate: float | None
    ) -> None:
        if learning_rate == 0.0:
            self.register_buffer(name, value)
            return

        parameter = nn.Parameter(value)
        parameter._optim = {"weight_decay": 0.0}  # type: ignore[attr-defined]
        if learning_rate is not None:
            parameter._optim["lr"] = learning_rate  # type: ignore[attr-defined]
        self.register_parameter(name, parameter)

    def forward(self, length: int) -> torch.Tensor:
        if length <= 0:
            raise ValueError("sequence length must be positive")

        dt = self.log_dt.exp()
        coefficients = torch.view_as_complex(self.coefficients)
        a = -self.log_a_real.exp() + 1j * self.a_imag
        dt_a = a * dt.unsqueeze(-1)

        steps = torch.arange(length, device=a.device, dtype=dt.dtype)
        vandermonde = torch.exp(dt_a.unsqueeze(-1) * steps)
        coefficients = coefficients * torch.expm1(dt_a) / a
        return 2.0 * torch.einsum("hn,hnl->hl", coefficients, vandermonde).real


class S4D(nn.Module):
    """Causal long-range sequence mixer using FFT convolution.

    Inputs and outputs use shape ``(batch, length, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        featurewise_dropout: bool = False,
        **kernel_kwargs: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.skip = nn.Parameter(torch.randn(d_model))
        self.kernel = S4DKernel(d_model, d_state=d_state, **kernel_kwargs)
        self.activation = nn.GELU()
        self.dropout = SequenceDropout(dropout, featurewise=featurewise_dropout)
        self.output_projection = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.size(-1) != self.d_model:
            raise ValueError(
                f"expected (batch, length, {self.d_model}), got {tuple(inputs.shape)}"
            )

        values = inputs.transpose(1, 2)
        length = values.size(-1)
        kernel = self.kernel(length)

        fft_dtype = (
            torch.float32
            if values.dtype in {torch.float16, torch.bfloat16}
            else values.dtype
        )
        kernel_fft = torch.fft.rfft(kernel.to(fft_dtype), n=2 * length)
        values_fft = torch.fft.rfft(values.to(fft_dtype), n=2 * length)
        mixed = torch.fft.irfft(values_fft * kernel_fft, n=2 * length)[..., :length]
        mixed = mixed.to(values.dtype)
        mixed = mixed + values * self.skip.to(values.dtype).unsqueeze(-1)
        mixed = self.dropout(self.activation(mixed))
        return self.output_projection(mixed).transpose(1, 2)
