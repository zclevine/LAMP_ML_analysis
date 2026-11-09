"""Inference-only speedups that do not change a single predicted number.

## The S4D kernel recompute

``S4DKernel.forward(L)`` materializes the SSM convolution kernel from scratch on
**every** forward pass::

    dtA = A * dt.unsqueeze(-1)                      # (H, N/2)
    K   = dtA.unsqueeze(-1) * torch.arange(L)       # (H, N/2, L)   <- allocates
    K   = 2 * einsum('hn,hnl->hl', C, torch.exp(K)).real

With ``d_model=64`` and ``d_state=64`` that is a ``(64, 32, L)`` complex
exponential — ~2M complex ``exp`` calls at ``L=1000`` — and ``S4D.forward`` then
takes ``rfft(k, n=2L)`` of the result. Both depend only on the module's
parameters and on ``L``, never on the input batch.

In training that recompute is mandatory: the parameters move every step and the
kernel has to be rebuilt to carry gradients. At inference the parameters are
frozen, so for a fixed ``L`` the kernel is a constant being rebuilt once per
batch, per S4D module, per seed. The shipped ensemble has 5 seeds x 2 S4D
modules (forward and backward lanes), so one design run pays for it 10 times per
batch.

This module caches ``k_f = rfft(kernel(L), n=2L)`` keyed by ``(L, device,
dtype)``. The arithmetic is bit-for-bit what the vendored code computes — the
same tensor, reused instead of rebuilt — so predictions are unchanged. A test
asserts that equality rather than taking it on trust.

## Why it is patched in rather than edited into the model

``models/lyra/`` is a verbatim copy of the training-time architecture, and its
value is that it stays verbatim: it is the thing parity is defined against. So
the cache is installed onto module *instances* here instead of being edited into
the vendored source.

## Safety

The cached forward falls back to the original, uncached path whenever
``self.training`` or ``torch.is_grad_enabled()``, and clears the cache on the
way through. Caching a kernel that still has an autograd graph attached, or one
built from parameters that are about to be updated, would be wrong; this keeps
both cases on the stock path. The cache is therefore only ever live under
``model.eval()`` + ``torch.no_grad()``, which is exactly how
:class:`~lamp_design.models.inference.LyraEnsemble` runs.

It does assume frozen parameters. If you mutate a loaded model's weights in
place without calling ``.train()``, call :func:`clear_kernel_cache` afterwards.
"""

from __future__ import annotations

import types
from typing import Dict, Tuple

import torch
import torch.nn as nn

from lamp_design.models.lyra.s4d import S4D

#: Distinct ``(L, device, dtype)`` kernels held per S4D module. Deterministic
#: geometries produce few distinct lengths, so this is generous; the bound only
#: exists so a pathological input cannot grow the cache without limit.
MAX_CACHED_KERNELS = 256

_CacheKey = Tuple[int, str, torch.dtype]


def _cached_s4d_forward(self: S4D, u: torch.Tensor, **kwargs) -> torch.Tensor:
    """``S4D.forward`` with the kernel FFT memoized. See module docstring.

    Mirrors the vendored implementation; the only change is that ``k_f`` may
    come from the cache. Anything that is not frozen-parameter inference is
    delegated straight back to the vendored method, so that path cannot drift
    from the original.
    """
    if self.training or torch.is_grad_enabled():
        self._lamp_kernel_cache.clear()
        return self._lamp_stock_forward(u, **kwargs)

    if not self.transposed:
        u = u.transpose(-1, -2)
    L = u.size(-1)

    key: _CacheKey = (L, str(u.device), u.dtype)
    k_f = self._lamp_kernel_cache.get(key)
    if k_f is None:
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L).detach()
        if len(self._lamp_kernel_cache) >= MAX_CACHED_KERNELS:
            self._lamp_kernel_cache.clear()
        self._lamp_kernel_cache[key] = k_f

    u_f = torch.fft.rfft(u, n=2 * L)
    y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]

    y = y + u * self.D.unsqueeze(-1)

    y = self.dropout(self.activation(y))
    y = self.output_linear(y)
    if not self.transposed:
        y = y.transpose(-1, -2)
    return y


def enable_kernel_cache(model: nn.Module) -> int:
    """Install the kernel cache on every :class:`S4D` module under ``model``.

    Idempotent. Returns the number of modules patched (0 if already enabled),
    so a caller can assert the optimization actually found something -- a silent
    0 would mean the architecture changed and the speedup is not being applied.
    """
    patched = 0
    for module in model.modules():
        if not isinstance(module, S4D):
            continue
        if getattr(module, "_lamp_kernel_cache", None) is not None:
            continue
        # Plain dict attribute: nn.Module.__setattr__ only intercepts
        # Parameters/Modules/Tensors, so this stays out of the state_dict.
        module._lamp_kernel_cache: Dict[_CacheKey, torch.Tensor] = {}
        module._lamp_stock_forward = module.forward
        module.forward = types.MethodType(_cached_s4d_forward, module)
        patched += 1
    return patched


def clear_kernel_cache(model: nn.Module) -> None:
    """Drop every cached kernel, leaving the cache enabled.

    Call this after mutating a loaded model's parameters in place.
    """
    for module in model.modules():
        cache = getattr(module, "_lamp_kernel_cache", None)
        if cache is not None:
            cache.clear()


def disable_kernel_cache(model: nn.Module) -> int:
    """Restore the stock ``S4D.forward``. Returns the number of modules restored."""
    restored = 0
    for module in model.modules():
        if getattr(module, "_lamp_kernel_cache", None) is None:
            continue
        del module._lamp_kernel_cache
        module.__dict__.pop("_lamp_stock_forward", None)
        # Dropping the instance attribute puts class-level S4D.forward back in play.
        module.__dict__.pop("forward", None)
        restored += 1
    return restored


def cached_kernel_count(model: nn.Module) -> int:
    """Total cached kernels across the model. For tests and diagnostics."""
    return sum(
        len(cache)
        for module in model.modules()
        if (cache := getattr(module, "_lamp_kernel_cache", None)) is not None
    )


__all__ = [
    "MAX_CACHED_KERNELS",
    "cached_kernel_count",
    "clear_kernel_cache",
    "disable_kernel_cache",
    "enable_kernel_cache",
]
