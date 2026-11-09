"""The kernel cache must be an optimization only -- never a numerical change.

The whole value of :mod:`lamp_design.models.fastpath` rests on one claim: that
``rfft(S4DKernel(L))`` is a pure function of the frozen parameters and ``L``, so
memoizing it changes nothing. These tests check that claim directly against the
stock vendored path rather than assuming it.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch

from _weights import WEIGHTS_MISSING_REASON, weights_or_none
from lamp_design.models.fastpath import (
    cached_kernel_count,
    clear_kernel_cache,
    disable_kernel_cache,
    enable_kernel_cache,
)
from lamp_design.models.lyra.lyra_base import Lyra
from lamp_design.models.lyra.s4d import S4D


def _model(seed: int = 0) -> Lyra:
    torch.manual_seed(seed)
    model = Lyra(d_input=19, d_output=1, d_model=64, dropout=0.2, ssm="s4d", bidirectional=True)
    model.eval()
    return model


def _batch(n: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, length, 19, generator=g)


def test_finds_the_s4d_modules():
    """A silent 0 would mean the speedup is not being applied at all."""
    model = _model()
    n = enable_kernel_cache(model)
    # bidirectional => forward and backward lanes, one S4D layer each
    assert n == 2, f"expected 2 S4D modules, patched {n}"
    assert n == sum(isinstance(m, S4D) for m in model.modules())


def test_enable_is_idempotent():
    model = _model()
    assert enable_kernel_cache(model) == 2
    assert enable_kernel_cache(model) == 0, "second call must not re-patch"


def test_predictions_are_bit_identical():
    """The point of the whole module."""
    model = _model()
    x = _batch(8, 180)
    with torch.no_grad():
        stock = model(x).clone()

    assert enable_kernel_cache(model) == 2
    with torch.no_grad():
        first = model(x).clone()   # populates the cache
        second = model(x).clone()  # reads it

    assert torch.equal(stock, first), "first cached pass differs from stock"
    assert torch.equal(stock, second), "cache reuse changed the output"


def test_identical_across_several_lengths_and_batch_sizes():
    """One cache entry per length; batch size must not leak into it."""
    model_a, model_b = _model(), _model()
    enable_kernel_cache(model_b)

    for length in (64, 101, 180, 181):
        for n in (1, 3, 16):
            x = _batch(n, length, seed=length + n)
            with torch.no_grad():
                assert torch.equal(model_a(x), model_b(x)), f"L={length} n={n}"

    # 4 distinct lengths x 2 S4D modules
    assert cached_kernel_count(model_b) == 8


def test_cache_is_bypassed_when_grad_is_enabled():
    """A cached kernel must never carry a stale autograd graph."""
    model = _model()
    enable_kernel_cache(model)
    x = _batch(4, 120)

    with torch.no_grad():
        model(x)
    assert cached_kernel_count(model) > 0

    # Grad enabled: must clear and take the stock path.
    out = model(x)
    assert out.requires_grad
    assert cached_kernel_count(model) == 0, "cache survived a grad-enabled pass"
    out.sum().backward()
    assert model.s4ds[0].kernel.C.grad is not None, "gradients did not reach the kernel"


def test_cache_is_bypassed_in_training_mode():
    model = _model()
    enable_kernel_cache(model)
    with torch.no_grad():
        model(_batch(4, 120))
    assert cached_kernel_count(model) > 0

    model.train()
    with torch.no_grad():
        model(_batch(4, 120))
    assert cached_kernel_count(model) == 0, "cached a kernel while training"


def test_clear_keeps_the_cache_enabled():
    model = _model()
    enable_kernel_cache(model)
    x = _batch(4, 120)
    with torch.no_grad():
        model(x)
    assert cached_kernel_count(model) > 0

    clear_kernel_cache(model)
    assert cached_kernel_count(model) == 0
    with torch.no_grad():
        model(x)
    assert cached_kernel_count(model) > 0, "clear() disabled the cache instead of emptying it"


def test_disable_restores_stock_forward():
    model = _model()
    enable_kernel_cache(model)
    with torch.no_grad():
        model(_batch(4, 120))

    assert disable_kernel_cache(model) == 2
    assert cached_kernel_count(model) == 0
    for module in model.modules():
        assert not hasattr(module, "_lamp_kernel_cache")

    reference = _model()
    x = _batch(4, 120)
    with torch.no_grad():
        assert torch.equal(reference(x), model(x))


def test_cache_does_not_pollute_state_dict():
    """Patched instances must still round-trip through ``load_state_dict``."""
    model = _model()
    keys_before = set(model.state_dict())
    enable_kernel_cache(model)
    with torch.no_grad():
        model(_batch(2, 90))
    assert set(model.state_dict()) == keys_before

    fresh = _model(seed=7)
    fresh.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.skipif(weights_or_none() is None, reason=WEIGHTS_MISSING_REASON)
def test_real_ensemble_predictions_unchanged():
    """End-to-end on the shipped checkpoints: same Tp, to the bit."""
    from lamp_design.models.inference import LyraEnsemble

    weights = weights_or_none()
    rng = np.random.default_rng(3)
    rows = [rng.random((L, 19)).astype(np.float32) for L in (150, 150, 167, 180, 180, 180)]

    plain = LyraEnsemble.load(weights, cache_kernels=False)
    cached = LyraEnsemble.load(weights, cache_kernels=True)
    assert cached_kernel_count(plain.members[0].model) == 0
    assert all(
        hasattr(m, "_lamp_kernel_cache")
        for member in cached.members
        for m in member.model.modules()
        if isinstance(m, S4D)
    )

    mean_a, std_a = plain.predict(rows)
    mean_b, std_b = cached.predict(rows)
    assert np.array_equal(mean_a, mean_b), "ensemble means changed"
    assert np.array_equal(std_a, std_b), "ensemble spreads changed"
