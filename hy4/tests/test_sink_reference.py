# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the exact sink helper shipped by the overlay."""
import importlib.util
from pathlib import Path

import pytest
import torch

p = Path(__file__).resolve().parents[1] / "overlay" / "sm80_sink_math.py"
spec = importlib.util.spec_from_file_location("hy4_sink_math", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
apply_sink = mod.apply_sink


@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sink_explicit_zero_value_reference(heads, dtype):
    g = torch.Generator().manual_seed(19)
    scores = torch.randn(5, heads, 11, generator=g) * 5
    values = torch.randn(5, heads, 11, 16, generator=g)
    sinks = torch.linspace(-15, 15, heads)
    # Include an all-masked row, an interior mask, and normal rows.
    scores[0] = -torch.inf
    scores[1, :, 3] = -torch.inf
    lse = torch.logsumexp(scores, dim=-1)
    real_probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    real_out = (real_probs.unsqueeze(-1) * values).sum(-2).to(dtype)
    with_sink = torch.cat([scores, sinks.expand(5, heads).unsqueeze(-1)], -1)
    p_real = torch.softmax(with_sink, dim=-1)[..., :-1]
    expected = (p_real.unsqueeze(-1) * values).sum(-2)
    actual = apply_sink(real_out, lse, sinks)
    tol = 0.012 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual.float(), expected, atol=tol, rtol=tol)
    assert actual[0].count_nonzero() == 0
    assert torch.isfinite(actual).all()


def test_extreme_sink_and_real_lse():
    out = torch.ones(2, 3, 8)
    lse = torch.tensor([[1000., -1000., 0.], [-torch.inf, 1000., -1000.]])
    sinks = torch.tensor([-1000., 1000., 0.])
    actual = apply_sink(out, lse, sinks)
    expected = torch.tensor([[1., 0., 0.5], [0., 0.5, 0.]])
    torch.testing.assert_close(actual[..., 0], expected)


def test_reject_nonlocal_or_wrong_dtype():
    with pytest.raises(ValueError):
        apply_sink(torch.ones(2, 4, 8), torch.zeros(2, 4), torch.zeros(8))
    with pytest.raises(ValueError):
        apply_sink(torch.ones(2, 4, 8), torch.zeros(2, 4), torch.zeros(4).bfloat16())
