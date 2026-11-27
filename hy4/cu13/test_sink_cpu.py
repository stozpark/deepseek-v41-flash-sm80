# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from sink_math import apply_attention_sink


@pytest.mark.parametrize("heads", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_virtual_zero_value_sink(heads, dtype):
    torch.manual_seed(17)
    score = torch.randn(5, heads, 17) * 5
    value = torch.randn(5, heads, 17, 32)
    score[0] = -torch.inf
    score[1, :, 3] = -torch.inf
    sinks = torch.linspace(-20., 20., heads)
    lse = score.logsumexp(-1)
    raw = (score.softmax(-1).nan_to_num(0).unsqueeze(-1) * value).sum(-2).to(dtype)
    augmented = torch.cat((score, sinks.expand(5, heads).unsqueeze(-1)), -1)
    expected = (augmented.softmax(-1)[..., :-1].unsqueeze(-1) * value).sum(-2)
    out, total = apply_attention_sink(raw, lse, sinks)
    tol = 0.012 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(total, augmented.logsumexp(-1), atol=2e-6, rtol=2e-6)
    assert out[0].count_nonzero() == 0
    out2, no_lse = apply_attention_sink(raw, lse, sinks, return_lse=False)
    assert no_lse is None and torch.equal(out, out2)


def test_extreme_and_shape():
    out = torch.ones(2, 3, 8)
    lse = torch.tensor([[1000., -1000., 0.], [-torch.inf, 1000., -1000.]])
    sinks = torch.tensor([-1000., 1000., 0.])
    expected = torch.tensor([[1., 0., 0.5], [0., 0.5, 0.]])
    actual, _ = apply_attention_sink(out, lse, sinks)
    torch.testing.assert_close(actual[..., 0], expected)
    with pytest.raises(ValueError):
        apply_attention_sink(out, lse, torch.zeros(4))
    with pytest.raises(ValueError):
        apply_attention_sink(out, lse, sinks.bfloat16())
