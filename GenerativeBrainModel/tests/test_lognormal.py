import torch
import torch.nn.functional as F


from GenerativeBrainModel.utils.lognormal import (
    LOGNORMAL_EPS,
    clamp_positive,
    softplus_std,
    lognormal_nll,
    lognormal_mean,
    lognormal_median,
)


def test_clamp_positive_prevents_zeros():
    x = torch.tensor([0.0, -1.0, 2.0])
    clamped = clamp_positive(x)
    assert torch.all(clamped >= LOGNORMAL_EPS)
    assert torch.isclose(clamped[-1], torch.tensor(2.0))


def test_softplus_std_positive():
    raw = torch.tensor([-10.0, 0.0, 10.0])
    std = softplus_std(raw)
    assert torch.all(std > 0.0)
    # For large positive raw, softplus approximates identity
    assert torch.isclose(std[-1], raw[-1], atol=1e-4, rtol=1e-4)


def test_lognormal_mean_matches_manual_computation():
    m = torch.tensor([[0.25, -0.5]])
    s_raw = torch.tensor([[0.1, 0.4]])
    mean = lognormal_mean(m, s_raw)
    s = softplus_std(s_raw)
    expected = torch.exp(m + 0.5 * s * s)
    assert torch.allclose(mean, expected, rtol=1e-5, atol=1e-6)


def test_lognormal_median_matches_manual():
    m = torch.tensor([0.2, -0.7])
    median = lognormal_median(m)
    expected = torch.exp(m)
    assert torch.allclose(median, expected, rtol=1e-6, atol=1e-7)


def test_lognormal_nll_reductions():
    y = torch.tensor([[0.2, 1.3], [2.0, 4.5]])
    m = torch.zeros_like(y)
    s_raw = torch.zeros_like(y)

    nll_mean = lognormal_nll(y, m, s_raw, reduction='mean')
    nll_sum = lognormal_nll(y, m, s_raw, reduction='sum')
    nll_none = lognormal_nll(y, m, s_raw, reduction='none')

    assert nll_none.shape == y.shape
    assert torch.isclose(nll_mean, nll_none.mean())
    assert torch.isclose(nll_sum, nll_none.sum())


def test_lognormal_nll_gradients_backward():
    torch.manual_seed(0)
    y = torch.rand(4, 3) + 0.05
    m = torch.randn(4, 3, requires_grad=True)
    s_raw = torch.randn(4, 3, requires_grad=True)

    loss = lognormal_nll(y, m, s_raw)
    loss.backward()

    assert m.grad is not None and torch.isfinite(m.grad).all()
    assert s_raw.grad is not None and torch.isfinite(s_raw.grad).all()



