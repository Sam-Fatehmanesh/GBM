# This file intentionally left as alias during migration to SAS tests.

import torch

from GenerativeBrainModel.utils.sas import (
    SAS_EPS,
    sas_nll,
    sas_rate_median,
    sas_log_median,
)


def test_sas_rate_median_positive():
    mu = torch.tensor([[0.0, 0.5]])
    log_sigma = torch.tensor([[0.1, -0.2]])
    eta = torch.zeros_like(mu)
    log_delta = torch.zeros_like(mu)
    median = sas_rate_median(mu, log_sigma, eta, log_delta)
    assert torch.all(median > 0)


def test_sas_log_median_reduces_to_mu_when_eta_zero():
    mu = torch.tensor([[0.3, -0.4]])
    log_sigma = torch.randn_like(mu)
    eta = torch.zeros_like(mu)
    log_delta = torch.zeros_like(mu)
    log_med = sas_log_median(mu, log_sigma, eta, log_delta)
    assert torch.allclose(log_med, mu, atol=1e-6)


def test_sas_nll_backprop():
    torch.manual_seed(0)
    rates = torch.rand(4, 3) + 0.05
    mu = torch.randn(4, 3, requires_grad=True)
    log_sigma = torch.randn(4, 3, requires_grad=True)
    eta = torch.randn(4, 3, requires_grad=True)
    log_delta = torch.randn(4, 3, requires_grad=True)

    loss = sas_nll(rates, mu, log_sigma, eta, log_delta)
    loss.backward()

    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert log_sigma.grad is not None and torch.isfinite(log_sigma.grad).all()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()
    assert log_delta.grad is not None and torch.isfinite(log_delta.grad).all()



