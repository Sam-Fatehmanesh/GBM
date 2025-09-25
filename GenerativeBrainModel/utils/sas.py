"""Utilities for the sinh–arcsinh (SAS) distribution on log-rates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SAS_EPS = 1e-6


def _positive(v: torch.Tensor) -> torch.Tensor:
    """Map unconstrained tensor to strictly positive via softplus."""
    return F.softplus(v) + SAS_EPS


def sas_nll(
    rates: torch.Tensor,
    mu: torch.Tensor,
    raw_log_sigma: torch.Tensor,
    eta: torch.Tensor,
    raw_log_delta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative log-likelihood for the SAS distribution on log-rates.

    Args:
        rates: Observed rates (non-negative), shape (...).
        mu: Location parameter for Y = log(rate + eps).
        raw_log_sigma: Unconstrained scale parameter. Mapped to positive via softplus.
        eta: Skew parameter.
        raw_log_delta: Unconstrained tail-weight parameter. Positive via softplus.
        eps: Numerical jitter added inside the log.

    Returns:
        Scalar tensor: mean negative log-likelihood over all elements.
    """

    sigma = _positive(raw_log_sigma)
    delta = _positive(raw_log_delta)

    log_rates = torch.log(rates.clamp_min(eps) + eps)
    z = (log_rates - mu) / sigma
    w = torch.asinh(z)
    t = eta + delta * w

    # log-density of SAS on Y = log(rate)
    log_pdf = (
        torch.log(delta)
        - torch.log(sigma)
        - 0.5 * torch.log(torch.tensor(2 * math.pi, device=rates.device, dtype=rates.dtype))
        - 0.5 * t.pow(2)
        - 0.5 * torch.log1p(z.pow(2))
    )

    # NLL on Y plus change of variables (already handled by z term)
    nll = -log_pdf
    return nll.mean()


def sas_log_median(mu: torch.Tensor, raw_log_sigma: torch.Tensor, eta: torch.Tensor, raw_log_delta: torch.Tensor) -> torch.Tensor:
    """Log-median of the SAS distribution on log-rates."""

    sigma = _positive(raw_log_sigma)
    delta = _positive(raw_log_delta)
    # Quantile at q=0.5 ⇒ Phi^{-1}(0.5) = 0
    log_median = mu + sigma * torch.sinh((-eta) / delta)
    return log_median


def sas_rate_median(mu: torch.Tensor, raw_log_sigma: torch.Tensor, eta: torch.Tensor, raw_log_delta: torch.Tensor) -> torch.Tensor:
    """Median in the rate domain."""

    log_med = sas_log_median(mu, raw_log_sigma, eta, raw_log_delta)
    return torch.exp(log_med)

