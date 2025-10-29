"""Utilities for the LogNormal distribution on rates.

Mirrors the API style in sas.py for convenience.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


LOGN_EPS = 1e-6


def _positive(v: torch.Tensor) -> torch.Tensor:
    """Map unconstrained tensor to strictly positive via softplus."""
    return F.softplus(v) + LOGN_EPS


def lognormal_nll(
    rates: torch.Tensor,
    mu: torch.Tensor,
    raw_log_sigma: torch.Tensor,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
    weight_by_rate: bool = False,
) -> torch.Tensor:
    """Negative log-likelihood for LogNormal on rate domain.

    Args:
        rates: Observed rates (non-negative), shape (...).
        mu: Location parameter for Y = log(rate + eps).
        raw_log_sigma: Unconstrained scale parameter. Mapped to positive via softplus.
        eps: Numerical jitter added inside the log.
        mask: Optional mask with same shape as rates; averages over masked elements.

    Returns:
        Scalar tensor: mean negative log-likelihood over elements (masked mean if mask provided).
    """

    sigma = _positive(raw_log_sigma)

    y = torch.log(rates.clamp_min(eps) + eps)
    z = (y - mu) / sigma

    log_two_pi = math.log(2.0 * math.pi)
    # LogNormal log-pdf on x with parameters (mu, sigma) for y = log x:
    # log f(x) = -0.5*((y-mu)^2/sigma^2) - log x - log sigma - 0.5*log(2*pi)
    log_pdf = (
        -0.5 * z.pow(2)
        - torch.log(sigma)
        - 0.5 * torch.as_tensor(log_two_pi, dtype=rates.dtype, device=rates.device)
        - y
    )

    nll = -log_pdf


    if weight_by_rate:
        y0 = torch.quantile(y.detach(), 0.5)  # median as a reference
        w = torch.exp(0.7*(y - y0)).clamp(max=20)
        nll = nll * w


    if mask is not None:
        mask = mask.to(dtype=nll.dtype, device=nll.device)
        if mask.shape != nll.shape:
            mask = mask.expand_as(nll)
        total = mask.sum().clamp_min(1.0)
        return (nll * mask).sum() / total


    return nll.mean()


def lognormal_log_median(mu: torch.Tensor, raw_log_sigma: torch.Tensor) -> torch.Tensor:
    """Log-median of the LogNormal (independent of sigma)."""
    return mu


def lognormal_rate_median(mu: torch.Tensor, raw_log_sigma: torch.Tensor) -> torch.Tensor:
    """Median in the rate domain for LogNormal."""
    return torch.exp(mu)


def lognormal_rate_mean(mu: torch.Tensor, raw_log_sigma: torch.Tensor) -> torch.Tensor:
    """Mean in the rate domain for LogNormal: E[X] = exp(mu + 0.5*sigma^2)."""
    sigma = _positive(raw_log_sigma)
    return torch.exp(mu + 0.5 * sigma.pow(2))


def sample_lognormal(mu: torch.Tensor, raw_log_sigma: torch.Tensor) -> torch.Tensor:
    """Sample from the LogNormal distribution, clamping to 2 stds on either side."""
    sigma = _positive(raw_log_sigma)#.clamp(min=0.0, max=1.0)
    eps = torch.randn_like(mu) # 86.4% of samples are within 1.5 stds
    return torch.exp(mu + sigma * eps)#.clamp(min=0.0, max=8.0)