LOGNORMAL_EPS = 1e-6


import torch
import torch.nn.functional as F


def clamp_positive(y: torch.Tensor, eps: float = LOGNORMAL_EPS) -> torch.Tensor:
    """Clamp target rates to avoid log(0)."""
    return y.clamp_min(eps)


def softplus_std(s_raw: torch.Tensor, eps: float = LOGNORMAL_EPS) -> torch.Tensor:
    """Map raw scale logits to positive standard deviation of log-rate."""
    return F.softplus(s_raw) + eps


def lognormal_nll(
    y: torch.Tensor,
    m_raw: torch.Tensor,
    s_raw: torch.Tensor,
    *,
    eps: float = LOGNORMAL_EPS,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Negative log-likelihood of heteroskedastic lognormal targets."""
    y = clamp_positive(y, eps)
    s = softplus_std(s_raw, eps)
    z = (y.log() - m_raw) / s
    nll = 0.5 * z * z + y.log() + s.log()
    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


def lognormal_mean(m_raw: torch.Tensor, s_raw: torch.Tensor, eps: float = LOGNORMAL_EPS) -> torch.Tensor:
    """Return the predictive mean rate implied by lognormal parameters."""
    s = softplus_std(s_raw, eps).float()
    mean = torch.exp(m_raw.float() + 0.5 * s * s)
    return mean.to(dtype=m_raw.dtype)


def lognormal_median(m_raw: torch.Tensor, eps: float = LOGNORMAL_EPS) -> torch.Tensor:
    """Return the predictive median rate."""
    median = torch.exp(m_raw.float())
    return median.to(dtype=m_raw.dtype)
