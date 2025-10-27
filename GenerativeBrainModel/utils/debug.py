import os
import torch


def debug_enabled() -> bool:
    return os.environ.get('GBM_DEBUG_NAN', '0') == '1'


def assert_no_nan(t: torch.Tensor, tag: str) -> None:
    if not isinstance(t, torch.Tensor):
        return
    if t.numel() == 0:
        return
    finite = torch.isfinite(t)
    if not finite.all():
        bad = (~finite).nonzero(as_tuple=False)
        count = int(bad.shape[0])
        dtype = str(t.dtype)
        shape = tuple(t.shape)
        # Try to summarize
        with torch.no_grad():
            t_clamped = torch.nan_to_num(t.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mx = float(t_clamped.abs().max().item()) if t_clamped.numel() > 0 else float('nan')
        raise RuntimeError(f"NaN/Inf detected at {tag}: count={count}, shape={shape}, dtype={dtype}, max(abs(finite))~{mx}")











