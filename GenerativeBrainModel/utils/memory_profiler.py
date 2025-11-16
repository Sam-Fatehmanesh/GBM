from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil

    _PROCESS = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil optional
    psutil = None
    _PROCESS = None

import torch


def _bytes_to_human(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _cpu_rss_bytes() -> int:
    if _PROCESS is not None:
        try:
            return int(_PROCESS.memory_info().rss)
        except Exception:  # pragma: no cover
            pass
    try:
        import resource

        # ru_maxrss is kilobytes on Linux
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # pragma: no cover
        return 0


def _cuda_mem_stats(device: torch.device) -> tuple[Optional[int], Optional[int], Optional[int]]:
    if device.type != "cuda":
        return None, None, None
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
        int(torch.cuda.max_memory_allocated(device)),
    )


def _nvidia_smi_memory_bytes(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"-i",
                str(index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=False,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        used_mb = float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return int(used_mb * 1024 * 1024)


@dataclass
class MemorySnapshot:
    timestamp: float
    label: str
    step: Optional[int]
    cpu_rss: int
    cuda_allocated: Optional[int]
    cuda_reserved: Optional[int]
    cuda_max_allocated: Optional[int]
    nvidia_used: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "ts": self.timestamp,
            "label": self.label,
            "step": self.step,
            "cpu_rss": self.cpu_rss,
            "cpu_rss_h": _bytes_to_human(self.cpu_rss),
        }
        if self.cuda_allocated is not None:
            data.update(
                {
                    "cuda_allocated": self.cuda_allocated,
                    "cuda_reserved": self.cuda_reserved,
                    "cuda_max_allocated": self.cuda_max_allocated,
                    "cuda_allocated_h": _bytes_to_human(self.cuda_allocated),
                    "cuda_reserved_h": _bytes_to_human(self.cuda_reserved or 0),
                    "cuda_max_allocated_h": _bytes_to_human(self.cuda_max_allocated or 0),
                }
            )
        if self.nvidia_used is not None:
            data.update(
                {
                    "nvidia_used": self.nvidia_used,
                    "nvidia_used_h": _bytes_to_human(self.nvidia_used),
                }
            )
        if self.extra:
            data["extra"] = self.extra
        return data


class MemoryProfiler:
    """Collects lightweight CPU/GPU memory snapshots and optional static estimates."""

    def __init__(
        self,
        enabled: bool,
        device: torch.device,
        log_path: Optional[Path] = None,
        log_interval: int = 50,
        verbose: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_interval = max(1, int(log_interval))
        self.verbose = verbose
        self.snapshots: list[MemorySnapshot] = []
        self._last_logged_step: Optional[int] = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def should_log(self, step: Optional[int]) -> bool:
        if not self.enabled:
            return False
        if step is None:
            return True
        if self._last_logged_step is None:
            return True
        return (step - self._last_logged_step) >= self.log_interval

    def snapshot(self, label: str, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        cpu_rss = _cpu_rss_bytes()
        cuda_alloc, cuda_res, cuda_max = _cuda_mem_stats(self.device)
        nvidia_used = _nvidia_smi_memory_bytes(self.device)
        snap = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            step=step,
            cpu_rss=cpu_rss,
            cuda_allocated=cuda_alloc,
            cuda_reserved=cuda_res,
            cuda_max_allocated=cuda_max,
            nvidia_used=nvidia_used,
            extra=extra or {},
        )
        self.snapshots.append(snap)
        self._last_logged_step = step
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(snap.to_dict()) + "\n")
        if self.verbose:
            summary = snap.to_dict()
            extra_str = f" extra={summary.get('extra')}" if summary.get("extra") else ""
            nvidia_str = (
                f" nvidia={summary.get('nvidia_used_h')}" if summary.get("nvidia_used_h") else ""
            )
            cuda_alloc = summary.get("cuda_allocated_h", "n/a")
            cuda_reserved = summary.get("cuda_reserved_h", "n/a")
            print(  # pragma: no cover - console output
                f"[mem] {label} step={step} cpu={summary['cpu_rss_h']} "
                f"cuda_alloc={cuda_alloc} cuda_reserved={cuda_reserved}{nvidia_str}{extra_str}"
            )

    def summary(self) -> Dict[str, Any]:
        if not self.snapshots:
            return {}
        cpu_vals = [snap.cpu_rss for snap in self.snapshots]
        cuda_vals = [snap.cuda_allocated or 0 for snap in self.snapshots if snap.cuda_allocated is not None]
        nvidia_vals = [snap.nvidia_used or 0 for snap in self.snapshots if snap.nvidia_used is not None]
        return {
            "count": len(self.snapshots),
            "cpu_rss_max": max(cpu_vals),
            "cpu_rss_max_h": _bytes_to_human(max(cpu_vals)),
            "cuda_allocated_max": max(cuda_vals) if cuda_vals else None,
            "cuda_allocated_max_h": _bytes_to_human(max(cuda_vals)) if cuda_vals else None,
            "nvidia_used_max": max(nvidia_vals) if nvidia_vals else None,
            "nvidia_used_max_h": _bytes_to_human(max(nvidia_vals)) if nvidia_vals else None,
        }


def element_size(dtype: torch.dtype) -> int:
    dummy = torch.tensor(0, dtype=dtype)
    return int(dummy.element_size())


def _bytes(num_items: int, item_size: int) -> int:
    return int(num_items) * int(item_size)


def estimate_batch_static_memory(
    *,
    batch_size: int,
    seq_len: int,
    num_neurons: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    cov_rank: int,
    dtype: torch.dtype,
    include_copula: bool,
) -> Dict[str, Any]:
    """Rough static estimate for a single forward/backward pass."""

    bytes_per_elem = element_size(dtype)
    seq_in = max(seq_len - 1, 1)
    base_tokens = batch_size * seq_in * num_neurons
    per_tensor = {}

    # Core tiled activations (before/after conv, per layer retention)
    per_tensor["input_tile"] = _bytes(base_tokens * d_model, bytes_per_elem)
    per_tensor["stimulus_token"] = _bytes(batch_size * seq_in * d_model, bytes_per_elem)
    per_tensor["layer_stack"] = per_tensor["input_tile"] * max(n_layers, 1)

    # Attention QKV and output (dense upper bound)
    per_tensor["attention_qkv"] = _bytes(base_tokens * d_model * 3, bytes_per_elem)
    per_tensor["attention_output"] = _bytes(base_tokens * d_model, bytes_per_elem)

    # Copula low-rank factors + float32 q (if enabled)
    if include_copula and cov_rank > 0:
        per_tensor["copula_factors"] = _bytes(base_tokens * cov_rank, bytes_per_elem)
        per_tensor["copula_q_float32"] = _bytes(base_tokens, 4)

    total = sum(per_tensor.values())
    breakdown_h = {k: _bytes_to_human(v) for k, v in per_tensor.items()}
    return {
        "total_bytes": total,
        "total_h": _bytes_to_human(total),
        "breakdown_bytes": per_tensor,
        "breakdown_h": breakdown_h,
        "batch": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_neurons": num_neurons,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "cov_rank": cov_rank,
        },
    }


