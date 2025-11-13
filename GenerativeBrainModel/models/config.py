from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GBMConfig:
    """Configuration for GBM architecture and temporal gating.

    Core:
    - d_model, n_heads, n_layers, d_stimuli: hidden width, heads, depth, stimulus dim.

    Temporal attention (per-neuron over time):
    - temporal_dropout_p: dropout probability inside causal SDPA.
    - temporal_max_bh_per_call: maximum (batch*heads) rows per FLASH call.
    - temporal_max_seq_len: maximum precomputed rotary table length.

    Gating (controls which timesteps are attended per neuron row):
    - gate_capacity: fixed top-k tokens per (B,T,•). Mutually exclusive with gate_fraction.
    - gate_fraction: fraction of valid neurons to keep per (B,T). Mutually exclusive with gate_capacity.
    - gate_temperature: Gumbel noise scale used during training for relaxed top-k.
    - state_proj_rank: rank of low-rank state projection added to x before gating dot.
    - stochastic_topk: enable Gumbel top-k sampling during training.

    Heads:
    - cov_rank: low-rank dimension for covariance factor head.
    """

    # Core architecture
    d_model: int
    n_heads: int
    n_layers: int
    d_stimuli: int

    # Temporal attention (NeuronCausalAttention)
    temporal_dropout_p: float = 0.0
    temporal_max_bh_per_call: int = 2**16 - 1
    temporal_max_seq_len: int = 512

    # Gating
    gate_capacity: Optional[int] = None
    gate_fraction: Optional[float] = None
    gate_temperature: float = 0.1
    state_proj_rank: int = 2
    stochastic_topk: bool = True

    # Other heads
    cov_rank: int = 32
