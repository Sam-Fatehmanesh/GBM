## Neuron-level gating for temporal self-attention — Summary of changes

### Overview
- Added per-timestep neuron gating for temporal self-attention to reduce memory and compute.
- Introduced a centralized configuration object (`GBMConfig`) to capture core model and gating parameters.
- Integrated gating into the temporal attention module (`NeuronCausalAttention`) and propagated it from the higher-level model stack.
- Restored a simple global mean-pooled state token for spatial attention context, per request.
- Added concise docstrings to new/modified APIs for clarity.

---

### Temporal gating (per-timestep, per-neuron)
For each batch `b`, timestep `t`, and neuron `n`, the gate produces a logit

$$
\mathrm{logit}[b,t,n] = g^\top \big( x[b,t,n] + W_{\text{out}} W_{\text{in}} \,\bar{x}[b,t] \big) + b_g
$$

where:
- $x[b,t,n] \in \mathbb{R}^{D}$ is the neuron token,
- $\bar{x}[b,t] = \frac{1}{|V_{b,t}|}\sum_{n\in V_{b,t}} x[b,t,n]$ is the masked mean over valid neurons at $(b,t)$,
- $W_{\text{in}} \in \mathbb{R}^{D\times r},\; W_{\text{out}} \in \mathbb{R}^{r\times D}$ form a rank-$r$ state projection (config: `state_proj_rank`),
- $g \in \mathbb{R}^{D},\; b_g \in \mathbb{R}$ are learned gating parameters.

Probability and selection:
- Gate probability: $\sigma(\text{logit})$.
- Training-time relaxed top‑k: add Gumbel noise with temperature $\tau$ to logits (config: `gate_temperature`, `stochastic_topk`).
- Capacity:
  - Either fixed `gate_capacity=K` (top‑K across neurons per $(b,t)$), or
  - Fractional `gate_fraction=\alpha$ (select $\lfloor \alpha \cdot |V_{b,t}| \rfloor$ per $(b,t)$).
  - Exactly one of the two must be set; if the selected set equals all valid neurons, the dense path is used.

---

### Varlen causal attention construction
Let the temporal attention operate per neuron timeline independently over time:
- For each neuron row (length $T$), construct a variable-length sequence consisting only of timesteps where the gate is True.
- Apply timewise rotary embeddings using either a contiguous $[0:\mathrm{len})$ slice or per-token indices for varlen-packed tokens.
- Use FlashAttention‑2 varlen (`flash_attn_varlen_qkvpacked`) when available (BF16/FP16 on CUDA), else fall back to per-sequence SDPA with `is_causal=True`.
- Scatter the attention outputs back to original $(b, n, t)$ positions; use dense path if gating degrades to identity.

Result: temporal attention cost scales with selected timesteps rather than full sequence length at all rows.

---

### Configuration consolidation (GBMConfig)
Added `GenerativeBrainModel/models/config.py`:
- Core: `d_model`, `n_heads`, `n_layers`, `d_stimuli`
- Temporal attention controls: `temporal_dropout_p`, `temporal_max_bh_per_call`, `temporal_max_seq_len`
- Gating: `gate_capacity | gate_fraction` (mutually exclusive), `gate_temperature`, `state_proj_rank`, `stochastic_topk`
- Heads: `cov_rank`

Propagation:
- `GBM(config=GBMConfig(...))` stores the config and passes it to each `SpatioTemporalNeuralAttention`, which forwards it into `NeuronCausalAttention`.
- If `config` is omitted, temporal attention runs dense (no gating) using safe defaults.

---

### Spatio-temporal stack adjustments
- Global state token: reinstated a simple global mean-pooled state token per timestep
  - Pipeline: `FFN(d_model, 3d_model)` $\to$ mean over neurons (keep dims) $\to$ `RMSNorm` $\to$ concatenate as $(B,T,1,D)$
  - Masks/positions are extended by one entry for this token; later removed before temporal attention.
- Spatial attention is unchanged functionally (still uses spike masks), followed by `FFN0`.
- Temporal attention is `NeuronCausalAttention` with gating as configured.
- A final `FFN1` completes the block.

---

### API notes and docstrings
Key functions/classes received short but informative docstrings:
- `GBMConfig` fields and semantics.
- `NeuronCausalAttention`:
  - `_apply_timewise_rope`, `_apply_timewise_rope_indexed` (rotary details),
  - `_compute_gate_mask` (formula, identity criteria),
  - `forward` and varlen fallbacks.
- `SpatioTemporalNeuralAttention.__init__/forward`: high-level pipeline.
- `GBM` class: stage-level summary (embeddings, attention stack with gating, heads).

---

### Training and inference impact
- Memory and compute reductions during temporal attention proportional to the fraction/capacity of selected timesteps.
- Deterministic dense fallback when gating selects all valid tokens, preserving baseline throughput/quality.
- Gating is differentiable via logits; selection is non-differentiable but noise-tempered during training to help learning.

---

### How to enable gating
1) Construct a `GBMConfig` with exactly one of:
   - `gate_capacity = K` (int $> 0$), or
   - `gate_fraction = \alpha$ ( $0 < \alpha \leq 1$ ).
2) Pass it into GBM: `model = GBM(..., config=cfg)`.
3) Optionally tune:
   - `gate_temperature` (e.g., 0.1) and `stochastic_topk=True` for relaxed top‑k in training,
   - `state_proj_rank` (small rank, e.g., 2) for the low-rank state projection.

---

### Files touched (high level)
- `GenerativeBrainModel/models/attention.py` — temporal gating, varlen attention, rotary handling; docstrings.
- `GenerativeBrainModel/models/config.py` — new `GBMConfig`.
- `GenerativeBrainModel/models/spatiotemporal.py` — forward config into temporal attention; global mean state token; docstrings.
- `GenerativeBrainModel/models/gbm.py` — hold/propagate config; minor cleanups; docstrings.
- `tests/test_attention_forward.py` — gating-aware tests (minor adjustments).

