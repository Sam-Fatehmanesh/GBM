import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
import numpy as np
# from mamba_ssm import Mamba2 as Mamba  # kept for parity
from torch.nn.attention import sdpa_kernel, SDPBackend


class SDPA_MHA(nn.Module):
    """
    Multi-head attention that uses **FLASH_ATTENTION only** and chunks the batch so
    each kernel launch sees a safe (B_chunk * H). No fallbacks to Efficient/Math.

    Notes:
    - Requires dtype in {bfloat16, float16} and head_dim % 8 == 0.
    - Tune `max_bh_per_call` if you change heads or see launch errors.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_p: float = 0.0,
        bias: bool = False,
        # From your probe: FLASH worked at B_chunk=16384 with H=8 → BH=131072
        max_bh_per_call: int = 131072,
        debug_once: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 8 == 0, "FLASH requires head_dim to be a multiple of 8"

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout_p = dropout_p
        self.max_bh_per_call = int(max_bh_per_call)
        self._debug_once = debug_once
        self._printed = False

    @staticmethod
    def _ensure_flash_dtype(x: torch.Tensor):
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                f"FLASH requires bf16/fp16; got {x.dtype}. "
                "Use autocast to bf16/fp16 or model.to(dtype=torch.bfloat16/float16)."
            )

    def _to_bhld(self, x):  # (B,L,D) -> (B,H,L,Dh)
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _from_bhld(self, x):  # (B,H,L,Dh) -> (B,L,D)
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

    def _flash_once(self, q, k, v, *, is_causal: bool):
        # Single-kernel call with FLASH only.
        BH = q.shape[0] * q.shape[1]
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=is_causal, dropout_p=self.dropout_p
                )
        except RuntimeError as e:
            msg = (
                f"FLASH launch refused. "
                f"q.shape={tuple(q.shape)} (B={q.shape[0]}, H={q.shape[1]}, L={q.shape[2]}, Dh={q.shape[3]}) "
                f"BH={BH}, causal={is_causal}, dtype={q.dtype}. "
                f"Try reducing (B_chunk * H) via `max_bh_per_call`, or head_dim, or ensure bf16/fp16."
            )
            if self._debug_once and not self._printed:
                print("[FLASH refused]", msg, "err=", e)
                self._printed = True
            raise

    def forward(self, q_in, k_in, v_in, *, is_causal: bool = False):
        # Inputs: (B, L, D)
        # Match input dtypes to module parameter dtype (e.g., bf16) to avoid Linear matmul dtype mismatches
        target_dtype = next(self.parameters()).dtype
        q_in = q_in.to(dtype=target_dtype)
        k_in = k_in.to(dtype=target_dtype)
        v_in = v_in.to(dtype=target_dtype)
        self._ensure_flash_dtype(q_in)
        self._ensure_flash_dtype(k_in)
        self._ensure_flash_dtype(v_in)

        q = self._to_bhld(self.q_proj(q_in)).contiguous()
        k = self._to_bhld(self.k_proj(k_in)).contiguous()
        v = self._to_bhld(self.v_proj(v_in)).contiguous()

        B, H, L, Dh = q.shape
        if H > self.max_bh_per_call:
            raise RuntimeError(
                f"H={H} exceeds max_bh_per_call={self.max_bh_per_call}; "
                f"increase max_bh_per_call or reduce heads."
            )
        # Choose B_chunk so (B_chunk * H) <= max_bh_per_call
        bstep = max(1, min(B, self.max_bh_per_call // H))
        out_chunks = []
        for i in range(0, B, bstep):
            qs = q[i:i + bstep]
            ks = k[i:i + bstep]
            vs = v[i:i + bstep]
            out_chunks.append(self._flash_once(qs, ks, vs, is_causal=is_causal))
        out = torch.cat(out_chunks, dim=0)  # (B,H,L,Dh)
        out = self._from_bhld(out)
        return self.out_proj(out)


class SpatialNeuralAttention(nn.Module):
    """
    Spatial attention over N neurons per (batch,time).
    Uses pad compaction (per (b,t)) to avoid masks so FLASH can run.
    """
    def __init__(self, d_model, n_heads, n_rope_features=32, max_bh_per_call: int = 131072, debug_once: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_rope_features = n_rope_features

        self.attn = SDPA_MHA(d_model, n_heads, dropout_p=0.0, bias=False,
                             max_bh_per_call=max_bh_per_call, debug_once=debug_once)
        self.norm = RMSNorm(d_model)
        self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

        dirs = torch.randn(n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        freqs = torch.logspace(np.log10(1.0), np.log10(10000.0), n_rope_features)
        self.register_buffer('rope_dirs', dirs, persistent=False)
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _directional_rope(self, positions):  # positions: (B, N, 3)
        # Match dtype/device to positions to avoid upcasts
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
        proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)   # (B,N,F)
        angles = proj * rope_freqs                                 # (B,N,F)
        rope_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)
        return rope_emb

    def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D); rope_emb: (B,N,2F)
        B_T, N, D = x.shape
        B = rope_emb.shape[0]
        T = B_T // B
        rope = self.rope_proj(rope_emb.to(dtype=x.dtype, device=x.device))  # (B,N,D)
        rope = rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)
        return x + rope

    def forward(self, x, point_positions, neuron_pad_mask=None):
        # x: (B,T,N,D)  | point_positions: (B,N,3) | neuron_pad_mask: (B,N) with 1=valid, 0=pad
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)  # (B*T,N,D)
        res = x
        xn = self.norm(x)

        rope_emb = self._directional_rope(point_positions)  # (B,N,2F)
        qk = self._apply_rope(xn, rope_emb)                 # queries/keys only

        if neuron_pad_mask is None:
            out = self.attn(qk, qk, xn, is_causal=False)    # (B*T,N,D)
        else:
            # Compact away padded neurons per (b,t) so FLASH sees only valid tokens
            pad_bt = (neuron_pad_mask == 0).unsqueeze(1).expand(B, T, N).reshape(B*T, N)
            out = torch.zeros_like(x)
            for i in range(B * T):
                keep = ~pad_bt[i]  # True where valid
                if keep.any():
                    qi = qk[i, keep].unsqueeze(0)  # (1, N_i, D)
                    ki = qk[i, keep].unsqueeze(0)
                    vi = xn[i, keep].unsqueeze(0)
                    oi = self.attn(qi, ki, vi, is_causal=False).squeeze(0)  # (N_i,D)
                    out[i, keep] = oi.to(out.dtype)
                # pad positions remain zero (residual passthrough)
        x = out + res
        return x.view(B, T, N, D)


class TemporalNeuralAttention(nn.Module):
    """
    Temporal attention over T per neuron. Row-skips padded neurons so FLASH doesn't see masks.
    """
    def __init__(self, d_model, n_heads, max_bh_per_call: int = 131072, debug_once: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn = SDPA_MHA(d_model, n_heads, dropout_p=0.0, bias=False,
                             max_bh_per_call=max_bh_per_call, debug_once=debug_once)
        self.norm = RMSNorm(d_model)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('rope_embeddings', inv_freq)  # (D/2,)

    def _apply_rotary_pos_emb(self, x, seq_dim=1):
        # x: (B*, T, D)
        seq_len = x.shape[seq_dim]; d = x.shape[-1]; half = d // 2
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)  # (T,)
        rope = self.rope_embeddings.to(dtype=x.dtype, device=x.device)
        sinus = torch.einsum("i,j->ij", pos, rope)     # (T, D/2)
        sin_pos, cos_pos = torch.sin(sinus), torch.cos(sinus)
        sin_pos = sin_pos.unsqueeze(0).expand(x.shape[0], seq_len, half)
        cos_pos = cos_pos.unsqueeze(0).expand(x.shape[0], seq_len, half)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd  = x_even * sin_pos + x_odd * cos_pos
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2], x_rot[..., 1::2] = x_rot_even, x_rot_odd
        return x_rot

    def forward(self, x, neuron_pad_mask=None):
        # x: (B,T,N,D) ; neuron_pad_mask: (B,N) with 1=valid, 0=pad
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N,T,D)
        res = x
        xn = self.norm(x)
        qk = self._apply_rotary_pos_emb(xn, seq_dim=1)

        if neuron_pad_mask is None:
            out = self.attn(qk, qk, xn, is_causal=True)           # (B*N,T,D)
        else:
            valid_rows = (neuron_pad_mask.reshape(-1) != 0)       # (B*N,)
            out = torch.zeros_like(x)
            if valid_rows.any():
                qk_valid = qk[valid_rows]
                xn_valid = xn[valid_rows]
                out_valid = self.attn(qk_valid, qk_valid, xn_valid, is_causal=True)
                out[valid_rows] = out_valid.to(out.dtype)
            # rows that were padded stay zero → residual passthrough below

        x = out + res
        x = x.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()   # back to (B,T,N,D)
        return x
