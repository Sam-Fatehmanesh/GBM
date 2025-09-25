import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
import numpy as np
# from mamba_ssm import Mamba2 as Mamba  # kept for parity
from torch.nn.attention import sdpa_kernel, SDPBackend

import math
# Require FlashAttention-2 varlen
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_varlen_qkv
import torch.distributed as dist





# ========================= Routing + FlashAttention (shared) =========================



class RoutingFlashMHA(nn.Module):
    """
    Shared-routing + FlashAttention-2(varlen), with:
      • Spherical routing features (LN no-affine + L2)
      • Persistent **EMA centroids** synchronized across DDP ranks
      • Balanced top-w per centroid (multi-membership)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        target_cluster_size: int = 384,    # w
        ema_decay: float = 0.999,          # λ
        bias: bool = False,
        ddp_sync: bool = True,             # turn on DDP all-reduce/broadcast
        ddp_pg=None,                       # optional process group
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 8 == 0

        self.w = int(target_cluster_size)
        self.ema_decay = float(ema_decay)
        self.ddp_sync = bool(ddp_sync)
        self.ddp_pg = ddp_pg

        # Fused QKV
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-4)
        if self.out_proj.bias is not None:
            nn.init.normal_(self.out_proj.bias, mean=0.0, std=1e-4)

        # Spherical routing norm over shared routing features (per-token Dh)
        self.route_ln = nn.LayerNorm(self.head_dim, elementwise_affine=False)

        # Persistent centroid bank as a BUFFER (not optimized): [K, Dh], unit-norm
        self.register_buffer("centroids", torch.empty(0, self.head_dim), persistent=True)

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def _ddp_available(self):
        return self.ddp_sync and dist.is_available() and dist.is_initialized()

    def _all_reduce_(self, t: torch.Tensor):
        if self._ddp_available():
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.ddp_pg)

    def _bcast_centroids_(self):
        if self._ddp_available() and self.centroids.numel() > 0:
            # Broadcast whole tensor (simple & robust)
            dist.broadcast(self.centroids, src=0, group=self.ddp_pg)

    def _ensure_centroids(self, device, dtype, K_needed: int, seed_feats: torch.Tensor):
        """
        Ensure we have at least K_needed centroids. If we grow, rank-0 seeds new rows and broadcasts them.
        seed_feats are assumed roughly unit-norm in (Dh,).
        """
        K_cur = int(self.centroids.shape[0])
        if K_needed <= K_cur:
            return

        K_new = K_needed
        # Resize (all ranks) and zero-fill new rows
        if K_cur == 0:
            centroids = torch.zeros(K_new, self.head_dim, device=device, dtype=dtype)
        else:
            centroids = torch.zeros(K_new, self.head_dim, device=device, dtype=dtype)
            centroids[:K_cur] = self.centroids.to(device=device, dtype=dtype)

        # Rank-0 seeds the new rows, others leave zeros; then broadcast
        if (not self._ddp_available()) or dist.get_rank(self.ddp_pg or dist.group.WORLD) == 0:
            if seed_feats.numel() > 0:
                idx = torch.linspace(0, max(seed_feats.size(0) - 1, 0),
                                     steps=K_new - K_cur, device=seed_feats.device).round().to(torch.long)
                new_c = seed_feats.index_select(0, idx).to(device=device, dtype=dtype)
                new_c = self._l2n(new_c)
            else:
                new_c = self._l2n(torch.randn(K_new - K_cur, self.head_dim, device=device, dtype=dtype))
            centroids[K_cur:K_new] = new_c

        self.centroids = centroids  # assign buffer
        self._bcast_centroids_()

    @torch.no_grad()
    def _ema_update_ddp(self, sums_f32: torch.Tensor, counts_f32: torch.Tensor, k_use: int):
        """
        DDP-safe EMA update:
          - all-reduce sums & counts
          - update centroids[:k_use] with fp32 EMA, then write back unit-norm in buffer dtype
        """
        if k_use == 0:
            return
        # All-reduce (SUM) across ranks
        self._all_reduce_(sums_f32)
        self._all_reduce_(counts_f32)

        have = (counts_f32.squeeze(1) > 0)
        if not bool(have.any()):
            return

        # Promote to fp32 for the EMA math
        c = self.centroids[:k_use].to(dtype=torch.float32)
        means = torch.zeros_like(c)
        means[have] = sums_f32[have] / counts_f32[have].clamp_min(1e-6)

        decay = self.ema_decay
        c[have] = self._l2n(decay * c[have] + (1.0 - decay) * means[have])

        # Write back in buffer dtype
        self.centroids[:k_use] = c.to(dtype=self.centroids.dtype)

    @torch._dynamo.disable
    def forward(
        self,
        x_compact: torch.Tensor,          # [Ttot, D]
        seqlens_tokens: torch.Tensor,     # [S]
    ):
        dtype = next(self.parameters()).dtype
        x = x_compact.to(dtype=dtype)

        Ttot, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        assert D == H * Dh
        S = int(seqlens_tokens.numel())

        # Fused QKV
        qkv_full = self.qkv(x).view(Ttot, 3, H, Dh)   # [Ttot, 3, H, Dh]
        qh, kh, vh = qkv_full[:, 0], qkv_full[:, 1], qkv_full[:, 2]

        # Spherical routing features: LN (no affine) + L2, shared across heads
        q_route = self.route_ln(qh.mean(1))
        k_route = self.route_ln(kh.mean(1))
        r = 0.5 * (q_route + k_route)
        r = self._l2n(r.detach())                     # [Ttot, Dh], unit-norm

        # Per-set bounds
        lens_dev = seqlens_tokens.to(device=x.device, dtype=torch.long)
        cu_tok = torch.zeros(S + 1, device=x.device, dtype=torch.long)
        cu_tok[1:] = lens_dev.cumsum(0)
        bounds = cu_tok.detach().cpu().tolist()

        # Determine local k_needed (max over sets)
        k_needed_local = 0
        for s in range(S):
            a = int(bounds[s]); b = int(bounds[s + 1])
            Ls = max(0, min(b, Ttot) - min(a, Ttot))
            if Ls <= 0:
                continue
            w_eff = max(1, min(self.w, Ls))
            k_s = max(1, (Ls + w_eff - 1) // w_eff)
            k_needed_local = max(k_needed_local, k_s)

        # Global max across ranks so we reduce with uniform shapes
        k_needed = k_needed_local
        if self._ddp_available():
            t = torch.tensor([k_needed_local], device=x.device, dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.ddp_pg)
            k_needed = int(t.item())

        # Ensure/grow centroid bank to k_needed (seed & broadcast if needed)
        seed_idx = torch.linspace(0, max(Ttot - 1, 0), steps=max(1, k_needed),
                                  device=r.device).round().to(torch.long)
        self._ensure_centroids(x.device, r.dtype, k_needed, seed_feats=r.index_select(0, seed_idx))

        # Build multi-membership top-w packing and accumulate EMA stats (fp32)
        order_chunks, lens_chunks = [], []
        ema_sums = torch.zeros(k_needed, Dh, device=x.device, dtype=torch.float32)
        ema_counts = torch.zeros(k_needed, 1, device=x.device, dtype=torch.float32)

        for s in range(S):
            a = int(bounds[s]); b = int(bounds[s + 1])
            Ls = max(0, min(b, Ttot) - min(a, Ttot))
            if Ls <= 0:
                continue
            a = min(a, Ttot)

            feats = r.narrow(0, a, Ls)                       # [L, Dh]
            w_eff = max(1, min(self.w, Ls))
            k_s = max(1, (Ls + w_eff - 1) // w_eff)

            C = self.centroids[:k_s]                          # [k_s, Dh]
            sims = feats @ C.T                                # [L, k_s]

            # Balanced top-w per centroid (multi-membership)
            top_idx = sims.topk(k=w_eff, dim=0, largest=True, sorted=False).indices  # [w_eff, k_s]
            kept_rel = top_idx.transpose(0, 1).reshape(-1)    # [k_s * w_eff]
            order_abs = kept_rel + a
            order_chunks.append(order_abs)
            lens_chunks.append(torch.full((k_s,), w_eff, device=x.device, dtype=torch.int32))

            # Nearest-centroid assign for EMA stats
            assign = sims.argmax(dim=1)                       # [L]
            # fp32 accumulation
            feats_f32 = feats.to(torch.float32)
            ema_sums[:k_s].index_add_(0, assign, feats_f32)
            ema_counts[:k_s] += torch.bincount(assign, minlength=k_s).to(torch.float32).unsqueeze(1)

        if not order_chunks:
            return self.out_proj(torch.zeros_like(x))

        order = torch.cat(order_chunks, dim=0)                # [Tdup]
        lens_t = torch.cat(lens_chunks, dim=0)                # [#clusters_total]
        cu_cls = torch.zeros(lens_t.numel() + 1, device=x.device, dtype=torch.int32)
        cu_cls[1:] = lens_t.cumsum(0)
        max_seqlen = int(lens_t.max().item())

        # QKV pack (reorder → stack)
        qh_sel = qh.index_select(0, order)
        kh_sel = kh.index_select(0, order)
        vh_sel = vh.index_select(0, order)
        qkv = torch.stack([qh_sel, kh_sel, vh_sel], dim=1).contiguous()  # [Tdup, 3, H, Dh]
        if qkv.dtype not in (torch.bfloat16, torch.float16):
            qkv = qkv.to(torch.bfloat16)

        out_packed = flash_varlen_qkv(qkv, cu_seqlens=cu_cls, max_seqlen=max_seqlen,
                                      dropout_p=0.0, softmax_scale=None, causal=False)  # [Tdup, H, Dh]

        # Merge duplicates by sum → average by count
        out_h = torch.zeros(Ttot, H, Dh, device=x.device, dtype=out_packed.dtype)
        out_h.index_add_(0, order, out_packed)
        counts = torch.zeros(Ttot, device=x.device, dtype=out_packed.dtype)
        counts.index_add_(0, order, torch.ones(order.numel(), device=x.device, dtype=out_packed.dtype))
        out_h = out_h / counts.clamp_min(1.0).view(-1, 1, 1)

        # DDP-safe EMA update (single pass across batch)
        # Freeze centroid EMA during inference to ensure deterministic eval behavior
        if self.training:
            with torch.no_grad():
                self._ema_update_ddp(ema_sums, ema_counts, k_needed)

        out_input = out_h.reshape(Ttot, D)
        weight_dtype = self.out_proj.weight.dtype
        if out_input.dtype != weight_dtype:
            out_input = out_input.to(dtype=weight_dtype)
        return self.out_proj(out_input)

class SpatialNeuralAttention(nn.Module):
    """
    Vectorized pad compaction → shared-routing (spherical, EMA, multi-membership top-w) → FA-2(varlen).
    No special global token; duplicates across clusters are allowed and scatter-added.
    """
    def __init__(self, d_model, n_heads, n_rope_features=32,
                 target_cluster_size: int = 384, ema_decay: float = 0.999):
        super().__init__()
        self.attn = RoutingFlashMHA(
            d_model, n_heads,
            target_cluster_size=target_cluster_size,
            ema_decay=ema_decay,
            bias=False,
        )

        self.norm = RMSNorm(d_model)
        self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

        dirs = torch.randn(n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        freqs = torch.logspace(math.log10(1.0), math.log10(10000.0), n_rope_features)
        self.register_buffer('rope_dirs', dirs, persistent=False)
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _directional_rope(self, positions):  # positions: (B, N, 3)
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
        proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)
        angles = proj * rope_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)

    def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D)
        B_T, N, D = x.shape
        B = rope_emb.shape[0]
        T = B_T // B
        rope = self.rope_proj(rope_emb.to(dtype=x.dtype, device=x.device))  # (B,N,D)
        return x + rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)

    def forward(self, x, point_positions, neuron_pad_mask=None):
        # x: (B,T,N,D)  (the last token can be your stimulus token, but it's no longer treated specially)
        B, T, N, D = x.shape
        S = B * T

        x_bt = x.reshape(S, N, D)
        res = x_bt
        xn = self.norm(x_bt)

        rope_emb = self._directional_rope(point_positions)
        qk = self._apply_rope(xn, rope_emb)  # RoPE on Q/K only

        # Vectorized compaction
        if neuron_pad_mask is None:
            keep = torch.ones((B, N), dtype=torch.bool, device=x.device)
        else:
            keep = (neuron_pad_mask != 0).to(torch.bool)
        keep_bt = keep.unsqueeze(1).expand(B, T, N).reshape(S, N)      # (S,N)
        lens = keep_bt.sum(dim=1).to(torch.int32)                      # [S]

        cu_tok = torch.zeros(S + 1, device=x.device, dtype=torch.long)
        cu_tok[1:] = lens.to(torch.long).cumsum(0)

        flat_mask = keep_bt.reshape(-1)
        # compile-friendly index build (avoid .nonzero())
        idx_flat = torch.arange(flat_mask.numel(), device=x.device, dtype=torch.long)[flat_mask]

        qk_flat = qk.reshape(S * N, D)
        xn_flat = xn.reshape(S * N, D)
        x_compact = qk_flat.index_select(0, idx_flat)                  # [Ttot, D]
        v_compact = xn_flat.index_select(0, idx_flat)                  # (not used directly, but x_compact is the source)

        # Router uses x_compact for Q/K/V sources internally (self-attn)
        out_compact = self.attn(x_compact, seqlens_tokens=lens)        # [Ttot, D]

        # Scatter back
        out_flat = torch.zeros(S * N, D, device=x.device, dtype=out_compact.dtype)
        out_flat.index_copy_(0, idx_flat, out_compact)
        y = out_flat.view(S, N, D) + res
        return y.view(B, T, N, D)


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
    def _ensure_flash_dtype(*tensors: torch.Tensor):
        for x in tensors:
            if x.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError(
                    f"FlashAttention requires bf16/fp16 (got {x.dtype})."
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
        # Run projections in module dtype (may be fp32), then ensure FLASH dtype on q/k/v.
        module_dtype = next(self.parameters()).dtype
        q = self._to_bhld(self.q_proj(q_in.to(dtype=module_dtype))).contiguous()
        k = self._to_bhld(self.k_proj(k_in.to(dtype=module_dtype))).contiguous()
        v = self._to_bhld(self.v_proj(v_in.to(dtype=module_dtype))).contiguous()
        # Ensure bf16/fp16 for FLASH
        if q.dtype not in (torch.bfloat16, torch.float16):
            q = q.to(torch.bfloat16)
        if k.dtype not in (torch.bfloat16, torch.float16):
            k = k.to(torch.bfloat16)
        if v.dtype not in (torch.bfloat16, torch.float16):
            v = v.to(torch.bfloat16)

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
        out = out.to(module_dtype)
        return self.out_proj(out)


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

        x = out.to(res.dtype) + res
        x = x.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()   # back to (B,T,N,D)
        return x
