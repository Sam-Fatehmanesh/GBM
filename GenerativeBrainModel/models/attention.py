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





# ========================= Routing + FlashAttention (shared) =========================

class RoutingFlashMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        target_cluster_size: int = 384,
        routing_iters: int = 1,
        dropout_p: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 8 == 0

        self.w = int(target_cluster_size)
        self.routing_iters = int(routing_iters)
        self.dropout_p = float(dropout_p)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    @staticmethod
    def _ensure_flash_dtype(*xs):
        for x in xs:
            if x.dtype not in (torch.bfloat16, torch.float16):
                raise RuntimeError(f"FlashAttention requires bf16/fp16 (got {x.dtype}).")

    @staticmethod
    def _l2n(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(
        self,
        q_in: torch.Tensor,             # [Ttot, D]
        k_in: torch.Tensor,             # [Ttot, D]
        v_in: torch.Tensor,             # [Ttot, D]
        seqlens_tokens: torch.Tensor,   # [S]
        global_idx_per_set: torch.Tensor,  # [S]
    ):
        # Cast to module dtype first, then force FLASH-friendly dtype (bf16/fp16)
        dtype = next(self.parameters()).dtype
        q_in = q_in.to(dtype=dtype)
        k_in = k_in.to(dtype=dtype)
        v_in = v_in.to(dtype=dtype)
        if q_in.dtype not in (torch.bfloat16, torch.float16):
            q_in = q_in.to(torch.bfloat16)
        if k_in.dtype not in (torch.bfloat16, torch.float16):
            k_in = k_in.to(torch.bfloat16)
        if v_in.dtype not in (torch.bfloat16, torch.float16):
            v_in = v_in.to(torch.bfloat16)

        Ttot, D = q_in.shape
        H, Dh = self.n_heads, self.head_dim
        assert D == H * Dh
        S = int(seqlens_tokens.numel())
        assert global_idx_per_set.numel() == S

        # QKV projections (vectorized)
        q_full = self.q_proj(q_in)                # [Ttot, D]
        k_full = self.k_proj(k_in)
        v_full = self.v_proj(v_in)
        qh = q_full.view(Ttot, H, Dh)             # [Ttot, H, Dh]
        kh = k_full.view(Ttot, H, Dh)
        vh = v_full.view(Ttot, H, Dh)

        # Routing features (stop grad so routing doesn’t build a graph)
        r = 0.5 * (self._l2n(qh.mean(1).detach()) + self._l2n(kh.mean(1).detach()))  # [Ttot, Dh]

        # ---- per-set offsets computed on device to guarantee consistency with Ttot ----
        lens_dev = seqlens_tokens.to(device=q_in.device, dtype=torch.long)             # [S]
        cu_tok = torch.zeros(S + 1, device=q_in.device, dtype=torch.long)
        cu_tok[1:] = lens_dev.cumsum(0)
        # Avoid per-iteration CUDA .item() syncs by materializing bounds on CPU once
        bounds = cu_tok.detach().cpu().tolist()                                        # length S+1

        order_aug_chunks = []
        lens_aug_chunks  = []

        # one reusable scalar tensor for ‘global’ write (avoid tiny allocs)
        one_f32 = torch.tensor(1.0, device=q_in.device, dtype=dtype)  # for counts later

        for s in range(S):
            a = int(bounds[s])
            b = int(bounds[s + 1])
            Ls = b - a
            if Ls <= 0:
                continue
            # Clamp bounds defensively to Ttot
            a = min(a, Ttot)
            b = min(b, Ttot)
            Ls = b - a
            if Ls <= 0:
                continue

            feats = r.narrow(0, a, Ls)                      # [L, Dh] on GPU
            # Ensure non-zero cluster widths and counts
            w = max(1, min(self.w, Ls))
            k = max(1, (Ls + w - 1) // w)

            # ---- k-means-lite (vectorized), small fixed iters ----
            # init centroids deterministically
            with torch.no_grad():
                # evenly spaced indices (compute on CPU, then clamp and move)
                init_idx_cpu = torch.linspace(0, max(Ls - 1, 0), steps=k).round().to(torch.long)
                if Ls > 0:
                    init_idx_cpu.clamp_(0, Ls - 1)
                centroids = feats[init_idx_cpu.to(feats.device)]          # [k, Dh]
                for _ in range(max(1, self.routing_iters)):
                    sims = feats @ centroids.T                            # [L, k]
                    assign = sims.argmax(dim=1)                           # [L]
                    sums = torch.zeros_like(centroids)                    # [k, Dh]
                    sums.index_add_(0, assign, feats)
                    counts = torch.bincount(assign, minlength=k).clamp_min(1).unsqueeze(1).to(sums.dtype)
                    centroids = sums / counts
                    centroids = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-6)

            sims = feats @ centroids.T                                    # [L, k]
            assign = sims.argmax(dim=1)                                   # [L]
            own = sims.gather(1, assign.unsqueeze(1)).squeeze(1)          # [L]

            # ---- balanced trimming (fully vectorized within the set) ----
            # sort by score desc, then by cluster id (stable)
            order1 = torch.argsort(-own)
            assign1 = assign[order1]
            order2 = torch.argsort(assign1, stable=True)
            perm = order1[order2]                                         # [L]
            cid_sorted = assign1[order2]                                  # [L]

            counts = torch.bincount(cid_sorted, minlength=k)              # [k]
            keep_counts = torch.minimum(counts, torch.tensor(w, device=counts.device))
            valid_clusters = keep_counts > 0
            if not bool(valid_clusters.any()):
                keep_counts = torch.tensor([Ls], device=counts.device)
                cid_sorted = torch.zeros(Ls, device=counts.device, dtype=torch.long)
                perm = torch.arange(Ls, device=counts.device, dtype=torch.long)

            # compute position within each cluster segment
            starts = (torch.cumsum(counts, dim=0) - counts)               # [k]
            starts_rep = torch.repeat_interleave(starts, counts.clamp_min(0))
            pos = torch.arange(cid_sorted.numel(), device=cid_sorted.device) - starts_rep
            keep_mask = pos < keep_counts[cid_sorted]
            kept = perm[keep_mask]                                        # [sum kept]

            # absolute indices in global compact space
            order_abs_kept = kept + a                                     # [sum kept]

            # ---- append global token after each cluster (vectorized) ----
            kc = keep_counts[keep_counts > 0]                              # [C]
            C = kc.numel()
            ins = kc.cumsum(0) + torch.arange(C, device=kc.device)         # [C]
            total_aug = int(kc.sum().item()) + C

            with_global = torch.empty(total_aug, device=q_in.device, dtype=torch.long)
            mask = torch.ones(total_aug, dtype=torch.bool, device=q_in.device)
            mask[ins] = False
            with_global[mask] = order_abs_kept

            gidx = int(global_idx_per_set[s].item())
            with_global[ins] = gidx

            order_aug_chunks.append(with_global)                           # [total_aug]
            lens_aug_chunks.append((kc + 1).to(torch.int32))               # +1 for global

        if not order_aug_chunks:
            return torch.zeros_like(q_in)

        # ---- concat across sets and call FlashAttention once ----
        order_aug = torch.cat(order_aug_chunks, dim=0)                     # [Tdup]
        lens_t = torch.cat(lens_aug_chunks, dim=0)                         # [#clusters_total]
        cu_cls = torch.zeros(lens_t.numel() + 1, device=q_in.device, dtype=torch.int32)
        cu_cls[1:] = lens_t.cumsum(0)
        max_seqlen = int(lens_t.max().item())

        qkv = torch.stack([qh, kh, vh], dim=1).contiguous()                # [Ttot, 3, H, Dh]
        qkv = qkv[order_aug]                                               # [Tdup, 3, H, Dh]
        if qkv.dtype not in (torch.bfloat16, torch.float16):
            qkv = qkv.to(torch.bfloat16)

        p = self.dropout_p if self.training and self.dropout_p > 0 else 0.0
        out_packed = flash_varlen_qkv(qkv, cu_seqlens=cu_cls, max_seqlen=max_seqlen,
                                      dropout_p=p, softmax_scale=None, causal=False)   # [Tdup, H, Dh]

        # merge duplicates (global per cluster) by MEAN — fully vectorized
        out_h = torch.zeros(Ttot, H, Dh, device=q_in.device, dtype=out_packed.dtype)
        out_h.index_add_(0, order_aug, out_packed)
        counts_merge = torch.zeros(Ttot, device=q_in.device, dtype=out_packed.dtype)
        counts_merge.index_add_(0, order_aug, one_f32.expand(order_aug.numel()))
        out_h = out_h / counts_merge.clamp_min(1.0).view(-1, 1, 1)

        return self.out_proj(out_h.reshape(Ttot, D))


class SpatialNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_rope_features=32,
                 target_cluster_size: int = 384, routing_iters: int = 1):
        super().__init__()
        self.attn = RoutingFlashMHA(
            d_model, n_heads,
            target_cluster_size=target_cluster_size,
            routing_iters=routing_iters,
            dropout_p=0.0,
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
        # x: (B,T,N,D) with last token = global
        B, T, N, D = x.shape
        S = B * T

        x_bt = x.reshape(S, N, D)
        res = x_bt
        xn = self.norm(x_bt)

        rope_emb = self._directional_rope(point_positions)
        qk = self._apply_rope(xn, rope_emb)  # RoPE on Q/K only

        if neuron_pad_mask is None:
            keep = torch.ones((B, N), dtype=torch.bool, device=x.device)
        else:
            keep = (neuron_pad_mask != 0).to(torch.bool)
        if not bool(keep[:, -1].all()):
            raise RuntimeError("Global token (last neuron) must be valid in neuron_pad_mask")

        keep_bt = keep.unsqueeze(1).expand(B, T, N).reshape(S, N)      # (S,N)
        lens = keep_bt.sum(dim=1).to(torch.int32)                      # [S]

        cu_tok = torch.zeros(S + 1, device=x.device, dtype=torch.long)
        cu_tok[1:] = lens.to(torch.long).cumsum(0)

        flat_mask = keep_bt.reshape(-1)
        idx_flat = flat_mask.nonzero(as_tuple=False).squeeze(1)

        # compact index of global (last) token per set in compacted space
        counts_before_last = keep_bt[:, :N-1].sum(dim=1).to(torch.long)  # [S]
        global_idx_per_set = cu_tok[:-1] + counts_before_last            # [S]

        qk_flat = qk.reshape(S * N, D)
        xn_flat = xn.reshape(S * N, D)
        qk_compact = qk_flat[idx_flat]                                   # [Ttot, D]
        v_compact  = xn_flat[idx_flat]                                   # [Ttot, D]

        out_compact = self.attn(
            qk_compact, qk_compact, v_compact,
            seqlens_tokens=lens,
            global_idx_per_set=global_idx_per_set
        )                                                                # [Ttot, D]

        out_flat = torch.zeros(S * N, D, device=x.device, dtype=out_compact.dtype)
        out_flat[idx_flat] = out_compact
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

        x = out + res
        x = x.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()   # back to (B,T,N,D)
        return x
