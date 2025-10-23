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
# Also get normal FlashAttention-2 
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

import torch.distributed as dist

MAX_BATCH_SIZE = 65535



# ========================= Routing + FlashAttention (shared) =========================



# class RoutingFlashMHA(nn.Module):
#     """
#     Shared-routing + FlashAttention-2(varlen), with:
#       • Spherical routing features (LN no-affine + L2)
#       • Persistent **EMA centroids** synchronized across DDP ranks
#       • Balanced top-w per centroid (multi-membership)
#     """
#     def __init__(
#         self,
#         d_model: int,
#         n_heads: int,
#         *,
#         target_cluster_size: int = 384,    # desired tokens per centroid (w)
#         num_centroids: int = 384,          # fixed number of spatial centroids
#         ema_decay: float = 0.999,          # λ
#         bias: bool = False,
#         ddp_sync: bool = True,             # turn on DDP all-reduce/broadcast
#         ddp_pg=None,                       # optional process group
#     ):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.d_model, self.n_heads = d_model, n_heads
#         self.head_dim = d_model // n_heads
#         assert self.head_dim % 8 == 0

#         self.w = int(target_cluster_size)
#         self.num_centroids = int(num_centroids)
#         self.ema_decay = float(ema_decay)
#         self.ddp_sync = bool(ddp_sync)
#         self.ddp_pg = ddp_pg

#         # Fused QKV
#         self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
#         self.out_proj = nn.Linear(d_model, d_model, bias=bias)
#         # nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-4)
#         # if self.out_proj.bias is not None:
#         #     nn.init.normal_(self.out_proj.bias, mean=0.0, std=1e-4)

#         # Spherical routing norm over shared routing features (per-token Dh)
#         self.route_ln = nn.LayerNorm(self.head_dim, elementwise_affine=False)

#         # Fixed spatial centroid bank (features + positions)
#         grid_pos, grid_feat = self._create_initial_centroids(self.num_centroids, self.head_dim, dtype=torch.float32)
#         self.register_buffer("centroids", grid_feat, persistent=True)           # (K, Dh)
#         self.register_buffer("centroid_pos", grid_pos, persistent=True)         # (K, 3)
#         self.register_buffer("centroid_pos_norm2", (grid_pos ** 2).sum(-1), persistent=True)

#         self.spatial_weight = 1.0

#     @staticmethod
#     def _l2n(x: torch.Tensor, eps: float = 1e-6):
#         return x / (x.norm(dim=-1, keepdim=True) + eps)

#     @staticmethod
#     def _create_initial_centroids(K: int, head_dim: int, dtype=torch.float32):
#         # Generate near-uniform grid in [-1,1]^3
#         if K <= 0:
#             raise ValueError("num_centroids must be positive")
#         n = int(round(K ** (1 / 3)))
#         if n ** 3 < K:
#             n += 1
#         axes = torch.linspace(-1.0, 1.0, steps=n)
#         grid = torch.stack(torch.meshgrid(axes, axes, axes, indexing='ij'), dim=-1).reshape(-1, 3)
#         grid = grid[:K]
#         grid_pos = grid.to(dtype=dtype)
#         repeats = (head_dim + 2) // 3
#         tiled = grid_pos.repeat(1, repeats)
#         centroids_feat = tiled[:, :head_dim]
#         centroids_feat = RoutingFlashMHA._l2n(centroids_feat)
#         return grid_pos, centroids_feat

#     def _ddp_available(self):
#         return self.ddp_sync and dist.is_available() and dist.is_initialized()

#     def _all_reduce_(self, t: torch.Tensor):
#         if self._ddp_available():
#             dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.ddp_pg)

#     def _bcast_centroids_(self):
#         if self._ddp_available() and self.centroids.numel() > 0:
#             # Broadcast whole tensor (simple & robust)
#             dist.broadcast(self.centroids, src=0, group=self.ddp_pg)
#             dist.broadcast(self.centroid_pos, src=0, group=self.ddp_pg)
#             dist.broadcast(self.centroid_pos_norm2, src=0, group=self.ddp_pg)

#     def _ensure_centroids(self, *args, **kwargs):
#         # Fixed centroid bank; nothing to do
#         return

#     @torch.no_grad()
#     def _ema_update_ddp(self, sums_f32: torch.Tensor, counts_f32: torch.Tensor, k_use: int):
#         """
#         DDP-safe EMA update:
#           - all-reduce sums & counts
#           - update centroids[:k_use] with fp32 EMA, then write back unit-norm in buffer dtype
#         """
#         if k_use == 0:
#             return
#         # All-reduce (SUM) across ranks
#         self._all_reduce_(sums_f32)
#         self._all_reduce_(counts_f32)

#         have = (counts_f32.squeeze(1) > 0)
#         if not bool(have.any()):
#             return

#         # Promote to fp32 for the EMA math
#         c = self.centroids[:k_use].to(dtype=torch.float32)
#         means = torch.zeros_like(c)
#         means[have] = sums_f32[have] / counts_f32[have].clamp_min(1e-6)

#         decay = self.ema_decay
#         c[have] = self._l2n(decay * c[have] + (1.0 - decay) * means[have])

#         # Write back in buffer dtype
#         self.centroids[:k_use] = c.to(dtype=self.centroids.dtype)

#     @torch._dynamo.disable
#     def forward(
#         self,
#         x_compact: torch.Tensor,          # [Ttot, D]
#         seqlens_tokens: torch.Tensor,     # [S]
#         *,
#         token_positions: torch.Tensor | None = None,
#     ):
#         dtype = next(self.parameters()).dtype
#         x = x_compact.to(dtype=dtype)

#         Ttot, D = x.shape
#         H, Dh = self.n_heads, self.head_dim
#         assert D == H * Dh
#         S = int(seqlens_tokens.numel())

#         # Fused QKV
#         qkv_full = self.qkv(x).view(Ttot, 3, H, Dh)   # [Ttot, 3, H, Dh]
#         qh, kh, vh = qkv_full[:, 0], qkv_full[:, 1], qkv_full[:, 2]

#         # Spherical routing features: LN (no affine) + L2, shared across heads
#         q_route = self.route_ln(qh.mean(1))
#         k_route = self.route_ln(kh.mean(1))
#         r = 0.5 * (q_route + k_route)
#         r = self._l2n(r.detach())                     # [Ttot, Dh], unit-norm

#         # Per-set bounds
#         lens_dev = seqlens_tokens.to(device=x.device, dtype=torch.long)
#         cu_tok = torch.zeros(S + 1, device=x.device, dtype=torch.long)
#         cu_tok[1:] = lens_dev.cumsum(0)
#         bounds = cu_tok.detach().cpu().tolist()

#         # Build multi-membership top-w packing and accumulate EMA stats (fp32)
#         order_chunks, lens_chunks = [], []
#         ema_sums = torch.zeros(self.num_centroids, Dh, device=x.device, dtype=torch.float32)
#         ema_counts = torch.zeros(self.num_centroids, 1, device=x.device, dtype=torch.float32)

#         for s in range(S):
#             a = int(bounds[s]); b = int(bounds[s + 1])
#             Ls = max(0, min(b, Ttot) - min(a, Ttot))
#             if Ls <= 0:
#                 continue
#             a = min(a, Ttot)

#             feats = r.narrow(0, a, Ls)                       # [L, Dh]
#             w_eff = max(1, min(self.w, Ls))
#             k_s = self.num_centroids
#             C = self.centroids.to(device=x.device, dtype=feats.dtype)
#             sims = feats @ C.T
#             pos_slice = None
#             if token_positions is not None:
#                 pos_slice = token_positions.narrow(0, a, Ls)
#                 cen_pos = self.centroid_pos.to(device=x.device, dtype=pos_slice.dtype)
#                 cen_norm = self.centroid_pos_norm2.to(device=x.device, dtype=pos_slice.dtype)
#                 sq_norm_token = (pos_slice ** 2).sum(-1, keepdim=True)
#                 spatial = sq_norm_token - 2 * (pos_slice @ cen_pos.T) + cen_norm
#                 sims = sims - (self.spatial_weight * spatial)

#             top_idx = sims.topk(k=w_eff, dim=0, largest=True, sorted=False).indices  # [w_eff, k_s]
#             base_tokens = top_idx.transpose(0, 1)  # (k_s, w_eff)
#             selected_local = torch.zeros(Ls, dtype=torch.bool, device=x.device)
#             selected_local.scatter_(0, base_tokens.reshape(-1), True)

#             nearest = sims.argmax(dim=1)
#             missing_local = (~selected_local).nonzero(as_tuple=False).flatten()
#             if missing_local.numel() > 0:
#                 nearest_missing = nearest.index_select(0, missing_local)
#                 added_counts = torch.bincount(nearest_missing, minlength=k_s)
#             else:
#                 nearest_missing = None
#                 added_counts = torch.zeros(k_s, dtype=torch.long, device=x.device)

#             lens_local = torch.full((k_s,), w_eff, dtype=torch.long, device=x.device) + added_counts
#             total_tokens = lens_local.sum()

#             start_offsets = torch.zeros(k_s + 1, dtype=torch.long, device=x.device)
#             torch.cumsum(lens_local, dim=0, out=start_offsets[1:])

#             order_abs = torch.empty(int(total_tokens.item()), dtype=torch.long, device=x.device)

#             base_pos = (start_offsets[:-1].unsqueeze(1) + torch.arange(w_eff, device=x.device, dtype=torch.long)).reshape(-1)
#             order_abs[base_pos] = base_tokens.reshape(-1) + a

#             if nearest_missing is not None:
#                 sort_idx = torch.argsort(nearest_missing)
#                 nearest_sorted = nearest_missing.index_select(0, sort_idx)
#                 missing_sorted = missing_local.index_select(0, sort_idx)
#                 offsets = start_offsets[:-1] + w_eff
#                 extra_offsets = torch.repeat_interleave(offsets, added_counts)
#                 prefix = torch.cumsum(added_counts, dim=0) - added_counts
#                 per_entry_prefix = torch.repeat_interleave(prefix, added_counts)
#                 local_rank = torch.arange(missing_local.numel(), device=x.device, dtype=torch.long) - per_entry_prefix
#                 extra_positions = extra_offsets + local_rank
#                 order_abs[extra_positions] = missing_sorted + a

#             order_chunks.append(order_abs)
#             lens_chunks.append(lens_local.to(torch.int32))

#         if not order_chunks:
#             return self.out_proj(torch.zeros_like(x))

#         order = torch.cat(order_chunks, dim=0)                # [Tdup]
#         lens_t = torch.cat(lens_chunks, dim=0)                # [#clusters_total]

#         # Ensure every token is assigned at least once
#         selected_mask = torch.zeros(Ttot, device=x.device, dtype=torch.bool)
#         selected_mask.index_fill_(0, order, True)
#         missing = (~selected_mask).nonzero(as_tuple=False).flatten()
#         if missing.numel() > 0:
#             sims_missing = (r[missing] @ self.centroids.T)
#             nearest = sims_missing.argmax(dim=1)
#             order = torch.cat([order, missing], dim=0)
#             counts_missing = torch.bincount(nearest, minlength=k_s).to(torch.float32)
#             ema_counts[:k_s, 0] += counts_missing
#             sums_missing = torch.zeros_like(ema_sums)
#             sums_missing.index_add_(0, nearest, r[missing].to(torch.float32))
#             ema_sums += sums_missing

#         cu_cls = torch.zeros(lens_t.numel() + 1, device=x.device, dtype=torch.int32)
#         cu_cls[1:] = lens_t.cumsum(0)
#         max_seqlen = int(lens_t.max().item())

#         # QKV pack (reorder → stack)
#         qh_sel = qh.index_select(0, order)
#         kh_sel = kh.index_select(0, order)
#         vh_sel = vh.index_select(0, order)
#         qkv = torch.stack([qh_sel, kh_sel, vh_sel], dim=1).contiguous()  # [Tdup, 3, H, Dh]
#         if qkv.dtype not in (torch.bfloat16, torch.float16):
#             qkv = qkv.to(torch.bfloat16)

#         out_packed = flash_varlen_qkv(qkv, cu_seqlens=cu_cls, max_seqlen=max_seqlen,
#                                       dropout_p=0.0, softmax_scale=None, causal=False)  # [Tdup, H, Dh]

#         # Merge duplicates by sum → average by count
#         out_h = torch.zeros(Ttot, H, Dh, device=x.device, dtype=out_packed.dtype)
#         out_h.index_add_(0, order, out_packed)
#         counts = torch.zeros(Ttot, device=x.device, dtype=out_packed.dtype)
#         counts.index_add_(0, order, torch.ones(order.numel(), device=x.device, dtype=out_packed.dtype))
#         out_h = out_h / counts.clamp_min(1.0).view(-1, 1, 1)

#         # DDP-safe EMA update (single pass across batch)
#         # Freeze centroid EMA during inference to ensure deterministic eval behavior
#         if self.training:
#             with torch.no_grad():
#                 self._ema_update_ddp(ema_sums, ema_counts, self.num_centroids)

#         out_input = out_h.reshape(Ttot, D)
#         weight_dtype = self.out_proj.weight.dtype
#         if out_input.dtype != weight_dtype:
#             out_input = out_input.to(dtype=weight_dtype)
#         return self.out_proj(out_input)

# class SpatialNeuralAttention(nn.Module):
#     """
#     Vectorized pad compaction → shared-routing (spherical, EMA, multi-membership top-w) → FA-2(varlen).
#     No special global token; duplicates across clusters are allowed and scatter-added.
#     """
#     def __init__(self, d_model, n_heads, n_rope_features=32,
#                  target_cluster_size: int = 384, ema_decay: float = 0.999):
#         super().__init__()
#         self.attn = RoutingFlashMHA(
#             d_model, n_heads,
#             target_cluster_size=target_cluster_size,
#             ema_decay=ema_decay,
#             bias=False,
#         )

#         self.norm = RMSNorm(d_model)
#         self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

#         dirs = torch.randn(n_rope_features, 3)
#         dirs = dirs / dirs.norm(dim=-1, keepdim=True)
#         freqs = torch.logspace(math.log10(1.0), math.log10(10000.0), n_rope_features)
#         self.register_buffer('rope_dirs', dirs, persistent=False)
#         self.register_buffer('rope_freqs', freqs, persistent=False)

#     def _directional_rope(self, positions):  # positions: (B, N, 3)
#         rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
#         rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
#         proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)
#         angles = proj * rope_freqs
#         return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)

#     def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D)
#         B_T, N, D = x.shape
#         B = rope_emb.shape[0]
#         T = B_T // B
#         rope = self.rope_proj(rope_emb.to(dtype=x.dtype, device=x.device))  # (B,N,D)
#         return x + rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)

#     def forward(self, x, point_positions, neuron_pad_mask=None):
#         # x: (B,T,N,D)  (the last token can be your stimulus token, but it's no longer treated specially)
#         B, T, N, D = x.shape
#         S = B * T

#         x_bt = x.reshape(S, N, D)
#         res = x_bt
#         xn = self.norm(x_bt)

#         rope_emb = self._directional_rope(point_positions)
#         qk = self._apply_rope(xn, rope_emb)  # RoPE on Q/K only

#         # Vectorized compaction
#         if neuron_pad_mask is None:
#             keep = torch.ones((B, N), dtype=torch.bool, device=x.device)
#         else:
#             keep = (neuron_pad_mask != 0).to(torch.bool)
#         keep_bt = keep.unsqueeze(1).expand(B, T, N).reshape(S, N)      # (S,N)
#         lens = keep_bt.sum(dim=1).to(torch.int32)                      # [S]

#         cu_tok = torch.zeros(S + 1, device=x.device, dtype=torch.long)
#         cu_tok[1:] = lens.to(torch.long).cumsum(0)

#         flat_mask = keep_bt.reshape(-1)
#         # compile-friendly index build (avoid .nonzero())
#         idx_flat = torch.arange(flat_mask.numel(), device=x.device, dtype=torch.long)[flat_mask]

#         qk_flat = qk.reshape(S * N, D)
#         xn_flat = xn.reshape(S * N, D)
#         x_compact = qk_flat.index_select(0, idx_flat)                  # [Ttot, D]
#         v_compact = xn_flat.index_select(0, idx_flat)                  # (not used directly, but x_compact is the source)

#         if point_positions is not None:
#             pos_bt = point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S * N, 3)
#             token_pos = pos_bt.index_select(0, idx_flat)
#         else:
#             token_pos = None

#         # Router uses x_compact for Q/K/V sources internally (self-attn)
#         out_compact = self.attn(x_compact, seqlens_tokens=lens, token_positions=token_pos)        # [Ttot, D]

#         # Scatter back
#         out_flat = torch.zeros(S * N, D, device=x.device, dtype=out_compact.dtype)
#         out_flat.index_copy_(0, idx_flat, out_compact)
#         y = out_flat.view(S, N, D) + res
#         return y.view(B, T, N, D)


# class SDPA_MHA(nn.Module):
#     """
#     Multi-head attention that uses **FLASH_ATTENTION only** and chunks the batch so
#     each kernel launch sees a safe (B_chunk * H). No fallbacks to Efficient/Math.

#     Notes:
#     - Requires dtype in {bfloat16, float16} and head_dim % 8 == 0.
#     - Tune `max_bh_per_call` if you change heads or see launch errors.
#     """
#     def __init__(
#         self,
#         d_model: int,
#         n_heads: int,
#         dropout_p: float = 0.0,
#         bias: bool = False,
#         # From your probe: FLASH worked at B_chunk=16384 with H=8 → BH=131072
#         max_bh_per_call: int = 131072,
#         debug_once: bool = False,
#     ):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
#         self.d_model, self.n_heads = d_model, n_heads
#         self.head_dim = d_model // n_heads
#         assert self.head_dim % 8 == 0, "FLASH requires head_dim to be a multiple of 8"

#         self.q_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.k_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.v_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.out_proj = nn.Linear(d_model, d_model, bias=bias)
#         self.dropout_p = dropout_p
#         self.max_bh_per_call = int(max_bh_per_call)
#         self._debug_once = debug_once
#         self._printed = False

#     @staticmethod
#     def _ensure_flash_dtype(*tensors: torch.Tensor):
#         for x in tensors:
#             if x.dtype not in (torch.bfloat16, torch.float16):
#                 raise RuntimeError(
#                     f"FlashAttention requires bf16/fp16 (got {x.dtype})."
#                 )

#     def _to_bhld(self, x):  # (B,L,D) -> (B,H,L,Dh)
#         B, L, D = x.shape
#         return x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

#     def _from_bhld(self, x):  # (B,H,L,Dh) -> (B,L,D)
#         B, H, L, Dh = x.shape
#         return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

#     def _flash_once(self, q, k, v, *, is_causal: bool):
#         # Single-kernel call with FLASH only.
#         BH = q.shape[0] * q.shape[1]
#         try:
#             with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#                 return F.scaled_dot_product_attention(
#                     q, k, v, is_causal=is_causal, dropout_p=self.dropout_p
#                 )
#         except RuntimeError as e:
#             msg = (
#                 f"FLASH launch refused. "
#                 f"q.shape={tuple(q.shape)} (B={q.shape[0]}, H={q.shape[1]}, L={q.shape[2]}, Dh={q.shape[3]}) "
#                 f"BH={BH}, causal={is_causal}, dtype={q.dtype}. "
#                 f"Try reducing (B_chunk * H) via `max_bh_per_call`, or head_dim, or ensure bf16/fp16."
#             )
#             if self._debug_once and not self._printed:
#                 print("[FLASH refused]", msg, "err=", e)
#                 self._printed = True
#             raise

#     def forward(self, q_in, k_in, v_in, *, is_causal: bool = False):
#         # Inputs: (B, L, D)
#         # Run projections in module dtype (may be fp32), then ensure FLASH dtype on q/k/v.
#         module_dtype = next(self.parameters()).dtype
#         q = self._to_bhld(self.q_proj(q_in.to(dtype=module_dtype))).contiguous()
#         k = self._to_bhld(self.k_proj(k_in.to(dtype=module_dtype))).contiguous()
#         v = self._to_bhld(self.v_proj(v_in.to(dtype=module_dtype))).contiguous()
#         # Ensure bf16/fp16 for FLASH
#         if q.dtype not in (torch.bfloat16, torch.float16):
#             q = q.to(torch.bfloat16)
#         if k.dtype not in (torch.bfloat16, torch.float16):
#             k = k.to(torch.bfloat16)
#         if v.dtype not in (torch.bfloat16, torch.float16):
#             v = v.to(torch.bfloat16)

#         B, H, L, Dh = q.shape
#         if H > self.max_bh_per_call:
#             raise RuntimeError(
#                 f"H={H} exceeds max_bh_per_call={self.max_bh_per_call}; "
#                 f"increase max_bh_per_call or reduce heads."
#             )
#         # Choose B_chunk so (B_chunk * H) <= max_bh_per_call
#         bstep = max(1, min(B, self.max_bh_per_call // H))
#         out_chunks = []
#         for i in range(0, B, bstep):
#             qs = q[i:i + bstep]
#             ks = k[i:i + bstep]
#             vs = v[i:i + bstep]
#             out_chunks.append(self._flash_once(qs, ks, vs, is_causal=is_causal))
#         out = torch.cat(out_chunks, dim=0)  # (B,H,L,Dh)
#         out = self._from_bhld(out)
#         out = out.to(module_dtype)
#         return self.out_proj(out)


# class TemporalNeuralAttention(nn.Module):
#     """
#     Temporal attention over T per neuron. Row-skips padded neurons so FLASH doesn't see masks.
#     """
#     def __init__(self, d_model, n_heads, max_bh_per_call: int = 131072, debug_once: bool = False):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.attn = SDPA_MHA(d_model, n_heads, dropout_p=0.0, bias=False,
#                              max_bh_per_call=max_bh_per_call, debug_once=debug_once)
#         self.norm = RMSNorm(d_model)
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
#         self.register_buffer('rope_embeddings', inv_freq)  # (D/2,)

#     def _apply_rotary_pos_emb(self, x, seq_dim=1):
#         # x: (B*, T, D)
#         seq_len = x.shape[seq_dim]; d = x.shape[-1]; half = d // 2
#         pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)  # (T,)
#         rope = self.rope_embeddings.to(dtype=x.dtype, device=x.device)
#         sinus = torch.einsum("i,j->ij", pos, rope)     # (T, D/2)
#         sin_pos, cos_pos = torch.sin(sinus), torch.cos(sinus)
#         sin_pos = sin_pos.unsqueeze(0).expand(x.shape[0], seq_len, half)
#         cos_pos = cos_pos.unsqueeze(0).expand(x.shape[0], seq_len, half)
#         x_even, x_odd = x[..., 0::2], x[..., 1::2]
#         x_rot_even = x_even * cos_pos - x_odd * sin_pos
#         x_rot_odd  = x_even * sin_pos + x_odd * cos_pos
#         x_rot = torch.empty_like(x)
#         x_rot[..., 0::2], x_rot[..., 1::2] = x_rot_even, x_rot_odd
#         return x_rot

#     def forward(self, x, neuron_pad_mask=None):
#         # x: (B,T,N,D) ; neuron_pad_mask: (B,N) with 1=valid, 0=pad
#         B, T, N, D = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N,T,D)
#         res = x
#         xn = self.norm(x)
#         qk = self._apply_rotary_pos_emb(xn, seq_dim=1)

#         if neuron_pad_mask is None:
#             out = self.attn(qk, qk, xn, is_causal=True)           # (B*N,T,D)
#         else:
#             valid_rows = (neuron_pad_mask.reshape(-1) != 0)       # (B*N,)
#             out = torch.zeros_like(x)
#             if valid_rows.any():
#                 qk_valid = qk[valid_rows]
#                 xn_valid = xn[valid_rows]
#                 out_valid = self.attn(qk_valid, qk_valid, xn_valid, is_causal=True)
#                 out[valid_rows] = out_valid.to(out.dtype)
#             # rows that were padded stay zero → residual passthrough below

#         x = out.to(res.dtype) + res
#         x = x.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()   # back to (B,T,N,D)
#         return x


# class SpikeAwareDirectedAttention(nn.Module):
#     """
#     Attention operating on sending and receiving neuron token groups. All reciving neurons are included in Q, only neurons which are deemed spiking are included in KV. Using FlashAttention-2's flash_varlen_qkv.
#     """
#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.out_proj = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, sending_neurons, receiving_neurons, spike_mask):
#         # Inputs: (B, L, D)

#         module_dtype = next(self.parameters()).dtype
#         q = self.q_proj(q_in.to(dtype=module_dtype))


class SpikeSparseConnectomeRoutingAttention(nn.Module):
    """
    Attention which uses routing to split all neuron tokens into receiving and sending neuron groups. All receiving neurons are included in Q, only sending neurons which are deemed spiking are included in KV. Using FlashAttention-2's flash_varlen_qkv.    
    """
    def __init__(self, d_model, n_heads, neuron_cluster_size, num_clusters_per_head,
                 ema_decay: float = 0.992, n_rope_features: int = 32, dropout: float = 0.0,
                 profile_memory: bool = False):
        super().__init__()

        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.neuron_cluster_size = neuron_cluster_size
        self.num_clusters_per_head = num_clusters_per_head
        self.ema_decay = ema_decay
        self.n_rope_features = n_rope_features
        self.total_cluster_count = n_heads * num_clusters_per_head
        self.dropout = dropout
        self.profile_memory = bool(profile_memory)

        self.norm = RMSNorm(d_model)
        self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

        dirs = torch.randn(n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        freqs = torch.logspace(math.log10(1.0), math.log10(10000.0), n_rope_features)
        self.register_buffer('rope_dirs', dirs, persistent=False)
        self.register_buffer('rope_freqs', freqs, persistent=False)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False) 


        # Fixed centroid banks: ensure these tensors are constructed with no gradients tracked
        input_centroids, output_centroids = self._create_initial_centroids(
            self.num_clusters_per_head, self.n_heads, self.head_dim
        )
        # Store centroids in bf16 to reduce memory and bandwidth during routing
        self.register_buffer("input_centroids", input_centroids.to(torch.bfloat16), persistent=True)
        self.register_buffer("output_centroids", output_centroids.to(torch.bfloat16), persistent=True)

        # === learned null output neuron token (lives in model space) ===
        self.null_output_vec = nn.Parameter(torch.randn(self.d_model))

    @torch.no_grad()
    def _build_spiking_keep_masks(
        self,
        output_idx: torch.Tensor,
        sending_neurons_mask_bt: torch.Tensor,
        S: int,
        H: int,
        C: int,
        K: int,
    ):
        non_spiking_mask, all_non_spiking = self._get_indicies_of_non_spiking_neurons_in_clusters(
            output_idx, sending_neurons_mask_bt
        )
        cluster_keep = ~all_non_spiking                    # (S,H,C)
        q_keep = cluster_keep[..., None].expand(S, H, C, K)
        kv_keep_real = cluster_keep[..., None] & (~non_spiking_mask)
        kv_keep_null = cluster_keep[..., None]              # (S,H,C,1)
        return q_keep, kv_keep_real, kv_keep_null, cluster_keep

    @torch.no_grad()
    def _build_varlen_masks_and_lengths(
        self,
        q_keep: torch.Tensor,
        kv_keep: torch.Tensor,
        S: int,
        H: int,
        Cn_q: int,
        Cn_k: int,
    ):
        q_keep_b  = q_keep.reshape(S * H, Cn_q)
        kv_keep_b = kv_keep.reshape(S * H, Cn_k)
        len_q = q_keep_b.sum(dim=1, dtype=torch.int32)
        len_k = kv_keep_b.sum(dim=1, dtype=torch.int32)
        keep_items = (len_q > 0) & (len_k > 0)
        return q_keep_b, kv_keep_b, len_q, len_k, keep_items

    # Centroids are features concatenated with random positions, initial positions are the same for both receiving and sending neurons
    @torch.no_grad()
    def _create_initial_centroids(self, num_centroids, num_heads, head_dim, position_weight = 0.999):
        input_centroid_features = torch.randn(num_heads, num_centroids, head_dim)
        output_centroid_features = torch.randn(num_heads, num_centroids, head_dim)
        input_centroid_features = input_centroid_features / input_centroid_features.norm(dim=-1, keepdim=True)
        output_centroid_features = output_centroid_features / output_centroid_features.norm(dim=-1, keepdim=True)
        # Centroid positions: sample uniform radius, theta, phi for points in unit sphere
        u = torch.rand(num_heads, num_centroids)
        v = torch.rand(num_heads, num_centroids)
        w = torch.rand(num_heads, num_centroids)
        r = u.pow(1/3)                          # radius: cube root for uniformity in volume
        theta = 2 * math.pi * v                # azimuthal angle: [0, 2pi)
        phi = torch.acos(2 * w - 1)            # polar angle: [0, pi]
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        centroid_positions = torch.stack([x, y, z], dim=-1)  # (num_heads, num_centroids, 3)

        # Final centroids are a weighted concatenation of random initial features and positions
        input_centroids = torch.cat([input_centroid_features  * (1 - position_weight), centroid_positions * position_weight], dim=-1)
        output_centroids = torch.cat([output_centroid_features  * (1 - position_weight), centroid_positions * position_weight], dim=-1)

        return input_centroids, output_centroids

    @torch.no_grad()
    def _directional_rope(self, positions):  # positions: (B, N, 3)
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
        proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)
        angles = proj * rope_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)

    @torch.no_grad()
    def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D)
        B_T, N, D = x.shape
        B = rope_emb.shape[0]
        T = B_T // B
        rope = self.rope_proj(rope_emb.to(dtype=x.dtype, device=x.device))  # (B,N,D)
        return x + rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)

    @torch.no_grad()
    def _calculate_cluster_cosine_scores(self, q, k, unit_point_positions):
        # q, k: (S, H, N, Dh) -- S is batch*timesteps, H heads, N neurons, Dh head dim
        # unit_point_positions: (B, N, 3) -- B is batch size
        # Compute cosine similarity of concatenated vectors [feat, pos] with spherical norm,
        # but avoid materializing (S,H,N,Dh+3) tensors. Keep fully vectorized, no loops.

        S, H, N, Dh = q.shape
        B = unit_point_positions.shape[0]
        T = S // B

        # Positions as (S,N,3). No expansion along H to save memory; rely on broadcasting in einsum.
        pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)  # (S,N,3)

        # Centroids (H,C,Dh+3) normalized; split into feature and position parts
        input_centroids_n  = F.normalize(self.input_centroids,  dim=-1)
        output_centroids_n = F.normalize(self.output_centroids, dim=-1)
        c_in_feat,  c_in_pos  = input_centroids_n[..., :Dh],  input_centroids_n[..., Dh:]
        c_out_feat, c_out_pos = output_centroids_n[..., :Dh], output_centroids_n[..., Dh:]

        # Token concat norms: ||[q,pos]|| per (S,H,N,1). Avoid expanding pos over H by broadcasting.
        q_norm2 = (q.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)          # (S,H,N,1)
        p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)     # (S,N,1)
        p_norm2 = p_norm2.unsqueeze(1)                                                # (S,1,N,1)
        concat_norm = (q_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=q.dtype)    # (S,H,N,1)

        # Dot products: (S,H,N,C) via broadcasting; no intermediate (S,H,N,Dh+3).
        q_dot_cin   = torch.einsum("b h n d, h c d -> b h n c", q,        c_in_feat)
        p_dot_cin   = torch.einsum("b n d,   h c d -> b h n c", pos_st,   c_in_pos)
        in_score    = (q_dot_cin + p_dot_cin) / concat_norm

        k_dot_cout  = torch.einsum("b h n d, h c d -> b h n c", k,        c_out_feat)
        p_dot_cout  = torch.einsum("b n d,   h c d -> b h n c", pos_st,   c_out_pos)
        out_score   = (k_dot_cout + p_dot_cout) / concat_norm

        # Zero-out rows for padded tokens (pos == 0) without allocating a full (S,H,N,C) mask.
        pos_zero = (pos_st == 0).all(dim=-1)                      # (S,N)
        if pos_zero.any():
            m = (~pos_zero).to(in_score.dtype).unsqueeze(1).unsqueeze(-1)  # (S,1,N,1)
            in_score  = in_score * m
            out_score = out_score * m
        return in_score, out_score

    @torch.no_grad()
    def _compute_input_scores(self, q, unit_point_positions):
        S, H, N, Dh = q.shape
        B = unit_point_positions.shape[0]
        T = S // B
        pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)
        # Ensure bf16 for routing matmuls
        q_bf16 = q.to(torch.bfloat16)
        input_centroids_n = F.normalize(self.input_centroids, dim=-1)
        c_in_feat, c_in_pos = input_centroids_n[..., :Dh], input_centroids_n[..., Dh:]
        q_norm2 = (q_bf16.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)
        p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True).unsqueeze(1)
        concat_norm = (q_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=q_bf16.dtype)
        q_dot_cin = torch.einsum("b h n d, h c d -> b h n c", q_bf16, c_in_feat)
        p_dot_cin = torch.einsum("b n d,   h c d -> b h n c", pos_st, c_in_pos)
        in_score = (q_dot_cin + p_dot_cin) / concat_norm
        pos_zero = (pos_st == 0).all(dim=-1)
        if pos_zero.any():
            m = (~pos_zero).to(in_score.dtype).unsqueeze(1).unsqueeze(-1)
            in_score = in_score * m
        return in_score

    @torch.no_grad()
    def _compute_output_scores(self, k, unit_point_positions):
        S, H, N, Dh = k.shape
        B = unit_point_positions.shape[0]
        T = S // B
        pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)
        # Ensure bf16 for routing matmuls
        k_bf16 = k.to(torch.bfloat16)
        output_centroids_n = F.normalize(self.output_centroids, dim=-1)
        c_out_feat, c_out_pos = output_centroids_n[..., :Dh], output_centroids_n[..., Dh:]
        k_norm2 = (k_bf16.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)
        p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True).unsqueeze(1)
        concat_norm = (k_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=k_bf16.dtype)
        k_dot_cout = torch.einsum("b h n d, h c d -> b h n c", k_bf16, c_out_feat)
        p_dot_cout = torch.einsum("b n d,   h c d -> b h n c", pos_st, c_out_pos)
        out_score = (k_dot_cout + p_dot_cout) / concat_norm
        pos_zero = (pos_st == 0).all(dim=-1)
        if pos_zero.any():
            m = (~pos_zero).to(out_score.dtype).unsqueeze(1).unsqueeze(-1)
            out_score = out_score * m
        return out_score

    @torch.no_grad()
    def _topk_indices_from_scores(self, scores: torch.Tensor, k: int):
        # scores: (S,H,N,C) → return (S,H,C,K) top-k indices along N
        return torch.topk(scores.permute(0, 1, 3, 2), k=k, dim=-1)[1]

    @torch.no_grad()
    def _topk_indices_blocked(self, compute_scores_fn, q_or_k: torch.Tensor, unit_point_positions: torch.Tensor, k: int, C: int, blocks: int = 4):
        # Vectorized two-stage top-k across blocks of clusters.
        # Returns (S,H,C,K) indices along N.
        S, H, N, Dh = q_or_k.shape
        block_size = max(1, (C + blocks - 1) // blocks)
        # First pass: per-block top-k (S,H,block_C,K)
        topk_vals = []
        topk_idxN = []
        for c0 in range(0, C, block_size):
            c1 = min(c0 + block_size, C)
            # Compute scores for the slice of centroids by temporarily slicing centroids buffers
            if compute_scores_fn is self._compute_input_scores:
                # Temporarily slice centroids
                full = self.input_centroids
                self.input_centroids = full[:, c0:c1]
                scores_block = self._compute_input_scores(q_or_k, unit_point_positions)  # (S,H,N,block_C)
                self.input_centroids = full
            else:
                full = self.output_centroids
                self.output_centroids = full[:, c0:c1]
                scores_block = self._compute_output_scores(q_or_k, unit_point_positions)  # (S,H,N,block_C)
                self.output_centroids = full

            # Top-k along N for this block
            vals, idx = torch.topk(scores_block.permute(0, 1, 3, 2), k=k, dim=-1)
            topk_vals.append(vals)        # (S,H,block_C,K)
            topk_idxN.append(idx)         # (S,H,block_C,K)

        # Concatenate across blocks on C dimension
        vals_all = torch.cat(topk_vals, dim=2)       # (S,H,C,K)
        idxN_all = torch.cat(topk_idxN, dim=2)       # (S,H,C,K)

        # Global top-k across N across blocks: merge K candidates per block → still need only indices, so take topk over vals
        # vals_all corresponds to scores at positions idxN_all per (S,H,C,·)
        # We already have top-k per block; to get exact top-k across blocks, take top-k over K*blocks candidates.
        Kb = vals_all.size(-1)
        vals_flat = vals_all.reshape(S, H, C, Kb)
        idx_candidates = idxN_all.reshape(S, H, C, Kb)
        vals_top, idx_top_inKb = torch.topk(vals_flat, k=k, dim=-1)
        # Gather corresponding indices in N
        idx_final = torch.gather(idx_candidates, -1, idx_top_inKb)
        return idx_final  # (S,H,C,K)

    @torch.no_grad()
    def _calculate_neuron_top_indices(self, input_centroid_cosine_score, output_centroid_cosine_score):
        # input_centroid_cosine_score: (B, n_heads, N, num_clusters_per_head)
        # output_centroid_cosine_score: (B, n_heads, N, num_clusters_per_head)
        # For each (B, n_heads, num_clusters_per_head), select indices of the top-k scoring neurons along the *neuron* dimension.
        k = self.neuron_cluster_size  # number of top neurons to select per cluster

        # Get shape info
        B, n_heads, N, num_clusters_per_head = input_centroid_cosine_score.shape

        # Permute to bring num_clusters_per_head as the "items" of interest, then topk over neurons
        # For the *receiving* neurons: topk over N
        # Result: (B, n_heads, num_clusters_per_head, k) -- indices in N
        input_neuron_indices = torch.topk(
            input_centroid_cosine_score.permute(0,1,3,2), k=k, dim=-1
        )[1]  # (B, n_heads, num_clusters_per_head, k)

        # For the *sending* neurons: topk over N
        output_neuron_indices = torch.topk(
            output_centroid_cosine_score.permute(0,1,3,2), k=k, dim=-1
        )[1]  # (B, n_heads, num_clusters_per_head, k)

        return input_neuron_indices, output_neuron_indices


    @torch.no_grad()
    def _build_cluster_tensors(self, q, k, v, input_cluster_neuron_indices, output_cluster_neuron_indices):
        """
        Gather the top-k neurons per cluster for Q (receivers) and KV (senders).
        NOTE: we DO NOT append the null token here; we do it later after masking so it never
        affects cluster keep/drop decisions.
        """
        _, _, num_clusters_per_head, k_sel = input_cluster_neuron_indices.shape
        D = q.size(-1)

        # Q clusters: gather along neuron axis
        input_clusters_q = torch.gather(
            q.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
            3,
            input_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)       # (S,H,C,K,D)
        )

        # K clusters
        output_clusters_k = torch.gather(
            k.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
            3,
            output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)      # (S,H,C,K,D)
        )

        # V clusters
        output_clusters_v = torch.gather(
            v.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
            3,
            output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)      # (S,H,C,K,D)
        )

        return input_clusters_q, output_clusters_k, output_clusters_v

    @torch.no_grad()
    def _get_indicies_of_non_spiking_neurons_in_clusters(self, output_cluster_neuron_indices, sending_neurons_mask_bt):
        # output_cluster_neuron_indices: (B, n_heads, num_clusters_per_head, k)
        # sending_neurons_mask_bt: (B, N)

        # For each cluster, check which neurons are non-spiking. 
        B, n_heads, num_clusters_per_head, k = output_cluster_neuron_indices.shape

        # Expand mask so we can gather using neuron indices
        sending_neurons_mask_expanded = sending_neurons_mask_bt.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)

        # For each item in output_cluster_neuron_indices, gather the mask (True if spiking, False otherwise)
        gathered_mask = torch.gather(
            sending_neurons_mask_expanded.expand(B, n_heads, num_clusters_per_head, -1),  # (B, n_heads, num_clusters_per_head, N)
            3,
            output_cluster_neuron_indices
        )  # (B, n_heads, num_clusters_per_head, k)

        # Invert mask: True where neuron is not spiking
        non_spiking_mask = ~gathered_mask.bool()  # (B, n_heads, num_clusters_per_head, k)

        # Create a bool mask for clusters which are all non-spiking (along k dim)
        all_non_spiking_in_cluster_mask = non_spiking_mask.all(dim=-1)  # (B, n_heads, num_clusters_per_head), bool

        # The output:
        # non_spiking_mask: (B, n_heads, num_clusters_per_head, k) -- True at (i,j,c,x) if x-th neuron in c-th cluster is non-spiking
        # all_non_spiking_in_cluster_mask: (B, n_heads, num_clusters_per_head) -- True if all neurons in cluster are non-spiking

        return non_spiking_mask, all_non_spiking_in_cluster_mask
        
    @torch.no_grad()
    def _ema_update_centroids(
        self,
        q: torch.Tensor,                      # (S,H,N,Dh)
        k: torch.Tensor,                      # (S,H,N,Dh)
        unit_point_positions: torch.Tensor,   # (B,N,3), L2-normalized
        input_neuron_indices: torch.Tensor,   # (S,H,C,K)  -> indices into N for Q
        output_neuron_indices: torch.Tensor,  # (S,H,C,K)  -> indices into N for K
        ):
        """
        Exponential moving average update for:
        - self.input_centroids  (H,C,Dh+3)  using Q + positions
        - self.output_centroids (H,C,Dh+3)  using K + positions

        Uses only the selected K neurons per centroid (top-k per cluster).
        Skips padded tokens whose position is exactly (0,0,0).
        """
        device = q.device
        dtype  = q.dtype
        S, H, N, Dh = q.shape
        _, _, C, K = input_neuron_indices.shape
        B3 = unit_point_positions.shape  # (B,N,3)

        # Expand positions from (B,N,3) -> (S,N,3) to align with q/k (S=B*T, N)
        B = B3[0]
        T = S // B
        pos_bt = unit_point_positions.to(device=device, dtype=dtype)              # (B,N,3)
        pos_st = pos_bt.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)          # (S,N,3)

        # Helper: gather per (S,H,C,K) from (S,N,·)
        def _gather_feats(feats_shnd, idx_shck):
            # feats_shnd: (S,H?,N,D) or (S,N,3) broadcast to (S,H,C,N,D)
            if feats_shnd.dim() == 4:  # (S,H,N,Dh)
                S_, H_, N_, D_ = feats_shnd.shape
                # (S,H,1,N,D) -> (S,H,C,N,D)
                feats_exp = feats_shnd.unsqueeze(2).expand(S_, H_, C, N_, D_)
            else:  # positions: (S,N,3)
                S_, N_, D_ = feats_shnd.shape
                # (S,1,1,N,3) -> (S,H,C,N,3)
                feats_exp = feats_shnd.unsqueeze(1).unsqueeze(2).expand(S_, H, C, N_, D_)
            return torch.gather(
                feats_exp,
                3,
                idx_shck.unsqueeze(-1).expand(-1, -1, -1, -1, feats_exp.size(-1))         # (S,H,C,K,D)
            )  # -> (S,H,C,K,D)

        # Gather Q/K features and corresponding positions for selected indices
        q_sel = _gather_feats(q, input_neuron_indices)         # (S,H,C,K,Dh)
        k_sel = _gather_feats(k, output_neuron_indices)        # (S,H,C,K,Dh)
        p_q   = _gather_feats(pos_st, input_neuron_indices)    # (S,H,C,K,3)
        p_k   = _gather_feats(pos_st, output_neuron_indices)   # (S,H,C,K,3)

        # Mask out padded tokens: position == (0,0,0)
        pad_q = (p_q.abs().sum(dim=-1) == 0)                   # (S,H,C,K)
        pad_k = (p_k.abs().sum(dim=-1) == 0)                   # (S,H,C,K)

        # Build augmented vectors [feat, pos] and L2-normalize
        def _augment_norm(feat, pos):
            aug = torch.cat([feat, pos], dim=-1)               # (S,H,C,K,Dh+3)
            aug = F.normalize(aug, dim=-1)
            return aug

        q_aug = _augment_norm(q_sel, p_q)                      # (S,H,C,K,Dh+3)
        k_aug = _augment_norm(k_sel, p_k)                      # (S,H,C,K,Dh+3)

        # Zero-out padded entries before summing
        q_aug = q_aug.masked_fill(pad_q.unsqueeze(-1), 0.0)
        k_aug = k_aug.masked_fill(pad_k.unsqueeze(-1), 0.0)

        # Sum over batch and K, then normalize by counts (avoid div by zero)
        def _reduce_to_centroid_mean(aug, pad_mask):
            counts = (~pad_mask).sum(dim=(0, 2, 3), keepdim=False)              # (H,)
            # Per-centroid counts:
            cnt_hc = (~pad_mask).sum(dim=(0, 3), keepdim=False)                 # (H,C)
            # Sum over S and K
            sum_hck = aug.sum(dim=(0, 3))                                       # (H,C,Dh+3)
            # If a centroid has zero valid samples in this step, skip its update later
            return sum_hck, cnt_hc

        sum_q, cnt_q = _reduce_to_centroid_mean(q_aug, pad_q)   # (H,C,Dh+3), (H,C)
        sum_k, cnt_k = _reduce_to_centroid_mean(k_aug, pad_k)   # (H,C,Dh+3), (H,C)

        # Compute means where count > 0
        eps = 1e-6
        mean_q = sum_q / cnt_q.clamp_min(1).unsqueeze(-1)       # (H,C,Dh+3)
        mean_k = sum_k / cnt_k.clamp_min(1).unsqueeze(-1)       # (H,C,Dh+3)

        # Current centroids
        mu_q = self.input_centroids.to(device=device, dtype=dtype)    # (H,C,Dh+3)
        mu_k = self.output_centroids.to(device=device, dtype=dtype)   # (H,C,Dh+3)

        # EMA
        decay = self.ema_decay
        upd_q = torch.where(
            (cnt_q > 0).unsqueeze(-1),
            decay * mu_q + (1.0 - decay) * mean_q,
            mu_q,                                                   # no-op if no samples
        )
        upd_k = torch.where(
            (cnt_k > 0).unsqueeze(-1),
            decay * mu_k + (1.0 - decay) * mean_k,
            mu_k,
        )

        # Re-normalize to unit length for spherical k-means behavior
        upd_q = F.normalize(upd_q, dim=-1)
        upd_k = F.normalize(upd_k, dim=-1)

        # Write back
        self.input_centroids.copy_(upd_q)
        self.output_centroids.copy_(upd_k)

            


    def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
        # x: (B, T, N, Dmodel)
        # point_positions: (B, N, 3)
        # spike_mask: (B, T, N)  True where neuron fired at that time step

        B, T, N, Dmodel = x.shape
        H = self.n_heads
        Dh = self.head_dim
        C = self.num_clusters_per_head
        K = self.neuron_cluster_size  # <-- fixed

        S = B * T
        device = x.device
        dtype = x.dtype

        # Optional memory profiler helpers (CUDA only)
        def _mem_ckpt(tag: str):
            if self.profile_memory and torch.cuda.is_available():
                try:
                    import torch._dynamo as _d
                    if hasattr(_d, 'is_compiling') and _d.is_compiling():
                        return
                except Exception:
                    pass
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated(x.device)
                reserved = torch.cuda.memory_reserved(x.device)
                print(f"[SpikeSparseConnectomeAttention][mem] {tag}: alloc={alloc/1e9:.2f}GB reserved={reserved/1e9:.2f}GB")

        _mem_ckpt("start")

        # --------- Projections ----------
        x_bt = x.reshape(S, N, Dmodel)                    # (S,N,D)
        sending_neurons_mask_bt = spike_mask.reshape(S, N)

        unit_point_positions = F.normalize(point_positions, dim=-1)  # (B,N,3)
        res = x_bt
        xn = self.norm(x_bt)

        rope_emb = self._directional_rope(unit_point_positions)       # (B,N,2F)
        qk_in = self._apply_rope(xn, rope_emb)                        # add RoPE to Q/K inputs

        q = self.q_proj(qk_in)    # (S,N,D)
        k = self.k_proj(qk_in)    # (S,N,D)  (null token will NOT use RoPE)
        v = self.v_proj(xn)       # (S,N,D)
        _mem_ckpt("after_projections")

        # Shape to (S,H,N,Dh)
        q = q.view(S, H, N, Dh)
        k = k.view(S, H, N, Dh)
        v = v.view(S, H, N, Dh)

        # --------- Clustering ----------
        # Compute input/output scores separately (no_grad) to reduce peak memory
        with torch.no_grad():
            # Blocked top-k to reduce peak memory during routing
            input_idx = self._topk_indices_blocked(self._compute_input_scores, q, unit_point_positions, K, C, blocks=4)
        _mem_ckpt("after_input_topk")
        if self.profile_memory and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        _mem_ckpt("after_input_topk")

        with torch.no_grad():
            output_idx = self._topk_indices_blocked(self._compute_output_scores, k, unit_point_positions, K, C, blocks=4)
        _mem_ckpt("after_output_topk")
        if self.profile_memory and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        _mem_ckpt("after_output_topk")


        # === EMA centroid update (training only) ===
        if self.training:
            self._ema_update_centroids(
                q, k, unit_point_positions,   # (S,H,N,Dh), (S,H,N,Dh), (B,N,3)
                input_idx,                    # (S,H,C,K)
                output_idx,                   # (S,H,C,K)
            )


        # Build cluster tensors (no null yet)
        q_cl, k_cl, v_cl = self._build_cluster_tensors(q, k, v, input_idx, output_idx)    # (S,H,C,K,Dh)
        _mem_ckpt("after_build_clusters")


        # --------- Spiking masks ----------
        q_keep, kv_keep_real, kv_keep_null, cluster_keep = self._build_spiking_keep_masks(
            output_idx, sending_neurons_mask_bt, S, H, C, K
        )

        # --------- Append the NULL output token (per kept cluster) ----------
        # Pass the learned null vector through K and V projections and split into heads
        null_k_model = self.k_proj(self.null_output_vec.to(device=device, dtype=dtype))   # (Dmodel,)
        null_v_model = self.v_proj(self.null_output_vec.to(device=device, dtype=dtype))   # (Dmodel,)
        null_k = null_k_model.view(H, Dh)  # (H,Dh)
        null_v = null_v_model.view(H, Dh)  # (H,Dh)

        # Expand to (S,H,C,1,Dh) and append as the LAST token in each output cluster
        null_k_exp = null_k.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(S, H, C, 1, Dh)
        null_v_exp = null_v.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(S, H, C, 1, Dh)

        k_cl_aug = torch.cat([k_cl, null_k_exp], dim=3)  # (S,H,C,K+1,Dh)
        v_cl_aug = torch.cat([v_cl, null_v_exp], dim=3)  # (S,H,C,K+1,Dh)
        _mem_ckpt("after_null_append")

        # Final KV keep mask (real + null). The null token is NEVER removed from remaining clusters.
        kv_keep = torch.cat([kv_keep_real, kv_keep_null], dim=-1)  # (S,H,C,K+1)

        # --------- Pack to varlen (allowing different Q/K lengths) ----------
        Cn_q = C * K
        Cn_k = C * (K + 1)

        def _reshape_to_bcn(t):  # (S,H,C,?,Dh) -> (S*H, C*?, Dh)
            S_, H_, C_, KK_, Dh_ = t.shape
            return t.reshape(S_*H_, C_*KK_, Dh_)

        q_bcn  = _reshape_to_bcn(q_cl)          # (S*H, Cn_q, Dh)
        k_bcn  = _reshape_to_bcn(k_cl_aug)      # (S*H, Cn_k, Dh)
        v_bcn  = _reshape_to_bcn(v_cl_aug)      # (S*H, Cn_k, Dh)
        _mem_ckpt("after_pack_varlen_inputs")

        q_keep_b, kv_keep_b, len_q, len_k, keep_items = self._build_varlen_masks_and_lengths(
            q_keep, kv_keep, S, H, Cn_q, Cn_k
        )
        if not keep_items.any():
            # Nothing to attend to for any (sample, head)
            return x

        q_keep_b  = q_keep_b[keep_items]
        kv_keep_b = kv_keep_b[keep_items]
        q_bcn     = q_bcn[keep_items]
        k_bcn     = k_bcn[keep_items]
        v_bcn     = v_bcn[keep_items]
        len_q     = len_q[keep_items]
        len_k     = len_k[keep_items]
        kept_item_ids = torch.arange(S*H, device=device)[keep_items]

        # Flatten to kept tokens
        mask_q_flat  = q_keep_b.reshape(-1)
        mask_kv_flat = kv_keep_b.reshape(-1)

        q_kept = q_bcn.reshape(-1, Dh)[mask_q_flat]   # (total_q, Dh)
        k_kept = k_bcn.reshape(-1, Dh)[mask_kv_flat]  # (total_k, Dh)
        v_kept = v_bcn.reshape(-1, Dh)[mask_kv_flat]  # (total_k, Dh)

        # Varlen metadata
        B_eff = q_keep_b.size(0)
        cu_seqlens_q = torch.zeros(B_eff + 1, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.zeros(B_eff + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = torch.cumsum(len_q, dim=0)
        cu_seqlens_k[1:] = torch.cumsum(len_k, dim=0)
        max_seqlen_q = int(len_q.max().item())
        max_seqlen_k = int(len_k.max().item())

        # FlashAttention-2 varlen (nheads=1; head-as-batch) with optional chunking
        max_batch = MAX_BATCH_SIZE
        p_drop = self.dropout if self.training else 0.0

        # Directly accumulate outputs into neuron space to avoid large intermediate buffers
        out_sum = torch.zeros(S, H, N, Dh, device=device, dtype=v_bcn.dtype)
        counts = torch.zeros(S, H, N, 1, device=device, dtype=v_bcn.dtype)
        idx_flat_full = input_idx.reshape(S * H, C * K)  # (S*H, C*K)
        out_sum_lin = out_sum.view(S * H * N, Dh)
        counts_lin = counts.view(S * H * N)

        if B_eff <= max_batch:
            q_packed  = q_kept.unsqueeze(1).contiguous()  # (total_q, 1, Dh)
            kv_packed = torch.stack([k_kept, v_kept], dim=1).unsqueeze(2).contiguous()  # (total_k, 2, 1, Dh)

            attn_out = flash_attn_varlen_kvpacked_func(
                q_packed, kv_packed,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                p_drop,
                softmax_scale=None,
                causal=False
            ).squeeze(1)  # (total_q, Dh)
            _mem_ckpt("after_flash_attn_chunk_or_full")

            cols = torch.arange(Cn_q, device=device).expand(B_eff, Cn_q)
            col_ids = cols[q_keep_b]                        # (total_q,)
            row_ids = torch.repeat_interleave(torch.arange(B_eff, device=device), len_q)
            t_orig = kept_item_ids[row_ids]                 # (total_q,)
            lin_neuron = idx_flat_full[t_orig, col_ids]      # (total_q,)
            lin_index = t_orig * N + lin_neuron
            out_sum_lin.index_add_(0, lin_index, attn_out.to(out_sum_lin.dtype))
            counts_lin.index_add_(0, lin_index, torch.ones(lin_index.numel(), device=device, dtype=counts_lin.dtype))
        else:
            # Chunk over sequence items so each chunk has at most max_batch sequences
            for start in range(0, B_eff, max_batch):
                end = min(start + max_batch, B_eff)
                n_seq = end - start

                # Slice masks and tensors for this chunk
                q_keep_chunk = q_keep_b[start:end]
                kv_keep_chunk = kv_keep_b[start:end]
                len_q_chunk = len_q[start:end]
                len_k_chunk = len_k[start:end]
                q_bcn_chunk = q_bcn[start:end]
                k_bcn_chunk = k_bcn[start:end]
                v_bcn_chunk = v_bcn[start:end]

                # Flatten kept tokens for this chunk
                mask_q_flat_chunk = q_keep_chunk.reshape(-1)
                mask_kv_flat_chunk = kv_keep_chunk.reshape(-1)
                q_kept_chunk = q_bcn_chunk.reshape(-1, Dh)[mask_q_flat_chunk]
                k_kept_chunk = k_bcn_chunk.reshape(-1, Dh)[mask_kv_flat_chunk]
                v_kept_chunk = v_bcn_chunk.reshape(-1, Dh)[mask_kv_flat_chunk]

                # Build varlen metadata for this chunk
                cu_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=device)
                cu_k = torch.zeros(n_seq + 1, dtype=torch.int32, device=device)
                cu_q[1:] = torch.cumsum(len_q_chunk, dim=0)
                cu_k[1:] = torch.cumsum(len_k_chunk, dim=0)
                max_q = int(len_q_chunk.max().item())
                max_k = int(len_k_chunk.max().item())

                q_packed_chunk = q_kept_chunk.unsqueeze(1).contiguous()
                kv_packed_chunk = torch.stack([k_kept_chunk, v_kept_chunk], dim=1).unsqueeze(2).contiguous()

                attn_out_chunk = flash_attn_varlen_kvpacked_func(
                    q_packed_chunk, kv_packed_chunk,
                    cu_q, cu_k,
                    max_q, max_k,
                    p_drop,
                    softmax_scale=None,
                    causal=False
                ).squeeze(1)  # (total_q_chunk, Dh)
                _mem_ckpt(f"after_flash_attn_chunk[{start}:{end}]")

                # Direct scatter for this chunk
                cols_chunk = torch.arange(Cn_q, device=device).expand(n_seq, Cn_q)
                col_ids_chunk = cols_chunk[q_keep_chunk]
                row_ids_chunk = torch.repeat_interleave(torch.arange(n_seq, device=device), len_q_chunk)
                t_orig_chunk = kept_item_ids[start:end][row_ids_chunk]
                lin_neuron_chunk = idx_flat_full[t_orig_chunk, col_ids_chunk]
                lin_index_chunk = t_orig_chunk * N + lin_neuron_chunk
                out_sum_lin.index_add_(0, lin_index_chunk, attn_out_chunk.to(out_sum_lin.dtype))
                counts_lin.index_add_(0, lin_index_chunk, torch.ones(lin_index_chunk.numel(), device=device, dtype=counts_lin.dtype))

        _mem_ckpt("after_scatter_q_positions")

        out_shnd = out_sum / counts.clamp_min(1.0)  # (S,H,N,Dh)

        # Final projection & residual
        out = out_shnd.reshape(B, T, H, N, Dh).permute(0, 1, 3, 2, 4).reshape(B, T, N, H*Dh)
        out = self.o_proj(out)
        out = out + x  # residual
        _mem_ckpt("end")
        return out


class SparseSpikeFullAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_rope_features: int = 32, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 3D RoPE
        self.n_rope_features = int(n_rope_features)
        dirs = torch.randn(self.n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        freqs = torch.logspace(math.log10(1.0), math.log10(10000.0), self.n_rope_features)
        self.register_buffer('rope_dirs', dirs, persistent=False)
        self.register_buffer('rope_freqs', freqs, persistent=False)

    @torch.no_grad()
    def _directional_rope(self, positions):  # positions: (B, N, 3)
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
        proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)
        angles = proj * rope_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)

    @torch.no_grad()
    def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D)
        B_T, N, D = x.shape
        B = rope_emb.shape[0]
        T = B_T // B
        # Project rope to model dim (lightweight, front-fill)
        rope = torch.zeros(B, N, D, dtype=x.dtype, device=x.device)
        F2 = rope_emb.shape[-1]
        rope[..., :F2] = rope_emb.to(dtype=x.dtype, device=x.device)
        return x + rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)

    def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
        # x: (B,T,N,D), positions: (B,N,3), masks as described
        B, T, N, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        S = B * T

        # Normalize and RoPE for Q/K inputs
        xn = self.norm(x)
        unit_pos = F.normalize(point_positions, dim=-1)
        rope_emb = self._directional_rope(unit_pos)
        xn_bt = xn.view(B, T, N, D).reshape(S, N, D)
        qk_in = self._apply_rope(xn_bt, rope_emb)

        # Build masks and compact indices BEFORE projections
        keep_bt = (neuron_pad_mask != 0).unsqueeze(1).expand(B, T, N).reshape(S, N)
        send_bt = (spike_mask != 0).reshape(S, N) & keep_bt

        lens_q = keep_bt.sum(dim=1, dtype=torch.int32)
        lens_k = send_bt.sum(dim=1, dtype=torch.int32)
        cu_q = torch.zeros(S + 1, dtype=torch.int32, device=x.device); cu_q[1:] = torch.cumsum(lens_q, 0)
        cu_k = torch.zeros(S + 1, dtype=torch.int32, device=x.device); cu_k[1:] = torch.cumsum(lens_k, 0)
        max_q = int(lens_q.max().item()) if S > 0 else 0
        max_k = int(lens_k.max().item()) if S > 0 else 0

        # Flatten indices
        arange_SN = torch.arange(S * N, device=x.device, dtype=torch.long).reshape(S, N)
        idx_q = arange_SN[keep_bt].reshape(-1)
        idx_kv = arange_SN[send_bt].reshape(-1)

        # Compact tensors
        qk_in_flat = qk_in.reshape(S * N, D)
        xn_flat = xn_bt.reshape(S * N, D)
        pre_q = qk_in_flat.index_select(0, idx_q)          # (total_q, D)
        pre_k = qk_in_flat.index_select(0, idx_kv)         # (total_k, D)
        pre_v = xn_flat.index_select(0, idx_kv)            # (total_k, D)

        # Projections
        q = self.q_proj(pre_q).view(-1, H, Dh)             # (total_q,H,Dh)
        k = self.k_proj(pre_k).view(-1, H, Dh)             # (total_k,H,Dh)
        v = self.v_proj(pre_v).view(-1, H, Dh)             # (total_k,H,Dh)

        # Pack and run FlashAttention-2 varlen
        q_packed = q.contiguous()                          # (total_q,H,Dh)
        kv_packed = torch.stack([k, v], dim=1).contiguous()  # (total_k,2,H,Dh)
        p_drop = self.dropout if self.training else 0.0
        attn_out = flash_attn_varlen_kvpacked_func(
            q_packed, kv_packed,
            cu_q, cu_k,
            max_q, max_k,
            p_drop,
            softmax_scale=None,
            causal=False
        )  # (total_q,H,Dh)

        # Scatter back to (S,N,D)
        out_heads = torch.zeros(S * N, H, Dh, device=x.device, dtype=attn_out.dtype)
        out_heads.index_copy_(0, idx_q, attn_out)
        out_D = out_heads.view(S * N, D)
        out = self.o_proj(out_D).view(B, T, N, D)
        return out + x


        
        
class NeuronCausalAttention(nn.Module):
    """
    Causal self-attention over time per neuron independently.
    - Operates on each neuron timeline (length T) with causal attention.
    - Respects neuron_pad_mask (B,N): skip padded neurons entirely to save compute.
    - Spike mask is ignored.
    - Launches at most (2^16 - 1) sequences per FLASH call by chunking rows.
    """
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0,
                 max_bh_per_call: int = (2**18 )):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout_p)
        self.max_bh_per_call = int(max_bh_per_call)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _to_bhld(self, x):  # (B*, L, D) -> (B*, H, L, Dh)
        Bx, L, D = x.shape
        return x.view(Bx, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _from_bhld(self, x):  # (B*, H, L, Dh) -> (B*, L, D)
        Bx, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(Bx, L, H * Dh)

    def forward(self, x: torch.Tensor,
                neuron_pad_mask: torch.Tensor ):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        xn = self.norm(x)
        xt = xn.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N, T, D)
        res = xt

        # Row filter from neuron_pad_mask
        valid_rows = (neuron_pad_mask.reshape(-1) != 0)

        out = torch.zeros_like(xt)
        if valid_rows.any():
            q_in = xt[valid_rows]
            k_in = q_in
            v_in = q_in

            q = self._to_bhld(self.q_proj(q_in))
            k = self._to_bhld(self.k_proj(k_in))
            v = self._to_bhld(self.v_proj(v_in))

            # Ensure FLASH dtype
            if q.dtype not in (torch.bfloat16, torch.float16):
                q = q.to(torch.bfloat16)
            if k.dtype not in (torch.bfloat16, torch.float16):
                k = k.to(torch.bfloat16)
            if v.dtype not in (torch.bfloat16, torch.float16):
                v = v.to(torch.bfloat16)

            Bv, H, L, Dh = q.shape
            if H > self.max_bh_per_call:
                raise RuntimeError(
                    f"n_heads={H} exceeds max_bh_per_call={self.max_bh_per_call}; reduce heads or increase cap"
                )
            # Choose row chunk so (rows_chunk * H) <= max_bh_per_call
            rows_step = max(1, min(Bv, self.max_bh_per_call // H))
            out_chunks = []
            for i in range(0, Bv, rows_step):
                qs = q[i:i + rows_step]
                ks = k[i:i + rows_step]
                vs = v[i:i + rows_step]
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out_chunk = F.scaled_dot_product_attention(
                        qs, ks, vs, is_causal=True, dropout_p=self.dropout_p
                    )
                out_chunks.append(out_chunk)
            out_valid = torch.cat(out_chunks, dim=0)  # (Bv,H,L,Dh)
            out_valid = self._from_bhld(out_valid).to(res.dtype)
            out[valid_rows] = out_valid

        out = out + res
        out = out.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return self.o_proj(out)
        