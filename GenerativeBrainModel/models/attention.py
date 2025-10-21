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


class SpikeSparseConnectomeAttention(nn.Module):
    """
    Attention which uses routing to split all neuron tokens into receiving and sending neuron groups. All receiving neurons are included in Q, only sending neurons which are deemed spiking are included in KV. Using FlashAttention-2's flash_varlen_qkv.    """
    def __init__(self, d_model, n_heads, neuron_cluster_size, num_clusters_per_head, ema_decay: float = 0.992, n_rope_features: int = 32):
        super().__init__()

        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        assert self.head_dim % 8 == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.neuron_cluster_size = neuron_cluster_size
        self.num_clusters_per_head = num_clusters_per_head
        self.ema_decay = ema_decay
        self.n_rope_features = n_rope_features
        self.total_cluster_count = n_heads * num_clusters_per_head

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


        # Spherical routing norm over shared routing features (per-token Dh)
        self.route_ln = nn.LayerNorm(self.head_dim, elementwise_affine=False)

        # Fixed spatial centroid bank (features concatenated with positions), two sets, one for sending neurons and one for receiving neurons, such that they overlap initially
        input_centroids, output_centroids = self._create_initial_centroids(self.num_clusters_per_head, self.n_heads, self.head_dim)
        self.register_buffer("input_centroids", input_centroids, persistent=True)           # (self.n_heads, self.num_clusters_per_head, self.head_dim + 3 for positions)
        self.register_buffer("output_centroids", output_centroids, persistent=True)           # (self.n_heads, self.num_clusters_per_head, self.head_dim + 3 for positions)

    # Centroids are features concatenated with random positions, initial positions are the same for both receiving and sending neurons
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

    def _calculate_cluster_cosine_scores(self, q, k, unit_point_positions, ):
        # q: (B, n_heads, N, D)
        # k: (B, n_heads, N, D)
        # unit_point_positions: (B, N, 3)
        # input_centroids: (n_heads, num_clusters_per_head, D + 3)
        # output_centroids: (n_heads, num_clusters_per_head, D + 3)
        # Each output tensor should be shape (B, n_heads, N, num_clusters_per_head)
        # - for each input vector (q or k) in the batch/location, give the score to each centroid per head

        B, n_heads, N, D = q.shape

        # Expand unit_point_positions for n_heads
        positions = unit_point_positions.unsqueeze(1).expand(B, n_heads, N, 3)

        q_vecs = F.normalize(torch.cat([q, positions], dim=-1), dim=-1)
        k_vecs = F.normalize(torch.cat([k, positions], dim=-1), dim=-1)

        input_centroids = F.normalize(self.input_centroids, dim=-1)     # (n_heads, num_clusters_per_head, D+3)
        output_centroids = F.normalize(self.output_centroids, dim=-1)   # (n_heads, num_clusters_per_head, D+3)

        # Compute cosine similarity between each vector (q/k) and each centroid for each head
        input_centroid_cosine_score = torch.einsum(
            "b h n d, h k d -> b h n k",
            q_vecs,                                   # (B, n_heads, N, D+3)
            input_centroids                           # (n_heads, num_clusters_per_head, D+3)
        )
        output_centroid_cosine_score = torch.einsum(
            "b h n d, h k d -> b h n k",
            k_vecs,                                   # (B, n_heads, N, D+3)
            output_centroids                          # (n_heads, num_clusters_per_head, D+3)
        )

        # If position is (0, 0, 0), set the corresponding score to zero, since padded neurons have exactly zero position by setting their cosine score to zero they are excluded from the routed attention
        zero_position_mask = (positions == 0).all(dim=-1)  # (B, n_heads, N)
        # Broadcast mask to (B, n_heads, N, num_clusters_per_head) such that we can mask the scores
        zero_position_mask = zero_position_mask.unsqueeze(-1).expand(-1, -1, -1, input_centroids.shape[1])
        input_centroid_cosine_score = input_centroid_cosine_score.masked_fill(zero_position_mask, 0.0)
        output_centroid_cosine_score = output_centroid_cosine_score.masked_fill(zero_position_mask, 0.0)

        return input_centroid_cosine_score, output_centroid_cosine_score

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



    def _build_cluster_tensors(self, q, k, v, input_cluster_neuron_indices, output_cluster_neuron_indices):
        # q: (B, n_heads, N, D)
        # k: (B, n_heads, sending_neurons_N, D)
        # v: (B, n_heads, sending_neurons_N, D)
        # input_cluster_neuron_indices: (B, n_heads, num_clusters_per_head, k)
        # output_cluster_neuron_indices: (B, n_heads, num_clusters_per_head, k)
        # For each cluster, gather the top-k neurons (by index) and form new cluster tensors.
        # Neurons may appear in multiple clusters.
        _, _, num_clusters_per_head, cluster_neuron_size = input_cluster_neuron_indices.shape

        # Gather the top-k neurons for each cluster
        input_clusters_q = torch.gather(q.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1), 3, input_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, cluster_neuron_size))
        # input_clusters_q: (B, n_heads, num_clusters_per_head, cluster_neuron_size, D)
        output_clusters_k = torch.gather(k.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1), 3, output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, cluster_neuron_size))
        # output_clusters_k: (B, n_heads, num_clusters_per_head, cluster_neuron_size, D)
        output_clusters_v = torch.gather(v.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1), 3, output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, cluster_neuron_size))
        # output_clusters_v: (B, n_heads, num_clusters_per_head, cluster_neuron_size, D)

        return input_clusters_q, output_clusters_k, output_clusters_v

    def get_indicies_of_non_spiking_neurons_in_clusters(self, output_cluster_neuron_indices, sending_neurons_mask_bt):
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


    def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
        # Inputs:
        # x: (B, T, N, D)
        # point_positions: (B, N, 3)
        # neuron_pad_mask: (B, N)
        # spike_mask: (B, T, N)

        B, T, N, D = x.shape
        S = B * T


        x_bt = x.reshape(S, N, D)
        sending_neurons_mask_bt = spike_mask.reshape(S, N)


        # Normalize the point positions to the unit sphere
        unit_point_positions = F.normalize(point_positions, dim=-1)

        res = x_bt
        xn = self.norm(x_bt)

        rope_emb = self._directional_rope(unit_point_positions)
        qk = self._apply_rope(xn, rope_emb)  # RoPE on Q/K only

        q = self.q_proj(qk)
        k = self.k_proj(qk)
        v = self.v_proj(xn)

        # Add head dimension to the q k v tensors such that they have shape (B, n_heads, -1, D)
        q = q.view(B, self.n_heads, N, self.head_dim)
        k = k.view(B, self.n_heads, N, self.head_dim)
        v = v.view(B, self.n_heads, N, self.head_dim)

        # Calculate the cosine scores between the query and key vectors and the input and output centroids
        input_centroid_cosine_score, output_centroid_cosine_score = self._calculate_cluster_cosine_scores(
            q, k, unit_point_positions
        ) # (S, n_heads, N, num_clusters_per_head), (S, n_heads, N, num_clusters_per_head)
        
        # Calculate the top-k neurons for each cluster
        input_cluster_neuron_indices, output_cluster_neuron_indices = self._calculate_cluster_top_indices(
            input_centroid_cosine_score, output_centroid_cosine_score
        ) # (S, n_heads, num_clusters_per_head, k), (S, n_heads, num_clusters_per_head, k)
        # Build the cluster tensors
        input_clusters_q, output_clusters_k, output_clusters_v = self._build_cluster_tensors(
            q, k, v, input_cluster_neuron_indices, output_cluster_neuron_indices
        )  # (S, n_heads, num_clusters_per_head, cluster_neuron_size, D)

        # Get non spiking neuron indices in each cluster for output clusters aka kv
        non_spiking_mask, all_non_spiking_in_cluster_mask = self.get_indicies_of_non_spiking_neurons_in_clusters(
            output_cluster_neuron_indices, 
            sending_neurons_mask_bt
        ) # (S, n_heads, num_clusters_per_head, k), (S, n_heads, num_clusters_per_head)

        # We flatten the cluster tensors such that they have shape (S * num_clusters_per_head * cluster_neuron_size, n_heads, D)
        input_clusters_q = input_clusters_q.reshape(S * self.num_clusters_per_head * self.cluster_neuron_size, self.n_heads, D)
        output_clusters_k = output_clusters_k.reshape(S * self.num_clusters_per_head * self.cluster_neuron_size, self.n_heads, D)
        output_clusters_v = output_clusters_v.reshape(S * self.num_clusters_per_head * self.cluster_neuron_size, self.n_heads, D)

        # We combine the k and v tensors such that they have shape (S * num_clusters_per_head * cluster_neuron_size, 2, n_heads, D)
        output_clusters_kv = torch.stack([output_clusters_k, output_clusters_v], dim=1) # (S * num_clusters_per_head * cluster_neuron_size, 2, n_heads, D)

        # We now reshape the all_non_spiking_in_cluster_mask 
        all_non_spiking_in_cluster_mask = all_non_spiking_in_cluster_mask.reshape(S *  self.num_clusters_per_head, self.n_heads).expand(S * self.num_clusters_per_head * self.cluster_neuron_size, self.n_heads)
        # We now mask the output_clusters_kv tensor such that we only keep the non-spiking neurons
        output_clusters_kv = output_clusters_kv[~all_non_spiking_in_cluster_mask]
        




        


        return x