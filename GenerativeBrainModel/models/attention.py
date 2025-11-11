import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
import numpy as np

# from mamba_ssm import Mamba2 as Mamba  # kept for parity
from torch.nn.attention import sdpa_kernel, SDPBackend
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled

import math

# Require FlashAttention-2 varlen
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func as flash_varlen_qkv,
)

# Also get normal FlashAttention-2
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func

import torch.distributed as dist

MAX_BATCH_SIZE = 65535


# ------------------------- TorchDynamo compile helpers -------------------------
@torch._dynamo.disable
def _compute_max_lens_tensors(lens_q: torch.Tensor, lens_k: torch.Tensor):
    """Return Python ints for max lens without polluting the captured graph."""
    max_q = int(lens_q.max().item()) if lens_q.numel() > 0 else 0
    max_k = int(lens_k.max().item()) if lens_k.numel() > 0 else 0
    return max_q, max_k


@torch._dynamo.disable
def _build_varlen_metadata_from_masks(keep_bt: torch.Tensor, send_bt: torch.Tensor):
    """
    Build indices and cu_seqlens from boolean masks outside the compiled region.
    Returns: idx_q, idx_kv, cu_q, cu_k, lens_q, lens_k, max_q, max_k
    """
    S, N = keep_bt.shape
    device = keep_bt.device
    lens_q = keep_bt.sum(dim=1, dtype=torch.int32)
    lens_k = send_bt.sum(dim=1, dtype=torch.int32)
    cu_q = torch.zeros(S + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(S + 1, dtype=torch.int32, device=device)
    if S > 0:
        cu_q[1:] = torch.cumsum(lens_q, dim=0)
        cu_k[1:] = torch.cumsum(lens_k, dim=0)
    max_q, max_k = _compute_max_lens_tensors(lens_q, lens_k)
    arange_SN = torch.arange(S * N, device=device, dtype=torch.long).reshape(S, N)
    idx_q = arange_SN[keep_bt].reshape(-1).contiguous()
    idx_kv = arange_SN[send_bt].reshape(-1).contiguous()
    return idx_q, idx_kv, cu_q, cu_k, lens_q, lens_k, max_q, max_k


@torch._dynamo.disable
def _index_copy_rows(
    out: torch.Tensor, out_valid: torch.Tensor, valid_rows_mask: torch.Tensor
):
    """Avoid boolean index_put in compiled region by using index_copy with int indices."""
    if valid_rows_mask.dtype is not torch.bool:
        valid_rows_mask = valid_rows_mask != 0
    if not bool(valid_rows_mask.any().item()):
        return out
    row_idx = torch.where(valid_rows_mask)[0].to(dtype=torch.long)
    return out.index_copy(0, row_idx, out_valid)


@torch._dynamo.disable
def _indices_from_mask(mask_1d: torch.Tensor):
    """Return 1D long indices of True positions from a boolean mask, out of graph."""
    if mask_1d.dtype is not torch.bool:
        mask_1d = mask_1d != 0
    return torch.where(mask_1d)[0].to(dtype=torch.long).contiguous()


# class SpikeSparseConnectomeRoutingAttention(nn.Module):
#     """
#     Attention which uses routing to split all neuron tokens into receiving and sending neuron groups. All receiving neurons are included in Q, only sending neurons which are deemed spiking are included in KV. Using FlashAttention-2's flash_varlen_qkv.
#     """
#     def __init__(self, d_model, n_heads, neuron_cluster_size, num_clusters_per_head,
#                  ema_decay: float = 0.992, n_rope_features: int = 32, dropout: float = 0.0,
#                  profile_memory: bool = False):
#         super().__init__()

#         assert d_model % n_heads == 0
#         self.head_dim = d_model // n_heads
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.neuron_cluster_size = neuron_cluster_size
#         self.num_clusters_per_head = num_clusters_per_head
#         self.ema_decay = ema_decay
#         self.n_rope_features = n_rope_features
#         self.total_cluster_count = n_heads * num_clusters_per_head
#         self.dropout = dropout
#         self.profile_memory = bool(profile_memory)

#         self.norm = RMSNorm(d_model)
#         self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

#         dirs = torch.randn(n_rope_features, 3)
#         dirs = dirs / dirs.norm(dim=-1, keepdim=True)
#         freqs = torch.logspace(math.log10(1.0), math.log10(10000.0), n_rope_features)
#         self.register_buffer('rope_dirs', dirs, persistent=False)
#         self.register_buffer('rope_freqs', freqs, persistent=False)

#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.o_proj = nn.Linear(d_model, d_model, bias=False)


#         # Fixed centroid banks: ensure these tensors are constructed with no gradients tracked
#         input_centroids, output_centroids = self._create_initial_centroids(
#             self.num_clusters_per_head, self.n_heads, self.head_dim
#         )
#         # Store centroids in bf16 to reduce memory and bandwidth during routing
#         self.register_buffer("input_centroids", input_centroids.to(torch.bfloat16), persistent=True)
#         self.register_buffer("output_centroids", output_centroids.to(torch.bfloat16), persistent=True)

#         # === learned null output neuron token (lives in model space) ===
#         self.null_output_vec = nn.Parameter(torch.randn(self.d_model))

#     @torch.no_grad()
#     def _build_spiking_keep_masks(
#         self,
#         output_idx: torch.Tensor,
#         sending_neurons_mask_bt: torch.Tensor,
#         S: int,
#         H: int,
#         C: int,
#         K: int,
#     ):
#         non_spiking_mask, all_non_spiking = self._get_indicies_of_non_spiking_neurons_in_clusters(
#             output_idx, sending_neurons_mask_bt
#         )
#         cluster_keep = ~all_non_spiking                    # (S,H,C)
#         q_keep = cluster_keep[..., None].expand(S, H, C, K)
#         kv_keep_real = cluster_keep[..., None] & (~non_spiking_mask)
#         kv_keep_null = cluster_keep[..., None]              # (S,H,C,1)
#         return q_keep, kv_keep_real, kv_keep_null, cluster_keep

#     @torch.no_grad()
#     def _build_varlen_masks_and_lengths(
#         self,
#         q_keep: torch.Tensor,
#         kv_keep: torch.Tensor,
#         S: int,
#         H: int,
#         Cn_q: int,
#         Cn_k: int,
#     ):
#         q_keep_b  = q_keep.reshape(S * H, Cn_q)
#         kv_keep_b = kv_keep.reshape(S * H, Cn_k)
#         len_q = q_keep_b.sum(dim=1, dtype=torch.int32)
#         len_k = kv_keep_b.sum(dim=1, dtype=torch.int32)
#         keep_items = (len_q > 0) & (len_k > 0)
#         return q_keep_b, kv_keep_b, len_q, len_k, keep_items

#     # Centroids are features concatenated with random positions, initial positions are the same for both receiving and sending neurons
#     @torch.no_grad()
#     def _create_initial_centroids(self, num_centroids, num_heads, head_dim, position_weight = 0.999):
#         input_centroid_features = torch.randn(num_heads, num_centroids, head_dim)
#         output_centroid_features = torch.randn(num_heads, num_centroids, head_dim)
#         input_centroid_features = input_centroid_features / input_centroid_features.norm(dim=-1, keepdim=True)
#         output_centroid_features = output_centroid_features / output_centroid_features.norm(dim=-1, keepdim=True)
#         # Centroid positions: sample uniform radius, theta, phi for points in unit sphere
#         u = torch.rand(num_heads, num_centroids)
#         v = torch.rand(num_heads, num_centroids)
#         w = torch.rand(num_heads, num_centroids)
#         r = u.pow(1/3)                          # radius: cube root for uniformity in volume
#         theta = 2 * math.pi * v                # azimuthal angle: [0, 2pi)
#         phi = torch.acos(2 * w - 1)            # polar angle: [0, pi]
#         x = r * torch.sin(phi) * torch.cos(theta)
#         y = r * torch.sin(phi) * torch.sin(theta)
#         z = r * torch.cos(phi)
#         centroid_positions = torch.stack([x, y, z], dim=-1)  # (num_heads, num_centroids, 3)

#         # Final centroids are a weighted concatenation of random initial features and positions
#         input_centroids = torch.cat([input_centroid_features  * (1 - position_weight), centroid_positions * position_weight], dim=-1)
#         output_centroids = torch.cat([output_centroid_features  * (1 - position_weight), centroid_positions * position_weight], dim=-1)

#         return input_centroids, output_centroids

#     @torch.no_grad()
#     def _directional_rope(self, positions):  # positions: (B, N, 3)
#         rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
#         rope_freqs = self.rope_freqs.to(dtype=positions.dtype, device=positions.device)
#         proj = torch.einsum('bnd,fd->bnf', positions, rope_dirs)
#         angles = proj * rope_freqs
#         return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B,N,2F)

#     @torch.no_grad()
#     def _apply_rope(self, x, rope_emb):  # x: (B*T,N,D)
#         B_T, N, D = x.shape
#         B = rope_emb.shape[0]
#         T = B_T // B
#         rope = self.rope_proj(rope_emb.to(dtype=x.dtype, device=x.device))  # (B,N,D)
#         return x + rope.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)

#     @torch.no_grad()
#     def _calculate_cluster_cosine_scores(self, q, k, unit_point_positions):
#         # q, k: (S, H, N, Dh) -- S is batch*timesteps, H heads, N neurons, Dh head dim
#         # unit_point_positions: (B, N, 3) -- B is batch size
#         # Compute cosine similarity of concatenated vectors [feat, pos] with spherical norm,
#         # but avoid materializing (S,H,N,Dh+3) tensors. Keep fully vectorized, no loops.

#         S, H, N, Dh = q.shape
#         B = unit_point_positions.shape[0]
#         T = S // B

#         # Positions as (S,N,3). No expansion along H to save memory; rely on broadcasting in einsum.
#         pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)  # (S,N,3)

#         # Centroids (H,C,Dh+3) normalized; split into feature and position parts
#         input_centroids_n  = F.normalize(self.input_centroids,  dim=-1)
#         output_centroids_n = F.normalize(self.output_centroids, dim=-1)
#         c_in_feat,  c_in_pos  = input_centroids_n[..., :Dh],  input_centroids_n[..., Dh:]
#         c_out_feat, c_out_pos = output_centroids_n[..., :Dh], output_centroids_n[..., Dh:]

#         # Token concat norms: ||[q,pos]|| per (S,H,N,1). Avoid expanding pos over H by broadcasting.
#         q_norm2 = (q.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)          # (S,H,N,1)
#         p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)     # (S,N,1)
#         p_norm2 = p_norm2.unsqueeze(1)                                                # (S,1,N,1)
#         concat_norm = (q_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=q.dtype)    # (S,H,N,1)

#         # Dot products: (S,H,N,C) via broadcasting; no intermediate (S,H,N,Dh+3).
#         q_dot_cin   = torch.einsum("b h n d, h c d -> b h n c", q,        c_in_feat)
#         p_dot_cin   = torch.einsum("b n d,   h c d -> b h n c", pos_st,   c_in_pos)
#         in_score    = (q_dot_cin + p_dot_cin) / concat_norm

#         k_dot_cout  = torch.einsum("b h n d, h c d -> b h n c", k,        c_out_feat)
#         p_dot_cout  = torch.einsum("b n d,   h c d -> b h n c", pos_st,   c_out_pos)
#         out_score   = (k_dot_cout + p_dot_cout) / concat_norm

#         # Zero-out rows for padded tokens (pos == 0) without allocating a full (S,H,N,C) mask.
#         pos_zero = (pos_st == 0).all(dim=-1)                      # (S,N)
#         if pos_zero.any():
#             m = (~pos_zero).to(in_score.dtype).unsqueeze(1).unsqueeze(-1)  # (S,1,N,1)
#             in_score  = in_score * m
#             out_score = out_score * m
#         return in_score, out_score

#     @torch.no_grad()
#     def _compute_input_scores(self, q, unit_point_positions):
#         S, H, N, Dh = q.shape
#         B = unit_point_positions.shape[0]
#         T = S // B
#         pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)
#         # Ensure bf16 for routing matmuls
#         q_bf16 = q.to(torch.bfloat16)
#         input_centroids_n = F.normalize(self.input_centroids, dim=-1)
#         c_in_feat, c_in_pos = input_centroids_n[..., :Dh], input_centroids_n[..., Dh:]
#         q_norm2 = (q_bf16.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)
#         p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True).unsqueeze(1)
#         concat_norm = (q_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=q_bf16.dtype)
#         q_dot_cin = torch.einsum("b h n d, h c d -> b h n c", q_bf16, c_in_feat)
#         p_dot_cin = torch.einsum("b n d,   h c d -> b h n c", pos_st, c_in_pos)
#         in_score = (q_dot_cin + p_dot_cin) / concat_norm
#         pos_zero = (pos_st == 0).all(dim=-1)
#         if pos_zero.any():
#             m = (~pos_zero).to(in_score.dtype).unsqueeze(1).unsqueeze(-1)
#             in_score = in_score * m
#         return in_score

#     @torch.no_grad()
#     def _compute_output_scores(self, k, unit_point_positions):
#         S, H, N, Dh = k.shape
#         B = unit_point_positions.shape[0]
#         T = S // B
#         pos_st = unit_point_positions.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)
#         # Ensure bf16 for routing matmuls
#         k_bf16 = k.to(torch.bfloat16)
#         output_centroids_n = F.normalize(self.output_centroids, dim=-1)
#         c_out_feat, c_out_pos = output_centroids_n[..., :Dh], output_centroids_n[..., Dh:]
#         k_norm2 = (k_bf16.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True)
#         p_norm2 = (pos_st.to(dtype=torch.float32) ** 2).sum(dim=-1, keepdim=True).unsqueeze(1)
#         concat_norm = (k_norm2 + p_norm2).clamp_min(1e-12).sqrt().to(dtype=k_bf16.dtype)
#         k_dot_cout = torch.einsum("b h n d, h c d -> b h n c", k_bf16, c_out_feat)
#         p_dot_cout = torch.einsum("b n d,   h c d -> b h n c", pos_st, c_out_pos)
#         out_score = (k_dot_cout + p_dot_cout) / concat_norm
#         pos_zero = (pos_st == 0).all(dim=-1)
#         if pos_zero.any():
#             m = (~pos_zero).to(out_score.dtype).unsqueeze(1).unsqueeze(-1)
#             out_score = out_score * m
#         return out_score

#     @torch.no_grad()
#     def _topk_indices_from_scores(self, scores: torch.Tensor, k: int):
#         # scores: (S,H,N,C) → return (S,H,C,K) top-k indices along N
#         return torch.topk(scores.permute(0, 1, 3, 2), k=k, dim=-1)[1]

#     @torch.no_grad()
#     def _topk_indices_blocked(self, compute_scores_fn, q_or_k: torch.Tensor, unit_point_positions: torch.Tensor, k: int, C: int, blocks: int = 4):
#         # Vectorized two-stage top-k across blocks of clusters.
#         # Returns (S,H,C,K) indices along N.
#         S, H, N, Dh = q_or_k.shape
#         block_size = max(1, (C + blocks - 1) // blocks)
#         # First pass: per-block top-k (S,H,block_C,K)
#         topk_vals = []
#         topk_idxN = []
#         for c0 in range(0, C, block_size):
#             c1 = min(c0 + block_size, C)
#             # Compute scores for the slice of centroids by temporarily slicing centroids buffers
#             if compute_scores_fn is self._compute_input_scores:
#                 # Temporarily slice centroids
#                 full = self.input_centroids
#                 self.input_centroids = full[:, c0:c1]
#                 scores_block = self._compute_input_scores(q_or_k, unit_point_positions)  # (S,H,N,block_C)
#                 self.input_centroids = full
#             else:
#                 full = self.output_centroids
#                 self.output_centroids = full[:, c0:c1]
#                 scores_block = self._compute_output_scores(q_or_k, unit_point_positions)  # (S,H,N,block_C)
#                 self.output_centroids = full

#             # Top-k along N for this block
#             vals, idx = torch.topk(scores_block.permute(0, 1, 3, 2), k=k, dim=-1)
#             topk_vals.append(vals)        # (S,H,block_C,K)
#             topk_idxN.append(idx)         # (S,H,block_C,K)

#         # Concatenate across blocks on C dimension
#         vals_all = torch.cat(topk_vals, dim=2)       # (S,H,C,K)
#         idxN_all = torch.cat(topk_idxN, dim=2)       # (S,H,C,K)

#         # Global top-k across N across blocks: merge K candidates per block → still need only indices, so take topk over vals
#         # vals_all corresponds to scores at positions idxN_all per (S,H,C,·)
#         # We already have top-k per block; to get exact top-k across blocks, take top-k over K*blocks candidates.
#         Kb = vals_all.size(-1)
#         vals_flat = vals_all.reshape(S, H, C, Kb)
#         idx_candidates = idxN_all.reshape(S, H, C, Kb)
#         vals_top, idx_top_inKb = torch.topk(vals_flat, k=k, dim=-1)
#         # Gather corresponding indices in N
#         idx_final = torch.gather(idx_candidates, -1, idx_top_inKb)
#         return idx_final  # (S,H,C,K)

#     @torch.no_grad()
#     def _calculate_neuron_top_indices(self, input_centroid_cosine_score, output_centroid_cosine_score):
#         # input_centroid_cosine_score: (B, n_heads, N, num_clusters_per_head)
#         # output_centroid_cosine_score: (B, n_heads, N, num_clusters_per_head)
#         # For each (B, n_heads, num_clusters_per_head), select indices of the top-k scoring neurons along the *neuron* dimension.
#         k = self.neuron_cluster_size  # number of top neurons to select per cluster

#         # Get shape info
#         B, n_heads, N, num_clusters_per_head = input_centroid_cosine_score.shape

#         # Permute to bring num_clusters_per_head as the "items" of interest, then topk over neurons
#         # For the *receiving* neurons: topk over N
#         # Result: (B, n_heads, num_clusters_per_head, k) -- indices in N
#         input_neuron_indices = torch.topk(
#             input_centroid_cosine_score.permute(0,1,3,2), k=k, dim=-1
#         )[1]  # (B, n_heads, num_clusters_per_head, k)

#         # For the *sending* neurons: topk over N
#         output_neuron_indices = torch.topk(
#             output_centroid_cosine_score.permute(0,1,3,2), k=k, dim=-1
#         )[1]  # (B, n_heads, num_clusters_per_head, k)

#         return input_neuron_indices, output_neuron_indices


#     @torch.no_grad()
#     def _build_cluster_tensors(self, q, k, v, input_cluster_neuron_indices, output_cluster_neuron_indices):
#         """
#         Gather the top-k neurons per cluster for Q (receivers) and KV (senders).
#         NOTE: we DO NOT append the null token here; we do it later after masking so it never
#         affects cluster keep/drop decisions.
#         """
#         _, _, num_clusters_per_head, k_sel = input_cluster_neuron_indices.shape
#         D = q.size(-1)

#         # Q clusters: gather along neuron axis
#         input_clusters_q = torch.gather(
#             q.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
#             3,
#             input_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)       # (S,H,C,K,D)
#         )

#         # K clusters
#         output_clusters_k = torch.gather(
#             k.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
#             3,
#             output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)      # (S,H,C,K,D)
#         )

#         # V clusters
#         output_clusters_v = torch.gather(
#             v.unsqueeze(2).expand(-1, -1, num_clusters_per_head, -1, -1),             # (S,H,C,N,D)
#             3,
#             output_cluster_neuron_indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)      # (S,H,C,K,D)
#         )

#         return input_clusters_q, output_clusters_k, output_clusters_v

#     @torch.no_grad()
#     def _get_indicies_of_non_spiking_neurons_in_clusters(self, output_cluster_neuron_indices, sending_neurons_mask_bt):
#         # output_cluster_neuron_indices: (B, n_heads, num_clusters_per_head, k)
#         # sending_neurons_mask_bt: (B, N)

#         # For each cluster, check which neurons are non-spiking.
#         B, n_heads, num_clusters_per_head, k = output_cluster_neuron_indices.shape

#         # Expand mask so we can gather using neuron indices
#         sending_neurons_mask_expanded = sending_neurons_mask_bt.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)

#         # For each item in output_cluster_neuron_indices, gather the mask (True if spiking, False otherwise)
#         gathered_mask = torch.gather(
#             sending_neurons_mask_expanded.expand(B, n_heads, num_clusters_per_head, -1),  # (B, n_heads, num_clusters_per_head, N)
#             3,
#             output_cluster_neuron_indices
#         )  # (B, n_heads, num_clusters_per_head, k)

#         # Invert mask: True where neuron is not spiking
#         non_spiking_mask = ~gathered_mask.bool()  # (B, n_heads, num_clusters_per_head, k)

#         # Create a bool mask for clusters which are all non-spiking (along k dim)
#         all_non_spiking_in_cluster_mask = non_spiking_mask.all(dim=-1)  # (B, n_heads, num_clusters_per_head), bool

#         # The output:
#         # non_spiking_mask: (B, n_heads, num_clusters_per_head, k) -- True at (i,j,c,x) if x-th neuron in c-th cluster is non-spiking
#         # all_non_spiking_in_cluster_mask: (B, n_heads, num_clusters_per_head) -- True if all neurons in cluster are non-spiking

#         return non_spiking_mask, all_non_spiking_in_cluster_mask

#     @torch.no_grad()
#     def _ema_update_centroids(
#         self,
#         q: torch.Tensor,                      # (S,H,N,Dh)
#         k: torch.Tensor,                      # (S,H,N,Dh)
#         unit_point_positions: torch.Tensor,   # (B,N,3), L2-normalized
#         input_neuron_indices: torch.Tensor,   # (S,H,C,K)  -> indices into N for Q
#         output_neuron_indices: torch.Tensor,  # (S,H,C,K)  -> indices into N for K
#         ):
#         """
#         Exponential moving average update for:
#         - self.input_centroids  (H,C,Dh+3)  using Q + positions
#         - self.output_centroids (H,C,Dh+3)  using K + positions

#         Uses only the selected K neurons per centroid (top-k per cluster).
#         Skips padded tokens whose position is exactly (0,0,0).
#         """
#         device = q.device
#         dtype  = q.dtype
#         S, H, N, Dh = q.shape
#         _, _, C, K = input_neuron_indices.shape
#         B3 = unit_point_positions.shape  # (B,N,3)

#         # Expand positions from (B,N,3) -> (S,N,3) to align with q/k (S=B*T, N)
#         B = B3[0]
#         T = S // B
#         pos_bt = unit_point_positions.to(device=device, dtype=dtype)              # (B,N,3)
#         pos_st = pos_bt.unsqueeze(1).expand(B, T, N, 3).reshape(S, N, 3)          # (S,N,3)

#         # Helper: gather per (S,H,C,K) from (S,N,·)
#         def _gather_feats(feats_shnd, idx_shck):
#             # feats_shnd: (S,H?,N,D) or (S,N,3) broadcast to (S,H,C,N,D)
#             if feats_shnd.dim() == 4:  # (S,H,N,Dh)
#                 S_, H_, N_, D_ = feats_shnd.shape
#                 # (S,H,1,N,D) -> (S,H,C,N,D)
#                 feats_exp = feats_shnd.unsqueeze(2).expand(S_, H_, C, N_, D_)
#             else:  # positions: (S,N,3)
#                 S_, N_, D_ = feats_shnd.shape
#                 # (S,1,1,N,3) -> (S,H,C,N,3)
#                 feats_exp = feats_shnd.unsqueeze(1).unsqueeze(2).expand(S_, H, C, N_, D_)
#             return torch.gather(
#                 feats_exp,
#                 3,
#                 idx_shck.unsqueeze(-1).expand(-1, -1, -1, -1, feats_exp.size(-1))         # (S,H,C,K,D)
#             )  # -> (S,H,C,K,D)

#         # Gather Q/K features and corresponding positions for selected indices
#         q_sel = _gather_feats(q, input_neuron_indices)         # (S,H,C,K,Dh)
#         k_sel = _gather_feats(k, output_neuron_indices)        # (S,H,C,K,Dh)
#         p_q   = _gather_feats(pos_st, input_neuron_indices)    # (S,H,C,K,3)
#         p_k   = _gather_feats(pos_st, output_neuron_indices)   # (S,H,C,K,3)

#         # Mask out padded tokens: position == (0,0,0)
#         pad_q = (p_q.abs().sum(dim=-1) == 0)                   # (S,H,C,K)
#         pad_k = (p_k.abs().sum(dim=-1) == 0)                   # (S,H,C,K)

#         # Build augmented vectors [feat, pos] and L2-normalize
#         def _augment_norm(feat, pos):
#             aug = torch.cat([feat, pos], dim=-1)               # (S,H,C,K,Dh+3)
#             aug = F.normalize(aug, dim=-1)
#             return aug

#         q_aug = _augment_norm(q_sel, p_q)                      # (S,H,C,K,Dh+3)
#         k_aug = _augment_norm(k_sel, p_k)                      # (S,H,C,K,Dh+3)

#         # Zero-out padded entries before summing
#         q_aug = q_aug.masked_fill(pad_q.unsqueeze(-1), 0.0)
#         k_aug = k_aug.masked_fill(pad_k.unsqueeze(-1), 0.0)

#         # Sum over batch and K, then normalize by counts (avoid div by zero)
#         def _reduce_to_centroid_mean(aug, pad_mask):
#             counts = (~pad_mask).sum(dim=(0, 2, 3), keepdim=False)              # (H,)
#             # Per-centroid counts:
#             cnt_hc = (~pad_mask).sum(dim=(0, 3), keepdim=False)                 # (H,C)
#             # Sum over S and K
#             sum_hck = aug.sum(dim=(0, 3))                                       # (H,C,Dh+3)
#             # If a centroid has zero valid samples in this step, skip its update later
#             return sum_hck, cnt_hc

#         sum_q, cnt_q = _reduce_to_centroid_mean(q_aug, pad_q)   # (H,C,Dh+3), (H,C)
#         sum_k, cnt_k = _reduce_to_centroid_mean(k_aug, pad_k)   # (H,C,Dh+3), (H,C)

#         # Compute means where count > 0
#         eps = 1e-6
#         mean_q = sum_q / cnt_q.clamp_min(1).unsqueeze(-1)       # (H,C,Dh+3)
#         mean_k = sum_k / cnt_k.clamp_min(1).unsqueeze(-1)       # (H,C,Dh+3)

#         # Current centroids
#         mu_q = self.input_centroids.to(device=device, dtype=dtype)    # (H,C,Dh+3)
#         mu_k = self.output_centroids.to(device=device, dtype=dtype)   # (H,C,Dh+3)

#         # EMA
#         decay = self.ema_decay
#         upd_q = torch.where(
#             (cnt_q > 0).unsqueeze(-1),
#             decay * mu_q + (1.0 - decay) * mean_q,
#             mu_q,                                                   # no-op if no samples
#         )
#         upd_k = torch.where(
#             (cnt_k > 0).unsqueeze(-1),
#             decay * mu_k + (1.0 - decay) * mean_k,
#             mu_k,
#         )

#         # Re-normalize to unit length for spherical k-means behavior
#         upd_q = F.normalize(upd_q, dim=-1)
#         upd_k = F.normalize(upd_k, dim=-1)

#         # Write back
#         self.input_centroids.copy_(upd_q)
#         self.output_centroids.copy_(upd_k)


#     def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
#         # x: (B, T, N, Dmodel)
#         # point_positions: (B, N, 3)
#         # spike_mask: (B, T, N)  True where neuron fired at that time step

#         B, T, N, Dmodel = x.shape
#         H = self.n_heads
#         Dh = self.head_dim
#         C = self.num_clusters_per_head
#         K = self.neuron_cluster_size  # <-- fixed

#         S = B * T
#         device = x.device
#         dtype = x.dtype

#         # Optional memory profiler helpers (CUDA only)
#         def _mem_ckpt(tag: str):
#             if self.profile_memory and torch.cuda.is_available():
#                 try:
#                     import torch._dynamo as _d
#                     if hasattr(_d, 'is_compiling') and _d.is_compiling():
#                         return
#                 except Exception:
#                     pass
#                 torch.cuda.synchronize()
#                 alloc = torch.cuda.memory_allocated(x.device)
#                 reserved = torch.cuda.memory_reserved(x.device)
#                 print(f"[SpikeSparseConnectomeAttention][mem] {tag}: alloc={alloc/1e9:.2f}GB reserved={reserved/1e9:.2f}GB")

#         _mem_ckpt("start")

#         # --------- Projections ----------
#         x_bt = x.reshape(S, N, Dmodel)                    # (S,N,D)
#         sending_neurons_mask_bt = spike_mask.reshape(S, N)

#         unit_point_positions = F.normalize(point_positions, dim=-1)  # (B,N,3)
#         res = x_bt
#         xn = self.norm(x_bt)

#         rope_emb = self._directional_rope(unit_point_positions)       # (B,N,2F)
#         qk_in = self._apply_rope(xn, rope_emb)                        # add RoPE to Q/K inputs

#         q = self.q_proj(qk_in)    # (S,N,D)
#         k = self.k_proj(qk_in)    # (S,N,D)  (null token will NOT use RoPE)
#         v = self.v_proj(xn)       # (S,N,D)
#         _mem_ckpt("after_projections")
#         if debug_enabled():
#             assert_no_nan(q, 'SpikeSparse.proj.q')
#             assert_no_nan(k, 'SpikeSparse.proj.k')
#             assert_no_nan(v, 'SpikeSparse.proj.v')

#         # Shape to (S,H,N,Dh)
#         q = q.view(S, H, N, Dh)
#         k = k.view(S, H, N, Dh)
#         v = v.view(S, H, N, Dh)

#         # --------- Clustering ----------
#         # Compute input/output scores separately (no_grad) to reduce peak memory
#         with torch.no_grad():
#             # Blocked top-k to reduce peak memory during routing
#             input_idx = self._topk_indices_blocked(self._compute_input_scores, q, unit_point_positions, K, C, blocks=4)
#         _mem_ckpt("after_input_topk")
#         if self.profile_memory and torch.cuda.is_available():
#             try:
#                 torch.cuda.empty_cache()
#             except Exception:
#                 pass
#         _mem_ckpt("after_input_topk")

#         with torch.no_grad():
#             output_idx = self._topk_indices_blocked(self._compute_output_scores, k, unit_point_positions, K, C, blocks=4)
#         _mem_ckpt("after_output_topk")
#         if self.profile_memory and torch.cuda.is_available():
#             try:
#                 torch.cuda.empty_cache()
#             except Exception:
#                 pass
#         _mem_ckpt("after_output_topk")


#         # === EMA centroid update (training only) ===
#         if self.training:
#             self._ema_update_centroids(
#                 q, k, unit_point_positions,   # (S,H,N,Dh), (S,H,N,Dh), (B,N,3)
#                 input_idx,                    # (S,H,C,K)
#                 output_idx,                   # (S,H,C,K)
#             )


#         # Build cluster tensors (no null yet)
#         q_cl, k_cl, v_cl = self._build_cluster_tensors(q, k, v, input_idx, output_idx)    # (S,H,C,K,Dh)
#         _mem_ckpt("after_build_clusters")
#         if debug_enabled():
#             assert_no_nan(q_cl, 'SpikeSparse.q_clusters')
#             assert_no_nan(k_cl, 'SpikeSparse.k_clusters')
#             assert_no_nan(v_cl, 'SpikeSparse.v_clusters')


#         # --------- Spiking masks ----------
#         q_keep, kv_keep_real, kv_keep_null, cluster_keep = self._build_spiking_keep_masks(
#             output_idx, sending_neurons_mask_bt, S, H, C, K
#         )

#         # --------- Append the NULL output token (per kept cluster) ----------
#         # Pass the learned null vector through K and V projections and split into heads
#         null_k_model = self.k_proj(self.null_output_vec.to(device=device, dtype=dtype))   # (Dmodel,)
#         null_v_model = self.v_proj(self.null_output_vec.to(device=device, dtype=dtype))   # (Dmodel,)
#         null_k = null_k_model.view(H, Dh)  # (H,Dh)
#         null_v = null_v_model.view(H, Dh)  # (H,Dh)

#         # Expand to (S,H,C,1,Dh) and append as the LAST token in each output cluster
#         null_k_exp = null_k.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(S, H, C, 1, Dh)
#         null_v_exp = null_v.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(S, H, C, 1, Dh)

#         k_cl_aug = torch.cat([k_cl, null_k_exp], dim=3)  # (S,H,C,K+1,Dh)
#         v_cl_aug = torch.cat([v_cl, null_v_exp], dim=3)  # (S,H,C,K+1,Dh)
#         _mem_ckpt("after_null_append")

#         # Final KV keep mask (real + null). The null token is NEVER removed from remaining clusters.
#         kv_keep = torch.cat([kv_keep_real, kv_keep_null], dim=-1)  # (S,H,C,K+1)

#         # --------- Pack to varlen (allowing different Q/K lengths) ----------
#         Cn_q = C * K
#         Cn_k = C * (K + 1)

#         def _reshape_to_bcn(t):  # (S,H,C,?,Dh) -> (S*H, C*?, Dh)
#             S_, H_, C_, KK_, Dh_ = t.shape
#             return t.reshape(S_*H_, C_*KK_, Dh_)

#         q_bcn  = _reshape_to_bcn(q_cl)          # (S*H, Cn_q, Dh)
#         k_bcn  = _reshape_to_bcn(k_cl_aug)      # (S*H, Cn_k, Dh)
#         v_bcn  = _reshape_to_bcn(v_cl_aug)      # (S*H, Cn_k, Dh)
#         _mem_ckpt("after_pack_varlen_inputs")

#         q_keep_b, kv_keep_b, len_q, len_k, keep_items = self._build_varlen_masks_and_lengths(
#             q_keep, kv_keep, S, H, Cn_q, Cn_k
#         )
#         if not keep_items.any():
#             # Nothing to attend to for any (sample, head)
#             return x

#         q_keep_b  = q_keep_b[keep_items]
#         kv_keep_b = kv_keep_b[keep_items]
#         q_bcn     = q_bcn[keep_items]
#         k_bcn     = k_bcn[keep_items]
#         v_bcn     = v_bcn[keep_items]
#         len_q     = len_q[keep_items]
#         len_k     = len_k[keep_items]
#         kept_item_ids = torch.arange(S*H, device=device)[keep_items]

#         # Flatten to kept tokens
#         mask_q_flat  = q_keep_b.reshape(-1)
#         mask_kv_flat = kv_keep_b.reshape(-1)

#         q_kept = q_bcn.reshape(-1, Dh)[mask_q_flat]   # (total_q, Dh)
#         k_kept = k_bcn.reshape(-1, Dh)[mask_kv_flat]  # (total_k, Dh)
#         v_kept = v_bcn.reshape(-1, Dh)[mask_kv_flat]  # (total_k, Dh)

#         # Varlen metadata
#         B_eff = q_keep_b.size(0)
#         cu_seqlens_q = torch.zeros(B_eff + 1, dtype=torch.int32, device=device)
#         cu_seqlens_k = torch.zeros(B_eff + 1, dtype=torch.int32, device=device)
#         cu_seqlens_q[1:] = torch.cumsum(len_q, dim=0)
#         cu_seqlens_k[1:] = torch.cumsum(len_k, dim=0)
#         max_seqlen_q, max_seqlen_k = _compute_max_lens_tensors(len_q, len_k)

#         # FlashAttention-2 varlen (nheads=1; head-as-batch) with optional chunking
#         max_batch = MAX_BATCH_SIZE
#         p_drop = self.dropout if self.training else 0.0

#         # Directly accumulate outputs into neuron space to avoid large intermediate buffers
#         out_sum = torch.zeros(S, H, N, Dh, device=device, dtype=v_bcn.dtype)
#         counts = torch.zeros(S, H, N, 1, device=device, dtype=v_bcn.dtype)
#         idx_flat_full = input_idx.reshape(S * H, C * K)  # (S*H, C*K)
#         out_sum_lin = out_sum.view(S * H * N, Dh)
#         counts_lin = counts.view(S * H * N)

#         if B_eff <= max_batch:
#             q_packed  = q_kept.unsqueeze(1).contiguous()  # (total_q, 1, Dh)
#             kv_packed = torch.stack([k_kept, v_kept], dim=1).unsqueeze(2).contiguous()  # (total_k, 2, 1, Dh)

#             attn_out = flash_attn_varlen_kvpacked_func(
#                 q_packed, kv_packed,
#                 cu_seqlens_q, cu_seqlens_k,
#                 max_seqlen_q, max_seqlen_k,
#                 p_drop,
#                 softmax_scale=None,
#                 causal=False
#             ).squeeze(1)  # (total_q, Dh)
#             _mem_ckpt("after_flash_attn_chunk_or_full")
#             if debug_enabled():
#                 assert_no_nan(attn_out, 'SpikeSparse.FA.out_full')

#             cols = torch.arange(Cn_q, device=device).expand(B_eff, Cn_q)
#             col_ids = cols[q_keep_b]                        # (total_q,)
#             row_ids = torch.repeat_interleave(torch.arange(B_eff, device=device), len_q)
#             t_orig = kept_item_ids[row_ids]                 # (total_q,)
#             lin_neuron = idx_flat_full[t_orig, col_ids]      # (total_q,)
#             lin_index = t_orig * N + lin_neuron
#             out_sum_lin.index_add_(0, lin_index, attn_out.to(out_sum_lin.dtype))
#             counts_lin.index_add_(0, lin_index, torch.ones(lin_index.numel(), device=device, dtype=counts_lin.dtype))
#         else:
#             # Chunk over sequence items so each chunk has at most max_batch sequences
#             for start in range(0, B_eff, max_batch):
#                 end = min(start + max_batch, B_eff)
#                 n_seq = end - start

#                 # Slice masks and tensors for this chunk
#                 q_keep_chunk = q_keep_b[start:end]
#                 kv_keep_chunk = kv_keep_b[start:end]
#                 len_q_chunk = len_q[start:end]
#                 len_k_chunk = len_k[start:end]
#                 q_bcn_chunk = q_bcn[start:end]
#                 k_bcn_chunk = k_bcn[start:end]
#                 v_bcn_chunk = v_bcn[start:end]

#                 # Flatten kept tokens for this chunk
#                 mask_q_flat_chunk = q_keep_chunk.reshape(-1)
#                 mask_kv_flat_chunk = kv_keep_chunk.reshape(-1)
#                 q_kept_chunk = q_bcn_chunk.reshape(-1, Dh)[mask_q_flat_chunk]
#                 k_kept_chunk = k_bcn_chunk.reshape(-1, Dh)[mask_kv_flat_chunk]
#                 v_kept_chunk = v_bcn_chunk.reshape(-1, Dh)[mask_kv_flat_chunk]

#                 # Build varlen metadata for this chunk
#                 cu_q = torch.zeros(n_seq + 1, dtype=torch.int32, device=device)
#                 cu_k = torch.zeros(n_seq + 1, dtype=torch.int32, device=device)
#                 cu_q[1:] = torch.cumsum(len_q_chunk, dim=0)
#                 cu_k[1:] = torch.cumsum(len_k_chunk, dim=0)
#                 max_q, max_k = _compute_max_lens_tensors(len_q_chunk, len_k_chunk)

#                 q_packed_chunk = q_kept_chunk.unsqueeze(1).contiguous()
#                 kv_packed_chunk = torch.stack([k_kept_chunk, v_kept_chunk], dim=1).unsqueeze(2).contiguous()

#                 attn_out_chunk = flash_attn_varlen_kvpacked_func(
#                     q_packed_chunk, kv_packed_chunk,
#                     cu_q, cu_k,
#                     max_q, max_k,
#                     p_drop,
#                     softmax_scale=None,
#                     causal=False
#                 ).squeeze(1)  # (total_q_chunk, Dh)
#                 _mem_ckpt(f"after_flash_attn_chunk[{start}:{end}]")
#                 if debug_enabled():
#                     assert_no_nan(attn_out_chunk, 'SpikeSparse.FA.out_chunk')

#                 # Direct scatter for this chunk
#                 cols_chunk = torch.arange(Cn_q, device=device).expand(n_seq, Cn_q)
#                 col_ids_chunk = cols_chunk[q_keep_chunk]
#                 row_ids_chunk = torch.repeat_interleave(torch.arange(n_seq, device=device), len_q_chunk)
#                 t_orig_chunk = kept_item_ids[start:end][row_ids_chunk]
#                 lin_neuron_chunk = idx_flat_full[t_orig_chunk, col_ids_chunk]
#                 lin_index_chunk = t_orig_chunk * N + lin_neuron_chunk
#                 out_sum_lin.index_add_(0, lin_index_chunk, attn_out_chunk.to(out_sum_lin.dtype))
#                 counts_lin.index_add_(0, lin_index_chunk, torch.ones(lin_index_chunk.numel(), device=device, dtype=counts_lin.dtype))

#         _mem_ckpt("after_scatter_q_positions")

#         out_shnd = out_sum / counts.clamp_min(1.0)  # (S,H,N,Dh)
#         if debug_enabled():
#             assert_no_nan(out_shnd, 'SpikeSparse.out_shnd')

#         # Final projection & residual
#         out = out_shnd.reshape(B, T, H, N, Dh).permute(0, 1, 3, 2, 4).reshape(B, T, N, H*Dh)
#         out = self.o_proj(out)
#         out = out + x  # residual
#         _mem_ckpt("end")
#         return out


def gumbel_topk(logits: torch.Tensor, k: int, temperature: float = 1.0):
    """
    Gumbel top-k sampling.
    Returns: (topk_values, topk_indices)
    """
    if temperature == 0.0:
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)
        return topk_values, topk_indices
    gumbel_noise = (
        -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8) / temperature
    )
    topk_values, topk_indices = torch.topk(logits + gumbel_noise, k, dim=-1)
    return topk_values, topk_indices


class SparseSpikeFullAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_rope_features: int = 32,  # number of (direction, freq) angles available for RoPE
        dropout: float = 0.0,
        # ---- new, all optional (keeps Dh fixed) ----
        pos_tail_dim: int = 16,  # Dp: channels per head reserved for positional tail (<= Dh)
        n_rff: int = 32,  # M: random Fourier features; tail uses a learned (2M -> Dp) compression
        rff_sigma: float = 1.0,  # bandwidth for RFF wrt your RMS-scaled coords
        pos_tail_scale: float = 0.1,  # γ: strength of positional kernel in logits
        n_rot_pairs: int
        | None = None,  # m: rotary pairs per head (first 2m dims rotated)
        topk_fraction: float = 0.1,  # fraction of valid neurons to select via Gumbel top-k
        gumbel_temperature: float = 1.0,  # temperature for Gumbel noise
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        # Top-k config
        self.topk_fraction = float(topk_fraction)
        self.gumbel_temperature = float(gumbel_temperature)

        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # ---- RoPE basis (post-projection; direction-sensitive) ----
        self.n_rope_features = int(n_rope_features)
        dirs = torch.randn(self.n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        # tune upper band to your scale; 1..1e4 is fine for wide range, but you can lower upper band
        freqs = torch.logspace(
            math.log10(1.0), math.log10(10000.0), self.n_rope_features
        )
        self.register_buffer("rope_dirs", dirs, persistent=False)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        # ---- In-place positional tail (relative bias inside dot product), keeps Dh fixed ----
        # Random Fourier features Ω ~ N(0, σ^2 I), φ(p) = [cos(Ωp), sin(Ωp)] ∈ R^{2M}
        self.n_rff = int(n_rff)
        Omega = torch.randn(self.n_rff, 3) * float(rff_sigma)
        self.register_buffer("rff_Omega", Omega, persistent=False)

        # compress 2M -> Dp once (shared across heads) + per-head gain
        self.pos_tail_dim = int(pos_tail_dim)  # Dp
        assert 0 <= self.pos_tail_dim <= self.head_dim, (
            "pos_tail_dim must be <= head_dim"
        )
        self.pos_C = nn.Linear(
            2 * self.n_rff, self.pos_tail_dim, bias=False
        )  # shared compression
        self.pos_head_gain = nn.Parameter(
            torch.ones(self.n_heads, self.pos_tail_dim)
        )  # per-head scaling
        self.pos_tail_scale = float(pos_tail_scale)  # γ

        # ---- Rotary slice config (first 2m dims rotated; tail occupies last Dp dims) ----
        # we must leave room: 2m + Dp <= Dh
        max_m = max(0, (self.head_dim - self.pos_tail_dim) // 2)
        self.n_rot_pairs = (
            int(n_rot_pairs) if n_rot_pairs is not None else min(16, max_m)
        )
        self.n_rot_pairs = min(self.n_rot_pairs, max_m)  # safety

    # -------- helpers --------
    @torch.no_grad()
    def _rope_angles(
        self, positions
    ):  # positions: (B, N, 3)  (do NOT normalize; you said RMS-scaled & centered)
        rope_dirs = self.rope_dirs.to(dtype=positions.dtype, device=positions.device)
        rope_freqs = self.rope_freqs.to(
            dtype=positions.dtype, device=positions.device
        )  # (F,)
        proj = torch.einsum("bnd,fd->bnf", positions, rope_dirs)  # (B,N,F)
        angles = proj * rope_freqs  # broadcast (B,N,F)
        return angles  # (B,N,F)

    @torch.no_grad()
    def _rff_phi(self, positions):  # positions: (B, N, 3)
        # φ(p) = [cos(Ωp), sin(Ωp)] with Ω: (M,3)
        Omega = self.rff_Omega.to(
            dtype=positions.dtype, device=positions.device
        )  # (M,3)
        proj = torch.einsum("bnd,md->bnm", positions, Omega)  # (B,N,M)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B,N,2M)

    @staticmethod
    def _apply_rotary_inplace(x, theta, m):
        """
        x: (total_tokens, H, Dh)
        theta: (total_tokens, F)  -> we use only first m angles
        m: number of rotary pairs (first 2m dims per head will be rotated)
        """
        if m <= 0:
            return x
        # pairwise rotate: (even, odd)
        x_even = x[..., : 2 * m : 2]  # (total, H, m)
        x_odd = x[..., 1 : 2 * m : 2]  # (total, H, m)
        sin = torch.sin(theta[:, :m]).unsqueeze(1)  # (total,1,m)
        cos = torch.cos(theta[:, :m]).unsqueeze(1)  # (total,1,m)
        xe = x_even * cos - x_odd * sin
        xo = x_even * sin + x_odd * cos
        x[..., : 2 * m : 2] = xe
        x[..., 1 : 2 * m : 2] = xo
        return x

    # @torch._dynamo.disable
    def forward(self, x, point_positions, neuron_pad_mask, spike_mask):
        # x: (B,T,N,D), positions: (B,N,3), masks as described
        B, T, N, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        S = B * T
        m = self.n_rot_pairs
        Dp = self.pos_tail_dim

        # 1) Normalize activations (not positions)
        xn = self.norm(x)

        # 2) Precompute RoPE angles and φ(p) once per (B,N)
        angles_BNF = self._rope_angles(point_positions)  # (B,N,F_rope)
        phi_BN2M = self._rff_phi(point_positions)  # (B,N,2M)

        # 3) Flatten to (S,N,*) for (B,T) "sequence"
        xn_bt = xn.view(B, T, N, D).reshape(S, N, D)

        # 4) Build masks and compact indices BEFORE projections
        #    Queries: all valid neurons. KV: provided boolean spike mask (senders).
        valid_bt = (neuron_pad_mask != 0).unsqueeze(1).expand(B, T, N).reshape(S, N)
        spiking_bt = (spike_mask != 0).reshape(S, N) & valid_bt
        keep_bt = valid_bt
        send_bt = spiking_bt
        idx_q, idx_kv, cu_q, cu_k, lens_q, lens_k, max_q, max_k = (
            _build_varlen_metadata_from_masks(keep_bt, send_bt)
        )

        # 5) Compact inputs for projections
        x_flat = xn_bt.reshape(S * N, D)
        pre_q = x_flat.index_select(0, idx_q)  # (total_q, D)
        pre_kv = x_flat.index_select(0, idx_kv)  # (total_k, D)

        # 6) Project to Q/K/V
        q = self.q_proj(pre_q).view(-1, H, Dh)  # (total_q,H,Dh)
        k = self.k_proj(pre_kv).view(-1, H, Dh)  # (total_k,H,Dh)
        v = self.v_proj(pre_kv).view(-1, H, Dh)  # (total_k,H,Dh)

        # 7) Gather angles for compacted tokens and apply RoPE to first 2m dims (post-proj)
        angles_SN = (
            angles_BNF.unsqueeze(1).expand(B, T, N, -1).reshape(S * N, -1)
        )  # (S*N, F_rope)
        theta_q = angles_SN.index_select(0, idx_q)  # (total_q, F_rope)
        theta_k = angles_SN.index_select(0, idx_kv)  # (total_k, F_rope)
        self._apply_rotary_inplace(q, theta_q, m)
        self._apply_rotary_inplace(k, theta_k, m)

        # 8) In-place positional tail (distance/sector kernel) in the LAST Dp dims of Q/K; zero same slice in V
        if Dp > 0:
            # compress φ(p) -> Dp using shared linear, then per-head gain; scale by sqrt(γ)
            phi_SN = (
                phi_BN2M.unsqueeze(1).expand(B, T, N, -1).reshape(S * N, -1)
            )  # (S*N, 2M)
            phi_q = phi_SN.index_select(0, idx_q)  # (total_q, 2M)
            phi_k = phi_SN.index_select(0, idx_kv)  # (total_k, 2M)

            tail_q = self.pos_C(phi_q)  # (total_q, Dp)
            tail_k = self.pos_C(phi_k)  # (total_k, Dp)

            # apply per-head gain and global scale √γ, then overwrite last Dp dims
            scale = self.pos_tail_scale**0.5
            # (total, H, Dp)
            q_tail = scale * (tail_q.unsqueeze(1) * self.pos_head_gain.unsqueeze(0))
            k_tail = scale * (tail_k.unsqueeze(1) * self.pos_head_gain.unsqueeze(0))

            q[..., Dh - Dp :] = q_tail
            k[..., Dh - Dp :] = k_tail
            v[..., Dh - Dp :] = 0.0  # ensure tail affects logits only

        # 9) FlashAttention-2 varlen (unchanged)
        q_packed = q.contiguous()  # (total_q,H,Dh)
        kv_packed = torch.stack([k, v], dim=1).contiguous()  # (total_k,2,H,Dh)
        p_drop = self.dropout if self.training else 0.0

        attn_out = flash_attn_varlen_kvpacked_func(
            q_packed,
            kv_packed,
            cu_q,
            cu_k,
            max_q,
            max_k,
            p_drop,
            softmax_scale=None,
            causal=False,
        )  # (total_q,H,Dh)

        # 10) (Optional) zero positional tail in out before o_proj (should already be ~0 since V tail is 0)
        if Dp > 0:
            zero_tail = (
                attn_out[..., :Dp].detach().new_zeros(attn_out.shape[:-1] + (Dp,))
            )
            attn_out = torch.cat([attn_out[..., : Dh - Dp], zero_tail], dim=-1)

        # 11) Scatter back to (S,N,D) and output proj + residual
        out_heads = torch.zeros(S * N, H, Dh, device=x.device, dtype=attn_out.dtype)
        out_heads = out_heads.index_copy(0, idx_q, attn_out)
        out_D = out_heads.view(S * N, D)
        out = self.o_proj(out_D).view(B, T, N, D)

        # Add residual connection
        return out + x


class NeuronCausalAttention(nn.Module):
    """
    Causal self-attention over time per neuron independently, with timewise RoPE.
    - Operates on each neuron timeline (length T) with causal attention.
    - Respects neuron_pad_mask (B,N): skip padded neurons entirely to save compute.
    - Spike mask is ignored.
    - Launches at most (2^18) sequences per FLASH call by chunking rows.
    - Adds rotary positional embeddings over time axis per neuron.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout_p: float = 0.0,
        max_bh_per_call: int = (2**16 - 1),
        max_seq_len: int = 512,
    ):
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

        # --- Timewise RoPE (rotary) ---
        self.max_seq_len = max_seq_len
        # Precompute sin/cos tables for rotary, with 32 rotary dims per head by default (or as many as fit)
        self.rotary_frac = 0.5
        self.rotary_dim = int(self.head_dim * self.rotary_frac)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        t = torch.arange(self.max_seq_len)
        freqs = torch.outer(t, inv_freq)  # (T, rotary_dim // 2)
        sin, cos = freqs.sin(), freqs.cos()
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def _to_bhld(self, x):  # (B*, L, D) -> (B*, H, L, Dh)
        Bx, L, D = x.shape
        return (
            x.view(Bx, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        )

    def _from_bhld(self, x):  # (B*, H, L, Dh) -> (B*, L, D)
        Bx, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(Bx, L, H * Dh)

    def _apply_timewise_rope(self, x, seq_len):
        """
        Apply rotary positional embeddings along the time axis.
        x: (batch*, H, seq_len, Dh)
        Only applies to the first rotary_dim of Dh.
        """
        rotary_dim = self.rotary_dim
        if rotary_dim == 0:
            return x
        # get sin/cos for this seq_len, shape (seq_len, rotary_dim // 2)
        sin = self.rope_sin[:seq_len].to(dtype=x.dtype, device=x.device)
        cos = self.rope_cos[:seq_len].to(dtype=x.dtype, device=x.device)
        # Shape to (1,1,seq_len, rotary_dim), broadcast along batch/H
        sin = sin[None, None, :, :]
        cos = cos[None, None, :, :]

        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim:]
        # Reshape for pairwise treatment
        x1 = x1.reshape(*x.shape[:-1], rotary_dim // 2, 2)
        sin, cos = sin, cos
        x1_0 = x1[..., 0]
        x1_1 = x1[..., 1]
        rope_x0 = x1_0 * cos - x1_1 * sin
        rope_x1 = x1_0 * sin + x1_1 * cos
        # Stack and reshape back
        x1_rope = torch.stack([rope_x0, rope_x1], dim=-1).reshape(
            *x.shape[:-1], rotary_dim
        )
        # Concatenate with remaining dims
        return torch.cat([x1_rope, x2], dim=-1)

    # @torch._dynamo.disable
    def forward(self, x: torch.Tensor, neuron_pad_mask: torch.Tensor):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        xn = self.norm(x)
        xt = xn.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N, T, D)
        res = xt

        # Row filter from neuron_pad_mask
        valid_rows = neuron_pad_mask.reshape(-1) != 0

        out = torch.zeros_like(xt)
        if valid_rows.any():
            q_in = xt[valid_rows]
            k_in = q_in
            v_in = q_in

            # Projections
            q = self._to_bhld(self.q_proj(q_in))  # (Bv, H, L, Dh)
            k = self._to_bhld(self.k_proj(k_in))
            v = self._to_bhld(self.v_proj(v_in))

            # Apply timewise RoPE to q/k
            q = self._apply_timewise_rope(q, seq_len=q.shape[2])
            k = self._apply_timewise_rope(k, seq_len=k.shape[2])

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
                qs = q[i : i + rows_step]
                ks = k[i : i + rows_step]
                vs = v[i : i + rows_step]
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out_chunk = F.scaled_dot_product_attention(
                        qs, ks, vs, is_causal=True, dropout_p=self.dropout_p
                    )
                out_chunks.append(out_chunk)
            out_valid = torch.cat(out_chunks, dim=0)  # (Bv,H,L,Dh)
            out_valid = self._from_bhld(out_valid).to(res.dtype)
            out = _index_copy_rows(out, out_valid, valid_rows)

        out = out + res
        out = out.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return self.o_proj(out)
