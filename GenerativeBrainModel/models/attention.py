import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv4dRMSNorm
from einops import rearrange
import numpy as np
from GenerativeBrainModel.models.mlp import MLP
from mamba_ssm import Mamba2 as Mamba



class SpatialRegionAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatialModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # --- Spatial Attention: operate over N regions for each timepoint ---
        # (B, T, N, D) -> (B, T, N, D)
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)
        res = x
        x = self.norm(x)
        x = self.spatial_model(x, x, x)[0]
        x = x + res
        # Output shape: (B, T, N, D)
        return x.view(B, T, N, D)

class TemporalRegionAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(TemporalModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.temporal_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)

        # Initialize RoPE embeddings for temporal modeling
        rope_embeddings = self._init_rope(d_model)
        self.register_buffer('rope_embeddings', rope_embeddings)


    def _init_rope(self, d_model):
        """Initialize RoPE embeddings for temporal sequences"""
        # Create fixed position embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        return inv_freq
        
    def _apply_rotary_pos_emb(self, x, seq_dim=1):
        """Apply rotary positional embeddings to input tensor"""
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.shape[seq_dim]
        
        # Create position indices
        position = torch.arange(seq_len, device=x.device).float()
        
        # Compute sinusoidal positions
        sinusoidal_pos = torch.einsum("i,j->ij", position, self.rope_embeddings)
        sin_pos = torch.sin(sinusoidal_pos)
        cos_pos = torch.cos(sinusoidal_pos)
        
        # Apply rotary embeddings
        x_rot = x.clone()
        x_rot[..., 0::2] = x[..., 0::2] * cos_pos - x[..., 1::2] * sin_pos
        x_rot[..., 1::2] = x[..., 0::2] * sin_pos + x[..., 1::2] * cos_pos
        
        return x_rot

    def forward(self, x):
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, N, T, D) -> (B*N, T, D)
        x = x.view(B * N, T, D)

        res = x
        x = self.norm(x)

        # Apply RoPE embeddings to temporal modeling
        x_rope = self._apply_rotary_pos_emb(x, seq_dim=1)
        T_temporal = x.shape[1]
        causal_mask = torch.triu(torch.ones(T_temporal, T_temporal, device=x.device), diagonal=1).bool()
        x = self.temporal_model(x_rope, x_rope, x, attn_mask=causal_mask)[0]
        x = x + res

        # --- Restore original axes ---
        # (B*N, T, D) -> (B, N, T, D)
        x = x.view(B, N, T, D)
        # Swap back: (B, N, T, D) -> (B, T, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

class VoxelAttention(nn.Module):
    """
    Self-attention over a 3-D voxel grid, applied *per* time-step.
    Input  : (B, T, C, W, H, D)
    Output : (B, T, C, W, H, D)     (residual connection)
    """
    def __init__(self, dim: int):
        """
        Args
        ----
        dim : number of channels C
        """
        super().__init__()
        self.norm   = Conv4dRMSNorm(dim)
        self.to_qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj   = nn.Conv3d(dim, dim,     kernel_size=1, bias=False)

        # zero-init so the block starts as an identity map
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, C, W, H, D)

        Returns
        -------
        Tensor, same shape as x
        """
        B, T, C, W, H, D = x.shape
        identity = x

        # ── 1. merge time into batch & normalise ──────────────────────────────
        x = self.norm(x)                       # Conv4dRMSNorm along channels
        x = x.view(B * T, C, W, H, D)          # (B·T, C, W, H, D)

        # ── 2. compute Q, K, V ────────────────────────────────────────────────
        qkv = self.to_qkv(x)                   # (B·T, 3C, W, H, D)
        qkv = qkv.reshape(B * T, 3, C, -1)     # (B·T, 3, C, N)  , N=W·H·D
        qkv = qkv.permute(0, 1, 3, 2)          # (B·T, 3, N, C)
        q, k, v = qkv.unbind(dim=1)            # each (B·T, N, C)

        # add synthetic head dimension expected by SDP-attention (heads=1)
        q = q.unsqueeze(1)                     # (B·T, 1, N, C)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # ── 3. scaled-dot-product attention (torch ≥ 2.0) ─────────────────────
        out = F.scaled_dot_product_attention(q, k, v)  # (B·T, 1, N, C)
        out = out.squeeze(1)                            # (B·T, N, C)

        # ── 4. reshape back to volume & project ───────────────────────────────
        out = out.permute(0, 2, 1).contiguous()        # (B·T, C, N)
        out = out.view(B * T, C, W, H, D)              # (B·T, C, W, H, D)
        out = self.proj(out)                           # 1×1×1 conv

        # ── 5. restore (B, T, …) layout + residual ────────────────────────────
        out = out.view(B, T, C, W, H, D)

        return out + identity

