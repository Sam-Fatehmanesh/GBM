import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from mamba_ssm import Mamba2 as Mamba
import os

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)        
        self.linear_2 = nn.Linear(d_model, d_ff)
        self.act = nn.SiLU()
        self.linear_3 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x1 = self.act(self.linear_1(x))
        x2 = self.linear_2(x)
        x = self.linear_3(x2 * x1)
        return x

class SpatialModel(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatialModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)
        res = x
        x = self.norm(x)
        x = self.spatial_model(x, x, x)[0]
        x = x + res
        # Output shape: (B, T, N, D)
        return x.view(B, T, N, D)

class TemporalModel(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(TemporalModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.temporal_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)

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



class SpatioTemporalBrainModel(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatioTemporalRegionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_model = SpatialModel(d_model, n_heads, n_regions)
        self.temporal_model = TemporalModel(d_model, n_heads, n_regions)
        self.norm = RMSNorm(d_model)

        self.FFN = FFN(d_model, d_model*2)
        
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

    def forward(self, x, apply_last_norm=True):
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape

        # --- Spatial Attention: operate over regions for each timepoint ---
        # (B, T, N, D) -> (B, T, N, D)
        x = self.spatial_model(x)
        

        x = self.temporal_model(x)

        res = x

        x = self.norm(x)

        # --- FFN block ---
    
        x = self.FFN(x)

        x = x + res
            
        # Output shape: (B, T, N, D)
        return x

