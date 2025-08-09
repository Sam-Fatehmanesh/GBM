import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv4dRMSNorm
from einops import rearrange
import numpy as np
from GenerativeBrainModel.models.mlp import MLP
from mamba_ssm import Mamba2 as Mamba



class SpatialNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_rope_features=32):
        super(SpatialNeuralAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_rope_features = n_rope_features
        self.spatial_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = RMSNorm(d_model)
        # Initialize rope projection layer explicitly to avoid device mismatches
        self.rope_proj = nn.Linear(2 * n_rope_features, d_model, bias=False)

        # Directional (isotropic) RoPE: directions and magnitudes
        # Directions: random unit vectors on S^2
        dirs = torch.randn(n_rope_features, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        # Magnitudes: log-spaced as in standard RoPE
        min_freq = 1.0
        max_freq = 10000.0
        freqs = torch.logspace(np.log10(min_freq), np.log10(max_freq), n_rope_features)
        self.register_buffer('rope_dirs', dirs, persistent=False)
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _directional_rope(self, positions):
        """
        positions: (B, N, 3)
        Returns: (B, N, 2 * n_rope_features)
        """
        # Project positions onto each direction: (B, N, n_rope_features)
        proj = torch.einsum('bnd,fd->bnf', positions, self.rope_dirs)  # (B, N, F)
        # Multiply by frequencies
        angles = proj * self.rope_freqs  # (B, N, F)
        # Sin/cos
        rope_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, N, 2F)
        return rope_emb

    def _apply_rope(self, x, rope_emb):
        """
        x: (B*T, N, D)
        rope_emb: (B, N, 2F)
        Returns: (B*T, N, D)
        """
        # Project RoPE embedding to d_model and add to x
        # rope_emb: (B, N, 2F) -> (B, N, D)
        B_T, N, D = x.shape
        B = rope_emb.shape[0]
        T = B_T // B
        rope_emb = self.rope_proj(rope_emb)  # (B, N, D)
        rope_emb = rope_emb.unsqueeze(1).expand(B, T, N, D).reshape(B_T, N, D)
        return x + rope_emb

    def forward(self, x, point_positions, neuron_pad_mask=None):
        # x: (B, T, N, D)
        # point_positions: (B, N, 3)
        # expects neuron_pad_mask of shape (batch_size, n_neurons)
        B, T, N, D = x.shape
        x = x.reshape(B * T, N, D)
        res = x
        x = self.norm(x)
        # Directional RoPE
        rope_emb = self._directional_rope(point_positions)  # (B, N, 2F)
        x_rope = self._apply_rope(x, rope_emb)  # (B*T, N, D)

        # Prepare key_padding_mask for MultiheadAttention
        # MultiheadAttention expects key_padding_mask of shape (B*T, N), True for PAD
        key_padding_mask = None
        if neuron_pad_mask is not None:
            # neuron_pad_mask: (B, N) with 1 for valid, 0 for pad
            # Convert to bool mask: True for pad, False for valid
            key_padding_mask = (neuron_pad_mask == 0)  # (B, N), bool
            # Repeat for each time step in T
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(B, T, N).reshape(B * T, N)

        x = self.spatial_model(x_rope, x_rope, x, key_padding_mask=key_padding_mask)[0]
        x = x + res
        return x.view(B, T, N, D)

class TemporalNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(TemporalNeuralAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
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
        d_model = x.shape[-1]
        half_dim = d_model // 2

        # Create position indices
        position = torch.arange(seq_len, device=x.device).float()  # (seq_len,)

        # Compute sinusoidal positions
        sinusoidal_pos = torch.einsum("i,j->ij", position, self.rope_embeddings)  # (seq_len, half_dim)
        sin_pos = torch.sin(sinusoidal_pos)  # (seq_len, half_dim)
        cos_pos = torch.cos(sinusoidal_pos)  # (seq_len, half_dim)

        # Expand sin/cos to match x's batch dimension
        # x: (B*N, T, D), sin/cos: (T, half_dim) -> (B*N, T, half_dim)
        batch_size = x.shape[0]
        sin_pos = sin_pos.unsqueeze(0).expand(batch_size, seq_len, half_dim)
        cos_pos = cos_pos.unsqueeze(0).expand(batch_size, seq_len, half_dim)

        # Apply rotary embeddings explicitly
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        # Interleave even and odd features
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        return x_rot

    def forward(self, x, neuron_pad_mask=None):
        # expects x of shape (batch_size, seq_len, n_neurons, d_model)
        # expects neuron_pad_mask of shape (batch_size, n_neurons)
        B, T, N, D = x.shape

        # (B, T, N, D) -> (B, N, T, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, N, T, D) -> (B*N, T, D)
        x = x.view(B * N, T, D)

        res = x
        x = self.norm(x)

        # Apply RoPE embeddings to temporal modeling
        x_rope = self._apply_rotary_pos_emb(x, seq_dim=1)
        T_temporal = x.shape[1]
        causal_mask = torch.triu(torch.ones(T_temporal, T_temporal, device=x.device), diagonal=1).bool()  # (T, T)

        # Prepare key_padding_mask for MultiheadAttention (for padded neurons)
        # MultiheadAttention expects key_padding_mask of shape (B*N, T), True for PAD
        key_padding_mask = None
        if neuron_pad_mask is not None:
            # neuron_pad_mask: (B, N) with 1 for valid, 0 for pad
            # We want to mask all time steps for padded neurons
            # So, for each (B, N), if neuron_pad_mask == 0, mask all T for that row
            neuron_pad_mask_flat = (neuron_pad_mask == 0).reshape(-1)  # (B*N,)
            key_padding_mask = neuron_pad_mask_flat.unsqueeze(1).expand(B * N, T)  # (B*N, T)
        # Pass both attn_mask (causal) and key_padding_mask (neuron padding)
        x = self.temporal_model(
            x_rope, x_rope, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )[0]
        x = x + res

        # --- Restore original axes ---
        # (B*N, T, D) -> (B, N, T, D)
        x = x.view(B, N, T, D)
        # Swap back: (B, N, T, D) -> (B, T, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x
