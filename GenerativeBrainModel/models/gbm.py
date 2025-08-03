import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.convnormencoder import ConvNormEncoder
from mamba_ssm import Mamba2 as Mamba
import pdb
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


class SpatioTemporalRegionModel(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatioTemporalRegionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.temporal_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)#Mamba(d_model=d_model)
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.norm_3 = RMSNorm(d_model)

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
        # (B, T, N, D) -> (B*T, N, D)
        
        x = x.reshape(B * T, N, D)

        res = x
        x = self.norm_3(x)

        x = self.spatial_model(x, x, x)[0]
        x = x + res
        

        # --- Swap T and N for temporal modeling ---
        # (B*T, N, D) -> (B, T, N, D)
        x = x.view(B, T, N, D)
        # Swap T and N: (B, T, N, D) -> (B, N, T, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, N, T, D) -> (B*N, T, D)
        x = x.view(B * N, T, D)

        res = x
        x = self.norm_1(x)

        # Apply RoPE embeddings to temporal modeling
        x_rope = self._apply_rotary_pos_emb(x, seq_dim=1)
        
        # Create causal attention mask for temporal modeling
        T_temporal = x.shape[1]
        causal_mask = torch.triu(torch.ones(T_temporal, T_temporal, device=x.device), diagonal=1).bool()
        x = self.temporal_model(x_rope, x_rope, x, attn_mask=causal_mask)[0]
        x = x + res
        
        

        # --- Restore original axes ---
        # (B*N, T, D) -> (B, N, T, D)
        x = x.view(B, N, T, D)
        # Swap back: (B, N, T, D) -> (B, T, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        res = x

        x = self.norm_2(x)

        # --- MLP block ---
        

        x = self.FFN(x)

        
        x = x + res
            
        # Output shape: (B, T, N, D)
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


class GBM(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, autoencoder_path=None, volume_size=(256, 128, 30), region_size=(32, 16, 2)):
        super(GBM, self).__init__()

        # asserting divisibility
        assert volume_size[0] % region_size[0] == 0 and volume_size[1] % region_size[1] == 0 and volume_size[2] % region_size[2] == 0, "volume_size must be divisible by region_size"
        # asserting that region_size is a factor of volume_size
        assert np.prod(volume_size) % np.prod(region_size) == 0, "region_size must be a factor of volume_size"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = np.prod(volume_size) // np.prod(region_size)
        self.region_size = region_size
        self.volume_size = volume_size
        self.n_layers = n_layers
        self.n_blocks_x = volume_size[0] // region_size[0]
        self.n_blocks_y = volume_size[1] // region_size[1]
        self.n_blocks_z = volume_size[2] // region_size[2]
        self.layers = nn.ModuleList([SpatioTemporalRegionModel(d_model, n_heads, self.n_regions) for _ in range(n_layers)])

        self.final_spatial_mixer = SpatialModel(d_model, n_heads, self.n_regions)


        # initialize the autoencoder which has frozen weights
        self.autoencoder = ConvNormEncoder(
            hidden_channels=d_model,
            volume_size=volume_size,
            region_size=region_size,
            num_frequencies=32,
            sigma=1.0
        )

        if autoencoder_path is not None:
            checkpoint = torch.load(autoencoder_path, map_location=torch.device('cpu'))
            # Handle both direct model state_dict and full training checkpoints
            if 'model_state_dict' in checkpoint:
                # Full training checkpoint - extract model weights
                model_state_dict = checkpoint['model_state_dict']
            else:
                # Direct model state_dict
                model_state_dict = checkpoint
            
            # Handle torch.compile() prefixes (_orig_mod.) in state dict keys
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                # Remove _orig_mod. prefix if present (from torch.compile)
                if key.startswith('_orig_mod.'):
                    cleaned_key = key[len('_orig_mod.'):]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value
            
            self.autoencoder.load_state_dict(cleaned_state_dict)


    def forward(self, x, get_logits=True,):
        # Takes as input sequences of shape (batch_size, seq_len, volume_size**)
        # Returns sequences of shape (batch_size, seq_len, volume_size**)
        B, T, *vol_size = x.shape

        assert list(vol_size) == list(self.volume_size), "volume_size must match the input volume size"

        # reshape to (batch_size, seq_len, n_regions, d_model) where the whole volume is divided up into 3d regions
        # Spatially-aware reshaping: split the volume into regions (macro-blocks), preserving spatial locality.
        # x: (B, T, X, Y, Z) where (X, Y, Z) == volume_size

        # Unpack region and volume sizes
        region_x, region_y, region_z = self.region_size
        vol_x, vol_y, vol_z = self.volume_size


        # Encode the regions
        jump_res = x
        x = self.autoencoder.encode(x, apply_norm=False) # (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        # Move hidden_channels to the end, then flatten spatial dims to n_regions
        x = x.permute(0, 1, 3, 4, 5, 2).reshape(B, T, self.n_regions, self.d_model) # (B, T, n_regions, d_model)

        

        for i, layer in enumerate(self.layers):
            x = layer(x)


        x = self.final_spatial_mixer(x)

        # Reshape n_regions back to (n_blocks_x, n_blocks_y, n_blocks_z, d_model)
        x = x.view(B, T, self.n_blocks_x, self.n_blocks_y, self.n_blocks_z, self.d_model)  # (B, T, n_blocks_x, n_blocks_y, n_blocks_z, d_model)

        # Move d_model to channel position for decoder: (B, T, d_model, n_blocks_x, n_blocks_y, n_blocks_z)
        x = x.permute(0, 1, 5, 2, 3, 4)

        # Decode the regions
        x = self.autoencoder.decode(x, get_logits=True, apply_norm=True)

        # Reshape back to original volume shape: (B, T, X, Y, Z)
        x = x.reshape(B, T, vol_x, vol_y, vol_z)

        x = x + torch.logit(jump_res, eps=1e-4)

        if get_logits:
            return x

        return torch.sigmoid(x)

    def autoregress(self, init_x, n_steps, context_len=12):
        # init_x: (B, T, X, Y, Z)
        # n_steps: number of steps to generate
        # context_len: number of steps to use as context
        B, T, *vol_size = init_x.shape
        assert T >= context_len, "context_len must be less than or equal to T"
        assert n_steps > 0, "n_steps must be greater than 0"
        assert context_len > 0, "context_len must be greater than 0"

        # Start with the initial sequence
        current_sequence = init_x
        
        # generate n_steps steps
        for i in range(n_steps):
            # get the context from the current sequence
            context = current_sequence[:, -context_len:]
            # generate the next step
            next_step = self.forward(context, get_logits=False)[:, -1:]  # (B, 1, X, Y, Z)
            # append the next step to the current sequence
            current_sequence = torch.cat([current_sequence, next_step], dim=1)

        # Return the full sequence including the original init_x and the generated steps
        return current_sequence
