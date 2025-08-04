import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import SpatialRegionAttention, TemporalRegionAttention
from mamba_ssm import Mamba2 as Mamba
import os


class SpatioTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatioTemporalRegionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_attention = SpatialRegionAttention(d_model, n_heads, n_regions)
        self.temporal_attention = TemporalRegionAttention(d_model, n_heads, n_regions)

        self.FFN0 = FFN(d_model, d_model*2)
        self.FFN1 = FFN(d_model, d_model*2)
        

    def forward(self, x):
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape

        x = self.FFN0(x)
        
        x = self.spatial_attention(x)

        x = self.FFN1(x)
        
        x = self.temporal_attention(x)
        
        # Output shape: (B, T, N, D)
        return x

