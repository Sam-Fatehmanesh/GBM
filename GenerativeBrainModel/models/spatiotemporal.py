import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import SpatialNeuralAttention, TemporalNeuralAttention
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
# from mamba_ssm import Mamba2 as Mamba
import os


class SpatioTemporalNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SpatioTemporalNeuralAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.spatial_attention = SpatialNeuralAttention(d_model, n_heads)
        self.temporal_attention = TemporalNeuralAttention(d_model, n_heads)
        self.conv = nn.Sequential(
                CausalResidualNeuralConv1d(d_model, kernel_size=9),
                CausalResidualNeuralConv1d(d_model, kernel_size=9),
            )
        

        
        self.FFN0 = FFN(d_model, d_model*3)
        self.FFN1 = FFN(d_model, d_model*3)
        

    def forward(self, x, point_positions, neuron_pad_mask=None):
        # expects x of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N, D = x.shape

        d_dtype = x.dtype
        if point_positions.dtype != d_dtype:
            point_positions = point_positions.to(d_dtype)
        if neuron_pad_mask is not None and neuron_pad_mask.dtype != d_dtype:
            neuron_pad_mask = neuron_pad_mask.to(d_dtype)

        x = self.conv(x)

        # x = self.spatial_attention(x, point_positions, neuron_pad_mask)

        x = self.FFN0(x)
        
        x = self.temporal_attention(x, neuron_pad_mask)

        x = self.FFN1(x)
        
        # Output shape: (B, T, N, D)
        return x

