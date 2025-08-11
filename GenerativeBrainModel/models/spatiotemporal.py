import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import SpatialNeuralAttention, TemporalNeuralAttention
# from mamba_ssm import Mamba2 as Mamba
import os


class SpatioTemporalNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SpatioTemporalNeuralAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.spatial_attention = SpatialNeuralAttention(d_model, n_heads)
        self.temporal_attention = TemporalNeuralAttention(d_model, n_heads)

        self.FFN0 = FFN(d_model, d_model*2)
        self.FFN1 = FFN(d_model, d_model*2)
        

    def forward(self, x, point_positions, neuron_pad_mask=None):
        # expects x of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N, D = x.shape

        x = self.spatial_attention(x, point_positions, neuron_pad_mask)

        x = self.FFN0(x)
        
        x = self.temporal_attention(x, neuron_pad_mask)

        x = self.FFN1(x)
        
        # Output shape: (B, T, N, D)
        return x

