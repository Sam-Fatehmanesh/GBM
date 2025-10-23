import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import SparseSpikeFullAttention, NeuronCausalAttention
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
# from mamba_ssm import Mamba2 as Mamba
import os

class GlobalMeanPool(nn.Module):
    """Pools across the neuron (N) dimension, keeping dims."""
    def forward(self, x):
        # x: (B, T, N, D)
        return x.mean(dim=2, keepdim=True)  # (B, T, 1, D)


class SpatioTemporalNeuralAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SpatioTemporalNeuralAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.spatial_attention = SparseSpikeFullAttention(d_model=d_model, n_heads=n_heads)
        self.temporal_attention = NeuronCausalAttention(d_model=d_model, n_heads=n_heads)

        
        self.FFN0 = FFN(d_model, d_model*3)
        self.FFN1 = FFN(d_model, d_model*3)



        self.FFN_global_mean_pool = nn.Sequential(
            FFN(d_model, d_model*3),
            GlobalMeanPool(),
            RMSNorm(d_model),
            # Output shape: (B, T, N, D) -> (B, T, 1, D)
        )


    def forward(self, x, point_positions, neuron_pad_mask):
        # expects x of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N, D = x.shape

        # d_dtype = x.dtype
        # if point_positions.dtype != d_dtype:
        #     point_positions = point_positions.to(d_dtype)
        # if neuron_pad_mask is not None and neuron_pad_mask.dtype != d_dtype:
        #     neuron_pad_mask = neuron_pad_mask.to(d_dtype)

        global_mean_pool = self.FFN_global_mean_pool(x)
        x = torch.cat([x, global_mean_pool], dim=2) # (B, T, n_neurons + 1, d_model)
        neuron_pad_mask = torch.cat([neuron_pad_mask, torch.ones(B, 1, device=neuron_pad_mask.device)], dim=1) # (B, n_neurons + 1)
        point_positions = torch.cat([point_positions, torch.zeros(B, 1, 3, device=point_positions.device, dtype=x.dtype)], dim=1) # (B, n_neurons + 1, 3)

        x = self.spatial_attention(x, point_positions, neuron_pad_mask)

        x = x[:, :, :-1, :]
        neuron_pad_mask = neuron_pad_mask[:, :-1]
        point_positions = point_positions[:, :-1, :]

        x = self.FFN0(x)
        
        x = self.temporal_attention(x, neuron_pad_mask)

        x = self.FFN1(x)
        
        # Output shape: (B, T, N, D)
        return x

