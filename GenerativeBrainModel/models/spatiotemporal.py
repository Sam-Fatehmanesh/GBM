import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import (
    SparseSpikeFullAttention,
    NeuronCausalAttention,
)
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled

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
        self.spatial_attention = SparseSpikeFullAttention(
            d_model=d_model, n_heads=n_heads
        )
        self.temporal_attention = NeuronCausalAttention(
            d_model=d_model, n_heads=n_heads
        )

        self.FFN0 = FFN(d_model, d_model * 3)
        self.FFN1 = FFN(d_model, d_model * 3)

        self.FFN_global_mean_pool = nn.Sequential(
            FFN(d_model, d_model * 3),
            GlobalMeanPool(),
            RMSNorm(d_model),
            # Output shape: (B, T, N, D) -> (B, T, 1, D)
        )

    def forward(self, x, point_positions, neuron_pad_mask, neuron_spike_probs):
        # expects x of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N, D = x.shape

        # d_dtype = x.dtype
        # if point_positions.dtype != d_dtype:
        #     point_positions = point_positions.to(d_dtype)
        # if neuron_pad_mask is not None and neuron_pad_mask.dtype != d_dtype:
        #     neuron_pad_mask = neuron_pad_mask.to(d_dtype)

        global_mean_pool = self.FFN_global_mean_pool(x)
        x = torch.cat([x, global_mean_pool], dim=2)  # (B, T, n_neurons + 1, d_model)
        neuron_pad_mask = torch.cat(
            [neuron_pad_mask, torch.ones(B, 1, device=neuron_pad_mask.device)], dim=1
        )  # (B, n_neurons + 1)
        point_positions = torch.cat(
            [
                point_positions,
                torch.zeros(B, 1, 3, device=point_positions.device, dtype=x.dtype),
            ],
            dim=1,
        )  # (B, n_neurons + 1, 3)

        # Add a one to the neuron_spike_probs to account for the stimulus token
        neuron_spike_probs = torch.cat(
            [
                neuron_spike_probs,
                torch.ones_like(neuron_spike_probs[:, :, :1]).to(torch.float32),
            ],
            dim=2,
        )  # (B, T, n_neurons + 1)
        # Sample the neuron spike probabilities (clamp to [0,1] in fp32 to avoid device-side asserts)
        probs_fp32 = torch.nan_to_num(
            neuron_spike_probs.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp_(0.0, 1.0)
        if debug_enabled():
            assert_no_nan(probs_fp32, "STNA.probs_fp32_before_bernoulli")
        neuron_spike_mask = torch.bernoulli(probs_fp32).to(torch.bool)

        x = self.spatial_attention(
            x, point_positions, neuron_pad_mask, neuron_spike_mask
        )
        if debug_enabled():
            assert_no_nan(x, "STNA.after_spatial")

        x = x[:, :, :-1, :]
        neuron_pad_mask = neuron_pad_mask[:, :-1]
        point_positions = point_positions[:, :-1, :]

        x = self.FFN0(x)

        x = self.temporal_attention(x, neuron_pad_mask)
        if debug_enabled():
            assert_no_nan(x, "STNA.after_temporal")

        x = self.FFN1(x)

        # Output shape: (B, T, N, D)
        return x
