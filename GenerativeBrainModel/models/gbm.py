import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.spatiotemporal import SpatioTemporalNeuralAttention
# cimport pdb
import os


LOGRATE_EPS = 1e-6


class GBM(nn.Module):
    def __init__(self, d_model, d_stimuli, n_heads, n_layers):
        super(GBM, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_stimuli = d_stimuli

        self.layers = nn.ModuleList([SpatioTemporalNeuralAttention(d_model, n_heads) for _ in range(n_layers)])
        
        self.stimuli_encoder = nn.Sequential(
            nn.Linear(d_stimuli, d_model),
            RMSNorm(d_model),
        )
        self.neuron_scalar_position_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            RMSNorm(d_model),
        )
        self.neuron_scalar_decoder_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 4),
        )



        # Parameter groups for optimizers
        # Treat encoders as "embed", attention stack as "body", decoder head as "head", muon optimizer is applied to the body only and adamw is applied to the embed and head
        self.embed = nn.ModuleDict({
            'stim': self.stimuli_encoder,
            'neuron': self.neuron_scalar_position_encoder,
        })
        self.body = self.layers
        self.head = nn.ModuleDict({
            'neuron': self.neuron_scalar_decoder_head,
        })


    def forward(self, x, x_stimuli, point_positions, neuron_pad_mask, get_logits=True, input_log_rates=False):
        # Takes as input sequences of shape x: (batch_size, seq_len, n_neurons)
        # x_stimuli: (batch_size, seq_len, d_stimuli)
        # point_positions: (batch_size, n_neurons, 3)
        # neuron_pad_mask: (batch_size, n_neurons)
        # Returns sequences of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N = x.shape

        if not input_log_rates:
            # Convert input spike rates to log-rates immediately for downstream processing
            x = torch.log(x.clamp_min(LOGRATE_EPS))

        target_dtype = next(self.parameters()).dtype

        x = x.unsqueeze(-1)



        params_dtype = target_dtype

        # Keep dtype consistent with model weights (bf16 when enabled)
        if x.dtype != params_dtype:
            x = x.to(params_dtype)
        # Ensure point_positions dtype matches as well
        if point_positions.dtype != params_dtype:
            point_positions = point_positions.to(params_dtype)

        # point_positions: (B, N, 3)
        # Compute centroid (mean) and RMS scale for each batch to normalize positions
        centroid = point_positions.mean(dim=1, keepdim=True)  # (B, 1, 3)
        pos_centered = point_positions - centroid              # (B, N, 3)
        # Compute RMS scale (root mean square distance from centroid)
        r2 = (pos_centered ** 2).sum(dim=2)                   # (B, N)
        scale = (r2.mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()).unsqueeze(-1)  # (B, 1, 1)
        rel_point_positions = pos_centered / scale             # (B, N, 3) -- zero mean, unit RMS

        # Concatenate the neuron scalars and the relative point positions by repeating for the entire sequence
        x = torch.cat([x, rel_point_positions.unsqueeze(1).repeat(1, T, 1, 1)], dim=3)

        # Encode the neuron scalars and the point positions
        x = self.neuron_scalar_position_encoder(x)

        # (B, T, d_stimuli) -> (B, T, 1, d_model), acts as a global stimulus embedding as a token for each time step
        x_stimuli = x_stimuli.to(params_dtype).unsqueeze(2)
        x_stimuli = self.stimuli_encoder(x_stimuli) # (B, T, 1, d_model)

        # concatenate the stimulus embedding to the input
        x = torch.cat([x, x_stimuli], dim=2) # (B, T, n_neurons + 1, d_model)
        
        # Adds an additional 1 value to neuron_pad_mask to account for the stimulus token
        neuron_pad_mask = torch.cat([neuron_pad_mask, torch.ones(B, 1, device=neuron_pad_mask.device)], dim=1) # (B, n_neurons + 1)

        # Adds a zero vector to point_positions to account for the stimulus token
        point_positions = torch.cat([point_positions, torch.zeros(B, 1, 3, device=point_positions.device, dtype=params_dtype)], dim=1) # (B, n_neurons + 1, 3)

        # Apply the layers
        for i, layer in enumerate(self.layers):
            x = layer(x, point_positions, neuron_pad_mask)

        # Remove the stimulus token
        x = x[:, :, :-1, :]


        # Decode the neuron scalars
        x = self.neuron_scalar_decoder_head(x)

        # SAS parameters: mu, log_sigma_raw, eta, log_delta_raw
        mu = x[..., 0]
        log_sigma_raw = x[..., 1]
        eta = x[..., 2]
        log_delta_raw = x[..., 3]

        if get_logits:
            return mu, log_sigma_raw, eta, log_delta_raw

        from GenerativeBrainModel.utils.sas import sas_rate_median

        median = sas_rate_median(mu, log_sigma_raw, eta, log_delta_raw)
        return median.to(mu.dtype)


    def autoregress(self, init_x, init_stimuli, point_positions, neuron_pad_mask, future_stimuli=None, n_steps=10, context_len=12):
        """
        Deterministic rollout using SAS median.
        init_x:         (B, T, N)   rates (Hz)
        init_stimuli:   (B, T, dS)
        point_positions:(B, N, 3)
        neuron_pad_mask:(B, N)
        future_stimuli: (B, n_steps, dS) or None -> zeros
        Returns:        (B, T + n_steps, N)
        """
        B, T, N = init_x.shape
        assert T >= context_len, "context_len must be ≤ T"
        assert n_steps > 0 and context_len > 0

        if future_stimuli is None:
            future_stimuli = torch.zeros(B, n_steps, self.d_stimuli, device=init_x.device, dtype=init_x.dtype)
        else:
            assert future_stimuli.shape == (B, n_steps, self.d_stimuli)

        # Convert initial rates to log space once
        current_log_rates = torch.log(init_x.clamp_min(LOGRATE_EPS))
        current_stimuli = init_stimuli

        from GenerativeBrainModel.utils.sas import sas_log_median

        for i in range(n_steps):
            # Use log-rates throughout
            context_log = current_log_rates[:, -context_len:]
            context_stim = current_stimuli[:, -context_len:, :]
            mu, log_sigma_raw, eta, log_delta_raw = self.forward(
                context_log,
                context_stim,
                point_positions,
                neuron_pad_mask,
                get_logits=True,
                input_log_rates=True,
            )
            # The model outputs parameters for log-rate, so we use sas_log_median to get the next log-rate
            next_log_rate = sas_log_median(
                mu[:, -1:, :],
                log_sigma_raw[:, -1:, :],
                eta[:, -1:, :],
                log_delta_raw[:, -1:, :],
            )
            # Keep everything in log space
            current_log_rates = torch.cat([current_log_rates, next_log_rate], dim=1)
            current_stimuli = torch.cat([current_stimuli, future_stimuli[:, i:i+1, :]], dim=1)

        # Return rates in Hz (exp of log-rates)
        return current_log_rates.exp()
