import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
from GenerativeBrainModel.models.spatiotemporal import SpatioTemporalNeuralAttention
# cimport pdb
import os


LOGRATE_EPS = 1e-6


class GBM(nn.Module):
    def __init__(self, d_model, d_stimuli, n_heads, n_layers,
                 num_neurons_total: int,
                 neuron_pad_id: int = 0):
        super(GBM, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_stimuli = d_stimuli
        self.neuron_pad_id = int(neuron_pad_id)

        self.layers = nn.ModuleList([SpatioTemporalNeuralAttention(d_model, n_heads) for _ in range(n_layers)])

        self.conv = nn.Sequential(
                CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=1),
                CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=2),
                CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=4),
            )
        
        self.stimuli_encoder = nn.Sequential(
            nn.Linear(d_stimuli, d_model),
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
        })
        self.body = self.layers
        self.head = nn.ModuleDict({
            'neuron': self.neuron_scalar_decoder_head,
        })

        # Per-neuron embeddings (added after conv stack)
        self.neuron_embed = nn.Embedding(int(num_neurons_total) + 1, d_model, padding_idx=self.neuron_pad_id)

    def forward(self, x, x_stimuli, point_positions, neuron_pad_mask, neuron_ids, get_logits=True, input_log_rates=False):
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


        # (B, T, d_stimuli) -> (B, T, 1, d_model), acts as a global stimulus embedding as a token for each time step
        x_stimuli = x_stimuli.to(params_dtype).unsqueeze(2)
        x_stimuli = self.stimuli_encoder(x_stimuli) # (B, T, 1, d_model)

        # Tile x to (B, T, n_neurons, d_model)
        x = x.repeat(1, 1, N, 1)

        # concatenate the stimulus embedding to the input
        x = torch.cat([x, x_stimuli], dim=2) # (B, T, n_neurons + 1, d_model)

        # Apply the convolutional layers
        x = self.conv(x)

        # Add per-neuron embeddings
        # neuron_ids: (B, N) long; pad entries should be neuron_pad_id (0)
        if neuron_ids.dtype != torch.long:
            neuron_ids = neuron_ids.to(torch.long)
        # Map arbitrary 64-bit IDs into embedding table range while preserving 0 as pad
        num_emb = int(self.neuron_embed.num_embeddings)
        if num_emb > 1:
            mod_base = num_emb - 1
            # ids==0 stays 0; real ids mapped to [1, num_emb-1]
            ids_mod = torch.where(
                neuron_ids == self.neuron_pad_id,
                torch.zeros_like(neuron_ids),
                ((neuron_ids - 1) % mod_base) + 1,
            )
        else:
            ids_mod = torch.zeros_like(neuron_ids)
        emb = self.neuron_embed(ids_mod)  # (B,N,d_embed)
        # Only add to the neuron tokens (exclude stimulus token at index N)
        x_neurons = x[:, :, :N, :]
        x_neurons = x_neurons + emb.unsqueeze(1)
        x = torch.cat([x_neurons, x[:, :, N:, :]], dim=2)
    
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



    def autoregress_sample(
        self,
        init_x,                 # (B, T, N)   rates (Hz)
        init_stimuli,           # (B, T, dS)
        point_positions,        # (B, N, 3)
        neuron_pad_mask,        # (B, N)
        future_stimuli=None,    # (B, n_steps, dS) or None -> zeros
        n_steps=10,
        context_len=12,
        eps=LOGRATE_EPS,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        return_log: bool = False,
    ):
        """
        Stochastic rollout using SAS sampling on log-rate:
        Y = mu + sigma * sinh((U - eta) / delta),  U ~ N(0, temperature^2).
        Returns: (B, T + n_steps, N) rates (or log-rates if return_log=True).
        """
        assert n_steps > 0 and context_len > 0
        B, T, N = init_x.shape
        assert T >= context_len, "context_len must be ≤ T"

        if future_stimuli is None:
            future_stimuli = torch.zeros(B, n_steps, self.d_stimuli, device=init_x.device, dtype=init_x.dtype)
        else:
            assert future_stimuli.shape == (B, n_steps, self.d_stimuli)

        params_dtype = next(self.parameters()).dtype
        current_log_rates = torch.log(init_x.clamp_min(eps) + eps).to(params_dtype)
        current_stimuli   = init_stimuli.to(params_dtype)
        point_positions   = point_positions.to(params_dtype)

        POS_EPS = 1e-8
        def _pos(x): return F.softplus(x) + POS_EPS

        for t in range(n_steps):
            # context
            context_log  = current_log_rates[:, -context_len:]
            context_stim = current_stimuli[:, -context_len:, :]

            # one-step-ahead params for log-rate
            mu, log_sigma_raw, eta, log_delta_raw = self.forward(
                context_log,
                context_stim,
                point_positions,
                neuron_pad_mask,
                get_logits=True,
                input_log_rates=True,
            )
            mu   = mu[:, -1:, :]            # (B,1,N)
            sig  = _pos(log_sigma_raw[:, -1:, :])
            eta  = eta[:, -1:, :]
            delt = _pos(log_delta_raw[:, -1:, :])

            # sample next log-rate
            U = torch.randn_like(mu, generator=generator) * float(temperature)
            next_log_rate = mu + sig * torch.sinh((U - eta) / delt)

            # append
            current_log_rates = torch.cat([current_log_rates, next_log_rate], dim=1)
            current_stimuli   = torch.cat([current_stimuli, future_stimuli[:, t:t+1, :].to(params_dtype)], dim=1)

        if return_log:
            return current_log_rates
        # back to rate domain with the SAME eps used going in
        return torch.exp(current_log_rates) - eps

