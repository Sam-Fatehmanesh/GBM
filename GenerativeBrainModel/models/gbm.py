import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
from GenerativeBrainModel.models.spatiotemporal import SpatioTemporalNeuralAttention
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled

# cimport pdb
import os


LOGRATE_EPS = 1e-6


class GBM(nn.Module):
    def __init__(
        self,
        d_model,
        d_stimuli,
        n_heads,
        n_layers,
        num_neurons_total: int,
        neuron_pad_id: int = 0,
        global_neuron_ids: torch.Tensor | None = None,
        cov_rank: int = 32,
        use_ffn_checkpoint: bool = False,
    ):
        super(GBM, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_stimuli = d_stimuli
        self.neuron_pad_id = int(neuron_pad_id)

        self.layers = nn.ModuleList(
            [
                SpatioTemporalNeuralAttention(
                    d_model, n_heads, use_ffn_checkpoint=use_ffn_checkpoint
                )
                for _ in range(n_layers)
            ]
        )

        self.conv = nn.Sequential(
            CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=1),
            CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=2),
            CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=4),
            # CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=8),
            # CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=16),
            # CausalResidualNeuralConv1d(d_model, kernel_size=5, dilation=32),
        )

        self.post_embed_conv_rmsnorm = RMSNorm(d_model)

        self.stimuli_encoder = nn.Sequential(
            nn.Linear(d_stimuli, d_model),
            RMSNorm(d_model),
        )

        self.neuron_scalar_decoder_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 4),
        )

        # Low-rank correlation/covariance factors head (for dependence in loss)
        self.cov_factor_head = nn.Linear(d_model, int(cov_rank), bias=False)
        # Initialize near-zero so early MVN term is small but learnable
        nn.init.normal_(self.cov_factor_head.weight, mean=0.0, std=1e-3)

        # Encodes 3D position with magnitude as an independent input feature.
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, d_model),  # [unit_dir_x, unit_dir_y, unit_dir_z, ||p||]
            RMSNorm(d_model),
        )

        # If a global mapping of unique neuron IDs is provided, build a compact embedding table
        if global_neuron_ids.dtype != torch.long:
            global_neuron_ids = global_neuron_ids.to(torch.long)
        # Sorted unique IDs for deterministic behavior
        uniq_ids, _ = torch.sort(global_neuron_ids.unique())
        # Build an ID->index map via a hash table (Python dict) for CPU lookups; moved to device on forward
        self.register_buffer("global_neuron_ids_sorted", uniq_ids, persistent=True)
        self.register_buffer(
            "has_global_id_map", torch.tensor([1], dtype=torch.uint8), persistent=False
        )
        self.neuron_embed = nn.Embedding(
            int(uniq_ids.numel()) + 1, d_model, padding_idx=self.neuron_pad_id
        )

        # Parameter groups for optimizers
        # Treat encoders as "embed", attention stack as "body", decoder head as "head", muon optimizer is applied to the body only and adamw is applied to the embed and head
        self.embed = nn.ModuleDict(
            {
                "pos": self.pos_encoder,
                "stim": self.stimuli_encoder,
                "neuron": self.neuron_embed,
            }
        )
        self.body = self.layers
        self.head = nn.ModuleDict(
            {
                "neuron": self.neuron_scalar_decoder_head,
                "cov": self.cov_factor_head,
            }
        )

    def forward(
        self,
        x,
        x_stimuli,
        point_positions,
        neuron_pad_mask,
        neuron_ids,
        neuron_spike_probs,
        get_logits=True,
        input_log_rates=False,
        return_factors: bool = False,
    ):
        # Takes as input sequences of shape x: (batch_size, seq_len, n_neurons)
        # x_stimuli: (batch_size, seq_len, d_stimuli)
        # point_positions: (batch_size, n_neurons, 3)
        # neuron_pad_mask: (batch_size, n_neurons)
        # neuron_spike_probs: (batch_size, seq_len, n_neurons)
        # Returns sequences of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N = x.shape

        if not input_log_rates:
            # Convert input spike rates to log-rates immediately for downstream processing
            x = torch.log(x.clamp_min(LOGRATE_EPS))
        if debug_enabled():
            assert_no_nan(x, "GBM.forward.x_log")

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
        pos_centered = point_positions - centroid  # (B, N, 3)
        # Compute RMS scale (root mean square distance from centroid)
        r2 = (pos_centered**2).sum(dim=2)  # (B, N)
        scale = (r2.mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()).unsqueeze(
            -1
        )  # (B, 1, 1)
        rel_point_positions = pos_centered / scale  # (B, N, 3) -- zero mean, unit RMS

        # (B, T, d_stimuli) -> (B, T, 1, d_model), acts as a global stimulus embedding as a token for each time step
        x_stimuli = x_stimuli.to(params_dtype).unsqueeze(2)
        x_stimuli = self.stimuli_encoder(x_stimuli)  # (B, T, 1, d_model)

        # Tile x to (B, T, n_neurons, d_model)
        x = x.repeat(1, 1, 1, self.d_model)

        # concatenate the stimulus embedding to the input
        x = torch.cat([x, x_stimuli], dim=2)  # (B, T, n_neurons + 1, d_model)

        # Apply the convolutional layers
        x = self.conv(x)
        if debug_enabled():
            assert_no_nan(x, "GBM.after_conv")

        # Add per-neuron embeddings
        # neuron_ids: (B, N) long; pad entries should be neuron_pad_id (0)
        if neuron_ids.dtype != torch.long:
            neuron_ids = neuron_ids.to(torch.long)


        if bool(self.has_global_id_map.item()):
            # Map true 64-bit IDs to compact indices via binary search over sorted unique IDs
            # Keep 0 as pad id if present
            ids_flat = neuron_ids.view(-1)
            # torch.searchsorted requires sorted 1D sequence
            idx = torch.searchsorted(self.global_neuron_ids_sorted, ids_flat)
            # Handle IDs not found: set to pad (0) when out-of-range or mismatch
            valid = (idx < self.global_neuron_ids_sorted.numel()) & (
                self.global_neuron_ids_sorted[idx] == ids_flat
            )
            idx = torch.where(
                valid, idx + 1, torch.zeros_like(idx)
            )  # +1 to keep 0 as pad index
            ids_compact = idx.view_as(neuron_ids)
            emb = self.neuron_embed(ids_compact)
        else:
            # Legacy modulo mapping
            num_emb = int(self.neuron_embed.num_embeddings)
            if num_emb > 1:
                mod_base = num_emb - 1
                ids_mod = torch.where(
                    neuron_ids == self.neuron_pad_id,
                    torch.zeros_like(neuron_ids),
                    ((neuron_ids - 1) % mod_base) + 1,
                )
            else:
                ids_mod = torch.zeros_like(neuron_ids)
            emb = self.neuron_embed(ids_mod)  # (B, N, d_embed)
        # Repeat embeddings across the time dimension
        emb = emb.unsqueeze(1).expand(-1, x.shape[1], -1, -1)  # (B, T, N, d_embed)
        # Add only to the neuron tokens (exclude stimulus token at index N)
        x[:, :, :N, :] = x[:, :, :N, :] + emb

        # Add position encoding
        point_magnitudes = torch.norm(rel_point_positions, dim=-1, keepdim=True)
        rel_point_positions_with_magnitude = torch.cat(
            [
                rel_point_positions.div(point_magnitudes.clamp_min(1e-8)),
                point_magnitudes,
            ],
            dim=-1,
        )

        x[:, :, :N, :] = x[:, :, :N, :] + self.pos_encoder(
            rel_point_positions_with_magnitude
        )  # (B, T, n_neurons, d_model)

        x = self.post_embed_conv_rmsnorm(x)
        if debug_enabled():
            assert_no_nan(x, "GBM.after_post_embed_norm")

        # Adds an additional 1 value to neuron_pad_mask to account for the stimulus token
        neuron_pad_mask = torch.cat(
            [neuron_pad_mask, torch.ones(B, 1, device=neuron_pad_mask.device)], dim=1
        )  # (B, n_neurons + 1)

        # Adds a zero vector to rel_point_positions to account for the stimulus token
        rel_point_positions = torch.cat(
            [
                rel_point_positions,
                torch.zeros(
                    B, 1, 3, device=rel_point_positions.device, dtype=params_dtype
                ),
            ],
            dim=1,
        )  # (B, n_neurons + 1, 3)

        # Add a one to the neuron_spike_probs to account for the stimulus token
        neuron_spike_probs = torch.cat(
            [
                neuron_spike_probs,
                torch.ones(
                    B, T, 1, device=neuron_spike_probs.device, dtype=torch.float32
                ),
            ],
            dim=2,
        )  # (B, T, n_neurons + 1)

        # Sample spike mask robustly: clamp to [0,1] in fp32 to avoid bf16 rounding issues
        if neuron_spike_probs.dtype != torch.float32:
            probs_fp32 = neuron_spike_probs.to(torch.float32)
        else:
            probs_fp32 = neuron_spike_probs
        probs_fp32 = torch.nan_to_num(
            probs_fp32, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp_(0.0, 1.0)
        if debug_enabled():
            assert_no_nan(probs_fp32, "GBM.probs_fp32_before_bernoulli")
        neuron_spike_probs = torch.bernoulli(probs_fp32).to(torch.bool)

        # Apply the layers
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_point_positions, neuron_pad_mask, neuron_spike_probs)
            if debug_enabled():
                assert_no_nan(x, f"GBM.layer_{i}_out")

        # Remove the stimulus token
        x = x[:, :, :-1, :]

        # Add the learned neuron embeddings back to the input
        # Only add the *last half* of the embedding, zeros elsewhere
        half = emb.shape[-1] // 2
        pad = emb[:, :, :N, :].new_zeros(emb[:, :, :N, :].shape)
        pad[..., half:] = emb[:, :, :N, half:]
        x[:, :, :N, :] = x[:, :, :N, :] + pad

        # Decode the neuron scalars and low-rank covariance factors from the same normalized hidden
        # Reuse the RMSNorm inside the sequential to keep representation consistent
        h = self.neuron_scalar_decoder_head[0](x)
        x = self.neuron_scalar_decoder_head[1](h)
        factors = self.cov_factor_head(h)  # (B, T, N, r)
        if debug_enabled():
            assert_no_nan(x, "GBM.head_out")

        # SAS parameters: mu, log_sigma_raw, eta, log_delta_raw
        mu = x[..., 0]
        log_sigma_raw = x[..., 1]
        eta = x[..., 2]
        log_delta_raw = x[..., 3]

        if get_logits:
            if return_factors:
                return mu, log_sigma_raw, eta, log_delta_raw, factors
            return mu, log_sigma_raw, eta, log_delta_raw

        from GenerativeBrainModel.utils.sas import sas_rate_median

        median = sas_rate_median(mu, log_sigma_raw, eta, log_delta_raw)
        return median.to(mu.dtype)

    def autoregress_sample(
        self,
        init_x,  # (B, T, N)   rates (Hz)
        init_stimuli,  # (B, T, dS)
        point_positions,  # (B, N, 3)
        neuron_pad_mask,  # (B, N)
        future_stimuli=None,  # (B, n_steps, dS) or None -> zeros
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
            future_stimuli = torch.zeros(
                B, n_steps, self.d_stimuli, device=init_x.device, dtype=init_x.dtype
            )
        else:
            assert future_stimuli.shape == (B, n_steps, self.d_stimuli)

        params_dtype = next(self.parameters()).dtype
        current_log_rates = torch.log(init_x.clamp_min(eps) + eps).to(params_dtype)
        current_stimuli = init_stimuli.to(params_dtype)
        point_positions = point_positions.to(params_dtype)

        POS_EPS = 1e-8

        def _pos(x):
            return F.softplus(x) + POS_EPS

        for t in range(n_steps):
            # context
            context_log = current_log_rates[:, -context_len:]
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
            mu = mu[:, -1:, :]  # (B,1,N)
            sig = _pos(log_sigma_raw[:, -1:, :])
            eta = eta[:, -1:, :]
            delt = _pos(log_delta_raw[:, -1:, :])

            # sample next log-rate
            U = torch.randn_like(mu, generator=generator) * float(temperature)
            next_log_rate = mu + sig * torch.sinh((U - eta) / delt)

            # append
            current_log_rates = torch.cat([current_log_rates, next_log_rate], dim=1)
            current_stimuli = torch.cat(
                [current_stimuli, future_stimuli[:, t : t + 1, :].to(params_dtype)],
                dim=1,
            )

        if return_log:
            return current_log_rates
        # back to rate domain with the SAME eps used going in
        return torch.exp(current_log_rates) - eps
