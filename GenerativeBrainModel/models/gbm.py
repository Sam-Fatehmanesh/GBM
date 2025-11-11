import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.conv import CausalResidualNeuralConv1d
from GenerativeBrainModel.models.spatiotemporal import SpatioTemporalNeuralAttention
from GenerativeBrainModel.models.attention import gumbel_topk
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
        topk_fraction: float = 0.1,
        gumbel_temperature: float = 1.0,
    ):
        super(GBM, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_stimuli = d_stimuli
        self.neuron_pad_id = int(neuron_pad_id)
        self.topk_fraction = float(topk_fraction)
        self.gumbel_temperature = float(gumbel_temperature)

        self.layers = nn.ModuleList(
            [SpatioTemporalNeuralAttention(d_model, n_heads) for _ in range(n_layers)]
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

        # Build spike sender mask:
        # - If a boolean mask is provided, use it directly.
        # - Otherwise, perform per-timepoint Gumbel top-k on provided scores (probabilities).
        if neuron_spike_probs.dtype == torch.bool:
            neuron_spike_mask = neuron_spike_probs
        else:
            if neuron_spike_probs.dtype != torch.float32:
                scores_fp32 = neuron_spike_probs.to(torch.float32)
            else:
                scores_fp32 = neuron_spike_probs
            scores_fp32 = torch.nan_to_num(
                scores_fp32, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)  # (B,T,N)
            # Mask invalid neurons before selection (use ONLY real neuron tokens, exclude the extra stimulus token)
            valid_btn = (
                neuron_pad_mask[:, :N].unsqueeze(1).expand(B, T, N) != 0
            )  # (B,T,N)
            scores_masked = scores_fp32.masked_fill(~valid_btn, float("-inf"))
            # Flatten to rows (B*T, N)
            S = B * T
            scores_SN = scores_masked.reshape(S, N)
            valid_SN = valid_btn.reshape(S, N)
            valid_counts = valid_SN.sum(dim=1)
            frac = max(0.0, min(1.0, self.topk_fraction))
            k_per_row = (valid_counts.to(torch.float32) * frac).round().to(torch.int64)
            k_per_row = torch.where(
                valid_counts > 0, k_per_row.clamp_min(1), torch.zeros_like(k_per_row)
            )
            K_max = int(k_per_row.max().item()) if k_per_row.numel() > 0 else 0
            if K_max > 0:
                _, topk_idx = gumbel_topk(
                    scores_SN, K_max, temperature=float(self.gumbel_temperature)
                )
                send_SN = torch.zeros_like(valid_SN, dtype=torch.bool)  # (S,N)
                row_ids = (
                    torch.arange(S, device=scores_SN.device)
                    .unsqueeze(1)
                    .expand(S, K_max)
                )
                pos = (
                    torch.arange(K_max, device=scores_SN.device)
                    .unsqueeze(0)
                    .expand(S, K_max)
                )
                keep_pos = pos < k_per_row.unsqueeze(1)
                if keep_pos.any():
                    rows_sel = row_ids[keep_pos]
                    cols_sel = topk_idx[keep_pos]
                    send_SN[rows_sel, cols_sel] = True
            else:
                send_SN = torch.zeros_like(valid_SN, dtype=torch.bool)
            neuron_spike_mask = send_SN.view(B, T, N)
        # Append True for the stimulus token (n_neurons + 1)
        neuron_spike_mask = torch.cat(
            [
                neuron_spike_mask,
                torch.ones(B, T, 1, device=neuron_spike_mask.device, dtype=torch.bool),
            ],
            dim=2,
        )

        # Apply the layers
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_point_positions, neuron_pad_mask, neuron_spike_mask)
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

    @torch.no_grad()
    def autoregress(
        self,
        init_context: torch.Tensor,  # (B, Tc, N) rates
        stim_full: torch.Tensor,  # (B, Tc+Tf, K)
        point_positions: torch.Tensor,  # (B, N, 3)
        neuron_pad_mask: torch.Tensor,  # (B, N)
        neuron_ids: torch.Tensor,  # (B, N)
        lam: torch.Tensor | None,  # (B, N)
        las: torch.Tensor | None,  # (B, N)
        Tf: int,
        sampling_rate_hz: float = 3.0,
        max_context_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout with an expanding context window up to max_context_len:
        - Keeps top-k sender selections fixed for past timepoints.
        - Only samples top-k for the newest AR step.
        Returns:
          pred rates: (B, Tf, N)
          spike counts for initial context: (B, Tc)
          spike counts for predicted future: (B, Tf)
        """
        device = init_context.device
        dtype_params = next(self.parameters()).dtype
        B, Tc, N = init_context.shape
        assert B >= 1 and Tf > 0 and Tc > 0
        sr_f = float(sampling_rate_hz)
        eps = 1e-7
        # Context buffers
        context_rates = init_context.clone()  # (B,Tc,N)
        # Build fixed top-k sender mask for existing context (temperature=0 for stability)
        prob_ctx = 1.0 - torch.exp(-context_rates.to(torch.float32) / sr_f)
        prob_ctx = torch.nan_to_num(prob_ctx, nan=0.0, posinf=1.0, neginf=0.0).clamp_(
            0.0, 1.0
        )
        valid_btn = neuron_pad_mask.unsqueeze(1).expand(B, Tc, N) != 0
        scores_masked = prob_ctx.masked_fill(~valid_btn, float("-inf"))
        S0 = B * Tc
        scores_SN = scores_masked.reshape(S0, N)
        valid_SN = valid_btn.reshape(S0, N)
        valid_counts = valid_SN.sum(dim=1)
        frac = max(0.0, min(1.0, self.topk_fraction))
        k_per_row = (valid_counts.to(torch.float32) * frac).round().to(torch.int64)
        k_per_row = torch.where(
            valid_counts > 0, k_per_row.clamp_min(1), torch.zeros_like(k_per_row)
        )
        K_max = int(k_per_row.max().item()) if k_per_row.numel() > 0 else 0
        if K_max > 0:
            _, topk_idx = gumbel_topk(scores_SN, K_max, temperature=0.0)
            send_SN = torch.zeros_like(valid_SN, dtype=torch.bool)  # (S0,N)
            row_ids = (
                torch.arange(S0, device=scores_SN.device).unsqueeze(1).expand(S0, K_max)
            )
            pos = (
                torch.arange(K_max, device=scores_SN.device)
                .unsqueeze(0)
                .expand(S0, K_max)
            )
            keep_pos = pos < k_per_row.unsqueeze(1)
            if keep_pos.any():
                rows_sel = row_ids[keep_pos]
                cols_sel = topk_idx[keep_pos]
                send_SN[rows_sel, cols_sel] = True
        else:
            send_SN = torch.zeros_like(valid_SN, dtype=torch.bool)
        mask_ctx = send_SN.view(B, Tc, N)  # boolean
        # Spike counts for initial context
        ctx_counts = mask_ctx.sum(dim=2).to(torch.float32)  # (B,Tc)
        preds: list[torch.Tensor] = []
        pred_counts: list[torch.Tensor] = []
        for t in range(Tf):
            # Determine window length
            win_len = int(context_rates.shape[1])
            if max_context_len is not None:
                win_len = min(win_len, int(max_context_len))
            x_in = context_rates[:, -win_len:, :]  # (B,win_len,N)
            x_log = torch.log(x_in.clamp_min(eps))
            if (
                lam is not None
                and las is not None
                and lam.numel() > 0
                and las.numel() > 0
            ):
                lam_e = lam[:, None, :].to(dtype=x_log.dtype)
                las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
                x_in_z = (x_log - lam_e) / las_e
            else:
                x_in_z = x_log
            stim_step = stim_full[
                :, (context_rates.shape[1] - win_len) : (context_rates.shape[1]), :
            ]
            # Build boolean sender mask window (B, win_len, N) from fixed mask_ctx
            mask_window = mask_ctx[:, -win_len:, :]
            # Dtypes for compute on CUDA
            if self.training is False and x_in_z.device.type == "cuda":
                x_in_z = x_in_z.to(dtype_params)
                stim_step = stim_step.to(dtype_params)
            # Forward using precomputed boolean sender mask
            mu, raw_log_sigma, _, _ = self.forward(
                x_in_z,
                stim_step,
                point_positions,
                neuron_pad_mask,
                neuron_ids,
                mask_window,  # boolean mask
                get_logits=True,
                input_log_rates=True,
            )
            mu_last = mu[:, -1:, :]
            sig_last = raw_log_sigma[:, -1:, :]
            # Sample next-step rate from LogNormal
            from GenerativeBrainModel.utils.lognormal import sample_lognormal

            samp = sample_lognormal(mu_last, sig_last)  # (B,1,N)
            preds.append(samp[:, 0, :].to(torch.float32))
            # Append prediction
            context_rates = torch.cat(
                [context_rates, samp.to(context_rates.dtype)], dim=1
            )
            # Compute top-k sender mask for the NEW step only (B,1,N)
            prob_next = 1.0 - torch.exp(-samp.to(torch.float32) / sr_f)
            prob_next = torch.nan_to_num(
                prob_next, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            valid_next = (neuron_pad_mask != 0).unsqueeze(1)  # (B,1,N)
            scores_next = prob_next.masked_fill(~valid_next, float("-inf"))  # (B,1,N)
            S1 = B * 1
            scores_next_SN = scores_next.view(S1, N)
            valid_next_SN = valid_next.view(S1, N)
            valid_counts_next = valid_next_SN.sum(dim=1)
            k_row_next = (
                (valid_counts_next.to(torch.float32) * frac).round().to(torch.int64)
            )
            k_row_next = torch.where(
                valid_counts_next > 0,
                k_row_next.clamp_min(1),
                torch.zeros_like(k_row_next),
            )
            K1 = int(k_row_next.max().item()) if k_row_next.numel() > 0 else 0
            if K1 > 0:
                _, topk_idx_n = gumbel_topk(
                    scores_next_SN, K1, temperature=float(self.gumbel_temperature)
                )
                send_next = torch.zeros_like(valid_next_SN, dtype=torch.bool)
                row_ids_n = (
                    torch.arange(S1, device=scores_next_SN.device)
                    .unsqueeze(1)
                    .expand(S1, K1)
                )
                pos_n = (
                    torch.arange(K1, device=scores_next_SN.device)
                    .unsqueeze(0)
                    .expand(S1, K1)
                )
                keep_pos_n = pos_n < k_row_next.unsqueeze(1)
                if keep_pos_n.any():
                    rows_sel_n = row_ids_n[keep_pos_n]
                    cols_sel_n = topk_idx_n[keep_pos_n]
                    send_next[rows_sel_n, cols_sel_n] = True
            else:
                send_next = torch.zeros_like(valid_next_SN, dtype=torch.bool)
            next_mask = send_next.view(B, 1, N)
            mask_ctx = torch.cat([mask_ctx, next_mask], dim=1)  # append boolean mask
            # Record spike count for newest predicted step
            pred_counts.append(next_mask.sum(dim=2)[:, 0].to(torch.float32))  # (B,)
        pred_rates = torch.stack(preds, dim=1)  # (B,Tf,N)
        pred_counts_T = (
            torch.stack(pred_counts, dim=1)
            if len(pred_counts) > 0
            else torch.zeros((B, 0), device=device)
        )
        return pred_rates, ctx_counts, pred_counts_T
