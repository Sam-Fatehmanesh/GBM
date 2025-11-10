import torch
from torch import nn
import torch.nn.functional as F
import math
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP


class BModel(nn.Module):
    def __init__(
        self,
        d_behavior,
        d_max_neurons,
        d_hidden: int = 128,
        fourier_bands: int = 6,
        include_raw_coords: bool = True,
    ):
        super(BModel, self).__init__()

        self.d_behavior = d_behavior
        self.d_max_neurons = d_max_neurons
        self.d_hidden = d_hidden
        self.n_bands = int(fourier_bands)
        self.include_raw_coords = bool(include_raw_coords)

        # Fourier bands (pi * 2^k)
        fb = torch.pow(2.0, torch.arange(self.n_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("fourier_bands", fb, persistent=False)

        # Input dim: spike + (optional raw coords) + 2*sin/cos*3*bands
        pos_ff_dim = 2 * 3 * self.n_bands
        self.input_dim = 1 + (3 if self.include_raw_coords else 0) + pos_ff_dim

        # Shared per-neuron encoder: f([spike, pos_feats]) -> h
        self.neuron_encoder = nn.Sequential(
            nn.Linear(self.input_dim, d_hidden),
            RMSNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            RMSNorm(d_hidden),
            nn.SiLU(),
        )
        # Per-time pooled processing and final head
        self.time_mlp = nn.Sequential(
            RMSNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            RMSNorm(d_hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            RMSNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_behavior),
        )

    def forward(self, x, point_positions, neuron_mask=None, get_logits=True):
        # x: (B, T, N)
        # point_positions: (B, N, 3)
        # neuron_mask: (B, N) 1 for real, 0 for padded
        B, T, N = x.shape  # Here T is the window length (e.g., 6)

        # Mask
        if neuron_mask is None:
            mask = torch.ones(
                (B, N), device=point_positions.device, dtype=point_positions.dtype
            )
        else:
            mask = neuron_mask.to(
                device=point_positions.device, dtype=point_positions.dtype
            )

        # Relative positions: masked centroid + RMS scale
        num = (point_positions * mask.unsqueeze(-1)).sum(dim=1)  # (B,3)
        den = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)  # (B,1)
        centroid = num / den  # (B,3)
        pos_centered = point_positions - centroid.unsqueeze(1)  # (B,N,3)
        r2 = (pos_centered.pow(2).sum(dim=-1) * mask).sum(dim=1) / den.squeeze(-1)
        scale = r2.sqrt().clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        pos_rel = pos_centered / scale  # (B,N,3)

        # Fourier features on normalized coords
        freqs = self.fourier_bands.to(
            device=pos_rel.device, dtype=pos_rel.dtype
        )  # (nb,)
        pos_exp = pos_rel.unsqueeze(-1) * freqs  # (B,N,3,nb)
        sin = torch.sin(pos_exp)
        cos = torch.cos(pos_exp)
        pos_ff = torch.cat([sin, cos], dim=-1).reshape(
            B, N, 3 * 2 * self.n_bands
        )  # (B,N,6*nb)

        if self.include_raw_coords:
            pos_feats = torch.cat([pos_rel, pos_ff], dim=-1)
        else:
            pos_feats = pos_ff

        # Normalize neuron activations per (B,T) across neurons using mask (z-score)
        mask_bt = mask.unsqueeze(1)  # (B,1,N)
        denom_n = mask_bt.sum(dim=2).clamp_min(1.0)  # (B,1)
        mean_bt = ((x * mask_bt).sum(dim=2) / denom_n).unsqueeze(-1)  # (B,T,1)
        var_bt = ((((x - mean_bt) ** 2) * mask_bt).sum(dim=2) / denom_n).unsqueeze(
            -1
        )  # (B,T,1)
        std_bt = torch.sqrt(var_bt.clamp_min(1e-8))  # (B,T,1)
        x_norm = (x - mean_bt) / std_bt  # (B,T,N)

        # Repeat across time and concat normalized spike scalar
        pos_feats_t = pos_feats.unsqueeze(1).repeat(1, T, 1, 1)  # (B,T,N,Dpos)
        feats = torch.cat([x_norm.unsqueeze(-1), pos_feats_t], dim=3)  # (B,T,N,Din)
        feats = feats.view(B * T * N, self.input_dim)
        enc = self.neuron_encoder(feats)
        enc = enc.view(B, T, N, self.d_hidden)

        # Mask on encoder device/dtype
        mask = mask.to(device=enc.device, dtype=enc.dtype)
        mask_bt = mask.unsqueeze(1).unsqueeze(-1)  # (B,1,N,1)
        enc = enc * mask_bt
        denom = mask_bt.sum(dim=2).clamp_min(1.0)  # (B,T,1)
        pooled_t = enc.sum(dim=2) / denom  # (B,T,H) per-time pooled features

        # Per-time processing
        proc_t = self.time_mlp(pooled_t)  # (B,T,H)
        # Pool across time (mean)
        out_feat = proc_t.mean(dim=1)  # (B,H)
        out = self.head(out_feat).unsqueeze(1)  # (B,1,d_behavior)
        return out
