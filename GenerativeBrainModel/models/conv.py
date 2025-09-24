import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import FFN


class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size=16):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # Use depthwise convolution: in_channels=out_channels=dim, groups=dim
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim,
            bias=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, a=5**0.5)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: (batch, dim, seqlen)
        # nn.Conv1d expects (batch, channels, length)
        out = self.conv(x)
        # Remove extra padding on the right for causality
        out = out[..., :x.shape[-1]]
        return out

class CausalResidualNeuralConv1d(nn.Module):
    def __init__(self, dim, kernel_size=16):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.norm = RMSNorm(dim)
        self.conv = CausalConv1d(dim, kernel_size)

    def forward(self, x):
        # x: (batch_size, seq_len, n_neurons, d_model)
        B, T, N, D = x.shape

        res = x
        x = self.norm(x)
        # reshape to (batch*N, channels=d_model, length=seq_len) for Conv1d
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N, D, T)
        x = self.conv(x)
        #x = F.silu(x)
        # restore to (batch, seq_len, n_neurons, d_model)
        x = x.view(B, N, D, T).permute(0, 3, 1, 2).contiguous()
        x = x + res

        return x
