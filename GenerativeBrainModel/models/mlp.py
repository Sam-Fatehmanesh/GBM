import torch
from torch import nn
from torch.nn import functional as F
from GenerativeBrainModel.models.rms import RMSNorm


class MLP(nn.Module):
    def __init__(self, layers_num, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = RMSNorm(hidden_size)
        self.input_activation = nn.GELU()

        self.hidden_layers = nn.ModuleList()
        for _ in range(layers_num - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                RMSNorm(hidden_size),
                nn.GELU(),
            )
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.input_activation(x)
        residual = x

        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            x = layer(x)
            x = residual + x

        # Output layer
        x = self.output_layer(x)

        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FFN, self).__init__()
        self.d_ff = int(d_ff)
        self.linear_fused = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.act = nn.SiLU()
        self.linear_3 = nn.Linear(d_ff, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        fused = self.linear_fused(x)
        x1 = fused[..., : self.d_ff]
        x2 = fused[..., self.d_ff :]
        x = self.linear_3(self.act(x1) * x2)
        x = x + res
        return x
