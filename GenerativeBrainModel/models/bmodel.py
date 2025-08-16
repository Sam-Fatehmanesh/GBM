import torch
from torch import nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm



class BModel(nn.Module):
    def __init__(self, d_behavior, d_max_neurons):
        super(BModel, self).__init__()

        self.d_behavior = d_behavior
        self.d_max_neurons = d_max_neurons


        self.behavior_predictor = nn.Linear(4*d_max_neurons, d_behavior)


    def forward(self, x, point_positions, get_logits=True):
        # x: (B, T, n_neurons)
        # point_positions: (B, N, 3)
        # Returns sequences of shape (batch_size, seq_len, d_behavior)

        B, T, N = x.shape

        x = x.unsqueeze(-1)

        x = torch.cat([x, point_positions.unsqueeze(1).repeat(1, T, 1, 1)], dim=3)

        # flatten the neurons and point positions
        x = x.view(B, T, -1)

        x = self.behavior_predictor(x)

        if get_logits:
            return x

        return torch.sigmoid(x)