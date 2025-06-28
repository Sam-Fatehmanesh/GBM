import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.cnn import CNNLayer, DeCNNLayer
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP

class AutoMLP(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(AutoMLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            MLP(2, input_size, 4096, hidden_size)
        )
        self.decoder = nn.Sequential(
            MLP(2, hidden_size, 4096, input_size),
            nn.Unflatten(1, (1, 256, 128)),
            nn.Sigmoid(),
        )

        
    def forward(self, x):
        # AutoCNN expects 4D input for CNN layers, not flattened
        # Input should be [batch_size, 1, 256, 128]
        if len(x.shape) == 3:  # [batch_size, height, width]
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Remove channel dimension if present
        if len(decoded.shape) == 4 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        
        return decoded 

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)