import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.cnn import CNNLayer, DeCNNLayer
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP

class AutoCNN(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(AutoCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            #256x128
            CNNLayer(1, 4, 5),
            nn.AvgPool2d(4),
            #64x32x4
            CNNLayer(4, 16, 5),
            nn.MaxPool2d(4),
            #16x8x16
            CNNLayer(16, 128, 5),
            nn.MaxPool2d(2),
            #8x4x128
            nn.Flatten(),
            MLP(1, 4096, 1024, 1024)
        )
        self.decoder = nn.Sequential(
            MLP(1, 1024, 1024, 4096),
            nn.Unflatten(1, (128, 8, 4)),
            # 8x4x128
            DeCNNLayer(128, 16, 2, kernel_size=5),
            # 16x8x16
            DeCNNLayer(16, 4, 4, kernel_size=5),
            # 64x32x4
            DeCNNLayer(4, 1, 4, kernel_size=5, last_activation=False),
            # 256x128x1
            nn.Sigmoid()
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