import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size=32768, hidden_size=2048):
        super(SimpleAutoencoder, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.norm = RMSNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        #self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.norm(self.encoder(x))

        # Apply batch normalization
        #encoded = self.batch_norm(encoded)
        
        # Decode
        decoded = torch.sigmoid(self.decoder(encoded))
        
        # Reshape back to grid
        decoded = decoded.view(batch_size, 256, 128)
        
        return decoded 