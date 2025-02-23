import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(SimpleAutoencoder, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = torch.sigmoid(self.decoder(encoded))
        
        # Reshape back to grid
        decoded = decoded.view(batch_size, 256, 128)
        
        return decoded 