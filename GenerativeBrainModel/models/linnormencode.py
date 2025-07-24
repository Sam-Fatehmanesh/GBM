import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.rms import RMSNorm
import numpy as np
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding

# Linear Normed Encoder/Decoder for 3D volume regions
class LinNormEncoder(nn.Module):
    def __init__(self, input_size=32*16*2, hidden_size=256, volume_size=(256, 128, 30), region_size=(32, 16, 2), num_frequencies=32, sigma=1.0):
        super(LinNormEncoder, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.norm = RMSNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        
        # Store volume and region dimensions for positional encoding
        self.volume_size = volume_size
        self.region_size = region_size
        self.hidden_size = hidden_size
        
        # Calculate number of regions along each axis
        self.n_blocks_x = volume_size[0] // region_size[0]
        self.n_blocks_y = volume_size[1] // region_size[1] 
        self.n_blocks_z = volume_size[2] // region_size[2]
        self.n_regions = self.n_blocks_x * self.n_blocks_y * self.n_blocks_z
        
        # 3D positional encoding
        self.pos_encoding = RandomFourier3DEncoding(num_frequencies, sigma)
        self.pos_projection = nn.Linear(2 * num_frequencies, hidden_size)
        
        # Create fixed positional coordinates for all regions
        self._create_region_positions()
        
    def _create_region_positions(self):
        """Create normalized 3D coordinates for each region center."""
        positions = []
        
        for i in range(self.n_blocks_x):
            for j in range(self.n_blocks_y):
                for k in range(self.n_blocks_z):
                    # Calculate center coordinates of each region
                    # Normalize to [0, 1] range
                    x = (i + 0.5) / self.n_blocks_x
                    y = (j + 0.5) / self.n_blocks_y  
                    z = (k + 0.5) / self.n_blocks_z
                    positions.append([x, y, z])
        
        # Convert to tensor and register as buffer (not trainable)
        positions = torch.tensor(positions, dtype=torch.float32)
        self.register_buffer('region_positions', positions)  # Shape: (n_regions, 3)
        
    def forward(self, x):
        # Store original shape for reshaping after decode
        original_shape = x.shape
        
        x = self.encode(x, apply_norm=True)
        # For autoencoder training with BCEWithLogitsLoss, return logits
        x = self.decode(x, apply_norm=False, get_logits=True)
        
        # Reshape back to original input shape
        x = x.reshape(original_shape)
        return x

    def encode(self, x, apply_norm=True):
        # Expects (B, T, n_regions, region_x, region_y, region_z)
        B, T, n_regions, region_x, region_y, region_z = x.shape
        x = x.reshape(B*T*n_regions, region_x*region_y*region_z)

        # Apply linear encoder 
        x = torch.bernoulli(torch.clamp(x, min=0.0, max=1.0))
        x = self.encoder(x)
        
        # Apply 3D positional encoding
        # Get positional encodings for all regions
        pos_encodings = self.pos_encoding(self.region_positions)  # (n_regions, 2*num_frequencies)
        pos_features = self.pos_projection(pos_encodings)  # (n_regions, hidden_size)
        
        # Expand positional features to match batch dimensions
        # pos_features: (n_regions, hidden_size) -> (B*T*n_regions, hidden_size)
        pos_features_expanded = pos_features.unsqueeze(0).expand(B*T, -1, -1).reshape(B*T*n_regions, self.hidden_size)
        
        # Add positional encoding to encoded features
        x = x + pos_features_expanded
        
        if apply_norm:
            x = self.norm(x)
        return x
    
    def decode(self, x, apply_norm=True, get_logits=False):
        if apply_norm:
            x = self.norm(x)
        x = self.decoder(x)
        if get_logits:
            return x
        return torch.sigmoid(x)

