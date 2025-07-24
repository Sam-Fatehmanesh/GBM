import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomFourier3DEncoding(nn.Module):
    """
    Random Fourier feature embeddings for 3D positional encoding.
    
    Maps each 3D point p∈R³ into a high-dimensional "frequency" space by:
    1. Sampling a fixed matrix B∈R^{F×3} with entries B_{ij}~N(0,σ²)
    2. Projecting the coordinate: compute the vector B·p∈R^F
    3. Encoding via sine and cosine: γ(p) = [sin(B·p), cos(B·p)]∈R^{2F}
    """
    def __init__(self, num_frequencies=64, sigma=1.0):
        super(RandomFourier3DEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        
        # Sample fixed matrix B with entries from N(0, σ²)
        # Shape: (num_frequencies, 3) for 3D coordinates
        B = torch.randn(num_frequencies, 3) * sigma
        self.register_buffer('B', B)  # Fixed, not trainable
        
    def forward(self, positions):
        """
        Args:
            positions: (batch_size, 3) tensor of 3D coordinates
            
        Returns:
            encodings: (batch_size, 2*num_frequencies) tensor of positional encodings
        """
        # positions: (batch_size, 3)
        # B: (num_frequencies, 3)
        # Compute B·p for each position: (batch_size, num_frequencies)
        projections = torch.matmul(positions, self.B.T)
        
        # Apply sine and cosine: [sin(B·p), cos(B·p)]
        sin_proj = torch.sin(projections)
        cos_proj = torch.cos(projections)
        
        # Concatenate: (batch_size, 2*num_frequencies)
        encoding = torch.cat([sin_proj, cos_proj], dim=1)
        
        return encoding
