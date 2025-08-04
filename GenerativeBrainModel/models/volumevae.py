import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv3dRMSNorm
from GenerativeBrainModel.models.attention import SpatialRegionAttention 
from GenerativeBrainModel.models.mlp import MLP, FFN


class VolumeVAE(nn.Module):
    """
    Convolutional autoencoder with RMS normalization and 3D positional encoding.
    
    Uses 3D convolution with kernel_size=region_size and stride=region_size to automatically
    extract regions from volumes, eliminating the need for manual volume-to-region conversion.
    """
    
    def __init__(self, hidden_channels=512, volume_size=(256, 128, 30), 
                 region_size=(32, 16, 2), num_frequencies=32, sigma=1.0, n_heads=8):
        super(VolumeVAE, self).__init__()
        
        # Store dimensions
        self.volume_size = volume_size
        self.region_size = region_size
        self.hidden_channels = hidden_channels
        self.input_channels = 1

        # Assertions, region sizes divide volume sizes
        assert volume_size[0] % region_size[0] == 0, "Region size must divide volume size along x axis"
        assert volume_size[1] % region_size[1] == 0, "Region size must divide volume size along y axis"
        assert volume_size[2] % region_size[2] == 0, "Region size must divide volume size along z axis"
        
        # Calculate number of regions along each axis
        self.n_blocks_x = volume_size[0] // region_size[0]
        self.n_blocks_y = volume_size[1] // region_size[1] 
        self.n_blocks_z = volume_size[2] // region_size[2]
        self.n_regions = self.n_blocks_x * self.n_blocks_y * self.n_blocks_z

        self.d_model = hidden_channels
        
        # Encoder: 3D conv with kernel_size=region_size, stride=region_size
        # This automatically extracts regions from volumes
        self.encoder_conv = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=hidden_channels,
            kernel_size=region_size,
            stride=region_size,
            padding=0
        )
        


        # Attention layers
        self.encoder_attention = SpatialRegionAttention(hidden_channels, n_heads=8, n_regions=32)
        self.decoder_attention = SpatialRegionAttention(hidden_channels, n_heads=8, n_regions=32)

        # FFN layers
        self.encoder_FFN0 = FFN(hidden_channels, hidden_channels*2)
        self.encoder_FFN1 = FFN(hidden_channels, hidden_channels*2)
        self.decoder_FFN0 = FFN(hidden_channels, hidden_channels*2)
        self.decoder_FFN1 = FFN(hidden_channels, hidden_channels*2)

        self.encoder_rms_norm = Conv3dRMSNorm(hidden_channels)
        self.decoder_rms_norm = Conv3dRMSNorm(hidden_channels)
        # Decoder: 3D transposed conv to reconstruct volumes
        self.decoder_conv = nn.ConvTranspose3d(
            in_channels=hidden_channels,
            out_channels=self.input_channels,
            kernel_size=region_size,
            stride=region_size,
            padding=0
        )


        
        
        # 3D positional encoding
        self.pos_encoding = RandomFourier3DEncoding(num_frequencies, sigma)
        self.pos_projection = nn.Linear(2 * num_frequencies, hidden_channels)
        
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
        
    def forward(self, x, get_logits=True):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
            
        Returns:
            Reconstructed volumes of shape (B, T, vol_x, vol_y, vol_z)
        """
        # Input shape: (B, T, vol_x, vol_y, vol_z)
        
        B, T, vol_x, vol_y, vol_z = x.shape
        
        # Encode
        encoded = self.encode(x)
        
        # Decode 
        decoded = self.decode(encoded, get_logits=get_logits)
                
        return decoded

    def linPatch(self, x):
        """
        Encode volumes to latent representation.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
            apply_norm: Whether to apply normalization
            
        Returns:
            Encoded tensor of shape (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        """

        B, T, vol_x, vol_y, vol_z = x.shape
 
        # Apply Bernoulli sampling
        # x = torch.bernoulli(torch.clamp(x, min=0.0, max=1.0))
        
        # Reshape for conv3d: (B, T, vol_x, vol_y, vol_z) -> (B*T, 1, vol_x, vol_y, vol_z)
        x = x.contiguous().view(B*T, self.input_channels, vol_x, vol_y, vol_z)

        # Apply 3D convolution - this automatically extracts regions
        x = self.encoder_conv(x)  # (B*T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        
        # Apply positional encoding
        # Get positional encodings for all regions
        pos_encodings = self.pos_encoding(self.region_positions)  # (n_regions, 2*num_frequencies)
        pos_features = self.pos_projection(pos_encodings)  # (n_regions, hidden_channels)
        
        # Reshape pos_features to match spatial dimensions
        # pos_features: (n_regions, hidden_channels) -> (n_blocks_x, n_blocks_y, n_blocks_z, hidden_channels)
        pos_features = pos_features.view(self.n_blocks_x, self.n_blocks_y, self.n_blocks_z, self.hidden_channels)
        # -> (hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        pos_features = pos_features.permute(3, 0, 1, 2)
        # -> (1, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        pos_features = pos_features.unsqueeze(0)
        
        # Expand to match batch size
        batch_size = x.shape[0]
        pos_features = pos_features.expand(batch_size, -1, -1, -1, -1)
        
        # Add positional encoding
        x = x + pos_features

        x = self.encoder_rms_norm(x)
        
        
        return x.view(B, T, self.hidden_channels, self.n_blocks_x, self.n_blocks_y, self.n_blocks_z) # (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)

    def linDepatch(self, x):
        """
        Decode latent representation back to volumes.
        
        Args:
            x: Encoded tensor of shape (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
            apply_norm: Whether to apply normalization
            get_logits: Whether to return logits (for BCEWithLogitsLoss) or probabilities
            
        Returns:
            Decoded tensor of shape (B, T, vol_x, vol_y, vol_z)
        """

        B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z = x.shape
        x = x.view(B*T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)  # (B*T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)

        x = self.decoder_rms_norm(x)
              
        # Apply transposed convolution to reconstruct volumes
        x = self.decoder_conv(x)  # (B*T, input_channels, vol_x, vol_y, vol_z)

        # Reshape back to original volume dimensions
        vol_x, vol_y, vol_z = self.volume_size
        x = x.view(B, T, vol_x, vol_y, vol_z)  # (B, T, vol_x, vol_y, vol_z)

        return x

    def encode(self, x):
        """
        Encode volumes to latent representation.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
            apply_norm: Whether to apply normalization
            
        Returns:
            Encoded tensor of shape (batch_size, seq_len, n_regions, d_model)
        """
        B, T, vol_x, vol_y, vol_z = x.shape

        x = self.linPatch(x) # (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)

        x = x.permute(0, 1, 3, 4, 5, 2).reshape(B, T, self.n_regions, self.d_model) # (B, T, n_regions, d_model)

        x = self.encoder_FFN0(x)

        x = self.encoder_attention(x)

        x = self.encoder_FFN1(x)
        
        return x

    def decode(self, x, get_logits=False):
        """
        Decode latent representation back to volumes.
        
        Args:
            x: Encoded tensor of shape (B, T, n_regions, d_model)
            get_logits: Whether to return logits (for BCEWithLogitsLoss) or probabilities
            
        Returns:
            Decoded tensor of shape (B, T, vol_x, vol_y, vol_z)
        """
        B, T, n_regions, d_model = x.shape
        
        x = self.decoder_FFN0(x)

        x = self.decoder_attention(x)

        x = self.decoder_FFN1(x)

        # Reshape n_regions back to (n_blocks_x, n_blocks_y, n_blocks_z, d_model)
        x = x.view(B, T, self.n_blocks_x, self.n_blocks_y, self.n_blocks_z, self.d_model)  # (B, T, n_blocks_x, n_blocks_y, n_blocks_z, d_model)

        # Move d_model to channel position for decoder: (B, T, d_model, n_blocks_x, n_blocks_y, n_blocks_z)
        x = x.permute(0, 1, 5, 2, 3, 4)

        x = self.linDepatch(x)

        if get_logits:
            return x
        return torch.sigmoid(x)
    
