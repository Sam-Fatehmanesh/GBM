import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv3dRMSNorm, Conv4dRMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import VoxelAttention
from GenerativeBrainModel.models.convolutional import CausalConv4D, ResCausalConv4D, SpatialDownsampleConv4D, SpatialUpsampleConv4D

# A 4D VAE 
class HyperVAE(nn.Module):
    def __init__(self, d_model, volume=(256, 128, 30)):
        super(HyperVAE, self).__init__()

        # Store dimensions
        self.volume = volume
        assert volume == (256, 128, 30), "volume must be (256, 128, 30)"
        self.d_model = d_model

        chan_dim_mlpr = [1, 2, 4]
        
        

        self.encoder = nn.Sequential(
            # (B, T, 1, 256, 128, 30)
            CausalConv4D(in_channels=1, out_channels=1, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            ResCausalConv4D(in_channels=1, out_channels=d_model * chan_dim_mlpr[0], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(5,5,5), stride=(4,4,2), in_channels=d_model*chan_dim_mlpr[0], out_channels=d_model*chan_dim_mlpr[0]),
            # (B, T, d_model, 64, 32, 15)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[0], out_channels=d_model*chan_dim_mlpr[1], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(5,5,5), stride=(2,2,1), in_channels=d_model*chan_dim_mlpr[1], out_channels=d_model*chan_dim_mlpr[1]),
            # (B, T, d_model*2, 32, 16, 15)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[1], out_channels=d_model*chan_dim_mlpr[2], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(3,3,3), stride=(2,1,1), in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*chan_dim_mlpr[2]),
            # (B, T, d_model*4, 16, 16, 15)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*chan_dim_mlpr[2], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            VoxelAttention(dim=d_model*chan_dim_mlpr[2]),
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*2, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            # (B, T, d_model*2, 16, 16, 15) channel dimention containing mu and logvar
        )

        self.decoder = nn.Sequential(
            # (B, T, d_model, 16, 16, 15) - Input is z from reparameterization (half of encoder output channels)
            ResCausalConv4D(in_channels=d_model, out_channels=d_model*chan_dim_mlpr[2], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            VoxelAttention(dim=d_model*chan_dim_mlpr[2]),
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*chan_dim_mlpr[2], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(3,3,3), stride=(2,1,1), in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*chan_dim_mlpr[2]),
            # (B, T, d_model*8, 64, 64, 15)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[2], out_channels=d_model*chan_dim_mlpr[1], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(5,5,5), stride=(2,2,1), in_channels=d_model*chan_dim_mlpr[1], out_channels=d_model*chan_dim_mlpr[1]),
            # (B, T, d_model*2, 128, 128, 15)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[1], out_channels=d_model*chan_dim_mlpr[0], spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(5,5,5), stride=(4,4,2), in_channels=d_model*chan_dim_mlpr[0], out_channels=d_model*chan_dim_mlpr[0]),
            # (B, T, d_model, 256, 128, 30)
            ResCausalConv4D(in_channels=d_model*chan_dim_mlpr[0], out_channels=1, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            #CausalConv4D(in_channels=d_model*chan_dim_mlpr[0], out_channels=1, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            # (B, T, 1, 256, 128, 30)
        )
        

    def encode(self, x):
        """
        Encode volumes to latent representation.
        
        Args:
            x: Input volumes of shape (B, T, C, vol_x, vol_y, vol_z) where C=1 for brain volumes
        
        Returns:
            mu, logvar: Encoded tensors of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
        """
        x = self.encoder(x)

        # Splits channels dim into mu and logvar
        mu, logvar = x.chunk(2, dim=2)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent representation.
        
        Args:
            mu: Mean of the latent representation of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
            logvar: Log variance of the latent representation of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)

        Returns:
            Sampled tensor of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
        """
        # Sample from standard normal distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        # Reparameterize
        z = mu + eps * std
        
        return z

    def decode(self, z):
        """
        Decode latent representation back to volumes.
        
        Args:
            z: Encoded tensor of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
        
        Returns:
            Decoded tensor of shape (B, T, 1, vol_x, vol_y, vol_z)
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z) where C=1 for brain volumes
        
        Returns:
            Reconstructed volumes of shape (B, T, vol_x, vol_y, vol_z)

        """
        # Add channel dimension
        x = x.unsqueeze(2)

        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction_logits = self.decode(z)

        # Remove channel dimension
        reconstruction_logits = reconstruction_logits.squeeze(2)

        return reconstruction_logits, mu, logvar


    def kl_loss(self, mu, logvar):
        """
        Compute KL divergence loss from unit normal prior and posterior.
        
        Args:
            mu: Mean of the latent representation of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
            logvar: Log variance of the latent representation of shape (B, T, d_model, vol_x//16, vol_y//8, vol_z//2)
        
        Returns:
            KL divergence loss
        """
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_loss / mu.shape[0]  # Normalize by batch size

    def reconstruction_loss(self, x, reconstruction_logits):
        """
        Compute reconstruction loss.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
            reconstruction: Reconstructed volumes of shape (B, T, vol_x, vol_y, vol_z)
        
        Returns:
            Reconstruction loss
        """
        # Compute BCE loss using F.binary_cross_entropy_with_logits
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction_logits, x, reduction='mean')

        return reconstruction_loss# / x.shape[0]  # Normalize by batch size
