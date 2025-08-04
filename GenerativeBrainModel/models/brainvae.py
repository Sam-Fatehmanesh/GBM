import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv3dRMSNorm, Conv4dRMSNorm
from GenerativeBrainModel.models.mlp import MLP, FFN
from GenerativeBrainModel.models.attention import VoxelAttention
from GenerativeBrainModel.models.convolutional import CausalConv4D, ResCausalConv4D, SpatialDownsampleConv4D, SpatialUpsampleConv4D

# A 4D VAE 
class HyperVolumeVAE(nn.Module):
    def __init__(self, d_model, volume=(256, 128, 30)):
        super(HyperVolumeVAE, self).__init__()

        # Store dimensions
        self.volume = volume
        assert volume == (256, 128, 30), "volume must be (256, 128, 30)"
        self.d_model = d_model
        
        

        self.encoder = nn.Sequential(
            # (B, T, 1, 256, 128, 30)
            CausalConv4D(in_channels=1, out_channels=d_model, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            ResCausalConv4D(in_channels=d_model, out_channels=d_model, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(5,5,5), stride=(2,1,1), in_channels=d_model, out_channels=d_model),
            # (B, T, d_model, 128, 128, 30)
            ResCausalConv4D(in_channels=d_model, out_channels=d_model*2, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(5,5,5), stride=(2,2,1), in_channels=d_model*2, out_channels=d_model*2),
            # (B, T, d_model*2, 64, 64, 30)
            ResCausalConv4D(in_channels=d_model*2, out_channels=d_model*8, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialDownsampleConv4D(kernel_size=(3,3,3), stride=(4,4,2), in_channels=d_model*8, out_channels=d_model*8),
            # (B, T, d_model*4, 16, 16, 15)
            ResCausalConv4D(in_channels=d_model*8, out_channels=d_model*8, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            VoxelAttention(dim=d_model*8),
            ResCausalConv4D(in_channels=d_model*8, out_channels=d_model*2, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            # (B, T, d_model*2, 16, 16, 15) channel dimention containing mu and logvar
        )

        self.decoder = nn.Sequential(
            # (B, T, d_model*2, 16, 16, 15)
            ResCausalConv4D(in_channels=d_model, out_channels=d_model*8, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            VoxelAttention(dim=d_model*8),
            ResCausalConv4D(in_channels=d_model*8, out_channels=d_model*8, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(3,3,3), stride=(4,4,2), in_channels=d_model*8, out_channels=d_model*28,
            # (B, T, d_model*8, 64, 64, 30)
            ResCausalConv4D(in_channels=d_model*8, out_channels=d_model*2, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(5,5,5), stride=(2,2,1), in_channels=d_model*2, out_channels=d_model*2),
            # (B, T, d_model*2, 128, 128, 30)
            ResCausalConv4D(in_channels=d_model*2, out_channels=d_model, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            SpatialUpsampleConv4D(kernel_size=(5,5,5), stride=(2,1,1), in_channels=d_model, out_channels=d_model),
            # (B, T, d_model, 256, 128, 30)
            ResCausalConv4D(in_channels=d_model, out_channels=d_model, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            CausalConv4D(in_channels = d_model, out_channels=1, spatial_kernel_size=3, temporal_kernel_size=4, stride=1, temporal_stride=1),
            )
        

    def encode(self, x):
        """
        Encode volumes to latent representation.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
        
        Returns:
            Encoded tensor of shape (B, T, vol_x//16, vol_y//8, vol_z//2)
        """
        B, T, vol_x, vol_y, vol_z = x.shape

        x = self.encoder(x)

        # Splits channels dim into mu and logvar
        mu, logvar = x.chunk(2, dim=2)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent representation.
        
        Args:
            mu: Mean of the latent representation of shape (B, T, vol_x//16, vol_y//8, vol_z//2)
            logvar: Log variance of the latent representation of shape (B, T, vol_x//16, vol_y//8, vol_z//2)

        Returns:
            Sampled tensor of shape (B, T, vol_x, vol_y, vol_z)
        """
        B, T, vol_x, vol_y, vol_z = mu.shape

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
            z: Encoded tensor of shape (B, T, vol_x, vol_y, vol_z)
        
        Returns:
            Decoded tensor of shape (B, T, vol_x_new, vol_y_new, vol_z_new)
        """
        B, T, vol_x, vol_y, vol_z = z.shape

        return self.decoder(z)


    