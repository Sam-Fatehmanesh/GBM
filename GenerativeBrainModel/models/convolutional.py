import torch
import torch.nn as nn
import torch.nn.functional as F
from GenerativeBrainModel.models.posencode import RandomFourier3DEncoding
from GenerativeBrainModel.models.rms import RMSNorm, Conv4dRMSNorm


# A 4d convolutional which is causal on the time dimension, operates on (B, T, C, vol_x, vol_y, vol_z)
class CausalConv4D(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_kernel_size=3, temporal_kernel_size=2, stride=1, temporal_stride=1, padding='same'):
        super(CausalConv4D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.stride = stride
        self.temporal_stride = temporal_stride
        self.padding = padding
        
        # Calculate spatial padding for 'same' convolution
        if padding == 'same':
            self.spatial_padding = spatial_kernel_size // 2
        else:
            self.spatial_padding = 0
            
        # For causal convolution, we pad only the left side (past) in the temporal dimension
        self.temporal_padding = temporal_kernel_size - 1
        
        # True 4D convolution weights: (out_channels, in_channels, temporal_kernel_size, spatial_kernel_size, spatial_kernel_size, spatial_kernel_size)
        self.weight = nn.Parameter(torch.randn(
            out_channels,
            in_channels, 
            temporal_kernel_size,
            spatial_kernel_size,
            spatial_kernel_size, 
            spatial_kernel_size
        ))
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using Kaiming normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        """
        Forward pass for causal 4D convolution using efficient tensor operations
        
        Args:
            x: Input tensor of shape (B, T, C, vol_x, vol_y, vol_z)
            
        Returns:
            Output tensor of shape (B, T, out_channels, vol_x, vol_y, vol_z)
        """
        batch_size, seq_len, in_channels, vol_x, vol_y, vol_z = x.shape
        
        # Apply causal temporal padding (pad only on the left/past side)
        if self.temporal_padding > 0:
            temporal_pad = torch.zeros(batch_size, self.temporal_padding, in_channels, vol_x, vol_y, vol_z, 
                                     device=x.device, dtype=x.dtype)
            x_padded = torch.cat([temporal_pad, x], dim=1)  # (B, T + temp_pad, C, vol_x, vol_y, vol_z)
        else:
            x_padded = x
            
        # Apply spatial padding 
        if self.spatial_padding > 0:
            x_padded = F.pad(x_padded, 
                           (self.spatial_padding, self.spatial_padding,  # vol_z padding
                            self.spatial_padding, self.spatial_padding,  # vol_y padding  
                            self.spatial_padding, self.spatial_padding), # vol_x padding
                           mode='constant', value=0)
        
        # Reshape input for efficient convolution: (B, C, T_padded, vol_x_padded, vol_y_padded, vol_z_padded)
        x_reshaped = x_padded.permute(0, 2, 1, 3, 4, 5).contiguous()
        
        # Unfold the 4D tensor to extract 4D patches
        # Unfold temporal dimension with temporal stride
        x_unfolded = x_reshaped.unfold(2, self.temporal_kernel_size, self.temporal_stride)  # (B, C, T_out, vol_x_pad, vol_y_pad, vol_z_pad, temp_kernel)
        
        # Unfold spatial dimensions
        x_unfolded = x_unfolded.unfold(3, self.spatial_kernel_size, self.stride)  # (..., vol_x_out, vol_y_pad, vol_z_pad, temp_kernel, spat_kernel_x)
        x_unfolded = x_unfolded.unfold(4, self.spatial_kernel_size, self.stride)  # (..., vol_x_out, vol_y_out, vol_z_pad, temp_kernel, spat_kernel_x, spat_kernel_y)
        x_unfolded = x_unfolded.unfold(5, self.spatial_kernel_size, self.stride)  # (..., vol_x_out, vol_y_out, vol_z_out, temp_kernel, spat_kernel_x, spat_kernel_y, spat_kernel_z)
        
        # Get output dimensions (both spatial and temporal)
        t_out = x_unfolded.shape[2]  # Actual temporal output dimension after striding
        if self.padding == 'same':
            vol_x_out, vol_y_out, vol_z_out = vol_x, vol_y, vol_z
        else:
            vol_x_out = x_unfolded.shape[3]
            vol_y_out = x_unfolded.shape[4] 
            vol_z_out = x_unfolded.shape[5]
            
        # Reshape unfolded tensor for matrix multiplication
        # x_unfolded: (B, C, T_out, vol_x_out, vol_y_out, vol_z_out, temp_kernel, spat_kernel_x, spat_kernel_y, spat_kernel_z)
        x_patches = x_unfolded.permute(0, 2, 3, 4, 5, 1, 6, 7, 8, 9).contiguous()  
        # (B, T_out, vol_x_out, vol_y_out, vol_z_out, C, temp_kernel, spat_kernel_x, spat_kernel_y, spat_kernel_z)
        
        x_patches = x_patches.view(batch_size, t_out, vol_x_out, vol_y_out, vol_z_out, -1)
        # (B, T_out, vol_x_out, vol_y_out, vol_z_out, C * temp_kernel * spat_kernel_x * spat_kernel_y * spat_kernel_z)
        
        # Reshape weights for matrix multiplication
        weight_reshaped = self.weight.view(self.out_channels, -1)
        # (out_channels, C * temp_kernel * spat_kernel_x * spat_kernel_y * spat_kernel_z)
        
        # Apply convolution via matrix multiplication
        # x_patches: (B, T_out, vol_x_out, vol_y_out, vol_z_out, kernel_features)
        # weight_reshaped: (out_channels, kernel_features)
        output = torch.einsum('btvxyk, ok -> btvxyo', x_patches, weight_reshaped)
        # (B, T_out, vol_x_out, vol_y_out, vol_z_out, out_channels)
        
        # Add bias
        output = output + self.bias.view(1, 1, 1, 1, 1, -1)
        
        # Permute to final output format: (B, T, out_channels, vol_x, vol_y, vol_z)
        output = output.permute(0, 1, 5, 2, 3, 4).contiguous()
        
        return output


class ResCausalConv4D(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, spatial_kernel_size=3, temporal_kernel_size=2, stride=1, temporal_stride=1):
        super(ResCausalConv4D, self).__init__()

        self.norm0 = Conv4dRMSNorm(in_channels)
        self.act0 = nn.SiLU()
        self.conv0 = CausalConv4D(in_channels, out_channels, spatial_kernel_size = spatial_kernel_size, temporal_kernel_size = temporal_kernel_size, stride = stride, temporal_stride = temporal_stride)
        self.norm1 = Conv4dRMSNorm(out_channels)
        self.act1 = nn.SiLU()
        self.conv1 = CausalConv4D(out_channels, out_channels, spatial_kernel_size = spatial_kernel_size, temporal_kernel_size = temporal_kernel_size, stride = stride, temporal_stride = temporal_stride)

        self.shortcut = nn.Identity() if in_channels == out_channels else CausalConv4D(in_channels, out_channels, spatial_kernel_size=1, temporal_kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, vol_x, vol_y, vol_z)
        
        # Apply residual connection
        shortcut = self.shortcut(x)

        # Apply conv0
        x = self.norm0(x)
        x = self.act0(x)
        x = self.conv0(x)
        
        # Apply conv1
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        return x + shortcut 
        
# Downsample using strided 3d convolution on (B, T, C, vol_x, vol_y, vol_z) but merging B and T dimensions
class SpatialDownsampleConv4D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=(2,2,2)):
        super(SpatialDownsampleConv4D, self).__init__()

        # Calculate padding for 'same' behavior with striding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for spatial downsampling using strided 3D convolution
        
        Args:
            x: Input tensor of shape (B, T, C, vol_x, vol_y, vol_z)
            
        Returns:
            Output tensor of shape (B, T, out_channels, vol_x//stride, vol_y//stride, vol_z//stride)
        """
        batch_size, seq_len, in_channels, vol_x, vol_y, vol_z = x.shape
        
        # Reshape input for efficient convolution: (B, T, C, vol_x, vol_y, vol_z) -> (B*T, C, vol_x, vol_y, vol_z)
        x = x.view(batch_size * seq_len, in_channels, vol_x, vol_y, vol_z)
        
        # Apply 3D convolution
        x = self.conv(x)  # (B*T, out_channels, vol_x//stride, vol_y//stride, vol_z//stride)
        
        # Reshape output back to original shape: (B*T, out_channels, vol_x//stride, vol_y//stride, vol_z//stride) -> (B, T, out_channels, vol_x//stride, vol_y//stride, vol_z//stride)
        new_vol_x, new_vol_y, new_vol_z = x.shape[-3:]
        x = x.view(batch_size, seq_len, self.out_channels, new_vol_x, new_vol_y, new_vol_z)
        
        return x

# Upsample using F.interpolate with nearest exact and 3d convolution on (B, T, C, vol_x, vol_y, vol_z) but merging B and T dimensions
class SpatialUpsampleConv4D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3,3,3), stride=(2,2,2)):
        super(SpatialUpsampleConv4D, self).__init__()

        # Calculate padding for 'same' behavior
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )


        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for spatial upsampling using F.interpolate with nearest exact and 3D convolution
        
        Args:
            x: Input tensor of shape (B, T, C, vol_x, vol_y, vol_z)
            
        Returns:
            Output tensor of shape (B, T, out_channels, vol_x*stride, vol_y*stride, vol_z*stride)
        """
        batch_size, seq_len, in_channels, vol_x, vol_y, vol_z = x.shape
        
        # Reshape input for efficient convolution: (B, T, C, vol_x, vol_y, vol_z) -> (B*T, C, vol_x, vol_y, vol_z)
        x = x.view(batch_size * seq_len, in_channels, vol_x, vol_y, vol_z)

        # Apply interpolation
        x = F.interpolate(x, scale_factor=self.stride, mode='nearest-exact')

        # Apply 3D convolution
        x = self.conv(x)  # (B*T, out_channels, vol_x*stride, vol_y*stride, vol_z*stride)
        
        # Reshape output back to original shape: (B*T, out_channels, vol_x*stride, vol_y*stride, vol_z*stride) -> (B, T, out_channels, vol_x*stride, vol_y*stride, vol_z*stride)
        new_vol_x, new_vol_y, new_vol_z = x.shape[-3:]
        x = x.view(batch_size, seq_len, self.out_channels, new_vol_x, new_vol_y, new_vol_z)
        
        return x
