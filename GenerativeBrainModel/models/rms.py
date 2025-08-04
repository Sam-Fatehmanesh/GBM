import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class Conv3dRMSNorm(nn.Module):
    """
    RMSNorm wrapper for 3D convolution tensors.
    
    Handles the dimension reshaping needed to apply RMSNorm to conv3d output tensors
    where the channel dimension is at index 1 instead of the last dimension.
    """
    
    def __init__(self, hidden_channels: int):
        super(Conv3dRMSNorm, self).__init__()
        self.rms_norm = RMSNorm(hidden_channels)
        
    def forward(self, x):
        """
        Apply RMSNorm to conv3d tensor.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W) where C is the channel dimension
            
        Returns:
            RMS normalized tensor of same shape
        """
        # Input shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape to (B*D*H*W, C) so channel dimension is last
        x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x_reshaped = x_reshaped.view(-1, C)  # (B*D*H*W, C)
        
        # Apply RMSNorm
        x_normed = self.rms_norm(x_reshaped)  # (B*D*H*W, C)
        
        # Reshape back to original shape
        x_normed = x_normed.view(B, D, H, W, C)  # (B, D, H, W, C)
        x_normed = x_normed.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        
        return x_normed