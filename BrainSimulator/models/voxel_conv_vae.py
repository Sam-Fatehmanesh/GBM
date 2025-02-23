import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DeConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class VoxelEncoder(nn.Module):
    def __init__(self, 
                 voxel_size,
                 in_channels=1,
                 base_channels=32,
                 latent_dim=256,
                 n_conv_blocks=4):
        super().__init__()
        
        self.voxel_size = voxel_size
        channels = [in_channels] + [base_channels * (2**i) for i in range(n_conv_blocks)]
        
        # Initial projection
        self.init_proj = Conv3DBlock(channels[0], channels[1], kernel_size=7, stride=2, padding=3)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(1, len(channels)-1):
            self.encoder_blocks.append(nn.Sequential(
                Conv3DBlock(channels[i], channels[i], kernel_size=3, stride=1, padding=1),
                Conv3DBlock(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1)
            ))
        
        # Calculate final spatial dimensions
        self.final_spatial_size = voxel_size // (2 ** (n_conv_blocks))
        final_channels = channels[-1]
        
        # Latent projections
        flattened_dim = final_channels * (self.final_spatial_size ** 3)
        self.mu = nn.Linear(flattened_dim, latent_dim)
        self.logvar = nn.Linear(flattened_dim, latent_dim)
    
    def forward(self, x):
        # Initial projection
        x = self.init_proj(x)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Flatten and project to latent space
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        return mu, logvar

class VoxelDecoder(nn.Module):
    def __init__(self,
                 voxel_size,
                 latent_dim=256,
                 base_channels=32,
                 out_channels=1,
                 n_conv_blocks=4):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.n_conv_blocks = n_conv_blocks
        
        # Calculate initial spatial size and channels
        self.initial_spatial_size = voxel_size // (2 ** n_conv_blocks)
        channels = [base_channels * (2**i) for i in range(n_conv_blocks)][::-1] + [out_channels]
        initial_channels = channels[0]
        
        # Initial projection from latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, initial_channels * (self.initial_spatial_size ** 3)),
            nn.GELU()
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels)-2):
            self.decoder_blocks.append(nn.Sequential(
                DeConv3DBlock(channels[i], channels[i], kernel_size=3, stride=1, padding=1),
                DeConv3DBlock(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1, output_padding=0)
            ))
        
        # Final convolution
        self.final_conv = nn.Sequential(
            DeConv3DBlock(channels[-2], channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        batch_size = z.shape[0]
        
        # Project and reshape
        x = self.latent_proj(z)
        x = x.view(batch_size, -1, self.initial_spatial_size, self.initial_spatial_size, self.initial_spatial_size)
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class VoxelConvVAE(nn.Module):
    def __init__(self,
                 voxel_size=64,
                 in_channels=1,
                 base_channels=32,
                 latent_dim=256,
                 n_conv_blocks=4):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim
        
        self.encoder = VoxelEncoder(
            voxel_size=voxel_size,
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            n_conv_blocks=n_conv_blocks
        )
        
        self.decoder = VoxelDecoder(
            voxel_size=voxel_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            out_channels=in_channels,
            n_conv_blocks=n_conv_blocks
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_param_count(self):
        """Return number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def configure_optimizers(self, learning_rate=1e-4, weight_decay=1e-5):
        """Configure optimizer with weight decay on non-bias/norm parameters."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        
        return torch.optim.AdamW([
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.}
        ], lr=learning_rate)

def create_voxel_conv_vae(size='base'):
    """Create a Voxel Conv VAE with the specified size configuration."""
    configs = {
        'small': {  # ~5M params
            'voxel_size': 64,
            'in_channels': 1,
            'base_channels': 16,
            'latent_dim': 128,
            'n_conv_blocks': 4
        },
        'base': {  # ~50M params
            'voxel_size': 64,
            'in_channels': 1,
            'base_channels': 64,
            'latent_dim': 256,
            'n_conv_blocks': 5
        },
        'large': {  # ~200M params
            'voxel_size': 128,
            'in_channels': 1,
            'base_channels': 128,
            'latent_dim': 512,
            'n_conv_blocks': 6
        }
    }
    
    if size not in configs:
        raise ValueError(f"Size must be one of {list(configs.keys())}")
    
    return VoxelConvVAE(**configs[size]) 