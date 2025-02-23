import torch
import torch.nn as nn
import torch.nn.functional as F
from .point_transformer import PointTransformerBlock, PatchEmbed

class PointTransformerEncoder(nn.Module):
    def __init__(self, 
                 in_chans=4,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 patch_size=32):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        self.layer_types = []  # Store layer types separately
        self.num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        curr_stage = 0
        curr_dim = embed_dim
        
        for i in range(len(depths)):
            # Add transformer blocks for this stage
            stage_depth = depths[i]
            stage_dim = dims[i]
            stage_heads = num_heads[i]
            
            # Add projection if dim changes
            if curr_dim != stage_dim:
                proj = nn.Sequential(
                    nn.Linear(curr_dim, stage_dim),
                    nn.LayerNorm(stage_dim)
                )
                self.layers.append(proj)
                self.layer_types.append('proj')
                curr_dim = stage_dim
            
            # Add transformer blocks
            for j in range(stage_depth):
                block = PointTransformerBlock(
                    dim=stage_dim,
                    num_heads=stage_heads,
                    mlp_ratio=4,
                    drop_path=0.2,
                    patch_size=patch_size // (2 ** i)
                )
                self.layers.append(block)
                self.layer_types.append('block')
            
            curr_stage += 1
    
    def forward(self, features):
        """
        Args:
            features: (B, N, D) tensor of combined point features
                     where D includes both spatial coordinates and neural values
        Returns:
            x: (B, N', C) encoded features
            pos: (B, N', 3) positions for decoder
            mask: (B, N') boolean mask for valid patches
        """
        # Initial patch embedding
        x, pos, mask = self.patch_embed(features)
        
        # Process through layers
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'block':
                x = layer(x, pos, mask)
                # Downsample positions and mask if next layer is projection
                if i + 1 < len(self.layer_types) and self.layer_types[i + 1] == 'proj':
                    pos = pos[:, ::2, :]
                    mask = mask[:, ::2] if mask is not None else None
            else:  # projection layer
                x = layer(x)
        
        return x, pos, mask

class PointTransformerDecoder(nn.Module):
    def __init__(self,
                 latent_dim,
                 embed_dim=256,
                 depths=[2, 2, 6, 2],
                 num_heads=[32, 16, 8, 4],
                 mlp_ratio=4.,
                 drop_path_rate=0.2,
                 patch_size=32,
                 out_chans=4):
        super().__init__()
        
        # Reverse dimensions from encoder
        dims = [embed_dim * (2**i) for i in range(len(depths)-1, -1, -1)]
        
        # Project from latent space to initial dimension
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, dims[0]),
            nn.LayerNorm(dims[0])
        )
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        self.layer_types = []  # Store layer types separately
        
        for i in range(len(depths)):
            # Add transformer blocks
            for _ in range(depths[i]):
                # Ensure dim is divisible by num_heads
                dim = dims[i]
                num_head = num_heads[i]
                assert dim % num_head == 0, f"Dimension {dim} must be divisible by num_heads {num_head}"
                
                block = PointTransformerBlock(
                    dim=dim,
                    num_heads=num_head,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rate,
                    patch_size=patch_size * (2 ** (len(depths)-1-i))
                )
                self.layers.append(block)
                self.layer_types.append('block')
            
            # Add projection if not last stage
            if i < len(depths) - 1:
                proj = nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.LayerNorm(dims[i+1])
                )
                self.layers.append(proj)
                self.layer_types.append('proj')
        
        # Final projection to output channels
        self.final_proj = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Linear(dims[-1] // 2, out_chans)
        )
        
        self.patch_size = patch_size
    
    def upsample_positions(self, pos, mask=None):
        """Upsample positions by interpolation between neighboring points."""
        B, N, _ = pos.shape
        
        # Duplicate each position
        pos_expanded = pos.unsqueeze(2).expand(-1, -1, 2, -1)
        pos_flat = pos_expanded.reshape(B, N*2, 3)
        
        if mask is not None:
            # Duplicate mask
            mask = mask.unsqueeze(2).expand(-1, -1, 2)
            mask = mask.reshape(B, N*2)
        
        return pos_flat, mask
    
    def forward(self, z, pos, mask=None):
        """
        Args:
            z: (B, N, latent_dim) latent vectors
            pos: (B, N, 3) position coordinates
            mask: (B, N) optional boolean mask for valid patches
        Returns:
            x: (B, N', out_chans) reconstructed features
        """
        # Project from latent space
        x = self.latent_proj(z)
        
        # Process through transformer blocks
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == 'block':
                x = layer(x, pos, mask)
                # Upsample positions and mask if next layer is projection
                if i + 1 < len(self.layer_types) and self.layer_types[i + 1] == 'proj':
                    pos, mask = self.upsample_positions(pos, mask)
            else:  # projection layer
                x = layer(x)
        
        # Final projection to output channels
        x = self.final_proj(x)
        
        # Apply final mask if provided
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return x

class PointTransformerVAE(nn.Module):
    def __init__(self,
                 latent_dim=256,
                 embed_dim=256,
                 encoder_depths=[2, 2, 6, 2],
                 decoder_depths=[2, 2, 6, 2],
                 num_heads=[4, 8, 16, 32],
                 mlp_ratio=4.,
                 drop_path_rate=0.2,
                 patch_size=32,
                 in_chans=4):  # 3 spatial + 1 value
        super().__init__()
        
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Ensure dimensions are compatible with number of heads
        for i, num_head in enumerate(num_heads):
            dim = embed_dim * (2 ** i)
            assert dim % num_head == 0, f"Dimension {dim} must be divisible by num_heads {num_head}"
        
        # Encoder
        self.encoder = PointTransformerEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=encoder_depths,
            num_heads=num_heads,
            patch_size=patch_size
        )
        
        # Latent space projections
        final_dim = embed_dim * (2 ** (len(encoder_depths) - 1))
        self.mu_proj = nn.Linear(final_dim, latent_dim)
        self.logvar_proj = nn.Linear(final_dim, latent_dim)
        
        # Decoder
        self.decoder = PointTransformerDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            depths=decoder_depths,
            num_heads=num_heads[::-1],  # Reverse order for decoder
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_size=patch_size,
            out_chans=in_chans
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, features):
        x, pos, mask = self.encoder(features)
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar, pos, mask
    
    def decode(self, z, pos, mask=None):
        return self.decoder(z, pos, mask)
    
    def forward(self, features):
        """
        Args:
            features: (B, N, D) tensor of combined point features
                     where D includes both spatial coordinates and neural values
        Returns:
            recon_features: (B, N, D) reconstructed features
            mu: (B, N, latent_dim) mean of latent distribution
            logvar: (B, N, latent_dim) log variance of latent distribution
        """
        # Encode
        mu, logvar, pos, mask = self.encode(features)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_features = self.decode(z, pos, mask)
        
        return recon_features, mu, logvar
    
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

def create_point_transformer_vae(size='base'):
    """Create a Point Transformer VAE with the specified size configuration."""
    configs = {
        'small': {  # ~5M params
            'latent_dim': 128,
            'embed_dim': 128,  # Must be divisible by all num_heads values
            'encoder_depths': [2, 2, 4, 2],
            'decoder_depths': [2, 4, 2, 2],
            'num_heads': [2, 4, 8, 16],
            'mlp_ratio': 2,
            'patch_size': 16,
            'in_chans': 4  # 3 spatial + 1 value
        },
        'base': {  # ~50M params
            'latent_dim': 256,
            'embed_dim': 256,  # Must be divisible by all num_heads values
            'encoder_depths': [2, 2, 6, 2],
            'decoder_depths': [2, 6, 2, 2],
            'num_heads': [4, 8, 16, 32],
            'mlp_ratio': 4,
            'patch_size': 32,
            'in_chans': 4  # 3 spatial + 1 value
        },
        'large': {  # ~200M params
            'latent_dim': 512,
            'embed_dim': 512,  # Must be divisible by all num_heads values
            'encoder_depths': [2, 2, 18, 2],
            'decoder_depths': [2, 18, 2, 2],
            'num_heads': [8, 16, 32, 64],
            'mlp_ratio': 4,
            'patch_size': 32,
            'in_chans': 4  # 3 spatial + 1 value
        }
    }
    
    if size not in configs:
        raise ValueError(f"Size must be one of {list(configs.keys())}")
    
    # Verify that embed_dim is divisible by all num_heads values
    config = configs[size]
    embed_dim = config['embed_dim']
    for num_head in config['num_heads']:
        assert embed_dim % num_head == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_head}"
    
    return PointTransformerVAE(**config) 