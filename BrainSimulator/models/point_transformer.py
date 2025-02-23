import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, n_heads=4, head_dim=32, mlp_ratio=4, drop_path=0.0, 
                 attn_drop=0.0, drop=0.0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Position encoding MLPs
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        # Position-wise encoding for attention
        self.pos_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + self.pos_mlp(pos)
    
    def forward(self, x, pos):
        # x: (B, N, C), pos: (B, N, 3)
        B, N, C = x.shape
        
        # Pre-norm
        shortcut = x
        x = self.norm1(x)
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Add positional encodings to queries and keys
        pos_q = self.pos_embed(pos).reshape(B, N, self.n_heads, -1).permute(0, 2, 1, 3)
        pos_k = self.pos_embed(pos).reshape(B, N, self.n_heads, -1).permute(0, 2, 1, 3)
        
        q = q + pos_q
        k = k + pos_k
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Residual connection
        x = shortcut + self.drop_path(x)
        
        # MLP block with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class PointTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop_path=0., patch_size=32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, f"dim must be divisible by num_heads"
        
        # Multi-head self attention
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # Position encoding projects from 3D coordinates to per-head position encodings
        self.pos_embed = nn.Sequential(
            nn.Linear(3, self.head_dim),  # Project to head dimension first
            nn.GELU(),
            nn.Linear(self.head_dim, self.head_dim)  # Keep in head dimension
        )
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, pos, mask=None):
        """
        Args:
            x: (B, N, C) input features
            pos: (B, N, 3) position coordinates
            mask: (B, N) optional boolean mask for valid patches
        """
        B, N, C = x.shape
        assert C == self.dim, f"Input dimension {C} doesn't match expected dimension {self.dim}"
        
        # Multi-head self attention
        shortcut = x
        x = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, N, D/H
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, N, D/H
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, N, D/H
        
        # Compute position encoding per head
        pos_enc = self.pos_embed(pos)  # B, N, D/H
        pos_enc = pos_enc.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # B, H, N, D/H
        
        # Add position encoding to keys
        k = k + pos_enc
        
        # Compute attention with optional masking
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # B, H, N, N
        
        if mask is not None:
            # Convert mask to attention mask
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # B, 1, 1, N
            attn_mask = attn_mask & attn_mask.transpose(-2, -1)  # B, 1, N, N
            attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        
        # Apply attention to V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
        x = self.proj(x)
        
        # Apply mask if provided
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        # Residual connection
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Apply mask again after MLP if provided
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=4, embed_dim=256, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Project features to embedding dimension
        self.proj = nn.Sequential(
            nn.Linear(in_chans, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Normalization layer
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, features):
        """
        Args:
            features: (B, N, D) tensor of combined point features
                     where D includes both spatial coordinates and neural values
        Returns:
            x: (B, N//patch_size, embed_dim) embedded features
            centroids: (B, N//patch_size, 3) centroid positions
            mask: (B, N//patch_size) boolean mask indicating valid patches
        """
        B, N, D = features.shape
        assert D == self.in_chans, f"Input feature dimension {D} doesn't match expected in_chans {self.in_chans}"
        
        # Pad if necessary
        if N % self.patch_size != 0:
            pad_size = self.patch_size - (N % self.patch_size)
            features = torch.cat([features, features[:, :pad_size]], dim=1)
            N = features.shape[1]
        
        # Reshape to patches
        features = features.reshape(B, N // self.patch_size, self.patch_size, D)
        
        # Compute centroids using only spatial coordinates
        centroids = features[:, :, :, :3].mean(dim=2)  # [B, N//patch_size, 3]
        
        # Create mask for valid patches (where at least one point exists)
        mask = (features[:, :, :, 0].abs().sum(dim=2) > 0)  # [B, N//patch_size]
        
        # Average features within each patch
        x = features.mean(dim=2)  # [B, N//patch_size, D]
        
        # Project to embedding dimension
        x = self.proj(x)  # [B, N//patch_size, embed_dim]
        
        # Apply normalization
        x = self.norm(x)
        
        return x, centroids, mask

def group_points(features, centroids, k):
    """Group points around centroids using k-nearest neighbors.
    
    Args:
        features: (B, N, D) tensor of input features
                 where D includes both spatial coordinates and neural values
        centroids: (B, C, 3) tensor of centroids (only spatial coordinates)
        k: number of points to group around each centroid
    
    Returns:
        grouped_features: (B, C, k, D) tensor of grouped features
    """
    B, N, D = features.shape
    B, C, _ = centroids.shape
    device = features.device
    
    # Extract spatial coordinates for distance computation
    points_xyz = features[:, :, :3]  # B, N, 3
    
    # Compute distances in chunks to save memory
    chunk_size = 1024
    n_chunks = (N + chunk_size - 1) // chunk_size
    
    dist_list = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, N)
        chunk_points = points_xyz[:, start_idx:end_idx, :]  # B, chunk_size, 3
        
        # Compute distances for this chunk using only spatial coordinates
        chunk_dist = torch.cdist(centroids, chunk_points)  # B, C, chunk_size
        dist_list.append(chunk_dist)
    
    # Concatenate all chunks
    dist = torch.cat(dist_list, dim=2)  # B, C, N
    
    # Get k nearest neighbors
    k = min(k, N)
    _, idx = torch.topk(dist, k=k, dim=2, largest=False)  # B, C, k
    
    # Gather nearest neighbors (including all features)
    idx = idx.unsqueeze(3).expand(-1, -1, -1, D)  # B, C, k, D
    grouped_features = torch.gather(features.unsqueeze(1).expand(-1, C, -1, -1), 2, idx)
    
    # Center points around centroids (only spatial coordinates)
    grouped_xyz = grouped_features[:, :, :, :3]
    centroids_xyz = centroids.unsqueeze(2)  # B, C, 1, 3
    grouped_xyz = grouped_xyz - centroids_xyz
    
    # Combine centered coordinates with remaining features
    grouped_features = torch.cat([
        grouped_xyz,
        grouped_features[:, :, :, 3:]
    ], dim=3)
    
    return grouped_features

class PointTransformerEncoder(nn.Module):
    def __init__(self,
                 in_chans=4,
                 embed_dim=256,
                 depths=[2, 2, 6, 2],
                 num_heads=[4, 8, 16, 32],
                 patch_size=32):
        super().__init__()
        
        # Initial patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        
        for i in range(len(depths)):
            self.layers.append(
                PointTransformerBlock(
                    dim=dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=4,
                    drop_path=0.2,
                    patch_size=patch_size // (2 ** i)
                )
            )
    
    def forward(self, features):
        """
        Args:
            features: (B, N, D) tensor of combined point features
                     where D includes both spatial coordinates and neural values
        """
        # Extract spatial coordinates for positional information
        pos = features[:, :, :3]
        
        # Patch embedding
        x = self.patch_embed(features)
        
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, pos)
            # Downsample positions for next layer
            pos = pos[:, ::2, :]
        
        return x, pos 