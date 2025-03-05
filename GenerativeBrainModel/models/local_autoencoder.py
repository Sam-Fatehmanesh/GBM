import torch
import torch.nn as nn
import torch.nn.functional as F


class LocallyConnectedAutoencoder(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(LocallyConnectedAutoencoder, self).__init__()
        
        # Define dimensions
        self.height, self.width = 256, 128
        self.patch_size = 16
        self.num_patches_h = self.height // self.patch_size
        self.num_patches_w = self.width // self.patch_size
        self.total_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.patch_size * self.patch_size
        self.hidden_per_patch = hidden_size // self.total_patches
        
        # Use grouped linear layers - much more efficient
        self.encoder = nn.Conv1d(
            in_channels=self.patch_dim,
            out_channels=self.hidden_per_patch,
            kernel_size=1,
            groups=self.total_patches,
            bias=True
        )
        
        self.decoder = nn.Conv1d(
            in_channels=self.hidden_per_patch,
            out_channels=self.patch_dim,
            kernel_size=1,
            groups=self.total_patches,
            bias=True
        )
        
    def encode(self, x):
        """Encode input using grouped convolutions."""
        # Handle different input shapes
        if len(x.shape) == 2:  # [batch, input_size]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.height, self.width)
            seq_len = None
        elif len(x.shape) == 3 and x.shape[-1] == self.height * self.width:  # [batch, seq, input_size]
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.view(batch_size, seq_len, self.height, self.width)
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, self.height, self.width)
        
        # Rearrange into patches
        # [batch, height, width] -> [batch, total_patches, patch_dim]
        x_patches = x.unfold(1, self.patch_size, self.patch_size) \
                     .unfold(2, self.patch_size, self.patch_size) \
                     .reshape(-1, self.total_patches, self.patch_dim)
        
        # Rearrange for grouped convolution
        # [batch, total_patches, patch_dim] -> [batch, patch_dim*total_patches, 1]
        x_grouped = x_patches.transpose(1, 2).reshape(-1, self.patch_dim * self.total_patches, 1)
        
        # Apply grouped convolution
        # [batch, patch_dim*total_patches, 1] -> [batch, hidden_per_patch*total_patches, 1]
        encoded = self.encoder(x_grouped)
        
        # Reshape to final output
        # [batch, hidden_per_patch*total_patches, 1] -> [batch, hidden_size]
        encoded = encoded.reshape(-1, self.hidden_per_patch * self.total_patches)
        
        # Reshape back if needed
        if seq_len is not None:
            encoded = encoded.reshape(batch_size, seq_len, -1)
            
        return encoded
    
    def decode(self, z):
        """Decode latent vectors using grouped convolutions."""
        # Handle different input shapes
        original_shape = z.shape
        if len(z.shape) == 3:  # [batch, seq, hidden]
            batch_size, seq_len = z.shape[0], z.shape[1]
            z = z.reshape(-1, z.shape[-1])
        else:
            batch_size = z.shape[0]
            seq_len = None
            
        # Reshape for grouped convolution
        # [batch, hidden_size] -> [batch, hidden_per_patch*total_patches, 1]
        z = z.reshape(-1, self.hidden_per_patch * self.total_patches, 1)
        
        # Apply grouped convolution
        # [batch, hidden_per_patch*total_patches, 1] -> [batch, patch_dim*total_patches, 1]
        decoded = self.decoder(z)
        
        # Reshape to patches
        # [batch, patch_dim*total_patches, 1] -> [batch, total_patches, patch_dim]
        decoded = decoded.reshape(-1, self.total_patches, self.patch_dim)
        
        # Reconstruct image from patches
        # [batch, total_patches, patch_dim] -> [batch, num_patches_h, num_patches_w, patch_size, patch_size]
        decoded = decoded.reshape(
            -1, self.num_patches_h, self.num_patches_w, self.patch_size, self.patch_size
        )
        
        # [batch, num_patches_h, num_patches_w, patch_size, patch_size] -> [batch, height, width]
        decoded = decoded.permute(0, 1, 3, 2, 4).reshape(-1, self.height, self.width)
        
        # Flatten to [batch, input_size]
        decoded = decoded.reshape(-1, self.height * self.width)
        
        # Reshape back if needed
        if seq_len is not None:
            decoded = decoded.reshape(batch_size, seq_len, -1)
            
        return decoded
    
    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return torch.sigmoid(decoded)