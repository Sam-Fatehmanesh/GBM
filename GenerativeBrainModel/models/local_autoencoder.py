import torch
import torch.nn as nn
import torch.nn.functional as F


class LocallyConnectedAutoencoder(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(LocallyConnectedAutoencoder, self).__init__()
        
        # Define dimensions
        self.height, self.width = 256, 128
        self.patch_size = 32
        self.num_patches_h = self.height // self.patch_size
        self.num_patches_w = self.width // self.patch_size
        self.total_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.patch_size * self.patch_size
        
        # Ensure hidden_per_patch is an integer
        self.hidden_per_patch = max(1, hidden_size // self.total_patches)
        # Adjust hidden_size to be divisible by total_patches
        self.hidden_size = self.hidden_per_patch * self.total_patches
        
        # Create weight matrices for each patch
        # Encoder: [total_patches, hidden_per_patch, patch_dim]
        self.encoder_weights = nn.Parameter(
            torch.Tensor(self.total_patches, self.hidden_per_patch, self.patch_dim)
        )
        self.encoder_bias = nn.Parameter(
            torch.Tensor(self.total_patches, self.hidden_per_patch)
        )
        
        # Decoder: [total_patches, patch_dim, hidden_per_patch]
        self.decoder_weights = nn.Parameter(
            torch.Tensor(self.total_patches, self.patch_dim, self.hidden_per_patch)
        )
        self.decoder_bias = nn.Parameter(
            torch.Tensor(self.total_patches, self.patch_dim)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize weight matrices using Xavier initialization"""
        nn.init.xavier_uniform_(self.encoder_weights)
        nn.init.xavier_uniform_(self.decoder_weights)
        nn.init.zeros_(self.encoder_bias)
        nn.init.zeros_(self.decoder_bias)
        
    def encode(self, x):
        """Encode input by processing all patches in parallel."""
        # Initialize seq_len to None by default
        seq_len = None
        
        # Store original shape
        original_shape = x.shape
        
        # Handle different input shapes
        if len(x.shape) == 2:  # [batch, input_size]
            batch_size = x.shape[0]
            x = x.view(batch_size, self.height, self.width)
        elif len(x.shape) == 3 and x.shape[-1] == self.height * self.width:  # [batch, seq, input_size]
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.view(batch_size, seq_len, self.height, self.width)
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, self.height, self.width)
            batch_size = x.shape[0]  # Update batch_size to include seq_len
        else:
            # Handle unexpected input shape
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)  # Flatten any other dimensions
            x = x.view(batch_size, self.height, self.width)
        
        # Extract patches: [batch, height, width] -> [batch, num_patches_h, num_patches_w, patch_size, patch_size]
        x_unfolded = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        
        # Reshape to [batch, total_patches, patch_dim]
        x_patches = x_unfolded.reshape(batch_size, self.total_patches, self.patch_dim)
        
        # Process all patches in parallel using einsum
        # [batch, total_patches, patch_dim] @ [total_patches, hidden_per_patch, patch_dim] -> [batch, total_patches, hidden_per_patch]
        encoded = torch.einsum('btp,thp->bth', x_patches, self.encoder_weights)
        
        # Add bias
        encoded = encoded + self.encoder_bias.unsqueeze(0)
        
        # Apply non-linearity
        # encoded = F.relu(encoded)
        
        # Reshape to [batch, hidden_size]
        encoded = encoded.reshape(batch_size, -1)
        
        # Reshape back if needed
        if seq_len is not None:
            encoded = encoded.reshape(original_shape[0], seq_len, -1)
            
        return encoded
    
    def decode(self, z):
        """Decode latent vectors by processing all patches in parallel."""
        # Handle different input shapes
        if len(z.shape) == 3:  # [batch, seq, hidden]
            batch_size, seq_len = z.shape[0], z.shape[1]
            z = z.reshape(-1, z.shape[-1])
            original_batch_size = batch_size
        else:
            batch_size = z.shape[0]
            seq_len = None
            original_batch_size = batch_size
        
        # Split latent vector into patches: [batch, total_patches, hidden_per_patch]
        z_patches = z.reshape(batch_size, self.total_patches, self.hidden_per_patch)
        
        # Process all patches in parallel using einsum
        # [batch, total_patches, hidden_per_patch] @ [total_patches, patch_dim, hidden_per_patch] -> [batch, total_patches, patch_dim]
        decoded = torch.einsum('bth,tph->btp', z_patches, self.decoder_weights)
        
        # Add bias
        decoded = decoded + self.decoder_bias.unsqueeze(0)
        
        # Reshape to image patches: [batch, num_patches_h, num_patches_w, patch_size, patch_size]
        decoded = decoded.reshape(batch_size, self.num_patches_h, self.num_patches_w, self.patch_size, self.patch_size)
        
        # Reconstruct image: [batch, height, width]
        decoded = decoded.permute(0, 1, 3, 2, 4).reshape(batch_size, self.height, self.width)
        
        # Flatten to [batch, input_size]
        decoded = decoded.reshape(batch_size, self.height * self.width)
        
        # Reshape back if needed
        if seq_len is not None:
            decoded = decoded.reshape(original_batch_size, seq_len, -1)
            
        return decoded
    
    def forward(self, x):
        # Store original shape for reshaping the output
        original_shape = x.shape
        
        # Encode and decode
        z = self.encode(x)
        decoded = self.decode(z)
        
        # Apply sigmoid
        decoded = torch.sigmoid(decoded)
        
        # Ensure output has the same shape as input
        return decoded.reshape(original_shape)