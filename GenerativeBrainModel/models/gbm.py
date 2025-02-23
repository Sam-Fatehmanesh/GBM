import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.mambacore import StackedMamba
from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder

class GBM(nn.Module):
    def __init__(self, mamba_layers=1, mamba_dim=1024, pretrained_ae_path="/home/user1/projects/BrainSim/trained_simpleAE/checkpoints/best_model.pt"):
        """Generative Brain Model combining pretrained autoencoder with Mamba for sequential prediction.
        
        Args:
            mamba_layers: Number of Mamba layers
            mamba_dim: Hidden dimension for Mamba (should match autoencoder latent dim)
            pretrained_ae_path: Path to pretrained autoencoder checkpoint
        """
        super(GBM, self).__init__()
        
        # Load pretrained autoencoder
        self.autoencoder = SimpleAutoencoder(input_size=256*128, hidden_size=mamba_dim)
        checkpoint = torch.load(pretrained_ae_path)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
        # Create Mamba sequence model
        self.mamba = StackedMamba(d_model=mamba_dim, num_layers=mamba_layers)
        
        # Save dimensions
        self.latent_dim = mamba_dim
        self.grid_size = (256, 128)
        self.flat_size = np.prod(self.grid_size)
    
    def encode(self, x_flat):
        """Encode flattened grids to latent vectors.
        
        Args:
            x_flat: Tensor of shape (batch_size, 256*128) or (batch_size, seq_len, 256*128)
        Returns:
            latents: Tensor of shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)
        """
        return self.autoencoder.encoder(x_flat)
    
    def decode(self, z):
        """Decode latent vectors to flattened grid probabilities.
        
        Args:
            z: Tensor of shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)
        Returns:
            grids_flat: Tensor of shape (batch_size, 256*128) or (batch_size, seq_len, 256*128)
        """
        return torch.sigmoid(self.autoencoder.decoder(z))
    
    def forward(self, x):
        """Forward pass predicting next frame in sequence.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of grids
        Returns:
            next_frame_probs: Tensor of shape (batch_size, seq_len-1, 256, 128) containing
                            predicted probabilities for next frame in sequence
        """
        batch_size, seq_len = x.shape[:2]
        
        # Flatten all grids
        x_flat = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 256*128)
        
        # Encode all frames to latent space
        latents = self.encode(x_flat)  # (batch_size, seq_len, latent_dim)
        
        # Pass through Mamba to get predictions
        mamba_out = self.mamba(latents)  # (batch_size, seq_len, latent_dim)
        
        # Decode predictions back to flattened grids
        predictions_flat = self.decode(mamba_out)  # (batch_size, seq_len, 256*128)
        
        # Reshape predictions back to grids
        predictions = predictions_flat.view(batch_size, seq_len, *self.grid_size)
        
        # Return predictions for frames 2 to N
        return predictions[:, :-1]  # (batch_size, seq_len-1, 256, 128)
    
    def compute_loss(self, pred, target):
        """Compute binary cross entropy loss between predictions and targets.
        
        Args:
            pred: Tensor of shape (batch_size, seq_len-1, 256, 128) containing predicted probabilities
            target: Tensor of shape (batch_size, seq_len-1, 256, 128) containing target binary grids
        Returns:
            loss: Scalar loss value
        """
        return F.binary_cross_entropy(pred, target)

