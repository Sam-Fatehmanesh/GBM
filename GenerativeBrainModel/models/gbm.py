import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.mambacore import StackedMamba
from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder

def binary_focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
    """
    Compute binary focal loss for a batch of predictions.
    
    Args:
      pred (torch.Tensor): raw logits
      target (torch.Tensor): binary labels (0 or 1) of the same shape as pred.
      alpha (float): balancing factor.
      gamma (float): focusing parameter.
      reduction (str): reduction method ('mean', 'sum', or 'none').
      eps (float): small value for numerical stability.
    
    Returns:
      torch.Tensor: focal loss value.
    """
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(pred)
    
    # Compute p_t depending on the true label
    p_t = p * target + (1 - p) * (1 - target)
    
    # Compute the loss
    loss = -alpha * (1 - p_t)**gamma * torch.log(p_t + eps)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class GBM(nn.Module):
    def __init__(self, mamba_layers=1, mamba_dim=1024, pretrained_ae_path="trained_simpleAE/checkpoints/best_model.pt"):
        """Generative Brain Model combining pretrained autoencoder with Mamba for sequential prediction.
        
        Args:
            mamba_layers: Number of Mamba layers
            mamba_dim: Hidden dimension for Mamba (should match autoencoder latent dim)
            pretrained_ae_path: Path to pretrained autoencoder checkpoint
        """
        super(GBM, self).__init__()
        
        # Load pretrained autoencoder
        self.autoencoder = SimpleAutoencoder(input_size=256*128, hidden_size=mamba_dim)
        checkpoint = torch.load(pretrained_ae_path, map_location='cpu')  # Load to CPU first for better memory management
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze autoencoder parameters
        # for param in self.autoencoder.parameters():
        #     param.requires_grad = False
            
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
        # Ensure input is float32 for autoencoder compatibility
        if x_flat.dtype != torch.float32:
            x_flat = x_flat.float()
            
        return self.autoencoder.encoder(x_flat)
    
    def decode(self, z):
        """Decode latent vectors to flattened grid logits.
        
        Args:
            z: Tensor of shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)
        Returns:
            grids_flat: Tensor of shape (batch_size, 256*128) or (batch_size, seq_len, 256*128)
        """
        # Return logits directly without sigmoid for use with binary_cross_entropy_with_logits
        return self.autoencoder.decoder(z)
    
    def forward(self, x):
        """Forward pass predicting next frame in sequence.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of grids
                Can be uint8, float16, or float32 data type
        Returns:
            next_frame_logits: Tensor of shape (batch_size, seq_len-1, 256, 128) containing
                            predicted logits for next frame in sequence
        """
        # Remember original data type for output consistency
        orig_dtype = x.dtype
        batch_size, seq_len = x.shape[:2]
        
        # Flatten all grids more efficiently
        x_flat = x.reshape(batch_size, seq_len, -1)  # Use reshape instead of view for better compatibility
        
        # Encode all frames to latent space - encode method will convert to float32
        latents = self.encode(x_flat)  # (batch_size, seq_len, latent_dim)
        
        # Pass through Mamba to get predictions
        mamba_out = self.mamba(latents)  # (batch_size, seq_len, latent_dim)
        
        # Decode predictions back to flattened grid logits (no sigmoid)
        predictions_flat = self.decode(mamba_out)  # (batch_size, seq_len, 256*128)
        
        # Reshape predictions back to grids more efficiently
        predictions = predictions_flat.reshape(batch_size, seq_len, *self.grid_size)
        
        # Return predictions for frames 2 to N
        # Match original dtype if needed
        if predictions.dtype != orig_dtype and orig_dtype != torch.float32:
            return predictions[:, :-1].to(orig_dtype)
        else:
            return predictions[:, :-1]  # (batch_size, seq_len-1, 256, 128)
    
    def compute_loss(self, pred, target):
        """Compute focal loss between predictions and targets to address class imbalance.
        
        Args:
            pred: Tensor of shape (batch_size, seq_len-1, 256, 128) containing predicted logits
            target: Tensor of shape (batch_size, seq_len-1, 256, 128) containing target binary grids
                   Can be uint8, float16, or float32
        Returns:
            loss: Scalar loss value
        """
        # Ensure target is float32 for loss calculation
        if target.dtype != torch.float32:
            target = target.float()
        
        # Ensure pred is float32 for loss calculation if it's not already
        if pred.dtype != torch.float32:
            pred = pred.float()
        
        # Flatten predictions and targets for more efficient loss computation
        batch_size, seq_len = pred.shape[:2]
        pred_flat = pred.reshape(-1, self.flat_size)
        target_flat = target.reshape(-1, self.flat_size)
        
        # For efficient memory usage, compute loss in chunks for very large batches
        if pred_flat.shape[0] > 32:  # Process in chunks for large batches
            total_loss = 0.0
            chunk_size = 32
            num_chunks = (pred_flat.shape[0] + chunk_size - 1) // chunk_size  # Ceiling division
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, pred_flat.shape[0])
                
                # Use binary focal loss instead of BCE
                chunk_loss = binary_focal_loss(
                    pred_flat[start_idx:end_idx], 
                    target_flat[start_idx:end_idx],
                    alpha=0.25,  # Balance parameter
                    gamma=2.0,   # Focusing parameter
                    reduction='sum'
                )
                total_loss += chunk_loss
                
            # Average over all samples
            return total_loss / (pred_flat.shape[0] * self.flat_size)
        else:
            # For smaller batches, compute all at once
            return binary_focal_loss(
                pred_flat, 
                target_flat,
                alpha=0.25,  # Balance parameter
                gamma=2.0    # Focusing parameter
            )
            
    def get_predictions(self, x):
        """Forward pass returning the sigmoid of logits for actual probabilities.
        Useful for inference when probabilities are needed rather than logits.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of grids
        Returns:
            next_frame_probs: Tensor of shape (batch_size, seq_len-1, 256, 128) containing
                            predicted probabilities for next frame in sequence
        """
        # Get logits from forward pass
        logits = self.forward(x)
        
        # Apply sigmoid to get probabilities
        return torch.sigmoid(logits)

