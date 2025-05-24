import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.mambacore import StackedMamba
from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder

class GBM(nn.Module):
    def __init__(self, mamba_layers=1, mamba_dim=1024, mamba_state_multiplier=1, pretrained_ae_path="trained_simpleAE/checkpoints/best_model.pt"):
        """Generative Brain Model combining pretrained autoencoder with Mamba for sequential prediction.
        
        Args:
            mamba_layers: Number of Mamba layers
            mamba_dim: Hidden dimension for Mamba (should match autoencoder latent dim)
            pretrained_ae_path: Path to pretrained autoencoder checkpoint
        """
        super(GBM, self).__init__()
        
        # Load pretrained autoencoder
        self.autoencoder = SimpleAutoencoder(input_size=256*128, hidden_size=mamba_dim)#LocallyConnectedAutoencoder(hidden_size=mamba_dim)#(input_size=256*128, hidden_size=mamba_dim)
        if mamba_dim == 1024:
            checkpoint = torch.load(pretrained_ae_path, map_location='cpu')  # Load to CPU first for better memory management
            self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Create Mamba sequence model
        self.mamba = StackedMamba(d_model=mamba_dim, num_layers=mamba_layers, state_multiplier=mamba_state_multiplier)
        
        # Save dimensions
        self.latent_dim = mamba_dim
        self.grid_size = (256, 128)
        self.flat_size = int(np.prod(self.grid_size))

    def encode(self, x_flat):
        """Encode flattened grids to latent vectors.
        
        Args:
            x_flat: Tensor of shape (batch_size, 256*128) or (batch_size, seq_len, 256*128)
        Returns:
            latents: Tensor of shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)
        """
        return self.autoencoder.encoder(x_flat)
    
    def decode(self, z):
        """Decode latent vectors to flattened grid logits.
        
        Args:
            z: Tensor of shape (batch_size, latent_dim) or (batch_size, seq_len, latent_dim)
        Returns:
            grids_flat: Tensor of shape (batch_size, 256*128) or (batch_size, seq_len, 256*128)
        """
        return  self.autoencoder.decoder(z) 
    
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

        # Remember original shape for output consistency
        original_shape = x.shape
        
        # Flatten all grids more efficiently
        x_flat = x.reshape(batch_size, seq_len, -1)  # Use reshape instead of view for better compatibility
        
        # Encode all frames to latent space - encode method will convert to float32
        latents = self.encode(x_flat)  # (batch_size, seq_len, latent_dim)
        
        # Pass through Mamba to get predictions
        mamba_out = self.mamba(latents)  # (batch_size, seq_len, latent_dim)
        
        # Decode predictions back to flattened grid logits (no sigmoid)
        predictions_flat = self.decode(mamba_out).reshape(batch_size, seq_len, *self.grid_size)  # (batch_size, seq_len, 256*128)
        
        # Reshape predictions back to grids more efficiently
        predictions = predictions_flat.reshape(batch_size, seq_len, *self.grid_size)
        
        # Return predictions for frames 2 to N
        # Match original dtype if needed
        if predictions.dtype != orig_dtype and orig_dtype != torch.float32:
            return predictions[:, :-1].to(orig_dtype)
        else:
            return predictions[:, :-1]  # (batch_size, seq_len-1, 256, 128)
    
    def compute_loss(self, pred, target):
        """Compute binary cross entropy loss between predictions and targets.
        
        Args:
            pred: Tensor of shape (batch_size, seq_len-1, 256, 128) containing predicted logits
            target: Tensor of shape (batch_size, seq_len-1, 256, 128) containing target binary or probability grids
                   Can be uint8, float16, or float32
        Returns:
            loss: Scalar loss value
        """
        
        return F.binary_cross_entropy_with_logits(pred, torch.clamp(target, min=0.0, max=1.0), reduction='mean')
            
    def get_predictions(self, x, temperature=1.0):
        """Forward pass returning the sigmoid of logits for actual probabilities.
        Useful for inference when probabilities are needed rather than logits.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of grids
            temperature: Temperature for sampling
        Returns:
            next_frame_probs: Tensor of shape (batch_size, seq_len-1, 256, 128) containing
                            predicted probabilities for next frame in sequence
        """
        # Get logits from forward pass
        logits = self.forward(x)

        # Add a small epsilon to the temperature to avoid division by zero
        temperature += 1e-8

        # Apply sigmoid to get probabilities
        return torch.sigmoid(logits / temperature)

    def sample_binary_predictions(self, x):
        """Sample predictions from the model.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of z plane spike probabilities
            num_samples: Number of samples to generate
        
        Returns:
            samples: Tensor of shape (batch_size, num_samples, seq_len-1, 256, 128) containing
                     generated samples
        """
        # Ensure input values are probabilities in range [0,1]
        # This prevents CUDA assertion failures with torch.bernoulli
        clamped_probs = torch.clamp(x, min=0.0, max=1.0)
        return torch.bernoulli(clamped_probs)

    def generate_autoregressive_brain(self, init_x, num_steps=30):
        """Generate a brain sequence from an initial grid sequence using autoregressive sampling.
        
        Args:
            init_x: Tensor of shape (batch_size, seq_len, 256, 128) containing sequence of z plane spike probabilities
            num_steps: Number of steps to generate
        Returns:
            samples: Tensor of shape (batch_size, seq_len, 256, 128) containing
                     generated samples
        """
        x = init_x
        for i in range(num_steps):
            # Get logits for the next frame
            logits = self.forward(x)[:, -1, :, :]
            # Convert logits to probabilities with sigmoid and ensure valid range
            probs = torch.clamp(torch.sigmoid(logits), min=0.0, max=1.0)
            # Sample binary values
            prediction = torch.bernoulli(probs).unsqueeze(1)
            # Concatenate with existing sequence
            x = torch.cat((x, prediction), dim=1)
        return x
          