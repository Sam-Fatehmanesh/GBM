import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.mambacore import StackedMamba
from GenerativeBrainModel.models.vae import VariationalAutoEncoder

# Generative Brain Model
class GBM(nn.Module):
    def __init__(self, d_model, num_layers, num_distributions, num_categories):
        super(GBM, self).__init__()

        # We'll set the VAE later when loading pretrained
        self.pretrained_vae = None
        
        # Save parameters
        self.num_distributions = num_distributions
        self.num_categories = num_categories

        self.mamba_core = nn.Sequential(
            nn.Linear(num_categories * num_distributions, d_model),
            StackedMamba(d_model, num_layers),
            nn.Linear(d_model, num_categories * num_distributions),
        )

        self.head_softmax = nn.Softmax(dim=-1)

    def load_pretrained_vae(self, path):
        """Load a pretrained VAE from a checkpoint file"""
        if self.pretrained_vae is None:
            raise ValueError("VAE must be set before loading weights")
            
        self.pretrained_vae.load_state_dict(torch.load(path))

    def encode_images(self, x):
        """
        Encode a sequence of images to latent space using pretrained VAE
        Args:
            x: Input tensor of shape (batch, seq_len, 1, height, width)
        Returns:
            Latent tensor of shape (batch, seq_len, latent_dim)
        """
        if self.pretrained_vae is None:
            raise ValueError("VAE not initialized")
            
        batch_size, seq_len = x.shape[:2]
        # Reshape to (batch * seq_len, 1, height, width)
        x_flat = x.reshape(-1, *x.shape[2:])
        # Encode
        latent_sample, _ = self.pretrained_vae.encode(x_flat)
        # Reshape back to (batch, seq_len, latent_dim)
        return latent_sample.reshape(batch_size, seq_len, -1)

    def decode_images(self, x):
        """
        Decode a sequence of latent vectors to images using pretrained VAE
        Args:
            x: Latent tensor of shape (batch, seq_len, latent_dim)
        Returns:
            Image tensor of shape (batch, seq_len, 1, height, width)
        """
        if self.pretrained_vae is None:
            raise ValueError("VAE not initialized")
            
        batch_size, seq_len = x.shape[:2]
        # Reshape to (batch * seq_len, latent_dim)
        x_flat = x.reshape(-1, x.shape[-1])
        # Decode
        decoded = self.pretrained_vae.decode(x_flat)
        # Reshape back to (batch, seq_len, 1, height, width)
        return decoded.reshape(batch_size, seq_len, *decoded.shape[1:])

    def forward(self, x):
        batch_size, seq_len, latent_dim = x.shape
        """
        Forward pass through the GBM model.
        Args:
            x: Input tensor of shape (batch, seq_len, latent_dim)
               This should be a sequence of latent vectors from the VAE
               The input to the model uses integer encoding
               The output uses one-hot encoding
        Returns:
            Predicted next latent vector(s) with same shape as input
        """
        x = self.mamba_core(x)
        x = x.view(batch_size, seq_len, self.num_distributions, -1)
        x = self.head_softmax(x)
        x = x.view(batch_size, seq_len, -1)
        return x
    
    def predict_image_to_image(self, x):
        """
        Predict the next frame(s) given a sequence of frames
        Args:
            x: Input image tensor of shape (batch, seq_len, 1, height, width)
        Returns:
            Predicted next frame(s) with shape (batch, seq_len, 1, height, width)
        """
        if self.pretrained_vae is None:
            raise ValueError("VAE not initialized")
        
        batch_size, seq_len, _, height, width = x.shape
            
        # Encode sequence to latent space
        latent_sequence = self.encode_images(x)
        # Predict next latent states
        predicted_latents = self.forward(latent_sequence)
        # Perform argmax to go from distribution to integer encoding
        predicted_latents = torch.argmax(predicted_latents.view(batch_size, seq_len, self.num_distributions, -1), dim=-1)
        # Convert to one-hot encodinglaser
        predicted_latents = F.one_hot(predicted_latents, num_classes=self.num_categories)
        # Reshape to (batch, seq_len, 1, height, width)
        predicted_latents = predicted_latents.view(batch_size, seq_len, -1)
        # Decode predicted latents back to images
        predicted_frames = self.decode_images(predicted_latents)
        return predicted_frames