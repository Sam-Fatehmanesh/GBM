import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from BrainSimulator.models.cnn import CNNLayer, DeCNNLayer
from BrainSimulator.models.mlp import MLP

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight through estimator
        
        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, encoding_indices.view(input_shape[:-1])

class VQVAE(nn.Module):
    def __init__(self, image_height, image_width, num_embeddings=65536, embedding_dim=1024, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.embedding_dim = embedding_dim
        
        self.scalings = [4, 4, 4]  # Down/Upsampling factors
        
        # Calculate padded dimensions to make them divisible by total scaling
        total_scaling = int(np.prod(self.scalings))
        self.padded_height = int(np.ceil(image_height / total_scaling) * total_scaling)
        self.padded_width = int(np.ceil(image_width / total_scaling) * total_scaling)
        
        # Calculate padding
        self.pad_height = int(self.padded_height - image_height)
        self.pad_width = int(self.padded_width - image_width)
        self.pad_top = int(self.pad_height // 2)
        self.pad_bottom = int(self.pad_height - self.pad_top)
        self.pad_left = int(self.pad_width // 2)
        self.pad_right = int(self.pad_width - self.pad_left)
        
        # Calculate sizes after CNN layers
        self.post_cnn_height = int(self.padded_height // total_scaling)
        self.post_cnn_width = int(self.padded_width // total_scaling)
        
        # Encoder
        self.encoder = nn.Sequential(
            CNNLayer(1, 64, 7),  # Increased initial channels
            nn.MaxPool2d(self.scalings[0], stride=self.scalings[0]),
            
            CNNLayer(64, 512, 5),  # Increased middle channels
            nn.MaxPool2d(self.scalings[1], stride=self.scalings[1]),
            
            CNNLayer(512, 1024, 3),  # Increased channels
            nn.MaxPool2d(self.scalings[2], stride=self.scalings[2]),
            
            # Add a final convolution to match embedding_dim
            nn.Conv2d(1024, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            
            # Add extra processing for higher-level features
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU()
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Initial processing of quantized vectors
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.GELU(),
            
            # Project to decoder channels
            nn.Conv2d(embedding_dim, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            
            DeCNNLayer(1024, 512, scale_factor=self.scalings[2], kernel_size=3),
            
            DeCNNLayer(512, 64, scale_factor=self.scalings[1], kernel_size=5),
            
            DeCNNLayer(64, 1, scale_factor=self.scalings[0], kernel_size=5, last_activation=False),
            
            nn.Sigmoid()
        )
    
    def pad_input(self, x):
        return F.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom), mode='constant', value=0)
    
    def unpad_output(self, x):
        if self.pad_top + self.pad_bottom > 0:
            x = x[:, :, self.pad_top:-self.pad_bottom] if self.pad_bottom > 0 else x[:, :, self.pad_top:]
        if self.pad_left + self.pad_right > 0:
            x = x[:, :, :, self.pad_left:-self.pad_right] if self.pad_right > 0 else x[:, :, :, self.pad_left:]
        return x
    
    def encode(self, x):
        # Pad input
        x = self.pad_input(x)
        # Encode
        z = self.encoder(x)
        # Quantize
        quantized, vq_loss, encoding_indices = self.vq(z)
        return quantized, vq_loss, encoding_indices
    
    def decode(self, quantized):
        x = self.decoder(quantized)
        # Remove padding
        x = self.unpad_output(x)
        return x
    
    def forward(self, x):
        quantized, vq_loss, encoding_indices = self.encode(x)
        x_recon = self.decode(quantized)
        return x_recon, vq_loss, encoding_indices 