import torch
from torch import nn
import torch.nn.functional as F
import pdb
from BrainSimulator.models.cnn import CNNLayer, DeCNNLayer
from BrainSimulator.models.mlp import MLP
from BrainSimulator.custom_functions.utils import STMNsampler
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_n, latent_size_sqrt=16):
        super(VariationalAutoEncoder, self).__init__()

        self.image_n = image_n
        self.image_latent_size_sqrt = int(latent_size_sqrt)
        self.image_latent_size = latent_size_sqrt**2
        
        
        self.scale_0 = 8
        self.scale_1 = 8

        self.scalings = [8, 8]


        self.latent_channels_size = self.image_latent_size_sqrt

        self.post_cnn_encoder_side_size = int(self.image_n / np.prod(self.scalings))
        self.post_cnn_encoder_size = (self.post_cnn_encoder_side_size**2) * self.latent_channels_size


        
        
        self.encoder = nn.Sequential(
            CNNLayer(1, 64, 3),
            nn.MaxPool2d(self.scalings[0], stride=self.scalings[0]),

            CNNLayer(64, self.latent_channels_size, 3),
            nn.MaxPool2d(self.scalings[1], stride=self.scalings[1]),

            nn.Flatten(),
            MLP(3, self.post_cnn_encoder_size, self.image_latent_size, self.image_latent_size)
        )

        


        self.softmax_act = nn.Softmax(dim=1)
        self.sampler = STMNsampler()




        # Decoder
        self.decoder = nn.Sequential(
            
            MLP(3, self.image_latent_size, self.image_latent_size, self.post_cnn_encoder_size),
            
            nn.Unflatten(1, (self.latent_channels_size, self.post_cnn_encoder_side_size, self.post_cnn_encoder_side_size)),

            DeCNNLayer(self.latent_channels_size, 64, kernel_size=self.scalings[1], stride=self.scalings[1], padding=0),

            DeCNNLayer(64, 1, kernel_size=self.scalings[0], stride=self.scalings[0], padding=0),
            
            
            nn.Sigmoid(),
        )



    def encode(self, x):
        batch_dim = x.shape[0]
        

        x = self.encoder(x)


        x = x.view(batch_dim * self.image_latent_size_sqrt, self.image_latent_size_sqrt)

        distributions = self.softmax_act(x)
        distributions = 0.99*distributions + 0.01*torch.ones_like(distributions)/self.image_latent_size_sqrt

        sample = self.sampler(distributions)

        sample = sample.view(batch_dim, self.image_latent_size)
        distributions = distributions.view(batch_dim, self.image_latent_size)        

        return sample, distributions

    def decode(self, z):


        z = self.decoder(z)


        return z
    
    def forward(self, x):
        
        latent_sample, latent_distribution = self.encode(x)
        

        decoded_out = self.decode(latent_sample)


        return decoded_out, latent_sample, latent_distribution