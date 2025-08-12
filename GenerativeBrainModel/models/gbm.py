import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.spatiotemporal import SpatioTemporalNeuralAttention
# cimport pdb
import os


class GBM(nn.Module):
    def __init__(self, d_model, d_stimuli, n_heads, n_layers):
        super(GBM, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for rotary embeddings"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_stimuli = d_stimuli

        self.layers = nn.ModuleList([SpatioTemporalNeuralAttention(d_model, n_heads) for _ in range(n_layers)])
        
        self.stimuli_encoder = nn.Sequential(
            nn.Linear(d_stimuli, d_model),
            RMSNorm(d_model),
        )
        self.neuron_scalar_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            RMSNorm(d_model),
        )
        self.neuron_scalar_decoder_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # Parameter groups for optimizers
        # Treat encoders as "embed", attention stack as "body", decoder head as "head", muon optimizer is applied to the body only and adamw is applied to the embed and head
        self.embed = nn.ModuleDict({
            'stim': self.stimuli_encoder,
            'neuron': self.neuron_scalar_encoder,
        })
        self.body = self.layers
        self.head = self.neuron_scalar_decoder_head


    def forward(self, x, x_stimuli, point_positions, neuron_pad_mask, get_logits=True):
        # Takes as input sequences of shape x: (batch_size, seq_len, n_neurons)
        # x_stimuli: (batch_size, seq_len, d_stimuli)
        # point_positions: (batch_size, n_neurons, 3)
        # neuron_pad_mask: (batch_size, n_neurons)
        # Returns sequences of shape (batch_size, seq_len, n_neurons, d_model)
        B, T, N = x.shape

        x = x.unsqueeze(-1)
        # Keep dtype consistent with model weights (bf16 when enabled)
        if x.dtype != next(self.parameters()).dtype:
            x = x.to(next(self.parameters()).dtype)
        x = self.neuron_scalar_encoder(x)

        # (B, T, d_stimuli) -> (B, T, 1, d_model), acts as a global stimulus embedding as a token for each time step
        x_stimuli = x_stimuli.to(next(self.parameters()).dtype).unsqueeze(2)
        x_stimuli = self.stimuli_encoder(x_stimuli) # (B, T, 1, d_model)

        # concatenate the stimulus embedding to the input
        x = torch.cat([x, x_stimuli], dim=2) # (B, T, n_neurons + 1, d_model)
        
        # Adds an additional 1 value to neuron_pad_mask to account for the stimulus token
        neuron_pad_mask = torch.cat([neuron_pad_mask, torch.ones(B, 1, device=neuron_pad_mask.device)], dim=1) # (B, n_neurons + 1)

        # Adds a zero vector to point_positions to account for the stimulus token
        point_positions = torch.cat([point_positions, torch.zeros(B, 1, 3, device=point_positions.device)], dim=1) # (B, n_neurons + 1, 3)

        # Apply the layers
        for i, layer in enumerate(self.layers):
            x = layer(x, point_positions, neuron_pad_mask)

        # Remove the stimulus token
        x = x[:, :, :-1, :]

        # Decode the neuron scalars
        x = self.neuron_scalar_decoder_head(x)

        # Reshape to (batch_size, seq_len, n_neurons)
        x = x.squeeze(-1)

        if get_logits:
            return x

        return torch.sigmoid(x)

    def autoregress(self, init_x, init_stimuli, point_positions, neuron_pad_mask, future_stimuli=None, n_steps=10, context_len=12):
        # init_x: (B, T, n_neurons)
        # init_stimuli: (B, T, d_stimuli)
        # init_point_positions: (B, n_neurons, 3)
        # neuron_pad_mask: (B, n_neurons)
        # future_stimuli: (B, n_steps, d_stimuli)
        # n_steps: number of steps to generate
        # context_len: number of steps to use as context

        B, T, N = init_x.shape
        
        assert T >= context_len, "context_len must be less than or equal to T"
        assert n_steps > 0, "n_steps must be greater than 0"
        assert context_len > 0, "context_len must be greater than 0"

        if future_stimuli is None:
            future_stimuli = torch.zeros(B, n_steps, self.d_stimuli, device=init_x.device)
        else:
            assert future_stimuli.shape[0] == B, "future_stimuli must have the same batch size as init_x"
            assert future_stimuli.shape[1] == n_steps, "future_stimuli must have n_steps steps"
            assert future_stimuli.shape[2] == self.d_stimuli, "future_stimuli must have d_stimuli features"

        # Start with the initial sequence
        current_neuron_scalars = init_x
        current_stimuli = init_stimuli

        # generate n_steps steps
        for i in range(n_steps):
            # get the context from the current sequence
            context = current_neuron_scalars[:, -context_len:]
            context_stim = current_stimuli[:, -context_len:, :]
            # generate the next step using aligned context windows
            next_step = self.forward(context, context_stim, point_positions, neuron_pad_mask, get_logits=False)[:, -1:]  # (B, 1, n_neurons)
            # append the next step to the current sequence
            current_neuron_scalars = torch.cat([current_neuron_scalars, next_step], dim=1)
            # update the stimuli
            current_stimuli = torch.cat([current_stimuli, future_stimuli[:, i:i+1, :]], dim=1)

        # Return the full sequence including the original init_x and the generated steps
        return current_neuron_scalars
