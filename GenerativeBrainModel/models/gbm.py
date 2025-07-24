import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from GenerativeBrainModel.models.rms import RMSNorm
from GenerativeBrainModel.models.mlp import MLP
from GenerativeBrainModel.models.linnormencode import LinNormEncoder
from mamba_ssm import Mamba2 as Mamba
import pdb
import os

class SpatioTemporalRegionModel(nn.Module):
    def __init__(self, d_model, n_heads, n_regions):
        super(SpatioTemporalRegionModel, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = n_regions
        self.spatial_model = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.temporal_model = Mamba(d_model=d_model)
        self.norm_1 = RMSNorm(d_model)
        self.norm_2 = RMSNorm(d_model)
        self.norm_3 = RMSNorm(d_model)
        self.mlp = MLP(layers_num=1, input_size=d_model, hidden_size=d_model*2, output_size=d_model)

    def forward(self, x):
        # expects x of shape (batch_size, seq_len, n_regions, d_model)
        B, T, N, D = x.shape

        # --- Spatial Attention: operate over regions for each timepoint ---
        # (B, T, N, D) -> (B*T, N, D)
        x = x.reshape(B * T, N, D)
        res = x
        x = self.spatial_model(x, x, x)[0]
        x = x + res
        x = self.norm_1(x)

        # --- Swap T and N for temporal modeling ---
        # (B*T, N, D) -> (B, T, N, D)
        x = x.view(B, T, N, D)
        # Swap T and N: (B, T, N, D) -> (B, N, T, D)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (B, N, T, D) -> (B*N, T, D)
        x = x.view(B * N, T, D)

        res = x
        x = self.temporal_model(x)
        x = x + res
        x = self.norm_2(x)

        # --- Restore original axes ---
        # (B*N, T, D) -> (B, N, T, D)
        x = x.view(B, N, T, D)
        # Swap back: (B, N, T, D) -> (B, T, N, D)
        x = x.permute(0, 2, 1, 3).contiguous()

        # --- MLP block ---
        res = x
        x = self.mlp(x)
        x = x + res
        x = self.norm_3(x)
        # Output shape: (B, T, N, D)
        return x


class GBM(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, autoencoder_path=None, volume_size=(256, 128, 30), region_size=(32, 16, 2)):
        super(GBM, self).__init__()

        # asserting divisibility
        assert volume_size[0] % region_size[0] == 0 and volume_size[1] % region_size[1] == 0 and volume_size[2] % region_size[2] == 0, "volume_size must be divisible by region_size"
        # asserting that region_size is a factor of volume_size
        assert np.prod(volume_size) % np.prod(region_size) == 0, "region_size must be a factor of volume_size"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_regions = np.prod(volume_size) // np.prod(region_size)
        self.region_size = region_size
        self.volume_size = volume_size
        self.n_layers = n_layers
        self.n_blocks_x = volume_size[0] // region_size[0]
        self.n_blocks_y = volume_size[1] // region_size[1]
        self.n_blocks_z = volume_size[2] // region_size[2]
        self.layers = nn.ModuleList([SpatioTemporalRegionModel(d_model, n_heads, self.n_regions) for _ in range(n_layers)])

        # initialize the autoencoder which has frozen weights
        self.autoencoder = LinNormEncoder(
            input_size=np.prod(region_size), 
            hidden_size=d_model,
            volume_size=volume_size,
            region_size=region_size
        )

        if autoencoder_path is not None:
            self.autoencoder = LinNormEncoder(
                input_size=np.prod(region_size), 
                hidden_size=d_model,
                volume_size=volume_size,
                region_size=region_size
            )
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))

        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Takes as input sequences of shape (batch_size, seq_len, volume_size**)
        # Returns sequences of shape (batch_size, seq_len, volume_size**)
        B, T, *vol_size = x.shape

        assert vol_size == self.volume_size, "volume_size must match the input volume size"

        # reshape to (batch_size, seq_len, n_regions, d_model) where the whole volume is divided up into 3d regions
        # Spatially-aware reshaping: split the volume into regions (macro-blocks), preserving spatial locality.
        # x: (B, T, X, Y, Z) where (X, Y, Z) == volume_size

        # Unpack region and volume sizes
        region_x, region_y, region_z = self.region_size
        vol_x, vol_y, vol_z = self.volume_size


        # Encode the regions
        x = self.autoencoder.encode(x, apply_norm=True) # (B, T, hidden_channels, n_blocks_x, n_blocks_y, n_blocks_z)
        # Move hidden_channels to the end, then flatten spatial dims to n_regions
        x = x.permute(0, 1, 3, 4, 5, 2).reshape(B, T, self.n_regions, self.d_model) # (B, T, n_regions, d_model)

        for layer in self.layers:
            x = layer(x) # (B, T, n_regions, d_model)

        # Reshape n_regions back to (n_blocks_x, n_blocks_y, n_blocks_z, d_model)
        x = x.view(B, T, self.n_blocks_x, self.n_blocks_y, self.n_blocks_z, self.d_model)  # (B, T, n_blocks_x, n_blocks_y, n_blocks_z, d_model)
        # Move d_model to channel position for decoder: (B, T, d_model, n_blocks_x, n_blocks_y, n_blocks_z)
        x = x.permute(0, 1, 5, 2, 3, 4)

        # Decode the regions
        x = self.autoencoder.decode(x, get_logits=True, apply_norm=False)

        # Reshape back to original volume shape: (B, T, X, Y, Z)
        x = x.reshape(B, T, vol_x, vol_y, vol_z)

        return x

# During training loss of each GBM is indepedent, during inference we average the output logits of all GBMs 


class EnsembleGBM(nn.Module):
    """
    Ensemble of GBM models that trains with independent losses but averages logits during inference.
    
    During training (self.training=True):
        - Returns list of outputs from each GBM for independent loss computation
        - Each GBM can have different gradients and learn different aspects
        
    During inference (self.training=False): 
        - Returns averaged logits from all GBMs for better predictions
        - Ensemble averaging typically improves generalization
    """
    
    def __init__(self, n_models, d_model, n_heads, n_layers, autoencoder_path=None, 
                 volume_size=(256, 128, 30), region_size=(32, 16, 2), 
                 different_seeds=True):
        """
        Args:
            n_models: Number of GBM models in the ensemble
            d_model: Hidden dimension for each GBM
            n_heads: Number of attention heads for each GBM
            n_layers: Number of layers for each GBM
            autoencoder_path: Path to pretrained autoencoder (shared across all models)
            volume_size: 3D volume dimensions
            region_size: 3D region dimensions  
            different_seeds: If True, initialize each model with different random seed
        """
        super(EnsembleGBM, self).__init__()
        
        self.n_models = n_models
        self.d_model = d_model
        self.volume_size = volume_size
        self.region_size = region_size
        
        # Create ensemble of GBM models
        self.models = nn.ModuleList()
        
        for i in range(n_models):
            # Set different random seed for each model if requested
            if different_seeds:
                torch.manual_seed(42 + i)  # Different seed for each model
            
            model = GBM(
                d_model=d_model,
                n_heads=n_heads, 
                n_layers=n_layers,
                autoencoder_path=autoencoder_path,
                volume_size=volume_size,
                region_size=region_size
            )
            
            self.models.append(model)
        
        # Reset random seed to original state
        if different_seeds:
            torch.manual_seed(torch.initial_seed())
    
    def forward(self, x):
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, *volume_size)
            
        Returns:
            If training: List of outputs [output_1, output_2, ..., output_n] for independent losses
            If inference: Single averaged output tensor for final prediction
        """
        outputs = []
        
        # Get output from each model
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        if self.training:
            # During training: return list of independent outputs
            # This allows computing independent losses for each model
            return outputs
        else:
            # During inference: return averaged logits
            # Stack outputs and compute mean across models
            stacked_outputs = torch.stack(outputs, dim=0)  # (n_models, batch_size, seq_len, *volume_size)
            averaged_output = torch.mean(stacked_outputs, dim=0)  # (batch_size, seq_len, *volume_size)
            return averaged_output
    
    def train_forward(self, x):
        """Explicitly get training outputs (list of independent model outputs)."""
        was_training = self.training
        self.train()
        outputs = self.forward(x)
        self.train(was_training)
        return outputs
    
    def eval_forward(self, x):
        """Explicitly get inference output (averaged logits).""" 
        was_training = self.training
        self.eval()
        output = self.forward(x)
        self.train(was_training)
        return output
    
    def get_model_outputs(self, x):
        """Get individual outputs from each model (regardless of training mode)."""
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        return outputs
    
    def freeze_autoencoders(self):
        """Freeze autoencoder weights in all models (if not already frozen)."""
        for model in self.models:
            for param in model.autoencoder.parameters():
                param.requires_grad = False
    
    def unfreeze_autoencoders(self):
        """Unfreeze autoencoder weights in all models."""
        for model in self.models:
            for param in model.autoencoder.parameters():
                param.requires_grad = True
    
    def load_ensemble_weights(self, checkpoint_paths):
        """
        Load weights for each model from separate checkpoints.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints (length must match n_models)
        """
        if len(checkpoint_paths) != self.n_models:
            raise ValueError(f"Expected {self.n_models} checkpoint paths, got {len(checkpoint_paths)}")
        
        for i, (model, path) in enumerate(zip(self.models, checkpoint_paths)):
            if path is not None:
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print(f"Loaded weights for model {i} from {path}")
    
    def save_ensemble_weights(self, checkpoint_dir, prefix="model"):
        """
        Save weights for each model to separate checkpoints.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            prefix: Prefix for checkpoint filenames
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        saved_paths = []
        for i, model in enumerate(self.models):
            path = os.path.join(checkpoint_dir, f"{prefix}_{i}.pth")
            torch.save(model.state_dict(), path)
            saved_paths.append(path)
            print(f"Saved model {i} weights to {path}")
        
        return saved_paths 


    