import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from BrainSimulator.custom_functions.utils import RMSNorm, twohot_exp_loss, logits_to_value
from BrainSimulator.models.mlp import MLP
class MAB(nn.Module):
    """Multi-head Attention Block."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = RMSNorm(dim_V)
            self.ln1 = RMSNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/np.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.gelu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    """Set Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    """Induced Set Attention Block."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H) + self.proj(X)

class PMA(nn.Module):
    """Pooling by Multi-head Attention."""
    def __init__(self, dim, pooled_dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, pooled_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(pooled_dim, dim, pooled_dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SpikeSetTransformerVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=32, num_heads=4, num_inds=32, num_seeds=1):
        super(SpikeSetTransformerVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder components
        self.encoder_layers = nn.Sequential(
            ISAB(input_dim, self.hidden_dim, num_heads, num_inds, ln=True),
            ISAB(self.hidden_dim, self.hidden_dim, num_heads, num_inds, ln=True),
            ISAB(self.hidden_dim, self.hidden_dim, num_heads, num_inds, ln=True),
        )
        
        # Project concatenated point+count embedding back to hidden_dim
        self.encoder_proj = MLP(2, self.hidden_dim+1, self.hidden_dim, self.hidden_dim)
        
        self.pool = PMA(self.hidden_dim, latent_dim, num_heads, num_seeds, ln=True)
        
        # Project to mean and log variance
        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)
        
        # Position embedding MLP
        self.pos_embed = MLP(1, 1, hidden_dim, latent_dim)
        
        # Decoder components
        self.decoder_transformer = nn.Sequential(
            ISAB(latent_dim, self.hidden_dim, num_heads, num_inds, ln=True),
            ISAB(self.hidden_dim, self.hidden_dim, num_heads, num_inds, ln=True),
            ISAB(self.hidden_dim, self.hidden_dim, num_heads, num_inds, ln=True),
        )
        self.to_coordinates = nn.Linear(self.hidden_dim, 2)  # Output 2D coordinates

    def encode(self, x, num_points):
        """Encode input points to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, num_points, 2) containing 2D coordinates
            num_points: Tensor of shape (batch_size,) containing number of points per sample
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        
        # Process through encoder layers
        h = self.encoder_layers(x)  # (batch_size, num_points, hidden_dim)
        
        # Concatenate log(num_points) to each point embedding
        log_points = torch.log(num_points + 1e-8).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
        log_points = log_points.expand(-1, h.size(1), 1)  # (batch_size, num_points, 1)
        h = torch.cat([h, log_points], dim=2)  # (batch_size, num_points, hidden_dim + 1)
        
        # Project back to hidden_dim
        h = self.encoder_proj(h)  # (batch_size, num_points, hidden_dim)
        
        # Pool and get latent distribution
        h = self.pool(h)  # (batch_size, 1, hidden_dim)
        h = h.view(batch_size, -1)  # (batch_size, hidden_dim)
        
        # Get mean and log variance
        mu = h # self.to_mu(h)  # (batch_size, latent_dim)
        logvar = self.to_logvar(h)  # (batch_size, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, points_per_sample):
        """Decode latent representation to point set.
        
        Args:
            z: (batch_size, latent_dim) latent vectors
            points_per_sample: (batch_size,) tensor of number of points to generate per sample
        
        Returns:
            points: (batch_size, max_points, 2) tensor of 2D coordinates with sentinel value -1 for padding
        """
        batch_size = z.size(0)
        device = z.device
        
        # Project latent to hidden dimension
        z_proj = z  # (batch_size, latent_dim)
        
        # Get maximum number of points to generate
        max_points = points_per_sample.max().item()
        
        # Create mask for valid points
        point_indices = torch.arange(max_points, device=device)
        mask = point_indices.unsqueeze(0) < points_per_sample.unsqueeze(1)  # (batch_size, max_points)
        
        # Create position indices and normalize them
        pos_indices = point_indices.float() / (max_points - 1)  # Normalize to [0, 1]
        pos_indices = pos_indices.unsqueeze(-1)  # (max_points, 1)
        
        # Generate position embeddings
        pos_embeddings = self.pos_embed(pos_indices)  # (max_points, latent_dim)
        
        # Expand z_proj and position embeddings
        z_tiled = z_proj.unsqueeze(1).expand(-1, max_points, -1)  # (batch_size, max_points, latent_dim)
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, max_points, latent_dim)
        
        # Add position embeddings to tiled latents
        decoder_input = z_tiled + pos_embeddings * mask.unsqueeze(-1)
        
        # Process through decoder
        decoded = self.decoder_transformer(decoder_input)
        points = torch.sigmoid(self.to_coordinates(decoded))  # Output 2D coordinates
        
        # Set invalid points to sentinel value -1
        points = torch.where(mask.unsqueeze(-1), points, torch.full_like(points, -1.0))
        
        return points
    
    def compute_kl_loss(self, mu, logvar):
        """Compute KL divergence loss."""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def forward(self, x):
        """Forward pass through the model."""
        # Get actual number of points per sample (non-zero points)
        actual_points = torch.sum(torch.any(x != -1, dim=-1), dim=1)
        
        # Encode input to latent space
        mu, logvar = self.encode(x, actual_points)
        z = mu#self.reparameterize(mu, logvar)
        
        # Decode using actual number of points
        points = self.decode(z, actual_points)
        
        return points, (mu, logvar) 