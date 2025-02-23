import torch
import numpy as np
import torch.nn.functional as F

def compute_1d_wasserstein(sorted_x, sorted_y):
    """Compute 1D Wasserstein distance between sorted arrays."""
    return torch.abs(sorted_x - sorted_y).mean()

def project_points(points, theta):
    """Project points onto direction theta."""
    return torch.matmul(points, theta)

def compute_1d_wasserstein_batch(sorted_x, sorted_y):
    """Compute 1D Wasserstein distance between sorted arrays in batch.
    
    Args:
        sorted_x: Tensor of shape (B, T, P) where B is batch size, T is number of quantile points, 
                 P is number of projections
        sorted_y: Tensor of same shape as sorted_x
    """
    # Compute mean absolute difference along quantile dimension
    return torch.abs(sorted_x - sorted_y).mean(dim=1)  # Returns (B, P)

def interpolate_quantile_function(sorted_values, counts, num_quantile_points=100):
    """Interpolate sorted values to fixed quantile grid using grid_sample.
    
    Args:
        sorted_values: Tensor of shape (B, N, P) where B is batch size, N is max points, P is projections
        counts: Tensor of shape (B,) containing number of valid points per sample
        num_quantile_points: Number of points to interpolate to on quantile grid
    
    Returns:
        Interpolated values of shape (B, T, P) where T is num_quantile_points
    """
    batch_size, max_points, num_projections = sorted_values.shape
    device = sorted_values.device
    
    # Create uniform quantile grid [0, 1]
    r = torch.linspace(0, 1, num_quantile_points, device=device)  # (T,)
    
    # Compute start indices for valid regions
    x_start = max_points - counts  # (B,)
    
    # For each sample, map quantile r to absolute position in valid region
    # Expand dimensions for broadcasting
    x_start = x_start.view(batch_size, 1)  # (B, 1)
    counts = counts.view(batch_size, 1)  # (B, 1)
    r = r.view(1, num_quantile_points)  # (1, T)
    
    # Compute absolute indices: x_start + (counts-1) * r
    indices = x_start + (counts - 1) * r  # (B, T)
    
    # Convert to normalized coordinates in [-1, 1] for grid_sample
    normalized_indices = (indices / (max_points - 1)) * 2 - 1  # (B, T)
    
    # Prepare grid for grid_sample: (B, H, W, 2) where H=num_quantile_points, W=1
    grid = normalized_indices.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1)
    grid = torch.cat([torch.zeros_like(grid), grid], dim=3)  # (B, T, 1, 2)
    
    # Process each projection separately to maintain proper dimensions
    interpolated_list = []
    for p in range(num_projections):
        # Extract single projection and reshape for grid_sample: (B, C=1, H, W)
        proj_values = sorted_values[:, :, p].unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1)
        
        # Interpolate using grid_sample
        proj_interp = F.grid_sample(
            proj_values, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, 1, T, 1)
        
        interpolated_list.append(proj_interp.squeeze(3).squeeze(1))  # (B, T)
    
    # Stack all interpolated projections
    interpolated = torch.stack(interpolated_list, dim=2)  # (B, T, P)
    
    return interpolated

class AdaptiveSlicedWasserstein:
    def __init__(self, N0=10, s=5, epsilon=0.01, max_projections=100, num_quantile_points=100):
        """
        Initialize ASW with parameters.
        
        Args:
            N0: Initial number of projections
            s: Number of additional projections per iteration
            epsilon: Error tolerance
            max_projections: Maximum number of projections allowed
            num_quantile_points: Number of points to use in quantile grid
        """
        self.N0 = N0
        self.s = s
        self.epsilon = epsilon
        self.max_projections = max_projections
        self.num_quantile_points = num_quantile_points
        
    def __call__(self, x, y, p=1):
        """
        Compute ASW distance between two point sets.
        
        Args:
            x: First point set of shape (B, N1, D)
            y: Second point set of shape (B, N2, D)
            p: Order of the Wasserstein distance
            
        Returns:
            ASW distance between x and y
        """
        device = x.device
        batch_size = x.size(0)
        dim = x.size(-1)
        
        # Create validity masks for x and y
        valid_x = torch.any(x != 0, dim=-1)  # (B, N1)
        valid_y = torch.any(y != 0, dim=-1)  # (B, N2)
        
        # Count valid points per sample
        counts_x = valid_x.sum(dim=1)  # (B,)
        counts_y = valid_y.sum(dim=1)  # (B,)
        
        # Initialize statistics
        N = self.N0
        total_sw = None
        total_sw_squared = None
        
        while N <= self.max_projections:
            # Generate random projections
            if N == self.N0:
                num_new = self.N0
            else:
                num_new = self.s
                
            # Sample random directions on unit sphere
            theta = torch.randn(num_new, dim, device=device)
            theta = theta / torch.norm(theta, dim=1, keepdim=True)  # (num_new, dim)
            
            # Project points for all samples in parallel
            x_proj = torch.matmul(x, theta.T)  # (B, N1, num_new)
            y_proj = torch.matmul(y, theta.T)  # (B, N2, num_new)
            
            # Set invalid points to large negative value for consistent sorting
            x_proj = torch.where(valid_x.unsqueeze(-1), x_proj, torch.tensor(-1e9, device=device))
            y_proj = torch.where(valid_y.unsqueeze(-1), y_proj, torch.tensor(-1e9, device=device))
            
            # Sort projected points
            x_sorted, _ = torch.sort(x_proj, dim=1)  # (B, N1, num_new)
            y_sorted, _ = torch.sort(y_proj, dim=1)  # (B, N2, num_new)
            
            # Interpolate quantile functions to fixed grid
            x_interp = interpolate_quantile_function(x_sorted, counts_x, self.num_quantile_points)
            y_interp = interpolate_quantile_function(y_sorted, counts_y, self.num_quantile_points)
            
            # Compute Wasserstein distances for all projections
            sw_values = compute_1d_wasserstein_batch(x_interp, y_interp)  # (B, num_new)
            
            # Update running statistics
            sw_mean = sw_values.mean(dim=1)  # (B,)
            sw_squared_mean = (sw_values ** 2).mean(dim=1)  # (B,)
            
            if total_sw is None:
                total_sw = sw_mean
                total_sw_squared = sw_squared_mean
            else:
                # Ensure consistent batch size for statistics update
                if sw_mean.size(0) != total_sw.size(0):
                    # Adjust statistics to match current batch size
                    if sw_mean.size(0) < total_sw.size(0):
                        sw_mean = sw_mean.expand(total_sw.size(0))
                        sw_squared_mean = sw_squared_mean.expand(total_sw.size(0))
                    else:
                        total_sw = total_sw.expand(sw_mean.size(0))
                        total_sw_squared = total_sw_squared.expand(sw_mean.size(0))
                
                total_sw = (N * total_sw + num_new * sw_mean) / (N + num_new)
                total_sw_squared = (N * total_sw_squared + num_new * sw_squared_mean) / (N + num_new)
            
            # Check stopping condition
            variance = total_sw_squared - total_sw ** 2
            threshold = ((N - 1) * self.epsilon ** 2) / 4
            if torch.all(variance <= threshold):
                break
                
            N += num_new
            
        # Return mean distance across batch
        return (total_sw ** (1/p)).mean() 