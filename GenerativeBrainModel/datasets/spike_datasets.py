import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class GridSpikeDataset(Dataset):
    def __init__(self, h5_file, split='train', train_ratio=0.95):
        """Dataset for spike data, converting each z-plane at each timepoint to a 256x128 binary grid.
        
        Args:
            h5_file: Path to the processed spike data h5 file
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training
        """
        # Load spike data and cell positions
        with h5py.File(h5_file, 'r') as f:
            self.spikes = f['spikes'][:]  # shape: (n_timepoints, n_cells)
            self.cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
            
        # Get unique z values (rounded to handle floating point precision)
        self.z_values = np.unique(np.round(self.cell_positions[:, 2], decimals=3))
        self.num_z = len(self.z_values)
        self.num_timepoints = self.spikes.shape[0]
        
        # Pre-compute z-plane masks
        self.z_masks = {}
        cells_per_z = {}
        for z_idx, z_level in enumerate(self.z_values):
            z_mask = (np.round(self.cell_positions[:, 2], decimals=3) == z_level)
            self.z_masks[z_idx] = z_mask
            cells_per_z[z_level] = np.sum(z_mask)
            
        # Normalize cell positions to [0, 1]
        self.cell_positions = (self.cell_positions - self.cell_positions.min(axis=0)) / \
                            (self.cell_positions.max(axis=0) - self.cell_positions.min(axis=0))
        
        # Convert to grid indices
        self.cell_x = np.floor(self.cell_positions[:, 0] * 255).astype(np.int32)  # 0-255
        self.cell_y = np.floor(self.cell_positions[:, 1] * 127).astype(np.int32)  # 0-127
        
        # Create indices for all possible (timepoint, z) combinations
        all_indices = [(t, z) for t in range(self.num_timepoints) 
                      for z in range(self.num_z)]
        
        # Split into train/test
        np.random.seed(42)
        np.random.shuffle(all_indices)
        split_idx = int(len(all_indices) * train_ratio)
        
        if split == 'train':
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]
            
        print(f"\nDataset {h5_file}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Cells per z-plane: {[cells_per_z[z] for z in self.z_values]}")
        print(f"Total samples ({split}): {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Returns:
            grid: Binary tensor of shape (256, 128) representing active cells in the z-plane
        """
        # Get timepoint and z-level from indices
        timepoint, z_idx = self.indices[idx]
        z_level = self.z_values[z_idx]
        
        # Get mask for cells in this z-plane
        z_mask = self.z_masks[z_idx]
        
        # Get spikes for this timepoint
        spikes_t = self.spikes[timepoint]
        
        # Create empty grid
        grid = np.zeros((256, 128), dtype=np.float32)
        
        # Consider a cell active if it has any non-zero spike value
        active_mask = (np.abs(spikes_t) > 1e-6) & z_mask
        
        # Set active cells to 1 in the grid
        grid[self.cell_x[active_mask], self.cell_y[active_mask]] = 1.0
        
        return torch.FloatTensor(grid)

class SyntheticSpikeDataset(Dataset):
    def __init__(self, num_samples=10000, grid_size=(256, 128)):
        """Dataset that generates synthetic spike data following the empirical distribution.
        
        Args:
            num_samples: Number of synthetic samples to generate
            grid_size: Size of the binary grid (height, width)
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        # Distribution parameters from empirical analysis
        self.mean_spikes = 73.42
        self.std_spikes = 98.25
        self.min_spikes = 0
        self.max_spikes = 2000  # Using 99th percentile instead of max to avoid outliers
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random binary grid with number of spikes following empirical distribution."""
        # Sample number of spikes from truncated normal distribution
        num_spikes = int(np.clip(
            np.random.normal(self.mean_spikes, self.std_spikes),
            self.min_spikes,
            self.max_spikes
        ))
        
        # Create empty grid
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        if num_spikes > 0:
            # Randomly select positions for spikes
            total_cells = self.grid_size[0] * self.grid_size[1]
            spike_indices = np.random.choice(total_cells, size=num_spikes, replace=False)
            
            # Convert to 2D indices
            spike_rows = spike_indices // self.grid_size[1]
            spike_cols = spike_indices % self.grid_size[1]
            
            # Set spikes
            grid[spike_rows, spike_cols] = 1.0
        
        return torch.FloatTensor(grid) 