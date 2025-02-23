import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SequentialSpikeDataset(Dataset):
    def __init__(self, h5_file, seq_len=30, split='train', train_ratio=0.95):
        """Dataset for sequential spike data, ordering frames by time and z-height.
        Higher z-planes and earlier timepoints come first in the sequence.
        
        Args:
            h5_file: Path to the processed spike data h5 file
            seq_len: Length of sequences to return
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training
        """
        # Load spike data and cell positions
        with h5py.File(h5_file, 'r') as f:
            self.spikes = f['spikes'][:]  # shape: (n_timepoints, n_cells)
            self.cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
            
        # Get unique z values (rounded to handle floating point precision)
        self.z_values = np.unique(np.round(self.cell_positions[:, 2], decimals=3))
        self.z_values = np.sort(self.z_values)[::-1]  # Sort descending (higher z first)
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
        
        # Create all possible sequence starting points
        # Each sequence needs seq_len consecutive timepoints
        valid_starts = []
        for t in range(self.num_timepoints - seq_len + 1):
            valid_starts.extend([(t, z) for z in range(self.num_z)])
        
        # Split into train/test
        np.random.seed(42)
        np.random.shuffle(valid_starts)
        split_idx = int(len(valid_starts) * train_ratio)
        
        if split == 'train':
            self.valid_starts = valid_starts[:split_idx]
        else:
            self.valid_starts = valid_starts[split_idx:]
            
        self.seq_len = seq_len
        
        print(f"\nSequential Dataset {h5_file}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Z-plane ordering: {self.z_values}")
        print(f"Cells per z-plane: {[cells_per_z[z] for z in self.z_values]}")
        print(f"Sequence length: {seq_len}")
        print(f"Total sequences ({split}): {len(self.valid_starts)}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """Get a sequence of binary grids.
        
        Returns:
            sequence: Tensor of shape (seq_len, 256, 128) containing binary grids
        """
        t_start, z_start = self.valid_starts[idx]
        
        # Get sequence of timepoints
        sequence = []
        for t in range(t_start, t_start + self.seq_len):
            # Create empty grid
            grid = np.zeros((256, 128), dtype=np.float32)
            
            # Get spikes for this timepoint
            spikes_t = self.spikes[t]
            
            # Get mask for cells in this z-plane
            z_mask = self.z_masks[z_start]
            
            # Consider a cell active if it has any non-zero spike value
            active_mask = (np.abs(spikes_t) > 1e-6) & z_mask
            
            # Set active cells to 1 in the grid
            grid[self.cell_x[active_mask], self.cell_y[active_mask]] = 1.0
            sequence.append(grid)
            
        return torch.FloatTensor(sequence) 