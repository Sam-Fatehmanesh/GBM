import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SequentialSpikeDataset(Dataset):
    def __init__(self, h5_file, seq_len=30, split='train', train_ratio=0.95):
        """Dataset for sequential spike data, ordering frames by time and z-height.
        For each timepoint, sequences cycle through all z-planes from top to bottom.
        If the subject has fewer z-planes than the sequence length requires,
        the sequence will be padded with zeros.
        
        Args:
            h5_file: Path to the processed spike data h5 file
            seq_len: Length of sequences to return (must be divisible by max_z_planes)
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
        
        # Calculate required timesteps based on sequence length and number of z-planes
        self.timesteps_per_sequence = seq_len // self.num_z
        if self.timesteps_per_sequence < 1:
            raise ValueError(f"Sequence length {seq_len} too short for {self.num_z} z-planes")
        
        # Store actual sequence length (may be longer than requested if padding is needed)
        self.seq_len = seq_len
        
        # Pre-compute z-plane masks and cell indices
        self.z_masks = {}
        self.z_cell_indices = {}  # Store pre-computed cell indices for each z-plane
        cells_per_z = {}
        
        # Get x,y ranges for proper normalization
        x_min, y_min = self.cell_positions[:, :2].min(axis=0)
        x_max, y_max = self.cell_positions[:, :2].max(axis=0)
        
        for z_idx, z_level in enumerate(self.z_values):
            # Create mask for this z-plane
            z_mask = (np.round(self.cell_positions[:, 2], decimals=3) == z_level)
            self.z_masks[z_idx] = z_mask
            cells_per_z[z_level] = np.sum(z_mask)
            
            # Get positions for cells in this z-plane
            cell_positions_z = self.cell_positions[z_mask]
            
            # Normalize x,y positions to [0,1] using actual min/max
            cell_positions_norm = np.zeros_like(cell_positions_z[:, :2])
            cell_positions_norm[:, 0] = (cell_positions_z[:, 0] - x_min) / (x_max - x_min)
            cell_positions_norm[:, 1] = (cell_positions_z[:, 1] - y_min) / (y_max - y_min)
            
            # Convert to grid indices with bounds checking
            cell_x = np.clip(np.floor(cell_positions_norm[:, 0] * 255).astype(np.int32), 0, 255)
            cell_y = np.clip(np.floor(cell_positions_norm[:, 1] * 127).astype(np.int32), 0, 127)
            
            # Store indices
            self.z_cell_indices[z_idx] = (cell_x, cell_y)
            
            # Verify indices are within bounds
            assert cell_x.max() < 256 and cell_x.min() >= 0, f"X indices out of bounds: {cell_x.min()}, {cell_x.max()}"
            assert cell_y.max() < 128 and cell_y.min() >= 0, f"Y indices out of bounds: {cell_y.min()}, {cell_y.max()}"
        
        # Create valid starting timepoints
        # Each sequence needs timesteps_per_sequence consecutive timepoints
        max_start = self.num_timepoints - self.timesteps_per_sequence
        if max_start < 0:
            raise ValueError(f"Sequence length too long for dataset. "
                           f"Max possible length: {self.num_timepoints * self.num_z}")
        valid_starts = list(range(max_start + 1))
        
        # Split into train/test
        np.random.seed(42)
        np.random.shuffle(valid_starts)
        split_idx = int(len(valid_starts) * train_ratio)
        
        if split == 'train':
            self.valid_starts = valid_starts[:split_idx]
        else:
            self.valid_starts = valid_starts[split_idx:]
            
        print(f"\nSequential Dataset {h5_file}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Z-plane ordering: {self.z_values}")
        print(f"Cells per z-plane: {[cells_per_z[z] for z in self.z_values]}")
        print(f"Sequence length: {seq_len} ({self.timesteps_per_sequence} timepoints × {self.num_z} z-planes)")
        print(f"Total sequences ({split}): {len(self.valid_starts)}")
        print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """Get a sequence of binary grids.
        
        Returns:
            sequence: Tensor of shape (seq_len, 256, 128) containing binary grids
            Each subsequence of num_z frames contains all z-planes for a single timepoint
            If this subject has fewer z-planes than required, the sequence will be padded
            with zeros to match the required length.
        """
        t_start = self.valid_starts[idx]
        
        # Pre-allocate the full sequence array (will be padded with zeros if needed)
        sequence = np.zeros((self.seq_len, 256, 128), dtype=np.float32)
        
        # For each timepoint in the sequence
        for t_idx in range(self.timesteps_per_sequence):
            t = t_start + t_idx
            
            # For each z-plane (top to bottom)
            for z_idx in range(self.num_z):
                # Calculate sequence index
                seq_idx = t_idx * self.num_z + z_idx
                
                # Skip if we've reached the required sequence length
                if seq_idx >= self.seq_len:
                    break
                
                # Get pre-computed cell indices for this z-plane
                cell_x_z, cell_y_z = self.z_cell_indices[z_idx]
                
                # Get spikes for this timepoint and z-plane
                spikes_t = self.spikes[t][self.z_masks[z_idx]]
                
                # Set active cells to 1 in the grid
                active_cells = np.abs(spikes_t) > 1e-6
                sequence[seq_idx, cell_x_z[active_cells], cell_y_z[active_cells]] = 1.0
        
        return torch.from_numpy(sequence).contiguous() 