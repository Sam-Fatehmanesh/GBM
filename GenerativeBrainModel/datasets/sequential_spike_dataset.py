import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

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
        # Store the file path instead of keeping the file open
        self.h5_file_path = h5_file
        
        # Open the file temporarily to get metadata
        with h5py.File(h5_file, 'r', libver='latest', swmr=True) as f:
            # Get shape information
            self.num_timepoints = f['spikes'].shape[0]
            
            # Load cell positions (small enough to keep in memory)
            self.cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
            
            # Get unique z values (rounded to handle floating point precision)
            z_values = np.unique(np.round(self.cell_positions[:, 2], decimals=3))
            self.z_values = np.sort(z_values)[::-1]  # Sort descending (higher z first)
            self.num_z = len(self.z_values)
        
        # Store actual sequence length
        self.seq_len = seq_len
        
        # Pre-compute z-plane masks and cell indices for each z-plane
        self.z_masks = {}
        self.z_cell_indices = {}  # Store pre-computed indices for faster access
        cells_per_z = {}
        
        # Normalize cell positions to [0, 1]
        normalized_positions = (self.cell_positions - self.cell_positions.min(axis=0)) / \
                            (self.cell_positions.max(axis=0) - self.cell_positions.min(axis=0))
        
        # Convert to grid indices
        self.cell_x = np.floor(normalized_positions[:, 0] * 255).astype(np.int32)  # 0-255
        self.cell_y = np.floor(normalized_positions[:, 1] * 127).astype(np.int32)  # 0-127
        
        # Pre-compute cell indices for each z-plane to avoid repeated computations
        for z_idx, z_level in enumerate(self.z_values):
            # Create mask for cells in this z-plane
            z_mask = (np.round(self.cell_positions[:, 2], decimals=3) == z_level)
            self.z_masks[z_idx] = z_mask
            cells_per_z[z_level] = np.sum(z_mask)
            
            # Store cell indices for this z-plane
            self.z_cell_indices[z_idx] = {
                'x': self.cell_x[z_mask],
                'y': self.cell_y[z_mask],
                'indices': np.where(z_mask)[0]  # Store the actual indices for faster lookup
            }
        
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
            
        print(f"\nSequential Dataset {os.path.basename(h5_file)}:")
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
            Each subsequence of num_z frames contains all z-planes for a single timepoint
            Frames are ordered by timepoint (outer) and z-plane (inner, high to low)
        """
        # Get starting timepoint and z-plane from valid_starts
        t_start, z_start = self.valid_starts[idx]
        
        # Pre-allocate the full sequence array (using float16 to reduce memory usage)
        sequence = np.zeros((self.seq_len, 256, 128), dtype=np.float16)
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            spikes = f['spikes']  # Reference to the dataset
            
            # Fill the sequence with frames
            seq_idx = 0
            t = t_start
            
            while seq_idx < self.seq_len:
                # For each z-plane starting from z_start
                for z_idx in range(self.num_z):
                    # Calculate actual z-plane index with wrapping
                    actual_z = (z_start + z_idx) % self.num_z
                    
                    # Skip if we've reached the required sequence length
                    if seq_idx >= self.seq_len:
                        break
                    
                    # Get pre-computed cell indices for this z-plane
                    cell_indices = self.z_cell_indices[actual_z]['indices']
                    
                    # Skip if no cells in this z-plane
                    if len(cell_indices) == 0:
                        seq_idx += 1
                        continue
                    
                    # Get spikes for this timepoint and z-plane directly from HDF5 file
                    # This reads only the needed data from disk without loading everything
                    spikes_t = spikes[t][cell_indices]
                    
                    # Set active cells to 1 in the grid
                    active_cells = np.abs(spikes_t) > 1e-6
                    if np.any(active_cells):  # Only process if there are active cells
                        active_x = self.z_cell_indices[actual_z]['x'][active_cells]
                        active_y = self.z_cell_indices[actual_z]['y'][active_cells]
                        sequence[seq_idx, active_x, active_y] = 1.0
                    
                    seq_idx += 1
                
                # Move to next timepoint
                t += 1
                
                # Break if we've reached the end of available timepoints
                if t >= self.num_timepoints:
                    break
        
        # Convert to torch tensor with contiguous memory layout
        # Convert to float32 for model compatibility
        return torch.from_numpy(sequence).float().contiguous() 