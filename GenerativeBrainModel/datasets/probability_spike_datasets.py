import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class SequentialSpikeProbabilityDataset(Dataset):
    def __init__(self, h5_file, seq_len=30, split='train', train_ratio=0.95, use_probabilities=True):
        """Dataset for sequential spike data that can return probabilities or binary spikes as targets.
        
        Args:
            h5_file: Path to the processed grid data h5 file (with both spike_grids and probability_grids)
            seq_len: Length of sequences to return
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training (ignored if is_train mask exists)
            use_probabilities: If True, use spike probabilities as targets. If False, use binary spikes.
        """
        self.h5_file_path = h5_file
        self.seq_len = seq_len
        self.use_probabilities = use_probabilities
        
        # Open the file temporarily to get metadata
        with h5py.File(h5_file, 'r', libver='latest', swmr=True) as f:
            # Check what datasets are available
            available_keys = list(f.keys())
            self.has_probabilities = 'probability_grids' in available_keys
            
            if use_probabilities and not self.has_probabilities:
                print(f"Warning: Probability grids not found in {h5_file}, falling back to binary spikes")
                self.use_probabilities = False
            
            # Get shape information
            if 'spike_grids' in available_keys:
                self.num_timepoints, self.num_z, self.height, self.width = f['spike_grids'].shape
            else:
                # Fallback to old format
                self.num_timepoints, self.num_z, self.height, self.width = f['grids'].shape
                self.has_probabilities = False
                self.use_probabilities = False
            
            # Load train/test split if available
            if 'is_train' in available_keys:
                is_train = f['is_train'][:]
                train_indices = np.where(is_train == 1)[0]
                test_indices = np.where(is_train == 0)[0]
            else:
                # Create split manually
                np.random.seed(42)
                indices = np.arange(self.num_timepoints)
                np.random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio)
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
        
        # Create all possible sequence starting points
        if split == 'train':
            valid_timepoints = train_indices
        else:
            valid_timepoints = test_indices
        
        # Filter timepoints that can start a full sequence
        self.valid_starts = []
        for t in valid_timepoints:
            if t + self.seq_len <= self.num_timepoints:
                # Allow sequences to start at any z-plane
                for z in range(self.num_z):
                    self.valid_starts.append((t, z))
        
        print(f"\nProbability Sequential Dataset {os.path.basename(h5_file)}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Grid size: {self.height}x{self.width}")
        print(f"Sequence length: {seq_len}")
        print(f"Total sequences ({split}): {len(self.valid_starts)}")
        print(f"Using probabilities as targets: {self.use_probabilities}")
        print(f"Has probability grids: {self.has_probabilities}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        """Get a sequence from the dataset.
        
        Returns:
            sequence: Tensor of shape (seq_len, height, width) representing spike data
                     Values are binary (0/1) if use_probabilities=False, or probabilities [0,max] if True
        """
        # Get starting timepoint and z-plane from valid_starts
        t_start, z_start = self.valid_starts[idx]
        
        # Create sequence array
        sequence = np.zeros((self.seq_len, self.height, self.width), dtype=np.float32)
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Choose data source based on configuration
            if self.use_probabilities and self.has_probabilities:
                grids = f['probability_grids']
            elif 'spike_grids' in f:
                grids = f['spike_grids']
            else:
                # Fallback to old format
                grids = f['grids']
            
            # Fill the sequence with frames
            seq_idx = 0
            t = t_start
            current_z = z_start
            
            while seq_idx < self.seq_len and t < self.num_timepoints:
                # Process the remaining z-planes in the current timepoint
                while current_z < self.num_z and seq_idx < self.seq_len:
                    # Get the grid for this timepoint and z-plane
                    grid = grids[t, current_z]
                    sequence[seq_idx] = grid
                    
                    seq_idx += 1
                    current_z += 1
                
                # Move to next timepoint
                t += 1
                
                # For all subsequent timepoints, always start at z=0
                current_z = 0
        
        # Convert to torch tensor with contiguous memory layout
        return torch.from_numpy(sequence).float().contiguous()


class GridSpikeProbabilityDataset(Dataset):
    def __init__(self, h5_file, split='train', train_ratio=0.95, use_probabilities=True):
        """Dataset for individual grid frames that can return probabilities or binary spikes.
        
        Args:
            h5_file: Path to the processed grid data h5 file
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training (ignored if is_train mask exists)
            use_probabilities: If True, use spike probabilities as targets. If False, use binary spikes.
        """
        self.h5_file_path = h5_file
        self.use_probabilities = use_probabilities
        
        # Open the file temporarily to get metadata
        with h5py.File(h5_file, 'r', libver='latest', swmr=True) as f:
            # Check what datasets are available
            available_keys = list(f.keys())
            self.has_probabilities = 'probability_grids' in available_keys
            
            if use_probabilities and not self.has_probabilities:
                print(f"Warning: Probability grids not found in {h5_file}, falling back to binary spikes")
                self.use_probabilities = False
            
            # Get shape information
            if 'spike_grids' in available_keys:
                self.num_timepoints, self.num_z, self.height, self.width = f['spike_grids'].shape
            else:
                # Fallback to old format
                self.num_timepoints, self.num_z, self.height, self.width = f['grids'].shape
                self.has_probabilities = False
                self.use_probabilities = False
            
            # Load train/test split if available
            if 'is_train' in available_keys:
                is_train = f['is_train'][:]
                train_indices = np.where(is_train == 1)[0]
                test_indices = np.where(is_train == 0)[0]
            else:
                # Create split manually
                np.random.seed(42)
                indices = np.arange(self.num_timepoints)
                np.random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio)
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
        
        # Create indices for all (timepoint, z) combinations in the split
        if split == 'train':
            valid_timepoints = train_indices
        else:
            valid_timepoints = test_indices
            
        self.indices = [(t, z) for t in valid_timepoints for z in range(self.num_z)]
        
        print(f"\nProbability Grid Dataset {os.path.basename(h5_file)}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Grid size: {self.height}x{self.width}")
        print(f"Total samples ({split}): {len(self.indices)}")
        print(f"Using probabilities as targets: {self.use_probabilities}")
        print(f"Has probability grids: {self.has_probabilities}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single grid from the dataset.
        
        Returns:
            grid: Tensor of shape (height, width) representing spike data
                  Values are binary (0/1) if use_probabilities=False, or probabilities [0,max] if True
        """
        # Get timepoint and z-level from indices
        timepoint, z_idx = self.indices[idx]
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Choose data source based on configuration
            if self.use_probabilities and self.has_probabilities:
                grids = f['probability_grids']
            elif 'spike_grids' in f:
                grids = f['spike_grids']
            else:
                # Fallback to old format
                grids = f['grids']
            
            # Get the grid for this timepoint and z-plane
            grid = grids[timepoint, z_idx]
        
        return torch.from_numpy(grid).float().contiguous()


class SyntheticSpikeProbabilityDataset(Dataset):
    def __init__(self, num_samples=10000, grid_size=(256, 128), probability_mode=False):
        """Dataset that generates synthetic spike data with optional probability values.
        
        Args:
            num_samples: Number of synthetic samples to generate
            grid_size: Size of the grid (height, width)
            probability_mode: If True, generate probability values. If False, generate binary values.
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.probability_mode = probability_mode
        
        # Distribution parameters from empirical analysis
        self.mean_spikes = 200.0
        self.std_spikes = 98.25
        self.min_spikes = 0
        self.max_spikes = 2000  # Using 99th percentile instead of max to avoid outliers
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random grid with number of spikes following empirical distribution."""
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
            
            if self.probability_mode:
                # Generate random probability values (similar to CASCADE range)
                spike_probs = np.random.exponential(scale=2.0, size=num_spikes)
                spike_probs = np.clip(spike_probs, 0.1, 30.0)  # Reasonable range
                grid[spike_rows, spike_cols] = spike_probs
            else:
                # Set binary spikes
                grid[spike_rows, spike_cols] = 1.0
        
        return torch.FloatTensor(grid) 