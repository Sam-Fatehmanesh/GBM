import os
import h5py
import torch
import torch.utils.data
import numpy as np
from typing import List


class ChunkedSpikeDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that loads data on-demand from HDF5 files."""
    
    def __init__(self, data_dir, subjects):
        """
        Create a dataset that loads data on-demand without memory limits.
        
        Args:
            data_dir: Directory containing subject folders
            subjects: List of subject IDs to include
        """
        self.data_dir = data_dir
        self.subjects = subjects
        
        # Build index of all samples (including z-planes)
        self.sample_index = []  # List of (h5_path, sample_idx, z_idx)
        
        for subject_id in subjects:
            subject_dir = os.path.join(data_dir, f"subject_{subject_id}")
            h5_path = os.path.join(subject_dir, "preaugmented_grids.h5")
            
            if not os.path.exists(h5_path):
                print(f"Warning: {h5_path} not found, skipping subject {subject_id}")
                continue
                
            try:
                with h5py.File(h5_path, 'r') as f:
                    grids = f['grids']
                    total_samples = grids.shape[0]
                    n_z_planes = grids.shape[1]  # Number of z-planes per sample
                    
                    # Add sample indices for each z-plane of each sample (ALL data)
                    for i in range(total_samples):
                        for z in range(n_z_planes):
                            self.sample_index.append((h5_path, i, z))
                    
                    total_slices = total_samples * n_z_planes
                    print(f"Added {total_samples} samples x {n_z_planes} z-planes = {total_slices} 2D slices from subject {subject_id}")
                    
            except Exception as e:
                print(f"Error accessing {h5_path}: {e}")
                continue
        
        if not self.sample_index:
            raise ValueError("No data was loaded. Check your data directory and subject IDs.")
        
        print(f"Total samples available: {len(self.sample_index)}")
        
        # No caching - load on demand
        self.sample_shape = None
        
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        h5_path, original_sample_idx, z_idx = self.sample_index[idx]
        
        # Load specific slice on-demand (no caching)
        with h5py.File(h5_path, 'r') as f:
            grids = f['grids']
            
            # Load only the specific slice we need
            slice_2d = np.array(grids[original_sample_idx, z_idx], dtype=np.float32)
            
            if self.sample_shape is None:
                self.sample_shape = slice_2d.shape
                print(f"2D slice shape: {self.sample_shape}")
        
        return torch.from_numpy(slice_2d)
    
    def get_sample_shape(self):
        """Return the shape of a single sample."""
        if self.sample_shape is not None:
            return self.sample_shape
        
        # Get shape from first sample
        if len(self.sample_index) > 0:
            sample = self[0]
            self.sample_shape = sample.shape
            return self.sample_shape
        
        return None


def create_simple_dataloaders(
    data_dir: str,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    max_samples_per_subject: int = 50000,  # Keep for compatibility but ignore
    num_workers: int = 4,
    **kwargs  # For backward compatibility
):
    """Create memory-efficient train and test dataloaders using random split of all data."""
    
    # Get all available subjects
    all_subjects = []
    for item in os.listdir(data_dir):
        if item.startswith('subject_') and os.path.isdir(os.path.join(data_dir, item)):
            subject_id = int(item.split('_')[1])
            all_subjects.append(subject_id)
    
    print(f"Found {len(all_subjects)} subjects: {all_subjects}")
    print(f"Using train/test split: {train_ratio:.1%}/{1-train_ratio:.1%}")
    
    # Create a single dataset with all subjects
    full_dataset = ChunkedSpikeDataset(
        data_dir=data_dir,
        subjects=all_subjects
    )
    
    # Split dataset randomly
    total_samples = len(full_dataset)
    train_size = int(total_samples * train_ratio)
    test_size = total_samples - train_size
    
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # Create random split
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(42)  # For reproducible splits
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues with HDF5
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Get sample shape from the underlying dataset
    sample_shape = full_dataset.get_sample_shape()
    
    return train_loader, test_loader, sample_shape 