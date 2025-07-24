"""
PyTorch DataLoader for 3D volumetric data.
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

class VolumeDataset(Dataset):
    """
    General PyTorch Dataset for 3D volumetric data with flexible sequence parameters.
    Supports different sequence lengths, strides, and timepoint configurations.
    Caches all data in CPU memory if specified for faster training.
    """
    
    def __init__(self, data_files: List[str], sequence_length: int = 1, stride: int = 1, 
                 max_timepoints_per_subject: Optional[int] = None, use_cache: bool = False):
        """
        Args:
            data_files: List of H5 file paths.
            sequence_length: Number of consecutive timepoints per sample.
            stride: Step size between sequence starts.
            max_timepoints_per_subject: Max timepoints to use per subject file.
            use_cache: If True, loads entire dataset into CPU memory.
        """
        self.data_files = data_files
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_timepoints_per_subject = max_timepoints_per_subject
        self.use_cache = use_cache
        
        self.sequences = []
        self.cached_data = {} if use_cache else None
        
        # This dictionary will be populated by each worker process independently
        self.worker_file_handles = {}
        
        self._build_sequence_index()
        
    def _build_sequence_index(self):
        """Builds an index of all available sequences and caches data in memory if requested."""
        print(f"Building sequence index... Caching enabled: {self.use_cache}")
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                volumes = f['volumes']
                total_timepoints = volumes.shape[0]
                
                num_timepoints = min(self.max_timepoints_per_subject, total_timepoints) if self.max_timepoints_per_subject is not None else total_timepoints

                # ------------------------------------------------------------------
                # Skip consecutive leading and trailing all-zero volumes
                # ------------------------------------------------------------------
                first_nonzero = 0
                while first_nonzero < num_timepoints and np.all(volumes[first_nonzero] == 0):
                    first_nonzero += 1

                if first_nonzero == num_timepoints:
                    print(f"Warning: All {num_timepoints} volumes in {Path(file_path).name} are zero – skipping file.")
                    continue  # Entire file is zeros

                last_nonzero = num_timepoints - 1
                while last_nonzero >= first_nonzero and np.all(volumes[last_nonzero] == 0):
                    last_nonzero -= 1

                if last_nonzero < first_nonzero:
                    # Should not happen, but safeguard
                    print(f"Warning: No non-zero volumes found in {Path(file_path).name} after scanning – skipping file.")
                    continue

                # Informational logging
                if first_nonzero > 0:
                    print(f"Skipping {first_nonzero} initial zero volumes in {Path(file_path).name} (start at {first_nonzero}).")
                if last_nonzero < num_timepoints - 1:
                    skipped_trailing = num_timepoints - 1 - last_nonzero
                    print(f"Skipping {skipped_trailing} trailing zero volumes in {Path(file_path).name} (end at {last_nonzero}).")

                effective_timepoints = last_nonzero - first_nonzero + 1

                # Cache only useful slice
                if self.use_cache and file_path not in self.cached_data:
                    print(f"Caching data from {Path(file_path).name} ({effective_timepoints} timepoints)...")
                    self.cached_data[file_path] = volumes[first_nonzero:last_nonzero+1]

                # Build sequence index within non-zero window
                max_start_idx = last_nonzero - self.sequence_length + 1
                if max_start_idx >= first_nonzero:
                    for start_idx in range(first_nonzero, max_start_idx + 1, self.stride):
                        self.sequences.append({'file_path': file_path, 'start_idx': start_idx})
    
    def __len__(self):
        return len(self.sequences)
    
    def _get_file_handle(self, file_path: str):
        """
        Retrieves an open HDF5 file handle. Each worker process maintains its
        own set of open file handles to avoid conflicts.
        """
        if file_path not in self.worker_file_handles:
            self.worker_file_handles[file_path] = h5py.File(file_path, 'r')
        return self.worker_file_handles[file_path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_info = self.sequences[idx]
        file_path = seq_info['file_path']
        start_idx = seq_info['start_idx']
        end_idx = start_idx + self.sequence_length
        
        if self.use_cache:
            volumes = self.cached_data[file_path][start_idx:end_idx]
        else:
            # Each worker gets its own persistent file handle
            f = self._get_file_handle(file_path)
            volumes = f['volumes'][start_idx:end_idx]
        
        volumes = torch.from_numpy(volumes.astype(np.float32))
        
        if self.sequence_length == 1:
            volumes = volumes.squeeze(0)
        
        # For autoencoder, input and target are the same
        return volumes, volumes

    def __del__(self):
        """Ensures file handles are closed when a worker process exits."""
        for handle in self.worker_file_handles.values():
            handle.close()

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test PyTorch DataLoaders.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        A tuple containing the training and testing dataloaders.
    """
    print("Using PyTorch DataLoader.")
    
    data_config = config['data']
    training_config = config['training']
    
    data_dir = Path(data_config['data_dir'])
    all_files = sorted([str(f) for f in data_dir.glob("*.h5")])
    
    test_subjects = data_config.get('test_subjects', [])
    test_files = [str(data_dir / f"{s}.h5") for s in test_subjects if (data_dir / f"{s}.h5").exists()]
    
    if not test_files and test_subjects:
        print(f"Warning: No valid files found for test subjects: {test_subjects}")

    train_files = [f for f in all_files if f not in test_files]
    
    if not test_files:
        print("No valid test subjects provided. Creating a random 80/20 train/test split.")
        num_test = max(1, len(all_files) // 5)
        random.seed(training_config.get('seed', 42))
        test_files = random.sample(all_files, num_test)
        train_files = [f for f in all_files if f not in test_files]

    # Get parameters from config
    sequence_length = training_config.get('sequence_length', 1)
    stride = training_config.get('stride', 1)
    max_timepoints_per_subject = training_config.get('max_timepoints_per_subject', None)
    use_cache = data_config.get('cache_data', True)

    train_dataset = VolumeDataset(
        train_files, 
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache
    )
    
    test_dataset = VolumeDataset(
        test_files,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('volumes_per_batch', 4),
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
        persistent_workers=True,
        prefetch_factor=8
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.get('volumes_per_batch', 4),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
        persistent_workers=True,
        prefetch_factor=8
    )
    
    return train_loader, test_loader

def get_volume_info(data_dir: str) -> Dict:
    """
    Get information about volumes in the data directory.
    
    Args:
        data_dir: Path to directory containing H5 files
        
    Returns:
        Dictionary with volume information
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.h5"))
    
    if not files:
        raise ValueError(f"No H5 files found in {data_dir}")
    
    # Get info from first file
    with h5py.File(files[0], 'r') as f:
        sample_volume = f['volumes'][0]
        volume_shape = sample_volume.shape
        dtype = sample_volume.dtype
        
        all_same_shape = True
        shapes = [volume_shape]
        
        for file_path in files[:5]:
            with h5py.File(file_path, 'r') as f2:
                shape = f2['volumes'][0].shape
                shapes.append(shape)
                if shape != volume_shape:
                    all_same_shape = False
    
    return {
        'num_files': len(files),
        'volume_shape': volume_shape,
        'volume_size': list(volume_shape),  # Add volume_size as a list for compatibility
        'dtype': str(dtype),
        'all_same_shape': all_same_shape,
        'sample_shapes': shapes,
        'files': [f.name for f in files[:10]]
    } 