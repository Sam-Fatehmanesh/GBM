#!/usr/bin/env python
# Script to inspect the order of z-indices and timepoints in the DALI dataset

import os
import h5py
import numpy as np
import torch
import glob
from tqdm import tqdm

# Add the current directory to the path so we can import the modules
import sys
sys.path.append('.')

# Import the DALI data loader
from GenerativeBrainModel.datasets.dali_spike_dataset import DALIBrainDataLoader, H5Loader

def inspect_h5_file_structure(h5_file_path):
    """Inspect the structure of an H5 file to understand its organization"""
    print(f"\nInspecting H5 file: {os.path.basename(h5_file_path)}")
    with h5py.File(h5_file_path, 'r') as f:
        # Print the keys in the file
        print(f"Keys in the file: {list(f.keys())}")
        
        # Get shape information
        spikes_shape = f['spikes'].shape
        print(f"Spikes shape: {spikes_shape}")
        
        # Load cell positions
        cell_positions = f['cell_positions'][:]
        
        # Get unique z values
        z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
        z_values = sorted(z_values)
        print(f"Number of unique z-planes: {len(z_values)}")
        print(f"Z-plane values (sorted): {z_values}")
        
        # Count cells per z-plane
        cells_per_z = {}
        for z in z_values:
            # Find cells in this z-plane (with rounding to handle floating point precision)
            mask = np.isclose(cell_positions[:, 2], z, rtol=1e-3)
            cells_per_z[z] = np.sum(mask)
        
        print(f"Cells per z-plane: {cells_per_z}")
        
        return spikes_shape, z_values, cells_per_z

def inspect_h5loader_sequence(h5_file_path, seq_len=30):
    """Inspect how H5Loader processes sequences"""
    print(f"\nInspecting H5Loader sequence generation for: {os.path.basename(h5_file_path)}")
    
    # Create an H5Loader instance
    loader = H5Loader(h5_file_path, seq_len=seq_len)
    
    # Get the total number of sequences
    total_sequences = len(loader)
    print(f"Total sequences available: {total_sequences}")
    
    # Sample a few sequences
    sample_indices = [0, 1, total_sequences//2, total_sequences-2, total_sequences-1]
    sample_indices = [i for i in sample_indices if i < total_sequences]
    
    for idx in sample_indices:
        # Get sample info for this index
        sample_info = {'idx_in_epoch': idx}
        print(f"\nSample {idx} info:")
        
        # Get the starting timepoint and z-plane
        t_start, z_start = loader.valid_starts[idx]
        print(f"  Starting timepoint: {t_start}")
        print(f"  Starting z-plane index: {z_start}")
        
        if hasattr(loader, 'z_values') and z_start < len(loader.z_values):
            z_value = loader.z_values[z_start]
            print(f"  Starting z-plane value: {z_value}")
        
        # Generate the sequence
        sequence = loader(sample_info)
        
        # Count active cells in each frame
        active_counts = np.sum(sequence > 0, axis=(1, 2))
        
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Active cells per frame: {active_counts}")
        
        # Detailed breakdown of the first few frames
        print(f"  Detailed frame breakdown:")
        for frame_idx in range(min(5, seq_len)):
            print(f"    Frame {frame_idx}: {np.sum(sequence[0, frame_idx] > 0)} active cells")
            
            # Calculate the actual timepoint and z-plane for this frame
            # This logic mirrors what happens in the H5Loader.__call__ method
            if frame_idx == 0:
                current_t = t_start
                current_z = z_start
            else:
                # Calculate z-plane index for this frame
                z_idx_in_cycle = frame_idx % loader.num_z
                current_z = (z_start + z_idx_in_cycle) % loader.num_z
                
                # Calculate timepoint for this frame
                frames_per_timepoint = loader.num_z
                timepoint_offset = frame_idx // frames_per_timepoint
                current_t = t_start + timepoint_offset
            
            if hasattr(loader, 'z_values') and current_z < len(loader.z_values):
                z_value = loader.z_values[current_z]
                print(f"      Timepoint: {current_t}, z-plane index: {current_z}, z-value: {z_value}")
            else:
                print(f"      Timepoint: {current_t}, z-plane index: {current_z}")

def create_simple_dataloader(h5_files, batch_size=8, seq_len=30):
    """Create a simple data loader without DALI for inspection"""
    print(f"\nCreating simple DataLoader with {len(h5_files)} H5 files")
    
    # Create H5Loaders for each file
    loaders = [H5Loader(f, seq_len=seq_len) for f in h5_files]
    
    # Get a batch from each loader
    batch_list = []
    for loader in loaders:
        # Get a sample
        if len(loader) > 0:
            sample_info = {'idx_in_epoch': 0}
            sequence = loader(sample_info)
            batch_list.append(sequence)
    
    # Stack the batches
    if batch_list:
        batch = np.concatenate(batch_list, axis=0)[:batch_size]
        print(f"Batch shape: {batch.shape}")
        
        # Analyze the batch
        print("\nBatch analysis:")
        for seq_idx in range(min(3, batch.shape[0])):
            sequence = batch[seq_idx]
            active_counts = np.sum(sequence > 0, axis=(1, 2))
            print(f"  Sequence {seq_idx}:")
            print(f"    Shape: {sequence.shape}")
            print(f"    Active cells per frame: {active_counts}")
            
            # Detailed breakdown of the first few frames
            print(f"    Detailed frame breakdown:")
            for frame_idx in range(min(5, seq_len)):
                active_count = np.sum(sequence[frame_idx] > 0)
                print(f"      Frame {frame_idx}: {active_count} active cells")
                
                # Show active cell positions if any
                if active_count > 0:
                    active_y, active_x = np.where(sequence[frame_idx] > 0)
                    print(f"        Active positions (y,x): {list(zip(active_y[:5], active_x[:5]))}{'...' if active_count > 5 else ''}")
    else:
        print("No batches could be created.")

def main():
    # Find all processed spike files
    spike_files = glob.glob(os.path.join('processed_spikes', '*_processed.h5'))
    
    if not spike_files:
        print("No processed spike files found in 'processed_spikes' directory.")
        return
    
    print(f"Found {len(spike_files)} processed spike files:")
    for f in spike_files:
        print(f"  {os.path.basename(f)}")
    
    # Inspect the structure of the first file
    first_file = spike_files[0]
    spikes_shape, z_values, cells_per_z = inspect_h5_file_structure(first_file)
    
    # Inspect how H5Loader processes sequences
    inspect_h5loader_sequence(first_file)
    
    # Create a simple data loader for inspection
    create_simple_dataloader(spike_files[:2])  # Use only the first 2 files to keep it manageable

if __name__ == "__main__":
    main() 