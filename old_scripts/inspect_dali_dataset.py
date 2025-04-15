#!/usr/bin/env python
# Script to inspect the DALI dataset used in train_gbm_dali.py

import os
import sys
import h5py
import numpy as np
import torch
import glob
from tqdm import tqdm

# Add the current directory to the path
sys.path.append('.')

# Import the DALI data loader
from GenerativeBrainModel.datasets.dali_spike_dataset import DALIBrainDataLoader, H5Loader

def analyze_sequence(loader, sample_idx, show_all_frames=False):
    """Analyze a sequence from the loader"""
    t_start, z_start = loader.valid_starts[sample_idx]
    print(f"\nSample {sample_idx}:")
    print(f"Starting timepoint: {t_start}, Starting z-plane index: {z_start}")
    
    if z_start < len(loader.z_values):
        z_value = loader.z_values[z_start]
        print(f"Starting z-plane value: {z_value}")
    
    sample_info = {'idx_in_epoch': sample_idx}
    sequence = loader(sample_info)
    
    print(f"Sequence shape: {sequence.shape}")
    
    # Analyze the sequence
    print("Frame analysis:")
    max_frames = sequence.shape[1] if show_all_frames else min(30, sequence.shape[1])
    
    # Create a table header
    print(f"{'Frame':<6} {'Active':<8} {'Timepoint':<10} {'Z-Index':<8} {'Z-Value':<8}")
    print("-" * 50)
    
    # Initialize variables to track the current timepoint and z-index
    current_t = t_start
    current_z = z_start
    
    for frame_idx in range(max_frames):
        # Count active cells in this frame
        active_count = np.sum(sequence[0, frame_idx] > 0)
        
        # Get the z-value for the current z-index
        z_value = loader.z_values[current_z] if current_z < len(loader.z_values) else None
        
        # Print in table format
        print(f"{frame_idx:<6} {active_count:<8} {current_t:<10} {current_z:<8} {z_value:<8}")
        
        # Update z-index for the next frame
        current_z += 1
        
        # If we've reached the end of z-planes for this timepoint
        if current_z >= loader.num_z:
            # Move to the next timepoint and reset z-index to 0
            current_t += 1
            current_z = 0
    
    return sequence

def main():
    # Find all processed spike files
    spike_files = glob.glob(os.path.join('processed_spikes', '*_processed.h5'))
    
    if not spike_files:
        print("No processed spike files found in 'processed_spikes' directory.")
        return
    
    print(f"Found {len(spike_files)} processed spike files:")
    for f in spike_files:
        print(f"  {os.path.basename(f)}")
    
    # Examine the first file to understand its structure
    first_file = spike_files[0]
    print(f"\nExamining file: {os.path.basename(first_file)}")
    
    with h5py.File(first_file, 'r') as f:
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
        print(f"Z-plane values (sorted): {z_values}")
        
        # Print a sample of the spikes data
        print(f"Sample of spikes data (first timepoint):")
        spikes_sample = f['spikes'][0]
        print(f"  Shape: {spikes_sample.shape}")
        print(f"  Non-zero elements: {np.count_nonzero(spikes_sample)}")
        print(f"  Min: {spikes_sample.min()}, Max: {spikes_sample.max()}")
    
    # Create an H5Loader to examine how it processes the data
    print("\nCreating H5Loader for the first file...")
    loader = H5Loader(first_file, seq_len=30)
    
    # Print information about the loader
    print(f"Number of z-planes: {loader.num_z}")
    print(f"Z-values: {loader.z_values}")
    print(f"Number of valid sequence starts: {len(loader.valid_starts)}")
    
    # Analyze multiple sequences to verify ordering
    print("\n=== ANALYZING MULTIPLE SEQUENCES TO VERIFY ORDERING ===")
    
    # Sample 1: First sequence
    analyze_sequence(loader, 0)
    
    # Sample 2: A sequence from the middle
    middle_idx = len(loader.valid_starts) // 2
    analyze_sequence(loader, middle_idx)
    
    # Sample 3: Last sequence
    last_idx = len(loader.valid_starts) - 1
    analyze_sequence(loader, last_idx)
    
    # Sample 4: A sequence starting at z-plane 0
    for idx, (t, z) in enumerate(loader.valid_starts):
        if z == 0:
            print("\nFound a sequence starting at z-plane 0:")
            analyze_sequence(loader, idx)
            break
    
    # Sample 5: A sequence that spans multiple timepoints
    print("\n=== ANALYZING A SEQUENCE THAT SPANS MULTIPLE TIMEPOINTS ===")
    # The sequence length is 30, and if there are fewer than 30 z-planes,
    # the sequence will span multiple timepoints
    if loader.num_z < 30:
        print(f"With {loader.num_z} z-planes and sequence length 30, sequences will span multiple timepoints")
        analyze_sequence(loader, 0, show_all_frames=True)
    else:
        print(f"With {loader.num_z} z-planes, a sequence of length 30 won't span multiple timepoints")

if __name__ == "__main__":
    main() 