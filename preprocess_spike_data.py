#!/usr/bin/env python3
"""
Preprocess spike data into pre-augmented grid format.

This script loads the original spike data, converts it into 256x128 grid format,
applies z-plane augmentations, and saves it to a new directory structure for faster loading.
"""

import os
import time
import numpy as np
import h5py
from tqdm import tqdm
import torch
import gc
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import shutil

def print_memory_stats(prefix=""):
    """Print memory usage statistics"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{prefix} Memory: RSS={mem_info.rss / 1e9:.2f}GB, VMS={mem_info.vms / 1e9:.2f}GB")
    except ImportError:
        print(f"{prefix} Memory: psutil not available")

def augment_z_frames(frame, z_index, height=256, width=128):
    """Apply z-index based augmentation to a frame.
    
    This function adds marker patterns in the top-right corner of each frame:
    - For z-index 0, adds a 2×2 marker
    - For z-index 1, adds a 2×3 marker
    - For z-index 2, adds a 2×4 marker
    And so on, with the width increasing with the z-index
    
    Args:
        frame: Numpy array of shape (height, width) 
        z_index: Current z-index for the frame
        height, width: Frame dimensions
        
    Returns:
        Augmented frame
    """
    # Clone the frame to avoid modifying the original
    augmented_frame = frame.copy()
    
    # Calculate marker dimensions based on z-index
    marker_height = 2  # Always 2 pixels high
    marker_width = z_index + 2  # z-index + 2 pixels wide
    
    # Ensure marker fits within the frame
    marker_width = min(marker_width, width)
    
    # Set the top-right corner pixels to 1
    augmented_frame[0:marker_height, width-marker_width:width] = 1
    
    return augmented_frame

def preprocess_file(h5_file_path, output_dir, split_ratio=0.95, splits=('train', 'test'), seed=42):
    """Process a single H5 file and save pre-augmented grids to output directory.
    
    Args:
        h5_file_path: Path to the processed spike data h5 file
        output_dir: Base output directory
        split_ratio: Ratio of data to use for training (default: 0.95)
        splits: Tuple of splits to generate (default: ('train', 'test'))
        seed: Random seed for reproducibility
    """
    print(f"Processing {h5_file_path}...")
    start_time = time.time()
    
    # Create subject-specific output directory
    subject_name = os.path.basename(h5_file_path).replace('_processed.h5', '')
    subject_dir = os.path.join(output_dir, subject_name)
    
    print(f"Saving to {subject_dir}")
    
    # Create output directory
    os.makedirs(subject_dir, exist_ok=True)
    
    # Open the file for reading
    with h5py.File(h5_file_path, 'r', libver='latest', swmr=True) as f:
        # Get shape information
        num_timepoints = f['spikes'].shape[0]
        
        # Load cell positions (small enough to keep in memory)
        cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
        
        # Get unique z values (rounded to handle floating point precision)
        z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
        # Sort in ascending order (lower z-planes first)
        z_values = np.sort(z_values)
        num_z = len(z_values)
        
        print(f"Found {num_timepoints} timepoints and {num_z} z-planes")
        
        # Normalize cell positions to [0, 1]
        normalized_positions = (cell_positions - cell_positions.min(axis=0)) / \
                          (cell_positions.max(axis=0) - cell_positions.min(axis=0))
        
        # Convert to grid indices
        cell_x = np.floor(normalized_positions[:, 0] * 255).astype(np.int32)  # 0-255
        cell_y = np.floor(normalized_positions[:, 1] * 127).astype(np.int32)  # 0-127
        
        # Pre-compute z-plane masks and cell indices for each z-plane
        z_masks = {}
        z_cell_indices = {}
        
        for z_idx, z_level in enumerate(z_values):
            # Create mask for cells in this z-plane
            z_mask = (np.round(cell_positions[:, 2], decimals=3) == z_level)
            z_masks[z_idx] = z_mask
            
            # Store cell indices for this z-plane
            z_cell_indices[z_idx] = {
                'x': cell_x[z_mask],
                'y': cell_y[z_mask],
                'indices': np.where(z_mask)[0]  # Store the actual indices for faster lookup
            }
        
        # Split timepoints into train and test sets
        np.random.seed(seed)
        
        # Create blocks of timepoints for more robust train/test split
        block_size = 330  # Use fixed block size 
        num_blocks = num_timepoints // block_size
        
        # Create block indices and shuffle
        block_indices = np.arange(num_blocks)
        np.random.shuffle(block_indices)
        
        # Determine number of blocks for test set
        num_test_blocks = max(1, int(num_blocks * (1 - split_ratio)))
        
        # Select test blocks
        test_block_indices = block_indices[:num_test_blocks]
        train_block_indices = block_indices[num_test_blocks:]
        
        # Create masks for timepoints
        test_timepoints = []
        train_timepoints = []
        
        # Assign timepoints to train/test based on blocks
        for block_idx in range(num_blocks):
            start_timepoint = block_idx * block_size
            end_timepoint = min((block_idx + 1) * block_size, num_timepoints)
            
            if block_idx in test_block_indices:
                test_timepoints.extend(range(start_timepoint, end_timepoint))
            else:
                train_timepoints.extend(range(start_timepoint, end_timepoint))
        
        # Handle any remaining timepoints (assign to training)
        if num_blocks * block_size < num_timepoints:
            remaining_timepoints = range(num_blocks * block_size, num_timepoints)
            train_timepoints.extend(remaining_timepoints)
        
        # Sort timepoints for easier processing
        test_timepoints.sort()
        train_timepoints.sort()
        
        print(f"Train set: {len(train_timepoints)} timepoints")
        print(f"Test set: {len(test_timepoints)} timepoints")
        
        # Create split-specific timepoint lists
        split_timepoints = {
            'train': train_timepoints,
            'test': test_timepoints
        }
        
        # Create a binary mask for train vs test
        # 1 = train, 0 = test
        is_train = np.zeros(num_timepoints, dtype=np.uint8)
        is_train[train_timepoints] = 1
        
        # Save metadata about the dataset
        metadata = {
            'num_timepoints': num_timepoints,
            'num_z_planes': num_z,
            'z_values': z_values,
            'train_timepoints': train_timepoints,
            'test_timepoints': test_timepoints,
            'is_train': is_train
        }
        
        # Save metadata to a separate file
        with h5py.File(os.path.join(subject_dir, 'metadata.h5'), 'w') as meta_file:
            meta_file.create_dataset('num_timepoints', data=num_timepoints)
            meta_file.create_dataset('num_z_planes', data=num_z)
            meta_file.create_dataset('z_values', data=z_values)
            meta_file.create_dataset('train_timepoints', data=train_timepoints)
            meta_file.create_dataset('test_timepoints', data=test_timepoints)
            meta_file.create_dataset('is_train', data=is_train)
        
        # Process each timepoint and z-plane
        spikes = f['spikes']
        
        # Create a single HDF5 file with all timepoints in original order
        all_timepoints = list(range(num_timepoints))
        
        # Create a combined output file
        output_file_path = os.path.join(subject_dir, 'preaugmented_grids.h5')
        with h5py.File(output_file_path, 'w') as out_f:
            # Create a dataset to store all grids
            # Shape: (num_timepoints, num_z, height, width)
            out_f.create_dataset('grids', 
                                shape=(num_timepoints, num_z, 256, 128), 
                                dtype=np.uint8,
                                chunks=(1, 1, 256, 128),  # Chunk by individual frames
                                compression='gzip',
                                compression_opts=1)  # Light compression for speed
            
            # Create dataset to store timepoint mapping (original indices)
            out_f.create_dataset('timepoint_indices', data=np.array(all_timepoints, dtype=np.int32))
            
            # Create dataset to store train/test mask
            out_f.create_dataset('is_train', data=is_train)
            
            # Process each timepoint
            for idx, t in enumerate(tqdm(all_timepoints, desc=f"Processing timepoints")):
                # Process each z-plane
                for z_idx in range(num_z):
                    # Get pre-computed cell indices for this z-plane
                    cell_indices = z_cell_indices[z_idx]['indices']
                    
                    # Create empty grid for this timepoint and z-plane
                    grid = np.zeros((256, 128), dtype=np.uint8)
                    
                    # Skip if no cells in this z-plane
                    if len(cell_indices) == 0:
                        # Still apply augmentation 
                        grid = augment_z_frames(grid, z_idx)
                        out_f['grids'][idx, z_idx] = grid
                        continue
                    
                    # Get spikes for this timepoint and z-plane
                    spikes_t = spikes[t][cell_indices]
                    
                    # Find active cells
                    active_cells = np.abs(spikes_t) > 1e-6
                    
                    if np.any(active_cells):  # Only process if there are active cells
                        active_x = z_cell_indices[z_idx]['x'][active_cells]
                        active_y = z_cell_indices[z_idx]['y'][active_cells]
                        
                        # Set active cells to 1 in the grid
                        for i in range(len(active_x)):
                            grid[active_x[i], active_y[i]] = 1
                    
                    # Apply z-plane augmentation
                    grid = augment_z_frames(grid, z_idx)
                    
                    # Save to output file
                    out_f['grids'][idx, z_idx] = grid
            
            # Add metadata to output file
            out_f.attrs['num_timepoints'] = num_timepoints
            out_f.attrs['num_z_planes'] = num_z
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return subject_dir

def main():
    parser = argparse.ArgumentParser(description='Preprocess spike data into pre-augmented grid format.')
    parser.add_argument('--input_dir', default='training_spike_data_2018', 
                        help='Directory containing processed spike data')
    parser.add_argument('--output_dir', default='preaugmented_training_spike_data_2018', 
                        help='Output directory for pre-augmented data')
    parser.add_argument('--split_ratio', type=float, default=0.95, 
                        help='Ratio of data to use for training')
    parser.add_argument('--workers', type=int, default=1, 
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all processed H5 files
    h5_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith("_processed.h5"):
            h5_files.append(os.path.join(args.input_dir, file))
    
    if not h5_files:
        print(f"No processed spike data files found in {args.input_dir}!")
        return
    
    print(f"Found {len(h5_files)} processed spike data files")
    
    # Process files
    if args.workers > 1:
        print(f"Processing files in parallel with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    preprocess_file, 
                    h5_file, 
                    args.output_dir, 
                    args.split_ratio,
                    ('train', 'test'),
                    args.seed
                ) 
                for h5_file in h5_files
            ]
            
            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing files"):
                future.result()
    else:
        print("Processing files sequentially")
        for h5_file in h5_files:
            preprocess_file(
                h5_file, 
                args.output_dir, 
                args.split_ratio,
                ('train', 'test'),
                args.seed
            )
    
    print(f"All files processed successfully. Output saved to {args.output_dir}")
    
    # Create a combined metadata file
    print("Creating combined metadata file...")
    combined_metadata = {}
    for subject_dir in os.listdir(args.output_dir):
        subject_path = os.path.join(args.output_dir, subject_dir)
        if os.path.isdir(subject_path):
            metadata_path = os.path.join(subject_path, 'metadata.h5')
            if os.path.exists(metadata_path):
                with h5py.File(metadata_path, 'r') as f:
                    subject_data = {
                        'num_timepoints': f['num_timepoints'][()],
                        'num_z_planes': f['num_z_planes'][()],
                        'z_values': f['z_values'][:],
                        'train_timepoints': f['train_timepoints'][:],
                        'test_timepoints': f['test_timepoints'][:],
                        'is_train': f['is_train'][:]
                    }
                    combined_metadata[subject_dir] = subject_data
    
    # Save combined metadata
    with h5py.File(os.path.join(args.output_dir, 'combined_metadata.h5'), 'w') as f:
        for subject, data in combined_metadata.items():
            subject_group = f.create_group(subject)
            for key, value in data.items():
                subject_group.create_dataset(key, data=value)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)  
    main() 