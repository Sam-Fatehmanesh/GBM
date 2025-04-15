#!/usr/bin/env python
# Script to test the z-index 0 augmentation in the DALI dataset

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from GenerativeBrainModel.datasets.dali_spike_dataset import augment_z0_frames

# Add the current directory to the path
sys.path.append('.')

# Import the DALI data loader
from GenerativeBrainModel.datasets.dali_spike_dataset import H5Loader

def test_augmentation():
    """Test the z-index 0 augmentation in the DALI dataset"""
    # Find all processed spike files
    spike_files = glob.glob(os.path.join('processed_spikes', '*_processed.h5'))
    
    if not spike_files:
        print("No processed spike files found in 'processed_spikes' directory.")
        return
    
    print(f"Found {len(spike_files)} processed spike files:")
    for f in spike_files:
        print(f"  {os.path.basename(f)}")
    
    # Use the first file for testing
    first_file = spike_files[0]
    print(f"\nTesting augmentation on file: {os.path.basename(first_file)}")
    
    # Create an H5Loader with a small sequence length for testing
    loader = H5Loader(first_file, seq_len=30)
    
    # Find a sequence that starts at z-index 0
    z0_idx = None
    for idx, (t, z) in enumerate(loader.valid_starts):
        if z == 0:
            z0_idx = idx
            print(f"Found sequence starting at z-index 0: index={idx}, timepoint={t}, z-index={z}")
            break
    
    if z0_idx is None:
        print("Could not find a sequence starting at z-index 0. Using the first sequence instead.")
        z0_idx = 0
    
    # Get the sequence
    sample_info = {'idx_in_epoch': z0_idx}
    sequence = loader(sample_info)
    
    print(f"Sequence shape: {sequence.shape}")
    
    # Create a directory for the test output
    os.makedirs('augmentation_test', exist_ok=True)
    
    # Track the current timepoint and z-index
    t_start, z_start = loader.valid_starts[z0_idx]
    print(f"Starting timepoint: {t_start}, Starting z-plane index: {z_start}")
    
    current_t = t_start
    current_z = z_start
    
    # Create a figure to display the frames
    plt.figure(figsize=(15, 10))
    
    # Display the first 10 frames
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(sequence[0, i], cmap='gray')
        plt.title(f"Frame {i}\nT={current_t}, Z={current_z}")
        plt.axis('off')
        
        # Update z-index for the next frame
        current_z += 1
        if current_z >= loader.num_z:
            current_t += 1
            current_z = 0
    
    plt.tight_layout()
    plt.savefig('augmentation_test/frames.png')
    print(f"Saved frame visualization to augmentation_test/frames.png")
    
    # Check if the augmentation is visible in the top right corner
    # for frames with z-index 0
    z0_frames = []
    current_t = t_start
    current_z = z_start
    
    for i in range(sequence.shape[1]):
        if current_z == 0:
            z0_frames.append(i)
        
        # Update z-index for the next frame
        current_z += 1
        if current_z >= loader.num_z:
            current_t += 1
            current_z = 0
    
    print(f"Frames with z-index 0: {z0_frames}")
    
    # Check if the top right corner is set to 1 for these frames
    for frame_idx in z0_frames:
        top_right = sequence[0, frame_idx, 0:2, 126:128]
        print(f"Frame {frame_idx} top right corner:\n{top_right}")
        
        # Zoom in on the top right corner
        plt.figure(figsize=(5, 5))
        plt.imshow(sequence[0, frame_idx, 0:10, 118:128], cmap='gray')
        plt.title(f"Frame {frame_idx} (z-index 0)\nTop Right Corner")
        plt.axis('on')
        plt.grid(True)
        plt.savefig(f'augmentation_test/frame_{frame_idx}_corner.png')
        plt.close()
    
    # Also check a few frames that are not z-index 0 for comparison
    non_z0_frames = [i for i in range(min(10, sequence.shape[1])) if i not in z0_frames]
    if non_z0_frames:
        print(f"\nChecking non-z0 frames for comparison: {non_z0_frames[:3]}")
        for frame_idx in non_z0_frames[:3]:
            top_right = sequence[0, frame_idx, 0:2, 126:128]
            print(f"Frame {frame_idx} top right corner:\n{top_right}")
            
            # Zoom in on the top right corner
            plt.figure(figsize=(5, 5))
            plt.imshow(sequence[0, frame_idx, 0:10, 118:128], cmap='gray')
            plt.title(f"Frame {frame_idx} (not z-index 0)\nTop Right Corner")
            plt.axis('on')
            plt.grid(True)
            plt.savefig(f'augmentation_test/frame_{frame_idx}_corner_non_z0.png')
            plt.close()
    
    print("Test completed. Check the augmentation_test directory for results.")

# Create a mock loader class
class MockLoader:
    num_z = 29

# Test with a 3D tensor
batch_3d = np.zeros((10, 256, 128))
print(f"Input 3D shape: {batch_3d.shape}")

# Apply augmentation
batch_aug_3d = augment_z0_frames(batch_3d, MockLoader())
print(f"Output 3D shape: {batch_aug_3d.shape}")

# Check if marker is present in first frame
if np.sum(batch_aug_3d[0, 0:2, -2:]) == 4:
    print("3D test passed - marker found in first frame")
else:
    print(f"3D test failed - marker not found. Top-right corner: {batch_aug_3d[0, 0:2, -2:]}")

# Test with a 4D tensor
batch_4d = np.zeros((2, 10, 256, 128))
print(f"\nInput 4D shape: {batch_4d.shape}")

# Apply augmentation
batch_aug_4d = augment_z0_frames(batch_4d, MockLoader())
print(f"Output 4D shape: {batch_aug_4d.shape}")

# Check if marker is present in first frame of first batch
if np.sum(batch_aug_4d[0, 0, 0:2, -2:]) == 4:
    print("4D test passed - marker found in first frame")
else:
    print(f"4D test failed - marker not found. Top-right corner: {batch_aug_4d[0, 0, 0:2, -2:]}")

print("\nTest completed successfully")

if __name__ == "__main__":
    test_augmentation() 