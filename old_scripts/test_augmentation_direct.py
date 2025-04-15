#!/usr/bin/env python
# Script to directly test the z-index-based augmentation function

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

# Add the current directory to the path
sys.path.append('.')

# Import the augmentation function
from GenerativeBrainModel.datasets.dali_spike_dataset import augment_z0_frames

def test_augmentation_direct():
    """Directly test the augmentation function"""
    # Create a simple batch with all zeros
    batch = np.zeros((1, 30, 256, 128), dtype=np.uint8)
    
    # Create a mock H5Loader class with the necessary attributes
    class MockH5Loader:
        def __init__(self):
            self.num_z = 29
            self.valid_starts = [(0, 0)]  # Start at timepoint 0, z-index 0
            self.z_values = np.arange(1, 30)  # Z-values from 1 to 29
    
    # Create a mock loader
    mock_loader = MockH5Loader()
    
    # Apply the augmentation
    augmented_batch = augment_z0_frames(batch, mock_loader)
    
    # Create a directory for the test output
    os.makedirs('augmentation_test_direct', exist_ok=True)
    
    # Check the augmentation for each frame
    current_z = 0
    
    for frame_idx in range(30):
        # Calculate the expected marker dimensions
        height = 2  # Always 2 cells high
        width = current_z + 2  # z-index + 2 cells wide
        
        # Get the actual marker
        marker_region = augmented_batch[0, frame_idx, 0:height, 128-width:128]
        
        print(f"Frame {frame_idx} (z-index {current_z}):")
        print(f"  Expected marker size: {height}×{width}")
        print(f"  Actual marker region:\n{marker_region}")
        print(f"  All ones: {np.all(marker_region == 1)}")
        
        # Visualize the frame
        plt.figure(figsize=(8, 8))
        plt.imshow(augmented_batch[0, frame_idx], cmap='gray')
        plt.title(f"Frame {frame_idx} (z-index {current_z})\nMarker: {height}×{width}")
        plt.axis('on')
        plt.grid(True)
        plt.savefig(f'augmentation_test_direct/frame_{frame_idx}_z{current_z}.png')
        plt.close()
        
        # Update z-index for the next frame
        current_z += 1
        if current_z >= mock_loader.num_z:
            current_z = 0
    
    # Create a composite image showing the first 10 frames
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(augmented_batch[0, i], cmap='gray')
        plt.title(f"Frame {i}\nz-index {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_test_direct/composite.png')
    plt.close()
    
    print("Test completed. Check the augmentation_test_direct directory for results.")

if __name__ == "__main__":
    test_augmentation_direct() 