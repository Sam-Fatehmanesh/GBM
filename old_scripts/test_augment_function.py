#!/usr/bin/env python
"""
Test script for the augmentation function in dali_spike_dataset.py
"""
import numpy as np
from GenerativeBrainModel.datasets.dali_spike_dataset import augment_z0_frames

# Create a mock loader class
class MockLoader:
    num_z = 29

# Test with a 3D tensor
batch_3d = np.zeros((10, 256, 128))
print(f"Input 3D shape: {batch_3d.shape}")

# Apply augmentation
batch_aug_3d = augment_z0_frames(batch_3d, MockLoader())
print(f"Output 3D shape: {batch_aug_3d.shape}")

# Check top-right corner of first frame
top_right = batch_aug_3d[0, 0:2, -2:]
print(f"First frame top-right corner shape: {top_right.shape}")
print(f"First frame top-right values:\n{top_right}")

# Print the first two rows of the first frame to see if markers are in the right place
print(f"First frame first two rows:\n{batch_aug_3d[0, 0:2]}")

# Test with a 4D tensor
batch_4d = np.zeros((2, 10, 256, 128))
print(f"\nInput 4D shape: {batch_4d.shape}")

# Apply augmentation
batch_aug_4d = augment_z0_frames(batch_4d, MockLoader())
print(f"Output 4D shape: {batch_aug_4d.shape}")

# Check top-right corner of first frame of first batch
top_right_4d = batch_aug_4d[0, 0, 0:2, -2:]
print(f"First batch, first frame top-right corner shape: {top_right_4d.shape}")
print(f"First batch, first frame top-right values:\n{top_right_4d}")

print("\nTest completed successfully") 