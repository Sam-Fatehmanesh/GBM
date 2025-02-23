#!/usr/bin/env python
import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from BrainSimulator.models.oasis import OASIS

# Parameters
OUTPUT_H = 500        # output image height for each view
OUTPUT_W = 500        # output image width for each view
VIDEO_FPS = 1         # 1 frame per second
DURATION = 600        # 600 timepoints (frames)
OUTPUT_VIDEO = "visualization/spike_activity_opencv.avi"

os.makedirs("visualization", exist_ok=True)

##############################
# Load cell positions from MAT file
##############################
print("Loading position data from MAT file...")
mat_data = loadmat('jdataraw/subject_1/data_full.mat')
print("\nExtracting cell positions...")
cell_xyz = mat_data['data'][0,0]['CellXYZ']
if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
    cell_xyz = cell_xyz[0,0]
print("Original cell_xyz shape:", cell_xyz.shape)

# Filter out invalid cells
print("\nChecking for invalid cells...")
if 'IX_inval_anat' in mat_data['data'][0,0].dtype.names:
    invalid_indices = mat_data['data'][0,0]['IX_inval_anat']
    if isinstance(invalid_indices, np.ndarray) and invalid_indices.dtype == np.object_:
        invalid_indices = invalid_indices[0,0].flatten()
    valid_mask = np.ones(cell_xyz.shape[0], dtype=bool)
    valid_mask[invalid_indices - 1] = False  # Convert from MATLAB 1-based indexing
    num_removed = np.sum(~valid_mask)
    cell_xyz = cell_xyz[valid_mask]
    print(f"Removed {num_removed} invalid cells")
print("Filtered cell_xyz shape:", cell_xyz.shape)

# Normalize positions to [0,1]
cell_xyz_norm = (cell_xyz - cell_xyz.min(axis=0)) / (cell_xyz.max(axis=0) - cell_xyz.min(axis=0))

# Pre-calculate pixel indices for two views
# Top view (XY): use x and y (columns 0 and 1)
top_x = np.floor(cell_xyz_norm[:, 0] * (OUTPUT_W - 1)).astype(np.int32)
top_y = np.floor(cell_xyz_norm[:, 1] * (OUTPUT_H - 1)).astype(np.int32)
# Side view (XZ): use x and z (columns 0 and 2)
side_x = np.floor(cell_xyz_norm[:, 0] * (OUTPUT_W - 1)).astype(np.int32)
side_z = np.floor(cell_xyz_norm[:, 2] * (OUTPUT_H - 1)).astype(np.int32)

##############################
# Load and process calcium data
##############################
print("\nLoading calcium time series data...")
with h5py.File('jdataraw/subject_1/TimeSeries.h5', 'r') as f:
    calcium_data = f['CellResp'][:DURATION]
print("Time series shape:", calcium_data.shape)

# Ensure that the number of cells matches
assert cell_xyz.shape[0] == calcium_data.shape[1], f"Mismatch: {cell_xyz.shape[0]} vs {calcium_data.shape[1]}"

# Parameters for OASIS
g = 0.95  # Calcium decay factor - should be calibrated for your indicator
sigma = np.std(calcium_data)  # Noise level estimate

# Initialize arrays for deconvolved spikes
spike_data = np.zeros_like(calcium_data)

# Process each cell's calcium trace
print("\nDeconvolving spikes from calcium traces...")
oasis = OASIS(g=g)
for i in tqdm(range(calcium_data.shape[1])):
    # Get calcium trace for this cell
    y = calcium_data[:, i]
    
    # Normalize trace
    y = (y - np.mean(y)) / np.std(y)
    
    # Run OASIS (now returns binary spikes)
    _, s = oasis.fit(y, sigma=sigma)
    
    # Store binary spikes
    spike_data[:, i] = s

##############################
# Create video using OpenCV
##############################
print("\nGenerating spike activity video...")

# Set up video writer
video_writer = cv2.VideoWriter(OUTPUT_VIDEO,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             VIDEO_FPS,
                             (OUTPUT_W * 2, OUTPUT_H))  # Two views side-by-side

# Define bin edges for 2D histograms
x_edges = np.linspace(0, OUTPUT_W, OUTPUT_W + 1)
y_edges = np.linspace(0, OUTPUT_H, OUTPUT_H + 1)

# For each timepoint, show binary spike presence per pixel
for t in tqdm(range(DURATION), desc="Processing timepoints"):
    activity = spike_data[t, :]  # Binary spike activity for all cells at time t (already 0 or 1)
    
    # Top view: check for any spikes in each pixel
    top_sum, _, _ = np.histogram2d(top_y, top_x, bins=[y_edges, x_edges], weights=activity)
    top_img = (top_sum > 0).astype(np.uint8) * 255  # Binary image: 0 or 255
    
    # Side view: check for any spikes in each pixel
    side_sum, _, _ = np.histogram2d(side_z, side_x, bins=[y_edges, x_edges], weights=activity)
    side_img = (side_sum > 0).astype(np.uint8) * 255  # Binary image: 0 or 255
    
    # Convert to color for text overlay (white for spikes, black for no spikes)
    top_rgb = cv2.applyColorMap(top_img, cv2.COLORMAP_HOT)
    side_rgb = cv2.applyColorMap(side_img, cv2.COLORMAP_HOT)
    
    # Add text labels
    cv2.putText(top_rgb, 'Top View (XY) - Spikes', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(side_rgb, 'Side View (XZ) - Spikes', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add timestamp
    cv2.putText(top_rgb, f'Time: {t/VIDEO_FPS:.1f}s', (10, OUTPUT_H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine views horizontally
    combined = np.hstack([top_rgb, side_rgb])
    video_writer.write(combined)

video_writer.release()
print(f"\nSpike activity video saved as: {OUTPUT_VIDEO}")

# Save spike data for future use
print("\nSaving spike data...")
with h5py.File('visualization/spike_data.h5', 'w') as f:
    f.create_dataset('spikes', data=spike_data)
print("Spike data saved as: visualization/spike_data.h5") 