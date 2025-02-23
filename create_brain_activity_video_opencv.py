#!/usr/bin/env python
import os
import cv2
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# Parameters
OUTPUT_H = 500        # output image height for each view
OUTPUT_W = 500        # output image width for each view
VIDEO_FPS = 1         # 1 frame per second
DURATION = 600        # 600 timepoints (frames)
OUTPUT_VIDEO = "visualization/brain_activity_opencv.avi"

os.makedirs("visualization", exist_ok=True)

##############################
# Load cell positions from MAT file (v5 file)
##############################
print("Loading position data from MAT file...")
mat_data = loadmat('jdataraw/subject_1/data_full.mat')
# Our file has a structure stored under 'data'; extract the field 'CellXYZ'
print("\nExtracting cell positions...")
cell_xyz = mat_data['data'][0,0]['CellXYZ']
if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
    cell_xyz = cell_xyz[0,0]
print("Original cell_xyz shape:", cell_xyz.shape)

# Filter out invalid cells using the field 'IX_inval_anat'
print("\nChecking for invalid cells...")
if 'IX_inval_anat' in mat_data['data'][0,0].dtype.names:
    invalid_indices = mat_data['data'][0,0]['IX_inval_anat']
    if isinstance(invalid_indices, np.ndarray) and invalid_indices.dtype == np.object_:
        invalid_indices = invalid_indices[0,0].flatten()
    valid_mask = np.ones(cell_xyz.shape[0], dtype=bool)
    valid_mask[invalid_indices - 1] = False  # Convert from MATLAB 1-based indexing to Python 0-based
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
# Load time series data (cell responses) from HDF5 file
##############################
print("\nLoading time series data...")
with h5py.File('jdataraw/subject_1/TimeSeries.h5', 'r') as f:
    # Load first 600 timepoints; cell_resp shape: (600, number_of_cells)
    cell_resp = f['CellResp'][:600]
print("Time series shape:", cell_resp.shape)

# Ensure that the number of cells in positions matches time series data
assert cell_xyz.shape[0] == cell_resp.shape[1], f"Mismatch: {cell_xyz.shape[0]} vs {cell_resp.shape[1]}"

##############################
# Create video using OpenCV
##############################
print("\nGenerating video using OpenCV...")

# Set up video writer:
video_writer = cv2.VideoWriter(OUTPUT_VIDEO,
                               cv2.VideoWriter_fourcc(*'XVID'),
                               VIDEO_FPS,
                               (OUTPUT_W * 2, OUTPUT_H))  # Two views side-by-side

# Define bin edges for our 2D histograms:
x_edges = np.linspace(0, OUTPUT_W, OUTPUT_W + 1)
y_edges = np.linspace(0, OUTPUT_H, OUTPUT_H + 1)

# For each timepoint, calculate the average activity per pixel in each view:
for t in tqdm(range(DURATION), desc="Processing timepoints"):
    activity = cell_resp[t, :]  # Activity for all cells at time t

    # Top view: accumulate activity using top_x, top_y
    top_sum, _, _ = np.histogram2d(top_y, top_x, bins=[y_edges, x_edges], weights=activity)
    top_count, _, _ = np.histogram2d(top_y, top_x, bins=[y_edges, x_edges])
    # Compute mean activity per pixel (avoid division by zero)
    top_img = np.where(top_count > 0, top_sum / top_count, 0)

    # Side view: accumulate activity using side_x, side_z
    side_sum, _, _ = np.histogram2d(side_z, side_x, bins=[y_edges, x_edges], weights=activity)
    side_count, _, _ = np.histogram2d(side_z, side_x, bins=[y_edges, x_edges])
    side_img = np.where(side_count > 0, side_sum / side_count, 0)

    # Normalize each image to [0,255] per frame
    def normalize(image):
        if image.max() > image.min():
            norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(image, dtype=np.uint8)
        return norm

    top_norm = normalize(top_img)
    side_norm = normalize(side_img)

    # Convert the gray image to color (to permit text overlays if needed)
    top_rgb = cv2.cvtColor(top_norm, cv2.COLOR_GRAY2BGR)
    side_rgb = cv2.cvtColor(side_norm, cv2.COLOR_GRAY2BGR)

    # Add text labels to each view
    cv2.putText(top_rgb, 'Top View (XY)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(side_rgb, 'Side View (XZ)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine the two views horizontally
    combined = np.hstack([top_rgb, side_rgb])
    video_writer.write(combined)

video_writer.release()
print(f"\nVideo saved as: {OUTPUT_VIDEO}") 