#!/usr/bin/env python
import os
import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from GenerativeBrainModel.models.oasis import OASIS

def process_subject(subject_dir, output_dir):
    """Process calcium data for a single subject using OASIS.
    
    Parameters
    ----------
    subject_dir : str
        Path to the subject's data directory
    output_dir : str
        Path to save the processed data
    """
    print(f"\nProcessing subject data from: {subject_dir}")
    
    # Load cell positions from MAT file
    print("Loading position data...")
    mat_data = loadmat(os.path.join(subject_dir, 'data_full.mat'))
    cell_xyz = mat_data['data'][0,0]['CellXYZ']
    if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
        cell_xyz = cell_xyz[0,0]
    print("Original cell_xyz shape:", cell_xyz.shape)
    
    # Filter out invalid cells
    print("Checking for invalid cells...")
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
    
    # Load calcium data
    print("Loading calcium time series data...")
    with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
        calcium_data = f['CellResp'][:]
    print("Time series shape:", calcium_data.shape)
    
    # Ensure that the number of cells matches
    assert cell_xyz.shape[0] == calcium_data.shape[1], \
        f"Mismatch: {cell_xyz.shape[0]} vs {calcium_data.shape[1]}"
    
    # Parameters for OASIS
    g = 0.95  # Calcium decay factor
    sigma = np.std(calcium_data)  # Noise level estimate
    
    # Initialize arrays for deconvolved spikes
    spike_data = np.zeros_like(calcium_data)
    calcium_denoised = np.zeros_like(calcium_data)
    
    # Process each cell's calcium trace
    print("Deconvolving spikes from calcium traces...")
    oasis = OASIS(g=g)
    for i in tqdm(range(calcium_data.shape[1])):
        # Get calcium trace for this cell
        y = calcium_data[:, i]
        
        # Normalize trace
        y = (y - np.mean(y)) / np.std(y)
        
        # Run OASIS
        c, s = oasis.fit(y, sigma=sigma)
        
        # Store results
        calcium_denoised[:, i] = c
        spike_data[:, i] = s
    
    # Save processed data
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    output_file = os.path.join(output_dir, f"{subject_name}_processed.h5")
    print(f"\nSaving processed data to: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Create groups
        f.create_dataset('spikes', data=spike_data)
        f.create_dataset('calcium_denoised', data=calcium_denoised)
        f.create_dataset('cell_positions', data=cell_xyz)
        
        # Save parameters as attributes
        f.attrs['g'] = g
        f.attrs['sigma'] = sigma
        f.attrs['num_timepoints'] = calcium_data.shape[0]
        f.attrs['num_cells'] = calcium_data.shape[1]
    
    return output_file

def main():
    # Create output directory
    output_dir = "processed_spikes"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subject directories
    data_root = "jdataraw"
    subject_dirs = []
    for entry in os.listdir(data_root):
        if entry.startswith("subject_"):
            subject_path = os.path.join(data_root, entry)
            if os.path.isdir(subject_path):
                subject_dirs.append(subject_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # Process each subject
    processed_files = []
    for subject_dir in sorted(subject_dirs):
        try:
            output_file = process_subject(subject_dir, output_dir)
            processed_files.append(output_file)
        except Exception as e:
            print(f"Error processing {subject_dir}: {str(e)}")
    
    print("\nProcessing complete!")
    print("Processed data saved to:")
    for f in processed_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main() 