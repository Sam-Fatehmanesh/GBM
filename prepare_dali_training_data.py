#!/usr/bin/env python
import os
import h5py
import numpy as np
from tqdm import tqdm
import shutil

def prepare_dali_training_file(input_file, output_dir):
    """
    Prepare an h5 file for DALI-based training from the 2018 processed spike data.
    
    Args:
        input_file (str): Path to the input h5 file
        output_dir (str): Directory to save the output file
    
    Returns:
        str: Path to the output file
    """
    print(f"Processing {os.path.basename(input_file)}")
    
    subject_name = os.path.basename(input_file).split('_spikes')[0]
    output_file = os.path.join(output_dir, f"{subject_name}_processed.h5")
    
    with h5py.File(input_file, 'r') as f_in:
        # Check if necessary datasets exist
        required_keys = ['spikes', 'cell_positions']
        for key in required_keys:
            if key not in f_in:
                print(f"Error: {key} not found in {input_file}")
                return None
        
        # Get data
        spikes = f_in['spikes'][:]  # Shape: (num_timepoints, num_cells)
        cell_positions = f_in['cell_positions'][:]  # Shape: (num_cells, 3)
        
        # Get attributes
        num_timepoints = f_in.attrs.get('num_timepoints', spikes.shape[0])
        num_cells = f_in.attrs.get('num_cells', spikes.shape[1])
        
        print(f"  - Spikes shape: {spikes.shape}")
        print(f"  - Cell positions shape: {cell_positions.shape}")
        
        # Check if there are any null or nan values in cell_positions
        if np.isnan(cell_positions).any():
            print(f"Warning: NaN values found in cell positions for {subject_name}")
            # Replace NaN values with 0 
            cell_positions = np.nan_to_num(cell_positions)
        
        # Create output file
        with h5py.File(output_file, 'w') as f_out:
            # Create datasets with same structure (preserve shape and order)
            f_out.create_dataset('spikes', data=spikes)
            f_out.create_dataset('cell_positions', data=cell_positions)
            
            # Save attributes
            f_out.attrs['num_timepoints'] = num_timepoints
            f_out.attrs['num_cells'] = num_cells
            f_out.attrs['subject'] = subject_name
    
    print(f"  - Created {output_file}")
    return output_file

def main():
    # Set up directories
    input_dir = "spike_processed_data_2018"
    output_dir = "training_spike_data_2018"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all spike h5 files
    input_files = []
    for file in os.listdir(input_dir):
        if file.endswith('_spikes.h5') or file.endswith('_spikes_built_in.h5'):
            input_files.append(os.path.join(input_dir, file))
    
    input_files.sort()
    print(f"Found {len(input_files)} spike data files")
    
    # Process each file
    output_files = []
    for input_file in tqdm(input_files, desc="Converting files"):
        output_file = prepare_dali_training_file(input_file, output_dir)
        if output_file:
            output_files.append(output_file)
    
    print("\nPreparation complete!")
    print(f"Created {len(output_files)} training-ready files in {output_dir}:")
    for output_file in output_files:
        print(f"  - {os.path.basename(output_file)}")

if __name__ == "__main__":
    main() 