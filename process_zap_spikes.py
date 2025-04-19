#!/usr/bin/env python
import os
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys

# Import from original OASIS library instead of custom class

from oasis.functions import estimate_parameters
from oasis.functions import deconvolve


def process_trace_file(input_file, cell_positions, output_dir):
    """Process calcium trace data for a condition using OASIS.
    
    Parameters
    ----------
    input_file : str
        Path to the h5 trace file
    cell_positions : numpy.ndarray
        Array containing the positions of all cells (x, y, z)
    output_dir : str
        Path to save the processed data
    """
    print(f"\nProcessing trace data from: {input_file}")
    
    condition_name = os.path.basename(input_file).replace('zap_traces_', '').replace('.h5', '')
    print(f"Condition: {condition_name}")
    
    # Load calcium data
    print("Loading calcium time series data...")
    with h5py.File(input_file, 'r') as f:
        # Inspect the file structure
        print("Keys in the file:", list(f.keys()))
        # Assume traces are stored in 'traces' dataset
        if 'traces' in f:
            calcium_data = f['traces'][:]
        else:
            # Try the first key if 'traces' doesn't exist
            first_key = list(f.keys())[0]
            calcium_data = f[first_key][:]
    
    print("Time series shape:", calcium_data.shape)
    
    # Make sure the calcium data and cell positions match in dimension
    num_cells = cell_positions.shape[0]
    if calcium_data.shape[1] != num_cells:
        print(f"WARNING: Mismatch in cell count: {calcium_data.shape[1]} in traces vs {num_cells} positions")
        # Use the minimum to avoid index errors
        num_cells = min(calcium_data.shape[1], num_cells)
        calcium_data = calcium_data[:, :num_cells]
        cell_positions = cell_positions[:num_cells, :]
    
    # Initialize arrays for deconvolved spikes and parameters
    num_timepoints, num_cells = calcium_data.shape
    spike_data = np.zeros_like(calcium_data)
    calcium_denoised = np.zeros_like(calcium_data)
    g_values = np.zeros(num_cells)
    b_values = np.zeros(num_cells)
    lam_values = np.zeros(num_cells)

    penalty = 1
    
    # Process each cell's calcium trace
    print("Deconvolving spikes from calcium traces...")
    for i in tqdm(range(num_cells)):
        # Get calcium trace for this cell
        y = calcium_data[:, i]
        
        # Check for NaN values
        if np.isnan(y).any():
            print(f"Warning: NaN values found in cell {i}, replacing with zeros")
            y = np.nan_to_num(y)
        
        try:
            
           
            y = y.astype(np.float64)
            
            #pdb.set_trace() 
            est = estimate_parameters(y, p=1, fudge_factor=0.98)
            g = est[0]
            sn = est[1]
            g_values[i] = g[0]
            c, s, b, g, lam  = deconvolve(y, g=g, sn=sn, penalty=1)

            threshold = 0.333 * sn
            s[s >= threshold] = 1
            s[s < threshold] = 0
            # convert spike amplitues to binary, aka 1 if greater than 0, 0 otherwise
            #s = np.where(s > 0, 1, 0)
            
            calcium_denoised[:, i] = c
            spike_data[:, i] = s

            b_values[i] = b
            lam_values[i] = lam
            
        except Exception as e:
            print(f"Error processing cell {i}: {str(e)}")
            # Leave as zeros in case of error
    
    # Save processed data
    output_file = os.path.join(output_dir, f"zap_spikes_{condition_name}.h5")
    print(f"\nSaving processed data to: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        f.create_dataset('spikes', data=spike_data)
        f.create_dataset('calcium_denoised', data=calcium_denoised)
        f.create_dataset('cell_positions', data=cell_positions)
        f.create_dataset('g_values', data=g_values)
        f.create_dataset('b_values', data=b_values)
        f.create_dataset('lambda_values', data=lam_values)
        
        # Save parameters as attributes
        f.attrs['penalty'] = penalty  # L1 penalty
        f.attrs['num_timepoints'] = num_timepoints
        f.attrs['num_cells'] = num_cells
        f.attrs['condition'] = condition_name
    
    return output_file

def main():
    # Create output directory
    output_dir = "spike_processed_data_zap"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the neuron positions
    print("Loading neuron positions...")
    position_file = "zap_neuron_positions/neuron_positions_binned_72.csv"
    positions_df = pd.read_csv(position_file)
    cell_positions = positions_df[['x', 'y', 'z']].values
    print(f"Loaded positions for {len(positions_df)} neurons")
    
    # Find all trace files
    data_root = "raw_trace_data_zap"
    trace_files = []
    for entry in os.listdir(data_root):
        if entry.startswith("zap_traces_") and entry.endswith(".h5"):
            trace_path = os.path.join(data_root, entry)
            if os.path.isfile(trace_path):
                trace_files.append(trace_path)
    
    print(f"Found {len(trace_files)} trace files")
    
    # Process each file
    processed_files = []
    for trace_file in sorted(trace_files):
        try:
            output_file = process_trace_file(trace_file, cell_positions, output_dir)
            processed_files.append(output_file)
        except Exception as e:
            print(f"Error processing {trace_file}: {str(e)}")
    
    print("\nProcessing complete!")
    print("Processed data saved to:")
    for f in processed_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()