#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from tqdm import tqdm
import argparse

# Import both OASIS implementations
from oasis.functions import deconvolve as oasis_lib_deconvolve  # From installed library
from GenerativeBrainModel.models.oasis import OASIS  # Built-in implementation

def process_and_visualize_subject(subject_dir, output_dir, oasis_method='library', num_neurons=10, 
                                  g=0.95, lambda_val=0, s_min=0):
    """Process and visualize calcium data for a single subject using specified OASIS method.
    
    Parameters
    ----------
    subject_dir : str
        Path to the subject's data directory
    output_dir : str
        Path to save the processed data and visualizations
    oasis_method : str
        Which OASIS implementation to use ('library' or 'built_in')
    num_neurons : int
        Number of neurons to visualize
    g : float
        Calcium decay factor (AR coefficient)
    lambda_val : float
        Sparsity penalty for L1 regularization
    s_min : float
        Minimum spike size (for L0 regularization)
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    print(f"\nProcessing subject data from: {subject_name}")
    
    # Load cell positions from MAT file
    print("Loading position data...")
    mat_data = loadmat(os.path.join(subject_dir, 'data_full.mat'))
    cell_xyz = mat_data['data'][0,0]['CellXYZ']
    if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
        cell_xyz = cell_xyz[0,0]
    
    # Filter out invalid cells if needed
    if 'IX_inval_anat' in mat_data['data'][0,0].dtype.names:
        invalid_indices = mat_data['data'][0,0]['IX_inval_anat']
        if isinstance(invalid_indices, np.ndarray) and invalid_indices.dtype == np.object_:
            invalid_indices = invalid_indices[0,0].flatten()
        valid_mask = np.ones(cell_xyz.shape[0], dtype=bool)
        valid_mask[invalid_indices - 1] = False  # Convert from MATLAB 1-based indexing
        cell_xyz = cell_xyz[valid_mask]
    
    # Load calcium data
    print("Loading calcium time series data...")
    with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
        calcium_data = f['CellResp'][:]
    
    # Ensure cell count matches
    assert cell_xyz.shape[0] == calcium_data.shape[1], \
        f"Cell count mismatch: {cell_xyz.shape[0]} vs {calcium_data.shape[1]}"
    
    # Initialize arrays for results
    num_timepoints, num_cells = calcium_data.shape
    spike_data = np.zeros_like(calcium_data)
    calcium_denoised = np.zeros_like(calcium_data)
    g_values = np.zeros(num_cells)
    
    # Process each cell's calcium trace
    print(f"Deconvolving spikes using {oasis_method} OASIS implementation...")
    
    if oasis_method == 'built_in':
        # Use built-in OASIS implementation
        oasis = OASIS(g=g)
        for i in tqdm(range(num_cells)):
            y = calcium_data[:, i]
            
            # Check for NaN values
            if np.isnan(y).any():
                print(f"Warning: NaN values found in cell {i}, replacing with zeros")
                y = np.nan_to_num(y)
            
            # Normalize trace to have min=0
            y_min = np.min(y)
            y = y - y_min
            
            # Run OASIS
            try:
                c, s = oasis.fit(y, sigma=np.std(y))
                calcium_denoised[:, i] = c
                spike_data[:, i] = s
                g_values[i] = g  # Use fixed g
            except Exception as e:
                print(f"Error processing cell {i}: {str(e)}")
    
    else:  # oasis_method == 'library'
        # Use installed OASIS library
        for i in tqdm(range(num_cells)):
            y = calcium_data[:, i]
            
            # Check for NaN values
            if np.isnan(y).any():
                print(f"Warning: NaN values found in cell {i}, replacing with zeros")
                y = np.nan_to_num(y)
            
            # Normalize trace to have min=0
            y_min = np.min(y)
            y = y - y_min
            
            try:
                # Use deconvolve function with specified parameters
                method = 'l0' if s_min > 0 else 'l1'
                
                # Handle the case where the deconvolve function returns a scalar
                # (which happens when there's an issue with the trace)
                result = oasis_lib_deconvolve(
                    y, 
                    g=g,  # Can be None to estimate g
                    penalty=lambda_val,
                    smin=s_min,
                    method=method
                )
                
                # Check if the result is a tuple (normal case) or a float (error case)
                if isinstance(result, tuple) and len(result) == 5:
                    c, s, b, g_est, lam = result
                    
                    # Store results
                    calcium_denoised[:, i] = c
                    spike_data[:, i] = s
                    g_values[i] = g_est if g is None else g
                else:
                    # Handle the case where deconvolve returns a scalar
                    print(f"Warning: Invalid result for cell {i}, returning zeros")
                    calcium_denoised[:, i] = 0
                    spike_data[:, i] = 0
                    g_values[i] = g if g is not None else 0.95
                
            except Exception as e:
                print(f"Error processing cell {i}: {str(e)}")
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{subject_name}_spikes_{oasis_method}.h5")
    print(f"Saving processed data to: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        f.create_dataset('spikes', data=spike_data)
        f.create_dataset('calcium_denoised', data=calcium_denoised)
        f.create_dataset('cell_positions', data=cell_xyz)
        f.create_dataset('g_values', data=g_values)
        
        # Save parameters as attributes
        f.attrs['oasis_method'] = oasis_method
        f.attrs['g_input'] = g
        f.attrs['lambda'] = lambda_val
        f.attrs['s_min'] = s_min
        f.attrs['num_timepoints'] = num_timepoints
        f.attrs['num_cells'] = num_cells
    
    # Create visualizations
    print("Generating visualizations...")
    pdf_file = os.path.join(output_dir, f"{subject_name}_visualization_{oasis_method}.pdf")
    
    with PdfPages(pdf_file) as pdf:
        # Randomly select neurons to visualize if there are more than requested
        if num_cells > num_neurons:
            neuron_indices = np.random.choice(num_cells, num_neurons, replace=False)
        else:
            neuron_indices = np.arange(num_cells)
        
        for idx in neuron_indices:
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            
            # Plot original trace
            ax[0].plot(calcium_data[:, idx], 'k', label='Original F')
            ax[0].set_title(f'Neuron {idx} - Original Calcium Trace')
            ax[0].set_ylabel('Fluorescence')
            ax[0].legend()
            
            # Plot denoised trace
            ax[1].plot(calcium_denoised[:, idx], 'b', label='Denoised Calcium')
            ax[1].set_title(f'Denoised Calcium Trace (g={g_values[idx]:.3f})')
            ax[1].set_ylabel('Fluorescence')
            ax[1].legend()
            
            # Plot spike data
            ax[2].plot(spike_data[:, idx], 'r', label='Inferred Spikes')
            ax[2].set_title('Inferred Spike Events')
            ax[2].set_xlabel('Time (frames)')
            ax[2].set_ylabel('Spike Amplitude')
            ax[2].legend()
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
        # Generate a summary page with both raw and processed data overlaid
        fig, axes = plt.subplots(num_neurons, 1, figsize=(12, 3*num_neurons))
        if num_neurons == 1:
            axes = [axes]  # Make it iterable
            
        for i, idx in enumerate(neuron_indices):
            # Normalize traces for better visualization
            raw = calcium_data[:, idx]
            denoised = calcium_denoised[:, idx]
            spikes = spike_data[:, idx]
            
            # Scale everything to 0-1 range for comparison
            raw_scaled = (raw - np.min(raw)) / (np.max(raw) - np.min(raw) + 1e-10)
            
            # Plot overlaid data
            axes[i].plot(raw_scaled, 'k', alpha=0.5, label='Original F')
            axes[i].plot(denoised, 'b', alpha=0.7, label='Denoised')
            
            # Plot spikes with different y-scale
            spike_max = np.max(spikes) if np.max(spikes) > 0 else 1
            axes[i].plot(spikes/spike_max, 'r', label='Spikes')
            
            axes[i].set_title(f'Neuron {idx} - Combined View (g={g_values[idx]:.3f})')
            axes[i].legend(loc='upper right')
            
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"Visualization saved to: {pdf_file}")
    return output_file, pdf_file

def main():
    parser = argparse.ArgumentParser(description='Process and visualize calcium imaging data with OASIS.')
    parser.add_argument('--data_root', type=str, default='raw_trace_data_2018',
                       help='Root directory containing subject data folders')
    parser.add_argument('--output_dir', type=str, default='spike_processed_data_2018',
                       help='Directory to save processed data and visualizations')
    parser.add_argument('--method', type=str, choices=['library', 'built_in', 'both'], default='both',
                       help='OASIS implementation to use')
    parser.add_argument('--num_subjects', type=int, default=None,
                       help='Number of subjects to process (None for all)')
    parser.add_argument('--num_neurons', type=int, default=10,
                       help='Number of neurons to visualize per subject')
    parser.add_argument('--g', type=float, default=0.95,
                       help='Calcium decay factor (AR coefficient). Set to None for auto-estimation')
    parser.add_argument('--lambda_val', type=float, default=0,
                       help='Sparsity penalty for L1 regularization')
    parser.add_argument('--s_min', type=float, default=0,
                       help='Minimum spike size (for L0 regularization)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = []
    for entry in os.listdir(args.data_root):
        if entry.startswith("subject_"):
            subject_path = os.path.join(args.data_root, entry)
            if os.path.isdir(subject_path):
                subject_dirs.append(subject_path)
    
    subject_dirs.sort()
    print(f"Found {len(subject_dirs)} subject directories")
    
    # Limit number of subjects if specified
    if args.num_subjects is not None:
        subject_dirs = subject_dirs[:args.num_subjects]
        print(f"Processing {len(subject_dirs)} subjects")
    
    # Process each subject
    methods = []
    if args.method == 'both':
        methods = ['library', 'built_in']
    else:
        methods = [args.method]
    
    results = []
    for subject_dir in subject_dirs:
        for method in methods:
            try:
                output_file, pdf_file = process_and_visualize_subject(
                    subject_dir, 
                    args.output_dir,
                    oasis_method=method,
                    num_neurons=args.num_neurons,
                    g=args.g,
                    lambda_val=args.lambda_val,
                    s_min=args.s_min
                )
                results.append((subject_dir, method, output_file, pdf_file))
            except Exception as e:
                print(f"Error processing {subject_dir} with {method}: {str(e)}")
    
    print("\nProcessing complete!")
    print("Results:")
    for subject_dir, method, output_file, pdf_file in results:
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        print(f"Subject {subject_name} ({method})")
        print(f"  - Data: {output_file}")
        print(f"  - Visualization: {pdf_file}")

if __name__ == "__main__":
    main() 