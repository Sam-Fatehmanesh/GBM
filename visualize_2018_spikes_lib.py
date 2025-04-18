#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from tqdm import tqdm
import argparse
import pdb
from oasis.functions import estimate_parameters
from oasis.functions import deconvolve

# Import only the built-in OASIS implementation

def process_and_visualize_subject(subject_dir, output_dir, num_neurons=10, g=0.028):
    """Process and visualize calcium data for a single subject using the built-in OASIS method.
    
    Parameters
    ----------
    subject_dir : str
        Path to the subject's data directory
    output_dir : str
        Path to save the processed data and visualizations
    num_neurons : int
        Number of neurons to visualize
    g : float
        Calcium decay factor (AR coefficient)
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    print(f"\nProcessing subject data from: {subject_name}")
    
    # Skip if this subject is already processed
    pdf_file = os.path.join(output_dir, f"{subject_name}_visualization.pdf")
    if os.path.exists(pdf_file):
        print(f"Subject {subject_name} is already processed. Skipping.")
        return None, pdf_file
    
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
    print(f"Deconvolving spikes using built-in OASIS implementation...")
    
    # Use built-in OASIS implementation
    # oasis = OASIS(g=g)
    for i in tqdm(range(num_cells)):
        y = calcium_data[:, i]
        
        # Check for NaN values
        if np.isnan(y).any():
            print(f"Warning: NaN values found in cell {i}, replacing with zeros")
            y = np.nan_to_num(y)
        
        # # Normalize trace to have min=0
        # y_min = np.min(y)
        # y = y - y_min
        
        # Run OASIS
        try:
            # c, s = oasis.fit(y, sigma=np.std(y))
            # convet to float64
            y = y.astype(np.float64)
            
            #pdb.set_trace() 
            est = estimate_parameters(y, p=1, fudge_factor=0.98)
            g = est[0]
            sn = est[1]
            g_values[i] = g[0]
            out = deconvolve(y, g=g, sn=sn, penalty=1)
            c = out[0]
            s = out[1]
            threshold = 0.333 * sn
            s[s >= threshold] = 1
            s[s < threshold] = 0
            # convert spike amplitues to binary, aka 1 if greater than 0, 0 otherwise
            #s = np.where(s > 0, 1, 0)
            
            calcium_denoised[:, i] = c
            spike_data[:, i] = s
        except Exception as e:
            print(f"Error processing cell {i}: {str(e)}")
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{subject_name}_spikes.h5")
    print(f"Saving processed data to: {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        f.create_dataset('spikes', data=spike_data)
        f.create_dataset('calcium_denoised', data=calcium_denoised)
        f.create_dataset('cell_positions', data=cell_xyz)
        f.create_dataset('g_values', data=g_values)
        
        # Save parameters as attributes
        f.attrs['g'] = g
        f.attrs['num_timepoints'] = num_timepoints
        f.attrs['num_cells'] = num_cells
    
    # Create visualizations
    print("Generating visualizations...")
    
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
    parser = argparse.ArgumentParser(description='Process and visualize calcium imaging data with built-in OASIS.')
    parser.add_argument('--data_root', type=str, default='raw_trace_data_2018',
                       help='Root directory containing subject data folders')
    parser.add_argument('--output_dir', type=str, default='spike_processed_data_2018',
                       help='Directory to save processed data and visualizations')
    parser.add_argument('--num_neurons', type=int, default=10,
                       help='Number of neurons to visualize per subject')
    parser.add_argument('--g', type=float, default=0.99,
                       help='Calcium decay factor (AR coefficient)')
    parser.add_argument('--skip_subjects', type=str, default='',
                       help='Comma-separated list of subjects to skip (already processed)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse skip subjects
    skip_subjects = [s.strip() for s in args.skip_subjects.split(',')]
    print(f"Skipping already processed subjects: {skip_subjects}")
    
    # Find all subject directories
    subject_dirs = []
    for entry in os.listdir(args.data_root):
        if entry.startswith("subject_") and entry not in skip_subjects:
            subject_path = os.path.join(args.data_root, entry)
            if os.path.isdir(subject_path):
                subject_dirs.append(subject_path)
    
    subject_dirs.sort()
    print(f"Found {len(subject_dirs)} subject directories to process")
    
    # Process each subject
    results = []
    for subject_dir in subject_dirs:
        try:
            output_file, pdf_file = process_and_visualize_subject(
                subject_dir, 
                args.output_dir,
                num_neurons=args.num_neurons,
                g=args.g
            )
            if output_file:  # Could be None if already processed
                results.append((subject_dir, output_file, pdf_file))
        except Exception as e:
            subject_name = os.path.basename(os.path.normpath(subject_dir))
            print(f"Error processing {subject_name}: {str(e)}")
    
    print("\nProcessing complete!")
    print("Results:")
    for subject_dir, output_file, pdf_file in results:
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        print(f"Subject {subject_name}")
        print(f"  - Data: {output_file}")
        print(f"  - Visualization: {pdf_file}")

if __name__ == "__main__":
    main() 