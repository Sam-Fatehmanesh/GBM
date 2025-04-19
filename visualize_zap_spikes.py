#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import argparse
import random

def visualize_zap_spikes(input_file, output_dir, num_neurons=10):
    """Visualize deconvolved spike data for ZAP datasets.
    
    Parameters
    ----------
    input_file : str
        Path to the processed h5 spike file
    output_dir : str
        Path to save the visualizations
    num_neurons : int
        Number of neurons to visualize
    """
    # Extract condition name from filename
    condition_name = os.path.basename(input_file).replace('zap_spikes_', '').replace('.h5', '')
    print(f"\nVisualizing spikes for condition: {condition_name}")
    
    # Skip if this condition is already processed
    pdf_file = os.path.join(output_dir, f"{condition_name}_visualization.pdf")
    if os.path.exists(pdf_file):
        print(f"Condition {condition_name} is already visualized. Skipping.")
        return pdf_file
    
    # Load processed data
    print(f"Loading processed data from: {input_file}")
    with h5py.File(input_file, 'r') as f:
        spike_data = f['spikes'][:]
        calcium_denoised = f['calcium_denoised'][:]
        cell_positions = f['cell_positions'][:]
        g_values = f['g_values'][:]
        
        # Get attributes
        num_timepoints = f.attrs['num_timepoints']
        num_cells = f.attrs['num_cells']
        
    print(f"Loaded data with {num_timepoints} timepoints and {num_cells} cells")
    
    # Now load the original trace data
    trace_file = input_file.replace('spike_processed_data_zap', 'raw_trace_data_zap').replace('zap_spikes_', 'zap_traces_')
    if os.path.exists(trace_file):
        print(f"Loading original trace data from: {trace_file}")
        with h5py.File(trace_file, 'r') as f:
            # Check keys in the file
            keys = list(f.keys())
            if 'traces' in keys:
                original_data = f['traces'][:]
            else:
                # Try the first key if 'traces' doesn't exist
                original_data = f[keys[0]][:]
            
            # Ensure shapes match
            if original_data.shape[1] > calcium_denoised.shape[1]:
                original_data = original_data[:, :calcium_denoised.shape[1]]
            elif original_data.shape[1] < calcium_denoised.shape[1]:
                print(f"Warning: Original data has fewer cells ({original_data.shape[1]}) than processed data ({calcium_denoised.shape[1]})")
                # Fill with zeros
                temp = np.zeros_like(calcium_denoised)
                temp[:, :original_data.shape[1]] = original_data
                original_data = temp
    else:
        print(f"Warning: Original trace file not found at {trace_file}")
        original_data = None
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    with PdfPages(pdf_file) as pdf:
        # Randomly select neurons to visualize if there are more than requested
        if num_cells > num_neurons:
            neuron_indices = random.sample(range(num_cells), num_neurons)
        else:
            neuron_indices = range(num_cells)
        
        # Create individual visualizations for each selected neuron
        for i, idx in enumerate(tqdm(neuron_indices, desc="Generating neuron plots")):
            if original_data is not None:
                fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                
                # Plot original calcium trace
                ax[0].plot(original_data[:, idx], 'k', label='Original Calcium')
                ax[0].set_title(f'Neuron {idx} - Original Calcium Trace')
                ax[0].set_ylabel('Fluorescence')
                ax[0].legend()
                
                # Plot denoised calcium trace
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
            else:
                fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                # Plot denoised calcium trace
                ax[0].plot(calcium_denoised[:, idx], 'b', label='Denoised Calcium')
                ax[0].set_title(f'Neuron {idx} - Denoised Calcium Trace (g={g_values[idx]:.3f})')
                ax[0].set_ylabel('Fluorescence')
                ax[0].legend()
                
                # Plot spike data
                ax[1].plot(spike_data[:, idx], 'r', label='Inferred Spikes')
                ax[1].set_title('Inferred Spike Events')
                ax[1].set_xlabel('Time (frames)')
                ax[1].set_ylabel('Spike Amplitude')
                ax[1].legend()
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
        # Generate a summary page with traces overlaid for each neuron
        for i in range(0, len(neuron_indices), 3):
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Handle case where there are fewer than 3 neurons left
            neurons_to_plot = neuron_indices[i:min(i+3, len(neuron_indices))]
            
            for j, idx in enumerate(neurons_to_plot):
                # Get the data for this neuron
                if original_data is not None:
                    original = original_data[:, idx]
                    # Scale the original signal for better visualization
                    original_scaled = (original - np.min(original)) / (np.max(original) - np.min(original) + 1e-10)
                    axes[j].plot(original_scaled, 'k', alpha=0.5, label='Original')
                
                denoised = calcium_denoised[:, idx]
                spikes = spike_data[:, idx]
                
                # Scale the denoised signal for better visualization
                denoised_scaled = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised) + 1e-10)
                
                # Plot overlaid data
                axes[j].plot(denoised_scaled, 'b', alpha=0.7, label='Denoised')
                
                # Plot spikes with different y-scale
                spike_max = np.max(spikes) if np.max(spikes) > 0 else 1
                axes[j].plot(spikes/spike_max, 'r', label='Spikes')
                
                # Add position information to the title
                pos = cell_positions[idx]
                axes[j].set_title(f'Neuron {idx} - Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - g={g_values[idx]:.3f}')
                axes[j].legend(loc='upper right')
                
            # Hide unused subplots
            for j in range(len(neurons_to_plot), 3):
                axes[j].set_visible(False)
                
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        
        # Add a page with spike rate statistics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Calculate spike rate per neuron (mean activity)
        spike_rates = np.mean(spike_data > 0, axis=0)
        
        # Plot 1: Histogram of spike rates
        axes[0, 0].hist(spike_rates, bins=30)
        axes[0, 0].set_title('Histogram of Spike Rates')
        axes[0, 0].set_xlabel('Spike Rate (probability of spiking per frame)')
        axes[0, 0].set_ylabel('Number of Neurons')
        
        # Plot 2: Scatter plot of spike rate vs. Z position
        axes[0, 1].scatter(cell_positions[:, 2], spike_rates, alpha=0.5)
        axes[0, 1].set_title('Spike Rate vs. Z Position')
        axes[0, 1].set_xlabel('Z Position')
        axes[0, 1].set_ylabel('Spike Rate')
        
        # Plot 3: Total spikes over time (averaged across neurons)
        total_spikes_per_frame = np.sum(spike_data > 0, axis=1)
        axes[1, 0].plot(total_spikes_per_frame)
        axes[1, 0].set_title('Total Active Neurons Over Time')
        axes[1, 0].set_xlabel('Time Frame')
        axes[1, 0].set_ylabel('Number of Active Neurons')
        
        # Plot 4: G values distribution
        axes[1, 1].hist(g_values, bins=30)
        axes[1, 1].set_title('Distribution of G Values')
        axes[1, 1].set_xlabel('G Value')
        axes[1, 1].set_ylabel('Number of Neurons')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    print(f"Visualization saved to: {pdf_file}")
    return pdf_file

def main():
    parser = argparse.ArgumentParser(description='Visualize ZAP spike data from processed h5 files.')
    parser.add_argument('--data_dir', type=str, default='spike_processed_data_zap',
                       help='Directory containing processed spike data files')
    parser.add_argument('--output_dir', type=str, default='spike_processed_data_zap/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_neurons', type=int, default=10,
                       help='Number of neurons to visualize per condition')
    parser.add_argument('--condition', type=str, default=None,
                       help='Specific condition to visualize (default: visualize all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all processed spike files
    spike_files = []
    for entry in os.listdir(args.data_dir):
        if entry.startswith("zap_spikes_") and entry.endswith(".h5"):
            spike_path = os.path.join(args.data_dir, entry)
            if os.path.isfile(spike_path):
                # If a specific condition is specified, only include that one
                if args.condition:
                    condition_name = entry.replace('zap_spikes_', '').replace('.h5', '')
                    if condition_name == args.condition:
                        spike_files.append(spike_path)
                else:
                    spike_files.append(spike_path)
    
    print(f"Found {len(spike_files)} processed spike files")
    
    # Process each file
    pdf_files = []
    for spike_file in sorted(spike_files):
        try:
            pdf_file = visualize_zap_spikes(
                spike_file, 
                args.output_dir,
                num_neurons=args.num_neurons
            )
            pdf_files.append(pdf_file)
        except Exception as e:
            condition_name = os.path.basename(spike_file).replace('zap_spikes_', '').replace('.h5', '')
            print(f"Error visualizing {condition_name}: {str(e)}")
    
    print("\nVisualization complete!")
    print("PDF visualizations saved to:")
    for pdf in pdf_files:
        print(f"  - {pdf}")

if __name__ == "__main__":
    main() 