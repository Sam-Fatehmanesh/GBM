#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

# Set a random seed for reproducibility
np.random.seed(42)

def visualize_spikes_for_condition(condition, num_neurons=16, time_window=250, start_at=None):
    """
    Visualize calcium traces and detected spikes for a sample of neurons.
    
    Parameters
    ----------
    condition : str
        The condition to visualize (e.g., 'gain', 'flash', etc.)
    num_neurons : int
        Number of neurons to visualize
    time_window : int
        Number of timepoints to show
    start_at : int or None
        If provided, start visualization at this specific timepoint
    """
    # Paths to the data files
    original_file = f"raw_trace_data_zap/zap_traces_{condition}.h5"
    processed_file = f"spike_processed_data_zap/zap_spikes_{condition}.h5"
    
    # Check if files exist
    if not os.path.exists(original_file) or not os.path.exists(processed_file):
        print(f"Data files for condition '{condition}' not found.")
        return
    
    # Load data
    with h5py.File(original_file, 'r') as f:
        if 'traces' in f:
            original_traces = f['traces'][:]
        else:
            # Try the first key if 'traces' doesn't exist
            first_key = list(f.keys())[0]
            original_traces = f[first_key][:]
    
    with h5py.File(processed_file, 'r') as f:
        spikes = f['spikes'][:]
        calcium_denoised = f['calcium_denoised'][:]
        g_values = f['g_values'][:] if 'g_values' in f else np.zeros(original_traces.shape[1])
        b_values = f['b_values'][:] if 'b_values' in f else np.zeros(original_traces.shape[1])
    
    # Get the total number of timepoints and neurons
    num_timepoints, total_neurons = original_traces.shape
    
    # Set start and end times for visualization window
    if start_at is not None:
        # Use the specific starting point if provided
        start_time = min(start_at, num_timepoints - time_window)
    elif time_window > 0 and time_window < num_timepoints:
        # Otherwise choose a random starting point
        start_time = np.random.randint(0, num_timepoints - time_window)
    else:
        start_time = 0
    
    end_time = min(start_time + time_window, num_timepoints)
    
    print(f"Visualizing time window [{start_time} - {end_time}] out of {num_timepoints} timepoints")
    
    # Randomly select neurons that have some activity
    active_neurons = []
    max_attempts = 100  # Limit attempts to find active neurons
    
    for _ in range(max_attempts):
        if len(active_neurons) >= num_neurons:
            break
            
        # Select a random neuron
        neuron_idx = np.random.randint(0, total_neurons)
        
        # Skip if already selected
        if neuron_idx in active_neurons:
            continue
        
        # Check if there's any spike activity
        neuron_spikes = spikes[start_time:end_time, neuron_idx]
        if np.sum(neuron_spikes) > 0:
            active_neurons.append(neuron_idx)
    
    # If not enough active neurons found, just pick random ones
    if len(active_neurons) < num_neurons:
        potential_neurons = list(set(range(total_neurons)) - set(active_neurons))
        additional_neurons = random.sample(potential_neurons, min(num_neurons - len(active_neurons), len(potential_neurons)))
        active_neurons.extend(additional_neurons)
    
    # Limit to requested number
    selected_neurons = active_neurons[:num_neurons]
    
    # Create plot
    plt.figure(figsize=(15, 3 * num_neurons))
    gs = GridSpec(num_neurons, 1, figure=plt.gcf(), hspace=0.4)
    
    # Time axis for plotting
    time_axis = np.arange(start_time, end_time)
    
    # Plot each selected neuron
    for i, neuron_idx in enumerate(selected_neurons):
        ax = plt.subplot(gs[i, 0])
        
        # Get the data for this neuron
        original_trace = original_traces[start_time:end_time, neuron_idx]
        denoised_trace = calcium_denoised[start_time:end_time, neuron_idx]
        neuron_spikes = spikes[start_time:end_time, neuron_idx]
        g_value = g_values[neuron_idx]
        b_value = b_values[neuron_idx]
        
        # Normalize original trace for better visualization
        norm_original = (original_trace - np.mean(original_trace)) / (np.std(original_trace) + 1e-10)
        
        # Plot original trace
        ax.plot(time_axis, norm_original, 'gray', alpha=0.7, label='Original Trace')
        
        # Plot denoised trace
        ax.plot(time_axis, denoised_trace, 'blue', alpha=0.8, label='Denoised')
        
        # Plot spikes
        # Scale spikes height for visibility
        spike_height = np.max(norm_original) + 0.5
        scaled_spikes = neuron_spikes * spike_height
        ax.vlines(time_axis[neuron_spikes > 0], 0, spike_height, color='red', linewidth=1, label='Spikes')
        
        # Add labels with OASIS parameters
        ax.set_ylabel(f"Neuron {neuron_idx}\ng={g_value:.3f}, b={b_value:.3f}")
        
        # Only add x-label for the bottom plot
        if i == num_neurons - 1:
            ax.set_xlabel("Time (frames)")
        
        # Add legend only for the first plot
        if i == 0:
            ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Calcium Traces and Detected Spikes for {condition.capitalize()} Condition\nTime Window: {start_time}-{end_time}", fontsize=16)
    
    # Create output directory if it doesn't exist
    output_dir = "spike_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_file = os.path.join(output_dir, f"spikes_{condition}_t{start_time}-{end_time}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_file}")
    return output_file

def main():
    # Create output directory
    output_dir = "spike_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of all conditions (excluding 'full')
    conditions = ['full']
    
    # Generate visualizations for each condition
    for condition in conditions:
        try:
            # Create two visualizations per condition with different starting points
            visualize_spikes_for_condition(condition, num_neurons=16, time_window=1000, start_at=0)  # Start at beginning
            middle_point = 300  # Choose a middle point for visualization
            visualize_spikes_for_condition(condition, num_neurons=16, time_window=200, start_at=middle_point)
        except Exception as e:
            print(f"Error generating visualization for {condition}: {str(e)}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 