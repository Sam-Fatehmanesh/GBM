#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_spike_distribution(spike_data):
    """Analyze the distribution of spikes across time.
    
    Parameters
    ----------
    spike_data : ndarray
        Binary spike data of shape (n_timepoints, n_cells)
    
    Returns
    -------
    dict
        Dictionary containing distribution statistics
    """
    # Count number of spikes at each time point
    spikes_per_time = np.sum(spike_data, axis=1)
    
    # Calculate basic statistics
    stats_dict = {
        'mean': np.mean(spikes_per_time),
        'median': np.median(spikes_per_time),
        'std': np.std(spikes_per_time),
        'min': np.min(spikes_per_time),
        'max': np.max(spikes_per_time),
        'percentile_25': np.percentile(spikes_per_time, 25),
        'percentile_75': np.percentile(spikes_per_time, 75),
        'total_spikes': np.sum(spike_data),
        'sparsity': np.mean(spike_data),  # Fraction of total possible spikes that occurred
    }
    
    return stats_dict, spikes_per_time

def plot_distributions(spikes_per_time, output_dir, subject_name):
    """Create plots of spike distributions.
    
    Parameters
    ----------
    spikes_per_time : ndarray
        Number of spikes at each time point
    output_dir : str
        Directory to save plots
    subject_name : str
        Name of the subject for plot titles
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram of spikes per time point
    ax1.hist(spikes_per_time, bins=50, density=True, alpha=0.7)
    ax1.axvline(np.mean(spikes_per_time), color='r', linestyle='--', 
                label=f'Mean: {np.mean(spikes_per_time):.1f}')
    ax1.axvline(np.median(spikes_per_time), color='g', linestyle='--', 
                label=f'Median: {np.median(spikes_per_time):.1f}')
    
    # Fit and plot normal distribution
    mu, std = stats.norm.fit(spikes_per_time)
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p, 'k', linewidth=2, label='Normal fit')
    
    ax1.set_title(f'{subject_name}: Distribution of Number of Spikes per Time Point')
    ax1.set_xlabel('Number of Cells Spiking')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Plot 2: Time series of number of spikes
    ax2.plot(spikes_per_time, alpha=0.7)
    ax2.axhline(np.mean(spikes_per_time), color='r', linestyle='--', 
                label=f'Mean: {np.mean(spikes_per_time):.1f}')
    ax2.fill_between(range(len(spikes_per_time)), 
                     np.percentile(spikes_per_time, 25) * np.ones_like(spikes_per_time),
                     np.percentile(spikes_per_time, 75) * np.ones_like(spikes_per_time),
                     alpha=0.2, label='25-75 percentile')
    
    ax2.set_title(f'{subject_name}: Number of Spikes Over Time')
    ax2.set_xlabel('Time Point')
    ax2.set_ylabel('Number of Cells Spiking')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{subject_name}_spike_distribution.png'))
    plt.close()

def main():
    # Create output directory for plots
    output_dir = "spike_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each subject's processed spike data
    processed_dir = "processed_spikes"
    for filename in os.listdir(processed_dir):
        if filename.endswith("_processed.h5"):
            subject_name = filename.replace("_processed.h5", "")
            print(f"\nAnalyzing {subject_name}...")
            
            # Load spike data
            with h5py.File(os.path.join(processed_dir, filename), 'r') as f:
                spike_data = f['spikes'][:]
                n_timepoints = f.attrs['num_timepoints']
                n_cells = f.attrs['num_cells']
            
            # Analyze distributions
            stats_dict, spikes_per_time = analyze_spike_distribution(spike_data)
            
            # Print statistics
            print("\nSpike Distribution Statistics:")
            print(f"Total number of cells: {n_cells}")
            print(f"Total number of timepoints: {n_timepoints}")
            print(f"Total number of spikes: {stats_dict['total_spikes']:,}")
            print(f"Mean spikes per timepoint: {stats_dict['mean']:.2f}")
            print(f"Median spikes per timepoint: {stats_dict['median']:.2f}")
            print(f"Standard deviation: {stats_dict['std']:.2f}")
            print(f"Range: {stats_dict['min']:.0f} - {stats_dict['max']:.0f}")
            print(f"25th percentile: {stats_dict['percentile_25']:.2f}")
            print(f"75th percentile: {stats_dict['percentile_75']:.2f}")
            print(f"Overall sparsity: {stats_dict['sparsity']*100:.2f}%")
            
            # Create plots
            plot_distributions(spikes_per_time, output_dir, subject_name)
            print(f"\nPlots saved to: {output_dir}/{subject_name}_spike_distribution.png")

if __name__ == "__main__":
    main() 