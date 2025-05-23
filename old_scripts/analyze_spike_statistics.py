#!/usr/bin/env python3
"""
Analyze spike statistics for each subject file in training_spike_data_2018.

This script reads all the processed spike files and calculates various
statistics to understand the distribution of spike data across subjects.
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from pathlib import Path

def analyze_spike_file(filepath):
    """Analyze a single spike file and return statistics."""
    subject_name = os.path.basename(filepath).replace('_processed.h5', '')
    
    with h5py.File(filepath, 'r') as f:
        # Get spike data
        spikes = f['spikes'][:]
        cell_positions = f['cell_positions'][:]
        
        # Get z distribution
        z_values = cell_positions[:, 2]
        z_counts = np.bincount(np.round(z_values * 10).astype(int))
        
        # Calculate spikes per timepoint
        total_spikes_per_timepoint = np.sum(spikes > 0, axis=1)
        avg_spikes_per_timepoint = np.mean(total_spikes_per_timepoint)
        median_spikes_per_timepoint = np.median(total_spikes_per_timepoint)
        
        # Basic statistics
        stats = {
            'subject': subject_name,
            'num_timepoints': spikes.shape[0],
            'num_cells': spikes.shape[1],
            'mean_spike_value': float(np.mean(spikes)),
            'median_spike_value': float(np.median(spikes)),
            'max_spike_value': float(np.max(spikes)),
            'min_spike_value': float(np.min(spikes)),
            'std_spike_value': float(np.std(spikes)),
            
            # Activity statistics
            'active_ratio': float(np.mean(spikes > 0)),  # Fraction of active cells
            'mean_active_value': float(np.mean(spikes[spikes > 0])) if np.any(spikes > 0) else 0,
            
            # Per-frame statistics
            'mean_active_cells_per_frame': float(np.mean(np.sum(spikes > 0, axis=1))),
            'max_active_cells_per_frame': int(np.max(np.sum(spikes > 0, axis=1))),
            'min_active_cells_per_frame': int(np.min(np.sum(spikes > 0, axis=1))),
            
            # Spikes per timepoint statistics
            'avg_spikes_per_timepoint': float(avg_spikes_per_timepoint),
            'median_spikes_per_timepoint': float(median_spikes_per_timepoint),
            'max_spikes_per_timepoint': int(np.max(total_spikes_per_timepoint)),
            'min_spikes_per_timepoint': int(np.min(total_spikes_per_timepoint)),
            
            # Z-plane statistics
            'num_z_planes': len(np.unique(np.round(z_values, decimals=3))),
            'min_z': float(np.min(z_values)),
            'max_z': float(np.max(z_values)),
            
            # File size
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
        }
        
    return stats

def main():
    # Set up directory
    input_dir = "training_spike_data_2018"
    output_dir = "spike_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all processed spike files
    h5_files = []
    for file in os.listdir(input_dir):
        if file.endswith('_processed.h5'):
            h5_files.append(os.path.join(input_dir, file))
    
    if not h5_files:
        print(f"No processed spike files found in {input_dir}")
        return
    
    print(f"Found {len(h5_files)} processed spike files")
    
    # Analyze each file
    all_stats = []
    for h5_file in tqdm(h5_files, desc="Analyzing files"):
        try:
            stats = analyze_spike_file(h5_file)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error analyzing {h5_file}: {str(e)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_stats)
    
    # Save statistics to CSV
    csv_path = os.path.join(output_dir, "spike_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved statistics to {csv_path}")
    
    # Create summary statistics
    summary = df.describe()
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")
    
    # Generate plots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    sns.barplot(data=df, x='subject', y='mean_spike_value')
    plt.title('Mean Spike Value by Subject')
    plt.xticks(rotation=90)
    
    plt.subplot(3, 2, 2)
    sns.barplot(data=df, x='subject', y='active_ratio')
    plt.title('Active Cell Ratio by Subject')
    plt.xticks(rotation=90)
    
    plt.subplot(3, 2, 3)
    sns.barplot(data=df, x='subject', y='num_cells')
    plt.title('Number of Cells by Subject')
    plt.xticks(rotation=90)
    
    plt.subplot(3, 2, 4)
    sns.barplot(data=df, x='subject', y='num_z_planes')
    plt.title('Number of Z-Planes by Subject')
    plt.xticks(rotation=90)
    
    # Add new plots for spikes per timepoint
    plt.subplot(3, 2, 5)
    sns.barplot(data=df, x='subject', y='avg_spikes_per_timepoint')
    plt.title('Average Spikes per Timepoint')
    plt.xticks(rotation=90)
    
    plt.subplot(3, 2, 6)
    sns.barplot(data=df, x='subject', y='median_spikes_per_timepoint')
    plt.title('Median Spikes per Timepoint')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spike_statistics.png"), dpi=300)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total subjects: {len(df)}")
    print(f"Total cells: {df['num_cells'].sum()}")
    print(f"Mean cells per subject: {df['num_cells'].mean():.1f}")
    print(f"Mean active ratio: {df['active_ratio'].mean():.4f}")
    print(f"Mean z-planes: {df['num_z_planes'].mean():.1f}")
    print(f"Average spikes per timepoint across all subjects: {df['avg_spikes_per_timepoint'].mean():.1f}")
    print(f"Median of median spikes per timepoint: {df['median_spikes_per_timepoint'].median():.1f}")
    
    # Display individual subject statistics
    print("\nPer-Subject Statistics:")
    for _, row in df.iterrows():
        print(f"\nSubject: {row['subject']}")
        print(f"  Cells: {row['num_cells']}")
        print(f"  Timepoints: {row['num_timepoints']}")
        print(f"  Z-planes: {row['num_z_planes']}")
        print(f"  Average spikes per timepoint: {row['avg_spikes_per_timepoint']:.1f}")
        print(f"  Median spikes per timepoint: {row['median_spikes_per_timepoint']:.1f}")
        print(f"  File size: {row['file_size_mb']:.1f} MB")

if __name__ == "__main__":
    main() 