#!/usr/bin/env python
import os
import h5py
import numpy as np
from tqdm import tqdm

def analyze_spike_balance(h5_file):
    """Analyze the balance of active vs inactive cells in spike data.
    
    Args:
        h5_file: Path to processed spike data H5 file
    """
    print(f"\nAnalyzing spike balance in: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        spikes = f['spikes'][:]  # shape: (n_timepoints, n_cells)
        cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
        
    # Get basic stats
    n_timepoints, n_cells = spikes.shape
    print(f"Number of timepoints: {n_timepoints}")
    print(f"Number of cells: {n_cells}")
    
    # Calculate overall activation rate
    active_mask = np.abs(spikes) > 1e-6
    total_active = np.sum(active_mask)
    total_elements = spikes.size
    
    activation_rate = total_active / total_elements
    print(f"\nOverall activation rate: {activation_rate:.6f}")
    print(f"Active elements: {total_active:,} / {total_elements:,}")
    print(f"Ratio of 1s to 0s: 1:{(1-activation_rate)/activation_rate:.1f}")
    
    # Get unique z values
    z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
    z_values = np.sort(z_values)[::-1]  # Sort descending
    
    print("\nActivation rates by z-plane:")
    print("-" * 40)
    
    # Calculate activation rate per z-plane
    for z_idx, z_level in enumerate(z_values):
        # Get mask for cells in this z-plane
        z_mask = (np.round(cell_positions[:, 2], decimals=3) == z_level)
        cells_in_plane = np.sum(z_mask)
        
        # Get spikes for cells in this z-plane
        plane_spikes = spikes[:, z_mask]
        active_in_plane = np.sum(np.abs(plane_spikes) > 1e-6)
        total_in_plane = plane_spikes.size
        
        plane_rate = active_in_plane / total_in_plane
        print(f"Z-plane {z_idx} (z={z_level:.3f}): {plane_rate:.6f}")
        print(f"  Cells in plane: {cells_in_plane}")
        print(f"  Active elements: {active_in_plane:,} / {total_in_plane:,}")
        print(f"  Ratio of 1s to 0s: 1:{(1-plane_rate)/plane_rate:.1f}")
        print("-" * 40)

def main():
    # Find all processed spike files
    processed_dir = "processed_spikes"
    spike_files = []
    for file in os.listdir(processed_dir):
        if file.endswith("_processed.h5"):
            spike_files.append(os.path.join(processed_dir, file))
    
    if not spike_files:
        print("No processed spike data files found!")
        return
    
    # Analyze each file
    for f in spike_files:
        analyze_spike_balance(f)

if __name__ == "__main__":
    main() 