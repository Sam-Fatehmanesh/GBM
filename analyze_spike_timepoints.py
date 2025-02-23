#!/usr/bin/env python
import h5py
import numpy as np
from tqdm import tqdm

def analyze_spike_data(h5_file):
    """Analyze spike data to understand z-plane activity patterns."""
    print(f"\nAnalyzing file: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        spikes = f['spikes'][:]  # (n_timepoints, n_cells)
        positions = f['cell_positions'][:]  # (n_cells, 3)
    
    # Basic dataset info
    n_timepoints, n_cells = spikes.shape
    print(f"Number of timepoints: {n_timepoints}")
    print(f"Number of cells: {n_cells}")
    
    # Analyze z-planes
    z_values = np.unique(np.round(positions[:, 2], decimals=3))
    print(f"\nFound {len(z_values)} unique z-planes")
    
    # Count cells per z-plane
    cells_per_z = {}
    for z in z_values:
        z_mask = (np.round(positions[:, 2], decimals=3) == z)
        cells_per_z[z] = np.sum(z_mask)
    
    print("\nCells per z-plane:")
    for z, count in sorted(cells_per_z.items()):
        print(f"z={z:.3f}: {count} cells")
    
    # Analyze activity patterns
    print("\nAnalyzing activity patterns...")
    empty_timepoints = 0
    total_active_cells = 0
    z_plane_stats = {z: {'empty_count': 0, 'total_spikes': 0} for z in z_values}
    
    for t in tqdm(range(n_timepoints)):
        spikes_t = spikes[t]  # Current timepoint's spikes
        active_cells = np.sum(np.abs(spikes_t) > 1e-6)
        total_active_cells += active_cells
        
        if active_cells == 0:
            empty_timepoints += 1
        
        # Check each z-plane
        for z in z_values:
            z_mask = (np.round(positions[:, 2], decimals=3) == z)
            z_spikes = spikes_t[z_mask]
            active_in_z = np.sum(np.abs(z_spikes) > 1e-6)
            
            if active_in_z == 0:
                z_plane_stats[z]['empty_count'] += 1
            z_plane_stats[z]['total_spikes'] += active_in_z
    
    # Report statistics
    print(f"\nEmpty timepoints (no spikes anywhere): {empty_timepoints} ({100*empty_timepoints/n_timepoints:.1f}%)")
    print(f"Average active cells per timepoint: {total_active_cells/n_timepoints:.1f}")
    
    print("\nPer z-plane statistics:")
    print("z-level | Cells | Empty % | Avg Active")
    print("-" * 50)
    for z in sorted(z_values):
        empty_percent = 100 * z_plane_stats[z]['empty_count'] / n_timepoints
        avg_active = z_plane_stats[z]['total_spikes'] / n_timepoints
        print(f"{z:7.3f} | {cells_per_z[z]:5d} | {empty_percent:7.1f}% | {avg_active:10.1f}")

def main():
    # Analyze first subject's data
    analyze_spike_data('processed_spikes/subject_1_processed.h5')

if __name__ == "__main__":
    main() 