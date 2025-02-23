import h5py
import numpy as np
from tqdm import tqdm

def analyze_empty_planes(h5_file):
    """Analyze percentage of z-planes with no spikes across timepoints."""
    print(f"\nAnalyzing file: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        spikes = f['spikes'][:]  # (n_timepoints, n_cells)
        positions = f['cell_positions'][:]  # (n_cells, 3)
        
    # Get unique z values
    z_values = np.unique(np.round(positions[:, 2], decimals=3))
    num_z = len(z_values)
    num_timepoints = spikes.shape[0]
    
    print(f"Number of z-planes: {num_z}")
    print(f"Number of timepoints: {num_timepoints}")
    print(f"Total cells: {len(positions)}")
    
    # Count empty planes
    total_planes = 0
    empty_planes = 0
    empty_by_z = {z: 0 for z in z_values}
    
    for t in tqdm(range(num_timepoints), desc="Processing timepoints"):
        spikes_t = spikes[t]  # Current timepoint's spikes
        
        for z in z_values:
            # Get cells in this z-plane
            z_mask = (np.round(positions[:, 2], decimals=3) == z)
            spikes_z = spikes_t[z_mask]
            
            total_planes += 1
            if not np.any(spikes_z > 0):
                empty_planes += 1
                empty_by_z[z] += 1
    
    empty_percentage = (empty_planes / total_planes) * 100
    print(f"\nOverall empty plane percentage: {empty_percentage:.1f}%")
    print(f"Total empty planes: {empty_planes} out of {total_planes}")
    
    print("\nEmpty percentage by z-plane:")
    print("Z-level | Empty % | Empty Count | Total Timepoints")
    print("-" * 50)
    for z in z_values:
        empty_pct = (empty_by_z[z] / num_timepoints) * 100
        print(f"{z:7.3f} | {empty_pct:7.1f}% | {empty_by_z[z]:11d} | {num_timepoints:15d}")

def main():
    # Analyze first subject
    h5_file = 'processed_spikes/subject_1_processed.h5'
    analyze_empty_planes(h5_file)

if __name__ == "__main__":
    main() 