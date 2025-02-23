import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def analyze_grid_approximation_positions(pos, z_level, grid_x, grid_y):
    """Analyze how many neurons get merged when mapping to a grid based on positions only.
    
    Returns detailed statistics about merging patterns.
    """
    # Get positions for specific z-level
    z = pos[:, 2]
    z_mask = (np.abs(z - z_level) < 1e-3)  # Use small epsilon for float comparison
    pos_z = pos[z_mask]
    
    if len(pos_z) == 0:
        return {
            'total_neurons': 0,
            'unique_cells': 0,
            'merged_neurons': 0,
            'merge_percentage': 0,
            'merge_distribution': [],
            'max_merge_count': 0,
            'mean_merge_count': 0,
            'median_merge_count': 0
        }
    
    # Normalize x,y to [0,1]
    x = (pos_z[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
    y = (pos_z[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min())
    
    # Convert to grid indices
    x_idx = (x * grid_x).astype(int).clip(0, grid_x-1)
    y_idx = (y * grid_y).astype(int).clip(0, grid_y-1)
    
    # Count neurons per grid cell
    grid_positions = np.stack([x_idx, y_idx], axis=1)
    unique_positions, counts = np.unique(grid_positions, axis=0, return_counts=True)
    
    # Calculate merging statistics
    total_neurons = len(pos_z)
    unique_cells = len(unique_positions)
    merged_neurons = total_neurons - unique_cells
    
    # Get distribution of merge counts (excluding single neurons)
    merge_counts = counts[counts > 1] - 1  # Subtract 1 to get number of additional neurons merged
    
    return {
        'total_neurons': total_neurons,
        'unique_cells': unique_cells,
        'merged_neurons': merged_neurons,
        'merge_percentage': 100 * merged_neurons / total_neurons if total_neurons > 0 else 0,
        'merge_distribution': merge_counts.tolist(),
        'max_merge_count': np.max(counts) if len(counts) > 0 else 0,
        'mean_merge_count': np.mean(counts) if len(counts) > 0 else 0,
        'median_merge_count': np.median(counts) if len(counts) > 0 else 0
    }

def plot_merge_statistics(all_stats, grid_sizes, output_dir):
    """Create plots visualizing the merging statistics."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Merge percentages across grid sizes
    plt.figure(figsize=(12, 6))
    grid_labels = [f'{x}x{y}' for x, y in grid_sizes]
    merge_percentages = [np.mean([stats[grid]['merge_percentage'] for stats in all_stats.values()]) 
                        for grid in grid_labels]
    
    plt.bar(grid_labels, merge_percentages)
    plt.title('Average Merge Percentage by Grid Size')
    plt.xlabel('Grid Size')
    plt.ylabel('Merge Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merge_percentages.png'))
    plt.close()
    
    # Plot 2: Distribution of merge counts
    plt.figure(figsize=(12, 6))
    for grid_size, color in zip(grid_labels, plt.cm.viridis(np.linspace(0, 1, len(grid_labels)))):
        all_merges = []
        for z_stats in all_stats.values():
            all_merges.extend(z_stats[grid_size]['merge_distribution'])
        
        if all_merges:
            plt.hist(all_merges, bins=20, alpha=0.5, label=grid_size, color=color, density=True)
    
    plt.title('Distribution of Merge Counts')
    plt.xlabel('Number of Additional Neurons Merged')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'merge_distribution.png'))
    plt.close()

def main():
    # Load data
    print("Loading data...")
    with h5py.File('processed_spikes/subject_1_processed.h5', 'r') as f:
        pos = f['cell_positions'][:]
    
    # Get unique z-levels (rounded to handle floating point precision)
    z_levels = np.unique(np.round(pos[:, 2], decimals=3))
    print(f"\nFound {len(z_levels)} unique z-levels")
    
    # Define grid sizes to test
    grid_sizes = [
        (64, 32),    # 2,048 cells
        (128, 64),   # 8,192 cells
        (160, 80),   # 12,800 cells
        (192, 96),   # 18,432 cells
        (224, 112),  # 25,088 cells
        (256, 128)   # 32,768 cells
    ]
    
    # Store all statistics
    all_stats = {}
    
    # Analyze each z-level
    print("\nAnalyzing neuron positions at each z-level...")
    for z in tqdm(z_levels):
        stats = {}
        for grid_x, grid_y in grid_sizes:
            stats[f'{grid_x}x{grid_y}'] = analyze_grid_approximation_positions(pos, z, grid_x, grid_y)
        all_stats[z] = stats
    
    # Create output directory for plots
    output_dir = "grid_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_merge_statistics(all_stats, grid_sizes, output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nAverage statistics across all z-planes:")
    print("Grid Size | Avg Total | Avg Unique | Avg Merged | Avg Merge % | Max Merged | Median Merged")
    print("-" * 85)
    
    for grid_x, grid_y in grid_sizes:
        grid_key = f'{grid_x}x{grid_y}'
        avg_total = np.mean([stats[grid_key]['total_neurons'] for stats in all_stats.values()])
        avg_unique = np.mean([stats[grid_key]['unique_cells'] for stats in all_stats.values()])
        avg_merged = np.mean([stats[grid_key]['merged_neurons'] for stats in all_stats.values()])
        avg_merge_pct = np.mean([stats[grid_key]['merge_percentage'] for stats in all_stats.values()])
        max_merged = np.max([stats[grid_key]['max_merge_count'] for stats in all_stats.values()])
        median_merged = np.median([stats[grid_key]['median_merge_count'] for stats in all_stats.values()])
        
        print(f"{grid_key:9} | {avg_total:9.1f} | {avg_unique:10.1f} | {avg_merged:10.1f} | "
              f"{avg_merge_pct:10.1f}% | {max_merged:10.1f} | {median_merged:13.1f}")
    
    # Save detailed statistics to file
    print("\nSaving detailed statistics to grid_analysis_results/detailed_stats.txt")
    with open(os.path.join(output_dir, "detailed_stats.txt"), "w") as f:
        f.write("Detailed Grid Analysis Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        for z in z_levels:
            f.write(f"\nZ-level: {z:.3f}\n")
            f.write("-" * 30 + "\n")
            for grid_x, grid_y in grid_sizes:
                grid_key = f'{grid_x}x{grid_y}'
                stats = all_stats[z][grid_key]
                f.write(f"\nGrid size {grid_key}:\n")
                f.write(f"  Total neurons: {stats['total_neurons']}\n")
                f.write(f"  Unique cells: {stats['unique_cells']}\n")
                f.write(f"  Merged neurons: {stats['merged_neurons']}\n")
                f.write(f"  Merge percentage: {stats['merge_percentage']:.1f}%\n")
                f.write(f"  Max neurons in one cell: {stats['max_merge_count']}\n")
                f.write(f"  Mean neurons per cell: {stats['mean_merge_count']:.2f}\n")
                f.write(f"  Median neurons per cell: {stats['median_merge_count']:.2f}\n")

if __name__ == "__main__":
    main() 