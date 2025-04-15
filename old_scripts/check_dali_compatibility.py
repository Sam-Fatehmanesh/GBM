#!/usr/bin/env python
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob
from tqdm import tqdm
import argparse

def inspect_h5_file(file_path):
    """
    Inspect the structure of an h5 file and verify its compatibility with the DALI dataset.
    
    Args:
        file_path (str): Path to the h5 file
    
    Returns:
        dict: Dictionary with inspection results
    """
    results = {
        'file_name': os.path.basename(file_path),
        'is_compatible': True,
        'issues': [],
        'structure': {},
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check required datasets
            required_datasets = ['spikes', 'cell_positions']
            for ds_name in required_datasets:
                if ds_name not in f:
                    results['is_compatible'] = False
                    results['issues'].append(f"Missing required dataset: {ds_name}")
            
            # Get structure information
            for key in f.keys():
                ds = f[key]
                results['structure'][key] = {
                    'shape': ds.shape,
                    'dtype': str(ds.dtype),
                }
                
                # Check for NaN values
                if key == 'cell_positions':
                    if np.isnan(ds[:]).any():
                        results['is_compatible'] = False
                        results['issues'].append(f"NaN values found in {key}")
            
            # Get attributes
            results['attributes'] = dict(f.attrs.items())
            
            # Check for correct shapes
            if 'spikes' in f and 'cell_positions' in f:
                spikes = f['spikes']
                cell_positions = f['cell_positions']
                
                # Check that num_cells dimensions match
                if spikes.shape[1] != cell_positions.shape[0]:
                    results['is_compatible'] = False
                    results['issues'].append(
                        f"Mismatch between spikes.shape[1] ({spikes.shape[1]}) and "
                        f"cell_positions.shape[0] ({cell_positions.shape[0]})"
                    )
                
                # Check that cell_positions has 3 dimensions (x, y, z)
                if cell_positions.shape[1] != 3:
                    results['is_compatible'] = False
                    results['issues'].append(
                        f"Expected cell_positions.shape[1] to be 3, got {cell_positions.shape[1]}"
                    )
    
    except Exception as e:
        results['is_compatible'] = False
        results['issues'].append(f"Error reading file: {str(e)}")
    
    return results

def create_visualization(file_path, output_dir, num_neurons=5, num_frames=100):
    """
    Create a visualization of the spike data in the h5 file.
    
    Args:
        file_path (str): Path to the h5 file
        output_dir (str): Directory to save the visualization
        num_neurons (int): Number of neurons to visualize
        num_frames (int): Number of frames to visualize
    
    Returns:
        str: Path to the output visualization file
    """
    subject_name = os.path.basename(file_path).split('_processed')[0]
    output_file = os.path.join(output_dir, f"{subject_name}_dali_check.pdf")
    
    with h5py.File(file_path, 'r') as f:
        # Get data
        spikes = f['spikes'][:]
        cell_positions = f['cell_positions'][:]
        
        # Get attributes
        num_timepoints = f.attrs.get('num_timepoints', spikes.shape[0])
        num_cells = f.attrs.get('num_cells', spikes.shape[1])
        
        # Limit to requested number of frames
        num_frames = min(num_frames, num_timepoints)
        spikes = spikes[:num_frames]
        
        # Select random neurons
        if num_cells > num_neurons:
            neuron_indices = np.random.choice(num_cells, num_neurons, replace=False)
        else:
            neuron_indices = np.arange(num_cells)
        
        # Create PDF with visualizations
        with PdfPages(output_file) as pdf:
            # Create file information page
            plt.figure(figsize=(10, 7))
            plt.axis('off')
            info_text = (
                f"File: {os.path.basename(file_path)}\n\n"
                f"Number of timepoints: {num_timepoints}\n"
                f"Number of cells: {num_cells}\n\n"
                f"Spikes shape: {spikes.shape}\n"
                f"Cell positions shape: {cell_positions.shape}\n\n"
                f"Attributes:\n"
            )
            
            for key, value in f.attrs.items():
                info_text += f"  - {key}: {value}\n"
            
            plt.text(0.05, 0.95, info_text, fontsize=12, va='top', family='monospace')
            pdf.savefig()
            plt.close()
            
            # Create spike activity visualization
            plt.figure(figsize=(15, 10))
            for i, neuron_idx in enumerate(neuron_indices):
                plt.subplot(num_neurons, 1, i+1)
                plt.plot(spikes[:, neuron_idx], 'r-', linewidth=1.5)
                plt.title(f"Neuron {neuron_idx} Spike Activity")
                plt.ylabel("Spike Activity")
                if i == num_neurons - 1:
                    plt.xlabel("Time (frames)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Create cell position visualization
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            pos = cell_positions[neuron_indices]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', marker='o')
            
            # Highlight selected neurons with a bigger marker
            for i, idx in enumerate(neuron_indices):
                ax.scatter(
                    cell_positions[idx, 0],
                    cell_positions[idx, 1],
                    cell_positions[idx, 2],
                    c='r', marker='o', s=100
                )
                ax.text(
                    cell_positions[idx, 0],
                    cell_positions[idx, 1],
                    cell_positions[idx, 2],
                    f"{idx}", fontsize=8
                )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Cell Positions (Showing {len(neuron_indices)} of {num_cells} neurons)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Create Z-plane distribution visualization
            plt.figure(figsize=(12, 8))
            z_values = cell_positions[:, 2]
            unique_z = np.unique(np.round(z_values, decimals=3))
            z_counts = [np.sum(np.isclose(z_values, z, atol=1e-3)) for z in unique_z]
            
            plt.bar(range(len(unique_z)), z_counts)
            plt.title("Number of Cells per Z-Plane")
            plt.xlabel("Z-Plane Index")
            plt.ylabel("Number of Cells")
            plt.xticks(range(len(unique_z)), [f"{z:.1f}" for z in unique_z], rotation=90)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Check DALI dataset compatibility')
    parser.add_argument('--data_dir', type=str, default='training_spike_data_2018',
                        help='Directory containing processed h5 files')
    parser.add_argument('--output_dir', type=str, default='training_spike_data_2018/checks',
                        help='Directory to save visualization files')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations for each file')
    parser.add_argument('--num_neurons', type=int, default=5,
                        help='Number of neurons to visualize')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all processed h5 files
    files = glob.glob(os.path.join(args.data_dir, '*_processed.h5'))
    files.sort()
    
    if not files:
        print(f"No processed h5 files found in {args.data_dir}")
        return
    
    print(f"Found {len(files)} processed h5 files")
    
    # Inspect each file
    all_compatible = True
    compatible_files = []
    incompatible_files = []
    
    for file_path in tqdm(files, desc="Checking files"):
        results = inspect_h5_file(file_path)
        
        if results['is_compatible']:
            compatible_files.append(file_path)
        else:
            incompatible_files.append((file_path, results['issues']))
            all_compatible = False
    
    # Print summary
    print("\nCompatibility Check Summary:")
    print(f"  - Total files: {len(files)}")
    print(f"  - Compatible files: {len(compatible_files)}")
    print(f"  - Incompatible files: {len(incompatible_files)}")
    
    if incompatible_files:
        print("\nIssues found in the following files:")
        for file_path, issues in incompatible_files:
            print(f"  - {os.path.basename(file_path)}:")
            for issue in issues:
                print(f"    - {issue}")
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        
        viz_files = []
        for file_path in tqdm(compatible_files, desc="Creating visualizations"):
            output_file = create_visualization(
                file_path,
                args.output_dir,
                num_neurons=args.num_neurons,
                num_frames=args.num_frames
            )
            viz_files.append(output_file)
        
        print(f"\nCreated {len(viz_files)} visualization files in {args.output_dir}")
    
    # Final judgment
    if all_compatible:
        print("\nAll files are compatible with the DALI dataset format!")
    else:
        print("\nSome files are not compatible with the DALI dataset format.")
        print("Please fix the issues and run this script again.")

if __name__ == "__main__":
    main() 