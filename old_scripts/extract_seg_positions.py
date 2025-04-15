#!/usr/bin/env python3

import os
import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import traceback

def download_json_from_gcs(url):
    """
    Download a JSON file from Google Cloud Storage
    """
    print(f"Attempting to download from: {url}")
    
    try:
        # Make the request
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Successfully downloaded data from {url}")
            return response.json()
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading JSON: {e}")
        traceback.print_exc()
        return None

def safely_print_json_structure(data, max_items=5):
    """
    Safely print the structure of a JSON object without printing the entire thing
    """
    print(f"JSON structure: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:max_items]} " + ("+ more..." if len(data.keys()) > max_items else ""))
        # Sample a few values to understand structure better
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                break
            print(f"Sample {key}: {type(value)}")
            if isinstance(value, (list, dict)):
                print(f"  {key} length/size: {len(value)}")
            elif isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and len(value) > 50:
                    print(f"  {key} value: {value[:50]}... (truncated)")
                else:
                    print(f"  {key} value: {value}")
    
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                keys = list(data[0].keys())
                print(f"First item keys: {keys[:max_items]}" + ("+ more..." if len(keys) > max_items else ""))
                # Print a few sample items
                print(f"Sample of first {min(max_items, len(data))} items:")
                for i in range(min(max_items, len(data))):
                    sample_dict = {k: data[i][k] for k in list(data[i].keys())[:3]}
                    print(f"  Item {i}: {sample_dict} " + ("+ more keys..." if len(data[i].keys()) > 3 else ""))
            elif len(data) > 0:
                sample_count = min(max_items, len(data))
                print(f"Sample of first {sample_count} items: " + str(data[:sample_count]))

def extract_positions_from_centroids_transposed(url, output_name):
    """
    Extract 3D positions from a centroids JSON file that has a transposed structure
    (neurons as columns rather than rows)
    """
    print(f"\nAttempting to extract positions from {output_name}...")
    
    # Download the JSON data
    data = download_json_from_gcs(url)
    
    if data is None:
        print(f"Failed to download {output_name}")
        return None
    
    # Process the JSON data
    output_dir = "neuron_positions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw JSON for inspection - without printing it
    json_file = os.path.join(output_dir, f"{output_name}.json")
    print(f"Saving raw data to {json_file}")
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Analyze structure without printing all content
    safely_print_json_structure(data)
    
    # Try to convert to proper positions DataFrame
    try:
        # Check if this is the expected format with centroids
        if isinstance(data, dict) and all(k in data for k in ['centroid_x', 'centroid_y', 'centroid_z']):
            print("Found centroid_x, centroid_y, centroid_z keys")
            
            # Get the neuron IDs
            neuron_ids = []
            if 'label' in data:
                # Try to extract neuron IDs from the label field
                label_data = data['label']
                if isinstance(label_data, dict):
                    neuron_ids = [int(label_data[str(i)]) for i in range(len(label_data))]
                    print(f"Extracted {len(neuron_ids)} neuron IDs from label field")
            
            if not neuron_ids:
                # If we couldn't extract IDs, just use sequential numbering
                num_neurons = len(data['centroid_x'])
                neuron_ids = list(range(1, num_neurons + 1))
                print(f"Using sequential IDs for {num_neurons} neurons")
            
            # Create a DataFrame with the proper structure
            positions = []
            x_data = data['centroid_x']
            y_data = data['centroid_y']
            z_data = data['centroid_z']
            
            # Get the number of neurons
            num_neurons = len(x_data)
            print(f"Found coordinates for {num_neurons} neurons")
            
            # Prepare position data
            position_data = {
                'neuron_id': neuron_ids,
                'x': [float(x_data[str(i)]) for i in range(num_neurons)],
                'y': [float(y_data[str(i)]) for i in range(num_neurons)],
                'z': [float(z_data[str(i)]) for i in range(num_neurons)]
            }
            
            # Create the DataFrame
            positions_df = pd.DataFrame(position_data)
            positions_df.set_index('neuron_id', inplace=True)
            
            print(f"Created positions DataFrame with shape: {positions_df.shape}")
            print("Sample of positions DataFrame:")
            print(positions_df.head())
            
            # Save positions
            positions_path = os.path.join(output_dir, f"positions_from_{output_name}.csv")
            # Save the full dataset as numpy array (more efficient)
            positions_df.to_csv(positions_path)
            print(f"Saved neuron positions to {positions_path}")
            
            # Save as numpy array
            positions_array = positions_df.values
            array_path = os.path.join(output_dir, f"positions_from_{output_name}.npy")
            np.save(array_path, positions_array)
            print(f"Saved position array with shape {positions_array.shape} to {array_path}")
            
            # Visualize in 3D - but use a small sample
            print("Creating 3D visualization with sample data...")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Only use up to 1000 points for visualization
            sample_size = min(1000, len(positions_df))
            viz_df = positions_df.iloc[:sample_size]
            ax.scatter(viz_df['x'], viz_df['y'], viz_df['z'], s=5, alpha=0.6)
            ax.set_title(f'Neuron Positions from {output_name} (Sample of {sample_size})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"positions_from_{output_name}_viz.png"), dpi=150)
            plt.close()
            
            return positions_df
            
        else:
            print(f"Data does not have the expected structure for {output_name}")
            return None
            
    except Exception as e:
        print(f"Error processing {output_name}: {e}")
        traceback.print_exc()
        return None

def visualize_combined_results(positions_dfs):
    """
    Create a combined visualization of all successful methods
    """
    if not positions_dfs:
        print("No position data available for visualization")
        return
    
    print("\nCreating combined visualization...")
    
    output_dir = "neuron_positions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    markers = ['o', '^', 's', 'D', 'v', 'p']
    
    for i, (name, df) in enumerate(positions_dfs.items()):
        # Sample data for visualization
        sample_size = min(1000, len(df))
        if sample_size < len(df):
            print(f"Using {sample_size} samples out of {len(df)} for {name}")
        
        # Get sample and normalize
        sample_df = df.iloc[:sample_size]
        normalized_df = (sample_df - sample_df.min()) / (sample_df.max() - sample_df.min())
        
        # Plot with a different color and marker for each method
        ax.scatter(
            normalized_df['x'], 
            normalized_df['y'], 
            normalized_df['z'],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=15,
            alpha=0.6,
            label=f"{name} (n={sample_size})"
        )
    
    ax.set_title('Combined Neuron Positions (Normalized)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_positions_viz.png"), dpi=150)
    plt.close()
    
    print(f"Saved combined visualization to {output_dir}/combined_positions_viz.png")

if __name__ == "__main__":
    # Try the centroids JSON files with the new approach
    positions_dfs = {}
    
    # Process dataframe_centroids.json
    url1 = "https://storage.googleapis.com/zapbench-release/volumes/20240930/segmentation/dataframe_centroids.json"
    df_centroids = extract_positions_from_centroids_transposed(url1, "dataframe_centroids")
    if df_centroids is not None:
        positions_dfs['centroids'] = df_centroids
    
    # Process dataframe_centroids_flat.json
    url2 = "https://storage.googleapis.com/zapbench-release/volumes/20240930/segmentation/dataframe_centroids_flat.json"
    df_centroids_flat = extract_positions_from_centroids_transposed(url2, "dataframe_centroids_flat")
    if df_centroids_flat is not None:
        positions_dfs['centroids_flat'] = df_centroids_flat
    
    # Determine the best source based on data completeness
    if positions_dfs:
        print("\nSuccessfully extracted positions from the following sources:")
        for name, df in positions_dfs.items():
            print(f"- {name}: {df.shape[0]} neurons with {df.shape[1]} dimensions")
        
        # Choose the method with the most neurons
        best_method = max(positions_dfs.items(), key=lambda x: x[1].shape[0])
        print(f"\nRecommended method: {best_method[0]} with {best_method[1].shape[0]} neurons")
        
        # Save final positions
        output_dir = "neuron_positions"
        final_path = os.path.join(output_dir, "final_neuron_positions.csv")
        best_method[1].to_csv(final_path)
        print(f"Saved final neuron positions to {final_path}")
        
        # Save as numpy array
        final_array = best_method[1].values
        np_path = os.path.join(output_dir, "final_neuron_positions.npy")
        np.save(np_path, final_array)
        print(f"Saved all neuron positions as numpy array with shape {final_array.shape} to {np_path}")
        
        # Create combined visualization
        visualize_combined_results(positions_dfs)
    else:
        print("\nFailed to extract positions from any source")
        sys.exit(1) 