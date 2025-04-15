#!/usr/bin/env python3

import os
import sys
import numpy as np
import h5py
import json
import tensorstore as ts
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback

def extract_positions_from_segmentation():
    """
    Extract 3D centroids for each neuron from the segmentation data.
    This attempts to access the GCS bucket directly.
    """
    print("Attempting to extract 3D positions from segmentation data...")
    
    # Path to the segmentation in GCS
    seg_path = "gs://zapbench-release/volumes/20240930/segmentation"
    
    try:
        # Try to open with zarr3 driver
        print(f"Opening segmentation from {seg_path}...")
        seg_obj = ts.open({
            'driver': 'zarr3',
            'kvstore': seg_path,
            'open': True
        }).result()
        
        print(f"Successfully opened segmentation with shape: {seg_obj.shape}")
        print(f"Data type: {seg_obj.dtype}")
        
        # Get basic information about the segmentation
        # This is likely a 3D volume with labeled neurons
        # Each unique value represents a different neuron
        
        # Define the output directory
        output_dir = "neuron_positions"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get a small chunk to analyze the structure
        chunk_size = [min(64, s) for s in seg_obj.shape]
        sample_chunk = seg_obj[0:chunk_size[0], 0:chunk_size[1], 0:chunk_size[2]].read().result()
        
        print(f"Sample chunk shape: {sample_chunk.shape}")
        unique_labels = np.unique(sample_chunk)
        print(f"Unique labels in sample chunk: {unique_labels}")
        
        # Save the sample chunk for inspection
        np.save(os.path.join(output_dir, "segmentation_sample.npy"), sample_chunk)
        
        # If the segmentation is too large to load entirely, we can process in chunks
        # For now, let's try to load small slices and calculate centroids
        
        # Determine the unique neuron labels throughout the volume
        # This could take time for a large volume, so we'll start with a subsample
        
        # Load a subset for analysis (adjust these indices based on the actual size)
        subset_shape = [min(128, s) for s in seg_obj.shape]
        subset = seg_obj[0:subset_shape[0], 0:subset_shape[1], 0:subset_shape[2]].read().result()
        
        # Find all unique neuron IDs in the subset
        neuron_ids = np.unique(subset)
        
        print(f"Found {len(neuron_ids)} unique labels in subset")
        
        # Remove background (usually labeled as 0)
        neuron_ids = neuron_ids[neuron_ids > 0]
        
        print(f"After removing background, found {len(neuron_ids)} neuron IDs")
        
        # Calculate centroids for each neuron in the subset
        centroids = {}
        for neuron_id in neuron_ids:
            # Create binary mask for this neuron
            neuron_mask = (subset == neuron_id)
            
            # Skip if no voxels for this neuron in the subset
            if not np.any(neuron_mask):
                continue
            
            # Calculate centroid
            centroid = center_of_mass(neuron_mask)
            
            # Store centroid
            centroids[int(neuron_id)] = centroid
        
        print(f"Calculated centroids for {len(centroids)} neurons")
        
        # Convert to a DataFrame for easier analysis
        centroids_df = pd.DataFrame.from_dict(
            {id: {'x': c[0], 'y': c[1], 'z': c[2]} for id, c in centroids.items()}, 
            orient='index'
        )
        
        # Save centroids to CSV
        centroids_path = os.path.join(output_dir, "neuron_centroids_from_segmentation.csv")
        centroids_df.to_csv(centroids_path)
        print(f"Saved neuron centroids to {centroids_path}")
        
        # Visualize first 1000 centroids in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        viz_df = centroids_df.iloc[:1000] if len(centroids_df) > 1000 else centroids_df
        ax.scatter(viz_df['x'], viz_df['y'], viz_df['z'], s=5, alpha=0.6)
        ax.set_title('Neuron Centroids from Segmentation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "neuron_centroids_3d_viz.png"), dpi=150)
        plt.close()
        
        return centroids_df
        
    except Exception as e:
        print(f"Error accessing segmentation: {e}")
        traceback.print_exc()
        return None

def extract_positions_from_dataframe():
    """
    Extract 3D centroids from a pre-downloaded segmentation dataframe.
    This is useful when we already have a dataframe with cell positions.
    """
    print("\nAttempting to extract 3D positions from segmentation dataframe...")
    
    output_dir = "neuron_positions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have a local copy of the segmentation dataframe
    df_path = os.path.join("raw_trace_data_zap", "segmentation_dataframe.csv")
    
    if not os.path.exists(df_path):
        print(f"Segmentation dataframe not found at {df_path}")
        
        # Try alternative locations or try to get it from ZAPBench
        try:
            from zapbench import constants
            from zapbench import data_utils
            
            print("ZAPBench package found, trying to get segmentation dataframe...")
            
            if hasattr(constants, 'SEGMENTATION_DATAFRAMES'):
                print(f"SEGMENTATION_DATAFRAMES exists with keys: {list(constants.SEGMENTATION_DATAFRAMES.keys())}")
                
                if constants.SEGMENTATION_DATAFRAMES:
                    df_name = list(constants.SEGMENTATION_DATAFRAMES.keys())[0]
                    print(f"Trying to get segmentation dataframe for '{df_name}'")
                    
                    df = data_utils.get_segmentation_dataframe(df_name)
                    print(f"Successfully got dataframe with shape: {df.shape}")
                    print(f"Columns: {df.columns.tolist()}")
                    
                    # Check if it has coordinate columns
                    coord_columns = [col for col in df.columns if 'centroid' in col.lower() or any(c in col.lower() for c in ['x', 'y', 'z'])]
                    
                    if coord_columns:
                        print(f"Found potential coordinate columns: {coord_columns}")
                        
                        # Save dataframe
                        os.makedirs("raw_trace_data_zap", exist_ok=True)
                        df.to_csv(df_path, index=False)
                        print(f"Saved segmentation dataframe to {df_path}")
                        
                        # Process the coordinate columns
                        position_data = df[coord_columns].values
                        
                        # Create position DataFrame
                        positions_df = pd.DataFrame(position_data, columns=coord_columns)
                        positions_df.index.name = 'neuron_id'
                        
                        # Save positions
                        positions_path = os.path.join(output_dir, "neuron_positions_from_dataframe.csv")
                        positions_df.to_csv(positions_path)
                        print(f"Saved neuron positions to {positions_path}")
                        
                        # Visualize in 3D
                        if len(coord_columns) >= 3:
                            fig = plt.figure(figsize=(10, 8))
                            ax = fig.add_subplot(111, projection='3d')
                            
                            viz_df = positions_df.iloc[:1000] if len(positions_df) > 1000 else positions_df
                            ax.scatter(viz_df[coord_columns[0]], viz_df[coord_columns[1]], viz_df[coord_columns[2]], s=5, alpha=0.6)
                            ax.set_title('Neuron Positions from DataFrame')
                            ax.set_xlabel(coord_columns[0])
                            ax.set_ylabel(coord_columns[1])
                            ax.set_zlabel(coord_columns[2])
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, "neuron_positions_3d_viz.png"), dpi=150)
                            plt.close()
                        
                        return positions_df
                    else:
                        print("No coordinate columns found in segmentation dataframe")
                        return None
                else:
                    print("SEGMENTATION_DATAFRAMES is empty")
                    return None
            else:
                print("SEGMENTATION_DATAFRAMES not found in constants")
                return None
                
        except ImportError:
            print("ZAPBench package not found")
            return None
            
    else:
        print(f"Loading segmentation dataframe from {df_path}")
        df = pd.read_csv(df_path)
        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract coordinate columns
        coord_columns = [col for col in df.columns if 'centroid' in col.lower() or any(c in col.lower() for c in ['x', 'y', 'z'])]
        
        if coord_columns:
            print(f"Found potential coordinate columns: {coord_columns}")
            
            # Process the coordinate columns
            position_data = df[coord_columns].values
            
            # Create position DataFrame
            positions_df = pd.DataFrame(position_data, columns=coord_columns)
            positions_df.index.name = 'neuron_id'
            
            # Save positions
            positions_path = os.path.join(output_dir, "neuron_positions_from_dataframe.csv")
            positions_df.to_csv(positions_path)
            print(f"Saved neuron positions to {positions_path}")
            
            # Visualize in 3D
            if len(coord_columns) >= 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                viz_df = positions_df.iloc[:1000] if len(positions_df) > 1000 else positions_df
                ax.scatter(viz_df[coord_columns[0]], viz_df[coord_columns[1]], viz_df[coord_columns[2]], s=5, alpha=0.6)
                ax.set_title('Neuron Positions from DataFrame')
                ax.set_xlabel(coord_columns[0])
                ax.set_ylabel(coord_columns[1])
                ax.set_zlabel(coord_columns[2])
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "neuron_positions_3d_viz.png"), dpi=150)
                plt.close()
            
            return positions_df
        else:
            print("No coordinate columns found in segmentation dataframe")
            return None

def extract_positions_from_embeddings():
    """
    Use PCA to extract 3D positions from high-dimensional position embeddings.
    This is a fallback when direct segmentation data is not available.
    """
    print("\nAttempting to extract 3D positions from position embeddings via PCA...")
    
    # Define output directory
    output_dir = "neuron_positions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have position embeddings downloaded
    embedding_path = os.path.join("raw_trace_data_zap", "position_embedding_sample.npy")
    
    if not os.path.exists(embedding_path):
        print(f"Position embeddings not found at {embedding_path}")
        
        # Try to access embeddings directly from GCS
        try:
            # Try with common paths for position embeddings
            embedding_gcs_path = "gs://zapbench-release/volumes/20240930/position_embedding"
            
            print(f"Trying to directly access position embeddings at: {embedding_gcs_path}")
            
            # Try to open with zarr3 driver
            tensorstore_obj = ts.open({
                'driver': 'zarr3',
                'kvstore': embedding_gcs_path,
                'open': True
            }).result()
            
            print(f"Successfully opened TensorStore with shape: {tensorstore_obj.shape}")
            print(f"Data type: {tensorstore_obj.dtype}")
            
            # Read the full data
            full_data = tensorstore_obj[:].read().result()
            print(f"Loaded position embeddings with shape: {full_data.shape}")
            
            # Save a copy for later use
            os.makedirs("raw_trace_data_zap", exist_ok=True)
            np.save(embedding_path, full_data)
            print(f"Saved position embeddings to {embedding_path}")
            
        except Exception as e:
            print(f"Error accessing position embeddings: {e}")
            
            # Try direct method
            try:
                from zapbench import constants
                from zapbench import data_utils
                
                print("ZAPBench package found, trying to get position embeddings...")
                
                if hasattr(constants, 'POSITION_EMBEDDING_SPECS'):
                    print(f"POSITION_EMBEDDING_SPECS exists with keys: {list(constants.POSITION_EMBEDDING_SPECS.keys())}")
                    
                    spec_name = list(constants.POSITION_EMBEDDING_SPECS.keys())[0]
                    spec = data_utils.get_position_embedding_spec(spec_name)
                    print(f"Got embedding spec: {spec}")
                    
                    tensorstore_obj = ts.open(spec).result()
                    print(f"Opened TensorStore with shape: {tensorstore_obj.shape}")
                    
                    full_data = tensorstore_obj[:].read().result()
                    print(f"Loaded position embeddings with shape: {full_data.shape}")
                    
                    os.makedirs("raw_trace_data_zap", exist_ok=True)
                    np.save(embedding_path, full_data)
                    print(f"Saved position embeddings to {embedding_path}")
                    
                else:
                    print("POSITION_EMBEDDING_SPECS not found in constants")
                    return None
            
            except ImportError:
                print("ZAPBench package not found")
                return None
    
    # Load the embeddings
    if os.path.exists(embedding_path):
        print(f"Loading position embeddings from {embedding_path}")
        embeddings = np.load(embedding_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Apply PCA to reduce to 3 dimensions
        print("Applying PCA to reduce embeddings to 3D positions...")
        pca = PCA(n_components=3)
        neuron_positions_3d = pca.fit_transform(embeddings)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Reduced embeddings to shape: {neuron_positions_3d.shape}")
        
        # Create DataFrame
        positions_df = pd.DataFrame(neuron_positions_3d, columns=['x', 'y', 'z'])
        positions_df.index.name = 'neuron_id'
        
        # Save positions
        positions_path = os.path.join(output_dir, "neuron_positions_from_pca.csv")
        positions_df.to_csv(positions_path)
        print(f"Saved neuron positions to {positions_path}")
        
        # Save the PCA components for reference
        components_path = os.path.join(output_dir, "pca_components.npy")
        np.save(components_path, pca.components_)
        
        # Visualize in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        viz_df = positions_df.iloc[:1000] if len(positions_df) > 1000 else positions_df
        ax.scatter(viz_df['x'], viz_df['y'], viz_df['z'], s=5, alpha=0.6)
        ax.set_title('Neuron Positions from PCA')
        ax.set_xlabel('X (PC1)')
        ax.set_ylabel('Y (PC2)')
        ax.set_zlabel('Z (PC3)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "neuron_positions_pca_3d_viz.png"), dpi=150)
        plt.close()
        
        return positions_df
    else:
        print("Failed to load or download position embeddings")
        return None

def compare_position_methods(df_segmentation=None, df_dataframe=None, df_pca=None):
    """
    Compare the different methods of obtaining 3D positions.
    """
    print("\nComparing position extraction methods...")
    
    output_dir = "neuron_positions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Count how many methods worked
    methods = []
    if df_segmentation is not None:
        methods.append(("segmentation", df_segmentation))
    if df_dataframe is not None:
        methods.append(("dataframe", df_dataframe))
    if df_pca is not None:
        methods.append(("pca", df_pca))
    
    print(f"Successfully extracted positions using {len(methods)} methods: {[m[0] for m in methods]}")
    
    if len(methods) <= 1:
        print("Not enough methods for comparison")
        return
    
    # Pairwise comparisons
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, df1 = methods[i]
            method2, df2 = methods[j]
            
            print(f"\nComparing {method1} vs {method2}:")
            
            # Both dataframes might use different scales or coordinate systems
            # We need to normalize both before comparison
            
            # Normalize method 1
            df1_values = df1.values
            df1_norm = (df1_values - df1_values.min(axis=0)) / (df1_values.max(axis=0) - df1_values.min(axis=0))
            
            # Normalize method 2
            df2_values = df2.values
            df2_norm = (df2_values - df2_values.min(axis=0)) / (df2_values.max(axis=0) - df2_values.min(axis=0))
            
            # Calculate correlation between the two methods
            # This assumes the neuron ordering is the same in both methods
            min_len = min(len(df1_norm), len(df2_norm))
            df1_subset = df1_norm[:min_len]
            df2_subset = df2_norm[:min_len]
            
            # Calculate correlation for each dimension
            for dim in range(df1_subset.shape[1]):
                corr = np.corrcoef(df1_subset[:, dim], df2_subset[:, dim])[0, 1]
                print(f"Dimension {dim} correlation: {corr:.4f}")
            
            # Visualize comparison
            fig = plt.figure(figsize=(15, 5))
            
            # Plot first dimension comparison
            ax1 = fig.add_subplot(131)
            ax1.scatter(df1_subset[:, 0], df2_subset[:, 0], s=5, alpha=0.5)
            ax1.set_title(f'Dimension 0: {method1} vs {method2}')
            ax1.set_xlabel(f'{method1} (normalized)')
            ax1.set_ylabel(f'{method2} (normalized)')
            
            # Plot second dimension comparison
            ax2 = fig.add_subplot(132)
            ax2.scatter(df1_subset[:, 1], df2_subset[:, 1], s=5, alpha=0.5)
            ax2.set_title(f'Dimension 1: {method1} vs {method2}')
            ax2.set_xlabel(f'{method1} (normalized)')
            ax2.set_ylabel(f'{method2} (normalized)')
            
            # Plot third dimension comparison
            ax3 = fig.add_subplot(133)
            ax3.scatter(df1_subset[:, 2], df2_subset[:, 2], s=5, alpha=0.5)
            ax3.set_title(f'Dimension 2: {method1} vs {method2}')
            ax3.set_xlabel(f'{method1} (normalized)')
            ax3.set_ylabel(f'{method2} (normalized)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{method1}_vs_{method2}.png"), dpi=150)
            plt.close()

if __name__ == "__main__":
    # Try the different methods
    df_segmentation = extract_positions_from_segmentation()
    df_dataframe = extract_positions_from_dataframe()
    df_pca = extract_positions_from_embeddings()
    
    # Use the best available method
    if df_dataframe is not None:
        print("\nRecommended method: Using position coordinates from segmentation dataframe")
        final_positions = df_dataframe
    elif df_segmentation is not None:
        print("\nRecommended method: Using centroids extracted from segmentation mask")
        final_positions = df_segmentation
    elif df_pca is not None:
        print("\nRecommended method: Using PCA-reduced position embeddings")
        final_positions = df_pca
    else:
        print("\nFailed to extract positions using any method")
        sys.exit(1)
    
    # Compare methods
    compare_position_methods(df_segmentation, df_dataframe, df_pca)
    
    # Save final positions in a standard format
    output_dir = "neuron_positions"
    final_path = os.path.join(output_dir, "final_neuron_positions.csv")
    final_positions.to_csv(final_path)
    print(f"\nSaved final neuron positions to {final_path}")
    
    # Also save as numpy array for easy loading
    final_array = final_positions.values
    np.save(os.path.join(output_dir, "final_neuron_positions.npy"), final_array)
    print(f"Also saved as numpy array with shape {final_array.shape}") 