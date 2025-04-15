#!/usr/bin/env python3

import os
import sys
import numpy as np
import tensorstore as ts
import json
import traceback

print("Testing ZAPBench position embedding access...")

try:
    # Try to import ZAPBench modules
    from zapbench import constants
    from zapbench import data_utils
    
    print("\nSuccessfully imported ZAPBench modules")
    
    # Check if POSITION_EMBEDDING_SPECS exists and what it contains
    print("\nChecking POSITION_EMBEDDING_SPECS in constants:")
    if hasattr(constants, 'POSITION_EMBEDDING_SPECS'):
        print(f"POSITION_EMBEDDING_SPECS exists with keys: {list(constants.POSITION_EMBEDDING_SPECS.keys())}")
        
        # Try to get a position embedding spec
        spec_name = list(constants.POSITION_EMBEDDING_SPECS.keys())[0]
        print(f"\nTrying to get position embedding spec for '{spec_name}'")
        
        try:
            spec = data_utils.get_position_embedding_spec(spec_name)
            print(f"Successfully got spec: {spec}")
            
            # Try to open the TensorStore
            print(f"\nTrying to open TensorStore with spec")
            tensorstore_obj = ts.open(spec).result()
            
            print(f"Successfully opened TensorStore with shape: {tensorstore_obj.shape}")
            print(f"Data type: {tensorstore_obj.dtype}")
            
            # Get a sample of the data
            sample_size = min(5, tensorstore_obj.shape[0])
            sample = tensorstore_obj[:sample_size].read().result()
            
            print(f"\nSample of position embeddings (first {sample_size}):")
            for i in range(sample_size):
                print(f"Neuron {i}: {sample[i]}")
                
            # Check if this looks like spatial coordinates
            print("\nAnalysis of position embedding data:")
            full_data = tensorstore_obj[:].read().result()
            print(f"Min values per dimension: {np.min(full_data, axis=0)}")
            print(f"Max values per dimension: {np.max(full_data, axis=0)}")
            print(f"Mean values per dimension: {np.mean(full_data, axis=0)}")
            print(f"Standard deviation per dimension: {np.std(full_data, axis=0)}")
            
            # Save a sample to file for later examination
            output_dir = "raw_trace_data_zap"
            os.makedirs(output_dir, exist_ok=True)
            
            sample_path = os.path.join(output_dir, "position_embedding_sample.npy")
            np.save(sample_path, full_data)
            print(f"\nSaved full position embedding data to {sample_path}")
            
            # Test the mapping to trace matrix
            print("\nChecking if position embeddings align with trace matrix...")
            
            trace_file = os.path.join(output_dir, "zap_traces_full.h5")
            import h5py
            with h5py.File(trace_file, 'r') as f:
                trace_shape = f['traces'].shape
                
            print(f"Trace matrix shape: {trace_shape}")
            print(f"Position embedding count: {tensorstore_obj.shape[0]}")
            
            if trace_shape[1] == tensorstore_obj.shape[0]:
                print("✓ Position embedding count matches neuron count in trace matrix!")
            else:
                print("✗ Position embedding count does NOT match neuron count in trace matrix")
                print(f"Trace neurons: {trace_shape[1]}, Position embeddings: {tensorstore_obj.shape[0]}")
            
        except Exception as e:
            print(f"Error getting or using position embedding spec: {e}")
            traceback.print_exc()
    else:
        print("POSITION_EMBEDDING_SPECS not found in constants")
        
    # Try alternative: check for segmentation dataframe
    print("\nChecking for segmentation dataframe:")
    if hasattr(constants, 'SEGMENTATION_DATAFRAMES'):
        print(f"SEGMENTATION_DATAFRAMES exists with keys: {list(constants.SEGMENTATION_DATAFRAMES.keys())}")
        
        # Try to get a segmentation dataframe
        if constants.SEGMENTATION_DATAFRAMES:
            df_name = list(constants.SEGMENTATION_DATAFRAMES.keys())[0]
            print(f"\nTrying to get segmentation dataframe for '{df_name}'")
            
            try:
                df = data_utils.get_segmentation_dataframe(df_name)
                print(f"Successfully got dataframe with shape: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                
                # Check if it has coordinate columns
                coord_columns = [col for col in df.columns if 'centroid' in col.lower() or any(c in col.lower() for c in ['x', 'y', 'z'])]
                if coord_columns:
                    print(f"\nFound potential coordinate columns: {coord_columns}")
                    
                    # Sample of the dataframe
                    print("\nSample of segmentation dataframe:")
                    print(df.head())
                    
                    # Save dataframe to file
                    df_path = os.path.join(output_dir, "segmentation_dataframe.csv")
                    df.to_csv(df_path, index=False)
                    print(f"\nSaved segmentation dataframe to {df_path}")
                else:
                    print("\nNo coordinate columns found in segmentation dataframe")
            except Exception as e:
                print(f"Error getting segmentation dataframe: {e}")
                traceback.print_exc()
        else:
            print("SEGMENTATION_DATAFRAMES is empty")
    else:
        print("SEGMENTATION_DATAFRAMES not found in constants")
        
except ImportError as e:
    print(f"Could not import ZAPBench modules: {e}")
    print("Trying to directly load the position data from known paths instead...")
    
    # Try direct access using common paths
    try:
        # Try with common paths for position embeddings
        embedding_path = "gs://zapbench-release/volumes/20240930/embeddings/position"
        
        print(f"\nTrying to directly access position embeddings at: {embedding_path}")
        
        try:
            # Try to open with zarr3 driver
            tensorstore_obj = ts.open({
                'driver': 'zarr3',
                'kvstore': embedding_path,
                'open': True
            }).result()
            
            print(f"Successfully opened TensorStore with shape: {tensorstore_obj.shape}")
            print(f"Data type: {tensorstore_obj.dtype}")
            
            # Get a sample of the data
            sample_size = min(5, tensorstore_obj.shape[0])
            sample = tensorstore_obj[:sample_size].read().result()
            
            print(f"\nSample of position embeddings (first {sample_size}):")
            for i in range(sample_size):
                print(f"Neuron {i}: {sample[i]}")
                
            # Save a sample to file for later examination
            output_dir = "raw_trace_data_zap"
            os.makedirs(output_dir, exist_ok=True)
            
            sample_path = os.path.join(output_dir, "position_embedding_sample.npy")
            full_data = tensorstore_obj[:].read().result()
            np.save(sample_path, full_data)
            print(f"\nSaved full position embedding data to {sample_path}")
            
        except Exception as e:
            print(f"Error directly accessing position embeddings: {e}")
            
        # Try with segmentation path
        seg_path = "gs://zapbench-release/volumes/20240930/segmentation"
        
        print(f"\nTrying to access segmentation at: {seg_path}")
        
        try:
            # Try to open with zarr3 driver
            seg_obj = ts.open({
                'driver': 'zarr3',
                'kvstore': seg_path,
                'open': True
            }).result()
            
            print(f"Successfully opened segmentation with shape: {seg_obj.shape}")
            print(f"Data type: {seg_obj.dtype}")
            
        except Exception as e:
            print(f"Error accessing segmentation: {e}")
            
    except Exception as e:
        print(f"Error with direct access attempts: {e}")
        
print("\nTest completed.") 