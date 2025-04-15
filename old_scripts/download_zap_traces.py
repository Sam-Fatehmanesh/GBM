#!/usr/bin/env python3

import os
import numpy as np
import h5py
import tensorstore as ts
from tqdm import tqdm
import json

# Define output directory
OUTPUT_DIR = "raw_trace_data_zap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting ZAPBench trace data download...")

# Open the trace dataset
try:
    ds_traces = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': 'gs://zapbench-release/volumes/20240930/traces'
    }).result()
    
    print(f"Opened trace dataset with shape {ds_traces.shape} and dtype {ds_traces.dtype}")
    
    # Get the full traces data
    print("Downloading full trace matrix...")
    full_traces = ds_traces[:, :].read().result()
    print(f"Downloaded full trace matrix with shape {full_traces.shape}")
    
    # Save as HDF5 file
    h5_path = os.path.join(OUTPUT_DIR, "zap_traces_full.h5")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("traces", data=full_traces)
        f.attrs["shape"] = full_traces.shape
        f.attrs["source"] = "gs://zapbench-release/volumes/20240930/traces"
    
    print(f"Saved full traces to {h5_path}")
    
    # Also download the condition information
    conditions = [
        "forward_swimming", "turning", "capture_swim", "spontaneous", 
        "visual_looming", "visual_dimming", "touch", "paralyzed"
    ]
    
    # Dictionary to store condition bounds
    condition_data = {}
    
    # Try to get condition bounds programmatically if zapbench is installed
    try:
        from zapbench import constants
        from zapbench import data_utils
        
        print("ZAPBench package found, extracting condition bounds...")
        for condition_id, condition_name in enumerate(constants.CONDITION_NAMES):
            inclusive_min, exclusive_max = data_utils.get_condition_bounds(condition_id)
            condition_data[condition_name] = {
                "min_idx": int(inclusive_min),
                "max_idx": int(exclusive_max),
                "length": int(exclusive_max - inclusive_min)
            }
            
            # Save individual condition data
            condition_traces = full_traces[inclusive_min:exclusive_max, :]
            condition_h5_path = os.path.join(OUTPUT_DIR, f"zap_traces_{condition_name}.h5")
            
            with h5py.File(condition_h5_path, 'w') as f:
                f.create_dataset("traces", data=condition_traces)
                f.attrs["condition"] = condition_name
                f.attrs["min_idx"] = inclusive_min
                f.attrs["max_idx"] = exclusive_max
                f.attrs["shape"] = condition_traces.shape
            
            print(f"Saved {condition_name} traces to {condition_h5_path} with {condition_traces.shape[0]} timepoints")
    except ImportError:
        print("ZAPBench package not found, skipping condition extraction")
    
    # Save condition data
    with open(os.path.join(OUTPUT_DIR, "condition_info.json"), 'w') as f:
        json.dump(condition_data, f, indent=2)
    
    print("Download complete!")

except Exception as e:
    print(f"Error downloading trace data: {e}") 