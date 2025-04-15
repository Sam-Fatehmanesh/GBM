#!/usr/bin/env python3

import h5py
import os
import numpy as np
import json

# Path to the trace data directory
DATA_DIR = "raw_trace_data_zap"

# Load condition info
with open(os.path.join(DATA_DIR, "condition_info.json"), 'r') as f:
    condition_info = json.load(f)

print("== ZAPBench Trace Data Summary ==")

# First check the full traces file
full_path = os.path.join(DATA_DIR, "zap_traces_full.h5")
with h5py.File(full_path, 'r') as f:
    traces = f['traces']
    print(f"\nFull Traces Dataset:")
    print(f"  Shape: {traces.shape}")
    print(f"  Dtype: {traces.dtype}")
    print(f"  Size: {traces.size * traces.dtype.itemsize / (1024**3):.2f} GB")
    
    # Basic statistics
    sample_traces = traces[:, :100]  # Sample for quick stats
    print(f"  Value range: [{np.min(sample_traces):.4f}, {np.max(sample_traces):.4f}]")
    print(f"  Mean: {np.mean(sample_traces):.4f}")
    print(f"  Std: {np.std(sample_traces):.4f}")
    
    # Check attributes
    print("  Attributes:")
    for attr in f.attrs:
        print(f"    {attr}: {f.attrs[attr]}")

# Check each condition file
print("\nCondition Datasets:")
condition_files = [f for f in os.listdir(DATA_DIR) if f.startswith("zap_traces_") and f != "zap_traces_full.h5"]

for condition_file in sorted(condition_files):
    file_path = os.path.join(DATA_DIR, condition_file)
    with h5py.File(file_path, 'r') as f:
        traces = f['traces']
        condition = condition_file.replace("zap_traces_", "").replace(".h5", "")
        print(f"\n  {condition}:")
        print(f"    Shape: {traces.shape}")
        print(f"    Timepoints: {traces.shape[0]}")
        print(f"    Neurons: {traces.shape[1]}")
        
        # Check attributes
        if 'condition' in f.attrs:
            print(f"    Condition (from attrs): {f.attrs['condition']}")
        if 'min_idx' in f.attrs and 'max_idx' in f.attrs:
            print(f"    Index range: [{f.attrs['min_idx']}, {f.attrs['max_idx']})")

print("\n== Comparison with GBM Data Structure ==")
print("ZAPBench trace structure:")
print("  - Shape: (timepoints, neurons)")
print("  - Data: Continuous neural activity traces (df/f)")
print("  - Organization: Separated by experimental conditions")

print("\nGBM spike data structure (from dali_spike_dataset.py):")
print("  - Shape: (timepoints, cells)")
print("  - Data: Binary spike events")
print("  - Organization: Cells organized by z-plane, processed into (batch, seq_len, 256, 128) grid format")

print("\nPotential integration approach:")
print("  1. Convert continuous ZAPBench traces to binary events using thresholding")
print("  2. Create spatial layout for ZAPBench neurons (not provided in original data)")
print("  3. Format into grid representation for compatibility with GBM model")
print("  4. Adjust DALIBrainDataLoader to handle both data sources") 