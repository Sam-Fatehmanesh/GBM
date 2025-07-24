#!/usr/bin/env python3
"""
Analyze the sparsity of volumes tensor in H5 files.
Calculate the fraction of non-zero entries across all H5 files.
"""

import h5py
import numpy as np
import os
from pathlib import Path

def analyze_h5_file(filepath):
    """Analyze a single H5 file and return volume statistics."""
    print(f"\nAnalyzing: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Print available keys to understand structure
            print(f"  Keys in file: {list(f.keys())}")
            
            if 'volumes' in f:
                volumes = f['volumes']
                print(f"  Volumes shape: {volumes.shape}")
                print(f"  Volumes dtype: {volumes.dtype}")
                
                # Calculate statistics
                total_entries = volumes.size
                non_zero_entries = np.count_nonzero(volumes)
                zero_entries = total_entries - non_zero_entries
                
                fraction_nonzero = non_zero_entries / total_entries
                fraction_zero = zero_entries / total_entries
                
                print(f"  Total entries: {total_entries:,}")
                print(f"  Non-zero entries: {non_zero_entries:,}")
                print(f"  Zero entries: {zero_entries:,}")
                print(f"  Fraction non-zero: {fraction_nonzero:.6f} ({fraction_nonzero*100:.4f}%)")
                print(f"  Fraction zero: {fraction_zero:.6f} ({fraction_zero*100:.4f}%)")
                
                if non_zero_entries > 0:
                    min_val = np.min(volumes[volumes > 0])
                    max_val = np.max(volumes)
                    mean_nonzero = np.mean(volumes[volumes > 0])
                    print(f"  Min non-zero value: {min_val:.6f}")
                    print(f"  Max value: {max_val:.6f}")
                    print(f"  Mean non-zero value: {mean_nonzero:.6f}")
                
                return {
                    'filepath': filepath,
                    'total_entries': total_entries,
                    'non_zero_entries': non_zero_entries,
                    'zero_entries': zero_entries,
                    'fraction_nonzero': fraction_nonzero,
                    'shape': volumes.shape
                }
            else:
                print(f"  No 'volumes' dataset found in {filepath}")
                return None
                
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

def main():
    """Main function to analyze all H5 files."""
    data_dir = Path("processed_spike_voxels_2018")
    
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist!")
        return
    
    # Find all H5 files
    h5_files = list(data_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {data_dir}")
        return
    
    print(f"Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"  - {f}")
    
    # Analyze each file
    results = []
    total_entries_all = 0
    total_nonzero_all = 0
    
    for h5_file in h5_files:
        result = analyze_h5_file(h5_file)
        if result:
            results.append(result)
            total_entries_all += result['total_entries']
            total_nonzero_all += result['non_zero_entries']
    
    # Overall statistics
    if results:
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS ACROSS ALL FILES:")
        print(f"{'='*60}")
        print(f"Total files analyzed: {len(results)}")
        print(f"Total entries across all files: {total_entries_all:,}")
        print(f"Total non-zero entries across all files: {total_nonzero_all:,}")
        print(f"Total zero entries across all files: {total_entries_all - total_nonzero_all:,}")
        
        overall_fraction_nonzero = total_nonzero_all / total_entries_all
        overall_fraction_zero = 1 - overall_fraction_nonzero
        
        print(f"\nOverall fraction non-zero: {overall_fraction_nonzero:.6f} ({overall_fraction_nonzero*100:.4f}%)")
        print(f"Overall fraction zero: {overall_fraction_zero:.6f} ({overall_fraction_zero*100:.4f}%)")
        print(f"Sparsity (fraction of zeros): {overall_fraction_zero:.6f}")
        
        # Individual file breakdown
        print(f"\nPER-FILE BREAKDOWN:")
        print(f"{'File':<20} {'Shape':<20} {'Non-zero %':<12} {'Total Entries':<15}")
        print("-" * 75)
        for result in results:
            filename = Path(result['filepath']).name
            shape_str = f"{result['shape']}"
            fraction_pct = result['fraction_nonzero'] * 100
            print(f"{filename:<20} {shape_str:<20} {fraction_pct:<12.4f} {result['total_entries']:<15,}")

if __name__ == "__main__":
    main() 