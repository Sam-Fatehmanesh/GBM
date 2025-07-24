#!/usr/bin/env python3
"""
Comprehensive analysis of processed spike data directories.
Provides detailed metadata on all subjects including sparsity, length, 
average spike probability, and other statistics.

Usage: python analyze_processed_spikes.py <directory_path>
"""

import h5py
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Try to import CuPy for GPU acceleration, fallback to NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - using GPU acceleration")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU (NumPy) processing")

def format_bytes(bytes_val):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def analyze_h5_file(filepath, verbose=False):
    """Analyze a single H5 file and return comprehensive statistics."""
    if verbose:
        print(f"\nAnalyzing: {filepath.name}")
    
    try:
        file_size = filepath.stat().st_size
        
        with h5py.File(filepath, 'r') as f:
            # Get basic file structure
            keys = list(f.keys())
            if verbose:
                print(f"  Keys in file: {keys}")
            
            if 'volumes' not in f:
                print(f"  WARNING: No 'volumes' dataset found in {filepath}")
                return None
                
            volumes_dataset = f['volumes']
            shape = volumes_dataset.shape
            dtype = volumes_dataset.dtype
            
            # Get metadata (handle scalar datasets properly)
            if 'num_timepoints' in f:
                num_timepoints = f['num_timepoints'][()]  # Read scalar value
            else:
                num_timepoints = shape[0]
                
            if 'volume_shape' in f:
                volume_shape = f['volume_shape'][:]  # Read array
            else:
                volume_shape = list(shape[1:])
            
            if verbose:
                print(f"  Volumes shape: {shape}")
                print(f"  Volumes dtype: {dtype}")
                print(f"  Number of timepoints: {num_timepoints}")
                print(f"  Volume dimensions (x,y,z): {volume_shape}")
            
            # Calculate basic statistics
            total_entries = volumes_dataset.size
            
            # For large datasets, sample a subset for statistics to avoid memory issues
            if total_entries > 50_000_000:  # 50M elements
                sample_size = min(1_000_000, total_entries // 100)  # Sample 1% or 1M, whichever is smaller
                
                # Sample random timepoints and spatial locations for efficiency
                n_timepoints_sample = min(100, shape[0])  # Sample max 100 timepoints
                time_indices = np.random.choice(shape[0], n_timepoints_sample, replace=False)
                time_indices = np.sort(time_indices)
                
                # Read sampled timepoints
                sample_data_list = []
                for t_idx in time_indices:
                    volume_slice = volumes_dataset[t_idx]  # Read one timepoint
                    # Further subsample spatial locations if needed
                    if volume_slice.size > sample_size // n_timepoints_sample:
                        flat_volume = volume_slice.flatten()
                        spatial_sample_size = sample_size // n_timepoints_sample
                        spatial_indices = np.random.choice(flat_volume.size, spatial_sample_size, replace=False)
                        sampled_volume = flat_volume[spatial_indices]
                    else:
                        sampled_volume = volume_slice.flatten()
                    sample_data_list.append(sampled_volume)
                
                # Combine all samples
                sample_data = np.concatenate(sample_data_list)
                
                # Transfer to GPU if available
                if GPU_AVAILABLE:
                    sample_data_gpu = cp.asarray(sample_data)
                    non_zero_sample = int(cp.count_nonzero(sample_data_gpu))
                    sample_fraction_nonzero = non_zero_sample / len(sample_data)
                    
                    # Value statistics from sample
                    if non_zero_sample > 0:
                        nonzero_values_gpu = sample_data_gpu[sample_data_gpu > 0]
                        min_val = float(cp.min(nonzero_values_gpu))
                        max_val = float(cp.max(sample_data_gpu))
                        mean_val = float(cp.mean(sample_data_gpu))
                        mean_nonzero_val = float(cp.mean(nonzero_values_gpu))
                        std_val = float(cp.std(sample_data_gpu))
                        median_nonzero_val = float(cp.median(nonzero_values_gpu))
                        # Clean up GPU memory
                        del sample_data_gpu, nonzero_values_gpu
                        if GPU_AVAILABLE:
                            cp.get_default_memory_pool().free_all_blocks()
                    else:
                        min_val = max_val = mean_val = mean_nonzero_val = std_val = median_nonzero_val = 0
                else:
                    non_zero_sample = np.count_nonzero(sample_data)
                    sample_fraction_nonzero = non_zero_sample / len(sample_data)
                    
                    # Value statistics from sample
                    if non_zero_sample > 0:
                        nonzero_values = sample_data[sample_data > 0]
                        min_val = np.min(nonzero_values)
                        max_val = np.max(sample_data)
                        mean_val = np.mean(sample_data)
                        mean_nonzero_val = np.mean(nonzero_values)
                        std_val = np.std(sample_data)
                        median_nonzero_val = np.median(nonzero_values)
                    else:
                        min_val = max_val = mean_val = mean_nonzero_val = std_val = median_nonzero_val = 0
                
                # Estimate total non-zero entries
                estimated_non_zero = int(sample_fraction_nonzero * total_entries)
                is_sampled = True
                
            else:
                # Load entire dataset for smaller files
                volumes_data = volumes_dataset[:]
                
                # Transfer to GPU if available
                if GPU_AVAILABLE:
                    volumes_data_gpu = cp.asarray(volumes_data)
                    non_zero_count = int(cp.count_nonzero(volumes_data_gpu))
                    estimated_non_zero = non_zero_count
                    sample_fraction_nonzero = non_zero_count / total_entries
                    
                    # Value statistics
                    if non_zero_count > 0:
                        nonzero_values_gpu = volumes_data_gpu[volumes_data_gpu > 0]
                        min_val = float(cp.min(nonzero_values_gpu))
                        max_val = float(cp.max(volumes_data_gpu))
                        mean_val = float(cp.mean(volumes_data_gpu))
                        mean_nonzero_val = float(cp.mean(nonzero_values_gpu))
                        std_val = float(cp.std(volumes_data_gpu))
                        median_nonzero_val = float(cp.median(nonzero_values_gpu))
                        # Clean up GPU memory
                        del volumes_data_gpu, nonzero_values_gpu
                        if GPU_AVAILABLE:
                            cp.get_default_memory_pool().free_all_blocks()
                    else:
                        min_val = max_val = mean_val = mean_nonzero_val = std_val = median_nonzero_val = 0
                else:
                    non_zero_count = np.count_nonzero(volumes_data)
                    estimated_non_zero = non_zero_count
                    sample_fraction_nonzero = non_zero_count / total_entries
                    
                    # Value statistics
                    if non_zero_count > 0:
                        nonzero_values = volumes_data[volumes_data > 0]
                        min_val = np.min(nonzero_values)
                        max_val = np.max(volumes_data)
                        mean_val = np.mean(volumes_data)
                        mean_nonzero_val = np.mean(nonzero_values)
                        std_val = np.std(volumes_data)
                        median_nonzero_val = np.median(nonzero_values)
                    else:
                        min_val = max_val = mean_val = mean_nonzero_val = std_val = median_nonzero_val = 0
                
                is_sampled = False
            
            # Calculate derived statistics
            fraction_nonzero = sample_fraction_nonzero
            fraction_zero = 1 - fraction_nonzero
            sparsity = fraction_zero
            
            # Memory estimates
            memory_per_timepoint = np.prod(volume_shape) * dtype.itemsize
            estimated_memory_gb = (total_entries * dtype.itemsize) / (1024**3)
            
            if verbose:
                print(f"  Total entries: {total_entries:,}")
                print(f"  {'Estimated' if is_sampled else 'Actual'} non-zero entries: {estimated_non_zero:,}")
                print(f"  Fraction non-zero: {fraction_nonzero:.6f} ({fraction_nonzero*100:.4f}%)")
                print(f"  Sparsity (zeros): {sparsity:.6f} ({sparsity*100:.4f}%)")
                if estimated_non_zero > 0:
                    print(f"  Min non-zero value: {min_val:.6f}")
                    print(f"  Max value: {max_val:.6f}")
                    print(f"  Mean value (all): {mean_val:.6f}")
                    print(f"  Mean non-zero value: {mean_nonzero_val:.6f}")
                    print(f"  Median non-zero value: {median_nonzero_val:.6f}")
                    print(f"  Std deviation: {std_val:.6f}")
                print(f"  File size: {format_bytes(file_size)}")
                print(f"  Estimated memory: {estimated_memory_gb:.2f} GB")
                if is_sampled:
                    print(f"  NOTE: Statistics based on random sample due to large dataset size")
            
            return {
                'filepath': str(filepath),
                'filename': filepath.name,
                'file_size_bytes': file_size,
                'file_size_human': format_bytes(file_size),
                'shape': shape,
                'dtype': str(dtype),
                'num_timepoints': int(num_timepoints),
                'volume_shape': list(volume_shape),
                'total_entries': int(total_entries),
                'estimated_non_zero_entries': int(estimated_non_zero),
                'fraction_nonzero': float(fraction_nonzero),
                'sparsity': float(sparsity),
                'min_nonzero_value': float(min_val) if estimated_non_zero > 0 else None,
                'max_value': float(max_val) if estimated_non_zero > 0 else None,
                'mean_value': float(mean_val),
                'mean_nonzero_value': float(mean_nonzero_val) if estimated_non_zero > 0 else None,
                'median_nonzero_value': float(median_nonzero_val) if estimated_non_zero > 0 else None,
                'std_deviation': float(std_val),
                'estimated_memory_gb': float(estimated_memory_gb),
                'memory_per_timepoint_bytes': int(memory_per_timepoint),
                'is_sampled': is_sampled,
                'keys': keys
            }
                
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None

def print_summary_table(results):
    """Print a nice summary table of all subjects."""
    if not results:
        return
    
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    
    # Header
    headers = ['Subject', 'Timepoints', 'Volume (X×Y×Z)', 'Sparsity %', 'Avg Prob', 'File Size', 'Memory (GB)']
    col_widths = [15, 12, 18, 12, 12, 12, 12]
    
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * 120)
    
    # Data rows
    for result in results:
        subject = result['filename'].replace('.h5', '')
        timepoints = f"{result['num_timepoints']:,}"
        volume = f"{result['volume_shape'][0]}×{result['volume_shape'][1]}×{result['volume_shape'][2]}"
        sparsity = f"{result['sparsity']*100:.2f}%"
        avg_prob = f"{result['mean_value']:.4f}" if result['mean_value'] is not None else "N/A"
        file_size = result['file_size_human']
        memory = f"{result['estimated_memory_gb']:.1f}"
        
        row = [subject, timepoints, volume, sparsity, avg_prob, file_size, memory]
        row_line = "".join(f"{str(r):<{w}}" for r, w in zip(row, col_widths))
        print(row_line)

def print_detailed_stats(results):
    """Print detailed statistics for each subject."""
    print(f"\n{'='*80}")
    print("DETAILED STATISTICS")
    print(f"{'='*80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['filename']}")
        print(f"   Shape: {result['shape']} | Data type: {result['dtype']}")
        print(f"   Timepoints: {result['num_timepoints']:,} | Volume: {result['volume_shape'][0]}×{result['volume_shape'][1]}×{result['volume_shape'][2]}")
        print(f"   Total entries: {result['total_entries']:,}")
        print(f"   Non-zero entries: {result['estimated_non_zero_entries']:,} ({'estimated' if result['is_sampled'] else 'exact'})")
        print(f"   Sparsity: {result['sparsity']*100:.4f}% | Non-zero: {result['fraction_nonzero']*100:.4f}%")
        
        if result['mean_nonzero_value'] is not None:
            print(f"   Values - Min: {result['min_nonzero_value']:.6f} | Max: {result['max_value']:.6f}")
            print(f"   Mean (all): {result['mean_value']:.6f} | Mean (non-zero): {result['mean_nonzero_value']:.6f}")
            print(f"   Median (non-zero): {result['median_nonzero_value']:.6f} | Std dev: {result['std_deviation']:.6f}")
        else:
            print(f"   All values are zero")
        
        print(f"   File size: {result['file_size_human']} | Memory: {result['estimated_memory_gb']:.2f} GB")
        print(f"   Memory per timepoint: {format_bytes(result['memory_per_timepoint_bytes'])}")

def print_aggregate_stats(results):
    """Print aggregate statistics across all subjects."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS ACROSS ALL SUBJECTS")
    print(f"{'='*80}")
    
    total_files = len(results)
    total_timepoints = sum(r['num_timepoints'] for r in results)
    total_entries = sum(r['total_entries'] for r in results)
    total_nonzero = sum(r['estimated_non_zero_entries'] for r in results)
    total_file_size = sum(r['file_size_bytes'] for r in results)
    total_memory = sum(r['estimated_memory_gb'] for r in results)
    
    overall_sparsity = (total_entries - total_nonzero) / total_entries
    overall_nonzero_fraction = total_nonzero / total_entries
    
    # Average values (weighted by number of entries)
    weighted_mean = sum(r['mean_value'] * r['total_entries'] for r in results) / total_entries
    
    # Get range of values across subjects
    all_means = [r['mean_value'] for r in results if r['mean_value'] is not None]
    all_sparsities = [r['sparsity'] for r in results]
    all_timepoints = [r['num_timepoints'] for r in results]
    
    print(f"Total subjects: {total_files}")
    print(f"Total timepoints: {total_timepoints:,}")
    print(f"Total entries: {total_entries:,}")
    print(f"Total non-zero entries: {total_nonzero:,}")
    print(f"Overall sparsity: {overall_sparsity:.6f} ({overall_sparsity*100:.4f}%)")
    print(f"Overall non-zero fraction: {overall_nonzero_fraction:.6f} ({overall_nonzero_fraction*100:.4f}%)")
    print(f"Weighted average spike probability: {weighted_mean:.6f}")
    print(f"Total disk usage: {format_bytes(total_file_size)}")
    print(f"Total estimated memory: {total_memory:.2f} GB")
    
    if all_timepoints:
        print(f"\nTimepoints per subject - Min: {min(all_timepoints):,} | Max: {max(all_timepoints):,} | Avg: {sum(all_timepoints)/len(all_timepoints):.1f}")
    
    if all_sparsities:
        print(f"Sparsity range - Min: {min(all_sparsities)*100:.2f}% | Max: {max(all_sparsities)*100:.2f}%")
    
    if all_means:
        print(f"Mean spike prob range - Min: {min(all_means):.6f} | Max: {max(all_means):.6f}")

def save_results_json(results, output_path):
    """Save results to JSON file."""
    # Add timestamp and summary
    output_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_subjects': len(results),
        'summary': {
            'total_timepoints': sum(r['num_timepoints'] for r in results),
            'total_entries': sum(r['total_entries'] for r in results),
            'total_nonzero_entries': sum(r['estimated_non_zero_entries'] for r in results),
            'overall_sparsity': (sum(r['total_entries'] for r in results) - sum(r['estimated_non_zero_entries'] for r in results)) / sum(r['total_entries'] for r in results),
            'total_file_size_bytes': sum(r['file_size_bytes'] for r in results),
            'total_memory_gb': sum(r['estimated_memory_gb'] for r in results)
        },
        'subjects': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze processed spike data directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_processed_spikes.py processed_spike_voxels_2018
  python analyze_processed_spikes.py /path/to/data --verbose --save-json results.json
        """
    )
    
    parser.add_argument('directory', help='Directory containing processed spike H5 files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output for each file')
    parser.add_argument('--save-json', help='Save detailed results to JSON file')
    parser.add_argument('--summary-only', action='store_true', help='Show only summary table')
    
    args = parser.parse_args()
    
    data_dir = Path(args.directory)
    
    if not data_dir.exists():
        print(f"ERROR: Directory {data_dir} does not exist!")
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory!")
        sys.exit(1)
    
    # Find all H5 files
    h5_files = list(data_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(h5_files)} H5 files in {data_dir}")
    if not args.summary_only:
        print(f"Files: {', '.join([f.name for f in h5_files])}")
    
    # Analyze each file
    print(f"\nAnalyzing {len(h5_files)} subjects...")
    results = []
    
    for h5_file in sorted(h5_files):
        result = analyze_h5_file(h5_file, verbose=args.verbose and not args.summary_only)
        if result:
            results.append(result)
    
    if not results:
        print("No valid H5 files could be analyzed!")
        sys.exit(1)
    
    # Print results
    print_summary_table(results)
    
    if not args.summary_only:
        print_detailed_stats(results)
    
    print_aggregate_stats(results)
    
    # Save to JSON if requested
    if args.save_json:
        save_results_json(results, args.save_json)
    
    print(f"\nAnalysis complete! Processed {len(results)} subjects.")

if __name__ == "__main__":
    main() 