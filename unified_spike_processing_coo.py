#!/usr/bin/env python3
"""
Unified Spike Processing Pipeline (COO Format)

This script combines the functionality of:
1. visualize_2018_spikes_cascade.py - CASCADE spike detection from calcium traces
2. prepare_dali_training_data.py - Data preparation for training
3. preprocess_spike_data.py - Grid conversion with z-plane augmentation

The unified pipeline processes raw calcium traces directly to final grid format
while generating visualization PDFs and eliminating intermediate files.

This version saves grid data in COO (Coordinate) sparse format for memory efficiency.
"""

import os
import time
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from tqdm import tqdm
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# CASCADE functional import
from neuralib.imaging.spikes.cascade import cascade_predict

# TensorFlow tuning (optional)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Parameters for ΔF/F baseline computation
SAMPLING_RATE = 2.0      # Hz
WINDOW_SEC    = 30.0     # seconds for sliding window
PERCENTILE    = 8        # percentile for F0
WINDOW_FRAMES = int(WINDOW_SEC * SAMPLING_RATE)

# Grid parameters
GRID_HEIGHT = 256
GRID_WIDTH = 128

def print_memory_stats(prefix=""):
    """Print memory usage statistics"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{prefix} Memory: RSS={mem_info.rss / 1e9:.2f}GB, VMS={mem_info.vms / 1e9:.2f}GB")
    except ImportError:
        print(f"{prefix} Memory: psutil not available")

def compute_dff(calcium_data):
    """
    Compute ΔF/F using an 8th-percentile, 30s sliding window per cell.
    
    Args:
        calcium_data: (T, N) array of calcium fluorescence traces
        
    Returns:
        dff: ΔF/F data of same shape
    """
    T, N = calcium_data.shape
    half_w = WINDOW_FRAMES // 2
    dff = np.zeros_like(calcium_data, dtype=np.float32)
    
    # Create sliding window indices for all timepoints
    starts = np.maximum(0, np.arange(T) - half_w)
    ends = np.minimum(T, np.arange(T) + half_w + 1)
    
    # Compute F0 for all cells at once using vectorized operations
    F0 = np.zeros((T, N), dtype=np.float32)
    for t in tqdm(range(T), desc="Computing baselines"):
        start, end = starts[t], ends[t]
        F0[t] = np.percentile(calcium_data[start:end], PERCENTILE, axis=0)
    
    # Compute ΔF/F for all cells
    dff = (calcium_data - F0)
    
    return dff

def run_cascade_inference(dff, batch_size=30000, model_type='Global_EXC_2Hz_smoothing500ms'):
    """
    Run CASCADE inference on ΔF/F data to get spike probabilities.
    
    Args:
        dff: (T, N) array of ΔF/F data
        batch_size: Batch size for CASCADE processing
        model_type: CASCADE model to use
        
    Returns:
        prob_data: (N, T) array of spike probabilities
        spike_data: (T, N) array of binary spikes
    """
    T, N = dff.shape
    
    # Prepare containers
    prob_data = np.zeros((N, T), dtype=np.float32)
    
    # Batched CASCADE inference
    print(f"  → Running CASCADE in batches of {batch_size}…")
    traces = dff.T.astype(np.float32)  # (N, T)
    
    for start in tqdm(range(0, N, batch_size), desc="CASCADE batches"):
        end = min(start + batch_size, N)
        batch = traces[start:end]
        
        batch_probs = cascade_predict(
            batch,
            model_type=model_type,
            threshold=1,
            padding=np.nan,
            verbose=False
        )
        
        batch_probs = np.atleast_2d(batch_probs)
        if batch_probs.shape != (end-start, T):
            raise ValueError(f"Unexpected batch_probs shape: {batch_probs.shape}, expected: {(end-start, T)}")
        
        prob_data[start:end] = batch_probs
    
    # Clamp probabilities and sample binary spikes
    print("  → Sampling binary spikes from probabilities…")
    rng = np.random.default_rng()
    prob_data = np.clip(prob_data, 0.0, 1.0)
    spike_data = (rng.random(prob_data.shape) < prob_data).astype(int).T  # (T, N)
    
    return prob_data, spike_data

def add_z_augmentation_markers(spike_coords, z_index, height=GRID_HEIGHT, width=GRID_WIDTH):
    """
    Add z-index based augmentation markers to spike coordinates.
    
    Args:
        spike_coords: List of existing spike coordinates for this timepoint/z-plane
        z_index: Current z-index for the frame
        height, width: Frame dimensions
        
    Returns:
        Updated spike_coords list with augmentation markers added
    """
    # Calculate marker dimensions based on z-index
    marker_height = 2  # Always 2 pixels high
    marker_width = z_index + 2  # z-index + 2 pixels wide
    
    # Ensure marker fits within the frame
    marker_width = min(marker_width, width)
    
    # Add marker coordinates to spike list
    augmentation_coords = []
    for x in range(marker_height):
        for y in range(width - marker_width, width):
            augmentation_coords.append([x, y])
    
    return spike_coords + augmentation_coords

def convert_to_coo_grids(spike_data, cell_positions, split_ratio=0.95, seed=42):
    """
    Convert spike data to COO (Coordinate) sparse format with z-plane augmentation and train/test split.
    
    Args:
        spike_data: (T, N) array of binary spikes
        cell_positions: (N, 3) array of cell positions
        split_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        coo_data: Dictionary containing COO sparse format data
        metadata: Dictionary containing dataset metadata
    """
    T, N = spike_data.shape
    
    # Handle NaN values in cell positions
    if np.isnan(cell_positions).any():
        print("Warning: NaN values found in cell positions, replacing with 0")
        cell_positions = np.nan_to_num(cell_positions)
    
    # Get unique z values (rounded to handle floating point precision)
    z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
    z_values = np.sort(z_values)  # Sort in ascending order
    num_z = len(z_values)
    
    print(f"Found {T} timepoints and {num_z} z-planes")
    
    # Normalize cell positions to [0, 1]
    pos_min = cell_positions.min(axis=0)
    pos_max = cell_positions.max(axis=0)
    pos_range = pos_max - pos_min
    
    # Handle case where range is 0 (all cells at same position)
    pos_range[pos_range == 0] = 1
    
    normalized_positions = (cell_positions - pos_min) / pos_range
    
    # Convert to grid indices
    cell_x = np.floor(normalized_positions[:, 0] * (GRID_HEIGHT - 1)).astype(np.int32)
    cell_y = np.floor(normalized_positions[:, 1] * (GRID_WIDTH - 1)).astype(np.int32)
    
    # Pre-compute z-plane masks and cell indices for each z-plane
    z_cell_indices = {}
    for z_idx, z_level in enumerate(z_values):
        z_mask = (np.round(cell_positions[:, 2], decimals=3) == z_level)
        z_cell_indices[z_idx] = {
            'x': cell_x[z_mask],
            'y': cell_y[z_mask],
            'indices': np.where(z_mask)[0]
        }
    
    # Create train/test split
    np.random.seed(seed)
    block_size = 330
    num_blocks = T // block_size
    
    # Create block indices and shuffle
    block_indices = np.arange(num_blocks)
    np.random.shuffle(block_indices)
    
    # Determine number of blocks for test set
    num_test_blocks = max(1, int(num_blocks * (1 - split_ratio)))
    
    # Select test blocks
    test_block_indices = block_indices[:num_test_blocks]
    train_block_indices = block_indices[num_test_blocks:]
    
    # Create masks for timepoints
    test_timepoints = []
    train_timepoints = []
    
    # Assign timepoints to train/test based on blocks
    for block_idx in range(num_blocks):
        start_timepoint = block_idx * block_size
        end_timepoint = min((block_idx + 1) * block_size, T)
        
        if block_idx in test_block_indices:
            test_timepoints.extend(range(start_timepoint, end_timepoint))
        else:
            train_timepoints.extend(range(start_timepoint, end_timepoint))
    
    # Handle any remaining timepoints (assign to training)
    if num_blocks * block_size < T:
        remaining_timepoints = range(num_blocks * block_size, T)
        train_timepoints.extend(remaining_timepoints)
    
    # Sort timepoints for easier processing
    test_timepoints.sort()
    train_timepoints.sort()
    
    print(f"Train set: {len(train_timepoints)} timepoints")
    print(f"Test set: {len(test_timepoints)} timepoints")
    
    # Create binary mask for train vs test
    is_train = np.zeros(T, dtype=np.uint8)
    is_train[train_timepoints] = 1
    
    # Convert to COO format
    spike_coords = []
    spike_values = []
    
    print("Converting spikes to COO sparse format...")
    for t in tqdm(range(T), desc="Processing timepoints"):
        for z_idx in range(num_z):
            # Get pre-computed cell indices for this z-plane
            cell_indices = z_cell_indices[z_idx]['indices']
            
            # Start with empty coordinate list for this timepoint/z-plane
            timepoint_coords = []
            
            # Process spikes if there are cells in this z-plane
            if len(cell_indices) > 0:
                # Get spikes for this timepoint and z-plane
                spikes_t = spike_data[t][cell_indices]
                
                # Find active cells
                active_cells = np.abs(spikes_t) > 1e-6
                
                if np.any(active_cells):
                    active_x = z_cell_indices[z_idx]['x'][active_cells]
                    active_y = z_cell_indices[z_idx]['y'][active_cells]
                    
                    # Add active cell coordinates
                    for i in range(len(active_x)):
                        timepoint_coords.append([active_x[i], active_y[i]])
            
            # Add z-plane augmentation markers
            timepoint_coords = add_z_augmentation_markers(timepoint_coords, z_idx)
            
            # Add coordinates to global list with timepoint and z-plane info
            for x, y in timepoint_coords:
                spike_coords.append([t, z_idx, x, y])
                spike_values.append(1)  # All spikes have value 1
    
    # Convert to numpy arrays
    spike_coords = np.array(spike_coords, dtype=np.int32) if spike_coords else np.empty((0, 4), dtype=np.int32)
    spike_values = np.array(spike_values, dtype=np.uint8) if spike_values else np.empty((0,), dtype=np.uint8)
    
    print(f"Total active pixels: {len(spike_coords)}")
    print(f"Sparsity: {len(spike_coords) / (T * num_z * GRID_HEIGHT * GRID_WIDTH) * 100:.4f}%")
    
    # Create COO data structure
    coo_data = {
        'spike_coords': spike_coords,  # (N_spikes, 4) with [t, z, x, y]
        'spike_values': spike_values,  # (N_spikes,) with spike values (all 1s)
        'grid_shape': np.array([T, num_z, GRID_HEIGHT, GRID_WIDTH], dtype=np.int32)
    }
    
    # Create metadata
    metadata = {
        'num_timepoints': T,
        'num_z_planes': num_z,
        'z_values': z_values,
        'train_timepoints': train_timepoints,
        'test_timepoints': test_timepoints,
        'is_train': is_train,
        'total_spikes': len(spike_coords),
        'sparsity_percent': len(spike_coords) / (T * num_z * GRID_HEIGHT * GRID_WIDTH) * 100
    }
    
    return coo_data, metadata

def coo_to_dense_grid(spike_coords, spike_values, grid_shape):
    """
    Convert COO sparse format back to dense grid format (for testing/verification).
    
    Args:
        spike_coords: (N_spikes, 4) array with [t, z, x, y] coordinates
        spike_values: (N_spikes,) array with spike values
        grid_shape: (4,) array with [T, Z, H, W] dimensions
        
    Returns:
        dense_grid: (T, Z, H, W) dense grid
    """
    T, Z, H, W = grid_shape
    dense_grid = np.zeros((T, Z, H, W), dtype=np.uint8)
    
    for i in range(len(spike_coords)):
        t, z, x, y = spike_coords[i]
        dense_grid[t, z, x, y] = spike_values[i]
    
    return dense_grid

def create_visualization_pdf(dff, prob_data, spike_data, subject_name, output_path, num_neurons=10):
    """
    Create visualization PDF showing ΔF/F, probabilities, and discrete spikes.
    
    Args:
        dff: (T, N) array of ΔF/F data
        prob_data: (N, T) array of spike probabilities  
        spike_data: (T, N) array of binary spikes
        subject_name: Name of the subject
        output_path: Path to save PDF
        num_neurons: Number of neurons to visualize
    """
    T, N = dff.shape
    
    print(f"  → Writing PDF to {output_path}")
    with PdfPages(output_path) as pdf:
        # Select random neurons to visualize
        sel = np.random.choice(N, min(num_neurons, N), replace=False)
        
        for idx in sel:
            fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            
            # ΔF/F trace
            ax[0].plot(dff[:, idx], 'k')
            ax[0].set_title(f'Neuron {idx}: ΔF/F')
            ax[0].set_ylabel('ΔF/F')
            
            # Spike probabilities
            ax[1].plot(prob_data[idx], 'm')
            ax[1].set_title('Inferred Spike Rate')
            ax[1].set_ylabel('Rate (spikes/frame)')
            
            # Discrete spikes
            ax[2].plot(spike_data[:, idx], 'r')
            ax[2].set_title('Discrete Spikes')
            ax[2].set_ylabel('Spike (0/1)')
            ax[2].set_xlabel('Frame')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def process_subject(subject_dir, output_dir, num_neurons=10, batch_size=30000, 
                   split_ratio=0.95, seed=42, cascade_model='Global_EXC_2Hz_smoothing500ms'):
    """
    Process a single subject through the complete pipeline.
    
    Args:
        subject_dir: Path to subject directory containing raw data
        output_dir: Output directory for processed data
        num_neurons: Number of neurons to visualize in PDF
        batch_size: Batch size for CASCADE processing
        split_ratio: Train/test split ratio
        seed: Random seed
        cascade_model: CASCADE model type to use
        
    Returns:
        subject_output_dir: Path to subject's output directory
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    print(f"\nProcessing subject: {subject_name}")
    
    # Create subject output directory
    subject_output_dir = os.path.join(output_dir, subject_name)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Check if already processed
    final_file = os.path.join(subject_output_dir, 'preaugmented_grids_coo.h5')
    pdf_file = os.path.join(subject_output_dir, f'{subject_name}_visualization.pdf')
    
    if os.path.exists(final_file) and os.path.exists(pdf_file):
        print("  → Already processed; skipping.")
        return subject_output_dir
    
    start_time = time.time()
    
    try:
        # Step 1: Load raw data
        print("  → Loading raw data...")
        
        # Load cell positions
        mat = loadmat(os.path.join(subject_dir, 'data_full.mat'))
        data0 = mat['data'][0, 0]
        cell_xyz = data0['CellXYZ']
        
        if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
            cell_xyz = cell_xyz[0, 0]
        
        # Handle invalid anatomical indices
        if 'IX_inval_anat' in data0.dtype.names:
            inval = data0['IX_inval_anat']
            if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                inval = inval[0, 0].flatten()
            mask = np.ones(cell_xyz.shape[0], bool)
            mask[np.array(inval, int) - 1] = False
            cell_xyz = cell_xyz[mask]
        
        # Load raw fluorescence traces
        with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
            calcium = f['CellResp'][:]  # shape = (T, N)
        
        T, N = calcium.shape
        assert N == cell_xyz.shape[0], f"Cell count mismatch: {N} vs {cell_xyz.shape[0]}"
        
        # Step 2: Compute ΔF/F
        print("  → Computing ΔF/F (8th-percentile, 30s sliding window)...")
        dff = compute_dff(calcium)
        
        # Step 3: Run CASCADE inference
        prob_data, spike_data = run_cascade_inference(dff, batch_size, cascade_model)
        
        # Step 4: Convert to COO sparse grid format
        print("  → Converting to COO sparse grid format with z-plane augmentation...")
        coo_data, metadata = convert_to_coo_grids(spike_data, cell_xyz, split_ratio, seed)
        
        # Step 5: Save final data in COO format
        print(f"  → Saving COO sparse data to {final_file}")
        with h5py.File(final_file, 'w') as f:
            # Save COO sparse data
            f.create_dataset('spike_coords', data=coo_data['spike_coords'],
                            compression='gzip', compression_opts=1)
            f.create_dataset('spike_values', data=coo_data['spike_values'],
                            compression='gzip', compression_opts=1)
            f.create_dataset('grid_shape', data=coo_data['grid_shape'])
            
            # Save timepoint indices and train/test mask
            f.create_dataset('timepoint_indices', 
                            data=np.arange(T, dtype=np.int32))
            f.create_dataset('is_train', data=metadata['is_train'])
            
            # Save attributes
            f.attrs['num_timepoints'] = metadata['num_timepoints']
            f.attrs['num_z_planes'] = metadata['num_z_planes']
            f.attrs['total_spikes'] = metadata['total_spikes']
            f.attrs['sparsity_percent'] = metadata['sparsity_percent']
            f.attrs['subject'] = subject_name
            f.attrs['cascade_model'] = cascade_model
            f.attrs['sampling_rate'] = SAMPLING_RATE
            f.attrs['window_sec'] = WINDOW_SEC
            f.attrs['percentile'] = PERCENTILE
            f.attrs['format'] = 'COO_sparse'
            f.attrs['grid_height'] = GRID_HEIGHT
            f.attrs['grid_width'] = GRID_WIDTH
        
        # Save metadata separately
        metadata_file = os.path.join(subject_output_dir, 'metadata.h5')
        with h5py.File(metadata_file, 'w') as f:
            f.create_dataset('num_timepoints', data=metadata['num_timepoints'])
            f.create_dataset('num_z_planes', data=metadata['num_z_planes'])
            f.create_dataset('z_values', data=metadata['z_values'])
            f.create_dataset('train_timepoints', data=metadata['train_timepoints'])
            f.create_dataset('test_timepoints', data=metadata['test_timepoints'])
            f.create_dataset('is_train', data=metadata['is_train'])
            f.attrs['total_spikes'] = metadata['total_spikes']
            f.attrs['sparsity_percent'] = metadata['sparsity_percent']
        
        # Step 6: Create visualization PDF
        print("  → Creating visualization PDF...")
        create_visualization_pdf(dff, prob_data, spike_data, subject_name, pdf_file, num_neurons)
        
        # Clean up memory
        del calcium, dff, prob_data, spike_data, coo_data
        gc.collect()
        
        processing_time = time.time() - start_time
        print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")
        print(f"  → Sparsity: {metadata['sparsity_percent']:.4f}%")
        
        return subject_output_dir
        
    except Exception as e:
        print(f"  → Error processing {subject_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Unified spike processing pipeline (COO sparse format): calcium traces → final grid format with visualizations'
    )
    parser.add_argument('--input_dir', type=str, default='raw_trace_data_2018',
                        help='Directory containing raw calcium trace data')
    parser.add_argument('--output_dir', type=str, default='processed_spike_grids_coo_2018',
                        help='Output directory for processed grid data')
    parser.add_argument('--num_neurons', type=int, default=10,
                        help='Number of neurons to visualize in PDFs')
    parser.add_argument('--batch_size', type=int, default=30000,
                        help='Batch size for CASCADE processing')
    parser.add_argument('--split_ratio', type=float, default=0.95,
                        help='Ratio of data to use for training')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cascade_model', type=str, default='Global_EXC_2Hz_smoothing500ms',
                        help='CASCADE model type to use')
    parser.add_argument('--skip', type=str, default='',
                        help='Comma-separated list of subjects to skip')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all subject directories
    skips = {s.strip() for s in args.skip.split(',') if s.strip()}
    subjects = sorted([
        os.path.join(args.input_dir, d)
        for d in os.listdir(args.input_dir)
        if d.startswith('subject_') and d not in skips and
        os.path.isdir(os.path.join(args.input_dir, d))
    ])
    
    if not subjects:
        print(f"No subject directories found in {args.input_dir}!")
        return
    
    print(f"Found {len(subjects)} subjects; skipping {skips}")
    
    # Process subjects
    processed_subjects = []
    
    if args.workers > 1:
        print(f"Processing subjects in parallel with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_subject,
                    subject_dir,
                    args.output_dir,
                    args.num_neurons,
                    args.batch_size,
                    args.split_ratio,
                    args.seed,
                    args.cascade_model
                )
                for subject_dir in subjects
            ]
            
            for future in tqdm(futures, desc="Processing subjects"):
                try:
                    result = future.result()
                    processed_subjects.append(result)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
    else:
        print("Processing subjects sequentially")
        for subject_dir in subjects:
            try:
                result = process_subject(
                    subject_dir,
                    args.output_dir,
                    args.num_neurons,
                    args.batch_size,
                    args.split_ratio,
                    args.seed,
                    args.cascade_model
                )
                processed_subjects.append(result)
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
    
    # Create combined metadata file
    print("\nCreating combined metadata file...")
    combined_metadata = {}
    total_spikes = 0
    total_pixels = 0
    
    for subject_dir in processed_subjects:
        subject_name = os.path.basename(subject_dir)
        metadata_path = os.path.join(subject_dir, 'metadata.h5')
        
        if os.path.exists(metadata_path):
            with h5py.File(metadata_path, 'r') as f:
                subject_data = {
                    'num_timepoints': f['num_timepoints'][()],
                    'num_z_planes': f['num_z_planes'][()],
                    'z_values': f['z_values'][:],
                    'train_timepoints': f['train_timepoints'][:],
                    'test_timepoints': f['test_timepoints'][:],
                    'is_train': f['is_train'][:],
                    'total_spikes': f.attrs['total_spikes'],
                    'sparsity_percent': f.attrs['sparsity_percent']
                }
                combined_metadata[subject_name] = subject_data
                
                # Accumulate totals for overall statistics
                total_spikes += subject_data['total_spikes']
                total_pixels += (subject_data['num_timepoints'] * 
                               subject_data['num_z_planes'] * 
                               GRID_HEIGHT * GRID_WIDTH)
    
    # Save combined metadata
    combined_metadata_path = os.path.join(args.output_dir, 'combined_metadata.h5')
    with h5py.File(combined_metadata_path, 'w') as f:
        for subject, data in combined_metadata.items():
            subject_group = f.create_group(subject)
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    subject_group.attrs[key] = value
                else:
                    subject_group.create_dataset(key, data=value)
        
        # Add global statistics
        f.attrs['total_subjects'] = len(processed_subjects)
        f.attrs['total_spikes'] = total_spikes
        f.attrs['total_pixels'] = total_pixels
        f.attrs['overall_sparsity_percent'] = (total_spikes / total_pixels * 100) if total_pixels > 0 else 0
        f.attrs['format'] = 'COO_sparse'
    
    print(f"\nUnified COO processing complete!")
    print(f"Processed {len(processed_subjects)} subjects")
    print(f"Output saved to: {args.output_dir}")
    print(f"Combined metadata: {combined_metadata_path}")
    
    # Print summary
    total_timepoints = sum(data['num_timepoints'] for data in combined_metadata.values())
    total_z_planes = sum(data['num_z_planes'] for data in combined_metadata.values())
    overall_sparsity = (total_spikes / total_pixels * 100) if total_pixels > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Total timepoints: {total_timepoints}")
    print(f"  Total z-planes: {total_z_planes}")
    print(f"  Grid size: {GRID_HEIGHT}x{GRID_WIDTH}")
    print(f"  Total spikes: {total_spikes:,}")
    print(f"  Overall sparsity: {overall_sparsity:.4f}%")
    print(f"  Memory savings vs dense: ~{100-overall_sparsity:.1f}%")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 