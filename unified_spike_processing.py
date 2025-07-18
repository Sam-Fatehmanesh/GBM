#!/usr/bin/env python3
"""
Unified Spike Processing Pipeline

This script combines the functionality of:
1. visualize_2018_spikes_cascade.py - CASCADE spike detection from calcium traces
2. prepare_dali_training_data.py - Data preparation for training
3. preprocess_spike_data.py - Grid conversion with z-plane augmentation

The unified pipeline processes either:
- Raw calcium traces directly to final grid format (original functionality)
- Pre-computed spike probabilities from CASCADE/OASIS to final grid format (new functionality)

While generating visualization PDFs and eliminating intermediate files.

Support for both OASIS (default) and CASCADE spike detection methods.
"""

import scipy
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

# Spike detection imports
try:
    from neuralib.imaging.spikes.cascade import cascade_predict
    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False

try:
    from oasis.functions import estimate_parameters, deconvolve
    OASIS_AVAILABLE = True
except ImportError:
    OASIS_AVAILABLE = False

# TensorFlow tuning (optional) - only if CASCADE is available
if CASCADE_AVAILABLE:
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

def load_precomputed_spike_data(precomputed_file, use_probabilities=False):
    """
    Load pre-computed spike probabilities and binary spikes from HDF5 file.
    
    Args:
        precomputed_file: Path to HDF5 file with pre-computed spike data
        use_probabilities: If True, use probabilities instead of binary spikes
        
    Returns:
        tuple: (calcium_data, prob_data, spike_data, cell_xyz, method_used)
            - calcium_data: (T, N) raw calcium data (if available, else None)
            - prob_data: (N, T) spike probabilities 
            - spike_data: (T, N) binary spikes or probabilities (based on use_probabilities flag)
            - cell_xyz: (N, 3) cell positions
            - method_used: string indicating 'cascade' or 'oasis'
    """
    print(f"  → Loading pre-computed spike data from {precomputed_file}")
    
    with h5py.File(precomputed_file, 'r') as f:
        # Load cell positions
        cell_xyz = f['cell_positions'][:]
        
        # Load spike data - format varies by method
        if 'probabilities' in f:
            # CASCADE format: has 'probabilities' and 'spikes'
            prob_data = f['probabilities'][:]  # (N, T)
            spike_data_binary = f['spikes'][:]        # (T, N) 
            method_used = 'cascade'
            
            # Handle NaN values in probabilities
            prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Choose between probabilities and binary based on flag
            if use_probabilities:
                # Use probabilities, transpose to match expected (T, N) format
                spike_data = prob_data.T  # (N, T) -> (T, N)
                print(f"    Using continuous probabilities instead of binary spikes")
            else:
                # Use binary spikes
                spike_data = spike_data_binary
            
            # Check if we have model_type attribute
            if 'model_type' in f.attrs:
                model_type = f.attrs['model_type']
                if isinstance(model_type, bytes):
                    model_type = model_type.decode()
                elif hasattr(model_type, 'item'):  # numpy scalar
                    model_type = model_type.item()
                    if isinstance(model_type, bytes):
                        model_type = model_type.decode()
                print(f"    Detected CASCADE data with model: {model_type}")
            else:
                print("    Detected CASCADE data")
                
        elif 'calcium_denoised' in f:
            # OASIS format: has 'calcium_denoised' (used as probabilities) and 'spikes'
            prob_data = f['calcium_denoised'][:].T  # (T, N) -> (N, T)
            spike_data_binary = f['spikes'][:]             # (T, N)
            method_used = 'oasis'
            
            # Handle NaN values in probabilities
            prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # For OASIS, probabilities are denoised calcium values
            if use_probabilities:
                spike_data = prob_data.T  # (N, T) -> (T, N)
                print(f"    Using continuous denoised calcium instead of binary spikes")
            else:
                spike_data = spike_data_binary
            
            if 'g' in f.attrs:
                g_val = f.attrs['g']
                if hasattr(g_val, 'item'):  # numpy scalar
                    g_val = g_val.item()
                print(f"    Detected OASIS data with g={g_val:.3f}")
            else:
                print("    Detected OASIS data")
                
        else:
            raise ValueError(f"Unrecognized spike data format in {precomputed_file}. Expected 'probabilities' (CASCADE) or 'calcium_denoised' (OASIS)")
        
        # Try to load raw calcium data if available (not always present)
        calcium_data = None
        if 'calcium_raw' in f:
            calcium_data = f['calcium_raw'][:]
        
        data_type = "probabilities" if use_probabilities else "binary spikes"
        print(f"    Loaded {spike_data.shape[1]} cells, {spike_data.shape[0]} timepoints ({data_type})")
        
    return calcium_data, prob_data, spike_data, cell_xyz, method_used

def compute_dff(calcium_data):
    """
    Compute ΔF/F using an 8th-percentile, 30s sliding window per cell.
    
    Args:
        calcium_data: (T, N) array of calcium fluorescence traces
        
    Returns:
        dff: ΔF/F data of same shape
    """
    """Compute ΔF/F using causal (past-only) sliding window."""
    T, N = calcium_data.shape
    dff = np.zeros_like(calcium_data, dtype=np.float32)
    
    # Use CAUSAL window - only past and current frames
    F0 = np.zeros((T, N), dtype=np.float32)
    for t in tqdm(range(T), desc="Computing baselines"):
        # Only use frames from max(0, t-window_size) to t (inclusive)
        start = max(0, t - WINDOW_FRAMES + 1)
        end = t + 1
        F0[t] = np.percentile(calcium_data[start:end], PERCENTILE, axis=0)
    
    dff = (calcium_data - F0)
    
    return dff

def run_cascade_inference(dff, batch_size=30000, model_type='Global_EXC_2Hz_smoothing500ms', keep_probabilities=False):
    """
    Run CASCADE inference on ΔF/F data to get spike probabilities.
    
    Args:
        dff: (T, N) array of ΔF/F data
        batch_size: Batch size for CASCADE processing
        model_type: CASCADE model to use
        keep_probabilities: If True, use probabilities directly instead of sampling binary spikes
        
    Returns:
        prob_data: (N, T) array of spike probabilities
        spike_data: (T, N) array of binary spikes (or probabilities if keep_probabilities=True)
    """
    if not CASCADE_AVAILABLE:
        raise ImportError("CASCADE not available. Please install neuralib or use OASIS method.")
    
    T, N = dff.shape
    
    # Dynamically adjust batch size based on number of neurons to avoid OOM
    if N > 50000:
        batch_size = min(batch_size, 5000)
        print(f"  → Large dataset detected ({N} neurons), reducing batch size to {batch_size}")
    elif N > 20000:
        batch_size = min(batch_size, 10000)
        print(f"  → Medium dataset detected ({N} neurons), reducing batch size to {batch_size}")
    
    # Prepare containers - use float16 to save memory
    prob_data = np.zeros((N, T), dtype=np.float16)
    
    # Batched CASCADE inference with memory cleanup
    print(f"  → Running CASCADE in batches of {batch_size}…")
    traces = dff.T.astype(np.float32)  # (N, T)
    
    # Delete original dff to free memory
    del dff
    gc.collect()
    
    for start in tqdm(range(0, N, batch_size), desc="CASCADE batches"):
        end = min(start + batch_size, N)
        batch = traces[start:end]
        
        try:
            batch_probs = cascade_predict(
                batch,
                model_type=model_type,
                threshold=0,
                padding=np.nan,
                verbose=False
            )
            
            batch_probs = np.atleast_2d(batch_probs)
            if batch_probs.shape != (end-start, T):
                raise ValueError(f"Unexpected batch_probs shape: {batch_probs.shape}, expected: {(end-start, T)}")
            
            # Handle NaN, inf values and ensure probabilities are in [0, 1] range
            batch_probs = np.nan_to_num(batch_probs, nan=0.0, posinf=1.0, neginf=0.0)
            batch_probs = np.clip(batch_probs, 0.0, 1.0)
            
            # Store as float16 to save memory
            prob_data[start:end] = batch_probs.astype(np.float16)
            
            # Clean up batch results
            del batch_probs, batch
            gc.collect()
            
        except Exception as e:
            print(f"Error processing batch {start}-{end}: {e}")
            # Fill with zeros if batch fails
            prob_data[start:end] = 0
            del batch
            gc.collect()
    
    # Clean up traces
    del traces
    gc.collect()
    
    # Handle spike data based on keep_probabilities flag
    if keep_probabilities:
        print("  → Using probabilities directly as spike data…")
        spike_data = prob_data.T.copy().astype(np.float16)  # (N, T) -> (T, N), matching expected format
    else:
        print("  → Sampling binary spikes from probabilities…")
        # Convert to float32 for binomial sampling with additional safety checks
        prob_for_sampling = prob_data.astype(np.float32)
        
        # Additional safety: handle any remaining NaN/inf values and ensure [0, 1] range
        prob_for_sampling = np.nan_to_num(prob_for_sampling, nan=0.0, posinf=1.0, neginf=0.0)
        prob_for_sampling = np.clip(prob_for_sampling, 0.0, 1.0)
        
        # Final validation before binomial sampling
        if np.any(np.isnan(prob_for_sampling)) or np.any(np.isinf(prob_for_sampling)):
            print("  → Warning: Still found NaN/inf values after cleanup, setting to 0")
            prob_for_sampling = np.nan_to_num(prob_for_sampling, nan=0.0, posinf=1.0, neginf=0.0)
        
        if np.any(prob_for_sampling < 0) or np.any(prob_for_sampling > 1):
            print("  → Warning: Found values outside [0,1] range after clipping, re-clipping")
            prob_for_sampling = np.clip(prob_for_sampling, 0.0, 1.0)
        
        # Now safe to use binomial sampling
        spike_data = np.random.binomial(1, prob_for_sampling).astype(np.uint8).T  # (N, T) -> (T, N)
        del prob_for_sampling
        gc.collect()
    
    return prob_data, spike_data

def run_oasis_inference(calcium_data, fudge_factor=0.98, penalty=1, threshold_factor=0.333):
    """
    Run OASIS deconvolution on raw calcium data to get spike probabilities.
    
    Args:
        calcium_data: (T, N) array of raw calcium fluorescence data
        fudge_factor: Fudge factor for parameter estimation
        penalty: Penalty parameter for deconvolution
        threshold_factor: Factor to multiply noise standard deviation for thresholding
        
    Returns:
        prob_data: (N, T) array of spike probabilities (denoised calcium)
        spike_data: (T, N) array of binary spikes
    """
    if not OASIS_AVAILABLE:
        raise ImportError("OASIS not available. Please install oasis-deconvolution or use CASCADE method.")
    
    T, N = calcium_data.shape
    
    # Initialize arrays for results
    spike_data = np.zeros_like(calcium_data, dtype=np.float32)
    calcium_denoised = np.zeros_like(calcium_data, dtype=np.float32)
    g_values = np.zeros(N)
    
    print("  → Running OASIS deconvolution...")
    
    for i in tqdm(range(N), desc="OASIS processing"):
        y = calcium_data[:, i].astype(np.float64)
        
        # Check for NaN values
        if np.isnan(y).any():
            print(f"Warning: NaN values found in cell {i}, replacing with zeros")
            y = np.nan_to_num(y)
        
        try:
            # Estimate parameters
            est = estimate_parameters(y, p=1, fudge_factor=fudge_factor)
            g = est[0]
            sn = est[1]
            g_values[i] = g[0] if len(g) > 0 else 0.95
            
            # Deconvolve
            out = deconvolve(y, g=g, sn=sn, penalty=penalty)
            c = out[0]  # Denoised calcium
            s = out[1]  # Spike amplitudes
            
            # Apply threshold to get binary spikes
            threshold = threshold_factor * sn
            s_binary = np.where(s >= threshold, 1, 0)
            
            calcium_denoised[:, i] = c
            spike_data[:, i] = s_binary
            
        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")
            # Use default values if processing fails
            calcium_denoised[:, i] = y
            spike_data[:, i] = 0
    
    # For OASIS, we use denoised calcium as "probabilities" and return binary spikes
    # Transpose to match CASCADE format: (N, T) for prob_data
    prob_data = calcium_denoised.T.astype(np.float32)
    
    return prob_data, spike_data

def run_spike_inference(data, method='oasis', **kwargs):
    """
    Run spike inference using the specified method.
    
    Args:
        data: Input data (ΔF/F for CASCADE, raw calcium for OASIS)
        method: 'cascade' or 'oasis'
        **kwargs: Additional arguments passed to the specific method
        
    Returns:
        prob_data: (N, T) array of spike probabilities
        spike_data: (T, N) array of binary spikes (or probabilities if keep_probabilities=True for CASCADE)
    """
    if method.lower() == 'cascade':
        return run_cascade_inference(data, **kwargs)
    elif method.lower() == 'oasis':
        return run_oasis_inference(data, **kwargs)
    else:
        raise ValueError(f"Unknown spike inference method: {method}. Use 'cascade' or 'oasis'.")

def augment_z_frames(frame, z_index, num_z_planes, height=GRID_HEIGHT, width=GRID_WIDTH, use_probabilities=False):
    """
    Apply z-index based augmentation to a frame.
    
    Args:
        frame: Numpy array of shape (height, width) 
        z_index: Current z-index for the frame
        num_z_planes: Total number of z-planes
        height, width: Frame dimensions
        use_probabilities: If True, the frame contains continuous probabilities
        
    Returns:
        Augmented frame
    """
    # Clone the frame to avoid modifying the original
    augmented_frame = frame.copy()
    
    # Zero out the entire z-plane marker area (top-right column down to num_z_planes)
    marker_area_height = min(num_z_planes, height)
    augmented_frame[0:marker_area_height, width-1] = 0
    
    # Set only the z-index pixel to 1 (or 1.0 for probabilities)
    if z_index < height:
        marker_value = 1.0 if use_probabilities else 1
        augmented_frame[z_index, width-1] = marker_value
    
    return augmented_frame

def convert_to_grids(spike_data, cell_positions, split_ratio=0.95, seed=42, use_probabilities=False):
    """
    Convert spike data to grid format with z-plane augmentation and train/test split.
    
    Args:
        spike_data: (T, N) array of binary spikes or continuous probabilities
        cell_positions: (N, 3) array of cell positions
        split_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        use_probabilities: If True, spike_data contains continuous probabilities
        
    Returns:
        grids: (T, num_z, height, width) array of grid data
        metadata: Dictionary containing dataset metadata
    """
    T, N = spike_data.shape
    
    # Handle NaN values in cell positions
    if np.isnan(cell_positions).any():
        print("Warning: NaN values found in cell positions, replacing with 0")
        cell_positions = np.nan_to_num(cell_positions)
    
    # Handle NaN values in spike data
    if np.isnan(spike_data).any():
        print("Warning: NaN values found in spike data, replacing with 0")
        spike_data = np.nan_to_num(spike_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get unique z values (rounded to handle floating point precision)
    z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
    z_values = np.sort(z_values)  # Sort in ascending order
    num_z = len(z_values)
    
    data_type = "probabilities" if use_probabilities else "binary spikes"
    print(f"Found {T} timepoints and {num_z} z-planes, processing {data_type}")
    
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
    
    # Create grids - use float32 for probabilities, uint8 for binary
    # Store probability grids in float16 to save space and speed up I/O
    grid_dtype = np.float16 if use_probabilities else np.uint8
    grids = np.zeros((T, num_z, GRID_HEIGHT, GRID_WIDTH), dtype=grid_dtype)
    
    print("Converting spikes to grid format...")
    for t in tqdm(range(T), desc="Processing timepoints"):
        for z_idx in range(num_z):
            # Get pre-computed cell indices for this z-plane
            cell_indices = z_cell_indices[z_idx]['indices']
            
            # Create empty grid for this timepoint and z-plane
            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=grid_dtype)
            
            # Skip if no cells in this z-plane
            if len(cell_indices) == 0:
                grid = augment_z_frames(grid, z_idx, num_z, use_probabilities=use_probabilities)
                grids[t, z_idx] = grid
                continue
            
            # Get spikes for this timepoint and z-plane
            spikes_t = spike_data[t][cell_indices]
            
            # Find active cells - different logic for binary vs probabilities
            if use_probabilities:
                # For probabilities, use any non-zero value
                active_cells = np.abs(spikes_t) > 1e-6
            else:
                # For binary, use threshold
                active_cells = np.abs(spikes_t) > 1e-6
            
            if np.any(active_cells):
                active_x = z_cell_indices[z_idx]['x'][active_cells]
                active_y = z_cell_indices[z_idx]['y'][active_cells]
                active_values = spikes_t[active_cells]
                
                # Set active cells in the grid
                for i in range(len(active_x)):
                    if use_probabilities:
                        # For probabilities, use the actual probability value
                        grid[active_x[i], active_y[i]] = active_values[i]
                    else:
                        # For binary, set to 1
                        grid[active_x[i], active_y[i]] = 1
            
            # Apply z-plane augmentation
            grid = augment_z_frames(grid, z_idx, num_z, use_probabilities=use_probabilities)
            grids[t, z_idx] = grid
    
    # Create metadata
    metadata = {
        'num_timepoints': T,
        'num_z_planes': num_z,
        'z_values': z_values,
        'train_timepoints': train_timepoints,
        'test_timepoints': test_timepoints,
        'is_train': is_train,
        'use_probabilities': use_probabilities,
        'data_type': data_type
    }
    
    return grids, metadata

def create_visualization_pdf(calcium_data, prob_data, spike_data, subject_name, output_path, 
                           num_neurons=10, method='oasis'):
    """
    Create visualization PDF showing calcium, probabilities, and discrete spikes.
    
    Args:
        calcium_data: (T, N) array of raw calcium data
        prob_data: (N, T) array of spike probabilities  
        spike_data: (T, N) array of binary spikes
        subject_name: Name of the subject
        output_path: Path to save PDF
        num_neurons: Number of neurons to visualize
        method: Spike detection method used
    """
    T, N = calcium_data.shape
    
    print(f"  → Writing PDF to {output_path}")
    with PdfPages(output_path) as pdf:
        # Select random neurons to visualize
        sel = np.random.choice(N, min(num_neurons, N), replace=False)
        
        for idx in sel:
            fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            
            # Raw calcium trace
            ax[0].plot(calcium_data[:, idx], 'k')
            ax[0].set_title(f'Neuron {idx}: Raw Calcium')
            ax[0].set_ylabel('Fluorescence')
            
            # Spike probabilities/denoised calcium
            if method.lower() == 'cascade':
                ax[1].plot(prob_data[idx], 'm')
                ax[1].set_title('Inferred Spike Rate (CASCADE)')
                ax[1].set_ylabel('Rate (spikes/frame)')
            else:  # OASIS
                ax[1].plot(prob_data[idx], 'b')
                ax[1].set_title('Denoised Calcium (OASIS)')
                ax[1].set_ylabel('Fluorescence')
            
            # Discrete spikes
            ax[2].plot(spike_data[:, idx], 'r')
            ax[2].set_title(f'Discrete Spikes ({method.upper()})')
            ax[2].set_ylabel('Spike (0/1)')
            ax[2].set_xlabel('Frame')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def process_subject(subject_dir, output_dir, num_neurons=10, batch_size=30000, 
                   split_ratio=0.95, seed=42, spike_method='oasis', 
                   cascade_model='Global_EXC_2Hz_smoothing500ms',
                   oasis_fudge_factor=0.98, oasis_penalty=1, oasis_threshold_factor=0.333,
                   precomputed_dir=None, keep_cascade_probabilities=False, use_probabilities=False,
                   test_run_neurons=None, original_sampling_rate=None, target_sampling_rate=None):
    """
    Process a single subject through the complete pipeline.
    
    Args:
        subject_dir: Path to subject directory containing raw data (ignored if precomputed_dir is provided)
        output_dir: Output directory for processed data
        num_neurons: Number of neurons to visualize in PDF
        batch_size: Batch size for CASCADE processing
        split_ratio: Train/test split ratio
        seed: Random seed
        spike_method: 'oasis' or 'cascade' for spike detection method (ignored if precomputed_dir is provided)
        cascade_model: CASCADE model type to use (if using CASCADE)
        oasis_fudge_factor: Fudge factor for OASIS parameter estimation
        oasis_penalty: Penalty parameter for OASIS deconvolution
        oasis_threshold_factor: Factor to multiply noise std for OASIS thresholding
        precomputed_dir: Path to directory with pre-computed spike data (if provided, skips spike inference)
        keep_cascade_probabilities: If True, use CASCADE probabilities directly instead of sampling binary spikes
        use_probabilities: If True, use probabilities instead of binary spikes in final output
        test_run_neurons: If specified, randomly select only this many neurons for testing
        original_sampling_rate: Original sampling rate of the calcium traces in Hz (required for CASCADE)
        target_sampling_rate: Target sampling rate for interpolation in Hz (required for CASCADE)
        
    Returns:
        subject_output_dir: Path to subject's output directory
    """
    if precomputed_dir:
        # Use pre-computed spike data
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        precomputed_file = os.path.join(precomputed_dir, f"{subject_name}_spikes.h5")
        
        if not os.path.exists(precomputed_file):
            print(f"Warning: Pre-computed file not found: {precomputed_file}")
            return None
            
        print(f"\nProcessing subject: {subject_name} using pre-computed spike data")
        
        # Create subject output directory
        subject_output_dir = os.path.join(output_dir, subject_name)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Check if already processed
        final_file = os.path.join(subject_output_dir, 'preaugmented_grids.h5')
        pdf_file = os.path.join(subject_output_dir, f'{subject_name}_visualization.pdf')
        
        if os.path.exists(final_file) and os.path.exists(pdf_file):
            print("  → Already processed; skipping.")
            return subject_output_dir
        
        start_time = time.time()
        
        try:
            # Load pre-computed spike data
            calcium_data, prob_data, spike_data, cell_xyz, method_used = load_precomputed_spike_data(precomputed_file, use_probabilities=use_probabilities)
            
            # Apply test run neuron selection if specified
            if test_run_neurons is not None:
                T, N = spike_data.shape
                if test_run_neurons < N:
                    print(f"  → Test run: selecting {test_run_neurons} random neurons out of {N}")
                    np.random.seed(seed)  # For reproducibility
                    selected_indices = np.random.choice(N, test_run_neurons, replace=False)
                    selected_indices = np.sort(selected_indices)  # Keep sorted for easier processing
                    
                    # Filter all data to selected neurons
                    spike_data = spike_data[:, selected_indices]
                    prob_data = prob_data[selected_indices, :]
                    cell_xyz = cell_xyz[selected_indices, :]
                    
                    if calcium_data is not None:
                        calcium_data = calcium_data[:, selected_indices]
                    
                    print(f"  → Reduced to {test_run_neurons} neurons for testing")
                else:
                    print(f"  → Test run: requested {test_run_neurons} neurons but only {N} available, using all")
            
            # If we don't have raw calcium data, create dummy data for visualization
            if calcium_data is None:
                print("  → No raw calcium data available, creating dummy data for visualization")
                T, N = spike_data.shape
                calcium_data = np.random.randn(T, N).astype(np.float32)
            
            # Continue with grid conversion and visualization
            print("  → Converting to grid format with z-plane augmentation...")
            grids, metadata = convert_to_grids(spike_data, cell_xyz, split_ratio, seed, use_probabilities=use_probabilities)
            
            # Save final data
            print(f"  → Saving final data to {final_file}")
            with h5py.File(final_file, 'w') as f:
                # Save main datasets
                f.create_dataset('grids', 
                                data=grids,
                                chunks=(1, 1, GRID_HEIGHT, GRID_WIDTH),
                                compression='gzip',
                                compression_opts=1)
                
                f.create_dataset('timepoint_indices', 
                                data=np.arange(spike_data.shape[0], dtype=np.int32))
                
                f.create_dataset('is_train', data=metadata['is_train'])
                
                # Save attributes
                f.attrs['num_timepoints'] = metadata['num_timepoints']
                f.attrs['num_z_planes'] = metadata['num_z_planes']
                f.attrs['subject'] = subject_name
                f.attrs['spike_method'] = method_used
                f.attrs['data_source'] = 'precomputed'
                f.attrs['precomputed_file'] = precomputed_file
                f.attrs['use_probabilities'] = use_probabilities
                f.attrs['data_type'] = metadata['data_type']
            
            # Save metadata separately
            metadata_file = os.path.join(subject_output_dir, 'metadata.h5')
            with h5py.File(metadata_file, 'w') as f:
                f.create_dataset('num_timepoints', data=metadata['num_timepoints'])
                f.create_dataset('num_z_planes', data=metadata['num_z_planes'])
                f.create_dataset('z_values', data=metadata['z_values'])
                f.create_dataset('train_timepoints', data=metadata['train_timepoints'])
                f.create_dataset('test_timepoints', data=metadata['test_timepoints'])
                f.create_dataset('is_train', data=metadata['is_train'])
                f.attrs['use_probabilities'] = use_probabilities
                f.attrs['data_type'] = metadata['data_type']
            
            # Create visualization PDF
            print("  → Creating visualization PDF...")
            create_visualization_pdf(calcium_data, prob_data, spike_data, subject_name, 
                                    pdf_file, num_neurons, method_used)
            
            # Clean up memory
            del calcium_data, prob_data, spike_data, grids
            gc.collect()
            
            processing_time = time.time() - start_time
            print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")
            
            return subject_output_dir
            
        except Exception as e:
            print(f"  → Error processing {subject_name}: {e}")
            raise
    
    else:
        # Original functionality - process raw calcium data
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        print(f"\nProcessing subject: {subject_name} using {spike_method.upper()}")
        
        # Create subject output directory
        subject_output_dir = os.path.join(output_dir, subject_name)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Check if already processed
        final_file = os.path.join(subject_output_dir, 'preaugmented_grids.h5')
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
            
            # Load fluorescence traces - prefer CellRespZ (already ΔF/F) if available
            with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
                if 'CellRespZ' in f:
                    calcium = f['CellRespZ'][:]  # May be (N, T) or (T, N)
                    is_already_dff = True
                    print("  → Using CellRespZ traces (already ΔF/F)")
                else:
                    calcium = f['CellResp'][:]  # May be (N, T) or (T, N)
                    is_already_dff = False
                    print("  → Using CellResp traces (raw fluorescence)")
            
            # Check dimensions and transpose if needed
            # The calcium data might be stored as (N, T) instead of (T, N)
            if calcium.shape[0] == cell_xyz.shape[0]:
                # First dimension matches number of cells, so data is (N, T)
                print(f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)")
                calcium = calcium.T
            elif calcium.shape[1] == cell_xyz.shape[0]:
                # Second dimension matches number of cells, so data is already (T, N)
                print(f"  → Calcium data is already (T, N) = {calcium.shape}")
            else:
                # Neither dimension matches - this is the problematic case
                print(f"  → Warning: Calcium shape {calcium.shape} doesn't match cell count {cell_xyz.shape[0]}")
                print(f"  → Assuming calcium is (T, N) and truncating to match")
                
            T, N_original = calcium.shape
            
            # Handle invalid anatomical indices
            anatomical_mask = np.ones(N_original, bool)
            if 'IX_inval_anat' in data0.dtype.names:
                inval = data0['IX_inval_anat']
                if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                    inval = inval[0, 0].flatten()
                # Only apply anatomical mask if indices are within range
                inval_indices = np.array(inval, int) - 1  # Convert to 0-based indexing
                valid_inval_indices = inval_indices[inval_indices < N_original]
                if len(valid_inval_indices) > 0:
                    anatomical_mask[valid_inval_indices] = False
            
            # Filter out neurons with invalid coordinates (NaN values)
            coordinate_mask = ~np.isnan(cell_xyz).any(axis=1)
            
            # Combine both masks
            valid_mask = anatomical_mask & coordinate_mask
            
            # Count how many neurons are being filtered out
            n_invalid_anatomical = np.sum(~anatomical_mask)
            n_invalid_coordinates = np.sum(~coordinate_mask)
            n_total_filtered = np.sum(~valid_mask)
            
            if n_invalid_anatomical > 0:
                print(f"  → Filtering out {n_invalid_anatomical} neurons with invalid anatomical indices")
            if n_invalid_coordinates > 0:
                print(f"  → Filtering out {n_invalid_coordinates} neurons with invalid coordinates (NaN values)")
            if n_total_filtered > 0:
                print(f"  → Total neurons filtered: {n_total_filtered} out of {N_original} ({n_total_filtered/N_original*100:.1f}%)")
            # Apply the combined mask to both cell positions and calcium data
            cell_xyz = cell_xyz[valid_mask]
            calcium = calcium[:, valid_mask]  # Filter columns (neurons)
            
            T, N = calcium.shape
            print(f"  → Retained {N} neurons with valid coordinates out of {N_original} total")
            
            # Apply test run neuron selection if specified
            if test_run_neurons is not None and test_run_neurons < N:
                print(f"  → Test run: selecting {test_run_neurons} random neurons out of {N}")
                np.random.seed(seed)  # For reproducibility
                selected_indices = np.random.choice(N, test_run_neurons, replace=False)
                selected_indices = np.sort(selected_indices)  # Keep sorted for easier processing
                
                # Filter calcium data and cell positions to selected neurons
                calcium = calcium[:, selected_indices]
                cell_xyz = cell_xyz[selected_indices, :]
                
                T, N = calcium.shape
                print(f"  → Reduced to {N} neurons for testing")
            elif test_run_neurons is not None:
                print(f"  → Test run: requested {test_run_neurons} neurons but only {N} available, using all")
            
            # Keep a copy of calcium for visualization (sample subset to save memory)
            calcium_for_viz = calcium[:, :min(num_neurons * 2, N)].copy()
            
            # Step 2: Run spike inference based on method
            if spike_method.lower() == 'cascade':
                # Use ΔF/F for CASCADE
                # if is_already_dff:
                #     print("  → Using existing ΔF/F traces for CASCADE...")
                #     dff = compute_dff(calcium)# calcium.copy()  # Make a copy to avoid modifying original
                # else:
                #     print("  → Computing ΔF/F (8th-percentile, 30s sliding window)...")
                #     dff = compute_dff(calcium)

                # perform linear interpolation on the calcium trace using scipy
                print(f"  → Interpolating calcium traces from {original_sampling_rate}Hz to {target_sampling_rate}Hz...")
                
                # Validate sampling rates are provided for CASCADE
                if original_sampling_rate is None or target_sampling_rate is None:
                    raise ValueError("--original_sampling_rate and --target_sampling_rate must be provided when using CASCADE")
                
                # Create original time points
                original_time = np.arange(T) / original_sampling_rate
                
                # Create new time points at target sampling rate
                new_T = int(T * target_sampling_rate / original_sampling_rate)
                new_time = np.arange(new_T) / target_sampling_rate
                
                # Interpolate each neuron's trace
                from scipy.interpolate import interp1d
                calcium_interpolated = np.zeros((new_T, N), dtype=calcium.dtype)
                
                for n in range(N):
                    interp_func = interp1d(original_time, calcium[:, n], kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    calcium_interpolated[:, n] = interp_func(new_time)
                
                # Replace calcium with interpolated version
                calcium = calcium_interpolated
                T = new_T
                print(f"  → Interpolated to {T} timepoints at {target_sampling_rate}Hz")


                # perform dff on the whole calcium trace
                dff = compute_dff(calcium)
                
                # Clean up calcium to free memory
                del calcium
                gc.collect()
                print_memory_stats("After ΔF/F computation")
                
                # Run CASCADE inference
                # If we want probabilities in final output, avoid binary sampling entirely
                cascade_keep_probs = keep_cascade_probabilities or use_probabilities
                prob_data, spike_data = run_spike_inference(
                    dff, method='cascade', 
                    batch_size=batch_size, 
                    model_type=cascade_model,
                    keep_probabilities=cascade_keep_probs
                )
                
                # Clean up after CASCADE
                del dff
                gc.collect()
                print_memory_stats("After CASCADE inference")
                
            elif spike_method.lower() == 'oasis':
                # OASIS expects raw calcium, but can also work with ΔF/F
                if is_already_dff:
                    print("  → Using existing ΔF/F traces for OASIS...")
                    dff = calcium.copy()  # Make a copy to avoid modifying original
                else:
                    print("  → Computing ΔF/F (8th-percentile, 30s sliding window)...")
                    dff = compute_dff(calcium)
                
                # Clean up calcium to free memory
                del calcium
                gc.collect()
                print_memory_stats("After ΔF/F computation")
                
                prob_data, spike_data = run_spike_inference(
                    dff, method='oasis',
                    fudge_factor=oasis_fudge_factor,
                    penalty=oasis_penalty, 
                    threshold_factor=oasis_threshold_factor
                )
                
                # Clean up after OASIS
                del dff
                gc.collect()
                print_memory_stats("After OASIS inference")
                
            else:
                raise ValueError(f"Unknown spike method: {spike_method}. Use 'oasis' or 'cascade'.")
            
            # Step 3: Convert to grid format
            print("  → Converting to grid format with z-plane augmentation...")
            grids, metadata = convert_to_grids(spike_data, cell_xyz, split_ratio, seed, use_probabilities=use_probabilities)
            
            # Step 4: Save final data
            print(f"  → Saving final data to {final_file}")
            with h5py.File(final_file, 'w') as f:
                # Save main datasets
                f.create_dataset('grids', 
                                data=grids,
                                chunks=(1, 1, GRID_HEIGHT, GRID_WIDTH),
                                compression='gzip',
                                compression_opts=1)
                
                f.create_dataset('timepoint_indices', 
                                data=np.arange(T, dtype=np.int32))
                
                f.create_dataset('is_train', data=metadata['is_train'])
                
                # Save attributes
                f.attrs['num_timepoints'] = metadata['num_timepoints']
                f.attrs['num_z_planes'] = metadata['num_z_planes']
                f.attrs['subject'] = subject_name
                f.attrs['spike_method'] = spike_method
                f.attrs['sampling_rate'] = SAMPLING_RATE
                f.attrs['data_source'] = 'raw_calcium'
                f.attrs['use_probabilities'] = use_probabilities
                f.attrs['data_type'] = metadata['data_type']
                f.attrs['is_already_dff'] = is_already_dff
                f.attrs['calcium_source'] = 'CellRespZ' if is_already_dff else 'CellResp'
                
                # Method-specific attributes
                if spike_method.lower() == 'cascade':
                    f.attrs['cascade_model'] = cascade_model
                    f.attrs['window_sec'] = WINDOW_SEC
                    f.attrs['percentile'] = PERCENTILE
                else:  # OASIS
                    f.attrs['oasis_fudge_factor'] = oasis_fudge_factor
                    f.attrs['oasis_penalty'] = oasis_penalty
                    f.attrs['oasis_threshold_factor'] = oasis_threshold_factor
                
                # Force write to disk
                f.flush()
            print(f"  → ✓ Saved grid data: {final_file}")
            
            # Save metadata separately
            metadata_file = os.path.join(subject_output_dir, 'metadata.h5')
            with h5py.File(metadata_file, 'w') as f:
                f.create_dataset('num_timepoints', data=metadata['num_timepoints'])
                f.create_dataset('num_z_planes', data=metadata['num_z_planes'])
                f.create_dataset('z_values', data=metadata['z_values'])
                f.create_dataset('train_timepoints', data=metadata['train_timepoints'])
                f.create_dataset('test_timepoints', data=metadata['test_timepoints'])
                f.create_dataset('is_train', data=metadata['is_train'])
                f.attrs['use_probabilities'] = use_probabilities
                f.attrs['data_type'] = metadata['data_type']
                
                # Force write to disk
                f.flush()
            print(f"  → ✓ Saved metadata: {metadata_file}")
            
            # Step 5: Create visualization PDF
            print("  → Creating visualization PDF...")
            create_visualization_pdf(calcium_for_viz, prob_data, spike_data, subject_name, 
                                    pdf_file, num_neurons, spike_method)
            print(f"  → ✓ Saved visualization: {pdf_file}")
            
            # Clean up memory
            del calcium_for_viz, prob_data, spike_data, grids
            gc.collect()
            
            processing_time = time.time() - start_time
            print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")
            
            return subject_output_dir
            
        except Exception as e:
            print(f"  → Error processing {subject_name}: {e}")
            raise

def update_combined_metadata(subject_output_dir, output_dir, processing_mode, args):
    """
    Update the combined metadata file with results from a single subject.
    This allows progressive saving so completed subjects aren't lost if the process crashes.
    """
    subject_name = os.path.basename(subject_output_dir)
    metadata_path = os.path.join(subject_output_dir, 'metadata.h5')
    combined_metadata_path = os.path.join(output_dir, 'combined_metadata.h5')
    
    if not os.path.exists(metadata_path):
        print(f"  → Warning: Metadata file not found for {subject_name}")
        return
    
    # Load subject metadata
    try:
        with h5py.File(metadata_path, 'r') as f:
            subject_data = {
                'num_timepoints': f['num_timepoints'][()],
                'num_z_planes': f['num_z_planes'][()],
                'z_values': f['z_values'][:],
                'train_timepoints': f['train_timepoints'][:],
                'test_timepoints': f['test_timepoints'][:],
                'is_train': f['is_train'][:]
            }
            if 'use_probabilities' in f.attrs:
                subject_data['use_probabilities'] = f.attrs['use_probabilities']
            if 'data_type' in f.attrs:
                subject_data['data_type'] = f.attrs['data_type']
    except Exception as e:
        print(f"  → Error reading metadata for {subject_name}: {e}")
        return
    
    # Update or create combined metadata file
    try:
        # Try to open existing file, create if doesn't exist
        mode = 'a' if os.path.exists(combined_metadata_path) else 'w'
        with h5py.File(combined_metadata_path, mode) as f:
            # Set global attributes only if creating new file
            if mode == 'w':
                if processing_mode == "precomputed":
                    f.attrs['data_source'] = 'precomputed'
                    f.attrs['precomputed_dir'] = args.precomputed_dir
                else:
                    f.attrs['data_source'] = 'raw_calcium'
                    f.attrs['spike_method'] = args.spike_method
                f.attrs['use_probabilities'] = args.use_probabilities
            
            # Remove existing subject group if it exists (for reprocessing)
            if subject_name in f:
                del f[subject_name]
            
            # Add subject data
            subject_group = f.create_group(subject_name)
            for key, value in subject_data.items():
                if key in ['use_probabilities', 'data_type']:
                    subject_group.attrs[key] = value
                else:
                    subject_group.create_dataset(key, data=value)
            
            # Force write to disk
            f.flush()
            
        print(f"  → ✓ Updated combined metadata for {subject_name}")
        
    except Exception as e:
        print(f"  → Error updating combined metadata for {subject_name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Unified spike processing pipeline: calcium traces → final grid format with visualizations'
    )
    parser.add_argument('--input_dir', type=str, default='raw_trace_data_2018',
                        help='Directory containing raw calcium trace data (ignored if --precomputed_dir is provided)')
    parser.add_argument('--output_dir', type=str, default='processed_spike_grids_2018',
                        help='Output directory for processed grid data')
    parser.add_argument('--precomputed_dir', type=str, default=None,
                        help='Directory containing pre-computed spike data (HDF5 files from CASCADE/OASIS). If provided, skips spike inference.')
    parser.add_argument('--spike_method', type=str, default='oasis', choices=['oasis', 'cascade'],
                        help='Spike detection method: oasis (default) or cascade (ignored if --precomputed_dir is provided)')
    parser.add_argument('--num_neurons', type=int, default=10,
                        help='Number of neurons to visualize in PDFs')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Batch size for CASCADE processing')
    parser.add_argument('--split_ratio', type=float, default=0.95,
                        help='Ratio of data to use for training')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_probabilities', action='store_true',
                        help='Use continuous probabilities instead of binary spikes in final output')
    
    # CASCADE-specific arguments
    parser.add_argument('--cascade_model', type=str, default='Global_EXC_2Hz_smoothing500ms',
                        help='CASCADE model type to use')
    parser.add_argument('--keep_cascade_probabilities', action='store_true',
                        help='Use CASCADE probabilities directly instead of sampling binary spikes')
    
    # OASIS-specific arguments
    parser.add_argument('--oasis_fudge_factor', type=float, default=0.98,
                        help='Fudge factor for OASIS parameter estimation')
    parser.add_argument('--oasis_penalty', type=int, default=1,
                        help='Penalty parameter for OASIS deconvolution')
    parser.add_argument('--oasis_threshold_factor', type=float, default=0.333,
                        help='Factor to multiply noise std for OASIS thresholding')
    
    parser.add_argument('--skip', type=str, default='',
                        help='Comma-separated list of subjects to skip')
    parser.add_argument('--test_run_neurons', type=int, default=None,
                        help='For testing: randomly select only this many neurons from each subject')
    parser.add_argument('--original_sampling_rate', type=float, default=None,
                        help='Original sampling rate of the calcium traces in Hz (e.g., 2.73) - required when using CASCADE')
    parser.add_argument('--target_sampling_rate', type=float, default=None,
                        help='Target sampling rate for interpolation in Hz (e.g., 2.5 or 3.0) - required when using CASCADE')
    
    args = parser.parse_args()
    
    # Validate CASCADE-specific requirements
    if not args.precomputed_dir and args.spike_method.lower() == 'cascade':
        if args.original_sampling_rate is None or args.target_sampling_rate is None:
            parser.error("--original_sampling_rate and --target_sampling_rate are required when using CASCADE")
    
    # Determine processing mode
    if args.precomputed_dir:
        print(f"Using pre-computed spike data from: {args.precomputed_dir}")
        if not os.path.exists(args.precomputed_dir):
            print(f"Error: Pre-computed directory not found: {args.precomputed_dir}")
            return
        input_dir = args.input_dir  # Still need this for subject directory structure
        processing_mode = "precomputed"
    else:
        print(f"Processing raw calcium data from: {args.input_dir}")
        # Check method availability for raw processing
        if args.spike_method.lower() == 'cascade' and not CASCADE_AVAILABLE:
            print("Error: CASCADE not available. Please install neuralib or use OASIS method.")
            return
        elif args.spike_method.lower() == 'oasis' and not OASIS_AVAILABLE:
            print("Error: OASIS not available. Please install oasis-deconvolution or use CASCADE method.")
            return
        input_dir = args.input_dir
        processing_mode = "raw"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all subject directories
    skips = {s.strip() for s in args.skip.split(',') if s.strip()}
    
    if processing_mode == "precomputed":
        # Find subjects based on available pre-computed files
        precomputed_files = [f for f in os.listdir(args.precomputed_dir) if f.endswith('_spikes.h5')]
        subject_names = [f.replace('_spikes.h5', '') for f in precomputed_files if f.replace('_spikes.h5', '') not in skips]
        subjects = [os.path.join(input_dir, name) for name in subject_names if name.startswith('subject_')]
        print(f"Found {len(precomputed_files)} pre-computed spike files")
    else:
        # Original logic for raw data
        subjects = sorted([
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.startswith('subject_') and d not in skips and
            os.path.isdir(os.path.join(input_dir, d))
        ])
    
    if not subjects:
        if processing_mode == "precomputed":
            print(f"No pre-computed spike files found in {args.precomputed_dir}!")
        else:
            print(f"No subject directories found in {input_dir}!")
        return
    
    print(f"Found {len(subjects)} subjects; skipping {skips}")
    if processing_mode == "precomputed":
        print(f"Using pre-computed spike data")
    else:
        print(f"Using spike detection method: {args.spike_method.upper()}")
    
    if args.use_probabilities:
        print("Using continuous probabilities instead of binary spikes in final output")
    else:
        print("Using binary spikes in final output")
    
    # Process subjects
    processed_subjects = []
    
    # Force sequential processing for CASCADE to avoid memory issues
    use_parallel = args.workers > 1 and not (processing_mode == "raw" and args.spike_method.lower() == 'cascade')
    
    if use_parallel:
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
                    args.spike_method,
                    args.cascade_model,
                    args.oasis_fudge_factor,
                    args.oasis_penalty,
                    args.oasis_threshold_factor,
                    args.precomputed_dir,
                    args.keep_cascade_probabilities,
                    args.use_probabilities,
                    args.test_run_neurons,
                    args.original_sampling_rate,
                    args.target_sampling_rate
                )
                for subject_dir in subjects
            ]
            
            for future in tqdm(futures, desc="Processing subjects"):
                try:
                    result = future.result()
                    if result is not None:
                        processed_subjects.append(result)
                        
                        # Update combined metadata file immediately after each subject
                        update_combined_metadata(result, args.output_dir, processing_mode, args)
                        
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
    else:
        if processing_mode == "raw" and args.spike_method.lower() == 'cascade':
            print("Processing subjects sequentially (CASCADE requires sequential processing to avoid memory issues)")
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
                    args.spike_method,
                    args.cascade_model,
                    args.oasis_fudge_factor,
                    args.oasis_penalty,
                    args.oasis_threshold_factor,
                    args.precomputed_dir,
                    args.keep_cascade_probabilities,
                    args.use_probabilities,
                    args.test_run_neurons,
                    args.original_sampling_rate,
                    args.target_sampling_rate
                )
                if result is not None:
                    processed_subjects.append(result)
                    
                    # Update combined metadata file immediately after each subject
                    update_combined_metadata(result, args.output_dir, processing_mode, args)
                    
                # Force garbage collection between subjects when using CASCADE
                if processing_mode == "raw" and args.spike_method.lower() == 'cascade':
                    gc.collect()
                    print_memory_stats("After processing subject")
                    
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
                gc.collect()  # Clean up after errors too
    
    # Load combined metadata file (progressively saved during processing)
    print("\nReading combined metadata file...")
    combined_metadata = {}
    combined_metadata_path = os.path.join(args.output_dir, 'combined_metadata.h5')
    
    if os.path.exists(combined_metadata_path):
        with h5py.File(combined_metadata_path, 'r') as f:
            for subject_name in f.keys():
                subject_group = f[subject_name]
                subject_data = {}
                
                # Load datasets
                for key in ['num_timepoints', 'num_z_planes', 'z_values', 'train_timepoints', 'test_timepoints', 'is_train']:
                    if key in subject_group:
                        subject_data[key] = subject_group[key][()]
                
                # Load attributes
                for key in ['use_probabilities', 'data_type']:
                    if key in subject_group.attrs:
                        subject_data[key] = subject_group.attrs[key]
                
                combined_metadata[subject_name] = subject_data
    else:
        print("  → Warning: Combined metadata file not found - this may indicate no subjects were successfully processed")
    
    print(f"\nUnified processing complete!")
    print(f"Processed {len(processed_subjects)} subjects using {processing_mode} data")
    if args.test_run_neurons is not None:
        print(f"Test mode: Used only {args.test_run_neurons} randomly selected neurons per subject")
    print(f"Output saved to: {args.output_dir}")
    print(f"Combined metadata: {combined_metadata_path}")
    
    # Print summary
    total_timepoints = sum(data['num_timepoints'] for data in combined_metadata.values())
    total_z_planes = sum(data['num_z_planes'] for data in combined_metadata.values())
    
    print(f"\nSummary:")
    if processing_mode == "precomputed":
        print(f"  Data source: Pre-computed spike data")
        print(f"  Pre-computed directory: {args.precomputed_dir}")
    else:
        print(f"  Data source: Raw calcium traces")
        print(f"  Spike detection method: {args.spike_method.upper()}")
    print(f"  Using probabilities: {args.use_probabilities}")
    print(f"  Total timepoints: {total_timepoints}")
    print(f"  Total z-planes: {total_z_planes}")
    print(f"  Grid size: {GRID_HEIGHT}x{GRID_WIDTH}")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 