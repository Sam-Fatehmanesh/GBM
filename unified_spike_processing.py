#!/usr/bin/env python3
"""
Spike Probability Processing Pipeline

This script processes calcium traces using CASCADE spike detection and saves
spike probabilities as time series data along with cell spatial positions.
No volumetric conversion is performed - data is saved directly as probability
time series and 3D position coordinates.

Features:
- CASCADE-only spike detection (no OASIS or binary spikes)
- Direct spike probability time series output (T, N) format
- Cell spatial positions preserved as (N, 3) coordinates
- YAML configuration support
- Continuous probability values only
- Configurable float16/float32 output data types
"""

import os
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from tqdm import tqdm
import gc
import multiprocessing as mp
import yaml
import argparse

# Spike detection imports
try:
    from neuralib.imaging.spikes.cascade import cascade_predict
    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False

# TensorFlow tuning (optional) - only if CASCADE is available
if CASCADE_AVAILABLE:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

def print_memory_stats(prefix=""):
    """Print memory usage statistics"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"{prefix} Memory: RSS={mem_info.rss / 1e9:.2f}GB, VMS={mem_info.vms / 1e9:.2f}GB")
    except ImportError:
        print(f"{prefix} Memory: psutil not available")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_default_config():
    """Create default configuration dictionary"""
    return {
        'data': {
            'input_dir': 'raw_trace_data_2018',
            'output_dir': 'processed_spike_voxels_2018',
            'skip_subjects': [],
            'test_run_neurons': None,
            'calcium_dataset': 'CellResp',  # Name of calcium trace dataset in TimeSeries.h5
            'is_raw': True,  # Whether data is raw (applies proper ΔF/F: (F-F0)/F0)
            'apply_baseline_subtraction': False,  # Whether to apply baseline subtraction only (F-F0)
            'window_length': 30.0,  # Window size in seconds for baseline computation
            'baseline_percentile': 8,  # Percentile for baseline computation
        },
        'processing': {
            'num_neurons_viz': 10,
            'batch_size': 5000,
            'workers': 1,
            'seed': 42,
            'original_sampling_rate': None,  # Required for CASCADE
            'target_sampling_rate': None,    # Required for CASCADE
            'return_to_original_rate': False,  # If True, downsample neural data back to original rate after CASCADE
        },
        'cascade': {
            'model_type': 'Global_EXC_2Hz_smoothing500ms',
        },
        'output': {
            'dtype': 'float16',  # Data type for probabilities and positions
            'include_additional_data': True,  # Include anat_stack, stimulus, behavior, eye data
        }
    }

def save_default_config(config_path):
    """Save default configuration to YAML file"""
    config = create_default_config()
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Default configuration saved to {config_path}")



def compute_baseline_correction(calcium_data, window_length=30.0, percentile=8, sampling_rate=2.0, 
                              is_raw=True, apply_baseline_subtraction=False):
    """
    Apply baseline correction using a causal sliding window per cell.
    
    Args:
        calcium_data: (T, N) array of calcium fluorescence traces
        window_length: Window size in seconds
        percentile: Percentile for baseline computation
        sampling_rate: Sampling rate in Hz
        is_raw: If True, compute proper ΔF/F: (F-F0)/F0
        apply_baseline_subtraction: If True and is_raw is False, compute baseline subtraction: F-F0
        
    Returns:
        processed_data: Processed data based on the specified method
    """
    T, N = calcium_data.shape
    window_frames = int(window_length * sampling_rate)
    
    # If neither processing is requested, return original data
    if not is_raw and not apply_baseline_subtraction:
        print("  → Using traces as-is (no baseline correction)")
        return calcium_data.copy()
    
    # Compute baseline F0 using sliding window
    F0 = np.zeros((T, N), dtype=np.float32)
    for t in tqdm(range(T), desc="Computing baselines"):
        # Only use frames from max(0, t-window_size) to t (inclusive)
        start = max(0, t - window_frames + 1)
        end = t + 1
        F0[t] = np.percentile(calcium_data[start:end], percentile, axis=0)
    
    # Apply the appropriate correction based on flags
    if is_raw:
        # Compute proper ΔF/F: (F - F0) / F0
        print("  → Computing ΔF/F: (F - F0) / F0")
        # Avoid division by zero
        F0_safe = np.where(F0 == 0, 1, F0)
        processed_data = (calcium_data - F0) / F0_safe
    elif apply_baseline_subtraction:
        # Compute baseline subtraction: F - F0
        print("  → Computing baseline subtraction: F - F0")
        processed_data = calcium_data - F0
    
    return processed_data

def run_cascade_inference(calcium_data, batch_size=5000, model_type='Global_EXC_2Hz_smoothing500ms', sampling_rate=2.0):
    """
    Run CASCADE inference on calcium data to get spike probabilities.
    
    Args:
        calcium_data: (T, N) array of calcium data (raw or baseline-subtracted)
        batch_size: Batch size for CASCADE processing
        model_type: CASCADE model to use
        sampling_rate: Sampling rate in Hz for converting spike rates to probabilities
        
    Returns:
        prob_data: (N, T) array of spike probabilities converted from CASCADE spike rates
    """
    if not CASCADE_AVAILABLE:
        raise ImportError("CASCADE not available. Please install neuralib.")
    
    T, N = calcium_data.shape
    
    # Dynamically adjust batch size based on number of neurons to avoid OOM
    if N > 50000:
        batch_size = min(batch_size, 5000)
        print(f"  → Large dataset detected ({N} neurons), reducing batch size to {batch_size}")
    elif N > 20000:
        batch_size = min(batch_size, 10000)
        print(f"  → Medium dataset detected ({N} neurons), reducing batch size to {batch_size}")
    
    # Prepare containers
    prob_data = np.zeros((N, T), dtype=np.float32)
    
    # Batched CASCADE inference
    print(f"  → Running CASCADE in batches of {batch_size}…")
    traces = calcium_data.T.astype(np.float32)  # (N, T)
    
    # Delete original calcium data to free memory
    del calcium_data
    gc.collect()
    
    for start in tqdm(range(0, N, batch_size), desc="CASCADE batches"):
        end = min(start + batch_size, N)
        batch = traces[start:end]
        
        try:
            batch_rates = cascade_predict(
                batch,
                model_type=model_type,
                threshold=0,
                padding=np.nan,
                verbose=False
            )
            
            batch_rates = np.atleast_2d(batch_rates)
            if batch_rates.shape != (end-start, T):
                raise ValueError(f"Unexpected batch_rates shape: {batch_rates.shape}, expected: {(end-start, T)}")
            
            # Handle NaN, inf values - CASCADE outputs spike rates in Hz
            batch_rates = np.nan_to_num(batch_rates, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert spike rates to probabilities using Poisson process: P = 1 - e^(-λ/F)
            # where λ is spike rate (Hz) and F is sampling rate (Hz)
            batch_probs = 1.0 - np.exp(-batch_rates / sampling_rate)
            
            # Ensure probabilities are in [0, 1] range (should be by construction, but safety check)
            batch_probs = np.clip(batch_probs, 0.0, 1.0)
            
            prob_data[start:end] = batch_probs
            
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
    
    return prob_data



def create_visualization_pdf(calcium_data, prob_data, subject_name, output_path, num_neurons=10):
    """
    Create visualization PDF showing calcium traces and spike probabilities.
    
    Args:
        calcium_data: (T, N) array of raw calcium data
        prob_data: (N, T) array of spike probabilities  
        subject_name: Name of the subject
        output_path: Path to save PDF
        num_neurons: Number of neurons to visualize
    """
    T, N = calcium_data.shape

    # Compute subject-level statistics from prob_data (expected shape: (N, T))
    per_neuron_mean = None
    per_neuron_var = None
    per_frame_mean = None
    per_frame_var = None
    try:
        per_neuron_mean = prob_data.mean(axis=1)
        per_neuron_var = prob_data.var(axis=1)
        per_frame_mean = prob_data.mean(axis=0)
        per_frame_var = prob_data.var(axis=0)
        global_mean = float(prob_data.mean())
        global_var = float(prob_data.var())
        sparsity_frac = float((prob_data < 1e-3).sum()) / float(prob_data.size)
        active_frac = 1.0 - sparsity_frac
        frac_active_gt_1pct = float((prob_data > 1e-2).mean())
        pn_mean_mean = float(per_neuron_mean.mean())
        pn_mean_median = float(np.median(per_neuron_mean))
        pn_var_mean = float(per_neuron_var.mean())
        pf_mean_mean = float(per_frame_mean.mean())
        pf_var_mean = float(per_frame_var.mean())
    except Exception:
        global_mean = global_var = sparsity_frac = active_frac = frac_active_gt_1pct = 0.0
        pn_mean_mean = pn_mean_median = pn_var_mean = pf_mean_mean = pf_var_mean = 0.0

    print(f"  → Writing PDF to {output_path}")
    with PdfPages(output_path) as pdf:
        # Summary metrics page
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle(f"Subject: {subject_name} — Summary Metrics", fontsize=12)

            # Text panel
            ax0 = axes[0, 0]
            ax0.axis('off')
            lines = [
                f"Neurons (N): {N}",
                f"Timepoints (T): {T}",
                f"Global mean prob: {global_mean:.6f}",
                f"Global var prob: {global_var:.6f}",
                f"Sparsity (<1e-3): {sparsity_frac:.4f}",
                f"Active frac: {active_frac:.4f}",
                f"Frac >1%: {frac_active_gt_1pct:.4f}",
                f"Per-neuron mean (mean/median): {pn_mean_mean:.6f} / {pn_mean_median:.6f}",
                f"Per-neuron var (mean): {pn_var_mean:.6f}",
                f"Per-frame mean (mean): {pf_mean_mean:.6f}",
                f"Per-frame var (mean): {pf_var_mean:.6f}",
            ]
            ax0.text(0.01, 0.98, "\n".join(lines), va='top', ha='left', fontsize=9)

            # Histogram: per-neuron mean
            ax1 = axes[0, 1]
            if per_neuron_mean is not None:
                ax1.hist(per_neuron_mean, bins=50, color='steelblue', alpha=0.8)
                ax1.set_title('Histogram: per-neuron mean prob')
            else:
                ax1.axis('off')

            # Per-frame mean over time
            ax2 = axes[1, 0]
            if per_frame_mean is not None:
                ax2.plot(per_frame_mean, color='darkmagenta', lw=0.8)
                ax2.set_title('Per-frame mean prob over time')
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Mean prob')
            else:
                ax2.axis('off')

            # Histogram: per-neuron variance
            ax3 = axes[1, 1]
            if per_neuron_var is not None:
                ax3.hist(per_neuron_var, bins=50, color='orange', alpha=0.8)
                ax3.set_title('Histogram: per-neuron variance')
            else:
                ax3.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        except Exception:
            pass

        # Select random neurons to visualize
        sel = np.random.choice(N, min(num_neurons, N), replace=False)
        
        for idx in sel:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            
            # Raw calcium trace
            ax[0].plot(calcium_data[:, idx], 'k')
            ax[0].set_title(f'Neuron {idx}: Raw Calcium')
            ax[0].set_ylabel('Fluorescence')
            
            # Spike probabilities
            ax[1].plot(prob_data[idx], 'm')
            ax[1].set_title('Spike Probabilities (CASCADE)')
            ax[1].set_ylabel('Probability')
            ax[1].set_xlabel('Frame')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def process_subject(subject_dir, config):
    """
    Process a single subject through the complete pipeline.
    
    Args:
        subject_dir: Path to subject directory containing raw data
        config: Configuration dictionary
        
    Returns:
        subject_output_dir: Path to subject's output directory
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    
    # Process raw calcium data
    print(f"\nProcessing subject: {subject_name} using CASCADE")
        
    # Create output directory
    os.makedirs(config['data']['output_dir'], exist_ok=True)

    # Check if already processed
    final_file = os.path.join(config['data']['output_dir'], f'{subject_name}.h5')
    pdf_file = os.path.join(config['data']['output_dir'], f'{subject_name}_visualization.pdf')

    if os.path.exists(final_file) and os.path.exists(pdf_file):
        print("  → Already processed; skipping.")
        return final_file

    start_time = time.time()
        
    try:
        # Load raw data
        print("  → Loading raw data...")
        
        # Load cell positions and additional datasets
        mat = loadmat(os.path.join(subject_dir, 'data_full.mat'))
        data0 = mat['data'][0, 0]
        cell_xyz = data0['CellXYZ']
        
        if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
            cell_xyz = cell_xyz[0, 0]

        # Try to load normalized positions if available
        cell_xyz_norm_available = False
        cell_xyz_norm = None
        if 'CellXYZ_norm' in data0.dtype.names:
            try:
                cell_xyz_norm = data0['CellXYZ_norm']
                if isinstance(cell_xyz_norm, np.ndarray) and cell_xyz_norm.dtype == np.object_:
                    cell_xyz_norm = cell_xyz_norm[0, 0]
                # Validate shape
                if isinstance(cell_xyz_norm, np.ndarray) and cell_xyz_norm.shape == cell_xyz.shape:
                    cell_xyz_norm_available = True
                    print("  → Found CellXYZ_norm; will use as base for normalization")
                else:
                    cell_xyz_norm = None
            except Exception:
                cell_xyz_norm = None
        
        # Load additional datasets
        print("  → Loading additional datasets from MATLAB file...")
        
        # Anatomical stack
        anat_stack = data0['anat_stack']
        if isinstance(anat_stack, np.ndarray) and anat_stack.dtype == np.object_:
            anat_stack = anat_stack[0, 0]
        
        # Sampling rate
        fpsec = data0['fpsec']
        if isinstance(fpsec, np.ndarray) and fpsec.dtype == np.object_:
            fpsec = fpsec[0, 0]
        fpsec = float(fpsec.item() if hasattr(fpsec, 'item') else fpsec)
        
        # Stimulus data
        stim_full = data0['stim_full']
        if isinstance(stim_full, np.ndarray) and stim_full.dtype == np.object_:
            stim_full = stim_full[0, 0]
        stim_full = np.squeeze(stim_full)  # Remove singleton dimensions
        
        # Behavioral data
        behavior_full = data0['Behavior_full']
        if isinstance(behavior_full, np.ndarray) and behavior_full.dtype == np.object_:
            behavior_full = behavior_full[0, 0]
        
        # Eye tracking data
        eye_full = data0['Eye_full']
        if isinstance(eye_full, np.ndarray) and eye_full.dtype == np.object_:
            eye_full = eye_full[0, 0]
        
        print(f"  → Loaded anat_stack: {anat_stack.shape}, fpsec: {fpsec} Hz")
        print(f"  → Loaded stim_full: {stim_full.shape}, behavior_full: {behavior_full.shape}, eye_full: {eye_full.shape}")
        
        # Load fluorescence traces
        calcium_dataset = config['data']['calcium_dataset']
        with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
            if calcium_dataset not in f:
                raise ValueError(f"Dataset '{calcium_dataset}' not found in TimeSeries.h5. Available datasets: {list(f.keys())}")
            calcium = f[calcium_dataset][:]
            print(f"  → Using {calcium_dataset} traces")
            
            # Prefer alignment via absIX if available
            have_abs_ix = 'absIX' in f
            if have_abs_ix:
                abs_ix = f['absIX'][:].flatten().astype(int) - 1  # 0-based
                N_abs = abs_ix.shape[0]
                # Ensure calcium is (T, N_abs)
                if calcium.shape[0] == N_abs and calcium.shape[1] != N_abs:
                    print(f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)")
                    calcium = calcium.T
                elif calcium.shape[1] == N_abs:
                    print(f"  → Calcium data is already (T, N) = {calcium.shape}")
                else:
                    print(f"  → Warning: absIX length {N_abs} does not match calcium shape {calcium.shape}; proceeding with calcium N")
                    N_abs = calcium.shape[1]
                    abs_ix = abs_ix[:N_abs]
                # If IX_inval_anat is provided, filter absIX/calcium consistently
                if 'IX_inval_anat' in data0.dtype.names:
                    inval = data0['IX_inval_anat']
                    try:
                        if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                            inval = inval[0, 0]
                        inval = np.array(inval).flatten().astype(int) - 1  # to 0-based
                        keep_mask_cols = ~np.isin(abs_ix, inval)
                        num_dropped = int((~keep_mask_cols).sum())
                        if num_dropped > 0:
                            print(f"  → IX_inval_anat: dropping {num_dropped} neurons from calcium/positions via absIX filter")
                            calcium = calcium[:, keep_mask_cols]
                            abs_ix = abs_ix[keep_mask_cols]
                            N_abs = abs_ix.shape[0]
                    except Exception as e:
                        print(f"  → Warning: IX_inval_anat filtering failed ({e}); continuing without it")
                # Select positions strictly by absIX to preserve index alignment
                positions_base = cell_xyz_norm if cell_xyz_norm_available else cell_xyz
                try:
                    cell_xyz = positions_base[abs_ix, :]
                except Exception as e:
                    print(f"  → Error selecting positions by absIX: {e}; falling back to first N positions")
                    cell_xyz = positions_base[:calcium.shape[1], :]
                # Optional coordinate sanity: drop NaN positions in lockstep with calcium
                if np.isnan(cell_xyz).any():
                    coord_mask = ~np.isnan(cell_xyz).any(axis=1)
                    num_drop = int((~coord_mask).sum())
                    if num_drop > 0:
                        print(f"  → Coordinate sanity: dropping {num_drop} NaN-position neurons")
                        cell_xyz = cell_xyz[coord_mask]
                        calcium = calcium[:, coord_mask]
                print(f"  → Final aligned shapes - calcium: {calcium.shape}, cell_xyz: {cell_xyz.shape}")
                T, N_original = calcium.shape
            else:
                # Check dimensions and transpose if needed using position count
                if calcium.shape[0] == cell_xyz.shape[0]:
                    print(f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)")
                    calcium = calcium.T
                elif calcium.shape[1] == cell_xyz.shape[0]:
                    print(f"  → Calcium data is already (T, N) = {calcium.shape}")
                else:
                    print(f"  → Warning: Calcium shape {calcium.shape} doesn't match cell count {cell_xyz.shape[0]}")
                    # Fallback: truncate to shared min count to preserve index ordering
                    T_tmp, N_calcium = calcium.shape
                    N_positions = cell_xyz.shape[0]
                    N_shared = min(N_calcium, N_positions)
                    calcium = calcium[:, :N_shared]
                    cell_xyz = cell_xyz[:N_shared, :]
                    print(f"  → Truncated to shared N={N_shared} to maintain index alignment")
                T, N_original = calcium.shape
            
            # If absIX was used, skip additional masking to preserve index alignment
            if have_abs_ix:
                N = calcium.shape[1]
                print(f"  → Retained {N} neurons (absIX-based alignment)")
            else:
                # Fallback simple coordinate NaN filter only if needed
                coordinate_mask = ~np.isnan(cell_xyz).any(axis=1)
                if not np.all(coordinate_mask):
                    calcium = calcium[:, coordinate_mask]
                    cell_xyz = cell_xyz[coordinate_mask]
                T, N = calcium.shape
                print(f"  → Retained {N} neurons after coordinate sanity check")

            # Prepare normalized positions (use CellXYZ_norm if available, else CellXYZ)
            # Always normalize to [0, 1] per axis after masking/selection
            # Select base positions for normalization preserving index alignment
            if cell_xyz_norm_available and have_abs_ix:
                base_positions = cell_xyz  # already selected via absIX and set into cell_xyz
            elif cell_xyz_norm_available and not have_abs_ix:
                # cell_xyz corresponds to filtered set; map via retained indices is already applied
                base_positions = cell_xyz_norm[:cell_xyz.shape[0]] if cell_xyz_norm.shape[0] != cell_xyz.shape[0] else cell_xyz_norm
            else:
                base_positions = cell_xyz
            
            # Apply test run neuron selection if specified
            test_run_neurons = config['data'].get('test_run_neurons')
            if test_run_neurons is not None and test_run_neurons < N:
                print(f"  → Test run: selecting {test_run_neurons} random neurons out of {N}")
                np.random.seed(config['processing']['seed'])
                selected_indices = np.random.choice(N, test_run_neurons, replace=False)
                selected_indices = np.sort(selected_indices)
                
                calcium = calcium[:, selected_indices]
                cell_xyz = cell_xyz[selected_indices, :]
                base_positions = base_positions[selected_indices, :]
                
                T, N = calcium.shape
                print(f"  → Reduced to {N} neurons for testing")

            # Compute normalization stats and normalized positions in [0, 1]
            # Replace NaNs before normalization
            if np.isnan(base_positions).any():
                base_positions = np.nan_to_num(base_positions)

            pos_min = base_positions.min(axis=0)
            pos_max = base_positions.max(axis=0)
            pos_range = pos_max - pos_min
            pos_range[pos_range == 0] = 1.0
            norm_positions = (base_positions - pos_min) / pos_range
            # Safety: clip to [0, 1]
            norm_positions = np.clip(norm_positions, 0.0, 1.0)
            
        # Keep a copy of calcium for visualization
        calcium_for_viz = calcium[:, :min(config['processing']['num_neurons_viz'] * 2, N)].copy()
            
        # Interpolate calcium traces and temporal data if needed
        orig_rate = config['processing']['original_sampling_rate']
        target_rate = config['processing']['target_sampling_rate']
        return_to_original = config['processing'].get('return_to_original_rate', False)
        
        # Store original dimensions for potential downsampling later
        original_T = T
        upsampled_T = T
        
        if orig_rate is not None and target_rate is not None:
            print(f"  → Interpolating calcium traces from {orig_rate}Hz to {target_rate}Hz...")
                
            # Create time points
            original_time = np.arange(T) / orig_rate
            upsampled_T = int(T * target_rate / orig_rate)
            new_time = np.arange(upsampled_T) / target_rate
            
            # Interpolate each neuron's trace using PCHIP
            from scipy.interpolate import PchipInterpolator
            calcium_interpolated = np.zeros((upsampled_T, N), dtype=calcium.dtype)
            
            for n in range(N):
                interp_func = PchipInterpolator(original_time, calcium[:, n], extrapolate=True)
                calcium_interpolated[:, n] = interp_func(new_time)
            
            calcium = calcium_interpolated
            T = upsampled_T
            print(f"  → Interpolated calcium to {T} timepoints at {target_rate}Hz")
            
            # Only interpolate non-neural data if we're NOT returning to original rate
            if not return_to_original:
                # Interpolate temporal datasets to match using hold interpolation
                print(f"  → Interpolating temporal datasets with hold interpolation (zero-order hold)...")
                from scipy.interpolate import interp1d
                
                # Hold interpolation for stimulus data (1D) - preserves discrete values
                if stim_full.shape[0] == len(original_time):
                    stim_hold_interp = interp1d(original_time, stim_full.astype(float), 
                                              kind='previous', bounds_error=False, 
                                              fill_value=(stim_full[0], stim_full[-1]))
                    stim_full = stim_hold_interp(new_time).astype(stim_full.dtype)
                
                # Hold interpolation for behavioral data (2D: behaviors x time)
                if behavior_full.shape[1] == len(original_time):
                    behavior_interp = np.zeros((behavior_full.shape[0], upsampled_T), dtype=behavior_full.dtype)
                    for b in range(behavior_full.shape[0]):
                        behav_hold_interp = interp1d(original_time, behavior_full[b, :], 
                                                   kind='previous', bounds_error=False,
                                                   fill_value=(behavior_full[b, 0], behavior_full[b, -1]))
                        behavior_interp[b, :] = behav_hold_interp(new_time)
                    behavior_full = behavior_interp
                
                # Hold interpolation for eye tracking data (2D: eye dimensions x time)
                if eye_full.shape[1] == len(original_time):
                    eye_interp = np.zeros((eye_full.shape[0], upsampled_T), dtype=eye_full.dtype)
                    for e in range(eye_full.shape[0]):
                        eye_hold_interp = interp1d(original_time, eye_full[e, :], 
                                                 kind='previous', bounds_error=False,
                                                 fill_value=(eye_full[e, 0], eye_full[e, -1]))
                        eye_interp[e, :] = eye_hold_interp(new_time)
                    eye_full = eye_interp
                
                print(f"  → Interpolated temporal data: stim_full {stim_full.shape}, behavior_full {behavior_full.shape}, eye_full {eye_full.shape}")
            else:
                print(f"  → Keeping temporal data at original rate (will downsample neural data later)")
        
        # Use actual sampling rate for further processing
        effective_sampling_rate = target_rate or orig_rate or fpsec

        # Apply baseline correction based on configuration
        processed_calcium = compute_baseline_correction(
            calcium,
            config['data']['window_length'],
            config['data']['baseline_percentile'],
            effective_sampling_rate,
            config['data']['is_raw'],
            config['data']['apply_baseline_subtraction']
        )
        
        # Clean up calcium to free memory
        del calcium
        gc.collect()
        
        # Run CASCADE inference
        prob_data = run_cascade_inference(
            processed_calcium,
            config['processing']['batch_size'],
            config['cascade']['model_type'],
            effective_sampling_rate
        )

        # Clean up after CASCADE
        del processed_calcium
        gc.collect()
        
        # Optionally downsample neural data back to original rate
        if return_to_original and orig_rate is not None and target_rate is not None and orig_rate != target_rate:
            print(f"  → Downsampling neural data from {target_rate}Hz back to {orig_rate}Hz with anti-aliasing...")
            from scipy.signal import resample_poly
            from fractions import Fraction
            
            # Calculate the rational resampling factors
            # We upsampled by target_rate/orig_rate, so we downsample by orig_rate/target_rate
            rate_ratio = Fraction(orig_rate).limit_denominator() / Fraction(target_rate).limit_denominator()
            up_factor = rate_ratio.numerator
            down_factor = rate_ratio.denominator
            
            print(f"  → Downsampling with factors: up={up_factor}, down={down_factor}")
            
            # Transpose prob_data to (T, N) for processing
            prob_data_T = prob_data.T  # Shape: (T, N)
            
            # Downsample each neuron's probability trace
            downsampled_prob_data = np.zeros((original_T, N), dtype=prob_data.dtype)
            
            for n in tqdm(range(N), desc="Downsampling neurons"):
                # Use resample_poly with proper anti-aliasing
                # The function automatically applies appropriate low-pass filtering
                downsampled_prob_data[:, n] = resample_poly(
                    prob_data_T[:, n], 
                    up=up_factor, 
                    down=down_factor,
                    axis=0
                )
            
            # Update prob_data back to (N, T) format and dimensions
            prob_data = downsampled_prob_data.T  # Back to (N, T)
            T = original_T
            effective_sampling_rate = orig_rate
            
            print(f"  → Downsampled to {T} timepoints at {effective_sampling_rate}Hz")
            
            # Clean up
            del prob_data_T, downsampled_prob_data
            gc.collect()
        
        # Prepare data for saving - transpose probabilities to (T, N) format and convert to configured dtype
        print("  → Preparing probability data for saving...")
        output_dtype = getattr(np, config['output']['dtype'])

        # Handle NaN values in probability data
        prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Transpose to (T, N) and convert to specified dtype
        prob_data_transposed = prob_data.T.astype(output_dtype)  # (T, N)

        # Use normalized positions computed above
        cell_positions = norm_positions.astype(output_dtype)  # (N, 3) in [0, 1]

        # Save final data
        print(f"  → Saving final data to {final_file}")
        with h5py.File(final_file, 'w') as f:
            # Save main datasets
            f.create_dataset('spike_probabilities',
                            data=prob_data_transposed,
                            chunks=(min(1000, T), min(100, N)),
                            compression='gzip',
                            compression_opts=1)

            f.create_dataset('cell_positions',
                             data=cell_positions,
                             compression='gzip',
                             compression_opts=1)

            f.create_dataset('timepoint_indices',
                             data=np.arange(T, dtype=np.int32))

            # Save metadata as datasets
            f.create_dataset('num_timepoints', data=T)
            f.create_dataset('num_neurons', data=N)

            # Sampling rate information (always included)
            f.create_dataset('original_sampling_rate_hz', data=fpsec)

            # Save additional datasets if requested
            if config['output'].get('include_additional_data', True):
                print(f"  → Saving additional datasets...")

                # Anatomical reference stack
                f.create_dataset('anat_stack',
                                 data=anat_stack,
                                 compression='gzip',
                                 compression_opts=1)

                # Temporal data (stimulus, behavior, eye tracking)
                f.create_dataset('stimulus_full',
                                 data=stim_full,
                                 compression='gzip',
                                 compression_opts=1)

                f.create_dataset('behavior_full',
                                 data=behavior_full,
                                 compression='gzip',
                                 compression_opts=1)

                f.create_dataset('eye_full',
                                 data=eye_full,
                                 compression='gzip',
                                 compression_opts=1)
            else:
                print(f"  → Skipping additional datasets per configuration")

            # Save attributes
            f.attrs['subject'] = subject_name
            f.attrs['data_source'] = 'raw_calcium'
            f.attrs['spike_dtype'] = config['output']['dtype']
            f.attrs['position_dtype'] = config['output']['dtype']
            f.attrs['positions_normalized'] = True
            f.attrs['positions_source'] = 'CellXYZ_norm' if cell_xyz_norm_available else 'CellXYZ_normalized_runtime'
            f.attrs['positions_min'] = pos_min.astype(np.float32)
            f.attrs['positions_max'] = pos_max.astype(np.float32)
            f.attrs['cascade_model'] = config['cascade']['model_type']
            f.attrs['calcium_dataset'] = config['data']['calcium_dataset']
            f.attrs['is_raw'] = config['data']['is_raw']
            f.attrs['apply_baseline_subtraction'] = config['data']['apply_baseline_subtraction']
            f.attrs['window_length'] = config['data']['window_length']
            f.attrs['baseline_percentile'] = config['data']['baseline_percentile']
            f.attrs['original_sampling_rate'] = config['processing'].get('original_sampling_rate', fpsec)
            f.attrs['target_sampling_rate'] = config['processing'].get('target_sampling_rate', fpsec)
            f.attrs['effective_sampling_rate'] = effective_sampling_rate
            f.attrs['matlab_fpsec'] = fpsec
            f.attrs['includes_additional_data'] = config['output'].get('include_additional_data', True)
            f.attrs['return_to_original_rate'] = return_to_original
            f.attrs['final_sampling_rate'] = effective_sampling_rate  # The actual final rate of all data
        
        # Create visualization PDF
        print("  → Creating visualization PDF...")
        create_visualization_pdf(calcium_for_viz, prob_data, subject_name, pdf_file, 
                               config['processing']['num_neurons_viz'])
            
        # Clean up memory
        del calcium_for_viz, prob_data, prob_data_transposed, cell_positions
        del anat_stack, stim_full, behavior_full, eye_full
        gc.collect()
        
        processing_time = time.time() - start_time
        print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")
        
        return final_file
            
    except Exception as e:
        print(f"  → Error processing {subject_name}: {e}")
        raise

def analyze_raw_data(config):
    """
    Analyze and output information about raw data without processing.
    
    Args:
        config: Configuration dictionary
    """
    print("="*80)
    print("RAW DATA ANALYSIS")
    print("="*80)
    
    # Find subjects
    subjects = sorted([
        os.path.join(config['data']['input_dir'], d)
        for d in os.listdir(config['data']['input_dir'])
        if d.startswith('subject_') and d not in config['data'].get('skip_subjects', []) and
        os.path.isdir(os.path.join(config['data']['input_dir'], d))
    ])
    
    if not subjects:
        print(f"No subjects found in {config['data']['input_dir']}")
        return
    
    print(f"Found {len(subjects)} subjects in {config['data']['input_dir']}")
    print(f"Skipping subjects: {config['data'].get('skip_subjects', [])}")
    print()
    
    # Analyze each subject
    for subject_dir in subjects:
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        print(f"Subject: {subject_name}")
        print("-" * 40)
        
        try:
            # Load cell positions
            mat_file = os.path.join(subject_dir, 'data_full.mat')
            if not os.path.exists(mat_file):
                print(f"  ERROR: data_full.mat not found")
                continue
                
            mat = loadmat(mat_file)
            data0 = mat['data'][0, 0]
            cell_xyz = data0['CellXYZ']
            
            if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
                cell_xyz = cell_xyz[0, 0]
            
            print(f"  Cell positions shape: {cell_xyz.shape}")
            
            # Analyze cell positions
            valid_coords = ~np.isnan(cell_xyz).any(axis=1)
            num_valid = np.sum(valid_coords)
            num_invalid = np.sum(~valid_coords)
            
            print(f"  Valid coordinate neurons: {num_valid}")
            print(f"  Invalid coordinate neurons: {num_invalid}")
            
            if num_valid > 0:
                valid_positions = cell_xyz[valid_coords]
                
                # Spatial bounds
                min_coords = valid_positions.min(axis=0)
                max_coords = valid_positions.max(axis=0)
                ranges = max_coords - min_coords
                
                print(f"  Spatial bounds:")
                print(f"    X: {min_coords[0]:.2f} to {max_coords[0]:.2f} (range: {ranges[0]:.2f})")
                print(f"    Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} (range: {ranges[1]:.2f})")
                print(f"    Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} (range: {ranges[2]:.2f})")
                
                # Spatial statistics
                mean_coords = valid_positions.mean(axis=0)
                std_coords = valid_positions.std(axis=0)
                
                print(f"  Spatial statistics:")
                print(f"    Mean: X={mean_coords[0]:.2f}, Y={mean_coords[1]:.2f}, Z={mean_coords[2]:.2f}")
                print(f"    Std:  X={std_coords[0]:.2f}, Y={std_coords[1]:.2f}, Z={std_coords[2]:.2f}")
            
            # Check for invalid anatomical indices
            if 'IX_inval_anat' in data0.dtype.names:
                inval = data0['IX_inval_anat']
                if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                    inval = inval[0, 0].flatten()
                inval_indices = np.array(inval, int) - 1  # Convert to 0-based
                print(f"  Invalid anatomical indices: {len(inval_indices)} neurons")
            else:
                print(f"  Invalid anatomical indices: None specified")
            
            # Load calcium data
            timeseries_file = os.path.join(subject_dir, 'TimeSeries.h5')
            if not os.path.exists(timeseries_file):
                print(f"  ERROR: TimeSeries.h5 not found")
                continue
                
            calcium_dataset = config['data']['calcium_dataset']
            with h5py.File(timeseries_file, 'r') as f:
                print(f"  Available datasets in TimeSeries.h5: {list(f.keys())}")
                
                if calcium_dataset not in f:
                    print(f"  ERROR: Dataset '{calcium_dataset}' not found")
                    continue
                    
                calcium_shape = f[calcium_dataset].shape
                calcium_dtype = f[calcium_dataset].dtype
                
                print(f"  Calcium dataset '{calcium_dataset}':")
                print(f"    Shape: {calcium_shape}")
                print(f"    Data type: {calcium_dtype}")
                
                # Determine if transposition is needed
                if calcium_shape[0] == cell_xyz.shape[0]:
                    print(f"    Format: (N, T) - will be transposed to (T, N)")
                    T, N = calcium_shape[1], calcium_shape[0]
                elif calcium_shape[1] == cell_xyz.shape[0]:
                    print(f"    Format: (T, N) - correct format")
                    T, N = calcium_shape[0], calcium_shape[1]
                else:
                    print(f"    WARNING: Shape mismatch with cell positions ({cell_xyz.shape[0]} cells)")
                    T, N = calcium_shape[0], calcium_shape[1]
                
                print(f"    Interpreted as: {T} timepoints, {N} neurons")
                
                # Sampling rate information
                orig_rate = config['processing'].get('original_sampling_rate')
                target_rate = config['processing'].get('target_sampling_rate')
                
                if orig_rate:
                    duration = T / orig_rate
                    print(f"    Duration: {duration:.2f} seconds at {orig_rate} Hz")
                    
                    if target_rate and target_rate != orig_rate:
                        new_T = int(T * target_rate / orig_rate)
                        new_duration = new_T / target_rate
                        print(f"    After resampling: {new_T} timepoints at {target_rate} Hz ({new_duration:.2f} seconds)")
                
                # Memory requirements estimation
                memory_gb = (T * N * 4) / (1024**3)  # Assuming float32
                print(f"    Memory requirement: ~{memory_gb:.2f} GB (float32)")
                
                # Sample some data statistics
                sample_size = min(1000, T)
                sample_indices = np.random.choice(T, sample_size, replace=False)
                sample_indices = np.sort(sample_indices)  # HDF5 requires sorted indices
                sample_data = f[calcium_dataset][sample_indices, :min(1000, N)]
                
                print(f"    Sample statistics (first {min(1000, N)} neurons, {sample_size} timepoints):")
                print(f"      Min: {sample_data.min():.4f}")
                print(f"      Max: {sample_data.max():.4f}")
                print(f"      Mean: {sample_data.mean():.4f}")
                print(f"      Std: {sample_data.std():.4f}")
                
                # Check for NaN/inf values
                num_nan = np.sum(np.isnan(sample_data))
                num_inf = np.sum(np.isinf(sample_data))
                if num_nan > 0 or num_inf > 0:
                    print(f"      WARNING: Found {num_nan} NaN and {num_inf} inf values in sample")
            
            # Additional MATLAB data analysis  
            print(f"  Additional MATLAB datasets:")
            
            # Sampling rate from MATLAB
            fpsec_val = data0['fpsec']
            if isinstance(fpsec_val, np.ndarray) and fpsec_val.dtype == np.object_:
                fpsec_val = fpsec_val[0, 0]
            fpsec_val = float(fpsec_val.item() if hasattr(fpsec_val, 'item') else fpsec_val)
            print(f"    MATLAB sampling rate (fpsec): {fpsec_val} Hz")
            
            # Anatomical stack
            anat_stack = data0['anat_stack']
            if isinstance(anat_stack, np.ndarray) and anat_stack.dtype == np.object_:
                anat_stack = anat_stack[0, 0]
            print(f"    Anatomical stack: {anat_stack.shape} ({anat_stack.dtype})")
            anat_memory_gb = (np.prod(anat_stack.shape) * anat_stack.itemsize) / (1024**3)
            print(f"      Memory requirement: ~{anat_memory_gb:.3f} GB")
            
            # Stimulus data
            stim_full = data0['stim_full']
            if isinstance(stim_full, np.ndarray) and stim_full.dtype == np.object_:
                stim_full = stim_full[0, 0]
            stim_full = np.squeeze(stim_full)
            print(f"    Stimulus data: {stim_full.shape} ({stim_full.dtype})")
            print(f"      Values: {stim_full.min()} to {stim_full.max()}")
            
            # Behavioral data
            behavior_full = data0['Behavior_full']
            if isinstance(behavior_full, np.ndarray) and behavior_full.dtype == np.object_:
                behavior_full = behavior_full[0, 0]
            print(f"    Behavioral data: {behavior_full.shape} ({behavior_full.dtype})")
            print(f"      {behavior_full.shape[0]} behavioral variables over {behavior_full.shape[1]} timepoints")
            
            # Eye tracking data
            eye_full = data0['Eye_full']
            if isinstance(eye_full, np.ndarray) and eye_full.dtype == np.object_:
                eye_full = eye_full[0, 0]
            print(f"    Eye tracking data: {eye_full.shape} ({eye_full.dtype})")
            print(f"      {eye_full.shape[0]} eye dimensions over {eye_full.shape[1]} timepoints")
            
            # Check temporal alignment
            if config['output'].get('include_additional_data', True):
                temporal_datasets = [
                    ("stimulus", stim_full.shape[0] if stim_full.ndim == 1 else stim_full.shape[1]),
                    ("behavior", behavior_full.shape[1]),
                    ("eye", eye_full.shape[1])
                ]
                print(f"    Temporal alignment check:")
                for name, length in temporal_datasets:
                    print(f"      {name}: {length} timepoints ({'✓' if length == T else '⚠'} vs calcium {T})")
            
            # Output format preview
            output_dtype = config['output']['dtype']
            
            print(f"  Output format settings:")
            print(f"    Data type: {output_dtype}")
            
            if num_valid > 0:
                # Estimate spike probability memory requirements - (T, N) format
                prob_memory_gb = (T * num_valid * (2 if output_dtype == 'float16' else 4)) / (1024**3)
                print(f"    Spike probabilities memory requirement: ~{prob_memory_gb:.2f} GB")
                
                # Estimate cell positions memory requirements - (N, 3) format
                pos_memory_gb = (num_valid * 3 * (2 if output_dtype == 'float16' else 4)) / (1024**3)
                print(f"    Cell positions memory requirement: ~{pos_memory_gb:.2f} GB")
                
                total_memory_gb = prob_memory_gb + pos_memory_gb
                print(f"    Total output memory requirement: ~{total_memory_gb:.2f} GB")
                
                print(f"    Output format: Spike probabilities (T={T}, N={num_valid}), Positions (N={num_valid}, 3)")
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Spike Probability Processing Pipeline'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--create-default-config', type=str, default=None,
                        help='Create default configuration file at specified path and exit')
    parser.add_argument('--raw_data_info', action='store_true',
                        help='Analyze and display raw data information without processing')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_default_config:
        save_default_config(args.create_default_config)
        return
    
    # Validate that config is provided when not creating default config
    if args.config is None:
        print("Error: --config is required when not using --create-default-config")
        print("Use --create-default-config to create a default configuration file")
        return
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Use --create-default-config to create a default configuration file")
        return
    
    config = load_config(args.config)
    
    # Handle raw data info request
    if args.raw_data_info:
        analyze_raw_data(config)
        return
    
    # Validate CASCADE availability
    if not CASCADE_AVAILABLE:
        print("Error: CASCADE not available. Please install neuralib.")
        return
    
    # Validate required parameters
    if not config['processing'].get('original_sampling_rate') or not config['processing'].get('target_sampling_rate'):
        print("Error: original_sampling_rate and target_sampling_rate are required for raw data processing")
        return
    
    # Create output directory
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    
    # Find subjects in raw data directory
    subjects = sorted([
    os.path.join(config['data']['input_dir'], d)
    for d in os.listdir(config['data']['input_dir'])
    if d.startswith('subject_') and d not in config['data'].get('skip_subjects', []) and
    os.path.isdir(os.path.join(config['data']['input_dir'], d))
    ])
    
    if not subjects:
        print(f"No subjects found in {config['data']['input_dir']}")
        return
    
    print(f"Found {len(subjects)} subjects; skipping {config['data']['skip_subjects']}")
    
    if config['data'].get('test_run_neurons'):
        print(f"Test mode: Using only {config['data']['test_run_neurons']} randomly selected neurons per subject")
    
    # Process subjects
    processed_subjects = []
    
    # Process subjects sequentially to avoid memory issues with CASCADE
    print("Processing subjects sequentially")
    for subject_dir in subjects:
        try:
            result = process_subject(subject_dir, config)
            if result is not None:
                processed_subjects.append(result)
                
        # Force garbage collection between subjects
                gc.collect()
                print_memory_stats("After processing subject")
                
        except Exception as e:
            print(f"Error processing {subject_dir}: {e}")
        gc.collect()

    print(f"\nSpike probability processing complete!")
    print(f"Processed {len(processed_subjects)} subjects")
    if config['data'].get('test_run_neurons'):
        print(f"Test mode: Used only {config['data']['test_run_neurons']} randomly selected neurons per subject")
    print(f"Output saved to: {config['data']['output_dir']}")
    print(f"Output format: Spike probabilities (T, N) and cell positions (N, 3)")
    print(f"Data type: {config['output']['dtype']}")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 