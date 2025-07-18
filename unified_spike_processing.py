#!/usr/bin/env python3
"""
3D Volumetric Spike Processing Pipeline

This script processes calcium traces using CASCADE spike detection and creates
3D volumetric representations of neural activity. Each timepoint becomes a 
single 3D volume with continuous probability values.

Features:
- CASCADE-only spike detection (no OASIS or binary spikes)
- 3D volumetric representation of neural activity
- YAML configuration support
- Continuous probability values only
- No augmentations or grid transformations
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
        },
        'cascade': {
            'model_type': 'Global_EXC_2Hz_smoothing500ms',
        },
        'volumization': {
            'volume_shape': [64, 64, 32],  # [x, y, z] dimensions
            'dtype': 'float16',
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

def run_cascade_inference(calcium_data, batch_size=5000, model_type='Global_EXC_2Hz_smoothing500ms'):
    """
    Run CASCADE inference on calcium data to get spike probabilities.
    
    Args:
        calcium_data: (T, N) array of calcium data (raw or baseline-subtracted)
        batch_size: Batch size for CASCADE processing
        model_type: CASCADE model to use
        
    Returns:
        prob_data: (N, T) array of spike probabilities
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

def create_3d_volumes(prob_data, cell_positions, volume_shape, dtype='float16'):
    """
    Convert spike probabilities to 3D volumetric format.
    
    Args:
        prob_data: (N, T) array of spike probabilities
        cell_positions: (N, 3) array of cell positions
        volume_shape: [x, y, z] dimensions of output volume
        dtype: Data type for output volumes
        
    Returns:
        volumes: (T, x, y, z) array of volumetric data
        metadata: Dictionary containing dataset metadata
    """
    N, T = prob_data.shape
    x_size, y_size, z_size = volume_shape
    
    # Handle NaN values in cell positions
    if np.isnan(cell_positions).any():
        print("Warning: NaN values found in cell positions, replacing with 0")
        cell_positions = np.nan_to_num(cell_positions)
    
    # Handle NaN values in probability data
    if np.isnan(prob_data).any():
        print("Warning: NaN values found in probability data, replacing with 0")
        prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Creating 3D volumes with shape {volume_shape} from {N} cells and {T} timepoints")
    
    # Normalize cell positions to [0, 1]
    pos_min = cell_positions.min(axis=0)
    pos_max = cell_positions.max(axis=0)
    pos_range = pos_max - pos_min
    
    # Handle case where range is 0 (all cells at same position)
    pos_range[pos_range == 0] = 1
    
    normalized_positions = (cell_positions - pos_min) / pos_range
    
    # Convert to volume indices
    volume_x = np.floor(normalized_positions[:, 0] * (x_size - 1)).astype(np.int32)
    volume_y = np.floor(normalized_positions[:, 1] * (y_size - 1)).astype(np.int32)
    volume_z = np.floor(normalized_positions[:, 2] * (z_size - 1)).astype(np.int32)
    
    # Create volumes
    volume_dtype = getattr(np, dtype)
    volumes = np.zeros((T, x_size, y_size, z_size), dtype=volume_dtype)
    
    print("Converting probabilities to 3D volumes...")
    for t in tqdm(range(T), desc="Processing timepoints"):
        # Get probabilities for this timepoint
        probs_t = prob_data[:, t]
        
        # Find cells with non-zero probabilities
        active_cells = np.abs(probs_t) > 1e-6
            
        if np.any(active_cells):
            active_x = volume_x[active_cells]
            active_y = volume_y[active_cells]
            active_z = volume_z[active_cells]
            active_probs = probs_t[active_cells]
                
            # Set active cells in the volume
            # If multiple cells map to same volume element, sum their probabilities
            for i in range(len(active_x)):
                volumes[t, active_x[i], active_y[i], active_z[i]] += active_probs[i]

    # Create metadata
    metadata = {
        'num_timepoints': T,
        'volume_shape': volume_shape,
        'dtype': dtype
    }
    
    return volumes, metadata

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
    
    print(f"  → Writing PDF to {output_path}")
    with PdfPages(output_path) as pdf:
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
        
        # Load cell positions
        mat = loadmat(os.path.join(subject_dir, 'data_full.mat'))
        data0 = mat['data'][0, 0]
        cell_xyz = data0['CellXYZ']
        
        if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
            cell_xyz = cell_xyz[0, 0]
        
        # Load fluorescence traces
        calcium_dataset = config['data']['calcium_dataset']
        with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
            if calcium_dataset not in f:
                raise ValueError(f"Dataset '{calcium_dataset}' not found in TimeSeries.h5. Available datasets: {list(f.keys())}")
            calcium = f[calcium_dataset][:]
            print(f"  → Using {calcium_dataset} traces")
            
            # Check dimensions and transpose if needed
            if calcium.shape[0] == cell_xyz.shape[0]:
                print(f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)")
                calcium = calcium.T
            elif calcium.shape[1] == cell_xyz.shape[0]:
                print(f"  → Calcium data is already (T, N) = {calcium.shape}")
            else:
                print(f"  → Warning: Calcium shape {calcium.shape} doesn't match cell count {cell_xyz.shape[0]}")
                # Handle size mismatch by identifying neurons with valid coordinates
                T, N_calcium = calcium.shape
                N_positions = cell_xyz.shape[0]
                
                # Find neurons with valid (non-NaN) coordinates
                valid_coord_mask = ~np.isnan(cell_xyz).any(axis=1)
                valid_coord_indices = np.where(valid_coord_mask)[0]
                
                print(f"  → Found {len(valid_coord_indices)} neurons with valid coordinates out of {N_positions}")
                
                # Determine which neurons to keep based on the overlap
                if N_calcium <= N_positions:
                    # More position data than calcium data
                    # Keep only the first N_calcium neurons that have valid coordinates
                    valid_calcium_coords = valid_coord_indices[valid_coord_indices < N_calcium]
                    print(f"  → Keeping {len(valid_calcium_coords)} neurons that have both calcium traces and valid coordinates")
                    
                    # Filter both arrays to match
                    calcium = calcium[:, valid_calcium_coords]
                    cell_xyz = cell_xyz[valid_calcium_coords, :]
                    
                else:
                    # More calcium data than position data
                    # Keep only neurons up to N_positions that have valid coordinates
                    valid_calcium_coords = valid_coord_indices
                    print(f"  → Keeping {len(valid_calcium_coords)} neurons that have both calcium traces and valid coordinates")
                    
                    # Filter both arrays to match
                    calcium = calcium[:, valid_calcium_coords]
                    cell_xyz = cell_xyz[valid_calcium_coords, :]
                
                print(f"  → Final aligned shapes - calcium: {calcium.shape}, cell_xyz: {cell_xyz.shape}")
                
            T, N_original = calcium.shape
            
            # Handle invalid anatomical indices
            anatomical_mask = np.ones(N_original, bool)
            if 'IX_inval_anat' in data0.dtype.names:
                inval = data0['IX_inval_anat']
                if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                    inval = inval[0, 0].flatten()
                inval_indices = np.array(inval, int) - 1  # Convert to 0-based indexing
                valid_inval_indices = inval_indices[inval_indices < N_original]
                if len(valid_inval_indices) > 0:
                    anatomical_mask[valid_inval_indices] = False
            
            # Filter out neurons with invalid coordinates (additional check)
            coordinate_mask = ~np.isnan(cell_xyz).any(axis=1)
            
            # Combine both masks
            valid_mask = anatomical_mask & coordinate_mask
            
            # Count filtered neurons
            n_invalid_anatomical = np.sum(~anatomical_mask)
            n_invalid_coordinates = np.sum(~coordinate_mask)
            n_total_filtered = np.sum(~valid_mask)
            
            if n_invalid_anatomical > 0:
                print(f"  → Filtering out {n_invalid_anatomical} neurons with invalid anatomical indices")
            if n_invalid_coordinates > 0:
                print(f"  → Filtering out {n_invalid_coordinates} neurons with invalid coordinates")
            if n_total_filtered > 0:
                print(f"  → Total neurons filtered: {n_total_filtered} out of {N_original} ({n_total_filtered/N_original*100:.1f}%)")
        
            # Apply masks
            cell_xyz = cell_xyz[valid_mask]
            calcium = calcium[:, valid_mask]
            
            T, N = calcium.shape
            print(f"  → Retained {N} neurons with valid coordinates")
            
            # Apply test run neuron selection if specified
            test_run_neurons = config['data'].get('test_run_neurons')
            if test_run_neurons is not None and test_run_neurons < N:
                print(f"  → Test run: selecting {test_run_neurons} random neurons out of {N}")
                np.random.seed(config['processing']['seed'])
                selected_indices = np.random.choice(N, test_run_neurons, replace=False)
                selected_indices = np.sort(selected_indices)
                
                calcium = calcium[:, selected_indices]
                cell_xyz = cell_xyz[selected_indices, :]
                
                T, N = calcium.shape
                print(f"  → Reduced to {N} neurons for testing")
            
        # Keep a copy of calcium for visualization
        calcium_for_viz = calcium[:, :min(config['processing']['num_neurons_viz'] * 2, N)].copy()
            
        # Interpolate calcium traces if needed
        orig_rate = config['processing']['original_sampling_rate']
        target_rate = config['processing']['target_sampling_rate']
        
        if orig_rate is not None and target_rate is not None:
            print(f"  → Interpolating calcium traces from {orig_rate}Hz to {target_rate}Hz...")
                
            # Create time points
            original_time = np.arange(T) / orig_rate
            new_T = int(T * target_rate / orig_rate)
            new_time = np.arange(new_T) / target_rate
            
            # Interpolate each neuron's trace
            from scipy.interpolate import interp1d
            calcium_interpolated = np.zeros((new_T, N), dtype=calcium.dtype)
            
            for n in range(N):
                interp_func = interp1d(original_time, calcium[:, n], kind='linear', 
                                        bounds_error=False, fill_value='extrapolate')
                calcium_interpolated[:, n] = interp_func(new_time)
            
            calcium = calcium_interpolated
            T = new_T
            print(f"  → Interpolated to {T} timepoints at {target_rate}Hz")

        # Apply baseline correction based on configuration
        processed_calcium = compute_baseline_correction(
            calcium,
            config['data']['window_length'],
            config['data']['baseline_percentile'],
            target_rate or orig_rate or 2.0,
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
            config['cascade']['model_type']
                )
                
                # Clean up after CASCADE
        del processed_calcium
        gc.collect()
        
        # Convert to 3D volumes
        print("  → Converting to 3D volumetric format...")
        volumes, metadata = create_3d_volumes(
            prob_data, 
            cell_xyz, 
            config['volumization']['volume_shape'],
            config['volumization']['dtype']
        )
            
        # Save final data
        print(f"  → Saving final data to {final_file}")
        with h5py.File(final_file, 'w') as f:
                # Save main datasets
            f.create_dataset('volumes', 
                            data=volumes,
                            chunks=(1, *config['volumization']['volume_shape']),
                                compression='gzip',
                                compression_opts=1)
                
            f.create_dataset('timepoint_indices', 
                            data=np.arange(T, dtype=np.int32))
                
            # Save metadata as datasets
            f.create_dataset('num_timepoints', data=metadata['num_timepoints'])
            f.create_dataset('volume_shape', data=metadata['volume_shape'])
                
            # Save attributes
            f.attrs['subject'] = subject_name
            f.attrs['data_source'] = 'raw_calcium'
            f.attrs['dtype'] = metadata['dtype']
            f.attrs['cascade_model'] = config['cascade']['model_type']
            f.attrs['calcium_dataset'] = config['data']['calcium_dataset']
            f.attrs['is_raw'] = config['data']['is_raw']
            f.attrs['apply_baseline_subtraction'] = config['data']['apply_baseline_subtraction']
            f.attrs['window_length'] = config['data']['window_length']
            f.attrs['baseline_percentile'] = config['data']['baseline_percentile']
        
        # Create visualization PDF
        print("  → Creating visualization PDF...")
        create_visualization_pdf(calcium_for_viz, prob_data, subject_name, pdf_file, 
                               config['processing']['num_neurons_viz'])
            
            # Clean up memory
        del calcium_for_viz, prob_data, volumes
        gc.collect()
        
        processing_time = time.time() - start_time
        print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")
        
        return final_file
            
    except Exception as e:
        print(f"  → Error processing {subject_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='3D Volumetric Spike Processing Pipeline'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file')
    parser.add_argument('--create-default-config', type=str, default=None,
                        help='Create default configuration file at specified path and exit')
    
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

    print(f"\n3D Volumetric processing complete!")
    print(f"Processed {len(processed_subjects)} subjects")
    if config['data'].get('test_run_neurons'):
        print(f"Test mode: Used only {config['data']['test_run_neurons']} randomly selected neurons per subject")
    print(f"Output saved to: {config['data']['output_dir']}")
    print(f"Volume shape: {config['volumization']['volume_shape']}")
    print(f"Data type: {config['volumization']['dtype']}")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 