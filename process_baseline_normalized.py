import h5py
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pdb

def calculate_slice_means(input_files, slice_interval=10):
    """
    Calculate mean values for each slice position across all volumes,
    after normalizing each frame between 0 and 1.
    
    Args:
        input_files: List of paths to input h5 files
        slice_interval: Number of frames between same slice positions
        
    Returns:
        slice_means: Array of shape (n_positions, height, width) containing mean values
    """
    # Get dimensions from first file
    with h5py.File(input_files[0], 'r') as f:
        first_data = f['default'][:]
        _, height, width = first_data.shape
    
    # Initialize arrays for accumulating means
    slice_means = np.zeros((slice_interval, height, width))
    slice_counts = np.zeros(slice_interval)
    
    # Calculate means across all files
    print("\nCalculating global means across all volumes...")
    global_frame_idx = 0
    
    for input_file in tqdm(input_files, desc='Processing volumes for means'):
        with h5py.File(input_file, 'r') as f:
            data = f['default'][:]
            n_frames = data.shape[0]
            
            for i in range(n_frames):
                pos = (global_frame_idx) % slice_interval
                # Normalize frame to [0,1]
                frame_min = data[i].min()
                frame_max = data[i].max()
                std = np.std(data[i])
                mean = np.mean(data[i])
                if frame_max > frame_min:
                    frame_norm = (data[i] - mean) / std
                    frame_norm = data[i]
                    slice_means[pos] += frame_norm
                    slice_counts[pos] += 1
                global_frame_idx += 1
    
    # Compute final means
    for pos in range(slice_interval):
        if slice_counts[pos] > 0:
            slice_means[pos] /= slice_counts[pos]
            
    return slice_means

def find_slice_max_values(input_files, slice_means, slice_interval=10):
    """
    Find maximum values for each slice position after normalization and baseline subtraction.
    
    Args:
        input_files: List of paths to input h5 files
        slice_means: Mean values for each slice position
        slice_interval: Number of frames between same slice positions
        
    Returns:
        slice_maxes: Array of maximum values for each slice position
    """
    slice_maxes = np.zeros(slice_interval)
    global_frame_idx = 0
    
    print("\nFinding maximum values for each slice position...")
    for input_file in tqdm(input_files, desc='Processing volumes for max values'):
        with h5py.File(input_file, 'r') as f:
            data = f['default'][:]
            n_frames = data.shape[0]
            
            for i in range(n_frames):
                pos = (global_frame_idx) % slice_interval
                
                # 1. Normalize frame to [0,1]
                frame_min = data[i].min()
                frame_max = data[i].max()
                std = np.std(data[i])
                mean = np.mean(data[i])
                if frame_max > frame_min:
                    frame_norm = (data[i] - mean) / std
                else:
                    frame_norm = np.zeros_like(data[i])
                frame_norm = data[i]
                
                # 2. Subtract baseline and clip
                frame_baselined = frame_norm - slice_means[pos]
                
                # Update maximum value for this position
                slice_maxes[pos] = max(slice_maxes[pos], frame_baselined.max())
                
                global_frame_idx += 1
    
    return slice_maxes

def process_volume(input_file, output_file, slice_means, slice_maxes, global_start_idx, slice_interval=10):
    """
    Process a single volume file:
    1. Normalize each frame using standard deviation
    2. Subtract the corresponding normalized baseline (mean)
    3. Clip values to [0,1]
    4. Scale to 255 using position-specific maximum values
    
    Args:
        input_file: Path to input h5 file
        output_file: Path to output h5 file
        slice_means: Mean values for each slice position
        slice_maxes: Maximum values for each slice position
        global_start_idx: The global frame index at the start of this volume
        slice_interval: Number of frames between same slice positions
    """
    with h5py.File(input_file, 'r') as f:
        data = f['default'][:]
        n_frames = data.shape[0]
        
        # Process each frame
        processed_data = np.zeros_like(data, dtype=np.float32)
        for i in range(n_frames):
            pos = (global_start_idx + i) % slice_interval
            
            # 1. Normalize frame using standard deviation
            frame_min = data[i].min()
            frame_std = np.std(data[i])
            mean = np.mean(data[i])

            if frame_std > 0:
                frame_norm = (data[i] - mean) / frame_std
            else:
                frame_norm = np.zeros_like(data[i])

            frame_norm = data[i]

            # 2. Subtract baseline (per-pixel, per-z-coordinate mean)
            frame_baselined = frame_norm - slice_means[pos]

            # 3. Clip negative values
            frame_clipped = np.maximum(frame_baselined, 0) 

            # linear normalization 0-1 with min and max
            processed_data[i] = frame_clipped / slice_maxes[pos]


        # Save processed data
        with h5py.File(output_file, 'w') as out_f:
            out_f.create_dataset('default', data=processed_data)

def process_recording(input_dir, output_dir, recording_name):
    """
    Process all h5 files from a single recording.
    
    Args:
        input_dir: Directory containing input h5 files
        output_dir: Base directory for processed outputs
        recording_name: Name of the recording (used for output subdirectory)
    """
    # Create output directory for this recording
    recording_output_dir = os.path.join(output_dir, f"{recording_name}_preprocessed")
    os.makedirs(recording_output_dir, exist_ok=True)
    
    # Get all input files
    input_files = sorted(glob(os.path.join(input_dir, '*.h5')))
    if not input_files:
        raise ValueError(f"No h5 files found in {input_dir}")
    
    # Calculate slice means using normalized frames
    slice_means = calculate_slice_means(input_files)
    
    # Find maximum values for each slice position
    slice_maxes = find_slice_max_values(input_files, slice_means)
    
    # Save the slice statistics for reference
    with h5py.File(os.path.join(recording_output_dir, 'slice_statistics.h5'), 'w') as f:
        f.create_dataset('slice_means', data=slice_means)
        f.create_dataset('slice_maxes', data=slice_maxes)
    
    # Process each file using the slice means and maxes
    print("\nProcessing individual volumes...")
    global_frame_idx = 0
    
    for input_file in tqdm(input_files, desc='Applying preprocessing'):
        # Create corresponding output filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(recording_output_dir, filename)
        
        # Process the volume
        process_volume(input_file, output_file, slice_means, slice_maxes, global_frame_idx)
        
        # Update global frame index
        with h5py.File(input_file, 'r') as f:
            global_frame_idx += f['default'].shape[0]

def tif_to_h5(tif_path, output_dir):
    """
    Convert a tif file to h5 files.
    
    Args:
        tif_path: Path to input tif file
        output_dir: Directory to save h5 files
    """
    from skimage import io
    
    # Read the tif file
    print(f"Reading tif file: {tif_path}")
    data = io.imread(tif_path)
    nfr_per_vol = 25
    prev_frame = 0
    total_num_frames = len(data)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting to h5 files...")
    for volume, next_frame in enumerate(
        tqdm(np.arange(nfr_per_vol, total_num_frames + nfr_per_vol, nfr_per_vol))
    ):
        h5_path = os.path.join(output_dir, f"volume{1000 + volume}.h5")
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset(
                "default", data=data[prev_frame:next_frame]
            )
        prev_frame = next_frame
    print(f"Conversion complete. H5 files saved in {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process brain recording data')
    parser.add_argument('--tif_file', type=str, help='Path to input tif file')
    parser.add_argument('--input_dir', type=str, help='Directory containing input h5 files')
    parser.add_argument('--output_dir', type=str, default='data', help='Base directory for outputs')
    parser.add_argument('--recording_name', type=str, required=True, help='Name of the recording (e.g. recording_1)')
    
    args = parser.parse_args()
    
    if args.tif_file:
        # If tif file provided, first convert to h5
        temp_h5_dir = os.path.join(args.output_dir, 'temp_h5')
        tif_to_h5(args.tif_file, temp_h5_dir)
        input_dir = temp_h5_dir
    else:
        input_dir = args.input_dir
        
    if not input_dir:
        raise ValueError("Either --tif_file or --input_dir must be provided")
    
    # Process the recording
    process_recording(input_dir, args.output_dir, args.recording_name)
    
    # Clean up temporary directory if it was created
    if args.tif_file and os.path.exists(temp_h5_dir):
        import shutil
        shutil.rmtree(temp_h5_dir)

if __name__ == '__main__':
    main() 