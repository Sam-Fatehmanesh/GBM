import os
import re
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage import io
import argparse
import shutil
import json
from datetime import datetime

def natural_sort_key(s):
    """Key function for natural sorting of strings with numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['raw_data', 'data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return {d: os.path.abspath(d) for d in dirs}

def detect_frames_per_volume(filename):
    """Try to detect frames per volume from filename pattern"""
    # Pattern: looking for numbers followed by "slices" or "z"
    patterns = [
        r'(\d+)[\-_]?slices',  # matches: 10slices, 10-slices, 10_slices
        r'(\d+)[\-_]?z',       # matches: 10z, 10-z, 10_z
        r'z(\d+)',             # matches: z10
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            return int(match.group(1))
    return None

def load_file_config(config_file="data/file_config.json"):
    """Load file-specific configurations"""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}

def save_file_config(config, config_file="data/file_config.json"):
    """Save file-specific configurations"""
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def tif_to_h5(tif_path, output_dir, frames_per_volume=25):
    """
    Convert a tif file to h5 files.
    
    Args:
        tif_path: Path to input tif file
        output_dir: Directory to save h5 files
        frames_per_volume: Number of frames per volume
    """
    print(f"Processing: {tif_path}")
    print(f"Frames per volume: {frames_per_volume}")
    print(f"Reading tif file...")
    data = io.imread(tif_path)
    
    total_frames = len(data)
    n_volumes = total_frames // frames_per_volume
    
    if total_frames % frames_per_volume != 0:
        print(f"Warning: Total frames ({total_frames}) is not divisible by frames_per_volume ({frames_per_volume})")
        print(f"Truncating to {n_volumes * frames_per_volume} frames")
    
    print(f"Converting {n_volumes} volumes to h5 files...")
    for volume in tqdm(range(n_volumes), desc="Converting volumes"):
        start_frame = volume * frames_per_volume
        end_frame = start_frame + frames_per_volume
        
        h5_path = os.path.join(output_dir, f"volume{1000 + volume}.h5")
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("default", data=data[start_frame:end_frame])

def process_baseline_normalized(input_dir, output_dir, slice_interval=25):
    """Process h5 files with baseline normalization"""
    from process_baseline_normalized import process_recording
    process_recording(input_dir, output_dir, os.path.basename(output_dir))

def get_next_recording_number():
    """Get the next available recording number"""
    existing_dirs = glob("data/recording_*")
    if not existing_dirs:
        return 1
    
    numbers = [int(re.search(r'recording_(\d+)', d).group(1)) 
              for d in existing_dirs if re.search(r'recording_(\d+)', d)]
    return max(numbers) + 1 if numbers else 1

def process_new_recordings(default_frames_per_volume=25, file_frames_map=None):
    """
    Process any new tif files found in raw_data directory
    
    Args:
        default_frames_per_volume: Default number of frames per volume
        file_frames_map: Dictionary mapping filenames to their frames per volume
    """
    dirs = ensure_directories()
    file_frames_map = file_frames_map or {}
    
    # Load existing file configuration
    file_config = load_file_config()
    
    # Get list of tif files that haven't been processed
    processed_log = "data/processed_files.txt"
    processed_files = set()
    if os.path.exists(processed_log):
        with open(processed_log, 'r') as f:
            processed_files = set(line.strip() for line in f)
    
    tif_files = sorted(glob(os.path.join(dirs['raw_data'], '*.tif')), 
                      key=natural_sort_key)
    new_files = [f for f in tif_files if f not in processed_files]
    
    if not new_files:
        print("No new tif files to process")
        return
    
    print(f"Found {len(new_files)} new tif files to process")
    
    # Process each new file
    for tif_file in new_files:
        filename = os.path.basename(tif_file)
        print(f"\nProcessing {filename}")
        
        # Determine frames per volume for this file
        frames = None
        
        # 1. Check command-line specified mapping
        if filename in file_frames_map:
            frames = file_frames_map[filename]
        
        # 2. Check saved configuration
        elif filename in file_config:
            frames = file_config[filename]
        
        # 3. Try to detect from filename
        else:
            frames = detect_frames_per_volume(filename)
        
        # 4. Use default if no other option found
        if frames is None:
            print(f"Could not detect frames per volume from filename. Using default: {default_frames_per_volume}")
            frames = default_frames_per_volume
        
        # Save configuration for this file
        file_config[filename] = frames
        save_file_config(file_config)
        
        recording_num = get_next_recording_number()
        
        # Create temporary directory for h5 files
        temp_dir = os.path.join(dirs['data'], 'temp_h5')
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Convert tif to h5
            tif_to_h5(tif_file, temp_dir, frames)
            
            # Process with baseline normalization
            output_dir = os.path.join(dirs['data'], f'recording_{recording_num}')
            process_baseline_normalized(temp_dir, dirs['data'], f'recording_{recording_num}')
            
            # Log the processed file
            with open(processed_log, 'a') as f:
                f.write(f"{tif_file}\n")
            
            print(f"Successfully processed {filename} as recording_{recording_num}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Process brain recording data')
    parser.add_argument('--default-frames', type=int, default=25,
                      help='Default number of frames per volume (default: 25)')
    parser.add_argument('--file-frames', nargs=2, action='append',
                      metavar=('FILENAME', 'FRAMES'),
                      help='Specify frames per volume for specific files')
    parser.add_argument('--reprocess-all', action='store_true',
                      help='Reprocess all tif files, even if already processed')
    
    args = parser.parse_args()
    
    # Convert file-frames arguments to dictionary
    file_frames_map = {}
    if args.file_frames:
        file_frames_map = {filename: int(frames) 
                          for filename, frames in args.file_frames}
    
    if args.reprocess_all:
        # Remove processed files log and all recording directories
        if os.path.exists("data/processed_files.txt"):
            os.remove("data/processed_files.txt")
        if os.path.exists("data/file_config.json"):
            os.remove("data/file_config.json")
        for d in glob("data/recording_*"):
            shutil.rmtree(d)
    
    process_new_recordings(args.default_frames, file_frames_map)

if __name__ == '__main__':
    main() 