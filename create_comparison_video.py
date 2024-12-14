import h5py
import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm

def normalize_frame(frame):
    """Normalize frame to 0-255 range"""
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max == frame_min:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)

def create_comparison_video(raw_files, processed_files, output_path, fps=1):
    """
    Create a video comparing raw and processed data side by side.
    Left: raw data normalized to 0-255
    Right: processed data normalized to 0-255
    Both sides are normalized independently for better visualization.
    
    Args:
        raw_files: List of paths to raw h5 files
        processed_files: List of paths to processed h5 files
        output_path: Path where to save the video
        fps: Frames per second for the video
    """
    # Get dimensions from first file
    with h5py.File(raw_files[0], 'r') as f:
        data = f['default'][:]
        height, width = data.shape[1:]
    
    # Initialize video writer for side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec for AVI
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    print("Creating comparison video...")
    for raw_file, proc_file in tqdm(zip(raw_files, processed_files), desc='Processing volumes'):
        with h5py.File(raw_file, 'r') as f_raw, h5py.File(proc_file, 'r') as f_proc:
            raw_data = f_raw['default'][:]
            proc_data = f_proc['default'][:]
            
            # Process each frame
            for raw_frame, proc_frame in zip(raw_data, proc_data):
                # Normalize both frames independently
                raw_norm = normalize_frame(raw_frame)
                proc_norm = normalize_frame(proc_frame)
                
                # Convert to RGB
                raw_rgb = cv2.cvtColor(raw_norm, cv2.COLOR_GRAY2BGR)
                proc_rgb = cv2.cvtColor(proc_norm, cv2.COLOR_GRAY2BGR)
                
                # Add labels
                cv2.putText(raw_rgb, 'Raw (Normalized)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(proc_rgb, 'Processed (Normalized)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combine frames side by side
                combined = np.hstack([raw_rgb, proc_rgb])
                
                # Write frame
                out.write(combined)
    
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    # Get first 10 files from both input and data directories
    raw_files = sorted(glob(os.path.join('input', 'volume*.h5')))[:10]
    proc_files = sorted(glob(os.path.join('data', 'volume*.h5')))[:10]
    
    if not raw_files or not proc_files:
        print("Could not find enough files in input or data directories!")
        return
    
    if len(raw_files) != len(proc_files):
        print("Mismatch between number of raw and processed files!")
        return
    
    print(f"Found {len(raw_files)} files to process")
    
    # Create video
    output_path = 'brain_scan_comparison.avi'  # Changed to .avi
    create_comparison_video(raw_files, proc_files, output_path)

if __name__ == '__main__':
    main() 