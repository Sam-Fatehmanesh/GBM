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

def create_sample_video(h5_files, output_path, fps=10, max_frames=100):
    """Create a video from h5 files"""
    # Get dimensions from first file
    with h5py.File(h5_files[0], 'r') as f:
        first_frame = f['default'][0]
        height, width = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Creating video with {min(max_frames, len(h5_files) * 25)} frames...")
    
    for h5_file in tqdm(h5_files):
        with h5py.File(h5_file, 'r') as f:
            data = f['default'][:]
            for frame in data:
                if frame_count >= max_frames:
                    break
                    
                # Normalize and convert frame
                frame_uint8 = normalize_frame(frame)
                
                # Convert to RGB (OpenCV requirement)
                frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
                
                # Write frame
                out.write(frame_rgb)
                frame_count += 1
                
        if frame_count >= max_frames:
            break
    
    out.release()
    print(f"Video saved as {output_path}")

def main():
    # Get all h5 files from recording_2
    h5_files = sorted(glob("data/recording_2_preprocessed/volume*.h5"))
    
    if not h5_files:
        raise ValueError("No h5 files found in recording_2_preprocessed directory")
    
    # Create output directory if it doesn't exist
    os.makedirs("sample_videos", exist_ok=True)
    
    # Create sample video
    output_path = "sample_videos/recording_2_sample.mp4"
    create_sample_video(h5_files, output_path, fps=1, max_frames=256)

if __name__ == "__main__":
    main() 