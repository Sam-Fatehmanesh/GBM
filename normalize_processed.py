import h5py
import numpy as np
from glob import glob
import os
from tqdm import tqdm

def normalize_volume(input_file, output_file):
    """
    Normalize a processed volume from 0-255 to 0-1 range.
    
    Args:
        input_file: Path to input h5 file (0-255 range)
        output_file: Path to output h5 file (0-1 range)
    """
    with h5py.File(input_file, 'r') as f:
        data = f['default'][:].astype(np.float32)
        # Normalize from 0-255 to 0-1
        data = data / 255.0
        
        # Save normalized data
        with h5py.File(output_file, 'w') as out_f:
            out_f.create_dataset('default', data=data)

def main():
    # Create output directory
    os.makedirs('data_normalized', exist_ok=True)
    
    # Get all processed files
    input_files = sorted(glob(os.path.join('data', 'volume*.h5')))
    
    if not input_files:
        print("No processed files found in data directory!")
        return
    
    print(f"Found {len(input_files)} files to normalize")
    
    # Process each file
    for input_file in tqdm(input_files, desc='Normalizing volumes'):
        # Create corresponding output filename
        filename = os.path.basename(input_file)
        output_file = os.path.join('data_normalized', filename)
        
        # Normalize the volume
        normalize_volume(input_file, output_file)
    
    print("Normalization complete. Normalized files are in data_normalized directory.")

if __name__ == '__main__':
    main() 