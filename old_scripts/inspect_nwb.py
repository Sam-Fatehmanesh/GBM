import h5py
import numpy as np

def print_h5_structure(h5_file, indent=0):
    """Recursively print the structure of an HDF5 file."""
    for key in h5_file.keys():
        print('  ' * indent + f'├── {key}')
        if isinstance(h5_file[key], h5py.Group):
            print_h5_structure(h5_file[key], indent + 1)
        elif isinstance(h5_file[key], h5py.Dataset):
            dataset = h5_file[key]
            print('  ' * indent + f'│   ├── Shape: {dataset.shape}')
            print('  ' * indent + f'│   ├── Dtype: {dataset.dtype}')
            print('  ' * indent + f'│   └── Chunks: {dataset.chunks}')

def main():
    nwb_path = "raw_trace_data_2019/sub-20161022-1_ses-20161022T151003_ophys.nwb"
    
    print(f"Examining NWB file structure: {nwb_path}")
    print("This will only read the metadata, not the actual data arrays.")
    
    try:
        with h5py.File(nwb_path, 'r') as f:
            print("\nFile structure:")
            print_h5_structure(f)
            
            # Print some basic metadata
            print("\nBasic metadata:")
            for key, value in f.attrs.items():
                print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error examining file: {str(e)}")

if __name__ == "__main__":
    main() 