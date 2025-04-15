#!/usr/bin/env python
import os
import torch
import numpy as np
import time
import glob
from tqdm import tqdm
import argparse

# Import the DALI dataset components
from GenerativeBrainModel.datasets.dali_spike_dataset import H5Loader, augment_z0_frames

def test_h5_loader(h5_file, seq_len=100, num_samples=10):
    """
    Test the H5Loader class by loading a few samples and measuring performance.
    
    Args:
        h5_file (str): Path to the h5 file
        seq_len (int): Length of sequences to load
        num_samples (int): Number of samples to load
    
    Returns:
        dict: Dictionary with test results
    """
    print(f"\nTesting H5Loader for {os.path.basename(h5_file)}")
    
    try:
        # Initialize loader for training data
        start_time = time.time()
        train_loader = H5Loader(h5_file, seq_len=seq_len, split='train')
        init_time = time.time() - start_time
        print(f"  - Loader initialization time: {init_time:.2f}s")
        print(f"  - Total sequences: {len(train_loader)}")
        print(f"  - Number of Z-planes: {train_loader.num_z}")
        
        # Load a few random samples and measure time
        load_times = []
        sample_shapes = []
        samples = []
        
        for i in tqdm(range(num_samples), desc="Loading samples"):
            # Generate a random sample index
            idx = np.random.randint(0, len(train_loader))
            
            # Create a class that mimics the SampleInfo object
            class SampleInfo:
                def __init__(self, idx):
                    self.idx_in_epoch = idx
            
            sample_info = SampleInfo(idx)
            
            # Load sample and measure time
            start_time = time.time()
            sample = train_loader(sample_info)
            load_time = time.time() - start_time
            
            sample_shapes.append(sample.shape)
            load_times.append(load_time)
            samples.append(sample)
        
        # Calculate statistics
        avg_load_time = np.mean(load_times)
        std_load_time = np.std(load_times)
        print(f"  - Average sample load time: {avg_load_time:.3f}s ± {std_load_time:.3f}s")
        print(f"  - Sample shape: {sample_shapes[0]}")
        
        # Try the augmentation function
        if len(samples) > 0:
            print("  - Testing augmentation function...")
            aug_sample = augment_z0_frames(samples[0], train_loader)
            print(f"  - Augmented sample shape: {aug_sample.shape}")
        
        # Initialize loader for testing data
        start_time = time.time()
        test_loader = H5Loader(h5_file, seq_len=seq_len, split='test')
        init_time = time.time() - start_time
        print(f"  - Test loader initialization time: {init_time:.2f}s")
        print(f"  - Test sequences: {len(test_loader)}")
        
        success = True
    
    except Exception as e:
        print(f"Error testing {h5_file}: {str(e)}")
        success = False
    
    return {
        'file': h5_file,
        'success': success,
        'avg_load_time': avg_load_time if 'avg_load_time' in locals() else None,
    }

def main():
    parser = argparse.ArgumentParser(description='Test DALI dataset loading with prepared data')
    parser.add_argument('--data_dir', type=str, default='training_spike_data_2018',
                        help='Directory containing processed h5 files')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Sequence length for loading')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to load per file')
    parser.add_argument('--num_files', type=int, default=3,
                        help='Number of files to test (set to 0 for all files)')
    
    args = parser.parse_args()
    
    # Find all processed h5 files
    files = glob.glob(os.path.join(args.data_dir, '*_processed.h5'))
    files.sort()
    
    if not files:
        print(f"No processed h5 files found in {args.data_dir}")
        return
    
    print(f"Found {len(files)} processed h5 files")
    
    # Limit the number of files to test if specified
    if args.num_files > 0 and args.num_files < len(files):
        print(f"Testing {args.num_files} of {len(files)} files")
        # Take a sample from the beginning, middle, and end
        if args.num_files == 1:
            files = [files[0]]
        elif args.num_files == 2:
            files = [files[0], files[-1]]
        elif args.num_files == 3:
            files = [files[0], files[len(files)//2], files[-1]]
        else:
            # Take a random sample
            indices = np.linspace(0, len(files)-1, args.num_files).astype(int)
            files = [files[i] for i in indices]
    
    # Test each file
    results = []
    for file_path in files:
        result = test_h5_loader(file_path, seq_len=args.seq_len, num_samples=args.num_samples)
        results.append(result)
    
    # Print summary
    print("\nTest Summary:")
    print(f"  - Total files tested: {len(results)}")
    print(f"  - Successful tests: {sum(1 for r in results if r['success'])}")
    print(f"  - Failed tests: {sum(1 for r in results if not r['success'])}")
    
    if all(r['success'] for r in results):
        print("\nAll tests passed! The data is ready for DALI-based training.")
    else:
        print("\nSome tests failed. Please check the issues and fix them.")

if __name__ == "__main__":
    # Initialize CUDA first to avoid issues with DALI
    if torch.cuda.is_available():
        torch.cuda.init()
    
    main() 