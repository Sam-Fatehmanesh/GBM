"""
Fast DALI-based data loader for preaugmented brain data.

This module provides a more efficient DALI-based data loader for preaugmented 
brain data that was preprocessed into grid format. This loader is designed to work
with the output of the preprocess_spike_data.py script and provides a significant
speed improvement over the original DALI loader.
"""

import os
import numpy as np
import torch
import h5py
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import gc
import time
from glob import glob

# Enable memory tracking for diagnostics
MEMORY_DIAGNOSTICS = False

def print_memory_stats(prefix=""):
    """Print GPU memory usage statistics"""
    if MEMORY_DIAGNOSTICS:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"{prefix} GPU Memory: Total={t/1e9:.2f}GB, Reserved={r/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free={f/1e9:.2f}GB")

class FastGridLoader:
    """Helper class to load preaugmented grid data for DALI's ExternalSource."""
    def __init__(self, h5_file_path, seq_len=330, split='train', stride=None):
        """Initialize FastGridLoader for efficient loading of preaugmented grid data.
        
        Args:
            h5_file_path: Path to the preaugmented grid data h5 file
            seq_len: Length of sequences to return
            split: 'train' or 'test' (should match the file path)
            stride: Stride between sequence starts. If None, sequences can start at any valid point.
                   If specified, enforces a minimum distance between sequence starts, which
                   controls the maximum allowed overlap. For a max overlap of O, set stride = seq_len - O.
        """
        self.h5_file_path = h5_file_path
        self.seq_len = seq_len
        self.split = split  # Store split for later checks
        self.stride = stride  # New parameter for striding
        
        # Track z-start values for the current batch
        self.current_batch_z_starts = []
        self.current_batch_indices = []
        
        # Extract subject name
        self.subject_name = os.path.basename(os.path.dirname(h5_file_path))
        
        # Open the file temporarily to get metadata
        with h5py.File(h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Get shape information
            self.num_timepoints = f.attrs['num_timepoints']
            self.num_z = f.attrs['num_z_planes']
            self.grids_shape = f['grids'].shape
            
            # Get mapping from index to original timepoint
            self.timepoint_indices = f['timepoint_indices'][:]
            
            # Get train/test mask (1 = train, 0 = test)
            self.is_train = f['is_train'][:]
            
            # Calculate total sequences for random access
            self.total_possible_sequences = self.num_timepoints * self.num_z
            
            # Create valid starting points based on timepoints and z-planes
            self.valid_starts = []
            for tp_idx in range(self.num_timepoints):
                # For each timepoint, we can start at any z-plane
                for z in range(self.num_z):
                    # Check if there are enough frames left for a complete sequence
                    
                    # Calculate how many frames we have from this starting point
                    # First, how many z-planes are left in this timepoint
                    frames_in_current_tp = self.num_z - z
                    
                    # How many complete timepoints we need after the first one
                    needed_complete_tps = (self.seq_len - frames_in_current_tp) // self.num_z
                    
                    # Any additional z-planes needed from the final timepoint
                    remaining_frames = (self.seq_len - frames_in_current_tp) % self.num_z
                    
                    # Total timepoints needed
                    total_tps_needed = 1 + needed_complete_tps
                    if remaining_frames > 0:
                        total_tps_needed += 1
                    
                    # Check if we have enough timepoints available
                    if tp_idx + total_tps_needed <= self.num_timepoints:
                        # Build the list of time-points this window uses
                        tp_span = list(range(tp_idx, tp_idx + total_tps_needed))
                        
                        # Check 1: Verify all frames are from the correct split
                        split_check = True
                        if split == 'train':
                            # For train split, all frames must have is_train == 1
                            for t in tp_span:
                                if self.is_train[t] != 1:
                                    split_check = False
                                    break
                        else:  # split == 'test'
                            # For test split, all frames must have is_train == 0
                            for t in tp_span:
                                if self.is_train[t] != 0:
                                    split_check = False
                                    break
                        
                        if not split_check:
                            continue  # Skip this start point if it crosses a split boundary
                        
                        # Check 2: Verify timepoints are consecutive in the original timeline
                        continuity_check = True
                        for i in range(len(tp_span) - 1):
                            # Original time indices should be consecutive
                            if self.timepoint_indices[tp_span[i]] + 1 != self.timepoint_indices[tp_span[i+1]]:
                                continuity_check = False
                                break
                        
                        if not continuity_check:
                            continue  # Skip this start point if it's not continuous in real time
                        
                        # If we passed both checks, this is a valid start point
                        self.valid_starts.append((tp_idx, z))
        
        # Apply striding if requested
        if self.stride is not None and self.stride > 1:
            # Calculate flat indices for all start points
            flat_starts = [tp * self.num_z + z for tp, z in self.valid_starts]
            
            # Sort by flat index and select points with the stride
            ordered = sorted(zip(flat_starts, self.valid_starts), key=lambda x: x[0])
            
            # Apply striding to enforce minimum distance between starts
            strided_starts = []
            if ordered:  # Check if we have any valid start points
                strided_starts.append(ordered[0][1])  # Always include first point
                last_flat_idx = ordered[0][0]
                
                for flat_idx, (tp, z) in ordered[1:]:
                    if flat_idx - last_flat_idx >= self.stride:
                        strided_starts.append((tp, z))
                        last_flat_idx = flat_idx
            
            self.valid_starts = strided_starts
            print(f"Applied striding with stride={self.stride}: reduced valid starts from {len(ordered)} to {len(self.valid_starts)}")
        
        # Shuffle valid starts
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(self.valid_starts)
        
        print(f"\nFast DALI Dataset (using {os.path.basename(h5_file_path)}):")
        print(f"Subject: {self.subject_name}")
        print(f"Split: {split}")
        print(f"Total z-planes: {self.num_z}")
        print(f"Total timepoints: {self.num_timepoints}")
        print(f"Sequence length: {seq_len}")
        if self.stride is not None:
            print(f"Stride: {self.stride} (max overlap: {seq_len - self.stride})")
        print(f"Total valid sequences: {len(self.valid_starts)}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __call__(self, sample_info):
        """Generate sequences from preaugmented grid data.
        
        Args:
            sample_info: Sample information containing index
            
        Returns:
            batch: Numpy array of shape (batch_size, seq_len, 256, 128) with binary spike data
        """
        start_time = time.time()
        
        # Extract indices from SampleInfo objects
        if hasattr(sample_info, 'idx_in_epoch'):
            # Single SampleInfo object
            idx = int(sample_info.idx_in_epoch)
            batch_size = 1
            indices = [idx]
        elif isinstance(sample_info, list):
            # List of SampleInfo objects or indices
            batch_size = len(sample_info)
            indices = []
            for s in sample_info:
                if hasattr(s, 'idx_in_epoch'):
                    indices.append(int(s.idx_in_epoch))
                else:
                    indices.append(int(s))
        else:
            # Fallback for other cases
            try:
                idx = int(sample_info)
                batch_size = 1
                indices = [idx]
            except (TypeError, ValueError):
                # If we can't convert to int, print debug info and use a default
                if MEMORY_DIAGNOSTICS:
                    print(f"Warning: Unexpected sample_info type: {type(sample_info)}")
                # Use index 0 as fallback
                batch_size = 1
                indices = [0]
        
        # Store batch indices for later reference
        self.current_batch_indices = indices.copy()
        
        # Clear previous batch z-starts and populate with new ones
        self.current_batch_z_starts = []
        
        # Create batch of sequences
        batch = np.zeros((batch_size, self.seq_len, 256, 128), dtype=np.uint8)
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            grids = f['grids']  # Reference to the dataset
            
            # Process each sample in the batch
            for batch_idx, idx in enumerate(indices):
                # Ensure idx is within valid range
                idx = idx % len(self.valid_starts)
                
                # Get starting timepoint and z-plane
                tp_start, z_start = self.valid_starts[idx]
                
                # Store z-start for this sequence
                self.current_batch_z_starts.append(z_start)
                
                # Fill the sequence with frames
                seq_idx = 0
                tp = tp_start
                current_z = z_start
                
                while seq_idx < self.seq_len and tp < self.num_timepoints:
                    # Process z-planes in the current timepoint
                    while current_z < self.num_z and seq_idx < self.seq_len:
                        # Access the precomputed grid directly
                        grid = grids[tp, current_z]
                        
                        # Copy to the batch
                        batch[batch_idx, seq_idx] = grid
                        
                        # Move to next z-plane
                        seq_idx += 1
                        current_z += 1
                    
                    # Move to next timepoint
                    tp += 1
                    
                    # Reset z-plane for the next timepoint
                    current_z = 0
        
        if MEMORY_DIAGNOSTICS:
            batch_size_mb = batch.nbytes / 1e6
            print(f"FastGridLoader: Batch size: {batch_size_mb:.2f} MB")
            print(f"FastGridLoader call took {time.time() - start_time:.2f}s for batch of {batch_size}")
        
        return batch
    
    def get_batch_z_starts(self):
        """Get the z-start values for the current batch."""
        return self.current_batch_z_starts

@pipeline_def
def create_fast_brain_data_pipeline(grid_loader, device="gpu"):
    """Create a DALI pipeline for preaugmented brain sequence data."""
    # Generate indices for the batch
    indices = fn.random.uniform(range=(0, len(grid_loader)), dtype=types.INT64)
    
    # Load the preaugmented grid data directly using external source
    sequences = fn.external_source(
        source=grid_loader,
        num_outputs=1,
        device="cpu",
        batch=False,
        parallel=True
    )
    
    # Make sure we're getting a single tensor, not a list
    sequences = sequences[0] if isinstance(sequences, list) else sequences
    
    # Transfer to GPU and cast to float16/float32 for model processing
    if device == "gpu":
        sequences = sequences.gpu()
        # Cast to float16 for memory efficiency during further processing
        sequences = fn.cast(sequences, dtype=types.FLOAT16)
    
    return sequences

class FastDALIBrainDataLoader:
    """Fast DALI-based data loader for preaugmented brain sequences."""
    def __init__(self, preaugmented_dir, batch_size=128, seq_len=330, split='train', 
                 device_id=0, num_threads=2, gpu_prefetch=1, seed=42, shuffle=True,
                 stride=None):
        """Initialize the FastDALIBrainDataLoader.
        
        Args:
            preaugmented_dir: Directory containing preaugmented data
            batch_size: Batch size for data loading
            seq_len: Length of sequences to return
            split: 'train' or 'test'
            device_id: GPU device ID
            num_threads: Number of threads for data loading
            gpu_prefetch: Number of batches to prefetch to GPU
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle the data
            max_overlap: Maximum allowed overlap between sequences (will be converted to stride)
            stride: Stride between sequence starts (directly sets the minimum gap between starts)
                   If stride is provided, it takes precedence over max_overlap.
        """
        self.batch_size = batch_size
        self.device_id = device_id
        
        # Store z-start information for the current batch
        self.current_batch_z_starts = []

        self.stride = stride
        
        # Find all preaugmented grid files
        self.grid_files = []
        for subject_dir in os.listdir(preaugmented_dir):
            subject_path = os.path.join(preaugmented_dir, subject_dir)
            if os.path.isdir(subject_path):
                grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
                if os.path.exists(grid_file):
                    self.grid_files.append(grid_file)
        
        if not self.grid_files:
            raise ValueError(f"No preaugmented grid files found in {preaugmented_dir}")
        
        print(f"Found {len(self.grid_files)} preaugmented grid files")
        
        # Calculate pipeline batch size
        self.pipeline_batch_size = max(8, self.batch_size // len(self.grid_files))
        
        if MEMORY_DIAGNOSTICS:
            print(f"Using pipeline batch size of {self.pipeline_batch_size} (total batch size: {self.batch_size})")
        
        # Create grid loaders for each file
        self.grid_loaders = [FastGridLoader(f, seq_len, split, stride=self.stride) for f in self.grid_files]
        
        # Create a pipeline for each file (do NOT build yet)
        self.pipelines = []
        for grid_loader in self.grid_loaders:
            pipe = create_fast_brain_data_pipeline(
                grid_loader=grid_loader,
                device="gpu",
                batch_size=self.pipeline_batch_size,
                num_threads=num_threads,
                device_id=device_id,
                seed=seed,
                prefetch_queue_depth=gpu_prefetch,
                enable_memory_stats=MEMORY_DIAGNOSTICS,
                py_start_method="spawn"
            )
            self.pipelines.append(pipe)
            
        # IMPORTANT: Start Python workers on all pipelines BEFORE any build()
        for pipe in self.pipelines:
            pipe.start_py_workers()

        # Now it is safe to build the pipelines (CUDA will be initialised after forking)
        for pipe in self.pipelines:
            pipe.build()
        
        # Create DALI iterators
        self.iterators = []
        for i, pipe in enumerate(self.pipelines):
            self.iterators.append(
                DALIGenericIterator(
                    [pipe],
                    ["sequences"],
                    # Use PARTIAL for all pipelines to avoid dropping data
                    last_batch_policy=LastBatchPolicy.PARTIAL,
                    prepare_first_batch=True,
                    # Add auto_reset to True to automatically reset the iterator
                    auto_reset=True
                )
            )
        
        # Calculate total length
        self.total_length = sum(len(grid_loader) for grid_loader in self.grid_loaders)
        # Calculate steps per epoch based on pipeline batch sizes
        total_pipeline_batch_size = self.pipeline_batch_size * len(self.pipelines)
        self.steps_per_epoch = self.total_length // total_pipeline_batch_size
        
        # Print basic information
        print(f"Created Fast DALI data loader with {len(self.pipelines)} pipelines")
        print(f"Total sequences: {self.total_length}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        if self.stride is not None:
            print(f"Using stride={self.stride} (max_overlap={seq_len - self.stride})")
        
        # Initialize iterator state
        self.current_iterator = 0
        self.current_iter = None
        self.accumulated_batch = None
        self.accumulated_count = 0
        self.current_batch_indices = []
        
        # Print memory stats after initialization
        print_memory_stats("After Fast DALI loader init:")
    
    def __iter__(self):
        """Return a combined iterator that yields from all pipelines in sequence."""
        self.current_iterator = 0
        self.current_iter = iter(self.iterators[0])
        self.accumulated_batch = None
        self.accumulated_count = 0
        self.current_batch_z_starts = []
        return self
    
    def __next__(self):
        """Get the next batch, accumulating from multiple pipelines if needed."""
        if MEMORY_DIAGNOSTICS:
            print_memory_stats("Before batch fetch:")
            start_time = time.time()
        
        # If we already have a full batch accumulated, return it
        if self.accumulated_batch is not None and self.accumulated_count >= self.batch_size:
            result = self.accumulated_batch[:self.batch_size]
            # Keep any remaining samples for the next batch
            if self.accumulated_count > self.batch_size:
                self.accumulated_batch = self.accumulated_batch[self.batch_size:]
                self.accumulated_count -= self.batch_size
            else:
                self.accumulated_batch = None
                self.accumulated_count = 0
                
            if MEMORY_DIAGNOSTICS:
                print(f"Returning accumulated batch: {result.shape}, dtype={result.dtype}")
                print_memory_stats("After returning accumulated batch:")
            
            # Update z-start information for the current batch
            self._update_z_starts_for_current_batch()
                
            return result
        
        # Otherwise, accumulate more data
        try:
            # Get next batch from current iterator
            if self.current_iter is None:
                self.current_iter = iter(self.iterators[self.current_iterator])
            
            batch = next(self.current_iter)
            data = batch[0]["sequences"]
            
            if MEMORY_DIAGNOSTICS:
                print(f"Got batch from DALI: {data.shape}, dtype={data.dtype}")
            
            # Keep data in float16 to reduce memory usage
            if data.dtype != torch.float16:
                data = data.to(torch.float16)
                if MEMORY_DIAGNOSTICS:
                    print(f"Converted data to float16: {data.shape}")
            
            # Accumulate the batch
            if self.accumulated_batch is None:
                self.accumulated_batch = data
                self.accumulated_count = data.size(0)
            else:
                self.accumulated_batch = torch.cat([self.accumulated_batch, data], dim=0)
                self.accumulated_count += data.size(0)
            
            # If we have enough data, return a batch
            if self.accumulated_count >= self.batch_size:
                result = self.accumulated_batch[:self.batch_size]
                # Keep any remaining samples for the next batch
                if self.accumulated_count > self.batch_size:
                    self.accumulated_batch = self.accumulated_batch[self.batch_size:]
                    self.accumulated_count -= self.batch_size
                else:
                    self.accumulated_batch = None
                    self.accumulated_count = 0
                    
                if MEMORY_DIAGNOSTICS:
                    print(f"Returning batch: {result.shape}, dtype={result.dtype}")
                    print(f"Batch fetch took {time.time() - start_time:.2f}s")
                    print_memory_stats("After returning batch:")
                
                # Update z-start information for the current batch
                self._update_z_starts_for_current_batch()
                
                return result
            
            # Otherwise, try to get more data from the next iterator
            self.current_iterator = (self.current_iterator + 1) % len(self.iterators)
            self.current_iter = iter(self.iterators[self.current_iterator])
            return self.__next__()
            
        except StopIteration:
            # Try the next iterator if available
            self.current_iterator = (self.current_iterator + 1) % len(self.iterators)
            
            # If we've gone through all iterators and still don't have a full batch
            if self.current_iterator == 0:
                if self.accumulated_batch is not None and self.accumulated_count > 0:
                    # Return a partial batch if we have any data
                    result = self.accumulated_batch
                    self.accumulated_batch = None
                    self.accumulated_count = 0
                    
                    if MEMORY_DIAGNOSTICS:
                        print(f"Returning partial batch: {result.shape}, dtype={result.dtype}")
                        print(f"Batch fetch took {time.time() - start_time:.2f}s")
                        print_memory_stats("After returning partial batch:")
                    
                    # Update z-start information for the current batch
                    self._update_z_starts_for_current_batch()
                    
                    return result
                else:
                    # Reset for next epoch and raise StopIteration
                    self.reset()
                    raise StopIteration
            
            # Try the next iterator
            self.current_iter = iter(self.iterators[self.current_iterator])
            return self.__next__()
    
    def _update_z_starts_for_current_batch(self):
        """Update the z-start information for the current batch."""
        # Clear previous batch z-starts
        self.current_batch_z_starts = []
        
        # Get z-start values from the current grid loader
        current_grid_loader = self.grid_loaders[self.current_iterator]
        if hasattr(current_grid_loader, 'current_batch_z_starts'):
            # Copy z-start values from the GridLoader
            self.current_batch_z_starts = current_grid_loader.current_batch_z_starts.copy()
    
    @property
    def batch_z_starts(self):
        """Get the z-start values for the current batch."""
        return self.current_batch_z_starts
    
    def get_z_starts(self):
        """Get z-start values for the current batch (convenience method)."""
        return self.current_batch_z_starts
    
    def __len__(self):
        return self.steps_per_epoch
    
    def reset(self):
        """Reset all iterators."""
        for iterator in self.iterators:
            iterator.reset()
        self.accumulated_batch = None
        self.accumulated_count = 0
        self.current_batch_z_starts = []
        
        if MEMORY_DIAGNOSTICS:
            # Force garbage collection to free up memory
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_stats("After reset:") 