"""
Probability-aware data loader for preaugmented brain data with spike probability targets.

This module extends the FastDALIBrainDataLoader to support loading probability grids
as targets instead of binary spike grids, enabling richer training signals.
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
import tempfile
from typing import List, Optional
from tqdm import tqdm

from .fast_dali_spike_dataset import FastDALIBrainDataLoader, MEMORY_DIAGNOSTICS, print_memory_stats


class ProbabilityGridLoader:
    """Helper class to load probability or spike grids for DALI's ExternalSource."""
    
    def __init__(self, h5_file_path, seq_len=330, split='train', stride=None, use_probabilities=True):
        """Initialize ProbabilityGridLoader for efficient loading of probability/spike grid data.
        
        Args:
            h5_file_path: Path to the preaugmented grid data h5 file  
            seq_len: Length of sequences to return
            split: 'train' or 'test' (should match the file path)
            stride: Stride between sequence starts
            use_probabilities: If True, load probability grids. If False, load binary spike grids.
        """
        self.h5_file_path = h5_file_path
        self.seq_len = seq_len
        self.split = split
        self.stride = stride
        self.use_probabilities = use_probabilities
        
        # Track z-start values for the current batch
        self.current_batch_z_starts = []
        self.current_batch_indices = []
        
        # Extract subject name
        self.subject_name = os.path.basename(os.path.dirname(h5_file_path))
        
        # Open the file temporarily to get metadata
        with h5py.File(h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Check what datasets are available
            available_keys = list(f.keys())
            self.has_probabilities = 'probability_grids' in available_keys
            
            if use_probabilities and not self.has_probabilities:
                tqdm.write(f"Warning: Probability grids not found in {h5_file_path}, falling back to binary spikes")
                self.use_probabilities = False
            
            # Determine which dataset to use
            if self.use_probabilities and self.has_probabilities:
                self.data_key = 'probability_grids'
                tqdm.write(f"Using probability grids as targets from {self.subject_name}")
            elif 'spike_grids' in available_keys:
                self.data_key = 'spike_grids'
                tqdm.write(f"Using binary spike grids as targets from {self.subject_name}")
            else:
                # Fallback to old format
                self.data_key = 'grids'
                self.has_probabilities = False
                self.use_probabilities = False
                tqdm.write(f"Using legacy grid format from {self.subject_name}")
            
            # Get shape information
            self.num_timepoints = f.attrs['num_timepoints']
            self.num_z = f.attrs['num_z_planes']
            self.grids_shape = f[self.data_key].shape
            
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
                    
                    # Check if we have enough timepoints left
                    if tp_idx + total_tps_needed <= self.num_timepoints:
                        # Check if this timepoint belongs to the correct split
                        is_train_timepoint = self.is_train[tp_idx] == 1
                        
                        if ((split == 'train' and is_train_timepoint) or 
                            (split == 'test' and not is_train_timepoint)):
                            
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
        
        print(f"\nProbability DALI Dataset (using {os.path.basename(h5_file_path)}):")
        print(f"Subject: {self.subject_name}")
        print(f"Split: {split}")
        print(f"Data source: {self.data_key}")
        print(f"Using probabilities: {self.use_probabilities}")
        print(f"Total z-planes: {self.num_z}")
        print(f"Total timepoints: {self.num_timepoints}")
        print(f"Sequence length: {seq_len}")
        if self.stride is not None:
            print(f"Stride: {self.stride} (max overlap: {seq_len - self.stride})")
        print(f"Total valid sequences: {len(self.valid_starts)}")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __call__(self, sample_info):
        """Generate sequences from preaugmented grid data (probability or binary).
        
        Args:
            sample_info: Sample information containing index
            
        Returns:
            batch: Numpy array of shape (batch_size, seq_len, 256, 128) with probability or binary data
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
        
        # For probability training: load both binary inputs and probability targets
        # For binary training: load only binary data
        input_batch = np.zeros((batch_size, self.seq_len, 256, 128), dtype=np.uint8)  # Always binary for model input
        
        if self.use_probabilities:
            target_batch = np.zeros((batch_size, self.seq_len, 256, 128), dtype=np.float16)  # Probabilities for targets
        else:
            target_batch = None  # Will use same as input for binary training
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Load binary spike grids for model input (always binary)
            if 'spike_grids' in f:
                input_grids = f['spike_grids']
            else:
                # Fallback to old format or thresholded probabilities
                input_grids = f[self.data_key]
            
            # Load target grids based on training mode
            if self.use_probabilities and 'probability_grids' in f:
                target_grids = f['probability_grids']
            else:
                target_grids = input_grids  # Use same as input
            
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
                        # Load input grid (always binary)
                        input_grid = input_grids[tp, current_z]
                        if input_grid.max() > 1:  # If probabilities, threshold to binary
                            input_grid = (input_grid > 0.5).astype(np.uint8)
                        input_batch[batch_idx, seq_idx] = input_grid
                        
                        # Load target grid (probabilities or binary)
                        if self.use_probabilities and target_batch is not None:
                            target_grid = target_grids[tp, current_z]
                            target_batch[batch_idx, seq_idx] = target_grid
                        
                        # Move to next z-plane
                        seq_idx += 1
                        current_z += 1
                    
                    # Move to next timepoint
                    tp += 1
                    
                    # Reset z-plane for the next timepoint
                    current_z = 0
        
        if MEMORY_DIAGNOSTICS:
            input_size_mb = input_batch.nbytes / 1e6
            target_size_mb = target_batch.nbytes / 1e6 if target_batch is not None else 0
            print(f"ProbabilityGridLoader: Input batch: {input_size_mb:.2f} MB, Target batch: {target_size_mb:.2f} MB")
            print(f"ProbabilityGridLoader call took {time.time() - start_time:.2f}s for batch of {batch_size}")
        
        if self.use_probabilities and target_batch is not None:
            return input_batch, target_batch, np.array(self.current_batch_z_starts, dtype=np.int64)
        else:
            return input_batch, np.array(self.current_batch_z_starts, dtype=np.int64)


@pipeline_def
def create_probability_brain_data_pipeline(grid_loader, device="gpu"):
    """Create a DALI pipeline for preaugmented brain sequence data with probability targets."""
    # Generate indices for the batch
    indices = fn.random.uniform(range=(0, len(grid_loader)), dtype=types.INT64)
    
    # Load the preaugmented grid data directly using external source
    if grid_loader.use_probabilities:
        outputs = fn.external_source(
            source=grid_loader,
            num_outputs=3,  # input_batch, target_batch, z_starts
            device="cpu",
            batch=True,
            parallel=True
        )
        # Unpack outputs: input_sequences, target_sequences, z_starts
        input_sequences = outputs[0]
        target_sequences = outputs[1]
        z_starts = outputs[2]
    else:
        outputs = fn.external_source(
            source=grid_loader,
            num_outputs=2,  # input_batch, z_starts
            device="cpu",
            batch=True,
            parallel=True
        )
        # Unpack outputs: input_sequences, z_starts
        input_sequences = outputs[0]
        target_sequences = input_sequences  # Use same for binary training
        z_starts = outputs[1]
    
    # Transfer to GPU and cast appropriately for model processing
    if device == "gpu":
        input_sequences = input_sequences.gpu()
        target_sequences = target_sequences.gpu()
        # Cast input to float16 for model (binary data)
        input_sequences = fn.cast(input_sequences, dtype=types.FLOAT16)
        # Cast targets appropriately (float16 for probabilities, float16 for binary)
        target_sequences = fn.cast(target_sequences, dtype=types.FLOAT16)
    
    return input_sequences, target_sequences, z_starts


class ProbabilityDALIBrainDataLoader(FastDALIBrainDataLoader):
    """Probability-aware DALI-based data loader for brain sequences with optional probability targets."""
    
    def __init__(self, preaugmented_dir, batch_size=128, seq_len=330, split='train', 
                 device_id=0, num_threads=2, gpu_prefetch=1, seed=42, shuffle=True,
                 stride=None, use_probabilities=True):
        """Initialize the ProbabilityDALIBrainDataLoader.
        
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
            stride: Stride between sequence starts
            use_probabilities: If True, use probability grids as targets. If False, use binary spikes.
        """
        self.use_probabilities = use_probabilities
        
        # Store parameters for grid loader creation
        self.preaugmented_dir = preaugmented_dir
        self.seq_len = seq_len
        self.split = split
        self.stride = stride
        
        # Call parent __init__ but we'll override some parts
        self.batch_size = batch_size
        self.device_id = device_id
        
        # Store z-start information for the current batch
        self.current_batch_z_starts = []
        
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
        
        # Create probability grid loaders for each file
        self.grid_loaders = [
            ProbabilityGridLoader(f, seq_len, split, stride=stride, use_probabilities=use_probabilities) 
            for f in self.grid_files
        ]
        
        # Create a pipeline for each file using the probability pipeline
        self.pipelines = []
        for grid_loader in self.grid_loaders:
            pipe = create_probability_brain_data_pipeline(
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
            
            # Start Python workers before building the pipeline
            pipe.start_py_workers()
            pipe.build()
            self.pipelines.append(pipe)
        
        # Create DALI iterators
        self.iterators = []
        for i, pipe in enumerate(self.pipelines):
            self.iterators.append(
                DALIGenericIterator(
                    [pipe],
                    ["input_sequences", "target_sequences", "z_starts"] if self.use_probabilities else ["input_sequences", "z_starts"],
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
        print(f"Created Probability DALI data loader with {len(self.pipelines)} pipelines")
        print(f"Using probabilities as targets: {self.use_probabilities}")
        print(f"Total sequences: {self.total_length}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        if self.stride is not None:
            print(f"Using stride={self.stride} (max_overlap={seq_len - self.stride})")
        
        # Initialize iterator state (inherits __iter__ and __next__ from parent)
        self.current_iterator = 0
        self.current_iter = None
        self.accumulated_batch = None
        self.accumulated_count = 0
        self.current_batch_indices = []
        
        # Print memory stats after initialization
        print_memory_stats("After Probability DALI loader init:")

    def __next__(self):
        """Get the next batch, handling separate input and target sequences for probability mode."""
        if MEMORY_DIAGNOSTICS:
            print_memory_stats("Before probability batch fetch:")
            start_time = time.time()
        
        # If we already have a full batch accumulated, return it
        if self.accumulated_batch is not None and self.accumulated_count >= self.batch_size:
            if self.use_probabilities:
                # Return separate input and target batches
                input_result = self.accumulated_batch[0][:self.batch_size]
                target_result = self.accumulated_batch[1][:self.batch_size]
                result = (input_result, target_result)
                
                # Keep any remaining samples for the next batch
                if self.accumulated_count > self.batch_size:
                    self.accumulated_batch = (
                        self.accumulated_batch[0][self.batch_size:],
                        self.accumulated_batch[1][self.batch_size:]
                    )
                    self.accumulated_count -= self.batch_size
                else:
                    self.accumulated_batch = None
                    self.accumulated_count = 0
            else:
                # Binary mode: return single tensor like parent class
                result = self.accumulated_batch[:self.batch_size]
                if self.accumulated_count > self.batch_size:
                    self.accumulated_batch = self.accumulated_batch[self.batch_size:]
                    self.accumulated_count -= self.batch_size
                else:
                    self.accumulated_batch = None
                    self.accumulated_count = 0
                
            if MEMORY_DIAGNOSTICS:
                if self.use_probabilities:
                    print(f"Returning probability batch: input={result[0].shape}, target={result[1].shape}")
                else:
                    print(f"Returning binary batch: {result.shape}")
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
            
            if self.use_probabilities:
                # Extract separate input and target data
                input_data = batch[0]["input_sequences"]
                target_data = batch[0]["target_sequences"]
                
                if MEMORY_DIAGNOSTICS:
                    print(f"Got probability batch from DALI: input={input_data.shape}, target={target_data.shape}")
                
                # Keep data in float16 to reduce memory usage
                if input_data.dtype != torch.float16:
                    input_data = input_data.to(torch.float16)
                if target_data.dtype != torch.float16:
                    target_data = target_data.to(torch.float16)
                
                # Accumulate the batch
                if self.accumulated_batch is None:
                    self.accumulated_batch = (input_data, target_data)
                    self.accumulated_count = input_data.size(0)
                else:
                    self.accumulated_batch = (
                        torch.cat([self.accumulated_batch[0], input_data], dim=0),
                        torch.cat([self.accumulated_batch[1], target_data], dim=0)
                    )
                    self.accumulated_count += input_data.size(0)
            else:
                # Binary mode: use only input sequences
                data = batch[0]["input_sequences"]
                
                if MEMORY_DIAGNOSTICS:
                    print(f"Got binary batch from DALI: {data.shape}")
                
                if data.dtype != torch.float16:
                    data = data.to(torch.float16)
                
                # Accumulate the batch
                if self.accumulated_batch is None:
                    self.accumulated_batch = data
                    self.accumulated_count = data.size(0)
                else:
                    self.accumulated_batch = torch.cat([self.accumulated_batch, data], dim=0)
                    self.accumulated_count += data.size(0)
            
            # If we have enough data, return a batch
            if self.accumulated_count >= self.batch_size:
                if self.use_probabilities:
                    # Return separate input and target batches
                    input_result = self.accumulated_batch[0][:self.batch_size]
                    target_result = self.accumulated_batch[1][:self.batch_size]
                    result = (input_result, target_result)
                    
                    # Keep any remaining samples
                    if self.accumulated_count > self.batch_size:
                        self.accumulated_batch = (
                            self.accumulated_batch[0][self.batch_size:],
                            self.accumulated_batch[1][self.batch_size:]
                        )
                        self.accumulated_count -= self.batch_size
                    else:
                        self.accumulated_batch = None
                        self.accumulated_count = 0
                else:
                    # Binary mode: return single tensor
                    result = self.accumulated_batch[:self.batch_size]
                    if self.accumulated_count > self.batch_size:
                        self.accumulated_batch = self.accumulated_batch[self.batch_size:]
                        self.accumulated_count -= self.batch_size
                    else:
                        self.accumulated_batch = None
                        self.accumulated_count = 0
                    
                if MEMORY_DIAGNOSTICS:
                    if self.use_probabilities:
                        print(f"Returning probability batch: input={result[0].shape}, target={result[1].shape}")
                    else:
                        print(f"Returning binary batch: {result.shape}")
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
                    if self.use_probabilities:
                        result = self.accumulated_batch  # Tuple of (input, target)
                    else:
                        result = self.accumulated_batch  # Single tensor
                    
                    self.accumulated_batch = None
                    self.accumulated_count = 0
                    
                    if MEMORY_DIAGNOSTICS:
                        if self.use_probabilities:
                            print(f"Returning partial probability batch: input={result[0].shape}, target={result[1].shape}")
                        else:
                            print(f"Returning partial binary batch: {result.shape}")
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

    def reset(self):
        """Reset all iterators, handling tuple accumulated batch for probability mode."""
        for iterator in self.iterators:
            iterator.reset()
        self.accumulated_batch = None
        self.accumulated_count = 0
        self.current_batch_z_starts = []
        
        if MEMORY_DIAGNOSTICS:
            # Force garbage collection to free up memory
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_stats("After probability reset:")


class SubjectFilteredProbabilityDALIBrainDataLoader(ProbabilityDALIBrainDataLoader):
    """ProbabilityDALIBrainDataLoader that can filter subjects based on include/exclude lists."""
    
    def __init__(
        self,
        preaugmented_dir: str,
        include_subjects: Optional[List[str]] = None,
        exclude_subjects: Optional[List[str]] = None,
        use_probabilities: bool = True,
        **kwargs
    ):
        """Initialize the SubjectFilteredProbabilityDALIBrainDataLoader.
        
        Args:
            preaugmented_dir: Directory containing preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            use_probabilities: If True, use probability grids as targets. If False, use binary spikes.
            **kwargs: Additional arguments to pass to ProbabilityDALIBrainDataLoader
        """
        self.original_dir = preaugmented_dir
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects if exclude_subjects else []
        
        # Create a temporary directory to store the filtered subject directories
        self.temp_dir = None
        filtered_dir = self._create_filtered_subject_dir(preaugmented_dir, include_subjects, exclude_subjects)
        
        # Initialize the parent class with the filtered directory
        super().__init__(filtered_dir, use_probabilities=use_probabilities, **kwargs)
        
        # Print subjects being used
        if include_subjects:
            tqdm.write(f"Using only subjects: {include_subjects}")
        if exclude_subjects:
            tqdm.write(f"Excluding subjects: {exclude_subjects}")
        
        tqdm.write(f"Probability target mode: {use_probabilities}")
    
    def _create_filtered_subject_dir(self, preaugmented_dir, include_subjects, exclude_subjects):
        """Create a temporary directory with symlinks to only the desired subject directories.
        
        Args:
            preaugmented_dir: Original directory containing all preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            
        Returns:
            filtered_dir: Path to a temporary directory containing only the filtered subjects
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="filtered_probability_subjects_")
        
        # Get all subjects in the preaugmented directory
        all_subjects = []
        for subject_dir in os.listdir(preaugmented_dir):
            subject_path = os.path.join(preaugmented_dir, subject_dir)
            if os.path.isdir(subject_path):
                # Check if this is a valid subject directory (contains preaugmented_grids.h5)
                grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
                if os.path.exists(grid_file):
                    all_subjects.append(subject_dir)
        
        # Filter subjects based on include/exclude lists
        if include_subjects is not None:
            filtered_subjects = [s for s in all_subjects if s in include_subjects]
        else:
            filtered_subjects = [s for s in all_subjects if s not in exclude_subjects]
        
        # If no subjects remain after filtering, raise an error
        if not filtered_subjects:
            raise ValueError("No subjects left after filtering!")
        
        # Create symlinks to the filtered subject directories
        for subject in filtered_subjects:
            # Use absolute path for the source to ensure symlinks work correctly
            src_path = os.path.abspath(os.path.join(preaugmented_dir, subject))
            dst_path = os.path.join(self.temp_dir, subject)
            os.symlink(src_path, dst_path)
        
        tqdm.write(f"Created filtered directory with {len(filtered_subjects)} subjects: {filtered_subjects}")
        
        return self.temp_dir
    
    def __del__(self):
        """Clean up temporary directory when the object is deleted."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Delete symlinks but not the actual data
                for item in os.listdir(self.temp_dir):
                    os.unlink(os.path.join(self.temp_dir, item))
                # Remove the temporary directory
                os.rmdir(self.temp_dir)
            except Exception as e:
                # Print but don't fail if cleanup encounters an error
                print(f"Warning: Error during cleanup of temporary directory: {e}") 