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

class H5Loader:
    """Helper class to load data from HDF5 files for DALI's ExternalSource."""
    def __init__(self, h5_file_path, seq_len=330, split='train', train_ratio=0.95):
        """Initialize H5Loader for efficient loading of spike data.
        
        This loader ensures that:
        1. Sequences can start at any z-plane for the first timepoint
        2. Each new timepoint in a sequence always starts at z=0 (most dorsal z-plane)
        3. Z-planes are processed in anatomical order for each timepoint
        4. Training and test sets are completely separated by timepoint blocks
        
        Args:
            h5_file_path: Path to the processed spike data h5 file
            seq_len: Length of sequences to return
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training
        """
        self.h5_file_path = h5_file_path
        self.seq_len = seq_len
        
        # Open the file temporarily to get metadata and precompute indices
        with h5py.File(h5_file_path, 'r', libver='latest', swmr=True) as f:
            # Get shape information
            self.num_timepoints = f['spikes'].shape[0]
            
            # Load cell positions (small enough to keep in memory)
            self.cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
            
            # Get unique z values (rounded to handle floating point precision)
            z_values = np.unique(np.round(self.cell_positions[:, 2], decimals=3))
            # Sort in ascending order (lower z-planes first)
            self.z_values = np.sort(z_values)  # Keep ascending order
            self.num_z = len(self.z_values)
        
        # Pre-compute z-plane masks and cell indices for each z-plane
        self.z_masks = {}
        self.z_cell_indices = {}  # Store pre-computed indices for faster access
        
        # Normalize cell positions to [0, 1]
        normalized_positions = (self.cell_positions - self.cell_positions.min(axis=0)) / \
                            (self.cell_positions.max(axis=0) - self.cell_positions.min(axis=0))
        
        # Convert to grid indices
        self.cell_x = np.floor(normalized_positions[:, 0] * 255).astype(np.int32)  # 0-255
        self.cell_y = np.floor(normalized_positions[:, 1] * 127).astype(np.int32)  # 0-127
        
        # Pre-compute cell indices for each z-plane to avoid repeated computations
        for z_idx, z_level in enumerate(self.z_values):
            # Create mask for cells in this z-plane
            z_mask = (np.round(self.cell_positions[:, 2], decimals=3) == z_level)
            self.z_masks[z_idx] = z_mask
            
            # Store cell indices for this z-plane
            self.z_cell_indices[z_idx] = {
                'x': self.cell_x[z_mask],
                'y': self.cell_y[z_mask],
                'indices': np.where(z_mask)[0]  # Store the actual indices for faster lookup
            }
        
        # Implement timepoint-based separation
        # 1. Create blocks of timepoints
        # Size of each block - make sure it's larger than sequence length to avoid boundary issues
        block_size = self.seq_len * 2 # Double sequence length for safety
        
        # Check if we have enough timepoints for proper blocking
        if self.num_timepoints < block_size * 2:  # Need at least 2 blocks
            # Fall back to simple ratio splitting if data is too small
            print(f"Warning: Not enough timepoints for blocking. Using simple ratio split.")
            self._create_valid_starts_with_ratio_split(train_ratio, split)
        else:
            # Calculate number of complete blocks
            num_blocks = self.num_timepoints // block_size
            
            # Create block indices
            block_indices = np.arange(num_blocks)
            
            # Shuffle blocks
            np.random.seed(42)  # Fixed seed for reproducibility
            np.random.shuffle(block_indices)
            
            # Determine how many blocks to use for test set (approx. 1-train_ratio)
            # Ensure at least one block for testing
            num_test_blocks = max(1, int(num_blocks * (1 - train_ratio)))
            
            # Select test blocks
            test_block_indices = block_indices[:num_test_blocks]
            train_block_indices = block_indices[num_test_blocks:]
            
            # Create masks for timepoints
            test_timepoints = []
            train_timepoints = []
            
            # Assign timepoints to train/test based on blocks
            for block_idx in range(num_blocks):
                start_timepoint = block_idx * block_size
                end_timepoint = min((block_idx + 1) * block_size, self.num_timepoints)
                
                if block_idx in test_block_indices:
                    test_timepoints.extend(range(start_timepoint, end_timepoint))
                else:
                    train_timepoints.extend(range(start_timepoint, end_timepoint))
            
            # Handle any remaining timepoints (assign to training by default)
            if num_blocks * block_size < self.num_timepoints:
                remaining_timepoints = range(num_blocks * block_size, self.num_timepoints)
                train_timepoints.extend(remaining_timepoints)
            
            # Sort timepoints for easier processing
            test_timepoints.sort()
            train_timepoints.sort()
            
            # Find valid starting points that ensure a complete sequence stays 
            # within either training or test data
            valid_starts = []
            
            if split == 'train':
                # Process training timepoints, finding continuous regions
                if len(train_timepoints) > 0:
                    # Find continuous regions
                    regions = []
                    region_start = train_timepoints[0]
                    current_region = [region_start]
                    
                    for i in range(1, len(train_timepoints)):
                        # If continuous, add to current region
                        if train_timepoints[i] == train_timepoints[i-1] + 1:
                            current_region.append(train_timepoints[i])
                        else:
                            # Region ended, store it and start a new one
                            regions.append(current_region)
                            current_region = [train_timepoints[i]]
                    
                    # Add the last region
                    if current_region:
                        regions.append(current_region)
                    
                    # For each continuous region, find valid starting points
                    for region in regions:
                        # Only consider regions long enough for a complete sequence
                        if len(region) >= self.seq_len:
                            # Find valid starting points within this region
                            # A valid starting point can be any timepoint that leaves room for a full sequence
                            for t_idx, t in enumerate(region):
                                # Check if there are enough remaining timepoints in this region for a full sequence
                                if t_idx + self.seq_len <= len(region):
                                    # This is a valid starting point
                                    valid_starts.extend([(t, z) for z in range(self.num_z)])
            else:
                # Process test timepoints, finding continuous regions
                if len(test_timepoints) > 0:
                    # Find continuous regions
                    regions = []
                    region_start = test_timepoints[0]
                    current_region = [region_start]
                    
                    for i in range(1, len(test_timepoints)):
                        # If continuous, add to current region
                        if test_timepoints[i] == test_timepoints[i-1] + 1:
                            current_region.append(test_timepoints[i])
                        else:
                            # Region ended, store it and start a new one
                            regions.append(current_region)
                            current_region = [test_timepoints[i]]
                    
                    # Add the last region
                    if current_region:
                        regions.append(current_region)
                    
                    # For each continuous region, find valid starting points
                    for region in regions:
                        # Only consider regions long enough for a complete sequence
                        if len(region) >= self.seq_len:
                            # Find valid starting points within this region
                            # A valid starting point can be any timepoint that leaves room for a full sequence
                            for t_idx, t in enumerate(region):
                                # Check if there are enough remaining timepoints in this region for a full sequence
                                if t_idx + self.seq_len <= len(region):
                                    # This is a valid starting point
                                    valid_starts.extend([(t, z) for z in range(self.num_z)])
            
            # Shuffle valid starts (within their respective train/test sets)
            np.random.shuffle(valid_starts)
            self.valid_starts = valid_starts
            
            # Print detailed information about the split
            print(f"\nTimepoint-based split for {os.path.basename(h5_file_path)}:")
            print(f"Total timepoints: {self.num_timepoints}")
            print(f"Block size: {block_size}")
            print(f"Number of blocks: {num_blocks}")
            print(f"Test blocks: {num_test_blocks} ({len(test_timepoints)} timepoints)")
            print(f"Train blocks: {num_blocks - num_test_blocks} ({len(train_timepoints)} timepoints)")
            print(f"Number of continuous regions: {len(regions)}")
            print(f"Average region length: {sum(len(r) for r in regions) / len(regions) if regions else 0:.1f} timepoints")
        
        # Only print basic dataset info, not diagnostics
        print(f"\nDALI-Based Dataset {os.path.basename(h5_file_path)}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Sequence length: {seq_len}")
        print(f"Total sequences ({split}): {len(self.valid_starts)}")
        
        # Only print z-plane ordering in diagnostic mode
        if MEMORY_DIAGNOSTICS:
            print(f"Z-plane ordering: {self.z_values}")
    
    def _create_valid_starts_with_ratio_split(self, train_ratio, split):
        """Legacy method for creating valid starts using simple ratio-based splitting.
        Used as fallback when there are not enough timepoints for block-based splitting."""
        # Create all possible sequence starting points
        valid_starts = []
        for t in range(self.num_timepoints - self.seq_len + 1):
            # Allow sequences to start at any z-index
            valid_starts.extend([(t, z) for z in range(self.num_z)])
        
        # Split into train/test
        np.random.seed(42)
        np.random.shuffle(valid_starts)
        split_idx = int(len(valid_starts) * train_ratio)
        
        if split == 'train':
            self.valid_starts = valid_starts[:split_idx]
        else:
            self.valid_starts = valid_starts[split_idx:]
        
        print(f"Warning: Using legacy ratio-based split due to insufficient timepoints.")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __call__(self, sample_info):
        """Generate dense format sequences directly on CPU.
        
        This method generates sequences where:
        - The first sequence can start at any z-plane
        - Z-planes are processed in anatomical order (0, 1, 2, ...) for each timepoint
        - Each new timepoint after the first always starts at z=0
        - No wrapping occurs within a timepoint
        
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
                    print(f"Sample info: {sample_info}")
                    print(f"Sample info attributes: {dir(sample_info)}")
                # Use index 0 as fallback
                batch_size = 1
                indices = [0]
        
        # Create dense batch directly on CPU
        # Using uint8 to save memory since data is binary
        batch = np.zeros((batch_size, self.seq_len, 256, 128), dtype=np.uint8)
        
        # Open the HDF5 file for reading
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as f:
            spikes = f['spikes']  # Reference to the dataset
            
            # Process each sample in the batch
            for batch_idx, idx in enumerate(indices):
                # Ensure idx is within valid range
                idx = idx % len(self.valid_starts)
                
                # Get starting timepoint and z-plane from valid_starts
                t_start, z_start = self.valid_starts[idx]
                
                # Fill the sequence with frames
                seq_idx = 0
                t = t_start
                current_z = z_start  # Start at the specified z-index
                
                while seq_idx < self.seq_len:
                    # Process the remaining z-planes in the current timepoint
                    while current_z < self.num_z and seq_idx < self.seq_len:
                        # Get pre-computed cell indices for this z-plane
                        cell_indices = self.z_cell_indices[current_z]['indices']
                        
                        # Skip if no cells in this z-plane
                        if len(cell_indices) == 0:
                            seq_idx += 1
                            current_z += 1
                            continue
                        
                        # Get spikes for this timepoint and z-plane directly from HDF5 file
                        spikes_t = spikes[t][cell_indices]
                        
                        # Find active cells
                        active_cells = np.abs(spikes_t) > 1e-6
                        
                        if np.any(active_cells):  # Only process if there are active cells
                            active_x = self.z_cell_indices[current_z]['x'][active_cells]
                            active_y = self.z_cell_indices[current_z]['y'][active_cells]
                            
                            # Set active cells to 1 in the dense representation
                            for i in range(len(active_x)):
                                batch[batch_idx, seq_idx, active_x[i], active_y[i]] = 1
                        
                        seq_idx += 1
                        current_z += 1
                    
                    # Move to next timepoint
                    t += 1
                    
                    # Break if we've reached the end of available timepoints
                    if t >= self.num_timepoints:
                        break
                    
                    # For all subsequent timepoints, always start at z=0
                    current_z = 0
        
        if MEMORY_DIAGNOSTICS:
            batch_size_mb = batch.nbytes / 1e6
            print(f"H5Loader: Dense uint8 format: {batch_size_mb:.2f} MB")
            print(f"H5Loader call took {time.time() - start_time:.2f}s for batch of {batch_size}")
        
        return batch

def augment_z0_frames(batch, loader):
    """Apply z-index based augmentation to frames in the batch.
    
    This function adds marker patterns in the top-right corner of each frame:
    - For z-index 0, adds a 2×2 marker
    - For z-index 1, adds a 2×3 marker
    - For z-index 2, adds a 2×4 marker
    And so on, with the width increasing with the z-index
    
    Args:
        batch: Numpy array of shape (batch_size, seq_len, height, width) or (seq_len, height, width)
        loader: H5Loader instance that contains z-index tracking information
        
    Returns:
        Augmented batch with marker patterns
    """
    # Handle different input shapes (DALI might send single samples with shape [seq_len, h, w])
    input_was_3d = False
    if len(batch.shape) == 3:
        # Single sample case: [seq_len, height, width]
        input_was_3d = True
        seq_len, height, width = batch.shape
        batch_size = 1
        # Reshape to [1, seq_len, height, width] for consistent processing
        batch = batch.reshape(1, seq_len, height, width)
    else:
        # Batch case: [batch_size, seq_len, height, width]
        batch_size, seq_len, height, width = batch.shape
    
    # Clone the batch to avoid modifying the original
    augmented_batch = batch.copy()
    
    # Loop through all samples in the batch
    for b in range(batch_size):
        # For each sample, track the current z-index
        current_z = 0  # Start at z=0
        
        # Loop through all frames in the sequence
        for f in range(seq_len):
            # Calculate marker dimensions based on z-index
            marker_height = 2  # Always 2 pixels high
            marker_width = current_z + 2  # z-index + 2 pixels wide
            
            # Ensure marker fits within the frame
            marker_width = min(marker_width, width)
            
            # Set the top-right corner pixels to 1
            augmented_batch[b, f, 0:marker_height, width-marker_width:width] = 1
            
            # Print debug information if memory diagnostics is enabled
            if MEMORY_DIAGNOSTICS:
                print(f"Augmenting frame {f} with z-index {current_z}, marker size: {marker_height}×{marker_width}")
            
            # Update z-index for the next frame
            current_z += 1
            if current_z >= loader.num_z:
                current_z = 0  # Wrap back to z=0
    
    # If input was a single sample (3D), return in original shape
    if input_was_3d:
        return augmented_batch[0]
    
    return augmented_batch

@pipeline_def
def create_brain_data_pipeline(h5_loader, device="gpu"):
    """Create a DALI pipeline for brain sequence data using dense format from CPU."""
    # Generate indices for the batch
    indices = fn.random.uniform(range=(0, len(h5_loader)), dtype=types.INT64)
    
    # Load the dense data directly using external source
    sequences = fn.external_source(
        source=h5_loader,
        num_outputs=1,
        device="cpu",
        batch=False,
        parallel=True
    )
    
    # Make sure we're getting a single tensor, not a list
    sequences = sequences[0] if isinstance(sequences, list) else sequences
    
    # Apply augmentation before transferring to GPU
    # Use Python function to augment frames based on z-index
    sequences = fn.python_function(sequences, function=lambda x: augment_z0_frames(x, h5_loader))
    
    # Transfer to GPU and cast to float16/float32 for model processing
    if device == "gpu":
        sequences = sequences.gpu()
        # Cast to float16 for memory efficiency during further processing
        sequences = fn.cast(sequences, dtype=types.FLOAT16)
    
    return sequences


class DALIBrainDataLoader:
    """DALI-based data loader for brain sequences, optimized for GPU performance."""
    def __init__(self, h5_files, batch_size=128, seq_len=330, split='train', 
                 train_ratio=0.95, device_id=0, num_threads=2, gpu_prefetch=1,
                 seed=42, shuffle=True):
        # Reduce batch size to avoid memory issues
        self.original_batch_size = batch_size
        # Start with a smaller batch size
        self.batch_size = batch_size#min(64, batch_size)
        
        if MEMORY_DIAGNOSTICS:
            print(f"MEMORY OPTIMIZATION: Using reduced batch size of {self.batch_size} (requested: {batch_size})")
        
        self.device_id = device_id
        
        # Calculate pipeline batch size - use smaller batches per pipeline to reduce memory pressure
        # Divide the total batch size by the number of files, with a minimum of 8 (reduced from 16)
        self.pipeline_batch_size = max(8, self.batch_size // len(h5_files))
        
        if MEMORY_DIAGNOSTICS:
            print(f"Using pipeline batch size of {self.pipeline_batch_size} (total batch size: {self.batch_size})")
        
        # Create H5Loaders for each file
        self.h5_loaders = [H5Loader(f, seq_len, split, train_ratio) for f in h5_files]
        
        # Create a pipeline for each file
        self.pipelines = []
        for h5_loader in self.h5_loaders:
            pipe = create_brain_data_pipeline(
                h5_loader=h5_loader,
                device="gpu",
                batch_size=self.pipeline_batch_size,  # Use smaller batch size per pipeline
                num_threads=num_threads,
                device_id=device_id,
                seed=seed,
                # Further reduce prefetch depth to minimize memory usage
                prefetch_queue_depth=1,
                # Set memory stats tracking for optimization
                enable_memory_stats=True,
                # Use 'spawn' instead of 'fork' to avoid CUDA initialization issues
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
                    ["sequences"],
                    # Use PARTIAL for all pipelines to avoid dropping data
                    last_batch_policy=LastBatchPolicy.PARTIAL,
                    prepare_first_batch=True,
                    # Add auto_reset to True to automatically reset the iterator
                    auto_reset=True
                )
            )
        
        # Calculate total length
        self.total_length = sum(len(h5_loader) for h5_loader in self.h5_loaders)
        # Calculate steps per epoch based on pipeline batch sizes
        total_pipeline_batch_size = self.pipeline_batch_size * len(self.pipelines)
        self.steps_per_epoch = self.total_length // total_pipeline_batch_size
        
        # Print basic information always
        print(f"Created DALI data loader with {len(self.pipelines)} pipelines")
        print(f"Total sequences: {self.total_length}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        
        # Initialize iterator state
        self.current_iterator = 0
        self.current_iter = None
        self.accumulated_batch = None
        self.accumulated_count = 0
        
        # Print memory stats after initialization
        print_memory_stats("After DALI loader init:")
    
    def __iter__(self):
        """Return a combined iterator that yields from all pipelines in sequence."""
        self.current_iterator = 0
        self.current_iter = iter(self.iterators[0])
        self.accumulated_batch = None
        self.accumulated_count = 0
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
                    
                    return result
                else:
                    # Reset for next epoch and raise StopIteration
                    self.reset()
                    raise StopIteration
            
            # Try the next iterator
            self.current_iter = iter(self.iterators[self.current_iterator])
            return self.__next__()
    
    def __len__(self):
        return self.steps_per_epoch
    
    def reset(self):
        """Reset all iterators."""
        for iterator in self.iterators:
            iterator.reset()
        self.accumulated_batch = None
        self.accumulated_count = 0
        
        if MEMORY_DIAGNOSTICS:
            # Force garbage collection to free up memory
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_stats("After reset:") 