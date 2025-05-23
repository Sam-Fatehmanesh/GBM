# Unified Spike Processing Pipeline (COO Sparse Format)

This script is a memory-efficient version of the unified spike processing pipeline that saves grid data in COO (Coordinate) sparse format instead of dense arrays.

## Key Differences from Dense Version

### **Memory Efficiency**
- **Sparse Storage**: Only stores coordinates of active spikes rather than full dense grids
- **Typical Savings**: 95-99% memory reduction for neural spike data
- **Scalability**: Can handle much larger datasets without memory issues

### **COO Format Structure**
Instead of dense (T, Z, H, W) grids, data is stored as:
- `spike_coords`: (N_spikes, 4) array with [timepoint, z_plane, x, y] coordinates
- `spike_values`: (N_spikes,) array with spike values (all 1s for binary spikes)
- `grid_shape`: (4,) array with [T, Z, H, W] dimensions

## Usage

### Basic Usage
```bash
python unified_spike_processing_coo.py
```

### Command Line Arguments
Same as the dense version, but outputs to `processed_spike_grids_coo_2018/` by default:

```bash
python unified_spike_processing_coo.py \
    --input_dir raw_trace_data_2018 \
    --output_dir processed_spike_grids_coo_2018 \
    --workers 4 \
    --batch_size 30000
```

## Output Data Structure

```
processed_spike_grids_coo_2018/
├── subject_1/
│   ├── preaugmented_grids_coo.h5      # Main COO sparse data
│   ├── metadata.h5                    # Subject metadata with sparsity stats
│   └── subject_1_visualization.pdf    # Spike detection plots
├── subject_2/
│   ├── preaugmented_grids_coo.h5
│   ├── metadata.h5
│   └── subject_2_visualization.pdf
├── combined_metadata.h5               # Combined metadata with global sparsity stats
└── ...
```

### Output Files

#### `preaugmented_grids_coo.h5`
Main sparse data file containing:
- `spike_coords`: (N_spikes, 4) int32 array with [t, z, x, y] coordinates
- `spike_values`: (N_spikes,) uint8 array with spike values (all 1s)
- `grid_shape`: (4,) int32 array with [T, Z, H, W] dimensions
- `timepoint_indices`: (T,) int32 array of original timepoint indices
- `is_train`: (T,) uint8 binary mask (1=train, 0=test)
- Attributes: `total_spikes`, `sparsity_percent`, `format='COO_sparse'`, etc.

#### Enhanced Metadata
- All original metadata fields
- `total_spikes`: Number of active pixels across all timepoints/z-planes
- `sparsity_percent`: Percentage of pixels that are active
- Memory savings statistics

## Converting Between Formats

### COO to Dense (for verification/testing)
```python
def coo_to_dense_grid(spike_coords, spike_values, grid_shape):
    """Convert COO sparse format back to dense grid format."""
    T, Z, H, W = grid_shape
    dense_grid = np.zeros((T, Z, H, W), dtype=np.uint8)
    
    for i in range(len(spike_coords)):
        t, z, x, y = spike_coords[i]
        dense_grid[t, z, x, y] = spike_values[i]
    
    return dense_grid
```

### Loading COO Data
```python
import h5py
import numpy as np

# Load COO sparse data
with h5py.File('subject_1/preaugmented_grids_coo.h5', 'r') as f:
    spike_coords = f['spike_coords'][:]  # (N_spikes, 4)
    spike_values = f['spike_values'][:]  # (N_spikes,)
    grid_shape = f['grid_shape'][:]      # (4,)
    is_train = f['is_train'][:]          # (T,)
    
    # Get sparsity info
    total_spikes = f.attrs['total_spikes']
    sparsity_percent = f.attrs['sparsity_percent']
    
    print(f"Total spikes: {total_spikes:,}")
    print(f"Sparsity: {sparsity_percent:.4f}%")
```

## Memory Comparison

For typical neural spike data:

| Format | Memory Usage | Typical Sparsity | Memory Savings |
|--------|--------------|------------------|----------------|
| Dense  | 100%         | 0.1-1%          | 0%             |
| COO    | ~1-5%        | 0.1-1%          | 95-99%         |

**Example**: A dataset with 10,000 timepoints, 20 z-planes, 256×128 grids:
- Dense: ~6.5 GB
- COO (0.5% sparsity): ~33 MB
- **Savings**: 99.5% memory reduction

## Integration with Training Pipelines

### Option 1: Convert to Dense During Loading
```python
# Load COO data and convert to dense for existing pipelines
def load_coo_as_dense(filepath):
    with h5py.File(filepath, 'r') as f:
        spike_coords = f['spike_coords'][:]
        spike_values = f['spike_values'][:]
        grid_shape = f['grid_shape'][:]
    
    return coo_to_dense_grid(spike_coords, spike_values, grid_shape)
```

### Option 2: Native Sparse Training (Recommended)
Modify training pipelines to work directly with COO format:
- Use PyTorch sparse tensors: `torch.sparse_coo_tensor()`
- Implement sparse-aware data loaders
- Use sparse-compatible loss functions

## Performance Benefits

### **Storage**
- File sizes reduced by 95-99%
- Faster I/O due to smaller files
- Reduced disk space requirements

### **Memory**
- Lower RAM usage during processing
- Can process larger datasets
- Reduced memory transfer overhead

### **Processing**
- Sparse operations can be faster for very sparse data
- Better cache locality for active pixels
- Potential for specialized sparse algorithms

## Sparsity Statistics

The script automatically computes and reports:
- **Per-subject sparsity**: Percentage of active pixels per subject
- **Overall sparsity**: Aggregate sparsity across all subjects
- **Memory savings**: Estimated memory reduction vs dense format

Example output:
```
Subject: subject_1
→ Sparsity: 0.3421%
→ Memory savings vs dense: ~99.7%

Summary:
  Total spikes: 2,847,392
  Overall sparsity: 0.4125%
  Memory savings vs dense: ~99.6%
```

## Use Cases

### **Ideal for:**
- Large-scale neural datasets
- Long time series with sparse activity
- Memory-constrained environments
- Storage-efficient archival

### **Consider Dense Format for:**
- Very dense spike patterns (>10% sparsity)
- Algorithms requiring dense operations
- Small datasets where memory isn't a concern

## Dependencies

Same as the dense version:
- `numpy`, `h5py`, `matplotlib`, `scipy`, `tqdm`
- `neuralib` (for CASCADE)
- `tensorflow` (for CASCADE backend)
- `psutil` (optional, for memory monitoring)

## Backward Compatibility

The COO format includes a utility function to convert back to dense format, ensuring compatibility with existing analysis pipelines while providing the memory benefits of sparse storage. 