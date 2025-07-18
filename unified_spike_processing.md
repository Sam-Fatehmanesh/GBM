# Data Documentation: 3D Volumetric Spike Processing Pipeline

## Overview

The `unified_spike_processing.py` script processes calcium imaging data through a complete pipeline that converts raw calcium traces into 3D volumetric representations of neural activity. Each timepoint becomes a single 3D volume with continuous spike probability values derived from CASCADE inference.

## Pipeline Summary

```
Raw Calcium Traces → Baseline Correction → CASCADE Inference → 3D Volumetric Representation
```

**Key Features:**
- CASCADE-only spike detection (no OASIS support)
- Configurable baseline correction (proper ΔF/F or baseline subtraction)
- 3D volumetric representation of neural activity
- YAML configuration system
- Continuous probability values only
- Memory-efficient processing with batch operations

---

## Input Data Format

### Directory Structure Expected

```
{input_dir}/
├── subject_001/
│   ├── data_full.mat
│   └── TimeSeries.h5
├── subject_002/
│   ├── data_full.mat
│   └── TimeSeries.h5
└── ...
```

### Required Files per Subject

#### 1. `data_full.mat` (MATLAB file)
**Required Contents:**
- `data[0,0]['CellXYZ']` - Cell position coordinates
  - **Format**: (N, 3) array where N = number of cells
  - **Columns**: [X, Y, Z] spatial coordinates
  - **Data Type**: Numeric (float/double)
  - **Units**: Arbitrary spatial units (will be normalized)

**Optional Contents:**
- `data[0,0]['IX_inval_anat']` - Invalid anatomical indices
  - **Format**: Array of 1-based indices of cells to exclude
  - **Purpose**: Anatomical filtering (converted to 0-based internally)

#### 2. `TimeSeries.h5` (HDF5 file)
**Required Dataset:**
- `{calcium_dataset}` - Calcium fluorescence traces
  - **Format**: (T, N) or (N, T) array where T = timepoints, N = cells
  - **Data Type**: Numeric (float32/float64)
  - **Units**: Fluorescence values (arbitrary units)
  - **Default Name**: `'CellResp'` (configurable via `calcium_dataset`)

**Common Dataset Names:**
- `'CellResp'` - Raw calcium fluorescence traces
- `'CellRespZ'` - Pre-processed calcium traces (e.g., z-scored)
- Custom names as specified in configuration

### Data Validation and Filtering

The script automatically:
1. **Validates dimensions** - Ensures calcium data matches cell count
2. **Filters invalid cells** - Removes cells with:
   - Invalid anatomical indices (if `IX_inval_anat` present)
   - NaN coordinates in position data
3. **Handles dimension mismatches** - Transposes data if needed
4. **Reports filtering statistics** - Shows how many cells were removed

---

## Configuration Format

### YAML Configuration Structure

```yaml
data:
  input_dir: 'raw_trace_data_2018'           # Input directory path
  output_dir: 'processed_spike_voxels_2018'  # Output directory path
  skip_subjects: []                          # List of subjects to skip
  test_run_neurons: null                     # Limit neurons for testing
  calcium_dataset: 'CellResp'                # Dataset name in TimeSeries.h5
  is_raw: true                               # Apply proper ΔF/F: (F-F0)/F0
  apply_baseline_subtraction: false          # Apply baseline subtraction: F-F0
  window_length: 30.0                        # Baseline window size (seconds)
  baseline_percentile: 8                     # Percentile for baseline computation

processing:
  num_neurons_viz: 10                        # Neurons to visualize in PDF
  batch_size: 5000                           # CASCADE batch size
  workers: 1                                 # Number of parallel workers
  seed: 42                                   # Random seed
  original_sampling_rate: 2.73               # Original sampling rate (Hz)
  target_sampling_rate: 2.5                  # Target sampling rate (Hz)

cascade:
  model_type: 'Global_EXC_2Hz_smoothing500ms'  # CASCADE model

volumization:
  volume_shape: [64, 64, 32]                 # [X, Y, Z] voxel dimensions
  dtype: 'float16'                           # Data type for volumes
```

### Processing Logic Options

| `is_raw` | `apply_baseline_subtraction` | **Result** |
|----------|------------------------------|------------|
| `true`   | `true` or `false`            | ΔF/F: `(F - F0) / F0` |
| `false`  | `true`                       | Baseline subtraction: `F - F0` |
| `false`  | `false`                      | No processing (original data) |

---

## Raw Data Analysis Mode

The `--raw_data_info` flag allows you to analyze the raw data structure and characteristics without running the full processing pipeline. This is useful for:

- **Data quality assessment** - Check for missing files, dimension mismatches, or invalid data
- **Parameter optimization** - Determine optimal volume dimensions and memory requirements
- **Troubleshooting** - Identify issues before running expensive processing

### Information Provided

For each subject found in the input directory, the analysis reports:

#### File Structure
- Presence of required files (`data_full.mat`, `TimeSeries.h5`)
- Available datasets in `TimeSeries.h5`
- Verification of configured `calcium_dataset` name

#### Cell Position Data
- **Total neurons**: Count of cells with position data
- **Valid/Invalid coordinates**: Cells with NaN vs. valid spatial coordinates
- **Spatial bounds**: Min/max coordinates in X, Y, Z dimensions
- **Spatial statistics**: Mean and standard deviation of cell positions
- **Invalid anatomical indices**: Count of cells marked for exclusion

#### Calcium Trace Data
- **Dataset shape**: Dimensions of calcium fluorescence data
- **Data type**: Storage format (float32, float64, etc.)
- **Format detection**: Whether data is (T, N) or (N, T) format
- **Temporal information**: Duration and sampling rate calculations
- **Memory requirements**: Estimated RAM usage during processing
- **Data statistics**: Min/max/mean/std of sample data
- **Data quality**: Detection of NaN or infinity values

#### Volumization Preview
- **Target volume shape**: Configured 3D volume dimensions
- **Volume memory requirements**: Estimated storage size
- **Spatial resolution**: Physical units per voxel
- **Neuron density**: Average neurons per voxel
- **Optimization warnings**: Suggestions for volume dimension adjustments

### Example Output

```
================================================================================
RAW DATA ANALYSIS
================================================================================
Found 5 subjects in raw_trace_data_2018
Skipping subjects: []

Subject: subject_001
----------------------------------------
  Cell positions shape: (45123, 3)
  Valid coordinate neurons: 44891
  Invalid coordinate neurons: 232
  Spatial bounds:
    X: 12.50 to 487.30 (range: 474.80)
    Y: 8.75 to 512.44 (range: 503.69)
    Z: 0.00 to 79.50 (range: 79.50)
  Spatial statistics:
    Mean: X=250.15, Y=260.87, Z=39.75
    Std:  X=137.42, Y=145.23, Z=22.95
  Invalid anatomical indices: 1247 neurons
  Available datasets in TimeSeries.h5: ['CellResp', 'CellRespZ', 'TimeStamps']
  Calcium dataset 'CellResp':
    Shape: (45123, 18750)
    Data type: float32
    Format: (N, T) - will be transposed to (T, N)
    Interpreted as: 18750 timepoints, 45123 neurons
    Duration: 6863.64 seconds at 2.73 Hz
    After resampling: 17159 timepoints at 2.5 Hz (6863.60 seconds)
    Memory requirement: ~3.40 GB (float32)
    Sample statistics (first 1000 neurons, 1000 timepoints):
      Min: 125.2341
      Max: 8943.7754
      Mean: 1247.3829
      Std: 891.2456
  Volumization settings:
    Target volume shape: [64, 64, 32]
    Data type: float16
    Volume memory requirement: ~0.70 GB
    Spatial resolution: X=7.42, Y=7.87, Z=2.48 units/voxel
    Average neurons per voxel: 3.44
```

### Optimization Recommendations

Based on the analysis output, you can optimize processing parameters:

**Volume Dimensions:**
- **Too sparse** (< 0.1 neurons/voxel): Consider smaller volume dimensions
- **Too dense** (> 10 neurons/voxel): Consider larger volume dimensions
- **Optimal**: 1-5 neurons per voxel for good spatial resolution

**Memory Management:**
- Large datasets (> 2 GB): Reduce `batch_size` in configuration
- Multiple subjects: Process sequentially to avoid memory issues
- Consider using `test_run_neurons` for initial parameter testing

**Data Quality:**
- NaN/inf values: May require data preprocessing
- Dimension mismatches: Check data loading and transposition logic
- Missing datasets: Verify `calcium_dataset` name in configuration

---

## Output Data Format

### Directory Structure Produced

```
{output_dir}/
├── subject_001.h5
├── subject_001_visualization.pdf
├── subject_002.h5
├── subject_002_visualization.pdf
└── ...
```

### HDF5 File Structure: `{subject_name}.h5`

#### Main Datasets

**`volumes`** - 3D volumetric neural activity data
- **Shape**: `(T, X, Y, Z)`
  - `T` = Number of timepoints
  - `X, Y, Z` = Volume dimensions from config (e.g., 64×64×32)
- **Data Type**: Configurable (default: `float16`)
- **Values**: Continuous spike probabilities `[0.0, 1.0]`
- **Compression**: gzip level 1
- **Chunking**: `(1, X, Y, Z)` for efficient timepoint access

**`timepoint_indices`** - Sequential timepoint indices
- **Shape**: `(T,)`
- **Data Type**: `int32`
- **Values**: `[0, 1, 2, ..., T-1]`

#### Metadata Datasets

**`num_timepoints`** - Total number of timepoints
- **Type**: Scalar integer

**`volume_shape`** - Volume dimensions
- **Shape**: `(3,)`
- **Values**: `[X, Y, Z]` dimensions

#### HDF5 Attributes

**Processing Information:**
- `subject`: Subject name (string)
- `data_source`: Always `'raw_calcium'`
- `cascade_model`: CASCADE model used (string)
- `calcium_dataset`: Dataset name used from TimeSeries.h5
- `dtype`: Data type used for volumes

**Baseline Correction Parameters:**
- `is_raw`: Whether ΔF/F was computed (boolean)
- `apply_baseline_subtraction`: Whether baseline subtraction was applied (boolean)
- `window_length`: Baseline window size in seconds (float)
- `baseline_percentile`: Percentile used for baseline (integer)

### PDF Visualization File: `{subject_name}_visualization.pdf`

**Contents:**
- Multiple pages (one per visualized neuron)
- Each page contains 2 subplots:
  1. **Raw Calcium Trace** - Original fluorescence signal over time
  2. **Spike Probabilities** - CASCADE-inferred probabilities over time
- Default: 10 randomly selected neurons (configurable)

---

## Data Processing Details

### Spatial Mapping

1. **Cell Position Normalization**
   - Cell coordinates normalized to `[0, 1]` range
   - Handles cases where spatial range is zero
   - Maps to volume indices using floor operation

2. **Volume Element Assignment**
   - Each cell maps to discrete volume element `[x, y, z]`
   - Multiple cells mapping to same element have probabilities **summed**
   - Empty volume elements contain `0.0`

### Temporal Processing

1. **Sampling Rate Conversion**
   - Linear interpolation between original and target sampling rates
   - Maintains temporal relationships
   - Uses scipy.interpolate.interp1d

2. **Baseline Correction**
   - Causal sliding window (past + current frames only)
   - Percentile-based baseline computation
   - Choice between ΔF/F or baseline subtraction

3. **CASCADE Inference**
   - Batched processing to avoid memory issues
   - Probability clipping to `[0, 1]` range
   - NaN/infinity handling with graceful fallback

### Memory Management

- **Batch Processing**: Large datasets processed in configurable batches
- **Memory Cleanup**: Explicit garbage collection between processing steps
- **Data Type Optimization**: Float16 storage for space efficiency
- **Streaming**: Timepoint-by-timepoint processing where possible

---

## Usage Examples

### Basic Usage
```bash
# Create default configuration
python unified_spike_processing.py --create-default-config config.yaml

# Analyze raw data information without processing
python unified_spike_processing.py --config config.yaml --raw_data_info

# Edit config.yaml as needed, then run full processing
python unified_spike_processing.py --config config.yaml
```

### Data Access Examples

```python
import h5py
import numpy as np

# Load processed data
with h5py.File('subject_001.h5', 'r') as f:
    # Main data
    volumes = f['volumes'][:]  # Shape: (T, X, Y, Z)
    timepoints = f['timepoint_indices'][:]
    
    # Metadata
    num_timepoints = f['num_timepoints'][()]
    volume_shape = f['volume_shape'][:]
    
    # Processing info
    subject = f.attrs['subject']
    cascade_model = f.attrs['cascade_model']
    is_raw = f.attrs['is_raw']

# Access specific timepoint
t = 100
brain_volume_t = volumes[t, :, :, :]  # Shape: (X, Y, Z)

# Access specific location over time
x, y, z = 32, 32, 16
time_series = volumes[:, x, y, z]  # Shape: (T,)

# Find active volume elements at timepoint
active_mask = volumes[t] > 0.01  # Threshold for "active"
active_coords = np.where(active_mask)
```

---

## Command Line Options

The script supports several command line options:

```bash
python unified_spike_processing.py [OPTIONS]

Options:
  --config PATH                 Path to YAML configuration file (required for processing)
  --create-default-config PATH  Create default configuration file at specified path and exit
  --raw_data_info              Analyze and display raw data information without processing
  -h, --help                   Show help message and exit
```

### Usage Patterns

1. **First-time setup**: Create default configuration and analyze data
   ```bash
   python unified_spike_processing.py --create-default-config config.yaml
   python unified_spike_processing.py --config config.yaml --raw_data_info
   ```

2. **Parameter optimization**: Use raw data analysis to optimize volume dimensions
   ```bash
   # Analyze data to determine optimal volume shape
   python unified_spike_processing.py --config config.yaml --raw_data_info
   
   # Edit config.yaml based on analysis recommendations
   # Run full processing
   python unified_spike_processing.py --config config.yaml
   ```

3. **Troubleshooting**: Check data quality before processing
   ```bash
   # Quick data validation
   python unified_spike_processing.py --config config.yaml --raw_data_info
   ```

---

## Technical Requirements

### Dependencies
- `numpy`, `h5py`, `scipy`, `matplotlib`, `tqdm`, `yaml`
- `neuralib` (for CASCADE inference)
- `tensorflow` (CASCADE backend)

### Hardware Considerations
- **Memory**: Scales with number of neurons and timepoints
- **GPU**: Optional for CASCADE inference acceleration
- **Storage**: Output files typically 10-100MB per subject

### Performance Characteristics
- **Processing Speed**: ~1-10 subjects per hour (depends on data size)
- **Memory Usage**: Peak usage during CASCADE inference
- **Disk I/O**: Optimized with HDF5 compression and chunking

---

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   ValueError: Dataset 'CellRespZ' not found in TimeSeries.h5
   ```
   **Solution**: Check `calcium_dataset` in config matches HDF5 dataset name

2. **Dimension Mismatch**
   ```
   Warning: Calcium shape doesn't match cell count
   ```
   **Solution**: Script attempts auto-correction; check data loading

3. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce `batch_size` in configuration

4. **CASCADE Not Available**
   ```
   ImportError: CASCADE not available
   ```
   **Solution**: Install neuralib package

### Data Quality Checks

The script automatically reports:
- Number of cells loaded vs. filtered
- Timepoint count and sampling rate conversion
- Volume dimensions and memory usage
- Processing method applied

---

## Version History

- **2025-07-18**: Initial 3D volumetric processing pipeline
- **2025-07-18**: Added proper ΔF/F support and reorganized configuration
- **2025-07-18**: Simplified file structure and merged metadata
- **2025-07-18**: Added Poisson spike rate to probability conversion
- **2025-07-18**: Added `--raw_data_info` flag for data analysis without processing 