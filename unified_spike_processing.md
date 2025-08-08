# Data Documentation: Spike Probability Processing Pipeline

## Overview

The `unified_spike_processing.py` script processes calcium imaging data through a complete pipeline that converts raw calcium traces into spike probability time series data. The output preserves the original neuron-level resolution by saving spike probabilities as (T, N) time series along with (N, 3) spatial position coordinates for each neuron.

## Pipeline Summary

```
Raw Calcium Traces → Baseline Correction → CASCADE Inference → Direct Probability Output
```

**Key Features:**
- CASCADE-only spike detection (no OASIS support)
- Configurable baseline correction (proper ΔF/F or baseline subtraction)
- Direct probability time series output (T, N) format
- Cell spatial positions preserved as (N, 3) coordinates
- YAML configuration system
- Continuous probability values only
- Configurable float16/float32 output data types
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
  return_to_original_rate: false             # Downsample neural data back to original rate after CASCADE

cascade:
  model_type: 'Global_EXC_2Hz_smoothing500ms'  # CASCADE model

output:
  dtype: 'float16'                           # Data type for spike probabilities and positions
  include_additional_data: true              # Include anat_stack, stimulus, behavior, eye data
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

#### Additional MATLAB Datasets  
- **MATLAB sampling rate**: Original fpsec value from data_full.mat
- **Anatomical stack**: 3D reference brain anatomy with memory requirements
- **Stimulus data**: Temporal stimulus condition codes with alignment info
- **Behavioral data**: Multi-variable behavioral measurements over time
- **Eye tracking data**: Eye position coordinates over time
- **Temporal alignment**: Verification that all temporal data matches calcium timepoints

#### Output Format Preview
- **Data type**: Storage format for probabilities and positions
- **Memory requirements**: Estimated storage size for time series and positions
- **Output dimensions**: Expected (T, N) and (N, 3) array sizes
- **Additional data inclusion**: Based on configuration settings

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
  Additional MATLAB datasets:
    MATLAB sampling rate (fpsec): 1.97 Hz
    Anatomical stack: (2048, 1116, 29) (uint16)
      Memory requirement: ~0.127 GB
    Stimulus data: (2880,) (uint8)
      Values: 0 to 3
    Behavioral data: (5, 2880) (float64)
      5 behavioral variables over 2880 timepoints
    Eye tracking data: (2, 2880) (float64)
      2 eye dimensions over 2880 timepoints
    Temporal alignment check:
      stimulus: 2880 timepoints (✓ vs calcium 2880)
      behavior: 2880 timepoints (✓ vs calcium 2880)
      eye: 2880 timepoints (✓ vs calcium 2880)
  Output format settings:
    Data type: float16
    Spike probabilities memory requirement: ~0.65 GB
    Cell positions memory requirement: ~0.00 GB  
    Total output memory requirement: ~0.65 GB
    Output format: Spike probabilities (T=17159, N=44891), Positions (N=44891, 3)
    Additional data inclusion: true
```

### Optimization Recommendations

Based on the analysis output, you can optimize processing parameters:

**Data Type Selection:**
- **float16**: Recommended for most use cases (50% memory savings, sufficient precision)  
- **float32**: Use for high-precision requirements or if numerical issues occur

**Sampling Rate Strategy:**
- **`return_to_original_rate: false`**: Use when you want higher temporal resolution in final output
- **`return_to_original_rate: true`**: Use when you want to preserve original experimental timing
- Anti-aliasing downsampling ensures no information loss when returning to original rate

**Memory Management:**
- Large datasets (> 2 GB): Reduce `batch_size` in configuration
- Multiple subjects: Process sequentially to avoid memory issues
- Consider using `test_run_neurons` for initial parameter testing
- Monitor spike probability memory requirements for very large datasets

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

**`spike_probabilities`** - Spike probability time series data
- **Shape**: `(T, N)`
  - `T` = Number of timepoints
  - `N` = Number of neurons
- **Data Type**: Configurable (default: `float16`)
- **Values**: Continuous spike probabilities `[0.0, 1.0]`
- **Compression**: gzip level 1
- **Chunking**: `(min(1000, T), min(100, N))` for efficient access

**`cell_positions`** - 3D spatial coordinates of neurons
- **Shape**: `(N, 3)`
  - `N` = Number of neurons
  - `3` = X, Y, Z coordinates
- **Data Type**: Configurable (default: `float16`)
- **Values**: Spatial coordinates in original units
- **Compression**: gzip level 1

**`timepoint_indices`** - Sequential timepoint indices
- **Shape**: `(T,)`
- **Data Type**: `int32`
- **Values**: `[0, 1, 2, ..., T-1]`

#### Metadata Datasets

**`num_timepoints`** - Total number of timepoints
- **Type**: Scalar integer

**`num_neurons`** - Total number of neurons
- **Type**: Scalar integer

**`original_sampling_rate_hz`** - Original sampling rate from MATLAB file
- **Type**: Scalar float (from fpsec field)

#### Additional Datasets (Optional)

The following datasets are included when `include_additional_data: true` in configuration:

**`anat_stack`** - 3D anatomical reference stack
- **Shape**: `(2048, 1116, 29)` (typical dimensions)
- **Data Type**: uint16
- **Purpose**: 3D brain anatomy reference for spatial context
- **Compression**: gzip level 1

**`stimulus_full`** - Stimulus condition labels over time
- **Shape**: `(T,)` - matches calcium timepoints after any interpolation  
- **Data Type**: uint8
- **Values**: Stimulus condition codes (typically 0-3)
- **Compression**: gzip level 1

**`behavior_full`** - Behavioral measurements over time
- **Shape**: `(5, T)` - 5 behavioral variables × timepoints
- **Data Type**: float64
- **Values**: Behavioral metrics (normalized 0-1 range)
- **Compression**: gzip level 1

**`eye_full`** - Eye tracking data over time
- **Shape**: `(2, T)` - 2 eye dimensions (X, Y) × timepoints
- **Data Type**: float64
- **Values**: Eye position coordinates
- **Compression**: gzip level 1

#### HDF5 Attributes

**Processing Information:**
- `subject`: Subject name (string)
- `data_source`: Always `'raw_calcium'`
- `cascade_model`: CASCADE model used (string)
- `calcium_dataset`: Dataset name used from TimeSeries.h5
- `spike_dtype`: Data type used for spike probabilities
- `position_dtype`: Data type used for cell positions
- `original_sampling_rate`: Original data sampling rate (Hz) from config
- `target_sampling_rate`: Target resampled rate (Hz) from config
- `effective_sampling_rate`: Actual sampling rate used for processing (Hz)
- `final_sampling_rate`: Final sampling rate of all output data (Hz)
- `matlab_fpsec`: Original MATLAB sampling rate from fpsec field (Hz)
- `return_to_original_rate`: Boolean indicating if neural data was downsampled back to original rate
- `includes_additional_data`: Boolean indicating if additional datasets are included

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

### Spatial Data Processing

1. **Cell Position Preservation**
   - Original spatial coordinates preserved without normalization
   - NaN values in positions replaced with zeros
   - No spatial discretization or mapping performed
   - Full-resolution position data maintained as (N, 3) array

### Temporal Processing

1. **Sampling Rate Conversion**
   
   **Two processing modes based on `return_to_original_rate` setting:**
   
   **Mode A: `return_to_original_rate: false` (default)**
   - **Neural data**: Smooth interpolation (PCHIP) to target sampling rate
   - **Non-neural data**: Hold interpolation (zero-order hold) to match neural data
   - Final output: All data at target sampling rate
   
   **Mode B: `return_to_original_rate: true`**  
   - **Neural data**: Upsampled to target rate → CASCADE processing → Anti-aliasing downsampled back to original rate
   - **Non-neural data**: No interpolation (stays at original rate)
   - Final output: All data at original sampling rate
   - Downsampling uses `scipy.signal.resample_poly` with proper anti-aliasing filter
   - Low-pass cutoff: ≤ 0.9 × (original_rate/2) Hz to prevent aliasing

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
    spike_probs = f['spike_probabilities'][:]  # Shape: (T, N)
    positions = f['cell_positions'][:]         # Shape: (N, 3)  
    timepoints = f['timepoint_indices'][:]     # Shape: (T,)
    
    # Metadata
    num_timepoints = f['num_timepoints'][()]
    num_neurons = f['num_neurons'][()]
    
    # Processing info
    subject = f.attrs['subject']
    cascade_model = f.attrs['cascade_model']
    is_raw = f.attrs['is_raw']

# Access spike probabilities for specific neuron over time
neuron_id = 100
neuron_time_series = spike_probs[:, neuron_id]  # Shape: (T,)
neuron_position = positions[neuron_id, :]       # Shape: (3,)

# Access all neurons at specific timepoint
t = 100
all_probs_t = spike_probs[t, :]  # Shape: (N,)

# Find active neurons at timepoint
active_mask = all_probs_t > 0.01  # Threshold for "active"
active_neurons = np.where(active_mask)[0]
active_positions = positions[active_neurons, :]

# Get spatial bounds of all neurons
min_coords = positions.min(axis=0)  # [x_min, y_min, z_min]
max_coords = positions.max(axis=0)  # [x_max, y_max, z_max]

# Access additional datasets (if included)
if f.attrs.get('includes_additional_data', False):
    # Anatomical reference
    anat_stack = f['anat_stack'][:]  # Shape: (2048, 1116, 29) typical
    
    # Temporal data aligned with spike probabilities
    stimulus = f['stimulus_full'][:]      # Shape: (T,)
    behavior = f['behavior_full'][:]      # Shape: (5, T)
    eye_data = f['eye_full'][:]           # Shape: (2, T)
    
    # Original sampling rate from MATLAB
    original_rate = f['original_sampling_rate_hz'][()]
    
    # Check processing mode
    return_to_original = f.attrs.get('return_to_original_rate', False)
    final_rate = f.attrs.get('final_sampling_rate', original_rate)
    
    print(f"Processing mode: {'Original rate' if return_to_original else 'Target rate'}")
    print(f"Final sampling rate: {final_rate} Hz")
    print(f"Stimulus at timepoint {t}: {stimulus[t]}")
    print(f"Behavior variables at timepoint {t}: {behavior[:, t]}")
    print(f"Eye position at timepoint {t}: {eye_data[:, t]}")
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

2. **Parameter optimization**: Use raw data analysis to check memory requirements and data types
   ```bash
   # Analyze data to determine optimal data types and memory usage
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

- **2025-01-18**: Added `return_to_original_rate` configuration option with anti-aliasing downsampling for neural data
- **2025-01-18**: Changed interpolation for non-neural data (stimulus, behavior, eye) to zero-order hold to preserve discrete values
- **2025-01-18**: Enhanced neuron filtering using IX_inval_anat with absIX mapping and added support for additional datasets (anat_stack, stimulus_full, behavior_full, eye_full)
- **2025-01-18**: Removed volumetric processing - now saves spike probabilities as (T, N) time series and cell positions as (N, 3) coordinates directly
- **2025-07-18**: Initial 3D volumetric processing pipeline
- **2025-07-18**: Added proper ΔF/F support and reorganized configuration
- **2025-07-18**: Simplified file structure and merged metadata
- **2025-07-18**: Added Poisson spike rate to probability conversion
- **2025-07-18**: Added `--raw_data_info` flag for data analysis without processing 