# Unified Spike Processing Pipeline

This script combines the functionality of three separate scripts into one unified pipeline:

1. **visualize_2018_spikes_cascade.py** - CASCADE spike detection from calcium traces
2. **prepare_dali_training_data.py** - Data preparation for training  
3. **preprocess_spike_data.py** - Grid conversion with z-plane augmentation

## Overview

The unified pipeline processes raw calcium fluorescence traces directly to final grid format while generating visualization PDFs and eliminating intermediate files and folders.

### Pipeline Steps

1. **Load Raw Data**: Reads calcium traces from `TimeSeries.h5` and cell positions from `data_full.mat`
2. **Compute ΔF/F**: Calculates ΔF/F using 8th percentile sliding window baseline
3. **CASCADE Inference**: Runs CASCADE model to detect spike probabilities
4. **Binary Spike Sampling**: Samples binary spikes from probabilities
5. **Grid Conversion**: Converts spikes to 256×128 grid format based on cell positions
6. **Z-plane Augmentation**: Adds z-index markers in top-right corners
7. **Train/Test Split**: Creates reproducible train/test splits using block-based approach
8. **Save Final Data**: Outputs grid data and metadata in HDF5 format
9. **Generate Visualizations**: Creates PDF plots showing ΔF/F, probabilities, and spikes

## Usage

### Basic Usage

```bash
python unified_spike_processing.py
```

This will process all subjects in `raw_trace_data_2018/` and output to `processed_spike_grids_2018/`.

### Command Line Arguments

```bash
python unified_spike_processing.py \
    --input_dir raw_trace_data_2018 \
    --output_dir processed_spike_grids_2018 \
    --num_neurons 10 \
    --batch_size 30000 \
    --split_ratio 0.95 \
    --workers 4 \
    --seed 42 \
    --cascade_model Global_EXC_2Hz_smoothing500ms \
    --skip subject_1,subject_2
```

### Arguments

- `--input_dir`: Directory containing raw calcium trace data (default: `raw_trace_data_2018`)
- `--output_dir`: Output directory for processed grid data (default: `processed_spike_grids_2018`) 
- `--num_neurons`: Number of neurons to visualize in PDFs (default: 10)
- `--batch_size`: Batch size for CASCADE processing (default: 30000)
- `--split_ratio`: Ratio of data to use for training (default: 0.95)
- `--workers`: Number of parallel workers (default: 1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--cascade_model`: CASCADE model type to use (default: `Global_EXC_2Hz_smoothing500ms`)
- `--skip`: Comma-separated list of subjects to skip (default: empty)

## Input Data Structure

The script expects input data in the following structure:

```
raw_trace_data_2018/
├── subject_1/
│   ├── data_full.mat      # Contains cell positions in CellXYZ field
│   └── TimeSeries.h5      # Contains calcium traces in CellResp dataset
├── subject_2/
│   ├── data_full.mat
│   └── TimeSeries.h5
└── ...
```

## Output Data Structure

The script produces the following output structure:

```
processed_spike_grids_2018/
├── subject_1/
│   ├── preaugmented_grids.h5      # Main grid data (T, Z, 256, 128)
│   ├── metadata.h5                # Subject metadata  
│   └── subject_1_visualization.pdf # Spike detection plots
├── subject_2/
│   ├── preaugmented_grids.h5
│   ├── metadata.h5
│   └── subject_2_visualization.pdf
├── combined_metadata.h5           # Combined metadata for all subjects
└── ...
```

### Output Files

#### `preaugmented_grids.h5`
Main data file containing:
- `grids`: (T, Z, 256, 128) uint8 array of grid data with z-plane augmentation
- `timepoint_indices`: (T,) int32 array of original timepoint indices  
- `is_train`: (T,) uint8 binary mask (1=train, 0=test)
- Attributes: `num_timepoints`, `num_z_planes`, `subject`, `cascade_model`, etc.

#### `metadata.h5`
Subject-specific metadata:
- `num_timepoints`: Total number of timepoints
- `num_z_planes`: Number of z-planes
- `z_values`: Unique z-coordinate values
- `train_timepoints`: Indices of training timepoints
- `test_timepoints`: Indices of test timepoints  
- `is_train`: Binary train/test mask

#### `subject_X_visualization.pdf`
Multi-page PDF with plots for randomly selected neurons showing:
- ΔF/F calcium trace
- CASCADE spike probabilities
- Binary spike train

#### `combined_metadata.h5`
Aggregated metadata for all processed subjects.

## Z-plane Augmentation

The script applies z-plane specific augmentation by adding marker patterns in the top-right corner:
- Z-plane 0: 2×2 marker
- Z-plane 1: 2×3 marker  
- Z-plane 2: 2×4 marker
- And so on...

This allows models to distinguish between different z-planes during training.

## Performance Features

- **Memory Efficient**: Processes data in chunks and cleans up memory after each subject
- **Parallel Processing**: Supports multi-worker parallel processing with `--workers` argument
- **Progress Tracking**: Uses tqdm progress bars for long-running operations
- **Resume Capability**: Skips already processed subjects automatically
- **Error Handling**: Continues processing other subjects if one fails

## Dependencies

- `numpy`
- `h5py` 
- `matplotlib`
- `scipy`
- `tqdm`
- `neuralib` (for CASCADE)
- `tensorflow` (for CASCADE backend)
- `psutil` (optional, for memory monitoring)

## Memory Requirements

Processing large datasets requires significant memory. Monitor memory usage and adjust `--batch_size` if needed. The script includes optional memory monitoring via `psutil`.

## Reproducibility

The script ensures reproducible results by:
- Setting random seeds for train/test splits and spike sampling
- Using deterministic algorithms where possible
- Saving all parameters and settings in output metadata

## Integration with Training

The output format is directly compatible with existing DALI-based training scripts. Use the `processed_spike_grids_2018/` directory as input to your training pipeline. 