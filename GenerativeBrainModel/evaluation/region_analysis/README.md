# Region-Specific Performance Analysis for Generative Brain Model (GBM)

This module provides comprehensive tools for evaluating GBM performance across different brain regions using zebrafish brain masks. The system supports both next-frame and long-horizon prediction evaluation with detailed visualizations and metrics.

## Overview

The region evaluation system is designed to provide detailed insights into how well the GBM model performs in different anatomical regions of the zebrafish brain. It handles the complexities of:

- **Z-marker detection**: Identifying frame boundaries using 2×2 top-right markers
- **Brain volume grouping**: Organizing frames into complete brain volumes
- **Region-specific metrics**: Calculating precision, recall, and F1 scores per brain region
- **Temporal analysis**: Tracking performance degradation over long prediction horizons
- **Rich visualizations**: Generating publication-ready plots and heatmaps

## Architecture

### Core Components

1. **TestDataLoader** (`test_data_loader.py`)
   - Loads test data and predictions from experiment HDF5 files
   - Separates model input data (sampled probabilities) from evaluation data (binary)
   - Handles both next-frame and long-horizon data preparation

2. **FrameProcessor** (`frame_processor.py`)
   - Detects z-zero frames using 2×2 markers with clear adjacent areas
   - Extracts valid 330-frame chunks with strict boundary validation
   - Provides chunk validation and statistics

3. **VolumeGrouper** (`volume_grouper.py`)
   - Groups frame sequences into complete brain volumes (z=0 to next z=0)
   - Validates volume completeness and consistency
   - Handles overlap between chunks and volumes

4. **PredictionRunner** (`prediction_runner.py`)
   - Loads trained GBM models from experiment checkpoints
   - Runs next-frame predictions (typically using pre-computed results)
   - Performs autoregressive long-horizon predictions
   - Handles probability sampling and thresholding

5. **RegionPerformanceEvaluator** (`region_performance_evaluator.py`)
   - Loads and processes zebrafish brain region masks
   - Calculates region-specific performance metrics (precision, recall, F1)
   - Supports both aggregate and temporal evaluation modes
   - Provides summary statistics across regions

6. **RegionPerformanceVisualizer** (`region_performance_visualizer.py`)
   - Generates bar plots for next-frame performance comparison
   - Creates heatmaps for long-horizon temporal performance
   - Produces temporal trend plots with error bars
   - Creates comprehensive dashboards combining all analyses

## Usage

### Basic Usage

```bash
# Evaluate both next-frame and long-horizon performance
python GenerativeBrainModel/scripts/evaluate_region_performance.py \
    --experiment_path experiments/gbm/20250523_112455 \
    --masks_path masks \
    --device cuda

# Evaluate only next-frame performance
python GenerativeBrainModel/scripts/evaluate_region_performance.py \
    --experiment_path experiments/gbm/20250523_112455 \
    --skip_long_horizon

# Custom configuration
python GenerativeBrainModel/scripts/evaluate_region_performance.py \
    --experiment_path experiments/gbm/20250523_112455 \
    --target_shape 30 256 128 \
    --threshold 0.6 \
    --time_window 15 \
    --chunk_size 330
```

### Command Line Arguments

**Required:**
- `--experiment_path`: Path to experiment directory containing `test_data_and_predictions.h5`

**Optional:**
- `--masks_path`: Path to brain region masks directory (default: `masks`)
- `--output_dir`: Base output directory (default: `experiments/region_eval`)
- `--target_shape`: Target shape for mask downsampling (default: `30 256 128`)
- `--threshold`: Threshold for binary conversion (default: `0.5`)
- `--chunk_size`: Expected chunk size in frames (default: `330`)
- `--time_window`: Time window for temporal evaluation (default: `10`)
- `--device`: Computation device (default: `cuda`)
- `--skip_next_frame`: Skip next-frame evaluation
- `--skip_long_horizon`: Skip long-horizon evaluation

## Input Requirements

### Experiment Directory Structure

Your experiment directory must contain:

```
experiment_directory/
├── test_data_and_predictions.h5    # Required: Test data and model predictions
├── checkpoints/                    # Optional: For long-horizon evaluation
│   ├── best_model.pt              # Preferred checkpoint
│   └── *.pt                       # Other checkpoints
└── ...
```

### HDF5 File Structure

The `test_data_and_predictions.h5` file should contain:

```
test_data_and_predictions.h5
├── predictions/
│   └── next_frame              # Shape: (T-1, H, W) - next frame predictions
├── test_data/
│   ├── sampled_probabilities   # Shape: (T, H, W) - model input data
│   └── binary_data            # Shape: (T, H, W) - ground truth for evaluation
└── metadata/
    └── (experiment metadata)
```

### Brain Masks

Brain region masks should be stored as TIF files in the masks directory:

```
masks/
├── prosencephalon.tif
├── mesencephalon.tif
├── rhombencephalon.tif
├── region_A.tif
├── region_B.tif
└── ...
```

Each mask file should be a 3D boolean array with the same spatial dimensions as your brain data.

## Output Structure

The evaluation generates a timestamped output directory:

```
experiments/region_eval/20240315_143022/
├── config.json                           # Evaluation configuration
├── evaluation_summary.json              # High-level summary
├── logs/
│   └── evaluation.log                   # Detailed logging
├── plots/                               # All visualizations
│   ├── next_frame_performance.png       # Bar plot: next-frame metrics by region
│   ├── region_comparison_f1.png         # Horizontal bar: regions ranked by F1
│   ├── long_horizon_heatmap_f1.png      # Heatmap: F1 scores over time×regions
│   ├── long_horizon_heatmap_precision.png
│   ├── long_horizon_heatmap_recall.png
│   ├── temporal_trends.png              # Line plot: metrics over time
│   ├── performance_dashboard.png        # Comprehensive 6-panel dashboard
│   └── next_frame_performance_table.csv # CSV table of all metrics
└── results/                             # Detailed numerical results
    ├── next_frame_results.json          # Per-region next-frame metrics
    ├── next_frame_summary.json          # Summary statistics
    ├── long_horizon_results.json        # Temporal results per region
    ├── long_horizon_summary.json        # Temporal summary statistics
    └── processing_info.json             # Frame processing details
```

## Evaluation Types

### 1. Next-Frame Prediction Evaluation

- **Purpose**: Evaluate model's ability to predict the immediate next frame
- **Data**: Uses pre-computed predictions from `test_data_and_predictions.h5`
- **Metrics**: Precision, recall, F1, accuracy per brain region
- **Aggregation**: Time-averaged performance across all frames
- **Visualization**: Bar plots comparing regions

### 2. Long-Horizon Prediction Evaluation

- **Purpose**: Evaluate model's ability to maintain prediction quality over long sequences
- **Method**: Autoregressive prediction using first 110 frames to predict next 220 frames
- **Metrics**: Same metrics calculated in temporal windows
- **Temporal Analysis**: Performance tracked over time to show degradation patterns
- **Visualization**: Heatmaps showing region×time performance matrices

## Key Features

### Data Handling
- **Proper Data Separation**: Model input (sampled probabilities) vs evaluation (binary thresholded)
- **Z-Marker Detection**: Robust detection of frame boundaries using 2×2 markers
- **Volume Validation**: Ensures complete brain volumes for meaningful analysis
- **Chunk Processing**: Handles 330-frame sequences with strict boundary validation

### Region Analysis
- **Mask Integration**: Seamless integration with zebrafish brain region masks
- **Multi-Scale Support**: Configurable downsampling for different resolutions
- **Region Statistics**: Detailed metrics including support, coverage, and pixel counts
- **Performance Weighting**: Support-weighted averages for meaningful region comparisons

### Visualizations
- **Publication Ready**: High-DPI plots with clean, professional styling
- **Multiple Formats**: Bar plots, heatmaps, line plots, and comprehensive dashboards
- **Interactive Elements**: Value annotations and clear legends
- **Comprehensive Coverage**: Individual metrics plus summary dashboards

### Robustness
- **Error Handling**: Graceful handling of missing data or failed components
- **Logging**: Detailed logging at multiple levels for debugging
- **Validation**: Extensive validation of inputs, processing steps, and outputs
- **Configuration Saving**: Complete parameter tracking for reproducibility

## Performance Metrics

For each brain region, the system calculates:

- **Precision**: TP / (TP + FP) - Accuracy of positive predictions
- **Recall**: TP / (TP + FN) - Coverage of actual positives  
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total - Overall classification accuracy
- **Support**: Number of positive examples in ground truth

Summary statistics include:
- **Mean metrics**: Simple averages across regions
- **Weighted metrics**: Support-weighted averages for population-representative metrics
- **Temporal trends**: Performance changes over prediction horizon

## Example Workflow

1. **Setup**: Train GBM model and generate test predictions
2. **Prepare**: Organize experiment directory with required files
3. **Evaluate**: Run region evaluation script
4. **Analyze**: Review generated plots and numerical results
5. **Compare**: Use results to compare different models or configurations

## Troubleshooting

### Common Issues

1. **Missing test data file**: Ensure `test_data_and_predictions.h5` exists in experiment directory
2. **Mask loading errors**: Check that mask files are valid TIF format and accessible
3. **Shape mismatches**: Verify that mask target_shape matches your data dimensions
4. **Memory issues**: Reduce target_shape or use CPU device for large datasets
5. **No z-markers found**: Adjust marker thresholds or check data quality

### Debug Mode

For detailed debugging, check the generated log file in `logs/evaluation.log` which contains:
- Component initialization status
- Data loading progress  
- Processing step details
- Error tracebacks
- Performance timing information

## Integration

This evaluation system integrates seamlessly with:
- **GBM training pipeline**: Uses standard experiment output format
- **Mask management**: Compatible with `ZebrafishMaskLoader` 
- **Visualization tools**: Generates publication-ready figures
- **Analysis workflows**: JSON output enables further analysis in Python/R

The modular design allows easy extension for new evaluation metrics, visualization types, or analysis methods. 