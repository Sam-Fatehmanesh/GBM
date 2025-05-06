# Simulated Optogenetics with GBM

This script simulates optogenetic stimulation of specific brain regions using a trained Generative Brain Model (GBM), then tracks the resulting activity across all brain regions over time.

## Overview

The simulation process:

1. Takes an initial seed sequence (e.g., 115 frames)
2. Activates a specified target region in the final frame of the seed
3. Runs autoregressive generation to see how activity propagates through the brain
4. Produces heatmaps showing region activity over time

## Requirements

- Same environment as the main GBM project
- A trained GBM model
- MapZBrain TIFF masks for brain regions
- Test data (either from an H5 file or via DALI loader)

## Usage

```bash
# Basic usage with predefined region list
python simulate_optogenetics.py \
  --exp-timestamp 20230815_153045 \
  --target-subject fish123 \
  --masks-dir /path/to/mapzbrain/masks/ \
  --output-dir results/opto_experiments/ \
  --region-list nucleus_of_the_medial_longitudinal_fascicle tectum nucleus_isthmi \
  --test-h5 /path/to/test_data_and_predictions.h5 \
  --seed-length 115 \
  --horizon-length 330 \
  --save-raw

# Run only a baseline simulation without stimulation
python simulate_optogenetics.py \
  --exp-timestamp 20230815_153045 \
  --target-subject fish123 \
  --masks-dir /path/to/mapzbrain/masks/ \
  --output-dir results/opto_baseline/ \
  --test-h5 /path/to/test_data_and_predictions.h5 \
  --baseline-only
```

## Command-line Arguments

- `--exp-timestamp`: Experiment folder timestamp (e.g., 20230815_153045)
- `--target-subject`: Target subject name used for finetuning
- `--masks-dir`: Directory containing MapZBrain TIFF masks
- `--output-dir`: Directory to save simulation results
- `--test-h5`: Path to saved test_data_and_predictions.h5 (required if not using --reuse-loader)
- `--reuse-loader`: Recreate DALI loader instead of using saved test data
- `--region-list`: List of regions to stimulate (TIFF filenames without .tif)
- `--seed-length`: Length of seed sequence (default: 115)
- `--horizon-length`: Maximum prediction horizon length (default: 330)
- `--baseline-only`: Run only a baseline simulation without stimulation
- `--device`: PyTorch device (default: cuda:0)
- `--save-raw`: Save raw activity data as CSV and pickle files

## Output Files

The script creates a timestamped directory under the specified output directory containing:

- **Heatmaps**: Visual representations of brain activity over time
  - One heatmap per stimulated region
  - A baseline heatmap with no stimulation
  - Both normalized and raw count versions
- **Data**: Raw activity timeseries (if --save-raw is used)
  - CSV files for each simulation run
  - A pickle file containing all activity data
- **Logs**: Detailed information about the simulation process

## Example Brain Region Pathways

The script is particularly useful for investigating the following neural pathways:

1. **nucleus_of_the_medial_longitudinal_fascicle_(pretectum,_basal_part)**
   - Projects to: rhombencephalon_(hindbrain), medulla_oblongata
   - Function: Produces graded ipsilateral tail deflections

2. **tectum**
   - Projects to: rhombencephalon_(hindbrain), nucleus_isthmi
   - Function: Drives distinct premotor ensembles in rhombencephalon

3. **nucleus_isthmi**
   - Projects to: tectum, pretectum
   - Function: Sustains prey-tracking by potentiating visual responses

4. **dorsal_habenula**
   - Projects to: interpeduncular_nucleus
   - Function: Mediates aversive-learning and fear-modulation circuits

5. **locus_coeruleus**
   - Projects to: posterior_tuberculum, preoptic_region, tectum, rhombencephalon_(hindbrain)
   - Function: Broad ascending projections to diencephalic centers

6. **superior_raphe**
   - Projects to: tectum
   - Function: Enhances visual-motion sensitivity in the tectal neuropil 