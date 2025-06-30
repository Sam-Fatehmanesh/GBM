# Generative Brain Model: Neural Activity Simulation and Prediction

This is a research project focused on modeling and predicting neural spike activity patterns in brain imaging data. 

## Project Overview

This project aims to:
1. Process neural spike data from multiple subjects
2. Create autoencoder models to compress and reconstruct neural activity patterns
3. Implement a Generative Brain Model (GBM) that can predict future neural activity states
4. Provide visualization tools for analyzing model performance

## Quick-Start Guide

### 1. Environment Setup
1. Install the NVIDIA driver and CUDA ≥ 11.8.
2. Create the project Conda environment (includes PyTorch + DALI):
```bash
conda env create -f environment.yml
conda activate gbm
```
3. (Optional) Export the environment for reproducibility:
```bash
conda env export --no-builds > environment.lock.yml
```

### 2. Data Preparation
The training scripts expect pre-processed spike-grid folders such as
`processed_spike_grids_2018_first_unified_test/` placed at the repository
root (same level as `GenerativeBrainModel/`). The raw → grid
conversion utilities are not included here; refer to the
`old_data/` examples for the required directory layout:
```
processed_spike_grids_2018_first_unified_test/
└── subject_*/
    └── *.npy               # (T, H, W) spike tensors
```

### 3. Training Workflows

#### 3.1 Autoencoder-Only
A lightweight convolutional auto-encoder can be trained with:
```bash
python GenerativeBrainModel/scripts/train_simple_autoencoder.py \
    --data_dir processed_spike_grids_2018_first_unified_test \
    --epochs 100 \
    --batch_size 128 \
    --output_dir experiments/autosae/$(date +%Y%m%d_%H%M%S)
```
Key flags (see `--help` for full list):
* `--latent_dim` – size of the bottleneck representation.
* `--num_workers` – DALI CPU workers per GPU.
* `--prefetch_queue` – DALI GPU prefetch depth (default 1).

#### 3.2 Two-Phase GBM (Pre-train → Fine-tune)
The main GBM sequence model is launched via the phase runner:
```bash
python GenerativeBrainModel/training/phase_runner.py \
    --config configs/gbm_zebrafish.yaml \
    --output_dir experiments/gbm/$(date +%Y%m%d_%H%M%S)
```
`configs/*.yaml` files describe optimiser, scheduler and architectural
hyper-parameters for both phases.

### 4. Evaluation & Visualisation
* **Quantitative**: run
  ```bash
  python GenerativeBrainModel/evaluation/evaluate_gbm.py \
      --checkpoint <ckpt_path> --split test
  ```
  which outputs CSV metrics and a matplotlib PDF.
* **Video reconstruction**: generate side-by-side videos of ground-truth
  vs. model output:
  ```bash
  python create_brain_video.py --checkpoint <ckpt_path> --out_dir videos/
  ```

### 5. Web Application
A FastAPI backend with streaming endpoints can be started for interactive
in-browser experiments:
```bash
uvicorn GenerativeBrainModel.webapp.main:app --host 0.0.0.0 --port 8000
```
Swagger docs are available at `http://<host>:8000/docs`.

### 6. Directory Conventions
```
.
├── GenerativeBrainModel/      # Core library code
│   ├── datasets/             # DALI + NumPy data loaders
│   ├── models/               # PyTorch + FlashAttention modules
│   ├── training/             # Training loops & schedulers
│   ├── evaluation/           # Metrics & plotting helpers
│   └── webapp/               # FastAPI service
├── experiments/              # Auto-generated experiment folders
├── processed_spike_grids_*/  # Pre-processed zebrafish data
├── environment.yml           # Conda spec
└── README.md
```

### 7. Logging & Checkpoints
* TensorBoard logs: `<output_dir>/logs/`
* Model checkpoints: `<output_dir>/checkpoints/epoch=XX.ckpt`
* `metrics.csv` and YAML summaries are written alongside checkpoints.

### 8. Reproducibility
All random seeds are initialised via the `--seed` argument (defaults to
42). Add `--deterministic` for deterministic cuDNN kernels.

### 9. Troubleshooting
* **CUDA Out-of-Memory**: lower `--batch_size` or use `--precision 16`.
* **DALI errors**: ensure matching CUDA & NVIDIA driver versions.
