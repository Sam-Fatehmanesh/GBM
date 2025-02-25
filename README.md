# BrainSim: Neural Activity Simulation and Prediction

BrainSim is a research project focused on modeling and predicting neural spike activity patterns in brain imaging data. The project implements various deep learning models to generate and predict sequential neural activity patterns.

## Project Overview

This project aims to:
1. Process neural spike data from multiple subjects
2. Create autoencoder models to compress and reconstruct neural activity patterns
3. Implement a Generative Brain Model (GBM) that can predict future neural activity states
4. Provide visualization tools for analyzing model performance

## Repository Structure

```
BrainSim/
├── processed_spikes/           # Processed HDF5 files containing spike data
├── processed_spikes.zip        # Compressed archive of spike data
├── GenerativeBrainModel/       # Main package
│   ├── models/                 # Neural network model definitions
│   │   ├── simple_autoencoder.py  # Simple autoencoder implementation
│   │   └── gbm.py              # Generative Brain Model implementation
│   ├── datasets/               # Dataset classes
│   │   └── sequential_spike_dataset.py  # Dataset for sequential spike data
│   ├── custom_functions/       # Utility functions
│   │   └── visualization.py    # Functions for creating plots and videos
│   └── scripts/                # Training and evaluation scripts
│       ├── train_simple_autoencoder.py  # Script to train the autoencoder
│       └── train_gbm.py        # Script to train the GBM
└── experiments/                # Output directory for experiment results
    ├── autoencoder/            # Autoencoder experiment results
    └── gbm/                    # GBM experiment results
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BrainSim.git
   cd BrainSim
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas matplotlib tqdm h5py opencv-python
   pip install mamba-ssm  # For state space models
   ```

## Usage

### Data Preparation
The project expects processed spike data in HDF5 format. Each file should contain:
- `spikes`: Binary spike events
- `cell_positions`: 3D coordinates of cells
- `timepoints`: Timestamps for each frame

### Training a Simple Autoencoder
```bash
python -m GenerativeBrainModel.scripts.train_simple_autoencoder
```

This trains an autoencoder to compress and reconstruct neural activity patterns. The model will be saved in the `experiments/autoencoder/` directory.

### Training the Generative Brain Model
```bash
python -m GenerativeBrainModel.scripts.train_gbm
```

This trains a GBM to predict future neural activity states based on past observations. The model and visualizations will be saved in the `experiments/gbm/` directory.

## Visualization

The training scripts automatically generate:
- Loss plots showing training and validation performance
- Videos comparing actual neural activity with model predictions
- Data check videos to verify dataset generation

Videos are saved in the `experiments/*/videos/` directories and can be viewed with any standard video player.

## Model Architecture

### Simple Autoencoder
- Encoder: Linear layer mapping from input size (32768) to hidden size (1024)
- Decoder: Linear layer mapping from hidden size back to input size
- Activation: Sigmoid function for binary output

### Generative Brain Model (GBM)
- Pretrained autoencoder for encoding/decoding neural activity patterns
- Mamba-based sequential predictor for modeling temporal dynamics
- Binary Cross Entropy loss function for training

## Performance Considerations

- The dataset generation process pads sequences to ensure uniform length
- Video generation is limited to a maximum number of frames to prevent excessive file sizes
- Training parameters can be adjusted in the respective script files

## License

MIT License

Copyright (c) 2023 BrainSim Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
