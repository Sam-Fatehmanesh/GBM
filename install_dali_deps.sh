#!/bin/bash
set -e

# Activate the conda environment
echo "Activating conda environment 'brainsim'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brainsim

# Get CUDA version from the environment
CUDA_VERSION=$(conda list | grep cudatoolkit | awk '{print $2}' | cut -d'.' -f1-2)
echo "Detected CUDA version: $CUDA_VERSION"

# Install NVIDIA DALI based on CUDA version
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "Installing NVIDIA DALI for CUDA 11.8..."
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    echo "Installing NVIDIA DALI for CUDA 12.1..."
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
else
    echo "Unsupported CUDA version: $CUDA_VERSION. Installing DALI for CUDA 12.0 (closest match)..."
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120
fi

# No need for TensorFlow plugin since we're only using PyTorch
echo "DALI for PyTorch has been installed successfully."

# Update the environment.yml file to include the new dependencies
echo "Updating environment.yml file..."
conda env export --name brainsim > environment.yml.new

# Add a note about manually installed packages
echo "# Note: NVIDIA DALI for PyTorch was installed via pip using install_dali_deps.sh" >> environment.yml.new
echo "# If recreating this environment, run install_dali_deps.sh after conda env create" >> environment.yml.new

# Replace the old environment.yml
mv environment.yml.new environment.yml

echo "Installation complete! NVIDIA DALI for PyTorch has been installed in the 'brainsim' environment."
echo "The environment.yml file has been updated to reflect the changes."
echo "To run the DALI-based training script, use: python -m GenerativeBrainModel.scripts.train_gbm_dali" 