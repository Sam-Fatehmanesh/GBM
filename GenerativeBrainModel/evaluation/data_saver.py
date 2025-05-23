"""Test data and prediction saving utilities."""

import os
import h5py
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from datetime import datetime


def save_test_data_and_predictions(model, data_loader, save_dir, num_samples=100, params=None):
    """
    Save a batch of test data and model predictions for later analysis.
    
    Args:
        model: The trained model
        data_loader: DataLoader containing test data
        save_dir: Directory to save the data and predictions
        num_samples: Number of binary samples to generate from the model
        params: Dictionary of model parameters and hyperparameters
    """
    # Get a batch of test data
    data_loader.reset()
    
    # Get first batch
    batch = next(iter(data_loader))
    
    # Get z-start values for each sequence from the dataloader
    # This uses the new property we added to the FastDALIBrainDataLoader
    z_starts = []
    if hasattr(data_loader, 'batch_z_starts') and data_loader.batch_z_starts:
        # Use the data loader's tracked z-start values
        z_starts = data_loader.batch_z_starts
        tqdm.write(f"Retrieved {len(z_starts)} z-start values from data loader")
    else:
        # Fallback to default assumption that all sequences start at z=0
        tqdm.write("No z-start values available from data loader, assuming z_start=0 for all sequences")
        z_starts = [0] * batch.shape[0]
    
    # Ensure batch is on CUDA
    if batch.device.type != 'cuda':
        batch = batch.cuda()
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        with autocast():
            # Get probability predictions from the model
            predictions = model.get_predictions(batch)
            
            # Generate sampled binary predictions
            sampled_predictions = model.sample_binary_predictions(predictions)
    
    # Choose a subset of samples for saving
    num_sequences = min(num_samples, batch.size(0))
    rand_indices = torch.randperm(batch.size(0))[:num_sequences]
    
    # Extract the selected samples
    test_data = batch[rand_indices].cpu().numpy()
    pred_probs = predictions[rand_indices].cpu().numpy()
    pred_samples = sampled_predictions[rand_indices].cpu().numpy()
    
    # Extract corresponding z_starts for selected samples
    selected_z_starts = [z_starts[i] for i in rand_indices.cpu().numpy()]
    
    # Create output directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Create metadata dictionary 
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': num_sequences,
        'test_data_shape': test_data.shape,
        'predictions_shape': pred_probs.shape,
        'samples_shape': pred_samples.shape
    }
    
    # Add model architecture details
    metadata['model_type'] = 'GBM'
    metadata['mamba_layers'] = model.mamba.num_layers
    metadata['mamba_dim'] = model.mamba.d_model
    metadata['latent_dim'] = model.latent_dim
    metadata['grid_size'] = list(model.grid_size)
    
    # Add model parameters from params dictionary if provided
    if params is not None:
        for key, value in params.items():
            # Only include simple types that can be stored as attributes in HDF5
            if isinstance(value, (str, int, float, bool, list, tuple)) and not key.startswith('_'):
                metadata[f'param_{key}'] = value
    
    # Save the data to HDF5 file
    output_file = os.path.join(save_dir, 'test_data_and_predictions.h5')
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        f.create_dataset('test_data', data=test_data)
        f.create_dataset('predicted_probabilities', data=pred_probs)
        f.create_dataset('predicted_samples', data=pred_samples)
        
        # Save z_start values for each sequence
        f.create_dataset('sequence_z_starts', data=np.array(selected_z_starts, dtype=np.int32))
        
        # Store metadata as both attributes and a separate metadata group
        meta_group = f.create_group('metadata')
        
        # Add attributes to the file
        for key, value in metadata.items():
            try:
                # Try to add as file attribute
                f.attrs[key] = value
                
                # Also store in metadata group
                if isinstance(value, (list, tuple)):
                    meta_group.create_dataset(key, data=value)
                else:
                    meta_group.attrs[key] = value
            except Exception as e:
                print(f"Warning: Could not save metadata {key}: {e}")
    
    tqdm.write(f"Saved {num_sequences} test samples and predictions to {output_file}")
    tqdm.write(f"Included sequence_z_starts indicating the starting z-plane index for each sequence")
    
    # Create a small metadata file with description
    with open(os.path.join(save_dir, 'test_data_info.txt'), 'w') as f:
        f.write(f"Test data and predictions saved on: {metadata['timestamp']}\n")
        f.write(f"Number of samples: {metadata['num_samples']}\n")
        f.write(f"Data shapes:\n")
        f.write(f"  test_data: {metadata['test_data_shape']}\n")
        f.write(f"  predicted_probabilities: {metadata['predictions_shape']}\n")
        f.write(f"  predicted_samples: {metadata['samples_shape']}\n")
        f.write(f"  sequence_z_starts: array of shape ({num_sequences},) containing starting z-plane for each sequence\n")
        f.write("\nModel architecture:\n")
        f.write(f"  Model type: {metadata['model_type']}\n")
        f.write(f"  Mamba layers: {metadata['mamba_layers']}\n")
        f.write(f"  Mamba dimension: {metadata['mamba_dim']}\n")
        f.write(f"  Latent dimension: {metadata['latent_dim']}\n")
        f.write(f"  Grid size: {metadata['grid_size']}\n")
        f.write("\nTraining parameters:\n")
        
        if params is not None:
            for key, value in params.items():
                if not key.startswith('_') and key not in ['preaugmented_dir']:
                    f.write(f"  {key}: {value}\n")
        
        f.write("\nData format:\n")
        f.write("  test_data: Original test data sequences\n")
        f.write("  predicted_probabilities: Model's predicted probabilities\n")
        f.write("  predicted_samples: Binary samples from the predicted probabilities\n")
        f.write("  sequence_z_starts: Starting z-plane index for each sequence\n")
        f.write("\nFile format: HDF5\n")
        f.write("Access data with: h5py.File('test_data_and_predictions.h5', 'r')\n") 