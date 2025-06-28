# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from GenerativeBrainModel.models.autocnn import AutoCNN
from GenerativeBrainModel.datasets.simple_spike_dataset import create_simple_dataloaders
from GenerativeBrainModel.custom_functions.visualization import create_autoencoder_comparison_video, update_loss_plot
from GenerativeBrainModel.utils.file_utils import create_experiment_dir, save_losses_to_csv
from GenerativeBrainModel.models.automlp import AutoMLP
from GenerativeBrainModel.models.autovit import AutoViT
from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder

# set seeds
torch.manual_seed(42)
np.random.seed(42)


def collect_pca_data(train_loader, max_samples=10000):
    """Collect data samples for PCA analysis."""
    print("Collecting data samples for PCA analysis...")
    
    collected_data = []
    sample_count = 0
    
    for batch in train_loader:
        if batch.device.type != 'cuda':
            batch = batch.cuda(non_blocking=True)
        
        # Flatten each frame
        batch_flat = batch.view(batch.shape[0], -1)
        
        # Add to collection
        for i in range(batch_flat.shape[0]):
            if sample_count >= max_samples:
                break
            collected_data.append(batch_flat[i].cpu().numpy())
            sample_count += 1
        
        if sample_count >= max_samples:
            break
    
    return np.array(collected_data)

class AutoencoderDatasetWrapper:
    """Simple wrapper for autoencoder video generation."""
    
    def __init__(self, data_loader, max_samples=1000):
        self.data = []
        self._collect_data(data_loader, max_samples)
    
    def _collect_data(self, data_loader, max_samples):
        sample_count = 0
        for batch in data_loader:
            for i in range(batch.shape[0]):
                if sample_count >= max_samples:
                    break
                # Keep original 2D shape for video generation
                frame = batch[i]  # Keep as [256, 128]
                self.data.append(frame)
                sample_count += 1
            if sample_count >= max_samples:
                break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_test_loss(model, test_loader, device, max_batches=10):
    """Evaluate test loss on a subset of test data."""
    model.eval()
    test_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            batch = batch.to(device)
            
            # Autoencoder handles dimension conversion automatically
            reconstructed = model(batch)
            loss = F.binary_cross_entropy(reconstructed, batch)
            test_loss += loss.item()
            batch_count += 1
    
    return test_loss / batch_count if batch_count > 0 else 0.0

def main():
    try:
        # Autoencoder parameters
        params = {
            'batch_size': 128, 
            'num_epochs': 1,  
            'learning_rate': 1e-4, 
            'hidden_size': 2048,
            'grad_clip_norm': 1.0,  # Gradient clipping max norm
            'data_dir': 'preaugmented_training_spike_data_2018_cascade',
            'train_ratio': 0.8,  # Use 80% for training, 20% for validation
            'num_workers': 10   
        }
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = create_experiment_dir('experiments/autolinear', timestamp)
        print(f"Saving experiment results to: {exp_dir}")
        
        # Create simple dataloaders
        print("Creating data loaders...")
        train_loader, test_loader, sample_shape = create_simple_dataloaders(
            data_dir=params['data_dir'],
            train_ratio=params['train_ratio'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers']
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Sample shape: {sample_shape}")
        
        # Validate that we have data
        if len(train_loader) == 0:
            raise ValueError("No training data loaded! Check data directory and HDF5 file structure.")
        if len(test_loader) == 0:
            raise ValueError("No test data loaded! Check HDF5 file structure.")
        if sample_shape is None:
            raise ValueError("Could not determine sample shape! Check HDF5 data format.")
        
        # Validate sample shape for autoencoder
        if len(sample_shape) == 2:
            height, width = sample_shape
            if height != 256 or width != 128:
                raise ValueError(f"autoencoder expects 256x128 input, got {height}x{width}")
            print(f"Input dimensions: {height} x {width} (suitable for autoencoder)")
        else:
            raise ValueError(f"autoencoder expects 2D input, got shape: {sample_shape}")
        
        # Create autoencoder model
        # The Autoencoder expects input_size parameter but doesn't use it (uses hardcoded 256x128)
        model = SimpleAutoencoder(hidden_size=params['hidden_size'])
        
        # Save model info
        with open(os.path.join(exp_dir, "model_info.txt"), "w") as f:
            f.write("Autoencoder Training\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {model}\n\n")
            
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nModel Statistics:\n")
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Input shape: {sample_shape}\n")
            f.write(f"Hidden size: {params['hidden_size']}\n")
            
            f.write(f"\n Architecture:\n")
            f.write("Encoder: 1->4->16->128 channels with pooling\n")
            f.write("Decoder: 128->16->4->1 channels with upsampling\n")
            f.write("Final activation: Sigmoid\n")
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training
        print("Starting autoencoder training...")
        train_losses = []
        test_losses = []
        raw_batch_losses = []
        intermediate_test_losses = []
        best_test_loss = float('inf')
        
        for epoch in range(params['num_epochs']):
            # Training
            model.train()
            epoch_loss = 0.0
            quarter_epoch_size = len(train_loader) // 4
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params["num_epochs"]}')
            
            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(device)
                
                # Forward pass - Autoencoder handles dimension conversion automatically
                reconstructed = model(batch)
                loss = F.binary_cross_entropy(reconstructed, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                clip_grad_norm_(model.parameters(), params['grad_clip_norm'])
                
                optimizer.step()
                
                # Track losses
                raw_loss = loss.item()
                raw_batch_losses.append(raw_loss)
                epoch_loss += raw_loss
                current_avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': raw_loss, 'avg': current_avg_loss})
                
                # Quarter epoch evaluation
                if (batch_idx + 1) % quarter_epoch_size == 0:
                    test_loss = evaluate_test_loss(model, test_loader, device)
                    intermediate_test_losses.append(test_loss)
                    model.train()  # Back to training mode
                    
                    # Update plots
                    current_train_losses = train_losses + [current_avg_loss]
                    current_test_losses = test_losses + [test_loss]
                    update_loss_plot(current_train_losses, current_test_losses, raw_batch_losses,
                                   os.path.join(exp_dir, 'plots', 'loss_plot.png'))
                    
                    print(f"Quarter-epoch {(batch_idx + 1) // quarter_epoch_size}: Train={current_avg_loss:.6f}, Test={test_loss:.6f}")
            
            # End of epoch
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Full test evaluation
            final_test_loss = evaluate_test_loss(model, test_loader, device, max_batches=len(test_loader))
            test_losses.append(final_test_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {final_test_loss:.6f}")
            
            # Save best model
            if final_test_loss < best_test_loss:
                best_test_loss = final_test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'params': params,
                    'best_test_loss': best_test_loss
                }, os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            
            # Create reconstruction video
            test_dataset_wrapper = AutoencoderDatasetWrapper(test_loader, max_samples=100)
            video_path = os.path.join(exp_dir, 'videos', f'reconstruction_epoch_{epoch+1:03d}.mp4')
            create_autoencoder_comparison_video(model, test_dataset_wrapper, video_path, num_frames=100)
            
            # Save losses
            save_losses_to_csv({
                'epoch': list(range(1, len(train_losses) + 1)),
                'train_loss': train_losses,
                'test_loss': test_losses
            }, os.path.join(exp_dir, 'logs', 'losses.csv'))
            
            if intermediate_test_losses:
                save_losses_to_csv({
                    'quarter_epoch': list(range(1, len(intermediate_test_losses) + 1)),
                    'test_loss': intermediate_test_losses
                }, os.path.join(exp_dir, 'logs', 'intermediate_test_losses.csv'))
        
        # Save final model
        torch.save({
            'epoch': params['num_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'params': params,
            'final_train_loss': avg_train_loss,
            'final_test_loss': final_test_loss
        }, os.path.join(exp_dir, 'checkpoints', 'final_model.pt'))
        
        # Final reconstruction video
        final_test_wrapper = AutoencoderDatasetWrapper(test_loader, max_samples=200)
        video_path = os.path.join(exp_dir, 'videos', 'final_reconstruction.mp4')
        create_autoencoder_comparison_video(model, final_test_wrapper, video_path, num_frames=200)
        
        print(" autoencoder training complete!")
        print(f"Results saved to: {exp_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 