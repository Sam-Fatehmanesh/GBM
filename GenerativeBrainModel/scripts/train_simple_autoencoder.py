# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set CUDA environment variables for better stability
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder
from GenerativeBrainModel.datasets.simple_spike_dataset import create_simple_dataloaders
from GenerativeBrainModel.custom_functions.visualization import create_autoencoder_comparison_video, update_loss_plot
from GenerativeBrainModel.utils.file_utils import create_experiment_dir, save_losses_to_csv

# Seeds are now set in main() function


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

def initialize_autoencoder_with_pca(model, data_matrix, hidden_size):
    """Initialize autoencoder weights using PCA components."""
    print(f"Running PCA on {data_matrix.shape[0]} samples with {data_matrix.shape[1]} features...")
    
    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data_matrix)
    
    # Run PCA
    pca = PCA(n_components=hidden_size)
    pca.fit(data_standardized)
    
    print(f"PCA explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Get PCA components
    components = pca.components_  # Shape: (hidden_size, input_size)
    
    # Initialize encoder and decoder weights
    with torch.no_grad():
        model.encoder.weight.data = torch.FloatTensor(components)
        model.encoder.bias.data.zero_()
        model.decoder.weight.data = torch.FloatTensor(components.T)
        model.decoder.bias.data = torch.FloatTensor(scaler.mean_)
    
    print("Autoencoder weights initialized with PCA components!")
    
    return {
        'pca': pca,
        'scaler': scaler,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_explained_variance': pca.explained_variance_ratio_.sum(),
        'n_components': hidden_size,
        'n_samples_used': data_matrix.shape[0]
    }

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

def validate_data_ranges(tensor, name="tensor"):
    """Validate tensor for valid probability ranges and NaN/inf values."""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains infinite values")
        return False
    if tensor.min() < 0.0:
        print(f"Warning: {name} contains values < 0.0 (min: {tensor.min().item():.6f})")
        return False
    if tensor.max() > 1.0:
        print(f"Warning: {name} contains values > 1.0 (max: {tensor.max().item():.6f})")
        return False
    return True

def evaluate_test_loss(model, test_loader, device, max_batches=10):
    """Evaluate test loss on a subset of test data."""
    model.eval()
    test_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break
                
            try:
                batch = batch.to(device, non_blocking=True)
                batch_flat = batch.view(batch.shape[0], -1)
                
                # Sample batch_flat from probabilities to binary
                # Clamp values to [0, 1] and handle NaN/inf to ensure valid probabilities
                batch_flat = torch.clamp(batch_flat, min=0.0, max=1.0)
                batch_flat = torch.nan_to_num(batch_flat, nan=0.0, posinf=1.0, neginf=0.0)
                batch_flat = torch.bernoulli(batch_flat)
                
                reconstructed = model(batch_flat)
                # Reshape reconstructed back to original shape for loss calculation
                reconstructed_reshaped = reconstructed.view(batch.shape)
                
                # Ensure target is properly clamped and cleaned
                target = torch.clamp(batch, min=0.0, max=1.0)
                target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
                loss = F.binary_cross_entropy(reconstructed_reshaped, target)
                test_loss += loss.item()
                batch_count += 1
                
                # Synchronize CUDA operations
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error in test evaluation at batch {batch_idx}: {e}")
                    # Clear CUDA cache and continue
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    continue
                else:
                    raise e
    
    return test_loss / batch_count if batch_count > 0 else 0.0

def main():
    try:
        # Set deterministic behavior for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train Simple Autoencoder on Spike Data')
        parser.add_argument('--data_dir', type=str, required=True,
                           help='Directory containing processed spike data')
        parser.add_argument('--batch_size', type=int, default=128,
                           help='Batch size for training (default: 128)')
        parser.add_argument('--num_epochs', type=int, default=1,
                           help='Number of training epochs (default: 1)')
        parser.add_argument('--learning_rate', type=float, default=1e-4,
                           help='Learning rate for optimizer (default: 1e-3)')
        parser.add_argument('--hidden_size', type=int, default=2048,
                           help='Hidden layer size for autoencoder (default: 2048)')
        parser.add_argument('--train_ratio', type=float, default=0.95,
                           help='Ratio of data to use for training (default: 0.95)')
        parser.add_argument('--num_workers', type=int, default=10,
                           help='Number of data loader workers (default: 10)')
        parser.add_argument('--exp_name', type=str, default=None,
                           help='Experiment name (default: timestamp)')
        
        args = parser.parse_args()
        
        # Validate data directory
        if not os.path.exists(args.data_dir):
            raise ValueError(f"Data directory not found: {args.data_dir}")
        
        # Convert to params dict
        params = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size,
            'data_dir': args.data_dir,
            'train_ratio': args.train_ratio,
            'num_workers': args.num_workers
        }
        
        # Print configuration
        print("Training Configuration:")
        print("=" * 50)
        for key, value in params.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        # Create experiment directory
        if args.exp_name:
            exp_name = args.exp_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exp_name = timestamp
        exp_dir = create_experiment_dir('experiments/autoencoder', exp_name)
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
        
        # Sample a batch to check data integrity
        print("Checking data integrity...")
        try:
            sample_batch = next(iter(train_loader))
            print(f"Sample batch shape: {sample_batch.shape}")
            print(f"Sample batch dtype: {sample_batch.dtype}")
            print(f"Sample batch min: {sample_batch.min().item():.6f}")
            print(f"Sample batch max: {sample_batch.max().item():.6f}")
            print(f"Sample batch mean: {sample_batch.mean().item():.6f}")
            print(f"Sample batch std: {sample_batch.std().item():.6f}")
            print(f"Sample batch NaN count: {torch.isnan(sample_batch).sum().item()}")
            print(f"Sample batch Inf count: {torch.isinf(sample_batch).sum().item()}")
            
            # Check if values are in valid probability range
            if sample_batch.min() < 0.0 or sample_batch.max() > 1.0:
                print("WARNING: Data contains values outside [0, 1] range!")
                print("This will cause issues with torch.bernoulli()")
                
        except Exception as e:
            print(f"Error checking data integrity: {e}")
        
        # Validate that we have data
        if len(train_loader) == 0:
            raise ValueError("No training data loaded! Check data directory and HDF5 file structure.")
        if len(test_loader) == 0:
            raise ValueError("No test data loaded! Check data directory and HDF5 file structure.")
        if sample_shape is None:
            raise ValueError("Could not determine sample shape! Check HDF5 data format.")
        
        # Calculate input size
        if len(sample_shape) == 3:
            channels, height, width = sample_shape
            input_size = channels * height * width
            print(f"Input dimensions: {channels} x {height} x {width} = {input_size}")
        elif len(sample_shape) == 2:
            height, width = sample_shape
            input_size = height * width
            print(f"Input dimensions: {height} x {width} = {input_size}")
        else:
            raise ValueError(f"Unexpected sample shape: {sample_shape}")
        
        # Create model
        model = SimpleAutoencoder(
            input_size=input_size,
            hidden_size=params['hidden_size']
        )
        
        # PCA initialization (disabled)
        print("Skipping PCA initialization...")
        pca_results = {
            'n_samples_used': 0,
            'n_components': 0,
            'total_explained_variance': 0.0,
            'explained_variance_ratio': [],
            'pca': None
        }
        
        # Save model info and PCA results
        with open(os.path.join(exp_dir, "model_info.txt"), "w") as f:
            f.write("Simple Autoencoder Training\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Model: {model}\n\n")
            
            f.write("Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nModel Statistics:\n")
            total_params = sum(p.numel() for p in model.parameters())
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Input size: {input_size}\n")
            f.write(f"Hidden size: {params['hidden_size']}\n")
        
            f.write(f"\nPCA Initialization:\n")
            f.write(f"Samples used: {pca_results['n_samples_used']:,}\n")
            f.write(f"Components: {pca_results['n_components']}\n")
            f.write(f"Explained variance: {pca_results['total_explained_variance']:.4f}\n")
        
        # Skip PCA analysis since it's disabled
        print("PCA analysis skipped")
        
        # Initialize CUDA properly with fallback to CPU
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            try:
                print(f"CUDA available: {torch.cuda.is_available()}")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"CUDA current device: {torch.cuda.current_device()}")
                print(f"CUDA device name: {torch.cuda.get_device_name()}")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Set CUDA device explicitly
                torch.cuda.set_device(0)
                
                # Initialize CUDA context
                torch.cuda.init()
                
                # Test CUDA with a simple operation
                test_tensor = torch.randn(10, 10).cuda()
                test_result = torch.matmul(test_tensor, test_tensor.T)
                torch.cuda.synchronize()
                
                device = torch.device("cuda:0")
                print(f"CUDA initialization successful")
                
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU")
                use_cuda = False
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Ensure model is in correct state
        model.train()
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training
        print("Starting training...")
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
                try:
                    # Move batch to device with proper error handling
                    batch = batch.to(device, non_blocking=True)
                    batch_flat = batch.view(batch.shape[0], -1)
                    
                    # Validate input data ranges (only for first few batches to avoid spam)
                    if batch_idx < 5:
                        if not validate_data_ranges(batch_flat, f"batch_{batch_idx}_input"):
                            print(f"Data range issues detected in batch {batch_idx}")
                            print(f"  Min: {batch_flat.min().item():.6f}, Max: {batch_flat.max().item():.6f}")
                            print(f"  NaN count: {torch.isnan(batch_flat).sum().item()}")
                            print(f"  Inf count: {torch.isinf(batch_flat).sum().item()}")
                    
                    # Sample batch_flat from probabilities to binary
                    # Clamp values to [0, 1] and handle NaN/inf to ensure valid probabilities
                    batch_flat = torch.clamp(batch_flat, min=0.0, max=1.0)
                    batch_flat = torch.nan_to_num(batch_flat, nan=0.0, posinf=1.0, neginf=0.0)
                    batch_flat = torch.bernoulli(batch_flat)
                    
                    # Ensure gradient computation is enabled
                    batch_flat.requires_grad_(True)
                    
                    # Forward pass with gradient checkpointing for memory efficiency
                    reconstructed = model(batch_flat)
                    
                    # Reshape reconstructed back to original shape for loss calculation
                    reconstructed_reshaped = reconstructed.view(batch.shape)
                    
                    # Calculate loss - ensure target is also properly clamped
                    target = torch.clamp(batch, min=0.0, max=1.0)
                    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
                    loss = F.binary_cross_entropy(reconstructed_reshaped, target)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Synchronize CUDA operations
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Track losses
                    raw_loss = loss.item()
                    raw_batch_losses.append(raw_loss)
                    epoch_loss += raw_loss
                    current_avg_loss = epoch_loss / (batch_idx + 1)
                    progress_bar.set_postfix({'loss': raw_loss, 'avg': current_avg_loss})
                    
                    # Clean up memory periodically
                    if batch_idx % 100 == 0 and device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"CUDA error at batch {batch_idx}: {e}")
                        print("Attempting to recover...")
                        
                        # Print debug info about the current batch
                        print(f"Batch shape: {batch.shape}")
                        print(f"Batch device: {batch.device}")
                        print(f"Batch dtype: {batch.dtype}")
                        print(f"Batch min: {batch.min().item():.6f}, max: {batch.max().item():.6f}")
                        print(f"Batch NaN count: {torch.isnan(batch).sum().item()}")
                        print(f"Batch Inf count: {torch.isinf(batch).sum().item()}")
                        
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        
                        # Reset CUDA context
                        torch.cuda.synchronize()
                        
                        # Skip this batch and continue
                        continue
                    else:
                        raise e
                
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
                    'pca_results': pca_results,
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
            'pca_results': pca_results,
            'final_train_loss': avg_train_loss,
            'final_test_loss': final_test_loss
        }, os.path.join(exp_dir, 'checkpoints', 'final_model.pt'))
        
        # Final reconstruction video
        final_test_wrapper = AutoencoderDatasetWrapper(test_loader, max_samples=200)
        video_path = os.path.join(exp_dir, 'videos', 'final_reconstruction.mp4')
        create_autoencoder_comparison_video(model, final_test_wrapper, video_path, num_frames=200)
        
        print("Training complete!")
        print(f"Results saved to: {exp_dir}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 