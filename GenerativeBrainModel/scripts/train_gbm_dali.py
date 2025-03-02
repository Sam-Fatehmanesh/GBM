# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set maximum memory allocations for DALI
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["DALI_PREALLOCATE_WIDTH"] = "4096" 
os.environ["DALI_PREALLOCATE_HEIGHT"] = "4096"
os.environ["DALI_PREALLOCATE_DEPTH"] = "8"
os.environ["DALI_TENSOR_ALLOCATOR_BLOCK_SIZE"] = str(512*1024*1024)  # 512MB blocks - reduced from 1GB

import torch
import torch.multiprocessing as mp
# Set sharing strategy to file_system to avoid shared memory issues
mp.set_sharing_strategy('file_system')
# Set the start method to spawn to avoid CUDA initialization issues
mp.set_start_method('spawn', force=True)

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import h5py
import gc
import time

# Import DALI components first, before initializing CUDA
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
try:
    # Attempt to enable DALI profiling for optimization
    from nvidia.dali.plugin.profiler import ProfilerTarget, Profiler
    has_profiler = True
except ImportError:
    has_profiler = False

# Enable memory diagnostics
MEMORY_DIAGNOSTICS = False

def print_memory_stats(prefix=""):
    """Print GPU memory usage statistics"""
    if MEMORY_DIAGNOSTICS:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"{prefix} GPU Memory: Total={t/1e9:.2f}GB, Reserved={r/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free={f/1e9:.2f}GB")

# Import our custom modules
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.dali_spike_dataset import DALIBrainDataLoader
from GenerativeBrainModel.custom_functions.visualization import create_comparison_video, update_loss_plot

# Initialize CUDA after importing DALI
torch.cuda.init()
# Pre-allocate CUDA memory to avoid fragmentation
# Reserve 512MB (reduced from 1GB) for continuous allocation
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()

# Enable tensor cores for better performance with fp16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable memory optimizations
torch.backends.cudnn.benchmark = True

def get_max_z_planes(spike_files):
    """Get the maximum number of z-planes across all subjects.
    
    Args:
        spike_files: List of paths to processed spike data H5 files
    
    Returns:
        max_z_planes: Maximum number of z-planes found across all subjects
    """
    max_z_planes = 0
    
    for h5_file in spike_files:
        with h5py.File(h5_file, 'r') as f:
            cell_positions = f['cell_positions'][:]
            # Get unique z values (rounded to handle floating point precision)
            z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
            max_z_planes = max(max_z_planes, len(z_values))
    
    if max_z_planes == 0:
        raise ValueError("No z-planes found in any of the spike files!")
    
    return max_z_planes

def create_experiment_dir():
    """Create a timestamped experiment directory with all necessary subdirectories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', 'gbm', timestamp)
    
    # Create main experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    return exp_dir

def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

def create_prediction_video(model, data_loader, output_path, num_frames=8):
    """Create a video of model predictions vs actual brain activity."""
    # Get a single batch
    data_loader.reset()
    batch = next(iter(data_loader))
    
    # Ensure batch is on CUDA
    if batch.device.type != 'cuda':
        batch = batch.cuda()
    
    # Convert uint8 to float32 for visualization if needed
    if batch.dtype == torch.uint8:
        batch_viz = batch.float()
    else:
        batch_viz = batch
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        # Model already handles different input types
        predictions = model.get_predictions(batch)
    
    # Convert predictions to numpy for visualization
    # We'll choose 2 random sequences from the batch for visualization
    rand_indices = torch.randint(0, batch.size(0), (2,))
    
    # Create comparison visualization
    create_comparison_video(
        actual=batch_viz[rand_indices].cpu().numpy(),
        predicted=predictions[rand_indices].cpu().numpy(),
        output_path=output_path,
        num_frames=num_frames,
        fps=5
    )

def main():
    try:
        # Parameters
        params = {
            'batch_size': 64,  # Reduced from 128 to 64 to save memory
            'num_epochs': 4,
            'learning_rate': 1e-4,
            'mamba_layers': 1,
            'mamba_dim': 1024,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95,
            'dali_num_threads': 2,  # Reduced from 4 to 2
            'gpu_prefetch': 1,      # Reduced from 2 to 1
            'use_float16': True,    # Enable float16 for memory efficiency
        }
        
        # Print initial memory stats
        print_memory_stats("Initial:")
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        tqdm.write(f"Saving experiment results to: {exp_dir}")
        
        # Create datasets from spike data
        processed_dir = "processed_spikes"
        spike_files = []
        for file in os.listdir(processed_dir):
            if file.endswith("_processed.h5"):
                spike_files.append(os.path.join(processed_dir, file))
        
        if not spike_files:
            raise ValueError("No processed spike data files found!")
        
        # Get maximum number of z-planes across all subjects
        max_z_planes = get_max_z_planes(spike_files)
        params['seq_len'] = params['timesteps_per_sequence'] * max_z_planes
        
        tqdm.write(f"Maximum z-planes across subjects: {max_z_planes}")
        tqdm.write(f"Total sequence length: {params['seq_len']} ({params['timesteps_per_sequence']} timepoints × {max_z_planes} z-planes)")
        
        # Create DALI data loaders with optimized parameters
        train_loader = DALIBrainDataLoader(
            spike_files,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            split='train',
            train_ratio=params['train_ratio'],
            device_id=0,  # Assuming single GPU setup
            num_threads=params['dali_num_threads'],
            gpu_prefetch=params['gpu_prefetch'],
            seed=42,
            shuffle=True
        )
        
        print_memory_stats("After train loader:")
        
        # Use fewer files for test loader to save memory
        test_files = spike_files[:5] if len(spike_files) > 5 else spike_files
        test_loader = DALIBrainDataLoader(
            test_files,  # Use subset of files for testing
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            split='test',
            train_ratio=params['train_ratio'],
            device_id=0,  # Assuming single GPU setup
            num_threads=params['dali_num_threads'],
            gpu_prefetch=params['gpu_prefetch'],
            seed=43,  # Different seed for test set
            shuffle=False
        )
        
        print_memory_stats("After test loader:")
        
        tqdm.write(f"Number of batches in train_loader: {len(train_loader)}")
        tqdm.write(f"Number of batches in test_loader: {len(test_loader)}")
        
        # Create model
        model = GBM(
            mamba_layers=params['mamba_layers'],
            mamba_dim=params['mamba_dim']
        )
        
        # Save model architecture and parameters
        with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
            f.write("Complete Model Architecture:\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model))
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("Model Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            # Add statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\nModel Statistics:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        model = model.to(device)
        
        print_memory_stats("After model creation:")
        
        # Skip data check video to save memory during diagnostics
        # create_data_check_video(model, test_loader, video_path)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Set up DALI profiler if available
        if has_profiler:
            profiler = Profiler()
            profiler.init(ProfilerTarget.CPU_GPU)
            profiler.start()
        
        # Training loop
        tqdm.write("Starting training...")
        train_losses = []
        test_losses = []
        raw_batch_losses = []  # Track individual batch losses
        best_test_loss = float('inf')
        
        # Create CUDA streams for overlapping operations
        compute_stream = torch.cuda.Stream()
        copy_stream = torch.cuda.Stream()
        
        # Enable automatic mixed precision for memory efficiency
        scaler = torch.cuda.amp.GradScaler(enabled=params['use_float16'])
        
        for epoch in range(params['num_epochs']):
            if MEMORY_DIAGNOSTICS:
                print_memory_stats(f"Beginning of epoch {epoch+1}:")
                
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # Reset DALI loader
            train_loader.reset()
            train_iter = iter(train_loader)
            
            # Training loop with tqdm for progress display
            train_loop = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{params['num_epochs']}")
            
            for batch_idx in train_loop:
                if MEMORY_DIAGNOSTICS:
                    print_memory_stats(f"Before batch {batch_idx+1}:")
                
                try:
                    # Get batch with automatic GPU transfer
                    batch = next(train_iter)
                    
                    # Ensure batch is on GPU
                    if batch.device.type != 'cuda':
                        batch = batch.cuda(non_blocking=True)
                    
                    # Process batch - model handles dtype conversion internally
                    optimizer.zero_grad()
                    
                    # Use automatic mixed precision for forward pass
                    with torch.cuda.amp.autocast(enabled=params['use_float16']):
                        # Get logits from model (no sigmoid)
                        predictions = model(batch)
                        # Binary cross entropy with logits loss
                        loss = model.compute_loss(predictions, batch[:, 1:])
                    
                    # Scale loss and backpropagate with mixed precision
                    scaler.scale(loss).backward()
                    # Update weights with gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Record raw batch loss
                    raw_batch_losses.append(loss.item())
                    
                    # Update tqdm display
                    train_loop.set_postfix(loss=loss.item())
                    
                    # Clean up memory
                    if batch_idx % 50 == 0:
                        # Forced garbage collection every 50 batches to control memory growth
                        torch.cuda.empty_cache()
                        gc.collect()
                
                except StopIteration:
                    break
            
            # Append actual epoch average loss
            avg_train_loss = epoch_loss / (batch_count)
            train_losses.append(avg_train_loss)
            
            # Clean up memory before evaluation
            torch.cuda.empty_cache()
            if MEMORY_DIAGNOSTICS:
                print_memory_stats(f"Before evaluation, epoch {epoch+1}:")
            
            # Testing
            model.eval()
            test_loss = 0.0
            test_batch_count = 0
            
            # Reset DALI loader for testing
            test_loader.reset()
            test_iter = iter(test_loader)
            
            with torch.no_grad():
                for test_batch_idx in range(len(test_loader)):
                    try:
                        # Get batch with automatic GPU transfer
                        batch = next(test_iter)
                        
                        # Ensure batch is on GPU
                        if batch.device.type != 'cuda':
                            batch = batch.cuda(non_blocking=True)
                        
                        # Model handles different input types internally
                        with torch.cuda.amp.autocast(enabled=params['use_float16']):
                            # Get logits from model (no sigmoid)
                            predictions = model(batch)
                            # Loss function handles type conversion
                            loss = model.compute_loss(predictions, batch[:, 1:])
                        
                        test_loss += loss.item()
                        test_batch_count += 1
                        
                    except StopIteration:
                        break
            
            if test_batch_count > 0:
                avg_test_loss = test_loss / test_batch_count
            else:
                avg_test_loss = float('inf')
                
            test_losses.append(avg_test_loss)
            
            # Clean up memory before video creation
            torch.cuda.empty_cache()
            if MEMORY_DIAGNOSTICS:
                print_memory_stats(f"Before video creation, epoch {epoch+1}:")
            
            # Create prediction video after each epoch
            video_path = os.path.join(exp_dir, 'videos', f'predictions_epoch_{epoch+1:03d}.mp4')
            create_prediction_video(model, test_loader, video_path)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'raw_batch_losses': raw_batch_losses,
                    'params': params,
                    'best_test_loss': best_test_loss
                }, os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            
            # Update plots and save losses
            update_loss_plot(train_losses, test_losses, raw_batch_losses,
                           os.path.join(exp_dir, 'plots', 'loss_plot.png'))
            save_losses_to_csv(
                {'epoch': list(range(1, len(train_losses) + 1)),
                 'train_loss': train_losses,
                 'test_loss': test_losses},
                os.path.join(exp_dir, 'logs', 'losses.csv')
            )
            
            # Save raw batch losses separately
            if raw_batch_losses:
                save_losses_to_csv(
                    {'batch': list(range(1, len(raw_batch_losses) + 1)),
                     'raw_loss': raw_batch_losses},
                    os.path.join(exp_dir, 'logs', 'raw_batch_losses.csv')
                )
            
            tqdm.write(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                      f"Test Loss = {avg_test_loss:.6f}")
        
        # Stop DALI profiler if active
        if has_profiler:
            profiler.stop()
            profile_path = os.path.join(exp_dir, 'logs', 'dali_profile.json')
            profiler.export(profile_path)
            
        tqdm.write("Training complete!")
        
        # Save final model and diagnostics
        torch.save({
            'epoch': params['num_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'raw_batch_losses': raw_batch_losses,
            'params': params,
            'final_train_loss': avg_train_loss,
            'final_test_loss': avg_test_loss
        }, os.path.join(exp_dir, 'checkpoints', 'final_model.pt'))
        
        # Create final prediction video
        video_path = os.path.join(exp_dir, 'videos', 'final_predictions.mp4')
        create_prediction_video(model, test_loader, video_path, max_seqs=5)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 