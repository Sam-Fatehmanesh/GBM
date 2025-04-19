#!/usr/bin/env python3
"""
Train GBM model with the fast DALI dataset loader.

This is a direct replacement for train_gbm_dali.py that uses the 
FastDALIBrainDataLoader to load pre-augmented data for faster training.
"""

# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set maximum memory allocations for DALI
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["DALI_PREALLOCATE_WIDTH"] = "4096" 
os.environ["DALI_PREALLOCATE_HEIGHT"] = "4096"
os.environ["DALI_PREALLOCATE_DEPTH"] = "8"
os.environ["DALI_TENSOR_ALLOCATOR_BLOCK_SIZE"] = str(512*1024*1024)  # 512MB blocks

import torch
import torch.multiprocessing as mp
# Set sharing strategy to file_system to avoid shared memory issues
mp.set_sharing_strategy('file_system')
# Set the start method to spawn to avoid CUDA initialization issues
mp.set_start_method('spawn', force=True)

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import h5py
import gc
import time

# Import DALI components
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
try:
    # Attempt to enable DALI profiling for optimization
    from nvidia.dali.plugin.profiler import ProfilerTarget, Profiler
    has_profiler = True
except ImportError:
    has_profiler = False

# Sets torch seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

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
from GenerativeBrainModel.datasets.fast_dali_spike_dataset import FastDALIBrainDataLoader
from GenerativeBrainModel.custom_functions.visualization import create_comparison_video, update_loss_plot

# Initialize CUDA after importing DALI
torch.cuda.init()
# Pre-allocate CUDA memory to avoid fragmentation
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()

# Enable tensor cores for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable memory optimizations
torch.backends.cudnn.benchmark = True

def get_max_z_planes(preaugmented_dir, split='train'):
    """Get the maximum number of z-planes from the preaugmented data.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        split: 'train' or 'test'
    
    Returns:
        max_z_planes: Maximum number of z-planes found across all subjects
    """
    max_z_planes = 0
    
    for subject_dir in os.listdir(preaugmented_dir):
        subject_path = os.path.join(preaugmented_dir, subject_dir)
        if os.path.isdir(subject_path):
            metadata_path = os.path.join(subject_path, 'metadata.h5')
            if os.path.exists(metadata_path):
                with h5py.File(metadata_path, 'r') as f:
                    if 'num_z_planes' in f:
                        max_z_planes = max(max_z_planes, f['num_z_planes'][()])
    
    if max_z_planes == 0:
        raise ValueError("No z-planes found in any of the preaugmented data files!")
    
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

def create_prediction_video(model, data_loader, output_path, num_frames=330):
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
        # Get probability predictions from the model
        predictions = model.get_predictions(batch)
        
        # Generate sampled binary predictions from the predicted probability distributions
        sampled_predictions = model.sample_binary_predictions(predictions)
    
    # Convert predictions to numpy for visualization
    # We'll choose multiple sequences from the batch for visualization (up to 100)
    num_sequences = min(100, batch.size(0))
    rand_indices = torch.randperm(batch.size(0))[:num_sequences]
    
    # Create comparison visualization with 1fps, including the sampled predictions
    create_comparison_video(
        actual=batch_viz[rand_indices].cpu().numpy(),
        predicted=predictions[rand_indices].cpu().numpy(),
        output_path=output_path,
        num_frames=num_frames,
        fps=1,  # Set to 1fps as requested
        sampled_predictions=sampled_predictions[rand_indices].cpu().numpy()
    )

def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    """Creates a learning rate scheduler with linear warmup and cosine decay to specified min_lr.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate to decay to
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Get base lr
        base_lr = optimizer.param_groups[0]['lr']
        
        # Convert min_lr to a fraction of base_lr for the lambda function
        # Prevent division by zero by ensuring base_lr is not zero
        min_lr_factor = min_lr / max(base_lr, 1e-8)
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine annealing after warmup with decay to min_lr
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between 1.0 and min_lr_factor instead of between 1.0 and 0.0
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

def main():
    try:
        # Parameters
        params = {
            'preaugmented_dir': 'preaugmented_training_spike_data_2018',  # Directory with preaugmented data
            'batch_size': 128, 
            'num_epochs': 1,
            'learning_rate': 6e-3,
            'weight_decay': 0.1,         # Weight decay for AdamW
            'warmup_ratio': 0.1,         # Warmup ratio (percentage of total steps)
            'min_lr': 1e-5,              # Minimum learning rate for cosine decay
            'mamba_layers': 8,
            'mamba_dim': 1024,
            'mamba_state_multiplier': 4,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95,
            'dali_num_threads': 2, 
            'gpu_prefetch': 1,      
            'use_float16': False,    # Disabled float16 for full precision training
            'seed': seed,
            'validation_per_epoch': 8,  # Number of validation evaluations per epoch
            'timestep_stride': 1/3,       # Maximum allowed overlap between sequences, or None for no restriction
        }

        assert params['timesteps_per_sequence'] >= params['timestep_stride'], "Timestep stride must be less than or equal to timesteps per sequence"
        
        # Print initial memory stats
        print_memory_stats("Initial:")
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        tqdm.write(f"Saving experiment results to: {exp_dir}")
        
        # Check if preaugmented directory exists
        if not os.path.exists(params['preaugmented_dir']):
            raise ValueError(f"Preaugmented directory '{params['preaugmented_dir']}' not found!")
        
        # Get maximum number of z-planes from preaugmented data
        max_z_planes = get_max_z_planes(params['preaugmented_dir'])
        params['seq_len'] = int(params['timesteps_per_sequence'] * max_z_planes)    
        params['seq_stride'] =  int(params['timestep_stride'] * max_z_planes)
        
        # Calculate stride based on max_overlap if specified
        if 'max_overlap' in params and params['max_overlap'] is not None:
            stride = params['seq_len'] - params['max_overlap']
            tqdm.write(f"Using stride={stride} based on max_overlap={params['max_overlap']}")
        else:
            stride = None
            tqdm.write("No stride specified, sequences can start at any valid point")
        
        tqdm.write(f"Maximum z-planes across subjects: {max_z_planes}")
        tqdm.write(f"Total sequence length: {params['seq_len']} ({params['timesteps_per_sequence']} timepoints × {max_z_planes} z-planes)")
        
        # Create DALI data loaders with the FastDALIBrainDataLoader
        train_loader = FastDALIBrainDataLoader(
            params['preaugmented_dir'],
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            split='train',
            device_id=0,  # Assuming single GPU setup
            num_threads=params['dali_num_threads'],
            gpu_prefetch=params['gpu_prefetch'],
            seed=42,
            shuffle=True,
            stride=params['seq_stride']  # Pass the max_overlap parameter
        )
        
        print_memory_stats("After train loader:")
        
        # Use same dir for test loader
        test_loader = FastDALIBrainDataLoader(
            params['preaugmented_dir'],
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            split='test',
            device_id=0,  # Assuming single GPU setup
            num_threads=params['dali_num_threads'],
            gpu_prefetch=params['gpu_prefetch'],
            seed=43,  # Different seed for test set
            shuffle=False,
            stride=params['seq_stride']  # Pass the max_overlap parameter
        )
        
        print_memory_stats("After test loader:")
        
        tqdm.write(f"Number of batches in train_loader: {len(train_loader)}")
        tqdm.write(f"Effective Batch Size: {train_loader.total_length//train_loader.steps_per_epoch}")
        tqdm.write(f"Number of batches in test_loader: {len(test_loader)}")
        tqdm.write(f"Effective Batch Size: {test_loader.total_length//test_loader.steps_per_epoch}")
        
        # Create model
        model = GBM(
            mamba_layers=params['mamba_layers'],
            mamba_dim=params['mamba_dim'],
            mamba_state_multiplier=params['mamba_state_multiplier']
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
        
        # Create initial comparison video before training starts
        tqdm.write("Creating initial comparison video with untrained model...")
        video_path = os.path.join(exp_dir, 'videos', 'predictions_initial.mp4')
        create_prediction_video(model, test_loader, video_path, num_frames=330)
        
        # Create optimizer - AdamW with weight decay
        optimizer = AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'], betas=(0.9, 0.95))
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * params['num_epochs']
        warmup_steps = int(total_steps * params['warmup_ratio'])
        
        # Create learning rate scheduler with linear warmup and cosine decay
        lr_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=params['min_lr'])
        
        tqdm.write(f"Using AdamW optimizer with weight decay: {params['weight_decay']}")
        tqdm.write(f"Learning rate scheduler: Linear warmup for {warmup_steps} steps, then cosine decay to {params['min_lr']}")
        tqdm.write(f"Total training steps: {total_steps}")
        
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
        
        # Add for frequent validation tracking
        validation_step_indices = []
        validation_losses = []
        
        # Add running average tracking for loss spike detection
        running_avg_loss = None
        running_avg_alpha = 0.95  # Exponential moving average factor
        
        # Create CUDA streams for overlapping operations
        compute_stream = torch.cuda.Stream()
        copy_stream = torch.cuda.Stream()
        
        # Variables to store previous model and optimizer states
        prev_model_state = None
        prev_optimizer_state = None
        prev_batch_loss = None
        
        # Use GradScaler for gradient stability even in FP32 mode
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Calculate quarter epoch size for video generation
        quarter_epoch_size = len(train_loader) // 4
        
        # Calculate validation frequency in batches
        validation_interval = len(train_loader) // params['validation_per_epoch']
        tqdm.write(f"Validating every {validation_interval} batches ({params['validation_per_epoch']} times per epoch)")
        
        for epoch in range(params['num_epochs']):
            if MEMORY_DIAGNOSTICS:
                print_memory_stats(f"Beginning of epoch {epoch+1}:")
                
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            skipped_batches = 0  # Track number of skipped batches
            reverted_batches = 0  # Track number of reverted batches
            
            # Reset DALI loader
            train_loader.reset()
            train_iter = iter(train_loader)
            
            # Training loop with tqdm for progress display
            train_loop = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{params['num_epochs']}")

            # Checkpoint batch index
            checkpoint_batch_idx = 0

            # Checkpoint per N batches
            checkpoint_batch_every = 64

            # Initialize running average loss
            running_avg_loss = None
            
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
                    
                    # Get logits from model (no sigmoid)
                    predictions = model(batch)
                    # BCE loss
                    loss = model.compute_loss(predictions, batch[:, 1:])
                    
                    # Check if loss is abnormally high compared to running average
                    current_loss = loss.item()
                    
                    # Initialize running average if not set
                    if running_avg_loss is None:
                        running_avg_loss = current_loss
                    
                    # Check if loss exceeds threshold (2x running average)
                    if current_loss > running_avg_loss * 2.0 and running_avg_loss > 0:
                        tqdm.write(f"Loss spike detected at batch {batch_idx+1}: {current_loss:.6f} (> {running_avg_loss:.6f} * 2)")
                        
                        # Check if we have previous states to revert to (not for the first two batches)
                        if prev_model_state is not None and batch_idx > 0:
                            # Revert model to its state from the checkpoint batch
                            tqdm.write(f"Reverting model to state from batch {checkpoint_batch_idx}")
                            model.load_state_dict(prev_model_state)
                            optimizer.load_state_dict(prev_optimizer_state)
                            
                            # Remove the previous batch's loss from our tracking
                            if len(raw_batch_losses) > 0:
                                # Remove the last batch loss as we're reverting its effect
                                removed_loss = raw_batch_losses.pop()
                                # Adjust the epoch loss to remove the effect of the last batch
                                if batch_count > 0:
                                    epoch_loss -= prev_batch_loss
                                    batch_count -= 1
                            
                            # Track reverted batches
                            reverted_batches += 1
                        
                        # Skip the current batch with the loss spike
                        skipped_batches += 1
                        
                        # Update running average with the spike value (with lower weight to reduce impact)
                        running_avg_loss = running_avg_loss * 0.99 + current_loss * 0.01
                        
                        # Record the high loss for monitoring but don't update model
                        raw_batch_losses.append(current_loss)
                        
                        # Update tqdm display with reversion info
                        train_loop.set_postfix(loss=current_loss, avg=running_avg_loss, 
                                               skipped=skipped_batches, reverted=reverted_batches)
                        
                        continue
                    
                    # If we get here, the batch is good. Save the current model and optimizer states
                    # before updating the model, so we can revert if needed next time
                    # Only save state every 32 batches to reduce memory overhead
                    if batch_idx % checkpoint_batch_every == 0:
                        prev_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        prev_optimizer_state = optimizer.state_dict()
                        prev_batch_loss = current_loss
                        checkpoint_batch_idx = batch_idx
                    
                    # Update running average with normal weight
                    running_avg_loss = running_avg_loss * running_avg_alpha + current_loss * (1 - running_avg_alpha)
                    
                    # Use scaler for gradient stability (even in FP32)
                    scaler.scale(loss).backward()
                    
                    # Update weights with gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Using gradient clip value 1.0
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update learning rate scheduler
                    lr_scheduler.step()
                    
                    # Log current learning rate every 500 batches
                    if batch_idx % 128 == 0:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        tqdm.write(f"Current learning rate: {current_lr:.8f}")
                    
                    # Update metrics
                    epoch_loss += current_loss
                    batch_count += 1
                    
                    # Record raw batch loss
                    raw_batch_losses.append(current_loss)
                    
                    # Update tqdm display
                    train_loop.set_postfix(loss=current_loss, avg=running_avg_loss, 
                                          skipped=skipped_batches, reverted=reverted_batches)
                    
                    # Run validation at specified intervals
                    global_step = epoch * len(train_loader) + batch_idx
                    if (batch_idx + 1) % validation_interval == 0 or batch_idx == len(train_loader) - 1:
                        tqdm.write(f"Running validation at step {global_step} (epoch {epoch+1}, batch {batch_idx+1})")
                        
                        # Clean up memory before evaluation
                        torch.cuda.empty_cache()
                        
                        # Switch to evaluation mode
                        model.eval()
                        test_loss = 0.0
                        test_batch_count = 0
                        
                        # Reset test data loader
                        test_loader.reset()
                        test_iter = iter(test_loader)
                        
                        with torch.no_grad():
                            for test_batch_idx in range(len(test_loader)):
                                try:
                                    # Get batch with automatic GPU transfer
                                    test_batch = next(test_iter)
                                    
                                    # Ensure batch is on GPU
                                    if test_batch.device.type != 'cuda':
                                        test_batch = test_batch.cuda(non_blocking=True)
                                    
                                    # Get predictions and calculate loss
                                    test_predictions = model(test_batch)
                                    test_batch_loss = model.compute_loss(test_predictions, test_batch[:, 1:])
                                    
                                    test_loss += test_batch_loss.item()
                                    test_batch_count += 1
                                    
                                except StopIteration:
                                    break
                        
                        # Calculate average test loss
                        if test_batch_count > 0:
                            avg_test_loss = test_loss / test_batch_count
                        else:
                            avg_test_loss = float('inf')
                            
                        # Record validation result
                        validation_step_indices.append(global_step)
                        validation_losses.append(avg_test_loss)
                        
                        # Switch back to training mode
                        model.train()
                        
                        tqdm.write(f"Validation loss at step {global_step}: {avg_test_loss:.6f}")
                        
                        # Update plots with more frequent validation
                        current_avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                        temp_train_losses = train_losses + [current_avg_loss]
                        
                        # Create or update plot
                        plt.figure(figsize=(12, 8))
                        
                        # Plot raw batch losses
                        plt.subplot(2, 1, 1)
                        plt.plot(raw_batch_losses, 'b-', alpha=0.3)
                        plt.title('Raw Batch Losses')
                        plt.xlabel('Batch')
                        plt.ylabel('Loss')
                        plt.grid(True)
                        
                        # Plot train and validation losses
                        plt.subplot(2, 1, 2)
                        
                        # Plot epoch-level train and test losses if available
                        if train_losses:
                            epochs = list(range(1, len(train_losses) + 1))
                            plt.plot(epochs, train_losses, 'b-', label='Train Loss (epoch)')
                        if test_losses:
                            epochs = list(range(1, len(test_losses) + 1))
                            plt.plot(epochs, test_losses, 'r-', label='Test Loss (epoch)')
                        
                        # Plot the more frequent validation losses
                        if validation_losses:
                            # Normalize steps to epoch fraction for plotting
                            epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
                            plt.plot(epoch_fractions, validation_losses, 'g--', marker='o', label='Validation Loss (frequent)')
                        
                        plt.title('Training and Validation Losses')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(exp_dir, 'plots', 'loss_plot.png'))
                        plt.close()
                        
                        # Save validation losses to CSV
                        if validation_losses:
                            save_losses_to_csv(
                                {'step': validation_step_indices,
                                 'epoch': [step / len(train_loader) for step in validation_step_indices],
                                 'validation_loss': validation_losses},
                                os.path.join(exp_dir, 'logs', 'validation_losses.csv')
                            )
                            
                    # Update loss graph and CSV every 128 batches
                    if (len(raw_batch_losses) % 128 == 0):
                        # Update raw batch losses
                        save_losses_to_csv(
                            {'batch': list(range(1, len(raw_batch_losses) + 1)),
                             'raw_loss': raw_batch_losses},
                            os.path.join(exp_dir, 'logs', 'raw_batch_losses.csv')
                        )
                        
                        tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Current Loss = {current_loss:.6f}, "
                                  f"Avg Loss = {running_avg_loss:.6f}, "
                                  f"Skipped = {skipped_batches}, Reverted = {reverted_batches}")
                    
                    # Generate comparison video every quarter epoch
                    if (batch_idx + 1) % quarter_epoch_size == 0:
                        quarter = (batch_idx + 1) // quarter_epoch_size
                        tqdm.write(f"Generating quarter epoch comparison video ({quarter}/4)...")
                        
                        # Clean up memory before video creation
                        torch.cuda.empty_cache()
                        
                        # Create prediction video
                        video_path = os.path.join(exp_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_quarter_{quarter}.mp4')
                        create_prediction_video(model, test_loader, video_path, num_frames=330)
                    
                    # Clean up memory
                    if batch_idx % 50 == 0:
                        # Forced garbage collection every 50 batches to control memory growth
                        torch.cuda.empty_cache()
                        gc.collect()
                
                except StopIteration:
                    break
                    
            # Append running average loss instead of calculating from epoch_loss
            avg_train_loss = running_avg_loss if running_avg_loss is not None else float('inf')
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
            video_path = os.path.join(exp_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_final.mp4')
            create_prediction_video(model, test_loader, video_path, num_frames=330)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'validation_losses': validation_losses,
                    'validation_steps': validation_step_indices,
                    'raw_batch_losses': raw_batch_losses,
                    'params': params,
                    'best_test_loss': best_test_loss,
                }, os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            
            # Create or update final plot with both epoch and frequent validation data
            plt.figure(figsize=(12, 8))
            
            # Plot raw batch losses
            plt.subplot(2, 1, 1)
            plt.plot(raw_batch_losses, 'b-', alpha=0.3)
            plt.title('Raw Batch Losses')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Plot train and validation losses
            plt.subplot(2, 1, 2)
            
            # Plot epoch-level train and test losses
            epochs = list(range(1, len(train_losses) + 1))
            plt.plot(epochs, train_losses, 'b-', label='Train Loss (epoch)')
            plt.plot(epochs, test_losses, 'r-', label='Test Loss (epoch)')
            
            # Plot the more frequent validation losses
            if validation_losses:
                # Normalize steps to epoch fraction for plotting
                epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
                plt.plot(epoch_fractions, validation_losses, 'g--', marker='o', label='Validation Loss (frequent)')
            
            plt.title('Training and Validation Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, 'plots', 'loss_plot.png'))
            plt.close()
            
            # Save all losses to CSV files
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
            
            # Save validation losses
            if validation_losses:
                save_losses_to_csv(
                    {'step': validation_step_indices,
                     'epoch': [step / len(train_loader) for step in validation_step_indices],
                     'validation_loss': validation_losses},
                    os.path.join(exp_dir, 'logs', 'validation_losses.csv')
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
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'validation_losses': validation_losses,
            'validation_steps': validation_step_indices,
            'raw_batch_losses': raw_batch_losses,
            'params': params,
            'final_train_loss': avg_train_loss,
            'final_test_loss': avg_test_loss,
        }, os.path.join(exp_dir, 'checkpoints', 'final_model.pt'))
        
        # Create final prediction video
        video_path = os.path.join(exp_dir, 'videos', 'final_predictions.mp4')
        create_prediction_video(model, test_loader, video_path, num_frames=330)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 