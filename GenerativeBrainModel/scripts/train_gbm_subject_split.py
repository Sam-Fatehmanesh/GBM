#!/usr/bin/env python3
"""
Train GBM model with a two-phase approach: pretrain on all subjects except the target, 
then finetune on the target subject only.

This script extends train_with_fast_dali_original_recipe.py to implement
subject-specific training after pretraining on a broader dataset.
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
import argparse
import shutil
import tempfile
from typing import List, Optional, Dict, Any, Tuple

# Set sharing strategy to file_system to avoid shared memory issues
mp.set_sharing_strategy('file_system')
# Set the start method to spawn to avoid CUDA initialization issues
mp.set_start_method('spawn', force=True)

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
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
import pdb

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

#--------------------------
# Helper Functions
#--------------------------

def print_memory_stats(prefix=""):
    """Print GPU memory usage statistics"""
    if MEMORY_DIAGNOSTICS:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"{prefix} GPU Memory: Total={t/1e9:.2f}GB, Reserved={r/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free={f/1e9:.2f}GB")

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

def create_experiment_dir(base_path, phase_name):
    """Create a phase experiment directory with all necessary subdirectories
    
    Args:
        base_path: Base path for the experiment (e.g., 'experiments/gbm/timestamp')
        phase_name: Name of the phase (e.g., 'pretrain', 'finetune')
        
    Returns:
        phase_dir: Path to the created phase directory
    """
    # Create phase directory
    phase_dir = os.path.join(base_path, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    os.makedirs(os.path.join(phase_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'logs'), exist_ok=True)
    
    return phase_dir

def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

def create_prediction_video(model, data_loader, output_path, num_frames=330):
    """Create a video of model predictions vs actual brain activity with color-coded bottom right panel."""
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
        with autocast():
            # Get probability predictions from the model
            predictions = model.get_predictions(batch)
            
            # Generate sampled binary predictions from the predicted probability distributions
            sampled_predictions = model.sample_binary_predictions(predictions)
    
    # Convert predictions to numpy for visualization
    # We'll choose multiple sequences from the batch for visualization (up to 100)
    num_sequences = min(100, batch.size(0))
    rand_indices = torch.randperm(batch.size(0))[:num_sequences]
    
    # Create color-coded comparison visualization
    create_color_coded_comparison_video(
        actual=batch_viz[rand_indices].cpu().numpy(),
        predicted=predictions[rand_indices].cpu().numpy(),
        sampled_predictions=sampled_predictions[rand_indices].cpu().numpy(),
        output_path=output_path,
        num_frames=num_frames,
        fps=1
    )

def create_color_coded_comparison_video(actual, predicted, sampled_predictions, output_path, num_frames=330, fps=1):
    """Create video comparing actual brain activity with model predictions, with color-coded bottom right panel.
    
    Args:
        actual: numpy array of shape (batch_size, seq_len, 256, 128) with actual data
        predicted: numpy array of shape (batch_size, seq_len-1, 256, 128) with predictions
        sampled_predictions: numpy array of shape (batch_size, seq_len-1, 256, 128) with sampled binary predictions
        output_path: Path to save the video
        num_frames: Maximum sequence length to use (default: 330 for full sequences)
        fps: Frames per second for the video (default: 1)
    """
    try:
        # Set up video parameters
        width = 128
        height = 256
        scale = 2
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 2x2 grid layout
        video_width = scaled_width * 2
        video_height = scaled_height * 2
        
        # Use H264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (video_width, video_height), 
                            isColor=True)
        
        if not out.isOpened():
            # Try an alternative codec silently
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (video_width, video_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer. Skipping video creation.")
                return
        
        # Process each sequence in the batch
        batch_size = min(actual.shape[0], 100)  # Limit to at most 100 sequences
        total_frames = 0
        max_total_frames = 1000  # Safety limit to prevent extremely large videos
        
        for batch_idx in range(batch_size):
            # Get current sequence
            current_seq = actual[batch_idx]
            predicted_seq = predicted[batch_idx]
            sampled_seq = sampled_predictions[batch_idx]
            
            # Use full sequence length, limited by available frames
            seq_len = min(current_seq.shape[0]-1, predicted_seq.shape[0], num_frames)
            
            # Check if adding this sequence would exceed the max frames limit
            frames_left = max_total_frames - total_frames
            if frames_left <= 0:
                # We've reached the maximum, stop adding sequences
                break
                
            # If this sequence would exceed the limit, truncate it
            if seq_len > frames_left:
                seq_len = frames_left
            
            for t in range(seq_len):
                try:
                    # Get current frame, prediction, and ground truth next frame
                    current = current_seq[t]
                    next_frame = current_seq[t+1]  # Ground truth
                    prediction = predicted_seq[t]   # Model prediction (probabilities)
                    sampled = sampled_seq[t]  # Sampled binary prediction
                    
                    # Ensure values are in appropriate ranges
                    if current.dtype == np.uint8:
                        current = current.astype(np.float32) / 255.0
                    if next_frame.dtype == np.uint8:
                        next_frame = next_frame.astype(np.float32) / 255.0
                    if prediction.dtype == np.uint8:
                        prediction = prediction.astype(np.float32) / 255.0
                    if sampled.dtype == np.uint8:
                        sampled = sampled.astype(np.float32) / 255.0
                    
                    # Convert binary values to 0/1 for classification
                    sampled_binary = (sampled > 0.5).astype(np.uint8)
                    actual_binary = (next_frame > 0.5).astype(np.uint8)
                    
                    # Debug: ensure we have the right dimensions
                    if current.shape != (height, width):
                        tqdm.write(f"Warning: Unexpected data shape {current.shape}, expected ({height}, {width})")
                    
                    # Create color-coded image for bottom right panel
                    # Green: True positive (predicted=1, actual=1)
                    # Red: False positive (predicted=1, actual=0)  
                    # Black: True negative and False negative (predicted=0 or actual=0)
                    color_coded = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Ensure masks have the same shape as the color_coded array's spatial dimensions
                    assert sampled_binary.shape == (height, width), f"sampled_binary shape {sampled_binary.shape} doesn't match expected ({height}, {width})"
                    assert actual_binary.shape == (height, width), f"actual_binary shape {actual_binary.shape} doesn't match expected ({height}, {width})"
                    
                    # True positives: Green
                    tp_mask = (sampled_binary == 1) & (actual_binary == 1)
                    color_coded[tp_mask] = [0, 255, 0]  # Green in BGR
                    
                    # False positives: Red
                    fp_mask = (sampled_binary == 1) & (actual_binary == 0)
                    color_coded[fp_mask] = [0, 0, 255]  # Red in BGR
                    
                    # True negatives and False negatives: Black (already initialized to black)
                    # No need to set these explicitly as they're already [0, 0, 0]
                    
                    # Convert other images to uint8
                    curr_img = (current * 255).astype(np.uint8)
                    pred_img = (prediction * 255).astype(np.uint8)
                    next_img = (next_frame * 255).astype(np.uint8)
                    
                    # Scale up images
                    curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    next_img = cv2.resize(next_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    color_coded_scaled = cv2.resize(color_coded, (scaled_width, scaled_height), 
                                                  interpolation=cv2.INTER_NEAREST)
                    
                    # Convert grayscale images to RGB
                    curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                    next_rgb = cv2.cvtColor(next_img, cv2.COLOR_GRAY2BGR)
                    # color_coded_scaled is already in BGR format
                    
                    # Add text labels
                    cv2.putText(curr_rgb, 'Current', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(pred_rgb, 'Predicted Spike Probabilities', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(next_rgb, 'Actual Next', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(color_coded_scaled, 'Predictions: TP=Green, FP=Red', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                    
                    # Add frame number and sequence info
                    cv2.putText(curr_rgb, f'Seq {batch_idx+1}/{batch_size}, Frame {t+1}/{seq_len}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Create 2x2 grid: current | predicted
                    #                  actual  | color_coded
                    top_row = np.hstack([curr_rgb, pred_rgb])
                    bottom_row = np.hstack([next_rgb, color_coded_scaled])
                    combined = np.vstack([top_row, bottom_row])
                    
                    out.write(combined)
                    total_frames += 1
                    
                except Exception as frame_error:
                    tqdm.write(f"Error processing frame {t} in sequence {batch_idx}: {str(frame_error)}")
                    # Skip this frame and continue
                    continue
            
            # Check if we've reached the maximum frames
            if total_frames >= max_total_frames:
                tqdm.write(f"Reached maximum frame limit ({max_total_frames}). Truncating video.")
                break
        
        out.release()
        # Simplify the output message
        tqdm.write(f"Color-coded video saved: {os.path.basename(output_path)} ({total_frames} frames, {batch_size} sequences)")
        
        # Verify the video file was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # If file is very small, it likely failed
                tqdm.write(f"Warning: Video file is unusually small ({file_size} bytes), generation may have failed")
        else:
            tqdm.write(f"Warning: Video file was not created at {output_path}")
        
    except Exception as e:
        tqdm.write(f"Error creating color-coded video: {str(e)}")
        import traceback
        traceback.print_exc()

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
        else:
            return 1.0
        
        # Cosine annealing after warmup with decay to min_lr
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between 1.0 and min_lr_factor instead of between 1.0 and 0.0
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

#--------------------------
# Subject-Filtered DataLoader
#--------------------------

class SubjectFilteredFastDALIBrainDataLoader(FastDALIBrainDataLoader):
    """FastDALIBrainDataLoader that can filter subjects based on include/exclude lists."""
    
    def __init__(
        self,
        preaugmented_dir: str,
        include_subjects: Optional[List[str]] = None,
        exclude_subjects: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the SubjectFilteredFastDALIBrainDataLoader.
        
        Args:
            preaugmented_dir: Directory containing preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            **kwargs: Additional arguments to pass to FastDALIBrainDataLoader
        """
        self.original_dir = preaugmented_dir
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects if exclude_subjects else []
        
        # Create a temporary directory to store the filtered subject directories
        self.temp_dir = None
        filtered_dir = self._create_filtered_subject_dir(preaugmented_dir, include_subjects, exclude_subjects)
        
        # Initialize the parent class with the filtered directory
        super().__init__(filtered_dir, **kwargs)
        
        # Print subjects being used
        if include_subjects:
            tqdm.write(f"Using only subjects: {include_subjects}")
        if exclude_subjects:
            tqdm.write(f"Excluding subjects: {exclude_subjects}")
    
    def _create_filtered_subject_dir(self, preaugmented_dir, include_subjects, exclude_subjects):
        """Create a temporary directory with symlinks to only the desired subject directories.
        
        Args:
            preaugmented_dir: Original directory containing all preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            
        Returns:
            filtered_dir: Path to a temporary directory containing only the filtered subjects
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="filtered_subjects_")
        
        # Get all subjects in the preaugmented directory
        all_subjects = []
        for subject_dir in os.listdir(preaugmented_dir):
            subject_path = os.path.join(preaugmented_dir, subject_dir)
            if os.path.isdir(subject_path):
                # Check if this is a valid subject directory (contains preaugmented_grids.h5)
                grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
                if os.path.exists(grid_file):
                    all_subjects.append(subject_dir)
        
        # Filter subjects based on include/exclude lists
        if include_subjects is not None:
            filtered_subjects = [s for s in all_subjects if s in include_subjects]
        else:
            filtered_subjects = [s for s in all_subjects if s not in exclude_subjects]
        
        # If no subjects remain after filtering, raise an error
        if not filtered_subjects:
            raise ValueError("No subjects left after filtering!")
        
        # Create symlinks to the filtered subject directories
        for subject in filtered_subjects:
            # Use absolute path for the source to ensure symlinks work correctly
            src_path = os.path.abspath(os.path.join(preaugmented_dir, subject))
            dst_path = os.path.join(self.temp_dir, subject)
            os.symlink(src_path, dst_path)
        
        tqdm.write(f"Created filtered directory with {len(filtered_subjects)} subjects: {filtered_subjects}")
        
        return self.temp_dir
    
    def __del__(self):
        """Clean up temporary directory when the object is deleted."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Delete symlinks but not the actual data
                for item in os.listdir(self.temp_dir):
                    os.unlink(os.path.join(self.temp_dir, item))
                # Remove the temporary directory
                os.rmdir(self.temp_dir)
            except Exception as e:
                # Print but don't fail if cleanup encounters an error
                print(f"Warning: Error during cleanup of temporary directory: {e}")

#--------------------------
# Phase Training Function
#--------------------------

def run_phase(
    phase_name: str,
    preaugmented_dir: str,
    params: Dict[str, Any],
    exp_root: str,
    subjects_include: Optional[List[str]] = None,
    subjects_exclude: Optional[List[str]] = None,
    init_checkpoint: Optional[str] = None
) -> str:
    """Run a single training phase (pretrain or finetune).
    
    Args:
        phase_name: Name of the phase ('pretrain' or 'finetune')
        preaugmented_dir: Directory containing preaugmented data
        params: Dictionary of training parameters
        exp_root: Root experiment directory
        subjects_include: List of subject names to include. If None, all subjects are included.
        subjects_exclude: List of subject names to exclude. If None, no subjects are excluded.
        init_checkpoint: Path to a checkpoint to initialize the model from
        
    Returns:
        best_checkpoint_path: Path to the best checkpoint from this phase
    """
    tqdm.write(f"\n{'='*20} Starting {phase_name.upper()} phase {'='*20}\n")
    
    # Create phase directory
    phase_dir = create_experiment_dir(exp_root, phase_name)
    tqdm.write(f"Saving {phase_name} results to: {phase_dir}")
    
    # Print initial memory stats
    print_memory_stats(f"Initial ({phase_name}):")
    
    # Get maximum number of z-planes from preaugmented data
    max_z_planes = get_max_z_planes(preaugmented_dir)
    params['seq_len'] = int(params['timesteps_per_sequence'] * max_z_planes)    
    params['seq_stride'] = int(params['timestep_stride'] * max_z_planes)
    
    tqdm.write(f"Maximum z-planes across subjects: {max_z_planes}")
    tqdm.write(f"Total sequence length: {params['seq_len']} ({params['timesteps_per_sequence']} timepoints × {max_z_planes} z-planes)")
    
    # Create DALI data loaders with the SubjectFilteredFastDALIBrainDataLoader
    train_loader = SubjectFilteredFastDALIBrainDataLoader(
        preaugmented_dir,
        include_subjects=subjects_include,
        exclude_subjects=subjects_exclude,
        batch_size=params['batch_size'],
        seq_len=params['seq_len'],
        split='train',
        device_id=0,  # Assuming single GPU setup
        num_threads=params['dali_num_threads'],
        gpu_prefetch=params['gpu_prefetch'],
        seed=42,
        shuffle=True,
        stride=params['seq_stride']
    )
    
    print_memory_stats(f"After train loader ({phase_name}):")
    
    # Use same subject filtering for test loader
    test_loader = SubjectFilteredFastDALIBrainDataLoader(
        preaugmented_dir,
        include_subjects=subjects_include,
        exclude_subjects=subjects_exclude,
        batch_size=params['batch_size'],
        seq_len=params['seq_len'],
        split='test',
        device_id=0,  # Assuming single GPU setup
        num_threads=params['dali_num_threads'],
        gpu_prefetch=params['gpu_prefetch'],
        seed=43,  # Different seed for test set
        shuffle=False,
        stride=params['seq_stride']
    )
    
    print_memory_stats(f"After test loader ({phase_name}):")
    
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
    
    # If init_checkpoint is provided, load weights from it
    if init_checkpoint:
        tqdm.write(f"Initializing model from checkpoint: {init_checkpoint}")
        try:
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            tqdm.write("Successfully loaded model weights from checkpoint.")
        except Exception as e:
            tqdm.write(f"Error loading checkpoint: {str(e)}")
            if phase_name == 'finetune':
                tqdm.write("WARNING: Starting finetuning from scratch!")
    
    # Save model architecture and parameters
    with open(os.path.join(phase_dir, "model_architecture.txt"), "w") as f:
        f.write(f"{phase_name.upper()} Phase Model Architecture:\n")
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
        
        # Add subject filtering info
        f.write(f"\nSubject Filtering:\n")
        if subjects_include:
            f.write(f"Including only subjects: {subjects_include}\n")
        if subjects_exclude:
            f.write(f"Excluding subjects: {subjects_exclude}\n")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")
    model = model.to(device)
    
    print_memory_stats(f"After model creation ({phase_name}):")
    
    # Create initial comparison video before training starts
    tqdm.write(f"Creating initial comparison video for {phase_name}...")
    video_path = os.path.join(phase_dir, 'videos', 'predictions_initial.mp4')
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
    tqdm.write(f"Starting {phase_name} training...")
    train_losses = []
    test_losses = []
    raw_batch_losses = []  # Track individual batch losses
    best_test_loss = float('inf')
    
    # Add for frequent validation tracking
    validation_step_indices = []
    validation_losses = []
    # Add F1 score tracking for validation and test
    validation_f1_scores = []
    test_f1_scores = []
    # Add recall and precision tracking for validation and test
    validation_recall_scores = []
    validation_precision_scores = []
    test_recall_scores = []
    test_precision_scores = []
    
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
    scaler = GradScaler()
    
    # Calculate quarter epoch size for video generation
    quarter_epoch_size = len(train_loader) // 4
    
    # Calculate validation frequency in batches
    validation_interval = len(train_loader) // params['validation_per_epoch']
    tqdm.write(f"Validating every {validation_interval} batches ({params['validation_per_epoch']} times per epoch)")
    
    # Path to save the best model checkpoint
    best_model_checkpoint_path = os.path.join(phase_dir, 'checkpoints', 'best_model.pt')
    
    for epoch in range(params['num_epochs']):
        if MEMORY_DIAGNOSTICS:
            print_memory_stats(f"Beginning of epoch {epoch+1} ({phase_name}):")
            
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        skipped_batches = 0  # Track number of skipped batches
        reverted_batches = 0  # Track number of reverted batches
        
        # Reset DALI loader
        train_loader.reset()
        train_iter = iter(train_loader)
        
        # Training loop with tqdm for progress display
        train_loop = tqdm(range(len(train_loader)), desc=f"{phase_name.capitalize()} Epoch {epoch+1}/{params['num_epochs']}")

        # Checkpoint batch index
        checkpoint_batch_idx = 0

        # Checkpoint per N batches
        checkpoint_batch_every = 64

        # Initialize running average loss
        running_avg_loss = None
        
        for batch_idx in train_loop:
            if MEMORY_DIAGNOSTICS:
                print_memory_stats(f"Before batch {batch_idx+1} ({phase_name}):")
            
            try:
                # Get batch with automatic GPU transfer
                batch = next(train_iter)
                
                # Ensure batch is on GPU
                if batch.device.type != 'cuda':
                    batch = batch.cuda(non_blocking=True)
                
                # Process batch - model handles dtype conversion internally
                optimizer.zero_grad()
                
                # Get logits from model (no sigmoid) with autocast for mixed precision
                with autocast():
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
                del loss, predictions, batch
                
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
                    
                    
                    # Switch to evaluation mode
                    model.eval()
                    test_loss = 0.0
                    test_batch_count = 0
                    total_tp, total_fp, total_fn = 0, 0, 0
                    
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
                                
                                # Get logits from model (no sigmoid)
                                with autocast():
                                    test_predictions = model(test_batch)
                                    test_batch_loss = model.compute_loss(test_predictions, test_batch[:, 1:])
                                
                                test_loss += test_batch_loss.item()
                                test_batch_count += 1
                                del test_batch_loss
                                
                                # Calculate F1 metrics using sigmoid on test_predictions
                                probs = torch.sigmoid(test_predictions)
                                del test_predictions
                                # Keep predictions as bool to save memory (1 byte vs 4 bytes for int)
                                preds = (probs > 0.5)
                                del probs
                                # Keep targets as bool as well
                                targets = test_batch[:, 1:].bool()
                                del test_batch
                                torch.cuda.empty_cache()
                                # Use boolean operations for memory efficiency
                                total_tp += (preds & targets).sum().item()
                                total_fp += (preds & ~targets).sum().item()
                                total_fn += (~preds & targets).sum().item()
                                # Clean up
                                del preds, targets
                            except StopIteration:
                                break
                    
                    
                    torch.cuda.empty_cache()
                    # Calculate average test loss
                    if test_batch_count > 0:
                        avg_test_loss = test_loss / test_batch_count
                    else:
                        avg_test_loss = float('inf')
                        
                    # Record validation result
                    validation_step_indices.append(global_step)
                    validation_losses.append(avg_test_loss)
                    
                    # Calculate F1 score
                    if total_tp + total_fp + total_fn > 0:
                        val_f1 = 2 * total_tp / float(2 * total_tp + total_fp + total_fn)
                    else:
                        val_f1 = 0.0
                    validation_f1_scores.append(val_f1)
                    tqdm.write(f"Validation F1 at step {global_step}: {val_f1:.6f}")
                    
                    # Calculate Recall and Precision
                    if total_tp + total_fn > 0:
                        val_recall = total_tp / float(total_tp + total_fn)
                    else:
                        val_recall = 0.0
                    
                    if total_tp + total_fp > 0:
                        val_precision = total_tp / float(total_tp + total_fp)
                    else:
                        val_precision = 0.0
                    
                    validation_recall_scores.append(val_recall)
                    validation_precision_scores.append(val_precision)
                    tqdm.write(f"Validation Recall at step {global_step}: {val_recall:.6f}")
                    tqdm.write(f"Validation Precision at step {global_step}: {val_precision:.6f}")
                    
                    # Switch back to training mode
                    model.train()
                    
                    tqdm.write(f"Validation loss at step {global_step}: {avg_test_loss:.6f}")
                    
                    # Update plots with more frequent validation
                    current_avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                    temp_train_losses = train_losses + [current_avg_loss]
                    
                    # Create or update plot
                    plt.figure(figsize=(12, 12))
                    
                    # Plot raw batch losses
                    plt.subplot(3, 1, 1)
                    plt.plot(raw_batch_losses, 'b-', alpha=0.3)
                    plt.title(f'{phase_name.capitalize()} Phase - Raw Batch Losses')
                    plt.xlabel('Batch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    
                    # Plot train and validation losses
                    plt.subplot(3, 1, 2)
                    
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
                    
                    plt.title(f'{phase_name.capitalize()} Phase - Training and Validation Losses')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(True)
                    
                    # Add subplot for F1 scores
                    plt.subplot(3, 1, 3)
                    
                    # Plot validation metrics
                    if validation_f1_scores:
                        epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
                        plt.plot(epoch_fractions, validation_f1_scores, 'm--', marker='x', label='Validation F1')
                        
                    if validation_recall_scores:
                        epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
                        plt.plot(epoch_fractions, validation_recall_scores, 'g--', marker='o', label='Validation Recall')
                        
                    if validation_precision_scores:
                        epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
                        plt.plot(epoch_fractions, validation_precision_scores, 'b--', marker='^', label='Validation Precision')
                    
                    # Plot test metrics if available
                    if test_f1_scores:
                        epochs = list(range(1, len(test_f1_scores) + 1))
                        plt.plot(epochs, test_f1_scores, 'k-', marker='s', label='Test F1')
                        
                    if test_recall_scores:
                        epochs = list(range(1, len(test_recall_scores) + 1))
                        plt.plot(epochs, test_recall_scores, 'r-', marker='d', label='Test Recall')
                        
                    if test_precision_scores:
                        epochs = list(range(1, len(test_precision_scores) + 1))
                        plt.plot(epochs, test_precision_scores, 'c-', marker='v', label='Test Precision')
                    
                    plt.title(f'{phase_name.capitalize()} Phase - Performance Metrics (F1, Recall, Precision)')
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.ylim(0, 1)  # All metrics range from 0 to 1
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(phase_dir, 'plots', 'loss_plot.png'))
                    plt.close()
                    
                    # Save validation losses to CSV
                    if validation_losses:
                        save_losses_to_csv(
                            {'step': validation_step_indices,
                             'epoch': [step / len(train_loader) for step in validation_step_indices],
                             'validation_loss': validation_losses,
                             'validation_f1': validation_f1_scores,
                             'validation_recall': validation_recall_scores,
                             'validation_precision': validation_precision_scores},
                            os.path.join(phase_dir, 'logs', 'validation_losses.csv')
                        )
                    
                    # Save best model if this is the best validation loss
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
                            'phase': phase_name,
                        }, best_model_checkpoint_path)
                        tqdm.write(f"Saved new best model with validation loss: {best_test_loss:.6f}")
                            
                # Update loss graph and CSV every 128 batches
                if (len(raw_batch_losses) % 128 == 0):
                    # Update raw batch losses
                    save_losses_to_csv(
                        {'batch': list(range(1, len(raw_batch_losses) + 1)),
                         'raw_loss': raw_batch_losses},
                        os.path.join(phase_dir, 'logs', 'raw_batch_losses.csv')
                    )
                    
                tqdm.write(f"{phase_name.capitalize()} Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Current Loss = {current_loss:.6f}, "
                          f"Avg Loss = {running_avg_loss:.6f}, "
                          f"Skipped = {skipped_batches}, Reverted = {reverted_batches}")
                
                # Generate comparison video every quarter epoch
                if (batch_idx + 1) % quarter_epoch_size == 0:
                    quarter = (batch_idx + 1) // quarter_epoch_size
                    tqdm.write(f"Generating quarter epoch comparison video ({quarter}/4)...")
                    
                    # Clean up memory before video creation
                    torch.cuda.empty_cache()
                    
                    # Create prediction video
                    video_path = os.path.join(phase_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_quarter_{quarter}.mp4')
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
            print_memory_stats(f"Before evaluation, epoch {epoch+1} ({phase_name}):")
        
        # Testing
        model.eval()
        test_loss = 0.0
        test_batch_count = 0
        
        # Reset DALI loader for testing
        test_loader.reset()
        test_iter = iter(test_loader)
        with torch.no_grad():
            test_loss = 0.0
            test_batch_count = 0
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for test_batch_idx in range(len(test_loader)):
                try:
                    # Get batch with automatic GPU transfer
                    batch = next(test_iter)
                    
                    # Ensure batch is on GPU
                    if batch.device.type != 'cuda':
                        batch = batch.cuda(non_blocking=True)
                    
                    # Get logits from model (no sigmoid)
                    with autocast():
                        predictions = model(batch)
                        # Loss function handles type conversion
                        loss = model.compute_loss(predictions, batch[:, 1:])
                    
                    test_loss += loss.item()
                    test_batch_count += 1
                    
                    # Calculate F1 metrics using sigmoid on predictions
                    probs = torch.sigmoid(predictions)
                    del predictions
                    # Keep predictions as bool to save memory (1 byte vs 4 bytes for int)
                    preds = (probs > 0.5)
                    del probs
                    # Keep targets as bool as well
                    targets = batch[:, 1:].bool()
                    del batch
                    torch.cuda.empty_cache()
                    # Use boolean operations for memory efficiency
                    total_tp += (preds & targets).sum().item()
                    total_fp += (preds & ~targets).sum().item()
                    total_fn += (~preds & targets).sum().item()
                    # Clean up
                    del preds, targets
                except StopIteration:
                    break
        
        # Calculate average test loss
        if test_batch_count > 0:
            avg_test_loss = test_loss / test_batch_count
        else:
            avg_test_loss = float('inf')
        test_losses.append(avg_test_loss)
        
        # Calculate F1 score
        if total_tp + total_fp + total_fn > 0:
            epoch_f1 = 2 * total_tp / float(2 * total_tp + total_fp + total_fn)
        else:
            epoch_f1 = 0.0
        test_f1_scores.append(epoch_f1)
        tqdm.write(f"Test F1 at epoch {epoch+1}: {epoch_f1:.6f}")
        
        # Calculate Recall and Precision
        if total_tp + total_fn > 0:
            epoch_recall = total_tp / float(total_tp + total_fn)
        else:
            epoch_recall = 0.0
        
        if total_tp + total_fp > 0:
            epoch_precision = total_tp / float(total_tp + total_fp)
        else:
            epoch_precision = 0.0
        
        test_recall_scores.append(epoch_recall)
        test_precision_scores.append(epoch_precision)
        tqdm.write(f"Test Recall at epoch {epoch+1}: {epoch_recall:.6f}")
        tqdm.write(f"Test Precision at epoch {epoch+1}: {epoch_precision:.6f}")
        
        # Clean up memory before video creation
        torch.cuda.empty_cache()
        if MEMORY_DIAGNOSTICS:
            print_memory_stats(f"Before video creation, epoch {epoch+1} ({phase_name}):")
        
        # Create prediction video after each epoch
        video_path = os.path.join(phase_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_final.mp4')
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
                'phase': phase_name,
            }, best_model_checkpoint_path)
            tqdm.write(f"Saved new best model with test loss: {best_test_loss:.6f}")
        
        # Create or update final plot with both epoch and frequent validation data
        plt.figure(figsize=(12, 12))
        
        # Plot raw batch losses
        plt.subplot(3, 1, 1)
        plt.plot(raw_batch_losses, 'b-', alpha=0.3)
        plt.title(f'{phase_name.capitalize()} Phase - Raw Batch Losses')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot train and validation losses
        plt.subplot(3, 1, 2)
        
        # Plot epoch-level train and test losses
        epochs = list(range(1, len(train_losses) + 1))
        plt.plot(epochs, train_losses, 'b-', label='Train Loss (epoch)')
        plt.plot(epochs, test_losses, 'r-', label='Test Loss (epoch)')
        
        # Plot the more frequent validation losses
        if validation_losses:
            # Normalize steps to epoch fraction for plotting
            epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
            plt.plot(epoch_fractions, validation_losses, 'g--', marker='o', label='Validation Loss (frequent)')
        
        plt.title(f'{phase_name.capitalize()} Phase - Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Add subplot for F1 scores
        plt.subplot(3, 1, 3)
        
        # Plot validation metrics
        if validation_f1_scores:
            epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
            plt.plot(epoch_fractions, validation_f1_scores, 'm--', marker='x', label='Validation F1')
            
        if validation_recall_scores:
            epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
            plt.plot(epoch_fractions, validation_recall_scores, 'g--', marker='o', label='Validation Recall')
            
        if validation_precision_scores:
            epoch_fractions = [step / len(train_loader) for step in validation_step_indices]
            plt.plot(epoch_fractions, validation_precision_scores, 'b--', marker='^', label='Validation Precision')
        
        # Plot test metrics if available
        if test_f1_scores:
            epochs = list(range(1, len(test_f1_scores) + 1))
            plt.plot(epochs, test_f1_scores, 'k-', marker='s', label='Test F1')
            
        if test_recall_scores:
            epochs = list(range(1, len(test_recall_scores) + 1))
            plt.plot(epochs, test_recall_scores, 'r-', marker='d', label='Test Recall')
            
        if test_precision_scores:
            epochs = list(range(1, len(test_precision_scores) + 1))
            plt.plot(epochs, test_precision_scores, 'c-', marker='v', label='Test Precision')
        
        plt.title(f'{phase_name.capitalize()} Phase - Performance Metrics (F1, Recall, Precision)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.ylim(0, 1)  # All metrics range from 0 to 1
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(phase_dir, 'plots', 'loss_plot.png'))
        plt.close()
        
        # Save all losses to CSV files
        save_losses_to_csv(
            {'epoch': list(range(1, len(train_losses) + 1)),
             'train_loss': train_losses,
             'test_loss': test_losses,
             'test_f1': test_f1_scores,
             'test_recall': test_recall_scores,
             'test_precision': test_precision_scores},
            os.path.join(phase_dir, 'logs', 'losses.csv')
        )
        
        # Save raw batch losses separately
        if raw_batch_losses:
            save_losses_to_csv(
                {'batch': list(range(1, len(raw_batch_losses) + 1)),
                 'raw_loss': raw_batch_losses},
                os.path.join(phase_dir, 'logs', 'raw_batch_losses.csv')
            )
        
        # Save validation losses
        if validation_losses:
            save_losses_to_csv(
                {'step': validation_step_indices,
                 'epoch': [step / len(train_loader) for step in validation_step_indices],
                 'validation_loss': validation_losses,
                 'validation_f1': validation_f1_scores,
                 'validation_recall': validation_recall_scores,
                 'validation_precision': validation_precision_scores},
                os.path.join(phase_dir, 'logs', 'validation_losses.csv')
            )
        
        tqdm.write(f"{phase_name.capitalize()} Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                  f"Test Loss = {avg_test_loss:.6f}")
    
    # Stop DALI profiler if active
    if has_profiler:
        profiler.stop()
        profile_path = os.path.join(phase_dir, 'logs', 'dali_profile.json')
        profiler.export(profile_path)
        
    tqdm.write(f"{phase_name.capitalize()} training complete!")
    
    # Save final model and diagnostics
    final_model_path = os.path.join(phase_dir, 'checkpoints', 'final_model.pt')
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
        'phase': phase_name,
    }, final_model_path)
    
    # Create final prediction video
    video_path = os.path.join(phase_dir, 'videos', 'final_predictions.mp4')
    create_prediction_video(model, test_loader, video_path, num_frames=330)
    
    # Save test data and predictions if this is the finetuning phase
    if phase_name == "finetune":
        tqdm.write("Saving test data and predictions for analysis...")
        save_data_dir = os.path.join(phase_dir, 'test_data')
        os.makedirs(save_data_dir, exist_ok=True)
        save_test_data_and_predictions(model, test_loader, save_data_dir, num_samples=100, params=params)
    
    # Return path to best model checkpoint for potential use in next phase
    return best_model_checkpoint_path

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

#--------------------------
# Main Function
#--------------------------

def main():
    """Main function to run the two-phase training process."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train GBM with a two-phase approach: pretrain on all subjects except target, then finetune on target")
        parser.add_argument("--preaugmented-dir", type=str, default="preaugmented_training_spike_data_2018", 
                            help="Directory containing preaugmented data")
        parser.add_argument("--target-subject", type=str, required=True,
                            help="Name of the target subject to hold out for finetuning")
        parser.add_argument("--num-epochs-pretrain", type=int, default=1,
                            help="Number of epochs for pretraining phase")
        parser.add_argument("--num-epochs-finetune", type=int, default=1, 
                            help="Number of epochs for finetuning phase")
        parser.add_argument("--batch-size", type=int, default=128,
                            help="Batch size for both phases")
        parser.add_argument("--learning-rate", type=float, default=6e-4,
                            help="Learning rate for both phases")
        parser.add_argument("--skip-pretrain", action="store_true",
                            help="Skip the pretrain phase and go directly to finetuning")
        parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                            help="Path to a pretrained checkpoint to start from (skips pretrain phase)")
        args = parser.parse_args()
        
        # Validate target subject exists
        target_subject_path = os.path.join(args.preaugmented_dir, args.target_subject)
        #pdb.set_trace()
        if not os.path.exists(target_subject_path) or not os.path.isdir(target_subject_path):
            raise ValueError(f"Target subject '{args.target_subject}' not found in {args.preaugmented_dir}")
        
        grid_file = os.path.join(target_subject_path, 'preaugmented_grids.h5')
        if not os.path.exists(grid_file):
            raise ValueError(f"Target subject '{args.target_subject}' does not have preaugmented_grids.h5 file")
        
        tqdm.write(f"Target subject validated: {args.target_subject}")
        
        # Base parameters for both phases
        base_params = {
            'preaugmented_dir': args.preaugmented_dir,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': 0.1,
            'warmup_ratio': 0.1,
            'min_lr': 1e-5,
            'mamba_layers': 8,
            'mamba_dim': 1024,
            'mamba_state_multiplier': 8,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95,
            'dali_num_threads': 2,
            'gpu_prefetch': 1,
            'use_float16': False,
            'seed': seed,
            'validation_per_epoch': 8,
            'timestep_stride': 1/3,
        }
        
        # Set phase-specific parameters
        pretrain_params = base_params.copy()
        pretrain_params['num_epochs'] = args.num_epochs_pretrain
        
        finetune_params = base_params.copy()
        finetune_params['num_epochs'] = args.num_epochs_finetune
        
        # Create timestamped experiment root directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_root = os.path.join('experiments', 'gbm', timestamp)
        os.makedirs(exp_root, exist_ok=True)
        
        # Save experiment metadata
        with open(os.path.join(exp_root, "experiment_info.txt"), "w") as f:
            f.write(f"Two-Phase GBM Training Experiment\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Target Subject: {args.target_subject}\n")
            f.write(f"Pretraining Epochs: {args.num_epochs_pretrain}\n")
            f.write(f"Finetuning Epochs: {args.num_epochs_finetune}\n\n")
            f.write(f"Base Parameters:\n")
            for key, value in base_params.items():
                f.write(f"  {key}: {value}\n")
        
        # Initialize pretrain checkpoint path
        pretrain_checkpoint = args.pretrain_checkpoint
        
        # Phase 1: Pretraining on all subjects except the target
        if not args.skip_pretrain and args.pretrain_checkpoint is None:
            tqdm.write(f"Running pretraining phase on all subjects except '{args.target_subject}'...")
            
            pretrain_checkpoint = run_phase(
                phase_name="pretrain",
                preaugmented_dir=args.preaugmented_dir,
                params=pretrain_params,
                exp_root=exp_root,
                subjects_exclude=[args.target_subject],
                subjects_include=None,
                init_checkpoint=None
            )
        elif args.pretrain_checkpoint:
            tqdm.write(f"Skipping pretraining, using provided checkpoint: {args.pretrain_checkpoint}")
        else:
            tqdm.write(f"Skipping pretraining phase as requested.")
            pretrain_checkpoint = None
        
        # Phase 2: Finetuning on the target subject only
        tqdm.write(f"Running finetuning phase on target subject '{args.target_subject}'...")
        
        finetune_checkpoint = run_phase(
            phase_name="finetune",
            preaugmented_dir=args.preaugmented_dir,
            params=finetune_params,
            exp_root=exp_root,
            subjects_include=[args.target_subject],
            subjects_exclude=None,
            init_checkpoint=pretrain_checkpoint
        )
        
        # Print final summary
        tqdm.write("\n" + "="*50)
        tqdm.write("Training complete!")
        tqdm.write(f"Experiment directory: {exp_root}")
        if pretrain_checkpoint:
            tqdm.write(f"Best pretrain checkpoint: {pretrain_checkpoint}")
        tqdm.write(f"Best finetune checkpoint: {finetune_checkpoint}")
        tqdm.write("="*50)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main() 