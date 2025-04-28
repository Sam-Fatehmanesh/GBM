#!/usr/bin/env python3
"""
GBM Brain Region Evaluation Script

This script evaluates a trained GBM model on test data, computing metrics for:
1. Next-frame prediction accuracy
2. Long-horizon autoregressive prediction accuracy

Metrics are computed globally and per brain region using MapZBrain masks.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import tifffile  # For reading multi-page TIFFs

import torch
from torch.utils.data import DataLoader

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import GBM model and dataset
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.fast_dali_spike_dataset import FastDALIBrainDataLoader
from GenerativeBrainModel.scripts.train_gbm_subject_split import SubjectFilteredFastDALIBrainDataLoader


# Configure logging
def setup_logging(log_file=None):
    """Set up logging to console and optionally to a file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    return logging.getLogger(__name__)


# Argument parsing
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate GBM model by brain region')
    
    # Required arguments
    parser.add_argument('--exp-timestamp', type=str, required=True,
                        help='Experiment folder timestamp (e.g., 20230815_153045)')
    parser.add_argument('--target-subject', type=str, required=True,
                        help='Target subject name used for finetuning')
    parser.add_argument('--masks-dir', type=str, required=True,
                        help='Directory containing MapZBrain TIFF masks')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--test-h5', type=str,
                           help='Path to saved test_data_and_predictions.h5')
    data_group.add_argument('--reuse-loader', action='store_true',
                           help='Recreate DALI loader instead of using saved test data')
    
    # Optional arguments
    parser.add_argument('--model-checkpoint', type=str,
                        help='Override default best_model.pt checkpoint')
    parser.add_argument('--horizon-length', type=int, default=330,
                        help='Maximum prediction horizon length (default: 330)')
    parser.add_argument('--seed-length', type=int,
                        help='Length of seed sequence for autoregressive generation (default: auto)')
    parser.add_argument('--multi-sample', type=int, default=1,
                        help='Number of autoregressive samples to generate and average (default: 1)')
    parser.add_argument('--region-group', type=str, default='all', choices=['all', 'major', 'grouped'],
                        help='Brain region grouping preset (default: all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='PyTorch device (default: cuda:0)')
    
    return parser.parse_args()


# Model loading
def load_model(timestamp: str, device: torch.device, checkpoint_file: Optional[str] = None) -> Tuple[GBM, Dict]:
    """
    Load the GBM model from a checkpoint.
    
    Args:
        timestamp: Experiment timestamp directory
        device: PyTorch device
        checkpoint_file: Optional path to specific checkpoint file
    
    Returns:
        model: Loaded GBM model
        params: Parameter dictionary from checkpoint
    """
    finetune_dir = os.path.join('experiments', 'gbm', timestamp, 'finetune')
    if not os.path.exists(finetune_dir):
        raise FileNotFoundError(f"Finetune directory not found: {finetune_dir}")
    
    if checkpoint_file is None:
        checkpoint_file = os.path.join(finetune_dir, 'checkpoints', 'best_model.pt')
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    logging.info(f"Loading model from checkpoint: {checkpoint_file}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    params = checkpoint['params']
    
    # Create model with same architecture
    model = GBM(
        mamba_layers=params['mamba_layers'],
        mamba_dim=params['mamba_dim'],
        mamba_state_multiplier=params['mamba_state_multiplier']
    )
    
    # Load weights and set to eval mode
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully. Architecture: mamba_layers={params['mamba_layers']}, "
                f"mamba_dim={params['mamba_dim']}, mamba_state_multiplier={params['mamba_state_multiplier']}")
    
    return model, params


# Data loading
def load_test_data_h5(test_h5_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load test data and predictions from HDF5 file.
    
    Args:
        test_h5_path: Path to HDF5 file
        
    Returns:
        test_data: Test input sequences (B, T, H, W)
        pred_probs: Predicted probabilities (B, T-1, H, W)
        pred_samples: Binary predictions (B, T-1, H, W)
    """
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(f"Test data file not found: {test_h5_path}")
    
    logging.info(f"Loading test data from: {test_h5_path}")
    
    with h5py.File(test_h5_path, 'r') as f:
        # Check for required datasets
        required_keys = ['test_data', 'predicted_probabilities', 'predicted_samples']
        for key in required_keys:
            if key not in f:
                raise KeyError(f"Required dataset '{key}' not found in H5 file")
        
        # Load datasets
        test_data = torch.from_numpy(f['test_data'][:])
        pred_probs = torch.from_numpy(f['predicted_probabilities'][:])
        pred_samples = torch.from_numpy(f['predicted_samples'][:])
        
        # Log dataset shapes
        logging.info(f"Test data shape: {test_data.shape}")
        logging.info(f"Predicted probabilities shape: {pred_probs.shape}")
        logging.info(f"Predicted samples shape: {pred_samples.shape}")
    
    return test_data, pred_probs, pred_samples


def create_test_loader(preaugmented_dir: str, target_subject: str, params: Dict) -> torch.Tensor:
    """
    Create a DALI data loader for test data.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        target_subject: Target subject name
        params: Parameter dictionary
        
    Returns:
        test_batch: First batch from test loader (B, T, H, W)
    """
    logging.info(f"Creating DALI loader for subject: {target_subject}")
    
    # Create test loader
    test_loader = SubjectFilteredFastDALIBrainDataLoader(
        preaugmented_dir,
        include_subjects=[target_subject],
        batch_size=params.get('batch_size', 32),
        seq_len=params.get('seq_len', 330),
        split='test',
        device_id=0,
        num_threads=params.get('dali_num_threads', 2),
        gpu_prefetch=params.get('gpu_prefetch', 1),
        seed=params.get('seed', 42) + 1,  # Different seed for test set
        shuffle=False,
        stride=params.get('seq_stride', 1)
    )
    
    # Get first batch
    logging.info(f"Getting test batch from loader (size: {len(test_loader)})")
    test_loader.reset()
    test_batch = next(iter(test_loader))
    
    logging.info(f"Test batch shape: {test_batch.shape}")
    
    return test_batch


# Brain region mask processing
def load_brain_region_masks(masks_dir: str, subj_z_planes: int, 
                           target_shape: Tuple[int, int] = (256, 128)) -> Dict[str, np.ndarray]:
    """
    Load and resize brain region masks from 3D TIFF files.
    
    Args:
        masks_dir: Directory containing mask TIFF files
        subj_z_planes: Number of z-planes for the subject
        target_shape: Target shape (height, width) for masks
        
    Returns:
        region_masks: Dictionary of resized 3D binary masks {region_name: mask_3d}
          where mask_3d has shape (Z, H, W)
    """
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Get list of TIFF files
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.tif')]
    if not mask_files:
        raise FileNotFoundError(f"No TIFF files found in masks directory: {masks_dir}")
    
    logging.info(f"Found {len(mask_files)} TIFF mask files in {masks_dir}")
    
    # Create whole brain mask (all True)
    region_masks = {
        'whole_brain': np.ones((subj_z_planes, *target_shape), dtype=bool)
    }
    
    # Process each mask file
    for mask_file in tqdm(mask_files, desc="Loading masks"):
        # Extract region name from filename
        region_name = os.path.splitext(mask_file)[0]
        
        # Load mask as 3D stack
        mask_path = os.path.join(masks_dir, mask_file)
        try:
            # Use tifffile to read potentially multi-page TIFF
            original_mask_3d = tifffile.imread(mask_path)
            
            # Handle single page TIFFs (add z dimension if needed)
            if original_mask_3d.ndim == 2:
                original_mask_3d = original_mask_3d[np.newaxis, :, :]
                logging.warning(f"Mask {mask_file} is 2D, expanding to 3D with single z-plane")
            
            if original_mask_3d.dtype != bool:
                original_mask_3d = original_mask_3d.astype(bool)
                
            # Target 3D shape
            target_3d_shape = (subj_z_planes, *target_shape)
            
            # Resize mask to target dimensions
            resized_mask_3d = resize_3d_binary_mask(original_mask_3d, target_3d_shape)
            
            # Check if mask is empty after resizing
            if not np.any(resized_mask_3d):
                logging.warning(f"Mask for region '{region_name}' is empty after resizing")
                continue
                
            # Store 3D mask
            region_masks[region_name] = resized_mask_3d
                
        except Exception as e:
            logging.error(f"Error processing mask {mask_file}: {str(e)}")
    
    logging.info(f"Successfully loaded {len(region_masks)} brain region masks")
    
    return region_masks


def resize_3d_binary_mask(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize a 3D binary mask to the target shape while preserving binary values.
    
    Args:
        mask: 3D binary mask array (Z, H, W)
        target_shape: Target shape (Z', H', W')
        
    Returns:
        resized_mask: Resized binary mask (Z', H', W')
    """
    # Convert to float for resize operation
    mask_float = mask.astype(float)
    
    # Resize using nearest neighbor interpolation for z-axis and bilinear for x,y
    resized = resize(mask_float, target_shape, order=1, 
                     mode='constant', anti_aliasing=False)
    
    # Convert back to binary
    binary_mask = resized > 0.5
    
    return binary_mask


def group_brain_regions(region_masks: Dict[str, np.ndarray], grouping: str) -> Dict[str, np.ndarray]:
    """
    Create grouped masks by combining related brain regions.
    
    Args:
        region_masks: Dictionary of 3D region masks
        grouping: Grouping strategy ('all', 'major', 'grouped')
        
    Returns:
        grouped_masks: Dictionary of grouped 3D masks
    """
    if grouping == 'all':
        return region_masks
    
    # Always include whole brain
    grouped_masks = {'whole_brain': region_masks['whole_brain']}
    
    # Define region groupings
    region_groups = {
        'forebrain': [
            'prosencephalon_(forebrain)', 'telencephalon', 
            'dorsal_telencephalon_(pallium)', 'ventral_telencephalon_(subpallium)',
            'preoptic_region', 'secondary_prosencephalon'
        ],
        'midbrain': [
            'mesencephalon_(midbrain)', 'midbrain', 'tectum_&_tori',
            'torus_longitudinalis', 'torus_semicircularis'
        ],
        'hindbrain': [
            'rhombencephalon_(hindbrain)', 'cerebellum',
            'medulla_oblongata', 'inferior_medulla_oblongata',
            'intermediate_medulla_oblongata', 'superior_medulla_oblongata'
        ],
        'thalamus': [
            'dorsal_thalamus_proper', 'ventral_thalamus__alar_part',
            'habenula', 'ventral_habenula', 'dorsal_habenula'
        ],
        'hypothalamus': [
            'rostral_hypothalamus', 'intermediate_hypothalamus_(entire)',
            'intermediate_hypothalamus_(remaining)'
        ],
        'retinal_arborization': [
            'retinal_arborization_field_1', 'retinal_arborization_field_2',
            'retinal_arborization_field_3', 'retinal_arborization_field_4',
            'retinal_arborization_field_5', 'retinal_arborization_field_6',
            'retinal_arborization_field_7', 'retinal_arborization_field_8',
            'retinal_arborization_field_9', 'retinal_arborization_field_10'
        ]
    }
    
    if grouping == 'major':
        # Include just the major brain divisions
        major_regions = [
            'prosencephalon_(forebrain)', 'mesencephalon_(midbrain)', 'rhombencephalon_(hindbrain)',
            'telencephalon', 'cerebellum', 'tectum_&_tori'
        ]
        for region in major_regions:
            if region in region_masks:
                grouped_masks[region] = region_masks[region]
        return grouped_masks
    
    # Get the shape from whole_brain mask for creating empty masks
    mask_shape = region_masks['whole_brain'].shape
    
    # Create union masks for each group
    for group_name, regions in region_groups.items():
        # Find available regions in this group
        available_regions = [r for r in regions if r in region_masks]
        
        if not available_regions:
            continue
            
        # Initialize empty mask
        group_mask = np.zeros(mask_shape, dtype=bool)
        
        # Union all regions in group
        for region in available_regions:
            group_mask |= region_masks[region]
            
        grouped_masks[group_name] = group_mask
    
    logging.info(f"Created {len(grouped_masks)} grouped region masks")
    
    return grouped_masks


# Binary prediction metrics
class BinaryMetrics:
    """Track binary prediction metrics (TP, TN, FP, FN) for a region."""
    
    def __init__(self, name: str):
        """
        Initialize a metrics tracker for a named region.
        
        Args:
            name: Name of the region
        """
        self.name = name
        self.tp = 0  # True positives
        self.tn = 0  # True negatives
        self.fp = 0  # False positives
        self.fn = 0  # False negatives
        
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        Update counters with new predictions and targets.
        
        Args:
            pred: Binary predictions tensor
            target: Binary target tensor
            mask: Optional binary mask tensor (same shape as pred/target)
        """
        # Default mask is all ones (entire image)
        if mask is None:
            mask = torch.ones_like(pred, dtype=bool)
            
        # Convert inputs to appropriate types
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy().astype(bool)
        else:
            pred_np = pred.astype(bool)
            
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy().astype(bool)
        else:
            target_np = target.astype(bool)
            
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = mask.astype(bool)
            
        # Apply mask and update counts
        pred_masked = pred_np[mask_np]
        target_masked = target_np[mask_np]
        
        # Update counts
        self.tp += np.sum((pred_masked == True) & (target_masked == True))
        self.tn += np.sum((pred_masked == False) & (target_masked == False))
        self.fp += np.sum((pred_masked == True) & (target_masked == False))
        self.fn += np.sum((pred_masked == False) & (target_masked == True))
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute binary classification metrics from counters.
        
        Returns:
            metrics: Dictionary of metric values
        """
        total = self.tp + self.tn + self.fp + self.fn
        
        # Handle empty regions
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'specificity': 0.0,
                'f1': 0.0,
                'balanced_acc': 0.0,
                'tp': self.tp,
                'tn': self.tn,
                'fp': self.fp,
                'fn': self.fn,
                'total_voxels': total
            }
        
        # Calculate accuracy
        accuracy = (self.tp + self.tn) / total
        
        # Calculate precision (positive predictive value)
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        
        # Calculate recall (sensitivity, true positive rate)
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        
        # Calculate specificity (true negative rate)
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Return all metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'balanced_acc': balanced_acc,
            'tp': self.tp,
            'tn': self.tn,
            'fp': self.fp,
            'fn': self.fn,
            'total_voxels': total
        }
        
    def reset(self):
        """Reset all counters to zero."""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


# Evaluation functions
def evaluate_next_frame(model: GBM, test_data: torch.Tensor, region_masks: Dict[str, np.ndarray], 
                        device: torch.device, subj_z_planes: int) -> Dict[str, Dict[str, float]]:
    """
    Evaluate normal next-frame prediction performance by region.
    
    Args:
        model: Trained GBM model
        test_data: Test data tensor (B, T, H, W)
        region_masks: Dictionary of 3D region masks (Z, H, W)
        device: PyTorch device
        subj_z_planes: Number of z-planes for the subject
        
    Returns:
        results: Dictionary of metrics for each region
    """
    logging.info("Evaluating next-frame prediction performance")
    
    # Move data to device if needed
    if test_data.device != device:
        test_data = test_data.to(device)
    
    # Initialize metrics for each region
    region_metrics = {region: BinaryMetrics(region) for region in region_masks}
    
    # Generate predictions
    with torch.no_grad():
        # Get probability predictions
        pred_probs = model.get_predictions(test_data)  # Shape: (B, T-1, H, W)
        # Convert to binary predictions
        pred_binary = model.sample_binary_predictions(pred_probs)
    
    # True targets are the frames after the first one
    targets = test_data[:, 1:]  # Shape: (B, T-1, H, W)
    
    # Process each region
    for region_name, region_mask_3d in tqdm(region_masks.items(), desc="Processing regions (next-frame)"):
        # Evaluate each batch and timestep
        for b in range(pred_binary.shape[0]):
            for t in range(pred_binary.shape[1]):
                # Calculate z-plane index
                z_idx = t % subj_z_planes
                
                # Get the corresponding 2D mask for this z-plane
                mask_2d = region_mask_3d[z_idx]
                region_mask_tensor = torch.tensor(mask_2d, device=device)
                
                # Update metrics for this sample, timestep, and region
                region_metrics[region_name].update(
                    pred_binary[b, t],
                    targets[b, t],
                    region_mask_tensor
                )
    
    # Compute final metrics for all regions
    results = {}
    for region_name, metric_tracker in region_metrics.items():
        results[region_name] = metric_tracker.compute_metrics()
    
    logging.info("Next-frame evaluation complete")
    
    return results


def evaluate_long_horizon(model: GBM, test_data: torch.Tensor, region_masks: Dict[str, np.ndarray],
                         device: torch.device, subj_z_planes: int, seed_length: Optional[int] = None, 
                         horizon_length: int = 330, per_step: bool = True, 
                         cumulative: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate long-horizon autoregressive prediction by region.
    
    Args:
        model: Trained GBM model
        test_data: Test data tensor (B, T, H, W)
        region_masks: Dictionary of 3D region masks (Z, H, W)
        device: PyTorch device
        subj_z_planes: Number of z-planes for the subject
        seed_length: Length of seed sequence
        horizon_length: Maximum prediction horizon
        per_step: Whether to compute per-step metrics
        cumulative: Whether to compute cumulative metrics
        
    Returns:
        results: Dictionary of metrics for each region
    """
    logging.info("Evaluating long-horizon prediction performance")
    
    # Move data to device if needed
    if test_data.device != device:
        test_data = test_data.to(device)
    
    # Determine seed length if not specified
    if seed_length is None:
        seed_length = min(115, test_data.shape[1] - 1)
    
    # Cap horizon length based on available data
    max_horizon = min(horizon_length - seed_length, test_data.shape[1] - seed_length)
    
    logging.info(f"Using seed length {seed_length}, predicting {max_horizon} steps")
    
    # Create metrics trackers
    if per_step:
        # Per-step metrics
        per_step_metrics = {
            region: [BinaryMetrics(f"{region}_h{h}") 
                    for h in range(1, max_horizon + 1)]
            for region in region_masks
        }
    
    if cumulative:
        # Cumulative metrics
        cumulative_metrics = {
            region: BinaryMetrics(f"{region}_cumulative") 
            for region in region_masks
        }
    
    # Process each sample in the batch
    num_samples = min(test_data.shape[0], 10)  # Process at most 10 samples for efficiency
    
    for sample_idx in tqdm(range(num_samples), desc="Processing samples (long-horizon)"):
        # Get seed sequence
        seed = test_data[sample_idx:sample_idx+1, :seed_length]
        
        # Generate autoregressive predictions
        with torch.no_grad():
            generated = model.generate_autoregressive_brain(
                seed, num_steps=max_horizon)
            
            # Extract the predicted part (after seed)
            predictions = generated[:, seed_length:seed_length+max_horizon]
        
        # Get ground truth for comparison
        targets = test_data[sample_idx:sample_idx+1, seed_length:seed_length+max_horizon]
        
        # For each brain region
        for region_name, region_mask_3d in region_masks.items():
            # Update per-step metrics if requested
            if per_step:
                for h in range(max_horizon):
                    # Calculate z-plane index for this horizon step
                    z_idx = (seed_length + h) % subj_z_planes
                    
                    # Get the corresponding 2D mask for this z-plane
                    mask_2d = region_mask_3d[z_idx]
                    region_mask_tensor = torch.tensor(mask_2d, device=device)
                    
                    per_step_metrics[region_name][h].update(
                        predictions[0, h],
                        targets[0, h],
                        region_mask_tensor
                    )
            
            # Update cumulative metrics if requested
            if cumulative:
                for h in range(max_horizon):
                    # Calculate z-plane index for this horizon step
                    z_idx = (seed_length + h) % subj_z_planes
                    
                    # Get the corresponding 2D mask for this z-plane
                    mask_2d = region_mask_3d[z_idx]
                    region_mask_tensor = torch.tensor(mask_2d, device=device)
                    
                    cumulative_metrics[region_name].update(
                        predictions[0, h],
                        targets[0, h],
                        region_mask_tensor
                    )
    
    # Compute and collect results
    results = {}
    
    if per_step:
        # Per-step metrics for each region
        results['per_step'] = {
            region: [metrics.compute_metrics() for metrics in metrics_list]
            for region, metrics_list in per_step_metrics.items()
        }
    
    if cumulative:
        # Cumulative metrics for each region
        results['cumulative'] = {
            region: metrics.compute_metrics()
            for region, metrics in cumulative_metrics.items()
        }
    
    logging.info("Long-horizon evaluation complete")
    
    return results


def run_multiple_samples(model: GBM, test_data: torch.Tensor, region_masks: Dict[str, np.ndarray],
                         device: torch.device, num_samples: int = 5, subj_z_planes: int = 30,
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run long-horizon evaluation multiple times and average results.
    
    Args:
        model: Trained GBM model
        test_data: Test data tensor (B, T, H, W)
        region_masks: Dictionary of 3D region masks
        device: PyTorch device
        num_samples: Number of stochastic samples to generate
        subj_z_planes: Number of z-planes for the subject
        **kwargs: Additional arguments for evaluate_long_horizon
        
    Returns:
        avg_results: Averaged results across samples
    """
    logging.info(f"Running long-horizon evaluation with {num_samples} samples")
    
    all_results = []
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Run evaluation
        results = evaluate_long_horizon(
            model, test_data, region_masks, device, 
            subj_z_planes=subj_z_planes, **kwargs
        )
        all_results.append(results)
    
    # Average results
    avg_results = {}
    
    # Average per-step results
    if 'per_step' in all_results[0]:
        avg_results['per_step'] = {}
        for region in all_results[0]['per_step']:
            avg_results['per_step'][region] = []
            for step in range(len(all_results[0]['per_step'][region])):
                # Average metrics across samples for this region and step
                avg_step = {}
                for metric in all_results[0]['per_step'][region][step]:
                    if metric in ['tp', 'tn', 'fp', 'fn', 'total_voxels']:
                        # For counters, we sum
                        avg_step[metric] = sum(res['per_step'][region][step][metric] 
                                              for res in all_results)
                    else:
                        # For metrics, we average
                        avg_step[metric] = sum(res['per_step'][region][step][metric] 
                                              for res in all_results) / num_samples
                avg_results['per_step'][region].append(avg_step)
    
    # Average cumulative results
    if 'cumulative' in all_results[0]:
        avg_results['cumulative'] = {}
        for region in all_results[0]['cumulative']:
            # Average metrics across samples for this region
            avg_results['cumulative'][region] = {}
            for metric in all_results[0]['cumulative'][region]:
                if metric in ['tp', 'tn', 'fp', 'fn', 'total_voxels']:
                    # For counters, we sum
                    avg_results['cumulative'][region][metric] = sum(
                        res['cumulative'][region][metric] for res in all_results)
                else:
                    # For metrics, we average
                    avg_results['cumulative'][region][metric] = sum(
                        res['cumulative'][region][metric] for res in all_results) / num_samples
    
    logging.info("Multi-sample evaluation complete")
    
    return avg_results


# Results saving and visualization
def save_results(results: Dict[str, Any], output_dir: str, prefix: str = '') -> None:
    """
    Save evaluation results to CSV files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        prefix: Optional prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save next-frame metrics
    if 'next_frame' in results:
        next_frame_rows = []
        for region, metrics in results['next_frame'].items():
            next_frame_rows.append({'region': region, **metrics})
        
        next_frame_df = pd.DataFrame(next_frame_rows)
        next_frame_path = os.path.join(output_dir, f'{prefix}next_frame_metrics.csv')
        next_frame_df.to_csv(next_frame_path, index=False)
        logging.info(f"Saved next-frame metrics to {next_frame_path}")
    
    # Save long-horizon cumulative metrics
    if 'long_horizon' in results and 'cumulative' in results['long_horizon']:
        cumulative_rows = []
        for region, metrics in results['long_horizon']['cumulative'].items():
            cumulative_rows.append({'region': region, **metrics})
        
        cumulative_df = pd.DataFrame(cumulative_rows)
        cumulative_path = os.path.join(output_dir, f'{prefix}long_horizon_cumulative.csv')
        cumulative_df.to_csv(cumulative_path, index=False)
        logging.info(f"Saved cumulative long-horizon metrics to {cumulative_path}")
    
    # Save long-horizon per-step metrics
    if 'long_horizon' in results and 'per_step' in results['long_horizon']:
        per_step_rows = []
        for region, steps_metrics in results['long_horizon']['per_step'].items():
            for step, metrics in enumerate(steps_metrics):
                per_step_rows.append({'region': region, 'horizon_step': step + 1, **metrics})
        
        per_step_df = pd.DataFrame(per_step_rows)
        per_step_path = os.path.join(output_dir, f'{prefix}long_horizon_per_step.csv')
        per_step_df.to_csv(per_step_path, index=False)
        logging.info(f"Saved per-step long-horizon metrics to {per_step_path}")


def create_accuracy_curves(results: Dict[str, Any], output_dir: str, prefix: str = '') -> None:
    """
    Create and save plots of accuracy vs horizon step.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        prefix: Optional prefix for filenames
    """
    if 'long_horizon' not in results or 'per_step' not in results['long_horizon']:
        logging.warning("No per-step results found. Skipping accuracy curve generation.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key regions to plot (whole brain + major subdivisions)
    key_regions = [
        'whole_brain', 'telencephalon', 'cerebellum', 'midbrain', 
        'rhombencephalon_(hindbrain)', 'prosencephalon_(forebrain)'
    ]
    available_key_regions = [r for r in key_regions if r in results['long_horizon']['per_step']]
    
    if not available_key_regions:
        available_key_regions = list(results['long_horizon']['per_step'].keys())[:5]  # First 5 regions
    
    # Plot accuracy vs horizon step
    plt.figure(figsize=(12, 8))
    
    for region in available_key_regions:
        steps = results['long_horizon']['per_step'][region]
        x = list(range(1, len(steps) + 1))
        y = [step['accuracy'] for step in steps]
        plt.plot(x, y, label=region)
    
    plt.title('Prediction Accuracy vs Horizon Step')
    plt.xlabel('Horizon Step')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    acc_path = os.path.join(output_dir, f'{prefix}accuracy_vs_horizon.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved accuracy curve plot to {acc_path}")
    
    # Plot F1 score vs horizon step
    plt.figure(figsize=(12, 8))
    
    for region in available_key_regions:
        steps = results['long_horizon']['per_step'][region]
        x = list(range(1, len(steps) + 1))
        y = [step['f1'] for step in steps]
        plt.plot(x, y, label=region)
    
    plt.title('F1 Score vs Horizon Step')
    plt.xlabel('Horizon Step')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    f1_path = os.path.join(output_dir, f'{prefix}f1_vs_horizon.png')
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved F1 curve plot to {f1_path}")


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of key metrics to the console.
    
    Args:
        results: Results dictionary
    """
    print("\n" + "="*80)
    print("GBM EVALUATION SUMMARY")
    print("="*80)
    
    # Print next-frame metrics
    if 'next_frame' in results:
        print("\nNEXT-FRAME PREDICTION METRICS:")
        print("-"*50)
        
        # Sort regions by accuracy
        sorted_regions = sorted(
            results['next_frame'].items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        # Print table header
        print(f"{'Region':<30} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-"*70)
        
        # Print metrics for each region
        for region, metrics in sorted_regions:
            print(f"{region:<30} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                 f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    # Print long-horizon cumulative metrics
    if 'long_horizon' in results and 'cumulative' in results['long_horizon']:
        print("\nLONG-HORIZON CUMULATIVE METRICS:")
        print("-"*50)
        
        # Sort regions by accuracy
        sorted_regions = sorted(
            results['long_horizon']['cumulative'].items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        # Print table header
        print(f"{'Region':<30} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-"*70)
        
        # Print metrics for each region
        for region, metrics in sorted_regions:
            print(f"{region:<30} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                 f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    print("\n" + "="*80)


# New function to get subject's number of z-planes
def get_subject_z_planes(preaugmented_dir: str, target_subject: str) -> int:
    """
    Get the number of z-planes for a specific subject from its metadata file.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        target_subject: Name of the target subject
        
    Returns:
        num_z_planes: Number of z-planes for the subject
    """
    metadata_path = os.path.join(preaugmented_dir, target_subject, 'metadata.h5')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found for subject {target_subject}: {metadata_path}")
    
    with h5py.File(metadata_path, 'r') as f:
        if 'num_z_planes' not in f:
            raise KeyError(f"num_z_planes not found in metadata file: {metadata_path}")
        
        num_z_planes = int(f['num_z_planes'][()])
    
    logging.info(f"Subject {target_subject} has {num_z_planes} z-planes")
    return num_z_planes


def resize_3d_binary_mask(mask: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize a 3D binary mask to the target shape while preserving binary values.
    
    Args:
        mask: 3D binary mask array (Z, H, W)
        target_shape: Target shape (Z', H', W')
        
    Returns:
        resized_mask: Resized binary mask (Z', H', W')
    """
    # Convert to float for resize operation
    mask_float = mask.astype(float)
    
    # Resize using nearest neighbor interpolation for z-axis and bilinear for x,y
    resized = resize(mask_float, target_shape, order=1, 
                     mode='constant', anti_aliasing=False)
    
    # Convert back to binary
    binary_mask = resized > 0.5
    
    return binary_mask


# Main function
def main():
    """Main function to run the evaluation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create a datetime subfolder in the output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {output_dir}")
    
    # Copy the original command used to run this script
    with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
        f.write(f"python {' '.join(sys.argv)}\n")
    
    # Set up logging
    log_file = os.path.join(output_dir, 'eval.log')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(log_file)
    
    # Log script configuration
    logging.info("Starting GBM brain region evaluation")
    logging.info(f"Arguments: {vars(args)}")
    logging.info(f"Results will be saved to: {output_dir}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Using CPU instead.")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")
    
    try:
        # Load model and parameters
        model, params = load_model(args.exp_timestamp, device, args.model_checkpoint)
        
        # Load test data
        if args.test_h5:
            test_data, _, _ = load_test_data_h5(args.test_h5)
        else:
            # Use DALI loader
            test_data = create_test_loader(
                params['preaugmented_dir'], 
                args.target_subject, 
                params
            )
        
        # Get subject's number of z-planes
        subj_z_planes = get_subject_z_planes(
            params['preaugmented_dir'], 
            args.target_subject
        )
        logging.info(f"Subject {args.target_subject} has {subj_z_planes} z-planes")
        
        # Load and process brain region masks
        region_masks = load_brain_region_masks(
            args.masks_dir,
            subj_z_planes,
            target_shape=(256, 128)
        )
        
        # Group regions if requested
        if args.region_group != 'all':
            region_masks = group_brain_regions(region_masks, args.region_group)
        
        # Evaluate next-frame prediction
        next_frame_results = evaluate_next_frame(
            model, test_data, region_masks, device, subj_z_planes
        )
        
        # Evaluate long-horizon prediction
        if args.multi_sample > 1:
            # Multiple sample runs
            long_horizon_results = run_multiple_samples(
                model, test_data, region_masks, device, num_samples=args.multi_sample,
                subj_z_planes=subj_z_planes, seed_length=args.seed_length, 
                horizon_length=args.horizon_length
            )
        else:
            # Single run
            long_horizon_results = evaluate_long_horizon(
                model, test_data, region_masks, device, subj_z_planes,
                seed_length=args.seed_length, horizon_length=args.horizon_length
            )
        
        # Combine results
        all_results = {
            'next_frame': next_frame_results,
            'long_horizon': long_horizon_results
        }
        
        # Save results
        save_results(all_results, output_dir)
        
        # Create visualizations
        create_accuracy_curves(all_results, output_dir)
        
        # Save parameters
        with open(os.path.join(output_dir, 'model_params.json'), 'w') as f:
            # Convert non-serializable objects to strings
            serializable_params = {}
            for k, v in params.items():
                try:
                    json.dumps({k: v})
                    serializable_params[k] = v
                except (TypeError, OverflowError):
                    serializable_params[k] = str(v)
            json.dump(serializable_params, f, indent=2)
        
        # Print summary
        print_summary(all_results)
        
        # Print final output path
        print(f"\nEvaluation complete. Results saved to: {output_dir}")
        logging.info(f"Evaluation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 