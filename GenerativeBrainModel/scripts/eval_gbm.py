#!/usr/bin/env python3
"""
GBM Model Evaluation Script

This script evaluates trained GBM models on test data with detailed region-based metrics.
It loads 3D brain masks for various regions and computes comprehensive evaluation metrics
including BCE loss, normalized activation magnitude error, and F1 scores.

Features:
- Config-based parameter management
- 3D brain mask loading for region-specific evaluation
- Batch-based evaluation on test subjects
- Time-series heatmaps for metrics visualization
- Per-region and whole-volume metrics computation
- Support for both single models and ensembles
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from GenerativeBrainModel.models.gbm import GBM, EnsembleGBM
from GenerativeBrainModel.dataloaders.volume_dataloader import VolumeDataset
from torch.utils.data import DataLoader

# Mask loading implementation (from user example)
import glob
import tifffile

class ZebrafishMaskLoader:
    """
    A class to load and manage 3D zebrafish brain masks from TIF files.
    
    This class loads all TIF mask files from a specified directory, downsamples them
    to a target resolution, and stores them as PyTorch tensors on a specified device.
    """
    
    def __init__(
        self, 
        masks_dir: str = "masks",
        target_shape: Tuple[int, int, int] = (30, 256, 128),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bool
    ):
        """
        Initialize the zebrafish mask loader.
        
        Args:
            masks_dir: Directory containing the TIF mask files
            target_shape: Target shape for downsampling (Z, Y, X). Default: (30, 256, 128)
            device: PyTorch device to load tensors onto. If None, uses CUDA if available
            dtype: Data type for the tensors. Default: torch.bool
        """
        self.masks_dir = masks_dir
        self.target_shape = target_shape
        self.dtype = dtype
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Dictionary to store loaded masks
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Original shape (will be detected from first loaded mask)
        self.original_shape: Optional[Tuple[int, int, int]] = None
        
        # Load all masks
        self._load_all_masks()
    
    def _load_all_masks(self) -> None:
        """Load all TIF mask files from the masks directory."""
        # Find all TIF files in the masks directory
        mask_files = glob.glob(os.path.join(self.masks_dir, "*.tif"))
        
        if not mask_files:
            raise ValueError(f"No TIF files found in directory: {self.masks_dir}")
        
        print(f"Loading {len(mask_files)} mask files from {self.masks_dir}")
        print(f"Target shape: {self.target_shape}")
        print(f"Device: {self.device}")
        
        for mask_file in sorted(mask_files):
            self._load_single_mask(mask_file)
        
        print(f"Successfully loaded {len(self.masks)} masks")
        
        # Print memory usage if on GPU
        if self.device.type == "cuda":
            memory_mb = sum(mask.element_size() * mask.nelement() for mask in self.masks.values()) / 1024**2
            print(f"Total GPU memory used by masks: {memory_mb:.2f} MB")
    
    def _load_single_mask(self, mask_file: str) -> None:
        """
        Load a single TIF mask file, downsample it, and store as tensor.
        
        Args:
            mask_file: Path to the TIF mask file
        """
        # Extract filename without extension for dictionary key
        filename = os.path.basename(mask_file)
        mask_name = os.path.splitext(filename)[0]
        
        try:
            # Load the mask using tifffile
            mask_data = tifffile.imread(mask_file)
            
            # Store original shape from first mask
            if self.original_shape is None:
                self.original_shape = mask_data.shape
                print(f"Original mask shape: {self.original_shape}")
            
            # Verify shape consistency
            if mask_data.shape != self.original_shape:
                print(f"Warning: {mask_name} has different shape {mask_data.shape} vs expected {self.original_shape}")
            
            # Convert to torch tensor
            mask_tensor = torch.from_numpy(mask_data.astype(np.float32))
            
            # Downsample if needed
            if mask_tensor.shape != self.target_shape:
                mask_tensor = self._downsample_mask(mask_tensor)
            
            # Convert to target dtype and move to device
            mask_tensor = mask_tensor.to(dtype=self.dtype, device=self.device)
            
            # Store in dictionary
            self.masks[mask_name] = mask_tensor
            
            # Print progress for large masks
            if mask_tensor.sum() > 0:  # Only print for non-empty masks
                fill_percentage = (mask_tensor.sum().item() / mask_tensor.numel()) * 100
                print(f"  {mask_name}: {fill_percentage:.2f}% filled")
        
        except Exception as e:
            print(f"Error loading {mask_file}: {str(e)}")
    
    def _downsample_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample a mask tensor to the target shape and reorder axes to match volume format.
        
        Args:
            mask_tensor: Input mask tensor of shape (Z, Y, X)
            
        Returns:
            Downsampled mask tensor of shape (X, Y, Z) to match volume dimensions
        """
        # First, apply cropping before downsampling
        # Gets rid of the first/last 100 along Y axis and first/last 20 along X axis
        mask_tensor = mask_tensor[:, 100:-275, 20:-20]

        # So we need: (Z, Y, X) -> (X, Y, Z) which is permute(2, 1, 0)
        mask_tensor = mask_tensor.permute(1, 2, 0)  # (Z, Y, X) -> (X, Y, Z)
        
        # Add batch dimension for interpolation: (1, 1, Z, Y, X)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        
        # Use trilinear interpolation for 3D downsampling
        # target_shape is (Z, Y, X), and F.interpolate expects size in same order as input
        # So we interpolate to target_shape which gives us (Z, Y, X) output
        downsampled = F.interpolate(
            mask_tensor, 
            size=self.target_shape,  # (X, Y, Z) = (256, 128, 30)
            mode='nearest'  # Preserves binary nature of masks
        )
        
        # Remove batch dimensions: result is (X, Y, Z) = (256, 128, 30)
        downsampled = downsampled.squeeze(0).squeeze(0)  # (1, 1, X, Y, Z) -> (X, Y, Z)
        
        return downsampled
    
    def get_mask(self, mask_name: str) -> torch.Tensor:
        """
        Get a specific mask by name.
        
        Args:
            mask_name: Name of the mask (filename without .tif extension)
            
        Returns:
            The mask tensor
            
        Raises:
            KeyError: If mask name not found
        """
        if mask_name not in self.masks:
            available_masks = list(self.masks.keys())
            raise KeyError(f"Mask '{mask_name}' not found. Available masks: {available_masks[:5]}...")
        
        return self.masks[mask_name]
    
    def get_all_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get all loaded masks.
        
        Returns:
            Dictionary of all mask tensors
        """
        return self.masks
    
    def list_masks(self) -> List[str]:
        """
        Get list of all available mask names.
        
        Returns:
            List of mask names
        """
        return list(self.masks.keys())


def create_default_eval_config() -> Dict[str, Any]:
    """
    Create default configuration for GBM evaluation.
    
    Returns:
        Dictionary with default evaluation configuration
    """
    return {
        'evaluation': {
            'name': 'gbm_evaluation',
            'description': 'Comprehensive evaluation of trained GBM models on test data',
            'model_path': None,  # Path to trained model checkpoint - REQUIRED
            'model_type': 'single',  # 'single' or 'ensemble'
            'ensemble_paths': [],  # List of paths for ensemble models
        },
        
        'data': {
            'data_dir': 'processed_spike_voxels_2018',  # Directory with test data
            'test_subjects': ['subject_1', 'subject_4', 'subject_5'],  # Test subjects to evaluate
            'max_timepoints_per_subject': None,  # Limit timepoints per subject (None = all)
            'sequence_length': 8,  # Must match training sequence length
            'stride': 2,  # Must match training stride
        },
        
        'masks': {
            'masks_dir': 'masks',  # Directory containing brain region TIF masks
            'target_shape': [30, 256, 128],  # Target shape for mask downsampling (Z, Y, X)
            'region_names': [],  # Specific regions to evaluate (empty = all available)
        },
        
        'evaluation_params': {
            'batch_size': 4,  # Batch size for evaluation
            'num_batches': None,  # Number of batches to evaluate (None = all)
            'threshold': 0.5,  # Threshold for F1 score computation
            'device': 'cuda',  # Device to run evaluation on
            'save_predictions': False,  # Whether to save model predictions
        },
        
        'metrics': {
            'compute_bce_loss': True,  # Binary cross-entropy loss
            'compute_magnitude_error': True,  # Normalized activation magnitude error
            'compute_f1_scores': True,  # F1 scores with thresholding
            'compute_per_timepoint': True,  # Compute metrics per timepoint
            'compute_time_averaged': True,  # Compute time-averaged metrics
        },
        
        'visualization': {
            'create_heatmaps': True,  # Create time-series heatmaps
            'heatmap_figsize': [12, 8],  # Figure size for heatmaps
            'save_plots': True,  # Save plots to files
            'show_plots': False,  # Display plots interactively
            'max_regions_in_heatmap': 20,  # Limit number of regions in heatmap for readability
        },
        
        'output': {
            'output_dir': 'eval_results',  # Output directory for results
            'save_detailed_results': True,  # Save detailed per-sample results
            'save_summary_csv': True,  # Save summary metrics to CSV
            'create_report': True,  # Create evaluation report
        }
    }


def load_eval_config(config_path: str) -> Dict[str, Any]:
    """
    Load evaluation configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    
    # Merge with defaults
    default_config = create_default_eval_config()
    config = deep_update(default_config, user_config)
    
    return config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Recursively update nested dictionary.
    
    Args:
        base_dict: Base dictionary
        update_dict: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def save_default_eval_config(output_path: str):
    """
    Save default evaluation configuration to YAML file.
    
    Args:
        output_path: Path to save the config file
    """
    config = create_default_eval_config()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Default evaluation configuration saved to: {output_path}")


class GBMEvaluator:
    """
    Comprehensive evaluator for GBM models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_output_directory()
        
        # Initialize components
        self.model = None
        self.mask_loader = None
        self.test_loader = None
        
        # Results storage
        self.results = defaultdict(list)
        self.region_names = []
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Set up compute device."""
        device_str = self.config['evaluation_params'].get('device', 'cuda')
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
    
    def setup_output_directory(self):
        """Set up output directory structure."""
        self.output_dir = Path(self.config['output']['output_dir'])
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / f"eval_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def load_model(self):
        """Load the trained GBM model."""
        self.logger.info("Loading model...")
        
        model_config = self.config['evaluation']
        model_type = model_config.get('model_type', 'single')
        
        if model_type == 'single':
            model_path = model_config['model_path']
            if not model_path or not Path(model_path).exists():
                raise ValueError(f"Model path not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration from checkpoint if available
            if 'config' in checkpoint:
                model_config_from_ckpt = checkpoint['config']['model']
                
                # Create model instance
                self.model = GBM(
                    d_model=model_config_from_ckpt['d_model'],
                    n_heads=model_config_from_ckpt['n_heads'],
                    n_layers=model_config_from_ckpt['n_layers'],
                    autoencoder_path=model_config_from_ckpt.get('autoencoder_path'),
                    volume_size=tuple(model_config_from_ckpt['volume_size']),
                    region_size=tuple(model_config_from_ckpt['region_size'])
                )
            else:
                raise ValueError("Checkpoint must contain model configuration")
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle torch.compile() prefixes
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    cleaned_key = key[len('_orig_mod.'):]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value
            
            self.model.load_state_dict(cleaned_state_dict)
            self.logger.info(f"Loaded single model from: {model_path}")
            
        elif model_type == 'ensemble':
            ensemble_paths = model_config.get('ensemble_paths', [])
            if not ensemble_paths:
                raise ValueError("Ensemble paths must be provided for ensemble evaluation")
            
            # Load first checkpoint to get model configuration
            first_checkpoint = torch.load(ensemble_paths[0], map_location=self.device)
            if 'config' not in first_checkpoint:
                raise ValueError("Checkpoint must contain model configuration")
            
            model_config_from_ckpt = first_checkpoint['config']['model']
            ensemble_size = len(ensemble_paths)
            
            # Create ensemble model
            self.model = EnsembleGBM(
                n_models=ensemble_size,
                d_model=model_config_from_ckpt['d_model'],
                n_heads=model_config_from_ckpt['n_heads'],
                n_layers=model_config_from_ckpt['n_layers'],
                autoencoder_path=model_config_from_ckpt.get('autoencoder_path'),
                volume_size=tuple(model_config_from_ckpt['volume_size']),
                region_size=tuple(model_config_from_ckpt['region_size']),
                different_seeds=model_config_from_ckpt.get('different_seeds', True)
            )
            
            # Load weights for each model in ensemble
            self.model.load_ensemble_weights(ensemble_paths)
            self.logger.info(f"Loaded ensemble with {ensemble_size} models")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded with {total_params:,} parameters")
    
    def load_masks(self):
        """Load brain region masks."""
        self.logger.info("Loading brain region masks...")
        
        masks_config = self.config['masks']
        masks_dir = masks_config['masks_dir']
        
        if not Path(masks_dir).exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Get target shape (convert from [Z, Y, X] to (Z, Y, X))
        target_shape = tuple(masks_config['target_shape'])
        
        # Load masks
        self.mask_loader = ZebrafishMaskLoader(
            masks_dir=masks_dir,
            target_shape=target_shape,
            device=self.device,
            dtype=torch.float32  # Use float for easier computation
        )
        
        # Get region names
        all_masks = self.mask_loader.list_masks()
        region_names = masks_config.get('region_names', [])
        
        if region_names:
            # Use specified regions
            self.region_names = [name for name in region_names if name in all_masks]
            missing = set(region_names) - set(all_masks)
            if missing:
                self.logger.warning(f"Requested regions not found: {missing}")
        else:
            # Use all available regions
            self.region_names = all_masks
        
        self.logger.info(f"Loaded {len(self.region_names)} brain region masks")
        self.logger.info(f"Region names: {self.region_names[:5]}..." if len(self.region_names) > 5 else f"Region names: {self.region_names}")
    
    def create_test_dataloader(self):
        """Create test data loader."""
        self.logger.info("Creating test data loader...")
        
        data_config = self.config['data']
        eval_config = self.config['evaluation_params']
        
        # Get test subject files
        data_dir = Path(data_config['data_dir'])
        test_subjects = data_config['test_subjects']
        
        test_files = []
        for subject in test_subjects:
            subject_file = data_dir / f"{subject}.h5"
            if subject_file.exists():
                test_files.append(str(subject_file))
            else:
                self.logger.warning(f"Test subject file not found: {subject_file}")
        
        if not test_files:
            raise ValueError("No valid test subject files found")
        
        # Create dataset
        test_dataset = VolumeDataset(
            data_files=test_files,
            sequence_length=data_config['sequence_length'],
            stride=data_config['stride'],
            max_timepoints_per_subject=data_config.get('max_timepoints_per_subject'),
            use_cache=False  # Don't cache for evaluation to save memory
        )
        
        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=eval_config['batch_size'],
            shuffle=False,  # No shuffling for evaluation
            num_workers=2,
            pin_memory=False
        )
        
        self.logger.info(f"Test dataset: {len(test_dataset)} sequences from {len(test_files)} subjects")
    
    def prepare_seq2seq_data(self, sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input and target sequences for seq2seq evaluation.
        
        Args:
            sequences: Tensor of shape (B, T, X, Y, Z)
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        B, T, X, Y, Z = sequences.shape
        
        if T < 2:
            raise ValueError(f"Sequence length must be at least 2 for seq2seq evaluation, got {T}")
        
        # Input: all timesteps except the last
        input_seq = sequences[:, :-1, :, :, :]  # (B, T-1, X, Y, Z)
        
        # Target: all timesteps except the first  
        target_seq = sequences[:, 1:, :, :, :]   # (B, T-1, X, Y, Z)
        
        return input_seq, target_seq
    
    def compute_volume_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics for a volume.
        
        Args:
            predictions: Model predictions (B, T, X, Y, Z)
            targets: Ground truth targets (B, T, X, Y, Z)
            mask: Optional mask to apply (X, Y, Z) or (Z, Y, X)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Apply mask if provided
        if mask is not None:
            # Mask should now have the correct shape (X, Y, Z) to match volume dimensions
            if mask.shape != predictions.shape[-3:]:
                raise ValueError(f"Mask shape {mask.shape} incompatible with volume shape {predictions.shape[-3:]}")
            
            # Apply mask
            mask_bool = mask.bool()
            pred_masked = predictions[..., mask_bool]
            target_masked = targets[..., mask_bool]
        else:
            pred_masked = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
            target_masked = targets.reshape(targets.shape[0], targets.shape[1], -1)
        
        # Convert to probabilities
        pred_probs = torch.sigmoid(pred_masked)
        
        metrics_config = self.config['metrics']
        threshold = self.config['evaluation_params']['threshold']
        
        # BCE Loss
        if metrics_config['compute_bce_loss']:
            bce_loss = F.binary_cross_entropy_with_logits(pred_masked, target_masked, reduction='mean')
            metrics['bce_loss'] = bce_loss.item()
        
        # Normalized activation magnitude error
        if metrics_config['compute_magnitude_error']:
            pred_sum = pred_probs.sum().item()
            target_sum = target_masked.sum().item()
            
            if target_sum > 0:
                magnitude_error = (pred_sum - target_sum) / target_sum
            else:
                magnitude_error = 0.0 if pred_sum == 0 else float('inf')
            
            metrics['magnitude_error'] = magnitude_error
        
        # F1 Score
        if metrics_config['compute_f1_scores']:
            pred_binary = (pred_probs >= threshold).float()
            target_binary = (target_masked >= threshold).float()
            
            tp = (pred_binary * target_binary).sum().item()
            fp = (pred_binary * (1 - target_binary)).sum().item()
            fn = ((1 - pred_binary) * target_binary).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics['f1_score'] = f1
            metrics['precision'] = precision
            metrics['recall'] = recall
        
        return metrics
    
    def compute_timepoint_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> List[Dict[str, float]]:
        """
        Compute metrics for each timepoint separately.
        
        Args:
            predictions: Model predictions (B, T, X, Y, Z)
            targets: Ground truth targets (B, T, X, Y, Z)
            mask: Optional mask to apply
            
        Returns:
            List of metric dictionaries, one per timepoint
        """
        B, T, X, Y, Z = predictions.shape
        timepoint_metrics = []
        
        for t in range(T):
            pred_t = predictions[:, t:t+1, :, :, :]  # (B, 1, X, Y, Z)
            target_t = targets[:, t:t+1, :, :, :]    # (B, 1, X, Y, Z)
            
            metrics_t = self.compute_volume_metrics(pred_t, target_t, mask)
            timepoint_metrics.append(metrics_t)
        
        return timepoint_metrics
    
    def create_3d_debug_visualizations(self, volume: torch.Tensor, mask: torch.Tensor, 
                                     volume_name: str, mask_name: str):
        """
        Create 3D visualizations of volume and mask to debug axis ordering.
        
        Args:
            volume: Volume tensor to visualize (X, Y, Z)
            mask: Mask tensor to visualize 
            volume_name: Name for the volume plot
            mask_name: Name for the mask plot
        """
        self.logger.info(f"Creating 3D debug visualizations...")
        self.logger.info(f"Volume shape: {volume.shape}")
        self.logger.info(f"Mask shape: {mask.shape}")
        
        # Create debug plots directory
        debug_dir = self.plots_dir / "debug_axis_ordering"
        debug_dir.mkdir(exist_ok=True)
        
        # Convert tensors to numpy and move to CPU
        if volume.is_cuda:
            volume = volume.cpu()
        if mask.is_cuda:
            mask = mask.cpu()
        
        volume_np = volume.numpy()
        mask_np = mask.numpy()
        
        # Create 3D scatter plots for both volume and mask
        fig = plt.figure(figsize=(15, 6))
        
        # Volume visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Get non-zero voxels for volume (threshold for visualization)
        threshold = np.percentile(volume_np[volume_np > 0], 95) if np.any(volume_np > 0) else 0.1
        vol_coords = np.where(volume_np > threshold)
        vol_values = volume_np[vol_coords]
        
        if len(vol_coords[0]) > 0:
            # Sample points if too many (for performance)
            if len(vol_coords[0]) > 5000:
                sample_idx = np.random.choice(len(vol_coords[0]), 5000, replace=False)
                vol_coords = (vol_coords[0][sample_idx], vol_coords[1][sample_idx], vol_coords[2][sample_idx])
                vol_values = vol_values[sample_idx]
            
            scatter1 = ax1.scatter(vol_coords[0], vol_coords[1], vol_coords[2], 
                                 c=vol_values, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter1, ax=ax1, shrink=0.5)
        
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis') 
        ax1.set_zlabel('Z axis')
        ax1.set_title(f'{volume_name}\nShape: {volume.shape}')
        
        # Mask visualization
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Get non-zero voxels for mask
        mask_coords = np.where(mask_np > 0.5)  # Assuming binary mask
        
        if len(mask_coords[0]) > 0:
            # Sample points if too many (for performance)
            if len(mask_coords[0]) > 5000:
                sample_idx = np.random.choice(len(mask_coords[0]), 5000, replace=False)
                mask_coords = (mask_coords[0][sample_idx], mask_coords[1][sample_idx], mask_coords[2][sample_idx])
            
            ax2.scatter(mask_coords[0], mask_coords[1], mask_coords[2], 
                       c='red', alpha=0.6, s=1)
        
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')
        ax2.set_title(f'{mask_name}\nShape: {mask.shape}')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = debug_dir / 'axis_ordering_debug.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved debug visualization: {plot_path}")
        
        # Also create individual slice views for better understanding
        self._create_slice_views(volume_np, mask_np, debug_dir, volume_name, mask_name)
        
        plt.close()
    
    def _create_slice_views(self, volume_np: np.ndarray, mask_np: np.ndarray, 
                           debug_dir: Path, volume_name: str, mask_name: str):
        """Create 2D slice views along each axis to understand orientation."""
        
        # Volume slices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Get middle slices
        mid_x = volume_np.shape[0] // 2
        mid_y = volume_np.shape[1] // 2  
        mid_z = volume_np.shape[2] // 2
        
        # Volume slices
        axes[0, 0].imshow(volume_np[mid_x, :, :], aspect='auto', origin='lower')
        axes[0, 0].set_title(f'{volume_name} - X={mid_x} slice\n(Y vs Z)')
        axes[0, 0].set_xlabel('Z axis')
        axes[0, 0].set_ylabel('Y axis')
        
        axes[0, 1].imshow(volume_np[:, mid_y, :], aspect='auto', origin='lower')
        axes[0, 1].set_title(f'{volume_name} - Y={mid_y} slice\n(X vs Z)')
        axes[0, 1].set_xlabel('Z axis')
        axes[0, 1].set_ylabel('X axis')
        
        axes[0, 2].imshow(volume_np[:, :, mid_z], aspect='auto', origin='lower')
        axes[0, 2].set_title(f'{volume_name} - Z={mid_z} slice\n(X vs Y)')
        axes[0, 2].set_xlabel('Y axis')
        axes[0, 2].set_ylabel('X axis')
        
        # Mask slices - should now have same shape as volume
        axes[1, 0].imshow(mask_np[mid_x, :, :], aspect='auto', origin='lower', cmap='Reds')
        axes[1, 0].set_title(f'{mask_name} - X={mid_x} slice\n(Y vs Z)')
        axes[1, 0].set_xlabel('Z axis')
        axes[1, 0].set_ylabel('Y axis')
        
        axes[1, 1].imshow(mask_np[:, mid_y, :], aspect='auto', origin='lower', cmap='Reds')
        axes[1, 1].set_title(f'{mask_name} - Y={mid_y} slice\n(X vs Z)')
        axes[1, 1].set_xlabel('Z axis')
        axes[1, 1].set_ylabel('X axis')
        
        axes[1, 2].imshow(mask_np[:, :, mid_z], aspect='auto', origin='lower', cmap='Reds')
        axes[1, 2].set_title(f'{mask_name} - Z={mid_z} slice\n(X vs Y)')
        axes[1, 2].set_xlabel('Y axis')
        axes[1, 2].set_ylabel('X axis')
        
        plt.tight_layout()
        
        # Save slice views
        slice_path = debug_dir / 'slice_views_debug.png'
        plt.savefig(slice_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved slice views: {slice_path}")
        
        plt.close()
    
    def run_evaluation(self):
        """Run the complete evaluation process."""
        self.logger.info("Starting GBM evaluation...")
        
        # Load components
        self.load_model()
        self.load_masks()
        self.create_test_dataloader()
        
        # Initialize results storage
        self.results = {
            'whole_volume': [],
            'regions': {region: [] for region in self.region_names},
            'timepoint_metrics': {
                'whole_volume': {},  # metric_name -> {timepoint_idx -> [list of values across batches]}
                'regions': {region: {} for region in self.region_names}  # region -> metric_name -> {timepoint_idx -> [list of values]}
            }
        }
        
        # Evaluation loop
        num_batches = self.config['evaluation_params'].get('num_batches')
        total_batches = len(self.test_loader)
        if num_batches:
            total_batches = min(num_batches, total_batches)
        
        self.logger.info(f"Evaluating on {total_batches} batches...")
        
        # Flag to create debug visualizations after first forward pass
        debug_visualizations_created = False
        
        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=total_batches, desc="Evaluating")
            
            for batch_idx, batch_data in pbar:
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Prepare data
                if isinstance(batch_data, (list, tuple)):
                    sequences, _ = batch_data
                else:
                    sequences = batch_data
                
                sequences = sequences.to(self.device)
                
                # Prepare seq2seq data
                input_seq, target_seq = self.prepare_seq2seq_data(sequences)
                
                # Model forward pass
                pred_logits = self.model(input_seq, get_logits=True)
                
                # Create debug visualizations after first forward pass
                if not debug_visualizations_created:
                    debug_visualizations_created = True
                    
                    # Create time-averaged volume from target data
                    time_averaged_volume = target_seq.mean(dim=(0, 1))  # Average over batch and time: (X, Y, Z)
                    
                    # Get whole_brain mask if available
                    try:
                        whole_brain_mask = self.mask_loader.get_mask('whole_brain')
                        self.create_3d_debug_visualizations(
                            time_averaged_volume, 
                            whole_brain_mask,
                            "Time-Averaged Volume (from targets)", 
                            "Whole Brain Mask"
                        )
                    except KeyError:
                        self.logger.warning("whole_brain mask not found, skipping debug visualization")
                        # Try with any available mask for debugging
                        if self.region_names:
                            first_mask = self.mask_loader.get_mask(self.region_names[0])
                            self.create_3d_debug_visualizations(
                                time_averaged_volume,
                                first_mask, 
                                "Time-Averaged Volume (from targets)",
                                f"First Available Mask ({self.region_names[0]})"
                            )
                    except Exception as e:
                        self.logger.warning(f"Could not create debug visualizations: {e}")
                
                # Compute metrics for whole volume (averaged over batch)
                whole_volume_metrics = self.compute_volume_metrics(pred_logits, target_seq)
                self.results['whole_volume'].append(whole_volume_metrics)
                
                # Compute timepoint-specific metrics for whole volume
                if self.config['metrics']['compute_per_timepoint']:
                    timepoint_metrics = self.compute_timepoint_metrics(pred_logits, target_seq)
                    # Store metrics per timepoint index, averaging over batch dimension
                    for t_idx, t_metrics_dict in enumerate(timepoint_metrics):
                        # t_metrics_dict is a dictionary like {'bce_loss': 0.1, 'f1_score': 0.8, ...}
                        for metric_name, metric_value in t_metrics_dict.items():
                            if metric_name not in self.results['timepoint_metrics']['whole_volume']:
                                self.results['timepoint_metrics']['whole_volume'][metric_name] = defaultdict(list)
                            self.results['timepoint_metrics']['whole_volume'][metric_name][t_idx].append(metric_value)
                
                # Compute metrics for each brain region
                for region_name in self.region_names:
                    try:
                        mask = self.mask_loader.get_mask(region_name)
                        
                        # Region metrics (averaged over time and batch)
                        region_metrics = self.compute_volume_metrics(pred_logits, target_seq, mask)
                        self.results['regions'][region_name].append(region_metrics)
                        
                        # Timepoint-specific region metrics
                        if self.config['metrics']['compute_per_timepoint']:
                            region_timepoint_metrics = self.compute_timepoint_metrics(
                                pred_logits, target_seq, mask
                            )
                            # Store metrics per timepoint index, averaging over batch dimension
                            for t_idx, t_metrics_dict in enumerate(region_timepoint_metrics):
                                for metric_name, metric_value in t_metrics_dict.items():
                                    if metric_name not in self.results['timepoint_metrics']['regions'][region_name]:
                                        self.results['timepoint_metrics']['regions'][region_name][metric_name] = defaultdict(list)
                                    self.results['timepoint_metrics']['regions'][region_name][metric_name][t_idx].append(metric_value)
                    
                    except Exception as e:
                        self.logger.warning(f"Error computing metrics for region {region_name}: {e}")
                
                # Update progress bar
                if self.results['whole_volume']:
                    latest_loss = self.results['whole_volume'][-1].get('bce_loss', 0)
                    pbar.set_postfix({'BCE Loss': f'{latest_loss:.4f}'})
                
                # Clean up GPU memory
                del sequences, input_seq, target_seq, pred_logits
                torch.cuda.empty_cache()
        
        self.logger.info("Evaluation completed!")
        
        # Generate results and visualizations
        self.compute_summary_statistics()
        if self.config['visualization']['create_heatmaps']:
            self.create_heatmaps()
        if self.config['output']['save_summary_csv']:
            self.save_results_to_csv()
        if self.config['output']['create_report']:
            self.create_evaluation_report()
    
    def compute_summary_statistics(self):
        """Compute summary statistics from evaluation results."""
        self.logger.info("Computing summary statistics...")
        
        self.summary_stats = {}
        
        # Whole volume statistics
        if self.results['whole_volume']:
            whole_vol_data = self.results['whole_volume']
            self.summary_stats['whole_volume'] = self._compute_stats_for_data(whole_vol_data)
        
        # Region statistics  
        self.summary_stats['regions'] = {}
        for region_name in self.region_names:
            if self.results['regions'][region_name]:
                region_data = self.results['regions'][region_name]
                self.summary_stats['regions'][region_name] = self._compute_stats_for_data(region_data)
    
    def _compute_stats_for_data(self, data_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistics (mean, std, min, max) for a list of metric dictionaries."""
        if not data_list:
            return {}
        
        # Get all metric names
        metric_names = set()
        for data in data_list:
            metric_names.update(data.keys())
        
        stats = {}
        for metric in metric_names:
            values = [data.get(metric, 0) for data in data_list]
            values = [v for v in values if not (np.isnan(v) or np.isinf(v))]  # Filter invalid values
            
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                stats[metric] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return stats
    
    def create_heatmaps(self):
        """Create time-series heatmaps for metrics visualization."""
        self.logger.info("Creating heatmaps...")
        
        if not self.config['metrics']['compute_per_timepoint']:
            self.logger.warning("Per-timepoint metrics not computed - skipping heatmaps")
            return
        
        viz_config = self.config['visualization']
        figsize = tuple(viz_config['heatmap_figsize'])
        max_regions = viz_config.get('max_regions_in_heatmap', 20)
        
        # Select top regions by activity for heatmap
        selected_regions = self.region_names[:max_regions]
        
        metrics_to_plot = []
        if self.config['metrics']['compute_bce_loss']:
            metrics_to_plot.append('bce_loss')
        if self.config['metrics']['compute_magnitude_error']:
            metrics_to_plot.append('magnitude_error')
        if self.config['metrics']['compute_f1_scores']:
            metrics_to_plot.append('f1_score')
        
        for metric_name in metrics_to_plot:
            try:
                self._create_metric_heatmap(metric_name, selected_regions, figsize)
            except Exception as e:
                self.logger.warning(f"Failed to create heatmap for {metric_name}: {e}")
    
    def _create_metric_heatmap(self, metric_name: str, region_names: List[str], figsize: Tuple[int, int]):
        """Create a heatmap for a specific metric."""
        # Collect timepoint data for each region
        heatmap_data = []
        row_labels = ['Whole Volume'] + region_names
        
        # Check if metric data exists
        if metric_name not in self.results['timepoint_metrics']['whole_volume']:
            self.logger.warning(f"No {metric_name} data available for heatmap")
            return
        
        # Determine number of timepoints from the data
        max_timepoints = 0
        whole_vol_metric_data = self.results['timepoint_metrics']['whole_volume'][metric_name]
        if whole_vol_metric_data:
            max_timepoints = max(max_timepoints, max(whole_vol_metric_data.keys()))
        
        for region in region_names:
            if metric_name in self.results['timepoint_metrics']['regions'][region]:
                region_metric_data = self.results['timepoint_metrics']['regions'][region][metric_name]
                if region_metric_data:
                    max_timepoints = max(max_timepoints, max(region_metric_data.keys()))
        
        if max_timepoints == 0:
            self.logger.warning(f"No timepoint data available for {metric_name} heatmap")
            return
        
        num_timepoints = max_timepoints + 1  # 0-indexed
        
        # Whole volume data - average across batches for each timepoint
        whole_vol_values = []
        for t in range(num_timepoints):
            if t in whole_vol_metric_data:
                metric_values = whole_vol_metric_data[t]  # List of values across batches
                if metric_values:
                    whole_vol_values.append(np.mean(metric_values))
                else:
                    whole_vol_values.append(np.nan)
            else:
                whole_vol_values.append(np.nan)
        heatmap_data.append(whole_vol_values)
        
        # Region data - average across batches for each timepoint
        for region in region_names:
            region_values = []
            if metric_name in self.results['timepoint_metrics']['regions'][region]:
                region_metric_data = self.results['timepoint_metrics']['regions'][region][metric_name]
                for t in range(num_timepoints):
                    if t in region_metric_data:
                        metric_values = region_metric_data[t]  # List of values across batches
                        if metric_values:
                            region_values.append(np.mean(metric_values))
                        else:
                            region_values.append(np.nan)
                    else:
                        region_values.append(np.nan)
            else:
                # No data for this metric in this region
                region_values = [np.nan] * num_timepoints
            heatmap_data.append(region_values)
        
        # Convert to numpy array
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        plt.figure(figsize=figsize)
        
        # Use appropriate colormap based on metric
        if metric_name == 'bce_loss':
            cmap = 'Reds'
            vmin, vmax = None, None
        elif metric_name == 'magnitude_error':
            cmap = 'RdBu_r'
            vmin, vmax = -1, 1  # Center around 0
        elif metric_name == 'f1_score':
            cmap = 'Greens'
            vmin, vmax = 0, 1
        else:
            cmap = 'viridis'
            vmin, vmax = None, None
        
        sns.heatmap(
            heatmap_array,
            yticklabels=row_labels,
            cmap=cmap,
            cbar_kws={'label': metric_name.replace('_', ' ').title()},
            vmin=vmin,
            vmax=vmax,
            fmt='.3f' if metric_name != 'f1_score' else '.2f'
        )
        
        plt.title(f'{metric_name.replace("_", " ").title()} Across Time and Brain Regions')
        plt.xlabel('Timepoint')
        plt.ylabel('Brain Region')
        plt.tight_layout()
        
        # Save plot
        if self.config['visualization']['save_plots']:
            plot_path = self.plots_dir / f'{metric_name}_heatmap.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved heatmap: {plot_path}")
        
        # Show plot if requested
        if self.config['visualization']['show_plots']:
            plt.show()
        else:
            plt.close()
    
    def save_results_to_csv(self):
        """Save summary results to CSV files."""
        self.logger.info("Saving results to CSV...")
        
        # Summary statistics CSV
        summary_data = []
        
        # Whole volume row
        if 'whole_volume' in self.summary_stats:
            whole_vol_stats = self.summary_stats['whole_volume']
            for metric, stats in whole_vol_stats.items():
                row = {
                    'region': 'whole_volume',
                    'metric': metric,
                    **stats
                }
                summary_data.append(row)
        
        # Region rows
        for region_name in self.region_names:
            if region_name in self.summary_stats['regions']:
                region_stats = self.summary_stats['regions'][region_name]
                for metric, stats in region_stats.items():
                    row = {
                        'region': region_name,
                        'metric': metric,
                        **stats
                    }
                    summary_data.append(row)
        
        # Save to CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = self.output_dir / 'evaluation_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            self.logger.info(f"Saved summary results: {summary_csv_path}")
    
    def create_evaluation_report(self):
        """Create a comprehensive evaluation report."""
        self.logger.info("Creating evaluation report...")
        
        report_path = self.output_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("GBM MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {self.config['evaluation']['model_type']}\n")
            f.write(f"Model Path: {self.config['evaluation']['model_path']}\n")
            f.write(f"Test Subjects: {self.config['data']['test_subjects']}\n")
            f.write(f"Number of Brain Regions: {len(self.region_names)}\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Whole volume results
            f.write("WHOLE VOLUME RESULTS\n")
            f.write("-" * 30 + "\n")
            if 'whole_volume' in self.summary_stats:
                whole_vol_stats = self.summary_stats['whole_volume']
                for metric, stats in whole_vol_stats.items():
                    f.write(f"{metric.upper()}:\n")
                    f.write(f"  Mean: {stats['mean']:.6f}\n")
                    f.write(f"  Std:  {stats['std']:.6f}\n")
                    f.write(f"  Min:  {stats['min']:.6f}\n")
                    f.write(f"  Max:  {stats['max']:.6f}\n")
                    f.write(f"  Count: {stats['count']}\n\n")
            
            # Best and worst performing regions
            f.write("REGION PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            if self.summary_stats.get('regions'):
                # Sort regions by F1 score if available
                if any('f1_score' in stats for stats in self.summary_stats['regions'].values()):
                    region_f1_scores = []
                    for region, stats in self.summary_stats['regions'].items():
                        if 'f1_score' in stats:
                            region_f1_scores.append((region, stats['f1_score']['mean']))
                    
                    region_f1_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    f.write("Top 5 Regions by F1 Score:\n")
                    for i, (region, f1_score) in enumerate(region_f1_scores[:5]):
                        f.write(f"  {i+1}. {region}: {f1_score:.4f}\n")
                    
                    f.write("\nBottom 5 Regions by F1 Score:\n")
                    for i, (region, f1_score) in enumerate(region_f1_scores[-5:]):
                        f.write(f"  {i+1}. {region}: {f1_score:.4f}\n")
                    f.write("\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 15 + "\n")
            f.write(yaml.dump(self.config, default_flow_style=False, indent=2))
        
        self.logger.info(f"Saved evaluation report: {report_path}")


def generate_default_config_file(output_path: str):
    """Generate a default configuration file."""
    save_default_eval_config(output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained GBM models.")
    
    # Config generation
    parser.add_argument(
        '--generate-config',
        type=str,
        metavar='PATH',
        help="Generate a default evaluation config file at the specified path and exit."
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help="Path to the YAML evaluation configuration file."
    )
    
    args = parser.parse_args()
    
    if args.generate_config:
        try:
            generate_default_config_file(args.generate_config)
            print(f"Default evaluation config file generated at: {args.generate_config}")
        except Exception as e:
            print(f"Error generating config file: {e}")
        return
    
    if not args.config:
        parser.error("The --config argument is required unless --generate-config is used.")
    
    # Load configuration
    try:
        config = load_eval_config(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    # Validate required parameters
    if not config['evaluation']['model_path'] and config['evaluation']['model_type'] == 'single':
        print("Error: model_path is required for single model evaluation")
        return
    
    if config['evaluation']['model_type'] == 'ensemble' and not config['evaluation']['ensemble_paths']:
        print("Error: ensemble_paths is required for ensemble evaluation")
        return
    
    # Run evaluation
    try:
        evaluator = GBMEvaluator(config)
        evaluator.run_evaluation()
        print(f"Evaluation completed! Results saved to: {evaluator.output_dir}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main() 