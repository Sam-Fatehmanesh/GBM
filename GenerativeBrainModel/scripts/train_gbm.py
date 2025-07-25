#!/usr/bin/env python3
"""
Main training script for GBM (Generative Brain Model) with seq2seq training.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import pdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from GenerativeBrainModel.models.gbm import GBM, EnsembleGBM
from GenerativeBrainModel.dataloaders.volume_dataloader import create_dataloaders, get_volume_info
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_validation_video


# Configuration utilities embedded in this script
def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary for GBM training.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        'experiment': {
            'name': 'gbm_training',
            'description': 'GBM (Generative Brain Model) sequence-to-sequence training on 3D volumetric spike data',
            'tags': ['gbm', 'seq2seq', '3d-volumes', 'spike-data', 'temporal-modeling']
        },
        
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [
                'subject_1',
                'subject_4', 
                'subject_5'
            ],
            'use_cache': False  # Test performance with caching disabled
        },
        
        'model': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'autoencoder_path': None,  # Path to pretrained autoencoder checkpoint
            'volume_size': [256, 128, 30],
            'region_size': [32, 16, 2],
            'ensemble_size': 1,  # Number of models in ensemble (1 = single model)
            'different_seeds': True  # Use different seeds for ensemble models
        },
        
        'training': {
            'volumes_per_batch': 4,  # Number of sequences per batch
            'num_epochs': 100,
            'learning_rate': 0.0005,  # Lower LR for fine-tuning
            'weight_decay': 1e-4,
            'scheduler': 'linear_warmup',  # Linear warmup for first 10% of batches
            'gradient_clip_norm': 1.0,
            'validation_frequency': 8,  # Number of times to run validation per epoch
            
            # Sequence parameters for GBM (seq2seq training)
            'sequence_length': 8,  # Length of input sequences (longer for temporal modeling)
            'stride': 2,  # Stride between sequences (overlap for more training data)
            'max_timepoints_per_subject': None,  # Max timepoints per subject file (None = use all available)
            
            # Hardware settings
            'use_gpu': True,
            'pin_memory': False,  # Large 3-D volumes + shared memory → bus errors; default off
            'persistent_workers': False,  # Safer default; can enable if system SHM is large
            'prefetch_factor': 2,  # Reduce SHM usage vs default 8
            'num_workers': 2,
            'mixed_precision': True,
            'compile_model': False,  # PyTorch 2.0 compile
            
            # Random seed
            'seed': 42
        },
        
        'loss': {
            'loss_function': 'bce',  # BCE for probability prediction
            'loss_weights': {
                'reconstruction': 1.0,
                'regularization': 0.0
            }
        },
        
        'logging': {
            'log_level': 'INFO',
            'log_frequency': 10,  # Log every N batches
            'save_checkpoint_frequency': 5,  # Save checkpoint every N epochs
            'keep_n_checkpoints': 3
        },
        
        'paths': {
            'output_base_dir': 'experiments/gbm',
            'checkpoint_dir': None,  # Will be set automatically
            'log_dir': None,  # Will be set automatically
            'plot_dir': None  # Will be set automatically
        }
    }


def generate_config_file(output_path: str, overrides: Dict = None) -> None:
    """
    Generate a YAML configuration file with optional overrides.
    
    Args:
        output_path: Path to save the config file
        overrides: Dictionary of config overrides
    """
    config = create_default_config()
    
    # Apply overrides if provided
    if overrides:
        config = deep_update(config, overrides)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    print(f"Configuration file saved to: {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    # Load user config
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)
    # Merge with defaults to fill missing keys
    default_config = create_default_config()
    config = deep_update(default_config, user_config)
    # Validate and set automatic paths
    config = setup_experiment_paths(config)
    return config


def setup_experiment_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up experiment directory structure and paths.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration with paths set
    """
    # Create timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['experiment']['name']
    
    # Base experiment directory
    base_dir = Path(config['paths']['output_base_dir']) / f"{experiment_name}_{timestamp}"
    
    # Set all paths
    config['paths']['checkpoint_dir'] = str(base_dir / 'checkpoints')
    config['paths']['log_dir'] = str(base_dir / 'logs')
    config['paths']['plot_dir'] = str(base_dir / 'logs' / 'plots')
    config['paths']['experiment_dir'] = str(base_dir)
    
    # Create directories
    for path_key in ['checkpoint_dir', 'log_dir', 'plot_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check data directory exists
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for H5 files
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        raise ValueError(f"No H5 files found in data directory: {data_dir}")
    
    # Validate test subjects exist
    test_subjects = config['data'].get('test_subjects', [])
    if test_subjects:
        missing_subjects = []
        for subject in test_subjects:
            if not (data_dir / f"{subject}.h5").exists():
                missing_subjects.append(subject)
        if missing_subjects:
            print(f"Warning: Test subjects not found: {missing_subjects}")
    
    # Validate volume/region size compatibility
    model_config = config['model']
    volume_size = model_config['volume_size']
    region_size = model_config['region_size']
    
    for i in range(3):
        if volume_size[i] % region_size[i] != 0:
            raise ValueError(
                f"Volume size {volume_size} not divisible by region size {region_size} "
                f"at dimension {i}"
            )
    
    # Check autoencoder path if provided
    autoencoder_path = model_config.get('autoencoder_path')
    if autoencoder_path and not Path(autoencoder_path).exists():
        raise ValueError(f"Autoencoder checkpoint not found: {autoencoder_path}")
    
    # Validate sequence length
    seq_len = config['training']['sequence_length']
    if seq_len < 2:
        raise ValueError(f"Sequence length must be at least 2 for seq2seq training, got {seq_len}")
    
    print(f"Configuration validation passed!")
    print(f"GBM Model: d_model={model_config['d_model']}, n_heads={model_config['n_heads']}, n_layers={model_config['n_layers']}")
    print(f"Sequence length: {seq_len}, Ensemble size: {model_config['ensemble_size']}")


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


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


class GBMTrainer:
    """
    Comprehensive GBM trainer with sequence-to-sequence training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_random_seeds()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.early_stopping_counter = 0
        
        # Setup experiment directory
        self.experiment_dir = Path(config['paths']['experiment_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        
        # Initialize metrics tracker
        self.metrics_tracker = CombinedMetricsTracker(
            log_dir=config['paths']['log_dir'],
            validation_threshold=0.5,
            ema_alpha=0.05  # EMA smoothing factor for training loss
        )
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['paths']['log_dir'])
        log_file = log_dir / 'training.log'
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def setup_device(self):
        """Set up compute device (GPU/CPU)."""
        if self.config['training']['use_gpu'] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
    
    def setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config['training']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Make sure operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.logger.info(f"Random seed set to: {seed}")
    
    def build_model(self):
        """Build and initialize the GBM model."""
        model_config = self.config['model']
        
        ensemble_size = model_config.get('ensemble_size', 1)
        
        if ensemble_size > 1:
            # Create ensemble model
            self.model = EnsembleGBM(
                n_models=ensemble_size,
                d_model=model_config['d_model'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                autoencoder_path=model_config.get('autoencoder_path'),
                volume_size=tuple(model_config['volume_size']),
                region_size=tuple(model_config['region_size']),
                different_seeds=model_config.get('different_seeds', True)
            )
            self.logger.info(f"Created EnsembleGBM with {ensemble_size} models")
        else:
            # Create single GBM model
            self.model = GBM(
                d_model=model_config['d_model'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                autoencoder_path=model_config.get('autoencoder_path'),
                volume_size=tuple(model_config['volume_size']),
                region_size=tuple(model_config['region_size'])
            )
            self.logger.info("Created single GBM model")
        
        autoencoder_path = model_config.get('autoencoder_path')
        if autoencoder_path:
            self.logger.info(f"Loaded pretrained autoencoder from: {autoencoder_path}")
        else:
            self.logger.warning("No autoencoder path provided - using randomly initialized autoencoder")
        
        self.logger.info(f"Model volume size: {model_config['volume_size']}, region size: {model_config['region_size']}")
        
        self.model.to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config['training'].get('compile_model', False):
            # Disable Dynamo compilation for Mamba layers which already ship Triton kernels
            try:
                import torch._dynamo as _dynamo
                from mamba_ssm import Mamba2 as Mamba

                def _disable_mamba(module: torch.nn.Module):
                    if isinstance(module, Mamba):
                        module.forward = _dynamo.disable(module.forward)
                self.model.apply(_disable_mamba)

                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with PyTorch 2.0 (Mamba layers excluded from compilation)")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        untrainable_params = total_params - trainable_params
        
        self.logger.info(f"Model created: {self.model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Untrainable parameters: {untrainable_params:,}")
        self.logger.info(f"Model size: {total_params * 4 / 1e6:.1f} MB (float32)")
        
        # Save architecture info to file
        self.save_architecture_info(total_params, trainable_params, untrainable_params)
    
    def save_architecture_info(self, total_params: int, trainable_params: int, untrainable_params: int):
        """
        Save detailed model architecture information to a text file.
        
        Args:
            total_params: Total number of parameters
            trainable_params: Number of trainable parameters
            untrainable_params: Number of untrainable parameters
        """
        architecture_file = self.experiment_dir / 'architecture.txt'
        
        with open(architecture_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GBM MODEL ARCHITECTURE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic model info
            f.write(f"Model Class: {self.model.__class__.__name__}\n")
            f.write(f"Experiment: {self.config['experiment']['name']}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            model_config = self.config['model']
            for key, value in model_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Parameter counts
            f.write("PARAMETER SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total Parameters:      {total_params:,}\n")
            f.write(f"  Trainable Parameters:   {trainable_params:,}\n")
            f.write(f"  Untrainable Parameters: {untrainable_params:,}\n")
            f.write(f"  Model Size (float32):   {total_params * 4 / 1e6:.2f} MB\n")
            f.write(f"  Model Size (float16):   {total_params * 2 / 1e6:.2f} MB\n\n")
            
            # Detailed parameter breakdown by layer
            f.write("PARAMETER BREAKDOWN BY LAYER:\n")
            f.write("-" * 40 + "\n")
            for name, param in self.model.named_parameters():
                param_count = param.numel()
                trainable = "✓" if param.requires_grad else "✗"
                f.write(f"  {name:<40} {param_count:>10,} [{trainable}] {tuple(param.shape)}\n")
            f.write("\n")
            
            # Buffer information (non-trainable tensors)
            f.write("REGISTERED BUFFERS:\n")
            f.write("-" * 40 + "\n")
            for name, buffer in self.model.named_buffers():
                buffer_count = buffer.numel() if buffer is not None else 0
                shape = tuple(buffer.shape) if buffer is not None else "None"
                f.write(f"  {name:<40} {buffer_count:>10,} {shape}\n")
            f.write("\n")
            
            # Model architecture string representation
            f.write("DETAILED MODEL ARCHITECTURE:\n")
            f.write("-" * 40 + "\n")
            f.write(str(self.model))
            f.write("\n\n")
            
            # Training configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            training_config = self.config['training']
            for key, value in training_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Hardware info
            f.write("HARDWARE INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Device: {self.device}\n")
            if self.device.type == 'cuda':
                f.write(f"  GPU Name: {torch.cuda.get_device_name()}\n")
                f.write(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
                f.write(f"  CUDA Version: {torch.version.cuda}\n")
            f.write(f"  PyTorch Version: {torch.__version__}\n")
            f.write(f"  Mixed Precision: {self.config['training'].get('mixed_precision', False)}\n")
            f.write(f"  Model Compilation: {self.config['training'].get('compile_model', False)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        self.logger.info(f"Model architecture saved to: {architecture_file}")
    
    def build_optimizer(self):
        """Build optimizer and scheduler."""
        training_config = self.config['training']
        
        # Optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        self.logger.info(f"Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'linear_warmup':
            # Calculate total batches for warmup (10% of first epoch)
            warmup_batches = int(0.1 * len(self.train_loader))
            
            def lr_lambda(batch):
                if batch < warmup_batches:
                    return float(batch) / float(max(1, warmup_batches))
                return 1.0
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            self.logger.info(f"Using Linear Warmup scheduler with {warmup_batches} warmup batches.")
        
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config['learning_rate'] * 0.01
            )
            self.logger.info("Using Cosine Annealing scheduler.")
            
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['num_epochs'] // 3,
                gamma=0.1
            )
            self.logger.info("Using Step LR scheduler.")
            
        else:
            self.scheduler = None
            self.logger.info("No scheduler will be used.")
        
        # Mixed precision scaler
        if training_config.get('mixed_precision', False):
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"Scheduler: {scheduler_type}")
    
    def build_dataloaders(self):
        """Build training and validation dataloaders."""
        self.logger.info("Building dataloaders...")
        
        # Get data info to determine volume size
        try:
            volume_info = get_volume_info(self.config['data']['data_dir'])
            # Check if volume_size is manually specified in model config
            if self.config['model'].get('volume_size') and self.config['model']['volume_size'] != volume_info.get('volume_size'):
                self.logger.warning(f"Using config volume_size {self.config['model']['volume_size']} instead of detected size {volume_info['volume_size']}.")
            elif not self.config['model'].get('volume_size'):
                # Auto-detect and set in model config
                self.config['model']['volume_size'] = volume_info['volume_size']
                self.logger.info(f"Auto-detected volume_size: {volume_info['volume_size']}")
            self.logger.info(f"Auto-detected dataset info: {volume_info}")
        except Exception as e:
            self.logger.error(f"Could not auto-detect volume info: {e}")
            if not self.config['model'].get('volume_size'):
                raise ValueError("Could not determine volume_size from data directory and it is not specified in model config.")
            self.logger.info(f"Using volume_size from model config: {self.config['model']['volume_size']}")

        self.train_loader, self.test_loader = create_dataloaders(self.config)
        self.logger.info(f"Train loader: {len(self.train_loader.dataset)} samples")
        self.logger.info(f"Test loader: {len(self.test_loader.dataset)} samples")
    
    def get_loss_function(self):
        """Get loss function based on configuration."""
        loss_type = self.config['loss']['loss_function'].lower()
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def prepare_seq2seq_data(self, sequences):
        """
        Prepare input and target sequences for seq2seq training.
        
        Args:
            sequences: Tensor of shape (B, T, X, Y, Z)
            
        Returns:
            Tuple of (input_sequences, target_sequences)
            - input_sequences: (B, T-1, X, Y, Z) - sequences[0:T-1]
            - target_sequences: (B, T-1, X, Y, Z) - sequences[1:T]
        """
        B, T, X, Y, Z = sequences.shape
        
        if T < 2:
            raise ValueError(f"Sequence length must be at least 2 for seq2seq training, got {T}")
        
        # Input: all timesteps except the last
        input_seq = sequences[:, :-1, :, :, :]  # (B, T-1, X, Y, Z)
        
        # Target: all timesteps except the first  
        target_seq = sequences[:, 1:, :, :, :]   # (B, T-1, X, Y, Z)
        
        return input_seq, target_seq

    def run_validation(self, epoch: int, batch_idx: int):
        """
        Run validation on entire test dataset and log metrics using the metrics tracker.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
        """
        self.logger.info(f"Running validation at epoch {epoch}, batch {batch_idx}")
        
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        loss_fn = self.get_loss_function()
        
        # Initialize PR AUC binned accumulators to avoid storing all predictions
        threshold = self.metrics_tracker.threshold
        device = self.device
        num_bins = 1000
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
        tp_counts = torch.zeros(num_bins, device=device)
        total_counts = torch.zeros(num_bins, device=device)
        total_positives = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(self.test_loader, desc=f"Validation E{epoch}B{batch_idx}", leave=False, ncols=100)
            
            for val_batch_data in val_pbar:
                # To device
                if isinstance(val_batch_data, (list, tuple)):
                    val_sequences, _ = val_batch_data
                else:
                    val_sequences = val_batch_data
                
                val_sequences = val_sequences.to(self.device, non_blocking=True)
                
                # Prepare seq2seq data
                val_input, val_target = self.prepare_seq2seq_data(val_sequences)
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    # Get predicted next volumes
                    val_output = self.model(val_input, get_logits=True)
                    val_loss = loss_fn(val_output, val_target)
                
                total_val_loss += val_loss.item()
                num_batches += 1
                
                # Convert logits to probabilities for metrics calculation
                val_probabilities = torch.sigmoid(val_output)
                
                # Flatten predictions and targets
                batch_preds = val_probabilities.flatten()
                batch_targets = val_target.flatten()
                # Binarize and accumulate for PR AUC
                binary_targets = (batch_targets >= threshold).float()
                total_positives += binary_targets.sum().item()
                bin_indices = torch.searchsorted(bin_edges[1:], batch_preds, right=False)
                tp_counts.scatter_add_(0, bin_indices, binary_targets)
                total_counts.scatter_add_(0, bin_indices, torch.ones_like(binary_targets))
                # Update progress bar
                val_pbar.set_postfix({'Val Loss': f'{total_val_loss/num_batches:.4f}'})
                # Free GPU memory from this validation batch
                del val_sequences, val_input, val_target, val_output, val_loss, val_probabilities, batch_preds, batch_targets
                torch.cuda.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_batches
        # Compute PR AUC from accumulators
        tp_flipped = torch.flip(tp_counts, [0])
        total_flipped = torch.flip(total_counts, [0])
        cumulative_tp = torch.cumsum(tp_flipped, dim=0)
        cumulative_fp = torch.cumsum(total_flipped - tp_flipped, dim=0)
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)
        recall = cumulative_tp / (total_positives + 1e-8)
        precision = torch.cat([torch.tensor([1.0], device=device), precision])
        recall = torch.cat([torch.tensor([0.0], device=device), recall])
        recall_diff = recall[1:] - recall[:-1]
        pr_auc = torch.sum(recall_diff * precision[:-1]).item()
        # Cleanup bins
        del tp_counts, total_counts, tp_flipped, total_flipped, cumulative_tp, cumulative_fp
        del precision, recall, recall_diff, bin_edges
        torch.cuda.empty_cache()
        # Log validation metrics directly
        self.metrics_tracker.csv_logger.log_metrics({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'validation_loss': avg_val_loss,
            'pr_auc': pr_auc
        })
        self.logger.info(f"Validation - Loss: {avg_val_loss:.6f}, PR AUC: {pr_auc:.4f}")
        # Generate updated plots
        if self.metrics_tracker.plot_generator is not None:
            try:
                self.metrics_tracker.plot_generator.generate_training_plots()
            except Exception as e:
                self.logger.warning(f"Failed to generate plots: {e}")
        metrics = {'validation_loss': avg_val_loss, 'pr_auc': pr_auc}
        
        # Check if this is the best validation loss and save best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_best_model(epoch, batch_idx, avg_val_loss)
            self.early_stopping_counter = 0
            self.logger.info(f"New best validation loss: {avg_val_loss:.6f}")
        else:
            self.early_stopping_counter += 1
        
        self.model.train()
        return avg_val_loss

    def train(self):
        """Main training loop."""
        self.logger.info("Starting GBM training...")
        
        # Build all components
        self.build_model()
        self.build_dataloaders()  # Ensure train_loader exists for scheduler
        self.build_optimizer()
        
        # Save configuration
        config_path = self.experiment_dir / 'config.yaml'
        save_config(self.config, str(config_path))
        
        num_epochs = self.config['training']['num_epochs']
        total_batches = len(self.train_loader)
        
        self.current_epoch = 0
        global_step = 0

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self.model.train()
            
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=True)
            
            # Calculate validation frequency for this epoch
            validation_frequency = self.config['training'].get('validation_frequency', 8)
            total_batches = len(self.train_loader)
            validation_interval = max(1, total_batches // validation_frequency)

            for batch_idx, batch_data in enumerate(pbar):
                global_step += 1

                # To device
                if isinstance(batch_data, (list, tuple)):
                    sequences, _ = batch_data
                else:
                    sequences = batch_data
                
                sequences = sequences.to(self.device, non_blocking=True)
                
                # Prepare seq2seq data
                input_seq, target_seq = self.prepare_seq2seq_data(sequences)
                
                # Forward/backward pass
                self.optimizer.zero_grad()
                
                with autocast(enabled=self.scaler is not None):
                    # Predict next volumes
                    output_seq = self.model(input_seq, get_logits=True)
                    loss = self.get_loss_function()(output_seq, target_seq)
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    if self.config['training'].get('gradient_clip_norm'):
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config['training']['gradient_clip_norm']
                        )
                    else:
                        grad_norm = None
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config['training'].get('gradient_clip_norm'):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config['training']['gradient_clip_norm']
                        )
                    else:
                        grad_norm = None
                    self.optimizer.step()

                # Scheduler per batch for linear warmup
                if self.scheduler and self.config['training']['scheduler'] == 'linear_warmup':
                    self.scheduler.step()
                
                # Update running loss
                running_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Current batch loss
                batch_loss = loss.item()
                
                # Log training metrics with EMA
                self.metrics_tracker.log_training_step(
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                    loss=loss.item(),
                    learning_rate=current_lr
                )
 
                # Update progress bar with EMA loss
                ema_loss = self.metrics_tracker.get_current_training_ema()
                pbar.set_postfix({
                    'Loss': f'{batch_loss:.6f}',
                    'EMA Loss': f'{ema_loss:.6f}' if ema_loss else 'N/A',
                })
                
                # Run validation at specified frequency
                if (batch_idx + 1) % validation_interval == 0 or batch_idx == total_batches - 1:
                    self.run_validation(epoch, batch_idx + 1)
                # Free GPU tensors for this batch
                del sequences, input_seq, target_seq, output_seq, loss

            # Scheduler step per epoch for other schedulers
            if self.scheduler and self.config['training']['scheduler'] != 'linear_warmup':
                self.scheduler.step()
        
            # Save checkpoint at end of epoch
            if epoch % self.config['logging']['save_checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, global_step)
        
        self.logger.info("GBM training completed!")
        
        # Generate validation comparison video
        try:
            self.logger.info("Generating validation comparison video...")
            video_path = create_validation_video(
                model=self.model,
                validation_loader=self.test_loader,
                device=self.device,
                experiment_dir=self.experiment_dir,
                video_name="gbm_validation_comparison.mp4"
            )
            self.logger.info(f"Validation comparison video saved to: {video_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate validation video: {e}")
            self.logger.error("Continuing without video generation...")

    def save_checkpoint(self, epoch: int, step: int):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            step: Current global step
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(state, checkpoint_path)
        self.logger.info(f"Checkpoint saved at epoch {epoch}")

        # Cleanup old checkpoints
        self.cleanup_checkpoints()
        
    def save_best_model(self, epoch: int, batch_idx: int, val_loss: float):
        """
        Save the best model based on validation loss.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            val_loss: Validation loss that triggered this save
        """
        best_model_name = "best_gbm_model.pth"
        best_model_path = self.checkpoint_dir / best_model_name
        
        # Remove previous best model if it exists
        if self.best_model_path and self.best_model_path.exists():
            self.best_model_path.unlink()
        
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'validation_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(state, best_model_path)
        self.best_model_path = best_model_path
        self.logger.info(f"Best GBM model saved with validation loss: {val_loss:.6f} at epoch {epoch}, batch {batch_idx}")
        
        return best_model_path
        
    def cleanup_checkpoints(self):
        """Remove old checkpoints to save space."""
        keep_n = self.config['logging']['keep_n_checkpoints']
        if keep_n <= 0:
            return
        
        # Get all checkpoint files, sort by epoch number
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        if len(checkpoints) > keep_n:
            for checkpoint in checkpoints[:-keep_n]:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train GBM on 3D volume sequences.")
    
    # Config generation
    parser.add_argument(
        '--generate-config', 
        type=str,
        metavar='PATH',
        help="Generate a default config file at the specified path and exit."
    )
    
    # Config file
    parser.add_argument(
        '--config', 
        type=str, 
        help="Path to the YAML configuration file."
    )
    
    args = parser.parse_args()
    
    if args.generate_config:
        try:
            generate_config_file(args.generate_config)
            print(f"Default GBM config file generated at: {args.generate_config}")
        except Exception as e:
            print(f"Error generating config file: {e}")
        return

    if not args.config:
        parser.error("The --config argument is required unless --generate-config is used.")
        
    # Load and set up configuration
    config = load_config(args.config)
    config = setup_experiment_paths(config)
    validate_config(config)
    
    # Start training
    trainer = GBMTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 