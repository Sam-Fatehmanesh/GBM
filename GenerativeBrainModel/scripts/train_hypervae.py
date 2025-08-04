#!/usr/bin/env python3
"""
Main training script for HyperVAE with VAE loss (reconstruction + KL divergence).
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

# Import our modules
from GenerativeBrainModel.models.hypervae import HyperVAE
from GenerativeBrainModel.dataloaders.volume_dataloader import create_dataloaders, get_volume_info
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_validation_video


# Configuration utilities embedded in this script
def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary for HyperVAE training.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        'experiment': {
            'name': 'hypervae_training',
            'description': 'HyperVAE training on 3D volumetric spike data with VAE loss',
            'tags': ['vae', 'hypervae', '4d-conv', '3d-volumes', 'spike-data']
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
            'd_model': 32,  # Start with smaller model for VAE
            'volume_size': [256, 128, 30],
        },
        
        'training': {
            'batch_size': 2,  # Smaller batch for 4D conv VAE
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'muon',  # Use Muon optimizer
            # Muon-specific settings
            'muon_lr': 0.02,  # Learning rate for Muon (hidden weights)
            'muon_momentum': 0.95,  # Momentum for Muon
            'muon_nesterov': True,  # Use Nesterov momentum for Muon
            'muon_ns_steps': 5,  # Number of steps for Muon
            # AdamW settings for non-hidden params when using Muon
            'adamw_lr': 3e-4,  # Learning rate for AdamW params (when using Muon)
            'adamw_betas': [0.9, 0.95],  # Beta values for AdamW
            'adamw_eps': 1e-8,           # Epsilon for AdamW
            # Scheduler
            'scheduler': 'warmup_cosine',  # Linear warmup + cosine annealing
            'min_lr_ratio': 0.01,  # Minimum LR as ratio of initial LR
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': 1,  # Number of batches to accumulate before optimizer step
            'validation_frequency': 4,  # Number of times to run validation per epoch
            
            # Sequence parameters for VAE training
            'sequence_length': 4,  # Multiple timepoints for temporal modeling
            'stride': 2,  # Stride between sequences
            'max_timepoints_per_subject': None,  # Max timepoints per subject file (None = use all available)
            
            # Hardware settings
            'use_gpu': True,
            'pin_memory': False,  # Large 4-D volumes + shared memory → bus errors; default off
            'persistent_workers': False,  # Safer default
            'prefetch_factor': 2,  # Reduce SHM usage
            'num_workers': 2,
            'mixed_precision': True,
            'compile_model': False,  # PyTorch 2.0 compile
            
            # Random seed
            'seed': 42
        },
        
        'loss': {
            'beta': 1.0,  # KL divergence weight in VAE loss (beta-VAE parameter)
            'reconstruction_loss': 'bce',  # 'mse', 'mae', 'bce' for reconstruction
            'kl_annealing': False,  # Whether to anneal KL loss during training
            'kl_annealing_steps': 1000,  # Number of steps to linearly increase KL weight from 0 to beta
        },
        
        'logging': {
            'log_level': 'INFO',
            'log_frequency': 10,  # Log every N batches
            'save_checkpoint_frequency': 5,  # Save checkpoint every N epochs
            'keep_n_checkpoints': 3
        },
        
        'paths': {
            'output_base_dir': 'experiments/hypervae',
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
    
    # Validate sequence length
    seq_len = config['training']['sequence_length']
    if seq_len < 1:
        raise ValueError(f"Sequence length must be at least 1 for VAE training, got {seq_len}")
    
    print(f"Configuration validation passed!")
    print(f"HyperVAE Model: d_model={config['model']['d_model']}")
    print(f"Sequence length: {seq_len}")
    print(f"VAE beta (KL weight): {config['loss']['beta']}")


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


class HyperVAETrainer:
    """
    Comprehensive HyperVAE trainer with VAE loss (reconstruction + KL divergence).
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
        self.optimizer_stepped = False  # Track if optimizer has been called at least once
        
        # Setup experiment directory
        self.experiment_dir = Path(config['paths']['experiment_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        
        # Initialize metrics tracker with VAE loss component tracking
        self.metrics_tracker = CombinedMetricsTracker(
            log_dir=config['paths']['log_dir'],
            validation_threshold=0.5,
            ema_alpha=0.05,  # EMA smoothing factor for training loss
            track_loss_components=True  # Enable VAE loss component tracking
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
        """Build and initialize the HyperVAE model."""
        model_config = self.config['model']
        
        # Create HyperVAE model
        self.model = HyperVAE(
            d_model=model_config['d_model'],
            volume=tuple(model_config['volume_size'])
        )
        
        self.logger.info(f"Created HyperVAE model")
        self.logger.info(f"Model volume size: {model_config['volume_size']}")
        
        self.model.to(self.device)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config['training'].get('compile_model', False):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with PyTorch 2.0")
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
            f.write("HYPERVAE MODEL ARCHITECTURE SUMMARY\n")
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
            
            # VAE-specific info
            f.write("VAE CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            loss_config = self.config['loss']
            for key, value in loss_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Detailed parameter breakdown by layer
            f.write("PARAMETER BREAKDOWN BY LAYER:\n")
            f.write("-" * 40 + "\n")
            for name, param in self.model.named_parameters():
                param_count = param.numel()
                trainable = "✓" if param.requires_grad else "✗"
                f.write(f"  {name:<40} {param_count:>10,} [{trainable}] {tuple(param.shape)}\n")
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
        """Build optimizer and scheduler based on config."""
        training_config = self.config['training']
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Get optimizer type from config
        optimizer_type = training_config.get('optimizer', 'muon').lower()
        
        if optimizer_type == 'adamw':
            # Use pure AdamW optimizer
            learning_rate = training_config.get('learning_rate', 0.001)
            adamw_betas = training_config.get('adamw_betas', [0.9, 0.999])
            adamw_eps = training_config.get('adamw_eps', 1e-8)
            weight_decay = training_config.get('weight_decay', 1e-4)
            
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay
            )
            
            self.logger.info(f"Using AdamW optimizer:")
            self.logger.info(f"  Learning rate: {learning_rate}")
            self.logger.info(f"  Betas: {adamw_betas}")
            self.logger.info(f"  Weight decay: {weight_decay}")
            self.logger.info(f"  Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
            
            # Scheduler setup for AdamW
            scheduler_type = training_config.get('scheduler', 'warmup_cosine')
            base_lr = learning_rate
            
            if scheduler_type == 'warmup_cosine':
                # Linear warmup followed by cosine annealing
                warmup_batches = int(0.1 * len(self.train_loader))  # 10% of first epoch for warmup
                total_batches = training_config['num_epochs'] * len(self.train_loader)
                cosine_batches = total_batches - warmup_batches  # Remaining batches for cosine annealing
                
                # Get minimum learning rate (default to 1% of initial LR)
                min_lr_ratio = training_config.get('min_lr_ratio', 0.01)
                min_lr = base_lr * min_lr_ratio
                
                def lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    else:
                        cosine_batch = batch - warmup_batches
                        cosine_progress = cosine_batch / cosine_batches
                        cosine_progress = min(cosine_progress, 1.0)
                        
                        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * cosine_progress))
                        return lr / base_lr
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
                
                self.logger.info(f"Using Warmup + Cosine Annealing scheduler:")
                self.logger.info(f"  Warmup batches: {warmup_batches}")
                self.logger.info(f"  Total batches: {total_batches}")
                self.logger.info(f"  Min LR ratio: {min_lr_ratio} (min_lr: {min_lr:.2e})")
            else:
                self.scheduler = None
                self.logger.info("No scheduler will be used.")
                
        elif optimizer_type == 'muon':
            # Use Muon optimizer with hybrid approach
            try:
                from muon import MuonWithAuxAdam
            except ImportError:
                raise ImportError("Muon optimizer not found. Please install with: pip install git+https://github.com/KellerJordan/Muon")
            
            # Separate parameters according to Muon guidelines with model-structure awareness
            hidden_weights = []  # Conv filters/weight matrices (except first conv) - use Muon
            adamw_params = []    # First conv, biases, norms, attention, final layers - use AdamW
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Identify first convolutional layer (should use AdamW, not Muon)
                is_first_conv = name == 'encoder.0.weight'  # First CausalConv4D in encoder
                
                # Categorize parameters based on Muon recommendations
                if param.ndim >= 2 and not is_first_conv:
                    # Conv filters and weight matrices (except first conv) - use Muon
                    hidden_weights.append(param)
                else:
                    # First conv, biases, norms, embeddings, final layers - use AdamW
                    adamw_params.append(param)
            
            # Get optimizer settings
            muon_lr = training_config.get('muon_lr', 0.02)
            muon_momentum = training_config.get('muon_momentum', 0.95)
            muon_nesterov = training_config.get('muon_nesterov', True)
            muon_ns_steps = training_config.get('muon_ns_steps', 5)
            
            # AdamW optimizer for non-hidden parameters
            adamw_lr = training_config.get('adamw_lr', 3e-4)
            adamw_betas = training_config.get('adamw_betas', [0.9, 0.95])
            
            # Create parameter groups for MuonWithAuxAdam
            param_groups = []
            
            # Hidden weights use Muon (excluding first conv)
            if hidden_weights:
                param_groups.append({
                    'params': hidden_weights,
                    'use_muon': True,
                    'lr': muon_lr,
                    'momentum': muon_momentum,
                    'weight_decay': training_config['weight_decay']
                })
            
            # All other parameters (first conv, biases, norms, etc.) use AdamW
            if adamw_params:
                param_groups.append({
                    'params': adamw_params,
                    'use_muon': False,
                    'lr': adamw_lr,
                    'betas': training_config.get('adamw_betas', [0.9, 0.95]),
                    'eps': training_config.get('adamw_eps', 1e-8),
                    'weight_decay': training_config['weight_decay']
                })
            
            self.optimizer = MuonWithAuxAdam(param_groups)
            
            self.logger.info(f"Using Muon optimizer (following recommended best practices):")
            self.logger.info(f"  Hidden weights ({len(hidden_weights)} params): Muon (LR: {muon_lr})")
            self.logger.info(f"  First conv + other params ({len(adamw_params)} params): AdamW (LR: {adamw_lr})")
            self.logger.info(f"  First conv layer excluded from Muon as recommended")
            
            self.logger.info(f"Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
            
            # Scheduler - warmup + cosine annealing for Muon
            scheduler_type = training_config.get('scheduler', 'warmup_cosine')
            base_muon_lr = muon_lr
            base_adamw_lr = adamw_lr
            
            if scheduler_type == 'warmup_cosine':
                # Linear warmup followed by cosine annealing
                warmup_batches = int(0.1 * len(self.train_loader))  # 10% of first epoch for warmup
                total_batches = training_config['num_epochs'] * len(self.train_loader)
                cosine_batches = total_batches - warmup_batches  # Remaining batches for cosine annealing
                
                # Get minimum learning rate (default to 1% of initial LR)
                min_lr_ratio = training_config.get('min_lr_ratio', 0.01)
                min_muon_lr = base_muon_lr * min_lr_ratio
                min_adamw_lr = base_adamw_lr * min_lr_ratio
                
                def muon_lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    else:
                        cosine_batch = batch - warmup_batches
                        cosine_progress = cosine_batch / cosine_batches
                        cosine_progress = min(cosine_progress, 1.0)
                        
                        lr = min_muon_lr + 0.5 * (base_muon_lr - min_muon_lr) * (1 + np.cos(np.pi * cosine_progress))
                        return lr / base_muon_lr
                
                def adamw_lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    else:
                        cosine_batch = batch - warmup_batches
                        cosine_progress = cosine_batch / cosine_batches
                        cosine_progress = min(cosine_progress, 1.0)
                        
                        lr = min_adamw_lr + 0.5 * (base_adamw_lr - min_adamw_lr) * (1 + np.cos(np.pi * cosine_progress))
                        return lr / base_adamw_lr
                
                lr_lambdas = [muon_lr_lambda] if len(param_groups) == 1 else [muon_lr_lambda, adamw_lr_lambda]
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambdas)
                
                min_lrs = f"Muon: {min_muon_lr:.2e}, AdamW: {min_adamw_lr:.2e}"
                self.logger.info(f"Using Warmup + Cosine Annealing scheduler:")
                self.logger.info(f"  Warmup batches: {warmup_batches}")
                self.logger.info(f"  Total batches: {total_batches}")
                self.logger.info(f"  Min LR ratio: {min_lr_ratio} (min_lrs: {min_lrs})")
            else:
                self.scheduler = None
                self.logger.info("No scheduler will be used.")
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: 'adamw', 'muon'")
        
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
    
    def compute_vae_loss(self, x, reconstruction_logits, mu, logvar, global_step=None):
        """
        Compute VAE loss (reconstruction loss + beta * KL divergence) using model's built-in methods.
        
        Args:
            x: Input volumes of shape (B, T, vol_x, vol_y, vol_z)
            reconstruction_logits: Reconstructed volume logits of shape (B, T, vol_x, vol_y, vol_z)
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            global_step: Current global step (for KL annealing)
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss, beta_used)
        """
        # Use model's built-in loss methods
        recon_loss = self.model.reconstruction_loss(x, reconstruction_logits)
        kl_loss = self.model.kl_loss(mu, logvar)
        
        # Beta weighting for KL loss (beta-VAE)
        beta = self.config['loss']['beta']
        
        # KL annealing (optional) - linearly increase beta from 0 to target beta over annealing_steps
        if self.config['loss']['kl_annealing'] and global_step is not None:
            annealing_steps = int(self.config['loss']['kl_annealing_steps'])
            if global_step <= annealing_steps:
                beta = beta * (global_step / annealing_steps)
        
        # Total VAE loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, beta

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
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        # Collect predictions and targets in smaller chunks to avoid memory issues
        prediction_chunks = []
        target_chunks = []
        max_chunk_size = 25000  # Process in chunks of ~25k elements
        current_chunk_size = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.test_loader, desc=f"Validation E{epoch}B{batch_idx}", leave=False, ncols=100)
            
            for val_batch_data in val_pbar:
                # To device
                if isinstance(val_batch_data, (list, tuple)):
                    val_sequences, _ = val_batch_data
                else:
                    val_sequences = val_batch_data
                
                val_sequences = val_sequences.to(self.device, non_blocking=True)
                
                # Handle sequence length = 1 case: ensure time dimension exists
                # Expected shape: (B, T, vol_x, vol_y, vol_z)
                # If sequence_length = 1 and we get (B, vol_x, vol_y, vol_z), add time dim
                if self.config['training']['sequence_length'] == 1 and val_sequences.dim() == 4:
                    val_sequences = val_sequences.unsqueeze(1)  # Add time dimension: (B, vol_x, vol_y, vol_z) -> (B, 1, vol_x, vol_y, vol_z)
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    reconstruction_logits, mu, logvar = self.model(val_sequences)
                    total_loss, recon_loss, kl_loss, beta = self.compute_vae_loss(
                        val_sequences, reconstruction_logits, mu, logvar, None
                    )
                
                total_val_loss += total_loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
                
                # Apply sigmoid to get probabilities for metrics calculation
                val_probabilities = torch.sigmoid(reconstruction_logits)
                
                # Flatten predictions and targets
                batch_preds = val_probabilities.flatten()
                batch_targets = val_sequences.flatten()
                
                # Add to current chunk
                if len(prediction_chunks) == 0 or current_chunk_size + batch_preds.numel() > max_chunk_size:
                    # Start new chunk
                    prediction_chunks.append([batch_preds])
                    target_chunks.append([batch_targets])
                    current_chunk_size = batch_preds.numel()
                else:
                    # Add to current chunk
                    prediction_chunks[-1].append(batch_preds)
                    target_chunks[-1].append(batch_targets)
                    current_chunk_size += batch_preds.numel()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Val Loss': f'{total_val_loss/num_batches:.4f}',
                    'Recon': f'{total_recon_loss/num_batches:.4f}',
                    'KL': f'{total_kl_loss/num_batches:.4f}'
                })
        
        # Calculate average validation losses
        avg_val_loss = total_val_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        # Concatenate chunks and compute metrics
        # This processes data in manageable chunks rather than one huge tensor
        all_predictions = torch.cat([torch.cat(chunk) for chunk in prediction_chunks])
        all_targets = torch.cat([torch.cat(chunk) for chunk in target_chunks])
        
        # Use metrics tracker to compute and log validation metrics
        metrics = self.metrics_tracker.log_validation_step(
            epoch=epoch,
            batch_idx=batch_idx,
            predictions=all_predictions,
            targets=all_targets,
            validation_loss=avg_val_loss
        )
        
        # Log VAE-specific metrics
        self.logger.info(f"Validation - Total Loss: {avg_val_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KL Loss: {avg_kl_loss:.6f}")
        
        # GPU memory cleanup for large tensors
        del all_predictions, all_targets, prediction_chunks, target_chunks
        torch.cuda.empty_cache()
        
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
        self.logger.info("Starting HyperVAE training...")
        
        # If not in distributed mode, ensure Muon optimizer sees world size = 1 and stub all_gather
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.get_world_size = lambda group=None: 1
                dist.get_rank = lambda group=None: 0
                # Stub all_gather so Muon step doesn't require init_process_group
                def _fake_all_gather(tensor_list, tensor, group=None):
                    # For world_size=1, just copy tensor to output list
                    tensor_list[0].copy_(tensor)
                dist.all_gather = _fake_all_gather
        except ImportError:
            pass
        
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
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}', leave=True)
            
            # Calculate validation frequency for this epoch
            validation_frequency = self.config['training'].get('validation_frequency', 4)
            total_batches = len(self.train_loader)
            validation_interval = max(1, total_batches // validation_frequency)

            # Gradient accumulation setup
            gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
            
            for batch_idx, batch_data in enumerate(pbar):
                # To device
                if isinstance(batch_data, (list, tuple)):
                    sequences, _ = batch_data
                else:
                    sequences = batch_data
                
                sequences = sequences.to(self.device, non_blocking=True)
                
                # Handle sequence length = 1 case: ensure time dimension exists
                # Expected shape: (B, T, vol_x, vol_y, vol_z)
                # If sequence_length = 1 and we get (B, vol_x, vol_y, vol_z), add time dim
                if self.config['training']['sequence_length'] == 1 and sequences.dim() == 4:
                    sequences = sequences.unsqueeze(1)  # Add time dimension: (B, vol_x, vol_y, vol_z) -> (B, 1, vol_x, vol_y, vol_z)
                
                with autocast(enabled=self.scaler is not None):
                    # Forward pass through VAE
                    reconstruction_logits, mu, logvar = self.model(sequences)
                    total_loss, recon_loss, kl_loss, beta = self.compute_vae_loss(
                        sequences, reconstruction_logits, mu, logvar, global_step
                    )
                    
                    # Scale loss by accumulation steps to maintain same effective loss
                    total_loss = total_loss / gradient_accumulation_steps
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Check if we should step the optimizer (every accumulation_steps or at end of epoch)
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                is_last_batch = batch_idx == len(pbar) - 1
                
                if is_accumulation_step or is_last_batch:
                    # Increment global step only when we actually step the optimizer
                    global_step += 1
                    
                    # Gradient clipping and optimizer step
                    if self.scaler:
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
                        if self.config['training'].get('gradient_clip_norm'):
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config['training']['gradient_clip_norm']
                            )
                        else:
                            grad_norm = None
                        self.optimizer.step()
                    
                    # Zero gradients after optimizer step
                    self.optimizer.zero_grad()
                    
                    # Mark that optimizer has stepped at least once
                    self.optimizer_stepped = True
                    
                    # Scheduler per optimization step (not per batch) for warmup schedulers
                    if self.scheduler and self.config['training']['scheduler'] == 'warmup_cosine' and self.optimizer_stepped:
                        self.scheduler.step()
                
                # Update running losses
                running_loss += total_loss.item()
                running_recon_loss += recon_loss.item()
                running_kl_loss += kl_loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Current batch losses
                batch_total_loss = total_loss.item()
                batch_recon_loss = recon_loss.item()
                batch_kl_loss = kl_loss.item()
                
                # Log training metrics with EMA and VAE loss components
                self.metrics_tracker.log_training_step(
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    reconstruction_loss=recon_loss.item(),
                    kl_loss=kl_loss.item(),
                    beta=beta
                )
 
                # Update progress bar with EMA loss and VAE components
                ema_loss = self.metrics_tracker.get_current_training_ema()
                pbar.set_postfix({
                    'Total': f'{batch_total_loss:.4f}',
                    'Recon': f'{batch_recon_loss:.4f}',
                    'KL': f'{batch_kl_loss:.4f}',
                    'β': f'{beta:.3f}',
                    'EMA': f'{ema_loss:.4f}' if ema_loss else 'N/A',
                })
                
                # Run validation at specified frequency
                if (batch_idx + 1) % validation_interval == 0 or batch_idx == total_batches - 1:
                    self.run_validation(epoch, batch_idx + 1)

            # Scheduler step per epoch for other schedulers
            if self.scheduler and self.config['training']['scheduler'] != 'warmup_cosine':
                self.scheduler.step()
        
            # Save checkpoint at end of epoch
            if epoch % self.config['logging']['save_checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, global_step)
        
        self.logger.info("HyperVAE training completed!")
        
        # Generate validation comparison video
        try:
            self.logger.info("Generating validation comparison video...")
            video_path = create_validation_video(
                model=self.model,
                validation_loader=self.test_loader,
                device=self.device,
                experiment_dir=self.experiment_dir,
                video_name="hypervae_validation_comparison.mp4"
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
        best_model_name = "best_hypervae_model.pth"
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
        self.logger.info(f"Best HyperVAE model saved with validation loss: {val_loss:.6f} at epoch {epoch}, batch {batch_idx}")
        
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
    parser = argparse.ArgumentParser(description="Train HyperVAE on 3D volume sequences.")
    
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
            print(f"Default HyperVAE config file generated at: {args.generate_config}")
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
    trainer = HyperVAETrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
