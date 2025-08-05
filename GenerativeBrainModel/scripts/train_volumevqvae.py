#!/usr/bin/env python3
"""
Main training script for VolumeVQVAE autoencoder.
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
from GenerativeBrainModel.models.volumevqvae import VolumeVQVAE
from GenerativeBrainModel.dataloaders.volume_dataloader import create_dataloaders, get_volume_info
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_validation_video


# Configuration utilities embedded in this script
def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary for VQ-VAE training.
    
    Returns:
        Dictionary with all configuration parameters
    """
    return {
        'experiment': {
            'name': 'VolumeVQVAE_training',
            'description': 'VolumeVQVAE autoencoder training on 3D volumetric spike data',
            'tags': ['vq-vae', 'autoencoder', 'quantization', '3d-volumes', 'spike-data']
        },
        
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [
                'subject_1',
                'subject_4', 
                'subject_5'
            ],
            'volume_size': None,  # MUST be provided if auto-detection fails
            'volumes_per_batch': 8,
            'max_timepoints_per_subject': None, # -1 for all
            'use_cache': False,
            'cache_data': False  # Test performance with caching disabled
        },
        
        'model': {
            'hidden_channels': 256,
            'volume_size': [256, 128, 30],
            'region_size': [32, 16, 2],
            'num_frequencies': 32,
            'sigma': 1.0,
            'n_heads': 8,
            'codebook_size': 512
        },
        
        'training': {
            'volumes_per_batch': 8,  # Number of full volumes per batch (effective batch size will be volumes_per_batch * n_regions)
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',  # Optimizer type: 'adamw' or 'muon'
            # Muon-specific settings (used when optimizer='muon')
            'muon_lr': 0.02,  # Learning rate for Muon (hidden weights)
            'muon_momentum': 0.95,  # Momentum for Muon
            'muon_nesterov': True,  # Use Nesterov momentum for Muon
            'muon_ns_steps': 5,  # Number of steps for Muon
            # AdamW settings for non-hidden params when using Muon
            'adamw_lr': 3e-4,  # Learning rate for AdamW params (when using Muon)
            'adamw_betas': [0.9, 0.95],  # Beta values for AdamW
            'adamw_eps': 1e-8,           # Epsilon for AdamW
            'scheduler': 'linear_warmup',  # Available: 'linear_warmup', 'warmup_cosine', 'cosine', 'step', or None
            'min_lr_ratio': 0.01,  # Minimum LR as ratio of initial LR (for warmup_cosine)
            'gradient_clip_norm': 1.0,
            'validation_frequency': 8,  # Number of times to run validation per epoch
            
            # Sequence parameters for autoencoder (single timepoint reconstruction)
            'sequence_length': 1,  # Single timepoint for autoencoder
            'stride': 1,  # Use every timepoint (step size between sequences)
            'max_timepoints_per_subject': None,  # Max timepoints per subject file (None = use all available)
            
            # Hardware settings
            'use_gpu': True,
            'num_workers': 4,  # DataLoader worker processes
            'pin_memory': True,  # Faster GPU transfer (keeps data in pinned CPU memory)
            'mixed_precision': True,
            'compile_model': False,  # PyTorch 2.0 compile
            
            # Random seed
            'seed': 42
        },
        
        'loss': {
            'loss_function': 'vq_vae',  # Uses custom VQ-VAE loss (reconstruction + commitment)
            'beta': 0.25,  # Commitment loss weight
            'loss_weights': {
                'reconstruction': 1.0,
                'commitment': 0.25
            }
        },
        
        'logging': {
            'log_level': 'INFO',
            'log_frequency': 10,  # Log every N batches
            'save_checkpoint_frequency': 10,  # Save checkpoint every N epochs
            'keep_n_checkpoints': 5
        },
        
        'paths': {
            'output_base_dir': 'experiments/volumevqvae',
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
    
    # Calculate and log input size for reference
    input_size = region_size[0] * region_size[1] * region_size[2]
    print(f"Configuration validation passed!")
    print(f"Calculated input_size: {input_size} (from region_size {region_size})")


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


class VQVAETrainer:
    """
    Comprehensive VQ-VAE trainer with all bells and whistles.
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
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.early_stopping_counter = 0
        
        # Setup experiment directory
        self.experiment_dir = Path(config['paths']['experiment_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        
        # Initialize VQ-VAE specific metrics tracker
        self.metrics_tracker = CombinedMetricsTracker(
            log_dir=config['paths']['log_dir'],
            validation_threshold=0.5,
            ema_alpha=0.05,  # EMA smoothing factor for training loss
            enable_plotting=True,
            track_loss_components=True,
            loss_component_type='vqvae'
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
        """Build and initialize the model."""
        model_config = self.config['model']
        
        # Use VolumeVQVAE
        self.model = VolumeVQVAE(
            hidden_channels=model_config['hidden_channels'],
            volume_size=model_config['volume_size'],
            region_size=model_config['region_size'],
            num_frequencies=model_config['num_frequencies'],
            sigma=model_config['sigma'],
            n_heads=model_config['n_heads'],
            codebook_size=model_config['codebook_size']
        )
        
        self.logger.info(f"Model volume size: {model_config['volume_size']}, region size: {model_config['region_size']}")
        self.logger.info(f"Codebook size: {model_config['codebook_size']}")
        
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
            f.write("MODEL ARCHITECTURE SUMMARY\n")
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
                f.write(f"  {name:<30} {param_count:>10,} [{trainable}] {tuple(param.shape)}\n")
            f.write("\n")
            
            # Buffer information (non-trainable tensors)
            f.write("REGISTERED BUFFERS:\n")
            f.write("-" * 40 + "\n")
            for name, buffer in self.model.named_buffers():
                buffer_count = buffer.numel() if buffer is not None else 0
                shape = tuple(buffer.shape) if buffer is not None else "None"
                f.write(f"  {name:<30} {buffer_count:>10,} {shape}\n")
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
            
            # VQ-VAE specific configuration
            f.write("VQ-VAE CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            loss_config = self.config['loss']
            for key, value in loss_config.items():
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
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Get optimizer type from config
        optimizer_type = training_config.get('optimizer', 'adamw').lower()
        
        if optimizer_type == 'adamw':
            # Use pure AdamW optimizer
            learning_rate = training_config.get('learning_rate', 0.001)
            weight_decay = training_config.get('weight_decay', 1e-5)
            
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            self.logger.info(f"Using AdamW optimizer:")
            self.logger.info(f"  Learning rate: {learning_rate}")
            self.logger.info(f"  Weight decay: {weight_decay}")
            self.logger.info(f"  Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
            
            # Scheduler setup for AdamW
            scheduler_type = training_config.get('scheduler', 'linear_warmup')
            base_lr = learning_rate
            
        elif optimizer_type == 'muon':
            # Use Muon optimizer with hybrid approach
            try:
                from muon import MuonWithAuxAdam
            except ImportError:
                raise ImportError("Muon optimizer not found. Please install with: pip install git+https://github.com/KellerJordan/Muon")
            
            # Separate parameters according to Muon guidelines for VQ-VAE architecture
            hidden_weights = []  # Conv filters/weight matrices - use Muon
            adamw_params = []    # Biases, norms, attention, codebook, final layers - use AdamW
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Categorize parameters based on Muon recommendations and VQ-VAE structure
                if param.ndim >= 2:
                    # Conv filters and weight matrices - use Muon
                    hidden_weights.append(param)
                else:
                    # Biases, norms, attention parameters, codebook - use AdamW
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
            
            # Hidden weights use Muon
            if hidden_weights:
                param_groups.append({
                    'params': hidden_weights,
                    'use_muon': True,
                    'lr': muon_lr,
                    'momentum': muon_momentum,
                    'weight_decay': training_config['weight_decay']
                })
            
            # All other parameters use AdamW
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
            
            self.logger.info(f"Using Muon optimizer:")
            self.logger.info(f"  Hidden weights ({len(hidden_weights)} params): Muon (LR: {muon_lr})")
            self.logger.info(f"  Other params ({len(adamw_params)} params): AdamW (LR: {adamw_lr})")
            
            self.logger.info(f"Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
            
            # Scheduler - support different schedulers for Muon
            scheduler_type = training_config.get('scheduler', 'linear_warmup')
            base_muon_lr = muon_lr
            base_adamw_lr = adamw_lr
            
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: 'adamw', 'muon'")
        
        # Scheduler
        if scheduler_type == 'linear_warmup':
            # Calculate total batches for warmup (10% of first epoch)
            warmup_batches = int(0.1 * len(self.train_loader))
            
            if optimizer_type == 'muon':
                # Different lambda functions for different parameter groups
                def muon_lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    return 1.0
                
                def adamw_lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    return 1.0
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, [muon_lr_lambda, adamw_lr_lambda])
            else:
                def lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    return 1.0
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
            self.logger.info(f"Using Linear Warmup scheduler with {warmup_batches} warmup batches.")
        
        elif scheduler_type == 'warmup_cosine':
            # Linear warmup followed by cosine annealing
            warmup_batches = int(0.1 * len(self.train_loader))  # 10% of first epoch for warmup
            total_batches = training_config['num_epochs'] * len(self.train_loader)
            cosine_batches = total_batches - warmup_batches  # Remaining batches for cosine annealing
            
            # Get minimum learning rate (default to 1% of initial LR)
            min_lr_ratio = training_config.get('min_lr_ratio', 0.01)
            
            if optimizer_type == 'muon':
                # Different lambda functions for different parameter groups
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
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, [muon_lr_lambda, adamw_lr_lambda])
                min_lrs = f"Muon: {min_muon_lr:.2e}, AdamW: {min_adamw_lr:.2e}"
            else:
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
                min_lrs = f"{min_lr:.2e}"
            
            self.logger.info(f"Using Warmup + Cosine Annealing scheduler:")
            self.logger.info(f"  Warmup batches: {warmup_batches}")
            self.logger.info(f"  Total batches: {total_batches}")
            self.logger.info(f"  Min LR ratio: {min_lr_ratio} (min_lrs: {min_lrs})")
        
        elif scheduler_type == 'cosine':
            if optimizer_type == 'muon':
                # For Muon, we use the base learning rates from each group
                eta_min_muon = base_muon_lr * 0.01
                eta_min_adamw = base_adamw_lr * 0.01
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['num_epochs'],
                    eta_min=[eta_min_muon, eta_min_adamw]
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=training_config['num_epochs'],
                    eta_min=base_lr * 0.01
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

    def run_validation(self, epoch: int, batch_idx: int):
        """
        Run validation on entire test dataset and log VQ-VAE specific metrics.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
        """
        self.logger.info(f"Running validation at epoch {epoch}, batch {batch_idx}")
        
        self.model.eval()
        total_val_loss = 0.0
        total_reconstruction_loss = 0.0
        total_commitment_loss = 0.0
        num_batches = 0
        
        # Collect predictions and targets in smaller chunks to avoid memory issues
        prediction_chunks = []
        target_chunks = []
        max_chunk_size = 25000  # Process in chunks of ~25k elements
        current_chunk_size = 0
        
        beta = self.config['loss']['beta']
        
        with torch.no_grad():
            val_pbar = tqdm(self.test_loader, desc=f"Validation E{epoch}B{batch_idx}", leave=False, ncols=100)
            
            for val_batch_data in val_pbar:
                # To device
                if isinstance(val_batch_data, (list, tuple)):
                    val_data, val_target = val_batch_data
                else:
                    val_data = val_batch_data
                    val_target = val_data  # Autoencoder target is input
                
                val_data = val_data.to(self.device, non_blocking=True)
                val_target = val_target.to(self.device, non_blocking=True)
                
                # Add sequence dimension T=1 for autoencoder: (B, X, Y, Z) -> (B, T=1, X, Y, Z)
                if len(val_data.shape) == 4:  # (B, X, Y, Z)
                    val_data = val_data.unsqueeze(1)  # (B, 1, X, Y, Z)
                    val_target = val_target.unsqueeze(1)  # (B, 1, X, Y, Z)
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    # VQ-VAE forward pass returns (reconstruction_logits, quantized_encoding, encoding)
                    reconstruction_logits, quantized_encoding, encoding = self.model(val_data)
                    
                    # Calculate separate losses
                    reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(
                        reconstruction_logits, val_target, reduction='mean'
                    )
                    commitment_loss = beta * nn.functional.mse_loss(
                        quantized_encoding.detach(), encoding, reduction='mean'
                    )
                    val_loss = reconstruction_loss + commitment_loss
                
                total_val_loss += val_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_commitment_loss += commitment_loss.item()
                num_batches += 1
                
                # Convert logits to probabilities for metrics calculation
                val_probabilities = torch.sigmoid(reconstruction_logits)
                
                # Flatten predictions and targets
                batch_preds = val_probabilities.flatten()
                batch_targets = val_target.flatten()
                
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
                    'Recon': f'{total_reconstruction_loss/num_batches:.4f}',
                    'Commit': f'{total_commitment_loss/num_batches:.4f}'
                })
        
        # Calculate average validation losses
        avg_val_loss = total_val_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        avg_commitment_loss = total_commitment_loss / num_batches
        
        # Concatenate chunks and compute metrics
        all_predictions = torch.cat([torch.cat(chunk) for chunk in prediction_chunks])
        all_targets = torch.cat([torch.cat(chunk) for chunk in target_chunks])
        
        # Use VQ-VAE specific metrics tracker
        metrics = self.metrics_tracker.log_validation_step(
            epoch=epoch,
            batch_idx=batch_idx,
            predictions=all_predictions,
            targets=all_targets,
            validation_loss=avg_val_loss
        )
        
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
        self.logger.info("Starting VQ-VAE training...")
        
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
        beta = self.config['loss']['beta']
        
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
                    data, target = batch_data
                else:
                    data = batch_data
                    target = data # Autoencoder target is input
                
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Add sequence dimension T=1 for autoencoder: (B, X, Y, Z) -> (B, T=1, X, Y, Z)
                if len(data.shape) == 4:  # (B, X, Y, Z)
                    data = data.unsqueeze(1)  # (B, 1, X, Y, Z)
                    target = target.unsqueeze(1)  # (B, 1, X, Y, Z)
                
                # Forward/backward pass
                self.optimizer.zero_grad()
                
                with autocast(enabled=self.scaler is not None):
                    # VQ-VAE forward pass returns (reconstruction_logits, quantized_encoding, encoding)
                    reconstruction_logits, quantized_encoding, encoding = self.model(data)
                    
                    # Calculate separate losses for logging
                    reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(
                        reconstruction_logits, target, reduction='mean'
                    )
                    commitment_loss = beta * nn.functional.mse_loss(
                        quantized_encoding.detach(), encoding, reduction='mean'
                    )
                    total_loss = reconstruction_loss + commitment_loss
                
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
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
                    total_loss.backward()
                    if self.config['training'].get('gradient_clip_norm'):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config['training']['gradient_clip_norm']
                        )
                    else:
                        grad_norm = None
                    self.optimizer.step()

                # Scheduler per batch for linear warmup and warmup_cosine
                if self.scheduler and self.config['training']['scheduler'] in ['linear_warmup', 'warmup_cosine']:
                    self.scheduler.step()
                
                # Update running loss
                running_loss += total_loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log training metrics with VQ-VAE specific losses
                self.metrics_tracker.log_training_step(
                    epoch=epoch,
                    batch_idx=batch_idx + 1,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    reconstruction_loss=reconstruction_loss.item(),
                    commitment_loss=commitment_loss.item(),
                    beta=beta
                )
 
                # Update progress bar with EMA losses
                ema_losses = self.metrics_tracker.get_current_component_emas()
                ema_total = self.metrics_tracker.get_current_training_ema()
                pbar.set_postfix({
                    'Total': f'{total_loss.item():.6f}',
                    'EMA': f'{ema_total:.6f}' if ema_total else 'N/A',
                    'Recon': f'{ema_losses.get("reconstruction_loss_ema", 0):.6f}' if ema_losses.get("reconstruction_loss_ema") else 'N/A',
                    'Commit': f'{ema_losses.get("commitment_loss_ema", 0):.6f}' if ema_losses.get("commitment_loss_ema") else 'N/A',
                })
                
                # Run validation at specified frequency
                if (batch_idx + 1) % validation_interval == 0 or batch_idx == total_batches - 1:
                    self.run_validation(epoch, batch_idx + 1)

            # Scheduler step per epoch for other schedulers
            if self.scheduler and self.config['training']['scheduler'] not in ['linear_warmup', 'warmup_cosine']:
                self.scheduler.step()
        
            # Save checkpoint at end of epoch
            if epoch % self.config['logging']['save_checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, global_step)
        
        self.logger.info("VQ-VAE training completed!")
        
        # Generate VQ-VAE specific plots
        try:
            self.logger.info("Generating VQ-VAE training plots...")
            plot_path = self.metrics_tracker.generate_vqvae_plots()
            self.logger.info(f"VQ-VAE training plots saved to: {plot_path}")
        except Exception as e:
            self.logger.error(f"Failed to generate VQ-VAE plots: {e}")
            self.logger.error("Continuing without VQ-VAE plot generation...")
        
        # Generate validation comparison video
        try:
            self.logger.info("Generating validation comparison video...")
            video_path = create_validation_video(
                model=self.model,
                validation_loader=self.test_loader,
                device=self.device,
                experiment_dir=self.experiment_dir,
                video_name="validation_comparison.mp4"
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
        best_model_name = "best_model.pth"
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
        self.logger.info(f"Best model saved with validation loss: {val_loss:.6f} at epoch {epoch}, batch {batch_idx}")
        
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
    parser = argparse.ArgumentParser(description="Train a VQ-VAE autoencoder on 3D volume data.")
    
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
            print(f"Default VQ-VAE config file generated at: {args.generate_config}")
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
    trainer = VQVAETrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
