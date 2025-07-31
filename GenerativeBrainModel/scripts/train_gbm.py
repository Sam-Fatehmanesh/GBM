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
            'gbm_checkpoint_path': None,  # Path to complete GBM checkpoint for continued training
            'reset_training_state': False,  # If True, reset epoch/step counters when loading GBM checkpoint
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
            # Muon-specific hyperparameters
            'muon_betas': [0.9, 0.95],    # Beta values for Muon optimizer
            'muon_eps': 1e-8,             # Epsilon for Muon optimizer
            'scheduler': 'warmup_cosine',  # Linear warmup + cosine annealing
            # Available schedulers: 'linear_warmup', 'warmup_cosine', 'warmup_lineardecay', 'cosine', 'step', or None
            'min_lr_ratio': 0.01,  # Minimum LR as ratio of initial LR (for warmup_cosine and warmup_lineardecay)
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': None,  # Number of batches to accumulate gradients over (None/0 = disabled)
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
            },
            # Two-Step Rollout Loss configuration
            'use_two_step_rollout': False,  # Enable/disable two-step rollout loss
            'two_step_alpha': 0.5,  # Weight for L2 loss component (α in L = L1 + α*L2)
            'two_step_curriculum': True,  # Enable curriculum learning for α
            'two_step_alpha_start': 0.1,  # Starting α value for curriculum
            'two_step_alpha_end': 1.0,  # Final α value for curriculum
            'two_step_ramp_fraction': 0.5  # Fraction of total training steps to ramp α (0.5 = first 50% of training)
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
    
    # Check GBM checkpoint path if provided
    gbm_checkpoint_path = model_config.get('gbm_checkpoint_path')
    if gbm_checkpoint_path and not Path(gbm_checkpoint_path).exists():
        raise ValueError(f"GBM checkpoint not found: {gbm_checkpoint_path}")
    
    # Validate sequence length
    seq_len = config['training']['sequence_length']
    use_two_step_rollout = config['loss'].get('use_two_step_rollout', False)
    
    if use_two_step_rollout:
        if seq_len < 2:
            raise ValueError(f"Sequence length must be at least 2 for two-step rollout training, got {seq_len}")
        print(f"Two-Step Rollout Loss enabled - will request {seq_len + 1} timesteps from dataloader")
    else:
        if seq_len < 2:
            raise ValueError(f"Sequence length must be at least 2 for seq2seq training, got {seq_len}")
    
    # Validate two-step rollout loss configuration
    if use_two_step_rollout:
        loss_config = config['loss']
        alpha = loss_config.get('two_step_alpha', 0.5)
        if not (0.0 <= alpha <= 10.0):  # Allow alpha > 1 for emphasis on L2
            raise ValueError(f"two_step_alpha must be between 0.0 and 10.0, got {alpha}")
        
        if loss_config.get('two_step_curriculum', False):
            alpha_start = loss_config.get('two_step_alpha_start', 0.1)
            alpha_end = loss_config.get('two_step_alpha_end', 1.0)
            ramp_fraction = loss_config.get('two_step_ramp_fraction', 0.5)
            
            if not (0.0 <= alpha_start <= 10.0):
                raise ValueError(f"two_step_alpha_start must be between 0.0 and 10.0, got {alpha_start}")
            if not (0.0 <= alpha_end <= 10.0):
                raise ValueError(f"two_step_alpha_end must be between 0.0 and 10.0, got {alpha_end}")
            if not (0.0 < ramp_fraction <= 1.0):
                raise ValueError(f"two_step_ramp_fraction must be between 0.0 and 1.0, got {ramp_fraction}")
            
            print(f"Two-step curriculum: α ramps from {alpha_start} to {alpha_end} over {ramp_fraction*100:.0f}% of training")
        else:
            print(f"Two-step rollout: fixed α = {alpha}")
    
    print(f"Configuration validation passed!")
    print(f"GBM Model: d_model={model_config['d_model']}, n_heads={model_config['n_heads']}, n_layers={model_config['n_layers']}")
    print(f"Sequence length: {seq_len}, Ensemble size: {model_config['ensemble_size']}")
    
    # Show checkpoint/autoencoder loading info
    gbm_checkpoint_path = model_config.get('gbm_checkpoint_path')
    autoencoder_path = model_config.get('autoencoder_path')
    reset_training_state = model_config.get('reset_training_state', False)
    
    if gbm_checkpoint_path:
        reset_msg = " (with training state reset)" if reset_training_state else " (continuing from checkpoint)"
        print(f"Will load complete GBM checkpoint: {gbm_checkpoint_path}{reset_msg}")
    elif autoencoder_path:
        print(f"Will load pretrained autoencoder: {autoencoder_path}")
    else:
        print("Will use randomly initialized model")


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
        self.loaded_from_checkpoint = False
        self.checkpoint_epoch = 0
        self.checkpoint_step = 0
        self.checkpoint_data = None  # Store checkpoint data for optimizer/scheduler loading
        
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
        gbm_checkpoint_path = model_config.get('gbm_checkpoint_path')
        
        if gbm_checkpoint_path and autoencoder_path:
            self.logger.info(f"Both GBM checkpoint and autoencoder path provided - GBM checkpoint takes precedence")
        elif autoencoder_path and not gbm_checkpoint_path:
            self.logger.info(f"Loaded pretrained autoencoder from: {autoencoder_path}")
        elif not autoencoder_path and not gbm_checkpoint_path:
            self.logger.warning("No autoencoder path or GBM checkpoint provided - using randomly initialized autoencoder")
        
        self.logger.info(f"Model volume size: {model_config['volume_size']}, region size: {model_config['region_size']}")
        
        self.model.to(self.device)
        
        # Load complete GBM checkpoint if provided (takes precedence over autoencoder_path)
        gbm_checkpoint_path = model_config.get('gbm_checkpoint_path')
        if gbm_checkpoint_path:
            self.load_gbm_checkpoint(gbm_checkpoint_path)
        
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
    
    def load_gbm_checkpoint(self, checkpoint_path: str):
        """
        Load a complete GBM checkpoint for continued training.
        
        Args:
            checkpoint_path: Path to the GBM checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        self.logger.info(f"Loading GBM checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle torch.compile() prefixes (_orig_mod.) in state dict keys
            model_state_dict = checkpoint['model_state_dict']
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                # Remove _orig_mod. prefix if present (from torch.compile)
                if key.startswith('_orig_mod.'):
                    cleaned_key = key[len('_orig_mod.'):]
                else:
                    cleaned_key = key
                cleaned_state_dict[cleaned_key] = value
            
            # Load model state
            self.model.load_state_dict(cleaned_state_dict)
            
            # Load training state (unless reset is requested)
            reset_training_state = self.config['model'].get('reset_training_state', False)
            if reset_training_state:
                self.checkpoint_epoch = 0
                self.checkpoint_step = 0
                self.best_val_loss = float('inf')
                self.logger.info("Training state reset - will start training from epoch 1")
            else:
                self.checkpoint_epoch = checkpoint.get('epoch', 0)
                self.checkpoint_step = checkpoint.get('step', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                
                # If this was the best model, update best_model_path
                if 'validation_loss' in checkpoint:
                    self.best_val_loss = checkpoint['validation_loss']
            
            self.loaded_from_checkpoint = True
            self.checkpoint_data = checkpoint  # Store for optimizer/scheduler loading
            
            self.logger.info(f"Successfully loaded GBM checkpoint:")
            self.logger.info(f"  Checkpoint epoch: {self.checkpoint_epoch}")
            self.logger.info(f"  Checkpoint step: {self.checkpoint_step}")
            self.logger.info(f"  Best validation loss: {self.best_val_loss}")
            
            # Check if config matches
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                current_model_config = self.config['model']
                checkpoint_model_config = checkpoint_config.get('model', {})
                
                # Compare key model parameters
                key_params = ['d_model', 'n_heads', 'n_layers', 'volume_size', 'region_size']
                for param in key_params:
                    current_val = current_model_config.get(param)
                    checkpoint_val = checkpoint_model_config.get(param)
                    if current_val != checkpoint_val:
                        self.logger.warning(f"Config mismatch for {param}: current={current_val}, checkpoint={checkpoint_val}")
            
        except Exception as e:
            self.logger.error(f"Failed to load GBM checkpoint: {e}")
            raise ValueError(f"Could not load GBM checkpoint from {checkpoint_path}: {e}")
    
    def load_optimizer_states(self):
        """
        Load optimizer and scheduler states from the stored checkpoint data.
        This should be called after the optimizer and scheduler are built.
        """
        if not self.checkpoint_data:
            self.logger.warning("No checkpoint data available for loading optimizer states")
            return
        
        # Check if training state should be reset
        reset_training_state = self.config['model'].get('reset_training_state', False)
        if reset_training_state:
            self.logger.info("Training state reset requested - using fresh optimizer/scheduler/scaler states")
            return
        
        try:
            # Load optimizer state
            if 'optimizer_state_dict' in self.checkpoint_data:
                self.optimizer.load_state_dict(self.checkpoint_data['optimizer_state_dict'])
                self.logger.info("Loaded optimizer state from checkpoint")
            else:
                self.logger.info("No optimizer state found in checkpoint - using fresh optimizer")
            
            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in self.checkpoint_data:
                self.scheduler.load_state_dict(self.checkpoint_data['scheduler_state_dict'])
                self.logger.info("Loaded scheduler state from checkpoint")
            elif self.scheduler:
                self.logger.info("No scheduler state found in checkpoint - using fresh scheduler")
            
            # Load scaler state
            if self.scaler and 'scaler_state_dict' in self.checkpoint_data:
                self.scaler.load_state_dict(self.checkpoint_data['scaler_state_dict'])
                self.logger.info("Loaded mixed precision scaler state from checkpoint")
            elif self.scaler:
                self.logger.info("No scaler state found in checkpoint - using fresh scaler")
                
        except Exception as e:
            self.logger.warning(f"Failed to load optimizer/scheduler states from checkpoint: {e}")
            self.logger.warning("Continuing with fresh optimizer/scheduler states")
    
    def build_optimizer(self):
        """Build optimizer and scheduler."""
        training_config = self.config['training']
        
        # Optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer_type = training_config.get('optimizer', 'adamw')
        
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
            self.logger.info(f"Using AdamW optimizer with learning rate {training_config['learning_rate']}")
        elif optimizer_type == 'muon':
            # Import Muon optimizer
            try:
                from muon import MuonWithAuxAdam
            except ImportError:
                raise ImportError("Muon optimizer not found. Please install with: pip install git+https://github.com/KellerJordan/Muon")
            
            # Separate parameters according to Muon guidelines
            hidden_weights = []
            hidden_gains_biases = []
            nonhidden_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Parameters from autoencoder should use AdamW (non-hidden)
                if 'autoencoder' in name:
                    nonhidden_params.append(param)
                # Hidden weights: parameters with ndim >= 2 (weight matrices, conv filters)
                elif param.ndim >= 2:
                    hidden_weights.append(param)
                # Hidden gains/biases: parameters with ndim < 2 (biases, layer norms, etc.)
                else:
                    hidden_gains_biases.append(param)
            
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
                # Muon group: params, use_muon, lr, momentum, weight_decay
                param_groups.append({
                    'params': hidden_weights,
                    'use_muon': True,
                    'lr': muon_lr,
                    'momentum': muon_momentum,
                    'weight_decay': training_config['weight_decay']
                })
            
            # Hidden gains/biases and non-hidden params use AdamW
            if hidden_gains_biases or nonhidden_params:
                # AdamW group in MuonWithAuxAdam: params, use_muon, lr, betas, eps, weight_decay
                param_groups.append({
                    'params': hidden_gains_biases + nonhidden_params,
                    'use_muon': False,
                    'lr': adamw_lr,
                    'betas': training_config.get('adamw_betas'),
                    'eps': training_config.get('adamw_eps'),
                    'weight_decay': training_config['weight_decay']
                })
            
            self.optimizer = MuonWithAuxAdam(param_groups)
            
            self.logger.info(f"Using Muon optimizer:")
            self.logger.info(f"  Hidden weights ({len(hidden_weights)} params): Muon (LR: {muon_lr})")
            self.logger.info(f"  Other params ({len(hidden_gains_biases) + len(nonhidden_params)} params): AdamW (LR: {adamw_lr})")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        self.logger.info(f"Optimizer parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        
        # For Muon optimizer, we need to handle multiple parameter groups with different base LRs
        if optimizer_type == 'muon':
            base_muon_lr = training_config.get('muon_lr', 0.02)
            base_adamw_lr = training_config.get('adamw_lr', 3e-4)
        else:
            base_lr = training_config['learning_rate']
        
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
        
        elif scheduler_type == 'warmup_lineardecay':
            # Linear warmup followed by linear decay
            warmup_batches = int(0.1 * len(self.train_loader))  # 10% of first epoch for warmup
            total_batches = training_config['num_epochs'] * len(self.train_loader)
            decay_batches = total_batches - warmup_batches  # Remaining batches for linear decay
            
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
                        decay_batch = batch - warmup_batches
                        decay_progress = decay_batch / decay_batches
                        decay_progress = min(decay_progress, 1.0)
                        
                        lr = base_muon_lr - (base_muon_lr - min_muon_lr) * decay_progress
                        return lr / base_muon_lr
                
                def adamw_lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    else:
                        decay_batch = batch - warmup_batches
                        decay_progress = decay_batch / decay_batches
                        decay_progress = min(decay_progress, 1.0)
                        
                        lr = base_adamw_lr - (base_adamw_lr - min_adamw_lr) * decay_progress
                        return lr / base_adamw_lr
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, [muon_lr_lambda, adamw_lr_lambda])
                min_lrs = f"Muon: {min_muon_lr:.2e}, AdamW: {min_adamw_lr:.2e}"
            else:
                min_lr = base_lr * min_lr_ratio
                
                def lr_lambda(batch):
                    if batch < warmup_batches:
                        return float(batch) / float(max(1, warmup_batches))
                    else:
                        decay_batch = batch - warmup_batches
                        decay_progress = decay_batch / decay_batches
                        decay_progress = min(decay_progress, 1.0)
                        
                        lr = base_lr - (base_lr - min_lr) * decay_progress
                        return lr / base_lr
                
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
                min_lrs = f"{min_lr:.2e}"
            
            self.logger.info(f"Using Warmup + Linear Decay scheduler:")
            self.logger.info(f"  Warmup batches: {warmup_batches}")
            self.logger.info(f"  Total batches: {total_batches}")
            self.logger.info(f"  Min LR ratio: {min_lr_ratio} (min_lrs: {min_lrs})")
        
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
        
        # Load optimizer and scheduler states from checkpoint if available
        if self.loaded_from_checkpoint:
            self.load_optimizer_states()
        
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

        # Check if Two-Step Rollout Loss is enabled and adjust sequence length accordingly
        use_two_step_rollout = self.config['loss'].get('use_two_step_rollout', False)
        original_sequence_length = self.config['training']['sequence_length']
        
        if use_two_step_rollout:
            # Request one extra timestep for two-step rollout loss
            self.config['training']['sequence_length'] = original_sequence_length + 1
            self.logger.info(f"Two-Step Rollout Loss enabled: requesting {original_sequence_length + 1} timesteps (original: {original_sequence_length})")
        
        self.train_loader, self.test_loader = create_dataloaders(self.config)
        
        # Restore original sequence length for model processing
        if use_two_step_rollout:
            self.config['training']['sequence_length'] = original_sequence_length
            
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
    
    def get_two_step_alpha(self, global_step: int, total_steps: int) -> float:
        """
        Calculate the alpha weight for two-step rollout loss with curriculum learning.
        
        Args:
            global_step: Current global training step (0-indexed)
            total_steps: Total training steps across all epochs
            
        Returns:
            Alpha weight for L2 loss component
        """
        loss_config = self.config['loss']
        
        if not loss_config.get('two_step_curriculum', False):
            # Fixed alpha value
            return loss_config.get('two_step_alpha', 0.5)
        
        # Curriculum learning: ramp alpha from start to end value during training
        alpha_start = loss_config.get('two_step_alpha_start', 0.1)
        alpha_end = loss_config.get('two_step_alpha_end', 1.0)
        
        # Get the fraction of training steps to use for ramping (default: 50%)
        ramp_fraction = loss_config.get('two_step_ramp_fraction', 0.5)
        ramp_steps = int(total_steps * ramp_fraction)
        
        if global_step < ramp_steps:
            # Linear interpolation from start to end during ramp period
            progress = global_step / max(1, ramp_steps - 1)
            alpha = alpha_start + progress * (alpha_end - alpha_start)
        else:
            # Use final alpha value after ramp period
            alpha = alpha_end
            
        return alpha
    
    def prepare_seq2seq_data(self, sequences, two_step_rollout=False):
        """
        Prepare input and target sequences for seq2seq training.
        
        Args:
            sequences: Tensor of shape (B, T, X, Y, Z)
            two_step_rollout: If True, prepare data for two-step rollout loss
            
        Returns:
            If two_step_rollout=False:
                Tuple of (input_sequences, target_sequences)
                - input_sequences: (B, T-1, X, Y, Z) - sequences[0:T-1]
                - target_sequences: (B, T-1, X, Y, Z) - sequences[1:T]
            If two_step_rollout=True:
                Tuple of (input_sequences, target_sequences, second_step_target)
                - input_sequences: (B, T-2, X, Y, Z) - sequences[0:T-2]
                - target_sequences: (B, T-2, X, Y, Z) - sequences[1:T-1]  
                - second_step_target: (B, T-2, X, Y, Z) - sequences[2:T]
        """
        B, T, X, Y, Z = sequences.shape
        
        if not two_step_rollout:
            # Standard seq2seq preparation
            if T < 2:
                raise ValueError(f"Sequence length must be at least 2 for seq2seq training, got {T}")
            
            # Input: all timesteps except the last
            input_seq = sequences[:, :-1, :, :, :]  # (B, T-1, X, Y, Z)
            
            # Target: all timesteps except the first  
            target_seq = sequences[:, 1:, :, :, :]   # (B, T-1, X, Y, Z)
            
            return input_seq, target_seq
        else:
            # Two-step rollout preparation  
            if T < 3:
                raise ValueError(f"Sequence length must be at least 3 for two-step rollout training, got {T}")
            
            # Input: all timesteps except the last two
            input_seq = sequences[:, :-2, :, :, :]  # (B, T-2, X, Y, Z)
            
            # Target for first step: timesteps 1 to T-2  
            target_seq = sequences[:, 1:-1, :, :, :]   # (B, T-2, X, Y, Z)
            
            # Target for second step: timesteps 2 to T-1
            second_step_target = sequences[:, 2:, :, :, :]   # (B, T-2, X, Y, Z)
            
            return input_seq, target_seq, second_step_target

    def run_validation(self, epoch: int, batch_idx: int, global_step: int, total_steps: int):
        """
        Run validation on entire test dataset and log metrics using the metrics tracker.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            global_step: Current global training step
            total_steps: Total training steps across all epochs
        """
        self.logger.info(f"Running validation at epoch {epoch}, batch {batch_idx}")
        
        self.model.eval()
        total_val_loss = 0.0
        num_batches = 0
        
        loss_fn = self.get_loss_function()
        
        # Initialize PR AUC binned accumulators to avoid storing all predictions
        threshold = self.metrics_tracker.validation_tracker.threshold
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
                
                # Check if two-step rollout loss is enabled
                use_two_step_rollout = self.config['loss'].get('use_two_step_rollout', False)
                
                # Prepare seq2seq data
                if use_two_step_rollout:
                    val_input, val_target, val_second_step_target = self.prepare_seq2seq_data(val_sequences, two_step_rollout=True)
                else:
                    val_input, val_target = self.prepare_seq2seq_data(val_sequences, two_step_rollout=False)
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    # Get predicted next volumes
                    val_output = self.model(val_input, get_logits=True)
                    val_l1_loss = loss_fn(val_output, val_target)
                    
                    if use_two_step_rollout:
                        # Two-step rollout validation loss
                        # Get current alpha weight
                        alpha = self.get_two_step_alpha(global_step, total_steps)
                        
                        # Detach the first step predictions
                        val_first_step_pred = torch.sigmoid(val_output.detach())
                        
                        # Create second step input
                        B, T_minus_2, X, Y, Z = val_input.shape
                        val_second_step_input = torch.cat([
                            val_input[:, 1:, :, :, :],  # (B, T-3, X, Y, Z)
                            val_first_step_pred[:, -1:, :, :, :]  # (B, 1, X, Y, Z)
                        ], dim=1)  # (B, T-2, X, Y, Z)
                        
                        # Forward pass for second step
                        val_second_step_output = self.model(val_second_step_input, get_logits=True)
                        val_l2_loss = loss_fn(val_second_step_output, val_second_step_target)
                        
                        # Combined validation loss
                        val_loss = val_l1_loss + alpha * val_l2_loss
                        
                        # Use first step predictions for metrics calculation
                        val_probabilities = torch.sigmoid(val_output)
                    else:
                        # Standard validation loss
                        val_loss = val_l1_loss
                        val_probabilities = torch.sigmoid(val_output)
                
                total_val_loss += val_loss.item()
                num_batches += 1
                
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
        # Log validation metrics directly to validation CSV
        self.metrics_tracker.validation_tracker.csv_logger.log_metrics({
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
        total_steps = num_epochs * total_batches
        
        # Set starting epoch and step based on checkpoint loading
        if self.loaded_from_checkpoint:
            start_epoch = self.checkpoint_epoch + 1
            global_step = self.checkpoint_step
            self.logger.info(f"Resuming training from epoch {start_epoch} (loaded from checkpoint)")
        else:
            start_epoch = 1
            global_step = 0
            
        self.current_epoch = 0

        for epoch in range(start_epoch, num_epochs + 1):
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
                
                # Check if two-step rollout loss is enabled
                use_two_step_rollout = self.config['loss'].get('use_two_step_rollout', False)
                
                # Prepare seq2seq data
                if use_two_step_rollout:
                    input_seq, target_seq, second_step_target = self.prepare_seq2seq_data(sequences, two_step_rollout=True)
                else:
                    input_seq, target_seq = self.prepare_seq2seq_data(sequences, two_step_rollout=False)
                
                # Get gradient accumulation settings
                grad_accum_steps = self.config['training'].get('gradient_accumulation_steps')
                use_grad_accum = grad_accum_steps is not None and grad_accum_steps > 1
                
                # Zero gradients at start of accumulation cycle
                if not use_grad_accum or (batch_idx % grad_accum_steps == 0):
                    self.optimizer.zero_grad()
                
                with autocast(enabled=self.scaler is not None):
                    # Predict next volumes (L1 loss)
                    output_seq = self.model(input_seq, get_logits=True)
                    loss_fn = self.get_loss_function()
                    l1_loss = loss_fn(output_seq, target_seq)
                    
                    if use_two_step_rollout:
                        # Two-step rollout loss implementation
                        # Get current alpha weight for curriculum learning
                        alpha = self.get_two_step_alpha(global_step, total_steps)
                        
                        # Detach the first step predictions to stop gradients
                        first_step_pred = torch.sigmoid(output_seq.detach())
                        
                        # Create input for second step: concatenate true input sequence with detached prediction
                        # input_seq: (B, T-2, X, Y, Z), first_step_pred: (B, T-2, X, Y, Z)
                        # We need to take the last timestep from first_step_pred and use it as the next input
                        B, T_minus_2, X, Y, Z = input_seq.shape
                        
                        # Create second step input by taking original input[1:] and appending detached prediction[-1:]
                        second_step_input = torch.cat([
                            input_seq[:, 1:, :, :, :],  # (B, T-3, X, Y, Z) - shift input by 1
                            first_step_pred[:, -1:, :, :, :]  # (B, 1, X, Y, Z) - last prediction as input
                        ], dim=1)  # (B, T-2, X, Y, Z)
                        
                        # Forward pass for second step
                        second_step_output = self.model(second_step_input, get_logits=True)
                        l2_loss = loss_fn(second_step_output, second_step_target)
                        
                        # Combined loss: L = L1 + α * L2
                        loss = l1_loss + alpha * l2_loss
                        
                        # Log alpha value periodically
                        if batch_idx % self.config['logging']['log_frequency'] == 0:
                            self.logger.debug(f"Two-step rollout: α={alpha:.3f}, L1={l1_loss.item():.6f}, L2={l2_loss.item():.6f}")
                    else:
                        # Standard single-step loss
                        loss = l1_loss
                    
                    # Scale loss by accumulation steps for proper averaging
                    if use_grad_accum:
                        loss = loss / grad_accum_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # Only step optimizer at end of accumulation cycle
                    if not use_grad_accum or ((batch_idx + 1) % grad_accum_steps == 0):
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
                    
                    # Only step optimizer at end of accumulation cycle
                    if not use_grad_accum or ((batch_idx + 1) % grad_accum_steps == 0):
                        if self.config['training'].get('gradient_clip_norm'):
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config['training']['gradient_clip_norm']
                            )
                        else:
                            grad_norm = None
                        self.optimizer.step()

                # Scheduler per batch for linear warmup (only when actually stepping)
                if (self.scheduler and self.config['training']['scheduler'] in ['linear_warmup', 'warmup_cosine', 'warmup_lineardecay'] and
                    (not use_grad_accum or ((batch_idx + 1) % grad_accum_steps == 0))):
                    self.scheduler.step()
                
                # Update running loss
                # Note: loss.item() already scaled by grad_accum_steps if accumulation is used
                actual_loss = loss.item() * (grad_accum_steps if use_grad_accum else 1)
                running_loss += actual_loss
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
                    self.run_validation(epoch, batch_idx + 1, global_step, total_steps)
                # Free GPU tensors for this batch
                del sequences, input_seq, target_seq, output_seq, loss

            # Scheduler step per epoch for other schedulers
            if self.scheduler and self.config['training']['scheduler'] not in ['linear_warmup', 'warmup_cosine', 'warmup_lineardecay']:
                self.scheduler.step()
        
            # Save checkpoint at end of every epoch
            self.save_checkpoint(epoch, global_step)
        
        self.logger.info("GBM training completed!")
        
        # Load best model for video generation
        try:
            best_model_for_video = self.load_best_model_for_inference()
        except (ValueError, FileNotFoundError) as e:
            self.logger.warning(f"Could not load best model checkpoint: {e}")
            self.logger.warning("Using current model state for video generation...")
            best_model_for_video = self.model
        
        # Generate validation comparison video
        try:
            self.logger.info("Generating validation comparison video...")
            self.logger.info("Using best model checkpoint for video generation...")
            video_path = create_validation_video(
                model=best_model_for_video,
                validation_loader=self.test_loader,
                device=self.device,
                experiment_dir=self.experiment_dir,
                video_name="gbm_validation_comparison.mp4",
                seq2seq=True  # GBM is a seq2seq model that predicts next frames
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

    def load_best_model_for_inference(self) -> torch.nn.Module:
        """
        Load the best model checkpoint for inference.
        
        Returns:
            The loaded model.
            
        Raises:
            ValueError: If no best model checkpoint path is available
            FileNotFoundError: If the best model checkpoint file doesn't exist
        """
        if self.best_model_path is None:
            raise ValueError("No best model checkpoint found. Please train the model first.")
        
        if not self.best_model_path.exists():
            raise FileNotFoundError(f"Best model checkpoint file not found: {self.best_model_path}")
        
        self.logger.info(f"Loading best model from: {self.best_model_path}")
        state_dict = torch.load(self.best_model_path, map_location=self.device)
        
        # Handle torch.compile() prefixes (_orig_mod.) in state dict keys
        model_state_dict = state_dict['model_state_dict']
        cleaned_state_dict = {}
        for key, value in model_state_dict.items():
            # Remove _orig_mod. prefix if present (from torch.compile)
            if key.startswith('_orig_mod.'):
                cleaned_key = key[len('_orig_mod.'):]
            else:
                cleaned_key = key
            cleaned_state_dict[cleaned_key] = value
        
        # Create a new model instance to avoid modifying the current model's state
        model_config = self.config['model']
        ensemble_size = model_config.get('ensemble_size', 1)
        
        if ensemble_size > 1:
            best_model = EnsembleGBM(
                n_models=ensemble_size,
                d_model=model_config['d_model'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                autoencoder_path=model_config.get('autoencoder_path'),
                volume_size=tuple(model_config['volume_size']),
                region_size=tuple(model_config['region_size']),
                different_seeds=model_config.get('different_seeds', True)
            )
        else:
            best_model = GBM(
                d_model=model_config['d_model'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                autoencoder_path=model_config.get('autoencoder_path'),
                volume_size=tuple(model_config['volume_size']),
                region_size=tuple(model_config['region_size'])
            )
        
        best_model.load_state_dict(cleaned_state_dict)
        best_model.to(self.device)
        
        # Log information about the best model
        best_val_loss = state_dict.get('validation_loss', 'unknown')
        best_epoch = state_dict.get('epoch', 'unknown')
        best_batch = state_dict.get('batch_idx', 'unknown')
        model_size = sum(p.numel() for p in best_model.parameters())
        
        self.logger.info(f"Best model loaded successfully:")
        self.logger.info(f"  Validation loss: {best_val_loss}")
        self.logger.info(f"  Epoch: {best_epoch}, Batch: {best_batch}")
        self.logger.info(f"  Model size: {model_size:,} parameters")
        
        return best_model


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