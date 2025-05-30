"""
PredictionRunner: Handle model predictions for both evaluation types.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import sys
import pdb

# Add GenerativeBrainModel to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.gbm import GBM

logger = logging.getLogger(__name__)


class PredictionRunner:
    """
    Run predictions using trained GBM models.
    
    Handles:
    - Model loading from experiment checkpoints
    - Next-frame predictions (using existing results)
    - Long-horizon autoregressive predictions
    - Probability thresholding and sampling
    """
    
    def __init__(self, 
                 experiment_path: str,
                 device: str = 'cuda',
                 threshold: float = 0.5):
        """
        Initialize PredictionRunner.
        
        Args:
            experiment_path: Path to experiment directory
            device: Device for computation ('cuda' or 'cpu')
            threshold: Threshold for binary conversion (default 0.5)
        """
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.threshold = threshold
        self.model = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load trained GBM model from experiment checkpoint."""
        # Look for checkpoints in multiple possible locations
        possible_checkpoint_dirs = [
            self.experiment_path / "checkpoints",  # Direct path
            self.experiment_path / "finetune" / "checkpoints",  # In finetune subdirectory
            self.experiment_path / "pretrain" / "checkpoints",  # In pretrain subdirectory
        ]
        
        checkpoint_dir = None
        for dir_path in possible_checkpoint_dirs:
            if dir_path.exists() and list(dir_path.glob("*.pt")):
                checkpoint_dir = dir_path
                logger.info(f"Found checkpoints directory at: {dir_path}")
                break
        
        if checkpoint_dir is None:
            available_dirs = "\n".join([f"  - {d}" for d in possible_checkpoint_dirs])
            logger.warning(f"No checkpoints directory found in any of these locations:\n{available_dirs}")
            return
        
        # Look for the best or latest checkpoint
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {checkpoint_dir}")
            return
        
        # Prefer best checkpoint, fallback to latest
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            checkpoint_path = best_checkpoint
        else:
            # Sort by modification time and take the latest
            checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Loading model from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            elif 'config' in checkpoint:
                model_config = checkpoint['config'].get('model', {})
            else:
                # Try to infer from model state dict
                logger.warning("Model config not found in checkpoint, using defaults")
                model_config = self._infer_model_config(checkpoint['model_state_dict'])
            
            # Create model
            self.model = GBM(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _infer_model_config(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Infer model configuration from state dict."""
        # Default GBM parameters
        config = {
            'mamba_layers': 8,  # Default values that work with most models
            'mamba_dim': 1024,
            'mamba_state_multiplier': 8,
            'pretrained_ae_path': "trained_simpleAE/checkpoints/best_model.pt"
        }
        
        # Try to infer mamba_dim from the state dict
        for name, param in state_dict.items():
            if 'mamba' in name and 'norm' in name and param.dim() == 1:
                config['mamba_dim'] = param.shape[0]
                logger.info(f"Inferred mamba_dim from {name}: {param.shape[0]}")
                break
            elif 'autoencoder.encoder' in name and 'weight' in name and param.dim() == 2:
                # Encoder output dimension should match mamba_dim
                config['mamba_dim'] = param.shape[0]
                logger.info(f"Inferred mamba_dim from encoder: {param.shape[0]}")
                break
        
        logger.info(f"Inferred model config: {config}")
        return config
    
    def is_model_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        return self.model is not None
    
    def run_next_frame_predictions(self, 
                                 input_frames: torch.Tensor) -> torch.Tensor:
        """
        Run next-frame predictions.
        
        Note: For most cases, this should use pre-computed predictions from
        test_data_and_predictions.h5 rather than running the model again.
        
        Args:
            input_frames: Input frames of shape (T, H, W) or (T, C, H, W)
            
        Returns:
            Predictions of shape (T, H, W)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Cannot run predictions.")
        
        logger.info(f"Running next-frame predictions for {input_frames.shape[0]} frames")
        
        predictions = []
        
        with torch.no_grad():
            for t in range(input_frames.shape[0]):
                input_frame = input_frames[t:t+1]  # Keep batch dimension
                
                # Run model prediction
                prediction = self.model(input_frame)
                
                # Apply threshold and convert to probabilities
                prediction = torch.sigmoid(prediction)
                
                predictions.append(prediction.squeeze(0))
        
        predictions = torch.stack(predictions, dim=0)
        
        logger.info(f"Next-frame predictions complete: {predictions.shape}")
        return predictions
    
    def run_long_horizon_predictions(self, 
                                   initial_frames: torch.Tensor,
                                   num_steps: int = 220) -> torch.Tensor:
        """
        Run long-horizon autoregressive predictions using the model's built-in method.
        
        Args:
            initial_frames: Initial frames of shape (T_init, H, W) - should be at least 110 frames
            num_steps: Number of steps to predict forward
            
        Returns:
            Predictions of shape (num_steps, H, W)
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Cannot run predictions.")
        
        logger.info(f"Running long-horizon prediction for {num_steps} steps")
        logger.info(f"Initial frames shape: {initial_frames.shape}")
        
        # Add batch dimension for model input: (T, H, W) -> (1, T, H, W)
        initial_batch = initial_frames.unsqueeze(0)

        
        with torch.no_grad():
            # Use the model's built-in autoregressive generation method
            #pdb.set_trace()
            full_sequence, probabilities = self.model.generate_autoregressive_brain(
                init_x=initial_batch,
                num_steps=num_steps
            )
            logger.debug(f"Model output shapes: full_sequence={full_sequence.shape}, probabilities={probabilities.shape}")
            
            # Use probability predictions instead of sampled binary predictions
            # probabilities shape: (batch_size, num_steps, H, W)
            # These are the raw model probabilities which should maintain consistent spike rates
            predictions = probabilities.squeeze(0)  # Remove batch dimension: (num_steps, H, W)
            
            # Log probability statistics to understand the distribution
            logger.info(f"Probability statistics:")
            logger.info(f"  Range: {predictions.min():.6f} to {predictions.max():.6f}")
            logger.info(f"  Mean: {predictions.mean():.6f}")
            logger.info(f"  Std: {predictions.std():.6f}")
            
            # Check if probabilities are reasonable
            if predictions.max() < 0.01:
                logger.warning("Very low probability values detected - model may be too conservative")
            elif predictions.min() > 0.99:
                logger.warning("Very high probability values detected - model may be too aggressive")
        
        logger.info(f"Long-horizon probability predictions complete: {predictions.shape}")
        return predictions
    
    def threshold_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert predictions to binary using threshold.
        
        Args:
            predictions: Continuous predictions
            
        Returns:
            Binary thresholded predictions
        """
        return (predictions > self.threshold).float()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_loaded():
            return {"model_loaded": False}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_loaded": True,
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(self.model.parameters()).device),
            "experiment_path": str(self.experiment_path)
        }
    
    def validate_predictions(self, 
                           predictions: torch.Tensor,
                           expected_shape: Tuple[int, ...]) -> bool:
        """
        Validate prediction outputs.
        
        Args:
            predictions: Model predictions
            expected_shape: Expected output shape
            
        Returns:
            True if predictions are valid
        """
        if predictions.shape != expected_shape:
            logger.error(f"Prediction shape mismatch: got {predictions.shape}, expected {expected_shape}")
            return False
        
        # Check for NaN or inf values
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            logger.error("Predictions contain NaN or inf values")
            return False
        
        # Check value range (should be probabilities between 0 and 1)
        if torch.any(predictions < 0) or torch.any(predictions > 1):
            logger.warning("Predictions outside [0, 1] range - may need sigmoid activation")
        
        return True 