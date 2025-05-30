"""
TestDataLoader: Load and separate model input vs evaluation data from GBM test results.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TestDataLoader:
    """
    Load test data and predictions from GBM experiment results.
    
    Handles separation of:
    - Model input data: sampled probabilities (for model predictions)
    - Evaluation data: thresholded binary data (for performance metrics)
    """
    
    def __init__(self, experiment_path: str, device: str = 'cuda'):
        """
        Initialize TestDataLoader.
        
        Args:
            experiment_path: Path to experiment directory containing test_data_and_predictions.h5
            device: Device for PyTorch tensors ('cuda' or 'cpu')
        """
        self.experiment_path = Path(experiment_path)
        self.device = device
        
        # Look for test data file in multiple possible locations
        possible_paths = [
            self.experiment_path / "test_data_and_predictions.h5",  # Direct path
            self.experiment_path / "test_data" / "test_data_and_predictions.h5",  # In test_data subdirectory
            self.experiment_path / "finetune" / "test_data" / "test_data_and_predictions.h5",  # In finetune/test_data
            self.experiment_path / "pretrain" / "test_data" / "test_data_and_predictions.h5",  # In pretrain/test_data
        ]
        
        self.test_file_path = None
        for path in possible_paths:
            if path.exists():
                self.test_file_path = path
                logger.info(f"Found test data file at: {path}")
                break
        
        if self.test_file_path is None:
            available_paths = "\n".join([f"  - {p}" for p in possible_paths])
            raise FileNotFoundError(f"Test data file not found in any of these locations:\n{available_paths}")
        
        # Load and cache data
        self._load_data()
    
    def _load_data(self):
        """Load all data from the HDF5 file."""
        logger.info(f"Loading test data from {self.test_file_path}")
        
        with h5py.File(self.test_file_path, 'r') as f:
            # Check available keys and adapt to actual file structure
            available_keys = list(f.keys())
            logger.info(f"Available keys in HDF5 file: {available_keys}")
            
            # Load test data (ground truth sequences) - these are probabilities that need clamping
            if 'test_data' in f:
                test_data_raw = torch.tensor(
                    f['test_data'][:],
                    dtype=torch.float32,
                    device=self.device
                )
                logger.info(f"Loaded test_data with shape: {test_data_raw.shape}")
                logger.info(f"Test data range: {test_data_raw.min():.6f} to {test_data_raw.max():.6f}")
                
                # Clamp test data to valid probability range [0, 1]
                test_data_clamped = torch.clamp(test_data_raw, 0.0, 1.0)
                logger.info(f"Clamped test data range: {test_data_clamped.min():.6f} to {test_data_clamped.max():.6f}")
                
            else:
                raise KeyError("'test_data' not found in HDF5 file")
            
            # Load predictions (model outputs) - should already be clamped probabilities
            if 'predicted_probabilities' in f:
                predictions_raw = torch.tensor(
                    f['predicted_probabilities'][:], 
                    dtype=torch.float32, 
                    device=self.device
                )
                logger.info(f"Loaded predicted_probabilities with shape: {predictions_raw.shape}")
                logger.info(f"Predictions range: {predictions_raw.min():.6f} to {predictions_raw.max():.6f}")
                
                # Ensure predictions are also clamped (should already be, but safety)
                self.next_frame_predictions = torch.clamp(predictions_raw, 0.0, 1.0)
                
                # Load sequence_z_starts if available (starting z-plane index per sequence)
                if 'sequence_z_starts' in f:
                    seq_z = f['sequence_z_starts'][:]
                    # Store raw sequence z-starts
                    self.sequence_z_starts = seq_z
                    logger.info(f"Loaded sequence_z_starts with shape: {seq_z.shape}")
                else:
                    self.sequence_z_starts = None
            
            else:
                raise KeyError("'predicted_probabilities' not found in HDF5 file")
            
            # Separate model input vs evaluation data properly:
            # Model input: sampled binary values from clamped probabilities (for autoregressive prediction)
            self.model_input_data = torch.bernoulli(test_data_clamped)
            
            # Evaluation data: binary ground truth from thresholded clamped probabilities (for metrics)
            self.evaluation_data = (test_data_clamped > 0.5).float()
            
            # Load metadata
            if 'metadata' in f:
                self.metadata = {}
                # Load attributes
                if hasattr(f['metadata'], 'attrs'):
                    self.metadata.update(dict(f['metadata'].attrs))
                # Load datasets within metadata group
                if hasattr(f['metadata'], 'keys'):
                    for key in f['metadata'].keys():
                        try:
                            # Try to load as array data
                            self.metadata[key] = f['metadata'][key][:]
                        except:
                            # If it fails, just store the reference
                            self.metadata[key] = str(f['metadata'][key])
            else:
                self.metadata = {}
            
        logger.info(f"Loaded data shapes:")
        logger.info(f"  Model input: {self.model_input_data.shape}")
        logger.info(f"  Evaluation data: {self.evaluation_data.shape}")
        logger.info(f"  Next frame predictions: {self.next_frame_predictions.shape}")
        
        # Verify data consistency
        if self.model_input_data.shape[0] != self.evaluation_data.shape[0]:
            raise ValueError("Model input and evaluation data must have same batch size")
        
        # Check that predictions align with expected temporal dimensions
        expected_pred_frames = self.model_input_data.shape[1] - 1  # Should predict T-1 frames
        if self.next_frame_predictions.shape[1] != expected_pred_frames:
            logger.warning(f"Prediction temporal dimension ({self.next_frame_predictions.shape[1]}) "
                         f"doesn't match expected ({expected_pred_frames}). Adjusting evaluation logic.")
            
        # Reshape flattened data into 4D volumes if grid_size metadata available
        if hasattr(self, 'metadata') and 'grid_size' in self.metadata:
            try:
                Z = int(self.metadata['grid_size'][2])
                T = self.model_input_data.shape[0] // Z
                H = self.model_input_data.shape[1]
                W = self.model_input_data.shape[2]
                # Create 4D volume data: (num_volumes, Z, H, W)
                self.model_input_data_4d = self.model_input_data.view(T, Z, H, W)
                self.evaluation_data_4d = self.evaluation_data.view(T, Z, H, W)
                logger.info(f"Reshaped data to 4D volumes: {T} volumes of size ({Z}, {H}, {W})")
            except Exception as e:
                logger.warning(f"Could not reshape data to 4D volumes: {e}")
        
        # Save first sequence z-start for convenience
        if self.sequence_z_starts is not None and len(self.sequence_z_starts) > 0:
            self.sequence_z_start = int(self.sequence_z_starts[0])
        else:
            self.sequence_z_start = 0
    
    def get_next_frame_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data for next-frame prediction evaluation.
        
        Returns:
            Tuple of (input_frames, ground_truth_frames, predictions)
            - input_frames: Model input data for frames t=1 to t=N (binary sampled values)
            - ground_truth_frames: Binary ground truth for frames t=2 to t=N+1  
            - predictions: Model predictions for frames t=2 to t=N+1 (probabilities)
        """
        # Handle 4D data (multiple sequences) correctly
        if self.model_input_data.ndim == 4:
            # 4D data: [N_sequences, T, H, W] -> flatten to [N_sequences * (T-1), H, W]
            N, T, H, W = self.model_input_data.shape
            
            # Input: frames 0 to T-2 for each sequence (binary sampled values for context)
            input_frames = self.model_input_data[:, :-1].reshape(-1, H, W)  # [N*(T-1), H, W]
            
            # Ground truth: frames 1 to T-1 for each sequence (binary for evaluation)
            ground_truth_frames = self.evaluation_data[:, 1:].reshape(-1, H, W)  # [N*(T-1), H, W]
            
            # Predictions: frames 1 to T-1 for each sequence (probabilities)
            predictions = self.next_frame_predictions.reshape(-1, H, W)  # [N*(T-1), H, W]
            
        else:
            # 3D data: [T, H, W] -> keep as is
            # Input: frames 0 to T-2 (binary sampled values for context)
            input_frames = self.model_input_data[:-1]  # t=0 to t=T-2
            
            # Ground truth: frames 1 to T-1 (binary for evaluation)
            ground_truth_frames = self.evaluation_data[1:]  # t=1 to t=T-1
            
            # Predictions: frames 1 to T-1 (probabilities)
            predictions = self.next_frame_predictions  # Should be [T-1, H, W]
        
        assert input_frames.shape[0] == ground_truth_frames.shape[0] == predictions.shape[0], \
            f"Temporal dimensions must match: input={input_frames.shape[0]}, gt={ground_truth_frames.shape[0]}, pred={predictions.shape[0]}"
        
        return input_frames, ground_truth_frames, predictions
    
    def get_long_horizon_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data for long-horizon prediction evaluation.
        
        Returns:
            Tuple of (initial_frames, ground_truth_continuation)
            - initial_frames: First 165 frames (binary sampled values for model input)
            - ground_truth_continuation: Next 165 frames (binary for evaluation)
        """
        # Handle multiple samples: return batch of sequences for long-horizon evaluation
        if self.model_input_data.ndim == 4:
            N, T, H, W = self.model_input_data.shape
            logger.info(f"Batch long-horizon detected: {N} sequences of {T} frames each")
            data_all = self.model_input_data  # (N, T, H, W)
            eval_all = self.evaluation_data    # (N, T, H, W)
        else:
            # Single sequence: make batch dimension
            data_all = self.model_input_data.unsqueeze(0)
            eval_all = self.evaluation_data.unsqueeze(0)
            N, T, H, W = data_all.shape
            logger.info(f"Single-sample long-horizon: 1 sequence of {T} frames")
        # Validate sufficient frames
        if T < 330:
            raise ValueError(f"Need at least 330 frames for long-horizon evaluation, got {T}")
        # Define chunk and initial sizes
        chunk_size = 330
        initial_size = 165
        # Extract initial and continuation batches
        initial_all = data_all[:, :initial_size, :, :]  # (N, 165, H, W)
        cont_all = eval_all[:, initial_size:chunk_size, :, :]  # (N, 165, H, W)
        logger.info(f"Long-horizon data extracted: initial_frames={initial_all.shape}, ground_truth_continuation={cont_all.shape}")
        return initial_all, cont_all
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        return self.metadata.copy()
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        return {
            'total_frames': self.model_input_data.shape[0],
            'spatial_shape': self.model_input_data.shape[1:],
            'device': str(self.device),
            'experiment_path': str(self.experiment_path),
            'metadata': self.get_metadata()
        } 