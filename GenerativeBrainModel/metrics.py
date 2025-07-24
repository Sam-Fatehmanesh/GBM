"""
Comprehensive metrics tracking system for GBM training.

This module provides reusable classes for:
- GPU-accelerated binary classification metrics
- CSV logging and management
- Exponential moving average tracking
- Validation metrics computation
"""

import csv
import torch
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style


class BinaryMetricsCalculator:
    """
    GPU-accelerated binary classification metrics calculator.
    
    Computes F1, precision, and recall using pure PyTorch tensor operations
    for maximum performance on GPU.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Binary classification threshold (default: 0.5)
        """
        self.threshold = threshold
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute binary classification metrics efficiently on GPU.
        
        Args:
            predictions: Prediction probabilities (0-1) on GPU
            targets: Target probabilities (0-1) on GPU
            
        Returns:
            Dictionary with pr_auc
        """
        # Convert to binary targets for PR AUC calculation
        binary_targets = (targets >= self.threshold).float()
        
        # Calculate PR AUC using memory-efficient binning approach
        pr_auc = self._compute_pr_auc_binned(predictions, binary_targets)
        
        # GPU memory cleanup
        del binary_targets
        torch.cuda.empty_cache()
        
        return {
            'pr_auc': pr_auc
        }
    
    def _compute_pr_auc_binned(self, predictions: torch.Tensor, binary_targets: torch.Tensor, 
                              num_bins: int = 1000, batch_size: int = 5000) -> float:
        """
        Compute PR AUC using memory-efficient batched binning approach.
        
        This method processes predictions in batches to avoid memory issues,
        incrementally building bin counts and computing PR AUC from accumulated counts.
        
        Args:
            predictions: Prediction probabilities (0-1) on GPU
            binary_targets: Binary targets (0-1) on GPU
            num_bins: Number of bins to discretize predictions (default: 1000)
            batch_size: Size of batches for processing (default: 5000)
            
        Returns:
            PR AUC score
        """
        total_positives = binary_targets.sum()
        
        if total_positives == 0:
            return 0.0
        
        # Create bins from 0 to 1
        device = predictions.device
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
        
        # Initialize accumulator arrays
        tp_counts = torch.zeros(num_bins, device=device)
        total_counts = torch.zeros(num_bins, device=device)
        
        # Process in batches with progress bar
        num_samples = predictions.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Only show progress bar if processing a significant amount of data
        show_progress = num_batches > 5
        
        batch_iterator = range(num_batches)
        if show_progress:
            from tqdm import tqdm
            batch_iterator = tqdm(batch_iterator, desc="Computing PR AUC", leave=False, ncols=80)
        
        for i in batch_iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Get batch
            batch_preds = predictions[start_idx:end_idx]
            batch_targets = binary_targets[start_idx:end_idx]
            
            # Discretize predictions into bin indices
            bin_indices = torch.searchsorted(bin_edges[1:], batch_preds, right=False)
            
            # Accumulate counts for this batch
            tp_counts.scatter_add_(0, bin_indices, batch_targets)
            total_counts.scatter_add_(0, bin_indices, torch.ones_like(batch_targets))
        
        # Calculate cumulative counts from highest threshold to lowest
        # (reverse order since higher indices = higher thresholds)
        tp_counts_flipped = torch.flip(tp_counts, [0])
        total_counts_flipped = torch.flip(total_counts, [0])
        
        cumulative_tp = torch.cumsum(tp_counts_flipped, dim=0)
        cumulative_fp = torch.cumsum(total_counts_flipped - tp_counts_flipped, dim=0)
        
        # Calculate precision and recall
        precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)
        recall = cumulative_tp / total_positives
        
        # Add starting point (recall=0, precision=1) for proper AUC calculation
        precision = torch.cat([torch.tensor([1.0], device=device), precision])
        recall = torch.cat([torch.tensor([0.0], device=device), recall])
        
        # Calculate AUC using trapezoidal rule
        recall_diff = recall[1:] - recall[:-1]
        auc = torch.sum(recall_diff * precision[:-1]).item()
        
        # GPU memory cleanup
        del tp_counts, total_counts, tp_counts_flipped, total_counts_flipped
        del cumulative_tp, cumulative_fp, precision, recall, recall_diff
        del bin_edges
        torch.cuda.empty_cache()
        
        return auc


class ExponentialMovingAverage:
    """
    Exponential moving average tracker for smooth loss curves.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize EMA tracker.
        
        Args:
            alpha: Smoothing factor (0 < alpha < 1). Lower values = more smoothing.
        """
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float) -> float:
        """
        Update EMA with new value.
        
        Args:
            new_value: New value to incorporate
            
        Returns:
            Updated EMA value
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def get_value(self) -> Optional[float]:
        """Get current EMA value."""
        return self.value
    
    def reset(self):
        """Reset EMA to None."""
        self.value = None


class CSVLogger:
    """
    Thread-safe CSV logger for metrics tracking.
    """
    
    def __init__(self, csv_path: Union[str, Path], fieldnames: List[str]):
        """
        Initialize CSV logger.
        
        Args:
            csv_path: Path to CSV file
            fieldnames: List of column names
        """
        self.csv_path = Path(csv_path)
        self.fieldnames = fieldnames
        self.logger = logging.getLogger(__name__)
        
        # Create CSV file with headers
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers."""
        # Create parent directory if it doesn't exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
        
        self.logger.info(f"CSV logger initialized: {self.csv_path}")
    
    def log_metrics(self, metrics_dict: Dict[str, Union[float, int, str]]):
        """
        Log metrics to CSV file.
        
        Args:
            metrics_dict: Dictionary of metrics to log
        """
        # Filter metrics to only include defined fieldnames
        filtered_metrics = {k: v for k, v in metrics_dict.items() if k in self.fieldnames}
        
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(filtered_metrics)


class ValidationMetricsTracker:
    """
    Comprehensive validation metrics tracker with CSV logging.
    
    Combines binary metrics calculation, CSV logging, and validation management
    for easy integration into training scripts.
    """
    
    def __init__(self, csv_path: Union[str, Path], threshold: float = 0.5):
        """
        Initialize validation metrics tracker.
        
        Args:
            csv_path: Path to save validation metrics CSV
            threshold: Binary classification threshold
        """
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_calculator = BinaryMetricsCalculator(threshold=threshold)
        
        # Define CSV fields for validation metrics
        fieldnames = ['epoch', 'batch_idx', 'validation_loss', 'pr_auc']
        self.csv_logger = CSVLogger(csv_path, fieldnames)
    
    def compute_and_log_validation(self, 
                                 epoch: int, 
                                 batch_idx: int, 
                                 predictions: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 validation_loss: float) -> Dict[str, float]:
        """
        Compute validation metrics and log to CSV.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            predictions: Model predictions (probabilities 0-1) on GPU
            targets: Target values (probabilities 0-1) on GPU
            validation_loss: Validation loss value
            
        Returns:
            Dictionary of computed metrics
        """
        # Compute binary classification metrics
        metrics = self.metrics_calculator.compute_metrics(predictions, targets)
        
        # Add validation loss to metrics
        metrics['validation_loss'] = validation_loss
        metrics['epoch'] = epoch
        metrics['batch_idx'] = batch_idx
        
        # Log to CSV
        self.csv_logger.log_metrics(metrics)
        
        # Log to console
        self.logger.info(f"Validation - Loss: {validation_loss:.6f}, "
                        f"PR AUC: {metrics['pr_auc']:.4f}")
        
        # GPU memory cleanup
        del predictions, targets
        torch.cuda.empty_cache()
        
        return metrics


class TrainingMetricsTracker:
    """
    Training metrics tracker with exponential moving average and CSV logging.
    
    Tracks training loss with EMA smoothing and logs to CSV for analysis.
    """
    
    def __init__(self, csv_path: Union[str, Path], ema_alpha: float = 0.1):
        """
        Initialize training metrics tracker.
        
        Args:
            csv_path: Path to save training metrics CSV
            ema_alpha: EMA smoothing factor for loss tracking
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize EMA tracker
        self.loss_ema = ExponentialMovingAverage(alpha=ema_alpha)
        
        # Define CSV fields for training metrics
        fieldnames = ['epoch', 'batch_idx', 'training_loss', 'training_loss_ema', 'learning_rate']
        self.csv_logger = CSVLogger(csv_path, fieldnames)
    
    def log_training_step(self, 
                         epoch: int, 
                         batch_idx: int, 
                         loss: float, 
                         learning_rate: float):
        """
        Log training step metrics including EMA loss.
        
        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            loss: Training loss for this batch
            learning_rate: Current learning rate
        """
        # Update EMA
        ema_loss = self.loss_ema.update(loss)
        
        # Prepare metrics dictionary
        metrics = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'training_loss': loss,
            'training_loss_ema': ema_loss,
            'learning_rate': learning_rate
        }
        
        # Log to CSV
        self.csv_logger.log_metrics(metrics)
    
    def get_current_ema_loss(self) -> Optional[float]:
        """Get current EMA loss value."""
        return self.loss_ema.get_value()
    
    def reset_ema(self):
        """Reset EMA tracker (useful for new epochs or experiments)."""
        self.loss_ema.reset()


class PlotGenerator:
    """
    Generates comprehensive training plots from CSV metrics files.
    
    Creates vertically stacked plots showing:
    1. Training batch loss and EMA loss
    2. Validation loss over time
    3. PR AUC metric
    """
    
    def __init__(self, 
                 training_csv_path: Union[str, Path], 
                 validation_csv_path: Union[str, Path],
                 plots_dir: Union[str, Path]):
        """
        Initialize plot generator.
        
        Args:
            training_csv_path: Path to training metrics CSV
            validation_csv_path: Path to validation metrics CSV  
            plots_dir: Directory to save plots
        """
        self.training_csv_path = Path(training_csv_path)
        self.validation_csv_path = Path(validation_csv_path)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style for better-looking plots
        style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 10),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,
            'grid.alpha': 0.3
        })
    
    def _load_csv_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and validation CSV data.
        
        Returns:
            Tuple of (training_df, validation_df)
        """
        training_df = pd.DataFrame()
        validation_df = pd.DataFrame()
        
        # Load training data if file exists and has content
        if self.training_csv_path.exists():
            try:
                training_df = pd.read_csv(self.training_csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                self.logger.warning(f"Could not load training CSV: {self.training_csv_path}")
        
        # Load validation data if file exists and has content
        if self.validation_csv_path.exists():
            try:
                validation_df = pd.read_csv(self.validation_csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                self.logger.warning(f"Could not load validation CSV: {self.validation_csv_path}")
        
        return training_df, validation_df
    
    def generate_training_plots(self) -> None:
        """
        Generate comprehensive training plots and save to plots directory.
        """
        training_df, validation_df = self._load_csv_data()
        
        # Skip plotting if no data available
        if training_df.empty and validation_df.empty:
            self.logger.warning("No data available for plotting")
            return
        
        # Calculate global batch indices for consistent x-axis
        max_batch_in_epoch1 = 1
        x_train = None
        x_val = None
        
        # Process training data
        if not training_df.empty:
            if 'epoch' in training_df.columns and 'batch_idx' in training_df.columns:
                # Estimate batches per epoch from the data
                max_batch_in_epoch1 = training_df[training_df['epoch'] == 1]['batch_idx'].max() if len(training_df[training_df['epoch'] == 1]) > 0 else 1
                training_df['global_batch'] = (training_df['epoch'] - 1) * max_batch_in_epoch1 + training_df['batch_idx']
                x_train = training_df['global_batch']
            else:
                x_train = range(len(training_df))
        
        # Process validation data with same epoch scaling
        if not validation_df.empty:
            if 'epoch' in validation_df.columns and 'batch_idx' in validation_df.columns:
                validation_df['global_batch'] = (validation_df['epoch'] - 1) * max_batch_in_epoch1 + validation_df['batch_idx']
                x_val = validation_df['global_batch']
            else:
                x_val = range(len(validation_df))
        
        # Calculate unified x-axis limits
        x_min = 0
        x_max = 1
        if x_train is not None and len(x_train) > 0:
            x_max = max(x_max, x_train.max())
        if x_val is not None and len(x_val) > 0:
            x_max = max(x_max, x_val.max())
        
        # Create figure with 3 vertically stacked subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        
        # Plot 1: Training Loss and EMA
        ax1 = axes[0]
        if not training_df.empty and 'training_loss' in training_df.columns and x_train is not None:
            ax1.plot(x_train, training_df['training_loss'], alpha=0.3, color='blue', label='Batch Loss')
            if 'training_loss_ema' in training_df.columns:
                ax1.plot(x_train, training_df['training_loss_ema'], color='darkblue', label='EMA Loss', linewidth=2)
        
        ax1.set_title('Training Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(x_min, x_max)  # Unified x-axis
        
        # Plot 2: Validation Loss
        ax2 = axes[1]
        if not validation_df.empty and 'validation_loss' in validation_df.columns and x_val is not None:
            ax2.plot(x_val, validation_df['validation_loss'], color='crimson', marker='o', markersize=4, label='Validation Loss')
        
        ax2.set_title('Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(x_min, x_max)  # Unified x-axis
        
        # Plot 3: PR AUC
        ax3 = axes[2]
        if not validation_df.empty and x_val is not None:
            if 'pr_auc' in validation_df.columns:
                ax3.plot(x_val, validation_df['pr_auc'], color='red', marker='d', markersize=4, label='PR AUC')
        
        ax3.set_title('PR AUC')
        ax3.set_ylabel('Score')
        ax3.set_xlabel('Global Batch Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)  # PR AUC is between 0 and 1
        ax3.set_xlim(x_min, x_max)  # Unified x-axis
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to: {plot_path}")


class CombinedMetricsTracker:
    """
    Combined metrics tracker for both training and validation.
    
    Provides a unified interface for tracking all metrics in a training run.
    """
    
    def __init__(self, 
                 log_dir: Union[str, Path], 
                 validation_threshold: float = 0.5,
                 ema_alpha: float = 0.1,
                 enable_plotting: bool = True):
        """
        Initialize combined metrics tracker.
        
        Args:
            log_dir: Directory to save CSV files
            validation_threshold: Binary classification threshold for validation
            ema_alpha: EMA smoothing factor for training loss
            enable_plotting: Whether to generate plots during training
        """
        log_dir = Path(log_dir)
        
        # Initialize individual trackers
        self.training_tracker = TrainingMetricsTracker(
            csv_path=log_dir / 'training_metrics.csv',
            ema_alpha=ema_alpha
        )
        
        self.validation_tracker = ValidationMetricsTracker(
            csv_path=log_dir / 'validation_metrics.csv',
            threshold=validation_threshold
        )
        
        # Initialize plot generator if enabled
        self.plot_generator = None
        if enable_plotting:
            plots_dir = log_dir / 'plots'
            self.plot_generator = PlotGenerator(
                training_csv_path=log_dir / 'training_metrics.csv',
                validation_csv_path=log_dir / 'validation_metrics.csv',
                plots_dir=plots_dir
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Combined metrics tracker initialized in: {log_dir}")
        if enable_plotting:
            self.logger.info(f"Plot generation enabled in: {plots_dir}")
    
    def log_training_step(self, epoch: int, batch_idx: int, loss: float, learning_rate: float):
        """Log training step metrics."""
        self.training_tracker.log_training_step(epoch, batch_idx, loss, learning_rate)
    
    def log_validation_step(self, 
                           epoch: int, 
                           batch_idx: int, 
                           predictions: torch.Tensor, 
                           targets: torch.Tensor, 
                           validation_loss: float) -> Dict[str, float]:
        """Log validation step metrics and generate plots."""
        # Log validation metrics
        metrics = self.validation_tracker.compute_and_log_validation(
            epoch, batch_idx, predictions, targets, validation_loss
        )
        
        # Generate plots if enabled
        if self.plot_generator is not None:
            try:
                self.plot_generator.generate_training_plots()
            except Exception as e:
                self.logger.warning(f"Failed to generate plots: {e}")
        
        return metrics
    
    def get_current_training_ema(self) -> Optional[float]:
        """Get current training loss EMA."""
        return self.training_tracker.get_current_ema_loss()
    
    def reset_training_ema(self):
        """Reset training loss EMA."""
        self.training_tracker.reset_ema() 