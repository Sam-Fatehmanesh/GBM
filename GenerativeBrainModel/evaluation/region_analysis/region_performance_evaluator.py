"""
RegionPerformanceEvaluator: Calculate region-specific performance metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add utils path for mask loader
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.masks import ZebrafishMaskLoader
from .volume_grouper import VolumeGrouper

logger = logging.getLogger(__name__)


class RegionPerformanceEvaluator:
    """
    Evaluate model performance across different brain regions.
    
    Handles:
    - Mask loading and region definition
    - Performance calculation per region
    - Temporal performance tracking for long-horizon evaluation
    - Proper thresholding for binary evaluation
    """
    
    def __init__(self, 
                 masks_path: str = "masks",
                 target_shape: Tuple[int, int, int] = (30, 256, 128),
                 threshold: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize RegionPerformanceEvaluator.
        
        Args:
            masks_path: Path to directory containing brain region masks
            target_shape: Target shape for mask downsampling
            threshold: Threshold for binary conversion of predictions
            device: Device for computations
        """
        self.masks_path = masks_path
        self.target_shape = target_shape
        self.threshold = threshold
        self.device = device
        
        # Load brain region masks
        self._load_masks()
    
    def _load_masks(self):
        """Load and prepare brain region masks."""
        logger.info(f"Loading brain masks from {self.masks_path}")
        
        try:
            self.mask_loader = ZebrafishMaskLoader(
                masks_dir=self.masks_path,
                target_shape=self.target_shape,
                device=self.device
            )
            
            # Get available region names
            self.region_names = self.mask_loader.list_masks()
            logger.info(f"Loaded {len(self.region_names)} brain regions")
            
            # Create combined masks for performance calculation
            self._prepare_region_masks()
            
            self.sequence_z_start = None  # To be set externally for dynamic mask per frame
            self.initial_length = None
            self.Z = None
            
        except Exception as e:
            logger.error(f"Failed to load masks: {e}")
            raise
    
    def _prepare_region_masks(self):
        """Prepare region masks for performance calculation."""
        self.region_masks = {}
        self.region_masks3d = {}
        
        for region_name in self.region_names:
            # Get mask and flatten to 2D for performance calculation
            mask_3d = self.mask_loader.get_mask(region_name)
            self.region_masks3d[region_name] = mask_3d.float()
            mask_2d = torch.sum(mask_3d, dim=0) > 0
            self.region_masks[region_name] = mask_2d.float()
        
        # Add EntireBrain region - a mask of all ones (entire frame)
        if len(self.region_masks) > 0:
            # Get shape from any existing mask for 2D version
            first_mask_2d = next(iter(self.region_masks.values()))
            entire_brain_mask_2d = torch.ones_like(first_mask_2d)
            self.region_masks['EntireBrain'] = entire_brain_mask_2d
            
            # Also create 3D version for dynamic mask access
            first_mask_3d = next(iter(self.region_masks3d.values()))
            entire_brain_mask_3d = torch.ones_like(first_mask_3d)
            self.region_masks3d['EntireBrain'] = entire_brain_mask_3d
            
            self.region_names.append('EntireBrain')
        
        logger.info(f"Prepared 2D masks for {len(self.region_masks)} regions (including EntireBrain)")
    
    def calculate_metrics(self, 
                         predictions: torch.Tensor, 
                         ground_truth: torch.Tensor,
                         mask: torch.Tensor) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for a specific region.
        
        Args:
            predictions: Binary predictions tensor
            ground_truth: Binary ground truth tensor  
            mask: Region mask tensor
            
        Returns:
            Dictionary with precision, recall, F1, and support
        """
        # Apply mask to get region-specific data - only consider pixels within the mask
        mask_bool = mask > 0  # Convert to boolean mask
        
        # Extract predictions and ground truth only for masked pixels
        pred_masked = predictions[mask_bool]
        gt_masked = ground_truth[mask_bool]
        
        mask_size = mask_bool.sum().item()
        
        # If mask is too small, return zero metrics
        if mask_size < 1:
            return {
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'support': 0.0, 'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                'total_pixels': mask_size
            }
        
        # Calculate confusion matrix components only within the masked region
        tp = torch.sum((pred_masked == 1) & (gt_masked == 1)).float()
        fp = torch.sum((pred_masked == 1) & (gt_masked == 0)).float()
        fn = torch.sum((pred_masked == 0) & (gt_masked == 1)).float()
        tn = torch.sum((pred_masked == 0) & (gt_masked == 0)).float()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Support (number of positive examples in ground truth within mask)
        support = torch.sum(gt_masked == 1).float()
        
        # Additional metrics (only consider pixels within the mask)
        total_pixels = mask_size
        accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
        
        # Convert to Python floats safely
        def safe_item(x):
            return x.item() if hasattr(x, 'item') else float(x)
        
        return {
            'precision': safe_item(precision),
            'recall': safe_item(recall),
            'f1': safe_item(f1),
            'accuracy': safe_item(accuracy),
            'support': safe_item(support),
            'tp': safe_item(tp),
            'fp': safe_item(fp),
            'fn': safe_item(fn),
            'tn': safe_item(tn),
            'total_pixels': safe_item(total_pixels)
        }
    
    def evaluate_next_frame_predictions(self, 
                                      predictions: torch.Tensor,
                                      ground_truth: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Evaluate next-frame predictions across all regions.
        
        Args:
            predictions: Predicted frames of shape (T, H, W) - should be binary spikes
            ground_truth: Ground truth frames of shape (T, H, W) - binary spikes
            
        Returns:
            Dictionary mapping region names to performance metrics
        """
        logger.info(f"Evaluating next-frame predictions for {predictions.shape[0]} frames")
        
        # Convert probabilities to binary using threshold
        if torch.all((predictions == 0) | (predictions == 1)):
            pred_binary = predictions.float()
        else:
            # Deterministic thresholding instead of random sampling
            pred_binary = (predictions > self.threshold).float()
            logger.info(f"Thresholding predictions at {self.threshold} for binary evaluation")
        
        gt_binary = ground_truth  # Should already be binary from TestDataLoader
        
        # Vectorized batch evaluation across frames
        final_results = {}
        T = pred_binary.shape[0]
        for region_name in self.region_names:
            mask = self.region_masks[region_name].bool()
            total_pixels_per_frame = mask.sum().item()
            # Flatten predictions and ground truth over time and space
            pred_flat = pred_binary[:, mask].reshape(-1)
            gt_flat = gt_binary[:, mask].reshape(-1)
            tp = ((pred_flat == 1) & (gt_flat == 1)).sum().float()
            fp = ((pred_flat == 1) & (gt_flat == 0)).sum().float()
            fn = ((pred_flat == 0) & (gt_flat == 1)).sum().float()
            tn = ((pred_flat == 0) & (gt_flat == 0)).sum().float()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (total_pixels_per_frame * T) if total_pixels_per_frame * T > 0 else 0.0
            support = tp + fn
            final_results[region_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'total_pixels': total_pixels_per_frame * T
            }
            # Convert any tensor values to Python scalars for JSON serialization
            for k, v in final_results[region_name].items():
                if isinstance(v, torch.Tensor):
                    final_results[region_name][k] = v.item()
        logger.info(f"Next-frame evaluation complete for {len(final_results)} regions")
        return final_results
    

    
    def evaluate_long_horizon_predictions_by_volumes(self, 
                                                   predictions: torch.Tensor,
                                                   ground_truth: torch.Tensor,
                                                   volume_boundaries: List[Tuple[int, int]]) -> Dict[str, List[Dict[str, float]]]:
        """
        Evaluate long-horizon predictions using actual brain volume boundaries.
        
        Args:
            predictions: Predicted frames of shape (T, H, W) - should be binary spikes
            ground_truth: Ground truth frames of shape (T, H, W) - binary spikes
            volume_boundaries: List of (start_idx, end_idx) tuples for brain volumes
            
        Returns:
            Dictionary mapping region names to lists of volume-based metrics
        """
        logger.info(f"Evaluating long-horizon predictions for {len(volume_boundaries)} brain volumes")
        
        # Convert probabilities to binary using threshold
        if torch.all((predictions == 0) | (predictions == 1)):
            pred_binary = predictions.float()
        else:
            # Deterministic thresholding instead of random sampling
            pred_binary = (predictions > self.threshold).float()
            logger.info(f"Thresholding predictions at {self.threshold} for binary evaluation")
        
        gt_binary = ground_truth  # Should already be binary from TestDataLoader

        # Debug breakpoint removed; now vectorized per-volume metrics
        
        T = predictions.shape[0]
        results = {region_name: [] for region_name in self.region_names}
        
        for volume_idx, (start_idx, end_idx) in enumerate(tqdm(volume_boundaries, desc="Volume-based evaluation", unit="volume")):
            # Ensure boundaries are within available data
            start_idx = max(0, start_idx)
            end_idx = min(T, end_idx)
            
            if start_idx >= end_idx:
                logger.warning(f"Invalid volume boundary: {start_idx} to {end_idx}")
                continue
            
            volume_size = end_idx - start_idx
            
            # Get data for this brain volume
            pred_volume = pred_binary[start_idx:end_idx]
            gt_volume = gt_binary[start_idx:end_idx]
            
            # Vectorized metrics per region for this volume
            for region_name in self.region_names:
                mask = self.region_masks[region_name].bool()
                total_pixels_per_frame = mask.sum().item()
                total_pixels_volume = total_pixels_per_frame * volume_size
                # Flatten predictions and ground truth across all frames and mask
                pred_flat = pred_volume[:, mask].reshape(-1)
                gt_flat = gt_volume[:, mask].reshape(-1)
                tp = ((pred_flat == 1) & (gt_flat == 1)).sum().float()
                fp = ((pred_flat == 1) & (gt_flat == 0)).sum().float()
                fn = ((pred_flat == 0) & (gt_flat == 1)).sum().float()
                tn = ((pred_flat == 0) & (gt_flat == 0)).sum().float()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / total_pixels_volume if total_pixels_volume > 0 else 0.0
                support = tp + fn
                metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'support': support,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'total_pixels': total_pixels_volume,
                    'volume_idx': volume_idx,
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'volume_size': volume_size
                }
                # Convert any tensor values to Python scalars for safe numpy ops
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        metrics[k] = v.item()
                results[region_name].append(metrics)
        
        logger.info(f"Long-horizon evaluation complete: {len(volume_boundaries)} brain volumes")
        return results
    
    def evaluate_long_horizon_predictions(self, 
                                        predictions: torch.Tensor,
                                        ground_truth: torch.Tensor,
                                        time_window: int = 10) -> Dict[str, List[Dict[str, float]]]:
        """
        Evaluate long-horizon predictions with temporal tracking.
        
        Args:
            predictions: Predicted frames of shape (T, H, W) - should be binary spikes
            ground_truth: Ground truth frames of shape (T, H, W) - binary spikes
            time_window: Size of time windows for evaluation
            
        Returns:
            Dictionary mapping region names to lists of temporal metrics
        """
        logger.info(f"Evaluating long-horizon predictions for {predictions.shape[0]} frames")
        
        # Convert probabilities to binary using threshold
        if torch.all((predictions == 0) | (predictions == 1)):
            pred_binary = predictions.float()
        else:
            # Deterministic thresholding instead of random sampling
            pred_binary = (predictions > self.threshold).float()
            logger.info(f"Thresholding predictions at {self.threshold} for binary evaluation")
        
        gt_binary = ground_truth  # Should already be binary from TestDataLoader
        
        T = predictions.shape[0]
        num_windows = T // time_window
        
        results = {region_name: [] for region_name in self.region_names}
        
        for window_idx in tqdm(range(num_windows), desc="Time-window evaluation", unit="window"):
            start_idx = window_idx * time_window
            end_idx = min((window_idx + 1) * time_window, T)
            window_size = end_idx - start_idx
            
            # Get data for this time window
            pred_window = pred_binary[start_idx:end_idx]
            gt_window = gt_binary[start_idx:end_idx]
            
            # Accumulate metrics across frames in this window (like next-frame evaluation)
            window_results = {region_name: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} 
                            for region_name in self.region_names}
            
            for t in range(window_size):
                pred_frame = pred_window[t]
                gt_frame = gt_window[t]
                
                for region_name in self.region_names:
                    # Use 2D masks for consistent evaluation with next-frame predictions
                    # This ensures all regions have meaningful mask coverage
                    mask = self.region_masks[region_name]
                    frame_metrics = self.calculate_metrics(pred_frame, gt_frame, mask)
                    
                    # Accumulate counts for this window
                    window_results[region_name]['tp'] += frame_metrics['tp']
                    window_results[region_name]['fp'] += frame_metrics['fp']
                    window_results[region_name]['fn'] += frame_metrics['fn']
                    window_results[region_name]['tn'] += frame_metrics['tn']
            
            # Calculate metrics for this time window
            for region_name in self.region_names:
                counts = window_results[region_name]
                tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
                
                # Total pixels for this window
                mask = self.region_masks[region_name]
                total_pixels_per_frame = torch.sum(mask).item()
                total_pixels_window = total_pixels_per_frame * window_size
                
                # Calculate metrics for this window
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / total_pixels_window if total_pixels_window > 0 else 0.0
                support = tp + fn  # Total positive examples
                
                metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'support': support,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'total_pixels': total_pixels_window,
                    'time_window': window_idx,
                    'start_frame': start_idx,
                    'end_frame': end_idx
                }
                
                results[region_name].append(metrics)
        
        logger.info(f"Long-horizon evaluation complete: {num_windows} time windows")
        return results
    
    def get_region_summary(self, 
                          evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Get summary statistics across all regions.
        
        Args:
            evaluation_results: Results from region evaluation
            
        Returns:
            Summary statistics dictionary
        """
        if not evaluation_results:
            return {}
        
        # Extract metrics across all regions and convert to Python floats
        precisions = [float(r['precision'].cpu().item()) if hasattr(r['precision'], 'cpu') else float(r['precision']) for r in evaluation_results.values()]
        recalls = [float(r['recall'].cpu().item()) if hasattr(r['recall'], 'cpu') else float(r['recall']) for r in evaluation_results.values()]
        f1_scores = [float(r['f1'].cpu().item()) if hasattr(r['f1'], 'cpu') else float(r['f1']) for r in evaluation_results.values()]
        accuracies = [float(r['accuracy'].cpu().item()) if hasattr(r['accuracy'], 'cpu') else float(r['accuracy']) for r in evaluation_results.values()]
        supports = [float(r['support'].cpu().item()) if hasattr(r['support'], 'cpu') else float(r['support']) for r in evaluation_results.values()]
        
        # Calculate weighted averages (weighted by support)
        total_support = sum(supports)
        
        if total_support > 0:
            weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
            weighted_recall = sum(rc * s for rc, s in zip(recalls, supports)) / total_support
            weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
            weighted_accuracy = sum(acc * s for acc, s in zip(accuracies, supports)) / total_support
        else:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
            weighted_accuracy = 0.0
        
        return {
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_f1': np.mean(f1_scores),
            'mean_accuracy': np.mean(accuracies),
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'weighted_accuracy': weighted_accuracy,
            'num_regions': len(evaluation_results),
            'total_support': total_support
        }
    
    def get_temporal_summary(self, 
                           temporal_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[float]]:
        """
        Get temporal summary across regions.
        
        Args:
            temporal_results: Results from long-horizon evaluation
            
        Returns:
            Temporal summary with metrics over time
        """
        if not temporal_results:
            return {}
        
        # Get number of time windows
        num_windows = len(next(iter(temporal_results.values())))
        
        temporal_summary = {
            'time_points': list(range(num_windows)),
            'mean_f1_over_time': [],
            'mean_precision_over_time': [],
            'mean_recall_over_time': [],
            'std_f1_over_time': [],
            'std_precision_over_time': [],
            'std_recall_over_time': []
        }
        
        for window_idx in range(num_windows):
            # Collect metrics across all regions for this time window
            window_f1s = []
            window_precisions = []
            window_recalls = []
            
            for region_results in temporal_results.values():
                if window_idx < len(region_results):
                    window_f1s.append(region_results[window_idx]['f1'])
                    window_precisions.append(region_results[window_idx]['precision'])
                    window_recalls.append(region_results[window_idx]['recall'])
            
            # Calculate statistics for this time window
            temporal_summary['mean_f1_over_time'].append(np.mean(window_f1s))
            temporal_summary['mean_precision_over_time'].append(np.mean(window_precisions))
            temporal_summary['mean_recall_over_time'].append(np.mean(window_recalls))
            temporal_summary['std_f1_over_time'].append(np.std(window_f1s))
            temporal_summary['std_precision_over_time'].append(np.std(window_precisions))
            temporal_summary['std_recall_over_time'].append(np.std(window_recalls))
        
        return temporal_summary
    
    def get_available_regions(self) -> List[str]:
        """Get list of available brain regions."""
        return self.region_names.copy()
    
    def get_region_info(self) -> Dict[str, Dict[str, float]]:
        """Get information about each region."""
        region_info = {}
        
        for region_name in self.region_names:
            mask = self.region_masks[region_name]
            total_pixels = torch.sum(mask).item()
            
            region_info[region_name] = {
                'total_pixels': total_pixels,
                'coverage_ratio': total_pixels / mask.numel()
            }
        
        return region_info 