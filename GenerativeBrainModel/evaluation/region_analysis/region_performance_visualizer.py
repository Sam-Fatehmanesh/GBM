"""
RegionPerformanceVisualizer: Generate visualizations for region-specific performance.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import os
import cv2
import torch
from GenerativeBrainModel.custom_functions.visualization import create_color_coded_comparison_video
import pdb

logger = logging.getLogger(__name__)


class RegionPerformanceVisualizer:
    """
    Create visualizations for region-specific performance analysis.
    
    Handles:
    - Bar plots for next-frame prediction aggregate performance
    - Heatmaps for long-horizon temporal performance
    - Comparison plots across different metrics
    - Publication-ready styling
    """
    
    def __init__(self, 
                 output_dir: str,
                 figsize: Tuple[int, int] = (16, 10),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8'):
        """
        Initialize RegionPerformanceVisualizer.
        
        Args:
            output_dir: Directory to save plots
            figsize: Default figure size (larger for legible region names)
            dpi: DPI for saved figures
            style: Matplotlib style
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized plots
        self.next_frame_dir = self.output_dir / "next_frame"
        self.long_horizon_dir = self.output_dir / "long_horizon"
        self.next_frame_dir.mkdir(exist_ok=True)
        self.long_horizon_dir.mkdir(exist_ok=True)
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up plotting style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-darkgrid')
        
        # Color palette for brain regions
        self.region_colors = plt.cm.Set3(np.linspace(0, 1, 20))  # Support up to 20 regions
    
    def plot_next_frame_performance(self, 
                                  evaluation_results: Dict[str, Dict[str, float]],
                                  metrics: List[str] = ['precision', 'recall', 'f1']) -> Dict[str, str]:
        """
        Create separate bar plots for each metric in next-frame prediction performance.
        
        Args:
            evaluation_results: Results from region evaluation
            metrics: List of metrics to plot
            
        Returns:
            Dictionary mapping metric names to saved plot paths
        """
        logger.info(f"Creating next-frame performance plots for {len(evaluation_results)} regions")
        
        # Prepare data
        regions = list(evaluation_results.keys())
        n_regions = len(regions)
        plot_paths = {}
        
        # Calculate larger figure size based on number of regions
        figsize = (max(16, n_regions * 0.6), 10)
        
        # Create separate plot for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get values for this metric
            values = [evaluation_results[region][metric] for region in regions]
            
            # Sort regions by performance for better visualization
            sorted_pairs = sorted(zip(regions, values), key=lambda x: x[1], reverse=True)
            sorted_regions, sorted_values = zip(*sorted_pairs) if sorted_pairs else ([], [])
            # Convert tensor values to Python floats to avoid CUDA numpy conversion errors
            sorted_values = [val.cpu().item() if isinstance(val, torch.Tensor) else float(val) for val in sorted_values]
            
            # Create color gradient based on performance
            colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(sorted_values)))
            
            # Create bar plot
            bars = ax.bar(range(len(sorted_regions)), sorted_values, 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, sorted_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Customize plot
            ax.set_xlabel('Brain Region', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=14, fontweight='bold')
            ax.set_title(f'Next-Frame {metric.capitalize()} Performance by Region', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(range(len(sorted_regions)))
            ax.set_xticklabels(sorted_regions, rotation=45, ha='right', fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot to next_frame directory
            filename = f"next_frame_{metric}_performance.png"
            filepath = self.next_frame_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            plot_paths[metric] = str(filepath)
            logger.info(f"Saved next-frame {metric} plot to {filepath}")
        
        return plot_paths
    
    def plot_long_horizon_heatmap(self, 
                                temporal_results: Dict[str, List[Dict[str, float]]],
                                metric: str = 'f1',
                                title: str = "Long-Horizon Performance Heatmap") -> str:
        """
        Create heatmap for long-horizon temporal performance.
        
        Args:
            temporal_results: Results from long-horizon evaluation
            metric: Metric to visualize
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        logger.info(f"Creating long-horizon heatmap for metric: {metric}")
        
        # Prepare data matrix
        regions = list(temporal_results.keys())
        
        if not regions:
            logger.warning("No temporal results to plot")
            return ""
        
        # Get number of time windows
        max_windows = max(len(results) for results in temporal_results.values())
        
        # Limit to top performing regions to make heat map readable
        # Calculate average F1 score for each region to sort
        region_avg_scores = {}
        for region in regions:
            region_results = temporal_results[region]
            if region_results:
                scores = [w.get(metric, 0) for w in region_results if metric in w]
                region_avg_scores[region] = np.mean(scores) if scores else 0
            else:
                region_avg_scores[region] = 0
        
        # Sort regions by average performance and take top 30 + EntireBrain
        sorted_regions = sorted(region_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Always include EntireBrain if present, then top 29 others
        display_regions = []
        if 'EntireBrain' in regions:
            display_regions.append('EntireBrain')
        
        for region, _ in sorted_regions:
            if region != 'EntireBrain' and len(display_regions) < 30:
                display_regions.append(region)
        
        # Create data matrix (regions × time windows)
        data_matrix = np.full((len(display_regions), max_windows), np.nan)
        
        for i, region in enumerate(display_regions):
            region_results = temporal_results[region]
            for j, window_result in enumerate(region_results):
                if j < max_windows and metric in window_result:
                    data_matrix[i, j] = window_result[metric]
        
        # Create better proportioned figure with capped width
        # Use 0.3 inch per window up to a max of 50 inches
        cell_width = 0.3
        max_fig_width = 50
        fig_width = max(12, min(max_windows * cell_width, max_fig_width))
        fig_height = max(8, len(display_regions) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap with proper alignment, scaling color from 0 to max value in the data
        vmin = 0
        vmax = np.nanmax(data_matrix)
        im = ax.imshow(data_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels to align with pixel centers
        ax.set_xticks(np.arange(max_windows))
        ax.set_yticks(np.arange(len(display_regions)))
        ax.set_xticklabels([f'V{i+1}' for i in range(max_windows)])  # Changed from T to V for Volume
        ax.set_yticklabels(display_regions)
        
        # Add grid lines between cells for better alignment visualization
        ax.set_xticks(np.arange(max_windows + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(display_regions) + 1) - 0.5, minor=True)
        # Disable major grid to avoid double lines
        ax.grid(False)
        # Draw only minor grid lines between cells
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f'{metric.capitalize()} Score', rotation=-90, va="bottom")
        
        # Remove text annotations for cleaner look
        # (Numbers removed per user request)
        
        # Set labels and title
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Brain Region')
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to long_horizon directory
        filename = f"long_horizon_heatmap_{metric}.png"
        filepath = self.long_horizon_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved long-horizon heatmap to {filepath}")
        return str(filepath)
    
    def plot_temporal_trends(self, 
                           temporal_summary: Dict[str, List[float]],
                           title: str = "Performance Trends Over Time") -> str:
        """
        Plot temporal trends with error bars.
        
        Args:
            temporal_summary: Summary statistics over time
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating temporal trends plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        time_points = temporal_summary['time_points']
        
        # Plot metrics with error bars
        metrics_to_plot = [
            ('mean_f1_over_time', 'std_f1_over_time', 'F1 Score'),
            ('mean_precision_over_time', 'std_precision_over_time', 'Precision'),
            ('mean_recall_over_time', 'std_recall_over_time', 'Recall')
        ]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(metrics_to_plot)))
        
        for i, (mean_key, std_key, label) in enumerate(metrics_to_plot):
            if mean_key in temporal_summary and std_key in temporal_summary:
                means = temporal_summary[mean_key]
                stds = temporal_summary[std_key]
                
                ax.errorbar(time_points, means, yerr=stds, 
                          label=label, marker='o', linewidth=2, 
                          color=colors[i], alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Performance Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to long_horizon directory
        filename = "temporal_trends.png"
        filepath = self.long_horizon_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved temporal trends plot to {filepath}")
        return str(filepath)
    
    def plot_region_comparison(self, 
                             evaluation_results: Dict[str, Dict[str, float]],
                             metric: str = 'f1',
                             sort_by_performance: bool = True,
                             title: str = "Region Performance Comparison") -> str:
        """
        Create horizontal bar plot comparing regions.
        
        Args:
            evaluation_results: Results from region evaluation
            metric: Metric to compare
            sort_by_performance: Whether to sort regions by performance
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        logger.info(f"Creating region comparison plot for metric: {metric}")
        
        # Prepare data
        regions = list(evaluation_results.keys())
        values = [evaluation_results[region][metric] for region in regions]
        
        if sort_by_performance:
            # Sort by performance
            sorted_pairs = sorted(zip(regions, values), key=lambda x: x[1], reverse=True)
            regions, values = zip(*sorted_pairs)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(regions) * 0.4)))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(regions))
        bars = ax.barh(y_pos, values, alpha=0.8, 
                      color=[self.region_colors[i % len(self.region_colors)] for i in range(len(regions))])
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=10)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions)
        ax.set_xlabel(f'{metric.capitalize()} Score')
        ax.set_title(title)
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to next_frame directory  
        filename = f"region_comparison_{metric}.png"
        filepath = self.next_frame_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved region comparison plot to {filepath}")
        return str(filepath)
    
    def create_summary_dashboard(self, 
                               next_frame_results: Dict[str, Dict[str, float]],
                               temporal_results: Dict[str, List[Dict[str, float]]],
                               temporal_summary: Dict[str, List[float]]) -> str:
        """
        Create a comprehensive dashboard with multiple subplots.
        
        Args:
            next_frame_results: Next-frame evaluation results
            temporal_results: Long-horizon evaluation results
            temporal_summary: Temporal summary statistics
            
        Returns:
            Path to saved dashboard
        """
        logger.info("Creating comprehensive performance dashboard")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Subplot 1: Next-frame performance bar plot
        ax1 = plt.subplot(2, 3, 1)
        regions = list(next_frame_results.keys())
        f1_scores = [next_frame_results[region]['f1'] for region in regions]
        bars = ax1.bar(range(len(regions)), f1_scores, alpha=0.8, color='skyblue')
        ax1.set_title('Next-Frame F1 Scores by Region')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('F1 Score')
        ax1.set_xticks(range(len(regions)))
        ax1.set_xticklabels(regions, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Temporal trends
        ax2 = plt.subplot(2, 3, 2)
        if temporal_summary and 'time_points' in temporal_summary:
            time_points = temporal_summary['time_points']
            if 'mean_f1_over_time' in temporal_summary:
                ax2.plot(time_points, temporal_summary['mean_f1_over_time'], 'o-', label='F1', linewidth=2)
            if 'mean_precision_over_time' in temporal_summary:
                ax2.plot(time_points, temporal_summary['mean_precision_over_time'], 's-', label='Precision', linewidth=2)
            if 'mean_recall_over_time' in temporal_summary:
                ax2.plot(time_points, temporal_summary['mean_recall_over_time'], '^-', label='Recall', linewidth=2)
            ax2.set_title('Performance Trends Over Time')
            ax2.set_xlabel('Time Window')
            ax2.set_ylabel('Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Region comparison (top 10)
        ax3 = plt.subplot(2, 3, 3)
        sorted_regions = sorted(regions, key=lambda r: next_frame_results[r]['f1'], reverse=True)[:10]
        sorted_f1s = [next_frame_results[region]['f1'] for region in sorted_regions]
        y_pos = np.arange(len(sorted_regions))
        ax3.barh(y_pos, sorted_f1s, alpha=0.8, color='lightcoral')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sorted_regions)
        ax3.set_xlabel('F1 Score')
        ax3.set_title('Top 10 Regions by F1 Score')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Subplot 4-6: Heatmaps for different metrics
        metrics_for_heatmap = ['f1', 'precision', 'recall']
        for i, metric in enumerate(metrics_for_heatmap):
            ax = plt.subplot(2, 3, 4 + i)
            
            if temporal_results:
                # Prepare heatmap data
                regions_subset = list(temporal_results.keys())[:15]  # Limit to first 15 regions
                max_windows = min(10, max(len(results) for results in temporal_results.values()))  # Limit to 10 windows
                
                data_matrix = np.full((len(regions_subset), max_windows), np.nan)
                
                for j, region in enumerate(regions_subset):
                    region_results = temporal_results[region]
                    for k, window_result in enumerate(region_results[:max_windows]):
                        if metric in window_result:
                            data_matrix[j, k] = window_result[metric]
                
                # Heatmap with scaling from 0 to max
                vmin = 0
                vmax = np.nanmax(data_matrix)
                im = ax.imshow(data_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_xticks(np.arange(max_windows))
                ax.set_yticks(np.arange(len(regions_subset)))
                ax.set_xticklabels([f'T{k+1}' for k in range(max_windows)])
                ax.set_yticklabels(regions_subset, fontsize=8)
                ax.set_title(f'{metric.capitalize()} Heatmap')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel(f'{metric.capitalize()}', rotation=-90, va="bottom")
        
        plt.tight_layout()
        
        # Save dashboard
        filename = "performance_dashboard.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance dashboard to {filepath}")
        return str(filepath)
    
    def save_results_table(self, 
                         evaluation_results: Dict[str, Dict[str, float]],
                         filename: str = "region_performance_table.csv") -> str:
        """
        Save results as a CSV table.
        
        Args:
            evaluation_results: Results from region evaluation
            filename: Output filename
            
        Returns:
            Path to saved CSV
        """
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(evaluation_results, orient='index')
        
        # Sort by F1 score
        df = df.sort_values('f1', ascending=False)
        
        # Save to CSV
        filepath = self.output_dir / filename
        df.to_csv(filepath, float_format='%.4f')
        
        logger.info(f"Saved results table to {filepath}")
        return str(filepath)
    
    def create_comparison_video(self, 
                              predictions: torch.Tensor,
                              ground_truth: torch.Tensor,
                              input_frames: Optional[torch.Tensor] = None,
                              filename: str = "prediction_comparison.mp4",
                              num_frames: int = 100,
                              fps: int = 2,
                              max_sequences: int = 5) -> str:
        """
        Create a comparison video of predicted vs true frames, similar to quarterly evaluations.
        
        Args:
            predictions: Predicted frames of shape (T, H, W) - binary spikes
            ground_truth: Ground truth frames of shape (T, H, W) - binary spikes  
            input_frames: Optional input frames for context (T, H, W)
            filename: Output video filename
            num_frames: Maximum number of frames to include per sequence
            fps: Frames per second for the video
            max_sequences: Maximum number of sequences to include
            
        Returns:
            Path to saved video
        """
        logger.info(f"Generating training-style comparison video")
        # Convert tensors to numpy arrays
        pred_np = predictions.cpu().numpy()
        gt_np = ground_truth.cpu().numpy()
        input_np = input_frames.cpu().numpy() if input_frames is not None else None
        sampled_np = (predictions > 0.5).cpu().numpy()
        
        # Remove last frame from predictions and sampled predictions only if they're the same length as input
        if input_np is not None and len(pred_np) == len(input_np):
            pred_np = pred_np[:-1]
            sampled_np = sampled_np[:-1]
        
        # Expand to 4D: (batch_size=1, seq_len, H, W)
        pred_np = np.expand_dims(pred_np, axis=0)
        sampled_np = np.expand_dims(sampled_np, axis=0)
        actual_np = np.expand_dims(input_np, axis=0) if input_np is not None else None
            # Ensure output directory exists
        video_path = self.output_dir / "videos" / filename
        video_path.parent.mkdir(parents=True, exist_ok=True)
        # Delegate to shared visualization utility
        #pdb.set_trace()
        create_color_coded_comparison_video(
            actual=actual_np,
            predicted=pred_np,
            sampled_predictions=sampled_np,
            output_path=str(video_path),
            num_frames=num_frames,
            fps=fps,
            threshold_left_panels=True
        )
        return str(video_path)
        # No fallback: exceptions propagate 