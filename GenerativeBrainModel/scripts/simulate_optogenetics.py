#!/usr/bin/env python3
"""
GBM Simulated Optogenetics Script

This script simulates optogenetic stimulation of specific brain regions using a trained GBM model,
then tracks the resulting activity across all brain regions over time.

The simulation:
1. Takes an initial seed sequence
2. Activates a target region in the final frame of the seed
3. Runs autoregressive generation to see how activity propagates
4. Produces heatmaps showing region activity over time
"""

import os
import sys
import json
import logging
import argparse
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import tifffile

import torch
from torch.utils.data import DataLoader

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import GBM model and dataset
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.fast_dali_spike_dataset import FastDALIBrainDataLoader
from GenerativeBrainModel.scripts.train_gbm_subject_split import SubjectFilteredFastDALIBrainDataLoader
from GenerativeBrainModel.scripts.evaluate_gbm import (
    setup_logging, load_model, load_test_data_h5, create_test_loader,
    load_brain_region_masks, get_subject_z_planes
)


# Argument parsing
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simulate optogenetic stimulation with GBM model')
    
    # Required arguments
    parser.add_argument('--exp-timestamp', type=str, required=True,
                        help='Experiment folder timestamp (e.g., 20230815_153045)')
    parser.add_argument('--target-subject', type=str, required=True,
                        help='Target subject name used for finetuning')
    parser.add_argument('--masks-dir', type=str, required=True,
                        help='Directory containing MapZBrain TIFF masks')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save simulation results')
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--test-h5', type=str,
                           help='Path to saved test_data_and_predictions.h5')
    data_group.add_argument('--reuse-loader', action='store_true',
                           help='Recreate DALI loader instead of using saved test data')
    
    # Optogenetic simulation settings
    parser.add_argument('--region-list', type=str, nargs='+', default=[],
                        help='List of regions to stimulate (TIFF filenames without .tif)')
    parser.add_argument('--seed-length', type=int, default=115,
                        help='Length of seed sequence (default: 115)')
    parser.add_argument('--horizon-length', type=int, default=330,
                        help='Maximum prediction horizon length (default: 330)')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Run only a baseline simulation without stimulation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='PyTorch device (default: cuda:0)')
    parser.add_argument('--save-raw', action='store_true',
                        help='Save raw activity data as CSV and pickle files')
    
    return parser.parse_args()


def normalize_activity(activity: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Normalize activity data for better visualization.
    
    Args:
        activity: Dictionary of activity data
            {run_name: {region_name: activity_timeseries}}
    
    Returns:
        normalized: Dictionary of normalized activity
    """
    normalized = {}
    
    for run_name, region_dict in activity.items():
        normalized[run_name] = {}
        for region_name, timeseries in region_dict.items():
            # Avoid division by zero by adding a small epsilon
            max_val = np.max(timeseries)
            if max_val > 0:
                normalized[run_name][region_name] = timeseries / max_val
            else:
                normalized[run_name][region_name] = timeseries
                
    return normalized


def extract_region_activity(
    predictions: Dict[str, torch.Tensor],
    region_masks: Dict[str, np.ndarray],
    z_starts: torch.Tensor,
    subj_z_planes: int,
    num_volumes: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract region-level activity time series from predictions, aggregated by volume.
    
    Args:
        predictions: Dictionary of predicted sequences for each run
            {run_name: sequence_tensor (1, T, H, W)}
        region_masks: Dictionary of 3D region masks (Z, H, W)
        z_starts: Starting z-plane indices for sequences
        subj_z_planes: Number of z-planes for the subject
        num_volumes: Total number of volumes (each volume spans all z-planes)
    
    Returns:
        activity: Dictionary of activity time series
            {run_name: {region_name: activity_timeseries}}
    """
    activity = {name: {} for name in predictions}
    
    for name, seq in predictions.items():
        # Convert to numpy for processing
        arr = seq.detach().cpu().numpy()[0]  # (T, H, W)
        logging.info(f"Processing activity for run: {name}, sequence shape: {arr.shape}")
        
        # For each region, apply mask and count active voxels at each timestep
        for region_name, mask3d in tqdm(region_masks.items(), desc=f"Extracting {name} region activity"):
            # Initialize array for volume-wise counts
            vol_counts = np.zeros(num_volumes, dtype=float)
            
            # Aggregate frames into volumes
            for t in range(arr.shape[0]):
                # Calculate which volume this frame belongs to
                vol_idx = t // subj_z_planes
                
                # Ensure we don't go beyond our allocated volumes
                if vol_idx >= num_volumes:
                    continue
                
                # Calculate z-plane index for this timestep
                z_idx = (z_starts[0].item() + t) % subj_z_planes
                
                # Apply mask and count active voxels, add to volume count
                vol_counts[vol_idx] += float(np.sum(arr[t] * mask3d[z_idx]))
            
            # Store as numpy array
            activity[name][region_name] = vol_counts
    
    return activity


def create_heatmaps(
    activity: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    seed_volumes: int,
    subj_z_planes: int,
    normalized: bool = True,
    include_regions: Optional[List[str]] = None
):
    """
    Create heatmaps of region activity over time (in volumes).
    
    Args:
        activity: Dictionary of activity data
            {run_name: {region_name: activity_timeseries}}
        output_dir: Directory to save heatmaps
        seed_volumes: Number of volumes in the seed sequence
        subj_z_planes: Number of z-planes per volume
        normalized: Whether to normalize activity (default: True)
        include_regions: Optional list of regions to include in heatmaps
    """
    # Create output directory for heatmaps with subfolders
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Create subfolders for different types of heatmaps
    normalized_dir = os.path.join(heatmap_dir, 'normalized')
    raw_dir = os.path.join(heatmap_dir, 'raw')
    difference_dir = os.path.join(heatmap_dir, 'difference')
    
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(difference_dir, exist_ok=True)
    
    # Store baseline data for difference maps
    baseline_data = None
    if 'baseline' in activity:
        baseline_data = activity['baseline']
    
    # Create a heatmap for each run
    for name, region_dict in activity.items():
        logging.info(f"Creating heatmap for run: {name}")
        
        # Filter regions if specified
        if include_regions:
            region_dict = {r: data for r, data in region_dict.items() if r in include_regions}
        
        # Sort regions by total activity to make patterns more visible
        regions = sorted(region_dict.keys(), 
                       key=lambda r: np.sum(region_dict[r]), 
                       reverse=True)
        
        # Stack data into a 2D array (num_regions × num_volumes)
        data = np.stack([region_dict[r] for r in regions], axis=0)
        
        # Determine x-axis labels for volumes
        num_volumes = data.shape[1]
        if num_volumes > 20:
            # If many volumes, label fewer ticks
            x_step = max(1, num_volumes // 10)
            xticks = np.arange(0, num_volumes, x_step)
        else:
            # If few volumes, label all
            xticks = np.arange(0, num_volumes)
        
        # Calculate the boundary position (one position before seed_volumes)
        boundary_pos = max(0, seed_volumes - 1)
        
        # Create normalized heatmap
        plot_data = data
        if normalized:
            # Normalize each row (region) to [0, 1]
            row_max = data.max(axis=1, keepdims=True)
            # Avoid division by zero
            row_max[row_max == 0] = 1.0
            plot_data = data / row_max
            cbar_label = 'Normalized activity'
            
            plt.figure(figsize=(14, max(6, 0.25 * len(regions))))
            g = sns.heatmap(
                plot_data, 
                cmap='magma', 
                yticklabels=regions, 
                xticklabels=xticks,
                cbar_kws={'label': cbar_label}
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=0)
            
            # Add vertical line at boundary_pos (right before stimulation)
            plt.axvline(x=boundary_pos, color='cyan', linestyle='--', linewidth=2)
            plt.text(boundary_pos + 0.2, len(regions) - 1, 'Stimulation/Prediction Start', 
                    color='cyan', fontsize=10, rotation=90, va='top')
            
            # Add shaded area for ground truth (init) data
            plt.axvspan(0, boundary_pos, alpha=0.1, color='green')
            plt.text(boundary_pos/2, len(regions) - 1, 'Ground Truth Data', 
                    color='green', fontsize=10, ha='center', va='top')
            
            plt.title(f'Brain Region Activity After {"" if name != "baseline" else "No"} Stimulation: {name}')
            plt.xlabel(f'Time (volumes) - each volume = {subj_z_planes} z-planes')
            plt.ylabel('Brain region')
            plt.tight_layout()
            
            # Save figure in normalized subfolder
            plt.savefig(os.path.join(normalized_dir, f'heatmap_{name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create raw count heatmap
        plt.figure(figsize=(14, max(6, 0.25 * len(regions))))
        g = sns.heatmap(
            data, 
            cmap='magma', 
            yticklabels=regions, 
            xticklabels=xticks,
            cbar_kws={'label': 'Active voxel count'}
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        
        # Add vertical line at boundary_pos (right before stimulation)
        plt.axvline(x=boundary_pos, color='cyan', linestyle='--', linewidth=2)
        plt.text(boundary_pos + 0.2, len(regions) - 1, 'Stimulation/Prediction Start', 
                color='cyan', fontsize=10, rotation=90, va='top')
        
        # Add shaded area for ground truth (init) data
        plt.axvspan(0, boundary_pos, alpha=0.1, color='green')
        plt.text(boundary_pos/2, len(regions) - 1, 'Ground Truth Data', 
                color='green', fontsize=10, ha='center', va='top')
        
        plt.title(f'Brain Region Activity (Raw Count) After {"" if name != "baseline" else "No"} Stimulation: {name}')
        plt.xlabel(f'Time (volumes) - each volume = {subj_z_planes} z-planes')
        plt.ylabel('Brain region')
        plt.tight_layout()
        
        # Save figure in raw subfolder
        plt.savefig(os.path.join(raw_dir, f'heatmap_{name}_raw.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create difference heatmaps (only if baseline exists and there are stimulated regions)
    if baseline_data is not None and len(activity) > 1:
        logging.info("Creating difference heatmaps (stimulated - baseline)")
        
        # For each stimulated region, create a difference map
        for name, region_dict in activity.items():
            # Skip baseline
            if name == 'baseline':
                continue
                
            # Calculate differences for each region
            diff_dict = {}
            for region_name in region_dict:
                if region_name in baseline_data:
                    # Ensure arrays are the same length
                    min_len = min(len(region_dict[region_name]), len(baseline_data[region_name]))
                    diff_dict[region_name] = region_dict[region_name][:min_len] - baseline_data[region_name][:min_len]
            
            # Only proceed if we have differences
            if not diff_dict:
                continue
                
            # Sort regions by absolute magnitude of difference
            regions = sorted(diff_dict.keys(), 
                           key=lambda r: np.abs(diff_dict[r]).sum(), 
                           reverse=True)
            
            # Stack data into a 2D array
            diff_data = np.stack([diff_dict[r] for r in regions], axis=0)
            
            # Create diverging colormap for differences
            # Define the colormap
            cmap = plt.cm.RdBu_r
            
            # Find the absolute maximum for symmetric color scaling
            vmax = np.abs(diff_data).max()
            vmin = -vmax
            
            # Create normalized difference heatmap
            plt.figure(figsize=(14, max(6, 0.25 * len(regions))))
            g = sns.heatmap(
                diff_data, 
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                center=0,
                yticklabels=regions, 
                xticklabels=xticks,
                cbar_kws={'label': 'Activity difference (stimulated - baseline)'}
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=0)
            
            # Add vertical line at boundary_pos (right before stimulation)
            plt.axvline(x=boundary_pos, color='cyan', linestyle='--', linewidth=2)
            plt.text(boundary_pos + 0.2, len(regions) - 1, 'Stimulation/Prediction Start', 
                    color='cyan', fontsize=10, rotation=90, va='top')
            
            # Add shaded area for ground truth (init) data
            plt.axvspan(0, boundary_pos, alpha=0.1, color='green')
            plt.text(boundary_pos/2, len(regions) - 1, 'Ground Truth Data', 
                    color='green', fontsize=10, ha='center', va='top')
            
            plt.title(f'Difference in Brain Activity: {name} - Baseline')
            plt.xlabel(f'Time (volumes) - each volume = {subj_z_planes} z-planes')
            plt.ylabel('Brain region')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(difference_dir, f'diff_{name}_vs_baseline.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
            # Also save a filtered version with only the top differences
            # Show only regions with significant differences
            abs_diffs = np.abs(diff_data).sum(axis=1)
            significant_threshold = np.percentile(abs_diffs, 75)  # Show top 25% most different regions
            sig_indices = np.where(abs_diffs > significant_threshold)[0]
            
            if len(sig_indices) > 0:
                sig_diff_data = diff_data[sig_indices]
                sig_regions = [regions[i] for i in sig_indices]
                
                plt.figure(figsize=(14, max(6, 0.25 * len(sig_regions))))
                g = sns.heatmap(
                    sig_diff_data, 
                    cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax,
                    center=0,
                    yticklabels=sig_regions, 
                    xticklabels=xticks,
                    cbar_kws={'label': 'Activity difference (stimulated - baseline)'}
                )
                g.set_xticklabels(g.get_xticklabels(), rotation=0)
                
                # Add vertical line at boundary_pos (right before stimulation)
                plt.axvline(x=boundary_pos, color='cyan', linestyle='--', linewidth=2)
                plt.text(boundary_pos + 0.2, len(sig_regions) - 1, 'Stimulation/Prediction Start', 
                        color='cyan', fontsize=10, rotation=90, va='top')
                
                # Add shaded area for ground truth (init) data
                plt.axvspan(0, boundary_pos, alpha=0.1, color='green')
                plt.text(boundary_pos/2, len(sig_regions) - 1, 'Ground Truth Data', 
                        color='green', fontsize=10, ha='center', va='top')
                
                plt.title(f'Significant Differences: {name} - Baseline (Top 25% Most Different Regions)')
                plt.xlabel(f'Time (volumes) - each volume = {subj_z_planes} z-planes')
                plt.ylabel('Brain region')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(difference_dir, f'sig_diff_{name}_vs_baseline.png'), dpi=300, bbox_inches='tight')
                plt.close()


def save_activity_data(activity: Dict[str, Dict[str, np.ndarray]], output_dir: str, subj_z_planes: int):
    """
    Save raw activity data to CSV and pickle files.
    
    Args:
        activity: Dictionary of activity data
            {run_name: {region_name: activity_timeseries}}
        output_dir: Directory to save data
        subj_z_planes: Number of z-planes per volume
    """
    # Create output directory for data
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save pickle file with all data
    with open(os.path.join(data_dir, 'activity_data.pkl'), 'wb') as f:
        pickle.dump(activity, f)
        
    # Also save metadata about the volumes
    with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'z_planes_per_volume': subj_z_planes,
            'explanation': 'Each unit on the time axis represents one complete volume (z-stack)'
        }, f, indent=2)
    
    # Save CSV file for each run
    for name, region_dict in activity.items():
        # Convert to DataFrame
        df = pd.DataFrame(region_dict)
        
        # Save to CSV
        csv_path = os.path.join(data_dir, f'activity_{name}.csv')
        df.to_csv(csv_path, index=True)
        logging.info(f"Saved activity data to {csv_path}")


def main():
    """Main function to run the optogenetic simulation."""
    global args
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, 'simulation.log')
    logger = setup_logging(log_file)
    
    # Log script configuration
    logging.info("Starting GBM optogenetic simulation")
    logging.info(f"Arguments: {vars(args)}")
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Using CPU instead.")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    logging.info(f"Using device: {device}")
    
    try:
        # Load model and parameters
        model, params = load_model(args.exp_timestamp, device, checkpoint_file=None)
        
        # Load test data
        z_starts = None
        if args.test_h5:
            test_data, pred_probs, pred_samples, z_starts = load_test_data_h5(args.test_h5)
            
            if z_starts is not None:
                if device.type == 'cuda':
                    z_starts = z_starts.to(device)
            else:
                logging.warning("No z-plane start indices found. Using default z_start=0.")
                z_starts = torch.zeros(test_data.shape[0], dtype=torch.long, device=device)
        else:
            # Use DALI loader
            test_data, z_starts = create_test_loader(
                params['preaugmented_dir'], 
                args.target_subject, 
                params
            )
            
            if z_starts is None:
                logging.warning("No z-plane start indices available. Using default z_start=0.")
                z_starts = torch.zeros(test_data.shape[0], dtype=torch.long, device=device)
        
        # Ensure test data is on the correct device
        if test_data.device != device:
            test_data = test_data.to(device)
        
        # Get subject's number of z-planes
        subj_z_planes = get_subject_z_planes(
            params['preaugmented_dir'], 
            args.target_subject
        )
        logging.info(f"Subject {args.target_subject} has {subj_z_planes} z-planes")
        
        # Calculate number of volumes based on z-planes
        num_frames = args.seed_length + (args.horizon_length - args.seed_length)
        num_volumes = int(np.ceil(num_frames / subj_z_planes))
        seed_volumes = int(np.ceil(args.seed_length / subj_z_planes))
        logging.info(f"Total frames: {num_frames}, Z-planes per volume: {subj_z_planes}")
        logging.info(f"Total volumes: {num_volumes}, Seed volumes: {seed_volumes}")
        
        # Load brain region masks
        region_masks = load_brain_region_masks(
            args.masks_dir,
            subj_z_planes,
            target_shape=(256, 128)
        )
        
        # Check that requested regions exist in the masks
        missing_regions = [r for r in args.region_list if r not in region_masks]
        if missing_regions:
            logging.warning(f"The following requested regions are not found in masks: {missing_regions}")
            if not args.baseline_only and not [r for r in args.region_list if r in region_masks]:
                logging.error("No valid regions to stimulate. Run with --baseline-only or provide valid regions.")
                return 1
        
        # Select a single sample to use as seed
        # Use the first sample in the batch
        seed = test_data[:1, :args.seed_length].clone()  # shape (1, seed_length, H, W)
        logging.info(f"Using seed sequence shape: {seed.shape}")
        
        # Prepare runs with stimulated regions
        runs = {}
        
        # Always include a baseline run with no stimulation
        runs['baseline'] = seed.clone()
        
        # Add stimulation runs if not baseline-only
        if not args.baseline_only:
            for region_name in args.region_list:
                if region_name not in region_masks:
                    continue
                
                logging.info(f"Preparing stimulation for region: {region_name}")
                mask3d = region_masks[region_name]  # shape (Z, 256, 128)
                
                # Pick the z-plane for the last seed frame
                z0 = (z_starts[0].item() + args.seed_length - 1) % subj_z_planes
                stim_mask2d = torch.tensor(mask3d[z0], device=device, dtype=seed.dtype)
                
                # Create a copy of the seed with the region stimulated in the last frame
                seed_opto = seed.clone()
                
                # Set all voxels in the target region to 1 (fully active)
                # Only modify where the mask is True
                last_frame = seed_opto[0, args.seed_length - 1]
                last_frame[stim_mask2d > 0] = 1.0
                
                runs[region_name] = seed_opto
        
        # Run autoregressive prediction for each run
        predictions = {}
        for name, seed_run in tqdm(runs.items(), desc="Running simulations"):
            logging.info(f"Generating predictions for {name}")
            try:
                # Generate autoregressive predictions
                with torch.no_grad():
                    horizon_steps = args.horizon_length - args.seed_length
                    generated = model.generate_autoregressive_brain(
                        seed_run, num_steps=horizon_steps
                    )
                    
                    # Ensure we have binary predictions
                    if generated.dtype != torch.bool and generated.dtype != torch.uint8:
                        # If probabilities, convert to binary
                        if torch.max(generated) <= 1.0 and torch.min(generated) >= 0.0:
                            generated = (generated > 0.5).float()
                
                # Store full sequence (seed + generated)
                predictions[name] = generated
                logging.info(f"Generated sequence shape for {name}: {generated.shape}")
                
            except Exception as e:
                logging.error(f"Error generating predictions for {name}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Extract region-level activity for each run, now aggregated by volume
        activity = extract_region_activity(
            predictions, region_masks, z_starts, subj_z_planes, num_volumes
        )
        
        # Create heatmaps using volumes as time units
        create_heatmaps(activity, output_dir, seed_volumes, subj_z_planes)
        
        # Optionally save raw activity data
        if args.save_raw:
            save_activity_data(activity, output_dir, subj_z_planes)
        
        # Print final output path
        print(f"\nSimulation complete. Results saved to: {output_dir}")
        logging.info(f"Simulation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during simulation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 