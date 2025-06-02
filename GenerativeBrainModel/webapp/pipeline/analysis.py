import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from GenerativeBrainModel.utils.masks import load_zebrafish_masks


def analyze_region_differences(
    baseline_dir: str,
    predictions_dir: str,
    regions: list = None,
    output_dir: str = None
) -> str:
    """
    Compute region-wise activation ratio: predicted_sum / baseline_count for each region.
    Loads activation_mask and predicted_probabilities, aggregates by region, and saves:
      - summary.json
      - summary.csv
      - heatmap.png (all regions)
      - heatmap_top12.png (top 12 regions by change magnitude)

    Returns the output directory path.
    """
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.join(predictions_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Load activation mask (3D boolean array: Z, Y, X)
    activation_path = os.path.join(baseline_dir, 'activation_mask.npy')
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"Activation mask not found: {activation_path}")
    activation_mask = np.load(activation_path).astype(bool)

    # Load sequence start offset to align slices to global z planes
    seq_start_path = os.path.join(baseline_dir, 'sequence_z_start.npy')
    if not os.path.exists(seq_start_path):
        raise FileNotFoundError(f"Sequence z-start file not found: {seq_start_path}")
    z_start = int(np.load(seq_start_path))

    # Load predicted probabilities (num_steps, Y, X)
    probs_path = os.path.join(predictions_dir, 'predicted_probabilities.npy')
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Predicted probabilities not found: {probs_path}")
    probabilities = np.load(probs_path).astype(float)

    # Load mask loader and region masks using activation mask dimensions
    # activation_mask shape: (Z, Y, X)
    Z_mask, Y_mask, X_mask = activation_mask.shape
    mask_loader = load_zebrafish_masks(target_shape=(Z_mask, Y_mask, X_mask))
    all_regions = mask_loader.list_masks()
    # Select regions to analyze
    if regions is None:
        regions_to_use = all_regions
    else:
        regions_to_use = [r for r in regions if r in all_regions]
        missing = set(regions) - set(regions_to_use)
        if missing:
            print(f"Warning: some regions not found and will be skipped: {missing}")

    # Determine number of predicted volumes
    Z = activation_mask.shape[0]
    num_steps = probabilities.shape[0]
    volumes_count = num_steps // Z
    if num_steps % Z != 0:
        volumes_count += 1

    # Compute per-volume metrics for each region
    summary = {}
    ratios_matrix = []
    region_magnitudes = []  # For finding top 12 regions
    
    for region in regions_to_use:
        region_mask3d = mask_loader.get_mask(region).cpu().numpy().astype(bool)
        baseline_count = int(np.sum(activation_mask & region_mask3d))
        region_ratios = []
        region_metrics = []
        
        for vol_idx in range(volumes_count):
            start = vol_idx * Z
            end = min(start + Z, num_steps)
            pred_sum = 0.0
            for t in range(start, end):
                z_local = t % Z
                z_global = z_start + z_local
                mask2d = activation_mask[z_global] & region_mask3d[z_global]
                if mask2d.any():
                    pred_sum += float(np.sum(probabilities[t][mask2d]))
            if baseline_count > 0:
                ratio = float((pred_sum - baseline_count) / baseline_count)
            else:
                ratio = 0.0
            region_ratios.append(ratio)
            region_metrics.append({
                'volume_idx': vol_idx,
                'baseline_count': baseline_count,
                'predicted_sum': pred_sum,
                'ratio': ratio
            })
        
        summary[region] = region_metrics
        ratios_matrix.append(region_ratios)
        
        # Calculate magnitude of change for this region (max absolute change across volumes)
        magnitude = max(abs(r) for r in region_ratios) if region_ratios else 0.0
        region_magnitudes.append((region, magnitude))

    # Save summary JSON (per-volume metrics)
    json_path = os.path.join(output_dir, 'summary.json')
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)

    # Save summary CSV (per-volume rows)
    csv_path = os.path.join(output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['region', 'volume_idx', 'baseline_count', 'predicted_sum', 'ratio'])
        for region in regions_to_use:
            for m in summary[region]:
                writer.writerow([region, m['volume_idx'], m['baseline_count'], m['predicted_sum'], m['ratio']])

    # Generate full heatmap of per-volume ratios (all regions x volumes)
    data_matrix = np.array(ratios_matrix)
    n_regions, n_volumes = data_matrix.shape
    
    # Full heatmap
    fig, ax = plt.subplots(figsize=(max(12, n_volumes * 0.5), max(8, n_regions * 0.3)))
    im = ax.imshow(data_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=np.nanmax(data_matrix))
    # Set ticks for volumes
    ax.set_xticks(np.arange(n_volumes))
    ax.set_xticklabels([f'V{i+1}' for i in range(n_volumes)], rotation=90)
    # Set ticks for regions
    ax.set_yticks(np.arange(n_regions))
    ax.set_yticklabels(regions_to_use)
    ax.set_title('Predicted Activation Ratio per Region and Volume (All Regions)')
    # Add minor gridlines between cells
    ax.set_xticks(np.arange(n_volumes + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_regions + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Predicted Sum / Baseline Count')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close(fig)

    # Generate top 12 regions heatmap
    # Sort regions by magnitude and take top 12
    region_magnitudes.sort(key=lambda x: x[1], reverse=True)
    top_12_regions = [region for region, magnitude in region_magnitudes[:12]]
    
    if len(top_12_regions) > 0:
        # Get indices of top 12 regions in the original list
        top_12_indices = [regions_to_use.index(region) for region in top_12_regions if region in regions_to_use]
        
        # Extract data for top 12 regions
        top_12_data = data_matrix[top_12_indices, :]
        
        # Generate top 12 heatmap
        fig, ax = plt.subplots(figsize=(max(12, n_volumes * 0.5), max(6, len(top_12_regions) * 0.4)))
        im = ax.imshow(top_12_data, aspect='auto', cmap='viridis', vmin=0, vmax=np.nanmax(data_matrix))
        # Set ticks for volumes
        ax.set_xticks(np.arange(n_volumes))
        ax.set_xticklabels([f'V{i+1}' for i in range(n_volumes)], rotation=90)
        # Set ticks for top 12 regions
        ax.set_yticks(np.arange(len(top_12_regions)))
        ax.set_yticklabels(top_12_regions)
        ax.set_title('Predicted Activation Ratio per Region and Volume (Top 12 Regions by Change Magnitude)')
        # Add minor gridlines between cells
        ax.set_xticks(np.arange(n_volumes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(top_12_regions) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Predicted Sum / Baseline Count')
        plt.tight_layout()
        heatmap_top12_path = os.path.join(output_dir, 'heatmap_top12.png')
        plt.savefig(heatmap_top12_path)
        plt.close(fig)

    return output_dir 