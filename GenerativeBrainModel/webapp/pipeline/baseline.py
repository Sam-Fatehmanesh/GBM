import os
import uuid
import h5py
import numpy as np
from GenerativeBrainModel.utils.masks import load_zebrafish_masks


def modified_baseline_sequence(
    experiment_path: str,
    regions: list,
    fraction: float,
    sample_idx: int = 0,
    output_dir: str = None
) -> str:
    """
    Load the first baseline sequence from test_data_and_predictions.h5, apply optogenetic activation
    to the last full brain volume for the specified regions and fraction, and save results.

    Returns the output directory path containing 'baseline_sequence.npy' and 'activation_mask.npy'.
    """
    # Locate the HDF5 file under experiment_path
    # Common paths: pretrain/test_data, finetune/test_data
    h5_path = None
    for phase in ['pretrain', 'finetune']:
        candidate = os.path.join(experiment_path, phase, 'test_data', 'test_data_and_predictions.h5')
        if os.path.exists(candidate):
            h5_path = candidate
            break
    # Fallback: search recursively
    if h5_path is None:
        import glob

        matches = glob.glob(os.path.join(experiment_path, '**', 'test_data_and_predictions.h5'), recursive=True)
        if matches:
            h5_path = matches[0]
    if h5_path is None:
        raise FileNotFoundError(f"HDF5 file 'test_data_and_predictions.h5' not found under: {experiment_path}")

    # Load baseline sequences
    with h5py.File(h5_path, 'r') as f:
        # test_data shape: (num_samples, seq_len, H, W)
        data = f['test_data'][:]  # load entire dataset into memory
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range")

    seq = data[sample_idx].astype(np.uint8).copy()  # shape (seq_len, H, W)
    seq_len, H, W = seq.shape

    # Load and combine masks
    mask_loader = load_zebrafish_masks()
    Z, Ym, Xm = mask_loader.target_shape
    if (Ym, Xm) != (H, W):
        raise ValueError(f"Unexpected frame shape {H,W}, expected {(Ym,Xm)}")

    # Sample activations per region mask
    activation = np.zeros((Z, Ym, Xm), dtype=bool)
    for region in regions:
        # Load region mask
        region_mask = mask_loader.get_mask(region).cpu().numpy().astype(bool)
        # Find all voxels in this region
        region_indices = np.argwhere(region_mask)
        # Determine number to activate for this region
        num_to_activate = int(len(region_indices) * fraction)
        # Randomly select region-specific voxels
        if num_to_activate > 0 and len(region_indices) > 0:
            chosen = np.random.choice(len(region_indices), size=num_to_activate, replace=False)
            for idx in chosen:
                z, y, x = region_indices[idx]
                activation[z, y, x] = True

    # Apply activation to the last volume in the sequence
    for frame_idx in range(seq_len - Z, seq_len):
        z = frame_idx - (seq_len - Z)
        mask2d = activation[z]
        # Set activated voxels to 1
        seq[frame_idx, mask2d] = 1

    # Prepare output directory
    if output_dir is None:
        job_id = uuid.uuid4().hex
        output_dir = os.path.join(experiment_path, 'webapp_job_' + job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Save baseline sequence and activation mask
    np.save(os.path.join(output_dir, 'baseline_sequence.npy'), seq)
    np.save(os.path.join(output_dir, 'activation_mask.npy'), activation)

    return output_dir 