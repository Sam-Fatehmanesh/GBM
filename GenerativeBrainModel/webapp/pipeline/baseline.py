import os
import uuid
import h5py
import numpy as np
from pathlib import Path
from GenerativeBrainModel.utils.masks import load_zebrafish_masks


def modified_baseline_sequence(
    experiment_path: str,
    regions: list,
    fraction: float,
    sample_idx: int = 0,
    output_dir: str = None,
    mask_loader=None
) -> str:
    """
    Load the first baseline sequence from test_data_and_predictions.h5, apply optogenetic activation
    to the last full brain volume for the specified regions and fraction, and save results.

    Args:
        mask_loader: Optional pre-loaded mask loader to avoid reloading masks

    Returns the output directory path containing 'baseline_sequence.npy' and 'activation_mask.npy'.
    """
    # Locate the HDF5 file under experiment_path (prefer finetune over pretrain)
    # Common paths: finetune/test_data, pretrain/test_data
    h5_path = None
    for phase in ['finetune', 'pretrain']:
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

    # Load baseline sequences and metadata
    with h5py.File(h5_path, 'r') as f:
        # test_data shape: (num_samples, seq_len, H, W)
        data = f['test_data'][:]  # load entire dataset into memory
        # Load sequence_z_starts if available (starting z-plane index per sequence)
        if 'sequence_z_starts' in f:
            z_starts_arr = f['sequence_z_starts'][:]
        else:
            z_starts_arr = np.zeros(data.shape[0], dtype=int)
        # Parse experiment_info.txt to find preaugmented_dir and target_subject
        exp_info = Path(experiment_path) / 'experiment_info.txt'
        preaug_dir = None
        subject = None
        if exp_info.exists():
            with open(exp_info) as ef:
                for line in ef:
                    lt = line.strip()
                    # Parse preaugmented directory
                    if lt.lower().startswith('preaugmented_dir:'):
                        preaug_dir = lt.split(':',1)[1].strip()
                    # Parse target subject (handles both 'target_subject:' and 'target subject:')
                    elif lt.lower().startswith('target_subject:') or lt.lower().startswith('target subject:'):
                        subject = lt.split(':',1)[1].strip()
        if not preaug_dir:
            raise RuntimeError(f"preaugmented_dir not specified in {exp_info}")
        if not subject or subject.lower() == 'none':
            raise RuntimeError(f"Target subject not specified in {exp_info}")
        # Locate metadata.h5 for this subject
        metadata_h5 = Path(preaug_dir) / subject / 'metadata.h5'
        if not metadata_h5.exists():
            raise FileNotFoundError(f"Subject metadata not found: {metadata_h5}")
        # Read Z (num_z_planes) from metadata
        with h5py.File(metadata_h5, 'r') as mf:
            Z = int(mf['num_z_planes'][()])
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range")

    # Determine the selected z_start for this sequence
    selected_z_start = int(z_starts_arr[sample_idx])

    seq = data[sample_idx].astype(np.uint8).copy()  # shape (seq_len, H, W)
    seq_len, H, W = seq.shape

    # Use provided mask loader or create new one if not provided
    if mask_loader is None:
        # Fallback: instantiate mask loader with dynamic Z and sequence spatial dims
        mask_loader = load_zebrafish_masks(target_shape=(Z, H, W))

    # Sample activations per region mask
    activation = np.zeros((Z, H, W), dtype=bool)
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

    # Inject activation into the last complete volume
    for frame_idx in range(seq_len - Z, seq_len):
        local_z = frame_idx - (seq_len - Z)
        global_z = selected_z_start + local_z
        # Only inject if global_z is within volume bounds
        if 0 <= global_z < Z:
            mask2d = activation[global_z]
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
    # Save the sequence z-start index for inference UI
    np.save(os.path.join(output_dir, 'sequence_z_start.npy'), selected_z_start)

    return output_dir 