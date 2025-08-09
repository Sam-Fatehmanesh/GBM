import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

# Set multiprocessing tensor sharing strategy to file_system to reduce shared memory usage
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass


class NeuralDataset(Dataset):
    """
    PyTorch Dataset for neuron-level spike probability time series.

    - Reads HDF5 files produced by unified_spike_processing.py
    - Returns time sequences of spike probabilities, padded to the group's max neurons
    - Includes per-sample neuron positions (padded) and neuron validity mask
    - Includes per-sample stimulus codes aligned with the time sequence
    """

    def __init__(
        self,
        data_files: List[str],
        pad_neurons_to: int,
        sequence_length: int = 1,
        stride: int = 1,
        max_timepoints_per_subject: Optional[int] = None,
        use_cache: bool = False,
        start_timepoint: Optional[int] = None,
        end_timepoint: Optional[int] = None,
    ) -> None:
        self.data_files = data_files
        self.pad_neurons_to = pad_neurons_to
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_timepoints_per_subject = max_timepoints_per_subject
        self.use_cache = use_cache
        self.start_timepoint = start_timepoint
        self.end_timepoint = end_timepoint

        self.sequences: List[Dict[str, int]] = []
        self.cached_spikes: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_positions: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_stimulus: Optional[Dict[str, np.ndarray]] = {} if use_cache else None

        # Each worker process has its own set of open file handles
        self.worker_file_handles: Dict[str, h5py.File] = {}

        self._build_sequence_index()

    def _build_sequence_index(self) -> None:
        """Build sequence index and optionally cache required data."""
        print(f"Building neural sequence index... Caching: {self.use_cache}")
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                spikes_ds = f['spike_probabilities']  # (T, N)
                T_total, N = spikes_ds.shape

                # Determine start/end bounds
                start_point = self.start_timepoint if self.start_timepoint is not None else 0
                end_point = self.end_timepoint if self.end_timepoint is not None else T_total
                end_point = min(end_point, T_total)

                if self.max_timepoints_per_subject is not None:
                    end_point = min(end_point, start_point + self.max_timepoints_per_subject)

                num_timepoints = end_point - start_point
                if num_timepoints < self.sequence_length:
                    print(f"Skipping {Path(file_path).name}: not enough timepoints ({num_timepoints}) for sequence_length={self.sequence_length}")
                    continue

                # Cache required slices
                if self.use_cache:
                    if file_path not in self.cached_spikes:
                        self.cached_spikes[file_path] = spikes_ds[start_point:end_point].astype(np.float32)
                    if file_path not in self.cached_positions:
                        pos = f['cell_positions'][:]  # (N, 3)
                        self.cached_positions[file_path] = pos.astype(np.float32)
                    # Stimulus may or may not exist
                    stim = f['stimulus_full'][:] if 'stimulus_full' in f else np.zeros((T_total,), dtype=np.uint8)
                    stim = stim[start_point:end_point]
                    self.cached_stimulus[file_path] = stim.astype(np.int64)

                # Create sequence indices
                max_start_idx = end_point - self.sequence_length
                for start_idx in range(start_point, max_start_idx + 1, self.stride):
                    self.sequences.append({'file_path': file_path, 'start_idx': start_idx})

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_file_handle(self, file_path: str) -> h5py.File:
        if file_path not in self.worker_file_handles:
            self.worker_file_handles[file_path] = h5py.File(file_path, 'r')
        return self.worker_file_handles[file_path]

    def _pad_neurons(self, data_2d: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
        """
        Pad along neuron dimension to self.pad_neurons_to.
        Expected input shapes:
          - (seq_len, N) for spikes (time-major)
          - (N, 3) for positions
        """
        if data_2d.ndim == 2:
            if data_2d.shape[-1] == self.pad_neurons_to:
                return data_2d
            if data_2d.shape[0] == self.pad_neurons_to:
                return data_2d  # already padded
        N = data_2d.shape[-1] if data_2d.shape[0] != self.pad_neurons_to else data_2d.shape[0]

        if data_2d.shape == (N, 3):
            # positions (N, 3) -> pad to (pad_N, 3)
            pad_N = self.pad_neurons_to
            out = np.zeros((pad_N, 3), dtype=data_2d.dtype)
            out[:N, :] = data_2d
            return out
        else:
            # spikes (seq_len, N) -> pad to (seq_len, pad_N)
            seq_len = data_2d.shape[0]
            pad_N = self.pad_neurons_to
            out = np.zeros((seq_len, pad_N), dtype=data_2d.dtype)
            out[:, :N] = data_2d
            return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_info = self.sequences[idx]
        file_path = seq_info['file_path']
        start_idx = seq_info['start_idx']
        end_idx = start_idx + self.sequence_length

        if self.use_cache:
            spikes_TN = self.cached_spikes[file_path]  # (T_slice, N)
            segment = spikes_TN[start_idx - (self.start_timepoint or 0): end_idx - (self.start_timepoint or 0)]  # (L, N)
            positions = self.cached_positions[file_path]  # (N, 3)
            stimulus = self.cached_stimulus[file_path][start_idx - (self.start_timepoint or 0): end_idx - (self.start_timepoint or 0)]
        else:
            f = self._get_file_handle(file_path)
            spikes_ds = f['spike_probabilities']  # (T, N)
            positions = f['cell_positions'][:]    # (N, 3)
            segment = spikes_ds[start_idx:end_idx].astype(np.float32)
            if 'stimulus_full' in f:
                stimulus = f['stimulus_full'][start_idx:end_idx].astype(np.int64)
            else:
                stimulus = np.zeros((self.sequence_length,), dtype=np.int64)

        # Determine valid neuron count before padding
        N_valid = positions.shape[0]

        # Pad spikes and positions to group max
        segment_padded = self._pad_neurons(segment, pad_value=0.0)  # (L, pad_N)
        positions_padded = self._pad_neurons(positions, pad_value=0.0)  # (pad_N, 3)

        # Neuron validity mask (1 for real neurons, 0 for padded)
        neuron_mask = np.zeros((self.pad_neurons_to,), dtype=np.float32)
        neuron_mask[:N_valid] = 1.0

        # Convert to tensors
        spikes_tensor = torch.from_numpy(segment_padded.astype(np.float32))  # (L, pad_N)
        positions_tensor = torch.from_numpy(positions_padded.astype(np.float32))  # (pad_N, 3)
        mask_tensor = torch.from_numpy(neuron_mask)
        stimulus_tensor = torch.from_numpy(stimulus.astype(np.int64))  # (L,)

        return {
            'spikes': spikes_tensor,            # (sequence_length, pad_N)
            'positions': positions_tensor,      # (pad_N, 3)
            'neuron_mask': mask_tensor,         # (pad_N,)
            'stimulus': stimulus_tensor,        # (sequence_length,)
            'file_path': file_path,
            'start_idx': start_idx,
        }

    def __del__(self) -> None:
        for handle in self.worker_file_handles.values():
            try:
                handle.close()
            except Exception:
                pass


def _group_files_by_rate(files: List[str]) -> Dict[float, List[str]]:
    """Group subject files by their final_sampling_rate attribute."""
    groups: Dict[float, List[str]] = {}
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                rate = float(f.attrs.get('final_sampling_rate', f.attrs.get('effective_sampling_rate', 0.0)))
        except Exception:
            rate = 0.0
        groups.setdefault(rate, []).append(fp)
    return groups


def _max_neurons_in_files(files: List[str]) -> int:
    max_n = 0
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'num_neurons' in f:
                    n = int(f['num_neurons'][()])
                else:
                    n = f['cell_positions'].shape[0]
                max_n = max(max_n, n)
        except Exception:
            continue
    return max_n


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/test DataLoaders for neural spike probability data.

    Behavior:
    - Groups files by final_sampling_rate and selects the group to use
      (by default the largest group; can be forced via config['data']['sampling_rate']).
    - Pads neuron dimension to the max neurons within the selected group.

    Returns:
      (train_loader, test_loader)
    """
    print("Using NeuralDataLoader.")

    data_config = config['data']
    training_config = config['training']

    data_dir = Path(data_config['data_dir'])
    all_files = sorted([str(f) for f in data_dir.glob("*.h5")])

    if not all_files:
        raise ValueError(f"No H5 files found in {data_dir}")

    # Group by sampling rate
    groups = _group_files_by_rate(all_files)
    print(f"Found {len(groups)} sampling-rate group(s): {[(rate, len(files)) for rate, files in groups.items()]}")

    # Optionally pick a specific rate
    desired_rate = data_config.get('sampling_rate', None)
    if desired_rate is not None:
        # find closest matching rate
        selected_rate = min(groups.keys(), key=lambda r: abs(r - float(desired_rate)))
    else:
        # choose the largest group by file count
        selected_rate = max(groups.keys(), key=lambda r: len(groups[r]))

    selected_files = groups[selected_rate]
    print(f"Selected sampling rate: {selected_rate} Hz with {len(selected_files)} files")

    # Train/test split
    test_subjects = data_config.get('test_subjects', [])
    test_files = [str(data_dir / f"{s}.h5") for s in test_subjects if (data_dir / f"{s}.h5").exists() and str(data_dir / f"{s}.h5") in selected_files]
    train_files = [f for f in selected_files if f not in test_files]

    if not test_files and test_subjects:
        print(f"Warning: No valid files found for test subjects in selected rate group: {test_subjects}")

    if not test_files:
        print("No valid test subjects provided. Creating a random 80/20 train/test split within selected rate group.")
        num_test = max(1, len(selected_files) // 5)
        random.seed(training_config.get('seed', 42))
        test_files = random.sample(selected_files, num_test)
        train_files = [f for f in selected_files if f not in test_files]

    # Compute padding size (max neurons) within selected files
    pad_neurons_to = _max_neurons_in_files(selected_files)
    print(f"Padding neuron dimension to: {pad_neurons_to}")

    # Parameters
    sequence_length = training_config.get('sequence_length', 1)
    stride = training_config.get('stride', 1)
    max_timepoints_per_subject = training_config.get('max_timepoints_per_subject', None)
    use_cache = data_config.get('use_cache', True)
    start_timepoint = training_config.get('start_timepoint', None)
    end_timepoint = training_config.get('end_timepoint', None)

    train_dataset = NeuralDataset(
        train_files,
        pad_neurons_to=pad_neurons_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
    )

    test_dataset = NeuralDataset(
        test_files,
        pad_neurons_to=pad_neurons_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
    )

    # Safer defaults for memory usage
    dl_kwargs = {
        'batch_size': training_config.get('batch_size', 4),
        'num_workers': training_config.get('num_workers', 2),
        'pin_memory': training_config.get('pin_memory', False),
        'persistent_workers': training_config.get('persistent_workers', False),
        'prefetch_factor': training_config.get('prefetch_factor', 2),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)

    return train_loader, test_loader
