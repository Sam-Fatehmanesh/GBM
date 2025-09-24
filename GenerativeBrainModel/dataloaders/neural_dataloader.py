import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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
        pad_stimuli_to: int,
        sequence_length: int = 1,
        stride: int = 1,
        max_timepoints_per_subject: Optional[int] = None,
        use_cache: bool = False,
        start_timepoint: Optional[int] = None,
        end_timepoint: Optional[int] = None,
        spikes_dataset_name: str = 'neuron_values',
        role: str = 'all',  # 'all' | 'train' | 'val' | 'test'
    ) -> None:
        self.data_files = data_files
        self.pad_stimuli_to = pad_stimuli_to
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_timepoints_per_subject = max_timepoints_per_subject
        self.use_cache = use_cache
        self.start_timepoint = start_timepoint
        self.end_timepoint = end_timepoint
        self.spikes_dataset_name = spikes_dataset_name
        self.role = role

        self.sequences: List[Dict[str, int]] = []
        self.cached_spikes: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_positions: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_stimulus: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        # Offset of cached slices within original file (for fast index math)
        self.cache_offset: Optional[Dict[str, int]] = {} if use_cache else None

        # Each worker process has its own set of open file handles
        self.worker_file_handles: Dict[str, h5py.File] = {}

        self._build_sequence_index()

    def _build_sequence_index(self) -> None:
        """Build sequence index and optionally cache required data."""
        print(f"Building neural sequence index... Caching: {self.use_cache}")
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                ds_name = self._resolve_spikes_dataset_name(f)
                spikes_ds = f[ds_name]  # (T, N)
                T_total, N = spikes_ds.shape

                # Read optional split attributes from file (0-based, [start,end) semantics)
                def _get_attr(name: str, default: Optional[int]) -> Optional[int]:
                    try:
                        return int(f.attrs[name]) if name in f.attrs else default
                    except Exception:
                        return default
                tr0 = _get_attr('split_train_start', None)
                tr1 = _get_attr('split_train_end', None)
                va0 = _get_attr('split_val_start', None)
                va1 = _get_attr('split_val_end', None)
                te0 = _get_attr('split_test_start', None)
                te1 = _get_attr('split_test_end', None)
                condition_name = None
                try:
                    if 'split_condition_name' in f.attrs:
                        v = f.attrs['split_condition_name']
                        condition_name = v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v)
                except Exception:
                    condition_name = None

                # Determine start/end bounds (global override optional)
                start_point = self.start_timepoint if self.start_timepoint is not None else 0
                end_point = self.end_timepoint if self.end_timepoint is not None else T_total
                end_point = min(end_point, T_total)

                if self.max_timepoints_per_subject is not None:
                    end_point = min(end_point, start_point + self.max_timepoints_per_subject)

                num_timepoints = end_point - start_point
                if num_timepoints < self.sequence_length:
                    print(f"Skipping {Path(file_path).name}: not enough timepoints ({num_timepoints}) for sequence_length={self.sequence_length}")
                    continue

                # --- Trim leading and trailing all-zero frames (rows) ---
                def _first_nonzero_row(ds, a, b, chunk=256) -> int:
                    i = a
                    eps = 0.0
                    while i < b:
                        j = min(i + chunk, b)
                        block = ds[i:j, :]
                        # any value non-zero in row
                        nz = np.any(np.abs(block) > eps, axis=1)
                        if nz.any():
                            return i + int(np.argmax(nz))
                        i = j
                    return b
                def _last_nonzero_row(ds, a, b, chunk=256) -> int:
                    i = b
                    eps = 0.0
                    while i > a:
                        j = max(a, i - chunk)
                        block = ds[j:i, :]
                        nz = np.any(np.abs(block) > eps, axis=1)
                        if nz.any():
                            # last index in [j,i) having nz
                            return j + (len(nz) - 1 - int(np.argmax(nz[::-1])))
                        i = j
                    return a - 1

                nz_start = _first_nonzero_row(spikes_ds, start_point, end_point)
                nz_last = _last_nonzero_row(spikes_ds, start_point, end_point)
                if nz_start >= end_point or nz_last < nz_start:
                    print(f"Skipping {Path(file_path).name}: all-zero in selected window [{start_point},{end_point})")
                    continue
                # convert last to exclusive end
                trimmed_start = nz_start
                trimmed_end = nz_last + 1
                if trimmed_start > start_point or trimmed_end < end_point:
                    lead = trimmed_start - start_point
                    tail = end_point - trimmed_end
                    print(f"Trimmed {Path(file_path).name}: lead_zero={lead}, tail_zero={tail}, kept={trimmed_end-trimmed_start}")
                else:
                    print(f"No margin zeros in {Path(file_path).name}")

                # Cache required slices
                if self.use_cache:
                    if file_path not in self.cached_spikes:
                        self.cached_spikes[file_path] = spikes_ds[trimmed_start:trimmed_end].astype(np.float32)
                        if self.cache_offset is not None:
                            self.cache_offset[file_path] = int(trimmed_start)
                    if file_path not in self.cached_positions:
                        pos = f['cell_positions'][:]  # (N, 3)
                        self.cached_positions[file_path] = pos.astype(np.float32)
                    # Stimulus: prefer 'stimuli_features' else 'stimulus_full'
                    if 'stimuli_features' in f:
                        stim = f['stimuli_features'][trimmed_start:trimmed_end]
                        self.cached_stimulus[file_path] = stim.astype(np.float32)
                    elif 'stimulus_full' in f:
                        stim = f['stimulus_full'][trimmed_start:trimmed_end]
                        self.cached_stimulus[file_path] = stim.astype(np.float32)
                    else:
                        self.cached_stimulus[file_path] = np.zeros((trimmed_end - trimmed_start, 1), dtype=np.float32)

                # Create sequence indices within selected role range and trimmed bounds
                role_ranges: List[Tuple[int, int, str]] = []
                if self.role in ('all', 'train') and tr0 is not None and tr1 is not None:
                    role_ranges.append((tr0, tr1, 'train'))
                if self.role in ('all', 'val') and va0 is not None and va1 is not None:
                    role_ranges.append((va0, va1, 'val'))
                if self.role in ('all', 'test') and te0 is not None and te1 is not None:
                    role_ranges.append((te0, te1, 'test'))
                # Fallback if no attrs present
                if not role_ranges:
                    role_ranges.append((0, T_total, 'all'))
                for a, b, role_name in role_ranges:
                    a2 = max(a, trimmed_start)
                    b2 = min(b, trimmed_end)
                    max_start_idx = b2 - self.sequence_length
                    for start_idx in range(a2, max_start_idx + 1, self.stride):
                        self.sequences.append({
                            'file_path': file_path,
                            'start_idx': start_idx,
                            'role': role_name,
                            'condition_name': condition_name or 'unknown'
                        })

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_file_handle(self, file_path: str) -> h5py.File:
        if file_path not in self.worker_file_handles:
            self.worker_file_handles[file_path] = h5py.File(file_path, 'r')
        return self.worker_file_handles[file_path]

    # Removed neuron pre-padding: padding is now handled in collate_fn per-batch

    def _pad_stimulus(self, stim_2d: np.ndarray) -> np.ndarray:
        """Pad stimulus along feature dimension to self.pad_stimuli_to.
        Expects (seq_len, K) and returns (seq_len, pad_K).
        """
        if stim_2d.ndim != 2:
            return stim_2d
        L, K = stim_2d.shape
        if K == self.pad_stimuli_to:
            return stim_2d
        out = np.zeros((L, self.pad_stimuli_to), dtype=stim_2d.dtype)
        out[:, :min(K, self.pad_stimuli_to)] = stim_2d[:, :min(K, self.pad_stimuli_to)]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_info = self.sequences[idx]
        file_path = seq_info['file_path']
        start_idx = seq_info['start_idx']
        end_idx = start_idx + self.sequence_length

        if self.use_cache:
            spikes_TN = self.cached_spikes[file_path]  # (T_slice, N)
            offset = self.cache_offset[file_path] if self.cache_offset is not None else (self.start_timepoint or 0)
            rel_start = start_idx - offset
            rel_end = end_idx - offset
            segment = spikes_TN[rel_start:rel_end]  # (L, N)
            positions = self.cached_positions[file_path]  # (N, 3)
            stimulus = self.cached_stimulus[file_path][rel_start:rel_end]  # (L, K_file)
        else:
            f = self._get_file_handle(file_path)
            ds_name = self._resolve_spikes_dataset_name(f)
            spikes_ds = f[ds_name]  # (T, N)
            positions = f['cell_positions'][:]    # (N, 3)
            segment = spikes_ds[start_idx:end_idx].astype(np.float32)
            if 'stimuli_features' in f:
                stimulus = f['stimuli_features'][start_idx:end_idx].astype(np.float32)
            elif 'stimulus_full' in f:
                stimulus = f['stimulus_full'][start_idx:end_idx].astype(np.float32)
            else:
                stimulus = np.zeros((self.sequence_length, 1), dtype=np.float32)

        # Pad stimulus to the global width across files
        stimulus = self._pad_stimulus(stimulus)

        # Determine valid neuron count before padding
        N_valid = positions.shape[0]

        # Do not pre-pad neurons; collate_fn will pad per-batch
        segment_padded = segment  # (L, N)
        positions_padded = positions  # (N, 3)

        # Neuron validity mask (1 for real neurons, 0 for padded)
        neuron_mask = np.ones((N_valid,), dtype=np.float32)

        # Convert to tensors
        spikes_tensor = torch.from_numpy(segment_padded.astype(np.float32))  # (L, pad_N)
        positions_tensor = torch.from_numpy(positions_padded.astype(np.float32))  # (pad_N, 3)
        mask_tensor = torch.from_numpy(neuron_mask)
        stimulus_tensor = torch.from_numpy(stimulus.astype(np.float32))  # (L, K_file)

        return {
            'spikes': spikes_tensor,            # (sequence_length, N)
            'positions': positions_tensor,      # (N, 3)
            'neuron_mask': mask_tensor,         # (N,)
            'stimulus': stimulus_tensor,        # (sequence_length, K_file)
            'file_path': file_path,
            'start_idx': start_idx,
            'split_role': seq_info.get('role', 'all'),
            'condition_name': seq_info.get('condition_name', 'unknown'),
        }

    def _resolve_spikes_dataset_name(self, f: h5py.File) -> str:
        """Resolve dataset name with backward compatibility.
        Priority: explicit self.spikes_dataset_name -> 'neuron_values' ->
        'spike_probabilities' -> 'spike_rates_hz' -> 'processed_calcium' -> 'zcalcium'.
        """
        # If explicitly provided name exists, honor it
        try:
            if self.spikes_dataset_name in f:
                return self.spikes_dataset_name
        except Exception:
            pass
        # Preferred new name
        if 'neuron_values' in f:
            return 'neuron_values'
        # Backward-compat names
        if 'spike_probabilities' in f:
            return 'spike_probabilities'
        if 'spike_rates_hz' in f:
            return 'spike_rates_hz'
        if 'processed_calcium' in f:
            return 'processed_calcium'
        if 'zcalcium' in f:
            return 'zcalcium'
        raise ValueError("No compatible spikes dataset found (tried neuron_values, spike_probabilities, spike_rates_hz, processed_calcium, zcalcium)")

    def __del__(self) -> None:
        for handle in self.worker_file_handles.values():
            try:
                handle.close()
            except Exception:
                pass


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


def _max_stimuli_in_files(files: List[str]) -> int:
    """Return max stimulus width (K) across files; if missing, assume 1."""
    max_k = 1
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'stimuli_features' in f:
                    ds = f['stimuli_features']
                    if ds.ndim == 2:
                        max_k = max(max_k, int(ds.shape[1]))
                elif 'stimulus_full' in f:
                    ds = f['stimulus_full']
                    if ds.ndim == 2:
                        max_k = max(max_k, int(ds.shape[1]))
                    elif ds.ndim == 1:
                        max_k = max(max_k, 1)
        except Exception:
            continue
    return max_k


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler], Optional[DistributedSampler]]:
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

    # Determine selected files: if only_test flag is set with explicit subjects, restrict to them
    only_test = bool(training_config.get('only_test', False))
    selected_files = all_files
    print(f"Found total files: {len(all_files)}")

    # Train/test split
    test_subjects = data_config.get('test_subjects', [])
    test_files = [str(data_dir / f"{s}.h5") for s in test_subjects if (data_dir / f"{s}.h5").exists() and str(data_dir / f"{s}.h5") in all_files]
    if only_test and test_files:
        selected_files = test_files
    train_files = [f for f in selected_files if f not in test_files]

    if not test_files and test_subjects:
        print(f"Warning: No valid files found for requested test subjects: {test_subjects}")

    # If files contain internal split attributes, use them for all roles and avoid random 80/20
    def _has_internal_splits(file_list: List[str]) -> bool:
        for fp in file_list:
            try:
                with h5py.File(fp, 'r') as f:
                    if all(k in f.attrs for k in (
                        'split_train_start','split_train_end','split_val_start','split_val_end','split_test_start','split_test_end'
                    )):
                        return True
            except Exception:
                continue
        return False

    if _has_internal_splits(selected_files):
        print("Detected per-file train/val/test split attributes; using internal ranges for all roles.")
        train_files = selected_files
        test_files = selected_files
    elif not test_files and not only_test:
        print("No valid test subjects provided. Creating a random 80/20 train/test split.")
        num_test = max(1, len(selected_files) // 5)
        random.seed(training_config.get('seed', 42))
        test_files = random.sample(selected_files, num_test)
        train_files = [f for f in selected_files if f not in test_files]
    elif only_test and test_files:
        # Mirror test set for train as well (eval-only usage); avoids touching unrelated files
        train_files = test_files

    # Compute padding sizes
    # - Neuron padding is handled per-batch, but we keep this for logging/debug
    pad_neurons_to = _max_neurons_in_files(selected_files)
    # - Stimulus padding should consider ALL subjects in the folder so selected subsets still align
    pad_stimuli_to = _max_stimuli_in_files(all_files)
    print(f"Padding neuron dimension to: {pad_neurons_to}")
    print(f"Padding stimulus dimension to: {pad_stimuli_to}")

    # Parameters
    sequence_length = training_config.get('sequence_length', 1)
    stride = training_config.get('stride', 1)
    max_timepoints_per_subject = training_config.get('max_timepoints_per_subject', None)
    use_cache = data_config.get('use_cache', True)
    start_timepoint = training_config.get('start_timepoint', None)
    end_timepoint = training_config.get('end_timepoint', None)

    # Determine which spikes dataset to read (defaults to neuron_values, can override)
    spikes_dataset_name = data_config.get('spikes_dataset_name', 'neuron_values')

    train_dataset = NeuralDataset(
        train_files,
        pad_stimuli_to=pad_stimuli_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        spikes_dataset_name=spikes_dataset_name,
        role='train',
    )

    val_dataset = NeuralDataset(
        test_files if training_config.get('val_on_test_plus_val', True) else train_files,
        pad_stimuli_to=pad_stimuli_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        spikes_dataset_name=spikes_dataset_name,
        role='val',
    )

    test_dataset = NeuralDataset(
        test_files,
        pad_stimuli_to=pad_stimuli_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        spikes_dataset_name=spikes_dataset_name,
        role='test',
    )

    # Safer defaults for memory usage
    num_workers = int(training_config.get('num_workers', 0))
    dl_kwargs = {
        'batch_size': training_config.get('batch_size', 4),
        'num_workers': num_workers,
        'pin_memory': training_config.get('pin_memory', False),
        'persistent_workers': training_config.get('persistent_workers', False) if num_workers > 0 else False,
        'prefetch_factor': training_config.get('prefetch_factor', 2) if num_workers > 0 else None,
        'pin_memory_device': 'cuda' if training_config.get('pin_memory', False) else '',
    }
    if num_workers == 0 and 'prefetch_factor' in dl_kwargs:
        del dl_kwargs['prefetch_factor']

    # Distributed samplers (if dist is initialized)
    train_sampler = None
    val_sampler = None
    test_sampler = None
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            train_sampler = DistributedSampler(NeuralDataset(
                train_files,
                pad_stimuli_to=pad_stimuli_to,
                sequence_length=sequence_length,
                stride=stride,
                max_timepoints_per_subject=max_timepoints_per_subject,
                use_cache=use_cache,
                start_timepoint=start_timepoint,
                end_timepoint=end_timepoint,
                role='train',
            ), shuffle=True)
            val_sampler = DistributedSampler(NeuralDataset(
                test_files if training_config.get('val_on_test_plus_val', True) else train_files,
                pad_stimuli_to=pad_stimuli_to,
                sequence_length=sequence_length,
                stride=stride,
                max_timepoints_per_subject=max_timepoints_per_subject,
                use_cache=use_cache,
                start_timepoint=start_timepoint,
                end_timepoint=end_timepoint,
                role='val',
            ), shuffle=False)
            test_sampler = DistributedSampler(NeuralDataset(
                test_files,
                pad_stimuli_to=pad_stimuli_to,
                sequence_length=sequence_length,
                stride=stride,
                max_timepoints_per_subject=max_timepoints_per_subject,
                use_cache=use_cache,
                start_timepoint=start_timepoint,
                end_timepoint=end_timepoint,
                role='test',
            ), shuffle=False)
    except Exception:
        pass

    def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Determine max neurons in this batch
        max_n = max(item['positions'].shape[0] for item in batch)
        # Determine stimulus width (already padded by dataset for K); unify L as given
        L = batch[0]['spikes'].shape[0]
        K = batch[0]['stimulus'].shape[1]

        def pad_spikes(x: torch.Tensor, target_n: int) -> torch.Tensor:
            Lx, Nx = x.shape
            if Nx == target_n:
                return x
            out = torch.zeros((Lx, target_n), dtype=x.dtype)
            out[:, :Nx] = x
            return out

        def pad_positions(p: torch.Tensor, target_n: int) -> torch.Tensor:
            Nx, D = p.shape
            if Nx == target_n:
                return p
            out = torch.zeros((target_n, D), dtype=p.dtype)
            out[:Nx, :] = p
            return out

        def pad_mask(m: torch.Tensor, target_n: int) -> torch.Tensor:
            Nx = m.shape[0]
            if Nx == target_n:
                return m
            out = torch.zeros((target_n,), dtype=m.dtype)
            out[:Nx] = m
            return out

        spikes = torch.stack([pad_spikes(it['spikes'], max_n) for it in batch], dim=0)
        positions = torch.stack([pad_positions(it['positions'], max_n) for it in batch], dim=0)
        masks = torch.stack([pad_mask(it['neuron_mask'], max_n) for it in batch], dim=0)
        stimulus = torch.stack([it['stimulus'] for it in batch], dim=0)  # (B, L, K) already padded across files

        return {
            'spikes': spikes,           # (B, L, max_n)
            'positions': positions,     # (B, max_n, 3)
            'neuron_mask': masks,       # (B, max_n)
            'stimulus': stimulus,       # (B, L, K)
            'file_path': [it['file_path'] for it in batch],
            'start_idx': torch.tensor([it['start_idx'] for it in batch], dtype=torch.long),
            'split_role': [it['split_role'] for it in batch],
            'condition_name': [it['condition_name'] for it in batch],
        }

    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_pad, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler, collate_fn=collate_pad, **dl_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, sampler=test_sampler, collate_fn=collate_pad, **dl_kwargs)

    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler
