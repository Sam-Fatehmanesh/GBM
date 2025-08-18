import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

# Align sharing strategy with neural_dataloader to reduce shared memory pressure
try:
    mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass


def _resample_1d_series(series: np.ndarray, new_len: int) -> np.ndarray:
    """Linearly resample a 1D array to new_len using numpy only.
    Preserves endpoints and handles degenerate lengths.
    """
    old_len = int(series.shape[0])
    if old_len == new_len:
        return series.astype(np.float32, copy=False)
    if old_len <= 1:
        return np.full((new_len,), float(series[0] if old_len == 1 else 0.0), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=old_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float64)
    out = np.interp(x_new, x_old, series.astype(np.float64))
    return out.astype(np.float32)


class BehaviorDataset(Dataset):
    """
    Dataset for predicting behavior from per-timepoint neural states and positions.

    - Hardcoded sequence_length = 1, stride = 1 (per-timepoint samples)
    - Trims leading/trailing all-zero neural rows
    - Upsamples behavior_full to the neural T and aligns within the trimmed window
    - Pads behavior feature dimension to `pad_behavior_to` (truncate or zero-pad)
    - Neuron padding is handled in `collate_fn` per-batch
    """

    def __init__(
        self,
        data_files: List[str],
        pad_behavior_to: int,
        use_cache: bool = False,
        max_timepoints_per_subject: Optional[int] = None,
        start_timepoint: Optional[int] = None,
        end_timepoint: Optional[int] = None,
    ) -> None:
        self.data_files = data_files
        self.pad_behavior_to = int(pad_behavior_to)
        self.use_cache = use_cache
        self.max_timepoints_per_subject = max_timepoints_per_subject
        self.start_timepoint = start_timepoint
        self.end_timepoint = end_timepoint

        # Sequence settings: window length (L) and stride (default 6 and 1)
        from typing import Any
        self.sequence_length: int = 6
        self.stride: int = 1
        self.sequences: List[Dict[str, int]] = []
        self.cached_spikes: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_positions: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cached_behavior: Optional[Dict[str, np.ndarray]] = {} if use_cache else None
        self.cache_offset: Optional[Dict[str, int]] = {} if use_cache else None

        self.worker_file_handles: Dict[str, h5py.File] = {}

        self._build_index()

    def _build_index(self) -> None:
        print(f"Building behavior dataset index... Caching: {self.use_cache}")
        for file_path in self.data_files:
            with h5py.File(file_path, 'r') as f:
                if 'spike_probabilities' not in f:
                    continue
                spikes_ds = f['spike_probabilities']  # (T, N)
                T_total, N = spikes_ds.shape

                # Determine bounds
                start_point = self.start_timepoint if self.start_timepoint is not None else 0
                end_point = self.end_timepoint if self.end_timepoint is not None else T_total
                end_point = min(end_point, T_total)

                if self.max_timepoints_per_subject is not None:
                    end_point = min(end_point, start_point + self.max_timepoints_per_subject)

                # Trim leading/trailing all-zero neural rows
                def _first_nonzero_row(ds, a, b, chunk=256) -> int:
                    i = a
                    eps = 0.0
                    while i < b:
                        j = min(i + chunk, b)
                        block = ds[i:j, :]
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
                            return j + (len(nz) - 1 - int(np.argmax(nz[::-1])))
                        i = j
                    return a - 1

                nz_start = _first_nonzero_row(spikes_ds, start_point, end_point)
                nz_last = _last_nonzero_row(spikes_ds, start_point, end_point)
                if nz_start >= end_point or nz_last < nz_start:
                    print(f"Skipping {Path(file_path).name}: all-zero in selected window [{start_point},{end_point})")
                    continue
                trimmed_start = nz_start
                trimmed_end = nz_last + 1

                if 'behavior_full' not in f:
                    print(f"Skipping {Path(file_path).name}: missing behavior_full")
                    continue

                # Prepare cached slices if requested
                if self.use_cache:
                    if file_path not in self.cached_spikes:
                        self.cached_spikes[file_path] = spikes_ds[trimmed_start:trimmed_end].astype(np.float32)
                        if self.cache_offset is not None:
                            self.cache_offset[file_path] = int(trimmed_start)
                    if file_path not in self.cached_positions:
                        pos = f['cell_positions'][:]
                        self.cached_positions[file_path] = pos.astype(np.float32)
                    # Load behavior and normalize to (B, Tb)
                    beh = f['behavior_full'][()]
                    if beh.ndim == 1:
                        beh_bt = beh.astype(np.float32)[None, :]
                    elif beh.ndim == 2:
                        if beh.shape[0] <= beh.shape[1]:
                            beh_bt = beh.astype(np.float32)
                        else:
                            beh_bt = beh.astype(np.float32).T
                    else:
                        print(f"Skipping {Path(file_path).name}: unexpected behavior_full ndim={beh.ndim}")
                        continue
                    # Sanitize raw behavior
                    try:
                        beh_bt = np.nan_to_num(beh_bt, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                    B, Tb = int(beh_bt.shape[0]), int(beh_bt.shape[1])
                    T_slice = int(trimmed_end - trimmed_start)
                    beh_resampled = np.zeros((B, T_total), dtype=np.float32)
                    for b in range(B):
                        beh_resampled[b] = _resample_1d_series(beh_bt[b], T_total)
                    # Sanitize after resampling
                    try:
                        beh_resampled = np.nan_to_num(beh_resampled, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                    beh_slice = beh_resampled[:, trimmed_start:trimmed_end]  # (B, T_slice)
                    # Pad/truncate behavior feature dimension to pad_behavior_to
                    if B != self.pad_behavior_to:
                        if B < self.pad_behavior_to:
                            pad = np.zeros((self.pad_behavior_to - B, T_slice), dtype=np.float32)
                            beh_slice = np.concatenate([beh_slice, pad], axis=0)
                        else:
                            beh_slice = beh_slice[:self.pad_behavior_to, :]
                    # Store as (T_slice, K)
                    self.cached_behavior[file_path] = beh_slice.T.astype(np.float32)

                # Create index entries for sliding window of length L with stride
                L = int(self.sequence_length)
                for s in range(trimmed_start, trimmed_end - L + 1, max(1, self.stride)):
                    curr = s + L - 1
                    self.sequences.append({'file_path': file_path, 'window_start': s, 'start_idx': curr})

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_file_handle(self, file_path: str) -> h5py.File:
        if file_path not in self.worker_file_handles:
            self.worker_file_handles[file_path] = h5py.File(file_path, 'r')
        return self.worker_file_handles[file_path]

    def _pad_behavior_dim(self, beh_1d: np.ndarray) -> np.ndarray:
        K = beh_1d.shape[0]
        if K == self.pad_behavior_to:
            return beh_1d
        out = np.zeros((self.pad_behavior_to,), dtype=beh_1d.dtype)
        out[:min(K, self.pad_behavior_to)] = beh_1d[:min(K, self.pad_behavior_to)]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        file_path = seq['file_path']
        # Current time index and window start
        t = seq['start_idx']
        s = seq.get('window_start', t)

        if self.use_cache:
            spikes_TN = self.cached_spikes[file_path]  # (T_slice, N)
            offset = self.cache_offset[file_path]
            rel_s = s - offset
            rel_t = t - offset
            L = self.sequence_length
            spikes_1N = spikes_TN[rel_s:rel_s + L]  # (L, N)
            positions = self.cached_positions[file_path]  # (N, 3)
            beh_TK = self.cached_behavior[file_path]      # (T_slice, K)
            beh_1K = beh_TK[rel_t]
        else:
            f = self._get_file_handle(file_path)
            spikes_ds = f['spike_probabilities']
            L = self.sequence_length
            spikes_1N = spikes_ds[s:s + L].astype(np.float32)
            try:
                spikes_1N = np.nan_to_num(spikes_1N, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            positions = f['cell_positions'][:].astype(np.float32)
            try:
                positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            beh = f['behavior_full'][()]
            # Normalize to (B, Tb)
            if beh.ndim == 1:
                beh_bt = beh.astype(np.float32)[None, :]
            elif beh.ndim == 2:
                if beh.shape[0] <= beh.shape[1]:
                    beh_bt = beh.astype(np.float32)
                else:
                    beh_bt = beh.astype(np.float32).T
            else:
                raise RuntimeError(f"Unexpected behavior_full ndim={beh.ndim} in {file_path}")
            # Sanitize raw behavior
            try:
                beh_bt = np.nan_to_num(beh_bt, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            B, Tb = int(beh_bt.shape[0]), int(beh_bt.shape[1])
            T_total = int(spikes_ds.shape[0])
            beh_resampled = np.zeros((B, T_total), dtype=np.float32)
            for b in range(B):
                beh_resampled[b] = _resample_1d_series(beh_bt[b], T_total)
            try:
                beh_resampled = np.nan_to_num(beh_resampled, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
            beh_1K = beh_resampled[:, t]
            beh_1K = self._pad_behavior_dim(beh_1K)

        # Build mask for valid neurons before padding
        N_valid = positions.shape[0]
        mask = np.ones((N_valid,), dtype=np.float32)

        spikes_tensor = torch.from_numpy(spikes_1N.astype(np.float32))      # (L, N)
        positions_tensor = torch.from_numpy(positions.astype(np.float32))    # (N, 3)
        mask_tensor = torch.from_numpy(mask)
        behavior_tensor = torch.from_numpy(beh_1K.astype(np.float32))        # (K,)

        return {
            'spikes': spikes_tensor,            # (1, N)
            'positions': positions_tensor,      # (N, 3)
            'neuron_mask': mask_tensor,         # (N,)
            'behavior': behavior_tensor,        # (K,)
            'file_path': file_path,
            'start_idx': t,
        }

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


def _max_behavior_dim(files: List[str]) -> int:
    max_k = 0
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'behavior_full' not in f:
                    continue
                beh = f['behavior_full']
                if beh.ndim == 1:
                    max_k = max(max_k, 1)
                elif beh.ndim == 2:
                    # Behavior treated as channels first (B, Tb) or (Tb, B)
                    max_k = max(max_k, int(min(beh.shape[0], beh.shape[1])))
        except Exception:
            continue
    if max_k == 0:
        # Fallback to 1 if no behavior available (should be filtered earlier)
        max_k = 1
    return max_k


def create_behavior_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """Create train/val DataLoaders for behavior prediction.

    Returns (train_loader, val_loader, train_sampler, val_sampler)
    """
    print("Using BehaviorDataLoader.")

    data_config = config['data']
    training_config = config['training']

    data_dir = Path(data_config['data_dir'])
    all_files = sorted([str(f) for f in data_dir.glob("*.h5")])
    if not all_files:
        raise ValueError(f"No H5 files found in {data_dir}")

    # Keep only files that contain behavior_full (avoid leaking file handles)
    filtered: List[str] = []
    for fp in all_files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'behavior_full' in f:
                    filtered.append(fp)
        except Exception:
            continue
    all_files = filtered
    if not all_files:
        raise ValueError(f"No H5 files with behavior_full found in {data_dir}")

    # Train/test split
    test_subjects = data_config.get('test_subjects', [])
    # Only keep requested subjects that also have behavior_full (i.e., are present in all_files)
    requested = [str(data_dir / f"{s}.h5") for s in test_subjects]
    test_files = [fp for fp in requested if fp in all_files]
    train_files = [f for f in all_files if f not in test_files]
    if not test_files:
        print("No valid test subjects provided. Creating a random 80/20 train/test split.")
        num_test = max(1, len(all_files) // 5)
        random.seed(training_config.get('seed', 42))
        test_files = random.sample(all_files, num_test)
        train_files = [f for f in all_files if f not in test_files]

    pad_behavior_to = _max_behavior_dim(all_files)
    pad_neurons_to = _max_neurons_in_files(all_files)
    print(f"Padding behavior dimension to: {pad_behavior_to}")
    print(f"(For logging) Max neuron count across files: {pad_neurons_to}")

    use_cache = data_config.get('use_cache', True)
    max_timepoints_per_subject = training_config.get('max_timepoints_per_subject', None)
    start_timepoint = training_config.get('start_timepoint', None)
    end_timepoint = training_config.get('end_timepoint', None)

    train_dataset = BehaviorDataset(
        train_files,
        pad_behavior_to=pad_behavior_to,
        use_cache=use_cache,
        max_timepoints_per_subject=max_timepoints_per_subject,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
    )

    val_dataset = BehaviorDataset(
        test_files,
        pad_behavior_to=pad_behavior_to,
        use_cache=use_cache,
        max_timepoints_per_subject=max_timepoints_per_subject,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
    )

    num_workers = int(training_config.get('num_workers', 0))
    dl_kwargs = {
        'batch_size': training_config.get('batch_size', 32),
        'num_workers': num_workers,
        'pin_memory': training_config.get('pin_memory', False),
        'persistent_workers': training_config.get('persistent_workers', False) if num_workers > 0 else False,
        'prefetch_factor': training_config.get('prefetch_factor', 2) if num_workers > 0 else None,
    }
    if num_workers == 0 and 'prefetch_factor' in dl_kwargs:
        del dl_kwargs['prefetch_factor']

    def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        target_n = pad_neurons_to
        K = batch[0]['behavior'].shape[0]

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

        spikes = torch.stack([pad_spikes(it['spikes'], target_n) for it in batch], dim=0)      # (B, 1, Npad)
        positions = torch.stack([pad_positions(it['positions'], target_n) for it in batch], dim=0)  # (B, Npad, 3)
        masks = torch.stack([pad_mask(it['neuron_mask'], target_n) for it in batch], dim=0)    # (B, Npad)
        behaviors = torch.stack([it['behavior'] for it in batch], dim=0)                    # (B, K)

        return {
            'spikes': spikes,
            'positions': positions,
            'neuron_mask': masks,
            'behavior': behaviors,
            'file_path': [it['file_path'] for it in batch],
            'start_idx': torch.tensor([it['start_idx'] for it in batch], dtype=torch.long),
        }

    # Build weighted sampler mixing uniform and magnitude-proportional sampling (50/50)
    # Precompute per-sample weights for the train dataset
    weights = None
    try:
        # magnitude score per sample = mean(|behavior|) at its index
        # Build per-sequence mapping: file_path,start_idx -> magnitude
        seq_weights = []
        for seq in train_dataset.sequences:
            fp = seq['file_path']
            t = seq['start_idx']
            if train_dataset.use_cache:
                beh_TK = train_dataset.cached_behavior.get(fp, None)
                if beh_TK is not None:
                    rel_t = t - train_dataset.cache_offset[fp]
                    if 0 <= rel_t < beh_TK.shape[0]:
                        mag = float(np.mean(np.abs(beh_TK[rel_t])))
                    else:
                        mag = 0.0
                else:
                    mag = 0.0
            else:
                # Fallback: uniform if uncached
                mag = 0.0
            seq_weights.append(mag)
        mags = np.array(seq_weights, dtype=np.float64)
        mags = np.nan_to_num(mags, nan=0.0, posinf=0.0, neginf=0.0)
        if mags.max() > 0:
            mags = mags / (mags.max() + 1e-12)
        # Mix 50/50 uniform and magnitude
        uni = np.full_like(mags, 1.0 / max(1, len(mags)))
        mix = 0.5 * uni + 0.5 * (mags / mags.sum() if mags.sum() > 0 else uni)
        weights = torch.from_numpy(mix.astype(np.float64))
        train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    except Exception:
        train_sampler = None

    train_loader = DataLoader(train_dataset, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_pad, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=None, collate_fn=collate_pad, **dl_kwargs)

    return train_loader, val_loader, None, None


