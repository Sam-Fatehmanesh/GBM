import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from GenerativeBrainModel.dataloaders.neural_dataloader import NeuralDataset


def _max_stimuli_in_file(fp: str) -> int:
    max_k = 1
    try:
        with h5py.File(fp, "r") as f:
            if "stimulus_full" in f:
                ds = f["stimulus_full"]
                if ds.ndim == 2:
                    max_k = int(ds.shape[1])
                elif ds.ndim == 1:
                    max_k = 1
    except Exception:
        pass
    return max_k


def _unique_neuron_ids_in_file(fp: str) -> torch.Tensor:
    try:
        with h5py.File(fp, "r") as f:
            if "neuron_global_ids" in f:
                arr = f["neuron_global_ids"][:]
            else:
                if "cell_positions" in f:
                    n = int(f["cell_positions"].shape[0])
                else:
                    n = int(f["num_neurons"][()]) if "num_neurons" in f else 0
                arr = np.arange(n, dtype=np.int64)
        return torch.from_numpy(np.array(arr, dtype=np.int64))
    except Exception:
        return torch.empty(0, dtype=torch.long)


def _collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_n = max(item["positions"].shape[0] for item in batch)
    L = batch[0]["spikes"].shape[0]
    K = batch[0]["stimulus"].shape[1]

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

    def pad_ids(ids: torch.Tensor, target_n: int) -> torch.Tensor:
        Nx = ids.shape[0]
        if Nx == target_n:
            return ids.to(torch.long)
        out = torch.zeros((target_n,), dtype=torch.long)
        out[:Nx] = ids.to(torch.long)
        return out

    def pad_vec_or_zero(
        v: torch.Tensor, target_n: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if v.numel() == 0:
            return torch.zeros((target_n,), dtype=dtype)
        Nx = v.shape[0]
        if Nx == target_n:
            return v.to(dtype)
        out = torch.zeros((target_n,), dtype=dtype)
        out[:Nx] = v.to(dtype)
        return out

    spikes = torch.stack([pad_spikes(it["spikes"], max_n) for it in batch], dim=0)
    positions = torch.stack(
        [pad_positions(it["positions"], max_n) for it in batch], dim=0
    )
    masks = torch.stack([pad_mask(it["neuron_mask"], max_n) for it in batch], dim=0)
    stimulus = torch.stack([it["stimulus"] for it in batch], dim=0)
    neuron_ids = torch.stack([pad_ids(it["neuron_ids"], max_n) for it in batch], dim=0)
    log_mean = torch.stack(
        [
            pad_vec_or_zero(it["log_activity_mean"], max_n, torch.float32)
            for it in batch
        ],
        dim=0,
    )
    log_std = torch.stack(
        [pad_vec_or_zero(it["log_activity_std"], max_n, torch.float32) for it in batch],
        dim=0,
    )

    return {
        "spikes": spikes,
        "positions": positions,
        "neuron_mask": masks,
        "stimulus": stimulus,
        "neuron_ids": neuron_ids,
        "log_activity_mean": log_mean,
        "log_activity_std": log_std,
        "file_path": [it["file_path"] for it in batch],
        "start_idx": torch.tensor([it["start_idx"] for it in batch], dtype=torch.long),
    }


def create_single_subject_dataloaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, None, None, torch.Tensor]:
    data_cfg = config["data"]
    train_cfg = config["training"]

    data_dir = Path(data_cfg["data_dir"])
    include_files = data_cfg.get("include_files", None)
    if include_files:
        subject_name = (
            include_files[0] if isinstance(include_files, list) else include_files
        )
    else:
        # fallback to subject_14.h5 if not provided
        subject_name = "subject_14.h5"
    subject_path = (data_dir / subject_name).resolve()
    if not subject_path.exists():
        raise FileNotFoundError(
            f"Single-subject dataloader: file not found: {subject_path}"
        )

    pad_stimuli_to = _max_stimuli_in_file(str(subject_path))

    sequence_length = train_cfg.get("sequence_length", 1)
    stride = train_cfg.get("stride", 1)
    max_timepoints_per_subject = train_cfg.get("max_timepoints_per_subject", None)
    use_cache = data_cfg.get("use_cache", True)
    start_timepoint = train_cfg.get("start_timepoint", None)
    end_timepoint = train_cfg.get("end_timepoint", None)
    spikes_dataset_name = data_cfg.get("spikes_dataset_name", "neuron_values")
    split_frac = float(train_cfg.get("test_split_fraction", 0.1))

    train_dataset = NeuralDataset(
        [str(subject_path)],
        pad_stimuli_to=pad_stimuli_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        spikes_dataset_name=spikes_dataset_name,
        split_role="train",
        test_split_fraction=split_frac,
    )

    val_dataset = NeuralDataset(
        [str(subject_path)],
        pad_stimuli_to=pad_stimuli_to,
        sequence_length=sequence_length,
        stride=stride,
        max_timepoints_per_subject=max_timepoints_per_subject,
        use_cache=use_cache,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        spikes_dataset_name=spikes_dataset_name,
        split_role="test",
        test_split_fraction=split_frac,
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    dl_kwargs = {
        "batch_size": train_cfg.get("batch_size", 4),
        "num_workers": num_workers,
        "pin_memory": train_cfg.get("pin_memory", False),
        "persistent_workers": train_cfg.get("persistent_workers", False)
        if num_workers > 0
        else False,
        "prefetch_factor": train_cfg.get("prefetch_factor", 2)
        if num_workers > 0
        else None,
        "pin_memory_device": "cuda" if train_cfg.get("pin_memory", False) else "",
    }
    if num_workers == 0 and "prefetch_factor" in dl_kwargs:
        del dl_kwargs["prefetch_factor"]

    train_loader = DataLoader(
        train_dataset, shuffle=True, sampler=None, collate_fn=_collate_pad, **dl_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, sampler=None, collate_fn=_collate_pad, **dl_kwargs
    )

    uniq_ids = _unique_neuron_ids_in_file(str(subject_path))
    return train_loader, val_loader, None, None, uniq_ids
