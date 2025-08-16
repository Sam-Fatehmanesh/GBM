#!/usr/bin/env python3
"""
Create 2D PCA and 2D UMAP embeddings for the entire processed spike-voxel dataset
(`processed_spike_voxels_2018`), treating each brain volume (single timepoint across
all neurons/voxels) as one data point. Points are color-coded by subject.

Optionally, compute PCA/UMAP on upsampled behavior time series (per-timepoint vectors
across behavior channels) using --do_behavior. Behavior is upsampled per subject to
match that subject's neural timepoints and then trimmed to the nonzero neural window.

Outputs under experiments/dataset_embeddings/<timestamp>/:
- logs/pca_subjects.png
- logs/umap_subjects.png
- logs/pca_embeddings.h5   (embedding, subject_labels, subjects)
- logs/umap_embeddings.h5  (embedding, subject_labels, subjects)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from GenerativeBrainModel.dataloaders.neural_dataloader import _max_neurons_in_files


def setup_experiment_dirs(base_dir: Path, name: str) -> Dict[str, Path]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / f"{name}_{ts}"
    logs_dir = exp_dir / 'logs'
    for p in [exp_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {'exp': exp_dir, 'logs': logs_dir}


def _trim_nonzero_rows(spikes_ds: np.ndarray) -> Tuple[int, int]:
    """Return [start, end) indices trimming leading/trailing all-zero rows.
    Mirrors the logic in `NeuralDataset._build_sequence_index` to match behavior.
    """
    T_total = spikes_ds.shape[0]
    def _first_nonzero_row(arr: np.ndarray, a: int, b: int, chunk: int = 256) -> int:
        i = a
        eps = 0.0
        while i < b:
            j = min(i + chunk, b)
            block = arr[i:j, :]
            nz = np.any(np.abs(block) > eps, axis=1)
            if nz.any():
                return i + int(np.argmax(nz))
            i = j
        return b
    def _last_nonzero_row(arr: np.ndarray, a: int, b: int, chunk: int = 256) -> int:
        i = b
        eps = 0.0
        while i > a:
            j = max(a, i - chunk)
            block = arr[j:i, :]
            nz = np.any(np.abs(block) > eps, axis=1)
            if nz.any():
                return j + (len(nz) - 1 - int(np.argmax(nz[::-1])))
            i = j
        return a - 1
    nz_start = _first_nonzero_row(spikes_ds, 0, T_total)
    nz_last = _last_nonzero_row(spikes_ds, 0, T_total)
    if nz_start >= T_total or nz_last < nz_start:
        return 0, 0
    return nz_start, nz_last + 1


def load_all_subject_timepoints(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load all subjects' timepoints into a single matrix (sum_T, max_N).

    Returns (X, subject_labels, subjects) where:
    - X: float32 array of shape (sum_T_kept, N_max)
    - subject_labels: int32 array of length sum_T_kept with subject index
    - subjects: list of subject names in label order
    """
    files = sorted([str(f) for f in data_dir.glob('*.h5')])
    if not files:
        raise ValueError(f"No H5 files found in {data_dir}")

    # Determine global neuron width
    n_max = _max_neurons_in_files(files)

    # Determine subjects list from filenames
    subjects: List[str] = []
    file_to_subject: Dict[str, int] = {}
    for fp in files:
        name = Path(fp).stem
        subjects.append(name)
        file_to_subject[fp] = len(subjects) - 1

    # First pass to accumulate total rows
    total_rows = 0
    kept_rows_per_file: Dict[str, Tuple[int, int]] = {}
    for fp in files:
        with h5py.File(fp, 'r') as f:
            ds = f['spike_probabilities']  # (T, N)
            # Load into memory for trimming
            arr = ds[()].astype(np.float32)
            start, end = _trim_nonzero_rows(arr)
            if end > start:
                total_rows += (end - start)
                kept_rows_per_file[fp] = (start, end)
            else:
                kept_rows_per_file[fp] = (0, 0)

    # Allocate outputs
    X = np.zeros((total_rows, n_max), dtype=np.float32)
    y = np.zeros((total_rows,), dtype=np.int32)

    # Second pass: fill matrix
    cursor = 0
    for fp in files:
        start, end = kept_rows_per_file[fp]
        if end <= start:
            continue
        with h5py.File(fp, 'r') as f:
            arr = f['spike_probabilities'][start:end].astype(np.float32)  # (Tk, N)
        Tk, N = arr.shape
        X[cursor:cursor+Tk, :N] = arr
        y[cursor:cursor+Tk] = file_to_subject[fp]
        cursor += Tk

    return X, y, subjects


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


def load_all_subject_behavior_timepoints(data_dir: Path, subjects: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load all subjects' behavior into a matrix (sum_T_kept, B).

    For each file:
      - Load spike_probabilities to determine T and the nonzero trim window [start,end)
      - Load behavior_full (expected (B, Tb) or (Tb, B))
      - Upsample each behavior channel to length T
      - Slice to [start:end) and append rows

    Returns (Xb, y) where:
      - Xb: float32 array of shape (sum_T_kept, B)
      - y:  int32 subject labels aligned to rows
    """
    files = sorted([str(f) for f in data_dir.glob('*.h5')])
    if not files:
        raise ValueError(f"No H5 files found in {data_dir}")

    subj_to_idx: Dict[str, int] = {name: i for i, name in enumerate(subjects)}

    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    B_dim: int = -1

    for fp in files:
        stem = Path(fp).stem
        if stem not in subj_to_idx:
            # Skip files not present in provided subjects mapping
            continue
        with h5py.File(fp, 'r') as f:
            if 'spike_probabilities' not in f:
                continue
            spikes = f['spike_probabilities'][()].astype(np.float32)
            T, _ = spikes.shape
            start, end = _trim_nonzero_rows(spikes)
            if end <= start:
                continue
            if 'behavior_full' not in f:
                # Skip subjects without behavior
                continue
            beh = f['behavior_full'][()]
            # Normalize shape to (B, Tb)
            if beh.ndim == 2:
                if beh.shape[0] <= beh.shape[1]:
                    # Likely (B, Tb)
                    B, Tb = int(beh.shape[0]), int(beh.shape[1])
                    beh_bt = beh.astype(np.float32)
                else:
                    # Likely (Tb, B)
                    Tb, B = int(beh.shape[0]), int(beh.shape[1])
                    beh_bt = beh.astype(np.float32).T
            elif beh.ndim == 1:
                # Single behavior channel
                B, Tb = 1, int(beh.shape[0])
                beh_bt = beh.astype(np.float32)[None, :]
            else:
                continue

            # Replace NaNs/infs in raw behavior before dimension alignment
            try:
                beh_bt = np.nan_to_num(beh_bt, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass

            # Establish consistent behavior dimension across subjects
            if B_dim < 0:
                B_dim = B
            else:
                # If inconsistent, pad/truncate to first observed B_dim
                if B != B_dim:
                    if B < B_dim:
                        pad = np.zeros((B_dim - B, Tb), dtype=np.float32)
                        beh_bt = np.concatenate([beh_bt, pad], axis=0)
                        B = B_dim
                    else:
                        beh_bt = beh_bt[:B_dim, :]
                        B = B_dim

            # Resample each behavior channel to neural length T
            beh_resampled = np.zeros((B, T), dtype=np.float32)
            for b in range(B):
                beh_resampled[b] = _resample_1d_series(beh_bt[b], T)
            # Safety: clean any residual NaNs/infs post-resampling
            beh_resampled = np.nan_to_num(beh_resampled, nan=0.0, posinf=0.0, neginf=0.0)

            # Slice to [start, end)
            beh_slice = beh_resampled[:, start:end].T  # (Tk, B)
            X_rows.append(beh_slice)
            y_rows.append(np.full((beh_slice.shape[0],), subj_to_idx[stem], dtype=np.int32))

    if not X_rows:
        raise ValueError("No behavior data found to embed.")

    Xb = np.concatenate(X_rows, axis=0)
    # Final safety: ensure no NaNs/infs for downstream PCA/UMAP
    Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    yb = np.concatenate(y_rows, axis=0)
    return Xb, yb


def main() -> None:
    parser = argparse.ArgumentParser(description='PCA/UMAP embeddings for entire processed dataset')
    parser.add_argument('--data_dir', type=str, default='processed_spike_voxels_2018', help='Directory with subject_*.h5 files')
    parser.add_argument('--output_dir', type=str, default='experiments/dataset_embeddings', help='Where to save outputs')
    parser.add_argument('--random_state', type=int, default=42)
    # Parametric UMAP controls (torch-based)
    parser.add_argument('--umap_n_neighbors', type=int, default=30)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--umap_subsample_frac', type=float, default=0.10, help='Fraction of rows to fit UMAP on')
    parser.add_argument('--umap_max_train_samples', type=int, default=100000, help='Cap on rows used to fit UMAP')
    parser.add_argument('--umap_epochs', type=int, default=10)
    parser.add_argument('--umap_batch_size', type=int, default=2048)
    parser.add_argument('--umap_lr', type=float, default=1e-3)
    parser.add_argument('--umap_use_gpu', action='store_true', help='Use GPU for UMAP if available')
    parser.add_argument('--umap_num_workers', type=int, default=1)
    parser.add_argument('--behavior_metric', type=str, default='cosine', choices=['euclidean','cosine','manhattan'], help='Distance metric for behavior UMAP')
    # Which embeddings to run
    parser.add_argument('--skip_neural', action='store_true', help='Skip neural embeddings')
    parser.add_argument('--do_behavior', action='store_true', help='Also compute behavior embeddings')
    # Coloring option for neural embeddings
    parser.add_argument('--color_by_behavior_magnitude', action='store_true', help='Color neural PCA/UMAP by per-timepoint magnitude of standardized behavior (L2 across channels) instead of by subject')
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    dirs = setup_experiment_dirs(base_dir, 'full_dataset_embeddings')

    data_dir = Path(args.data_dir)
    print(f"Loading all timepoints from: {data_dir}")
    X = None
    y = None
    subjects: List[str] = []
    if not args.skip_neural:
        # Load full neural matrix and subject labels
        X, y, subjects = load_all_subject_timepoints(data_dir)
        print(f"Loaded neural matrix shape: {X.shape}; subjects: {len(subjects)}")
    elif args.do_behavior:
        # Behavior-only path: just enumerate subjects without loading neural matrix
        files = sorted([str(f) for f in data_dir.glob('*.h5')])
        subjects = [Path(fp).stem for fp in files]
        print(f"Enumerated subjects for behavior-only embeddings: {len(subjects)}")

    # Optionally pre-load behavior and compute standardized magnitude aligned to neural rows
    beh_mag = None  # type: ignore[assignment]
    Xb_std = None   # type: ignore[assignment]
    yb = None       # type: ignore[assignment]
    mu_b = None     # type: ignore[assignment]
    std_b_safe = None  # type: ignore[assignment]
    if (not args.skip_neural) and (args.color_by_behavior_magnitude or args.do_behavior):
        print("Loading and aligning behavior time series across subjects for coloring/embeddings…")
        Xb, yb = load_all_subject_behavior_timepoints(data_dir, subjects)
        # Standardize behavior features globally (z-score), guard zero-std
        mu_b = Xb.mean(axis=0, dtype=np.float64)
        std_b = Xb.std(axis=0, dtype=np.float64)
        std_b_safe = np.where(std_b < 1e-8, 1.0, std_b)
        Xb_std = ((Xb - mu_b.astype(np.float32)) / std_b_safe.astype(np.float32)).astype(np.float32)
        if args.color_by_behavior_magnitude:
            # L2 norm across standardized behavior channels per timepoint
            beh_mag = np.sqrt((Xb_std.astype(np.float32) ** 2).sum(axis=1))

    # PCA 2D (compute, plot, save H5) BEFORE UMAP for NEURAL
    from sklearn.decomposition import PCA
    if not args.skip_neural:
        pca = PCA(n_components=2, random_state=args.random_state)
        pca_2d = pca.fit_transform(X)
    # Plot helpers
    def plot_scatter(points: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
        plt.figure(figsize=(9, 5))
        cmap = plt.get_cmap('tab20')
        num_classes = len(subjects)
        for idx, subj in enumerate(subjects):
            sel = (labels == idx)
            color = cmap(idx % cmap.N)
            plt.scatter(points[sel, 0], points[sel, 1], s=1.0, alpha=0.6, marker='.', label=subj, color=color)
        plt.title(title)
        plt.legend(markerscale=6, fontsize='small', ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
    def plot_scatter_continuous(points: np.ndarray, values: np.ndarray, title: str, out_path: Path, cmap_name: str = 'viridis', log_scale: bool = False) -> None:
        plt.figure(figsize=(9, 5))
        norm = None
        if log_scale:
            eps = 1e-6
            vmin = float(max(eps, np.min(values[values > 0]) if np.any(values > 0) else eps))
            vmax = float(max(vmin * (1.0 + 1e-6), np.max(values) + eps))
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        sc = plt.scatter(
            points[:, 0], points[:, 1], c=values, s=1.0, alpha=0.7, marker='.',
            cmap=plt.get_cmap(cmap_name), norm=norm
        )
        plt.title(title)
        cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
        cbar.set_label('Behavior magnitude (L2, standardized' + (', log scale' if log_scale else '') + ')')
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()

    if not args.skip_neural:
        if args.color_by_behavior_magnitude and beh_mag is not None:
            plot_scatter_continuous(
                pca_2d, beh_mag,
                'PCA (2D) - entire dataset (colored by behavior magnitude, log scale)',
                dirs['logs'] / 'pca_behavior_magnitude.png',
                log_scale=True
            )
        else:
            plot_scatter(pca_2d, y, 'PCA (2D) - entire dataset', dirs['logs'] / 'pca_subjects.png')
        # Save PCA to H5
        with h5py.File(dirs['logs'] / 'pca_embeddings.h5', 'w') as f:
            f.create_dataset('embedding', data=pca_2d, compression='gzip', compression_opts=1)
            f.create_dataset('subject_labels', data=y.astype(np.int32), compression='gzip', compression_opts=1)
            str_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('subjects', data=np.array(subjects, dtype=object), dtype=str_dt)
            if args.color_by_behavior_magnitude and beh_mag is not None:
                f.create_dataset('behavior_magnitude', data=beh_mag.astype(np.float32))

    # Prepare RNG for any UMAP section
    rng = np.random.default_rng(args.random_state)
    # Import parametric UMAP if needed for neural and/or behavior
    if (not args.skip_neural) or args.do_behavior:
        # Compatibility shim: some versions of umap-pytorch call np.product; alias to np.prod if missing
        if not hasattr(np, 'product'):
            try:
                np.product = np.prod  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            import torch
            from umap_pytorch import PUMAP
        except Exception as e:
            raise RuntimeError(
                'Parametric UMAP requires the umap-pytorch package. Install with: pip install umap-pytorch'
            ) from e
        num_gpus = 1 if (args.umap_use_gpu and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 0

    # UMAP 2D (Parametric, torch) - fit on 10% uniform subset, then transform full dataset (NEURAL)
    if not args.skip_neural:
        n_total = X.shape[0]
        n_train = int(np.ceil(args.umap_subsample_frac * n_total))
        n_train = int(min(max(1, n_train), args.umap_max_train_samples))
        subset_idx = rng.choice(n_total, size=n_train, replace=False)
        X_subset = torch.from_numpy(X[subset_idx])

    # Initialize and fit PUMAP
    pumap = PUMAP(
        encoder=None,
        decoder=None,
        n_neighbors=int(args.umap_n_neighbors),
        min_dist=float(args.umap_min_dist),
        n_components=2,
        lr=float(args.umap_lr),
        epochs=int(args.umap_epochs),
        batch_size=int(args.umap_batch_size),
        num_workers=int(args.umap_num_workers),
        num_gpus=int(num_gpus),
        random_state=int(args.random_state),
        match_nonparametric_umap=False,
    )

    if not args.skip_neural:
        print(f"Fitting parametric UMAP on subset: {n_train}/{n_total} rows (frac={n_train/n_total:.3f})")
        pumap.fit(X_subset)

        # Transform full dataset in batches to control memory
        N = X.shape[0]
        bs = int(args.umap_batch_size)
        umap_full = np.zeros((N, 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, N, bs):
                end = min(N, start + bs)
                chunk = torch.from_numpy(X[start:end])
                emb = pumap.transform(chunk)
                if hasattr(emb, 'detach'):
                    emb = emb.detach().cpu().numpy()
                else:
                    emb = np.asarray(emb)
                umap_full[start:end] = emb.astype(np.float32)

        # Plot and save
        if args.color_by_behavior_magnitude and beh_mag is not None:
            plot_scatter_continuous(
                umap_full, beh_mag,
                'UMAP (2D, parametric) - entire dataset (colored by behavior magnitude, log scale)',
                dirs['logs'] / 'umap_behavior_magnitude.png',
                log_scale=True
            )
        else:
            plot_scatter(umap_full, y, 'UMAP (2D, parametric) - entire dataset', dirs['logs'] / 'umap_subjects.png')
        with h5py.File(dirs['logs'] / 'umap_embeddings.h5', 'w') as f:
            f.create_dataset('embedding', data=umap_full, compression='gzip', compression_opts=1)
            f.create_dataset('subject_labels', data=y.astype(np.int32), compression='gzip', compression_opts=1)
            str_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('subjects', data=np.array(subjects, dtype=object), dtype=str_dt)
            # Save subset indices used for training for reproducibility
            f.create_dataset('train_subset_indices', data=subset_idx.astype(np.int64))
            if args.color_by_behavior_magnitude and beh_mag is not None:
                f.create_dataset('behavior_magnitude', data=beh_mag.astype(np.float32))

    # Behavior embeddings if requested
    if args.do_behavior:
        if Xb_std is None or yb is None or mu_b is None or std_b_safe is None:
            print("Loading and aligning behavior time series across subjects…")
            Xb, yb = load_all_subject_behavior_timepoints(data_dir, subjects)
            print(f"Loaded behavior matrix shape: {Xb.shape} (features=behavior channels)")
            # Standardize behavior features globally (z-score), guard zero-std
            mu_b = Xb.mean(axis=0, dtype=np.float64)
            std_b = Xb.std(axis=0, dtype=np.float64)
            std_b_safe = np.where(std_b < 1e-8, 1.0, std_b)
            Xb_std = ((Xb - mu_b.astype(np.float32)) / std_b_safe.astype(np.float32)).astype(np.float32)
        else:
            print(f"Loaded behavior matrix (precomputed) shape: {Xb_std.shape} (features standardized)")
        # Add tiny jitter to break ties for highly repetitive rows
        Xb_std = Xb_std + (1e-6 * np.random.default_rng(args.random_state).standard_normal(Xb_std.shape).astype(np.float32))

        # PCA on behavior
        pca_b = PCA(n_components=2, random_state=args.random_state)
        pca_b_2d = pca_b.fit_transform(Xb_std)
        plot_scatter(pca_b_2d, yb, 'PCA (2D) - behavior (entire dataset)', dirs['logs'] / 'pca_behavior_subjects.png')
        with h5py.File(dirs['logs'] / 'pca_behavior_embeddings.h5', 'w') as f:
            f.create_dataset('embedding', data=pca_b_2d, compression='gzip', compression_opts=1)
            f.create_dataset('subject_labels', data=yb.astype(np.int32), compression='gzip', compression_opts=1)
            str_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('subjects', data=np.array(subjects, dtype=object), dtype=str_dt)
            f.create_dataset('feature_mean', data=mu_b.astype(np.float32))
            f.create_dataset('feature_std', data=std_b_safe.astype(np.float32))

        # UMAP on behavior (parametric, fit on subset)
        n_total_b = Xb_std.shape[0]
        n_train_b = int(np.ceil(args.umap_subsample_frac * n_total_b))
        n_train_b = int(min(max(1, n_train_b), args.umap_max_train_samples))
        subset_idx_b = rng.choice(n_total_b, size=n_train_b, replace=False)
        Xb_subset = torch.from_numpy(Xb_std[subset_idx_b])

        pumap_b = PUMAP(
            encoder=None,
            decoder=None,
            n_neighbors=int(args.umap_n_neighbors),
            min_dist=float(args.umap_min_dist),
            n_components=2,
            metric=str(args.behavior_metric),
            lr=float(args.umap_lr),
            epochs=int(args.umap_epochs),
            batch_size=int(args.umap_batch_size),
            num_workers=int(args.umap_num_workers),
            num_gpus=int(num_gpus),
            random_state=int(args.random_state),
            match_nonparametric_umap=False,
        )
        print(f"Fitting parametric UMAP (behavior) on subset: {n_train_b}/{n_total_b} rows (frac={n_train_b/n_total_b:.3f})")
        pumap_b.fit(Xb_subset)

        Nb = Xb_std.shape[0]
        bs_b = int(args.umap_batch_size)
        umap_b_full = np.zeros((Nb, 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, Nb, bs_b):
                end = min(Nb, start + bs_b)
                chunk = torch.from_numpy(Xb_std[start:end])
                emb = pumap_b.transform(chunk)
                if hasattr(emb, 'detach'):
                    emb = emb.detach().cpu().numpy()
                else:
                    emb = np.asarray(emb)
                umap_b_full[start:end] = emb.astype(np.float32)

        plot_scatter(umap_b_full, yb, 'UMAP (2D, parametric) - behavior (entire dataset)', dirs['logs'] / 'umap_behavior_subjects.png')
        with h5py.File(dirs['logs'] / 'umap_behavior_embeddings.h5', 'w') as f:
            f.create_dataset('embedding', data=umap_b_full, compression='gzip', compression_opts=1)
            f.create_dataset('subject_labels', data=yb.astype(np.int32), compression='gzip', compression_opts=1)
            str_dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('subjects', data=np.array(subjects, dtype=object), dtype=str_dt)
            f.create_dataset('train_subset_indices', data=subset_idx_b.astype(np.int64))
            f.create_dataset('feature_mean', data=mu_b.astype(np.float32))
            f.create_dataset('feature_std', data=std_b_safe.astype(np.float32))

    print(f"Saved embeddings and plots under: {dirs['logs']}")


if __name__ == '__main__':
    main()


