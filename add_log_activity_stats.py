#!/usr/bin/env python3
"""
Add per-neuron log(activity) mean/std to existing GBM H5 outputs.

Log activity is computed as:
  log_vals = log(max(activity, eps))
where activity is the stored `neuron_values` (probabilities, rates, or z-scored
signal as indicated by `neuron_values_semantics`). The mean and std are taken
over time per neuron and saved as datasets:
  - log_activity_mean: (N,)
  - log_activity_std:  (N,)

Attributes added/updated:
  - includes_log_activity_stats: True/False
  - log_activity_eps: eps used during computation

Usage:
  python add_log_activity_stats.py --data-dir processed_spike_rates_2018 --eps 1e-7 --overwrite
"""

import os
import argparse
import gc
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm


def process_file(path: str, eps: float, overwrite: bool) -> Optional[str]:
    try:
        with h5py.File(path, 'r+') as f:
            if ('log_activity_mean' in f) and ('log_activity_std' in f) and (not overwrite):
                return None

            if 'neuron_values' not in f:
                return None

            vals_ds = f['neuron_values']  # (T, N)
            T, N = int(vals_ds.shape[0]), int(vals_ds.shape[1])

            # Compute in chunks over time to limit memory
            chunk_T = min(2000, T)
            sum_logs = np.zeros((N,), dtype=np.float64)
            sum_sq_logs = np.zeros((N,), dtype=np.float64)
            count = 0

            for t0 in range(0, T, chunk_T):
                t1 = min(T, t0 + chunk_T)
                slab = vals_ds[t0:t1, :].astype(np.float32, copy=False)
                np.maximum(slab, eps, out=slab)
                logs = np.log(slab, dtype=np.float32)
                sum_logs += logs.sum(axis=0, dtype=np.float64)
                sum_sq_logs += np.square(logs, dtype=np.float64).sum(axis=0, dtype=np.float64)
                count += (t1 - t0)
                del slab, logs
                gc.collect()

            mean = (sum_logs / float(count)).astype(np.float32)
            var = (sum_sq_logs / float(count) - np.square(mean)).astype(np.float32)
            std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)

            if 'log_activity_mean' in f:
                del f['log_activity_mean']
            if 'log_activity_std' in f:
                del f['log_activity_std']
            f.create_dataset('log_activity_mean', data=mean, compression='gzip', compression_opts=1)
            f.create_dataset('log_activity_std', data=std, compression='gzip', compression_opts=1)
            f.attrs['includes_log_activity_stats'] = True
            f.attrs['log_activity_eps'] = float(eps)
        return path
    except Exception as e:
        print(f"  ERROR updating {os.path.basename(path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Add per-neuron log(activity) mean/std to GBM H5 outputs')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing subject_*.h5 files')
    parser.add_argument('--eps', type=float, default=1e-7, help='Epsilon to avoid log(0)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite log stats if already present')
    parser.add_argument('--subjects-glob', type=str, default='subject_*.h5', help='Glob to select subject files')
    args = parser.parse_args()

    import glob
    files = sorted(glob.glob(os.path.join(args.data_dir, args.subjects_glob)))
    if not files:
        print(f"No files matching {args.subjects_glob} under {args.data_dir}")
        return

    print(f"Found {len(files)} files. Computing log(activity) stats (eps={args.eps})…")
    updated = 0
    for p in tqdm(files, desc='Files'):
        res = process_file(p, args.eps, args.overwrite)
        if res is not None:
            updated += 1
    print(f"Done. Updated {updated}/{len(files)} files.")


if __name__ == '__main__':
    main()









