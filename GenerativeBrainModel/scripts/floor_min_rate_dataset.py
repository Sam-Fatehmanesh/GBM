#!/usr/bin/env python3
"""
Make a clamped copy of a processed neural data folder (outputs of unified_spike_processing.py).

For each HDF5 file in the input directory:
 - Clamp the dataset 'neuron_values' to a minimum floor (default: 1e-5)
 - Stream over time to avoid high memory usage
 - Recompute per-neuron log(activity) mean/std on the floored values
 - Write a new H5 in the output directory with all other datasets/attrs copied
 - Store the floor as both an attribute 'min_rate_floor' and a scalar dataset 'min_rate_floor'

This works for both 'probabilities' and 'rates_hz' semantics; it does not change semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import shutil

import numpy as np
import h5py


def copy_non_values(src: h5py.File, dst: h5py.File, exclude: set[str]) -> None:
    """Copy all top-level datasets/groups from src to dst except those in exclude.
    Uses h5py's built-in cross-file copy to preserve chunking/compression where possible.
    """
    for name, obj in src.items():
        if name in exclude:
            continue
        try:
            src.copy(name, dst, name=name)
        except Exception:
            # Fallback: manual create for simple arrays
            if isinstance(obj, h5py.Dataset):
                dst.create_dataset(name, data=obj[()])
            else:
                grp = dst.create_group(name)
                for k, v in obj.attrs.items():
                    grp.attrs[k] = v


def clamp_and_rewrite(src_path: Path, dst_path: Path, floor: float, log_eps: float = 1e-7,
                      time_chunk: int = 4096) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # Copy attributes first
        for k, v in src.attrs.items():
            try:
                dst.attrs[k] = v
            except Exception:
                pass

        # Copy everything except the datasets we will regenerate
        exclude = {'neuron_values', 'log_activity_mean', 'log_activity_std'}
        copy_non_values(src, dst, exclude)

        if 'neuron_values' not in src:
            raise RuntimeError(f"Missing 'neuron_values' in {src_path}")

        d_in = src['neuron_values']
        T, N = int(d_in.shape[0]), int(d_in.shape[1])
        out_dtype = d_in.dtype

        # Create destination dataset with similar chunking/compression when available
        chunks = d_in.chunks if d_in.chunks is not None else (min(1024, T), min(256, N))
        compression = d_in.compression or 'gzip'
        compression_opts = d_in.compression_opts if d_in.compression_opts is not None else 1
        d_out = dst.create_dataset('neuron_values', shape=(T, N), dtype=out_dtype,
                                   chunks=chunks, compression=compression, compression_opts=compression_opts)

        # Streaming stats for log(activity)
        sum_logs = np.zeros((N,), dtype=np.float64)
        sumsq_logs = np.zeros((N,), dtype=np.float64)
        total = 0

        # Process by time chunks
        for t0 in range(0, T, time_chunk):
            t1 = min(T, t0 + time_chunk)
            slab = d_in[t0:t1, :].astype(np.float32, copy=False)  # (Lt, N)
            # Clamp to floor
            np.maximum(slab, floor, out=slab)
            # Write out
            d_out[t0:t1, :] = slab.astype(out_dtype, copy=False)
            # Accumulate log stats
            logs = np.log(np.maximum(slab, log_eps, dtype=np.float32))
            sum_logs += logs.sum(axis=0, dtype=np.float64)
            sumsq_logs += np.square(logs, dtype=np.float32).sum(axis=0, dtype=np.float64)
            total += (t1 - t0)

        # Compute mean/std
        if total <= 0:
            lam = np.zeros((N,), dtype=np.float32)
            las = np.zeros((N,), dtype=np.float32)
        else:
            mean = sum_logs / float(total)
            var = np.maximum(sumsq_logs / float(total) - mean * mean, 1e-12)
            lam = mean.astype(np.float32)
            las = np.sqrt(var, dtype=np.float64).astype(np.float32)

        dst.create_dataset('log_activity_mean', data=lam, compression='gzip', compression_opts=1)
        dst.create_dataset('log_activity_std', data=las, compression='gzip', compression_opts=1)

        # Update attributes
        dst.attrs['includes_log_activity_stats'] = True
        dst.attrs['log_activity_eps'] = float(log_eps)
        dst.attrs['min_rate_floor'] = float(floor)
        # Also store as a scalar dataset for ease of loading
        dst.create_dataset('min_rate_floor', data=np.array([float(floor)], dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description='Clamp min rates in processed H5s and recompute log stats')
    ap.add_argument('--input_dir', type=str, required=True, help='Directory containing source H5 files')
    ap.add_argument('--output_dir', type=str, required=True, help='Directory to write clamped H5 files')
    ap.add_argument('--min_rate', type=float, default=1e-5, help='Minimum rate/probability floor (default: 1e-5)')
    ap.add_argument('--log_eps', type=float, default=1e-7, help='Epsilon used when taking logs for stats')
    ap.add_argument('--pattern', type=str, default='*.h5', help='Glob pattern for H5 files (default: *.h5)')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite output files if they exist')
    args = ap.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(args.pattern))
    if not files:
        print(f"No files matched in {src_dir} with pattern {args.pattern}")
        return

    print(f"Clamping min rate to {args.min_rate} and recomputing log stats for {len(files)} files…")
    for fp in files:
        if not fp.is_file():
            continue
        out_fp = dst_dir / fp.name
        if out_fp.exists() and not args.overwrite:
            print(f"[skip] {out_fp} exists. Use --overwrite to replace.")
            continue
        try:
            clamp_and_rewrite(fp, out_fp, float(args.min_rate), float(args.log_eps))
            print(f"[ok] {fp.name} → {out_fp.name}")
        except Exception as e:
            print(f"[error] {fp.name}: {e}")


if __name__ == '__main__':
    main()



