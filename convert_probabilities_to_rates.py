#!/usr/bin/env python3
"""
Convert spike probabilities (T, N) to non-clipped spike rates in Hz (T, N).

This utility reads subject H5 files produced by unified_spike_processing.py
containing `spike_probabilities` and writes new H5 files that additionally
contain `spike_rates_hz`, computed via the inverse of the original mapping:

  probability = 1 - exp(-rate_hz / sampling_rate_hz)

Inverse used here (numerically stable):

  rate_hz = -sampling_rate_hz * log1p(-probability)

Notes:
- Probabilities are clipped to [0, 1 - eps] during conversion to avoid inf
  results at probability == 1.0. eps is configurable.
- All other datasets and file attributes are preserved in the output files.
- Generates a per-subject PDF with rate-focused summary plots.

Usage:
  python convert_probabilities_to_rates.py \
    --input-dir /home/user/gbm3/GBM3/processed_spike_voxels_2018 \
    --output-dir /home/user/gbm3/GBM3/processed_spike_rates_2018 \
    --dtype float32 --num-neurons-viz 10
"""

import os
import sys
import argparse
import math
import gc
from typing import Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_sampling_rate_hz(f: h5py.File) -> float:
    """Best-effort retrieval of the effective sampling rate in Hz.

    Preference order:
      1) f.attrs['final_sampling_rate']
      2) f.attrs['effective_sampling_rate']
      3) f.attrs['target_sampling_rate']
      4) f.attrs['original_sampling_rate']
      5) f['original_sampling_rate_hz'][()] (dataset)
      6) f.attrs['matlab_fpsec']
    """
    for key in (
        "final_sampling_rate",
        "effective_sampling_rate",
        "target_sampling_rate",
        "original_sampling_rate",
    ):
        if key in f.attrs:
            try:
                return float(f.attrs[key])
            except Exception:
                pass
    # Dataset fallback
    if "original_sampling_rate_hz" in f:
        try:
            return float(f["original_sampling_rate_hz"][()])
        except Exception:
            pass
    if "matlab_fpsec" in f.attrs:
        try:
            return float(f.attrs["matlab_fpsec"])
        except Exception:
            pass
    raise RuntimeError("Could not determine sampling rate (Hz) from file metadata")


def _create_rates_pdf(
    rates_shape: Tuple[int, int],
    per_neuron_mean: Optional[np.ndarray],
    per_neuron_var: Optional[np.ndarray],
    per_frame_mean: Optional[np.ndarray],
    subject_name: str,
    out_pdf_path: str,
    sample_traces: Optional[np.ndarray] = None,
):
    """Create a PDF summarizing spike rates.

    - rates_shape: (T, N)
    - per_neuron_mean: (N,)
    - per_neuron_var: (N,)
    - per_frame_mean: (T,)
    - sample_traces: (T, K) for K sample neurons (optional)
    """
    T, N = rates_shape
    # Global stats (robust to None)
    try:
        global_mean = (
            float(np.mean(per_neuron_mean)) if per_neuron_mean is not None else None
        )
        global_var = (
            float(np.mean(per_neuron_var)) if per_neuron_var is not None else None
        )
    except Exception:
        global_mean = None
        global_var = None

    with PdfPages(out_pdf_path) as pdf:
        # Summary page
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle(f"Subject: {subject_name} — Spike Rates (Hz)", fontsize=12)

            # Text panel
            ax0 = axes[0, 0]
            ax0.axis("off")
            lines = [
                f"Neurons (N): {N}",
                f"Timepoints (T): {T}",
                f"Global mean rate (Hz): {global_mean:.6f}"
                if global_mean is not None
                else "Global mean rate (Hz): N/A",
                f"Global var rate (Hz^2): {global_var:.6f}"
                if global_var is not None
                else "Global var rate (Hz^2): N/A",
                f"Per-neuron mean rate — mean: {np.mean(per_neuron_mean):.6f}"
                if per_neuron_mean is not None
                else "Per-neuron mean rate: N/A",
            ]
            ax0.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)

            # Histogram: per-neuron mean rate
            ax1 = axes[0, 1]
            if per_neuron_mean is not None:
                ax1.hist(per_neuron_mean, bins=50, color="steelblue", alpha=0.8)
                ax1.set_title("Histogram: per-neuron mean rate (Hz)")
            else:
                ax1.axis("off")

            # Per-frame mean rate over time
            ax2 = axes[1, 0]
            if per_frame_mean is not None:
                ax2.plot(per_frame_mean, color="darkmagenta", lw=0.8)
                ax2.set_title("Per-frame mean rate (Hz) over time")
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("Mean rate (Hz)")
            else:
                ax2.axis("off")

            # Placeholder for per-neuron variance histogram
            ax3 = axes[1, 1]
            if per_neuron_var is not None:
                ax3.hist(per_neuron_var, bins=50, color="orange", alpha=0.8)
                ax3.set_title("Histogram: per-neuron variance (Hz^2)")
            else:
                ax3.axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        except Exception:
            pass

        # Sample neuron traces
        if sample_traces is not None and sample_traces.size > 0:
            K = sample_traces.shape[1]
            for i in range(K):
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                    ax.plot(sample_traces[:, i], color="tab:purple", lw=0.8)
                    ax.set_title(f"Spike rate trace (Hz) — neuron {i + 1} of {K}")
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Rate (Hz)")
                    ax.grid(alpha=0.2)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception:
                    pass


def _copy_all_datasets_and_attrs(src: h5py.File, dst: h5py.File):
    """Copy all datasets and root attributes from src into dst.

    Existing objects with the same name in dst will be overwritten.
    """
    # Copy datasets/groups at root
    for key in list(src.keys()):
        try:
            if key in dst:
                del dst[key]
            src.copy(key, dst, name=key)
        except Exception as e:
            print(f"  → Warning: failed to copy dataset/group '{key}': {e}")
    # Copy attributes
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception as e:
            print(f"  → Warning: failed to copy attribute '{k}': {e}")


def convert_file(
    src_path: str,
    dst_path: str,
    dtype: str = "float32",
    num_neurons_viz: int = 10,
    chunk_rows: int = 1000,
    eps: float = 1e-7,
):
    """Convert a single subject H5 file from probabilities to rates and write output."""
    subject_name = os.path.basename(src_path).replace(".h5", "")

    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        # Copy everything first
        _copy_all_datasets_and_attrs(fin, fout)

        # Ensure probabilities exist
        if "spike_probabilities" not in fin:
            raise RuntimeError(f"'spike_probabilities' dataset missing in {src_path}")

        probs_ds = fin["spike_probabilities"]  # (T, N)
        T, N = int(probs_ds.shape[0]), int(probs_ds.shape[1])

        sampling_rate_hz = _read_sampling_rate_hz(fin)

        # Prepare output dataset
        np_dtype = getattr(np, dtype)
        rates_ds = fout.create_dataset(
            "spike_rates_hz",
            shape=(T, N),
            dtype=np_dtype,
            chunks=(min(chunk_rows, T), min(100, N)),
            compression="gzip",
            compression_opts=1,
        )

        # Statistics accumulators
        per_neuron_sum = np.zeros((N,), dtype=np.float64)
        per_neuron_sq_sum = np.zeros((N,), dtype=np.float64)
        per_frame_mean = np.zeros((T,), dtype=np.float32)

        # Select neurons to visualize
        rng = np.random.default_rng(42)
        nsel = max(0, min(num_neurons_viz, N))
        sel_idx = (
            rng.choice(N, nsel, replace=False)
            if nsel > 0
            else np.array([], dtype=np.int64)
        )
        sample_traces = np.zeros((T, nsel), dtype=np.float32) if nsel > 0 else None

        # Convert in time-chunks to limit memory
        for start in tqdm(range(0, T, chunk_rows), desc=f"Converting {subject_name}"):
            end = min(start + chunk_rows, T)
            # Read probabilities slice (end-start, N)
            p_chunk = probs_ds[start:end, :].astype(np.float32, copy=False)
            # Clip to avoid log(0) when p==1
            np.clip(p_chunk, 0.0, 1.0 - eps, out=p_chunk)
            # rate_hz = -F * log1p(-p)
            r_chunk = -float(sampling_rate_hz) * np.log1p(-p_chunk)

            # Write to output
            rates_ds[start:end, :] = r_chunk.astype(np_dtype, copy=False)

            # Update stats
            per_neuron_sum += r_chunk.sum(axis=0, dtype=np.float64)
            per_neuron_sq_sum += np.square(r_chunk, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )
            per_frame_mean[start:end] = r_chunk.mean(axis=1).astype(np.float32)

            # Collect selected traces
            if nsel > 0:
                sample_traces[start:end, :] = r_chunk[:, sel_idx].astype(
                    np.float32, copy=False
                )

            # Free memory
            del p_chunk, r_chunk
            gc.collect()

        # Finalize stats
        per_neuron_mean = (per_neuron_sum / float(T)).astype(np.float32)
        per_neuron_var = (
            per_neuron_sq_sum / float(T) - np.square(per_neuron_mean)
        ).astype(np.float32)

        # File-level metadata
        fout.attrs["includes_rates"] = True
        fout.attrs["rate_unit"] = "Hz"
        fout.attrs["probability_to_rate_formula"] = (
            "rate_hz = -sampling_rate_hz * log1p(-probability)"
        )
        fout.attrs["conversion_source"] = os.path.basename(src_path)

    # Create PDF visualization (post-close to ensure data is flushed)
    out_pdf_path = dst_path.replace(".h5", "_rates_visualization.pdf")
    try:
        _create_rates_pdf(
            rates_shape=(T, N),
            per_neuron_mean=per_neuron_mean,
            per_neuron_var=per_neuron_var,
            per_frame_mean=per_frame_mean,
            subject_name=subject_name,
            out_pdf_path=out_pdf_path,
            sample_traces=sample_traces,
        )
    except Exception as e:
        print(f"  → Warning: failed to create PDF for {subject_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert spike probabilities to rates (Hz) and replicate subject files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input subject H5 files (probabilities)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write converted subject H5 files (with rates)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Output dtype for rates",
    )
    parser.add_argument(
        "--num-neurons-viz",
        type=int,
        default=10,
        help="Number of neurons to visualize per subject",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1000,
        help="Time-chunk rows to process at once",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-7,
        help="Epsilon to avoid log(0) at probability=1.0",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    parser.add_argument(
        "--subjects-glob",
        type=str,
        default="subject_*.h5",
        help="Glob to select subject files within input-dir",
    )

    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Enumerate subjects
    import glob

    subject_files = sorted(glob.glob(os.path.join(in_dir, args.subjects_glob)))
    if not subject_files:
        print(f"No subject files matching {args.subjects_glob} in {in_dir}")
        return

    print(f"Found {len(subject_files)} subject files in {in_dir}")
    for src in subject_files:
        base = os.path.basename(src)
        dst = os.path.join(out_dir, base)
        pdf = dst.replace(".h5", "_rates_visualization.pdf")
        if (not args.overwrite) and os.path.exists(dst) and os.path.exists(pdf):
            print(f"Skipping {base} (already converted)")
            continue
        try:
            print(f"\nConverting: {base}")
            convert_file(
                src_path=src,
                dst_path=dst,
                dtype=args.dtype,
                num_neurons_viz=args.num_neurons_viz,
                chunk_rows=args.chunk_rows,
                eps=args.eps,
            )
            print(f"  → Wrote: {dst}")
            print(f"  → PDF:  {pdf}")
        except Exception as e:
            print(f"  ERROR converting {base}: {e}")
        finally:
            gc.collect()

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
