#!/usr/bin/env python3
"""
Spike Probability Processing Pipeline

This script processes calcium traces using CASCADE spike detection and saves
spike probabilities as time series data along with cell spatial positions.
No volumetric conversion is performed - data is saved directly as probability
time series and 3D position coordinates.

Features:
- CASCADE-only spike detection (no OASIS or binary spikes)
- Direct spike probability time series output (T, N) format
- Cell spatial positions preserved as (N, 3) coordinates
- YAML configuration support
- Continuous probability values only
- Configurable float16/float32 output data types
"""

import os
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from tqdm import tqdm
import gc
import multiprocessing as mp
import yaml
import argparse

# Spike detection imports
try:
    from neuralib.imaging.spikes.cascade import cascade_predict

    CASCADE_AVAILABLE = True
except ImportError:
    CASCADE_AVAILABLE = False

# TensorFlow tuning (optional) - only if CASCADE is available
if CASCADE_AVAILABLE:
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


def print_memory_stats(prefix=""):
    """Print memory usage statistics"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(
            f"{prefix} Memory: RSS={mem_info.rss / 1e9:.2f}GB, VMS={mem_info.vms / 1e9:.2f}GB"
        )
    except ImportError:
        print(f"{prefix} Memory: psutil not available")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_default_config():
    """Create default configuration dictionary"""
    return {
        "data": {
            "input_dir": "raw_trace_data_2018",
            "output_dir": "processed_spike_voxels_2018",
            "zapdata": False,  # When true, use ZapBench Zarr stores instead of per-subject raw dirs
            "skip_subjects": [],
            "test_run_neurons": None,
            "calcium_dataset": "CellResp",  # Name of calcium trace dataset in TimeSeries.h5
            "is_raw": True,  # Whether data is raw (applies proper ΔF/F: (F-F0)/F0)
            "apply_baseline_subtraction": False,  # Whether to apply baseline subtraction only (F-F0)
            "window_length": 30.0,  # Window size in seconds for baseline computation
            "baseline_percentile": 8,  # Percentile for baseline computation
            # ZapBench Zarr store locations (used when zapdata is true)
            "zap": {
                "traces_store": "/home/user/gbm3/GBM3/zapdata/traces",
                "segmentation_xy_store": "/home/user/gbm3/GBM3/zapdata/segmentation_xy",
                "stimuli_store": "/home/user/gbm3/GBM3/zapdata/stimuli_raw/stimuli_and_ephys.10chFlt",
                "stimuli_features_store": "/home/user/gbm3/GBM3/zapdata/stimuli_features",
                "subject_name": "zapbench",
                "split_by_stimuli": False,
            },
        },
        "processing": {
            "num_neurons_viz": 10,
            "batch_size": 5000,
            "workers": 1,
            "seed": 42,
            "skip_cascade": False,
            "original_sampling_rate": None,  # Required for CASCADE
            "target_sampling_rate": None,  # Required for CASCADE
            "return_to_original_rate": False,  # If True, downsample neural data back to original rate after CASCADE
            "convert_rates_to_probabilities": True,  # If False, keep CASCADE outputs as rates (Hz)
        },
        "cascade": {
            "model_type": "Global_EXC_2Hz_smoothing500ms",
        },
        "output": {
            "dtype": "float16",  # Data type for probabilities and positions
            "include_additional_data": True,  # Include anat_stack, stimulus, behavior, eye data
            "compute_log_activity_stats": False,  # If True, save per-neuron log(activity) mean/std
            "log_activity_eps": 1e-7,  # Epsilon for log to avoid log(0)
        },
    }


def save_default_config(config_path):
    """Save default configuration to YAML file"""
    config = create_default_config()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Default configuration saved to {config_path}")


def compute_baseline_correction(
    calcium_data,
    window_length=30.0,
    percentile=8,
    sampling_rate=2.0,
    is_raw=True,
    apply_baseline_subtraction=False,
):
    """
    Apply baseline correction using a causal sliding window per cell.

    Args:
        calcium_data: (T, N) array of calcium fluorescence traces
        window_length: Window size in seconds
        percentile: Percentile for baseline computation
        sampling_rate: Sampling rate in Hz
        is_raw: If True, compute proper ΔF/F: (F-F0)/F0
        apply_baseline_subtraction: If True and is_raw is False, compute baseline subtraction: F-F0

    Returns:
        processed_data: Processed data based on the specified method
    """
    T, N = calcium_data.shape
    window_frames = int(window_length * sampling_rate)

    # If neither processing is requested, return original data
    if not is_raw and not apply_baseline_subtraction:
        print("  → Using traces as-is (no baseline correction)")
        return calcium_data.copy()

    # Compute baseline F0 using sliding window
    F0 = np.zeros((T, N), dtype=np.float32)
    for t in tqdm(range(T), desc="Computing baselines"):
        # Only use frames from max(0, t-window_size) to t (inclusive)
        start = max(0, t - window_frames + 1)
        end = t + 1
        F0[t] = np.percentile(calcium_data[start:end], percentile, axis=0)

    # Apply the appropriate correction based on flags
    if is_raw:
        # Compute proper ΔF/F: (F - F0) / F0
        print("  → Computing ΔF/F: (F - F0) / F0")
        # Avoid division by zero
        F0_safe = np.where(F0 == 0, 1, F0)
        processed_data = (calcium_data - F0) / F0_safe
    elif apply_baseline_subtraction:
        # Compute baseline subtraction: F - F0
        print("  → Computing baseline subtraction: F - F0")
        processed_data = calcium_data - F0

    return processed_data


def run_cascade_inference(
    calcium_data,
    batch_size=5000,
    model_type="Global_EXC_2Hz_smoothing500ms",
    sampling_rate=2.0,
    convert_rates_to_probabilities=True,
):
    """
    Run CASCADE inference on calcium data to get spike rates or probabilities.

    Args:
        calcium_data: (T, N) array of calcium data (raw or baseline-subtracted)
        batch_size: Batch size for CASCADE processing
        model_type: CASCADE model to use
        sampling_rate: Sampling rate in Hz for converting spike rates to probabilities
        convert_rates_to_probabilities: If True, convert rates (Hz) to probabilities

    Returns:
        out_data: (N, T) array of spike probabilities or rates depending on flag
    """
    if not CASCADE_AVAILABLE:
        raise ImportError("CASCADE not available. Please install neuralib.")

    T, N = calcium_data.shape

    # Dynamically adjust batch size based on number of neurons to avoid OOM
    if N > 50000:
        batch_size = min(batch_size, 5000)
        print(
            f"  → Large dataset detected ({N} neurons), reducing batch size to {batch_size}"
        )
    elif N > 20000:
        batch_size = min(batch_size, 10000)
        print(
            f"  → Medium dataset detected ({N} neurons), reducing batch size to {batch_size}"
        )

    # Prepare containers
    out_data = np.zeros((N, T), dtype=np.float32)

    # Batched CASCADE inference
    print(f"  → Running CASCADE in batches of {batch_size}…")
    traces = calcium_data.T.astype(np.float32)  # (N, T)

    # Delete original calcium data to free memory
    del calcium_data
    gc.collect()

    for start in tqdm(range(0, N, batch_size), desc="CASCADE batches"):
        end = min(start + batch_size, N)
        batch = traces[start:end]

        try:
            batch_rates = cascade_predict(
                batch, model_type=model_type, threshold=0, padding=np.nan, verbose=False
            )

            batch_rates = np.atleast_2d(batch_rates)
            if batch_rates.shape != (end - start, T):
                raise ValueError(
                    f"Unexpected batch_rates shape: {batch_rates.shape}, expected: {(end - start, T)}"
                )

            # Handle NaN, inf values - CASCADE outputs spike rates in Hz
            batch_rates = np.nan_to_num(batch_rates, nan=0.0, posinf=0.0, neginf=0.0)

            if convert_rates_to_probabilities:
                # Convert spike rates to probabilities using Poisson process: P = 1 - e^(-λ/F)
                # where λ is spike rate (Hz) and F is sampling rate (Hz)
                batch_out = 1.0 - np.exp(-batch_rates / sampling_rate)
                # Ensure probabilities are in [0, 1] range (should be by construction, but safety check)
                batch_out = np.clip(batch_out, 0.0, 1.0)
            else:
                # Keep as rates (Hz)
                batch_out = batch_rates

            out_data[start:end] = batch_out

            # Clean up batch results
            del batch_out, batch
            gc.collect()

        except Exception as e:
            print(f"Error processing batch {start}-{end}: {e}")
            # Fill with zeros if batch fails
            out_data[start:end] = 0
            del batch
            gc.collect()

    # Clean up traces
    del traces
    gc.collect()

    return out_data


def create_visualization_pdf(
    calcium_data,
    prob_data,
    subject_name,
    output_path,
    num_neurons=10,
    stim=None,
    behavior=None,
    eye=None,
    is_probability=True,
    value_label=None,
):
    """
    Create visualization PDF showing calcium traces and spike probabilities.

    Args:
        calcium_data: (T, N) array of raw calcium data
        prob_data: (N, T) array of spike probabilities
        subject_name: Name of the subject
        output_path: Path to save PDF
        num_neurons: Number of neurons to visualize
    """
    T, N = calcium_data.shape

    # Compute subject-level statistics from prob_data (expected shape: (N, T))
    per_neuron_mean = None
    per_neuron_var = None
    per_frame_mean = None
    per_frame_var = None
    try:
        per_neuron_mean = prob_data.mean(axis=1)
        per_neuron_var = prob_data.var(axis=1)
        per_frame_mean = prob_data.mean(axis=0)
        per_frame_var = prob_data.var(axis=0)
        global_mean = float(prob_data.mean())
        global_var = float(prob_data.var())
        if is_probability:
            sparsity_frac = float((prob_data < 1e-3).sum()) / float(prob_data.size)
            active_frac = 1.0 - sparsity_frac
            frac_active_gt_1pct = float((prob_data > 1e-2).mean())
        else:
            sparsity_frac = active_frac = frac_active_gt_1pct = 0.0
        pn_mean_mean = float(per_neuron_mean.mean())
        pn_mean_median = float(np.median(per_neuron_mean))
        pn_var_mean = float(per_neuron_var.mean())
        pf_mean_mean = float(per_frame_mean.mean())
        pf_var_mean = float(per_frame_var.mean())
    except Exception:
        global_mean = global_var = sparsity_frac = active_frac = frac_active_gt_1pct = (
            0.0
        )
        pn_mean_mean = pn_mean_median = pn_var_mean = pf_mean_mean = pf_var_mean = 0.0

    print(f"  → Writing PDF to {output_path}")
    with PdfPages(output_path) as pdf:
        # Summary metrics page
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle(f"Subject: {subject_name} — Summary Metrics", fontsize=12)

            # Text panel
            ax0 = axes[0, 0]
            ax0.axis("off")
            lines = [
                f"Neurons (N): {N}",
                f"Timepoints (T): {T}",
                (
                    f"Global mean prob: {global_mean:.6f}"
                    if is_probability
                    else f"Global mean signal: {global_mean:.6f}"
                ),
                (
                    f"Global var prob: {global_var:.6f}"
                    if is_probability
                    else f"Global var signal: {global_var:.6f}"
                ),
                *(
                    [
                        f"Sparsity (<1e-3): {sparsity_frac:.4f}",
                        f"Active frac: {active_frac:.4f}",
                        f"Frac >1%: {frac_active_gt_1pct:.4f}",
                    ]
                    if is_probability
                    else []
                ),
                f"Per-neuron mean (mean/median): {pn_mean_mean:.6f} / {pn_mean_median:.6f}",
                f"Per-neuron var (mean): {pn_var_mean:.6f}",
                f"Per-frame mean (mean): {pf_mean_mean:.6f}",
                f"Per-frame var (mean): {pf_var_mean:.6f}",
            ]
            ax0.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9)

            # Histogram: per-neuron mean
            ax1 = axes[0, 1]
            if per_neuron_mean is not None:
                ax1.hist(per_neuron_mean, bins=50, color="steelblue", alpha=0.8)
                ax1.set_title(
                    "Histogram: per-neuron mean"
                    + (" prob" if is_probability else " signal")
                )
            else:
                ax1.axis("off")

            # Per-frame mean over time
            ax2 = axes[1, 0]
            if per_frame_mean is not None:
                ax2.plot(per_frame_mean, color="darkmagenta", lw=0.8)
                ax2.set_title(
                    "Per-frame mean "
                    + ("prob" if is_probability else "signal")
                    + " over time"
                )
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("Mean " + ("prob" if is_probability else "signal"))
            else:
                ax2.axis("off")

            # Histogram: per-neuron variance
            ax3 = axes[1, 1]
            if per_neuron_var is not None:
                ax3.hist(per_neuron_var, bins=50, color="orange", alpha=0.8)
                ax3.set_title("Histogram: per-neuron variance")
            else:
                ax3.axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        except Exception:
            pass

        # Helper to align auxiliary time series length to calcium T
        def _align_length(arr, time_len):
            try:
                if arr is None:
                    return None
                if hasattr(arr, "ndim"):
                    if arr.ndim == 1:
                        L = arr.shape[0]
                        if L != time_len:
                            Lnew = min(L, time_len)
                            return arr[:Lnew]
                        return arr
                    elif arr.ndim == 2:
                        L = arr.shape[1]
                        if L != time_len:
                            Lnew = min(L, time_len)
                            return arr[:, :Lnew]
                        return arr
                return arr
            except Exception:
                return None

        stim = _align_length(stim, T)
        behavior = _align_length(behavior, T)
        eye = _align_length(eye, T)

        # Stimulus page (overlay 2D features or step plot for 1D)
        if stim is not None:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                if (
                    hasattr(stim, "ndim")
                    and int(getattr(stim, "ndim", 1)) == 2
                    and stim.shape[1] > 1
                ):
                    # Expect (T, D). Overlay all columns.
                    T_st = stim.shape[0]
                    x = np.arange(T_st)
                    for j in range(stim.shape[1]):
                        ax.plot(
                            x, np.asarray(stim[:, j]).astype(float), lw=0.6, alpha=0.8
                        )
                    ax.set_title(f"Stimuli features (D={stim.shape[1]})")
                    ax.set_ylabel("Feature value")
                else:
                    ax.plot(
                        np.asarray(stim).astype(float),
                        drawstyle="steps-post",
                        color="tab:green",
                        lw=0.8,
                    )
                    ax.set_title("Stimulus over time")
                    ax.set_ylabel("Stimulus")
                ax.set_xlabel("Frame")
                ax.grid(alpha=0.2)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception:
                pass

        # Behavior page(s)
        if behavior is not None:
            try:
                B = behavior.shape[0] if behavior.ndim == 2 else 0
                if B > 0:
                    cols = 3
                    rows = int(np.ceil(B / cols))
                    fig, axes = plt.subplots(
                        rows, cols, figsize=(12, 3 * rows), sharex=True
                    )
                    axes = np.atleast_1d(axes).reshape(rows, cols)
                    for i in range(rows * cols):
                        r = i // cols
                        c = i % cols
                        ax = axes[r, c]
                        if i < B:
                            ax.plot(behavior[i], lw=0.8, color="tab:blue")
                            ax.set_title(f"Behavior {i + 1}")
                            ax.grid(alpha=0.2)
                        else:
                            ax.axis("off")
                    axes[-1, 0].set_xlabel("Frame")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

        # Eye tracking page(s)
        if eye is not None:
            try:
                E = eye.shape[0] if eye.ndim == 2 else 0
                if E > 0:
                    cols = min(E, 3)
                    rows = int(np.ceil(E / cols))
                    fig, axes = plt.subplots(
                        rows, cols, figsize=(12, 3 * rows), sharex=True
                    )
                    axes = np.atleast_1d(axes)
                    axes = axes.reshape(rows, cols)
                    for i in range(rows * cols):
                        r = i // cols
                        c = i % cols
                        ax = axes[r, c]
                        if i < E:
                            ax.plot(eye[i], lw=0.8, color="tab:orange")
                            ax.set_title(f"Eye {i + 1}")
                            ax.grid(alpha=0.2)
                        else:
                            ax.axis("off")
                    axes[-1, 0].set_xlabel("Frame")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

        # Select random neurons to visualize
        sel = np.random.choice(N, min(num_neurons, N), replace=False)

        for idx in sel:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            # Raw calcium trace
            ax[0].plot(calcium_data[:, idx], "k")
            ax[0].set_title(f"Neuron {idx}: Raw Calcium")
            ax[0].set_ylabel("Fluorescence")

            # Second panel: probabilities or processed calcium
            ax[1].plot(prob_data[idx], "m")
            if is_probability:
                ax[1].set_title("Spike Probabilities (CASCADE)")
                ax[1].set_ylabel("Probability")
            else:
                ax[1].set_title(
                    value_label
                    if value_label is not None
                    else "Processed Calcium (baseline-corrected)"
                )
                ax[1].set_ylabel("Signal")
            ax[1].set_xlabel("Frame")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def process_subject(subject_dir, config):
    """
    Process a single subject through the complete pipeline.

    Args:
        subject_dir: Path to subject directory containing raw data
        config: Configuration dictionary

    Returns:
        subject_output_dir: Path to subject's output directory
    """
    subject_name = os.path.basename(os.path.normpath(subject_dir))

    # Process raw calcium data
    print(
        f"\nProcessing subject: {subject_name} using {'CASCADE' if not config['processing'].get('skip_cascade', False) else 'z-scored calcium (skip CASCADE)'}"
    )

    # Create output directory
    os.makedirs(config["data"]["output_dir"], exist_ok=True)

    # Check if already processed
    final_file = os.path.join(config["data"]["output_dir"], f"{subject_name}.h5")
    pdf_file = os.path.join(
        config["data"]["output_dir"], f"{subject_name}_visualization.pdf"
    )

    if os.path.exists(final_file) and os.path.exists(pdf_file):
        print("  → Already processed; skipping.")
        return final_file

        start_time = time.time()

    try:
        # Load raw data
        print("  → Loading raw data...")

        # Load cell positions and additional datasets
        mat = loadmat(os.path.join(subject_dir, "data_full.mat"))
        data0 = mat["data"][0, 0]
        cell_xyz = data0["CellXYZ"]

        if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
            cell_xyz = cell_xyz[0, 0]

        # Try to load normalized positions if available
        cell_xyz_norm_available = False
        cell_xyz_norm = None
        if "CellXYZ_norm" in data0.dtype.names:
            try:
                cell_xyz_norm = data0["CellXYZ_norm"]
                if (
                    isinstance(cell_xyz_norm, np.ndarray)
                    and cell_xyz_norm.dtype == np.object_
                ):
                    cell_xyz_norm = cell_xyz_norm[0, 0]
                # Validate shape
                if (
                    isinstance(cell_xyz_norm, np.ndarray)
                    and cell_xyz_norm.shape == cell_xyz.shape
                ):
                    cell_xyz_norm_available = True
                    print("  → Found CellXYZ_norm; will use as base for normalization")
                else:
                    cell_xyz_norm = None
            except Exception:
                cell_xyz_norm = None

        # Load additional datasets (robust to missing fields)
        print("  → Loading additional datasets from MATLAB file...")

        def _get_field(obj, name):
            try:
                if name in obj.dtype.names:
                    val = obj[name]
                    if isinstance(val, np.ndarray) and val.dtype == np.object_:
                        val = val[0, 0]
                    return val
            except Exception:
                pass
            return None

        anat_stack = _get_field(data0, "anat_stack")

        # Sampling rate (fallback to config if missing)
        fpsec_val = _get_field(data0, "fpsec")
        if fpsec_val is not None:
            fpsec = float(fpsec_val.item() if hasattr(fpsec_val, "item") else fpsec_val)
        else:
            fpsec = float(config["processing"].get("original_sampling_rate") or 1.0)
            print(
                f"  → fpsec missing; using config original_sampling_rate = {fpsec} Hz"
            )

        # Stimulus data (may be missing)
        stim_full = _get_field(data0, "stim_full")
        if stim_full is not None:
            stim_full = np.squeeze(stim_full)

        # Behavioral data (may be missing)
        behavior_full = _get_field(data0, "Behavior_full")

        # Eye tracking data (may be missing)
        eye_full = _get_field(data0, "Eye_full")

        def _shape(x):
            try:
                return x.shape
            except Exception:
                return None

        print(f"  → Loaded anat_stack: {_shape(anat_stack)}, fpsec: {fpsec} Hz")
        print(
            f"  → Loaded stim_full: {_shape(stim_full)}, behavior_full: {_shape(behavior_full)}, eye_full: {_shape(eye_full)}"
        )

        # Load fluorescence traces
        calcium_dataset = config["data"]["calcium_dataset"]
        with h5py.File(os.path.join(subject_dir, "TimeSeries.h5"), "r") as f:
            if calcium_dataset not in f:
                raise ValueError(
                    f"Dataset '{calcium_dataset}' not found in TimeSeries.h5. Available datasets: {list(f.keys())}"
                )
            calcium = f[calcium_dataset][:]
            print(f"  → Using {calcium_dataset} traces")

            # Prefer alignment via absIX if available
            have_abs_ix = "absIX" in f
            if have_abs_ix:
                abs_ix = f["absIX"][:].flatten().astype(int) - 1  # 0-based
                N_abs = abs_ix.shape[0]
                # Ensure calcium is (T, N_abs)
                if calcium.shape[0] == N_abs and calcium.shape[1] != N_abs:
                    print(
                        f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)"
                    )
                    calcium = calcium.T
                elif calcium.shape[1] == N_abs:
                    print(f"  → Calcium data is already (T, N) = {calcium.shape}")
                else:
                    print(
                        f"  → Warning: absIX length {N_abs} does not match calcium shape {calcium.shape}; proceeding with calcium N"
                    )
                    N_abs = calcium.shape[1]
                    abs_ix = abs_ix[:N_abs]
                # If IX_inval_anat is provided, filter absIX/calcium consistently
                if "IX_inval_anat" in data0.dtype.names:
                    inval = data0["IX_inval_anat"]
                    try:
                        if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                            inval = inval[0, 0]
                        inval = np.array(inval).flatten().astype(int) - 1  # to 0-based
                        keep_mask_cols = ~np.isin(abs_ix, inval)
                        num_dropped = int((~keep_mask_cols).sum())
                        if num_dropped > 0:
                            print(
                                f"  → IX_inval_anat: dropping {num_dropped} neurons from calcium/positions via absIX filter"
                            )
                            calcium = calcium[:, keep_mask_cols]
                            abs_ix = abs_ix[keep_mask_cols]
                            N_abs = abs_ix.shape[0]
                    except Exception as e:
                        print(
                            f"  → Warning: IX_inval_anat filtering failed ({e}); continuing without it"
                        )
                # Select positions strictly by absIX to preserve index alignment
                positions_base = cell_xyz_norm if cell_xyz_norm_available else cell_xyz
                try:
                    cell_xyz = positions_base[abs_ix, :]
                except Exception as e:
                    print(
                        f"  → Error selecting positions by absIX: {e}; falling back to first N positions"
                    )
                    cell_xyz = positions_base[: calcium.shape[1], :]
                # Optional coordinate sanity: drop NaN positions in lockstep with calcium
                if np.isnan(cell_xyz).any():
                    coord_mask = ~np.isnan(cell_xyz).any(axis=1)
                    num_drop = int((~coord_mask).sum())
                    if num_drop > 0:
                        print(
                            f"  → Coordinate sanity: dropping {num_drop} NaN-position neurons"
                        )
                        cell_xyz = cell_xyz[coord_mask]
                        calcium = calcium[:, coord_mask]
                print(
                    f"  → Final aligned shapes - calcium: {calcium.shape}, cell_xyz: {cell_xyz.shape}"
                )
                T, N_original = calcium.shape
            else:
                # Check dimensions and transpose if needed using position count
                if calcium.shape[0] == cell_xyz.shape[0]:
                    print(
                        f"  → Calcium data is (N, T) = {calcium.shape}, transposing to (T, N)"
                    )
                    calcium = calcium.T
                elif calcium.shape[1] == cell_xyz.shape[0]:
                    print(f"  → Calcium data is already (T, N) = {calcium.shape}")
                else:
                    print(
                        f"  → Warning: Calcium shape {calcium.shape} doesn't match cell count {cell_xyz.shape[0]}"
                    )
                    # Fallback: truncate to shared min count to preserve index ordering
                    T_tmp, N_calcium = calcium.shape
                    N_positions = cell_xyz.shape[0]
                    N_shared = min(N_calcium, N_positions)
                    calcium = calcium[:, :N_shared]
                    cell_xyz = cell_xyz[:N_shared, :]
                    print(
                        f"  → Truncated to shared N={N_shared} to maintain index alignment"
                    )
                T, N_original = calcium.shape

            # If absIX was used, skip additional masking to preserve index alignment
            if have_abs_ix:
                N = calcium.shape[1]
                print(f"  → Retained {N} neurons (absIX-based alignment)")
            else:
                # Fallback simple coordinate NaN filter only if needed
                coordinate_mask = ~np.isnan(cell_xyz).any(axis=1)
                if not np.all(coordinate_mask):
                    calcium = calcium[:, coordinate_mask]
                    cell_xyz = cell_xyz[coordinate_mask]
                T, N = calcium.shape
                print(f"  → Retained {N} neurons after coordinate sanity check")

            # Prepare normalized positions (use CellXYZ_norm if available, else CellXYZ)
            # Always normalize to [0, 1] per axis after masking/selection
            # Select base positions for normalization preserving index alignment
            if cell_xyz_norm_available and have_abs_ix:
                base_positions = (
                    cell_xyz  # already selected via absIX and set into cell_xyz
                )
            elif cell_xyz_norm_available and not have_abs_ix:
                # cell_xyz corresponds to filtered set; map via retained indices is already applied
                base_positions = (
                    cell_xyz_norm[: cell_xyz.shape[0]]
                    if cell_xyz_norm.shape[0] != cell_xyz.shape[0]
                    else cell_xyz_norm
                )
            else:
                base_positions = cell_xyz

            # Apply test run neuron selection if specified
            test_run_neurons = config["data"].get("test_run_neurons")
            if test_run_neurons is not None and test_run_neurons < N:
                print(
                    f"  → Test run: selecting {test_run_neurons} random neurons out of {N}"
                )
                np.random.seed(config["processing"]["seed"])
                selected_indices = np.random.choice(N, test_run_neurons, replace=False)
                selected_indices = np.sort(selected_indices)

                calcium = calcium[:, selected_indices]
                cell_xyz = cell_xyz[selected_indices, :]
                base_positions = base_positions[selected_indices, :]

                T, N = calcium.shape
                print(f"  → Reduced to {N} neurons for testing")

            # Compute normalization stats and normalized positions in [0, 1]
            # Replace NaNs before normalization
            if np.isnan(base_positions).any():
                base_positions = np.nan_to_num(base_positions)

            pos_min = base_positions.min(axis=0)
            pos_max = base_positions.max(axis=0)
            pos_range = pos_max - pos_min
            pos_range[pos_range == 0] = 1.0
            norm_positions = (base_positions - pos_min) / pos_range
            # Safety: clip to [0, 1]
            norm_positions = np.clip(norm_positions, 0.0, 1.0)

        # (moved) We'll take a copy for visualization after any interpolation so lengths match

        # Interpolate calcium traces and temporal data if needed
        orig_rate = config["processing"]["original_sampling_rate"]
        target_rate = config["processing"]["target_sampling_rate"]
        return_to_original = config["processing"].get("return_to_original_rate", False)

        # Store original dimensions for potential downsampling later
        original_T = T
        upsampled_T = T

        if orig_rate is not None and target_rate is not None:
            print(
                f"  → Interpolating calcium traces from {orig_rate}Hz to {target_rate}Hz..."
            )

            # Create time points
            original_time = np.arange(T) / orig_rate
            upsampled_T = int(T * target_rate / orig_rate)
            new_time = np.arange(upsampled_T) / target_rate

            # Interpolate each neuron's trace using PCHIP
            from scipy.interpolate import PchipInterpolator

            calcium_interpolated = np.zeros((upsampled_T, N), dtype=calcium.dtype)

            for n in range(N):
                interp_func = PchipInterpolator(
                    original_time, calcium[:, n], extrapolate=True
                )
                calcium_interpolated[:, n] = interp_func(new_time)

            calcium = calcium_interpolated
            T = upsampled_T
            print(f"  → Interpolated calcium to {T} timepoints at {target_rate}Hz")

            # Only interpolate non-neural data if we're NOT returning to original rate
            if not return_to_original:
                # Interpolate temporal datasets to match using hold interpolation
                print(
                    f"  → Interpolating temporal datasets with hold interpolation (zero-order hold)..."
                )
                from scipy.interpolate import interp1d

                # Hold interpolation for stimulus data (1D) - preserves discrete values
                if stim_full is not None and stim_full.shape[0] == len(original_time):
                    stim_hold_interp = interp1d(
                        original_time,
                        stim_full.astype(float),
                        kind="previous",
                        bounds_error=False,
                        fill_value=(stim_full[0], stim_full[-1]),
                    )
                    stim_full = stim_hold_interp(new_time).astype(stim_full.dtype)

                # Hold interpolation for behavioral data (2D: behaviors x time)
                if behavior_full is not None and behavior_full.shape[1] == len(
                    original_time
                ):
                    behavior_interp = np.zeros(
                        (behavior_full.shape[0], upsampled_T), dtype=behavior_full.dtype
                    )
                    for b in range(behavior_full.shape[0]):
                        behav_hold_interp = interp1d(
                            original_time,
                            behavior_full[b, :],
                            kind="previous",
                            bounds_error=False,
                            fill_value=(behavior_full[b, 0], behavior_full[b, -1]),
                        )
                        behavior_interp[b, :] = behav_hold_interp(new_time)
                    behavior_full = behavior_interp

                # Hold interpolation for eye tracking data (2D: eye dimensions x time)
                if eye_full is not None and eye_full.shape[1] == len(original_time):
                    eye_interp = np.zeros(
                        (eye_full.shape[0], upsampled_T), dtype=eye_full.dtype
                    )
                    for e in range(eye_full.shape[0]):
                        eye_hold_interp = interp1d(
                            original_time,
                            eye_full[e, :],
                            kind="previous",
                            bounds_error=False,
                            fill_value=(eye_full[e, 0], eye_full[e, -1]),
                        )
                        eye_interp[e, :] = eye_hold_interp(new_time)
                    eye_full = eye_interp

                print(
                    f"  → Interpolated temporal data: stim_full {_shape(stim_full)}, behavior_full {_shape(behavior_full)}, eye_full {_shape(eye_full)}"
                )
            else:
                print(
                    f"  → Keeping temporal data at original rate (will downsample neural data later)"
                )

        # Use actual sampling rate for further processing
        effective_sampling_rate = target_rate or orig_rate or fpsec

        # Now take a copy for visualization that matches current calcium timeline
        calcium_for_viz = calcium[
            :, : min(config["processing"]["num_neurons_viz"] * 2, N)
        ].copy()

        # Depending on config, either run CASCADE (with baseline correction) or skip and use z-scored calcium directly
        skip_cascade = bool(config["processing"].get("skip_cascade", False))
        if not skip_cascade:
            # Apply baseline correction based on configuration (for CASCADE mode only)
            processed_calcium = compute_baseline_correction(
                calcium,
                config["data"]["window_length"],
                config["data"]["baseline_percentile"],
                effective_sampling_rate,
                config["data"]["is_raw"],
                config["data"]["apply_baseline_subtraction"],
            )

            # Clean up calcium to free memory
            del calcium
            gc.collect()

            convert_flag = bool(
                config["processing"].get("convert_rates_to_probabilities", True)
            )
            prob_data = run_cascade_inference(
                processed_calcium,
                config["processing"]["batch_size"],
                config["cascade"]["model_type"],
                effective_sampling_rate,
                convert_rates_to_probabilities=convert_flag,
            )
            # Clean up after CASCADE
            del processed_calcium
            gc.collect()
            zscore_mean = None
            zscore_std = None
        else:
            # No baseline correction in no-cascade mode; z-score normalize calcium per neuron
            # calcium shape: (T, N) → compute per-neuron params along axis=0
            zscore_mean = np.mean(calcium, axis=0).astype(np.float32)
            zscore_std = np.std(calcium, axis=0).astype(np.float32)
            std_safe = np.where(zscore_std == 0.0, 1.0, zscore_std)
            z_calcium = (calcium - zscore_mean) / std_safe
            # Output is (N, T)
            prob_data = z_calcium.T
            # Clean up
            del z_calcium, calcium
            gc.collect()

        # Optionally downsample neural data back to original rate
        if (
            return_to_original
            and orig_rate is not None
            and target_rate is not None
            and orig_rate != target_rate
        ):
            print(
                f"  → Downsampling neural data from {target_rate}Hz back to {orig_rate}Hz with anti-aliasing..."
            )
            from scipy.signal import resample_poly
            from fractions import Fraction

            # Calculate the rational resampling factors
            # We upsampled by target_rate/orig_rate, so we downsample by orig_rate/target_rate
            rate_ratio = (
                Fraction(orig_rate).limit_denominator()
                / Fraction(target_rate).limit_denominator()
            )
            up_factor = rate_ratio.numerator
            down_factor = rate_ratio.denominator

            print(f"  → Downsampling with factors: up={up_factor}, down={down_factor}")

            # Transpose prob_data to (T, N) for processing
            prob_data_T = prob_data.T  # Shape: (T, N)

            # Downsample each neuron's probability trace
            downsampled_prob_data = np.zeros((original_T, N), dtype=prob_data.dtype)

            for n in tqdm(range(N), desc="Downsampling neurons"):
                # Use resample_poly with proper anti-aliasing
                # The function automatically applies appropriate low-pass filtering
                downsampled_prob_data[:, n] = resample_poly(
                    prob_data_T[:, n], up=up_factor, down=down_factor, axis=0
                )

            # Update prob_data back to (N, T) format and dimensions
            prob_data = downsampled_prob_data.T  # Back to (N, T)
            T = original_T
            effective_sampling_rate = orig_rate

            print(f"  → Downsampled to {T} timepoints at {effective_sampling_rate}Hz")

            # Clean up
            del prob_data_T, downsampled_prob_data
            gc.collect()

        # Prepare data for saving - transpose to (T, N) format and convert to configured dtype
        print("  → Preparing output data for saving...")
        output_dtype = getattr(np, config["output"]["dtype"])

        # Handle NaN values in output data
        prob_data = np.nan_to_num(prob_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Transpose to (T, N) and convert to specified dtype
        prob_data_transposed = prob_data.T.astype(output_dtype)  # (T, N)

        # Optionally compute per-neuron log(activity) mean/std for CASCADE outputs (rates or probabilities)
        log_activity_mean = None
        log_activity_std = None
        if (not skip_cascade) and bool(
            config["output"].get("compute_log_activity_stats", False)
        ):
            try:
                eps = float(config["output"].get("log_activity_eps", 1e-7))
                vals = np.maximum(
                    prob_data_transposed.astype(np.float32, copy=False), eps
                )
                logs = np.log(vals)
                log_activity_mean = logs.mean(axis=0).astype(np.float32)
                log_activity_std = logs.std(axis=0).astype(np.float32)
                del vals, logs
                gc.collect()
            except Exception as _e:
                log_activity_mean = None
                log_activity_std = None

        # Use normalized positions computed above
        cell_positions = norm_positions.astype(output_dtype)  # (N, 3) in [0, 1]

        # Save final data
        print(f"  → Saving final data to {final_file}")
        with h5py.File(final_file, "w") as f:
            # Save main datasets
            f.create_dataset(
                "neuron_values",
                data=prob_data_transposed,
                chunks=(min(1000, T), min(100, N)),
                compression="gzip",
                compression_opts=1,
            )

            f.create_dataset(
                "cell_positions",
                data=cell_positions,
                compression="gzip",
                compression_opts=1,
            )

            f.create_dataset("timepoint_indices", data=np.arange(T, dtype=np.int32))

            # Save metadata as datasets
            f.create_dataset("num_timepoints", data=T)
            f.create_dataset("num_neurons", data=N)

            # Sampling rate information (always included)
            f.create_dataset("original_sampling_rate_hz", data=fpsec)

            # Save z-score parameters if in no-cascade mode
            if skip_cascade and zscore_mean is not None and zscore_std is not None:
                f.create_dataset(
                    "zscore_mean",
                    data=zscore_mean.astype(output_dtype),
                    compression="gzip",
                    compression_opts=1,
                )
                f.create_dataset(
                    "zscore_std",
                    data=zscore_std.astype(output_dtype),
                    compression="gzip",
                    compression_opts=1,
                )

            # Save log(activity) stats if computed
            if (log_activity_mean is not None) and (log_activity_std is not None):
                f.create_dataset(
                    "log_activity_mean",
                    data=log_activity_mean,
                    compression="gzip",
                    compression_opts=1,
                )
                f.create_dataset(
                    "log_activity_std",
                    data=log_activity_std,
                    compression="gzip",
                    compression_opts=1,
                )

            # Save additional datasets if requested
            if config["output"].get("include_additional_data", True):
                print(f"  → Saving additional datasets...")

                # Anatomical reference stack (ensure numeric dtype)
                if anat_stack is not None:
                    try:
                        print(
                            f"  → Saving anat_stack (type={type(anat_stack)}, dtype={getattr(anat_stack, 'dtype', None)}, shape={getattr(anat_stack, 'shape', None)})"
                        )
                    except Exception:
                        pass
                    try:
                        if (
                            isinstance(anat_stack, np.ndarray)
                            and anat_stack.dtype == np.object_
                        ):
                            anat_stack = np.asarray(anat_stack, dtype=np.float32)
                        f.create_dataset(
                            "anat_stack",
                            data=anat_stack,
                            compression="gzip",
                            compression_opts=1,
                        )
                    except Exception as e:
                        print(
                            f"  → Warning: skipping anat_stack save due to dtype issue: {e}"
                        )

                # Temporal data (stimulus, behavior, eye tracking)
                # Save stimulus_full as one-hot float array directly (T, K)
                K = None
                if stim_full is not None:
                    try:
                        print(
                            f"  → Preparing stimulus_full (type={type(stim_full)}, dtype={getattr(stim_full, 'dtype', None)}, shape={getattr(stim_full, 'shape', None)})"
                        )
                    except Exception:
                        pass
                    try:
                        stim_int = np.asarray(stim_full).astype(np.int64).reshape(-1)
                        unique_labels = np.unique(stim_int)
                        label_to_compact = {
                            int(lbl): i for i, lbl in enumerate(unique_labels.tolist())
                        }
                        compact_labels = np.vectorize(
                            lambda x: label_to_compact[int(x)]
                        )(stim_int)
                        K = int(len(unique_labels))
                        stim_onehot = np.eye(K, dtype=np.float32)[
                            compact_labels
                        ]  # (T, K)
                    except Exception:
                        K = 1
                        stim_onehot = np.zeros(
                            (stim_full.shape[0], K), dtype=np.float32
                        )
                    try:
                        print(
                            f"  → Saving stimulus_full one-hot (dtype={stim_onehot.dtype}, shape={getattr(stim_onehot, 'shape', None)})"
                        )
                        stim_onehot = np.asarray(stim_onehot, dtype=np.float32)
                        f.create_dataset(
                            "stimulus_full",
                            data=stim_onehot,
                            compression="gzip",
                            compression_opts=1,
                        )
                        f.attrs["stimulus_num_classes"] = int(K)
                    except Exception as e:
                        print(
                            f"  → Warning: skipping stimulus_full save due to dtype issue: {e}"
                        )

                if behavior_full is not None:
                    try:
                        print(
                            f"  → Saving behavior_full (type={type(behavior_full)}, dtype={getattr(behavior_full, 'dtype', None)}, shape={getattr(behavior_full, 'shape', None)})"
                        )
                    except Exception:
                        pass
                    try:
                        if (
                            isinstance(behavior_full, np.ndarray)
                            and behavior_full.dtype == np.object_
                        ):
                            behavior_full = np.asarray(behavior_full, dtype=np.float32)
                        f.create_dataset(
                            "behavior_full",
                            data=behavior_full,
                            compression="gzip",
                            compression_opts=1,
                        )
                    except Exception as e:
                        print(
                            f"  → Warning: skipping behavior_full save due to dtype issue: {e}"
                        )

                if eye_full is not None:
                    try:
                        print(
                            f"  → Saving eye_full (type={type(eye_full)}, dtype={getattr(eye_full, 'dtype', None)}, shape={getattr(eye_full, 'shape', None)})"
                        )
                    except Exception:
                        pass
                    try:
                        if (
                            isinstance(eye_full, np.ndarray)
                            and eye_full.dtype == np.object_
                        ):
                            eye_full = np.asarray(eye_full, dtype=np.float32)
                        f.create_dataset(
                            "eye_full",
                            data=eye_full,
                            compression="gzip",
                            compression_opts=1,
                        )
                    except Exception as e:
                        print(
                            f"  → Warning: skipping eye_full save due to dtype issue: {e}"
                        )
            else:
                print(f"  → Skipping additional datasets per configuration")

            # Save attributes
            f.attrs["subject"] = subject_name
            f.attrs["data_source"] = "raw_calcium"
            f.attrs["spike_dtype"] = config["output"]["dtype"]
            f.attrs["position_dtype"] = config["output"]["dtype"]
            f.attrs["positions_normalized"] = True
            f.attrs["positions_source"] = (
                "CellXYZ_norm"
                if cell_xyz_norm_available
                else "CellXYZ_normalized_runtime"
            )
            f.attrs["positions_min"] = pos_min.astype(np.float32)
            f.attrs["positions_max"] = pos_max.astype(np.float32)
            f.attrs["cascade_model"] = (
                config["cascade"]["model_type"] if not skip_cascade else "skipped"
            )
            f.attrs["calcium_dataset"] = config["data"]["calcium_dataset"]
            f.attrs["is_raw"] = config["data"]["is_raw"]
            f.attrs["apply_baseline_subtraction"] = config["data"][
                "apply_baseline_subtraction"
            ]
            f.attrs["window_length"] = config["data"]["window_length"]
            f.attrs["baseline_percentile"] = config["data"]["baseline_percentile"]
            f.attrs["skip_cascade"] = skip_cascade
            # Ensure sampling rate attributes are numeric (h5py cannot store None)
            _orig_sr_attr = config["processing"].get("original_sampling_rate")
            _orig_sr_attr = (
                float(_orig_sr_attr) if _orig_sr_attr is not None else float(fpsec)
            )
            _tgt_sr_attr = config["processing"].get("target_sampling_rate")
            _tgt_sr_attr = float(_tgt_sr_attr) if _tgt_sr_attr is not None else np.nan
            f.attrs["original_sampling_rate"] = _orig_sr_attr
            f.attrs["target_sampling_rate"] = _tgt_sr_attr
            f.attrs["effective_sampling_rate"] = effective_sampling_rate
            f.attrs["matlab_fpsec"] = fpsec
            f.attrs["includes_additional_data"] = config["output"].get(
                "include_additional_data", True
            )
            f.attrs["return_to_original_rate"] = return_to_original
            f.attrs["final_sampling_rate"] = (
                effective_sampling_rate  # The actual final rate of all data
            )
            # Semantics of neuron_values
            f.attrs["neuron_values_semantics"] = (
                "probabilities"
                if (
                    (not skip_cascade)
                    and bool(
                        config["processing"].get("convert_rates_to_probabilities", True)
                    )
                )
                else ("rates_hz" if (not skip_cascade) else "zscored_signal")
            )
            # Log stats metadata
            if (log_activity_mean is not None) and (log_activity_std is not None):
                f.attrs["includes_log_activity_stats"] = True
                f.attrs["log_activity_eps"] = float(
                    config["output"].get("log_activity_eps", 1e-7)
                )
            else:
                f.attrs["includes_log_activity_stats"] = False

        # Add per-neuron global IDs (random 64-bit) for embedding lookup
        try:
            rng = np.random.default_rng(seed=None)
            neuron_global_ids = rng.integers(
                low=1, high=np.iinfo(np.int64).max, size=(N,), dtype=np.int64
            )
            with h5py.File(final_file, "a") as f:
                if "neuron_global_ids" not in f:
                    f.create_dataset("neuron_global_ids", data=neuron_global_ids)
        except Exception as _e:
            pass

        # Create visualization PDF
        print("  → Creating visualization PDF...")
        is_probability = (not skip_cascade) and bool(
            config["processing"].get("convert_rates_to_probabilities", True)
        )
        value_label = (
            "Spike Rates (Hz)"
            if (not skip_cascade and not is_probability)
            else ("Z-scored Calcium" if skip_cascade else None)
        )
        create_visualization_pdf(
            calcium_for_viz,
            prob_data,
            subject_name,
            pdf_file,
            config["processing"]["num_neurons_viz"],
            stim=(
                stimuli_features
                if "stimuli_features" in locals() and stimuli_features is not None
                else (stim_full if stim_full is not None else None)
            ),
            behavior=behavior_full,
            eye=eye_full,
            is_probability=is_probability,
            value_label=value_label,
        )

        # Clean up memory
        del calcium_for_viz, prob_data, prob_data_transposed, cell_positions
        del anat_stack, stim_full, behavior_full, eye_full
        gc.collect()

        processing_time = time.time() - start_time
        print(f"  → Completed {subject_name} in {processing_time:.2f} seconds")

        return final_file

    except Exception as e:
        print(f"  → Error processing {subject_name}: {e}")
        raise


def analyze_raw_data(config):
    """
    Analyze and output information about raw data without processing.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("RAW DATA ANALYSIS")
    print("=" * 80)

    # Find subjects
    subjects = sorted(
        [
            os.path.join(config["data"]["input_dir"], d)
            for d in os.listdir(config["data"]["input_dir"])
            if d.startswith("subject_")
            and d not in config["data"].get("skip_subjects", [])
            and os.path.isdir(os.path.join(config["data"]["input_dir"], d))
        ]
    )

    if not subjects:
        print(f"No subjects found in {config['data']['input_dir']}")
        return

    print(f"Found {len(subjects)} subjects in {config['data']['input_dir']}")
    print(f"Skipping subjects: {config['data'].get('skip_subjects', [])}")
    print()

    # Analyze each subject
    for subject_dir in subjects:
        subject_name = os.path.basename(os.path.normpath(subject_dir))
        print(f"Subject: {subject_name}")
        print("-" * 40)

        try:
            # Load cell positions
            mat_file = os.path.join(subject_dir, "data_full.mat")
            if not os.path.exists(mat_file):
                print(f"  ERROR: data_full.mat not found")
                continue

            mat = loadmat(mat_file)
            data0 = mat["data"][0, 0]
            cell_xyz = data0["CellXYZ"]

            if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
                cell_xyz = cell_xyz[0, 0]

            print(f"  Cell positions shape: {cell_xyz.shape}")

            # Analyze cell positions
            valid_coords = ~np.isnan(cell_xyz).any(axis=1)
            num_valid = np.sum(valid_coords)
            num_invalid = np.sum(~valid_coords)

            print(f"  Valid coordinate neurons: {num_valid}")
            print(f"  Invalid coordinate neurons: {num_invalid}")

            if num_valid > 0:
                valid_positions = cell_xyz[valid_coords]

                # Spatial bounds
                min_coords = valid_positions.min(axis=0)
                max_coords = valid_positions.max(axis=0)
                ranges = max_coords - min_coords

                print(f"  Spatial bounds:")
                print(
                    f"    X: {min_coords[0]:.2f} to {max_coords[0]:.2f} (range: {ranges[0]:.2f})"
                )
                print(
                    f"    Y: {min_coords[1]:.2f} to {max_coords[1]:.2f} (range: {ranges[1]:.2f})"
                )
                print(
                    f"    Z: {min_coords[2]:.2f} to {max_coords[2]:.2f} (range: {ranges[2]:.2f})"
                )

                # Spatial statistics
                mean_coords = valid_positions.mean(axis=0)
                std_coords = valid_positions.std(axis=0)

                print(f"  Spatial statistics:")
                print(
                    f"    Mean: X={mean_coords[0]:.2f}, Y={mean_coords[1]:.2f}, Z={mean_coords[2]:.2f}"
                )
                print(
                    f"    Std:  X={std_coords[0]:.2f}, Y={std_coords[1]:.2f}, Z={std_coords[2]:.2f}"
                )

            # Check for invalid anatomical indices
            if "IX_inval_anat" in data0.dtype.names:
                inval = data0["IX_inval_anat"]
                if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
                    inval = inval[0, 0].flatten()
                inval_indices = np.array(inval, int) - 1  # Convert to 0-based
                print(f"  Invalid anatomical indices: {len(inval_indices)} neurons")
            else:
                print(f"  Invalid anatomical indices: None specified")

            # Load calcium data
            timeseries_file = os.path.join(subject_dir, "TimeSeries.h5")
            if not os.path.exists(timeseries_file):
                print(f"  ERROR: TimeSeries.h5 not found")
                continue

            calcium_dataset = config["data"]["calcium_dataset"]
            with h5py.File(timeseries_file, "r") as f:
                print(f"  Available datasets in TimeSeries.h5: {list(f.keys())}")

                if calcium_dataset not in f:
                    print(f"  ERROR: Dataset '{calcium_dataset}' not found")
                    continue

                calcium_shape = f[calcium_dataset].shape
                calcium_dtype = f[calcium_dataset].dtype

                print(f"  Calcium dataset '{calcium_dataset}':")
                print(f"    Shape: {calcium_shape}")
                print(f"    Data type: {calcium_dtype}")

                # Determine if transposition is needed
                if calcium_shape[0] == cell_xyz.shape[0]:
                    print(f"    Format: (N, T) - will be transposed to (T, N)")
                    T, N = calcium_shape[1], calcium_shape[0]
                elif calcium_shape[1] == cell_xyz.shape[0]:
                    print(f"    Format: (T, N) - correct format")
                    T, N = calcium_shape[0], calcium_shape[1]
                else:
                    print(
                        f"    WARNING: Shape mismatch with cell positions ({cell_xyz.shape[0]} cells)"
                    )
                    T, N = calcium_shape[0], calcium_shape[1]

                print(f"    Interpreted as: {T} timepoints, {N} neurons")

                # Sampling rate information
                orig_rate = config["processing"].get("original_sampling_rate")
                target_rate = config["processing"].get("target_sampling_rate")

                if orig_rate:
                    duration = T / orig_rate
                    print(f"    Duration: {duration:.2f} seconds at {orig_rate} Hz")

                    if target_rate and target_rate != orig_rate:
                        new_T = int(T * target_rate / orig_rate)
                        new_duration = new_T / target_rate
                        print(
                            f"    After resampling: {new_T} timepoints at {target_rate} Hz ({new_duration:.2f} seconds)"
                        )

                # Memory requirements estimation
                memory_gb = (T * N * 4) / (1024**3)  # Assuming float32
                print(f"    Memory requirement: ~{memory_gb:.2f} GB (float32)")

                # Sample some data statistics
                sample_size = min(1000, T)
                sample_indices = np.random.choice(T, sample_size, replace=False)
                sample_indices = np.sort(sample_indices)  # HDF5 requires sorted indices
                sample_data = f[calcium_dataset][sample_indices, : min(1000, N)]

                print(
                    f"    Sample statistics (first {min(1000, N)} neurons, {sample_size} timepoints):"
                )
                print(f"      Min: {sample_data.min():.4f}")
                print(f"      Max: {sample_data.max():.4f}")
                print(f"      Mean: {sample_data.mean():.4f}")
                print(f"      Std: {sample_data.std():.4f}")

                # Check for NaN/inf values
                num_nan = np.sum(np.isnan(sample_data))
                num_inf = np.sum(np.isinf(sample_data))
                if num_nan > 0 or num_inf > 0:
                    print(
                        f"      WARNING: Found {num_nan} NaN and {num_inf} inf values in sample"
                    )

            # Additional MATLAB data analysis
            print(f"  Additional MATLAB datasets:")

            # Sampling rate from MATLAB
            fpsec_val = data0["fpsec"]
            if isinstance(fpsec_val, np.ndarray) and fpsec_val.dtype == np.object_:
                fpsec_val = fpsec_val[0, 0]
            fpsec_val = float(
                fpsec_val.item() if hasattr(fpsec_val, "item") else fpsec_val
            )
            print(f"    MATLAB sampling rate (fpsec): {fpsec_val} Hz")

            # Anatomical stack
            anat_stack = data0["anat_stack"]
            if isinstance(anat_stack, np.ndarray) and anat_stack.dtype == np.object_:
                anat_stack = anat_stack[0, 0]
            print(f"    Anatomical stack: {anat_stack.shape} ({anat_stack.dtype})")
            anat_memory_gb = (np.prod(anat_stack.shape) * anat_stack.itemsize) / (
                1024**3
            )
            print(f"      Memory requirement: ~{anat_memory_gb:.3f} GB")

            # Stimulus data
            stim_full = data0["stim_full"]
            if isinstance(stim_full, np.ndarray) and stim_full.dtype == np.object_:
                stim_full = stim_full[0, 0]
            stim_full = np.squeeze(stim_full)
            print(f"    Stimulus data: {stim_full.shape} ({stim_full.dtype})")
            print(f"      Values: {stim_full.min()} to {stim_full.max()}")

            # Behavioral data
            behavior_full = data0["Behavior_full"]
            if (
                isinstance(behavior_full, np.ndarray)
                and behavior_full.dtype == np.object_
            ):
                behavior_full = behavior_full[0, 0]
            print(f"    Behavioral data: {behavior_full.shape} ({behavior_full.dtype})")
            print(
                f"      {behavior_full.shape[0]} behavioral variables over {behavior_full.shape[1]} timepoints"
            )

            # Eye tracking data
            eye_full = data0["Eye_full"]
            if isinstance(eye_full, np.ndarray) and eye_full.dtype == np.object_:
                eye_full = eye_full[0, 0]
            print(f"    Eye tracking data: {eye_full.shape} ({eye_full.dtype})")
            print(
                f"      {eye_full.shape[0]} eye dimensions over {eye_full.shape[1]} timepoints"
            )

            # Check temporal alignment
            if config["output"].get("include_additional_data", True):
                temporal_datasets = [
                    (
                        "stimulus",
                        stim_full.shape[0]
                        if stim_full.ndim == 1
                        else stim_full.shape[1],
                    ),
                    ("behavior", behavior_full.shape[1]),
                    ("eye", eye_full.shape[1]),
                ]
                print(f"    Temporal alignment check:")
                for name, length in temporal_datasets:
                    print(
                        f"      {name}: {length} timepoints ({'✓' if length == T else '⚠'} vs calcium {T})"
                    )

            # Output format preview
            output_dtype = config["output"]["dtype"]

            print(f"  Output format settings:")
            print(f"    Data type: {output_dtype}")

            if num_valid > 0:
                # Estimate spike probability memory requirements - (T, N) format
                prob_memory_gb = (
                    T * num_valid * (2 if output_dtype == "float16" else 4)
                ) / (1024**3)
                print(
                    f"    Spike probabilities memory requirement: ~{prob_memory_gb:.2f} GB"
                )

                # Estimate cell positions memory requirements - (N, 3) format
                pos_memory_gb = (
                    num_valid * 3 * (2 if output_dtype == "float16" else 4)
                ) / (1024**3)
                print(f"    Cell positions memory requirement: ~{pos_memory_gb:.2f} GB")

                total_memory_gb = prob_memory_gb + pos_memory_gb
                print(f"    Total output memory requirement: ~{total_memory_gb:.2f} GB")

                print(
                    f"    Output format: Spike probabilities (T={T}, N={num_valid}), Positions (N={num_valid}, 3)"
                )

        except Exception as e:
            print(f"  ERROR: {e}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Spike Probability Processing Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--create-default-config",
        type=str,
        default=None,
        help="Create default configuration file at specified path and exit",
    )
    parser.add_argument(
        "--raw_data_info",
        action="store_true",
        help="Analyze and display raw data information without processing",
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_default_config:
        save_default_config(args.create_default_config)
        return

    # Validate that config is provided when not creating default config
    if args.config is None:
        print("Error: --config is required when not using --create-default-config")
        print("Use --create-default-config to create a default configuration file")
        return

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Use --create-default-config to create a default configuration file")
        return

    config = load_config(args.config)

    # Handle raw data info request
    if args.raw_data_info:
        analyze_raw_data(config)
        return

    # If using ZapBench data, route to the specialized path and skip CASCADE validations
    if bool(config["data"].get("zapdata", False)):
        try:
            result = process_zapbench(config)
            if result:
                print(f"ZapBench processing complete: {result}")
        except Exception as e:
            print(f"Error processing ZapBench data: {e}")
            raise
        return

    # Validate CASCADE availability when needed (non-ZapBench path)
    if not config["processing"].get("skip_cascade", False):
        if not CASCADE_AVAILABLE:
            print(
                "Error: CASCADE not available. Please install neuralib or set processing.skip_cascade: true"
            )
            return

    # Validate required parameters
    require_rates = not config["processing"].get("skip_cascade", False)
    if require_rates and (
        not config["processing"].get("original_sampling_rate")
        or not config["processing"].get("target_sampling_rate")
    ):
        print(
            "Error: original_sampling_rate and target_sampling_rate are required unless processing.skip_cascade is true"
        )
        return

    # Create output directory
    os.makedirs(config["data"]["output_dir"], exist_ok=True)

    # Find subjects in raw data directory
    subjects = sorted(
        [
            os.path.join(config["data"]["input_dir"], d)
            for d in os.listdir(config["data"]["input_dir"])
            if d.startswith("subject_")
            and d not in config["data"].get("skip_subjects", [])
            and os.path.isdir(os.path.join(config["data"]["input_dir"], d))
        ]
    )

    if not subjects:
        print(f"No subjects found in {config['data']['input_dir']}")
        return

    print(f"Found {len(subjects)} subjects; skipping {config['data']['skip_subjects']}")

    if config["data"].get("test_run_neurons"):
        print(
            f"Test mode: Using only {config['data']['test_run_neurons']} randomly selected neurons per subject"
        )

    # Process subjects
    processed_subjects = []

    # Process subjects sequentially to avoid memory issues with CASCADE
    print("Processing subjects sequentially")
    for subject_dir in subjects:
        try:
            result = process_subject(subject_dir, config)
            if result is not None:
                processed_subjects.append(result)

                # Force garbage collection between subjects
                gc.collect()
                print_memory_stats("After processing subject")

        except Exception as e:
            print(f"Error processing {subject_dir}: {e}")
        gc.collect()

    print(f"\nProcessing complete!")
    print(f"Processed {len(processed_subjects)} subjects")
    if config["data"].get("test_run_neurons"):
        print(
            f"Test mode: Used only {config['data']['test_run_neurons']} randomly selected neurons per subject"
        )
    print(f"Output saved to: {config['data']['output_dir']}")
    print(f"Output format: Neuron values (T, N) and cell positions (N, 3)")
    print(f"Data type: {config['output']['dtype']}")


# -----------------------------
# ZapBench (Zarr) data support
# -----------------------------


def process_zapbench(config):
    """
    Process ZapBench Zarr stores: traces (T,N) and segmentation_xy (X,Y,Z labels) to produce
    an H5 with datasets: neuron_values (T,N_kept), cell_positions (N_kept,3) in [0,1].

    - Computes per-label 3D centroids (x,y,z) over all labeled voxels (no projection to 2D)
    - Aligns labels 1..N to trace columns 0..N-1; drops neurons without positions
    - Streams writes to H5 to avoid loading entire traces into memory
    """
    import numpy as np
    import h5py
    from pathlib import Path

    try:
        from zarrita.array import Array  # type: ignore
        from zarrita.store import LocalStore  # type: ignore
        from zarrita.group import make_store_path  # type: ignore
    except Exception as e:
        raise ImportError(
            "Reading Zarr v3 requires 'zarrita' (pip install zarrita)"
        ) from e

    zap_cfg = config["data"].get("zap", {})
    traces_store = Path(
        zap_cfg.get("traces_store", "/home/user/gbm3/GBM3/zapdata/traces")
    )
    seg_store = Path(
        zap_cfg.get(
            "segmentation_xy_store", "/home/user/gbm3/GBM3/zapdata/segmentation_xy"
        )
    )
    subject_name = str(zap_cfg.get("subject_name", "zapbench"))
    # Optional stimuli sources
    stim_store = Path(
        zap_cfg.get(
            "stimuli_store",
            "/home/user/gbm3/GBM3/zapdata/stimuli_raw/stimuli_and_ephys.10chFlt",
        )
    )
    stim_feat_store = Path(
        zap_cfg.get(
            "stimuli_features_store", "/home/user/gbm3/GBM3/zapdata/stimuli_features"
        )
    )

    print(f"Loading ZapBench traces from: {traces_store}")
    tr = Array.open(make_store_path(LocalStore(str(traces_store))))
    if len(tr.shape) != 2:
        raise ValueError(f"Expected traces shape (T,N); got {tr.shape}")
    T, N = int(tr.shape[0]), int(tr.shape[1])
    print(f"  Traces shape: (T={T}, N={N})")

    print(f"Loading ZapBench segmentation from: {seg_store}")
    seg = Array.open(make_store_path(LocalStore(str(seg_store))))
    if len(seg.shape) != 3:
        raise ValueError(f"Expected segmentation_xy shape (X,Y,Z); got {seg.shape}")
    X, Y, Z = int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])
    print(f"  Segmentation shape: (X={X}, Y={Y}, Z={Z})")

    # Stimuli loading (optional)
    stimuli_labels = None  # (T,) int; derived from condition identity dims
    stimuli_onehot = None  # (T, K) float32
    stimuli_features = None  # (T, 26) float32
    # Preferred: 26-D stimuli_features Zarr v2
    if stim_feat_store.exists():
        try:
            import zarr

            sf = zarr.open_array(str(stim_feat_store), mode="r")
            if sf.ndim != 2 or sf.shape[1] != 26:
                raise ValueError(
                    f"stimuli_features shape expected (T,26), got {sf.shape}"
                )
            stimuli_features = np.asarray(sf, dtype=np.float32)
            # Trim/pad to T
            if stimuli_features.shape[0] != T:
                L = min(T, stimuli_features.shape[0])
                stimuli_features = (
                    np.pad(stimuli_features[:L], ((0, T - L), (0, 0)), mode="edge")
                    if L < T
                    else stimuli_features[:T]
                )
            # Derive condition identity labels using fixed identity indices per spec.
            # Spec (1-based dims converted to 0-based indices):
            #  - After each encoding, a single binary dim indicates the condition identity.
            #    Examples given: Gain identity at dim 2 (index 1), Dots at dim 4 (index 3), Flash at dim 6 (index 5),
            #    Taxis uses dims 7-8 for left/right intensities; identity use dim 9? (index 8) is unclear.
            #    Turning uses dims 10-12 with 10 as velocity (continuous) and 11-12 sincos; identity likely dim 13 (index 12) not stated.
            #    Position uses 14-16 one-hot, with 17 delay; identity likely a separate binary (dim 18? index 17) per text.
            #    Open loop identity at dim 19 (index 18), Rotation identity at 20 (index 19), Dark identity at 22 (index 21).
            # Given ambiguities, we will robustly infer identity dims as binary columns that toggle piecewise-constantly.
            vals = stimuli_features
            is01 = (vals.min(axis=0) >= 0.0) & (vals.max(axis=0) <= 1.0)
            candidate_ids = np.where(is01)[0].tolist()
            if candidate_ids:
                # Prefer explicit identity indices from spec when present (0-based):
                # gain=1, dots=3, flash=5, taxis=8, turning=12, position=17, open_loop=18, rotation=20, dark=21
                identity_idx = [1, 3, 5, 8, 12, 17, 18, 20, 21]
                ids = [i for i in identity_idx if i < vals.shape[1]]
                use = ids if ids else candidate_ids
                # Map used identity indices to canonical condition names
                name_map_all = {
                    1: "gain",
                    3: "dots",
                    5: "flash",
                    8: "taxis",
                    12: "turning",
                    17: "position",
                    18: "open_loop",
                    20: "rotation",
                    21: "dark",
                }
                used_names = [name_map_all.get(i, f"cond{i}") for i in use]
                active = vals[:, use]
                idx = np.argmax(active, axis=1)
                maxv = active[np.arange(active.shape[0]), idx]
                labels = np.array(use, dtype=np.int64)[idx]
                # backfill when no identity set
                last = labels[0]
                for t in range(active.shape[0]):
                    if maxv[t] < 0.5:
                        labels[t] = last
                    else:
                        last = labels[t]
                uniq = np.unique(labels)
                # Preserve the order of `use` for compactification to keep names aligned
                order = [u for u in use if u in uniq]
                mapc = {int(lbl): i for i, lbl in enumerate(order)}
                compact = np.vectorize(lambda x: mapc[int(x)])(labels)
                K = len(order)
                stimuli_onehot = np.eye(K, dtype=np.float32)[compact]
                stimuli_labels = labels
                print(
                    f"  Stimuli features loaded: shape={vals.shape}; derived K={K} identity labels from {len(use)} identity dims"
                )
                # Save names and identity indices for downstream consumers
                condition_names = [name_map_all.get(i, f"cond{i}") for i in order]
                identity_indices_used = order
        except Exception as e:
            print(
                f"  → Warning: failed to load stimuli_features ({e}); falling back to stimuli_store if available"
            )

    # Fallback: stimuli_store labels or 10ch raw
    if (stimuli_onehot is None) and stim_store.exists():
        try:
            print(f"Loading ZapBench stimuli from: {stim_store}")
            # Try Zarr v3 first; if it fails, fall back to flat float32 10-channel file
            st = None
            try:
                st = Array.open(make_store_path(LocalStore(str(stim_store))))
            except Exception:
                st = None
            if st is not None:
                if len(st.shape) == 1:
                    stimuli_labels = np.asarray(st[:]).astype(np.int64)
                elif len(st.shape) == 2:
                    stimuli_labels = np.asarray(st[:, 0]).astype(np.int64)
                else:
                    raise ValueError(f"Unsupported stimuli shape: {st.shape}")
                if stimuli_labels.shape[0] != T:
                    L = min(T, stimuli_labels.shape[0])
                    stimuli_labels = stimuli_labels[:L]
                    if L < T:
                        pad = np.full(
                            (T - L,), stimuli_labels[-1] if L > 0 else 0, dtype=np.int64
                        )
                        stimuli_labels = np.concatenate([stimuli_labels, pad], axis=0)
                uniq = np.unique(stimuli_labels)
                label_to_compact = {int(lbl): i for i, lbl in enumerate(uniq.tolist())}
                compact = np.vectorize(lambda x: label_to_compact[int(x)])(
                    stimuli_labels
                )
                K = int(len(uniq))
                stimuli_onehot = np.eye(K, dtype=np.float32)[compact]
                print(
                    f"  Stimuli loaded: K={K} classes, length={stimuli_labels.shape[0]}"
                )
            else:
                # Flat file fallback: float32 10-channel (Tstim, 10)
                size = os.path.getsize(str(stim_store))
                ch = 10
                item = 4
                Tstim = size // (ch * item)
                mm = np.memmap(
                    str(stim_store), dtype=np.float32, mode="r", shape=(Tstim, ch)
                )
                # Aggregate to neural frames by averaging over bins per frame
                # Map each neural frame t to [start:end) indices in stimuli
                edges = np.linspace(0, Tstim, num=T + 1, dtype=np.int64)
                means = np.zeros((T, ch), dtype=np.float32)
                for t_idx in range(T):
                    s = int(edges[t_idx])
                    e = int(edges[t_idx + 1])
                    if e <= s:
                        e = min(Tstim, s + 1)
                    mm_slice = mm[s:e]
                    means[t_idx] = mm_slice.mean(axis=0)
                # Heuristic label: combine binary channels (approx) and quantized analog ch2
                b4 = (means[:, 4] > 0.5).astype(np.int32)
                b6 = (means[:, 6] > 0.5).astype(np.int32)
                b9 = (means[:, 9] > 0.5).astype(np.int32)
                # Quantize ch2 (0..~4) into 5 bins
                ch2 = np.clip(means[:, 2], 0.0, 4.999)
                q2 = (ch2 // 1.0).astype(np.int32)  # 0..4
                bitmask = b4 + 2 * b6 + 4 * b9
                stimuli_labels = (q2 + 5 * bitmask).astype(np.int64)
                uniq = np.unique(stimuli_labels)
                mapc = {int(lbl): i for i, lbl in enumerate(uniq.tolist())}
                compact = np.vectorize(lambda x: mapc[int(x)])(stimuli_labels)
                K = int(len(uniq))
                stimuli_onehot = np.eye(K, dtype=np.float32)[compact]
                print(
                    f"  Stimuli derived from 10ch float: K={K} classes, length={stimuli_labels.shape[0]}"
                )
        except Exception as e:
            print(
                f"  → Warning: failed to load stimuli ({e}); continuing without stimuli"
            )

    # Compute one 3D centroid per label id (background=0) by averaging all voxel coordinates
    print("Computing per-neuron 3D centroids (x,y,z) over all labeled voxels…")
    sum_x = np.zeros(N + 1, dtype=np.float64)
    sum_y = np.zeros(N + 1, dtype=np.float64)
    sum_z = np.zeros(N + 1, dtype=np.float64)
    count = np.zeros(N + 1, dtype=np.int64)

    for z in range(Z):
        slab = np.asarray(seg[:, :, z : z + 1])[:, :, 0]
        nz = slab != 0
        if not nz.any():
            continue
        labels = slab[nz].astype(np.int64)
        xs, ys = np.nonzero(nz)
        # Accumulate per-label counts and coordinate sums
        cnt = np.bincount(labels, minlength=N + 1)
        sx = np.bincount(labels, weights=xs.astype(np.float64), minlength=N + 1)
        sy = np.bincount(labels, weights=ys.astype(np.float64), minlength=N + 1)
        sz = np.bincount(
            labels, weights=np.full(labels.shape[0], float(z)), minlength=N + 1
        )
        count[: cnt.size] += cnt
        sum_x[: sx.size] += sx
        sum_y[: sy.size] += sy
        sum_z[: sz.size] += sz

    has_pos = count > 0
    has_pos[0] = False  # background
    num_with_pos = int(has_pos.sum())
    print(f"  Neurons with positions: {num_with_pos} / {N}")
    if num_with_pos == 0:
        raise RuntimeError("No labeled neurons found in segmentation_xy")

    # Build positions for labels 1..N mapped to indices 0..N-1; drop missing
    cx = np.zeros(N, dtype=np.float32)
    cy = np.zeros(N, dtype=np.float32)
    cz = np.zeros(N, dtype=np.float32)
    valid_idx = np.where(has_pos)[0]
    cx[valid_idx - 1] = (sum_x[valid_idx] / count[valid_idx]).astype(np.float32)
    cy[valid_idx - 1] = (sum_y[valid_idx] / count[valid_idx]).astype(np.float32)
    cz[valid_idx - 1] = (sum_z[valid_idx] / count[valid_idx]).astype(np.float32)

    # Select only neurons that have positions (mask over 0..N-1)
    keep_mask = np.zeros(N, dtype=bool)
    keep_mask[valid_idx - 1] = True
    keep_indices = np.nonzero(keep_mask)[0]
    N_keep = int(keep_mask.sum())
    print(f"  Keeping {N_keep} neurons aligned between traces and segmentation")

    # Normalize positions to [0,1] using array bounds
    pos = np.stack([cx[keep_mask], cy[keep_mask], cz[keep_mask]], axis=1).astype(
        np.float32
    )
    mins = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    maxs = np.array(
        [max(1.0, X - 1), max(1.0, Y - 1), max(1.0, Z - 1)], dtype=np.float32
    )
    pos_norm = pos / maxs  # already >=0
    pos_norm = np.clip(pos_norm, 0.0, 1.0)

    # Prepare output H5
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    final_file = os.path.join(output_dir, f"{subject_name}.h5")
    pdf_file = os.path.join(output_dir, f"{subject_name}_visualization.pdf")
    print(f"Saving ZapBench output to {final_file}")

    output_dtype = getattr(np, config["output"]["dtype"])

    # Determine processing path
    skip_cascade = bool(config["processing"].get("skip_cascade", False))
    orig_rate = config["processing"].get("original_sampling_rate")
    tgt_rate = config["processing"].get("target_sampling_rate")
    return_to_original = bool(
        config["processing"].get("return_to_original_rate", False)
    )

    # Helper to assemble a small sample for the PDF
    def read_small_sample(K: int) -> np.ndarray:
        K = min(K, N_keep)
        vals = np.zeros((T, K), dtype=np.float32)
        chunk_T = min(1000, T)
        for s in range(0, T, chunk_T):
            e = min(T, s + chunk_T)
            slab_all = np.asarray(tr[s:e, :])
            vals[s:e, :] = slab_all[:, keep_indices[:K]]
        return vals

    if skip_cascade:
        # Two-pass streaming z-score: first pass stats, second pass write
        print("  → Skip CASCADE: computing per-neuron z-score (streaming two-pass)")
        mean = np.zeros(N_keep, dtype=np.float64)
        M2 = np.zeros(N_keep, dtype=np.float64)
        ncount = 0
        chunk_T = min(1000, T)
        for t0 in range(0, T, chunk_T):
            t1 = min(T, t0 + chunk_T)
            slab_all = np.asarray(tr[t0:t1, :])
            slab = slab_all[:, keep_indices].astype(np.float64)
            # Update Welford by rows
            for row in slab:
                ncount += 1
                delta = row - mean
                mean += delta / ncount
                delta2 = row - mean
                M2 += delta2 * delta
        var = M2 / max(1, ncount - 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        mean32 = mean.astype(np.float32)
        std32 = std.astype(np.float32)

        with h5py.File(final_file, "w") as f:
            chunk_N = min(256, N_keep)
            dset_vals = f.create_dataset(
                "neuron_values",
                shape=(T, N_keep),
                dtype=output_dtype,
                chunks=(chunk_T, chunk_N),
                compression="gzip",
                compression_opts=1,
            )
            # Second pass: write z-scored
            for t0 in range(0, T, chunk_T):
                t1 = min(T, t0 + chunk_T)
                slab_all = np.asarray(tr[t0:t1, :])
                slab = slab_all[:, keep_indices].astype(np.float32)
                slab = (slab - mean32) / std32
                slab = np.nan_to_num(slab, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    output_dtype
                )
                dset_vals[t0:t1, :] = slab

            # Positions and metadata
            f.create_dataset(
                "cell_positions",
                data=pos_norm.astype(output_dtype),
                compression="gzip",
                compression_opts=1,
            )
            f.create_dataset("timepoint_indices", data=np.arange(T, dtype=np.int32))
            f.create_dataset("num_timepoints", data=T)
            f.create_dataset("num_neurons", data=N_keep)
            # Save stimulus features and labels if available
            if stimuli_features is not None:
                f.create_dataset(
                    "stimuli_features",
                    data=stimuli_features.astype(np.float32),
                    compression="gzip",
                    compression_opts=1,
                )
            if stimuli_onehot is not None:
                f.create_dataset(
                    "stimulus_full",
                    data=stimuli_onehot.astype(np.float32),
                    compression="gzip",
                    compression_opts=1,
                )
                f.attrs["stimulus_num_classes"] = int(stimuli_onehot.shape[1])
                # Save mapping (if available)
                if "condition_names" in locals():
                    try:
                        f.attrs["stimulus_condition_names"] = np.array(
                            condition_names, dtype="S"
                        )
                        f.attrs["stimulus_identity_indices"] = np.array(
                            identity_indices_used, dtype=np.int32
                        )
                    except Exception:
                        pass
            # Attributes
            f.attrs["subject"] = subject_name
            f.attrs["data_source"] = "zapbench_zarr"
            f.attrs["positions_normalized"] = True
            f.attrs["positions_min"] = mins
            f.attrs["positions_max"] = maxs
            f.attrs["includes_additional_data"] = False
            f.attrs["final_sampling_rate"] = float(orig_rate) if orig_rate else np.nan
            f.attrs["neuron_values_semantics"] = "zscored_signal"

            # Save z-score params
            f.create_dataset(
                "zscore_mean",
                data=mean32.astype(output_dtype),
                compression="gzip",
                compression_opts=1,
            )
            f.create_dataset(
                "zscore_std",
                data=std32.astype(output_dtype),
                compression="gzip",
                compression_opts=1,
            )

        # PDF
        try:
            K = config["processing"]["num_neurons_viz"] * 2
            values_small = read_small_sample(K)
            create_visualization_pdf(
                values_small,
                values_small.T,
                subject_name,
                pdf_file,
                num_neurons=config["processing"]["num_neurons_viz"],
                stim=(stimuli_labels if stimuli_labels is not None else None),
                behavior=None,
                eye=None,
                is_probability=False,
                value_label="Z-scored ZapBench Signal",
            )
        except Exception as e:
            print(f"  → Skipping PDF viz due to error: {e}")
        # Optional split-by-stimuli (skip_cascade path)
        if bool(zap_cfg.get("split_by_stimuli", False)) and (
            stimuli_onehot is not None
        ):
            print("  → Splitting output by contiguous stimulus segments…")
            _split_and_save_by_stimuli(final_file, output_dir)

        return final_file

    # CASCADE path: load full traces (T,N_keep) into memory to apply baseline and CASCADE
    if not CASCADE_AVAILABLE:
        raise RuntimeError(
            "CASCADE not available. Set processing.skip_cascade: true or install neuralib."
        )
    if not orig_rate or not tgt_rate:
        raise RuntimeError(
            "original_sampling_rate and target_sampling_rate are required for CASCADE path"
        )

    print("  → Loading full traces to memory for baseline correction and CASCADE…")
    traces_full = np.zeros((T, N_keep), dtype=np.float32)
    chunk_T = min(1000, T)
    for t0 in range(0, T, chunk_T):
        t1 = min(T, t0 + chunk_T)
        slab_all = np.asarray(tr[t0:t1, :])
        traces_full[t0:t1, :] = slab_all[:, keep_indices]

    # Baseline correction if configured (same as subject pipeline)
    processed_calcium = compute_baseline_correction(
        traces_full,
        config["data"]["window_length"],
        config["data"]["baseline_percentile"],
        float(orig_rate),
        bool(config["data"]["is_raw"]),
        bool(config["data"]["apply_baseline_subtraction"]),
    )
    del traces_full
    gc.collect()

    # CASCADE inference to probabilities or rates
    convert_flag = bool(
        config["processing"].get("convert_rates_to_probabilities", True)
    )
    prob_data = run_cascade_inference(
        processed_calcium,
        int(config["processing"]["batch_size"]),
        config["cascade"]["model_type"],
        float(tgt_rate),
        convert_rates_to_probabilities=convert_flag,
    )  # returns (N_keep, T)
    del processed_calcium
    gc.collect()

    # Optional return to original rate
    effective_sampling_rate = float(tgt_rate)
    if return_to_original and float(orig_rate) != float(tgt_rate):
        print(f"  → Downsampling probabilities from {tgt_rate}Hz to {orig_rate}Hz…")
        from scipy.signal import resample_poly
        from fractions import Fraction

        rate_ratio = (
            Fraction(int(orig_rate)).limit_denominator()
            / Fraction(int(tgt_rate)).limit_denominator()
        )
        up_factor = rate_ratio.numerator
        down_factor = rate_ratio.denominator
        prob_data_T = prob_data.T  # (T, N)
        T_out = int(T * float(orig_rate) / float(tgt_rate))
        downsampled = np.zeros((T_out, N_keep), dtype=np.float32)
        for n in tqdm(range(N_keep), desc="Downsampling neurons"):
            downsampled[:, n] = resample_poly(
                prob_data_T[:, n], up_factor, down_factor, axis=0
            )
        prob_data = downsampled.T  # (N, T_out)
        T = T_out
        effective_sampling_rate = float(orig_rate)

    # Write output
    with h5py.File(final_file, "w") as f:
        dset_vals = f.create_dataset(
            "neuron_values",
            data=prob_data.T.astype(output_dtype),
            chunks=(min(1000, T), min(256, N_keep)),
            compression="gzip",
            compression_opts=1,
        )
        f.create_dataset(
            "cell_positions",
            data=pos_norm.astype(output_dtype),
            compression="gzip",
            compression_opts=1,
        )
        f.create_dataset("timepoint_indices", data=np.arange(T, dtype=np.int32))
        f.create_dataset("num_timepoints", data=T)
        f.create_dataset("num_neurons", data=N_keep)
        f.create_dataset("original_sampling_rate_hz", data=float(orig_rate))
        # Optional log(activity) stats for CASCADE outputs
        if bool(config["output"].get("compute_log_activity_stats", False)):
            try:
                eps = float(config["output"].get("log_activity_eps", 1e-7))
                vals = np.maximum(prob_data.T.astype(np.float32, copy=False), eps)
                logs = np.log(vals)
                lam = logs.mean(axis=0).astype(np.float32)
                las = logs.std(axis=0).astype(np.float32)
                f.create_dataset(
                    "log_activity_mean",
                    data=lam,
                    compression="gzip",
                    compression_opts=1,
                )
                f.create_dataset(
                    "log_activity_std", data=las, compression="gzip", compression_opts=1
                )
                del vals, logs
                gc.collect()
                f.attrs["includes_log_activity_stats"] = True
                f.attrs["log_activity_eps"] = eps
            except Exception:
                f.attrs["includes_log_activity_stats"] = False
        # Save stimulus features and labels if available
        if "stimuli_features" in locals() and (stimuli_features is not None):
            f.create_dataset(
                "stimuli_features",
                data=stimuli_features.astype(np.float32),
                compression="gzip",
                compression_opts=1,
            )
        if stimuli_onehot is not None:
            f.create_dataset(
                "stimulus_full",
                data=stimuli_onehot.astype(np.float32),
                compression="gzip",
                compression_opts=1,
            )
            f.attrs["stimulus_num_classes"] = int(stimuli_onehot.shape[1])
            if "condition_names" in locals():
                try:
                    f.attrs["stimulus_condition_names"] = np.array(
                        condition_names, dtype="S"
                    )
                    f.attrs["stimulus_identity_indices"] = np.array(
                        identity_indices_used, dtype=np.int32
                    )
                except Exception:
                    pass
        f.attrs["subject"] = subject_name
        f.attrs["data_source"] = "zapbench_zarr"
        f.attrs["positions_normalized"] = True
        f.attrs["positions_min"] = mins
        f.attrs["positions_max"] = maxs
        f.attrs["includes_additional_data"] = False
        f.attrs["final_sampling_rate"] = effective_sampling_rate
        f.attrs["cascade_model"] = config["cascade"]["model_type"]
        f.attrs["neuron_values_semantics"] = (
            "probabilities" if convert_flag else "rates_hz"
        )

    # PDF using probabilities or rates
    try:
        K = config["processing"]["num_neurons_viz"] * 2
        values_small = read_small_sample(K)
        is_probability = bool(convert_flag)
        value_label = None if is_probability else "Spike Rates (Hz)"
        create_visualization_pdf(
            values_small,
            prob_data[:K, :],
            subject_name,
            pdf_file,
            num_neurons=config["processing"]["num_neurons_viz"],
            stim=(stimuli_labels if stimuli_labels is not None else None),
            behavior=None,
            eye=None,
            is_probability=is_probability,
            value_label=value_label,
        )
    except Exception as e:
        print(f"  → Skipping PDF viz due to error: {e}")

    # Optional split-by-stimuli
    if bool(zap_cfg.get("split_by_stimuli", False)) and (stimuli_onehot is not None):
        print("  → Splitting output by contiguous stimulus segments…")
        _split_and_save_by_stimuli(final_file, output_dir)

    return final_file


def _split_and_save_by_stimuli(h5_path: str, output_dir: str):
    """Split a ZapBench H5 by contiguous stimulus labels into separate H5+PDF files.

    Uses `stimulus_full` argmax labels if present. If absent, attempts to use
    `stimuli_features` identity dimensions (columns that are binary in {0,1})
    to derive a condition label per timepoint.
    """
    import h5py
    import numpy as np
    from pathlib import Path

    path = Path(h5_path)
    with h5py.File(str(path), "r") as f:
        feats = None
        cond_names = None
        if "stimulus_full" in f:
            stim_onehot = f["stimulus_full"][:]  # (T, K)
            labels = np.argmax(stim_onehot, axis=1).astype(np.int64)
            # Try to load names for nicer filenames
            try:
                if "stimulus_condition_names" in f.attrs:
                    cond_names = [
                        s.decode("utf-8")
                        if isinstance(s, (bytes, bytearray))
                        else str(s)
                        for s in f.attrs["stimulus_condition_names"]
                    ]
            except Exception:
                cond_names = None
            # Also load full stimuli_features if present to drive the PDF panel
            try:
                if "stimuli_features" in f:
                    feats = f["stimuli_features"][:]
            except Exception:
                feats = None
        elif "stimuli_features" in f:
            feats = f["stimuli_features"][:]  # (T, 26)
            is01 = (feats.min(axis=0) >= 0.0) & (feats.max(axis=0) <= 1.0)
            cand = np.where(is01)[0]
            if cand.size == 0:
                print(
                    "  → No identity-like dims in stimuli_features; skipping split-by-stimuli"
                )
                return
            active = feats[:, cand]
            which = np.argmax(active, axis=1)
            # map actual dim index to compact label
            uniq = np.unique(which)
            mapc = {int(u): i for i, u in enumerate(uniq.tolist())}
            labels = np.vectorize(lambda x: mapc[int(x)])(which).astype(np.int64)
            # Build names from dim indices if possible
            cond_names = [f"dim{int(i)}" for i in uniq.tolist()]
        else:
            print("  → No stimulus data found; skipping split-by-stimuli")
            return
        values = f["neuron_values"][:]  # (T, N)
        positions = f["cell_positions"][:]  # (N,3)
        T, N = values.shape
        # Find contiguous runs
        starts = [0]
        for t in range(1, T):
            if labels[t] != labels[t - 1]:
                starts.append(t)
        starts.append(T)
        # Save each run
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            label = int(labels[s])
            label_name = None
            if (
                "cond_names" in locals()
                and cond_names is not None
                and label < len(cond_names)
            ):
                label_name = cond_names[label]
            tag = label_name if label_name else f"stim{label}"
            out_base = Path(output_dir) / f"{path.stem}_{tag}_seg{i}"
            out_h5 = str(out_base.with_suffix(".h5"))
            out_pdf = str(Path(output_dir) / f"{out_base.stem}_visualization.pdf")
            with h5py.File(out_h5, "w") as g:
                g.create_dataset(
                    "neuron_values",
                    data=values[s:e].astype(values.dtype),
                    compression="gzip",
                    compression_opts=1,
                )
                g.create_dataset(
                    "cell_positions",
                    data=positions,
                    compression="gzip",
                    compression_opts=1,
                )
                g.create_dataset(
                    "timepoint_indices", data=np.arange(e - s, dtype=np.int32)
                )
                g.create_dataset("num_timepoints", data=int(e - s))
                g.create_dataset("num_neurons", data=int(N))
                # Save one-hot labels if present
                if "stim_onehot" in locals():
                    g.create_dataset(
                        "stimulus_full",
                        data=stim_onehot[s:e],
                        compression="gzip",
                        compression_opts=1,
                    )
                # Save full 26-D features if present in source file
                if "feats" in locals():
                    g.create_dataset(
                        "stimuli_features",
                        data=feats[s:e].astype(np.float32),
                        compression="gzip",
                        compression_opts=1,
                    )
                g.attrs.update(
                    {
                        k: f.attrs[k]
                        for k in f.attrs.keys()
                        if k not in ("num_timepoints",)
                    }
                )
                # Add train/val/test split ranges per paper (TAXIS fully test)
                L = int(e - s)
                g.attrs["split_index_semantics"] = (
                    "0-based inclusive start, exclusive end"
                )
                g.attrs["split_condition_name"] = (
                    label_name if label_name else f"stim{label}"
                )
                if str(g.attrs["split_condition_name"]).lower() == "taxis":
                    tr0, tr1 = 0, 0
                    va0, va1 = 0, 0
                    te0, te1 = 0, L
                else:
                    n_train = int(np.floor(0.7 * L))
                    n_val = int(np.floor(0.1 * L))
                    n_test = L - n_train - n_val
                    tr0, tr1 = 0, n_train
                    va0, va1 = tr1, tr1 + n_val
                    te0, te1 = va1, va1 + n_test
                g.attrs["split_train_start"] = np.int32(tr0)
                g.attrs["split_train_end"] = np.int32(tr1)
                g.attrs["split_val_start"] = np.int32(va0)
                g.attrs["split_val_end"] = np.int32(va1)
                g.attrs["split_test_start"] = np.int32(te0)
                g.attrs["split_test_end"] = np.int32(te1)
            # Create a compact PDF
            try:
                # Use full transposed values for prob panel indexing safety
                stim_for_pdf = feats[s:e] if "feats" in locals() else labels[s:e]
                create_visualization_pdf(
                    values[s:e],
                    values[s:e].T,
                    f.attrs.get("subject", "zapbench"),
                    out_pdf,
                    num_neurons=min(10, N),
                    stim=stim_for_pdf,
                    behavior=None,
                    eye=None,
                    is_probability=False,
                    value_label="Segment Signal",
                )
            except Exception as ex:
                print(f"  → Failed to write segment PDF {out_pdf}: {ex}")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
