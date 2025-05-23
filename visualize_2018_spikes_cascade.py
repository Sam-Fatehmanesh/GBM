#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from scipy.signal import find_peaks
from tqdm import tqdm
import argparse

# CASCADE functional import
from neuralib.imaging.spikes.cascade import cascade_predict

# TensorFlow tuning (optional)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Parameters for ΔF/F baseline
SAMPLING_RATE = 2.0      # Hz
WINDOW_SEC    = 30.0     # seconds for sliding window
PERCENTILE    = 8        # percentile for F0
WINDOW_FRAMES = int(WINDOW_SEC * SAMPLING_RATE)

def compute_dff(calcium_data):
    """
    Compute ΔF/F using an 8th-percentile, 30s sliding window per cell.
    calcium_data F/F0: (T, N) array
    returns dff of same shape.
    """
    T, N = calcium_data.shape
    half_w = WINDOW_FRAMES // 2
    dff = np.zeros_like(calcium_data, dtype=np.float32)
    
    # Create sliding window indices for all timepoints
    starts = np.maximum(0, np.arange(T) - half_w)
    ends = np.minimum(T, np.arange(T) + half_w + 1)
    
    # Compute F0 for all cells at once using vectorized operations
    F0 = np.zeros((T, N), dtype=np.float32)
    for t in tqdm(range(T), desc="Computing baselines"):
        start, end = starts[t], ends[t]
        F0[t] = np.percentile(calcium_data[start:end], PERCENTILE, axis=0)
    
    # # Handle divide-by-zero cases
    # F0_means = np.nanmean(F0, axis=0)
    # mask = F0 <= 0
    # rows, cols = np.where(mask)
    # F0[rows, cols] = F0_means[cols]
    
    # Compute ΔF/F for all cells
    dff = (calcium_data - F0)
    
    return dff

def process_and_visualize_subject(subject_dir, output_dir, num_neurons=10, batch_size=30000):
    subject_name = os.path.basename(os.path.normpath(subject_dir))
    print(f"\nProcessing subject: {subject_name}")

    os.makedirs(output_dir, exist_ok=True)
    h5_out  = os.path.join(output_dir, f"{subject_name}_spikes.h5")
    pdf_out = os.path.join(output_dir, f"{subject_name}_visualization.pdf")
    if os.path.exists(pdf_out):
        print("  → Already done; skipping.")
        return

    # Load cell positions
    mat   = loadmat(os.path.join(subject_dir, 'data_full.mat'))
    data0 = mat['data'][0,0]
    cell_xyz = data0['CellXYZ']
    if isinstance(cell_xyz, np.ndarray) and cell_xyz.dtype == np.object_:
        cell_xyz = cell_xyz[0,0]
    if 'IX_inval_anat' in data0.dtype.names:
        inval = data0['IX_inval_anat']
        if isinstance(inval, np.ndarray) and inval.dtype == np.object_:
            inval = inval[0,0].flatten()
        mask = np.ones(cell_xyz.shape[0], bool)
        mask[np.array(inval, int)-1] = False
        cell_xyz = cell_xyz[mask]

    # Load raw fluorescence traces
    with h5py.File(os.path.join(subject_dir, 'TimeSeries.h5'), 'r') as f:
        calcium = f['CellResp'][:]  # shape = (T, N)

    T, N = calcium.shape
    assert N == cell_xyz.shape[0], "Cell count mismatch"

    # Compute ΔF/F
    print("  → Computing ΔF/F (8th-percentile, 30 s sliding window)…")
    dff = compute_dff(calcium)  # shape = (T, N)

    # Prepare containers
    prob_data  = np.zeros((N, T), dtype=np.float32)
    spike_data = np.zeros((T, N), dtype=int)

    # Batched CASCADE inference
    print(f"  → Running CASCADE in batches of {batch_size}…")
    traces = dff.T.astype(np.float32)  # (N, T)
    for start in tqdm(range(0, N, batch_size), desc="Batches"):
        end = min(start + batch_size, N)
        batch = traces[start:end]
        batch_probs = cascade_predict(
            batch,
            model_type='Global_EXC_2Hz_smoothing500ms',
            threshold=1,
            padding=np.nan,
            verbose=True
        )
        batch_probs = np.atleast_2d(batch_probs)
        if batch_probs.shape != (end-start, T):
            raise ValueError("Unexpected batch_probs shape")
        prob_data[start:end] = batch_probs

    # Threshold probabilities at 0.5
    print("  → Thresholding binary spikes from probabilities…")
    spike_data = (prob_data > 0.5).astype(int)

    # Save HDF5
    print(f"  → Saving H5 to {h5_out}")
    with h5py.File(h5_out, 'w') as f:
        f.create_dataset('spikes',         data=spike_data)
        f.create_dataset('probabilities',  data=prob_data)
        f.create_dataset('cell_positions', data=cell_xyz)
        f.attrs['model_type']     = 'Global_EXC_2Hz_smoothing500ms'
        f.attrs['sampling_rate']  = SAMPLING_RATE
        f.attrs['window_sec']     = WINDOW_SEC
        f.attrs['percentile']     = PERCENTILE
        f.attrs['num_timepoints'] = T
        f.attrs['num_cells']      = N

    # Regenerate PDF
    print(f"  → Writing PDF to {pdf_out}")
    with PdfPages(pdf_out) as pdf:
        sel = np.random.choice(N, num_neurons, replace=False) if N>num_neurons else np.arange(N)
        for idx in sel:
            fig, ax = plt.subplots(3,1,figsize=(8,9),sharex=True)
            ax[0].plot(dff[:,idx], 'k')
            ax[0].set_title(f'Neuron {idx}: ΔF/F')
            ax[0].set_ylabel('ΔF/F')

            ax[1].plot(prob_data[idx], 'm')
            ax[1].set_title('Inferred Spike Rate')
            ax[1].set_ylabel('Rate (spikes/frame)')

            ax[2].plot(spike_data[:,idx], 'r')
            ax[2].set_title('Discrete Spikes')
            ax[2].set_ylabel('Spike (0/1)')
            ax[2].set_xlabel('Frame')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  → Done: {subject_name}")

def main():
    parser = argparse.ArgumentParser(
        description='CASCADE inference on ΔF/F with sliding-window baseline'
    )
    parser.add_argument('--data_root',   type=str, default='raw_trace_data_2018')
    parser.add_argument('--output_dir',  type=str, default='spike_processed_data_2018')
    parser.add_argument('--num_neurons', type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=30000)
    parser.add_argument('--skip',        type=str, default='')
    args = parser.parse_args()

    skips = {s.strip() for s in args.skip.split(',') if s.strip()}
    subjects = sorted(
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if d.startswith('subject_') and d not in skips
    )
    print(f"Found {len(subjects)} subjects; skipping {skips}")

    for sd in subjects:
        try:
            process_and_visualize_subject(
                sd,
                args.output_dir,
                args.num_neurons,
                args.batch_size
            )
        except Exception as e:
            print(f"Error processing {sd}: {e}")

if __name__ == '__main__':
    main()
