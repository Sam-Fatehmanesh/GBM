#!/usr/bin/env python
import os
import argparse
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import loadmat
from scipy.signal import find_peaks
from tqdm import tqdm

def resample_and_write_h5(src_h5, dst_h5, seed=None):
    # Load probabilities and original spikes shape
    with h5py.File(src_h5, 'r') as f:
        prob = f['probabilities'][:]           # shape (neurons, frames) or (frames, neurons)
        orig_spikes_shape = f['spikes'].shape
        # load other datasets & attrs
        others = {k: f[k][:] for k in f if k not in ('spikes','probabilities')}
        attrs = dict(f.attrs)

    # clamp & sample
    p = np.clip(prob, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    sampled = (rng.random(p.shape) < p).astype(np.int8)

    # orient sampled to match orig_spikes_shape
    if sampled.shape != orig_spikes_shape:
        if sampled.T.shape == orig_spikes_shape:
            sampled = sampled.T
        else:
            raise ValueError(f"Cannot match sampled {sampled.shape} to original {orig_spikes_shape}")

    # write new H5
    os.makedirs(os.path.dirname(dst_h5), exist_ok=True)
    with h5py.File(dst_h5, 'w') as f:
        f.create_dataset('spikes', data=sampled, compression='gzip')
        f.create_dataset('probabilities', data=prob, compression='gzip')
        for k, v in others.items():
            f.create_dataset(k, data=v, compression='gzip')
        for k, v in attrs.items():
            f.attrs[k] = v

def regenerate_pdf(subject, raw_root, h5_path, pdf_path, num_neurons):
    """
    Re-generate the PDF for one subject using raw traces, probabilities, and new spikes.
    raw_root: path to root of raw data (with subject subdirs)
    """
    # load raw calcium
    ts_h5 = os.path.join(raw_root, subject, 'TimeSeries.h5')
    with h5py.File(ts_h5, 'r') as f:
        calcium = f['CellResp'][:]   # shape = (frames, neurons)

    # load new probs & spikes
    with h5py.File(h5_path, 'r') as f:
        prob   = f['probabilities'][:]
        spikes = f['spikes'][:]

    # orient prob and spikes so that indexing is [neuron, frame]
    # if prob shape=(frames, neurons), transpose
    if prob.shape[0] == calcium.shape[0] and prob.shape[1] == calcium.shape[1]:
        prob = prob.T
    if spikes.shape[0] == calcium.shape[0] and spikes.shape[1] == calcium.shape[1]:
        spikes = spikes.T

    # sanity
    N = calcium.shape[1]
    T = calcium.shape[0]
    assert prob.shape == (N, T)
    assert spikes.shape == (N, T)

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        sel = np.random.choice(N, num_neurons, replace=False) if N>num_neurons else np.arange(N)
        for idx in sel:
            fig, ax = plt.subplots(3,1,figsize=(8,9),sharex=True)
            ax[0].plot(calcium[:,idx], 'k')
            ax[0].set_title(f'{subject} Neuron {idx}: Raw ΔF/F')
            ax[0].set_ylabel('Fluor.')

            ax[1].plot(prob[idx], 'm')
            ax[1].set_title('Spike Rate (expected spikes/frame)')
            ax[1].set_ylabel('Rate')

            ax[2].plot(spikes[idx], 'r')
            ax[2].set_title('Sampled Spikes (0/1)')
            ax[2].set_ylabel('Spike')
            ax[2].set_xlabel('Frame')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Copy output folder and replace spikes by sampling probabilities, regenerating PDFs."
    )
    parser.add_argument('-raw_root',
                        help="Root of raw data (contains subject_x/TimeSeries.h5)")
    parser.add_argument('-input_output_dir',
                        help="Existing CASCADE output folder (contains *.h5 and *.pdf)")
    parser.add_argument('-new_output_dir',
                        help="Where to write the new H5s and PDFs")
    parser.add_argument('--num_neurons', type=int, default=10,
                        help="Neurons to plot per PDF")
    args = parser.parse_args()

    os.makedirs(args.new_output_dir, exist_ok=True)

    # First process all H5 files
    h5_files = [f for f in os.listdir(args.input_output_dir) if f.endswith('.h5')]
    for fname in tqdm(h5_files, desc="Resampling H5 files"):
        src = os.path.join(args.input_output_dir, fname)
        dst = os.path.join(args.new_output_dir, fname)
        subj = os.path.splitext(fname)[0].replace('_spikes','')
        resample_and_write_h5(src, dst, 42 + h5_files.index(fname))

    # Then process all PDF files
    pdf_files = [f for f in os.listdir(args.input_output_dir) if f.endswith('.pdf')]
    for fname in tqdm(pdf_files, desc="Regenerating PDFs"):
        src = os.path.join(args.input_output_dir, fname)
        dst = os.path.join(args.new_output_dir, fname)
        subj = fname.replace('_visualization.pdf','')
        h5_file = os.path.join(args.new_output_dir, f"{subj}_spikes.h5")
        regenerate_pdf(subj, args.raw_root, h5_file, dst, args.num_neurons)

    # Finally copy any remaining files
    other_files = [f for f in os.listdir(args.input_output_dir) 
                  if not f.endswith('.h5') and not f.endswith('.pdf')]
    for fname in tqdm(other_files, desc="Copying other files"):
        src = os.path.join(args.input_output_dir, fname)
        dst = os.path.join(args.new_output_dir, fname)
        shutil.copy2(src, dst)

    print("All done.")

if __name__ == '__main__':
    main()