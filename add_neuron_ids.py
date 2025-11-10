#!/usr/bin/env python3
"""
Add per-neuron global IDs (random 64-bit ints) to existing processed H5 files.

Usage:
  python add_neuron_ids.py --data_dir processed_spike_rates_2018
"""

import argparse
import os
from pathlib import Path
import numpy as np
import h5py


def add_ids_to_file(h5_path: Path) -> bool:
    try:
        with h5py.File(str(h5_path), "a") as f:
            if "neuron_global_ids" in f:
                return False
            if "num_neurons" in f:
                N = int(f["num_neurons"][()])
            else:
                if "cell_positions" in f:
                    N = int(f["cell_positions"].shape[0])
                else:
                    raise ValueError(
                        "Cannot infer N; 'num_neurons' or 'cell_positions' missing"
                    )
            rng = np.random.default_rng(seed=None)
            ids = rng.integers(
                low=1, high=np.iinfo(np.int64).max, size=(N,), dtype=np.int64
            )
            f.create_dataset("neuron_global_ids", data=ids)
        return True
    except Exception as e:
        print(f"Failed to add neuron_global_ids to {h5_path}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed H5 files",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted([p for p in data_dir.glob("*.h5")])
    print(f"Found {len(files)} files in {data_dir}")
    added = 0
    for fp in files:
        changed = add_ids_to_file(fp)
        added += int(changed)
    print(f"Added neuron_global_ids to {added} files")


if __name__ == "__main__":
    main()
