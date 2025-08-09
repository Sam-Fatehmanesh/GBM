#!/usr/bin/env python3
"""
GBM Sliding Next-Step Evaluation

Runs a trained GBM on full subject timelines using sliding windows:
- For each window of length sequence_length, autoregress one next frame
- Compare the predicted next frame to the true next frame

Saves results to experiments/eval2_results/eval_<timestamp> with:
- Per-subject plots:
  - sorted_true_vs_pred_scatter.png (binned/ordered true vs mean predicted with y=x reference)
  - true_vs_pred_hexbin.png (2D density over [0,1]x[0,1])
  - true_vs_pred_scatter_subset.png (random subset scatter)
- Summary CSV/JSON of metrics: MAE, MSE, BCE(prob), Pearson r

Config-driven (YAML). Use --generate-config to write a default config skeleton.
"""

import os
import sys
import math
import json
import yaml
import time
import h5py
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

# Ensure project root is on path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.volume_dataloader import VolumeDataset
from torch.utils.data import DataLoader


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dicts (right-biased merge)."""
    result = base_dict.copy()
    for key, value in (update_dict or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def create_default_config() -> Dict[str, Any]:
    return {
        'evaluation': {
            'name': 'gbm_sliding_next_eval',
            'model_path': None,
            'strict_load': False,
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018_causal',
            'subjects': ['subject_1'],
            'use_cache': False,
        },
        'sliding': {
            'sequence_length': 8,
            'stride': 1,
            'batch_size': 2,
            'num_workers': 2,
        },
        'analysis': {
            'hexbin_bins': 200,
            'plot_bins_sorted': 200,
            'scatter_max_points': 200000,
            'random_seed': 42,
        },
        'embedding': {
            'sample_size': 2000,
            'random_seed': 42,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'umap_metric': 'euclidean'
        },
        'device': 'cuda',
        'output': {
            'base_dir': 'experiments/eval2_results',
            'save_plots': True,
            'save_csv': True,
            'save_json': True,
        },
    }


def save_default_config(path: str) -> None:
    cfg = create_default_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)


class SlidingNextEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self._setup_device()
        self._setup_output_dirs()
        self.model: Optional[GBM] = None

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_device(self) -> None:
        device_cfg = self.config.get('device', 'cuda')
        if device_cfg == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")

    def _setup_output_dirs(self) -> None:
        base_dir = Path(self.config['output']['base_dir'])
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.eval_dir = base_dir / f"eval_{timestamp}"
        self.plots_dir = self.eval_dir / 'plots'
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output dir: {self.eval_dir}")

    def load_model(self) -> None:
        model_path = self.config['evaluation']['model_path']
        if not model_path or not Path(model_path).exists():
            raise ValueError(f"Model path not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint must contain a 'config' with model settings")
        model_cfg = checkpoint['config']['model']

        self.model = GBM(
            d_model=model_cfg['d_model'],
            n_heads=model_cfg['n_heads'],
            n_layers=model_cfg['n_layers'],
            autoencoder_path=model_cfg.get('autoencoder_path'),
            volume_size=tuple(model_cfg['volume_size']),
            region_size=tuple(model_cfg['region_size'])
        )

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Clean compile prefixes only when present
        prefix = '_orig_mod.'
        cleaned = {}
        for k, v in state_dict.items():
            cleaned_key = k[len(prefix):] if k.startswith(prefix) else k
            cleaned[cleaned_key] = v

        # Allow non-strict loading if checkpoint/module names differ across versions
        strict = bool(self.config['evaluation'].get('strict_load', False))
        missing, unexpected = self.model.load_state_dict(cleaned, strict=strict)
        if not strict:
            if missing:
                self.logger.warning(f"Missing keys when loading model (non-strict): {len(missing)} keys")
            if unexpected:
                self.logger.warning(f"Unexpected keys when loading model (non-strict): {len(unexpected)} keys")
        self.model.to(self.device)
        self.model.eval()
        nparams = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded ({nparams:,} params)")

    def _create_subject_loader(self, subject_h5: Path) -> DataLoader:
        seq_len = int(self.config['sliding']['sequence_length'])
        stride = int(self.config['sliding']['stride'])
        use_cache = bool(self.config['data'].get('use_cache', False))

        dataset = VolumeDataset(
            data_files=[str(subject_h5)],
            sequence_length=seq_len + 1,  # include next frame as target
            stride=stride,
            max_timepoints_per_subject=None,
            use_cache=use_cache,
            start_timepoint=None,
            end_timepoint=None,
        )

        loader = DataLoader(
            dataset,
            batch_size=int(self.config['sliding']['batch_size']),
            shuffle=False,
            num_workers=int(self.config['sliding']['num_workers']),
            pin_memory=False,
        )
        return loader

    def _init_accumulators(self, voxel_count: int, expected_sequences: int) -> Dict[str, Any]:
        analysis_cfg = self.config['analysis']
        random.seed(analysis_cfg.get('random_seed', 42))
        np.random.seed(analysis_cfg.get('random_seed', 42))

        max_points = int(analysis_cfg.get('scatter_max_points', 200000))
        # Estimate sampling probability for subset scatter
        total_points_est = max(1, voxel_count * expected_sequences)
        p_sample = min(1.0, max_points / total_points_est)

        return {
            'count': 0,
            'sum_abs': 0.0,
            'sum_sq': 0.0,
            'sum_bce': 0.0,
            'sum_x': 0.0,   # true
            'sum_y': 0.0,   # pred
            'sum_x2': 0.0,
            'sum_y2': 0.0,
            'sum_xy': 0.0,
            'hex_counts': np.zeros((analysis_cfg['hexbin_bins'], analysis_cfg['hexbin_bins']), dtype=np.int64),
            'bin_sum_pred': np.zeros(analysis_cfg['plot_bins_sorted'], dtype=np.float64),
            'bin_sum_true': np.zeros(analysis_cfg['plot_bins_sorted'], dtype=np.float64),
            'bin_count': np.zeros(analysis_cfg['plot_bins_sorted'], dtype=np.int64),
            'subset_true': [],
            'subset_pred': [],
            'p_sample': p_sample,
        }

    def _update_accumulators(self, acc: Dict[str, Any], true_next: torch.Tensor, pred_next: torch.Tensor) -> None:
        # true_next, pred_next: (B, 1, X, Y, Z) probabilities on same device
        with torch.no_grad():
            y_true = true_next.reshape(-1).detach().to(torch.float32).cpu().numpy()
            y_pred = pred_next.reshape(-1).detach().to(torch.float32).cpu().numpy()

        # Clip to avoid log(0)
        eps = 1e-7
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        # Basic metrics
        diff = y_pred - y_true
        acc['sum_abs'] += float(np.abs(diff).sum())
        acc['sum_sq'] += float((diff * diff).sum())
        bce = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        acc['sum_bce'] += float(bce.sum())

        acc['sum_x'] += float(y_true.sum())
        acc['sum_y'] += float(y_pred.sum())
        acc['sum_x2'] += float((y_true * y_true).sum())
        acc['sum_y2'] += float((y_pred * y_pred).sum())
        acc['sum_xy'] += float((y_true * y_pred).sum())
        acc['count'] += y_true.size

        # Hexbin counts
        nb = acc['hex_counts'].shape[0]
        xi = np.clip((y_true * nb).astype(int), 0, nb - 1)
        yi = np.clip((y_pred * nb).astype(int), 0, nb - 1)
        # Efficient bincount on flattened 2D indices
        indices = xi * nb + yi
        binc = np.bincount(indices, minlength=nb * nb)
        acc['hex_counts'] += binc.reshape(nb, nb)

        # Binned sorted plot along truth axis
        pb = acc['bin_sum_pred'].shape[0]
        bi = np.clip((y_true * pb).astype(int), 0, pb - 1)
        # Accumulate by bins
        for b in range(pb):
            mask = (bi == b)
            if mask.any():
                acc['bin_sum_true'][b] += float(y_true[mask].sum())
                acc['bin_sum_pred'][b] += float(y_pred[mask].sum())
                acc['bin_count'][b] += int(mask.sum())

        # Subset scatter sampling
        if acc['p_sample'] < 1.0:
            keep = np.random.rand(y_true.size) < acc['p_sample']
            if keep.any():
                acc['subset_true'].append(y_true[keep])
                acc['subset_pred'].append(y_pred[keep])
        else:
            acc['subset_true'].append(y_true)
            acc['subset_pred'].append(y_pred)

    def _finalize_metrics(self, acc: Dict[str, Any]) -> Dict[str, float]:
        n = max(1, acc['count'])
        mae = acc['sum_abs'] / n
        mse = acc['sum_sq'] / n
        bce = acc['sum_bce'] / n
        # Pearson r
        num = n * acc['sum_xy'] - acc['sum_x'] * acc['sum_y']
        den_x = n * acc['sum_x2'] - acc['sum_x'] * acc['sum_x']
        den_y = n * acc['sum_y2'] - acc['sum_y'] * acc['sum_y']
        denom = math.sqrt(max(den_x, 0.0) * max(den_y, 0.0))
        r = float(num / denom) if denom > 0 else 0.0
        return {'mae': mae, 'mse': mse, 'bce': bce, 'pearson_r': r, 'count': n}

    def _plot_subject(self, subject_dir: Path, acc: Dict[str, Any], metrics: Dict[str, float]) -> None:
        if not self.config['output'].get('save_plots', True):
            return

        subject_dir.mkdir(parents=True, exist_ok=True)
        # 1) Sorted binned scatter (truth vs mean predicted)
        bin_sum_true = acc['bin_sum_true']
        bin_sum_pred = acc['bin_sum_pred']
        bin_count = acc['bin_count']
        mask = bin_count > 0
        x_vals = np.zeros_like(bin_sum_true)
        y_vals = np.zeros_like(bin_sum_pred)
        x_vals[mask] = bin_sum_true[mask] / bin_count[mask]
        y_vals[mask] = bin_sum_pred[mask] / bin_count[mask]

        order = np.argsort(x_vals[mask])
        x_sorted = x_vals[mask][order]
        y_sorted = y_vals[mask][order]

        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, y_sorted, linestyle='none', marker='o', markersize=3, alpha=0.7, label='Binned means')
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal y=x')
        plt.xlabel('True activation (binned mean, ascending)')
        plt.ylabel('Predicted activation (mean)')
        plt.title('True vs Predicted (Binned)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(subject_dir / 'sorted_true_vs_pred_scatter.png', dpi=200)
        plt.close()

        # 2) Hexbin-like heatmap from counts
        counts = acc['hex_counts']
        plt.figure(figsize=(7, 6))
        plt.imshow(counts.T, origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='viridis')
        plt.colorbar(label='Count')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('True vs Predicted Density')
        plt.tight_layout()
        plt.savefig(subject_dir / 'true_vs_pred_hexbin.png', dpi=200)
        plt.close()

        # 3) Scatter subset
        if acc['subset_true'] and acc['subset_pred']:
            x_sub = np.concatenate(acc['subset_true'])
            y_sub = np.concatenate(acc['subset_pred'])
            # Limit to at most configured scatter_max_points
            max_pts = int(self.config['analysis'].get('scatter_max_points', 200000))
            if x_sub.size > max_pts:
                idx = np.random.choice(x_sub.size, max_pts, replace=False)
                x_sub = x_sub[idx]
                y_sub = y_sub[idx]
            plt.figure(figsize=(7, 6))
            plt.scatter(x_sub, y_sub, s=1, alpha=0.05)
            plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title('True vs Predicted (Subset)')
            plt.tight_layout()
            plt.savefig(subject_dir / 'true_vs_pred_scatter_subset.png', dpi=200)
            plt.close()

        # 4) Save simple metrics text
        with open(subject_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    def evaluate_subject(self, subject_name: str) -> Dict[str, float]:
        data_dir = Path(self.config['data']['data_dir'])
        subject_h5 = data_dir / f"{subject_name}.h5"
        if not subject_h5.exists():
            raise FileNotFoundError(f"Missing subject file: {subject_h5}")

        # Prepare loader
        loader = self._create_subject_loader(subject_h5)
        # Infer volume size and expected sequences
        with h5py.File(subject_h5, 'r') as f:
            vol_shape = f['volumes'][0].shape
            total_timepoints = f['volumes'].shape[0]
        voxel_count = int(np.prod(vol_shape))
        expected_sequences = len(loader.dataset)
        acc = self._init_accumulators(voxel_count=voxel_count, expected_sequences=expected_sequences)

        seq_len = int(self.config['sliding']['sequence_length'])

        subject_dir = self.eval_dir / f"sliding_subject" / subject_name
        plots_dir = subject_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Prepare H5 for saving next-step truths and predictions
        h5_path = subject_dir / 'next_step_truth_pred.h5'
        if h5_path.exists():
            h5_path.unlink()
        h5_file = h5py.File(h5_path, 'w')
        d_true = h5_file.create_dataset(
            'true', shape=(expected_sequences, *vol_shape), dtype='float32',
            chunks=(1, *vol_shape), compression=None
        )
        d_pred = h5_file.create_dataset(
            'pred', shape=(expected_sequences, *vol_shape), dtype='float32',
            chunks=(1, *vol_shape), compression=None
        )
        write_offset = 0

        self.logger.info(f"Evaluating subject {subject_name} with {expected_sequences} windows (T={total_timepoints}, seq_len={seq_len})")

        with torch.no_grad():
            for batch_idx, (batch_seq, _) in enumerate(tqdm(loader, desc=f"{subject_name}", total=len(loader))):
                # batch_seq: (B, T=seq_len+1, X, Y, Z)
                batch_seq = batch_seq.to(self.device, non_blocking=False)
                init_x = batch_seq[:, :seq_len]  # (B, seq_len, ...)
                true_next = batch_seq[:, -1:]    # (B, 1, ...)

                # Autoregress one step
                generated = self.model.autoregress(init_x, n_steps=1, context_len=seq_len)
                pred_next = generated[:, -1:]  # (B, 1, X, Y, Z) probabilities

                # Accumulate statistics and plot data
                self._update_accumulators(acc, true_next=true_next, pred_next=pred_next)

                # Save to H5
                B = true_next.shape[0]
                d_true[write_offset:write_offset+B] = true_next.squeeze(1).detach().cpu().numpy().astype(np.float32)
                d_pred[write_offset:write_offset+B] = pred_next.squeeze(1).detach().cpu().numpy().astype(np.float32)
                write_offset += B

                # Free up GPU
                del batch_seq, init_x, true_next, generated, pred_next
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Ensure H5 is written
        h5_file.flush()
        h5_file.close()

        metrics = self._finalize_metrics(acc)

        # Save per-subject CSV/JSON and plots
        if self.config['output'].get('save_csv', True):
            df = pd.DataFrame([{'subject': subject_name, **metrics}])
            df.to_csv(subject_dir / 'summary.csv', index=False)
        if self.config['output'].get('save_json', True):
            with open(subject_dir / 'summary.json', 'w') as f:
                json.dump({'subject': subject_name, **metrics}, f, indent=2)

        self._plot_subject(subject_dir=plots_dir, acc=acc, metrics=metrics)

        # Create PCA and UMAP on a sample from saved H5
        try:
            self._create_embeddings(subject_dir, h5_path)
        except Exception as e:
            self.logger.warning(f"Failed to create embeddings for {subject_name}: {e}")

        return metrics

    def _create_embeddings(self, subject_dir: Path, h5_path: Path) -> None:
        emb_cfg = self.config['embedding']
        rng = np.random.RandomState(emb_cfg.get('random_seed', 42))

        with h5py.File(h5_path, 'r') as f:
            true_ds = f['true']
            pred_ds = f['pred']
            n = true_ds.shape[0]
            if n == 0:
                return
            k = int(min(emb_cfg.get('sample_size', 2000), n))
            idx = rng.choice(n, size=k, replace=False)

            # Load and flatten
            X_true = true_ds[idx].reshape(k, -1)
            X_pred = pred_ds[idx].reshape(k, -1)

        # Stack and labels
        X = np.vstack([X_true, X_pred])
        y = np.array([0] * k + [1] * k)  # 0=true, 1=pred

        # PCA
        pca = PCA(n_components=2, random_state=emb_cfg.get('random_seed', 42))
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(7, 6))
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s=6, alpha=0.4, label='True')
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s=6, alpha=0.4, label='Pred')
        plt.title('PCA (2D) of True vs Pred Next-Step Volumes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(subject_dir / 'pca_true_pred_2d.png', dpi=200)
        plt.close()

        # UMAP
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(emb_cfg.get('umap_n_neighbors', 15)),
                min_dist=float(emb_cfg.get('umap_min_dist', 0.1)),
                metric=str(emb_cfg.get('umap_metric', 'euclidean')),
                random_state=emb_cfg.get('random_seed', 42)
            )
            X_umap = reducer.fit_transform(X)
            plt.figure(figsize=(7, 6))
            plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], s=6, alpha=0.4, label='True')
            plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], s=6, alpha=0.4, label='Pred')
            plt.title('UMAP (2D) of True vs Pred Next-Step Volumes')
            plt.legend()
            plt.tight_layout()
            plt.savefig(subject_dir / 'umap_true_pred_2d.png', dpi=200)
            plt.close()
        else:
            self.logger.warning("UMAP is not installed. Skipping UMAP plot. Install 'umap-learn' to enable.")

    def run(self) -> None:
        self.load_model()
        subjects = self.config['data']['subjects']
        all_rows = []
        for subj in subjects:
            m = self.evaluate_subject(subj)
            all_rows.append({'subject': subj, **m})

        if all_rows and self.config['output'].get('save_csv', True):
            pd.DataFrame(all_rows).to_csv(self.eval_dir / 'summary_all_subjects.csv', index=False)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(create_default_config(), user_cfg)


def main():
    parser = argparse.ArgumentParser(description='Run GBM sliding next-step evaluation')
    parser.add_argument('--generate-config', type=str, help='Write a default config YAML and exit')
    parser.add_argument('--config', type=str, help='Path to evaluation YAML config')
    args = parser.parse_args()

    if args.generate_config:
        save_default_config(args.generate_config)
        print(f"Default config saved to: {args.generate_config}")
        return

    if not args.config:
        parser.error('The --config argument is required unless --generate-config is used.')

    cfg = load_config(args.config)
    if not cfg['evaluation']['model_path']:
        raise ValueError('evaluation.model_path must be set in the config')

    evaluator = SlidingNextEvaluator(cfg)
    evaluator.run()
    print(f"Done. Results under: {evaluator.eval_dir}")


if __name__ == '__main__':
    main()


