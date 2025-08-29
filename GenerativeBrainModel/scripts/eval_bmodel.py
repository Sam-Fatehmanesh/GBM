#!/usr/bin/env python3
"""
Evaluate per-dimension BModel checkpoints on a target subject and visualize behavior-space PCA.

Workflow:
- Given a train_bmodel experiment directory, load the LAST epoch checkpoint per behavior dim (dim_0, dim_1, ...).
- Build a dataloader for a single target subject using BehaviorDataset (upsampled behaviors).
- Fit a 2D PCA on the subject's original behavior vectors across time (using true behavior dims only).
- Run all per-dim models to predict behaviors from neural data on the same timepoints, project with the fitted PCA.
- Generate a large set of random neuron spike windows (uniform [0,1]) with the subject's neuron positions/mask,
  run through the models to synthesize behaviors, and project with the same PCA.
- Save a scatter plot overlaying original, predicted, and random-generated behavior distributions in 2D PCA space.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from GenerativeBrainModel.models.bmodel import BModel
from GenerativeBrainModel.dataloaders.behavior_dataloader import create_behavior_dataloaders, _max_behavior_dim
import h5py


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'bmodel_behavior_evaluation',
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'target_subject': None,   # e.g., 'subject_14' (REQUIRED)
            'use_cache': True,
        },
        'model': {
            'train_experiment_dir': None,  # REQUIRED: path to train_bmodel output dir (contains checkpoints/)
        },
        'eval': {
            'batch_size': 128,
            'num_workers': 2,
            'use_gpu': True,
            'random_samples': 2000,
            'pca_max_samples': None,     # optional cap; None = use all
            'random_seed': 42,
            'clamp_predictions': True,
            'disable_predictions': False,  # when True: skip model loading, subject predictions, and random generation
            'enable_random_generation': False,  # when True: generate random spikes -> behaviors
            'behavior_first': False,       # when True: generate novel targets in PCA space and optimize neuron logits
            'novel_behavior_samples': 1000,
            'novel_batch_size': 64,
            'num_mod_images': 5,           # number of samples to visualize (each with 6 timesteps)
            'image_grid': {                # output XY grid for images (Z is mean-pooled)
                'x': 512,
                'y': 256,
            },
            'behavior_opt': {
                'steps': 200,
                'lr': 1e-1,
                'clamp_targets': True,
                'delta_penalty': 1e-3,    # L1 penalty weight on positive-only neural modifications
            },
        },
        'logging': {
            'log_level': 'INFO',
        },
    }


def deep_update(base: Dict, updates: Dict) -> Dict:
    result = base.copy()
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def setup_experiment_dirs(base_dir: Path, name: str) -> Dict[str, Path]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / f"{name}_{ts}"
    logs_dir = exp_dir / 'logs'
    for p in [exp_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {'exp': exp_dir, 'logs': logs_dir}


def build_logger(log_dir: Path, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger('eval_bmodel')
    logger.setLevel(getattr(logging, level))
    fh = logging.FileHandler(log_dir / 'evaluation.log')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def free_memory(device: torch.device) -> None:
    """Aggressively free CPU and GPU caches after large computations."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if isinstance(device, torch.device) and device.type == 'cuda':
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _subject_behavior_dim(subject_h5: Path) -> int:
    with h5py.File(subject_h5, 'r') as f:
        if 'behavior_full' not in f:
            raise ValueError(f"behavior_full not found in {subject_h5}")
        beh = f['behavior_full']
        if beh.ndim == 1:
            return 1
        elif beh.ndim == 2:
            return int(min(beh.shape[0], beh.shape[1]))
        else:
            raise ValueError(f"Unexpected behavior_full ndim={beh.ndim} in {subject_h5}")


def _list_dim_dirs(ckpt_root: Path) -> List[int]:
    dims: List[int] = []
    if not ckpt_root.exists():
        return dims
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith('dim_'):
            try:
                d = int(p.name.split('_', 1)[1])
                dims.append(d)
            except Exception:
                pass
    return sorted(set(dims))


def _last_epoch_checkpoint(dim_dir: Path) -> Optional[Path]:
    # Prefer epoch_*.pth with highest integer; fallback to best.pth
    epoch_files = list(dim_dir.glob('epoch_*.pth'))
    best_file = dim_dir / 'best.pth'
    last_path: Optional[Path] = None
    best_epoch = -1
    for fp in epoch_files:
        m = re.search(r"epoch_(\d+)\.pth$", fp.name)
        if not m:
            continue
        try:
            ep = int(m.group(1))
        except Exception:
            continue
        if ep > best_epoch:
            best_epoch = ep
            last_path = fp
    if last_path is None and best_file.exists():
        last_path = best_file
    return last_path


def load_bmodel_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[BModel, Dict[str, Any]]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_in = state.get('config', {}) if isinstance(state, dict) else {}
    sd_in = state['model'] if (isinstance(state, dict) and 'model' in state) else state

    # Normalize possible wrappers like 'module.' or '_orig_mod.'
    def _normalize_key(k: str) -> str:
        changed = True
        while changed:
            changed = False
            if k.startswith('module.'):
                k = k[len('module.'):]
                changed = True
            if k.startswith('_orig_mod.'):
                k = k[len('_orig_mod.'):]
                changed = True
        return k

    if isinstance(sd_in, dict):
        sd = {_normalize_key(k): v for k, v in sd_in.items()}
    else:
        sd = sd_in

    # Determine d_max_neurons from config if available
    model_cfg = cfg_in.get('model', {}) if isinstance(cfg_in, dict) else {}
    d_max_neurons = model_cfg.get('d_max_neurons', None)
    if d_max_neurons is None:
        # Fallback: try to infer from any linear layer expecting input shaped by neurons (not reliable). Keep None.
        pass

    model = BModel(d_behavior=1, d_max_neurons=int(d_max_neurons) if d_max_neurons is not None else 0)
    model.to(device)
    model.eval()
    # strict=False allows minor mismatches if buffers differ
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        # Retry on CPU tensors if device issues
        model.load_state_dict({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}, strict=False)
    return model, cfg_in


def build_subject_loader(data_dir: Path, subject: str, batch_size: int, num_workers: int, use_cache: bool) -> torch.utils.data.DataLoader:
    # Build a config compatible with create_behavior_dataloaders and use its val_loader for the single subject
    cfg = {
        'data': {
            'data_dir': str(data_dir),
            'test_subjects': [subject],
            'use_cache': bool(use_cache),
        },
        'training': {
            'batch_size': int(batch_size),
            'num_workers': int(num_workers),
            # BehaviorDataset has fixed sequence_length=6 and stride=1 internally
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
        }
    }
    _, val_loader, *_ = create_behavior_dataloaders(cfg)
    return val_loader


def fit_pca_2d(X: np.ndarray, random_state: int = 42) -> Tuple[object, np.ndarray]:
    """Fit a 2D PCA on X (n_samples, n_features). Returns (pca_like, X2d).
    Falls back to SVD if sklearn is unavailable.
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        X2d = pca.fit_transform(X)
        return pca, X2d
    except Exception:
        # Center and SVD fallback
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:2].T  # (D,2)
        X2d = Xc @ comps
        class _PCAFallback:
            def __init__(self, comps, mean):
                self.components_ = comps.T
                self.mean_ = mean
            def transform(self, Y):
                return (Y - self.mean_) @ comps
        return _PCAFallback(comps, X.mean(axis=0, keepdims=True)), X2d


def main():
    parser = argparse.ArgumentParser(description='Evaluate BModel behavior predictors and visualize PCA')
    parser.add_argument('--config', type=str, required=False, help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    # Validate required inputs
    train_dir = cfg['model'].get('train_experiment_dir')
    target_subject = cfg['data'].get('target_subject')
    data_dir = Path(cfg['data'].get('data_dir', 'processed_spike_voxels_2018'))
    if not train_dir:
        raise ValueError('model.train_experiment_dir must be specified in the config')
    if not target_subject:
        raise ValueError('data.target_subject must be specified in the config (e.g., subject_14)')
    train_dir = Path(train_dir)
    ckpt_root = train_dir / 'checkpoints'

    # Setup experiment dirs
    base_dir = Path('experiments/bmodel_behavior_eval')
    dirs = setup_experiment_dirs(base_dir, cfg['experiment']['name'])
    with open(dirs['exp'] / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)
    logger = build_logger(dirs['logs'], cfg['logging'].get('log_level', 'INFO'))

    # Device & seeds
    device = torch.device('cuda' if (cfg['eval']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    np.random.seed(int(cfg['eval'].get('random_seed', 42)))
    torch.manual_seed(int(cfg['eval'].get('random_seed', 42)))

    disable_preds = bool(cfg['eval'].get('disable_predictions', False))

    # Determine dims: available model dims and subject behavior dims
    all_dim_ids = _list_dim_dirs(ckpt_root) if not disable_preds else []
    if not disable_preds and not all_dim_ids:
        raise ValueError(f"No dim_* directories found under {ckpt_root}")
    subj_h5 = data_dir / f"{target_subject}.h5"
    if not subj_h5.exists():
        raise FileNotFoundError(f"Target subject file not found: {subj_h5}")
    K_subject = _subject_behavior_dim(subj_h5)
    if disable_preds:
        dims_to_use = []
        logger.info("Predictions disabled: will only compute PCA on original behavior and skip model/random generations.")
    else:
        dims_to_use = [d for d in all_dim_ids if d < K_subject]
        if not dims_to_use:
            raise ValueError(f"No overlapping dims to evaluate (subject has {K_subject} dims, available models: {all_dim_ids})")
        logger.info(f"Evaluating dims 0..{len(dims_to_use)-1} (subject K={K_subject}, available models={len(all_dim_ids)})")

    # Load last-epoch checkpoint for each required dim
    dim_models: List[Tuple[int, BModel]] = []
    if not disable_preds:
        for d in dims_to_use:
            dim_dir = ckpt_root / f"dim_{d}"
            last_ckpt = _last_epoch_checkpoint(dim_dir)
            if last_ckpt is None:
                logger.warning(f"No epoch_* checkpoint found in {dim_dir}; skipping dim {d}")
                continue
            model, _ = load_bmodel_from_checkpoint(last_ckpt, device)
            dim_models.append((d, model))
        if not dim_models:
            raise ValueError("Failed to load any per-dim models.")

    # Build subject-only dataloader
    val_loader = build_subject_loader(data_dir, target_subject, cfg['eval']['batch_size'], cfg['eval']['num_workers'], cfg['data']['use_cache'])

    # Accumulate original and predicted behaviors
    originals: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    first_positions: Optional[torch.Tensor] = None
    first_mask: Optional[torch.Tensor] = None
    seq_len_L: Optional[int] = None
    first_spikes_window: Optional[torch.Tensor] = None
    clamp_pred = bool(cfg['eval'].get('clamp_predictions', True))
    # Per-neuron running stats over the target subject (for random sampling distribution)
    running_sum = None  # type: Optional[np.ndarray]
    running_sumsq = None  # type: Optional[np.ndarray]
    running_count = None  # type: Optional[np.ndarray]
    # Global per-timepoint mean activation target (scalar)
    global_sum_all = 0.0
    global_count_all = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Predict subject behaviors'):
            spikes = batch['spikes'].to(device)       # (B, L, N)
            positions = batch['positions'].to(device) # (B, N, 3)
            mask = batch['neuron_mask'].to(device)    # (B, N)
            beh = batch['behavior'][:, :K_subject].to(device)  # (B, K_subject)

            if first_positions is None:
                first_positions = positions[0].detach().clone()
                first_mask = mask[0].detach().clone()
                seq_len_L = int(spikes.shape[1])
                try:
                    first_spikes_window = spikes[0].detach().clone()
                except Exception:
                    first_spikes_window = None

            if not disable_preds:
                # Predict per dim and stack
                pred_per_dim: List[torch.Tensor] = []
                for d, m in dim_models:
                    out = m(spikes, positions, mask, get_logits=True).squeeze(1).squeeze(-1)  # (B,1,1)->(B,)
                    if clamp_pred:
                        out = out.clamp(0.0, 1.0)
                    pred_per_dim.append(out)
                if not pred_per_dim:
                    continue
                pred_mat = torch.stack(pred_per_dim, dim=1)  # (B, K_use)
                # originals are already upsampled/padded in the loader; slice to K_use for safety
                K_use = pred_mat.shape[1]
                orig_mat = beh[:, :K_use]
                predictions.append(pred_mat.detach().cpu().numpy())
                originals.append(orig_mat.detach().cpu().numpy())
            else:
                # When predictions disabled, still collect true behavior for PCA
                originals.append(beh.detach().cpu().numpy())

            # Update per-neuron stats on CPU using mask
            try:
                sp_np = spikes.detach().cpu().numpy()   # (B,L,Npad)
                m_np = mask.detach().cpu().numpy()       # (B,Npad)
                # Broadcast mask over time dimension
                m_exp = m_np[:, None, :].astype(np.float64)
                sp64 = sp_np.astype(np.float64)
                sum_add = (sp64 * m_exp).sum(axis=(0, 1))  # (Npad,)
                sumsq_add = ((sp64 ** 2) * m_exp).sum(axis=(0, 1))
                cnt_add = m_exp.sum(axis=(0, 1))          # (Npad,) equals (#valid in batch) * L
                if running_sum is None:
                    Npad_local = int(sum_add.shape[0])
                    running_sum = np.zeros((Npad_local,), dtype=np.float64)
                    running_sumsq = np.zeros((Npad_local,), dtype=np.float64)
                    running_count = np.zeros((Npad_local,), dtype=np.float64)
                running_sum += sum_add
                running_sumsq += sumsq_add
                running_count += cnt_add
                # Global accumulators for mean activation across all frames/neurons
                global_sum_all += float((sp64 * m_exp).sum())
                global_count_all += float(m_exp.sum())
            except Exception:
                pass

    if not originals:
        raise RuntimeError("No samples collected from the subject loader.")

    # Free GPU memory used during subject prediction phase
    try:
        del spikes, positions, mask, beh, pred_mat, orig_mat, pred_per_dim  # type: ignore[name-defined]
    except Exception:
        pass
    free_memory(device)

    originals_np = np.concatenate(originals, axis=0)  # (T, K_use)
    predictions_np = np.concatenate(predictions, axis=0)  # (T, K_use)

    # Optional sample cap for PCA fit
    pca_max = cfg['eval'].get('pca_max_samples', None)
    if pca_max is not None:
        n = int(pca_max)
        originals_np = originals_np[:n]
        predictions_np = predictions_np[:n]

    # Fit PCA on originals; transform predicted
    pca, originals_2d = fit_pca_2d(originals_np, random_state=int(cfg['eval'].get('random_seed', 42)))
    pred_2d = pca.transform(predictions_np) if (not disable_preds and predictions_np.size > 0) else np.zeros((0, 2), dtype=np.float32)
    free_memory(device)

    # Random-generated behaviors using uniform/Beta spikes
    if first_positions is None or first_mask is None or seq_len_L is None:
        raise RuntimeError("Failed to capture positions/mask/sequence length from loader.")
    if disable_preds:
        gen_2d = np.zeros((0, 2), dtype=np.float32)
        # Save outputs and plot originals only
        np.savez_compressed(
            dirs['logs'] / 'pca_embeddings.npz',
            originals_2d=originals_2d,
            predicted_2d=pred_2d,
            random_2d=gen_2d,
        )
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 6))
            plt.scatter(originals_2d[:, 0], originals_2d[:, 1], s=4, alpha=0.5, label='Original')
            plt.title(f"Behavior PCA (subject={target_subject})")
            plt.legend(markerscale=3)
            plt.tight_layout()
            plt.savefig(dirs['logs'] / 'behavior_pca_scatter.png', dpi=140, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")
        logger.info("Done (predictions disabled). Saved PCA of original behavior only.")
        return

    rand_total = int(cfg['eval'].get('random_samples', 2000))
    rand_bs = int(cfg['eval'].get('random_batch_size', cfg['eval'].get('batch_size', 128)))
    Npad = int(first_positions.shape[0])
    K_use = len(dim_models)

    # Random sampling distribution: prefer a global Beta from config, else fall back to per-neuron MoM
    rb = cfg['eval'].get('random_beta', None)
    global_beta = False
    if isinstance(rb, dict) and (rb.get('mean') is not None) and (rb.get('var') is not None):
        m = float(rb['mean'])
        v = float(rb['var'])
        eps = 1e-9
        m = float(np.clip(m, eps, 1.0 - eps))
        v = float(np.clip(v, eps, m * (1.0 - m) - eps))
        k = (m * (1.0 - m) / v) - 1.0
        k = float(np.clip(k, 2.0, 1e9))
        alpha_s = float(np.clip(m * k, 1e-6, 1e12))
        beta_s = float(np.clip((1.0 - m) * k, 1e-6, 1e12))
        alpha_t = torch.tensor(alpha_s, device=device, dtype=torch.float32)
        beta_t = torch.tensor(beta_s, device=device, dtype=torch.float32)
        global_beta = True
    else:
        # Build per-neuron Beta(a,b) from observed mean/variance across the subject (MoM)
        if running_sum is None or running_count is None or running_sumsq is None:
            mean_per_neuron = np.full((Npad,), 0.02, dtype=np.float64)
            var_per_neuron = np.full((Npad,), 0.02 * (1 - 0.02) * 0.1, dtype=np.float64)
        else:
            mean_per_neuron = running_sum / np.maximum(1.0, running_count)
            second_moment = running_sumsq / np.maximum(1.0, running_count)
            var_per_neuron = np.maximum(0.0, second_moment - mean_per_neuron ** 2)
        eps = 1e-6
        mean_per_neuron = np.clip(mean_per_neuron, eps, 1.0 - eps)
        max_var = mean_per_neuron * (1.0 - mean_per_neuron)
        var_per_neuron = np.clip(var_per_neuron, 1e-8, max_var - 1e-8)
        k = (mean_per_neuron * (1.0 - mean_per_neuron) / var_per_neuron) - 1.0
        k = np.clip(k, 2.0, 1e7)
        alpha_np = mean_per_neuron * k
        beta_np = (1.0 - mean_per_neuron) * k
        alpha_t = torch.from_numpy(alpha_np.astype(np.float32)).to(device)
        beta_t = torch.from_numpy(beta_np.astype(np.float32)).to(device)

    gen_np_parts: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, rand_total, rand_bs), desc='Random generation (chunks)'):
            b = min(rand_bs, rand_total - start)
            # Sample Beta per neuron, broadcast to (b,L,N)
            try:
                dist = torch.distributions.Beta(alpha_t, beta_t)
                if global_beta:
                    spikes_rand = dist.sample((b, seq_len_L, Npad))
                else:
                    spikes_rand = dist.sample((b, seq_len_L))  # broadcasts per-neuron (N)
                spikes_rand = spikes_rand.clamp(0.0, 1.0)
            except Exception:
                # Fallback to uniform if Beta fails
                spikes_rand = torch.rand((b, seq_len_L, Npad), device=device, dtype=torch.float32)
            pos_rand = first_positions.unsqueeze(0).repeat(b, 1, 1).to(device)
            mask_rand = first_mask.unsqueeze(0).repeat(b, 1).to(device)

            gen_list: List[torch.Tensor] = []
            for d, m in dim_models:
                out = m(spikes_rand, pos_rand, mask_rand, get_logits=True).squeeze(1).squeeze(-1)
                if clamp_pred:
                    out = out.clamp(0.0, 1.0)
                gen_list.append(out)
            if gen_list:
                gen_mat = torch.stack(gen_list, dim=1)  # (b, K_use)
                gen_np_parts.append(gen_mat.detach().cpu().numpy())

            # Free chunk tensors
            try:
                del spikes_rand, pos_rand, mask_rand, gen_list, gen_mat  # type: ignore[name-defined]
            except Exception:
                pass
            free_memory(device)

    if gen_np_parts:
        gen_np = np.concatenate(gen_np_parts, axis=0)
    else:
        gen_np = np.zeros((0, K_use), dtype=np.float32)

    gen_2d = pca.transform(gen_np) if gen_np.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    free_memory(device)

    # Optional behavior-first optimization of neuron logits to hit random PCA targets
    novel_2d = np.zeros((0, 2), dtype=np.float32)
    target_2d = np.zeros((0, 2), dtype=np.float32)
    if (not disable_preds) and bool(cfg['eval'].get('behavior_first', False)):
        # Sample novel target behaviors uniformly in [0,1]^K and optimize positive-only additive neural modifications
        num_samples = int(cfg['eval'].get('novel_behavior_samples', 1000))
        bsz = int(cfg['eval'].get('novel_batch_size', 64))
        steps = int(cfg['eval']['behavior_opt'].get('steps', 200))
        lr = float(cfg['eval']['behavior_opt'].get('lr', 1e-1))
        clamp_t = bool(cfg['eval']['behavior_opt'].get('clamp_targets', True))
        delta_penalty = float(cfg['eval']['behavior_opt'].get('delta_penalty', 1e-3))
        K_use = len(dim_models)
        # Target mean activation per timepoint (scalar from data)
        target_frame_mean = float(global_sum_all / max(1.0, global_count_all))

        collected_novel = []
        collected_target2d = []
        collected_deltas: List[np.ndarray] = []
        collected_base_win: List[np.ndarray] = []
        collected_modified_win: List[np.ndarray] = []
        loss_traces: List[List[float]] = []
        # Prepare base window (L,N) from subject; fallback to zeros if unavailable
        if first_spikes_window is None:
            Npad_local = int(first_positions.shape[0])
            base_win_cpu = torch.zeros((int(seq_len_L), Npad_local), dtype=torch.float32)
        else:
            base_win_cpu = first_spikes_window.detach().cpu().to(dtype=torch.float32)
        base_win = base_win_cpu.to(device)
        base_win = base_win.clamp(0.0, 1.0)
        mask_vec = first_mask.to(device).to(dtype=base_win.dtype)  # (N)
        for start in tqdm(range(0, num_samples, bsz), desc='Behavior-first (optimize spikes)'):
            m = min(bsz, num_samples - start)
            X_target = np.random.uniform(0.0, 1.0, size=(m, K_use)).astype(np.float32)
            if clamp_t:
                X_target = np.clip(X_target, 0.0, 1.0)
            # Trainable positive-only additive modification per neuron (constant over time)
            Npad = int(first_positions.shape[0])
            Lw = int(seq_len_L)
            delta_param = torch.zeros((m, Npad), device=device, dtype=torch.float32, requires_grad=True)
            pos_b = first_positions.unsqueeze(0).repeat(m, 1, 1).to(device)
            mask_b = first_mask.unsqueeze(0).repeat(m, 1).to(device)

            opt = torch.optim.Adam([delta_param], lr=lr)
            loss_fn = nn.L1Loss()
            trace: List[float] = []
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                # Positive-only delta via softplus; zero where neuron is masked-out
                delta_now = F.softplus(delta_param) * mask_vec.unsqueeze(0)
                # Add to base window and clamp to [0,1]
                base_rep = base_win.unsqueeze(0).repeat(m, 1, 1)
                spikes_const = torch.clamp(base_rep + delta_now.unsqueeze(1).repeat(1, Lw, 1), 0.0, 1.0)
                total_loss = 0.0
                for d_idx, (d, mdl) in enumerate(dim_models):
                    out = mdl(spikes_const, pos_b, mask_b, get_logits=True).squeeze(1).squeeze(-1)
                    tgt = torch.from_numpy(X_target[:, d_idx]).to(device)
                    total_loss = total_loss + loss_fn(out.float(), tgt.float())
                # Mean-activation regularization (equal weight to behavior loss)
                mask_bf = mask_b.to(dtype=spikes_const.dtype)
                denom = mask_bf.sum(dim=1).clamp_min(1.0)
                # Compute mean over neurons of the per-neuron average (time-averaged)
                mean_now = (spikes_const.mean(dim=1) * mask_bf).sum(dim=1) / denom
                target_vec = torch.full((m,), target_frame_mean, device=device, dtype=spikes_const.dtype)
                total_loss = total_loss + loss_fn(mean_now, target_vec)
                # Penalize mean added activation per sample over valid neurons (no extra time scaling)
                valid_count = mask_vec.sum().clamp_min(1.0)
                delta_mean = delta_now.sum(dim=1) / valid_count
                total_loss = total_loss + (delta_penalty * delta_mean.mean())
                total_loss.backward()
                opt.step()
                try:
                    trace.append(float(total_loss.detach().cpu().item()))
                except Exception:
                    pass

            with torch.no_grad():
                delta_final = F.softplus(delta_param) * mask_vec.unsqueeze(0)
                probs_final = torch.clamp(base_win.unsqueeze(0).repeat(m, 1, 1) + delta_final.unsqueeze(1).repeat(1, Lw, 1), 0.0, 1.0)
                spikes_final = probs_final
                pred_dims = []
                for d_idx, (d, mdl) in enumerate(dim_models):
                    out = mdl(spikes_final, pos_b, mask_b, get_logits=True).squeeze(1).squeeze(-1)
                    pred_dims.append(out.clamp(0.0, 1.0))
                pred_mat = torch.stack(pred_dims, dim=1).detach().cpu().numpy()  # (m, K_use)
                collected_novel.append(pred_mat)
                collected_target2d.append(pca.transform(X_target))
                # Save learned neural modifications and windows
                collected_deltas.append(delta_final.detach().cpu().numpy())  # (m,N)
                collected_base_win.append(base_win.detach().cpu().numpy())   # (L,N)
                collected_modified_win.append(spikes_final.detach().cpu().numpy())  # (m,L,N)
                if trace:
                    loss_traces.append(trace)

            # free per-batch temps
            try:
                del delta_param, pos_b, mask_b, spikes_const, spikes_final, pred_dims, pred_mat
            except Exception:
                pass
            free_memory(device)

        if collected_novel:
            novel_beh = np.concatenate(collected_novel, axis=0)
            novel_2d = pca.transform(novel_beh)
            target_2d = np.concatenate(collected_target2d, axis=0)
            try:
                # Write neural modification diffs NPZ once per run
                deltas_all = np.concatenate(collected_deltas, axis=0) if collected_deltas else np.zeros((0, int(first_positions.shape[0])), dtype=np.float32)
                base_win_unique = base_win.detach().cpu().numpy()
                np.savez_compressed(
                    dirs['logs'] / 'behavior_first_neural_mods.npz',
                    delta_per_sample=deltas_all,            # (num_samples, N)
                    base_spikes_window=base_win_unique,     # (L, N)
                    positions=first_positions.detach().cpu().numpy(),
                    neuron_mask=first_mask.detach().cpu().numpy(),
                )
            except Exception:
                pass
        # Save a few modification images (2D XY, Z mean-pooled)
        try:
            import matplotlib.pyplot as plt
            plots_dir = dirs['logs'] / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            num_img = int(cfg['eval'].get('num_mod_images', 5))
            grid_cfg = cfg['eval'].get('image_grid', {"x": 512, "y": 256})
            GX = int(grid_cfg.get('x', 512))
            GY = int(grid_cfg.get('y', 256))
            # Use only valid neurons
            pos_np = first_positions.detach().cpu().numpy()
            mask_np = first_mask.detach().cpu().numpy().astype(bool)
            xy = pos_np[mask_np, :2]
            xs = np.clip((xy[:, 0] * (GX - 1)).astype(np.int32), 0, GX - 1)
            ys = np.clip((xy[:, 1] * (GY - 1)).astype(np.int32), 0, GY - 1)
            # Gather deltas (may be large); limit to first num_img
            if collected_deltas:
                deltas_show = np.concatenate(collected_deltas, axis=0)  # (S, N)
                S = deltas_show.shape[0]
                pick = min(num_img, S)
                for s in range(pick):
                    delta_s = deltas_show[s][mask_np]  # (N_valid,)
                    # Build XY mean image by accumulating sums and counts
                    img_sum = np.zeros((GY, GX), dtype=np.float32)
                    img_cnt = np.zeros((GY, GX), dtype=np.float32)
                    # vectorized accumulation via flat indices
                    flat_idx = ys * GX + xs
                    np.add.at(img_sum.ravel(), flat_idx, delta_s)
                    np.add.at(img_cnt.ravel(), flat_idx, 1.0)
                    img = np.divide(img_sum, np.clip(img_cnt, 1.0, None))
                    # Clip to [0,1] and scale to absolute [0,255] (no per-image normalization)
                    img = np.clip(img, 0.0, 1.0) * 255.0
                    # Save one image per each of the 6 timesteps (same if delta is time-constant)
                    for t in range(int(seq_len_L)):
                        out_path = plots_dir / f"modification_sample{s}_t{t}.png"
                        plt.imsave(out_path, img, cmap='gray', vmin=0.0, vmax=255.0)
        except Exception:
            pass
        # Plot average optimization loss over steps
        try:
            if loss_traces:
                import matplotlib.pyplot as plt
                import numpy as _np
                max_len = max(len(t) for t in loss_traces)
                arr = _np.full((len(loss_traces), max_len), _np.nan, dtype=_np.float64)
                for i, t in enumerate(loss_traces):
                    arr[i, :len(t)] = _np.array(t, dtype=_np.float64)
                mean_loss = _np.nanmean(arr, axis=0)
                std_loss = _np.nanstd(arr, axis=0)
                steps_axis = _np.arange(1, max_len + 1)
                with open(dirs['logs'] / 'behavior_opt_loss.csv', 'w') as fcsv:
                    fcsv.write('step,loss_mean,loss_std\n')
                    for s, mu_l, sd_l in zip(steps_axis, mean_loss, std_loss):
                        fcsv.write(f"{int(s)},{float(mu_l):.6f},{float(sd_l):.6f}\n")
                plt.figure(figsize=(7,4))
                plt.plot(steps_axis, mean_loss, label='mean loss')
                plt.fill_between(steps_axis, mean_loss - std_loss, mean_loss + std_loss, color='tab:blue', alpha=0.2)
                plt.xlabel('Optimization step')
                plt.ylabel('L1 loss (sum over dims)')
                plt.title('Behavior-first optimization loss')
                plt.tight_layout()
                plt.savefig(dirs['logs'] / 'behavior_opt_loss.png', dpi=140, bbox_inches='tight')
                plt.close()
        except Exception:
            pass

    # Save arrays and plot
    np.savez_compressed(
        dirs['logs'] / 'pca_embeddings.npz',
        originals_2d=originals_2d,
        predicted_2d=pred_2d,
        random_2d=gen_2d,
        novel_2d=novel_2d,
        novel_target_2d=target_2d,
    )

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        plt.scatter(originals_2d[:, 0], originals_2d[:, 1], s=4, alpha=0.5, label='Original')
        if pred_2d.shape[0] > 0:
            plt.scatter(pred_2d[:, 0], pred_2d[:, 1], s=4, alpha=0.5, label='Predicted')
        if gen_2d.shape[0] > 0 and bool(cfg['eval'].get('enable_random_generation', False)):
            plt.scatter(gen_2d[:, 0], gen_2d[:, 1], s=4, alpha=0.5, label='Random-generated')
        if novel_2d.shape[0] > 0:
            plt.scatter(novel_2d[:, 0], novel_2d[:, 1], s=10, alpha=0.8, label='Behavior-first novel')
            plt.scatter(target_2d[:, 0], target_2d[:, 1], s=10, alpha=0.8, label='Behavior-first target')
        plt.title(f"Behavior PCA (subject={target_subject})")
        plt.legend(markerscale=3)
        plt.tight_layout()
        plt.savefig(dirs['logs'] / 'behavior_pca_scatter.png', dpi=140, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    logger.info("Done. Saved PCA embeddings and scatter plot.")


if __name__ == '__main__':
    main()


