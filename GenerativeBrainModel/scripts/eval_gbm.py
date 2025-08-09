#!/usr/bin/env python3
"""
Neuron-based GBM Evaluation Script

Evaluates a trained GBM model on neuron-level spike probability data produced by
unified_spike_processing.py. Computes loss and PR-AUC, writes CSV logs, and
produces comparison videos (next-step and autoregression) using the median-Z plane.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.metrics import CombinedMetricsTracker, pr_auc_binned
from GenerativeBrainModel.visualizations import create_nextstep_video, create_autoregression_video


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'gbm_neural_evaluation',
            'description': 'Evaluation of GBM on neuron-level sequences',
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [],               # if empty: random split used by dataloader
            'sampling_rate': None,             # optional: force a sampling rate group
            'use_cache': False,
        },
        'model': {
            'checkpoint': None,                # REQUIRED: path to checkpoint saved by train_gbm.py
        },
        'eval': {
            'batch_size': 4,
            'sequence_length': 12,             # must match training
            'stride': 3,                       # must match training
            'num_workers': 2,
            'use_gpu': True,
            'threshold': 0.5,
            'num_batches': None,               # limit batches for quick eval
            'make_videos': True,
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
    videos_dir = exp_dir / 'videos'
    for p in [exp_dir, logs_dir, videos_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {'exp': exp_dir, 'logs': logs_dir, 'videos': videos_dir}


def build_logger(log_dir: Path, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger('eval_gbm')
    logger.setLevel(getattr(logging, level))
    fh = logging.FileHandler(log_dir / 'evaluation.log')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> GBM:
    state = torch.load(ckpt_path, map_location=device)
    cfg_in_ckpt = state.get('config', {})
    model_cfg = cfg_in_ckpt.get('model', None)
    if model_cfg is None:
        raise ValueError('Checkpoint missing model config')
    model = GBM(
        d_model=model_cfg['d_model'],
        d_stimuli=model_cfg.get('d_stimuli', 1),
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
    ).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, tracker: CombinedMetricsTracker, threshold: float, num_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc='Eval'), 1):
        if num_batches and batch_idx > num_batches:
            break
        spikes = batch['spikes'].to(device)           # (B, L, N)
        positions = batch['positions'].to(device)     # (B, N, 3)
        mask = batch['neuron_mask'].to(device)        # (B, N)
        stim = batch['stimulus'].to(device).float()   # (B, L)

        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1].unsqueeze(-1)
        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        loss = loss_fn(logits, x_tgt)
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        probs = torch.sigmoid(logits)
        tracker.log_validation(epoch=0, batch_idx=batch_idx, predictions=probs, targets=x_tgt, val_loss=float(loss))

    avg_loss = total_loss / max(1, total_batches)
    return {'val_loss': avg_loss}


@torch.no_grad()
def make_videos(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, videos_dir: Path, sequence_length: int, num_batches: int = 1) -> None:
    # Use a few batches to render videos
    it = iter(loader)
    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        spikes = batch['spikes'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['neuron_mask'].to(device)
        stim = batch['stimulus'].to(device).float()

        # Next-step video
        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1].unsqueeze(-1)
        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        probs = torch.sigmoid(logits)
        create_nextstep_video(x_tgt, probs, positions, videos_dir / f'nextstep_batch{i+1}.mp4')

        # Autoregression video: use half as context
        context_len = min(8, spikes.shape[1] // 2)
        n_steps = min(16, spikes.shape[1] - context_len)
        if n_steps > 0:
            init_x = spikes[:, :context_len, :]
            init_stim = stim[:, :context_len].unsqueeze(-1)
            future_stim = stim[:, context_len:context_len + n_steps].unsqueeze(-1)
            gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
            create_autoregression_video(gen_seq[:, context_len:, :], positions, videos_dir / f'autoreg_batch{i+1}.mp4')


@torch.no_grad()
def evaluate_autoregression(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, threshold: float, logs_dir: Path, num_batches: Optional[int] = None) -> Dict[str, List[float]]:
    """
    Compute per-horizon autoregression metrics (BCE and PR-AUC) to assess degradation
    as steps increase. Uses half of the available sequence as context by default.
    """
    per_step_loss_sums: List[float] = []
    per_step_auc_sums: List[float] = []
    per_step_counts: List[int] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc='Eval AR'), 1):
        if num_batches and batch_idx > num_batches:
            break
        spikes = batch['spikes'].to(device)           # (B, L, N)
        positions = batch['positions'].to(device)     # (B, N, 3)
        mask = batch['neuron_mask'].to(device)        # (B, N)
        stim = batch['stimulus'].to(device).float()   # (B, L)

        B, L, N = spikes.shape
        context_len = min(8, L // 2)
        n_steps = L - context_len
        if n_steps <= 0:
            continue

        init_x = spikes[:, :context_len, :]
        init_stim = stim[:, :context_len].unsqueeze(-1)
        future_stim = stim[:, context_len:].unsqueeze(-1)
        gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
        gen_only = gen_seq[:, context_len:, :]   # (B, n_steps, N), probabilities in [0,1]
        tgt_only = spikes[:, context_len:, :]    # (B, n_steps, N)

        # Ensure accumulator sizes
        if len(per_step_loss_sums) < n_steps:
            extend_by = n_steps - len(per_step_loss_sums)
            per_step_loss_sums.extend([0.0] * extend_by)
            per_step_auc_sums.extend([0.0] * extend_by)
            per_step_counts.extend([0] * extend_by)

        # Per-horizon metrics
        for k in range(n_steps):
            preds_k = gen_only[:, k, :].detach()
            targs_k = tgt_only[:, k, :].detach()
            # BCE on probabilities
            loss_k = F.binary_cross_entropy(preds_k, targs_k, reduction='mean').item()
            # PR AUC using binned computation
            auc_k = pr_auc_binned(preds_k.flatten(), targs_k.flatten(), threshold=threshold)
            per_step_loss_sums[k] += loss_k
            per_step_auc_sums[k] += auc_k
            per_step_counts[k] += 1

    # Averages
    per_step_loss = [per_step_loss_sums[k] / max(1, per_step_counts[k]) for k in range(len(per_step_counts))]
    per_step_auc = [per_step_auc_sums[k] / max(1, per_step_counts[k]) for k in range(len(per_step_counts))]

    # Write CSV
    out_csv = logs_dir / 'autoregression_per_step.csv'
    with open(out_csv, 'w') as f:
        f.write('step,bce,pr_auc,count\n')
        for k in range(len(per_step_counts)):
            f.write(f"{k+1},{per_step_loss[k]:.6f},{per_step_auc[k]:.6f},{per_step_counts[k]}\n")

    # Optional quick plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        steps = np.arange(1, len(per_step_counts) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = 'tab:red'
        ax1.set_xlabel('Horizon (steps)')
        ax1.set_ylabel('BCE', color=color)
        ax1.plot(steps, per_step_loss, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('PR AUC', color=color)
        ax2.plot(steps, per_step_auc, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        fig.savefig(logs_dir / 'autoregression_per_step.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass

    return {'per_step_bce': per_step_loss, 'per_step_pr_auc': per_step_auc}


def main():
    parser = argparse.ArgumentParser(description='Evaluate GBM on neuron sequences')
    parser.add_argument('--config', type=str, required=False, help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    # Setup experiment dirs
    base_dir = Path('experiments/gbm_neural_eval')
    dirs = setup_experiment_dirs(base_dir, cfg['experiment']['name'])
    with open(dirs['exp'] / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2)
    logger = build_logger(dirs['logs'], cfg['logging'].get('log_level', 'INFO'))

    # Device
    device = torch.device('cuda' if (cfg['eval']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    # Load model
    ckpt = cfg['model']['checkpoint']
    if ckpt is None or not Path(ckpt).exists():
        raise ValueError('Please specify model.checkpoint path to a training checkpoint')
    model = load_model_from_checkpoint(Path(ckpt), device)

    # Use sampling rate group from checkpoint if available
    state = torch.load(ckpt, map_location='cpu')
    ckpt_cfg = state.get('config', {})
    if 'data' in ckpt_cfg and ckpt_cfg['data'].get('sampling_rate') is not None:
        cfg['data']['sampling_rate'] = ckpt_cfg['data']['sampling_rate']

    # Build dataloaders; we'll use the test loader to evaluate
    # Ensure eval-specific sequence params are reflected
    cfg['training'] = cfg.get('training', {})
    cfg['training']['batch_size'] = cfg['eval']['batch_size']
    cfg['training']['sequence_length'] = cfg['eval']['sequence_length']
    cfg['training']['stride'] = cfg['eval']['stride']
    cfg['training']['num_workers'] = cfg['eval']['num_workers']
    train_loader, test_loader = create_dataloaders(cfg)

    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.0, val_threshold=cfg['eval']['threshold'], enable_plots=True)
    metrics = evaluate(model, test_loader, device, tracker, threshold=cfg['eval']['threshold'], num_batches=cfg['eval']['num_batches'])
    logger.info(f"Evaluation metrics: {metrics}")
    tracker.plot_training()

    # Autoregression per-step metrics
    try:
        ar_metrics = evaluate_autoregression(model, test_loader, device, threshold=cfg['eval']['threshold'], logs_dir=dirs['logs'], num_batches=cfg['eval']['num_batches'])
        logger.info(f"Autoregression per-step metrics saved. First steps: BCE={ar_metrics['per_step_bce'][:3]}, AUC={ar_metrics['per_step_pr_auc'][:3]}")
    except Exception as e:
        logger.warning(f"Autoregression metrics failed: {e}")

    if cfg['eval']['make_videos']:
        try:
            make_videos(model, test_loader, device, dirs['videos'], sequence_length=cfg['eval']['sequence_length'], num_batches=1)
        except Exception as e:
            logger.warning(f"Video generation failed: {e}")


if __name__ == '__main__':
    main()


