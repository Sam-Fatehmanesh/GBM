#!/usr/bin/env python3
"""
Training script for behavior prediction from single-timepoint neural states.

- Uses BehaviorDataset with sequence_length=1, stride=1
- Upsamples behavior_full to neural T and aligns after trimming neural zero margins
- Trains BModel to map (spikes_t, positions) -> behavior_t in [0,1]
- Logs training/validation losses and produces predicted vs true plots per behavior dim
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from GenerativeBrainModel.models.bmodel import BModel
from GenerativeBrainModel.dataloaders.behavior_dataloader import create_behavior_dataloaders, _max_neurons_in_files, _max_behavior_dim


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'bmodel_behavior_training',
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [],
            'use_cache': True,
        },
        'model': {
            'd_behavior': None,      # inferred from data
            'd_max_neurons': None,   # inferred from data
        },
        'training': {
            'batch_size': 128,
            'num_epochs': 15,
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'seed': 42,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'use_gpu': True,
            'max_timepoints_per_subject': None,
            'start_timepoint': None,
            'end_timepoint': None,
            'validation_frequency': 2,
        },
        'logging': {
            'log_level': 'INFO',
        },
    }


def setup_experiment_dirs(base_dir: Path, name: str) -> Dict[str, Path]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / f"{name}_{ts}"
    log_dir = exp_dir / 'logs'
    plots_dir = log_dir / 'plots'
    ckpt_dir = exp_dir / 'checkpoints'
    for p in [exp_dir, log_dir, plots_dir, ckpt_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {'exp': exp_dir, 'logs': log_dir, 'plots': plots_dir, 'ckpt': ckpt_dir}


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def build_logger(log_dir: Path, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger('train_bmodel')
    logger.setLevel(getattr(logging, level))
    fh = logging.FileHandler(log_dir / 'training.log')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_architecture_file(model: nn.Module, dirs: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    out_path = dirs['exp'] / 'architecture.txt'
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params
    with open(out_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BMODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for k, v in cfg['model'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total:        {total_params:,}\n")
        f.write(f"  Trainable:    {trainable_params:,}\n")
        f.write(f"  Non-trainable:{untrainable_params:,}\n\n")
        f.write("MODEL STRUCTURE (repr)\n")
        f.write("-" * 40 + "\n")
        f.write(str(model))
        f.write("\n")


def train_one_epoch(model: BModel, loader, device: torch.device, optimizer, epoch: int, logger: logging.Logger) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}", mininterval=0.1, smoothing=0.05)
    for batch in pbar:
        spikes = batch['spikes'].to(device)            # (B, 1, N)
        positions = batch['positions'].to(device)      # (B, N, 3)
        behavior = batch['behavior'].to(device)        # (B, K)

        # Flatten spikes across neuron dim with positions like model expects
        # BModel expects x: (B, T, N) with T=1
        x = spikes
        optimizer.zero_grad()
        logits = model(x, positions, get_logits=True)              # (B, 1, d_behavior)
        logits = logits.squeeze(1)                                 # (B, d_behavior)
        loss = loss_fn(logits.float(), behavior.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        pbar.set_postfix({'loss': f"{float(loss.detach().cpu().item()):.6f}"})
    avg = total_loss / max(1, total_batches)
    logger.info(f"Epoch {epoch} train loss: {avg:.6f}")
    return avg


@torch.no_grad()
def validate(model: BModel, loader, device: torch.device, epoch: int, logger: logging.Logger) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0
    # Aggregate per-file predictions and truths
    per_file: Dict[str, Dict[str, list]] = {}
    for i, batch in enumerate(tqdm(loader, desc=f"Val E{epoch}")):
        spikes = batch['spikes'].to(device)
        positions = batch['positions'].to(device)
        behavior = batch['behavior'].to(device)
        logits = model(spikes, positions, get_logits=True).squeeze(1)
        loss = loss_fn(logits.float(), behavior.float())
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        truths = behavior.detach().cpu().numpy()
        paths = batch['file_path']
        starts = batch['start_idx'].detach().cpu().numpy()
        for b in range(probs.shape[0]):
            fp = paths[b]
            if fp not in per_file:
                per_file[fp] = {'idx': [], 'pred': [], 'true': []}
            per_file[fp]['idx'].append(int(starts[b]))
            per_file[fp]['pred'].append(probs[b])
            per_file[fp]['true'].append(truths[b])
    avg = total_loss / max(1, total_batches)
    logger.info(f"Epoch {epoch} val loss: {avg:.6f}")
    # Build a single subject aggregate (first file)
    if per_file:
        first_fp = sorted(per_file.keys())[0]
        order = np.argsort(np.array(per_file[first_fp]['idx']))
        pred = np.stack(per_file[first_fp]['pred'], axis=0)[order]  # (T, K)
        true = np.stack(per_file[first_fp]['true'], axis=0)[order]  # (T, K)
        return avg, {'file_path': first_fp, 'pred': pred, 'true': true}
    return avg, None


def plot_pred_vs_true(sample: Dict[str, np.ndarray], plots_dir: Path, epoch: int) -> None:
    y_true = sample['true']   # (T, K)
    y_pred = sample['pred']   # (T, K)
    T, K = y_true.shape
    cols = min(4, K)
    rows = int(np.ceil(K / cols))
    plt.figure(figsize=(4 * cols, 2.4 * rows))
    time = np.arange(T)
    for k in range(K):
        ax = plt.subplot(rows, cols, k + 1)
        ax.plot(time, y_true[:, k], label='true', linewidth=1)
        ax.plot(time, y_pred[:, k], label='pred', linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'Dim {k}')
        if k % cols == 0:
            ax.set_ylabel('0..1')
        if k // cols == rows - 1:
            ax.set_xlabel('time')
        if k == 0:
            ax.legend(fontsize='small')
    plt.tight_layout()
    out_path = plots_dir / f'pred_vs_true_timeseries_epoch_{epoch}.png'
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train behavior predictor (BModel) from neural states')
    parser.add_argument('--config', type=str, required=False, default='configs/train_bmodel.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        # Shallow update
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    # Dirs & logger
    base_dir = Path('experiments/bmodel_behavior')
    dirs = setup_experiment_dirs(base_dir, cfg['experiment']['name'])
    save_config(cfg, dirs['exp'] / 'config.yaml')
    logger = build_logger(dirs['logs'], cfg['logging'].get('log_level', 'INFO'))

    # Device & seeds
    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    set_seeds(cfg['training']['seed'])
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    # Data
    train_loader, val_loader, _, _ = create_behavior_dataloaders(cfg)

    # Infer dims from data
    data_dir = Path(cfg['data']['data_dir'])
    all_files = sorted([str(f) for f in data_dir.glob('*.h5')])
    d_behavior = _max_behavior_dim(all_files)
    d_max_neurons = _max_neurons_in_files(all_files)
    model = BModel(d_behavior=d_behavior, d_max_neurons=d_max_neurons).to(device)

    # Save architecture summary
    try:
        write_architecture_file(model, dirs, cfg)
    except Exception as e:
        logger.warning(f"Failed to write architecture file: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

    best_val = float('inf')
    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        train_one_epoch(model, train_loader, device, optimizer, epoch, logger)
        val_loss, sample = validate(model, val_loader, device, epoch, logger)
        if sample is not None:
            plot_pred_vs_true(sample, dirs['plots'], epoch)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'config': cfg}, dirs['ckpt'] / 'best.pth')
        # Also save per-epoch checkpoint
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'config': cfg}, dirs['ckpt'] / f'epoch_{epoch}.pth')

    logger.info("Training complete.")


if __name__ == '__main__':
    main()


