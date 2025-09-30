#!/usr/bin/env python3
"""
Evaluation script that mirrors train_gbm's aggregated validation.

Loads a checkpoint, builds dataloaders exactly like training, and runs the
same aggregated validation routine used at end-of-epoch in train_gbm.py.

Outputs a single-row validation.csv entry (epoch=1,batch_idx=1) and plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.scripts.eval_gbm import load_model_from_checkpoint, setup_experiment_dirs, build_logger
from GenerativeBrainModel.scripts.train_gbm import deep_update
from GenerativeBrainModel.utils.sas import sas_nll
from tqdm import tqdm


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'gbm_neural_evaluation_trainstyle',
        },
        'data': {
            'data_dir': 'processed_spike_rates_2018',
            'test_subjects': [],
            'use_cache': True,
            # Optional domain flags (not required for SAS; kept for compatibility)
            'spikes_are_rates': False,
            'spikes_are_zcalcium': False,
            # Optional: override dataset name if needed (defaults handled by dataloader)
            # 'spikes_dataset_name': 'neuron_values',
        },
        'model': {
            'checkpoint': None,
        },
        'eval': {
            'batch_size': 1,
            'sequence_length': 12,
            'stride': 1,
            'num_workers': 4,
            'use_gpu': True,
        },
        'logging': {
            'log_level': 'INFO',
        },
    }


def align_eval_with_ckpt(cfg: Dict[str, Any]) -> None:
    """Align eval sequence_length/stride/test_subjects with training config saved in checkpoint when not overridden."""
    ckpt_path = cfg.get('model', {}).get('checkpoint')
    if not ckpt_path:
        return
    try:
        state_meta = torch.load(Path(ckpt_path), map_location='cpu', weights_only=False)
        ckpt_cfg = state_meta.get('config', {}) if isinstance(state_meta, dict) else {}
        tr_cfg = ckpt_cfg.get('training', {}) if isinstance(ckpt_cfg, dict) else {}
        data_cfg = ckpt_cfg.get('data', {}) if isinstance(ckpt_cfg, dict) else {}
        # Only override if eval didn't specify
        if 'sequence_length' not in cfg['eval'] or cfg['eval']['sequence_length'] is None:
            cfg['eval']['sequence_length'] = tr_cfg.get('sequence_length', cfg['eval'].get('sequence_length'))
        if 'stride' not in cfg['eval'] or cfg['eval']['stride'] is None:
            cfg['eval']['stride'] = tr_cfg.get('stride', cfg['eval'].get('stride'))
        # Inherit explicit test subjects from training config if not provided
        if (not cfg['data'].get('test_subjects')) and isinstance(data_cfg, dict) and data_cfg.get('test_subjects'):
            cfg['data']['test_subjects'] = list(data_cfg['test_subjects'])
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate GBM using train-style aggregated validation')
    parser.add_argument('--config', type=str, required=False, help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    # Align eval params with training config in checkpoint if missing
    align_eval_with_ckpt(cfg)

    # Setup experiment dirs and logger
    base_dir = Path('experiments/gbm_neural_eval_trainstyle')
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
    if not ckpt or not Path(ckpt).exists():
        raise ValueError('Please specify model.checkpoint (path to checkpoint saved by train_gbm.py)')
    model = load_model_from_checkpoint(Path(ckpt), device)
    if device.type == 'cuda':
        model = model.to(dtype=torch.bfloat16)
    # Mirror flags used in train/eval (optional; kept for compatibility)
    try:
        model.spikes_are_rates = bool(cfg['data'].get('spikes_are_rates', False))
    except Exception:
        model.spikes_are_rates = False
    try:
        model.spikes_are_zcalcium = bool(cfg['data'].get('spikes_are_zcalcium', False))
    except Exception:
        model.spikes_are_zcalcium = False

    # Build dataloaders exactly like training (using eval params threaded into training keys)
    cfg['training'] = cfg.get('training', {})
    cfg['training']['batch_size'] = cfg['eval']['batch_size']
    cfg['training']['sequence_length'] = cfg['eval']['sequence_length']
    cfg['training']['stride'] = cfg['eval']['stride']
    cfg['training']['num_workers'] = cfg['eval']['num_workers']
    cfg['training']['only_test'] = False
    # If explicit test subjects are provided and exist, restrict loader to test-only
    data_dir = Path(cfg['data']['data_dir'])
    test_subjects = cfg['data'].get('test_subjects', [])
    if isinstance(test_subjects, list) and len(test_subjects) > 0:
        missing = [s for s in test_subjects if not (data_dir / f"{s}.h5").exists()]
        if missing:
            raise ValueError(f"Requested test subjects not found in {data_dir}: {missing}")
        cfg['training']['only_test'] = True

    train_loader, val_loader, *_ = create_dataloaders(cfg)
    if len(val_loader) == 0:
        raise RuntimeError(
            'Validation loader returned zero batches. Verify data_dir, sequence_length, stride, and test_subjects.'
        )

    # Metrics tracker
    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.01, val_threshold=0.5, enable_plots=True)

    # Run train-style aggregated validation without storing all predictions (streaming average to avoid OOM)
    total_loss = 0.0
    total_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation E1"):
            spikes = batch['spikes'].to(device).to(torch.bfloat16)
            positions = batch['positions'].to(device)
            mask = batch['neuron_mask'].to(device)
            stim = batch['stimulus'].to(device).to(torch.bfloat16)

            x_in = spikes[:, :-1, :]
            x_tgt = spikes[:, 1:, :].float()
            stim_in = stim[:, :-1, :]

            mu_val, log_sigma_val, eta_val, log_delta_val = model(x_in, stim_in, positions, mask, get_logits=True)
            mask_exp = mask[:, None, :]
            if mask_exp.shape != x_tgt.shape:
                mask_exp = mask_exp.expand_as(x_tgt)
            loss = sas_nll(
                x_tgt.float(),
                mu_val.float(),
                log_sigma_val.float(),
                eta_val.float(),
                log_delta_val.float(),
                mask=mask_exp.float(),
            )
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    # Log a single aggregate row; AUC disabled, so predictions/targets are unused
    try:
        tracker.log_validation(epoch=1, batch_idx=1, predictions=torch.zeros(1), targets=torch.zeros(1), val_loss=avg_loss, compute_auc=False)
        logger.info({ 'val_loss': avg_loss })
        tracker.plot_training()
    except Exception:
        pass


if __name__ == '__main__':
    main()



