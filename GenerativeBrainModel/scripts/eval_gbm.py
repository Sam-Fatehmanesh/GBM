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
import h5py

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.metrics import CombinedMetricsTracker, pr_auc_binned
from GenerativeBrainModel.visualizations import create_nextstep_video, create_autoregression_video
import numpy as np
import torch.nn.functional as F
import h5py
import numpy as np
import torch.nn.functional as F


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'gbm_neural_evaluation',
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [],               # if empty: random split used by dataloader
            'use_cache': False,
        },
        'model': {
            'checkpoint': None,                # REQUIRED: path to checkpoint saved by train_gbm.py
        },
        'eval': {
            'batch_size': 4,
            'sequence_length': 12,             # must match training
            'stride': 1,                       # use stride=1 to cover dataset for PCA/UMAP
            'num_workers': 2,
            'use_gpu': True,
            'threshold': 0.5,
            'num_batches_nextstep': None,      # batches to use for next-step eval/embeddings (None=all)
            'num_batches_ar': None,            # batches to use for AR metrics (None=all)
            'make_videos': True,               # also controls H5 saving when True
            'load_from': None,                 # reuse saved eval dir (videos/*.h5) to recompute plots/metrics
            'pca_umap_max_samples': 20000,     # cap timepoints to embed
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
    """Robustly load a GBM model from checkpoint.
    - Strips 'module.' prefixes (DDP checkpoints)
    - Infers model config from state_dict if missing or incomplete
    - Loads with strict=False but warns if many keys are missing
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_in_ckpt = state.get('config', {}) if isinstance(state, dict) else {}
    model_cfg = cfg_in_ckpt.get('model', None) if isinstance(cfg_in_ckpt, dict) else None
    sd_in = state['model'] if (isinstance(state, dict) and 'model' in state) else state

    # Normalize keys from DDP and torch.compile wrappers
    def _normalize_key(k: str) -> str:
        # Strip optional leading wrappers repeatedly
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

    # Helper: infer minimal config from state_dict when missing or incomplete
    def infer_config_from_state_dict(sdict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        d_model = None
        # Prefer head weight [N, d_model] where N may be 1 in some setups
        for key, tensor in sdict.items():
            if key.endswith('head.1.weight') or key.endswith('neuron_scalar_decoder_head.1.weight'):
                if tensor.ndim == 2:
                    d_model = int(tensor.shape[1])
                    break
        if d_model is None:
            for key, tensor in sdict.items():
                if 'stimuli_encoder.0.weight' in key and tensor.ndim == 2:
                    d_model = int(tensor.shape[0])
                    break
        if d_model is None:
            raise ValueError('Unable to infer d_model from checkpoint state_dict')
        d_stimuli = 1
        for key, tensor in sdict.items():
            if 'stimuli_encoder.0.weight' in key and tensor.ndim == 2:
                d_stimuli = int(tensor.shape[1])
                break
        layer_indices = set()
        for k in sdict.keys():
            if k.startswith('layers.'):
                try:
                    idx = int(k.split('.')[1])
                    layer_indices.add(idx)
                except Exception:
                    pass
        n_layers = max(layer_indices) + 1 if layer_indices else 1
        n_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)
        return {'d_model': d_model, 'd_stimuli': d_stimuli, 'n_layers': n_layers, 'n_heads': n_heads}

    # Build model config
    if not isinstance(model_cfg, dict) or any(k not in model_cfg for k in ('d_model', 'n_heads', 'n_layers')) or model_cfg.get('d_stimuli') in (None, 0):
        inferred = infer_config_from_state_dict(sd)
        d_model = inferred['d_model'] if not (isinstance(model_cfg, dict) and model_cfg.get('d_model')) else model_cfg['d_model']
        d_stimuli = inferred['d_stimuli'] if not (isinstance(model_cfg, dict) and model_cfg.get('d_stimuli')) else model_cfg['d_stimuli']
        n_layers = inferred['n_layers'] if not (isinstance(model_cfg, dict) and model_cfg.get('n_layers') is not None) else model_cfg['n_layers']
        n_heads = inferred['n_heads'] if not (isinstance(model_cfg, dict) and model_cfg.get('n_heads') is not None) else model_cfg['n_heads']
    else:
        d_model = model_cfg['d_model']
        d_stimuli = model_cfg.get('d_stimuli', 1)
        n_layers = model_cfg['n_layers']
        n_heads = model_cfg['n_heads']

    model = GBM(d_model=d_model, d_stimuli=d_stimuli, n_heads=n_heads, n_layers=n_layers).to(device)

    # Pre-shape routing centroid buffers to match checkpoint (avoid size-mismatch errors)
    try:
        for k, v in list(sd.items()):
            if not k.endswith('.centroids'):
                continue
            module_path = k.rsplit('.', 1)[0]  # drop '.centroids'
            # Walk the module tree
            cur = model
            for part in module_path.split('.'):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    # Handle numeric indices for ModuleList
                    try:
                        idx = int(part)
                        cur = cur[idx]
                    except Exception:
                        cur = None
                        break
            if cur is not None and hasattr(cur, 'centroids'):
                try:
                    setattr(cur, 'centroids', v.detach().clone().to(device=next(model.parameters()).device))
                except Exception:
                    pass
    except Exception:
        pass

    # Load weights (strict=False) and warn if many keys are missing
    incompatible = model.load_state_dict(sd, strict=False)
    try:
        missing = getattr(incompatible, 'missing_keys', [])
        unexpected = getattr(incompatible, 'unexpected_keys', [])
        total = len(sd.keys())
        loaded = total - len(unexpected)
        if total > 0 and loaded / total < 0.5:
            print(f"[eval_gbm] Warning: Loaded only {loaded}/{total} keys from checkpoint. Model config may be mismatched.")
            if len(unexpected) > 0:
                print(f"[eval_gbm] Unexpected keys (first 5): {unexpected[:5]}")
            if len(missing) > 0:
                print(f"[eval_gbm] Missing keys (first 5): {missing[:5]}")
    except Exception:
        pass

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
        spikes = batch['spikes'].to(device).to(torch.bfloat16)           # (B, L, N)
        positions = batch['positions'].to(device)     # (B, N, 3)
        mask = batch['neuron_mask'].to(device)        # (B, N)
        stim = batch['stimulus'].to(device).to(torch.bfloat16)           # (B, L, K)

        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1, :]
        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        loss = loss_fn(logits.float(), x_tgt.float())
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        probs = torch.sigmoid(logits)
        tracker.log_validation(epoch=0, batch_idx=batch_idx, predictions=probs, targets=x_tgt.float(), val_loss=float(loss.detach().cpu().item()))

    avg_loss = total_loss / max(1, total_batches)
    return {'val_loss': avg_loss}


@torch.no_grad()
def make_videos(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, videos_dir: Path, sequence_length: int, num_batches: int = 1, use_double_window_for_ar: bool = False) -> None:
    # Use a few batches to render videos
    it = iter(loader)
    # Aggregators for a single unified H5 output
    ns_truth_list, ns_pred_list, ns_pos_list = [], [], []
    ns_mask_list = []
    ar_truth_list, ar_pred_list = [], []
    ar_context_list, ar_nsteps_list = [], []
    for i in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        spikes = batch['spikes'].to(device).to(torch.bfloat16)
        positions = batch['positions'].to(device)
        mask = batch['neuron_mask'].to(device)
        stim = batch['stimulus'].to(device).to(torch.bfloat16)

        # Next-step video
        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1, :]
        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        probs = torch.sigmoid(logits)
        create_nextstep_video(x_tgt, probs, positions, videos_dir / f'nextstep_batch{i+1}.mp4')

        # Accumulate next-step truth/pred/positions
        ns_truth_list.append(x_tgt.detach().cpu())
        ns_pred_list.append(probs.detach().cpu())
        ns_pos_list.append(positions.detach().cpu())
        ns_mask_list.append(mask.detach().cpu())

        # Autoregression video
        L = int(spikes.shape[1])
        if use_double_window_for_ar:
            # Expect windows of size 2L0; use first half as context, second half as truth horizon
            context_len = L // 2
            n_steps = context_len
        else:
            # Backward compat: use full-1 context inside single window
            context_len = L - 1 if L > 1 else 1
            n_steps = context_len
        if n_steps > 0:
            init_x = spikes[:, :context_len, :]
            init_stim = stim[:, :context_len, :]
            future_real = stim[:, context_len:context_len + n_steps, :]
            if future_real.shape[1] < n_steps:
                pad = torch.zeros((future_real.shape[0], n_steps - future_real.shape[1], future_real.shape[2]), device=future_real.device, dtype=future_real.dtype)
                future_stim = torch.cat([future_real, pad], dim=1)
            else:
                future_stim = future_real
            gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
            gen_only = gen_seq[:, -n_steps:, :]
            truth_only = spikes[:, context_len:context_len + n_steps, :]
            create_autoregression_video(gen_only, positions, videos_dir / f'autoreg_batch{i+1}.mp4', truth=truth_only)
            # Accumulate AR truth/pred and metadata
            ar_truth_list.append(truth_only.detach().cpu())
            ar_pred_list.append(gen_only.detach().cpu())
            ar_context_list.append(int(context_len))
            ar_nsteps_list.append(int(n_steps))

    # Write a single unified H5 with all collected arrays
    try:
        out_h5 = videos_dir / 'eval_data.h5'
        with h5py.File(out_h5, 'w') as f:
            if ns_truth_list:
                f.create_dataset('truth_next_step', data=torch.cat(ns_truth_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('pred_next_step', data=torch.cat(ns_pred_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                # For positions, keep only the first (assumed constant per subject); else stack
                try:
                    pos_cat = torch.cat(ns_pos_list, dim=0).numpy()
                    f.create_dataset('positions', data=pos_cat, compression='gzip', compression_opts=1)
                except Exception:
                    f.create_dataset('positions', data=ns_pos_list[0].numpy(), compression='gzip', compression_opts=1)
                try:
                    f.create_dataset('neuron_mask', data=torch.cat(ns_mask_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                except Exception:
                    pass
            if ar_truth_list:
                f.create_dataset('truth_future', data=torch.cat(ar_truth_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('pred_future', data=torch.cat(ar_pred_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('ar_context_len', data=np.array(ar_context_list, dtype=np.int32))
                f.create_dataset('ar_n_steps', data=np.array(ar_nsteps_list, dtype=np.int32))
    except Exception:
        pass


@torch.no_grad()
def evaluate_autoregression(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, threshold: float, logs_dir: Path, num_batches: Optional[int] = None, use_double_window: bool = True) -> Dict[str, List[float]]:
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
        spikes = batch['spikes'].to(device).to(torch.bfloat16)   # (B, L, N)
        positions = batch['positions'].to(device)               # (B, N, 3)
        mask = batch['neuron_mask'].to(device)                  # (B, N)
        stim = batch['stimulus'].to(device).to(torch.bfloat16)  # (B, L, K)

        B, L, N = spikes.shape
        if use_double_window:
            # windows are size 2L0; use first half as context, second half as horizon
            context_len = L // 2
            n_steps = context_len
        else:
            context_len = L - 1 if L > 1 else 1
            n_steps = context_len
        if n_steps <= 0:
            continue

        init_x = spikes[:, :context_len, :]
        init_stim = stim[:, :context_len, :]
        future_stim = stim[:, context_len:, :]
        gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
        gen_only = gen_seq[:, -n_steps:, :]   # (B, n_steps, N), probabilities in [0,1]
        tgt_only = spikes[:, context_len:context_len + n_steps, :]    # (B, n_steps, N)

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

    # Load model if running online; otherwise allow offline analysis via load_from
    ckpt = cfg['model']['checkpoint']
    load_from = cfg['eval'].get('load_from')
    model: Optional[GBM] = None
    if ckpt and Path(ckpt).exists():
        model = load_model_from_checkpoint(Path(ckpt), device)
        if device.type == 'cuda':
            model = model.to(dtype=torch.bfloat16)
    else:
        if not load_from:
            raise ValueError('Please specify model.checkpoint or eval.load_from (path to eval_data.h5)')

    # Align eval defaults with training config in checkpoint when not overridden
    try:
        state_meta = torch.load(Path(ckpt), map_location='cpu', weights_only=False)
        ckpt_cfg = state_meta.get('config', {}) if isinstance(state_meta, dict) else {}
        tr_cfg = ckpt_cfg.get('training', {}) if isinstance(ckpt_cfg, dict) else {}
        data_cfg = ckpt_cfg.get('data', {}) if isinstance(ckpt_cfg, dict) else {}
        # Only override if eval didn't specify
        if 'sequence_length' not in cfg['eval'] or cfg['eval']['sequence_length'] is None:
            cfg['eval']['sequence_length'] = tr_cfg.get('sequence_length', cfg['eval']['sequence_length'])
        if 'stride' not in cfg['eval'] or cfg['eval']['stride'] is None:
            cfg['eval']['stride'] = tr_cfg.get('stride', cfg['eval']['stride'])
        if cfg['data'].get('test_subjects') in (None, [], ()):  # inherit test subjects if present
            if isinstance(data_cfg, dict) and data_cfg.get('test_subjects'):
                cfg['data']['test_subjects'] = list(data_cfg['test_subjects'])
    except Exception:
        pass

    # If offline (load_from), skip dataloaders and metrics that require the model
    offline_mode = (model is None)

    # Build next-step dataloader with context size L = eval.sequence_length
    cfg['training'] = cfg.get('training', {})
    cfg['training']['batch_size'] = cfg['eval']['batch_size']
    cfg['training']['sequence_length'] = cfg['eval']['sequence_length']
    cfg['training']['stride'] = cfg['eval']['stride']
    cfg['training']['num_workers'] = cfg['eval']['num_workers']
    cfg['training']['only_test'] = True  # enforce strict use of provided test subjects
    # Enforce explicit test subjects if provided; do NOT fall back to random split
    data_dir = Path(cfg['data']['data_dir'])
    test_subjects = cfg['data'].get('test_subjects', [])
    if isinstance(test_subjects, list) and len(test_subjects) > 0:
        missing = [s for s in test_subjects if not (data_dir / f"{s}.h5").exists()]
        if missing:
            raise ValueError(f"Requested test subjects not found in {data_dir}: {missing}")
    _, ns_loader, *_ = create_dataloaders(cfg)

    # Build AR dataloader with window size 2L (first L context, second L horizon)
    import copy as _copy
    cfg_ar = _copy.deepcopy(cfg)
    L_ctx = int(cfg['eval']['sequence_length'])
    cfg_ar['training']['sequence_length'] = 2 * L_ctx
    _, ar_loader, *_ = create_dataloaders(cfg_ar)

    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.0, val_threshold=cfg['eval']['threshold'], enable_plots=True)
    if not offline_mode:
        metrics = evaluate(model, ns_loader, device, tracker, threshold=cfg['eval']['threshold'], num_batches=cfg['eval']['num_batches_nextstep'])
        logger.info(f"Evaluation metrics: {metrics}")
        tracker.plot_training()

    # Autoregression per-step metrics
    if not offline_mode:
        try:
            ar_metrics = evaluate_autoregression(model, ar_loader, device, threshold=cfg['eval']['threshold'], logs_dir=dirs['logs'], num_batches=cfg['eval']['num_batches_ar'])
            logger.info(f"Autoregression per-step metrics saved. First steps: BCE={ar_metrics['per_step_bce'][:3]}, AUC={ar_metrics['per_step_pr_auc'][:3]}")
        except Exception as e:
            logger.warning(f"Autoregression metrics failed: {e}")

    if cfg['eval']['make_videos'] and not offline_mode:
        try:
            # Use AR loader so the unified H5 contains both next-step and AR using 2L windows
            make_videos(model, ar_loader, device, dirs['videos'], sequence_length=cfg['eval']['sequence_length'], num_batches=1, use_double_window_for_ar=True)
        except Exception as e:
            logger.warning(f"Video generation failed: {e}")

    # PCA/UMAP embeddings of next-step Truth vs Pred over stride=1 validation
    try:
        from sklearn.decomposition import PCA
        import umap
        max_samples = int(cfg['eval'].get('pca_umap_max_samples', 20000))
        all_truth, all_pred = [], []
        truth_means_list, pred_means_list = [], []
        if offline_mode and load_from and Path(load_from).exists():
            # Offline: load precomputed datasets
            with h5py.File(load_from, 'r') as f:
                if 'truth_next_step' in f and 'pred_next_step' in f:
                    t = torch.from_numpy(f['truth_next_step'][()])
                    p = torch.from_numpy(f['pred_next_step'][()])
                    all_truth.append(t)
                    all_pred.append(p)
                    # Optional neuron_mask for masked means
                    nm = torch.from_numpy(f['neuron_mask'][()]) if 'neuron_mask' in f else None
                    if nm is not None:
                        B = t.shape[0]
                        Lm = t.shape[1]
                        mask_rep = nm[:, None, :].expand(B, Lm, nm.shape[-1]).to(t.dtype)
                        denom = mask_rep.sum(dim=-1).clamp_min(1)
                        truth_means_list.append((t * mask_rep).sum(dim=-1) / denom)
                        pred_means_list.append((p * mask_rep).sum(dim=-1) / denom)
                    else:
                        truth_means_list.append(t.mean(dim=-1))
                        pred_means_list.append(p.mean(dim=-1))
        else:
            with torch.no_grad():
                for batch in tqdm(ns_loader, desc='Embed next-step'):
                    spikes = batch['spikes'].to(device).to(torch.bfloat16)
                    positions = batch['positions'].to(device)
                    mask = batch['neuron_mask'].to(device)
                    stim = batch['stimulus'].to(device).to(torch.bfloat16)
                    x_in = spikes[:, :-1, :]
                    x_tgt = spikes[:, 1:, :]
                    stim_in = stim[:, :-1, :]
                    logits = model(x_in, stim_in, positions, mask, get_logits=True)
                    probs = torch.sigmoid(logits).detach()  # keep on device for masked means
                    all_truth.append(x_tgt.detach().float().cpu())
                    all_pred.append(probs.float().cpu())
                    mask_exp = mask[:, None, :].to(x_tgt.dtype)
                    mask_rep = mask_exp.expand(-1, x_tgt.shape[1], -1)
                    denom = mask_rep.sum(dim=-1).clamp_min(1)
                    truth_means_list.append((x_tgt.float() * mask_rep).sum(dim=-1) / denom)
                    pred_means_list.append((probs.float() * mask_rep).sum(dim=-1) / denom)
                # No early stop: use all next-step timepoints across all batches
        if all_truth:
            truth_np = torch.cat([t.reshape(-1, t.shape[-1]) for t in all_truth], dim=0).numpy()
            pred_np  = torch.cat([p.reshape(-1, p.shape[-1]) for p in all_pred], dim=0).numpy()
            n = min(len(truth_np), max_samples)
            truth_np = truth_np[:n]
            pred_np  = pred_np[:n]
            # PCA 2D
            pca = PCA(n_components=2, random_state=42)
            truth_pca = pca.fit_transform(truth_np)
            pred_pca  = pca.transform(pred_np)
            # UMAP 2D
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
            umap_truth = reducer.fit_transform(truth_np)
            umap_pred  = reducer.transform(pred_np)
            # Plots
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(10,4))
            axs[0].scatter(truth_pca[:,0], truth_pca[:,1], s=2, alpha=0.5, label='Truth')
            axs[0].scatter(pred_pca[:,0],  pred_pca[:,1],  s=2, alpha=0.5, label='Pred')
            axs[0].set_title('PCA (2D)')
            axs[0].legend(markerscale=4)
            axs[1].scatter(umap_truth[:,0], umap_truth[:,1], s=2, alpha=0.5, label='Truth')
            axs[1].scatter(umap_pred[:,0],  umap_pred[:,1],  s=2, alpha=0.5, label='Pred')
            axs[1].set_title('UMAP (2D)')
            axs[1].legend(markerscale=4)
            plt.tight_layout()
            plt.savefig(dirs['logs'] / 'embedding_pca_umap.png', dpi=120, bbox_inches='tight')
            plt.close(fig)

            # Mean activation comparison using masked means (scatter over all timepoints)
            truth_mean = torch.cat([t.reshape(-1) for t in truth_means_list], dim=0).numpy()
            pred_mean  = torch.cat([p.reshape(-1) for p in pred_means_list], dim=0).numpy()
            order = np.argsort(truth_mean)
            truth_sorted = truth_mean[order]
            pred_sorted  = pred_mean[order]
            plt.figure(figsize=(6,4))
            plt.scatter(truth_sorted, pred_sorted, s=6, alpha=0.6, label='Predicted vs True')
            # Ideal line y=x over same domain
            x_min, x_max = float(truth_sorted.min()), float(truth_sorted.max())
            xs = np.linspace(x_min, x_max, 100)
            plt.plot(xs, xs, 'k--', label='Ideal y=x', alpha=0.6)
            plt.xlabel('True mean activation')
            plt.ylabel('Predicted mean activation')
            plt.title('Next-step mean activation: Predicted vs True')
            plt.legend()
            plt.tight_layout()
            plt.savefig(dirs['logs'] / 'mean_activation_nextstep.png', dpi=120, bbox_inches='tight')
            plt.close()
    except Exception as e:
        logger.warning(f"Embedding (PCA/UMAP) failed: {e}")


if __name__ == '__main__':
    main()


