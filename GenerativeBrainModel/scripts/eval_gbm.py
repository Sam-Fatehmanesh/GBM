#!/usr/bin/env python3
"""
Neuron-based GBM Evaluation Script

Evaluates a trained GBM model on neuron-level spike data produced by
unified_spike_processing.py. Computes losses and spike-rate metrics, writes CSV logs,
and produces comparison videos (next-step and autoregression) using the median-Z plane.
Includes relative performance vs a copy-last-timestep baseline.
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
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_nextstep_video, create_autoregression_video
from GenerativeBrainModel.utils.sas import sas_nll, sas_rate_median
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
            'spikes_are_rates': False,
            'spikes_are_zcalcium': False,
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
            'sampling_rate_hz': 3.0,           # used to compute rate metrics from probabilities
            'num_batches_nextstep': None,      # batches to use for next-step eval/embeddings (None=all)
            'num_batches_ar': None,            # batches to use for AR metrics (None=all)
            'make_videos': True,               # also controls H5 saving when True
            'load_from': None,                 # reuse saved eval dir (videos/*.h5) to recompute plots/metrics
            'pca_umap_max_samples': 20000,     # cap timepoints to embed
            # spike trace plots (truth over full 2L, AR preds over horizon)
            'num_trace_images': 100,
            'neurons_per_trace_image': 4,
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
    # Flags will be set by caller based on eval config
    model.spikes_are_rates = False
    model.spikes_are_zcalcium = False

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
def evaluate(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, tracker: CombinedMetricsTracker, threshold: float, num_batches: Optional[int] = None, *, spikes_are_rates: bool = False, spikes_are_zcalcium: bool = False, sampling_rate_hz: float = 3.0) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    mae_rate_sums = 0.0
    mae_rate_counts = 0
    mae_z_sums = 0.0
    mae_z_counts = 0
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
        mu_raw, log_sigma_raw, eta_raw, log_delta_raw = model(x_in, stim_in, positions, mask, get_logits=True)
        loss = sas_nll(x_tgt.float(), mu_raw.float(), log_sigma_raw.float(), eta_raw.float(), log_delta_raw.float())
        preds = sas_rate_median(mu_raw, log_sigma_raw, eta_raw, log_delta_raw)
        abs_err = (preds - x_tgt.float()).abs()
        mae_rate_sums += float(abs_err.mean().detach().cpu().item())
        mae_rate_counts += 1
        probs = preds
        targs_for_tracker = x_tgt.float()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        tracker.log_validation(epoch=0, batch_idx=batch_idx, predictions=probs, targets=targs_for_tracker, val_loss=float(loss.detach().cpu().item()), compute_auc=False)

    avg_loss = total_loss / max(1, total_batches)
    avg_mae_rate = (mae_rate_sums / max(1, mae_rate_counts)) if mae_rate_counts > 0 else float('nan')
    avg_mae = (mae_z_sums / max(1, mae_z_counts)) if mae_z_counts > 0 else float('nan')
    out = {'val_loss': avg_loss, 'mae_rate': avg_mae_rate}
    return out


@torch.no_grad()
def make_videos(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, videos_dir: Path, sequence_length: int, num_batches: int = 1, use_double_window_for_ar: bool = False, *, spikes_are_rates: bool = False, spikes_are_zcalcium: bool = False, sampling_rate_hz: float = 3.0) -> None:
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
        x_tgt = spikes[:, 1:, :].float()
        stim_in = stim[:, :-1, :]
        mu_raw, log_sigma_raw, eta_raw, log_delta_raw = model(x_in, stim_in, positions, mask, get_logits=True)
        preds_next = sas_rate_median(mu_raw, log_sigma_raw, eta_raw, log_delta_raw).float()
        truth_next = x_tgt.float()
        create_nextstep_video(truth_next, preds_next, positions, videos_dir / f'nextstep_batch{i+1}.mp4')

        # Accumulate next-step truth/pred/positions
        ns_truth_list.append(truth_next.detach().cpu())
        ns_pred_list.append(preds_next.detach().cpu())
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
            gen_only = gen_seq[:, -n_steps:, :].float()
            truth_only = spikes[:, context_len:context_len + n_steps, :].float()
            create_autoregression_video(gen_only, positions, videos_dir / f'autoreg_batch{i+1}.mp4', truth=truth_only)
            ar_truth_list.append(truth_only.detach().cpu())
            ar_pred_list.append(gen_only.detach().cpu())
            ar_context_list.append(int(context_len))
            ar_nsteps_list.append(int(n_steps))

    # Write a single unified H5 with all collected arrays
    try:
        out_h5 = videos_dir / 'eval_data.h5'
        with h5py.File(out_h5, 'w') as f:
            if ns_truth_list:
                f.create_dataset('truth_next_step', data=torch.cat(ns_truth_list, dim=0).to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('pred_next_step', data=torch.cat(ns_pred_list, dim=0).to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                # For positions, keep only the first (assumed constant per subject); else stack
                try:
                    pos_cat = torch.cat(ns_pos_list, dim=0).to(torch.float32).numpy()
                    f.create_dataset('positions', data=pos_cat, compression='gzip', compression_opts=1)
                except Exception:
                    f.create_dataset('positions', data=ns_pos_list[0].to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                try:
                    f.create_dataset('neuron_mask', data=torch.cat(ns_mask_list, dim=0).numpy(), compression='gzip', compression_opts=1)
                except Exception:
                    pass
            if ar_truth_list:
                f.create_dataset('truth_future', data=torch.cat(ar_truth_list, dim=0).to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('pred_future', data=torch.cat(ar_pred_list, dim=0).to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('ar_context_len', data=np.array(ar_context_list, dtype=np.int32))
                f.create_dataset('ar_n_steps', data=np.array(ar_nsteps_list, dtype=np.int32))
    except Exception:
        pass


@torch.no_grad()
def evaluate_autoregression(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, threshold: float, logs_dir: Path, num_batches: Optional[int] = None, use_double_window: bool = True, sampling_rate_hz: float = 3.0, *, spikes_are_rates: bool = False, spikes_are_zcalcium: bool = False) -> Dict[str, List[float]]:
    """
    Compute per-horizon autoregression metrics to assess degradation as steps increase.
    Also compute relative performance vs a baseline that copies the last context frame.
    Uses half of the available sequence as context by default.
    Also tracks the best performing AR sample across all batches.
    """
    per_step_loss_sums: List[float] = []  # BCE in prob mode; MSE in zcalcium; N/A in rates
    # PR AUC removed
    per_step_counts: List[int] = []
    # Additional spike-rate metric (MAE only)
    per_step_mae_rate_sums: List[float] = []
    # Additional zcalcium MAE per step
    per_step_mae_z_sums: List[float] = []
    # Baseline (copy-last) spike-rate MAE sums
    per_step_mae_rate_baseline_sums: List[float] = []

    # Best AR sample tracking
    best_sample_score = float('inf')  # Lower is better (mean MAE rate)
    best_sample_data = None

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
        gen_only = gen_seq[:, -n_steps:, :]
        tgt_only = spikes[:, context_len:context_len + n_steps, :].float()

        # Ensure accumulator sizes
        if len(per_step_loss_sums) < n_steps:
            extend_by = n_steps - len(per_step_loss_sums)
            per_step_loss_sums.extend([0.0] * extend_by)
            # no PR AUC accumulator
            per_step_counts.extend([0] * extend_by)
            per_step_mae_rate_sums.extend([0.0] * extend_by)
            per_step_mae_z_sums.extend([0.0] * extend_by)
            per_step_mae_rate_baseline_sums.extend([0.0] * extend_by)

        # Per-sample tracking for best AR sample
        for b in range(B):
            # Compute overall performance score for this sample (mean MAE rate across all steps)
            sample_mae_rates = []
            for k in range(n_steps):
                pred_k = gen_only[b:b+1, k, :].detach()  # Keep batch dim for consistency
                targ_k = tgt_only[b:b+1, k, :].detach()
                
                # Convert to rates if needed
                eps = 1e-7
                if spikes_are_rates:
                    pred_rate = pred_k.float()
                    true_rate = targ_k.float()
                else:
                    pred_clamped = pred_k.clamp(0.0, 1.0 - eps)
                    targ_clamped = targ_k.clamp(0.0, 1.0 - eps)
                    pred_rate = -sampling_rate_hz * torch.log1p(-pred_clamped)
                    true_rate = -sampling_rate_hz * torch.log1p(-targ_clamped)
                
                # Compute MAE rate for this step, considering mask
                if 'neuron_mask' in batch:
                    sample_mask = mask[b:b+1, :]  # (1, N)
                    valid_mae = ((pred_rate - true_rate).abs() * sample_mask.float()).sum() / sample_mask.sum().clamp_min(1)
                else:
                    valid_mae = (pred_rate - true_rate).abs().mean()
                sample_mae_rates.append(valid_mae.item())
            
            # Overall score for this sample (mean across all steps)
            sample_score = np.mean(sample_mae_rates) if sample_mae_rates else float('inf')
            
            # Update best sample if this one is better
            if sample_score < best_sample_score:
                best_sample_score = sample_score
                best_sample_data = {
                    'initial_context': init_x[b:b+1, :, :].detach().cpu(),  # (1, context_len, N)
                    'true_future': tgt_only[b:b+1, :, :].detach().cpu(),    # (1, n_steps, N) 
                    'pred_future': gen_only[b:b+1, :, :].detach().cpu(),    # (1, n_steps, N)
                    'positions': positions[b:b+1, :, :].detach().cpu(),     # (1, N, 3)
                    'neuron_mask': mask[b:b+1, :].detach().cpu() if 'neuron_mask' in batch else torch.ones(1, init_x.shape[-1], dtype=torch.bool),  # (1, N)
                    'stimulus_context': init_stim[b:b+1, :, :].detach().cpu() if init_stim is not None else None,  # (1, context_len, K)
                    'stimulus_future': future_stim[b:b+1, :n_steps, :].detach().cpu() if future_stim is not None and future_stim.shape[1] >= n_steps else None,  # (1, n_steps, K)
                    'context_len': context_len,
                    'n_steps': n_steps,
                    'sample_score': sample_score,
                    'batch_idx': batch_idx,
                    'sample_idx': b
                }

        # Per-horizon metrics
        for k in range(n_steps):
            preds_k = gen_only[:, k, :].detach()
            targs_k = tgt_only[:, k, :].detach()
            # Track optional losses
            if spikes_are_zcalcium:
                loss_k = F.mse_loss(preds_k.float(), targs_k.float(), reduction='mean').item()
                per_step_loss_sums[k] += loss_k
                per_step_counts[k] += 1
            elif not spikes_are_rates:
                # MAE in log-rate domain as proxy loss
                abs_err_loss = (preds_k.float() - targs_k.float()).abs().mean().item()
                per_step_loss_sums[k] += abs_err_loss
                per_step_counts[k] += 1

            # Spike-rate metrics are only defined for prob/rates modes
            if spikes_are_rates:
                pred_rate = preds_k.float()
                true_rate = targs_k.float()
                abs_err = (pred_rate - true_rate).abs()
                mae_rate_k = abs_err.mean().item()
                per_step_mae_rate_sums[k] += mae_rate_k
            elif not spikes_are_zcalcium:
                eps = 1e-7
                preds_k_clamped = preds_k.clamp(0.0, 1.0 - eps)
                targs_k_clamped = targs_k.clamp(0.0, 1.0 - eps)
                pred_rate = -sampling_rate_hz * torch.log1p(-preds_k_clamped)
                true_rate = -sampling_rate_hz * torch.log1p(-targs_k_clamped)
                abs_err = (pred_rate - true_rate).abs()
                mae_rate_k = abs_err.mean().item()
                per_step_mae_rate_sums[k] += mae_rate_k
            else:
                # zcalcium MAE per step (identity domain)
                abs_err = (preds_k.float() - targs_k.float()).abs()
                per_step_mae_z_sums[k] += abs_err.mean().item()
            # Baseline: copy last context frame forward
            if spikes_are_rates or (not spikes_are_zcalcium):
                last_ctx = init_x[:, -1, :]  # (B, N)
                if spikes_are_rates:
                    last_ctx_rate = last_ctx.float()
                    true_rate_for_baseline = true_rate
                else:
                    eps = 1e-7
                    last_ctx_clamped = last_ctx.clamp(0.0, 1.0 - eps)
                    last_ctx_rate = -sampling_rate_hz * torch.log1p(-last_ctx_clamped)
                    true_rate_for_baseline = true_rate
                abs_err_baseline = (last_ctx_rate - true_rate_for_baseline).abs()
                mae_rate_baseline_k = abs_err_baseline.mean().item()
                per_step_mae_rate_baseline_sums[k] += mae_rate_baseline_k
            

    # Averages
    per_step_loss = []
    for k in range(len(per_step_counts)):
        if spikes_are_zcalcium:
            per_step_loss.append(per_step_loss_sums[k] / max(1, per_step_counts[k]))
        elif spikes_are_rates:
            per_step_loss.append(float('nan'))
        else:
            per_step_loss.append(per_step_loss_sums[k] / max(1, per_step_counts[k]))
    per_step_mae_rate = [per_step_mae_rate_sums[k] / max(1, per_step_counts[k]) for k in range(len(per_step_counts))]
    per_step_mae_z = [per_step_mae_z_sums[k] / max(1, per_step_counts[k]) for k in range(len(per_step_counts))]
    per_step_mae_rate_baseline = [per_step_mae_rate_baseline_sums[k] / max(1, per_step_counts[k]) for k in range(len(per_step_counts))]
    # Relative performance: 1 - (model MAE / baseline MAE)
    per_step_relative = []
    for m, b in zip(per_step_mae_rate, per_step_mae_rate_baseline):
        if b is not None and float(b) > 1e-12 and np.isfinite(b) and np.isfinite(m):
            per_step_relative.append(1.0 - (float(m) / float(b)))
        else:
            per_step_relative.append(float('nan'))

    # Write CSV
    out_csv = logs_dir / 'autoregression_per_step.csv'
    with open(out_csv, 'w') as f:
        if spikes_are_zcalcium:
            f.write('step,mse,mae,count\n')
            for k in range(len(per_step_counts)):
                f.write(f"{k+1},{per_step_loss[k]:.6f},{per_step_mae_z[k]:.6f},{per_step_counts[k]}\n")
        else:
            f.write('step,loss,mae_rate,rel_perf,count\n')
            for k in range(len(per_step_counts)):
                rp = per_step_relative[k]
                rp_str = f"{rp:.6f}" if np.isfinite(rp) else 'nan'
                f.write(f"{k+1},{per_step_loss[k]:.6f},{per_step_mae_rate[k]:.6f},{rp_str},{per_step_counts[k]}\n")

    # Optional quick plot
    try:
        import matplotlib.pyplot as plt
        steps = np.arange(1, len(per_step_counts) + 1)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = 'tab:red'
        ax1.set_xlabel('Horizon (steps)')
        if spikes_are_zcalcium:
            ax1.set_ylabel('MSE', color=color)
            ax1.plot(steps, per_step_loss, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            color = 'tab:purple'
            ax2.set_ylabel('MAE (zcalcium)', color=color)
            ax2.plot(steps, per_step_mae_z, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        elif not spikes_are_rates:
            ax1.set_ylabel('Avg loss', color=color)
            ax1.plot(steps, per_step_loss, color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('MAE rate', color=color)
            ax2.plot(steps, per_step_mae_rate, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            # Only MAE in rates mode
            ax1.set_ylabel('MAE rate', color='tab:green')
            ax1.plot(steps, per_step_mae_rate, color='tab:green')
            ax1.tick_params(axis='y', labelcolor='tab:green')
        fig.tight_layout()
        fig.savefig(logs_dir / 'autoregression_per_step.png', dpi=120, bbox_inches='tight')
        plt.close(fig)

        # Relative performance plot
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Relative performance (1 - model/baseline)')
        ax.plot(steps, per_step_relative, color='tab:blue')
        ax.axhline(0.0, color='k', linestyle='--', linewidth=1)
        fig2.tight_layout()
        fig2.savefig(logs_dir / 'autoregression_relative_per_step.png', dpi=120, bbox_inches='tight')
        plt.close(fig2)
    except Exception:
        pass

    return {
        'per_step_loss': per_step_loss,
        'per_step_mae_rate': per_step_mae_rate,
        'per_step_rel_perf': per_step_relative,
        'best_sample_data': best_sample_data,
        'best_sample_score': best_sample_score,
    }


@torch.no_grad()
def create_best_ar_sample_video(best_sample_data: Dict[str, Any], videos_dir: Path) -> bool:
    """
    Create a video specifically for the best performing AR sample.
    Shows the full sequence: initial context + true future + predicted future.
    """
    if best_sample_data is None:
        return False
    
    try:
        # Extract data
        init_context = best_sample_data['initial_context']  # (1, context_len, N)
        true_future = best_sample_data['true_future']       # (1, n_steps, N)
        pred_future = best_sample_data['pred_future']       # (1, n_steps, N)
        positions = best_sample_data['positions']           # (1, N, 3)
        neuron_mask = best_sample_data['neuron_mask']       # (1, N)
        context_len = best_sample_data['context_len']
        n_steps = best_sample_data['n_steps']
        sample_score = best_sample_data['sample_score']
        batch_idx = best_sample_data['batch_idx']
        sample_idx = best_sample_data['sample_idx']
        
        # Import the video creation function
        from GenerativeBrainModel.visualizations import create_autoregression_video
        
        # Create video filename with performance info
        video_name = f'best_ar_sample_batch{batch_idx}_idx{sample_idx}_score{sample_score:.4f}.mp4'
        video_path = videos_dir / video_name
        
        # Ensure batched shapes expected by create_autoregression_video: (B, T, N) and (B, N, 3)
        if pred_future.ndim == 2:
            pred_future = pred_future.unsqueeze(0)
        if true_future is not None and true_future.ndim == 2:
            true_future = true_future.unsqueeze(0)
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)

        # Create the video - show predicted future against ground truth
        create_autoregression_video(
            pred_future,           # (B, n_steps, N)
            positions,             # (B, N, 3)
            video_path,
            truth=true_future      # (B, n_steps, N)
        )
        
        # Also save the data as an H5 file for further analysis
        try:
            import h5py
            h5_path = videos_dir / f'best_ar_sample_batch{batch_idx}_idx{sample_idx}_data.h5'
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('initial_context', data=init_context.to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('true_future', data=true_future.to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('pred_future', data=pred_future.to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('positions', data=positions.to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                f.create_dataset('neuron_mask', data=neuron_mask.numpy(), compression='gzip', compression_opts=1)
                
                # Save stimulus data if available
                if best_sample_data.get('stimulus_context') is not None:
                    f.create_dataset('stimulus_context', data=best_sample_data['stimulus_context'].to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                if best_sample_data.get('stimulus_future') is not None:
                    f.create_dataset('stimulus_future', data=best_sample_data['stimulus_future'].to(torch.float32).numpy(), compression='gzip', compression_opts=1)
                
                # Save metadata
                f.attrs['context_len'] = context_len
                f.attrs['n_steps'] = n_steps
                f.attrs['sample_score'] = sample_score
                f.attrs['batch_idx'] = batch_idx
                f.attrs['sample_idx'] = sample_idx
                f.attrs['description'] = f'Best performing AR sample from batch {batch_idx}, sample {sample_idx} with MAE rate score {sample_score:.6f}'
                
        except Exception as e:
            print(f"Warning: Could not save best sample H5 data: {e}")
            
    except Exception as e:
        print(f"Warning: Could not create best AR sample video: {e}")
        return False

    return True


@torch.no_grad()
def save_spike_trace_plots(
    model: GBM,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: Path,
    num_images: int = 10,
    neurons_per_image: int = 4,
) -> None:
    """Save line plots of spike probabilities for individual neurons.
    Each image contains 4 subplots. For each selected neuron, plot:
      - Truth over the full 2L window
      - AR prediction over the horizon (second L), aligned to the same x-axis
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    images_saved = 0

    for batch in tqdm(loader, desc='Spike trace plots'):
        if images_saved >= num_images:
            break
        spikes = batch['spikes'].to(device).to(torch.bfloat16)   # (B, 2L, N)
        positions = batch['positions'].to(device)
        mask = batch['neuron_mask'].to(device).bool()
        stim = batch['stimulus'].to(device).to(torch.bfloat16)

        B, Ltot, N = spikes.shape
        context_len = int(Ltot // 2)
        n_steps = int(context_len)
        if n_steps <= 0:
            continue

        init_x = spikes[:, :context_len, :]
        init_stim = stim[:, :context_len, :]
        future_stim = stim[:, context_len:, :]
        gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
        gen_only = gen_seq[:, -n_steps:, :]  # (B, L, N) preds in [0,1]

        for b in range(B):
            if images_saved >= num_images:
                break
            valid_idx = torch.where(mask[b])[0].detach().cpu().numpy()
            if valid_idx.size == 0:
                continue
            k = int(min(neurons_per_image, valid_idx.size))
            chosen = np.random.choice(valid_idx, size=k, replace=False)

            fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
            axes = axes.reshape(-1)
            t_full = np.arange(Ltot)
            t_pred = np.arange(context_len, context_len + n_steps)
            for ax_idx, neuron_id in enumerate(chosen):
                ax = axes[ax_idx]
                truth_np = spikes[b, :, neuron_id].detach().to(torch.float32).cpu().numpy()
                pred_np = gen_only[b, :, neuron_id].detach().to(torch.float32).cpu().numpy()
                ax.plot(t_full, truth_np, label='Truth', color='tab:green', linewidth=1.5)
                ax.plot(t_pred, pred_np, label='AR Pred', color='tab:orange', linewidth=1.5)
                ax.axvline(context_len - 0.5, color='k', linestyle='--', linewidth=1)
                ax.set_ylabel('Spike prob')
                ax.set_title(f'Neuron {int(neuron_id)}')
                ymin = float(min(np.min(truth_np), np.min(pred_np)))
                ymax = float(max(np.max(truth_np), np.max(pred_np)))
                if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
                    ymin, ymax = 0.0, 1.0
                # Add small padding for visibility
                pad = 0.02 * max(1e-6, (ymax - ymin))
                ax.set_ylim(ymin - pad, ymax + pad)
            # Hide unused axes if fewer than 4
            for j in range(k, 4):
                fig.delaxes(axes[j])
            axes_to_label = axes[:k]
            for ax in axes_to_label:
                ax.legend(loc='upper right', fontsize=8)
            axes_to_label[-1].set_xlabel('Time (steps)')
            fig.suptitle('Spike probability: Truth vs AR Prediction')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = out_dir / f'spike_traces_{images_saved + 1:02d}.png'
            fig.savefig(out_path, dpi=140, bbox_inches='tight')
            plt.close(fig)
            images_saved += 1

        # Free memory sooner
        del spikes, positions, mask, stim, gen_seq, gen_only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        # Set mode flags from eval config (mirrors training)
        try:
            model.spikes_are_rates = bool(cfg['data'].get('spikes_are_rates', False))
        except Exception:
            model.spikes_are_rates = False
        try:
            model.spikes_are_zcalcium = bool(cfg['data'].get('spikes_are_zcalcium', False))
        except Exception:
            model.spikes_are_zcalcium = False
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
    cfg['training']['only_test'] = False
    max_tp = cfg['eval'].get('max_timepoints_per_subject')
    if max_tp is not None:
        cfg['training']['max_timepoints_per_subject'] = max_tp
    cfg['training']['num_workers'] = cfg['eval']['num_workers']
    # Enforce explicit test subjects if provided; do NOT fall back to random split
    data_dir = Path(cfg['data']['data_dir'])
    test_subjects = cfg['data'].get('test_subjects', [])
    if isinstance(test_subjects, list) and len(test_subjects) > 0:
        missing = [s for s in test_subjects if not (data_dir / f"{s}.h5").exists()]
        if missing:
            raise ValueError(f"Requested test subjects not found in {data_dir}: {missing}")
        cfg['training']['only_test'] = True
    _, ns_loader, *_ = create_dataloaders(cfg)

    if len(ns_loader) == 0:
        raise RuntimeError(
            "Next-step evaluation loader returned zero batches. "
            "Verify data_dir, sequence_length, stride, and test_subjects cover at least one window."
        )

    # Build AR dataloader with window size 2L (first L context, second L horizon)
    import copy as _copy
    cfg_ar = _copy.deepcopy(cfg)
    L_ctx = int(cfg['eval']['sequence_length'])
    cfg_ar['training']['sequence_length'] = 2 * L_ctx
    if max_tp is not None:
        cfg_ar['training']['max_timepoints_per_subject'] = max_tp
    _, ar_loader, *_ = create_dataloaders(cfg_ar)

    if len(ar_loader) == 0:
        raise RuntimeError(
            "Autoregression evaluation loader returned zero batches with context length 2L. "
            "Consider reducing sequence_length or stride."
        )

    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.0, val_threshold=cfg['eval']['threshold'], enable_plots=True)
    if not offline_mode:
        metrics = evaluate(
            model,
            ns_loader,
            device,
            tracker,
            threshold=cfg['eval']['threshold'],
            num_batches=cfg['eval']['num_batches_nextstep'],
            spikes_are_rates=bool(cfg['data'].get('spikes_are_rates', False)),
            spikes_are_zcalcium=bool(cfg['data'].get('spikes_are_zcalcium', False)),
            sampling_rate_hz=float(cfg['eval'].get('sampling_rate_hz', 3.0)),
        )
        logger.info(f"Evaluation metrics: {metrics}")
        tracker.plot_training()

    # Autoregression per-step metrics
    if not offline_mode:
        try:
            ar_metrics = evaluate_autoregression(
                model,
                ar_loader,
                device,
                threshold=cfg['eval']['threshold'],
                logs_dir=dirs['logs'],
                num_batches=cfg['eval']['num_batches_ar'],
                use_double_window=True,
                sampling_rate_hz=float(cfg['eval'].get('sampling_rate_hz', 3.0)),
                spikes_are_rates=bool(cfg['data'].get('spikes_are_rates', False)),
                spikes_are_zcalcium=bool(cfg['data'].get('spikes_are_zcalcium', False)),
            )
            logger.info(f"Autoregression per-step metrics saved. First steps MAE_rate={ar_metrics['per_step_mae_rate'][:3]}")
            
            # Create video for best performing AR sample
            if ar_metrics.get('best_sample_data') is not None:
                best_score = ar_metrics.get('best_sample_score', float('inf'))
                best_data = ar_metrics['best_sample_data']
                logger.info(f"Best AR sample found: batch {best_data['batch_idx']}, sample {best_data['sample_idx']} with MAE rate score {best_score:.6f}")
                
                # Create video for the best AR sample
                try:
                    created = create_best_ar_sample_video(best_data, dirs['videos'])
                    video_filename = f"best_ar_sample_batch{best_data['batch_idx']}_idx{best_data['sample_idx']}_score{best_score:.4f}.mp4"
                    if created:
                        logger.info(f"Created video for best AR sample: {dirs['videos'] / video_filename}")
                    else:
                        logger.warning("Best AR sample video creation reported failure.")
                except Exception as e:
                    logger.warning(f"Failed to create best AR sample video: {e}")
            else:
                logger.warning("No best AR sample data found")
                
        except Exception as e:
            logger.warning(f"Autoregression metrics failed: {e}")

    if cfg['eval']['make_videos'] and not offline_mode:
        try:
            # Use AR loader so the unified H5 contains both next-step and AR using 2L windows
            make_videos(
                model,
                ar_loader,
                device,
                dirs['videos'],
                sequence_length=cfg['eval']['sequence_length'],
                num_batches=1,
                use_double_window_for_ar=True,
                spikes_are_rates=bool(cfg['data'].get('spikes_are_rates', False)),
                spikes_are_zcalcium=bool(cfg['data'].get('spikes_are_zcalcium', False)),
                sampling_rate_hz=float(cfg['eval'].get('sampling_rate_hz', 3.0)),
            )
        except Exception as e:
            logger.warning(f"Video generation failed: {e}")

    # Spike trace plots: 10 images, 4 neurons each (truth full 2L, AR preds over horizon)
    if not offline_mode:
        try:
            plots_dir = dirs['logs'] / 'plots'
            save_spike_trace_plots(
                model,
                ar_loader,
                device,
                plots_dir,
                num_images=int(cfg['eval'].get('num_trace_images', 10)),
                neurons_per_image=int(cfg['eval'].get('neurons_per_trace_image', 4)),
            )
        except Exception as e:
            logger.warning(f"Spike trace plotting failed: {e}")

    # Next-step: loss vs context size (per output position)
    try:
        import matplotlib.pyplot as plt
        # Determine L-1 positions
        L_ctx = int(cfg['eval']['sequence_length'])
        Tpos = max(1, L_ctx - 1)
        step_loss_sums = torch.zeros(Tpos, dtype=torch.float64)
        step_loss_counts = torch.zeros(Tpos, dtype=torch.float64)
        # Spike-rate metric accumulator (MAE only)
        step_mae_rate_sums = torch.zeros(Tpos, dtype=torch.float64)
        # Baseline (copy-last) spike-rate MAE accumulator
        step_mae_rate_baseline_sums = torch.zeros(Tpos, dtype=torch.float64)
        sampling_rate_hz = float(cfg['eval'].get('sampling_rate_hz', 3.0))

        if offline_mode and load_from and Path(load_from).exists():
            with h5py.File(load_from, 'r') as f:
                t = torch.from_numpy(f['truth_next_step'][()]).float()   # (Bsum, Tpos, N)
                p = torch.from_numpy(f['pred_next_step'][()]).float()    # (Bsum, Tpos, N)
                mask = torch.from_numpy(f['neuron_mask'][()]).bool() if 'neuron_mask' in f else None  # (Bsum, N)
                if mask is not None:
                    mask_rep = mask[:, None, :].expand(t.shape[0], t.shape[1], t.shape[2])
                else:
                    mask_rep = torch.ones_like(t, dtype=torch.bool)
                # BCE only when not in rates mode
                bce = F.binary_cross_entropy(p, t, reduction='none') if not bool(cfg['data'].get('spikes_are_rates', False)) else torch.zeros_like(t)
                # Spike-rate errors
                eps = 1e-7
                if bool(cfg['data'].get('spikes_are_rates', False)):
                    p_rate = p
                    t_rate = t
                else:
                    p_clamped = p.clamp(0.0, 1.0 - eps)
                    t_clamped = t.clamp(0.0, 1.0 - eps)
                    p_rate = -sampling_rate_hz * torch.log1p(-p_clamped)
                    t_rate = -sampling_rate_hz * torch.log1p(-t_clamped)
                abs_err = (p_rate - t_rate).abs()
                # Baseline: copy previous timestep truth rate to predict next (defined for j>=1)
                baseline_rate = t_rate[:, :-1, :]  # (Bsum, Tpos-1, N)
                for j in range(min(Tpos, bce.shape[1])):
                    sel = mask_rep[:, j, :]
                    if not bool(cfg['data'].get('spikes_are_rates', False)):
                        step_loss_sums[j] += bce[:, j, :][sel].sum().double().cpu()
                    step_loss_counts[j] += sel.sum().double().cpu()
                    step_mae_rate_sums[j] += abs_err[:, j, :][sel].sum().double().cpu()
                    if j - 1 >= 0 and j - 1 < baseline_rate.shape[1]:
                        abs_err_bl = (baseline_rate[:, j - 1, :] - t_rate[:, j, :]).abs()
                        step_mae_rate_baseline_sums[j] += abs_err_bl[sel].sum().double().cpu()
        else:
            with torch.no_grad():
                used = 0
                limit = cfg['eval'].get('num_batches_nextstep')
                for batch in tqdm(ns_loader, desc='Next-step loss vs context'):
                    spikes = batch['spikes'].to(device).to(torch.bfloat16)
                    mask = batch['neuron_mask'].to(device).bool()
                    stim = batch['stimulus'].to(device).to(torch.bfloat16)
                    x_in = spikes[:, :-1, :]
                    x_tgt = spikes[:, 1:, :].float()
                    mu_val, log_sigma_val, eta_val, log_delta_val = model(x_in, stim[:, :-1, :], batch['positions'].to(device), mask, get_logits=True)
                    preds = sas_rate_median(mu_val, log_sigma_val, eta_val, log_delta_val)
                    mask_rep = mask[:, None, :].expand(preds.shape[0], preds.shape[1], preds.shape[2])
                    # Loss surrogate (MAE in rates)
                    bce = (preds.float() - x_tgt.float()).abs()
                    eps = 1e-7
                    if bool(cfg['data'].get('spikes_are_rates', False)):
                        p_rate = preds.float()
                        t_rate = x_tgt.float()
                    else:
                        p_clamped = preds.clamp(0.0, 1.0 - eps)
                        t_clamped = x_tgt.clamp(0.0, 1.0 - eps)
                        p_rate = -sampling_rate_hz * torch.log1p(-p_clamped)
                        t_rate = -sampling_rate_hz * torch.log1p(-t_clamped)
                    abs_err = (p_rate - t_rate).abs()
                    # Baseline: copy previous timestep truth rate to predict next
                    baseline_rate = t_rate[:, :-1, :]  # aligns with next-step positions j>=1
                    for j in range(min(Tpos, bce.shape[1])):
                        sel = mask_rep[:, j, :]
                        if not bool(cfg['data'].get('spikes_are_rates', False)):
                            step_loss_sums[j] += bce[:, j, :][sel].sum().double().cpu()
                        step_loss_counts[j] += sel.sum().double().cpu()
                        step_mae_rate_sums[j] += abs_err[:, j, :][sel].sum().double().cpu()
                        if j - 1 >= 0 and j - 1 < baseline_rate.shape[1]:
                            abs_err_bl = (baseline_rate[:, j - 1, :] - t_rate[:, j, :]).abs()
                            step_mae_rate_baseline_sums[j] += abs_err_bl[sel].sum().double().cpu()
                    used += 1
                    if limit and used >= int(limit):
                        break

        ctx_sizes = np.arange(1, Tpos + 1)
        step_loss = (step_loss_sums / step_loss_counts.clamp_min(1)).cpu().numpy()
        step_mae_rate = (step_mae_rate_sums / step_loss_counts.clamp_min(1)).cpu().numpy()
        step_mae_rate_baseline = (step_mae_rate_baseline_sums / step_loss_counts.clamp_min(1)).cpu().numpy()
        # Relative performance per position
        rel_perf = np.full_like(step_mae_rate, np.nan, dtype=np.float64)
        valid = (step_mae_rate_baseline > 1e-12) & np.isfinite(step_mae_rate_baseline) & np.isfinite(step_mae_rate)
        rel_perf[valid] = 1.0 - (step_mae_rate[valid] / step_mae_rate_baseline[valid])
        # Save CSV
        with open(dirs['logs'] / 'nextstep_loss_vs_context.csv', 'w') as fcsv:
            fcsv.write('context,loss,mae_rate,rel_perf,count\n')
            for c, loss, mae_r, rp, cnt in zip(ctx_sizes, step_loss, step_mae_rate, rel_perf, step_loss_counts.cpu().numpy()):
                rp_str = f"{float(rp):.6f}" if np.isfinite(rp) else 'nan'
                fcsv.write(f"{int(c)},{float(loss):.6f},{float(mae_r):.6f},{rp_str},{int(cnt)}\n")
        # Plot scatter
        plt.figure(figsize=(6,4))
        plt.scatter(ctx_sizes, step_loss, s=16, alpha=0.8)
        plt.xlabel('Context length (tokens)')
        plt.ylabel('Next-step BCE loss')
        plt.title('Next-step loss vs context size')
        plt.tight_layout()
        plt.savefig(dirs['logs'] / 'nextstep_loss_vs_context.png', dpi=120, bbox_inches='tight')
        plt.close()
        # Plot spike-rate MAE
        try:
            plt.figure(figsize=(6,4))
            plt.scatter(ctx_sizes, step_mae_rate, s=16, alpha=0.8)
            plt.xlabel('Context length (tokens)')
            plt.ylabel('Mean absolute spike rate error (Hz)')
            plt.title('Next-step spike-rate MAE vs context size')
            plt.tight_layout()
            plt.savefig(dirs['logs'] / 'nextstep_mae_rate_vs_context.png', dpi=120, bbox_inches='tight')
            plt.close()
        except Exception:
            pass
        # Plot relative performance vs context
        try:
            plt.figure(figsize=(6,4))
            plt.scatter(ctx_sizes, rel_perf, s=16, alpha=0.8)
            plt.axhline(0.0, color='k', linestyle='--', linewidth=1)
            plt.xlabel('Context length (tokens)')
            plt.ylabel('Relative performance (1 - model/baseline)')
            plt.title('Next-step relative performance vs context size')
            plt.tight_layout()
            plt.savefig(dirs['logs'] / 'nextstep_relative_vs_context.png', dpi=120, bbox_inches='tight')
            plt.close()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Next-step loss vs context plotting failed: {e}")

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
                        truth_means_list.append(((t * mask_rep).sum(dim=-1) / denom).detach().cpu())
                        pred_means_list.append(((p * mask_rep).sum(dim=-1) / denom).detach().cpu())
                    else:
                        truth_means_list.append(t.mean(dim=-1).detach().cpu())
                        pred_means_list.append(p.mean(dim=-1).detach().cpu())
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
                    mu_val, log_sigma_val, eta_val, log_delta_val = model(x_in, stim_in, positions, mask, get_logits=True)
                    pred_mean = sas_rate_median(mu_val, log_sigma_val, eta_val, log_delta_val).detach()
                    all_truth.append(x_tgt.detach().float().cpu())
                    all_pred.append(pred_mean.float().cpu())
                    mask_exp = mask[:, None, :].to(x_tgt.dtype)
                    mask_rep = mask_exp.expand(-1, x_tgt.shape[1], -1)
                    denom = mask_rep.sum(dim=-1).clamp_min(1)
                    truth_means_list.append(((x_tgt.float() * mask_rep).sum(dim=-1) / denom).detach().cpu())
                    pred_means_list.append(((pred_mean.float() * mask_rep).sum(dim=-1) / denom).detach().cpu())
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

            # Calibration plot: true mean rate (sorted) vs predicted mean rate
            try:
                order = np.argsort(truth_mean)
                true_rates_sorted = truth_mean[order]
                pred_rates_sorted = pred_mean[order]
                if true_rates_sorted.size > 1:
                    r_val = float(np.corrcoef(true_rates_sorted, pred_rates_sorted)[0, 1])
                else:
                    r_val = float('nan')
                plt.figure(figsize=(6, 4))
                plt.plot(true_rates_sorted, pred_rates_sorted, color='tab:blue', linewidth=1.5)
                plt.xlabel('True mean rate (sorted, Hz)')
                plt.ylabel('Predicted mean rate (Hz)')
                plt.title('Calibration: True vs Predicted Mean Rates')
                plt.annotate(
                    f"r = {r_val:.4f}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8),
                )
                plt.tight_layout()
                plt.savefig(dirs['logs'] / 'calibration_true_vs_predicted_rates.png', dpi=120, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Calibration plot failed: {e}")

    except Exception as e:
        logger.warning(f"Embedding (PCA/UMAP) failed: {e}")


if __name__ == '__main__':
    main()


