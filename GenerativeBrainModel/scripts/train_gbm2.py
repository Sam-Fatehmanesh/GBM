#!/usr/bin/env python3
"""
Clean training script for GBM:
 - Dataloader provides spikes (rates), stimulus, positions, neuron_mask, neuron_ids, and optional per-neuron log stats.
 - Inputs to model are log(rates) z-normalized per neuron using provided log_activity_mean/std when available.
 - Model predicts LogNormal parameters (mu, raw_log_sigma) and is trained with LogNormal NLL.
 - Uses Muon optimizer for attention body and AdamW for embeddings/head.
 - Validates periodically and writes scalar plots and a 16-neuron autoregressive visualization comparing prediction vs truth.
 - Train/test split is last X% of time per subject, already enforced by the dataset construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.utils.lognormal import lognormal_nll, lognormal_rate_median
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {'name': 'gbm2_neural_training'},
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'use_cache': True,
        },
        'model': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_stimuli': None,  # infer from data
            'num_neurons_total': 4_000_000,  # capacity of neuron embedding table (>= max distinct IDs)
        },
        'training': {
            'batch_size': 2,
            'num_epochs': 20,
            'learning_rate': 5e-4,          # AdamW for non-body
            'muon_lr': 2e-2,                # Muon for attention body
            'weight_decay': 1e-4,
            'sequence_length': 12,
            'stride': 3,
            'num_workers': 0,
            'pin_memory': False,
            'test_split_fraction': 0.1,
            'use_gpu': True,
            'compile_model': False,
            'validation_frequency': 8,
            'val_sample_batches': 32,
            'gradient_clip_norm': 1.0,
            'seed': 42,
            'plots_dir': 'experiments/gbm2/plots',
            'sampling_rate_hz': 3.0,
        }
    }


def deep_update(base: Dict, updates: Dict) -> Dict:
    result = base.copy()
    for k, v in updates.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: GBM, cfg: Dict[str, Any]) -> optim.Optimizer:
    try:
        from muon import MuonWithAuxAdam
    except ImportError as e:
        raise ImportError("Muon optimizer not found. Install: pip install git+https://github.com/KellerJordan/Muon") from e

    # Body vs non-body
    hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2 and p.requires_grad]
    nonhidden_params = []
    for m in model.embed.values():
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]
    for m in model.head.values():
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]

    muon_lr = float(cfg.get('muon_lr', 2e-2))
    muon_wd = float(cfg.get('weight_decay', 1e-4))
    adamw_lr = float(cfg.get('learning_rate', 5e-4))
    adamw_wd = float(cfg.get('weight_decay', 1e-4))
    adamw_betas = (0.9, 0.95)

    param_groups = []
    if hidden_weights:
        param_groups.append(dict(params=hidden_weights, use_muon=True, lr=muon_lr, weight_decay=muon_wd))
    if hidden_gains_biases or nonhidden_params:
        param_groups.append(dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                                 lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_wd))
    return MuonWithAuxAdam(param_groups)


@torch.no_grad()
def make_validation_plots(step_dir: Path,
                          context_truth: torch.Tensor,
                          future_truth: torch.Tensor,
                          future_pred: torch.Tensor,
                          title: str,
                          max_neurons: int = 16) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    K = min(max_neurons, N)
    idx = torch.linspace(0, N - 1, K).round().long().tolist()
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K), sharex=True)
    axes = np.atleast_1d(axes)
    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)
    for r, ax in enumerate(axes):
        j = idx[r]
        ax.plot(x_context, context_truth[:, j].cpu().numpy(), color='gray', lw=1.0, label='context truth')
        ax.plot(x_future, future_truth[:, j].cpu().numpy(), color='tab:blue', lw=1.0, label='future truth')
        ax.plot(x_future, future_pred[:, j].cpu().numpy(), color='tab:orange', lw=1.0, label='future pred')
        ax.set_ylabel(f"n{j}")
        if r == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('time')
    fig.suptitle(title)
    fig.tight_layout()
    out = step_dir / 'val_autoreg_16neurons.png'
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def autoregressive_rollout(model: GBM,
                           init_context: torch.Tensor,   # (1, Tc, N) rates
                           stim_full: torch.Tensor,      # (1, Tc+Tf, K)
                           positions: torch.Tensor,      # (1, N, 3)
                           neuron_mask: torch.Tensor,    # (1, N)
                           neuron_ids: torch.Tensor,     # (1, N)
                           lam: torch.Tensor | None,     # (1, N)
                           las: torch.Tensor | None,     # (1, N)
                           device: torch.device,
                           Tf: int,
                           sampling_rate_hz: float = 3.0) -> torch.Tensor:      # returns (Tf, N) rates
    model.eval()
    context = init_context.clone()  # (1, Tc, N)
    preds = []
    eps = 1e-7
    for t in range(Tf):
        # Use last frame as input (next-step model); or use last Lc window if desired
        x_in = context[:, -1:, :]  # (1,1,N)
        x_log = torch.log(x_in.clamp_min(eps))
        if (lam is not None) and (lam.numel() > 0) and (las is not None) and (las.numel() > 0):
            lam_e = lam[:, None, :].to(dtype=x_log.dtype)
            las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
            x_in_z = (x_log - lam_e) / las_e
        else:
            x_in_z = x_log
        stim_step = stim_full[:, (context.shape[1]-1):(context.shape[1]), :]  # align single step
        if device.type == 'cuda':
            x_in_z = x_in_z.to(torch.bfloat16)
            stim_step = stim_step.to(torch.bfloat16)
        spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / float(sampling_rate_hz))
        mu, raw_log_sigma, _, _ = model(x_in_z, stim_step, positions, neuron_mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True)
        med = lognormal_rate_median(mu, raw_log_sigma)  # (1,1,N)
        preds.append(med[:, 0, :].to(torch.float32))
        # Append prediction to context
        context = torch.cat([context, med.to(context.dtype)], dim=1)
    return torch.cat(preds, dim=0)  # (Tf,N)


def main():
    ap = argparse.ArgumentParser(description='Train GBM (clean) with LogNormal NLL and Muon optimizer')
    ap.add_argument('--config', type=str, default=None, help='YAML config for overrides')
    args = ap.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    set_seeds(int(cfg['training']['seed']))

    # Allow Muon optimizer to run single-process by stubbing minimal torch.distributed APIs
    try:
        import torch.distributed as dist
        if dist.is_available() and not dist.is_initialized():
            dist.get_world_size = lambda group=None: 1
            dist.get_rank = lambda group=None: 0
            def _fake_all_gather(tensor_list, tensor, group=None):
                if tensor_list is None or len(tensor_list) == 0:
                    return
                # Ensure tensor_list[0] receives a copy of tensor
                if len(tensor_list) == 1:
                    if tensor_list[0].shape == tensor.shape:
                        tensor_list[0].copy_(tensor)
                    else:
                        tensor_list[0].resize_(tensor.shape).copy_(tensor)
                else:
                    for i in range(len(tensor_list)):
                        tensor_list[i].resize_(tensor.shape).copy_(tensor)
            dist.all_gather = _fake_all_gather
    except Exception:
        pass

    # Data
    train_loader, val_loader, _, _ = create_dataloaders(cfg)

    # Model
    try:
        sample = next(iter(train_loader))
        d_stimuli = int(sample['stimulus'].shape[-1])
        max_n = int(sample['positions'].shape[1])
    except Exception:
        d_stimuli = cfg['model'].get('d_stimuli') or 1
        max_n = 100_000

    model = GBM(d_model=cfg['model']['d_model'], d_stimuli=d_stimuli,
                n_heads=cfg['model']['n_heads'], n_layers=cfg['model']['n_layers'],
                num_neurons_total=int(cfg['model']['num_neurons_total'])).to(device)

    # Run bf16 on CUDA
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        model = model.to(dtype=torch.bfloat16)

    # Enable autograd anomaly and grad NaN/Inf checks in debug mode
    if debug_enabled():
        try:
            torch.autograd.set_detect_anomaly(True)
        except Exception:
            pass
        def _check_grad(pname):
            def hook(grad):
                assert_no_nan(grad, f'grad.{pname}')
                return grad
            return hook
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(_check_grad(n))

    if bool(cfg['training'].get('compile_model', False)):
        try:
            model = torch.compile(model, dynamic=True)
        except Exception:
            pass

    optimizer = build_optimizer(model, cfg['training'])

    # Per-run directory under experiments/gbm2/<YYYYmmdd_HHMMSS>
    base_dir = Path('experiments/gbm2')
    run_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = run_dir / 'plots'
    ckpt_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'
    for p in (plots_dir, ckpt_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    # Save resolved config for reproducibility
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)
    # Write a simple architecture summary
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(run_dir / 'architecture.txt', 'w') as f:
            f.write(f"Total params: {total_params:,}\nTrainable: {trainable_params:,}\n\n")
            f.write(str(model))
            f.write("\n")
    except Exception:
        pass

    # ---- Loss tracking state (for plotting) ----
    train_batch_losses: list[float] = []
    train_batch_ema: list[float] = []
    val_points: list[tuple[int, float]] = []  # (global_step, val_loss)
    ema_beta: float = 0.98
    ema_value: float | None = None
    global_step: int = 0

    def _update_loss_plot():
        try:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
            ax0, ax1 = axes
            if train_batch_losses:
                xs = list(range(1, len(train_batch_losses) + 1))
                ax0.plot(xs, train_batch_losses, color='tab:blue', alpha=0.35, lw=0.8, label='batch loss')
            if train_batch_ema:
                xs = list(range(1, len(train_batch_ema) + 1))
                ax0.plot(xs, train_batch_ema, color='tab:orange', lw=1.5, label='EMA')
            ax0.set_title('Training loss (batch + EMA)')
            ax0.set_ylabel('loss')
            ax0.legend(loc='upper right', fontsize=8)
            # Val subplot
            if val_points:
                vx, vy = zip(*val_points)
                ax1.plot(vx, vy, marker='o', linestyle='-', color='tab:green', lw=1.0, ms=3)
            ax1.set_title('Validation loss (per validation event)')
            ax1.set_xlabel('global step')
            ax1.set_ylabel('val loss')
            fig.tight_layout()
            fig.savefig(plots_dir / 'loss_curves.png')
            plt.close(fig)
        except Exception:
            pass

    def train_or_val_loop(loader, epoch: int, train: bool) -> float:
        nonlocal global_step, ema_value
        total_loss = 0.0
        count = 0
        mdl = model.train() if train else model.eval()
        # Intra-epoch validation triggers (evenly spaced + step 4)
        val_freq_local = int(cfg['training'].get('validation_frequency') or 0)
        val_sample_batches_local = int(cfg['training'].get('val_sample_batches') or 1)
        triggers: set[int] = set()
        total_steps = len(loader)
        if train and val_freq_local > 0 and total_steps > 0:
            for j in range(1, val_freq_local + 1):
                step = max(1, min(total_steps, round(j * total_steps / (val_freq_local + 1))))
                triggers.add(int(step))
            if total_steps >= 4:
                triggers.add(4)
        pbar = tqdm(loader, desc=('Train' if train else 'Val'))
        for batch in pbar:
            spikes = batch['spikes'].to(device)            # (B, L, N) rates
            positions = batch['positions'].to(device)      # (B, N, 3)
            mask = batch['neuron_mask'].to(device)         # (B, N)
            stim = batch['stimulus'].to(device)            # (B, L, K)
            neuron_ids = batch['neuron_ids'].to(device)    # (B, N)
            lam = batch.get('log_activity_mean', torch.empty(0)).to(device)
            las = batch.get('log_activity_std', torch.empty(0)).to(device)
            sr = float(cfg['training'].get('sampling_rate_hz', 3.0))

            # Prepare autoregressive pairs
            x_in = spikes[:, :-1, :]   # (B, L-1, N)
            x_tg = spikes[:, 1:, :].float()
            if debug_enabled():
                assert_no_nan(x_in, 'batch.x_in_raw')
                assert_no_nan(x_tg, 'batch.x_tg_raw')
            # Guard against NaNs/Infs in raw data
            x_in = torch.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0)
            x_tg = torch.nan_to_num(x_tg, nan=0.0, posinf=0.0, neginf=0.0)
            stim_in = stim[:, :-1, :]

            # Spike probabilities from rates for attention routing
            spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
            # Robustify probs to [0,1] and finite before model use
            spike_probs = torch.nan_to_num(spike_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            if debug_enabled():
                assert_no_nan(spike_probs, 'routing.spike_probs')

            # z-normalize log input when stats available: z = (log(x+eps) - mean)/std
            eps = 1e-7
            x_log = torch.log(x_in.clamp_min(eps))
            if debug_enabled():
                assert_no_nan(x_log, 'pre.zlog.x_log')
            if lam.numel() > 0 and las.numel() > 0:
                lam_e = lam[:, None, :].to(dtype=x_log.dtype)
                las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
                x_in_z = (x_log - lam_e) / las_e
            else:
                x_in_z = x_log
            # Ensure inputs are finite before entering model
            x_in_z = torch.nan_to_num(x_in_z, nan=0.0, posinf=0.0, neginf=0.0)
            if debug_enabled():
                assert_no_nan(x_in_z, 'pre.model.x_in_z')

            if device.type == 'cuda':
                x_in_z = x_in_z.to(torch.bfloat16)
                stim_in = stim_in.to(torch.bfloat16)

            if train:
                optimizer.zero_grad()

            mu, raw_log_sigma, _, _ = model(x_in_z, stim_in, positions, mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True)
            if debug_enabled():
                assert_no_nan(mu, 'model.mu')
                assert_no_nan(raw_log_sigma, 'model.raw_log_sigma')
            # Post-check for non-finite params
            if not torch.isfinite(mu).all() or not torch.isfinite(raw_log_sigma).all():
                mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
                raw_log_sigma = torch.nan_to_num(raw_log_sigma, nan=0.0, posinf=0.0, neginf=0.0)

            # Z-normalized target for loss in Normal(z) domain
            eps_n = 1e-7
            y_tg = torch.log(x_tg.clamp_min(eps_n))  # (B, L-1, N)
            if lam.numel() > 0 and las.numel() > 0:
                lam_e_loss = lam[:, None, :].to(dtype=y_tg.dtype)
                las_e_loss = las[:, None, :].to(dtype=y_tg.dtype).clamp_min(1e-6)
                z_tg = (y_tg - lam_e_loss) / las_e_loss
            else:
                z_tg = y_tg

            # Normal NLL on z_tg with model's (mu, sigma) in z-domain
            sigma_y = F.softplus(raw_log_sigma.to(torch.float32)) + 1e-6
            if lam.numel() > 0 and las.numel() > 0:
                mu_z = (mu.to(torch.float32) - lam_e_loss.to(torch.float32)) / las_e_loss.to(torch.float32)
                sigma_z = sigma_y / las_e_loss.to(torch.float32)
            else:
                mu_z = mu.to(torch.float32)
                sigma_z = sigma_y
            z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
            nll = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)
            # Build mask for loss to match target shape
            mask_exp = mask[:, None, :].expand_as(x_tg).float()
            if mask_exp is not None:
                total_m = mask_exp.sum().clamp_min(1.0)
                loss = (nll * mask_exp).sum() / total_m
            else:
                loss = nll.mean()
            if debug_enabled():
                assert_no_nan(loss, 'loss.nll')

            if train:
                loss.backward()
                if cfg['training'].get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_norm'])
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            count += 1

            # Update trackers and tqdm
            global_step += 1
            batch_loss_f = float(loss.detach().cpu())
            train_batch_losses.append(batch_loss_f)
            ema_value = (ema_value * ema_beta + batch_loss_f * (1.0 - ema_beta)) if (ema_value is not None) else batch_loss_f
            train_batch_ema.append(float(ema_value))
            pbar.set_postfix({'loss': f"{batch_loss_f:.4f}", 'ema': f"{float(ema_value):.4f}"})

            # Lightweight intra-epoch validation at configured steps
            if train and (count in triggers):
                model.eval()
                try:
                    total_v = 0.0
                    vb = 0
                    final_ctx = None
                    final_truth = None
                    final_pred = None
                    with torch.no_grad():
                        for vbatch in val_loader:
                            spikes_v = vbatch['spikes'].to(device)
                            positions_v = vbatch['positions'].to(device)
                            mask_v = vbatch['neuron_mask'].to(device)
                            stim_v = vbatch['stimulus'].to(device)
                            neuron_ids_v = vbatch['neuron_ids'].to(device)
                            lam_v = vbatch.get('log_activity_mean', torch.empty(0)).to(device)
                            las_v = vbatch.get('log_activity_std', torch.empty(0)).to(device)
                            sr_v = float(cfg['training'].get('sampling_rate_hz', 3.0))

                            x_in_v = spikes_v[:, :-1, :]
                            x_tg_v = spikes_v[:, 1:, :].float()
                            stim_in_v = stim_v[:, :-1, :]

                            eps_v = 1e-7
                            x_log_v = torch.log(x_in_v.clamp_min(eps_v))
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                lam_e_v = lam_v[:, None, :].to(dtype=x_log_v.dtype)
                                las_e_v = las_v[:, None, :].to(dtype=x_log_v.dtype).clamp_min(1e-6)
                                x_in_z_v = (x_log_v - lam_e_v) / las_e_v
                            else:
                                x_in_z_v = x_log_v

                            if device.type == 'cuda':
                                x_in_z_v = x_in_z_v.to(torch.bfloat16)
                                stim_in_v = stim_in_v.to(torch.bfloat16)

                            spike_probs_v = 1.0 - torch.exp(-x_in_v.float() / sr_v)
                            mu_v, raw_log_sigma_v, _, _ = model(x_in_z_v, stim_in_v, positions_v, mask_v, neuron_ids_v, spike_probs_v, get_logits=True, input_log_rates=True)
                            # Z-normalized validation target and Normal NLL in z-domain
                            y_tg_v = torch.log(x_tg_v.clamp_min(1e-7))
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                lam_e_v2 = lam_v[:, None, :].to(dtype=y_tg_v.dtype)
                                las_e_v2 = las_v[:, None, :].to(dtype=y_tg_v.dtype).clamp_min(1e-6)
                                z_tg_v = (y_tg_v - lam_e_v2) / las_e_v2
                            else:
                                z_tg_v = y_tg_v
                            sigma_y_v = F.softplus(raw_log_sigma_v.to(torch.float32)) + 1e-6
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                mu_z_v = (mu_v.to(torch.float32) - lam_e_v2.to(torch.float32)) / las_e_v2.to(torch.float32)
                                sigma_z_v = sigma_y_v / las_e_v2.to(torch.float32)
                            else:
                                mu_z_v = mu_v.to(torch.float32)
                                sigma_z_v = sigma_y_v
                            z_err_v = (z_tg_v.to(torch.float32) - mu_z_v) / sigma_z_v
                            nll_v = 0.5 * z_err_v.pow(2) + torch.log(sigma_z_v) + 0.5 * math.log(2.0 * math.pi)
                            mask_exp_v = mask_v[:, None, :].expand_as(x_tg_v).float()
                            vloss = (nll_v * mask_exp_v).sum() / mask_exp_v.sum().clamp_min(1.0)
                            total_v += float(vloss.detach().cpu().item())
                            vb += 1

                            # prepare small rollout visualization
                            Bv, Lm1_v, Nv = x_in_v.shape
                            Tc_v = max(1, Lm1_v // 2)
                            Tf_v = min(Lm1_v - Tc_v, 16) if (Lm1_v - Tc_v) > 0 else 1
                            init_context_v = x_in_v[0:1, :Tc_v, :].to(torch.float32)
                            stim_full_v = stim_v[0:1, :Tc_v+Tf_v, :]
                            pred_future_v = autoregressive_rollout(model, init_context_v, stim_full_v, positions_v[0:1], mask_v[0:1], neuron_ids_v[0:1], lam_v[0:1] if lam_v.numel()>0 else None, las_v[0:1] if las_v.numel()>0 else None, device, Tf_v, sampling_rate_hz=sr_v)
                            final_ctx = init_context_v[0].cpu()
                            final_truth = x_in_v[0, Tc_v:Tc_v+Tf_v, :].cpu()
                            final_pred = pred_future_v.cpu()
                            if vb >= val_sample_batches_local:
                                break

                    val_loss_step = total_v / max(1, vb)
                    val_points.append((global_step, float(val_loss_step)))
                    # Log step-level validation
                    with open(logs_dir / 'loss.txt', 'a') as f:
                        f.write(f"epoch {epoch}, step {count}, train {float(loss.detach().cpu().item()):.6f}, val {val_loss_step:.6f}\n")
                    if final_ctx is not None and final_truth is not None and final_pred is not None:
                        step_dir = Path(plots_dir) / f"epoch_{epoch}_step_{count}"
                        make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"E{epoch} step {count} autoreg 16 neurons")
                    _update_loss_plot()
                except Exception:
                    pass
                model.train()
        return total_loss / max(1, count)

    best_val = float('inf')
    val_freq = int(cfg['training'].get('validation_frequency') or 0)
    val_sample_batches = int(cfg['training'].get('val_sample_batches') or 1)

    for epoch in range(1, int(cfg['training']['num_epochs']) + 1):
        train_loss = train_or_val_loop(train_loader, epoch, train=True)

        # Periodic validation within epoch
        if val_freq > 0:
            # sample a few batches, aggregate mean
            model.eval()
            total = 0.0
            vb = 0
            final_ctx = None
            final_truth = None
            final_pred = None
            with torch.no_grad():
                for batch in val_loader:
                    spikes = batch['spikes'].to(device)
                    positions = batch['positions'].to(device)
                    mask = batch['neuron_mask'].to(device)
                    stim = batch['stimulus'].to(device)
                    neuron_ids = batch['neuron_ids'].to(device)
                    lam = batch.get('log_activity_mean', torch.empty(0)).to(device)
                    las = batch.get('log_activity_std', torch.empty(0)).to(device)
                    sr = float(cfg['training'].get('sampling_rate_hz', 3.0))

                    x_in = spikes[:, :-1, :]
                    x_tg = spikes[:, 1:, :].float()
                    stim_in = stim[:, :-1, :]

                    eps = 1e-7
                    x_log = torch.log(x_in.clamp_min(eps))
                    if lam.numel() > 0 and las.numel() > 0:
                        lam_e = lam[:, None, :].to(dtype=x_log.dtype)
                        las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
                        x_in_z = (x_log - lam_e) / las_e
                    else:
                        x_in_z = x_log

                    if device.type == 'cuda':
                        x_in_z = x_in_z.to(torch.bfloat16)
                        stim_in = stim_in.to(torch.bfloat16)

                    # Spike probabilities for routing during validation
                    spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
                    spike_probs = torch.nan_to_num(spike_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                    mu, raw_log_sigma, _, _ = model(x_in_z, stim_in, positions, mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True)
                    # Z-normalized validation target and Normal NLL in z-domain
                    y_tg = torch.log(x_tg.clamp_min(1e-7))
                    if lam.numel() > 0 and las.numel() > 0:
                        lam_e2 = lam[:, None, :].to(dtype=y_tg.dtype)
                        las_e2 = las[:, None, :].to(dtype=y_tg.dtype).clamp_min(1e-6)
                        z_tg = (y_tg - lam_e2) / las_e2
                    else:
                        z_tg = y_tg
                    sigma_y = F.softplus(raw_log_sigma.to(torch.float32)) + 1e-6
                    if lam.numel() > 0 and las.numel() > 0:
                        mu_z = (mu.to(torch.float32) - lam_e2.to(torch.float32)) / las_e2.to(torch.float32)
                        sigma_z = sigma_y / las_e2.to(torch.float32)
                    else:
                        mu_z = mu.to(torch.float32)
                        sigma_z = sigma_y
                    z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                    nll = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)
                    mask_exp = mask[:, None, :].expand_as(x_tg).float()
                    vloss = (nll * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)
                    total += float(vloss.detach().cpu().item())
                    vb += 1
                    # keep one for plot: multi-step autoregressive rollout using last half as horizon
                    B, Lm1, N = x_in.shape
                    Tc = max(1, Lm1 // 2)
                    Tf = min(Lm1 - Tc, 16) if (Lm1 - Tc) > 0 else 1
                    init_context = x_in[0:1, :Tc, :].to(torch.float32)
                    stim_full = stim[0:1, :Tc+Tf, :]
                    pred_future = autoregressive_rollout(model, init_context, stim_full, positions[0:1], mask[0:1], neuron_ids[0:1], lam[0:1] if lam.numel()>0 else None, las[0:1] if las.numel()>0 else None, device, Tf, sampling_rate_hz=sr)
                    final_ctx = init_context[0].cpu()
                    final_truth = x_in[0, Tc:Tc+Tf, :].cpu()
                    final_pred = pred_future.cpu()
                    if vb >= val_sample_batches:
                        break
            val_loss = total / max(1, vb)

            # Write scalar text and plot
            with open(logs_dir / 'loss.txt', 'a') as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            val_points.append((global_step, float(val_loss)))
            _update_loss_plot()
            # Autoregressive visualization (context + future) for 16 neurons on the kept batch
            if final_ctx is not None and final_truth is not None and final_pred is not None:
                step_dir = Path(plots_dir) / f"epoch_{epoch}"
                make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"E{epoch} autoreg 16 neurons")

            # Track best
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_dir / 'best_model.pth')

        else:
            # End-of-epoch validation only
            val_loss = train_or_val_loop(val_loader, epoch, train=False)
            with open(logs_dir / 'loss.txt', 'a') as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_dir / 'best_model.pth')

    print("Training complete. Best val:", best_val)


if __name__ == '__main__':
    main()


