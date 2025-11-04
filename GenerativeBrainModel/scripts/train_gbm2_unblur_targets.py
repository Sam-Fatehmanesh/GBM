#!/usr/bin/env python3
"""
Train GBM with blurred inputs but UNBLURRED targets.
This mirrors train_gbm2.py but applies spatial blur only to the input sequence x_in,
while targets x_tg remain from the original (unblurred) spikes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
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
from GenerativeBrainModel.dataloaders.single_subject_dataloader import create_single_subject_dataloaders
from GenerativeBrainModel.utils.lognormal import sample_lognormal
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {'name': 'gbm2_neural_training_unblur_targets'},
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'use_cache': True,
            'include_files': None,
            'spatial_blur_voxel_size': 0.0,
        },
        'model': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_stimuli': None,
            'num_neurons_total': 4_000_000,
            'cov_rank': 32,
        },
        'training': {
            'batch_size': 2,
            'num_epochs': 20,
            'learning_rate': 5e-4,
            'muon_lr': 2e-2,
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
            'loss_r_min': 0.0,
            'copula_weight': 0.05,
            'copula_warmup_floor': 1e-3,
            'copula_jitter': 1e-3,
            'copula_detach_burnin_steps': 0,
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


def _maybe_empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def _apply_spatial_voxel_blur(spikes: torch.Tensor,
                               positions: torch.Tensor,
                               mask: torch.Tensor,
                               voxel_size: float) -> torch.Tensor:
    if voxel_size is None or float(voxel_size) <= 0.0:
        return spikes
    B, L, N = spikes.shape
    device = spikes.device
    dtype = spikes.dtype
    out = torch.zeros_like(spikes)
    for b in range(B):
        valid = (mask[b] != 0)
        if not bool(valid.any().item()):
            continue
        pos_bn = positions[b, valid].to(torch.float32)
        vox = torch.floor(pos_bn / float(voxel_size)).to(torch.long)
        vx_min, _ = vox.min(dim=0)
        vox0 = vox - vx_min
        sx = int(vox0[:, 0].max().item()) + 1
        sy = int(vox0[:, 1].max().item()) + 1
        code = vox0[:, 0] + vox0[:, 1] * sx + vox0[:, 2] * (sx * sy)
        num_groups = int(code.max().item()) + 1 if code.numel() > 0 else 0
        spikes_bn = spikes[b, :, valid].to(torch.float32)
        if num_groups == 0:
            out[b, :, valid] = spikes_bn.to(dtype)
            continue
        sums = spikes_bn.new_zeros((L, num_groups))
        sums.index_add_(1, code, spikes_bn)
        counts = torch.bincount(code, minlength=num_groups).to(sums.dtype)
        counts = counts.clamp_min(1)
        means = sums / counts.unsqueeze(0)
        blurred = means.index_select(1, code)
        out[b, :, valid] = blurred.to(dtype)
    return out


@torch.no_grad()
def make_validation_plots(step_dir: Path,
                          context_truth: torch.Tensor,
                          future_truth: torch.Tensor,
                          future_pred: torch.Tensor,
                          title: str,
                          max_neurons: int = 16) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    # ensure same horizon length for truth and pred
    Tf_tg = int(future_truth.shape[0])
    Tf_pr = int(future_pred.shape[0])
    Tf = min(Tf_tg, Tf_pr)
    future_truth = future_truth[:Tf]
    future_pred = future_pred[:Tf]
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
                           init_context: torch.Tensor,
                           stim_full: torch.Tensor,
                           positions: torch.Tensor,
                           neuron_mask: torch.Tensor,
                           neuron_ids: torch.Tensor,
                           lam: torch.Tensor | None,
                           las: torch.Tensor | None,
                           device: torch.device,
                           Tf: int,
                           sampling_rate_hz: float = 3.0) -> torch.Tensor:
    model.eval()
    context = init_context.clone()
    Lc = int(init_context.shape[1])
    preds = []
    eps = 1e-7
    sr_f = float(sampling_rate_hz)
    prob_init = 1.0 - torch.exp(-context.to(torch.float32) / sr_f)
    prob_init = torch.nan_to_num(prob_init, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    spike_mask_ctx = torch.bernoulli(prob_init).to(torch.float32)
    for t in range(Tf):
        x_in = context[:, -Lc:, :]
        x_log = torch.log(x_in.clamp_min(eps))
        if (lam is not None) and (lam.numel() > 0) and (las is not None) and (las.numel() > 0):
            lam_e = lam[:, None, :].to(dtype=x_log.dtype)
            las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
            x_in_z = (x_log - lam_e) / las_e
        else:
            x_in_z = x_log
        stim_step = stim_full[:, (context.shape[1]-Lc):(context.shape[1]), :]
        if device.type == 'cuda':
            x_in_z = x_in_z.to(torch.bfloat16)
            stim_step = stim_step.to(torch.bfloat16)
        spike_probs_window = spike_mask_ctx[:, -Lc:, :]
        mu, raw_log_sigma, _, _ = model(x_in_z, stim_step, positions, neuron_mask, neuron_ids, spike_probs_window, get_logits=True, input_log_rates=True)
        mu_last = mu[:, -1:, :]
        sig_last = raw_log_sigma[:, -1:, :]
        samp = sample_lognormal(mu_last, sig_last)
        preds.append(samp[:, 0, :].to(torch.float32))
        context = torch.cat([context, samp.to(context.dtype)], dim=1)
        next_prob = 1.0 - torch.exp(-samp.to(torch.float32) / sr_f)
        next_prob = torch.nan_to_num(next_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        next_mask = torch.bernoulli(next_prob).to(torch.float32)
        spike_mask_ctx = torch.cat([spike_mask_ctx, next_mask], dim=1)
    return torch.cat(preds, dim=0)


def main():
    ap = argparse.ArgumentParser(description='Train GBM with blurred inputs and UNBLURRED targets')
    ap.add_argument('--config', type=str, default=None, help='YAML config for overrides')
    args = ap.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    set_seeds(int(cfg['training']['seed']))

    try:
        import torch.distributed as dist
        if dist.is_available() and not dist.is_initialized():
            dist.get_world_size = lambda group=None: 1
            dist.get_rank = lambda group=None: 0
            def _fake_all_gather(tensor_list, tensor, group=None):
                if tensor_list is None or len(tensor_list) == 0:
                    return
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

    train_loader, val_loader, _, _, unique_neuron_ids = create_single_subject_dataloaders(cfg)

    try:
        sample = next(iter(train_loader))
        d_stimuli = int(sample['stimulus'].shape[-1])
    except Exception:
        d_stimuli = cfg['model'].get('d_stimuli') or 1

    model = GBM(d_model=cfg['model']['d_model'], d_stimuli=d_stimuli,
                n_heads=cfg['model']['n_heads'], n_layers=cfg['model']['n_layers'],
                num_neurons_total=int(cfg['model']['num_neurons_total']),
                global_neuron_ids=unique_neuron_ids,
                cov_rank=int(cfg['model'].get('cov_rank', 32))).to(device)

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        model = model.to(dtype=torch.bfloat16)

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

    # Optimizer (reuse factory from original script)
    try:
        from GenerativeBrainModel.scripts.train_gbm2 import build_optimizer
        optimizer = build_optimizer(model, cfg['training'])
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training'].get('learning_rate', 5e-4)), weight_decay=float(cfg['training'].get('weight_decay', 1e-4)))

    base_dir = Path('experiments/gbm2')
    run_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = run_dir / 'plots'
    ckpt_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'
    for p in (plots_dir, ckpt_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)

    train_batch_losses: list[float] = []
    train_batch_ema: list[float] = []
    ema_total_value: float | None = None
    val_points: list[tuple[int, float]] = []
    ema_beta: float = 0.98
    ema_value: float | None = None
    global_step: int = 0
    total_train_steps: int = int(cfg['training']['num_epochs']) * max(1, len(train_loader))

    def _lambda_copula_for_step(step: int) -> float:
        base = float(cfg['training'].get('copula_weight', 0.0) or 0.0)
        if base <= 0.0:
            return 0.0
        floor = float(cfg['training'].get('copula_warmup_floor', 1e-3))
        floor = max(0.0, min(floor, 1.0))
        frac = min(1.0, max(0.0, step / max(1, total_train_steps)))
        scale = floor + (1.0 - floor) * frac
        return base * scale

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
        nonlocal global_step, ema_value, ema_total_value
        total_loss = 0.0
        count = 0
        model.train() if train else model.eval()
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
            spikes = batch['spikes'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['neuron_mask'].to(device)
            stim = batch['stimulus'].to(device)
            neuron_ids = batch['neuron_ids'].to(device)
            lam = batch.get('log_activity_mean', torch.empty(0)).to(device)
            las = batch.get('log_activity_std', torch.empty(0)).to(device)
            sr = float(cfg['training'].get('sampling_rate_hz', 3.0))

            # Keep a copy of original spikes for targets
            spikes_orig = spikes

            # Optional blur only for inputs
            blur_vox = float(cfg['data'].get('spatial_blur_voxel_size', 0.0) or 0.0)
            spikes_blur = _apply_spatial_voxel_blur(spikes, positions, mask, blur_vox) if blur_vox > 0.0 else spikes

            x_in = spikes_blur[:, :-1, :]
            x_tg = spikes_orig[:, 1:, :].float()

            x_in = torch.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0)
            x_tg = torch.nan_to_num(x_tg, nan=0.0, posinf=0.0, neginf=0.0)
            stim_in = stim[:, :-1, :]

            spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
            spike_probs = torch.nan_to_num(spike_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

            eps = 1e-7
            x_log = torch.log(x_in.clamp_min(eps))
            if lam.numel() > 0 and las.numel() > 0:
                lam_e = lam[:, None, :].to(dtype=x_log.dtype)
                las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
                x_in_z = (x_log - lam_e) / las_e
            else:
                x_in_z = x_log
            x_in_z = torch.nan_to_num(x_in_z, nan=0.0, posinf=0.0, neginf=0.0)

            if device.type == 'cuda':
                x_in_z = x_in_z.to(torch.bfloat16)
                stim_in = stim_in.to(torch.bfloat16)

            if train:
                optimizer.zero_grad()

            out = model(x_in_z, stim_in, positions, mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True, return_factors=True)
            if isinstance(out, tuple) and len(out) == 5:
                mu, raw_log_sigma, _eta, _delta, factors = out
            else:
                mu, raw_log_sigma, _a, _b = out
                factors = None

            eps_n = 1e-7
            y_tg = torch.log(x_tg.clamp_min(eps_n))
            if lam.numel() > 0 and las.numel() > 0:
                lam_e_loss = lam[:, None, :].to(dtype=y_tg.dtype)
                las_e_loss = las[:, None, :].to(dtype=y_tg.dtype).clamp_min(1e-6)
                z_tg = (y_tg - lam_e_loss) / las_e_loss
            else:
                z_tg = y_tg

            sigma_y = F.softplus(raw_log_sigma.to(torch.float32)) + 1e-6
            if lam.numel() > 0 and las.numel() > 0:
                mu_z = (mu.to(torch.float32) - lam_e_loss.to(torch.float32)) / las_e_loss.to(torch.float32)
                sigma_z = sigma_y / las_e_loss.to(torch.float32)
            else:
                mu_z = mu.to(torch.float32)
                sigma_z = sigma_y

            r_min = float(cfg['training'].get('loss_r_min', 0.0) or 0.0)
            if r_min > 0.0:
                y_min = math.log(r_min)
                if lam.numel() > 0 and las.numel() > 0:
                    z_min = (torch.tensor(y_min, dtype=mu_z.dtype, device=mu_z.device) - lam_e_loss.to(mu_z.dtype)) / las_e_loss.to(mu_z.dtype)
                else:
                    z_min = torch.tensor(y_min, dtype=mu_z.dtype, device=mu_z.device)
                is_cens = (x_tg <= r_min)
                z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                nll_pdf = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)
                alpha = (z_min - mu_z) / sigma_z
                log_cdf = torch.log((0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))).clamp_min(1e-12))
                nll_cdf = -log_cdf
                nll = torch.where(is_cens, nll_cdf, nll_pdf)
            else:
                z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                nll = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)

            mask_exp = mask[:, None, :].expand_as(x_tg).float()
            total_m = mask_exp.sum().clamp_min(1.0)
            loss = (nll * mask_exp).sum() / total_m

            # No copula in this minimal variant (optional): reuse loss only
            loss_total = loss

            if train:
                loss_total.backward()
                if cfg['training'].get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip_norm'])
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            count += 1

            global_step += 1
            batch_loss_f = float(loss.detach().cpu())
            train_batch_losses.append(batch_loss_f)
            ema_value = (ema_value * ema_beta + batch_loss_f * (1.0 - ema_beta)) if (ema_value is not None) else batch_loss_f
            batch_total_loss_f = float(loss_total.detach().cpu())
            ema_total_value = (ema_total_value * ema_beta + batch_total_loss_f * (1.0 - ema_beta)) if (ema_total_value is not None) else batch_total_loss_f
            train_batch_ema.append(float(ema_value))
            pbar.set_postfix({'loss': f"{batch_loss_f:.4f}", 'ema': f"{float(ema_value):.4f}"})

            # Intra-epoch validation
            if train and (count in triggers):
                _maybe_empty_cuda_cache()
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

                            spikes_orig_v = spikes_v
                            blur_vox_v = float(cfg['data'].get('spatial_blur_voxel_size', 0.0) or 0.0)
                            spikes_blur_v = _apply_spatial_voxel_blur(spikes_v, positions_v, mask_v, blur_vox_v) if blur_vox_v > 0.0 else spikes_v

                            x_in_v = spikes_blur_v[:, :-1, :]
                            x_tg_v = spikes_orig_v[:, 1:, :].float()
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
                            out_v = model(x_in_z_v, stim_in_v, positions_v, mask_v, neuron_ids_v, spike_probs_v, get_logits=True, input_log_rates=True, return_factors=True)
                            if isinstance(out_v, tuple) and len(out_v) == 5:
                                mu_v, raw_log_sigma_v, _e_v, _d_v, factors_v = out_v
                            else:
                                mu_v, raw_log_sigma_v, _aa_v, _bb_v = out_v
                                factors_v = None
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
                            r_min_v = float(cfg['training'].get('loss_r_min', 0.0) or 0.0)
                            if r_min_v > 0.0:
                                y_min_v = math.log(r_min_v)
                                if lam_v.numel() > 0 and las_v.numel() > 0:
                                    z_min_v = (torch.tensor(y_min_v, dtype=mu_z_v.dtype, device=mu_z_v.device) - lam_e_v2.to(mu_z_v.dtype)) / las_e_v2.to(mu_z_v.dtype)
                                else:
                                    z_min_v = torch.tensor(y_min_v, dtype=mu_z_v.dtype, device=mu_z_v.device)
                                is_cens_v = (x_tg_v <= r_min_v)
                                z_err_v = (z_tg_v.to(torch.float32) - mu_z_v) / sigma_z_v
                                nll_pdf_v = 0.5 * z_err_v.pow(2) + torch.log(sigma_z_v) + 0.5 * math.log(2.0 * math.pi)
                                alpha_v = (z_min_v - mu_z_v) / sigma_z_v
                                log_cdf_v = torch.log((0.5 * (1.0 + torch.erf(alpha_v / math.sqrt(2.0)))).clamp_min(1e-12))
                                nll_cdf_v = -log_cdf_v
                                nll_v = torch.where(is_cens_v, nll_cdf_v, nll_pdf_v)
                            else:
                                z_err_v = (z_tg_v.to(torch.float32) - mu_z_v) / sigma_z_v
                                nll_v = 0.5 * z_err_v.pow(2) + torch.log(sigma_z_v) + 0.5 * math.log(2.0 * math.pi)
                            mask_exp_v = mask_v[:, None, :].expand_as(x_tg_v).float()
                            vloss = (nll_v * mask_exp_v).sum() / mask_exp_v.sum().clamp_min(1.0)
                            total_v += float(vloss.detach().cpu().item())
                            vb += 1

                            # small rollout viz: context from blurred inputs, truth from original
                            Bv, Lm1_v, Nv = x_in_v.shape
                            Tf_v = min(Lm1_v, 16) if Lm1_v > 0 else 1
                            Tc_v = max(1, Lm1_v - Tf_v)
                            init_context_v = x_in_v[0:1, :Tc_v, :].to(torch.float32)
                            stim_full_v = stim_v[0:1, :Tc_v+Tf_v, :]
                            pred_future_v = autoregressive_rollout(model, init_context_v, stim_full_v, positions_v[0:1], mask_v[0:1], neuron_ids_v[0:1], lam_v[0:1] if lam_v.numel()>0 else None, las_v[0:1] if las_v.numel()>0 else None, device, Tf_v, sampling_rate_hz=sr_v)
                            final_ctx = init_context_v[0].cpu()
                            final_truth = x_tg_v[0, Tc_v:Tc_v+Tf_v, :].cpu()
                            final_pred = pred_future_v.cpu()
                            if vb >= val_sample_batches_local:
                                break

                    val_loss_step = total_v / max(1, vb)
                    val_points.append((global_step, float(val_loss_step)))
                    with open(logs_dir / 'loss.txt', 'a') as f:
                        tr = float(ema_total_value) if (ema_total_value is not None) else float(loss_total.detach().cpu().item())
                        f.write(f"epoch {epoch}, step {count}, train {tr:.6f}, val {val_loss_step:.6f}\n")
                    if final_ctx is not None and final_truth is not None and final_pred is not None:
                        step_dir = Path(plots_dir) / f"epoch_{epoch}_step_{count}"
                        make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"E{epoch} step {count} autoreg 16 neurons")
                    _update_loss_plot()
                except Exception:
                    pass
                model.train()
                _maybe_empty_cuda_cache()
        return total_loss / max(1, count)

    best_val = float('inf')
    val_freq = int(cfg['training'].get('validation_frequency') or 0)
    val_sample_batches = int(cfg['training'].get('val_sample_batches') or 1)

    for epoch in range(1, int(cfg['training']['num_epochs']) + 1):
        train_loss = train_or_val_loop(train_loader, epoch, train=True)

        if val_freq > 0:
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

                    spikes_orig = spikes
                    blur_vox = float(cfg['data'].get('spatial_blur_voxel_size', 0.0) or 0.0)
                    spikes_blur = _apply_spatial_voxel_blur(spikes, positions, mask, blur_vox) if blur_vox > 0.0 else spikes

                    x_in = spikes_blur[:, :-1, :]
                    x_tg = spikes_orig[:, 1:, :].float()
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

                    spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
                    out_e = model(x_in_z, stim_in, positions, mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True, return_factors=True)
                    if isinstance(out_e, tuple) and len(out_e) == 5:
                        mu, raw_log_sigma, _ee, _dd, factors_e = out_e
                    else:
                        mu, raw_log_sigma, _aaa, _bbb = out_e
                        factors_e = None
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
                    r_min_e = float(cfg['training'].get('loss_r_min', 0.0) or 0.0)
                    if r_min_e > 0.0:
                        y_min_e = math.log(r_min_e)
                        if lam.numel() > 0 and las.numel() > 0:
                            z_min_e = (torch.tensor(y_min_e, dtype=mu_z.dtype, device=mu_z.device) - lam_e2.to(mu_z.dtype)) / las_e2.to(mu_z.dtype)
                        else:
                            z_min_e = torch.tensor(y_min_e, dtype=mu_z.dtype, device=mu_z.device)
                        is_cens_e = (x_tg <= r_min_e)
                        z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                        nll_pdf = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)
                        alpha_e = (z_min_e - mu_z) / sigma_z
                        log_cdf_e = torch.log((0.5 * (1.0 + torch.erf(alpha_e / math.sqrt(2.0)))).clamp_min(1e-12))
                        nll_cdf = -log_cdf_e
                        nll = torch.where(is_cens_e, nll_cdf, nll_pdf)
                    else:
                        z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                        nll = 0.5 * z_err.pow(2) + torch.log(sigma_z) + 0.5 * math.log(2.0 * math.pi)
                    mask_exp = mask[:, None, :].expand_as(x_tg).float()
                    vloss = (nll * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)
                    total += float(vloss.detach().cpu().item())
                    vb += 1

                    B, Lm1, N = x_in.shape
                    Tf = min(Lm1, 16) if Lm1 > 0 else 1
                    Tc = max(1, Lm1 - Tf)
                    init_context = x_in[0:1, :Tc, :].to(torch.float32)
                    stim_full = stim[0:1, :Tc+Tf, :]
                    pred_future = autoregressive_rollout(model, init_context, stim_full, positions[0:1], mask[0:1], neuron_ids[0:1], lam[0:1] if lam.numel()>0 else None, las[0:1] if las.numel()>0 else None, device, Tf, sampling_rate_hz=sr)
                    final_ctx = init_context[0].cpu()
                    final_truth = x_tg[0, Tc:Tc+Tf, :].cpu()
                    final_pred = pred_future.cpu()
                    if vb >= val_sample_batches:
                        break
            val_loss = total / max(1, vb)
            with open(logs_dir / 'loss.txt', 'a') as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            val_points.append((global_step, float(val_loss)))
            _update_loss_plot()
            _maybe_empty_cuda_cache()
            if final_ctx is not None and final_truth is not None and final_pred is not None:
                step_dir = Path(plots_dir) / f"epoch_{epoch}"
                make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"E{epoch} autoreg 16 neurons")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_dir / 'best_model.pth')

        else:
            val_loss = train_or_val_loop(val_loader, epoch, train=False)
            with open(logs_dir / 'loss.txt', 'a') as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch, 'model': model.state_dict()}, ckpt_dir / 'best_model.pth')
            _maybe_empty_cuda_cache()

    print("Training complete. Best val:", best_val)
    _maybe_empty_cuda_cache()


if __name__ == '__main__':
    main()


