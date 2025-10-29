#!/usr/bin/env python3
"""
Evaluation/validation + autoregressive (AR) script for GBM2 experiments.

Given an experiments/gbm2/<YYYYmmdd_HHMMSS> directory, this script:
 - Loads the saved config.yaml
 - Builds the dataloaders
 - Constructs the model and restores the checkpoint (best_model.pth by default)
 - Computes validation loss over a configurable number of batches
 - Produces one or more AR visualizations (16 neurons) like train_gbm2.py
 - Writes outputs back into the same experiment directory under plots/ and logs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any
import copy
import math

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled
from GenerativeBrainModel.utils.lognormal import lognormal_rate_median, sample_lognormal, lognormal_rate_mean   


def _set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _strip_orig_mod_prefix(state_dict: dict) -> dict:
    if all(not k.startswith('_orig_mod.') for k in state_dict.keys()):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            nk = k[len('_orig_mod.') :]
        else:
            nk = k
        out[nk] = v
    return out


def _load_state_dict_robust(model: torch.nn.Module, state: dict) -> None:
    """Load a possibly-compiled checkpoint robustly by:
    - stripping _orig_mod. prefix if present
    - intersecting by keys and matching tensor shapes
    - loading with strict=False
    """
    model_sd = model.state_dict()
    state = _strip_orig_mod_prefix(state)
    filtered = {k: v for k, v in state.items() if (k in model_sd and getattr(v, 'shape', None) == model_sd[k].shape)}
    missing = [k for k in model_sd.keys() if k not in filtered]
    unexpected = [k for k in state.keys() if k not in filtered]
    if (len(filtered) == 0):
        raise RuntimeError('No matching parameters found between checkpoint and current model. Ensure code version matches the checkpoint.')
    model.load_state_dict(filtered, strict=False)
    try:
        print(f"[eval] loaded params: matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}")
    except Exception:
        pass

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
def make_mu_sigma_scatter(step_dir: Path,
                          mu_last: torch.Tensor,      # (N,) in log-rate (z or y domain depending on caller)
                          sigma_last: torch.Tensor,   # (N,) positive
                          valid_mask: torch.Tensor,   # (N,) bool or float
                          title: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    x = mu_last.to(torch.float32).detach().cpu().numpy()
    y = sigma_last.to(torch.float32).detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x, y, s=6, alpha=0.6, color='tab:blue', edgecolors='none')
    ax.set_xlabel('predicted mu (log-rate)')
    ax.set_ylabel('predicted sigma (log-rate)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(step_dir / 'mu_sigma_scatter.png')
    plt.close(fig)

@torch.no_grad()
def make_mean_rate_plot(step_dir: Path,
                        context_truth: torch.Tensor,
                        future_truth: torch.Tensor,
                        future_pred: torch.Tensor,
                        title: str) -> None:
    """Plot mean rate over neurons vs time for context, true future, and AR prediction."""
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    # Means across neurons
    ctx_mean = context_truth.to(torch.float32).mean(dim=1).cpu().numpy() if context_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
    fut_mean = future_truth.to(torch.float32).mean(dim=1).cpu().numpy() if future_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
    pred_mean = future_pred.to(torch.float32).mean(dim=1).cpu().numpy() if future_pred.numel() > 0 else np.zeros((0,), dtype=np.float32)

    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if ctx_mean.size:
        ax.plot(x_context, ctx_mean, color='gray', lw=1.5, label='context mean rate')
    if fut_mean.size:
        ax.plot(x_future, fut_mean, color='tab:blue', lw=1.5, label='future truth mean')
    if pred_mean.size:
        ax.plot(x_future, pred_mean, color='tab:orange', lw=1.5, label='future pred mean')
    ax.set_xlabel('time')
    ax.set_ylabel('mean rate')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    out = step_dir / 'mean_rate_over_time.png'
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_next_step_scatter(step_dir: Path,
                           true_last: torch.Tensor,   # (N,)
                           pred_last: torch.Tensor,   # (N,)
                           valid_mask: torch.Tensor,  # (N,) bool or float
                           title: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    t = true_last.to(torch.float32).detach().cpu().numpy()
    p = pred_last.to(torch.float32).detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    t = t[m]
    p = p[m]
    if t.size == 0:
        return
    # sort by true ascending
    order = np.argsort(t)
    t_sorted = t[order]
    p_sorted = p[order]
    # Pearson r
    if t_sorted.size >= 2:
        r = float(np.corrcoef(t_sorted, p_sorted)[0, 1])
    else:
        r = float('nan')
    # limits
    lo = float(min(t_sorted.min(), p_sorted.min()))
    hi = float(max(t_sorted.max(), p_sorted.max()))
    pad = 0.02 * (hi - lo + 1e-8)
    lo -= pad
    hi += pad
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(t_sorted, p_sorted, s=6, alpha=0.6, color='tab:orange', edgecolors='none')
    ax.plot([lo, hi], [lo, hi], color='gray', lw=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('true next-step rate')
    ax.set_ylabel('predicted next-step rate')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(step_dir / 'next_step_scatter.png')
    plt.close(fig)

@torch.no_grad()
def make_log_rate_scatter(step_dir: Path,
                          true_last: torch.Tensor,   # (N,) rate
                          pred_last: torch.Tensor,   # (N,) rate
                          valid_mask: torch.Tensor,  # (N,) bool or float
                          title: str,
                          eps: float = 1e-7) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    t = torch.log(true_last.clamp_min(eps)).to(torch.float32).detach().cpu().numpy()
    p = torch.log(pred_last.clamp_min(eps)).to(torch.float32).detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    t = t[m]
    p = p[m]
    if t.size == 0:
        return
    order = np.argsort(t)
    t_sorted = t[order]
    p_sorted = p[order]
    if t_sorted.size >= 2:
        r = float(np.corrcoef(t_sorted, p_sorted)[0, 1])
    else:
        r = float('nan')
    lo = float(min(t_sorted.min(), p_sorted.min()))
    hi = float(max(t_sorted.max(), p_sorted.max()))
    pad = 0.02 * (hi - lo + 1e-8)
    lo -= pad
    hi += pad
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(t_sorted, p_sorted, s=6, alpha=0.6, color='tab:green', edgecolors='none')
    ax.plot([lo, hi], [lo, hi], color='gray', lw=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('log true next-step rate')
    ax.set_ylabel('log predicted next-step rate')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(step_dir / 'log_next_step_scatter.png')
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
    Lc = int(init_context.shape[1])
    # Pre-sample spike masks for the initial context as 0/1 floats (deterministic under Bernoulli when re-used)
    sr_f = float(sampling_rate_hz)
    spike_prob_init = 1.0 - torch.exp(-context.to(torch.float32) / sr_f)
    spike_prob_init = torch.nan_to_num(spike_prob_init, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    spike_mask_ctx = torch.bernoulli(spike_prob_init).to(torch.float32)  # (1, Tc, N) 0/1 floats
    preds = []
    eps = 1e-7
    for t in range(Tf):
        # Use sliding window of last Lc steps
        x_in = context[:, -Lc:, :]  # (1,Lc,N)
        x_log = torch.log(x_in.clamp_min(eps))
        lam_e = lam[:, None, :].to(dtype=x_log.dtype)
        las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
        x_in_z = (x_log - lam_e) / las_e

        # Align stimuli for the sliding window
        stim_step = stim_full[:, (context.shape[1]-Lc):(context.shape[1]), :]
        if device.type == 'cuda':
            x_in_z = x_in_z.to(torch.bfloat16)
            stim_step = stim_step.to(torch.bfloat16)
        # Use pre-sampled spike mask as probabilities (0/1) so internal Bernoulli is deterministic
        spike_probs_window = spike_mask_ctx[:, -Lc:, :]  # (1, Lc, N) float32 0/1
        mu, raw_log_sigma, _, _ = model(x_in_z, stim_step, positions, neuron_mask, neuron_ids, spike_probs_window, get_logits=True, input_log_rates=True)
        # Take last-time params and sample from LogNormal
        mu_last = mu[:, -1:, :]
        sig_last = raw_log_sigma[:, -1:, :]
        samp = sample_lognormal(mu_last, sig_last)  # (1,1,N) rates
        preds.append(samp[:, 0, :].to(torch.float32))
        # Append predicted rate and its pre-sampled spike mask
        context = torch.cat([context, samp.to(context.dtype)], dim=1)
        next_prob = 1.0 - torch.exp(-samp.to(torch.float32) / sr_f)
        #next_prob = torch.nan_to_num(next_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        next_mask = torch.bernoulli(next_prob).to(torch.float32)  # (1,1,N)
        spike_mask_ctx = torch.cat([spike_mask_ctx, next_mask], dim=1)
    return torch.cat(preds, dim=0)  # (Tf,N)


def main():
    ap = argparse.ArgumentParser(description='Evaluate GBM (validation + AR) from an experiment directory')
    ap.add_argument('--exp_dir', type=str, required=True, help='Path to experiments/gbm2/<YYYYmmdd_HHMMSS>')
    ap.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path; defaults to <exp_dir>/checkpoints/best_model.pth')
    ap.add_argument('--val_batches', type=int, default=None, help='Number of validation batches to average (defaults to cfg.training.val_sample_batches or all)')
    ap.add_argument('--horizon', type=int, default=16, help='AR prediction horizon steps for visualization')
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg_path = exp_dir / 'config.yaml'
    ckpt_path = Path(args.ckpt) if args.ckpt else (exp_dir / 'checkpoints' / 'best_model.pth')
    plots_dir = exp_dir / 'plots'
    logs_dir = exp_dir / 'logs'
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    # Match training RNG behavior to keep validation comparable
    try:
        _set_seeds(int(cfg['training'].get('seed', 42)))
    except Exception:
        pass

    # Build an eval config with longer sequence length to include future truth beyond original window
    horizon = int(args.horizon)
    cfg_eval = copy.deepcopy(cfg)
    try:
        cfg_eval['training']['sequence_length'] = int(cfg_eval['training']['sequence_length']) + horizon
    except Exception:
        pass

    # Dataloaders (use extended window for eval so we have future ground truth)
    # Also receive global unique neuron IDs for collision-free embeddings
    train_loader, val_loader, _, _, unique_neuron_ids = create_dataloaders(cfg_eval)

    # Infer d_stimuli from a sample if needed
    try:
        sample = next(iter(train_loader))
        d_stimuli = int(sample['stimulus'].shape[-1])
    except Exception:
        d_stimuli = cfg['model'].get('d_stimuli') or 1

    model = GBM(d_model=cfg['model']['d_model'], d_stimuli=d_stimuli,
                n_heads=cfg['model']['n_heads'], n_layers=cfg['model']['n_layers'],
                num_neurons_total=int(cfg['model']['num_neurons_total']),
                global_neuron_ids=unique_neuron_ids).to(device)

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        model = model.to(dtype=torch.bfloat16)

    # Load checkpoint (then optionally compile, mirroring training)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('model', ckpt)
    _load_state_dict_robust(model, sd)
    if bool(cfg['training'].get('compile_model', False)):
        try:
            model = torch.compile(model, dynamic=True)
        except Exception:
            pass
    model.eval()

    # Validation pass and one AR visualization
    total = 0.0
    vb = 0
    final_ctx = None
    final_truth = None
    final_pred = None

    val_batches_limit = args.val_batches if args.val_batches is not None else int(cfg['training'].get('val_sample_batches') or 0)
    if val_batches_limit <= 0:
        val_batches_limit = 1_000_000_000  # effectively all

    sr = float(cfg['training'].get('sampling_rate_hz', 3.0))
    orig_seq_len = int(cfg['training'].get('sequence_length', 48))
    Lm1_orig = max(1, orig_seq_len - 1)

    with torch.no_grad():
        for batch in val_loader:
            spikes = batch['spikes'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['neuron_mask'].to(device)
            stim = batch['stimulus'].to(device)
            neuron_ids = batch['neuron_ids'].to(device)
            lam = batch.get('log_activity_mean', torch.empty(0)).to(device)
            las = batch.get('log_activity_std', torch.empty(0)).to(device)

            # Full windows from extended sequence
            x_in_full = spikes[:, :-1, :]
            x_tg_full = spikes[:, 1:, :].float()
            stim_in_full = stim[:, :-1, :]
            # Crop to original training window length for loss parity
            Lm1_use = min(Lm1_orig, int(x_in_full.shape[1]))
            x_in = x_in_full[:, :Lm1_use, :]
            x_tg = x_tg_full[:, :Lm1_use, :]
            stim_in = stim_in_full[:, :Lm1_use, :]

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
            spike_probs = torch.nan_to_num(spike_probs, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            mu, raw_log_sigma, _, _ = model(x_in_z, stim_in, positions, mask, neuron_ids, spike_probs, get_logits=True, input_log_rates=True)

            # Normal NLL in z-domain
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

            # Prepare one AR visualization with future truth beyond original window
            if final_ctx is None:
                B, Lm1_full, N = x_in_full.shape
                Tf = 32#min(horizon, max(1, Lm1_full - Lm1_orig)) if Lm1_full > Lm1_orig else min(horizon, Lm1_full)
                Tc = Lm1_orig
                # Ensure bounds
                Tf = max(1, min(Tf, max(1, Lm1_full - Tc)))
                init_context = x_in_full[0:1, :Tc, :].to(torch.float32)
                stim_full = stim[0:1, :Tc+Tf, :]
                pred_future = autoregressive_rollout(model, init_context, stim_full, positions[0:1], mask[0:1], neuron_ids[0:1], lam[0:1] if lam.numel()>0 else None, las[0:1] if las.numel()>0 else None, device, Tf, sampling_rate_hz=sr)
                final_ctx = init_context[0].cpu()
                final_truth = x_in_full[0, Tc:Tc+Tf, :].cpu()
                final_pred = pred_future.cpu()

                # Next-step scatter for last step of teacher-forced window
                true_last = x_tg[0, -1, :]                      # (N,)
                pred_last = sample_lognormal(mu[0:1, -1:, :], raw_log_sigma[0:1, -1:, :])[0, 0, :]
                step_dir = plots_dir / 'eval_ar'
                make_next_step_scatter(step_dir, true_last, pred_last, mask[0] != 0, title='Eval next-step scatter (last step)')
                # Mu vs Sigma scatter (last step, z/LogNormal params)
                mu_last = mu[0, -1, :].to(torch.float32)
                sigma_last = (F.softplus(raw_log_sigma[0, -1, :].to(torch.float32)) + 1e-6)
                make_mu_sigma_scatter(step_dir, mu_last, sigma_last, mask[0] != 0, title='Eval mu vs sigma (last step)')
                # Log-rate scatter (last step)
                make_log_rate_scatter(step_dir, true_last, pred_last, mask[0] != 0, title='Eval log-rate scatter (last step)')

            if vb >= val_batches_limit:
                break

    val_loss = total / max(1, vb)
    with open(logs_dir / 'eval_loss.txt', 'a') as f:
        f.write(f"val {val_loss:.6f} over {vb} batches\n")

    # Plot AR if we captured one
    if final_ctx is not None and final_truth is not None and final_pred is not None:
        step_dir = plots_dir / 'eval_ar'
        make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"Eval autoreg 16 neurons")
        make_mean_rate_plot(step_dir, final_ctx, final_truth, final_pred, title=f"Eval mean rate over time")

    print(f"Eval complete. Val loss: {val_loss:.6f} over {vb} batches. Outputs in: {exp_dir}")


if __name__ == '__main__':
    main()


