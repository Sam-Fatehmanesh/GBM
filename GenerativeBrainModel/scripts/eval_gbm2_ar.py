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
from matplotlib.colors import LogNorm

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
def make_log_true_vs_log_sigma_scatter(step_dir: Path,
                                       true_last: torch.Tensor,    # (N,) rate
                                       sigma_last: torch.Tensor,   # (N,) sigma in log-rate domain (positive)
                                       valid_mask: torch.Tensor,   # (N,) bool or float
                                       title: str,
                                       eps: float = 1e-7) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    # log transforms
    x = torch.log(true_last.clamp_min(eps)).to(torch.float32).detach().cpu().numpy()
    y = torch.log(sigma_last.clamp_min(eps)).to(torch.float32).detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    if xs.size >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
    else:
        r = float('nan')
    lo_x = float(xs.min()); hi_x = float(xs.max())
    pad_x = 0.02 * (hi_x - lo_x + 1e-8)
    lo_x -= pad_x; hi_x += pad_x
    lo_y = float(ys.min()); hi_y = float(ys.max())
    pad_y = 0.02 * (hi_y - lo_y + 1e-8)
    lo_y -= pad_y; hi_y += pad_y
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    hb = ax.hexbin(xs, ys, gridsize=80, extent=(lo_x, hi_x, lo_y, hi_y), cmap='magma', bins='log', mincnt=1)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel('log true next-step rate')
    ax.set_ylabel('log predicted sigma (log-rate)')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('count (log)')
    fig.tight_layout()
    fig.savefig(step_dir / 'log_true_vs_log_sigma_scatter.png')
    plt.close(fig)


@torch.no_grad()
def make_log_true_vs_log_mu_scatter(step_dir: Path,
                                    true_last: torch.Tensor,   # (N,) rate
                                    mu_last: torch.Tensor,     # (N,) predicted mean in log-rate domain
                                    valid_mask: torch.Tensor,  # (N,) bool or float
                                    title: str,
                                    eps: float = 1e-7) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    # x-axis: log of true rate; y-axis: predicted mu (already log-rate)
    x = torch.log(true_last.clamp_min(eps)).to(torch.float32).detach().cpu().numpy()
    y = mu_last.to(torch.float32).detach().cpu().numpy()
    m = valid_mask.detach().cpu().numpy().astype(bool)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    if xs.size >= 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
    else:
        r = float('nan')
    # Build square extent to align the y=x reference line
    lo_x = float(xs.min()); hi_x = float(xs.max())
    lo_y = float(ys.min()); hi_y = float(ys.max())
    lo = min(lo_x, lo_y)
    hi = max(hi_x, hi_y)
    pad = 0.02 * (hi - lo + 1e-8)
    lo -= pad
    hi += pad
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    hb = ax.hexbin(xs, ys, gridsize=80, extent=(lo, hi, lo, hi), cmap='magma', bins='log', mincnt=1)
    ax.plot([lo, hi], [lo, hi], color='gray', lw=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('log true next-step rate')
    ax.set_ylabel('predicted mu (log-rate)')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('count (log)')
    fig.tight_layout()
    fig.savefig(step_dir / 'log_true_vs_log_mu_scatter.png')
    plt.close(fig)

@torch.no_grad()
def make_mu_sigma_scatter(step_dir: Path,
                         mu_last: torch.Tensor,      # (N,) in log-rate (z or y domain depending on caller)
                         sigma_last: torch.Tensor,   # (N,) positive
                         true_last: torch.Tensor,    # (N,) true target rate (unused here)
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
    hb = ax.hexbin(x, y, gridsize=80, cmap='viridis', bins='log', mincnt=1)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('count (log)')
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
def make_median_rate_plot(step_dir: Path,
                          context_truth: torch.Tensor,
                          future_truth: torch.Tensor,
                          future_pred: torch.Tensor,
                          title: str) -> None:
    """Plot median rate over neurons vs time for context, true future, and AR prediction."""
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    # Medians across neurons
    ctx_med = context_truth.to(torch.float32).median(dim=1).values.cpu().numpy() if context_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
    fut_med = future_truth.to(torch.float32).median(dim=1).values.cpu().numpy() if future_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
    pred_med = future_pred.to(torch.float32).median(dim=1).values.cpu().numpy() if future_pred.numel() > 0 else np.zeros((0,), dtype=np.float32)

    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if ctx_med.size:
        ax.plot(x_context, ctx_med, color='gray', lw=1.5, label='context median rate')
    if fut_med.size:
        ax.plot(x_future, fut_med, color='tab:blue', lw=1.5, label='future truth median')
    if pred_med.size:
        ax.plot(x_future, pred_med, color='tab:orange', lw=1.5, label='future pred median')
    ax.set_xlabel('time')
    ax.set_ylabel('median rate')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    out = step_dir / 'median_rate_over_time.png'
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_mean_rate_plots_stacked(step_dir: Path,
                                 contexts: list[torch.Tensor],
                                 futures: list[torch.Tensor],
                                 preds: list[torch.Tensor],
                                 title: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    K = len(contexts)
    if K == 0:
        return
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(K):
        context_truth = contexts[i]
        future_truth = futures[i]
        future_pred = preds[i]
        Tc = int(context_truth.shape[0])
        Tf = int(future_truth.shape[0])
        ctx_mean = context_truth.to(torch.float32).mean(dim=1).cpu().numpy() if context_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
        fut_mean = future_truth.to(torch.float32).mean(dim=1).cpu().numpy() if future_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
        pred_mean = future_pred.to(torch.float32).mean(dim=1).cpu().numpy() if future_pred.numel() > 0 else np.zeros((0,), dtype=np.float32)
        x_context = np.arange(Tc)
        x_future = np.arange(Tc, Tc + Tf)
        ax = axes[i]
        if ctx_mean.size:
            ax.plot(x_context, ctx_mean, color='gray', lw=1.2, label='context mean rate')
        if fut_mean.size:
            ax.plot(x_future, fut_mean, color='tab:blue', lw=1.2, label='future truth mean')
        if pred_mean.size:
            ax.plot(x_future, pred_mean, color='tab:orange', lw=1.2, label='future pred mean')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel(f'AR {i+1}')
    axes[-1].set_xlabel('time')
    fig.suptitle(title)
    fig.tight_layout()
    out = step_dir / 'mean_rate_over_time_stacked.png'
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_median_rate_plots_stacked(step_dir: Path,
                                   contexts: list[torch.Tensor],
                                   futures: list[torch.Tensor],
                                   preds: list[torch.Tensor],
                                   title: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    K = len(contexts)
    if K == 0:
        return
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(K):
        context_truth = contexts[i]
        future_truth = futures[i]
        future_pred = preds[i]
        Tc = int(context_truth.shape[0])
        Tf = int(future_truth.shape[0])
        ctx_med = context_truth.to(torch.float32).median(dim=1).values.cpu().numpy() if context_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
        fut_med = future_truth.to(torch.float32).median(dim=1).values.cpu().numpy() if future_truth.numel() > 0 else np.zeros((0,), dtype=np.float32)
        pred_med = future_pred.to(torch.float32).median(dim=1).values.cpu().numpy() if future_pred.numel() > 0 else np.zeros((0,), dtype=np.float32)
        x_context = np.arange(Tc)
        x_future = np.arange(Tc, Tc + Tf)
        ax = axes[i]
        if ctx_med.size:
            ax.plot(x_context, ctx_med, color='gray', lw=1.2, label='context median rate')
        if fut_med.size:
            ax.plot(x_future, fut_med, color='tab:blue', lw=1.2, label='future truth median')
        if pred_med.size:
            ax.plot(x_future, pred_med, color='tab:orange', lw=1.2, label='future pred median')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel(f'AR {i+1}')
    axes[-1].set_xlabel('time')
    fig.suptitle(title)
    fig.tight_layout()
    out = step_dir / 'median_rate_over_time_stacked.png'
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_spike_counts_plots_stacked(step_dir: Path,
                                    ctx_counts_list: list[torch.Tensor],
                                    true_counts_list: list[torch.Tensor],
                                    pred_counts_list: list[torch.Tensor],
                                    title: str) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    K = len(ctx_counts_list)
    if K == 0:
        return
    fig, axes = plt.subplots(K, 1, figsize=(10, 2*K), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(K):
        ctx_counts = ctx_counts_list[i]
        true_counts = true_counts_list[i]
        pred_counts = pred_counts_list[i]
        Tc = int(ctx_counts.shape[0])
        Tf = int(pred_counts.shape[0])
        x_context = np.arange(Tc)
        x_future = np.arange(Tc, Tc + Tf)
        ax = axes[i]
        if Tc > 0:
            ax.plot(x_context, ctx_counts.to(torch.float32).cpu().numpy(), color='gray', lw=1.2, label='context spike count')
        if Tf > 0 and true_counts.numel() == Tf:
            ax.plot(x_future, true_counts.to(torch.float32).cpu().numpy(), color='tab:blue', lw=1.2, label='future truth spike count')
        if Tf > 0:
            ax.plot(x_future, pred_counts.to(torch.float32).cpu().numpy(), color='tab:orange', lw=1.2, label='future pred spike count')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel(f'AR {i+1}')
    axes[-1].set_xlabel('time')
    fig.suptitle(title)
    fig.tight_layout()
    out = step_dir / 'spike_counts_over_time_stacked.png'
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
    hb = ax.hexbin(t_sorted, p_sorted, gridsize=80, extent=(lo, hi, lo, hi), cmap='inferno', bins='log', mincnt=1)
    ax.plot([lo, hi], [lo, hi], color='gray', lw=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('true next-step rate')
    ax.set_ylabel('predicted next-step rate')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    ax.legend(loc='lower right', fontsize=8)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('count (log)')
    fig.tight_layout()
    fig.savefig(step_dir / 'next_step_scatter.png')
    plt.close(fig)


@torch.no_grad()
def make_spike_counts_plot(step_dir: Path,
                           ctx_counts: torch.Tensor,        # (Tc,)
                           future_truth_counts: torch.Tensor,# (Tf,)
                           future_pred_counts: torch.Tensor, # (Tf,)
                           title: str) -> None:
    """Plot number of spikes per timestep (sum of spike mask across neurons)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc = int(ctx_counts.shape[0])
    Tf = int(future_pred_counts.shape[0])

    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if Tc > 0:
        ax.plot(x_context, ctx_counts.to(torch.float32).cpu().numpy(), color='gray', lw=1.5, label='context spike count')
    if Tf > 0 and future_truth_counts.numel() == Tf:
        ax.plot(x_future, future_truth_counts.to(torch.float32).cpu().numpy(), color='tab:blue', lw=1.5, label='future truth spike count')
    if Tf > 0:
        ax.plot(x_future, future_pred_counts.to(torch.float32).cpu().numpy(), color='tab:orange', lw=1.5, label='future pred spike count')
    ax.set_xlabel('time')
    ax.set_ylabel('# spikes')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(step_dir / 'spike_counts_over_time.png')
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
    hb = ax.hexbin(t_sorted, p_sorted, gridsize=80, extent=(lo, hi, lo, hi), cmap='viridis', bins='log', mincnt=1)
    ax.plot([lo, hi], [lo, hi], color='gray', lw=1.2, linestyle='--', label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('log true next-step rate')
    ax.set_ylabel('log predicted next-step rate')
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}", transform=ax.transAxes, ha='left', va='top')
    ax.legend(loc='lower right', fontsize=8)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('count (log)')
    fig.tight_layout()
    fig.savefig(step_dir / 'log_next_step_scatter.png')
    plt.close(fig)


@torch.no_grad()
def make_per_index_loss_plot(step_dir: Path,
                             per_index_mean: np.ndarray,
                             title: str) -> None:
    """Plot mean next-step loss vs sequence index (proxy for available context length)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    if per_index_mean.size == 0:
        return
    x = np.arange(1, per_index_mean.size + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(x, per_index_mean, color='tab:purple', lw=1.5)
    ax.set_xlabel('sequence index t (predicts t+1)')
    ax.set_ylabel('mean NLL')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(step_dir / 'per_index_next_step_loss.png')
    plt.close(fig)


@torch.no_grad()
def make_pca_population_plot(step_dir: Path,
                             true_list: list[torch.Tensor],  # list of (N,)
                             pred_list: list[torch.Tensor],  # list of (N,)
                             mask_list: list[torch.Tensor],  # list of (N,) bool
                             title: str) -> None:
    """Compute a 2D PCA on next-step TRUE rate vectors (samples x neurons),
    then project PRED vectors with the same PCA, and plot both distributions.
    Uses only neurons valid across all samples to avoid missing-data handling.
    """
    def _dbg(msg: str):
        try:
            print(f"[eval_pca] {msg}")
        except Exception:
            pass
    if len(true_list) == 0 or len(pred_list) == 0:
        _dbg(f"no samples: len(true_list)={len(true_list)}, len(pred_list)={len(pred_list)}")
        return
    try:
        X_true = torch.stack([t.to(torch.float32) for t in true_list], dim=0)  # (S,N)
        X_pred = torch.stack([p.to(torch.float32) for p in pred_list], dim=0)  # (S,N)
        M = torch.stack([m.to(torch.bool) for m in mask_list], dim=0)          # (S,N)
    except Exception as e:
        _dbg(f"exception during stack: {e}")
        return
    # Require at least 2 samples and 2 neurons
    S, N = int(X_true.shape[0]), int(X_true.shape[1])
    if S < 2 or N < 2:
        _dbg(f"insufficient samples/neurons: S={S}, N={N}")
        return
    # Keep neurons observed in at least 20% of samples
    counts = M.sum(dim=0)
    thresh = max(2, int(0.2 * S))
    keep_cols = counts >= thresh
    if not bool(keep_cols.any().item()):
        _dbg("no columns meet observation threshold")
        return
    Xtrue = X_true[:, keep_cols]
    Xpred = X_pred[:, keep_cols]
    Mkeep = M[:, keep_cols]
    K = int(Xtrue.shape[1])
    if K < 2:
        _dbg(f"kept cols <2: K={K}")
        return
    # Compute column means over available TRUE entries and impute missing with mean
    denom = Mkeep.sum(dim=0).clamp_min(1).to(Xtrue.dtype)
    col_mean = (Xtrue.masked_fill(~Mkeep, 0.0).sum(dim=0) / denom)  # (K,)
    Xtrue_imp = torch.where(Mkeep, Xtrue, col_mean.unsqueeze(0))
    Xpred_imp = torch.where(Mkeep, Xpred, col_mean.unsqueeze(0))  # use same mean for pred
    # Center with true means
    mu = col_mean.unsqueeze(0)
    Xtrue_c = (Xtrue_imp - mu).cpu().numpy()
    Xpred_c = (Xpred_imp - mu).cpu().numpy()
    _dbg(f"PCA input shapes: true={Xtrue_c.shape}, pred={Xpred_c.shape}")
    # PCA via SVD on TRUE
    try:
        U, Svals, VT = np.linalg.svd(Xtrue_c, full_matrices=False)
    except Exception as e:
        _dbg(f"svd failed: {e}")
        return
    comps = VT[:2].T  # (K,2)
    Zt = Xtrue_c @ comps  # (S,2)
    Zp = Xpred_c @ comps  # (S,2)
    # Single-axes scatter overlay (true vs pred) with shared limits
    lo = np.minimum(Zt.min(axis=0), Zp.min(axis=0))
    hi = np.maximum(Zt.max(axis=0), Zp.max(axis=0))
    pad = 0.05 * (hi - lo + 1e-8)
    lo -= pad
    hi += pad
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(Zt[:, 0], Zt[:, 1], s=6, alpha=0.35, color='tab:blue', label='true')
    ax.scatter(Zp[:, 0], Zp[:, 1], s=6, alpha=0.35, color='tab:orange', label='pred')
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(step_dir / 'pca_population_distribution.png')
    plt.close(fig)


@torch.no_grad()
def make_pca_population_plot_per_subject(step_dir: Path,
                                         true_list: list[torch.Tensor],
                                         pred_list: list[torch.Tensor],
                                         mask_list: list[torch.Tensor],
                                         subjects: list[str],
                                         title: str) -> None:
    def _dbg(msg: str):
        try:
            print(f"[eval_pca_by_subj] {msg}")
        except Exception:
            pass
    if len(true_list) == 0:
        _dbg('no samples')
        return
    try:
        X_true = torch.stack([t.to(torch.float32) for t in true_list], dim=0)  # (S,N)
        X_pred = torch.stack([p.to(torch.float32) for p in pred_list], dim=0)  # (S,N)
        M = torch.stack([m.to(torch.bool) for m in mask_list], dim=0)          # (S,N)
    except Exception as e:
        _dbg(f'stack error: {e}')
        return
    S, N = int(X_true.shape[0]), int(X_true.shape[1])
    if S < 2 or N < 2:
        _dbg(f'insufficient S={S} N={N}')
        return
    counts = M.sum(dim=0)
    thresh = max(2, int(0.2 * S))
    keep_cols = counts >= thresh
    if not bool(keep_cols.any().item()):
        _dbg('no kept cols')
        return
    Xtrue = X_true[:, keep_cols]
    Xpred = X_pred[:, keep_cols]
    Mkeep = M[:, keep_cols]
    K = int(Xtrue.shape[1])
    if K < 2:
        _dbg(f'K<2: {K}')
        return
    denom = Mkeep.sum(dim=0).clamp_min(1).to(Xtrue.dtype)
    col_mean = (Xtrue.masked_fill(~Mkeep, 0.0).sum(dim=0) / denom)
    Xtrue_imp = torch.where(Mkeep, Xtrue, col_mean.unsqueeze(0))
    Xpred_imp = torch.where(Mkeep, Xpred, col_mean.unsqueeze(0))
    mu = col_mean.unsqueeze(0)
    Xtrue_c = (Xtrue_imp - mu).cpu().numpy()
    Xpred_c = (Xpred_imp - mu).cpu().numpy()
    # PCA on true (global)
    try:
        U, Svals, VT = np.linalg.svd(Xtrue_c, full_matrices=False)
    except Exception as e:
        _dbg(f'svd error: {e}')
        return
    comps = VT[:2].T
    Zt = Xtrue_c @ comps
    Zp = Xpred_c @ comps
    lo = np.minimum(Zt.min(axis=0), Zp.min(axis=0))
    hi = np.maximum(Zt.max(axis=0), Zp.max(axis=0))
    pad = 0.05 * (hi - lo + 1e-8)
    lo -= pad; hi += pad
    # Grid layout
    import math
    S_eff = Zt.shape[0]
    cols = min(5, max(1, int(math.ceil(math.sqrt(S_eff)))))
    rows = int(math.ceil(S_eff / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows*cols):
        r = i // cols; c = i % cols
        ax = axes[r, c]
        if i < S_eff:
            ax.scatter(Zt[i, 0], Zt[i, 1], s=14, color='tab:blue', label='true')
            ax.scatter(Zp[i, 0], Zp[i, 1], s=14, color='tab:orange', label='pred')
            ax.set_title(subjects[i] if i < len(subjects) else f'subj_{i+1}', fontsize=8)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        if r == rows-1:
            ax.set_xlabel('PC1')
        if c == 0:
            ax.set_ylabel('PC2')
        if i == 0:
            ax.legend(fontsize=7, loc='best')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(step_dir / 'pca_population_distribution_by_subject.png')
    plt.close(fig)


@torch.no_grad()
def make_laststep_pca_grid_per_subject(step_dir: Path,
                                       subj_to_true: dict,
                                       subj_to_pred: dict,
                                       subj_to_masks: dict) -> None:
    names = list(subj_to_true.keys())
    if len(names) == 0:
        print('[eval_pca_laststep] no subjects collected')
        return
    import math
    S = len(names)
    cols = min(5, max(1, int(math.ceil(math.sqrt(S)))))
    rows = int(math.ceil(S / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3.6*rows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(rows, cols)
    for i, subj in enumerate(names):
        r = i // cols; c = i % cols
        ax = axes[r, c]
        tl = subj_to_true.get(subj, [])
        pl = subj_to_pred.get(subj, [])
        ml = subj_to_masks.get(subj, [])
        ax.axis('off')
        try:
            if len(tl) < 2:
                ax.set_title(f'{subj} (n<2)', fontsize=8)
                continue
            X_true = torch.stack([t.to(torch.float32) for t in tl], dim=0)
            X_pred = torch.stack([p.to(torch.float32) for p in pl], dim=0) if len(pl)==len(tl) else None
            M = torch.stack([m.to(torch.bool) for m in ml], dim=0) if len(ml)==len(tl) else torch.ones_like(X_true, dtype=torch.bool)
            keep = M.all(dim=0)
            if not bool(keep.any().item()):
                ax.set_title(f'{subj} (no common)', fontsize=8)
                continue
            Xt = X_true[:, keep]
            Xp = X_pred[:, keep] if X_pred is not None else None
            if int(Xt.shape[1]) < 2:
                ax.set_title(f'{subj} (K<2)', fontsize=8)
                continue
            mu = Xt.mean(dim=0, keepdim=True)
            Xt_c = (Xt - mu).cpu().numpy()
            Xp_c = (Xp - mu).cpu().numpy() if Xp is not None else None
            U, Svals, VT = np.linalg.svd(Xt_c, full_matrices=False)
            comps = VT[:2].T
            Zt = Xt_c @ comps
            ax.scatter(Zt[:,0], Zt[:,1], s=10, alpha=0.7, color='tab:blue', label='true')
            if Xp_c is not None:
                Zp = Xp_c @ comps
                ax.scatter(Zp[:,0], Zp[:,1], s=10, alpha=0.7, color='tab:orange', label='pred')
            ax.set_title(subj, fontsize=8)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            if i == 0:
                ax.legend(fontsize=7, loc='best')
            ax.axis('on')
        except Exception as e:
            ax.set_title(f'{subj} (err)', fontsize=8)
    for j in range(len(names), rows*cols):
        axes[j//cols, j%cols].axis('off')
    fig.suptitle('Last-step PCA per subject (true in blue, pred in orange)')
    fig.tight_layout()
    fig.savefig(step_dir / 'pca_laststep_per_subject.png')
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
                           sampling_rate_hz: float = 3.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns:
    - pred rates: (Tf, N)
    - spike counts for context window: (Tc,)
    - spike counts for predicted future: (Tf,)
    """
    model.eval()
    context = init_context.clone()  # (1, Tc, N)
    Lc = int(init_context.shape[1])
    # Pre-sample spike masks for the initial context as 0/1 floats (deterministic under Bernoulli when re-used)
    sr_f = float(sampling_rate_hz)
    spike_prob_init = 1.0 - torch.exp(-context.to(torch.float32) / sr_f)
    spike_prob_init = torch.nan_to_num(spike_prob_init, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    spike_mask_ctx = torch.bernoulli(spike_prob_init).to(torch.float32)  # (1, Tc, N) 0/1 floats
    preds = []
    # Precompute context spike counts per timestep (sum over neurons)
    ctx_counts = spike_mask_ctx[:, :Lc, :].sum(dim=2).to(torch.float32)[0]  # (Tc,)
    pred_counts: list[torch.Tensor] = []
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
        # record spike count for this predicted step
        pred_counts.append(next_mask[0, 0, :].sum().to(torch.float32))
    return torch.cat(preds, dim=0), ctx_counts, torch.stack(pred_counts, dim=0)  # (Tf,N), (Tc,), (Tf,)


def main():
    ap = argparse.ArgumentParser(description='Evaluate GBM (validation + AR) from an experiment directory')
    ap.add_argument('--exp_dir', type=str, required=True, help='Path to experiments/gbm2/<YYYYmmdd_HHMMSS>')
    ap.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path; defaults to <exp_dir>/checkpoints/best_model.pth')
    ap.add_argument('--val_batches', type=int, default=None, help='Number of validation batches to average (defaults to cfg.training.val_sample_batches or all)')
    ap.add_argument('--horizon', type=int, default=64, help='AR prediction horizon steps for visualization')
    ap.add_argument('--num_ars', type=int, default=16, help='Number of unique AR rollouts to run and visualize')
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
    vb_loss_count = 0
    final_ctx = None
    final_truth = None
    final_pred = None
    first_ctx_counts = None
    first_true_counts = None
    first_pred_counts = None
    # Accumulators for multiple AR runs
    num_ars = int(args.num_ars)
    ar_contexts: list[torch.Tensor] = []
    ar_truths: list[torch.Tensor] = []
    ar_preds: list[torch.Tensor] = []
    ar_ctx_counts: list[torch.Tensor] = []
    ar_pred_counts: list[torch.Tensor] = []
    ar_true_counts: list[torch.Tensor] = []
    # Accumulators for ensembled scatter/density
    scat_true_last: list[torch.Tensor] = []
    scat_pred_last: list[torch.Tensor] = []
    scat_mu_last: list[torch.Tensor] = []
    scat_sigma_last: list[torch.Tensor] = []
    scat_mask_last: list[torch.Tensor] = []
    scat_subjects: list[str] = []
    # Accumulators for per-subject temporal PCA (use full teacher-forced window per subject)
    subj_true_mats: list[torch.Tensor] = []   # each: (Lm1_use,N)
    subj_pred_mats: list[torch.Tensor] = []   # each: (Lm1_use,N)
    subj_masks: list[torch.Tensor] = []       # each: (N,)
    subj_names: list[str] = []
    # Accumulators for subject-wise last-step PCA (across all batches)
    subj_last_true: dict[str, list[torch.Tensor]] = {}
    subj_last_pred: dict[str, list[torch.Tensor]] = {}
    subj_last_masks: dict[str, list[torch.Tensor]] = {}

    val_batches_limit = args.val_batches if args.val_batches is not None else int(cfg['training'].get('val_sample_batches') or 0)
    if val_batches_limit <= 0:
        val_batches_limit = 1_000_000_000  # effectively all

    sr = float(cfg['training'].get('sampling_rate_hz', 3.0))
    orig_seq_len = int(cfg['training'].get('sequence_length', 48))
    Lm1_orig = max(1, orig_seq_len - 1)

    # Accumulators for mean next-step loss vs sequence index across validation batches
    per_index_sum = torch.zeros(Lm1_orig, dtype=torch.float64)
    per_index_count = torch.zeros(Lm1_orig, dtype=torch.float64)

    with torch.no_grad():
        # Track one AR per subject/file
        seen_subjects: set[str] = set()
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

            # Accumulate per-sequence-index masked loss (sum and count) for averaging later
            per_pos_sum = (nll * mask_exp).sum(dim=2).to(torch.float64)   # (B, Lm1_use)
            per_pos_cnt = (mask_exp.sum(dim=2)).to(torch.float64)         # (B, Lm1_use)
            per_index_sum[:Lm1_use] += per_pos_sum.sum(dim=0).cpu()
            per_index_count[:Lm1_use] += per_pos_cnt.sum(dim=0).cpu()
            # Count only up to the requested number of validation batches for loss
            if vb_loss_count < val_batches_limit:
                total += float(vloss.detach().cpu().item())
                vb_loss_count += 1

            # Per-subject LAST-STEP accumulation for PCA (do this for all batches)
            file_paths = batch.get('file_path', [])
            B = int(x_in_full.shape[0])
            for bi in range(B):
                try:
                    from pathlib import Path as _P
                    subj_id = _P(file_paths[bi]).name if file_paths else f"subj"
                except Exception:
                    subj_id = f"subj"
                tl = x_tg[bi, -1, :].detach().cpu()
                # median prediction for stability (teacher-forced last step)
                pred_med = lognormal_rate_median(mu[bi:bi+1, -1:, :].detach().cpu(), raw_log_sigma[bi:bi+1, -1:, :].detach().cpu())[0,0,:]
                mk = (mask[bi] != 0).detach().cpu()
                subj_last_true.setdefault(subj_id, []).append(tl)
                subj_last_pred.setdefault(subj_id, []).append(pred_med)
                subj_last_masks.setdefault(subj_id, []).append(mk)

            # Collect up to num_ars unique-subject AR visualizations
            if len(ar_contexts) < num_ars:
                B, Lm1_full, N = x_in_full.shape
                Tf_all = min(horizon, max(1, Lm1_full - Lm1_orig)) if Lm1_full > Lm1_orig else min(horizon, Lm1_full)
                Tc = Lm1_orig
                Tf_all = max(1, min(Tf_all, max(1, Lm1_full - Tc)))

                # Iterate within batch to find first sample from a new subject
                for bi in range(B):
                    if len(ar_contexts) >= num_ars:
                        break
                    try:
                        fp = file_paths[bi]
                    except Exception:
                        fp = None
                    if fp is not None and fp in seen_subjects:
                        continue

                    # Mark subject as seen
                    if fp is not None:
                        seen_subjects.add(fp)

                    # Prepare sample-specific tensors
                    init_context = x_in_full[bi:bi+1, :Tc, :].to(torch.float32)
                    stim_full = stim[bi:bi+1, :Tc+Tf_all, :]
                    pred_future, ctx_spike_counts, pred_spike_counts = autoregressive_rollout(
                        model, init_context, stim_full,
                        positions[bi:bi+1], mask[bi:bi+1], neuron_ids[bi:bi+1],
                        lam[bi:bi+1] if lam.numel()>0 else None,
                        las[bi:bi+1] if las.numel()>0 else None,
                        device, Tf_all, sampling_rate_hz=sr)

                    # Save the first AR for legacy single-run plots
                    if final_ctx is None:
                        final_ctx = init_context[0].cpu()
                        final_truth = x_in_full[bi, Tc:Tc+Tf_all, :].cpu()
                        final_pred = pred_future.cpu()
                        first_ctx_counts = ctx_spike_counts.cpu()
                        # Compute true-future spike counts for first AR
                        prob_true_first = 1.0 - torch.exp(-final_truth.to(torch.float32) / sr)
                        prob_true_first = torch.nan_to_num(prob_true_first, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                        true_future_mask_first = torch.bernoulli(prob_true_first).to(torch.float32)
                        first_true_counts = true_future_mask_first.sum(dim=1).cpu()
                        first_pred_counts = pred_spike_counts.cpu()

                    # Accumulate for stacked over-time plots
                    ar_contexts.append(init_context[0].cpu())
                    ar_truths.append(x_in_full[bi, Tc:Tc+Tf_all, :].cpu())
                    ar_preds.append(pred_future.cpu())
                    ar_ctx_counts.append(ctx_spike_counts.cpu())
                    # Compute true-future spike counts using Bernoulli mask from truth rates
                    prob_true = 1.0 - torch.exp(-ar_truths[-1].to(torch.float32) / sr)
                    prob_true = torch.nan_to_num(prob_true, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                    true_future_mask = torch.bernoulli(prob_true).to(torch.float32)
                    true_spike_counts = true_future_mask.sum(dim=1)  # (Tf,)
                    ar_true_counts.append(true_spike_counts.cpu())
                    ar_pred_counts.append(pred_spike_counts.cpu())

                    # Accumulate for ensembled scatter/density (last teacher-forced step)
                    true_last = x_tg[bi, -1, :]  # (N,)
                    pred_last = sample_lognormal(mu[bi:bi+1, -1:, :], raw_log_sigma[bi:bi+1, -1:, :])[0, 0, :]
                    mu_last = mu[bi, -1, :].to(torch.float32)
                    sigma_last = (F.softplus(raw_log_sigma[bi, -1, :].to(torch.float32)) + 1e-6)
                    scat_true_last.append(true_last.detach().cpu())
                    scat_pred_last.append(pred_last.detach().cpu())
                    scat_mu_last.append(mu_last.detach().cpu())
                    scat_sigma_last.append(sigma_last.detach().cpu())
                    scat_mask_last.append((mask[bi] != 0).detach().cpu())
                    try:
                        from pathlib import Path as _P
                        subj = _P(fp).name if fp is not None else f"subj_{len(scat_subjects)+1}"
                    except Exception:
                        subj = f"subj_{len(scat_subjects)+1}"
                    scat_subjects.append(subj)

                    # Per-subject temporal PCA data (use all teacher-forced next steps)
                    subj_true = x_tg[bi, :Lm1_use, :].detach().cpu()        # (Lm1_use,N)
                    # Predict next-step rates per time using median for stability
                    subj_pred = lognormal_rate_median(mu[bi, :Lm1_use, :].detach().cpu(), raw_log_sigma[bi, :Lm1_use, :].detach().cpu())
                    subj_mask = (mask[bi] != 0).detach().cpu()              # (N,)
                    subj_true_mats.append(subj_true)
                    subj_pred_mats.append(subj_pred)
                    subj_masks.append(subj_mask)
                    subj_names.append(subj)

            # Break when both: (1) collected requested ARs and (2) finished requested val loss batches
            if (vb_loss_count >= val_batches_limit) and (len(ar_contexts) >= num_ars):
                break

    val_loss = total / max(1, vb_loss_count)
    with open(logs_dir / 'eval_loss.txt', 'a') as f:
        f.write(f"val {val_loss:.6f} over {vb_loss_count} batches\n")

    # Plot AR if we captured at least one
    step_dir = plots_dir / 'eval_ar'
    if final_ctx is not None and final_truth is not None and final_pred is not None:
        make_validation_plots(step_dir, final_ctx, final_truth, final_pred, title=f"Eval autoreg 16 neurons")
        make_mean_rate_plot(step_dir, final_ctx, final_truth, final_pred, title=f"Eval mean rate over time (first AR)")
        make_median_rate_plot(step_dir, final_ctx, final_truth, final_pred, title=f"Eval median rate over time (first AR)")
        if first_ctx_counts is not None and first_true_counts is not None and first_pred_counts is not None:
            make_spike_counts_plot(step_dir, first_ctx_counts, first_true_counts, first_pred_counts, title=f"Eval spike counts over time (first AR)")
    # Stacked over-time plots across multiple AR runs
    if len(ar_contexts) > 0:
        make_mean_rate_plots_stacked(step_dir, ar_contexts, ar_truths, ar_preds, title=f"Eval mean rate over time (stacked {len(ar_contexts)} ARs)")
        make_median_rate_plots_stacked(step_dir, ar_contexts, ar_truths, ar_preds, title=f"Eval median rate over time (stacked {len(ar_contexts)} ARs)")
        make_spike_counts_plots_stacked(step_dir, ar_ctx_counts, ar_true_counts, ar_pred_counts, title=f"Eval spike counts over time (stacked {len(ar_contexts)} ARs)")

        # Ensembled scatter/density plots (concatenate across runs)
        true_all = torch.cat(scat_true_last, dim=0) if len(scat_true_last) else torch.empty(0)
        pred_all = torch.cat(scat_pred_last, dim=0) if len(scat_pred_last) else torch.empty(0)
        mu_all = torch.cat(scat_mu_last, dim=0) if len(scat_mu_last) else torch.empty(0)
        sigma_all = torch.cat(scat_sigma_last, dim=0) if len(scat_sigma_last) else torch.empty(0)
        mask_all = torch.cat(scat_mask_last, dim=0) if len(scat_mask_last) else torch.empty(0, dtype=torch.bool)
        if true_all.numel() > 0:
            make_next_step_scatter(step_dir, true_all, pred_all, mask_all, title=f"Eval next-step scatter (ensembled {len(scat_true_last)} ARs)")
            make_mu_sigma_scatter(step_dir, mu_all, sigma_all, true_all, mask_all, title=f"Eval mu vs sigma (ensembled {len(scat_true_last)} ARs)")
            make_log_rate_scatter(step_dir, true_all, pred_all, mask_all, title=f"Eval log-rate scatter (ensembled {len(scat_true_last)} ARs)")
            make_log_true_vs_log_sigma_scatter(step_dir, true_all, sigma_all, mask_all, title=f"Eval log(true) vs log(sigma) (ensembled {len(scat_true_last)} ARs)")
            make_log_true_vs_log_mu_scatter(step_dir, true_all, mu_all, mask_all, title=f"Eval log(true) vs mu (log-rate) (ensembled {len(scat_true_last)} ARs)")
            pass

    # New: Subject-wise PCA using last-step vectors across all batches
    try:
        make_laststep_pca_grid_per_subject(step_dir, subj_last_true, subj_last_pred, subj_last_masks)
    except Exception as e:
        try:
            print(f"[eval_pca_laststep] exception: {e}")
        except Exception:
            pass

    # Plot mean next-step loss vs sequence index (more context to the right)
    if per_index_count.sum() > 0:
        per_idx_mean = (per_index_sum / per_index_count.clamp_min(1.0)).cpu().numpy()
        make_per_index_loss_plot(step_dir, per_idx_mean, title='Mean next-step NLL vs sequence index')

    print(f"Eval complete. Val loss: {val_loss:.6f} over {vb_loss_count} batches. Outputs in: {exp_dir}")


if __name__ == '__main__':
    main()


