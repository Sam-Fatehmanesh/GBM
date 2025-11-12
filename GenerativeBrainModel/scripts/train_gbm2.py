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
from GenerativeBrainModel.utils.lognormal import (
    lognormal_nll,
    lognormal_rate_median,
    sample_lognormal,
)
from GenerativeBrainModel.utils.debug import assert_no_nan, debug_enabled


def create_default_config() -> Dict[str, Any]:
    return {
        "experiment": {"name": "gbm2_neural_training"},
        "data": {
            "data_dir": "processed_spike_voxels_2018",
            "use_cache": True,
        },
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "d_stimuli": None,  # infer from data
            "num_neurons_total": 4_000_000,  # capacity of neuron embedding table (>= max distinct IDs)
            "cov_rank": 32,  # rank of low-rank correlation factors U
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 20,
            "learning_rate": 5e-4,  # AdamW for non-body
            "muon_lr": 2e-2,  # Muon for attention body
            "weight_decay": 1e-4,
            "sequence_length": 12,
            "stride": 3,
            "num_workers": 0,
            "pin_memory": False,
            "test_split_fraction": 0.1,
            "use_gpu": True,
            "compile_model": False,
            "validation_frequency": 8,
            "val_sample_batches": 32,
            "gradient_clip_norm": 1.0,
            "seed": 42,
            "plots_dir": "experiments/gbm2/plots",
            "sampling_rate_hz": 3.0,
            # Left-censored LogNormal threshold for loss (rates below are treated as "≤ r_min")
            # Set to 0.0 to disable; e.g., 1e-2 to ignore exact values below 0.01
            "loss_r_min": 0.0,
            # Gaussian copula dependence term (correlation-only)
            "copula_weight": 0.05,
            # Start the copula term at floor*copula_weight and ramp linearly to copula_weight by run end
            "copula_warmup_floor": 1e-3,
            # Jitter added to correlation base (R derived from L = (1+jitter)I + U U^T)
            "copula_jitter": 1e-3,
            # Optional burn-in steps where μ,σ are detached in copula term to stabilize early learning
            "copula_detach_burnin_steps": 0,
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
    """Free cached CUDA memory at safe points to reduce VRAM pressure.
    Called only outside hot training loops to avoid perf impact.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass


def build_optimizer(model: GBM, cfg: Dict[str, Any]) -> optim.Optimizer:
    try:
        from muon import MuonWithAuxAdam
    except ImportError as e:
        raise ImportError(
            "Muon optimizer not found. Install: pip install git+https://github.com/KellerJordan/Muon"
        ) from e

    # Body vs non-body
    hidden_weights = [
        p for p in model.body.parameters() if p.ndim >= 2 and p.requires_grad
    ]
    hidden_gains_biases = [
        p for p in model.body.parameters() if p.ndim < 2 and p.requires_grad
    ]
    nonhidden_params = []
    for m in model.embed.values():
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]
    for m in model.head.values():
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]

    muon_lr = float(cfg.get("muon_lr", 2e-2))
    muon_wd = float(cfg.get("weight_decay", 1e-4))
    adamw_lr = float(cfg.get("learning_rate", 5e-4))
    adamw_wd = float(cfg.get("weight_decay", 1e-4))
    adamw_betas = (0.9, 0.95)

    param_groups = []
    if hidden_weights:
        param_groups.append(
            dict(params=hidden_weights, use_muon=True, lr=muon_lr, weight_decay=muon_wd)
        )
    if hidden_gains_biases or nonhidden_params:
        param_groups.append(
            dict(
                params=hidden_gains_biases + nonhidden_params,
                use_muon=False,
                lr=adamw_lr,
                betas=adamw_betas,
                weight_decay=adamw_wd,
            )
        )
    return MuonWithAuxAdam(param_groups)


def _copula_lowrank_nll(
    q: torch.Tensor, factors: torch.Tensor, mask_exp: torch.Tensor, jitter: float = 1e-3
) -> torch.Tensor:
    """Gaussian copula dependence NLL with low-rank correlation.
    q: (B, L-1, N) standardized residuals; factors U: (B, L-1, N, r)
    mask_exp: (B, L-1, N) float mask in {0,1}. Returns mean over (B, L-1).
    R := S^{-1/2} L S^{-1/2} with L = a I + U U^T, a = 1 + jitter, s_i = diag(L).
    Loss per (B,t): 0.5 [logdet R + q^T (R^{-1} - I) q], normalized by N_eff.
    """
    dtype = torch.float32
    q = q.to(dtype)
    U = factors.to(dtype)
    mask_exp = mask_exp.to(dtype)
    a = 1.0 + float(jitter)

    # Apply mask
    q = q * mask_exp
    U = U * mask_exp[..., None]

    # s_i = a + ||U_i||^2
    row_norm2 = (U * U).sum(dim=-1)
    s = a + row_norm2
    s = torch.nan_to_num(s, nan=a, posinf=1e6, neginf=a)
    s_sqrt = s.sqrt()

    # Small r x r system: M = I + (1/a) U^T U
    UTU = torch.matmul(U.transpose(-1, -2), U)
    rdim = int(U.shape[-1])
    I_r = torch.eye(rdim, dtype=dtype, device=U.device).expand_as(UTU)
    M = 0.5 * (
        I_r + (1.0 / a) * UTU + (I_r + (1.0 / a) * UTU).transpose(-1, -2)
    )  # symm

    # Cholesky or eig fallback
    try:
        chol = torch.linalg.cholesky(M)
        logdetM = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1e-12)
        ).sum(dim=-1)
        Minv = torch.cholesky_inverse(chol)
    except Exception:
        w, Q = torch.linalg.eigh(M + 1e-4 * I_r)
        w = w.clamp_min(1e-8)
        logdetM = torch.log(w).sum(dim=-1)
        Minv = torch.matmul(
            Q, torch.matmul(torch.diag_embed(1.0 / w), Q.transpose(-1, -2))
        )

    # logdet R = logdet L - sum log s_i
    Ndim = q.shape[-1]
    logdetL = Ndim * math.log(a) + logdetM
    logdetR = logdetL - torch.log(s.clamp_min(1e-12)).sum(dim=-1)

    # Quadratic form
    y = s_sqrt * q
    y2 = (y * y).sum(dim=-1)
    q2 = (q * q).sum(dim=-1)
    v = torch.matmul(U.transpose(-1, -2), y[..., None])
    vTMinvV = (v.squeeze(-1) * torch.matmul(Minv, v).squeeze(-1)).sum(dim=-1)
    yTLinvY = (1.0 / a) * y2 - (1.0 / (a * a)) * vTMinvV
    quad = yTLinvY - q2

    nll = 0.5 * (logdetR + quad)
    N_eff = mask_exp.sum(dim=-1).clamp_min(1.0)
    return (nll / N_eff).mean()


@torch.no_grad()
def make_validation_plots(
    step_dir: Path,
    context_truth: torch.Tensor,
    future_truth: torch.Tensor,
    future_pred: torch.Tensor,
    title: str,
    max_neurons: int = 16,
) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    K = min(max_neurons, N)
    idx = torch.linspace(0, N - 1, K).round().long().tolist()
    fig, axes = plt.subplots(K, 1, figsize=(10, 2 * K), sharex=True)
    axes = np.atleast_1d(axes)
    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)
    for r, ax in enumerate(axes):
        j = idx[r]
        ax.plot(
            x_context,
            context_truth[:, j].cpu().numpy(),
            color="gray",
            lw=1.0,
            label="context truth",
        )
        ax.plot(
            x_future,
            future_truth[:, j].cpu().numpy(),
            color="tab:blue",
            lw=1.0,
            label="future truth",
        )
        ax.plot(
            x_future,
            future_pred[:, j].cpu().numpy(),
            color="tab:orange",
            lw=1.0,
            label="future pred",
        )
        ax.set_ylabel(f"n{j}")
        if r == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    out = step_dir / "val_autoreg_16neurons.png"
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_mean_rate_plot(
    step_dir: Path,
    context_truth: torch.Tensor,
    future_truth: torch.Tensor,
    future_pred: torch.Tensor,
    title: str,
) -> None:
    """Plot mean rate over neurons vs time for context, true future, and AR prediction."""
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    ctx_mean = (
        context_truth.to(torch.float32).mean(dim=1).cpu().numpy()
        if context_truth.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    fut_mean = (
        future_truth.to(torch.float32).mean(dim=1).cpu().numpy()
        if future_truth.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    pred_mean = (
        future_pred.to(torch.float32).mean(dim=1).cpu().numpy()
        if future_pred.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if ctx_mean.size:
        ax.plot(x_context, ctx_mean, color="gray", lw=1.5, label="context mean rate")
    if fut_mean.size:
        ax.plot(x_future, fut_mean, color="tab:blue", lw=1.5, label="future truth mean")
    if pred_mean.size:
        ax.plot(
            x_future, pred_mean, color="tab:orange", lw=1.5, label="future pred mean"
        )
    ax.set_xlabel("time")
    ax.set_ylabel("mean rate")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = step_dir / "mean_rate_over_time.png"
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def make_median_rate_plot(
    step_dir: Path,
    context_truth: torch.Tensor,
    future_truth: torch.Tensor,
    future_pred: torch.Tensor,
    title: str,
) -> None:
    """Plot median rate over neurons vs time for context, true future, and AR prediction."""
    step_dir.mkdir(parents=True, exist_ok=True)
    Tc, N = context_truth.shape
    Tf = future_truth.shape[0]
    ctx_med = (
        context_truth.to(torch.float32).median(dim=1).values.cpu().numpy()
        if context_truth.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    fut_med = (
        future_truth.to(torch.float32).median(dim=1).values.cpu().numpy()
        if future_truth.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    pred_med = (
        future_pred.to(torch.float32).median(dim=1).values.cpu().numpy()
        if future_pred.numel() > 0
        else np.zeros((0,), dtype=np.float32)
    )
    x_context = np.arange(Tc)
    x_future = np.arange(Tc, Tc + Tf)
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if ctx_med.size:
        ax.plot(x_context, ctx_med, color="gray", lw=1.5, label="context median rate")
    if fut_med.size:
        ax.plot(
            x_future, fut_med, color="tab:blue", lw=1.5, label="future truth median"
        )
    if pred_med.size:
        ax.plot(
            x_future, pred_med, color="tab:orange", lw=1.5, label="future pred median"
        )
    ax.set_xlabel("time")
    ax.set_ylabel("median rate")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = step_dir / "median_rate_over_time.png"
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def autoregressive_rollout(
    model: GBM,
    init_context: torch.Tensor,  # (1, Tc, N) rates
    stim_full: torch.Tensor,  # (1, Tc+Tf, K)
    positions: torch.Tensor,  # (1, N, 3)
    neuron_mask: torch.Tensor,  # (1, N)
    neuron_ids: torch.Tensor,  # (1, N)
    lam: torch.Tensor | None,  # (1, N)
    las: torch.Tensor | None,  # (1, N)
    device: torch.device,
    Tf: int,
    sampling_rate_hz: float = 3.0,
) -> torch.Tensor:  # returns (Tf, N) rates
    model.eval()
    context = init_context.clone()  # (1, Tc, N)
    Lc = int(init_context.shape[1])
    preds = []
    eps = 1e-7
    # Pre-sample spike mask for initial context (0/1 floats)
    sr_f = float(sampling_rate_hz)
    prob_init = 1.0 - torch.exp(-context.to(torch.float32) / sr_f)
    prob_init = torch.nan_to_num(prob_init, nan=0.0, posinf=1.0, neginf=0.0).clamp_(
        0.0, 1.0
    )
    spike_mask_ctx = torch.bernoulli(prob_init).to(torch.float32)  # (1,Tc,N)
    for t in range(Tf):
        # Use sliding window of last Lc steps
        x_in = context[:, -Lc:, :]  # (1,Lc,N)
        x_log = torch.log(x_in.clamp_min(eps))
        if (
            (lam is not None)
            and (lam.numel() > 0)
            and (las is not None)
            and (las.numel() > 0)
        ):
            lam_e = lam[:, None, :].to(dtype=x_log.dtype)
            las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
            x_in_z = (x_log - lam_e) / las_e
        else:
            x_in_z = x_log
        # Align stimuli to the sliding window
        stim_step = stim_full[:, (context.shape[1] - Lc) : (context.shape[1]), :]
        if device.type == "cuda":
            x_in_z = x_in_z.to(torch.bfloat16)
            stim_step = stim_step.to(torch.bfloat16)
        # Use pre-sampled spike mask window (0/1 floats) to make routing deterministic
        spike_probs_window = spike_mask_ctx[:, -Lc:, :]
        mu, raw_log_sigma, _, _ = model(
            x_in_z,
            stim_step,
            positions,
            neuron_mask,
            neuron_ids,
            spike_probs_window,
            get_logits=True,
            input_log_rates=True,
        )
        # Sample from LogNormal for stochastic rollout
        mu_last = mu[:, -1:, :]
        sig_last = raw_log_sigma[:, -1:, :]
        samp = sample_lognormal(mu_last, sig_last)  # (1,1,N)
        preds.append(samp[:, 0, :].to(torch.float32))
        # Append prediction and its pre-sampled spike mask
        context = torch.cat([context, samp.to(context.dtype)], dim=1)
        next_prob = 1.0 - torch.exp(-samp.to(torch.float32) / sr_f)
        next_prob = torch.nan_to_num(next_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp_(
            0.0, 1.0
        )
        next_mask = torch.bernoulli(next_prob).to(torch.float32)
        spike_mask_ctx = torch.cat([spike_mask_ctx, next_mask], dim=1)
    return torch.cat(preds, dim=0)  # (Tf,N)


def main():
    ap = argparse.ArgumentParser(
        description="Train GBM (clean) with LogNormal NLL and Muon optimizer"
    )
    ap.add_argument(
        "--config", type=str, default=None, help="YAML config for overrides"
    )
    args = ap.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, "r") as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    device = torch.device(
        "cuda" if (cfg["training"]["use_gpu"] and torch.cuda.is_available()) else "cpu"
    )
    set_seeds(int(cfg["training"]["seed"]))

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

    # Data (with global unique neuron IDs)
    train_loader, val_loader, _, _, unique_neuron_ids = create_dataloaders(cfg)

    # Model
    try:
        sample = next(iter(train_loader))
        d_stimuli = int(sample["stimulus"].shape[-1])
        max_n = int(sample["positions"].shape[1])
    except Exception:
        d_stimuli = cfg["model"].get("d_stimuli") or 1
        max_n = 100_000

    model = GBM(
        d_model=cfg["model"]["d_model"],
        d_stimuli=d_stimuli,
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        num_neurons_total=int(cfg["model"]["num_neurons_total"]),
        global_neuron_ids=unique_neuron_ids,
        cov_rank=int(cfg["model"].get("cov_rank", 32)),
    ).to(device)

    # Run bf16 on CUDA
    if device.type == "cuda":
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
                assert_no_nan(grad, f"grad.{pname}")
                return grad

            return hook

        for n, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(_check_grad(n))

    if bool(cfg["training"].get("compile_model", False)):
        try:
            model = torch.compile(model, dynamic=True)
        except Exception:
            pass

    optimizer = build_optimizer(model, cfg["training"])

    # Per-run directory under experiments/gbm2/<YYYYmmdd_HHMMSS>
    base_dir = Path("experiments/gbm2")
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = run_dir / "plots"
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    for p in (plots_dir, ckpt_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    # Save resolved config for reproducibility
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)
    # Write a simple architecture summary
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(run_dir / "architecture.txt", "w") as f:
            f.write(
                f"Total params: {total_params:,}\nTrainable: {trainable_params:,}\n\n"
            )
            f.write(str(model))
            f.write("\n")
    except Exception:
        pass

    # ---- Loss tracking state (for plotting) ----
    train_batch_losses: list[float] = []
    train_batch_ema: list[float] = []
    # Track EMA of TOTAL loss (independent + lambda*copula)
    ema_total_value: float | None = None
    val_points: list[tuple[int, float]] = []  # (global_step, val_loss)
    ema_beta: float = 0.98
    ema_value: float | None = None
    global_step: int = 0
    total_train_steps: int = int(cfg["training"]["num_epochs"]) * max(
        1, len(train_loader)
    )

    def _lambda_copula_for_step(step: int) -> float:
        base = float(cfg["training"].get("copula_weight", 0.0) or 0.0)
        if base <= 0.0:
            return 0.0
        floor = float(cfg["training"].get("copula_warmup_floor", 1e-3))
        floor = max(0.0, min(floor, 1.0))
        frac = min(1.0, max(0.0, step / max(1, total_train_steps)))
        # Linear ramp from floor to 1.0 over the full run
        scale = floor + (1.0 - floor) * frac
        return base * scale

    def _update_loss_plot():
        try:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                sharex=False,
                gridspec_kw={"height_ratios": [2, 1]},
            )
            ax0, ax1 = axes
            if train_batch_losses:
                xs = list(range(1, len(train_batch_losses) + 1))
                ax0.plot(
                    xs,
                    train_batch_losses,
                    color="tab:blue",
                    alpha=0.35,
                    lw=0.8,
                    label="batch loss",
                )
            if train_batch_ema:
                xs = list(range(1, len(train_batch_ema) + 1))
                ax0.plot(xs, train_batch_ema, color="tab:orange", lw=1.5, label="EMA")
            ax0.set_title("Training loss (batch + EMA)")
            ax0.set_ylabel("loss")
            ax0.legend(loc="upper right", fontsize=8)
            # Val subplot
            if val_points:
                vx, vy = zip(*val_points)
                ax1.plot(
                    vx, vy, marker="o", linestyle="-", color="tab:green", lw=1.0, ms=3
                )
            ax1.set_title("Validation loss (per validation event)")
            ax1.set_xlabel("global step")
            ax1.set_ylabel("val loss")
            fig.tight_layout()
            fig.savefig(plots_dir / "loss_curves.png")
            plt.close(fig)
        except Exception:
            pass

    def train_or_val_loop(loader, epoch: int, train: bool) -> float:
        nonlocal global_step, ema_value, ema_total_value
        total_loss = 0.0
        count = 0
        mdl = model.train() if train else model.eval()
        # Intra-epoch validation triggers (evenly spaced + step 4)
        val_freq_local = int(cfg["training"].get("validation_frequency") or 0)
        val_sample_batches_local = int(cfg["training"].get("val_sample_batches") or 1)
        triggers: set[int] = set()
        total_steps = len(loader)
        if train and val_freq_local > 0 and total_steps > 0:
            for j in range(1, val_freq_local + 1):
                step = max(
                    1, min(total_steps, round(j * total_steps / (val_freq_local + 1)))
                )
                triggers.add(int(step))
            if total_steps >= 4:
                triggers.add(4)
        pbar = tqdm(loader, desc=("Train" if train else "Val"))
        for batch in pbar:
            spikes = batch["spikes"].to(device)  # (B, L, N) rates
            positions = batch["positions"].to(device)  # (B, N, 3)
            mask = batch["neuron_mask"].to(device)  # (B, N)
            stim = batch["stimulus"].to(device)  # (B, L, K)
            neuron_ids = batch["neuron_ids"].to(device)  # (B, N)
            lam = batch.get("log_activity_mean", torch.empty(0)).to(device)
            las = batch.get("log_activity_std", torch.empty(0)).to(device)
            sr = float(cfg["training"].get("sampling_rate_hz", 3.0))

            # Prepare autoregressive pairs
            x_in = spikes[:, :-1, :]  # (B, L-1, N)
            x_tg = spikes[:, 1:, :].float()
            if debug_enabled():
                assert_no_nan(x_in, "batch.x_in_raw")
                assert_no_nan(x_tg, "batch.x_tg_raw")
            # Guard against NaNs/Infs in raw data
            x_in = torch.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0)
            x_tg = torch.nan_to_num(x_tg, nan=0.0, posinf=0.0, neginf=0.0)
            stim_in = stim[:, :-1, :]

            # Spike probabilities from rates for attention routing
            spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
            # Robustify probs to [0,1] and finite before model use
            spike_probs = torch.nan_to_num(
                spike_probs, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            if debug_enabled():
                assert_no_nan(spike_probs, "routing.spike_probs")

            # z-normalize log input when stats available: z = (log(x+eps) - mean)/std
            eps = 1e-7
            x_log = torch.log(x_in.clamp_min(eps))
            if debug_enabled():
                assert_no_nan(x_log, "pre.zlog.x_log")
            if lam.numel() > 0 and las.numel() > 0:
                lam_e = lam[:, None, :].to(dtype=x_log.dtype)
                las_e = las[:, None, :].to(dtype=x_log.dtype).clamp_min(1e-6)
                x_in_z = (x_log - lam_e) / las_e
            else:
                x_in_z = x_log
            # Ensure inputs are finite before entering model
            x_in_z = torch.nan_to_num(x_in_z, nan=0.0, posinf=0.0, neginf=0.0)
            if debug_enabled():
                assert_no_nan(x_in_z, "pre.model.x_in_z")

            if device.type == "cuda":
                x_in_z = x_in_z.to(torch.bfloat16)
                stim_in = stim_in.to(torch.bfloat16)

            if train:
                optimizer.zero_grad()

            out = model(
                x_in_z,
                stim_in,
                positions,
                mask,
                neuron_ids,
                spike_probs,
                get_logits=True,
                input_log_rates=True,
                return_factors=True,
            )
            if isinstance(out, tuple) and len(out) == 5:
                mu, raw_log_sigma, _eta, _delta, factors = out
            else:
                mu, raw_log_sigma, _a, _b = out
                factors = None
            if debug_enabled():
                assert_no_nan(mu, "model.mu")
                assert_no_nan(raw_log_sigma, "model.raw_log_sigma")
            # Post-check for non-finite params
            if not torch.isfinite(mu).all() or not torch.isfinite(raw_log_sigma).all():
                mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
                raw_log_sigma = torch.nan_to_num(
                    raw_log_sigma, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Z-normalized target for loss in Normal(z) domain
            eps_n = 1e-7
            y_tg = torch.log(x_tg.clamp_min(eps_n))  # (B, L-1, N)
            if lam.numel() > 0 and las.numel() > 0:
                lam_e_loss = lam[:, None, :].to(dtype=y_tg.dtype)
                las_e_loss = las[:, None, :].to(dtype=y_tg.dtype).clamp_min(1e-6)
                z_tg = (y_tg - lam_e_loss) / las_e_loss
            else:
                z_tg = y_tg

            # Normal NLL on z_tg with optional left-censoring below r_min
            sigma_y = F.softplus(raw_log_sigma.to(torch.float32)) + 1e-6
            if lam.numel() > 0 and las.numel() > 0:
                mu_z = (
                    mu.to(torch.float32) - lam_e_loss.to(torch.float32)
                ) / las_e_loss.to(torch.float32)
                sigma_z = sigma_y / las_e_loss.to(torch.float32)
            else:
                mu_z = mu.to(torch.float32)
                sigma_z = sigma_y

            r_min = float(cfg["training"].get("loss_r_min", 0.0) or 0.0)
            if r_min > 0.0:
                y_min = math.log(r_min)
                if lam.numel() > 0 and las.numel() > 0:
                    z_min = (
                        torch.tensor(y_min, dtype=mu_z.dtype, device=mu_z.device)
                        - lam_e_loss.to(mu_z.dtype)
                    ) / las_e_loss.to(mu_z.dtype)
                else:
                    z_min = torch.tensor(y_min, dtype=mu_z.dtype, device=mu_z.device)
                is_cens = x_tg <= r_min
                z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                nll_pdf = (
                    0.5 * z_err.pow(2)
                    + torch.log(sigma_z)
                    + 0.5 * math.log(2.0 * math.pi)
                )
                alpha = (z_min - mu_z) / sigma_z
                # log CDF of standard normal using erf; clamp for stability
                log_cdf = torch.log(
                    (0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))).clamp_min(1e-12)
                )
                nll_cdf = -log_cdf
                nll = torch.where(is_cens, nll_cdf, nll_pdf)
            else:
                z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                nll = (
                    0.5 * z_err.pow(2)
                    + torch.log(sigma_z)
                    + 0.5 * math.log(2.0 * math.pi)
                )
            # Build mask for loss to match target shape
            mask_exp = mask[:, None, :].expand_as(x_tg).float()
            if mask_exp is not None:
                total_m = mask_exp.sum().clamp_min(1.0)
                loss = (nll * mask_exp).sum() / total_m
            else:
                loss = nll.mean()
            if debug_enabled():
                assert_no_nan(loss, "loss.nll")

            # Optional low-rank MVN coupling loss
            lambda_cop = _lambda_copula_for_step(global_step)
            if (lambda_cop != 0.0) and (factors is not None):
                burn = int(cfg["training"].get("copula_detach_burnin_steps", 0) or 0)
                mu_c = mu_z.detach() if global_step < burn else mu_z
                sig_c = sigma_z.detach() if global_step < burn else sigma_z
                q = (z_tg.to(torch.float32) - mu_c) / sig_c
                jitter = float(cfg["training"].get("copula_jitter", 1e-3))
                loss_cop = _copula_lowrank_nll(q, factors, mask_exp, jitter=jitter)
                loss_total = loss + lambda_cop * loss_cop
            else:
                loss_total = loss

            if train:
                loss_total.backward()
                if cfg["training"].get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["training"]["gradient_clip_norm"]
                    )
                optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            count += 1

            # Update trackers and tqdm
            global_step += 1
            batch_loss_f = float(loss.detach().cpu())
            train_batch_losses.append(batch_loss_f)
            ema_value = (
                (ema_value * ema_beta + batch_loss_f * (1.0 - ema_beta))
                if (ema_value is not None)
                else batch_loss_f
            )
            # Update EMA of total loss for logging
            batch_total_loss_f = float(loss_total.detach().cpu())
            ema_total_value = (
                (ema_total_value * ema_beta + batch_total_loss_f * (1.0 - ema_beta))
                if (ema_total_value is not None)
                else batch_total_loss_f
            )
            train_batch_ema.append(float(ema_value))
            pbar.set_postfix(
                {"loss": f"{batch_loss_f:.4f}", "ema": f"{float(ema_value):.4f}"}
            )

            # Lightweight intra-epoch validation at configured steps
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
                            spikes_v = vbatch["spikes"].to(device)
                            positions_v = vbatch["positions"].to(device)
                            mask_v = vbatch["neuron_mask"].to(device)
                            stim_v = vbatch["stimulus"].to(device)
                            neuron_ids_v = vbatch["neuron_ids"].to(device)
                            lam_v = vbatch.get("log_activity_mean", torch.empty(0)).to(
                                device
                            )
                            las_v = vbatch.get("log_activity_std", torch.empty(0)).to(
                                device
                            )
                            sr_v = float(cfg["training"].get("sampling_rate_hz", 3.0))

                            x_in_v = spikes_v[:, :-1, :]
                            x_tg_v = spikes_v[:, 1:, :].float()
                            stim_in_v = stim_v[:, :-1, :]

                            eps_v = 1e-7
                            x_log_v = torch.log(x_in_v.clamp_min(eps_v))
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                lam_e_v = lam_v[:, None, :].to(dtype=x_log_v.dtype)
                                las_e_v = (
                                    las_v[:, None, :]
                                    .to(dtype=x_log_v.dtype)
                                    .clamp_min(1e-6)
                                )
                                x_in_z_v = (x_log_v - lam_e_v) / las_e_v
                            else:
                                x_in_z_v = x_log_v

                            if device.type == "cuda":
                                x_in_z_v = x_in_z_v.to(torch.bfloat16)
                                stim_in_v = stim_in_v.to(torch.bfloat16)

                            spike_probs_v = 1.0 - torch.exp(-x_in_v.float() / sr_v)
                            out_v = model(
                                x_in_z_v,
                                stim_in_v,
                                positions_v,
                                mask_v,
                                neuron_ids_v,
                                spike_probs_v,
                                get_logits=True,
                                input_log_rates=True,
                                return_factors=True,
                            )
                            if isinstance(out_v, tuple) and len(out_v) == 5:
                                mu_v, raw_log_sigma_v, _e_v, _d_v, factors_v = out_v
                            else:
                                mu_v, raw_log_sigma_v, _aa_v, _bb_v = out_v
                                factors_v = None
                            # Z-normalized validation target with optional left-censored NLL
                            y_tg_v = torch.log(x_tg_v.clamp_min(1e-7))
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                lam_e_v2 = lam_v[:, None, :].to(dtype=y_tg_v.dtype)
                                las_e_v2 = (
                                    las_v[:, None, :]
                                    .to(dtype=y_tg_v.dtype)
                                    .clamp_min(1e-6)
                                )
                                z_tg_v = (y_tg_v - lam_e_v2) / las_e_v2
                            else:
                                z_tg_v = y_tg_v
                            sigma_y_v = (
                                F.softplus(raw_log_sigma_v.to(torch.float32)) + 1e-6
                            )
                            if lam_v.numel() > 0 and las_v.numel() > 0:
                                mu_z_v = (
                                    mu_v.to(torch.float32) - lam_e_v2.to(torch.float32)
                                ) / las_e_v2.to(torch.float32)
                                sigma_z_v = sigma_y_v / las_e_v2.to(torch.float32)
                            else:
                                mu_z_v = mu_v.to(torch.float32)
                                sigma_z_v = sigma_y_v
                            r_min_v = float(
                                cfg["training"].get("loss_r_min", 0.0) or 0.0
                            )
                            if r_min_v > 0.0:
                                y_min_v = math.log(r_min_v)
                                if lam_v.numel() > 0 and las_v.numel() > 0:
                                    z_min_v = (
                                        torch.tensor(
                                            y_min_v,
                                            dtype=mu_z_v.dtype,
                                            device=mu_z_v.device,
                                        )
                                        - lam_e_v2.to(mu_z_v.dtype)
                                    ) / las_e_v2.to(mu_z_v.dtype)
                                else:
                                    z_min_v = torch.tensor(
                                        y_min_v,
                                        dtype=mu_z_v.dtype,
                                        device=mu_z_v.device,
                                    )
                                is_cens_v = x_tg_v <= r_min_v
                                z_err_v = (
                                    z_tg_v.to(torch.float32) - mu_z_v
                                ) / sigma_z_v
                                nll_pdf_v = (
                                    0.5 * z_err_v.pow(2)
                                    + torch.log(sigma_z_v)
                                    + 0.5 * math.log(2.0 * math.pi)
                                )
                                alpha_v = (z_min_v - mu_z_v) / sigma_z_v
                                log_cdf_v = torch.log(
                                    (
                                        0.5
                                        * (1.0 + torch.erf(alpha_v / math.sqrt(2.0)))
                                    ).clamp_min(1e-12)
                                )
                                nll_cdf_v = -log_cdf_v
                                nll_v = torch.where(is_cens_v, nll_cdf_v, nll_pdf_v)
                            else:
                                z_err_v = (
                                    z_tg_v.to(torch.float32) - mu_z_v
                                ) / sigma_z_v
                                nll_v = (
                                    0.5 * z_err_v.pow(2)
                                    + torch.log(sigma_z_v)
                                    + 0.5 * math.log(2.0 * math.pi)
                                )
                            mask_exp_v = mask_v[:, None, :].expand_as(x_tg_v).float()
                            vloss = (
                                nll_v * mask_exp_v
                            ).sum() / mask_exp_v.sum().clamp_min(1.0)
                            lambda_cop_v = _lambda_copula_for_step(global_step)
                            if (lambda_cop_v != 0.0) and (factors_v is not None):
                                q_v = (z_tg_v.to(torch.float32) - mu_z_v) / sigma_z_v
                                jitter_v = float(
                                    cfg["training"].get("copula_jitter", 1e-3)
                                )
                                vloss_cop = _copula_lowrank_nll(
                                    q_v, factors_v, mask_exp_v, jitter=jitter_v
                                )
                                vloss = vloss + lambda_cop_v * vloss_cop
                            total_v += float(vloss.detach().cpu().item())
                            vb += 1

                            # prepare small rollout visualization using full available context with sliding window
                            Bv, Lm1_v, Nv = x_in_v.shape
                            Tf_v = min(Lm1_v, 16) if Lm1_v > 0 else 1
                            Tc_v = max(1, Lm1_v - Tf_v)
                            init_context_v = x_in_v[0:1, :Tc_v, :].to(torch.float32)
                            stim_full_v = stim_v[0:1, : Tc_v + Tf_v, :]
                            pred_future_v, _ctx_ct_v, _pred_ct_v = model.autoregress(
                                init_context_v,
                                stim_full_v,
                                positions_v[0:1],
                                mask_v[0:1],
                                neuron_ids_v[0:1],
                                lam_v[0:1] if lam_v.numel() > 0 else None,
                                las_v[0:1] if las_v.numel() > 0 else None,
                                Tf_v,
                                sampling_rate_hz=sr_v,
                                max_context_len=int(
                                    cfg["training"].get("sequence_length", 12)
                                )
                                - 1,
                            )
                            final_ctx = init_context_v[0].cpu()
                            final_truth = x_in_v[0, Tc_v : Tc_v + Tf_v, :].cpu()
                            final_pred = pred_future_v[0].cpu()
                            if vb >= val_sample_batches_local:
                                break

                    val_loss_step = total_v / max(1, vb)
                    val_points.append((global_step, float(val_loss_step)))
                    # Log step-level validation
                    with open(logs_dir / "loss.txt", "a") as f:
                        # Log EMA of TOTAL loss for the train field
                        tr = (
                            float(ema_total_value)
                            if (ema_total_value is not None)
                            else float(loss_total.detach().cpu().item())
                        )
                        f.write(
                            f"epoch {epoch}, step {count}, train {tr:.6f}, val {val_loss_step:.6f}\n"
                        )
                    if (
                        final_ctx is not None
                        and final_truth is not None
                        and final_pred is not None
                    ):
                        step_dir = Path(plots_dir) / f"epoch_{epoch}_step_{count}"
                        make_validation_plots(
                            step_dir,
                            final_ctx,
                            final_truth,
                            final_pred,
                            title=f"E{epoch} step {count} autoreg 16 neurons",
                        )
                        # Save a checkpoint for this validation run (per-step)
                        try:
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "step_in_epoch": count,
                                    "model": model.state_dict(),
                                },
                                ckpt_dir / f"epoch_{epoch}_step_{count}.pth",
                            )
                        except Exception as _e_ckpt_step:
                            try:
                                print(
                                    f"[train_gbm2][AR E{epoch} step {count}] ckpt save failed: {_e_ckpt_step}"
                                )
                            except Exception:
                                pass
                        # Mean/median over time plots (context and future: true/pred)
                        try:
                            make_mean_rate_plot(
                                step_dir,
                                final_ctx,
                                final_truth,
                                final_pred,
                                title=f"E{epoch} step {count} mean rate over time",
                            )
                            make_median_rate_plot(
                                step_dir,
                                final_ctx,
                                final_truth,
                                final_pred,
                                title=f"E{epoch} step {count} median rate over time",
                            )
                        except Exception as _e_stats:
                            try:
                                print(
                                    f"[train_gbm2][AR E{epoch} step {count}] mean/median plot failed: {_e_stats}"
                                )
                            except Exception:
                                pass
                    _update_loss_plot()
                    # If we failed to prepare an AR visualization, raise with context
                    if final_ctx is None or final_truth is None or final_pred is None:
                        raise RuntimeError(
                            "No AR visualization produced during intra-epoch validation: "
                            "no eligible batch/window met requirements (check sequence_length-1 vs horizon, "
                            "val_sample_batches, or dataset windowing)."
                        )
                except Exception as e:
                    # Print error before re-raising to stop silently skipping AR
                    try:
                        print(f"[train_gbm2][intra-epoch val] Exception: {e}")
                    except Exception:
                        pass
                    raise
                model.train()
                # Free cached VRAM after validation + plotting to reduce peak usage
                _maybe_empty_cuda_cache()
        return total_loss / max(1, count)

    best_val = float("inf")
    val_freq = int(cfg["training"].get("validation_frequency") or 0)
    val_sample_batches = int(cfg["training"].get("val_sample_batches") or 1)

    for epoch in range(1, int(cfg["training"]["num_epochs"]) + 1):
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
                    spikes = batch["spikes"].to(device)
                    positions = batch["positions"].to(device)
                    mask = batch["neuron_mask"].to(device)
                    stim = batch["stimulus"].to(device)
                    neuron_ids = batch["neuron_ids"].to(device)
                    lam = batch.get("log_activity_mean", torch.empty(0)).to(device)
                    las = batch.get("log_activity_std", torch.empty(0)).to(device)
                    sr = float(cfg["training"].get("sampling_rate_hz", 3.0))

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

                    if device.type == "cuda":
                        x_in_z = x_in_z.to(torch.bfloat16)
                        stim_in = stim_in.to(torch.bfloat16)

                    # Spike probabilities for routing during validation
                    spike_probs = 1.0 - torch.exp(-x_in.to(torch.float32) / sr)
                    spike_probs = torch.nan_to_num(
                        spike_probs, nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp_(0.0, 1.0)
                    out_e = model(
                        x_in_z,
                        stim_in,
                        positions,
                        mask,
                        neuron_ids,
                        spike_probs,
                        get_logits=True,
                        input_log_rates=True,
                        return_factors=True,
                    )
                    if isinstance(out_e, tuple) and len(out_e) == 5:
                        mu, raw_log_sigma, _ee, _dd, factors_e = out_e
                    else:
                        mu, raw_log_sigma, _aaa, _bbb = out_e
                        factors_e = None
                    # Z-normalized validation target with optional left-censored NLL
                    y_tg = torch.log(x_tg.clamp_min(1e-7))
                    if lam.numel() > 0 and las.numel() > 0:
                        lam_e2 = lam[:, None, :].to(dtype=y_tg.dtype)
                        las_e2 = las[:, None, :].to(dtype=y_tg.dtype).clamp_min(1e-6)
                        z_tg = (y_tg - lam_e2) / las_e2
                    else:
                        z_tg = y_tg
                    sigma_y = F.softplus(raw_log_sigma.to(torch.float32)) + 1e-6
                    if lam.numel() > 0 and las.numel() > 0:
                        mu_z = (
                            mu.to(torch.float32) - lam_e2.to(torch.float32)
                        ) / las_e2.to(torch.float32)
                        sigma_z = sigma_y / las_e2.to(torch.float32)
                    else:
                        mu_z = mu.to(torch.float32)
                        sigma_z = sigma_y
                    r_min_e = float(cfg["training"].get("loss_r_min", 0.0) or 0.0)
                    if r_min_e > 0.0:
                        y_min_e = math.log(r_min_e)
                        if lam.numel() > 0 and las.numel() > 0:
                            z_min_e = (
                                torch.tensor(
                                    y_min_e, dtype=mu_z.dtype, device=mu_z.device
                                )
                                - lam_e2.to(mu_z.dtype)
                            ) / las_e2.to(mu_z.dtype)
                        else:
                            z_min_e = torch.tensor(
                                y_min_e, dtype=mu_z.dtype, device=mu_z.device
                            )
                        is_cens_e = x_tg <= r_min_e
                        z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                        nll_pdf = (
                            0.5 * z_err.pow(2)
                            + torch.log(sigma_z)
                            + 0.5 * math.log(2.0 * math.pi)
                        )
                        alpha_e = (z_min_e - mu_z) / sigma_z
                        log_cdf_e = torch.log(
                            (
                                0.5 * (1.0 + torch.erf(alpha_e / math.sqrt(2.0)))
                            ).clamp_min(1e-12)
                        )
                        nll_cdf = -log_cdf_e
                        nll = torch.where(is_cens_e, nll_cdf, nll_pdf)
                    else:
                        z_err = (z_tg.to(torch.float32) - mu_z) / sigma_z
                        nll = (
                            0.5 * z_err.pow(2)
                            + torch.log(sigma_z)
                            + 0.5 * math.log(2.0 * math.pi)
                        )
                    mask_exp = mask[:, None, :].expand_as(x_tg).float()
                    vloss = (nll * mask_exp).sum() / mask_exp.sum().clamp_min(1.0)
                    lambda_cop_e = _lambda_copula_for_step(global_step)
                    if (lambda_cop_e != 0.0) and (factors_e is not None):
                        q_e = (z_tg.to(torch.float32) - mu_z) / sigma_z
                        jitter_e = float(cfg["training"].get("copula_jitter", 1e-3))
                        vloss_cop = _copula_lowrank_nll(
                            q_e, factors_e, mask_exp, jitter=jitter_e
                        )
                        vloss = vloss + lambda_cop_e * vloss_cop
                    total += float(vloss.detach().cpu().item())
                    vb += 1
                    # keep one for plot: multi-step AR using full context with sliding window
                    B, Lm1, N = x_in.shape
                    Tf = min(Lm1, 16) if Lm1 > 0 else 1
                    Tc = max(1, Lm1 - Tf)
                    init_context = x_in[0:1, :Tc, :].to(torch.float32)
                    stim_full = stim[0:1, : Tc + Tf, :]
                    pred_future, _ctx_ct, _pred_ct = model.autoregress(
                        init_context,
                        stim_full,
                        positions[0:1],
                        mask[0:1],
                        neuron_ids[0:1],
                        lam[0:1] if lam.numel() > 0 else None,
                        las[0:1] if las.numel() > 0 else None,
                        Tf,
                        sampling_rate_hz=sr,
                        max_context_len=int(cfg["training"].get("sequence_length", 12))
                        - 1,
                    )
                    final_ctx = init_context[0].cpu()
                    final_truth = x_in[0, Tc : Tc + Tf, :].cpu()
                    final_pred = pred_future[0].cpu()
                    if vb >= val_sample_batches:
                        break
            val_loss = total / max(1, vb)

            # Write scalar text and plot
            with open(logs_dir / "loss.txt", "a") as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            val_points.append((global_step, float(val_loss)))
            _update_loss_plot()
            # Safe VRAM cleanup after plotting and optional checkpoint save
            _maybe_empty_cuda_cache()
            # Autoregressive visualization (context + future) for 16 neurons on the kept batch
            if (
                final_ctx is not None
                and final_truth is not None
                and final_pred is not None
            ):
                step_dir = Path(plots_dir) / f"epoch_{epoch}"
                make_validation_plots(
                    step_dir,
                    final_ctx,
                    final_truth,
                    final_pred,
                    title=f"E{epoch} autoreg 16 neurons",
                )
                # Save a checkpoint for this validation run (per-epoch)
                try:
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model": model.state_dict(),
                        },
                        ckpt_dir / f"epoch_{epoch}.pth",
                    )
                except Exception as _e_ckpt_epoch:
                    try:
                        print(
                            f"[train_gbm2][AR E{epoch}] ckpt save failed: {_e_ckpt_epoch}"
                        )
                    except Exception:
                        pass
                # Mean/median over time plots (context and future: true/pred)
                try:
                    make_mean_rate_plot(
                        step_dir,
                        final_ctx,
                        final_truth,
                        final_pred,
                        title=f"E{epoch} mean rate over time",
                    )
                    make_median_rate_plot(
                        step_dir,
                        final_ctx,
                        final_truth,
                        final_pred,
                        title=f"E{epoch} median rate over time",
                    )
                except Exception as _e_stats2:
                    try:
                        print(
                            f"[train_gbm2][AR E{epoch}] mean/median plot failed: {_e_stats2}"
                        )
                    except Exception:
                        pass
            else:
                raise RuntimeError(
                    "No AR visualization produced for this epoch: no eligible validation sample/window "
                    "met requirements (ensure sequence_length-1 > 0 and that horizon fits within available context)."
                )

            # Track best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()},
                    ckpt_dir / "best_model.pth",
                )

        else:
            # End-of-epoch validation only
            val_loss = train_or_val_loop(val_loader, epoch, train=False)
            with open(logs_dir / "loss.txt", "a") as f:
                f.write(f"epoch {epoch}, train {train_loss:.6f}, val {val_loss:.6f}\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()},
                    ckpt_dir / "best_model.pth",
                )
            # Safe VRAM cleanup after end-of-epoch validation
            _maybe_empty_cuda_cache()

    print("Training complete. Best val:", best_val)
    # Final cleanup
    _maybe_empty_cuda_cache()


if __name__ == "__main__":
    main()
