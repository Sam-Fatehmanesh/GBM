#!/usr/bin/env python3
"""
Training script for neuron-level GBM with logging, plots, and videos.

Changes vs. previous:
- No AMP/autocast; model runs in bf16 directly; losses in fp32.
- Optional torch.compile with dynamic=True (fewer recompiles).
- CUDA prefetcher casts spikes/stim to bf16 to avoid cast kernels on the default stream.
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
import random
import yaml
from tqdm import tqdm

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_nextstep_video, create_autoregression_video
import pdb


# ------------------------------- CUDA prefetcher (casts to bf16) -------------------------------

class CUDAPrefetchLoader:
    """Wrap a DataLoader to prefetch the next batch to CUDA on a dedicated stream.
    Casts spikes/stim to bf16 to minimize cast kernels on the default stream.
    """
    def __init__(self, loader: torch.utils.data.DataLoader, device: torch.device, *, cast_bf16: bool = True):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.cast_bf16 = cast_bf16

    def __len__(self):
        return len(self.loader)

    def _to_device(self, batch):
        non_blocking = True
        out = {}
        out['spikes'] = batch['spikes'].to(self.device, non_blocking=non_blocking)
        out['positions'] = batch['positions'].to(self.device, non_blocking=non_blocking)
        out['neuron_mask'] = batch['neuron_mask'].to(self.device, non_blocking=non_blocking)
        out['stimulus'] = batch['stimulus'].to(self.device, non_blocking=non_blocking)
        if self.cast_bf16:
            out['spikes'] = out['spikes'].to(torch.bfloat16)
            out['stimulus'] = out['stimulus'].to(torch.bfloat16)
        out['file_path'] = batch['file_path']
        out['start_idx'] = batch['start_idx']
        return out

    def __iter__(self):
        if self.stream is None:
            for b in self.loader:
                yield self._to_device(b)
            return
        first = True
        next_batch = None
        for b in self.loader:
            with torch.cuda.stream(self.stream):
                next_batch = self._to_device(b)
            if not first:
                torch.cuda.current_stream().wait_stream(self.stream)
                yield cur
            else:
                first = False
            cur = next_batch
        torch.cuda.current_stream().wait_stream(self.stream)
        if next_batch is not None:
            yield next_batch


# ---------------------------------- Default config & utilities ----------------------------------

def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'gbm_neural_training',
        },
        'data': {
            'data_dir': 'processed_spike_voxels_2018',
            'test_subjects': [],
            'use_cache': False,
        },
        'model': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_stimuli': None,  # inferred from data (stimulus_onehot width)
        },
        'training': {
            'batch_size': 2,
            'num_epochs': 50,
            'learning_rate': 5e-4,
            'muon_lr': 2e-2,
            'adamw_betas': (0.9, 0.95),
            'weight_decay': 1e-4,
            'scheduler': 'warmup_cosine',
            'min_lr_ratio': 0.01,
            'sequence_length': 12,
            'stride': 3,
            'max_timepoints_per_subject': None,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2,
            'use_gpu': True,
            'distributed': False,
            'backend': 'nccl',
            'compile_model': False,     # set True to enable torch.compile(dynamic=True)
            'seed': 42,
            'validation_frequency': 8,
            'val_sample_batches': 64,
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': None,
            'profile': False,
            'profile_steps': 50,
            'profile_dir': './tb_prof',
            # Extra loss term weight: mean squared difference between per-timestep mean
            # predicted activation and target activation (masked over valid neurons).
            'mean_activation_mse_weight': 0.1,
            # Scheduled sampling settings
            'scheduled_sampling': {
                'enable': False,
                # Linear schedule over global steps from step 0 to final training step
                'start_prob': 0.0,
                'end_prob': 0.2,
                # If True, sample Bernoulli from predicted probabilities when substituting.
                # If False, feed raw probabilities.
                'sample': True,
            },
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


def sanitized_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config without non-serializable runtime objects.
    Removes training.scheduler_obj (LambdaLR) and global_step counter.
    """
    import copy
    cfg_copy = copy.deepcopy(cfg)
    # If a full config dict is passed
    if isinstance(cfg_copy, dict) and 'training' in cfg_copy and isinstance(cfg_copy['training'], dict):
        tr = cfg_copy['training']
        tr.pop('scheduler_obj', None)
        tr.pop('global_step', None)
    # If just the training dict is passed
    elif isinstance(cfg_copy, dict):
        cfg_copy.pop('scheduler_obj', None)
        cfg_copy.pop('global_step', None)
    return cfg_copy

def setup_experiment_dirs(base_dir: Path, name: str) -> Dict[str, Path]:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = base_dir / f"{name}_{ts}"
    log_dir = exp_dir / 'logs'
    plots_dir = log_dir / 'plots'
    videos_dir = exp_dir / 'videos'
    ckpt_dir = exp_dir / 'checkpoints'
    for p in [exp_dir, log_dir, plots_dir, videos_dir, ckpt_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {'exp': exp_dir, 'logs': log_dir, 'plots': plots_dir, 'videos': videos_dir, 'ckpt': ckpt_dir}


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def build_logger(log_dir: Path, level: str = 'INFO') -> logging.Logger:
    logger = logging.getLogger('train_gbm')
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------- Optimizer & scheduler -------------------------------------

def build_optimizer(model: GBM, cfg: Dict[str, Any]) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """Muon for hidden weights; AdamW for the rest."""
    try:
        from muon import MuonWithAuxAdam
    except ImportError as e:
        raise ImportError("Muon optimizer not found. Install: pip install git+https://github.com/KellerJordan/Muon") from e

    # Hidden weights: parameters with ndim >= 2 from the attention body (layers)
    hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2 and p.requires_grad]
    # Hidden gains/biases: parameters with ndim < 2 from the body
    hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2 and p.requires_grad]
    # Non-hidden: embeddings + head
    nonhidden_params = []
    for m in model.embed.values():
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]
    for m in model.head.values():   
        nonhidden_params += [p for p in m.parameters() if p.requires_grad]

    muon_lr = cfg.get('muon_lr', 0.02)
    muon_weight_decay = cfg.get('weight_decay', 1e-4)
    adamw_lr = cfg.get('learning_rate', 3e-4)
    adamw_betas = tuple(cfg.get('adamw_betas', (0.9, 0.95)))
    adamw_weight_decay = cfg.get('weight_decay', 1e-4)

    param_groups = []
    if hidden_weights:
        param_groups.append(dict(params=hidden_weights, use_muon=True, lr=muon_lr, weight_decay=muon_weight_decay))
    if hidden_gains_biases or nonhidden_params:
        param_groups.append(dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_weight_decay))

    opt = MuonWithAuxAdam(param_groups)

    sched_type = cfg.get('scheduler', None)
    scheduler = None
    if sched_type == 'warmup_cosine':
        scheduler = 'warmup_cosine_placeholder'
    elif sched_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['num_epochs'],
                                                         eta_min=adamw_lr * cfg.get('min_lr_ratio', 0.01))
    elif sched_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg['num_epochs'] // 3), gamma=0.1)
    return opt, scheduler


def write_architecture_file(model: nn.Module, dirs: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    """Write an architecture summary (model repr and params) at run start."""
    exp_dir: Path = dirs['exp']
    out_path = exp_dir / 'architecture.txt'
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params

    with open(out_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GBM MODEL ARCHITECTURE SUMMARY\n")
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
        f.write("PARAMETER BREAKDOWN (name, shape, count, trainable)\n")
        f.write("-" * 40 + "\n")
        for name, p in model.named_parameters():
            f.write(f"  {name:<60} {tuple(p.shape)!s:<20} {p.numel():>12,}  {'✓' if p.requires_grad else '✗'}\n")
        f.write("\nMODEL STRUCTURE (repr)\n")
        f.write("-" * 40 + "\n")
        f.write(str(model))
        f.write("\n")


# ------------------------------------------- Train / Val -------------------------------------------

def train_one_epoch(
    model: GBM,
    loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer,
    scheduler,
    tracker: CombinedMetricsTracker,
    epoch: int,
    cfg: Dict[str, Any],
    best_loss: float,
    best_ckpt: Optional[Path],
    ckpt_dir: Path,
    videos_dir: Path,
) -> Tuple[float, Optional[Path]]:
    model.train()
    grad_accum = cfg.get('gradient_accumulation_steps') or 1
    val_freq = int(cfg.get('validation_frequency') or 0)
    val_sample_batches = int(cfg.get('val_sample_batches') or 1)

    # Wrap loader with CUDA prefetch to overlap H2D with compute (and cast to bf16)
    prefetch = (device.type == 'cuda')
    wrapped_loader = CUDAPrefetchLoader(loader, device, cast_bf16=True) if prefetch else loader
    pbar = tqdm(wrapped_loader, desc=f"Epoch {epoch}", mininterval=0.1, smoothing=0.05)

    # Determine validation trigger steps (exact count per epoch), and always include step 4 if possible
    triggers: set[int] = set()
    total_steps = len(loader)
    if val_freq > 0:
        for j in range(1, val_freq + 1):
            step = max(1, min(total_steps, round(j * total_steps / (val_freq + 1))))
            triggers.add(int(step))
    if total_steps >= 4:
        triggers.add(4)

    use_profiler = bool(cfg.get('profile', False)) and torch.cuda.is_available()
    prof = None
    if use_profiler:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=cfg.get('profile_steps', 50), repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(cfg.get('profile_dir', './tb_prof'))),
            record_shapes=True, profile_memory=True, with_stack=True, with_modules=True
        )
        prof.__enter__()

    for batch_idx, batch in enumerate(pbar, 1):
        # Batches are already on device and in bf16 via the prefetcher
        spikes = batch['spikes']           # (B, L, N) bf16
        positions = batch['positions']     # (B, N, 3) fp32
        mask = batch['neuron_mask']        # (B, N) bool/int
        stim = batch['stimulus']           # (B, L, K) bf16

        # Prepare seq2seq (input: 0..L-2, target: 1..L-1)
        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :].float()   # cast to fp32 for loss
        stim_in = stim[:, :-1, :]

        if batch_idx % grad_accum == 1:
            optimizer.zero_grad()

        # Scheduled sampling: optionally replace part of x_in with model predictions from a
        # teacher-forced pass (previous-step predictions only). Training-only.
        ss_cfg = cfg.get('scheduled_sampling', {}) or {}
        ss_enable = bool(ss_cfg.get('enable', False))
        if ss_enable:
            start_p = float(ss_cfg.get('start_prob', 0.0))
            end_p = float(ss_cfg.get('end_prob', 0.2))
            # Drive schedule by global batch step (increments every batch), spanning entire training
            step_now = int(cfg.get('ss_global_step', 0))
            train_len = int(cfg.get('train_loader_len', 0)) or int(len(loader))
            total_steps = max(1, int(cfg.get('num_epochs', 1)) * train_len)
            start_s = 0
            end_s = total_steps
            if step_now <= start_s:
                ss_p = start_p
            elif step_now >= end_s:
                ss_p = end_p
            else:
                frac = (step_now - start_s) / max(1, (end_s - start_s))
                ss_p = start_p + frac * (end_p - start_p)
            if ss_p > 0.0:
                with torch.no_grad():
                    logits_tf = model(x_in, stim_in, positions, mask, get_logits=True)
                    probs_tf = torch.sigmoid(logits_tf.float()).to(dtype=x_in.dtype)
                # Build previous-step predictions aligned to inputs: at t>0 use pred for t from probs_tf[:, t-1]
                prev_preds = torch.zeros_like(x_in)
                prev_preds[:, 1:, :] = probs_tf[:, :-1, :]
                if bool(ss_cfg.get('sample', True)):
                    # Sample Bernoulli spikes from probabilities
                    prev_sampled = torch.bernoulli(prev_preds.to(torch.float32)).to(dtype=x_in.dtype)
                    substitute = prev_sampled
                else:
                    substitute = prev_preds
                # Per-batch, per-time mask (not per-neuron) to replace entire frames; keep t=0 untouched
                replace_mask = (torch.rand(x_in.shape[0], x_in.shape[1], 1, device=x_in.device) < ss_p).to(x_in.dtype)
                replace_mask[:, 0, :] = 0  # never replace the first input step
                x_in = x_in * (1 - replace_mask) + substitute * replace_mask
            # Increment global step counter every batch
            cfg['ss_global_step'] = step_now + 1

        logits = model(x_in, stim_in, positions, mask, get_logits=True)  # (B, L-1, N) bf16
        bce_loss = nn.BCEWithLogitsLoss()(logits.float(), x_tgt)         # fp32
        # Mean activation MSE term
        with torch.no_grad():
            mask_rep = mask[:, None, :].expand(x_tgt.shape[0], x_tgt.shape[1], x_tgt.shape[2]).float()
            denom = mask_rep.sum(dim=-1).clamp_min(1.0)
        pred_mean = torch.sigmoid(logits.float())
        pred_mean = (pred_mean * mask_rep).sum(dim=-1) / denom
        targ_mean = (x_tgt * mask_rep).sum(dim=-1) / denom
        mean_mse = torch.mean((pred_mean - targ_mean) ** 2)
        alpha = float(cfg.get('mean_activation_mse_weight', 0.0) or 0.0)
        loss = bce_loss + alpha * mean_mse
        (loss / grad_accum).backward()

        if batch_idx % grad_accum == 0:
            if cfg.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
            optimizer.step()
            # Per-batch scheduler (warmup_cosine) → step AFTER optimizer.step()
            if scheduler == 'warmup_cosine_placeholder' and 'train_loader_len' in cfg:
                if 'scheduler_obj' not in cfg:
                    warm = int(0.1 * cfg['train_loader_len'])
                    total = cfg['num_epochs'] * cfg['train_loader_len']
                    base_lr = cfg['learning_rate']
                    min_lr = base_lr * cfg.get('min_lr_ratio', 0.01)
                    def lr_lambda(step):
                        if step < warm:
                            return float(step) / float(max(1, warm))
                        prog = (step - warm) / max(1, total - warm)
                        return (min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * min(prog, 1.0)))) / base_lr
                    cfg['scheduler_obj'] = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                    cfg['global_step'] = 0
                cfg['global_step'] += 1
                cfg['scheduler_obj'].step()

        lr_now = optimizer.param_groups[0]['lr']
        tracker.log_train(epoch, batch_idx, float(loss.detach().cpu().item()), lr_now)
        ema_now = tracker.loss_ema.get()
        pbar.set_postfix({
            'loss': f"{float(loss.detach().cpu().item()):.6f}",
            'ema': f"{ema_now:.6f}" if ema_now is not None else 'N/A'
        })

        if prof is not None:
            prof.step()

        # Lightweight validation at frequency
        if batch_idx in triggers:
            model.eval()
            loss_fn = nn.BCEWithLogitsLoss()
            vb = 0
            total_vloss = 0.0
            preds_all = []
            targs_all = []
            for vbatch in val_loader:
                spikes_v = vbatch['spikes'].to(device).to(torch.bfloat16)
                positions_v = vbatch['positions'].to(device)
                mask_v = vbatch['neuron_mask'].to(device)
                stim_v = vbatch['stimulus'].to(device).to(torch.bfloat16)
                x_in_v = spikes_v[:, :-1, :]
                x_tgt_v = spikes_v[:, 1:, :].float()
                stim_in_v = stim_v[:, :-1, :]
                with torch.no_grad():
                    logits_v = model(x_in_v, stim_in_v, positions_v, mask_v, get_logits=True)
                    vloss_bce = loss_fn(logits_v.float(), x_tgt_v)
                    # Validation mean activation MSE (same as train)
                    with torch.no_grad():
                        mask_rep_v = mask_v[:, None, :].expand(x_tgt_v.shape[0], x_tgt_v.shape[1], x_tgt_v.shape[2]).float()
                        denom_v = mask_rep_v.sum(dim=-1).clamp_min(1.0)
                        probs_v = torch.sigmoid(logits_v.float())
                        pred_mean_v = (probs_v * mask_rep_v).sum(dim=-1) / denom_v
                        targ_mean_v = (x_tgt_v * mask_rep_v).sum(dim=-1) / denom_v
                        mean_mse_v = torch.mean((pred_mean_v - targ_mean_v) ** 2)
                    vloss = vloss_bce + float(cfg.get('mean_activation_mse_weight', 0.0) or 0.0) * mean_mse_v
                    probs_v = torch.sigmoid(logits_v)
                total_vloss += float(vloss.detach().cpu().item())
                preds_all.append(probs_v.detach())
                targs_all.append(x_tgt_v.detach())
                vb += 1
                if vb >= val_sample_batches:
                    break
            if vb > 0:
                avg_vloss = total_vloss / vb
                preds_all = torch.cat([p.flatten() for p in preds_all], dim=0)
                targs_all = torch.cat([t.flatten() for t in targs_all], dim=0)
                tracker.log_validation(epoch, batch_idx, preds_all, targs_all, avg_vloss)
                # Sample video
                try:
                    sample_batch = next(iter(val_loader))
                    # Ensure a consistent per-sample shape for videos to avoid cat-size issues
                    spikes0 = sample_batch['spikes'][0:1]
                    stim0 = sample_batch['stimulus'][0:1]
                    pos0 = sample_batch['positions'][0:1]
                    mask0 = sample_batch['neuron_mask'][0:1]
                    # Diagnostics
                    print(f"[VideoDebug] spikes {tuple(spikes0.shape)} stim {tuple(stim0.shape)} pos {tuple(pos0.shape)} mask {tuple(mask0.shape)}")
                    # Truncate to a common L if mismatch exists
                    Lx = spikes0.shape[1]
                    Ls = stim0.shape[1]
                    L = min(Lx, Ls)
                    if Lx != Ls:
                        print(f"[VideoDebug] Truncating sequence length from spikes {Lx} / stim {Ls} to L={L}")
                    spikes0 = spikes0[:, :L]
                    stim0 = stim0[:, :L]
                    sample_batch = {
                        'spikes': spikes0,
                        'positions': pos0,
                        'neuron_mask': mask0,
                        'stimulus': stim0,
                    }
                    generate_epoch_videos(model, sample_batch, device, videos_dir, epoch=f"{epoch}_step{batch_idx}")
                except Exception as e:
                    # Log and continue; video generation should not break training
                    import traceback
                    traceback.print_exc()
                    print(f"Video generation failed at step {batch_idx}: {e}")
                # Save best checkpoint if improved
                if avg_vloss < best_loss:
                    best_loss = avg_vloss
                    ckpt_path = ckpt_dir / f"best_step_{epoch}_{batch_idx}.pth"
                    torch.save({
                        'epoch': epoch,
                        'step': batch_idx,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': sanitized_config({'training': cfg})
                    }, ckpt_path)
                    best_ckpt = ckpt_path
            model.train()
            # Update plots immediately after frequent validation
            try:
                tracker.plot_training()
            except Exception:
                pass

    if prof is not None:
        prof.__exit__(None, None, None)
    return best_loss, best_ckpt


@torch.no_grad()
def validate(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device,
             tracker: CombinedMetricsTracker, epoch: int, training_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Run end-of-epoch validation and log a SINGLE aggregated entry to CSV/plots.
    Computes average loss (with optional mean-activation MSE term) and PR-AUC over all batches.
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_batches = 0
    preds_all = []
    targs_all = []
    for batch in tqdm(loader, desc=f"Validation E{epoch}"):
        spikes = batch['spikes'].to(device).to(torch.bfloat16)
        positions = batch['positions'].to(device)
        mask = batch['neuron_mask'].to(device)
        stim = batch['stimulus'].to(device).to(torch.bfloat16)

        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :].float()
        stim_in = stim[:, :-1, :]

        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        vloss_bce = loss_fn(logits.float(), x_tgt)
        with torch.no_grad():
            mask_rep = mask[:, None, :].expand(x_tgt.shape[0], x_tgt.shape[1], x_tgt.shape[2]).float()
            denom = mask_rep.sum(dim=-1).clamp_min(1.0)
        probs = torch.sigmoid(logits.float())
        pred_mean = (probs * mask_rep).sum(dim=-1) / denom
        targ_mean = (x_tgt * mask_rep).sum(dim=-1) / denom
        mean_mse = torch.mean((pred_mean - targ_mean) ** 2)
        alpha = float(training_cfg.get('mean_activation_mse_weight', 0.0) or 0.0)
        loss = vloss_bce + alpha * mean_mse
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        preds_all.append(probs.detach())
        targs_all.append(x_tgt.detach())

    avg_loss = total_loss / max(1, total_batches)
    if preds_all and targs_all:
        preds_all = torch.cat([p.reshape(-1) for p in preds_all], dim=0)
        targs_all = torch.cat([t.reshape(-1) for t in targs_all], dim=0)
        # Log a single aggregated row (batch_idx=1)
        tracker.log_validation(epoch, 1, preds_all, targs_all, avg_loss)
    return {'val_loss': avg_loss}


def generate_epoch_videos(model: GBM, batch: Dict[str, torch.Tensor], device: torch.device,
                           videos_dir: Path, epoch: int | str) -> None:
    model.eval()
    # Keep spikes in fp32 to preserve small probabilities for visualization
    spikes = batch['spikes'].to(device).float()                 # (B, L, N)
    positions = batch['positions'].to(device)                   # (B, N, 3)
    mask = batch['neuron_mask'].to(device)
    stim = batch['stimulus'].to(device).float()                 # (B, L, K)

    # Next-step comparison on last step of input
    x_in = spikes[:, :-1, :]
    x_tgt = spikes[:, 1:, :]
    stim_in = stim[:, :-1, :]
    logits = model(x_in, stim_in, positions, mask, get_logits=True)
    probs = torch.sigmoid(logits)
    nextstep_path = videos_dir / f'nextstep_epoch_{epoch}.mp4'
    #pdb.set_trace()
    create_nextstep_video(x_tgt, probs, positions, nextstep_path)

    # Autoregression demo (double-panel): use half window as context, half as truth horizon
    L_full = int(spikes.shape[1])
    context_len = max(1, L_full // 2)
    n_steps = context_len
    if n_steps > 0:
        init_x = spikes[0:1, :context_len, :]
        init_stim = stim[0:1, :context_len, :]
        # Build future stimulus of length n_steps from the second half; pad with zeros if needed
        K = stim.shape[-1]
        future_real = stim[0:1, context_len: context_len + n_steps, :]
        pad_len = n_steps - future_real.shape[1]
        if pad_len > 0:
            future_pad = torch.zeros((1, pad_len, K), device=stim.device, dtype=stim.dtype)
            future_stim = torch.cat([future_real, future_pad], dim=1)
        else:
            future_stim = future_real
        pos0 = positions[0:1]
        mask0 = mask[0:1]
        gen_seq = model.autoregress(init_x, init_stim, pos0, mask0, future_stim,
                                     n_steps=n_steps, context_len=context_len)
        ar_path = videos_dir / f'autoreg_epoch_{epoch}.mp4'
        truth_horizon = spikes[0:1, context_len: context_len + n_steps, :]
        create_autoregression_video(gen_seq[:, -n_steps:, :], pos0, ar_path, truth=truth_horizon)


# ---------------------------------------------- Main ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train GBM on neuron sequences')
    parser.add_argument('--config', type=str, required=False, help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    # Dirs & logger
    base_dir = Path('experiments/gbm_neural')
    dirs = setup_experiment_dirs(base_dir, cfg['experiment']['name'])
    save_config(cfg, dirs['exp'] / 'config.yaml')
    logger = build_logger(dirs['logs'], cfg['logging'].get('log_level', 'INFO'))

    # Device & seeds
    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    set_seeds(cfg['training']['seed'])
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # If not distributed, stub minimal torch.distributed funcs so Muon works single-process
    try:
        import torch.distributed as dist
        if dist.is_available() and not dist.is_initialized():
            dist.get_world_size = lambda group=None: 1
            dist.get_rank = lambda group=None: 0
            def _fake_all_gather(tensor_list, tensor, group=None):
                if tensor_list is None or len(tensor_list) == 0:
                    return
                if tensor_list[0].shape == tensor.shape:
                    tensor_list[0].copy_(tensor)
                else:
                    tensor_list[0].resize_(tensor.shape).copy_(tensor)
            dist.all_gather = _fake_all_gather
    except Exception:
        pass

    # Data
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(cfg)
    cfg['training']['train_loader_len'] = len(train_loader)

    # Model
    mcfg = cfg['model']
    # Infer d_stimuli from data one-hot width if not provided
    inferred_d_stimuli: Optional[int] = None
    try:
        sample_batch = next(iter(train_loader))
        inferred_d_stimuli = int(sample_batch['stimulus'].shape[-1])
    except Exception:
        pass
    d_stimuli = mcfg['d_stimuli'] if mcfg['d_stimuli'] is not None else (inferred_d_stimuli or 1)

    model = GBM(d_model=mcfg['d_model'], d_stimuli=d_stimuli,
                n_heads=mcfg['n_heads'], n_layers=mcfg['n_layers']).to(device)

    # Run the whole model in bf16 on CUDA (norms/centroids handled internally in fp32)
    if device.type == 'cuda':
        model = model.to(dtype=torch.bfloat16)

    # Optional torch.compile with dynamic shapes
    if cfg['training'].get('compile_model', False):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Initialize DDP if requested
    ddp_enabled = bool(cfg['training'].get('distributed', False))
    if ddp_enabled and torch.cuda.device_count() > 1:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend=cfg['training'].get('backend', 'nccl'))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
            logger.info("Initialized DistributedDataParallel")
        except Exception as e:
            logger.warning(f"Failed to initialize DDP, continuing single-process: {e}")

    # Optimizer & scheduler
    optimizer, scheduler = build_optimizer(model, cfg['training'])

    # Metrics
    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.01, val_threshold=0.5, enable_plots=True)
    try:
        write_architecture_file(model, dirs, cfg)
    except Exception as e:
        logger.warning(f"Failed to write architecture file: {e}")

    best_loss = float('inf')
    best_ckpt: Optional[Path] = None

    num_epochs = cfg['training']['num_epochs']
    for epoch in range(1, num_epochs + 1):
        # Shuffle between epochs for distributed samplers
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        best_loss, best_ckpt = train_one_epoch(
            model, train_loader, val_loader, device, optimizer, scheduler,
            tracker, epoch, cfg['training'], best_loss, best_ckpt, dirs['ckpt'], dirs['videos']
        )

        val_metrics = validate(model, val_loader, device, tracker, epoch, cfg['training'])
        logger.info(f"Epoch {epoch} - Val: {val_metrics}")

        # Save checkpoint
        ckpt_path = dirs['ckpt'] / f'epoch_{epoch}.pth'
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'config': sanitized_config(cfg)}, ckpt_path)

        # Videos on a small sampled batch
        try:
            sample_batch = next(iter(val_loader))
            generate_epoch_videos(model, sample_batch, device, dirs['videos'], epoch)
        except Exception as e:
            logger.warning(f"Epoch {epoch} video generation failed: {e}")

        # Track best
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            best_ckpt = ckpt_path

        # Update plots
        tracker.plot_training()

    logger.info("Training complete.")
    # Best checkpoint videos
    if best_ckpt is not None:
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state['model'])
        try:
            sample_batch = next(iter(val_loader))
            generate_epoch_videos(model, sample_batch, device, dirs['videos'], epoch='best')
        except Exception as e:
            logger.warning(f"Best checkpoint video generation failed: {e}")


if __name__ == '__main__':
    main()
