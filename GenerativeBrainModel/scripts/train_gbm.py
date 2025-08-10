#!/usr/bin/env python3
"""
Training script for neuron-level GBM with logging, plots, and videos.

Features:
- Date-time experiment folders with logs, CSVs, and plots
- Per-epoch validation and comparison videos (next-step + autoregression)
- Best-checkpoint video generation at the end
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
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import random
import yaml
from tqdm import tqdm

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.metrics import CombinedMetricsTracker
from GenerativeBrainModel.visualizations import create_nextstep_video, create_autoregression_video


def create_default_config() -> Dict[str, Any]:
    return {
        'experiment': {
            'name': 'gbm_neural_training',
            # Minimal experiment metadata; unused keys removed
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
            'd_stimuli': None,  # will be inferred from data (stimulus_onehot width)
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
            'mixed_precision': True,      # enable AMP
            'amp_dtype': 'bf16',          # 'bf16' | 'fp16' | 'none'
            'distributed': False,         # enable DistributedDataParallel (single-node multi-GPU)
            'backend': 'nccl',            # DDP backend
            'compile_model': False,
            'seed': 42,
            'validation_frequency': 8,          # run small validation every N training batches
            'val_sample_batches': 64,           # number of batches to use for frequent validation
            'gradient_clip_norm': 1.0,
            'gradient_accumulation_steps': None,
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


def build_optimizer(model: GBM, cfg: Dict[str, Any]) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """Build Muon optimizer for hidden weights and AdamW for others, per Muon guidance."""
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
    nonhidden_params += [p for p in model.head.parameters() if p.requires_grad]

    # Construct parameter groups
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['num_epochs'], eta_min=adamw_lr * cfg.get('min_lr_ratio', 0.01))
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


def train_one_epoch(model: GBM, loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, device: torch.device, optimizer, scheduler, scaler: Optional[GradScaler], tracker: CombinedMetricsTracker, epoch: int, cfg: Dict[str, Any]) -> None:
    model.train()
    grad_accum = cfg.get('gradient_accumulation_steps') or 1
    val_freq = int(cfg.get('validation_frequency') or 0)
    val_sample_batches = int(cfg.get('val_sample_batches') or 1)
    # Resolve AMP dtype locally from cfg
    amp_dtype_cfg = (cfg.get('amp_dtype') or 'bf16').lower()
    amp_dtype = torch.bfloat16 if amp_dtype_cfg == 'bf16' else (torch.float16 if amp_dtype_cfg == 'fp16' else torch.float32)
    use_amp = bool(cfg.get('mixed_precision', True))
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar, 1):
        spikes = batch['spikes'].to(device)           # (B, L, N)
        positions = batch['positions'].to(device)     # (B, N, 3)
        mask = batch['neuron_mask'].to(device)        # (B, N)
        stim = batch['stimulus'].to(device)   # (B, L, K)

        # Prepare seq2seq (input: 0..L-2, target: 1..L-1)
        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1, :]  # already one-hot: (B, L-1, K)

        if batch_idx % grad_accum == 1:
            optimizer.zero_grad()

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(x_in, stim_in, positions, mask, get_logits=True)  # (B, L-1, N)
            loss = nn.BCEWithLogitsLoss()(logits, x_tgt)
            loss_to_backprop = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
            if batch_idx % grad_accum == 0:
                scaler.unscale_(optimizer)
                if cfg.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
                scaler.step(optimizer)
                scaler.update()
        else:
            loss_to_backprop.backward()
            if batch_idx % grad_accum == 0:
                if cfg.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
                optimizer.step()

        # Per-batch scheduler (warmup_cosine)
        if scheduler == 'warmup_cosine_placeholder' and 'train_loader_len' in cfg:
            # Setup on first use
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
        # Update tqdm with live loss and EMA
        ema_now = tracker.loss_ema.get()
        pbar.set_postfix({
            'loss': f"{float(loss.detach().cpu().item()):.6f}",
            'ema': f"{ema_now:.6f}" if ema_now is not None else 'N/A'
        })

        # Lightweight validation at frequency
        if val_freq > 0 and batch_idx % val_freq == 0:
            model.eval()
            loss_fn = nn.BCEWithLogitsLoss()
            # sample a few batches from val_loader
            vb = 0
            total_vloss = 0.0
            for vbatch in val_loader:
                spikes_v = vbatch['spikes'].to(device)
                positions_v = vbatch['positions'].to(device)
                mask_v = vbatch['neuron_mask'].to(device)
                stim_v = vbatch['stimulus'].to(device)
                x_in_v = spikes_v[:, :-1, :]
                x_tgt_v = spikes_v[:, 1:, :]
                stim_in_v = stim_v[:, :-1, :]
                with torch.no_grad():
                    logits_v = model(x_in_v, stim_in_v, positions_v, mask_v, get_logits=True)
                    vloss = loss_fn(logits_v, x_tgt_v)
                    probs_v = torch.sigmoid(logits_v)
                total_vloss += float(vloss.detach().cpu().item())
                vb += 1
                tracker.log_validation(epoch, batch_idx, probs_v, x_tgt_v, float(vloss))
                if vb >= val_sample_batches:
                    break
            model.train()
            # Update plots immediately after frequent validation
            try:
                tracker.plot_training()
            except Exception:
                pass


@torch.no_grad()
def validate(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, tracker: CombinedMetricsTracker, epoch: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    loss_fn = nn.BCEWithLogitsLoss()
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Validation E{epoch}"), 1):
        spikes = batch['spikes'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['neuron_mask'].to(device)
        stim = batch['stimulus'].to(device)

        x_in = spikes[:, :-1, :]
        x_tgt = spikes[:, 1:, :]
        stim_in = stim[:, :-1, :]

        logits = model(x_in, stim_in, positions, mask, get_logits=True)
        loss = loss_fn(logits, x_tgt)
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        probs = torch.sigmoid(logits)
        tracker.log_validation(epoch, batch_idx, probs, x_tgt)

    avg_loss = total_loss / max(1, total_batches)
    return {'val_loss': avg_loss}


def generate_epoch_videos(model: GBM, batch: Dict[str, torch.Tensor], device: torch.device, videos_dir: Path, epoch: int) -> None:
    model.eval()
    spikes = batch['spikes'].to(device)              # (B, L, N)
    positions = batch['positions'].to(device)        # (B, N, 3)
    mask = batch['neuron_mask'].to(device)
    stim = batch['stimulus'].to(device).float()  # (B, L, K)

    # Next-step comparison on last step of input
    x_in = spikes[:, :-1, :]
    x_tgt = spikes[:, 1:, :]
    stim_in = stim[:, :-1, :]
    logits = model(x_in, stim_in, positions, mask, get_logits=True)
    probs = torch.sigmoid(logits)
    nextstep_path = videos_dir / f'nextstep_epoch_{epoch}.mp4'
    create_nextstep_video(x_tgt, probs, positions, nextstep_path)

    # Autoregression demo: use last context_len frames and generate n_steps
    context_len = min(8, x_in.shape[1])
    n_steps = min(16, spikes.shape[1] - context_len)
    if n_steps > 0:
        init_x = spikes[:, :context_len, :]
        init_stim = stim[:, :context_len, :]
        future_stim = stim[:, context_len:context_len + n_steps, :]
        gen_seq = model.autoregress(init_x, init_stim, positions, mask, future_stim, n_steps=n_steps, context_len=context_len)
        ar_path = videos_dir / f'autoreg_epoch_{epoch}.mp4'
        create_autoregression_video(gen_seq[:, context_len:, :], positions, ar_path)


def main():
    parser = argparse.ArgumentParser(description='Train GBM on neuron sequences')
    parser.add_argument('--config', type=str, required=False, help='Path to YAML config')
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, 'r') as f:
            user = yaml.safe_load(f)
        cfg = deep_update(cfg, user)

    # Setup experiment dirs
    base_dir = Path('experiments/gbm_neural')
    dirs = setup_experiment_dirs(base_dir, cfg['experiment']['name'])
    save_config(cfg, dirs['exp'] / 'config.yaml')
    logger = build_logger(dirs['logs'], cfg['logging'].get('log_level', 'INFO'))

    # Device & seeds
    #torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if (cfg['training']['use_gpu'] and torch.cuda.is_available()) else 'cpu')
    set_seeds(cfg['training']['seed'])
    if device.type == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    # If not in distributed mode, stub minimal torch.distributed functions so Muon works in single-process
    try:
        import torch.distributed as dist
        if dist.is_available() and not dist.is_initialized():
            dist.get_world_size = lambda group=None: 1
            dist.get_rank = lambda group=None: 0
            def _fake_all_gather(tensor_list, tensor, group=None):
                if tensor_list is None:
                    return
                if len(tensor_list) == 0:
                    return
                if tensor_list[0].shape == tensor.shape:
                    tensor_list[0].copy_(tensor)
                else:
                    # Fallback: resize and copy if shape differs
                    tensor_list[0].resize_(tensor.shape).copy_(tensor)
            dist.all_gather = _fake_all_gather
    except Exception:
        pass

    # Data
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(cfg)
    cfg['training']['train_loader_len'] = len(train_loader)

    # Model
    mcfg = cfg['model']
    # Infer d_stimuli from data one-hot width if not provided (uses global padded width)
    inferred_d_stimuli: Optional[int] = None
    try:
        sample_batch = next(iter(train_loader))
        inferred_d_stimuli = int(sample_batch['stimulus'].shape[-1])
    except Exception:
        pass
    d_stimuli = mcfg['d_stimuli'] if mcfg['d_stimuli'] is not None else (inferred_d_stimuli or 1)
    model = GBM(d_model=mcfg['d_model'], d_stimuli=d_stimuli, n_heads=mcfg['n_heads'], n_layers=mcfg['n_layers']).to(device)
    # If bf16 requested, move model to bf16 to ensure Linear weights match bf16 inputs under autocast
    amp_dtype_cfg = (cfg['training'].get('amp_dtype') or 'bf16').lower()
    if cfg['training'].get('mixed_precision', True) and amp_dtype_cfg == 'bf16' and device.type == 'cuda':
        model = model.to(dtype=torch.bfloat16)
    if cfg['training'].get('compile_model', False):
        try:
            model = torch.compile(model)
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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            logger.info("Initialized DistributedDataParallel")
        except Exception as e:
            logger.warning(f"Failed to initialize DDP, continuing single-process: {e}")

    # Optimizer & scheduler
    optimizer, scheduler = build_optimizer(model, cfg['training'])
    amp_dtype_cfg = (cfg['training'].get('amp_dtype') or 'bf16').lower()
    amp_dtype = torch.bfloat16 if amp_dtype_cfg == 'bf16' else (torch.float16 if amp_dtype_cfg == 'fp16' else torch.float32)
    scaler = GradScaler(enabled=cfg['training'].get('mixed_precision', False) and amp_dtype is torch.float16)

    # Metrics
    tracker = CombinedMetricsTracker(log_dir=dirs['logs'], ema_alpha=0.05, val_threshold=0.5, enable_plots=True)
    # Persist architecture summary once per run
    try:
        write_architecture_file(model, dirs, cfg)
    except Exception as e:
        logger.warning(f"Failed to write architecture file: {e}")

    best_loss = float('inf')
    best_ckpt = None

    num_epochs = cfg['training']['num_epochs']
    for epoch in range(1, num_epochs + 1):
        # Shuffle between epochs for distributed samplers
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, val_loader, device, optimizer, scheduler, scaler, tracker, epoch, cfg['training'])
        val_metrics = validate(model, val_loader, device, tracker, epoch)
        logger.info(f"Epoch {epoch} - Val: {val_metrics}")

        # Save checkpoint
        ckpt_path = dirs['ckpt'] / f'epoch_{epoch}.pth'
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': cfg}, ckpt_path)

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


