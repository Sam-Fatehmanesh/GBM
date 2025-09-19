#!/usr/bin/env python3
"""
Trajectory planning with GBM + BModel:

Goal: Given a desired 2D trajectory over T steps (e.g., swirl), derive target behavior
velocities, generate a GBM autoregressive neural sequence from an initial context, and
optimize a minimal positive-only delta on the GBM-generated neural sequence so that
the BModel-predicted behavior matches the target velocities, thus yielding the target path.

Pipeline:
  1) Build a desired 2D path (swirl) for T steps.
  2) Convert path → target behaviors (bilateral forward speed + left/right unilateral):
       - forward_speed ~ per-step distance (scaled to [0,1]) → Bilateral (B)
       - heading_rate ~ per-step heading change (scaled) → (RU - LU)
       - Set LU = -0.5 * delta, RU = +0.5 * delta; then L = B + LU, R = B + RU; clip [0,1]
  3) Load an initial 12-step neural context (from subject H5) + positions/mask + stimulus (zeros if missing).
  4) Load GBM checkpoint and autoregress horizon T (sliding context of 12 via model.autoregress).
  5) Load BModel per-dimension checkpoints and build a sliding window (length 6) evaluator over the T frames.
  6) Optimize positive-only per-neuron, per-time delta (softplus) with L1 penalty to align BModel predictions
     with target behaviors across all T steps. Keep spikes in [0,1].
  7) Save artifacts (delta, modified spikes, predictions/targets, trajectory plots).

Notes:
  - Behavior column mapping is hard-coded for the dataset as: {L:0, R:4, B:3, LU:1, RU:2}.
  - BModel expects a fixed window length of 6; we use reflect padding to produce a prediction at each step.
  - Scales (speed_scale, turn_scale) should be tuned so that targets stay within [0,1].

Usage example:
  python -m GenerativeBrainModel.scripts.optimize_trajectory_gbm_bmodel \
    --data-dir processed_spike_voxels_2018 \
    --subject subject_14 \
    --gbm-checkpoint experiments/gbm_neural/.../checkpoints/best_step_*.pth \
    --bmodel-train-dir experiments/bmodel_behavior/bmodel_behavior_training_... \
    --horizon 128 --context-len 12 --speed-scale 0.1 --turn-scale 0.2
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reduce CUDA fragmentation early (must be set before first CUDA alloc)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.models.bmodel import BModel


def _read_sampling_rate_hz(f: h5py.File) -> float:
    for key in (
        'final_sampling_rate', 'effective_sampling_rate', 'target_sampling_rate', 'original_sampling_rate'
    ):
        if key in f.attrs:
            try:
                return float(f.attrs[key])
            except Exception:
                pass
    if 'original_sampling_rate_hz' in f:
        try:
            return float(f['original_sampling_rate_hz'][()])
        except Exception:
            pass
    if 'matlab_fpsec' in f.attrs:
        try:
            return float(f.attrs['matlab_fpsec'])
        except Exception:
            pass
    return float('nan')


def load_context_from_subject(h5_path: Path, context_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load first context_len frames of spikes and stimulus (or zeros), plus positions/mask.
    Returns (spikes_ctx [1,L,N], stim_ctx [1,L,K], positions [1,N,3], mask [1,N], K)."""
    with h5py.File(h5_path, 'r') as f:
        spikes = f['spike_probabilities'][()].astype(np.float32)
        T, N = spikes.shape
        L = min(context_len, T)
        spikes_ctx = spikes[:L]
        if 'stimulus_full' in f:
            stim_full = f['stimulus_full'][()].astype(np.float32)
            # Normalize to (T, K)
            if stim_full.ndim == 2:
                if stim_full.shape[0] == T:
                    stim_TK = stim_full
                elif stim_full.shape[1] == T:
                    stim_TK = stim_full.T
                else:
                    K = 1
                    stim_TK = np.zeros((T, K), dtype=np.float32)
            else:
                K = 1
                stim_TK = np.zeros((T, K), dtype=np.float32)
        else:
            K = 1
            stim_TK = np.zeros((T, K), dtype=np.float32)
        stim_ctx = stim_TK[:L]
        positions = f['cell_positions'][()].astype(np.float32)
        if np.isnan(positions).any():
            positions = np.nan_to_num(positions)
        mask = np.ones((positions.shape[0],), dtype=np.float32)
        return (
            spikes_ctx[None, ...],
            stim_ctx[None, ...],
            positions[None, ...],
            mask[None, ...],
            stim_ctx.shape[1],
        )


def load_gbm_from_checkpoint(ckpt_path: Path, device: torch.device, spikes_are_rates: bool = False) -> GBM:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_in_ckpt = state.get('config', {}) if isinstance(state, dict) else {}
    model_cfg = cfg_in_ckpt.get('model', None) if isinstance(cfg_in_ckpt, dict) else None
    sd_in = state['model'] if (isinstance(state, dict) and 'model' in state) else state

    def _normalize_key(k: str) -> str:
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

    sd = {_normalize_key(k): v for k, v in (sd_in.items() if isinstance(sd_in, dict) else sd_in)} if isinstance(sd_in, dict) else sd_in

    def infer_config_from_state_dict(sdict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        d_model = None
        for key, tensor in sdict.items():
            if key.endswith('neuron_scalar_decoder_head.1.weight') and tensor.ndim == 2:
                d_model = int(tensor.shape[1]); break
        if d_model is None:
            for key, tensor in sdict.items():
                if 'stimuli_encoder.0.weight' in key and tensor.ndim == 2:
                    d_model = int(tensor.shape[0]); break
        if d_model is None:
            raise ValueError('Unable to infer d_model')
        d_stimuli = 1
        for key, tensor in sdict.items():
            if 'stimuli_encoder.0.weight' in key and tensor.ndim == 2:
                d_stimuli = int(tensor.shape[1]); break
        layer_indices = set()
        for k in sdict.keys():
            if k.startswith('layers.'):
                try:
                    layer_indices.add(int(k.split('.')[1]))
                except Exception:
                    pass
        n_layers = max(layer_indices) + 1 if layer_indices else 1
        n_heads = 8 if d_model % 8 == 0 else (4 if d_model % 4 == 0 else 2)
        return {'d_model': d_model, 'd_stimuli': d_stimuli, 'n_layers': n_layers, 'n_heads': n_heads}

    if not isinstance(model_cfg, dict) or any(k not in model_cfg for k in ('d_model', 'n_heads', 'n_layers')) or model_cfg.get('d_stimuli') in (None, 0):
        cfg_inf = infer_config_from_state_dict(sd)
        d_model, d_stimuli, n_layers, n_heads = cfg_inf['d_model'], cfg_inf['d_stimuli'], cfg_inf['n_layers'], cfg_inf['n_heads']
    else:
        d_model = model_cfg['d_model']; d_stimuli = model_cfg.get('d_stimuli', 1)
        n_layers = model_cfg['n_layers']; n_heads = model_cfg['n_heads']

    model = GBM(d_model=d_model, d_stimuli=d_stimuli, n_heads=n_heads, n_layers=n_layers).to(device)
    try:
        model.spikes_are_rates = bool(spikes_are_rates)
    except Exception:
        model.spikes_are_rates = False
    # Pre-shape centroid buffers if present
    try:
        for k, v in list(sd.items()):
            if not k.endswith('.centroids'):
                continue
            module_path = k.rsplit('.', 1)[0]
            cur = model
            for part in module_path.split('.'):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    try:
                        cur = cur[int(part)]
                    except Exception:
                        cur = None; break
            if cur is not None and hasattr(cur, 'centroids'):
                setattr(cur, 'centroids', v.detach().clone().to(device=next(model.parameters()).device))
    except Exception:
        pass
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def list_dim_dirs(base_path: Path) -> List[int]:
    """Return available behavior dims from a BModel training run.
    Accepts either the experiment dir or the checkpoints dir.
    """
    # If user passed the experiment root, look under 'checkpoints'
    ckpt_root = base_path / 'checkpoints'
    # If they passed the checkpoints directory directly, use it
    if base_path.name == 'checkpoints' and base_path.is_dir():
        ckpt_root = base_path
    out: List[int] = []
    if ckpt_root.exists():
        for p in ckpt_root.iterdir():
            if p.is_dir() and p.name.startswith('dim_'):
                try:
                    out.append(int(p.name.split('_', 1)[1]))
                except Exception:
                    pass
    return sorted(set(out))


def last_epoch_checkpoint(dim_dir: Path) -> Optional[Path]:
    epoch_files = list(dim_dir.glob('epoch_*.pth'))
    best = dim_dir / 'best.pth'
    last: Optional[Path] = None
    best_ep = -1
    for fp in epoch_files:
        import re
        m = re.search(r"epoch_(\d+)\.pth$", fp.name)
        if not m:
            continue
        try:
            ep = int(m.group(1))
        except Exception:
            continue
        if ep > best_ep:
            best_ep = ep; last = fp
    if last is None and best.exists():
        last = best
    return last


def load_bmodels(train_dir: Path, device: torch.device, dims: List[int]) -> List[Tuple[int, BModel]]:
    out: List[Tuple[int, BModel]] = []
    # Accept either experiment dir or its checkpoints dir
    ckpt_root = train_dir / 'checkpoints'
    if train_dir.name == 'checkpoints' and train_dir.is_dir():
        ckpt_root = train_dir
    for d in dims:
        ddir = ckpt_root / f'dim_{d}'
        ck = last_epoch_checkpoint(ddir)
        if ck is None:
            continue
        state = torch.load(ck, map_location=device, weights_only=False)
        sd_in = state['model'] if (isinstance(state, dict) and 'model' in state) else state
        def _normalize_key(k: str) -> str:
            changed = True
            while changed:
                changed = False
                if k.startswith('module.'):
                    k = k[len('module.'):]; changed = True
                if k.startswith('_orig_mod.'):
                    k = k[len('_orig_mod.'):]; changed = True
            return k
        sd = {_normalize_key(k): v for k, v in (sd_in.items() if isinstance(sd_in, dict) else sd_in)} if isinstance(sd_in, dict) else sd_in
        model = BModel(d_behavior=1, d_max_neurons=0).to(device)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            model.load_state_dict({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}, strict=False)
        model.eval()
        # Freeze weights; we only optimize the activation deltas
        for p in model.parameters():
            p.requires_grad_(False)
        out.append((d, model))
    if not out:
        # Provide a helpful error with directory listing
        try:
            children = [p.name for p in ckpt_root.iterdir()]
        except Exception:
            children = []
        raise RuntimeError(f'No BModel checkpoints loaded in {ckpt_root}. Found entries: {children}')
    return out


def sliding_predict_bmodel(seq_BTN: torch.Tensor, positions_BN3: torch.Tensor, mask_BN: torch.Tensor, models: List[Tuple[int, BModel]], window_len: int = 6) -> torch.Tensor:
    """Predict behaviors per time step using rolling windows (reflect padding).
    Returns (T, K_use) on the same device as inputs and preserves autograd so
    gradients can flow to seq_BTN (for delta optimization).
    """
    B, T, N = seq_BTN.shape
    K_use = len(models)
    pad = window_len - 1
    x = seq_BTN[0]  # (T, N)
    x_pad = torch.cat([torch.flip(x[:pad], dims=[0]), x], dim=0)  # (T+pad, N)
    pos = positions_BN3[0:1]
    msk = mask_BN[0:1]
    step_preds: List[torch.Tensor] = []
    for t in range(T):
        win = x_pad[t:t + window_len]
        win_b = win.unsqueeze(0)  # (1, L, N)
        dim_out: List[torch.Tensor] = []
        for _, mdl in models:
            dtype_m = next(mdl.parameters()).dtype
            dim_out.append(
                mdl(win_b.to(dtype=dtype_m), pos.to(dtype=dtype_m), msk, get_logits=True)
                .squeeze(1)
                .squeeze(-1)
            )
        step_preds.append(torch.stack(dim_out, dim=0).view(-1))  # (K_use,)
    preds = torch.stack(step_preds, dim=0)  # (T, K_use)
    return preds


def build_swirl(T: int, base_radius: float = 0.2, turns: float = 1.5, growth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a small 2D swirl (spiral) path of T steps.
    - base_radius: initial radius (small)
    - turns: number of rotations over T
    - growth: radial growth factor (0 → constant radius; >0 → slowly expanding)
    Returns (x[T], y[T])."""
    t = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    theta = (2.0 * math.pi * turns) * t
    r = base_radius * (1.0 + growth * t)
    x = (r * np.cos(theta)).astype(np.float32)
    y = (r * np.sin(theta)).astype(np.float32)
    # Start at origin (0,0)
    x = x - x[0]
    y = y - y[0]
    return x, y


def path_to_behaviors(x: np.ndarray, y: np.ndarray, speed_scale: float, turn_scale: float) -> np.ndarray:
    """Convert (x,y) path to 5-dim behavior targets per step; dims order:
       [L, R, B, LU, RU] = [0,4,3,1,2].
       Bilateral: forward speed (normed by speed_scale). Heading rate -> RU-LU via turn_scale.
    """
    T = x.shape[0]
    dx = np.diff(x, prepend=x[0]).astype(np.float32)
    dy = np.diff(y, prepend=y[0]).astype(np.float32)
    speed = np.sqrt(dx * dx + dy * dy) / max(1e-6, float(speed_scale))
    speed = np.clip(speed, 0.0, 1.0).astype(np.float32)
    # Heading angle
    theta = np.arctan2(np.maximum(1e-12, dy), np.maximum(1e-12, dx)).astype(np.float32)
    dtheta = np.diff(theta, prepend=theta[0]).astype(np.float32)
    delta = dtheta / max(1e-6, float(turn_scale))  # desired RU-LU
    # Center to keep within [-1,1]
    delta = np.clip(delta, -1.0, 1.0).astype(np.float32)
    RU = np.clip(+0.5 * delta, -1.0, 1.0)
    LU = np.clip(-0.5 * delta, -1.0, 1.0)
    B = speed
    # Convert LU/RU to 0..1 contributions relative to B: L = B + LU, R = B + RU then clip
    L = np.clip(B + LU, 0.0, 1.0)
    R = np.clip(B + RU, 0.0, 1.0)
    # Assemble [L, R, B, LU, RU] then remap to K=5 indices [0..4] with mapping
    # mapping: L:0, R:4, B:3, LU:1, RU:2
    K = 5
    out = np.zeros((T, K), dtype=np.float32)
    out[:, 0] = L
    out[:, 4] = R
    out[:, 3] = B
    out[:, 1] = np.clip(B + LU, 0.0, 1.0) - B  # LU contribution around B retained via difference
    out[:, 2] = np.clip(B + RU, 0.0, 1.0) - B
    return out


def integrate_trajectory(B: np.ndarray, LU: np.ndarray, RU: np.ndarray, dt: float, turn_scale_angle: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    T = B.shape[0]
    x = np.zeros((T,), dtype=np.float32)
    y = np.zeros((T,), dtype=np.float32)
    th = np.zeros((T,), dtype=np.float32)
    omega = turn_scale_angle * (RU - LU).astype(np.float32)
    for t in range(T - 1):
        th[t + 1] = th[t] + omega[t] * dt
        x[t + 1] = x[t] + B[t] * math.cos(float(th[t])) * dt
        y[t + 1] = y[t] + B[t] * math.sin(float(th[t])) * dt
    return x, y


def main():
    ap = argparse.ArgumentParser(description='Optimize GBM neural sequence to match a target 2D trajectory via BModel behaviors')
    ap.add_argument('--data-dir', type=str, default='processed_spike_voxels_2018')
    ap.add_argument('--subject', type=str, default='subject_14')
    ap.add_argument('--gbm-checkpoint', type=str, required=True)
    ap.add_argument('--bmodel-train-dir', type=str, required=True, help='Directory containing per-dim checkpoints in checkpoints/dim_*/')
    ap.add_argument('--context-len', type=int, default=12)
    ap.add_argument('--horizon', type=int, default=128)
    ap.add_argument('--speed-scale', type=float, default=0.1, help='Divisor scaling for forward speed normalization')
    ap.add_argument('--turn-scale', type=float, default=0.2, help='Divisor scaling for heading-rate normalization')
    ap.add_argument('--l1-delta', type=float, default=0.0, help='L1 penalty weight on positive delta')
    ap.add_argument('--steps', type=int, default=400, help='Optimization steps')
    ap.add_argument('--lr', type=float, default=1e-1, help='Learning rate for delta optimizer')
    # Always use GPU + bf16 per request; will error if CUDA unavailable
    ap.add_argument('--out-dir', type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this script. Please run on a GPU machine.')
    device = torch.device('cuda')

    # Allow Muon optimizer to run in single-process mode without initializing DDP
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

    # Output dirs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_out = Path(args.out_dir) if args.out_dir else Path('experiments/trajectory_optimization') / ts
    logs_dir = base_out / 'logs'; data_dir_out = base_out / 'data'
    for p in (base_out, logs_dir, data_dir_out): p.mkdir(parents=True, exist_ok=True)

    # Load context and metadata
    subj_h5 = Path(args.data_dir) / f"{args.subject}.h5"
    spikes_ctx_np, stim_ctx_np, pos_np, mask_np, K_stim = load_context_from_subject(subj_h5, args.context_len)
    dt_hz = float('nan')
    try:
        with h5py.File(subj_h5, 'r') as f:
            sr = _read_sampling_rate_hz(f)
            dt_hz = (1.0 / sr) if (np.isfinite(sr) and sr > 0) else 1.0
    except Exception:
        dt_hz = 1.0

    # Load GBM
    gbm = load_gbm_from_checkpoint(Path(args.gbm_checkpoint), device, spikes_are_rates=False)
    gbm = gbm.to(dtype=torch.bfloat16)
    # Freeze GBM parameters; we only optimize delta on inputs
    for p in gbm.parameters():
        p.requires_grad_(False)
    spikes_ctx = torch.from_numpy(spikes_ctx_np).to(device, dtype=torch.bfloat16)
    stim_ctx = torch.from_numpy(stim_ctx_np).to(device, dtype=torch.bfloat16)
    pos = torch.from_numpy(pos_np).to(device, dtype=torch.bfloat16)
    mask = torch.from_numpy(mask_np).to(device)  # keep as float/bool mask; used as !=0

    # Ensure stimulus feature width matches model.d_stimuli (pad/truncate)
    dS = int(getattr(gbm, 'd_stimuli', K_stim or 1))
    # Pad/truncate context stimulus to dS
    L_ctx = stim_ctx.shape[1]
    stim_ctx_fixed = torch.zeros((1, L_ctx, dS), device=device, dtype=torch.bfloat16)
    copy_w = min(int(stim_ctx.shape[2]), dS)
    if copy_w > 0:
        stim_ctx_fixed[:, :, :copy_w] = stim_ctx.to(dtype=stim_ctx_fixed.dtype)[:, :, :copy_w]
    stim_ctx = stim_ctx_fixed

    # Future stimulus zeros with correct width
    fut_stim = torch.zeros((1, int(args.horizon), dS), device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        full_seq = gbm.autoregress(
            init_x=spikes_ctx, init_stimuli=stim_ctx, point_positions=pos, neuron_pad_mask=mask,
            future_stimuli=fut_stim, n_steps=int(args.horizon), context_len=int(args.context_len)
        )  # (1, L+T, N)
    gen_only = full_seq[:, -int(args.horizon):, :]  # (1,T,N), stays bf16 on CUDA

    # Load BModels per behavior dimension
    dims_all = list_dim_dirs(Path(args.bmodel_train_dir))
    if not dims_all:
        raise RuntimeError('No BModel dim_* directories found')
    # Use first 5 dims if available
    dims_use = dims_all[:5]
    bmodels = load_bmodels(Path(args.bmodel_train_dir), device, dims_use)
    # Ensure BModels are bf16 on CUDA
    bmodels = [(d, m.to(dtype=torch.bfloat16)) for d, m in bmodels]

    # Build target behaviors from a small swirl path
    x_tgt, y_tgt = build_swirl(int(args.horizon), base_radius=0.2, turns=1.5, growth=0.3)
    beh_tgt = path_to_behaviors(x_tgt, y_tgt, speed_scale=float(args.speed_scale), turn_scale=float(args.turn_scale))  # (T,5)

    # Initial predictions from BModel on GBM output
    with torch.no_grad():
        preds_init = sliding_predict_bmodel(gen_only, pos, mask, bmodels, window_len=6)  # (T, K_use)

    # Optimize positive-only per-neuron, per-time delta
    B, T, N = gen_only.shape
    L = int(spikes_ctx.shape[1])
    # Keep optimization master param in fp32 for numerical stability; cast to bf16 only for model calls
    # Apply delta over the L-step pre-AR context (not the generated horizon)
    delta_param = torch.zeros((L, N), device=device, dtype=torch.float32, requires_grad=True)
    # Use Muon optimizer (as in train_gbm.py). Requires `muon` to be installed.
    try:
        from muon import MuonWithAuxAdam
    except ImportError as e:
        raise ImportError("Muon optimizer not found. Install: pip install git+https://github.com/KellerJordan/Muon") from e
    # Single param group using Muon (no AdamW fallback as requested)
    opt = MuonWithAuxAdam([
        dict(params=[delta_param], use_muon=True, lr=float(args.lr), weight_decay=0.0)
    ])
    l1_w = float(args.l1_delta)
    beh_tgt_t = torch.from_numpy(beh_tgt).to(device=device, dtype=torch.float32)

    def predict_with_delta(delta_LN: torch.Tensor) -> torch.Tensor:
        # Modify only the context spikes, then autoregress to produce horizon with stop-grad through time
        delta_pos = F.softplus(delta_LN.to(dtype=torch.float32)) * mask[0].to(dtype=torch.float32)  # (L,N)
        spikes_ctx_mod = torch.clamp(spikes_ctx[0].to(dtype=torch.float32) + delta_pos, 0.0, 1.0).to(dtype=torch.bfloat16).unsqueeze(0)
        cur_x = spikes_ctx_mod  # (1, L, N)
        cur_stim = stim_ctx
        steps: List[torch.Tensor] = []
        T_h = int(args.horizon)
        L_ctx = int(args.context_len)
        # For small horizons, allow full BPTT (no detach) to improve learning signal
        allow_full_bptt = (T_h <= max(24, L_ctx))
        for i in range(T_h):
            ctx_x = cur_x[:, -L_ctx:, :]
            ctx_s = cur_stim[:, -L_ctx:, :]
            # Single-step forward with gradients w.r.t. ctx_x using checkpointing to save memory
            def _step(x_ctx, s_ctx):
                return gbm.forward(x_ctx, s_ctx, pos, mask, get_logits=False)[:, -1:, :]
            try:
                nxt = checkpoint(_step, ctx_x, ctx_s, use_reentrant=False)
            except TypeError:
                nxt = checkpoint(_step, ctx_x, ctx_s)
            steps.append(nxt.squeeze(1))  # (1,N)
            # Detach only when horizon is large; for small horizons keep graph for stronger signal
            cur_x = torch.cat([cur_x, (nxt if allow_full_bptt else nxt.detach())], dim=1)
            cur_stim = torch.cat([cur_stim, fut_stim[:, i:i+1, :]], dim=1)
        gen_only_mod = torch.stack(steps, dim=1)  # (1, T, N)
        preds = sliding_predict_bmodel(gen_only_mod, pos, mask, bmodels, window_len=6)  # (T,K)
        return preds

    pbar = tqdm(range(int(args.steps)), desc='Optimize deltas')
    best_loss = float('inf')
    best_delta: Optional[torch.Tensor] = None
    # Loss histories for plotting
    loss_hist_total: list[float] = []
    loss_hist_pred: list[float] = []
    loss_hist_l1: list[float] = []
    for _ in pbar:
        opt.zero_grad(set_to_none=True)
        preds = predict_with_delta(delta_param)  # (T,K) with gradients
        preds_t = preds.to(device=device, dtype=torch.float32)
        # Loss over selected dims only (mapping indices) – use float32 target/preds
        keep_idx = torch.tensor([0, 4, 3, 1, 2], device=device, dtype=torch.long)
        # Use L1 loss for behavior matching
        loss_pred = F.l1_loss(preds_t[:, keep_idx], beh_tgt_t[:, keep_idx])
        # L1 on positive delta (after softplus) but compute via param proxy to keep gradients smooth
        l1_penalty = torch.mean(F.softplus(delta_param))
        l1_term = l1_w * l1_penalty
        total = loss_pred + l1_term
        total.backward()
        opt.step()
        cur = float(total.detach().cpu().item())
        # Record losses
        loss_hist_total.append(cur)
        loss_hist_pred.append(float(loss_pred.detach().cpu().item()))
        loss_hist_l1.append(float(l1_term.detach().cpu().item()))
        pbar.set_postfix({'loss': f"{cur:.6f}"})
        if cur < best_loss:
            best_loss = cur
            best_delta = delta_param.detach().cpu().clone()

    if best_delta is None:
        best_delta = delta_param.detach().cpu().clone()

    # Compose modified context spikes and save
    best_delta_pos = F.softplus(best_delta.to(device=device, dtype=torch.float32)) * mask[0].to(device=device, dtype=torch.float32)
    spikes_ctx_mod = torch.clamp(spikes_ctx[0].to(dtype=torch.float32) + best_delta_pos, 0.0, 1.0).detach().cpu().numpy()  # (L,N)
    # Also generate horizon spikes from modified context for reference
    with torch.no_grad():
        spikes_ctx_mod_bf16 = torch.from_numpy(spikes_ctx_mod).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
        full_seq_best = gbm.autoregress(
            init_x=spikes_ctx_mod_bf16, init_stimuli=stim_ctx, point_positions=pos, neuron_pad_mask=mask,
            future_stimuli=fut_stim, n_steps=int(args.horizon), context_len=int(args.context_len)
        )
        gen_only_best = full_seq_best[:, -int(args.horizon):, :].detach().to(torch.float32).cpu().numpy()
    preds_final = (
        predict_with_delta(best_delta.to(device=device, dtype=torch.float32))
        .detach()
        .to(torch.float32)
        .cpu()
        .numpy()
    )  # (T,K)

    # Save arrays
    np.savez_compressed(
        data_dir_out / 'results.npz',
        target_behaviors=beh_tgt,
        preds_init=preds_init.detach().to(torch.float32).cpu().numpy(),
        preds_final=preds_final,
        gen_spikes=gen_only[0].detach().to(torch.float32).cpu().numpy(),
        delta_best=best_delta.detach().cpu().numpy(),
        spikes_modified=spikes_ctx_mod,
        gen_spikes_mod=gen_only_best[0],
        positions=pos[0].detach().to(torch.float32).cpu().numpy(),
        neuron_mask=mask[0].detach().cpu().numpy(),
        swirl_x=x_tgt, swirl_y=y_tgt,
    )

    # Plot trajectories: target vs achieved (from preds_final behaviors)
    try:
        import matplotlib.pyplot as plt
        # Integrate predicted final behaviors into positions
        B_idx, LU_idx, RU_idx = 3, 1, 2
        x_pred, y_pred = integrate_trajectory(preds_final[:, B_idx], preds_final[:, LU_idx], preds_final[:, RU_idx], dt=dt_hz, turn_scale_angle=1.0)
        plt.figure(figsize=(7, 7), dpi=220)
        plt.plot(x_tgt, y_tgt, label='target', color='tab:green', linewidth=2)
        plt.plot(x_pred, y_pred, label='achieved', color='tab:orange', linewidth=2)
        plt.axis('equal'); plt.grid(alpha=0.3)
        plt.title('2D trajectory (swirl): target vs achieved')
        plt.legend()
        plt.tight_layout()
        plt.savefig(logs_dir / 'trajectory_compare.png', dpi=300)
        plt.close()
        # Behavior traces
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=220)
        t = np.arange(int(args.horizon))
        axs[0].plot(t, beh_tgt[:, 3], label='B target'); axs[0].plot(t, preds_final[:, 3], label='B pred', alpha=0.8)
        axs[0].set_ylabel('Bilateral'); axs[0].legend()
        axs[1].plot(t, beh_tgt[:, 1], label='LU target'); axs[1].plot(t, preds_final[:, 1], label='LU pred', alpha=0.8)
        axs[1].set_ylabel('Left-unilateral'); axs[1].legend()
        axs[2].plot(t, beh_tgt[:, 2], label='RU target'); axs[2].plot(t, preds_final[:, 2], label='RU pred', alpha=0.8)
        axs[2].set_ylabel('Right-unilateral'); axs[2].legend(); axs[2].set_xlabel('time step')
        fig.tight_layout(); fig.savefig(logs_dir / 'behaviors_compare.png', dpi=300); plt.close(fig)
        # Optimization loss curve (total, pred, l1 components)
        try:
            steps = np.arange(len(loss_hist_total))
            plt.figure(figsize=(8, 4), dpi=200)
            plt.plot(steps, loss_hist_total, label='total', linewidth=1.5)
            plt.plot(steps, loss_hist_pred, label='pred_l1', linewidth=1.0)
            plt.plot(steps, loss_hist_l1, label='l1_term', linewidth=1.0)
            plt.xlabel('optimization step')
            plt.ylabel('loss')
            plt.title('Optimization loss over steps')
            plt.legend()
            plt.tight_layout()
            plt.savefig(logs_dir / 'optimization_loss.png', dpi=300)
            plt.close()
            # Also save CSV
            try:
                with open(logs_dir / 'optimization_loss.csv', 'w') as fcsv:
                    fcsv.write('step,total,pred_l1,l1_term\n')
                    for i in range(len(loss_hist_total)):
                        fcsv.write(f"{i},{loss_hist_total[i]:.8f},{loss_hist_pred[i]:.8f},{loss_hist_l1[i]:.8f}\n")
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass

    # Write a simple summary
    with open(base_out / 'summary.txt', 'w') as f:
        f.write(f"Best loss: {best_loss:.6f}\n")
        f.write(f"Context len: {args.context_len}, Horizon: {args.horizon}\n")
        f.write(f"Speed scale: {args.speed_scale}, Turn scale: {args.turn_scale}, L1 delta: {args.l1_delta}\n")
        f.write(f"GBM: {args.gbm_checkpoint}\nBModel train dir: {args.bmodel_train_dir}\n")


if __name__ == '__main__':
    main()


