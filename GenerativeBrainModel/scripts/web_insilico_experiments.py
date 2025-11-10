#!/usr/bin/env python3
"""
Web In-Silico Experiments for GBM

- Loads a trained GBM and a good AR sample H5 produced by eval_gbm.py
- Serves a Flask + Three.js UI to visualize initial context and AR outputs at 3 Hz
- Lets users select brain regions and add a spike-rate delta to the last context frame
- Clips the full initial sequence to [0,1] before AR (when configured)
- Batches up to 5 experiment requests and runs them together on the GPU
- Returns results, allows NPZ download and per-region relative activity heatmap
"""

from __future__ import annotations

import os
import uuid
import json
import time
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template_string, send_file, session, make_response
import yaml
import requests
import shutil

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders  # for types only; not used directly
from GenerativeBrainModel.scripts.eval_gbm2_ar import autoregressive_rollout as eval_ar_rollout

# --------------------------
# Config and global state
# --------------------------

@dataclass
class ServerConfig:
    host: str = '0.0.0.0'
    port: int = 8054
    debug: bool = False
    secret_key: str = 'change-me'
    turnstile_site_key: str = ''  # optional; if set, UI renders Turnstile widget
    disable_verification: bool = False  # if True, skip all verification paths

@dataclass
class DataConfig:
    ar_h5: str = ''            # Path to eval_gbm2_ar per-AR H5 (context, truth, pred, stim, ids, etc.)
    mask_h5: str = ''          # Optional atlas H5 for region overlays
    spikes_are_rates: bool = False
    sampling_rate_hz: float = 3.0
    clip_01_before_ar: bool = False
    baseline_npz: str = ''     # Optional; if empty, will fall back to AR H5 future truth rates

@dataclass
class ModelConfig:
    checkpoint: str = ''       # If empty, derived from AR H5 experiment dir
    use_gpu: bool = True
    dtype: str = 'bfloat16'  # 'float32' | 'bfloat16'

@dataclass
class ARConfig:
    sequence_length: int = 12
    default_ar_steps: int = 12

@dataclass
class UIConfig:
    max_neurons_render: int = 100000
    max_batch_size: int = 5

@dataclass
class StorageConfig:
    output_root: str = 'experiments/insilico_web'

@dataclass
class AppConfig:
    server: ServerConfig
    data: DataConfig
    model: ModelConfig
    ar: ARConfig
    ui: UIConfig
    storage: StorageConfig


def load_yaml_config(path: str) -> AppConfig:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    server = ServerConfig(**cfg.get('server', {}))
    data = DataConfig(**cfg.get('data', {}))
    model = ModelConfig(**cfg.get('model', {}))
    ar = ARConfig(**cfg.get('ar', {}))
    ui = UIConfig(**cfg.get('ui', {}))
    storage = StorageConfig(**cfg.get('storage', {}))
    return AppConfig(server=server, data=data, model=model, ar=ar, ui=ui, storage=storage)


# --------------------------
# Robust model loading (adapted from eval_gbm)
# --------------------------

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
    model_sd = model.state_dict()
    state = _strip_orig_mod_prefix(state)
    filtered = {k: v for k, v in state.items() if (k in model_sd and getattr(v, 'shape', None) == model_sd[k].shape)}
    if len(filtered) == 0:
        raise RuntimeError('No matching parameters between checkpoint and current model.')
    model.load_state_dict(filtered, strict=False)


def find_experiment_root_from_ar_h5(ar_h5_path: str) -> str:
    """Ascend from the AR H5 path until a directory containing config.yaml and checkpoints/ is found."""
    cur = os.path.abspath(os.path.dirname(ar_h5_path))
    while True:
        if os.path.isfile(os.path.join(cur, 'config.yaml')) and os.path.isdir(os.path.join(cur, 'checkpoints')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise ValueError(f"Could not find experiment root from AR H5 path: {ar_h5_path}")


def load_model_from_experiment(exp_dir: str, device: torch.device, d_stimuli: int) -> GBM:
    """Build GBM using experiment config and checkpoint, ensuring shapes match."""
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = state.get('model', state)
    # Determine embedding size and the EXACT global ID mapping used during training
    ne_key = None
    gid_key = None
    stim_w_key = None
    for k in sd.keys():
        if k.endswith('neuron_embed.weight'):
            ne_key = k
        if k.endswith('global_neuron_ids_sorted'):
            gid_key = k
        if k.endswith('stimuli_encoder.0.weight'):
            stim_w_key = k
    if ne_key is None:
        raise ValueError("Checkpoint is missing neuron_embed.weight")
    # Use the saved mapping; this is critical so searchsorted finds the right indices
    if gid_key is None:
        raise ValueError("Checkpoint is missing global_neuron_ids_sorted buffer needed for correct neuron ID mapping")
    global_ids_tensor = sd[gid_key].to(torch.long)
    if global_ids_tensor.ndim != 1 or global_ids_tensor.numel() < 1:
        raise ValueError("global_neuron_ids_sorted in checkpoint is malformed")
    # Enforce d_stimuli to EXACTLY match checkpoint to avoid random init due to shape mismatch
    if stim_w_key is not None:
        try:
            d_stimuli_ckpt = int(sd[stim_w_key].shape[1])
            d_stimuli = d_stimuli_ckpt
        except Exception:
            pass
    d_model = int(cfg['model']['d_model'])
    n_heads = int(cfg['model']['n_heads'])
    n_layers = int(cfg['model']['n_layers'])
    cov_rank = int(cfg['model'].get('cov_rank', 32))
    num_neurons_total = int(cfg['model'].get('num_neurons_total', 4_000_000))
    model = GBM(
        d_model=d_model, d_stimuli=int(d_stimuli),
        n_heads=n_heads, n_layers=n_layers,
        num_neurons_total=num_neurons_total,
        global_neuron_ids=global_ids_tensor,
        cov_rank=cov_rank
    ).to(device)
    _load_state_dict_robust(model, sd)
    model.eval()
    return model


# --------------------------
# Data loading utilities
# --------------------------

@dataclass
class BestSample:
    initial_context: np.ndarray  # (1, L, N)
    pred_future: Optional[np.ndarray]  # (1, Tgen, N) or None
    positions_norm: np.ndarray  # (1, N, 3) or (N, 3)
    neuron_mask: np.ndarray  # (1, N) or (N,)
    stim_context: Optional[np.ndarray]  # (1, L, K) or None
    stim_future: Optional[np.ndarray]  # (1, Tgen, K) or None
    context_len: int
    base_n_steps: int
    neuron_ids: Optional[np.ndarray] = None
    log_activity_mean: Optional[np.ndarray] = None  # (1, N) or (N,)
    log_activity_std: Optional[np.ndarray] = None   # (1, N) or (N,)
    training_lm1: Optional[int] = None


# --------------------------
# Path resolution helpers
# --------------------------

def load_ar_run_h5(path: str) -> BestSample:
    """Load a per-AR run H5 produced by eval_gbm2_ar.py (ar_runs/*.h5)."""
    assert os.path.exists(path), f"AR H5 not found: {path}"
    with h5py.File(path, 'r') as f:
        required = ('context_rates', 'positions', 'neuron_mask')
        for r in required:
            if r not in f:
                raise ValueError(f"AR H5 missing dataset '{r}': {path}")
        ctx = f['context_rates'][()]           # (Tc, N)
        pos = f['positions'][()]               # (N, 3)
        mask = f['neuron_mask'][()]            # (N,)
        fut_truth = f['future_truth_rates'][()] if 'future_truth_rates' in f else None  # (Tf,N)
        fut_pred = f['future_pred_rates'][()] if 'future_pred_rates' in f else None     # (Tf,N)
        stim_ctx = f['stimulus_context'][()] if 'stimulus_context' in f else None       # (Tc,K)
        stim_fut = f['stimulus_future'][()] if 'stimulus_future' in f else None         # (Tf,K)
        neuron_ids = f['neuron_ids'][()] if 'neuron_ids' in f else None
        lam = f['log_activity_mean'][()] if 'log_activity_mean' in f else None
        las = f['log_activity_std'][()] if 'log_activity_std' in f else None
        Tc = int(f.attrs.get('Tc', ctx.shape[0]))
        Tf = int(f.attrs.get('Tf', fut_pred.shape[0] if isinstance(fut_pred, np.ndarray) else (fut_truth.shape[0] if isinstance(fut_truth, np.ndarray) else 0)))
        training_lm1 = int(f.attrs.get('training_Lm1', Tc))
    init_ctx = ctx[None, ...].astype(np.float32)             # (1,Tc,N)
    pos = pos[None, ...].astype(np.float32)                  # (1,N,3)
    mask = mask[None, ...].astype(bool)                      # (1,N)
    stim_ctx = None if stim_ctx is None else stim_ctx[None, ...].astype(np.float32)
    stim_fut = None if stim_fut is None else stim_fut[None, ...].astype(np.float32)
    fut_pred = None if fut_pred is None else fut_pred[None, ...].astype(np.float32)
    return BestSample(
        initial_context=init_ctx,
        pred_future=fut_pred,
        positions_norm=pos,
        neuron_mask=mask,
        stim_context=stim_ctx,
        stim_future=stim_fut,
        context_len=Tc,
        base_n_steps=Tf,
        neuron_ids=None if neuron_ids is None else neuron_ids.astype(np.int64),
        log_activity_mean=None if lam is None else lam[None, ...].astype(np.float32) if lam.ndim == 1 else lam.astype(np.float32),
        log_activity_std=None if las is None else las[None, ...].astype(np.float32) if las.ndim == 1 else las.astype(np.float32),
        training_lm1=training_lm1
    )


def resolve_experiment_from_ar_h5(path: str) -> Tuple[str, str]:
    """Return (exp_dir, ckpt_path) derived from AR H5 location."""
    exp_dir = find_experiment_root_from_ar_h5(path)
    ckpt = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    if not os.path.exists(ckpt):
        raise ValueError(f"Checkpoint not found under experiment: {ckpt}")
    return exp_dir, ckpt


@dataclass
class MaskData:
    label_volume: np.ndarray  # kept for compatibility, unused
    region_names: List[str]
    grid_shape: Tuple[int, int, int]


# --------------------------
# Experiment queue and worker
# --------------------------

@dataclass
class ExperimentJob:
    job_id: str
    session_id: str
    neuron_indices: List[int]
    add_rate_hz: float
    ar_steps: int
    status: str = 'pending'  # pending | running | complete | error
    error: Optional[str] = None
    result_npz_path: Optional[str] = None
    result_frames: Optional[np.ndarray] = None  # (Tgen, N)


class ExperimentQueue:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self._queue: List[ExperimentJob] = []
        self._lock = threading.Lock()

    def enqueue(self, job: ExperimentJob) -> int:
        with self._lock:
            self._queue.append(job)
            return len([j for j in self._queue if j.status == 'pending'])

    def get_pending_batch(self) -> List[ExperimentJob]:
        with self._lock:
            pending = [j for j in self._queue if j.status == 'pending']
            take = pending[: self.max_batch_size]
            for j in take:
                j.status = 'running'
            return take

    def find(self, job_id: str) -> Optional[ExperimentJob]:
        with self._lock:
            for j in self._queue:
                if j.job_id == job_id:
                    return j
            return None

    def position_in_queue(self, job_id: str) -> int:
        with self._lock:
            pending_ids = [j.job_id for j in self._queue if j.status == 'pending']
            try:
                idx = pending_ids.index(job_id)
                return idx + 1
            except ValueError:
                return 0


# --------------------------
# Flask app setup
# --------------------------

app = Flask(__name__)
# Secure session cookie flags
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

# Global runtime objects (populated in main)
CFG: AppConfig
DEVICE: torch.device
MODEL: Optional[GBM] = None
BEST: BestSample
MASK: MaskData
REGION_TO_NEURON: Dict[int, np.ndarray]
QUEUE_OBJ: ExperimentQueue
SESSION_TO_JOB: Dict[str, str] = {}
RUNTIME_LOCK = threading.Lock()
BASELINE_FUTURE: Optional[np.ndarray] = None  # (Tbase, N) baseline used for heatmap/NPZ


def cleanup_old_experiment_dirs(root_dir: str, max_age_hours: int = 4) -> None:
    """Delete subdirectories under root_dir older than max_age_hours.
    Safe-guards: only removes directories directly under root_dir.
    """
    try:
        now = time.time()
        cutoff = now - float(max_age_hours) * 3600.0
        if not os.path.isdir(root_dir):
            return
        for entry in os.listdir(root_dir):
            full = os.path.join(root_dir, entry)
            if not os.path.isdir(full):
                continue
            try:
                mtime = os.path.getmtime(full)
            except Exception:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(full)
                except Exception:
                    pass
    except Exception:
        pass


def to_torch(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def get_model_dtype(cfg_dtype: str) -> torch.dtype:
    if cfg_dtype == 'float32':
        return torch.float32
    if cfg_dtype == 'bfloat16':
        return torch.bfloat16
    return torch.float32


# --------------------------
# Background worker
# --------------------------

def worker_loop():
    global MODEL
    dtype = get_model_dtype(CFG.model.dtype)
    sampling_rate_hz = float(CFG.data.sampling_rate_hz)
    while True:
        batch = QUEUE_OBJ.get_pending_batch()
        if not batch:
            time.sleep(0.25)
            continue
        try:
            # Build batched init sequences
            L = BEST.initial_context.shape[1]
            N = BEST.initial_context.shape[2]
            B = len(batch)

            init_x = np.repeat(BEST.initial_context, repeats=B, axis=0)  # (B,L,N)
            init_stim = BEST.stim_context if BEST.stim_context is not None else np.zeros((1, L, MODEL.d_stimuli), dtype=np.float32)
            init_stim = np.repeat(init_stim, repeats=B, axis=0)
            neuron_mask = BEST.neuron_mask[0].astype(bool)
            # Build future stimulus per job
            future_stim_list: List[np.ndarray] = []
            requested_steps: List[int] = []
            # Prepare shared tensors
            ids_np = BEST.neuron_ids if BEST.neuron_ids is not None else np.arange(N, dtype=np.int64)[None, :]
            ids_np = np.repeat(ids_np[None, ...] if ids_np.ndim == 1 else ids_np, repeats=B, axis=0) if ids_np.ndim == 1 else np.repeat(ids_np, repeats=B, axis=0)
            lam_np = BEST.log_activity_mean if BEST.log_activity_mean is not None else None
            las_np = BEST.log_activity_std if BEST.log_activity_std is not None else None

            for b_idx, job in enumerate(batch):
                # Gather indices from selected regions
                if job.neuron_indices:
                    region_indices = np.array([int(i) for i in job.neuron_indices if 0 <= int(i) < N], dtype=np.int64)
                    if region_indices.size > 0:
                        region_indices = region_indices[neuron_mask[region_indices]]
                else:
                    region_indices = np.array([], dtype=np.int64)

                # Compute delta in rate (Hz) domain (clamped 0..10 from UI)
                delta = float(job.add_rate_hz)

                if region_indices.size > 0 and delta > 0.0:
                    init_x[b_idx, -1, region_indices] = init_x[b_idx, -1, region_indices] + delta

                if CFG.data.clip_01_before_ar:
                    init_x[b_idx] = np.clip(init_x[b_idx], 0.0, 1.0)

                n_steps = int(job.ar_steps)
                requested_steps.append(n_steps)
                if BEST.stim_future is not None and BEST.stim_future.shape[1] >= n_steps:
                    future_stim_list.append(BEST.stim_future[:, :n_steps, :][0])  # (n_steps, K)
                elif BEST.stim_future is not None and BEST.stim_future.shape[1] > 0:
                    pad_len = n_steps - BEST.stim_future.shape[1]
                    K = BEST.stim_future.shape[2]
                    pad = np.zeros((pad_len, K), dtype=np.float32)
                    future_stim_list.append(np.concatenate([BEST.stim_future[0], pad], axis=0))
                else:
                    # No stimulus available; zeros
                    future_stim_list.append(np.zeros((n_steps, MODEL.d_stimuli), dtype=np.float32))

            # Determine the maximum steps to run in one AR call; run per distinct n_steps groups to avoid masking
            steps_to_jobs: Dict[int, List[int]] = {}
            for idx, n in enumerate(requested_steps):
                steps_to_jobs.setdefault(n, []).append(idx)

            results_per_job: Dict[int, np.ndarray] = {}
            for n_steps, job_indices in steps_to_jobs.items():
                # Run exactly like eval for each job independently
                max_context_len = int(BEST.training_lm1) if isinstance(BEST.training_lm1, int) and BEST.training_lm1 > 0 else int(BEST.context_len)
                for job_index in job_indices:
                    init_context = torch.from_numpy(init_x[job_index:job_index+1]).to(device=DEVICE, dtype=torch.float32)  # (1,L,N)
                    stim_ctx_j = torch.from_numpy(init_stim[job_index:job_index+1]).to(device=DEVICE, dtype=torch.float32)  # (1,L,K)
                    fut_stim_j = torch.from_numpy(future_stim_list[job_index][None, ...]).to(device=DEVICE, dtype=torch.float32)
                    stim_full_j = torch.cat([stim_ctx_j, fut_stim_j], dim=1)  # (1, L+n_steps, K)
                    pos_j = to_torch(BEST.positions_norm[0:1], device=DEVICE, dtype=dtype)  # (1,N,3)
                    mask_j = torch.from_numpy(BEST.neuron_mask[0:1]).to(device=DEVICE)
                    ids_j = torch.from_numpy(ids_np[job_index:job_index+1]).to(device=DEVICE, dtype=torch.long)
                    if lam_np is None:
                        lam_j = None
                    else:
                        lam_arr = lam_np if lam_np.ndim == 2 else lam_np[None, ...]
                        lam_j = torch.from_numpy(lam_arr).to(device=DEVICE, dtype=torch.float32)
                    if las_np is None:
                        las_j = None
                    else:
                        las_arr = las_np if las_np.ndim == 2 else las_np[None, ...]
                        las_j = torch.from_numpy(las_arr).to(device=DEVICE, dtype=torch.float32)
                    pred_rates, _ctx_counts, _pred_counts = eval_ar_rollout(
                        MODEL, init_context, stim_full_j, pos_j, mask_j, ids_j,
                        lam_j if lam_j is not None else None,
                        las_j if las_j is not None else None,
                        DEVICE, n_steps, sampling_rate_hz, max_context_len
                    )
                    results_per_job[job_index] = pred_rates.detach().cpu().numpy().astype(np.float32)

            # Persist outputs per job
            for j_idx, job in enumerate(batch):
                try:
                    gen_future = results_per_job[j_idx]  # (n_steps, N)
                    job.result_frames = gen_future
                    # Save NPZ: make directory unique with timestamp + job_id (never inside experiment dir)
                    out_dir = os.path.join(CFG.storage.output_root, f"{time.strftime('%Y%m%d_%H%M%S')}_{job.job_id}")
                    os.makedirs(out_dir, exist_ok=True)
                    npz_path = os.path.join(out_dir, f'{job.job_id}.npz')
                    np.savez_compressed(
                        npz_path,
                        initial_context=BEST.initial_context[0],
                        generated_future=gen_future,
                        baseline_future=(BASELINE_FUTURE if BASELINE_FUTURE is not None else (BEST.pred_future[0] if BEST.pred_future is not None else np.zeros((0, BEST.initial_context.shape[2]), dtype=np.float32))),
                        positions=BEST.positions_norm[0],
                        neuron_mask=BEST.neuron_mask[0],
                        selected_neurons=np.array(job.neuron_indices, dtype=np.int32),
                        add_rate_hz=np.array([job.add_rate_hz], dtype=np.float32),
                        sampling_rate_hz=np.array([CFG.data.sampling_rate_hz], dtype=np.float32),
                    )
                    job.result_npz_path = npz_path
                    job.status = 'complete'
                except Exception as e:
                    job.status = 'error'
                    job.error = str(e)
            # Periodic cleanup after processing a batch job
            try:
                cleanup_old_experiment_dirs(CFG.storage.output_root, max_age_hours=24)
            except Exception:
                pass
        except Exception as e:
            # Mark all as error
            for job in batch:
                job.status = 'error'
                job.error = str(e)


# --------------------------
# Flask routes
# --------------------------

@app.route('/')
def index():
    import time as _time
    ts = int(_time.time())
    html_template = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Zebrafish Brain In-Silico Experiment Platform - v{{ ts }}</title>
  <style>
    body { margin: 0; padding: 0; overflow: hidden; font-family: Arial, sans-serif; }
    #container { width: 100vw; height: 100vh; display: flex; }
    #controls { width: 360px; background: transparent; color: #d6ecff; padding: 15px; overflow-y: auto; box-sizing: border-box; position: relative; z-index: 1001; }
    #viewport { flex: 1; position: relative; }
    #canvas { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; display: block; z-index: 0; }
    .control-group { margin-bottom: 18px; }
    .control-group h3 { margin: 0 0 10px 0; color: #00e8ff; }
    label { font-size: 12px; }
    input[type="range"] { width: 100%; }
    button { background: linear-gradient(90deg, #0e2030, #0f2638); color: #ecf8ff; border: 1px solid #33c2ff; padding: 8px 12px; border-radius: 6px; cursor: pointer; margin: 2px; box-shadow: 0 0 10px rgba(51,194,255,0.28), 0 0 14px rgba(255,120,80,0.12); }
    button:hover { filter: brightness(1.07); box-shadow: 0 0 12px rgba(51,194,255,0.36), 0 0 16px rgba(255,120,80,0.20); border-color: #55d2ff; }
    .region-list { max-height: 240px; overflow-y: auto; border: 1px solid rgba(51,194,255,0.35); padding: 10px; border-radius: 6px; background: transparent; box-shadow: inset 0 0 12px rgba(255,120,80,0.08); }
    .region-item { margin: 3px 0; font-size: 11px; }
    #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); background: rgba(0,0,0,0.8); color: white; padding: 20px; border-radius: 10px; z-index: 1000; }
 
    /* Neon cyan sliders with subtle orange hints */
    input[type="range"] { -webkit-appearance: none; height: 6px; border-radius: 4px; background: #2a2a3a; outline: none; }
    input[type="range"]::-webkit-slider-runnable-track { height: 6px; border-radius: 4px; background: linear-gradient(90deg, rgba(0,232,255,0.7), rgba(102,204,255,0.45)); box-shadow: inset 0 0 6px rgba(255,120,80,0.12); }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #0e0e16; border: 2px solid #33c2ff; box-shadow: 0 0 10px rgba(51,194,255,0.34), 0 0 12px rgba(255,120,80,0.16); margin-top: -6px; }
    input[type="range"]::-moz-range-track { height: 6px; border-radius: 4px; background: linear-gradient(90deg, rgba(0,232,255,0.7), rgba(102,204,255,0.45)); box-shadow: inset 0 0 6px rgba(255,120,80,0.12); }
    input[type="range"]::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #0e0e16; border: 2px solid #33c2ff; box-shadow: 0 0 10px rgba(51,194,255,0.34), 0 0 12px rgba(255,120,80,0.16); }
 
    /* Checkbox accent color */
    input[type="checkbox"] { accent-color: #33c2ff; }
 
    /* Scrollbar styling for region list (WebKit) */
    .region-list::-webkit-scrollbar { width: 10px; }
    .region-list::-webkit-scrollbar-track { background: rgba(20,20,28,0.25); border-radius: 6px; }
    .region-list::-webkit-scrollbar-thumb { background: linear-gradient(180deg, rgba(0,191,255,0.7), rgba(102,204,255,0.5)); border-radius: 6px; box-shadow: inset 0 0 6px rgba(255,120,80,0.12); }
 
    /* Video controls bar styling */
    #video-controls { background: linear-gradient(0deg, rgba(10,10,14,0.92), rgba(10,10,14,0.68)); border-top: 1px solid #2a2a3a; }
    #time-display { color: #c8cce0; }
 
    /* Orange/red hint accents for interactivity (stronger, always visible on focus/active) */
    button:focus-visible, button:active { outline: 2px solid #ff6a3d; outline-offset: 2px; box-shadow: 0 0 12px rgba(255,106,61,0.40), 0 0 16px rgba(255,106,61,0.22); }
    input[type="range"]:focus-visible { box-shadow: 0 0 10px rgba(255,106,61,0.30), inset 0 0 8px rgba(255,106,61,0.18); }
    input[type="checkbox"]:focus-visible { outline: 2px solid #ff6a3d; outline-offset: 2px; }
    .region-list { transition: border-color 0.15s ease, box-shadow 0.15s ease; }
    .region-list:hover { border-color: rgba(255,106,61,0.55); box-shadow: inset 0 0 16px rgba(255,106,61,0.14), 0 0 10px rgba(51,194,255,0.14); }
    .region-item label { transition: color 0.12s ease, text-shadow 0.12s ease; }
    .region-item label:hover { color: #ffc2ae; text-shadow: 0 0 6px rgba(255,106,61,0.35); }
    .region-item input[type="checkbox"]:checked + label { color: #ffe0d6; text-shadow: 0 0 6px rgba(255,106,61,0.25); }
 
    /* Mobile layout */
    #toggle-controls { display: none; position: fixed; top: 10px; left: 10px; z-index: 1003; background: rgba(14,32,48,0.9); color: #ecf8ff; border: 1px solid #33c2ff; border-radius: 8px; padding: 8px 10px; font-size: 14px; }
    @media (max-width: 768px) {
      #controls { width: 100vw; max-height: 70vh; position: fixed; left: 0; top: 0; transform: translateY(-100%); transition: transform 200ms ease-in-out; padding: 12px 14px; background: rgba(0,0,0,0.65); backdrop-filter: blur(4px); }
      #controls.open { transform: translateY(0); }
      #toggle-controls { display: inline-block; }
      .control-group h3 { font-size: 16px; }
      button { padding: 10px 14px; font-size: 14px; }
      input[type="range"] { height: 8px; }
      input[type="range"]::-webkit-slider-thumb { width: 20px; height: 20px; margin-top: -8px; }
      input[type="range"]::-moz-range-thumb { width: 20px; height: 20px; }
      .region-list { max-height: 40vh; }
      #status-text { bottom: 72px; font-size: 18px; }
      #video-controls { gap: 10px; padding: 8px 10px; }
      #time-display { width: 140px; font-size: 13px; }
    }
    
    /* Invert-colors toggle */
    body.invert-colors { filter: invert(1) hue-rotate(180deg); }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  {% if ts_key %}
  <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
  {% endif %}
</head>
<body>
  <div id="container"> 
    <button id="toggle-controls">Menu</button>
    <div id="controls">
      <h2 style="margin-top:0;">Zebrafish Brain In-Silico Experiment Platform</h2>

      <div class="control-group">
        <h3>Model</h3>
        <div>
          <button id="open-preprint">Open Model Paper</button>
        </div>
      </div>

      <div class="control-group">
        <h3>Playback</h3>
         <div class="slider-container">
           <label for="point-size">Point Size</label>
           <input type="range" id="point-size" min="0.5" max="4.0" value="2.0" step="0.1" />
         </div>
         <div class="slider-container">
           <label for="opacity">Opacity</label>
           <input type="range" id="opacity" min="0.1" max="1.0" value="0.85" step="0.05" />
         </div>
         <div style="margin-top:8px;">
           <button id="toggle-invert">Invert Colors</button>
         </div>
       </div>

      <div class="control-group">
        <h3>Stimulation</h3>
        <div>
          <label>Spike rate delta (Hz):</label>
          <input type="number" id="rate-delta" value="0" step="0.1" min="0" max="10" />
        </div>
        <div class="slider-container" style="margin-top:6px;">
          <label for="stim-size">Cube size</label>
          <input type="range" id="stim-size" min="2" max="60" value="18" step="1" />
        </div>
        <div class="slider-container" style="margin-top:6px;">
          <label for="ar-steps">AR Steps: <span id="ar-steps-val">{{ ar_default }}</span></label>
          <input type="range" id="ar-steps" min="1" max="10" value="{{ ar_default }}" step="1" />
        </div>
        {% if ts_key %}
        <div style="margin:6px 0;">
          <div id="turnstile-widget" class="cf-turnstile" data-sitekey="{{ ts_key }}" data-theme="auto" data-action="run_experiment" data-callback="onTsSuccess" data-expired-callback="onTsExpired" data-error-callback="onTsError"></div>
        </div>
        {% endif %}
        <div>
          <button id="run-experiment">Run Experiment</button>
        </div>
        <div id="queue-status" style="margin-top:8px;font-size:12px;color:#ccc;">Queue: idle</div>
        <div id="result-links" style="display:none;margin-top:8px;">
          <button id="download-npz">Download initial and predicted activity data (NPZ)</button>
        </div>
      </div>

    </div>

    <div id="viewport" style="position:relative; z-index: 1;">
      <div id="loading">Loading...</div>
      <canvas id="canvas"></canvas>
        <div id="status-text" style="position:absolute; bottom:56px; left:50%; transform:translateX(-50%); color:#aaaaaa; font-size:24px; background:rgba(0,0,0,0.4); padding:4px 8px; border-radius:4px;">Viewing initial brain activity</div>
        <div id="instructions" style="position:absolute; bottom:8px; left:8px; max-width:320px; background:rgba(0,0,0,0.60); color:#ddd; font-size:11px; line-height:1.25; padding:6px 8px; border-radius:4px; box-shadow: 0 0 6px rgba(0,0,0,0.35); z-index: 1002;">
          <div style="font-weight:600; color:#f0f0f0; margin-bottom:4px;">Quick guide</div>
          <div style="margin:0;">
            • Right‑click to place/lock the cube. Use “Cube size” to adjust.<br/>
            • “Spike Δ” sets stimulation strength. “AR Steps” sets rollout length.<br/>
            • Run Experiment → generates future frames; Space toggles play/pause.<br/>
            • Reset to Initial → clears cube and returns to the context.
          </div>
        </div>
        <div id="video-controls" style="position:absolute; bottom:0; left:0; right:0; background:linear-gradient(0deg, rgba(10,10,14,0.92), rgba(10,10,14,0.68)); padding:6px 8px; display:flex; align-items:center; gap:8px; z-index: 1001;">
          <button id="play-pause">Pause</button>
          <div id="progress-wrap" style="position:relative; flex:1; display:flex; align-items:center;">
            <div id="progress-segments" style="position:absolute; left:6px; right:6px; height:10px; border-radius:5px; z-index:0; pointer-events:none; background:#0f0f1a; border:1px solid #333;">
              <div id="seg-initial" style="position:absolute; left:0; top:0; bottom:0; width:100%; background:linear-gradient(90deg, #4ea3ff, #66ccff); border-radius:5px; box-shadow: inset 0 0 4px rgba(78,163,255,0.5);"></div>
              <div id="seg-generated" style="position:absolute; right:0; top:0; bottom:0; width:0%; background:#bb33ff; border-radius:5px; box-shadow: 0 0 12px rgba(187,51,255,1.0); display:none;"></div>
            </div>
             <input type="range" id="progress" min="0" max="0" step="0.001" value="0" style="flex:1; position:relative; z-index:1; background:transparent;">
             <div id="progress-legend" style="position:absolute; left:6px; right:6px; bottom:-16px; display:flex; justify-content:space-between; font-size:11px; color:#c8cce0; pointer-events:none;">
               <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4ea3ff;margin-right:6px;"></span>Initial</span>
              <span>Generated<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#bb33ff;margin-left:6px;"></span></span>
             </div>
           </div>
          <span id="time-display" style="color:#c8cce0; font-size:12px; width:120px; text-align:right;">00:00 / 00:00</span>
        </div>
        <div style="position:absolute; top:8px; right:12px; z-index:1002;">
          <button id="reset-view">Reset to Initial</button>
        </div>
      </div>

  <script>
    let scene, camera, renderer, controls, points, positionsAttr, colorsAttr, geometry, material;
    let positions = null; // [[x,y,z], ...] scaled for hard-coded anisotropy
    let neuronMask = null; // [bool]
    let regionNames = [];
    let regionMeshes = {}; // regionId -> Points
    let initialFrames = null; // [[N], ...]
    let generatedFrames = null; // [[N], ...]
    let playbackFrames = null;
    let isPlaying = true;
    let playbackTimeSec = 0.0;
    let totalDurationSec = 0.0;
    const fps = {{ fps }};
    const verificationEnabled = {{ 1 if ts_key else 0 }};
    let stimCube = null;
    let stimCenter = null;
    let stimSize = 18.0; // cube edge length (same units as positions after scaling)
    let stimLocked = false;
    let selectedNeuronIndices = [];
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 3.0 };
    const mouse = new THREE.Vector2(-2, -2); // offscreen initially

    async function fetchJSON(url) { const r = await fetch(url); return await r.json(); }

    function initThree() {
      const canvas = document.getElementById('canvas');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);
      camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
      camera.position.set(150,150,150);
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(50,50,50);
      controls.update();
      const light = new THREE.AmbientLight(0x404040, 0.8); scene.add(light);
      // Stimulation cube
      const boxGeom = new THREE.BoxGeometry(1.0, 1.0, 1.0);
      const boxMat = new THREE.MeshBasicMaterial({ color: 0xff8800, transparent: true, opacity: 0.18, depthWrite: false });
      stimCube = new THREE.Mesh(boxGeom, boxMat);
      stimCube.visible = false;
      scene.add(stimCube);
      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });
      // Mouse events
      renderer.domElement.addEventListener('mousemove', (e) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 1.0 * 2 - 1;
        mouse.y = -(((e.clientY - rect.top) / rect.height) * 1.0) * 2 + 1;
      });
      renderer.domElement.addEventListener('contextmenu', (e) => {
        // Right-click: fix current cube center and compute neuron indices inside cube
        e.preventDefault();
        if (!positionsAttr) return;
        // If there is a raycast hit, snap cube to that point
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(points);
        if (intersects && intersects.length > 0) {
          const p = intersects[0].point;
          stimCube.position.copy(p);
          stimCube.visible = true;
        }
        // Lock and gather indices
        stimLocked = true;
        const n = positionsAttr.count;
        selectedNeuronIndices = [];
        const cx = stimCube.position.x, cy = stimCube.position.y, cz = stimCube.position.z;
        const half = stimSize * 0.5;
        for (let i=0;i<n;i++) {
          const x = positionsAttr.getX(i), y = positionsAttr.getY(i), z = positionsAttr.getZ(i);
          const dx = Math.abs(x - cx), dy = Math.abs(y - cy), dz = Math.abs(z - cz);
          if (dx <= half && dy <= half && dz <= half) selectedNeuronIndices.push(i);
        }
      }, false);
      animate();
    }

    function hslToRgb(h, s, l) {
      let r, g, b;
      if (s === 0) { r = g = b = l; } else {
        const hue2rgb = (p, q, t) => {
          if (t < 0) t += 1; if (t > 1) t -= 1;
          if (t < 1/6) return p + (q - p) * 6 * t;
          if (t < 1/2) return q;
          if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
          return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1/3); g = hue2rgb(p, q, h); b = hue2rgb(p, q, h - 1/3);
      }
      return [r, g, b];
    }

    function magmaColor(v) {
      // Simple magma-like: black -> deep purple -> red -> orange -> yellow
      const t = Math.min(1, Math.max(0, v));
      // Base hue progression (approx): 0.85 -> 0.75 -> 0.1
      let h;
      if (t < 0.33) h = 0.85 - 0.10 * (t / 0.33);
      else if (t < 0.66) h = 0.75 - 0.45 * ((t - 0.33) / 0.33);
      else h = 0.30 - 0.20 * ((t - 0.66) / 0.34);
      const s = 0.95;
      const l = 0.05 + 0.9 * Math.pow(t, 0.7);
      const [r,g,b] = hslToRgb(h, s, l);
      return [r, g, b];
    }

    function createPoints() {
      const n = positions.length;
      geometry = new THREE.BufferGeometry();
      const posArray = new Float32Array(n * 3);
      const colArray = new Float32Array(n * 3);
      const alphaArray = new Float32Array(n);
      for (let i=0;i<n;i++) {
        const x = positions[i][0];
        const y = positions[i][1];
        const z = positions[i][2];
        posArray[i*3] = x; posArray[i*3+1] = y; posArray[i*3+2] = z;
        const c = magmaColor(0.0);
        colArray[i*3] = c[0]; colArray[i*3+1] = c[1]; colArray[i*3+2] = c[2];
        alphaArray[i] = 0.0;
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colArray, 3));
      geometry.setAttribute('aAlpha', new THREE.BufferAttribute(alphaArray, 1));
      positionsAttr = geometry.getAttribute('position');
      colorsAttr = geometry.getAttribute('color');

      const initialPointSize = parseFloat((document.getElementById('point-size') || {}).value) || 2.5;
      const uniforms = {
        uPointSize: { value: initialPointSize },
        uGlobalOpacity: { value: 0.85 }
      };

      const vs = `
        attribute float aAlpha;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uPointSize;
        void main() {
          vColor = color;
          vAlpha = aAlpha;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uPointSize * (300.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `;

      const fs = `
        precision mediump float;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uGlobalOpacity;
        void main() {
          float alpha = vAlpha * uGlobalOpacity;
          gl_FragColor = vec4(vColor, alpha);
        }
      `;

      material = new THREE.ShaderMaterial({
        uniforms: uniforms,
        vertexShader: vs,
        fragmentShader: fs,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        vertexColors: true,
        alphaTest: 0.0
      });

      points = new THREE.Points(geometry, material);
      scene.add(points);
    }

    function updateStimCube() {
      if (!points) return;
      if (stimLocked) {
        // Keep cube where it was clicked; only ensure scale matches current slider
        if (stimCube) stimCube.scale.set(stimSize, stimSize, stimSize);
        return;
      }
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(points);
      if (intersects && intersects.length > 0) {
        const p = intersects[0].point;
        stimCube.position.copy(p);
        stimCube.visible = true;
        stimCube.scale.set(stimSize, stimSize, stimSize);
      } else {
        stimCube.visible = false;
      }
    }

    function updateColorsForFrame(values) {
      const n = Math.min(values.length, colorsAttr.count);
      const alphaAttr = geometry.getAttribute('aAlpha');
      for (let i=0;i<n;i++) {
        const v = Math.max(0.0, Math.min(1.0, values[i]));
        const [r,g,b] = magmaColor(v);
        colorsAttr.setX(i, r); colorsAttr.setY(i, g); colorsAttr.setZ(i, b);
        // Fade only the lowest activations: below threshold -> interpolate to a minimum alpha
        const fadeThreshold = 0.10; // 10% of scale
        const minAlpha = 0.20;      // still visible but see-through
        let a;
        if (v <= fadeThreshold) {
          a = minAlpha + (v / fadeThreshold) * (1.0 - minAlpha);
        } else {
          a = 1.0;
        }
        alphaAttr.setX(i, a);
      }
      colorsAttr.needsUpdate = true;
      alphaAttr.needsUpdate = true;
    }

    function formatTime(t) {
      const s = Math.max(0, Math.floor(t));
      const m = Math.floor(s / 60);
      const rs = s % 60;
      return `${m.toString().padStart(2,'0')}:${rs.toString().padStart(2,'0')}`;
    }

    function preparePlayback() {
      totalDurationSec = (playbackFrames && playbackFrames.length ? playbackFrames.length / fps : 0);
      playbackTimeSec = Math.min(playbackTimeSec, totalDurationSec > 0 ? totalDurationSec - (1 / fps) : 0);
      const progress = document.getElementById('progress');
      progress.max = totalDurationSec.toFixed(3);
      progress.step = (1.0 / fps).toFixed(3);
      progress.value = playbackTimeSec.toFixed(3);
      const td = document.getElementById('time-display');
      td.textContent = `${formatTime(playbackTimeSec)} / ${formatTime(totalDurationSec)}`;
      // Update segment bar to indicate initial vs generated
      try {
        const initLen = (initialFrames && initialFrames.length) ? initialFrames.length : 0;
        const totalLen = (playbackFrames && playbackFrames.length) ? playbackFrames.length : initLen;
        const ratio = totalLen > 0 ? Math.max(0, Math.min(1, initLen / totalLen)) : 1;
        const pct = (ratio * 100).toFixed(2) + '%';
        const segInit = document.getElementById('seg-initial');
        const segGen = document.getElementById('seg-generated');
        if (segInit && segGen) {
          const hasGen = (totalLen > initLen);
          if (hasGen) {
            // Show both segments: initial takes left portion, generated takes right
            segInit.style.width = pct;
            segGen.style.width = (100 - ratio * 100).toFixed(2) + '%';
            segGen.style.display = 'block';
          } else {
            // Only initial segment, full width
            segInit.style.width = '100%';
            segGen.style.display = 'none';
          }
        }
      } catch (e) {}
    }

    function renderFrameAtTime(tSec) {
      if (!playbackFrames || playbackFrames.length === 0) return;
      const totalFrames = playbackFrames.length;
      let idx = Math.floor(tSec * fps);
      if (idx >= totalFrames) idx = totalFrames - 1;
      if (idx < 0) idx = 0;
      updateColorsForFrame(playbackFrames[idx]);
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      if (isPlaying && playbackFrames && playbackFrames.length > 0) {
        const now = performance.now()/1000.0;
        if (!animate.lastTime) animate.lastTime = now;
        const delta = now - animate.lastTime;
        if (delta > 0) {
          animate.lastTime = now;
          playbackTimeSec += delta;
          if (playbackTimeSec >= totalDurationSec) playbackTimeSec = 0.0; // loop
          renderFrameAtTime(playbackTimeSec);
          const progress = document.getElementById('progress');
          progress.value = playbackTimeSec.toFixed(3);
          const td = document.getElementById('time-display');
          td.textContent = `${formatTime(playbackTimeSec)} / ${formatTime(totalDurationSec)}`;
        }
      }
      updateStimCube();
      renderer.render(scene, camera);
    }

    function regionColorForId(regionId) {
      // Stable pseudo-random color based on regionId (HSL)
      let seed = (regionId * 9301 + 49297) % 233280;
      function rand() { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }
      const h = rand();
      const s = 0.85;
      const l = 0.55;
      const color = new THREE.Color();
      color.setHSL(h, s, l);
      return color;
    }

    function createRegionMesh(regionId, vertices) {
      if (!vertices || vertices.length === 0) return;
      const n = vertices.length;
      const g = new THREE.BufferGeometry();
      const pos = new Float32Array(n * 3);
      for (let i=0;i<n;i++) { pos[i*3]=vertices[i][0]; pos[i*3+1]=vertices[i][1]; pos[i*3+2]=vertices[i][2]; }
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const m = new THREE.PointsMaterial({ color: regionColorForId(regionId), size: 1.4, transparent: true, opacity: 0.55, sizeAttenuation: true });
      const mesh = new THREE.Points(g, m);
      scene.add(mesh);
      regionMeshes[regionId] = mesh;
    }

    async function onRegionCheckboxChange(e) {
      if (e.target.type !== 'checkbox') return;
      const rid = parseInt(e.target.value);
      if (e.target.checked) {
        if (!regionMeshes[rid]) {
          try {
            const resp = await fetch(`/api/region_surface/${rid}`);
            const data = await resp.json();
            createRegionMesh(rid, data.vertices || []);
          } catch (err) { /* ignore */ }
        } else {
          regionMeshes[rid].visible = true;
        }
      } else {
        if (regionMeshes[rid]) regionMeshes[rid].visible = false;
      }
    }

    async function loadInit() {
      const meta = await fetchJSON('/api/init_data');
      positions = meta.positions; neuronMask = meta.neuron_mask; initialFrames = meta.initial_frames;
      createPoints();
      playbackFrames = initialFrames; isPlaying = true; playbackTimeSec = 0.0; preparePlayback();
      renderFrameAtTime(playbackTimeSec);
      document.getElementById('loading').style.display = 'none';
    }

    // ---- Turnstile helpers ----
    let tsLastToken = null;
    function onTsSuccess(token) { tsLastToken = token; }
    function onTsExpired() { tsLastToken = null; }
    function onTsError() { tsLastToken = null; }
    function getTurnstileToken() {
      try {
        if (window.turnstile && typeof window.turnstile.getResponse === 'function') {
          const token = window.turnstile.getResponse(); // first widget
          if (token && token.length > 0) return token;
        }
      } catch (e) { /* ignore */ }
      const input = document.querySelector('input[name="cf-turnstile-response"]');
      return (input && input.value) ? input.value : tsLastToken;
    }
    function resetTurnstile() {
      try { if (window.turnstile && typeof window.turnstile.reset === 'function') window.turnstile.reset(); } catch (e) { /* ignore */ }
      tsLastToken = null;
    }

    async function submitExperiment() {
      const rateDelta = parseFloat(document.getElementById('rate-delta').value);
      const arSteps = parseInt(document.getElementById('ar-steps').value);
      const cfToken = verificationEnabled ? getTurnstileToken() : null;
      if (verificationEnabled && !cfToken) { alert('Please complete the verification.'); return; }
      // No automatic hover selection: only use indices from explicit right-click lock
      let resp, data;
      try {
        resp = await fetch('/api/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ neuron_indices: selectedNeuronIndices, add_rate_hz: rateDelta, ar_steps: arSteps, cf_turnstile_token: cfToken }) });
        data = await resp.json();
      } catch (e) {
        if (verificationEnabled) resetTurnstile();
        alert('Network error submitting experiment');
        return;
      } finally {
        // Always refresh token after an attempt (tokens are single-use)
        if (verificationEnabled) resetTurnstile();
      }
      if (!resp.ok) {
        if (verificationEnabled && resp.status === 403 && (data && (data.error === 'verification_failed' || data.error === 'verification_error' || data.error === 'verification_required'))) {
          alert('Verification failed or expired. Please complete the verification again.');
        } else {
          alert((data && data.error) || 'Submit failed');
        }
        return;
      }
      pollStatus(data.job_id);
    }

    async function pollStatus(jobId) {
      const qs = document.getElementById('queue-status');
      const links = document.getElementById('result-links');
      const statusText = document.getElementById('status-text');
      links.style.display = 'none';
      const timer = setInterval(async () => {
        const r = await fetchJSON('/api/status');
        if (r.job_id && r.job_id !== jobId) return; // another job took over
        if (r.status === 'pending') { qs.textContent = `Queue position: ${r.position}`; }
        else if (r.status === 'running') { qs.textContent = 'Running...'; }
        else if (r.status === 'complete') {
          clearInterval(timer);
          qs.textContent = 'Complete (loading results...)';
          const fr = await fetchJSON(`/api/result_frames/${jobId}`);
          generatedFrames = fr.generated_frames;
          playbackFrames = initialFrames.concat(generatedFrames);
          playbackTimeSec = 0.0; preparePlayback();
          renderFrameAtTime(playbackTimeSec);
          statusText.textContent = 'Viewing initial + generated activity';
          links.style.display = 'block';
          const heatmapName = r.heatmap || `${jobId}.png`;
          const heatmapInfo = r.heatmap_info || null;
          document.getElementById('download-npz').onclick = () => window.location = `/api/download/${jobId}`;
          document.getElementById('open-heatmap').onclick = async () => {
            const overlay = document.getElementById('heatmap-overlay');
            const img = document.getElementById('heatmap-img');
            const err = document.getElementById('heatmap-error');
            const note = document.getElementById('heatmap-note');
            const candidates = [`/api/heatmap/${heatmapName}`, `/api/heatmap/${jobId}.png`];
            let loaded = false;
            for (const url of candidates) {
              try {
                const r = await fetch(url, { cache: 'no-store' });
                if (r.ok) {
                  const blob = await r.blob();
                  img.src = URL.createObjectURL(blob);
                  img.style.display = 'inline-block';
                  err.style.display = 'none';
                  if (heatmapInfo && heatmapInfo.baseline_steps && heatmapInfo.requested_steps && heatmapInfo.baseline_steps < heatmapInfo.requested_steps) {
                    note.textContent = `Note: baseline data is limited to ${heatmapInfo.baseline_steps} steps; showing first ${heatmapInfo.used_steps} of ${heatmapInfo.requested_steps} requested steps.`;
                    note.style.display = 'block';
                  } else {
                    note.style.display = 'none';
                  }
                  overlay.style.display = 'block';
                  loaded = true;
                  // Reset zoom/pan
                  const content = document.getElementById('heatmap-content');
                  content.style.transform = 'translate(0px, 0px) scale(1)';
                  let scale = 1.0, tx = 0, ty = 0; let dragging=false, sx=0, sy=0;
                  const viewport = document.getElementById('heatmap-viewport');
                  function apply() { content.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; }
                  viewport.onwheel = (e) => {
                    e.preventDefault();
                    const rect = viewport.getBoundingClientRect();
                    const cx = e.clientX - rect.left; const cy = e.clientY - rect.top;
                    const prevScale = scale;
                    const delta = Math.sign(e.deltaY) * -0.1; // invert for natural zoom
                    scale = Math.max(0.2, Math.min(8.0, scale * (1 + delta)));
                    const k = scale / prevScale - 1;
                    tx -= k * cx; ty -= k * cy;
                    apply();
                  };
                  viewport.onmousedown = (e) => { dragging=true; sx=e.clientX; sy=e.clientY; viewport.style.cursor='grabbing'; };
                  window.onmouseup = () => { dragging=false; viewport.style.cursor='grab'; };
                  window.onmousemove = (e) => { if (!dragging) return; tx += (e.clientX - sx); ty += (e.clientY - sy); sx=e.clientX; sy=e.clientY; apply(); };
                  break;
                }
              } catch (e) {}
            }
            if (!loaded) {
              img.style.display = 'none';
              err.style.display = 'block';
              if (note) note.style.display = 'none';
              overlay.style.display = 'block';
            }
          };
          // Now mark queue status as complete
          qs.textContent = 'Complete';
          // Refresh verification token for next submission
          resetTurnstile();
        } else if (r.status === 'error') {
          clearInterval(timer); qs.textContent = 'Error: ' + (r.error || '');
          statusText.textContent = 'Viewing initial brain activity';
          resetTurnstile();
        } else {
          qs.textContent = 'Queue: idle';
          statusText.textContent = 'Viewing initial brain activity';
          resetTurnstile();
        }
      }, 1000);
    }

    document.addEventListener('DOMContentLoaded', () => {
      initThree();
      loadInit();
      // Bind radius slider
      const sr = document.getElementById('stim-size');
      if (sr) {
        stimSize = parseFloat(sr.value);
        sr.oninput = (e) => {
          stimSize = parseFloat(e.target.value);
          if (stimCube) stimCube.scale.set(stimSize, stimSize, stimSize);
        };
      }
      // Bind AR steps value display
      const arSlider = document.getElementById('ar-steps');
      const arLabelVal = document.getElementById('ar-steps-val');
      if (arSlider && arLabelVal) {
        const syncAr = () => { arLabelVal.textContent = String(parseInt(arSlider.value)); };
        syncAr();
        arSlider.oninput = syncAr;
        arSlider.onchange = syncAr;
      }
      // Mobile controls toggle
      const toggleBtn = document.getElementById('toggle-controls');
      const controlsPanel = document.getElementById('controls');
      if (toggleBtn && controlsPanel) {
        const isMobile = window.matchMedia('(max-width: 768px)').matches;
        if (isMobile) {
          controlsPanel.classList.remove('open');
          toggleBtn.addEventListener('click', () => {
            controlsPanel.classList.toggle('open');
          });
          // Close on outside tap
          document.addEventListener('click', (e) => {
            if (!controlsPanel.contains(e.target) && e.target !== toggleBtn) {
              controlsPanel.classList.remove('open');
            }
          }, true);
        } else {
          controlsPanel.classList.add('open');
        }
      }
      document.getElementById('play-pause').onclick = () => { isPlaying = !isPlaying; document.getElementById('play-pause').textContent = isPlaying ? 'Pause' : 'Play'; };
      document.getElementById('progress').oninput = (e) => {
        const t = parseFloat(e.target.value);
        playbackTimeSec = isFinite(t) ? t : 0.0;
        renderFrameAtTime(playbackTimeSec);
        const td = document.getElementById('time-display');
        td.textContent = `${formatTime(playbackTimeSec)} / ${formatTime(totalDurationSec)}`;
      };
      document.getElementById('point-size').oninput = (e) => { if (material) material.uniforms.uPointSize.value = parseFloat(e.target.value); };
      document.getElementById('opacity').oninput = (e) => { if (material) material.uniforms.uGlobalOpacity.value = parseFloat(e.target.value); };
      const invertBtn = document.getElementById('toggle-invert');
      if (invertBtn) {
        invertBtn.onclick = () => {
          const inverted = document.body.classList.toggle('invert-colors');
          invertBtn.textContent = inverted ? 'Normal Colors' : 'Invert Colors';
        };
      }
      const preprintBtn = document.getElementById('open-preprint');
      if (preprintBtn) {
        preprintBtn.onclick = () => { window.open('https://arxiv.org/abs/2510.27366', '_blank', 'noopener'); };
      }
      document.getElementById('run-experiment').onclick = () => { submitExperiment(); };
      document.getElementById('reset-view').onclick = () => {
        generatedFrames = null;
        playbackFrames = initialFrames;
        playbackTimeSec = 0.0; preparePlayback();
        renderFrameAtTime(playbackTimeSec);
        const statusText = document.getElementById('status-text');
        statusText.textContent = 'Viewing initial brain activity';
        // Clear stimulation cube and selection
        selectedNeuronIndices = [];
        stimLocked = false;
        if (stimCube) { stimCube.visible = false; }
      };
      // Spacebar toggles play/pause unless typing in text/number inputs, textarea, or contenteditable
      window.addEventListener('keydown', (e) => {
        const el = e.target;
        const tag = (el && el.tagName) ? el.tagName.toUpperCase() : '';
        let typing = false;
        if (tag === 'INPUT') {
          const type = (el && el.type) ? String(el.type).toLowerCase() : '';
          // Allow spacebar when focused on range sliders, block for text/number and others
          typing = !(type === 'range');
        } else if (tag === 'TEXTAREA' || (el && el.isContentEditable)) {
          typing = true;
        }
        if ((e.code === 'Space' || e.key === ' ' || e.key === 'Spacebar') && !typing) {
          e.preventDefault();
          isPlaying = !isPlaying;
          const btn = document.getElementById('play-pause');
          if (btn) btn.textContent = isPlaying ? 'Pause' : 'Play';
        }
      });
      document.getElementById('close-heatmap').onclick = () => {
        const overlay = document.getElementById('heatmap-overlay');
        overlay.style.display = 'none';
      };
    });
  </script>
</body>
</html>
"""
    ts_key_render = None if getattr(CFG.server, 'disable_verification', False) else (getattr(CFG.server, 'turnstile_site_key', '') or None)
    response = make_response(render_template_string(html_template, ts=ts, ar_default=int(CFG.ar.default_ar_steps), fps=float(CFG.data.sampling_rate_hz), ts_key=ts_key_render))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/init_data')
def api_init_data():
    # Prepare a small subset of initial frames for default playback (loop full context)
    L = int(BEST.initial_context.shape[1])
    N = int(BEST.initial_context.shape[2])
    frames = BEST.initial_context[0].tolist()  # list of L lists of length N
    # Hard-coded anisotropic scaling: X = 2*Y, Z = (2/3)*Y, normalized so X spans ~300
    pos = BEST.positions_norm[0]
    ratio = np.array([2.0, 1.0, 2.0/3.0], dtype=np.float32)
    scale = 300.0 / ratio[0]
    pos_scaled = (pos * ratio * scale).astype(np.float32).tolist()
    return jsonify({
        'initial_frames': frames,
        'positions': pos_scaled,
        'neuron_mask': BEST.neuron_mask[0].astype(bool).tolist(),
        'context_len': int(BEST.context_len),
        'num_neurons': N,
        'fps': float(CFG.data.sampling_rate_hz)
    })


@app.route('/api/submit', methods=['POST'])
def api_submit():
    body = request.get_json(force=True)
    # Input structure validation
    if not isinstance(body, dict):
        return jsonify({'error': 'invalid_json'}), 400
    neuron_indices = body.get('neuron_indices')
    add_rate_hz_raw = body.get('add_rate_hz')
    ar_steps_raw = body.get('ar_steps')
    if neuron_indices is None:
        neuron_indices = []
    if not isinstance(neuron_indices, list):
        return jsonify({'error': 'invalid_neuron_indices'}), 400
    if any((not isinstance(r, (int, float))) for r in neuron_indices):
        return jsonify({'error': 'invalid_neuron_indices_types'}), 400
    try:
        add_rate_hz = float(add_rate_hz_raw)
    except Exception:
        return jsonify({'error': 'invalid_add_rate_hz'}), 400
    try:
        ar_steps = int(ar_steps_raw if ar_steps_raw is not None else CFG.ar.default_ar_steps)
    except Exception:
        return jsonify({'error': 'invalid_ar_steps'}), 400
    # Hard caps
    if add_rate_hz < 0.0:
        return jsonify({'error': 'invalid_add_rate_range'}), 400
    MAX_AR_STEPS = 128
    if ar_steps < 1 or ar_steps > MAX_AR_STEPS:
        return jsonify({'error': f'ar_steps_out_of_range_max_{MAX_AR_STEPS}'}), 400
    if len(neuron_indices) > 200000:
        return jsonify({'error': 'too_many_neurons'}), 400
    # Turnstile verification (can be disabled by config/flag)
    if not getattr(CFG.server, 'disable_verification', False):
        ts_secret = os.environ.get('TURNSTILE_SECRET', '').strip()
        ts_site = getattr(CFG.server, 'turnstile_site_key', '')
        cf_token = body.get('cf_turnstile_token')
        if not ts_secret or not ts_site:
            return jsonify({'error': 'verification_not_configured'}), 503
        if not cf_token:
            return jsonify({'error': 'verification_required'}), 403
        try:
            v = requests.post('https://challenges.cloudflare.com/turnstile/v0/siteverify', data={
                'secret': ts_secret,
                'response': cf_token,
                'remoteip': request.remote_addr
            }, timeout=5)
            v.raise_for_status()
            vj = v.json()
            if not vj.get('success'):
                return jsonify({'error': 'verification_failed'}), 403
        except Exception:
            return jsonify({'error': 'verification_error'}), 403

    # One job at a time per session
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid

    with RUNTIME_LOCK:
        existing_id = SESSION_TO_JOB.get(sid)
        if existing_id:
            job = QUEUE_OBJ.find(existing_id)
            if job and job.status in ('pending', 'running'):
                return jsonify({'error': 'You already have a running/pending experiment'}), 400

    job_id = str(uuid.uuid4())
    job = ExperimentJob(
        job_id=job_id,
        session_id=sid,
        neuron_indices=[int(r) for r in neuron_indices],
        add_rate_hz=add_rate_hz,
        ar_steps=ar_steps,
    )
    pos = QUEUE_OBJ.enqueue(job)
    with RUNTIME_LOCK:
        SESSION_TO_JOB[sid] = job_id
    return jsonify({'job_id': job_id, 'position': pos})


@app.route('/api/status')
def api_status():
    sid = session.get('sid')
    if not sid or sid not in SESSION_TO_JOB:
        return jsonify({'status': 'idle'})
    job_id = SESSION_TO_JOB[sid]
    job = QUEUE_OBJ.find(job_id)
    if not job:
        return jsonify({'status': 'idle'})
    if job.status == 'pending':
        pos = QUEUE_OBJ.position_in_queue(job_id)
        return jsonify({'job_id': job_id, 'status': job.status, 'position': pos})
    if job.status == 'running':
        return jsonify({'job_id': job_id, 'status': job.status})
    if job.status == 'complete':
        return jsonify({'job_id': job_id, 'status': job.status})
    return jsonify({'job_id': job_id, 'status': job.status, 'error': job.error})

@app.route('/api/result_frames/<job_id>')
def api_result_frames(job_id: str):
    job = QUEUE_OBJ.find(job_id)
    if not job or job.status != 'complete' or job.result_frames is None:
        return jsonify({'error': 'Not available'}), 404
    return jsonify({'generated_frames': job.result_frames.tolist()})


@app.route('/api/download/<job_id>')
def api_download_npz(job_id: str):
    job = QUEUE_OBJ.find(job_id)
    if not job or not job.result_npz_path:
        return jsonify({'error': 'Not available'}), 404
    # Path safety: ensure file exists and is under the configured output root
    full_path = os.path.realpath(job.result_npz_path)
    root_real = os.path.realpath(CFG.storage.output_root)
    if not full_path.startswith(root_real + os.sep) and full_path != root_real:
        return jsonify({'error': 'invalid_path'}), 400
    if not os.path.exists(full_path):
        return jsonify({'error': 'missing_file'}), 404
    return send_file(full_path, as_attachment=True)


    # Heatmap and region-surface endpoints removed


# --------------------------
# Main
# --------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Zebrafish Brain In-Silico Experiment Platform Web UI')
    parser.add_argument('--config', required=False, help='Path to YAML config')
    parser.add_argument('--ar_h5', required=False, help='Path to eval_gbm2_ar per-AR H5 (overrides data.ar_h5 in config)')
    parser.add_argument('--no_verification', action='store_true', help='Disable verification (turnstile) regardless of config')
    args = parser.parse_args()

    global CFG, DEVICE, MODEL, BEST, MASK, REGION_TO_NEURON, QUEUE_OBJ, BASELINE_FUTURE

    if args.config:
        CFG = load_yaml_config(args.config)
    else:
        # Minimal default config; must provide --ar_h5
        assert args.ar_h5, "--ar_h5 is required when no --config is provided"
        CFG = AppConfig(
            server=ServerConfig(),
            data=DataConfig(ar_h5=args.ar_h5, sampling_rate_hz=3.0, clip_01_before_ar=True),
            model=ModelConfig(),
            ar=ARConfig(),
            ui=UIConfig(),
            storage=StorageConfig()
        )
    if args.ar_h5:
        CFG.data.ar_h5 = args.ar_h5
    if args.no_verification:
        CFG.server.disable_verification = True

    # App secret key for per-session control
    app.secret_key = CFG.server.secret_key

    # Device and dtype
    use_gpu = CFG.model.use_gpu and torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_gpu else 'cpu')

    # Load AR run H5 and experiment config/ckpt derived from it
    assert CFG.data.ar_h5 and os.path.exists(CFG.data.ar_h5), f"data.ar_h5 not found: {CFG.data.ar_h5}"
    BEST = load_ar_run_h5(CFG.data.ar_h5)
    # Sync sampling rate with AR H5 attribute to match eval behavior exactly
    try:
        with h5py.File(CFG.data.ar_h5, 'r') as _hf:
            _sr = _hf.attrs.get('sampling_rate_hz', None)
            if _sr is not None:
                CFG.data.sampling_rate_hz = float(_sr)
    except Exception:
        pass
    exp_dir, ckpt_guess = resolve_experiment_from_ar_h5(CFG.data.ar_h5)
    # Determine d_stimuli from AR H5 stimuli (context or future)
    d_stimuli = 1
    try:
        if BEST.stim_context is not None and BEST.stim_context.shape[-1] > 0:
            d_stimuli = int(BEST.stim_context.shape[-1])
        elif BEST.stim_future is not None and BEST.stim_future.shape[-1] > 0:
            d_stimuli = int(BEST.stim_future.shape[-1])
    except Exception:
        d_stimuli = 1
    # Model checkpoint: use config-provided if present, else from experiment dir
    ckpt_path = CFG.model.checkpoint if CFG.model.checkpoint else ckpt_guess
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    # Build model using experiment config and robust state load
    MODEL = load_model_from_experiment(exp_dir, DEVICE, d_stimuli=d_stimuli)
    if DEVICE.type == 'cuda':
        if CFG.model.dtype == 'bfloat16':
            MODEL = MODEL.to(dtype=torch.bfloat16)
        else:
            MODEL = MODEL.to(dtype=torch.float32)
    # Keep trained global ID mapping (searchsorted) from checkpoint to ensure correct neuron embeddings

    # Optional mask for region overlays
    if CFG.data.mask_h5 and os.path.exists(CFG.data.mask_h5):
        MASK = load_mask_h5(CFG.data.mask_h5)
        REGION_TO_NEURON = map_neurons_to_regions(BEST.positions_norm, MASK)
    else:
        MASK = MaskData(label_volume=np.zeros((1,1,1), dtype=np.int32), region_names=[], grid_shape=(1,1,1))
        REGION_TO_NEURON = {}

    # Ensure output root
    os.makedirs(CFG.storage.output_root, exist_ok=True)

    # Baseline for heatmap/NPZ outputs (optional). Prefer NPZ if provided; else use AR H5 future truth.
    BASELINE_FUTURE = None
    base_npz = (getattr(CFG.data, 'baseline_npz', '') or '').strip()
    if base_npz and os.path.exists(base_npz):
        data_npz = np.load(base_npz)
        assert 'baseline_future' in data_npz, f"Baseline NPZ missing key 'baseline_future': {base_npz}"
        bf = data_npz['baseline_future']
        assert isinstance(bf, np.ndarray) and bf.ndim == 2, f"Baseline 'baseline_future' must be 2D (T,N). Got shape: {getattr(bf, 'shape', None)}"
        BASELINE_FUTURE = bf.astype(np.float32)
    else:
        try:
            # Use future truth from AR run as baseline if available
            with h5py.File(CFG.data.ar_h5, 'r') as f:
                if 'future_truth_rates' in f:
                    BASELINE_FUTURE = f['future_truth_rates'][()].astype(np.float32)
        except Exception:
            BASELINE_FUTURE = None

    # Start worker thread
    QUEUE_OBJ = ExperimentQueue(max_batch_size=int(CFG.ui.max_batch_size))
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    # Run server
    app.run(host=CFG.server.host, port=int(CFG.server.port), debug=CFG.server.debug)


if __name__ == '__main__':
    main()
