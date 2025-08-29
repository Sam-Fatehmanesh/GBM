#!/usr/bin/env python3
"""
Web In-Silico Experiments for GBM

- Loads a trained GBM and a AR sample H5 produced by eval_gbm.py
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

from GenerativeBrainModel.models.gbm import GBM

# --------------------------
# Config and global state
# --------------------------

@dataclass
class ServerConfig:
    host: str = '0.0.0.0'
    port: int = 8054
    debug: bool = False
    secret_key: str = 'change-me'

@dataclass
class DataConfig:
    best_sample_h5: str = ''
    mask_h5: str = ''
    spikes_are_rates: bool = False
    sampling_rate_hz: float = 3.0
    clip_01_before_ar: bool = True

@dataclass
class ModelConfig:
    checkpoint: str = ''
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

def load_model_from_checkpoint(ckpt_path: str, device: torch.device, spikes_are_rates: bool) -> GBM:
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

    if isinstance(sd_in, dict):
        sd = {_normalize_key(k): v for k, v in sd_in.items()}
    else:
        sd = sd_in

    def infer_config_from_state_dict(sdict: Dict[str, torch.Tensor]) -> Dict[str, int]:
        d_model = None
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
    try:
        model.spikes_are_rates = bool(spikes_are_rates)
    except Exception:
        model.spikes_are_rates = False

    # Pre-shape routing centroid buffers to match checkpoint (avoid size-mismatch errors)
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

    model.load_state_dict(sd, strict=False)
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


# --------------------------
# Path resolution helpers
# --------------------------

def resolve_best_sample_h5_path(path: str) -> str:
    """Resolve a usable best-sample H5 path.
    Accepts:
      - Direct path to best_ar_sample_*_data.h5
      - Direct path to any H5: if it lacks required datasets, try sibling matches
      - Directory path to an eval run: search videos/ for best_ar_sample_*_data.h5
    Returns a valid file path or raises a ValueError with guidance.
    """
    if os.path.isdir(path):
        # Search videos/ for best_ar_sample_*_data.h5
        candidates: List[Tuple[float, str]] = []
        for root, _dirs, files in os.walk(path):
            if os.path.basename(root) != 'videos':
                continue
            for fn in files:
                if fn.startswith('best_ar_sample_') and fn.endswith('_data.h5'):
                    full = os.path.join(root, fn)
                    candidates.append((os.path.getmtime(full), full))
        if not candidates:
            raise ValueError(f"No best_ar_sample_*_data.h5 found under directory: {path}. Run eval_gbm.py to produce best sample outputs.")
        candidates.sort(reverse=True)
        return candidates[0][1]

    if os.path.isfile(path):
        try:
            with h5py.File(path, 'r') as f:
                if 'initial_context' in f:
                    return path
                # Try siblings
        except Exception:
            pass
        parent = os.path.dirname(path)
        sibs = [fn for fn in os.listdir(parent) if fn.startswith('best_ar_sample_') and fn.endswith('_data.h5')]
        if sibs:
            sibs_full = [os.path.join(parent, fn) for fn in sibs]
            sibs_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return sibs_full[0]
        raise ValueError(
            f"H5 at {path} does not contain 'initial_context'. Expected a file like 'best_ar_sample_*_data.h5' produced by eval_gbm.py."
        )

    raise ValueError(f"Path not found: {path}")


def load_best_sample_h5(path: str) -> BestSample:
    resolved = resolve_best_sample_h5_path(path)
    with h5py.File(resolved, 'r') as f:
        if 'initial_context' not in f or 'positions' not in f or 'neuron_mask' not in f:
            available = list(f.keys())
            raise ValueError(
                "Best-sample H5 missing required datasets. Required: 'initial_context', 'positions', 'neuron_mask'. "
                f"Available: {available}. File: {resolved}"
            )
        init_ctx = f['initial_context'][()]  # (1, L, N)
        pred_fut = f['pred_future'][()] if 'pred_future' in f else None
        pos = f['positions'][()]
        if pos.ndim == 2:
            pos = pos[None, ...]
        neuron_mask = f['neuron_mask'][()] if 'neuron_mask' in f else np.ones((pos.shape[0], pos.shape[1]), dtype=bool)
        stim_ctx = f['stimulus_context'][()] if 'stimulus_context' in f else None
        stim_fut = f['stimulus_future'][()] if 'stimulus_future' in f else None
        context_len = int(f.attrs.get('context_len', init_ctx.shape[1]))
        base_n_steps = int(f.attrs.get('n_steps', pred_fut.shape[1] if pred_fut is not None else context_len))
    return BestSample(
        initial_context=init_ctx.astype(np.float32),
        pred_future=None if pred_fut is None else pred_fut.astype(np.float32),
        positions_norm=pos.astype(np.float32),
        neuron_mask=neuron_mask.astype(bool),
        stim_context=None if stim_ctx is None else stim_ctx.astype(np.float32),
        stim_future=None if stim_fut is None else stim_fut.astype(np.float32),
        context_len=context_len,
        base_n_steps=base_n_steps,
    )


@dataclass
class MaskData:
    label_volume: np.ndarray  # (X,Y,Z) int
    region_names: List[str]
    grid_shape: Tuple[int, int, int]


def load_mask_h5(path: str) -> MaskData:
    with h5py.File(path, 'r') as f:
        label_volume = f['label_volume'][()].astype(np.int32)
        region_names = [name.decode() if isinstance(name, bytes) else str(name) for name in f['region_names'][()]]
        grid_shape = tuple(label_volume.shape)
    return MaskData(label_volume=label_volume, region_names=region_names, grid_shape=grid_shape)


def map_neurons_to_regions(positions_norm: np.ndarray, mask: MaskData) -> Dict[int, np.ndarray]:
    # positions_norm: (1, N, 3) in [0,1] range
    pos = positions_norm[0]
    X, Y, Z = mask.grid_shape
    # Clamp to valid voxel indices
    ix = np.clip(np.round(pos[:, 0] * (X - 1)).astype(int), 0, X - 1)
    iy = np.clip(np.round(pos[:, 1] * (Y - 1)).astype(int), 0, Y - 1)
    iz = np.clip(np.round(pos[:, 2] * (Z - 1)).astype(int), 0, Z - 1)
    labels = mask.label_volume[ix, iy, iz]
    region_to_indices: Dict[int, List[int]] = {}
    for idx, rid in enumerate(labels.tolist()):
        if rid <= 0:
            continue
        if rid not in region_to_indices:
            region_to_indices[rid] = []
        region_to_indices[rid].append(idx)
    return {rid: np.array(indices, dtype=np.int64) for rid, indices in region_to_indices.items()}


# --------------------------
# Experiment queue and worker
# --------------------------

@dataclass
class ExperimentJob:
    job_id: str
    session_id: str
    selected_region_ids: List[int]
    add_rate_hz: float
    ar_steps: int
    status: str = 'pending'  # pending | running | complete | error
    error: Optional[str] = None
    result_npz_path: Optional[str] = None
    heatmap_path: Optional[str] = None
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

            for b_idx, job in enumerate(batch):
                # Gather indices from selected regions
                if job.selected_region_ids:
                    indices_sets = [REGION_TO_NEURON.get(rid, np.array([], dtype=np.int64)) for rid in job.selected_region_ids]
                    if indices_sets:
                        region_indices = np.unique(np.concatenate(indices_sets))
                        region_indices = region_indices[neuron_mask[region_indices]]  # respect mask
                    else:
                        region_indices = np.array([], dtype=np.int64)
                else:
                    region_indices = np.array([], dtype=np.int64)

                # Compute delta in the correct domain
                if CFG.data.spikes_are_rates:
                    delta = float(job.add_rate_hz)
                else:
                    # Convert delta rate to probability increment
                    delta = 1.0 - math.exp(-float(job.add_rate_hz) / max(1e-6, sampling_rate_hz))

                if region_indices.size > 0:
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
                bsize = len(job_indices)
                # Build tensors
                x_t = to_torch(init_x[job_indices], device=DEVICE, dtype=dtype)
                stim_t = to_torch(init_stim[job_indices], device=DEVICE, dtype=dtype)
                pos_t = to_torch(BEST.positions_norm[0][None, ...].repeat(bsize, axis=0), device=DEVICE, dtype=dtype)
                mask_t = torch.from_numpy(BEST.neuron_mask[0][None, :].repeat(bsize, axis=0)).to(device=DEVICE)
                fut_stim_np = np.stack([future_stim_list[j] for j in job_indices], axis=0)  # (B, n_steps, K)
                fut_stim_t = to_torch(fut_stim_np, device=DEVICE, dtype=dtype)

                with torch.no_grad():
                    gen_seq = MODEL.autoregress(
                        init_x=x_t,
                        init_stimuli=stim_t,
                        point_positions=pos_t,
                        neuron_pad_mask=mask_t,
                        future_stimuli=fut_stim_t,
                        n_steps=n_steps,
                        context_len=BEST.context_len,
                    )  # (B, L + n_steps, N) in model's native domain
                gen_only = gen_seq[:, -n_steps:, :]  # (B, n_steps, N)
                results = gen_only.detach().float().cpu().numpy()
                for loc_idx, job_index in enumerate(job_indices):
                    results_per_job[job_index] = results[loc_idx]

            # Persist outputs per job
            for j_idx, job in enumerate(batch):
                try:
                    gen_future = results_per_job[j_idx]  # (n_steps, N)
                    job.result_frames = gen_future
                    # Save NPZ
                    out_dir = os.path.join(CFG.storage.output_root, time.strftime('%Y%m%d_%H%M%S'))
                    os.makedirs(out_dir, exist_ok=True)
                    npz_path = os.path.join(out_dir, f'{job.job_id}.npz')
                    np.savez_compressed(
                        npz_path,
                        initial_context=BEST.initial_context[0],
                        generated_future=gen_future,
                        baseline_future=BEST.pred_future[0] if BEST.pred_future is not None else np.zeros((0, BEST.initial_context.shape[2]), dtype=np.float32),
                        positions=BEST.positions_norm[0],
                        neuron_mask=BEST.neuron_mask[0],
                        selected_regions=np.array(job.selected_region_ids, dtype=np.int32),
                        add_rate_hz=np.array([job.add_rate_hz], dtype=np.float32),
                        sampling_rate_hz=np.array([CFG.data.sampling_rate_hz], dtype=np.float32),
                    )
                    job.result_npz_path = npz_path

                    # Heatmap
                    heatmap_path = os.path.join(out_dir, f'{job.job_id}_heatmap.png')
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        # Compute baseline vs generated sums for selected regions
                        if BEST.pred_future is not None:
                            base = BEST.pred_future[0]  # (Tbase, N)
                            Texp = gen_future.shape[0]
                            Tbase = base.shape[0]
                            T = min(Texp, Tbase)
                            base = base[:T]
                            exp = gen_future[:T]
                            # Build heatmap matrix (R x T)
                            rows = []
                            row_labels = []
                            eps = 1e-6
                            # Whole brain (all valid neurons by mask)
                            try:
                                valid_idx = np.where(BEST.neuron_mask[0].astype(bool))[0]
                            except Exception:
                                valid_idx = np.arange(base.shape[1], dtype=np.int64)
                            if valid_idx.size > 0:
                                base_sum_wb = base[:, valid_idx].sum(axis=1)
                                exp_sum_wb = exp[:, valid_idx].sum(axis=1)
                                rel_wb = (exp_sum_wb - base_sum_wb) / np.maximum(base_sum_wb, eps)
                                rows.append(rel_wb.astype(np.float32))
                            else:
                                rows.append(np.zeros((T,), dtype=np.float32))
                            row_labels.append('Whole brain')

                            # All regions
                            for rid in range(1, len(MASK.region_names) + 1):
                                idxs = REGION_TO_NEURON.get(rid, np.array([], dtype=np.int64))
                                if idxs.size == 0:
                                    rows.append(np.zeros((T,), dtype=np.float32))
                                    row_labels.append(f'{MASK.region_names[rid - 1]} (empty)')
                                    continue
                                base_sum = base[:, idxs].sum(axis=1)
                                exp_sum = exp[:, idxs].sum(axis=1)
                                rel = (exp_sum - base_sum) / np.maximum(base_sum, eps)
                                rows.append(rel.astype(np.float32))
                                row_labels.append(MASK.region_names[rid - 1])
                            mat = np.stack(rows, axis=0) if rows else np.zeros((1, 1), dtype=np.float32)
                            fig, ax = plt.subplots(figsize=(max(6, T * 0.25), max(4, len(row_labels) * 0.22)))
                            im = ax.imshow(mat, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0, interpolation='nearest')
                            ax.set_xlabel('AR timestep')
                            ax.set_ylabel('Region')
                            ax.set_yticks(np.arange(len(row_labels)))
                            ax.set_yticklabels(row_labels)
                            fig.colorbar(im, ax=ax, shrink=0.8, label='(exp - base)/base')
                            fig.tight_layout()
                            fig.savefig(heatmap_path, dpi=130, bbox_inches='tight')
                            plt.close(fig)
                            job.heatmap_path = heatmap_path
                        else:
                            job.heatmap_path = None
                    except Exception as e:
                        job.heatmap_path = None
                    job.status = 'complete'
                except Exception as e:
                    job.status = 'error'
                    job.error = str(e)
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
    #controls { width: 360px; background: #151521; color: #d8d8e6; padding: 15px; overflow-y: auto; box-sizing: border-box; }
    #viewport { flex: 1; position: relative; }
    #canvas { width: 100%; height: 100%; display: block; }
    .control-group { margin-bottom: 18px; }
    .control-group h3 { margin: 0 0 10px 0; color: #ffb38a; }
    label { font-size: 12px; }
    input[type="range"] { width: 100%; }
    button { background: linear-gradient(90deg, #3a273a, #402a2a); color: #f7f3f0; border: 1px solid #ff9f6e; padding: 8px 12px; border-radius: 6px; cursor: pointer; margin: 2px; box-shadow: 0 0 10px rgba(255,159,110,0.22), 0 0 14px rgba(255,155,209,0.16); }
    button:hover { filter: brightness(1.08); box-shadow: 0 0 12px rgba(255,179,130,0.30), 0 0 16px rgba(255,155,209,0.22); border-color: #ffc08e; }
    .region-list { max-height: 240px; overflow-y: auto; border: 1px solid #2a2a3a; padding: 10px; border-radius: 6px; background: rgba(13,13,20,0.6); }
    .region-item { margin: 3px 0; font-size: 11px; }
    #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); background: rgba(0,0,0,0.8); color: white; padding: 20px; border-radius: 10px; z-index: 1000; }
 
    /* Warm neon sliders */
    input[type="range"] { -webkit-appearance: none; height: 6px; border-radius: 4px; background: #2a2a3a; outline: none; }
    input[type="range"]::-webkit-slider-runnable-track { height: 6px; border-radius: 4px; background: linear-gradient(90deg, rgba(255,120,160,0.5), rgba(255,180,90,0.35)); }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #0e0e16; border: 2px solid #ff9bd1; box-shadow: 0 0 10px rgba(255,155,209,0.32), 0 0 12px rgba(255,179,90,0.20); margin-top: -6px; }
    input[type="range"]::-moz-range-track { height: 6px; border-radius: 4px; background: linear-gradient(90deg, rgba(255,120,160,0.5), rgba(255,180,90,0.35)); }
    input[type="range"]::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #0e0e16; border: 2px solid #ff9bd1; box-shadow: 0 0 10px rgba(255,155,209,0.32), 0 0 12px rgba(255,179,90,0.20); }
 
    /* Checkbox accent color */
    input[type="checkbox"] { accent-color: #ff9bd1; }
 
    /* Scrollbar styling for region list (WebKit) */
    .region-list::-webkit-scrollbar { width: 10px; }
    .region-list::-webkit-scrollbar-track { background: rgba(20,20,28,0.45); border-radius: 6px; }
    .region-list::-webkit-scrollbar-thumb { background: linear-gradient(180deg, rgba(255,155,209,0.7), rgba(255,179,90,0.5)); border-radius: 6px; }
 
    /* Video controls bar styling */
    #video-controls { background: linear-gradient(0deg, rgba(10,10,14,0.92), rgba(10,10,14,0.68)); border-top: 1px solid #2a2a3a; }
    #time-display { color: #e6d8cf; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="container"> 
    <div id="controls">
      <h2 style="margin-top:0;">Zebrafish Brain In-Silico Experiment Platform</h2>

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
       </div>

      <div class="control-group">
        <h3>Regions</h3>
        <div style="margin: 6px 0;">
          <button id="select-all">Select All</button>
          <button id="clear-all">Clear All</button>
        </div>
        <div class="region-list" id="region-list"></div>
      </div>

      <div class="control-group">
        <h3>Experiment</h3>
        <div>
          <label>Spike-rate delta (Hz):</label>
          <input type="number" id="rate-delta" value="0.2" step="0.05" min="0" />
        </div>
        <div>
          <label>AR Steps:</label>
          <input type="number" id="ar-steps" value="{{ ar_default }}" step="1" min="1" />
        </div>
        <div>
          <button id="run-experiment">Run Experiment</button>
        </div>
        <div id="queue-status" style="margin-top:8px;font-size:12px;color:#ccc;">Queue: idle</div>
        <div id="result-links" style="display:none;margin-top:8px;">
          <button id="download-npz">Download NPZ</button>
          <button id="open-heatmap">Open Heatmap</button>
        </div>
      </div>
    </div>

    <div id="viewport">
      <div id="loading">Loading...</div>
      <canvas id="canvas"></canvas>
      <div id="heatmap-overlay" style="position:absolute; top:10%; left:10%; width:80%; height:80%; background:rgba(0,0,0,0.85); border:1px solid #444; border-radius:6px; display:none; z-index: 1002; padding:8px; box-sizing:border-box;">
        <div style="display:flex; justify-content:space-between; align-items:center; color:#dddddd; margin-bottom:6px;">
          <span>Autoregression Heatmap (scroll to zoom, drag to pan)</span>
          <button id="close-heatmap">Close</button>
        </div>
        <div id="heatmap-viewport" style="position:relative; width:100%; height:calc(100% - 32px); overflow:hidden; background:#111; cursor:grab;">
          <div id="heatmap-content" style="position:absolute; top:0; left:0; transform-origin: 0 0;">
            <img id="heatmap-img" alt="Heatmap" style="display:none; user-select:none; pointer-events:none;" />
          </div>
          <div id="heatmap-error" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); display:none; color:#ff8888; font-size:14px;">Heatmap not available.</div>
        </div>
      </div>
        <div id="status-text" style="position:absolute; bottom:56px; left:50%; transform:translateX(-50%); color:#aaaaaa; font-size:24px; background:rgba(0,0,0,0.4); padding:4px 8px; border-radius:4px;">Viewing initial brain activity</div>
        <div id="video-controls" style="position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.6); padding:6px 8px; display:flex; align-items:center; gap:8px; z-index: 1001;">
          <button id="play-pause">Pause</button>
          <input type="range" id="progress" min="0" max="0" step="0.001" value="0" style="flex:1;">
          <span id="time-display" style="color:#cccccc; font-size:12px; width:120px; text-align:right;">00:00 / 00:00</span>
        </div>
        <div style="position:absolute; top:8px; right:12px; z-index:1001;">
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

    async function fetchJSON(url) { const r = await fetch(url); return await r.json(); }

    function initThree() {
      const canvas = document.getElementById('canvas');
      const viewport = document.getElementById('viewport');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);
      camera = new THREE.PerspectiveCamera(75, viewport.clientWidth / viewport.clientHeight, 0.1, 10000);
      camera.position.set(150,150,150);
      renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
      renderer.setSize(viewport.clientWidth, viewport.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(50,50,50);
      controls.update();
      const light = new THREE.AmbientLight(0x404040, 0.8); scene.add(light);
      window.addEventListener('resize', () => {
        camera.aspect = viewport.clientWidth / viewport.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(viewport.clientWidth, viewport.clientHeight);
      });
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

      const uniforms = {
        uPointSize: { value: 1.0 },
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

    function populateRegionList() {
      const rl = document.getElementById('region-list');
      rl.innerHTML = '';
      for (let i=0;i<regionNames.length;i++) {
        const div = document.createElement('div');
        div.className = 'region-item';
        div.innerHTML = `<input type="checkbox" value="${i+1}" id="region-${i+1}"/> <label for="region-${i+1}">${regionNames[i]}</label>`
        rl.appendChild(div);
      }
      document.getElementById('select-all').onclick = async () => {
        document.querySelectorAll('#region-list input[type=checkbox]').forEach(cb => cb.checked = true);
        // Fetch and show all
        const boxes = document.querySelectorAll('#region-list input[type=checkbox]');
        for (const cb of boxes) { await onRegionCheckboxChange({ target: cb }); }
      };
      document.getElementById('clear-all').onclick = () => {
        document.querySelectorAll('#region-list input[type=checkbox]').forEach(cb => cb.checked = false);
        Object.values(regionMeshes).forEach(m => m.visible = false);
      };
      rl.addEventListener('change', onRegionCheckboxChange);
    }

    async function loadInit() {
      const meta = await fetchJSON('/api/init_data');
      positions = meta.positions; neuronMask = meta.neuron_mask; regionNames = meta.region_names; initialFrames = meta.initial_frames;
      populateRegionList();
      createPoints();
      playbackFrames = initialFrames; isPlaying = true; playbackTimeSec = 0.0; preparePlayback();
      renderFrameAtTime(playbackTimeSec);
      document.getElementById('loading').style.display = 'none';
    }

    async function submitExperiment() {
      const selected = Array.from(document.querySelectorAll('#region-list input[type=checkbox]:checked')).map(cb => parseInt(cb.value));
      const rateDelta = parseFloat(document.getElementById('rate-delta').value);
      const arSteps = parseInt(document.getElementById('ar-steps').value);
      const resp = await fetch('/api/submit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ regions: selected, add_rate_hz: rateDelta, ar_steps: arSteps }) });
      const data = await resp.json();
      if (!resp.ok) { alert(data.error || 'Submit failed'); return; }
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
          qs.textContent = 'Complete';
          const fr = await fetchJSON(`/api/result_frames/${jobId}`);
          generatedFrames = fr.generated_frames;
          playbackFrames = initialFrames.concat(generatedFrames);
          playbackTimeSec = 0.0; preparePlayback();
          renderFrameAtTime(playbackTimeSec);
          statusText.textContent = 'Viewing initial + generated activity';
          links.style.display = 'block';
          const heatmapName = r.heatmap || `${jobId}.png`;
          document.getElementById('download-npz').onclick = () => window.location = `/api/download/${jobId}.npz`;
          document.getElementById('open-heatmap').onclick = async () => {
            const overlay = document.getElementById('heatmap-overlay');
            const img = document.getElementById('heatmap-img');
            const err = document.getElementById('heatmap-error');
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
              overlay.style.display = 'block';
            }
          };
        } else if (r.status === 'error') {
          clearInterval(timer); qs.textContent = 'Error: ' + (r.error || '');
          statusText.textContent = 'Viewing initial brain activity';
        } else {
          qs.textContent = 'Queue: idle';
          statusText.textContent = 'Viewing initial brain activity';
        }
      }, 1000);
    }

    document.addEventListener('DOMContentLoaded', () => {
      initThree();
      loadInit();
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
      document.getElementById('run-experiment').onclick = () => { submitExperiment(); };
      document.getElementById('reset-view').onclick = () => {
        generatedFrames = null;
        playbackFrames = initialFrames;
        playbackTimeSec = 0.0; preparePlayback();
        renderFrameAtTime(playbackTimeSec);
        const statusText = document.getElementById('status-text');
        statusText.textContent = 'Viewing initial brain activity';
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
    response = make_response(render_template_string(html_template, ts=ts, ar_default=int(CFG.ar.default_ar_steps), fps=float(CFG.data.sampling_rate_hz)))
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
        'region_names': MASK.region_names,
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
    selected_regions = body.get('regions', [])
    add_rate_hz = float(body.get('add_rate_hz', 0.0))
    ar_steps = int(body.get('ar_steps', CFG.ar.default_ar_steps))

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
        selected_region_ids=[int(r) for r in selected_regions],
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
        return jsonify({'job_id': job_id, 'status': job.status, 'heatmap': os.path.basename(job.heatmap_path) if job.heatmap_path else None})
    return jsonify({'job_id': job_id, 'status': job.status, 'error': job.error})


@app.route('/api/result_frames/<job_id>')
def api_result_frames(job_id: str):
    job = QUEUE_OBJ.find(job_id)
    if not job or job.status != 'complete' or job.result_frames is None:
        return jsonify({'error': 'Not available'}), 404
    return jsonify({'generated_frames': job.result_frames.tolist()})


@app.route('/api/download/<path:filename>')
def api_download_npz(filename: str):
    full_path = os.path.join(CFG.storage.output_root, filename)
    if not os.path.exists(full_path):
        # Also search dated subdirs
        for root, _dirs, files in os.walk(CFG.storage.output_root):
            if filename in files:
                full_path = os.path.join(root, filename)
                break
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(full_path, as_attachment=True)


@app.route('/api/heatmap/<path:filename>')
def api_heatmap(filename: str):
    full_path = os.path.join(CFG.storage.output_root, filename)
    if not os.path.exists(full_path):
        for root, _dirs, files in os.walk(CFG.storage.output_root):
            if filename in files:
                full_path = os.path.join(root, filename)
                break
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(full_path, mimetype='image/png')


def _extract_region_surface_vertices_scaled(mask: MaskData, region_id: int, subsample: int = 4) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return np.empty((0, 3), dtype=np.float32)
    region_mask = (mask.label_volume == region_id)
    if not np.any(region_mask):
        return np.empty((0, 3), dtype=np.float32)
    if subsample > 1:
        region_mask = region_mask[::subsample, ::subsample, ::subsample]
    interior = ndimage.binary_erosion(region_mask)
    surface = region_mask & (~interior)
    coords = np.array(np.where(surface)).T.astype(np.float32)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if subsample > 1:
        coords *= subsample
    X, Y, Z = mask.grid_shape
    # normalize to [0,1]
    coords[:, 0] /= float(X)
    coords[:, 1] /= float(Y)
    coords[:, 2] /= float(Z)
    # apply the same anisotropic scaling as neurons: X=2*Y, Z=2/3*Y then normalize so X spans ~300
    ratio = np.array([2.0, 1.0, 2.0/3.0], dtype=np.float32)
    scale = 300.0 / ratio[0]
    coords = coords * ratio * scale
    return coords.astype(np.float32)


@app.route('/api/region_surface/<int:region_id>')
def api_region_surface(region_id: int):
    try:
        if region_id < 1 or region_id > len(MASK.region_names):
            return jsonify({'error': f'invalid region id {region_id}'}), 400
        verts = _extract_region_surface_vertices_scaled(MASK, region_id, subsample=4)
        return jsonify({'region_id': int(region_id), 'vertices': verts.tolist()})
    except Exception as e:
        return jsonify({'region_id': int(region_id), 'vertices': []})


# --------------------------
# Main
# --------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Zebrafish Brain In-Silico Experiment Platform Web UI')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    global CFG, DEVICE, MODEL, BEST, MASK, REGION_TO_NEURON, QUEUE_OBJ

    CFG = load_yaml_config(args.config)

    # App secret key for per-session control
    app.secret_key = CFG.server.secret_key

    # Device and dtype
    use_gpu = CFG.model.use_gpu and torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_gpu else 'cpu')

    # Load model
    assert os.path.exists(CFG.model.checkpoint), f"Checkpoint not found: {CFG.model.checkpoint}"
    MODEL = load_model_from_checkpoint(CFG.model.checkpoint, DEVICE, spikes_are_rates=CFG.data.spikes_are_rates)
    if DEVICE.type == 'cuda':
        if CFG.model.dtype == 'bfloat16':
            MODEL = MODEL.to(dtype=torch.bfloat16)
        else:
            MODEL = MODEL.to(dtype=torch.float32)

    # Load best sample and mask
    assert os.path.exists(CFG.data.best_sample_h5), f"Best sample H5 not found: {CFG.data.best_sample_h5}"
    BEST = load_best_sample_h5(CFG.data.best_sample_h5)
    assert os.path.exists(CFG.data.mask_h5), f"Mask H5 not found: {CFG.data.mask_h5}"
    MASK = load_mask_h5(CFG.data.mask_h5)
    REGION_TO_NEURON = map_neurons_to_regions(BEST.positions_norm, MASK)

    # Ensure output root
    os.makedirs(CFG.storage.output_root, exist_ok=True)

    # Start worker thread
    QUEUE_OBJ = ExperimentQueue(max_batch_size=int(CFG.ui.max_batch_size))
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    # Run server
    app.run(host=CFG.server.host, port=int(CFG.server.port), debug=CFG.server.debug)


if __name__ == '__main__':
    main()
