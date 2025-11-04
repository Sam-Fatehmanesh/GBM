#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path
import time
import yaml
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
 
# Ensure MKL threading layer is compatible before importing numpy/h5py
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import shutil


def _list_run_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)


def _find_new_run_dir(before: List[Path], after: List[Path]) -> Path:
    before_set = set(p.name for p in before)
    for p in after[::-1]:
        if p.name not in before_set:
            return p
    # fallback to latest
    return after[-1] if after else None


def _cleanup_cuda_memory():
    try:
        import gc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass


def _write_intraepoch_csv_from_loss(loss_txt: Path, out_csv: Path) -> None:
    try:
        if not loss_txt.exists():
            return
        import re
        rows = []
        with open(loss_txt, 'r') as f:
            for line in f:
                # match lines like: "epoch 1, step 308, train 1.400141, val 1.231044"
                m = re.search(r"epoch\s+(\d+),\s*step\s+(\d+),\s*train\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*val\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if m:
                    rows.append({
                        'epoch': int(m.group(1)),
                        'step': int(m.group(2)),
                        'train': float(m.group(3)),
                        'val': float(m.group(4)),
                    })
        if rows:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            import csv as _csv
            with open(out_csv, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=['epoch','step','train','val'])
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    except Exception:
        pass


def _read_last_val_loss(val_csv: Path) -> float | None:
    # Prefer val_events.csv if present
    if val_csv.exists():
        try:
            last_val = None
            with open(val_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row or len(row) < 4:
                        continue
                    try:
                        last_val = float(row[3])
                    except Exception:
                        pass
            if last_val is not None:
                return last_val
        except Exception:
            pass
    # Fallback to logs/loss.txt parsing (find last "val" number)
    loss_txt = val_csv.parent / 'loss.txt'
    if loss_txt.exists():
        try:
            with open(loss_txt, 'r') as f:
                lines = f.readlines()
            for line in reversed(lines):
                if 'val' in line:
                    # extract last float on the line
                    import re
                    floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if floats:
                        return float(floats[-1])
        except Exception:
            pass
    return None


@torch.no_grad()
def _apply_spatial_voxel_blur_torch(spikes_LN: np.ndarray,
                                    positions_N3: np.ndarray,
                                    mask_N: np.ndarray,
                                    voxel_size: float,
                                    device: torch.device | None = None) -> np.ndarray:
    if voxel_size is None or float(voxel_size) <= 0.0:
        return spikes_LN.astype(np.float32, copy=False)
    dev = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # to torch
    spikes = torch.from_numpy(spikes_LN.astype(np.float32, copy=False)).to(dev)      # (L,N)
    positions = torch.from_numpy(positions_N3.astype(np.float32, copy=False)).to(dev)
    mask = torch.from_numpy(mask_N.astype(np.float32, copy=False)).to(dev)
    L, N = spikes.shape
    out = torch.zeros_like(spikes)
    valid = (mask != 0)
    if not bool(valid.any().item()):
        return spikes_LN.astype(np.float32, copy=False)
    pos_v = positions[valid]  # (Nv,3)
    vox = torch.floor(pos_v / float(voxel_size)).to(torch.long)
    vx_min, _ = vox.min(dim=0)
    vox0 = vox - vx_min
    sx = int(vox0[:, 0].max().item()) + 1
    sy = int(vox0[:, 1].max().item()) + 1
    code = vox0[:, 0] + vox0[:, 1] * sx + vox0[:, 2] * (sx * sy)  # (Nv,)
    num_groups = int(code.max().item()) + 1 if code.numel() > 0 else 0
    spikes_v = spikes[:, valid]  # (L,Nv)
    if num_groups == 0:
        out[:, valid] = spikes_v
        return out.detach().cpu().numpy()
    sums = spikes_v.new_zeros((L, num_groups))
    sums.index_add_(1, code, spikes_v)
    counts = torch.bincount(code, minlength=num_groups).to(sums.dtype)
    counts = counts.clamp_min(1)
    means = sums / counts.unsqueeze(0)
    blurred = means.index_select(1, code)  # (L,Nv)
    out[:, valid] = blurred
    return out.detach().cpu().numpy()


def _make_blur_visualization(sample_h5: Path,
                             subject_file: str,
                             blur_levels: List[float],
                             out_dir: Path,
                             time_index: int | None = None,
                             max_points: int = 1000) -> None:
    fp = sample_h5 / subject_file
    if not fp.exists():
        return
    try:
        with h5py.File(fp, 'r') as f:
            ds_name = 'neuron_values' if 'neuron_values' in f else (
                'spike_probabilities' if 'spike_probabilities' in f else (
                'spike_rates_hz' if 'spike_rates_hz' in f else (
                'processed_calcium' if 'processed_calcium' in f else 'zcalcium')))
            spikes_TN = f[ds_name][:]  # (T,N)
            positions_N3 = f['cell_positions'][:]
    except Exception:
        return
    T, N = spikes_TN.shape
    t = time_index if time_index is not None else min(100, T - 1)
    base = spikes_TN[t].astype(np.float32)
    mask = np.ones((N,), dtype=np.float32)
    rng = np.random.default_rng(0)
    if N > max_points:
        idxs = rng.choice(N, size=max_points, replace=False)
    else:
        idxs = np.arange(N)
    plt.figure(figsize=(12, 8))
    cols = int(np.ceil((len(blur_levels) + 1) / 2))
    rows = 2
    # panel 0: original
    ax = plt.subplot(rows, cols, 1)
    ax.scatter(base[idxs], base[idxs], s=4, alpha=0.5)
    lim = [0, max(1e-6, float(np.percentile(base[idxs], 99.5)))]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_title('No blur (reference)')
    ax.set_xlabel('original')
    ax.set_ylabel('blurred')
    # panels: each blur
    for i, bv in enumerate(blur_levels, start=2):
        blurred = _apply_spatial_voxel_blur_torch(spikes_TN[t:t+1, :], positions_N3, mask, bv)[0]
        ax = plt.subplot(rows, cols, i)
        ax.scatter(base[idxs], blurred[idxs], s=4, alpha=0.5)
        lim2 = [0, max(1e-6, float(np.percentile(np.concatenate([base[idxs], blurred[idxs]]), 99.5)))]
        ax.plot(lim2, lim2, 'k--', lw=1)
        ax.set_title(f'voxel_size={bv}')
        ax.set_xlabel('original')
        ax.set_ylabel('blurred')
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'blur_effect_scatter.png', dpi=150)
    plt.close()


def _rasterize_to_grid(values_N: np.ndarray,
                       positions_N3: np.ndarray,
                       plane: str = 'xy',
                       grid_hw: Tuple[int, int] = (64, 64)) -> np.ndarray:
    H, W = grid_hw
    if plane not in ('xy', 'yz', 'xz'):
        plane = 'xy'
    if plane == 'xy':
        P = positions_N3[:, [0, 1]]
    elif plane == 'yz':
        P = positions_N3[:, [1, 2]]
    else:
        P = positions_N3[:, [0, 2]]
    # normalize to [0,1] using min/max
    pmin = P.min(axis=0)
    pmax = P.max(axis=0)
    span = np.maximum(pmax - pmin, 1e-6)
    U = (P - pmin) / span
    xs = np.clip((U[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    ys = np.clip((U[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    grid_sum = np.zeros((H, W), dtype=np.float32)
    grid_cnt = np.zeros((H, W), dtype=np.float32)
    np.add.at(grid_sum, (ys, xs), values_N)
    np.add.at(grid_cnt, (ys, xs), 1.0)
    grid = grid_sum / np.clip(grid_cnt, 1.0, None)
    return grid


def _make_blur_videos(sample_h5: Path,
                      subject_file: str,
                      blur_levels: List[float],
                      out_dir: Path,
                      plane: str = 'xy',
                      grid_hw: Tuple[int, int] = (512, 512),
                      frames: int = 240,
                      fps: int = 3) -> None:
    fp = sample_h5 / subject_file
    if not fp.exists():
        return
    try:
        with h5py.File(fp, 'r') as f:
            ds_name = 'neuron_values' if 'neuron_values' in f else (
                'spike_probabilities' if 'spike_probabilities' in f else (
                'spike_rates_hz' if 'spike_rates_hz' in f else (
                'processed_calcium' if 'processed_calcium' in f else 'zcalcium')))
            spikes_TN = f[ds_name][:].astype(np.float32)  # (T,N)
            positions_N3 = f['cell_positions'][:].astype(np.float32)
    except Exception:
        return
    T, N = spikes_TN.shape
    if T <= 0 or N <= 0:
        return
    use_frames = min(frames, T)
    # time indices spaced across sequence
    idxs = np.linspace(0, T - 1, use_frames).astype(np.int64)
    mask = np.ones((N,), dtype=np.float32)
    # normalize colormap limits for consistency: use 99.5th percentile of original values across sampled frames
    sample_vals = spikes_TN[idxs].reshape(-1)
    vmax = float(np.percentile(sample_vals, 99.5)) if sample_vals.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    for bv in blur_levels:
        out_path = out_dir / f'blur_video_vox{bv}.mp4'
        writer = None
        try:
            # Prefer FFMPEG for MP4; requires imageio-ffmpeg backend
            writer = imageio.get_writer(out_path, fps=int(fps), codec='libx264', format='FFMPEG')
        except Exception:
            # fallback to GIF with duration per frame
            out_path = out_dir / f'blur_video_vox{bv}.gif'
            try:
                writer = imageio.get_writer(out_path, duration=1.0/float(fps), loop=0, format='GIF')
            except Exception:
                writer = None
        for t in idxs:
            orig = spikes_TN[t]
            blurred = _apply_spatial_voxel_blur_torch(orig[None, :], positions_N3, mask, float(bv))[0]
            grid_o = _rasterize_to_grid(orig, positions_N3, plane=plane, grid_hw=grid_hw)
            grid_b = _rasterize_to_grid(blurred, positions_N3, plane=plane, grid_hw=grid_hw)
            # map to 0..1
            im_o = np.clip(grid_o / vmax, 0.0, 1.0)
            im_b = np.clip(grid_b / vmax, 0.0, 1.0)
            # stack side-by-side in RGB
            im_o_rgb = plt.cm.viridis(im_o)[..., :3]
            im_b_rgb = plt.cm.viridis(im_b)[..., :3]
            spacer = np.ones((grid_hw[0], 4, 3), dtype=np.float32)
            frame = np.concatenate([im_o_rgb, spacer, im_b_rgb], axis=1)
            if writer is not None:
                writer.append_data((frame * 255).astype(np.uint8))
        if writer is not None:
            writer.close()


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _repo_root() -> Path:
    # This file: <root>/GenerativeBrainModel/scripts/blur_sweep_subject14.py
    p = Path(__file__).resolve()
    # parents[0]=scripts, [1]=GenerativeBrainModel, [2]=project root
    return p.parents[2]


def _run_training_once(train_script: Path, cfg: Dict[str, Any], tmp_cfg_path: Path) -> Path | None:
    tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2, sort_keys=False)
    base_runs = Path('experiments/gbm2')
    before = _list_run_dirs(base_runs)
    cmd = [sys.executable, str(train_script), '--config', str(tmp_cfg_path)]
    try:
        env = os.environ.copy()
        env.setdefault('MKL_THREADING_LAYER', 'GNU')
        env.setdefault('OMP_NUM_THREADS', '1')
        env.setdefault('MKL_NUM_THREADS', '1')
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        # Ensure Python can import the local package
        repo = _repo_root()
        py_path = str(repo)
        if 'PYTHONPATH' in env and env['PYTHONPATH']:
            env['PYTHONPATH'] = py_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = py_path
        subprocess.run(cmd, check=True, env=env, cwd=str(repo))
    except subprocess.CalledProcessError as e:
        return None
    after = _list_run_dirs(base_runs)
    run_dir = _find_new_run_dir(before, after)
    _cleanup_cuda_memory()
    return run_dir


def _sweep(config: Dict[str, Any]) -> None:
    sweep_cfg = config.get('sweep', {})
    blur_levels: List[float] = [float(x) for x in sweep_cfg.get('blur_levels', [0.0, 2.0])]
    seeds: List[int] = [int(x) for x in sweep_cfg.get('seeds', [42, 123, 999])]
    subject_file: str = sweep_cfg.get('subject_file', 'subject_14.h5')
    out_root = Path(sweep_cfg.get('output_dir', 'experiments/gbm2_blur_sweep'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = out_dir / 'tmp_cfgs'

    # Load base training config (model/training defaults), defaulting to configs/config_gbm.yaml if present
    base_cfg_path = config.get('base_train_config', None)
    repo = _repo_root()
    if base_cfg_path is None:
        default_base = repo / 'configs' / 'config_gbm.yaml'
        base_cfg_path = str(default_base) if default_base.exists() else None
    base_cfg: Dict[str, Any] = {}
    if base_cfg_path is not None and Path(base_cfg_path).exists():
        with open(base_cfg_path, 'r') as bf:
            base_cfg = yaml.safe_load(bf) or {}

    train_overrides = config.get('train_overrides', {})
    # Ensure include_files and data_dir are set for the subject
    train_overrides = _deep_update(train_overrides, {
        'data': {
            'include_files': [subject_file],
        }
    })

    use_unblur_targets = bool(sweep_cfg.get('predict_unblurred_targets', True))
    train_script = _repo_root() / (
        'GenerativeBrainModel/scripts/train_gbm2_unblur_targets.py' if use_unblur_targets
        else 'GenerativeBrainModel/scripts/train_gbm2.py'
    )

    # Pre-training blur visualizations (scatter + videos)
    data_dir_vis = Path(train_overrides.get('data', {}).get('data_dir', 'processed_spike_voxels_2018'))
    _make_blur_visualization(data_dir_vis, subject_file, blur_levels, out_dir)
    _make_blur_videos(data_dir_vis, subject_file, blur_levels, out_dir, plane='xy', grid_hw=(512,512), frames=240, fps=3)
    _cleanup_cuda_memory()

    results: List[Tuple[float, int, float]] = []  # (blur, seed, final_val)
    per_run_records: List[Dict[str, Any]] = []

    total_runs = len(blur_levels) * len(seeds)
    sweep_pbar = tqdm(total=total_runs, desc='Sweep runs', unit='run')

    for bv in blur_levels:
        for seed in seeds:
            cfg = {
                'data': {
                    'spatial_blur_voxel_size': float(bv),
                },
                'training': {
                    'seed': int(seed),
                }
            }
            # Merge order: base (e.g., config_gbm.yaml) -> user overrides -> run-specific flags
            cfg_full = _deep_update(base_cfg, {'experiment': {'name': f'blur_sweep_b{bv}_s{seed}'}})
            cfg_full = _deep_update(cfg_full, train_overrides)
            cfg_full = _deep_update(cfg_full, cfg)
            # Preflight: ensure only subject_14.h5 is selected
            dd = cfg_full.get('data', {}).get('data_dir', 'processed_spike_voxels_2018')
            data_dir_abs = (_repo_root() / dd).resolve()
            include_list = cfg_full.get('data', {}).get('include_files', None)
            all_fs = sorted([p for p in data_dir_abs.glob('*.h5')])
            if include_list:
                include_set = {str(x) for x in include_list}
                kept = [p for p in all_fs if p.name in include_set]
            else:
                kept = all_fs
            if len(kept) != 1:
                raise RuntimeError(f"Expected exactly 1 subject after include_files filter, got {len(kept)} in {data_dir_abs}. include_files={include_list}")
            tmp_cfg = tmp_cfg_dir / f'config_blur{bv}_seed{seed}.yaml'
            run_dir = _run_training_once(train_script, cfg_full, tmp_cfg)
            if run_dir is None:
                sweep_pbar.update(1)
                sweep_pbar.set_postfix({'blur': bv, 'seed': seed, 'status': 'failed'})
                continue
            val_csv = run_dir / 'logs' / 'val_events.csv'
            last_val = _read_last_val_loss(val_csv)
            # Write per-run intra-epoch CSV from loss.txt
            _write_intraepoch_csv_from_loss(run_dir / 'logs' / 'loss.txt', run_dir / 'logs' / 'val_intraepoch.csv')
            results.append((float(bv), int(seed), float(last_val) if last_val is not None else float('nan')))
            per_run_records.append({
                'blur_voxel_size': float(bv),
                'seed': int(seed),
                'run_dir': str(run_dir),
                'val_csv': str(val_csv),
                'final_val_loss': float(last_val) if last_val is not None else float('nan'),
            })
            _cleanup_cuda_memory()
            sweep_pbar.update(1)
            sweep_pbar.set_postfix({'blur': bv, 'seed': seed, 'val': (None if last_val is None else float(last_val))})

            # Copy run loss.txt into sweep folder with descriptive name
            try:
                loss_src = run_dir / 'logs' / 'loss.txt'
                if loss_src.exists():
                    loss_dst = out_dir / f'loss_blur{bv}_seed{seed}.txt'
                    shutil.copyfile(str(loss_src), str(loss_dst))
            except Exception:
                pass

    sweep_pbar.close()
    # write per-run CSV
    per_run_csv = out_dir / 'per_run_results.csv'
    with open(per_run_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['blur_voxel_size', 'seed', 'final_val_loss', 'run_dir', 'val_csv'])
        writer.writeheader()
        for rec in per_run_records:
            writer.writerow(rec)

    # aggregate
    agg_rows = []
    for bv in sorted(set(x[0] for x in results)):
        vals = [r[2] for r in results if r[0] == bv and np.isfinite(r[2])]
        if len(vals) == 0:
            mean, std = float('nan'), float('nan')
        else:
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append({'blur_voxel_size': float(bv), 'final_val_mean': mean, 'final_val_std': std, 'n_runs': len(vals)})

    agg_csv = out_dir / 'aggregate_results.csv'
    with open(agg_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['blur_voxel_size', 'final_val_mean', 'final_val_std', 'n_runs'])
        writer.writeheader()
        for row in sorted(agg_rows, key=lambda r: r['blur_voxel_size']):
            writer.writerow(row)

    # plot final val vs blur
    xs = [row['blur_voxel_size'] for row in sorted(agg_rows, key=lambda r: r['blur_voxel_size'])]
    ys = [row['final_val_mean'] for row in sorted(agg_rows, key=lambda r: r['blur_voxel_size'])]
    es = [row['final_val_std'] for row in sorted(agg_rows, key=lambda r: r['blur_voxel_size'])]
    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, ys, yerr=es, fmt='-o', capsize=4)
    plt.xlabel('Spatial blur voxel size')
    plt.ylabel('Final validation loss')
    plt.title('Final val vs spatial blur')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'final_val_vs_blur.png', dpi=150)
    plt.close()

    # (visualizations already generated pre-training)

    print(f"Sweep complete. Outputs: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description='Sweep spatial blur voxel size and measure final validation loss (subject 14).')
    ap.add_argument('--config', type=str, default='configs/blur_sweep_subject14_test.yaml', help='YAML sweep config')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    _sweep(cfg)


if __name__ == '__main__':
    main()


