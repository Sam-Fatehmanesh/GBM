"""
Neuron-based visualization helpers: create comparison videos of next-step and
autoregressive predictions using 2D scatter at median Z plane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import cv2
from tqdm import tqdm


def _render_frame_2d(points_xy: np.ndarray, values_uint8: np.ndarray, width: int = 512, height: int = 512) -> np.ndarray:
    """
    Mean-pool along Z by projecting all 3D points to (x,y) and averaging values
    of collisions in image space. `values_uint8` must be in [0,255].
    """
    W, H = int(width), int(height)
    xs = np.clip((points_xy[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    ys = np.clip((points_xy[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    vals = values_uint8.astype(np.float32)

    sums = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)
    np.add.at(sums, (ys, xs), vals)
    np.add.at(counts, (ys, xs), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        img = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def _select_median_z(points_xyz: np.ndarray):
    """Return mask of neurons at the median Z slice (closest to median).
    Falls back to using all neurons if the slice is too sparse.
    """
    z = points_xyz[:, 2]
    # Handle NaNs gracefully
    if not np.isfinite(z).all() or z.size == 0:
        return np.ones(points_xyz.shape[0], dtype=bool)
    z_med = float(np.median(z))
    abs_diff = np.abs(z - z_med)
    min_diff = float(np.min(abs_diff))
    mask = abs_diff <= (min_diff + 1e-8)
    # If the median plane has too few points, render all neurons (dense projection)
    if mask.sum() < max(50, int(0.01 * points_xyz.shape[0])):
        mask = np.ones(points_xyz.shape[0], dtype=bool)
    return mask


def create_nextstep_video(
    original: torch.Tensor,  # (B,T,N)
    predicted: torch.Tensor,  # (B,T,N)
    positions: torch.Tensor,  # (B,N,3) normalized [0,1]
    out_path: Path,
    fps: int = 2,
    panel_width: int = 512,
    panel_height: int = 256,
) -> Path:
    """Create side-by-side next-step comparison video using median Z plane."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer mp4v, but fall back to common codecs if unavailable
    frame_size = (panel_width * 2, panel_height)
    def _open_writer(path: Path):
        for codec in ('mp4v', 'avc1', 'H264', 'XVID', 'MJPG'):
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if w.isOpened():
                return w
        return cv2.VideoWriter(str(path), 0, fps, frame_size)
    writer = _open_writer(out_path)

    try:
        # Ensure numpy float32 for math, then we convert to uint8 per-frame
        original = original.detach().to(torch.float32).cpu().numpy()
        predicted = predicted.detach().to(torch.float32).cpu().numpy()
        positions = positions.detach().to(torch.float32).cpu().numpy()

        B, T, N = original.shape
        if B == 0 or T == 0:
            return out_path
        font = cv2.FONT_HERSHEY_SIMPLEX
        print(f"[VideoGen] nextstep: B={B}, T={T}, N={N}, panel=({panel_width},{panel_height})")
        for b in tqdm(range(B), desc='Video Batches'):
            pos_b = positions[b]  # (N,3)
            xy = pos_b[:, :2]     # use all neurons and mean-pool overlaps along Z
            if not np.isfinite(xy).all():
                print(f"[VideoGen] NaNs in positions for batch {b}; replacing with zeros.")
                xy = np.nan_to_num(xy, nan=0.0)
            for t in range(T):
                orig_vals = original[b, t]
                pred_vals = predicted[b, t]
                if not np.isfinite(orig_vals).all() or orig_vals.size == 0:
                    print(f"[VideoGen] orig invalid at b={b} t={t}: shape={orig_vals.shape} minmax={[float(np.nanmin(orig_vals)) if orig_vals.size else None, float(np.nanmax(orig_vals)) if orig_vals.size else None]}")
                if not np.isfinite(pred_vals).all() or pred_vals.size == 0:
                    print(f"[VideoGen] pred invalid at b={b} t={t}: shape={pred_vals.shape} minmax={[float(np.nanmin(pred_vals)) if pred_vals.size else None, float(np.nanmax(pred_vals)) if pred_vals.size else None]}")
                # Normalize each independently to [0,255]
                def _to_u8(a: np.ndarray) -> np.ndarray:
                    if a.size == 0:
                        return a.astype(np.uint8)
                    a = a.astype(np.float32)
                    amin = float(a.min())
                    amax = float(a.max())
                    if not np.isfinite(amin) or not np.isfinite(amax):
                        return np.zeros_like(a, dtype=np.uint8)
                    if amax <= amin + 1e-12:
                        # If completely flat, produce a faint 1-LSB image to make presence visible
                        return np.full_like(a, 1, dtype=np.uint8)
                    a = (a - amin) / (amax - amin)
                    a = np.clip(a * 255.0, 0.0, 255.0)
                    return a.astype(np.uint8)
                orig_u8 = _to_u8(orig_vals)
                pred_u8 = _to_u8(pred_vals)
                orig_img = _render_frame_2d(xy, orig_u8, panel_width, panel_height)
                pred_img = _render_frame_2d(xy, pred_u8, panel_width, panel_height)
                frame = np.concatenate([orig_img, pred_img], axis=1)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Labels: left Truth, right Pred, batch/time indices
                cv2.putText(frame_bgr, 'Truth', (10, 24), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, 'Pred', (panel_width + 10, 24), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, f'batch {b}  t {t}', (10, panel_height - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame_bgr)
    finally:
        writer.release()

    return out_path


def create_autoregression_video(
    generated: torch.Tensor,  # (B,T,N) generated after context
    positions: torch.Tensor,  # (B,N,3)
    out_path: Path,
    truth: Optional[torch.Tensor] = None,  # (B,T,N) ground-truth for same horizon
    fps: int = 2,
    panel_width: int = 512,
    panel_height: int = 256,
) -> Path:
    """Create autoregression comparison video (Truth | AR Pred) using mean-pooled XY."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    two_panel = truth is not None
    frame_size = (panel_width * 2, panel_height) if two_panel else (panel_width, panel_height)
    def _open_writer(path: Path):
        for codec in ('mp4v', 'avc1', 'H264', 'XVID', 'MJPG'):
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
            if w.isOpened():
                return w
        return cv2.VideoWriter(str(path), 0, fps, frame_size)
    writer = _open_writer(out_path)

    try:
        generated = generated.detach().to(torch.float32).cpu().numpy()
        positions = positions.detach().to(torch.float32).cpu().numpy()
        truth_np = None if truth is None else truth.detach().to(torch.float32).cpu().numpy()
        B, T_gen, N = generated.shape
        T = T_gen if truth_np is None else min(T_gen, truth_np.shape[1])
        if B == 0 or T == 0:
            return out_path
        font = cv2.FONT_HERSHEY_SIMPLEX
        for b in tqdm(range(B), desc='AR Video Batches'):
            pos_b = positions[b]
            xy = pos_b[:, :2]
            for t in range(T):
                vals_pred = generated[b, t]
                # Normalize to [0,255]
                def to_img(u: np.ndarray) -> np.ndarray:
                    if u.size == 0:
                        return np.zeros((panel_height, panel_width), dtype=np.uint8)
                    amin = float(u.min()); amax = float(u.max())
                    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin + 1e-12:
                        return np.zeros((panel_height, panel_width), dtype=np.uint8)
                    u8 = ((u - amin) / (amax - amin) * 255.0).clip(0,255).astype(np.uint8)
                    return _render_frame_2d(xy, u8, panel_width, panel_height)
                pred_img = to_img(vals_pred)
                if two_panel:
                    vals_truth = truth_np[b, t]
                    truth_img = to_img(vals_truth)
                    frame = np.concatenate([truth_img, pred_img], axis=1)
                else:
                    frame = pred_img
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if two_panel:
                    cv2.putText(frame_bgr, 'Truth', (10, 24), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame_bgr, 'AR Pred', (panel_width + 10, 24), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, f'batch {b}  t {t}', (10, panel_height - 10), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame_bgr)
    finally:
        writer.release()

    return out_path


