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


def _render_frame_2d(points_xy: np.ndarray, values: np.ndarray, img_size: int = 512) -> np.ndarray:
    """
    Render neuron values as a scatter image.
    points_xy: (N,2) in [0,1] range.
    values: (N,) in [0,1].
    """
    H = W = img_size
    img = np.zeros((H, W), dtype=np.float32)

    xs = np.clip((points_xy[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    ys = np.clip((points_xy[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    vals = np.clip(values.astype(np.float32), 0.0, 1.0)

    # Accumulate max to visualize strongest activity per pixel (vectorized)
    np.maximum.at(img, (ys, xs), vals)

    img_u8 = (img * 255.0).astype(np.uint8)
    return img_u8


def _select_median_z(points_xyz: np.ndarray):
    """Return mask of neurons at the median Z slice (closest to median)."""
    z = points_xyz[:, 2]
    z_med = np.median(z)
    # choose those within minimal absolute difference to median (exact plane if normalized)
    abs_diff = np.abs(z - z_med)
    min_diff = np.min(abs_diff)
    mask = abs_diff <= (min_diff + 1e-8)
    return mask


def create_nextstep_video(
    original: torch.Tensor,  # (B,T,N)
    predicted: torch.Tensor,  # (B,T,N)
    positions: torch.Tensor,  # (B,N,3) normalized [0,1]
    out_path: Path,
    fps: int = 2,
    img_size: int = 512,
) -> Path:
    """Create side-by-side next-step comparison video using median Z plane."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (img_size * 2, img_size))

    try:
        original = original.detach().cpu().numpy()
        predicted = predicted.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy()

        B, T, N = original.shape
        for b in tqdm(range(B), desc='Video Batches'):
            pos_b = positions[b]  # (N,3)
            mask = _select_median_z(pos_b)
            xy = pos_b[mask, :2]
            for t in range(T):
                orig_vals = original[b, t, mask]
                pred_vals = predicted[b, t, mask]
                orig_img = _render_frame_2d(xy, orig_vals, img_size)
                pred_img = _render_frame_2d(xy, pred_vals, img_size)
                frame = np.concatenate([orig_img, pred_img], axis=1)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer.write(frame_bgr)
    finally:
        writer.release()

    return out_path


def create_autoregression_video(
    generated: torch.Tensor,  # (B,T,N) generated after context
    positions: torch.Tensor,  # (B,N,3)
    out_path: Path,
    fps: int = 2,
    img_size: int = 512,
) -> Path:
    """Create autoregression-only video using median Z plane (no ground truth)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (img_size, img_size))

    try:
        generated = generated.detach().cpu().numpy()
        positions = positions.detach().cpu().numpy()
        B, T, N = generated.shape
        for b in tqdm(range(B), desc='AR Video Batches'):
            pos_b = positions[b]
            mask = _select_median_z(pos_b)
            xy = pos_b[mask, :2]
            for t in range(T):
                vals = generated[b, t, mask]
                img = _render_frame_2d(xy, vals, img_size)
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                writer.write(frame_bgr)
    finally:
        writer.release()

    return out_path


