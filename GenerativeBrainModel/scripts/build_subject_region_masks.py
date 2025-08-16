#!/usr/bin/env python3
"""
Build subject-specific voxel masks for zebrafish brain regions by
registering a reference pointcloud to subject neuron positions using CPD, then
transforming each region's reference mask to the subject space and voxelizing.

Output: For each subject, an HDF5 file subject_X_mask.h5 containing:
 - group "regions/<region_name>" boolean masks stored as (X, Y, Z)
 - dataset "label_volume" int16 labels (0 = background, 1..R = regions), stored as (X, Y, Z)
 - dataset "region_names" array of UTF-8 strings mapping label ids to names
 - group "cpd" with registration parameters and metadata

Coordinate conventions:
 - Target grid is X,Y,Z = 512,256,128 and stored as array shape (X, Y, Z).
 - Reference TIF stacks load with shape (Z, Y, X). We convert to normalized
   physical coordinates in [0,1]^3 as (x, y, z) by dividing by (X-1, Y-1, Z-1).
 - Subject neuron positions are expected in H5 dataset 'cell_positions' with
   normalized [0,1] coordinates per axis (x, y, z). If not normalized, we
   normalize to [0,1] using per-axis min/max within the file.

Dependencies: numpy, h5py, tifffile, scipy, scikit-image, pycpd

Usage example:
  python -m GenerativeBrainModel.scripts.build_subject_region_masks \
    --subjects-glob \
      "/home/user/gbm3/GBM3/processed_spike_voxels_2018/subject_*.h5" \
    --masks-dir \
      "/home/user/gbm3/GBM3/masks" \
    --output-dir \
      "/home/user/gbm3/GBM3/processed_spike_voxels_2018_masks" \
    --cpd \
      affine

Notes:
 - If scikit-image is missing, fallback threshold uses percentile.
 - For speed, CPD is fit against a random subset of points, and we use
   gaussian_filter on occupancy grids to approximate Gaussian splatting.
"""

import os
import sys
import glob
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import h5py
import tifffile

from scipy.ndimage import gaussian_filter, binary_fill_holes

try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import ball, binary_closing, remove_small_holes
    from skimage.measure import label as label_components
except Exception:
    threshold_otsu = None
    ball = None
    binary_closing = None
    remove_small_holes = None
    label_components = None

try:
    from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
except Exception as e:
    RigidRegistration = AffineRegistration = DeformableRegistration = None


def _ensure_deps():
    missing = []
    if RigidRegistration is None:
        missing.append("pycpd")
    try:
        import scipy
    except Exception:
        missing.append("scipy")
    try:
        import tifffile as _tf
    except Exception:
        missing.append("tifffile")
    try:
        import h5py as _h
    except Exception:
        missing.append("h5py")
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}. Please install them.")


def load_reference_points_from_tif(tif_path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    vol = tifffile.imread(tif_path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D TIF at {tif_path}, got shape {vol.shape}")
    # Identify foreground coordinates (z, y, x)
    zz, yy, xx = np.nonzero(vol)
    Z, Y, X = vol.shape
    if zz.size == 0:
        return np.zeros((0, 3), dtype=np.float32), (Z, Y, X)
    # Normalize to [0,1] as (x, y, z)
    pts = np.stack([
        xx.astype(np.float32) / max(1, X - 1),
        yy.astype(np.float32) / max(1, Y - 1),
        zz.astype(np.float32) / max(1, Z - 1),
    ], axis=1)
    return pts, (Z, Y, X)


def load_subject_positions(subject_h5: str) -> np.ndarray:
    with h5py.File(subject_h5, "r") as f:
        if "cell_positions" not in f:
            raise ValueError(f"cell_positions not found in {subject_h5}")
        pos = f["cell_positions"][:].astype(np.float32)
        # pos expected shape (N,3) as (x,y,z)
        # If not normalized, normalize to [0,1]
        attrs = dict(f.attrs)
        normalized = bool(attrs.get("positions_normalized", False))
        if not normalized:
            pos_min = np.nanmin(pos, axis=0)
            pos_max = np.nanmax(pos, axis=0)
            rng = np.maximum(pos_max - pos_min, 1e-6)
            pos = (pos - pos_min) / rng
        # Sanitize
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        pos = np.clip(pos, 0.0, 1.0)
        return pos


def sample_points(points: np.ndarray, max_points: int, seed: int = 42) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def fit_cpd_transform(
    target_X: np.ndarray,
    source_Y: np.ndarray,
    method: str = "affine",
    w: float = 0.1,
    max_iters: int = 100,
    tolerance: float = 1e-5,
    low_rank: bool = False,
    beta: float = 2.0,
    alpha: float = 2.0,
    verbose: bool = True,
):
    if method == "rigid":
        reg = RigidRegistration(X=target_X, Y=source_Y, w=w, max_iterations=max_iters, tolerance=tolerance)
    elif method == "affine":
        reg = AffineRegistration(X=target_X, Y=source_Y, w=w, max_iterations=max_iters, tolerance=tolerance)
    elif method == "deformable":
        reg = DeformableRegistration(
            X=target_X,
            Y=source_Y,
            w=w,
            max_iterations=max_iters,
            tolerance=tolerance,
            low_rank=low_rank,
            beta=beta,
            alpha=alpha,
        )
    else:
        raise ValueError("method must be one of: rigid, affine, deformable")
    TY, params = reg.register()
    if verbose:
        if isinstance(params, tuple):
            # rigid/affine: (s, R, t)
            pass
    return reg, TY, params


def points_to_voxel_indices(points_xyzn: np.ndarray, grid_xyz: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = grid_xyz
    # points as (x,y,z) in [0,1]
    xi = np.clip(np.round(points_xyzn[:, 0] * (X - 1)).astype(np.int32), 0, X - 1)
    yi = np.clip(np.round(points_xyzn[:, 1] * (Y - 1)).astype(np.int32), 0, Y - 1)
    zi = np.clip(np.round(points_xyzn[:, 2] * (Z - 1)).astype(np.int32), 0, Z - 1)
    return xi, yi, zi


def build_density_grid(points_xyzn: np.ndarray, grid_shape_zyx: Tuple[int, int, int], sigma_vox: float = 1.25) -> np.ndarray:
    Z, Y, X = grid_shape_zyx
    if points_xyzn.size == 0:
        return np.zeros((Z, Y, X), dtype=np.float32)
    xi, yi, zi = points_to_voxel_indices(points_xyzn, (X, Y, Z))
    occ = np.zeros((Z, Y, X), dtype=np.float32)
    # Accumulate occupancy
    np.add.at(occ, (zi, yi, xi), 1.0)
    # Gaussian smoothing approximates splatting
    den = gaussian_filter(occ, sigma=sigma_vox, mode="nearest")
    return den.astype(np.float32)


def postprocess_binary(mask: np.ndarray, close_radius: int = 1, min_hole_vol: int = 8) -> np.ndarray:
    out = mask.copy()
    if binary_closing is not None and ball is not None:
        out = binary_closing(out, ball(close_radius))
    # Fill internal holes slice-wise and in 3D
    out = binary_fill_holes(out)
    if remove_small_holes is not None:
        out = remove_small_holes(out, area_threshold=int(min_hole_vol), connectivity=3)
    # Keep largest component if available
    if label_components is not None:
        lbl = label_components(out, connectivity=3)
        if lbl.max() > 1:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            keep = np.argmax(counts)
            out = (lbl == keep)
    return out.astype(bool)


def _compute_threshold(den: np.ndarray) -> float:
    if threshold_otsu is not None:
        try:
            vals = den[den > 0]
            if vals.size >= 256:
                return float(threshold_otsu(vals))
        except Exception:
            pass
    return float(np.percentile(den, 99))


def save_subject_masks(
    output_h5: str,
    label_vol: np.ndarray,
    region_names: List[str],
    subject_h5: str,
    cpd_info: Dict,
    grid_shape_zyx: Tuple[int, int, int],
):
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        # Regions
        grp_regions = f.create_group("regions")
        # Write exclusive masks derived from label volume one-by-one to limit memory
        for idx, name in enumerate(region_names):
            m_zyx = (label_vol == (idx + 1))
            # store as (X,Y,Z)
            m_xyz = np.transpose(m_zyx, (2, 1, 0))
            ds = grp_regions.create_dataset(name, data=m_xyz.astype(np.uint8), compression="gzip", compression_opts=1)
            ds.attrs["dtype"] = "bool"

        # Labels
        # store as (X,Y,Z)
        label_xyz = np.transpose(label_vol, (2, 1, 0))
        f.create_dataset("label_volume", data=label_xyz, compression="gzip", compression_opts=1)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("region_names", data=np.array(region_names, dtype=object), dtype=dt)

        # Metadata
        f.attrs["source_subject_file"] = os.path.abspath(subject_h5)
        f.attrs["grid_shape_zyx"] = np.array(grid_shape_zyx, dtype=np.int32)
        f.attrs["grid_shape_xyz"] = np.array([grid_shape_zyx[2], grid_shape_zyx[1], grid_shape_zyx[0]], dtype=np.int32)
        f.attrs["axes_order"] = "stored=(X,Y,Z); coords=(x,y,z)"
        f.attrs["generator"] = "build_subject_region_masks.py"

        # CPD params
        grp_cpd = f.create_group("cpd")
        for k, v in cpd_info.items():
            if isinstance(v, (str, int, float, np.integer, np.floating)):
                grp_cpd.attrs[k] = v
            else:
                try:
                    grp_cpd.create_dataset(k, data=np.array(v))
                except Exception:
                    grp_cpd.attrs[k] = json.dumps(v)


def main():
    parser = argparse.ArgumentParser(description="Build subject region voxel masks via CPD registration")
    parser.add_argument("--subjects-glob", type=str, default="/home/user/gbm3/GBM3/processed_spike_voxels_2018/*.h5",
                        help="Glob for subject H5 files containing 'cell_positions'")
    parser.add_argument("--masks-dir", type=str, default="/home/user/gbm3/GBM3/masks",
                        help="Directory of reference region TIF masks, including whole_brain.tif")
    parser.add_argument("--output-dir", type=str, default="/home/user/gbm3/GBM3/processed_spike_voxels_2018_masks",
                        help="Directory to write per-subject mask H5 files")
    parser.add_argument("--grid", type=str, default="512,256,128",
                        help="Grid as X,Y,Z (saved as X,Y,Z). Default 512,256,128")
    parser.add_argument("--cpd", type=str, choices=["rigid", "affine", "deformable"], default="affine",
                        help="CPD registration type")
    parser.add_argument("--cpd-w", type=float, default=0.1, help="CPD noise parameter w in [0,1)")
    parser.add_argument("--cpd-iters", type=int, default=100, help="Max iterations for CPD")
    parser.add_argument("--cpd-tol", type=float, default=1e-5, help="Tolerance for CPD convergence")
    parser.add_argument("--cpd-beta", type=float, default=2.0, help="Beta for deformable")
    parser.add_argument("--cpd-alpha", type=float, default=2.0, help="Alpha for deformable")
    parser.add_argument("--cpd-low-rank", action="store_true", help="Use low-rank for deformable")
    parser.add_argument("--max-cpd-points", type=int, default=20000,
                        help="Max points to sample from subject and reference for CPD")
    parser.add_argument("--sigma-vox", type=float, default=1.25, help="Gaussian sigma (in voxels) for splatting")
    parser.add_argument("--close-radius", type=int, default=1, help="Binary closing ball radius (voxels)")
    parser.add_argument("--min-hole-vol", type=int, default=8, help="Minimum hole volume to remove (voxels)")
    parser.add_argument("--percentile-thr", type=float, default=99.0, help="Percentile for threshold fallback")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    _ensure_deps()

    # Grid
    gx, gy, gz = [int(x) for x in args.grid.split(",")]
    grid_xyz = (gx, gy, gz)
    grid_zyx = (gz, gy, gx)

    # Region list
    all_tifs = sorted(glob.glob(os.path.join(args.masks_dir, "*.tif")))
    if not all_tifs:
        raise RuntimeError(f"No TIFs found in {args.masks_dir}")

    # Build CPD reference by concatenating all regions EXCEPT whole_brain and eye/retina-related
    def _is_excluded_for_ref(name: str) -> bool:
        n = name.lower()
        # Exclude only exact base names: whole_brain and retina
        return n == "whole_brain" or n == "retina"

    ref_pts_list: List[np.ndarray] = []
    for p in all_tifs:
        base = os.path.splitext(os.path.basename(p))[0]
        if _is_excluded_for_ref(base):
            continue
        pts, _ = load_reference_points_from_tif(p)
        if pts.size:
            ref_pts_list.append(pts)
    if not ref_pts_list:
        raise RuntimeError("No reference points collected from region TIFs (after exclusions)")
    ref_pts = np.concatenate(ref_pts_list, axis=0)

    # Region files for mask generation: exclude only whole_brain (we still label retina unless desired otherwise)
    mask_files = [p for p in all_tifs if os.path.basename(p) != "whole_brain.tif"]

    subjects = sorted(glob.glob(args.subjects_glob))
    if not subjects:
        print(f"No subject files matched {args.subjects_glob}")
        return

    print(f"Found {len(subjects)} subjects; {len(mask_files)} regions (CPD ref from {sum(1 for p in all_tifs if not _is_excluded_for_ref(os.path.splitext(os.path.basename(p))[0]))} files)")
    print(f"Grid internal (Z,Y,X)={grid_zyx}, saved (X,Y,Z)={grid_xyz}")

    for subject_h5 in subjects:
        subject_name = os.path.splitext(os.path.basename(subject_h5))[0]
        out_path = os.path.join(args.output_dir, f"{subject_name}_mask.h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Skipping {subject_name}: {out_path} exists (use --overwrite to replace)")
            continue

        print(f"\nSubject: {subject_name}")
        subj_pts = load_subject_positions(subject_h5)
        if subj_pts.shape[0] < 10:
            print(f"  Too few subject points ({subj_pts.shape[0]}), skipping")
            continue

        # Fit CPD on subsamples for speed
        subj_sample = sample_points(subj_pts, args.max_cpd_points)
        ref_sample = sample_points(ref_pts, args.max_cpd_points)
        print(f"  CPD fit: target (subject) {subj_sample.shape}, source (reference) {ref_sample.shape}, method={args.cpd}")
        reg, TY, params = fit_cpd_transform(
            subj_sample, ref_sample,
            method=args.cpd,
            w=args.cpd_w,
            max_iters=args.cpd_iters,
            tolerance=args.cpd_tol,
            low_rank=args.cpd_low_rank,
            beta=args.cpd_beta,
            alpha=args.cpd_alpha,
            verbose=True,
        )

        # Streamed multi-label construction to reduce memory
        z_dim, y_dim, x_dim = grid_zyx
        best_den = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)
        best_lbl = np.zeros((z_dim, y_dim, x_dim), dtype=np.int16)
        region_names: List[str] = []
        for idx, tif_path in enumerate(mask_files):
            name = os.path.splitext(os.path.basename(tif_path))[0]
            region_names.append(name)
            pts, _ = load_reference_points_from_tif(tif_path)
            if pts.shape[0] == 0:
                continue
            try:
                pts_t = reg.transform_point_cloud(pts)
            except Exception:
                pts_t = pts
            pts_t = np.clip(pts_t, 0.0, 1.0)
            den = build_density_grid(pts_t, grid_zyx, sigma_vox=args.sigma_vox)
            thr = _compute_threshold(den)
            mask = den >= thr
            mask = postprocess_binary(mask, close_radius=args.close_radius, min_hole_vol=args.min_hole_vol)
            update = mask & (den > best_den)
            if np.any(update):
                best_den[update] = den[update]
                best_lbl[update] = np.int16(idx + 1)
        label_vol = best_lbl

        # Prepare CPD info to save
        cpd_info: Dict = {
            "method": args.cpd,
            "w": args.cpd_w,
            "iters": args.cpd_iters,
            "tol": args.cpd_tol,
            "beta": args.cpd_beta,
            "alpha": args.cpd_alpha,
            "low_rank": bool(args.cpd_low_rank),
            "ref_points_used": int(ref_sample.shape[0]),
            "subj_points_used": int(subj_sample.shape[0]),
        }
        if isinstance(params, tuple) and len(params) == 3:
            s_reg, R_reg, t_reg = params
            cpd_info["scale"] = float(s_reg)
            cpd_info["rotation"] = np.array(R_reg, dtype=np.float32)
            cpd_info["translation"] = np.array(t_reg, dtype=np.float32)

        print(f"  Saving {out_path}")
        save_subject_masks(out_path, label_vol, region_names, subject_h5, cpd_info, grid_zyx)


if __name__ == "__main__":
    # Allow module entry via -m as well
    main()


