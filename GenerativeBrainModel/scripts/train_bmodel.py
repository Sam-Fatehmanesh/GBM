#!/usr/bin/env python3
"""
Training script for behavior prediction from single-timepoint neural states.

- Uses BehaviorDataset with sequence_length=1, stride=1
- Upsamples behavior_full to neural T and aligns after trimming neural zero margins
- Trains BModel to map (spikes_t, positions) -> behavior_t in [0,1]
- Logs training/validation losses and produces predicted vs true plots per behavior dim
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
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from GenerativeBrainModel.models.bmodel import BModel
from GenerativeBrainModel.dataloaders.behavior_dataloader import (
    create_behavior_dataloaders,
    _max_neurons_in_files,
    _max_behavior_dim,
)


def create_default_config() -> Dict[str, Any]:
    return {
        "experiment": {
            "name": "bmodel_behavior_training",
        },
        "data": {
            "data_dir": "processed_spike_voxels_2018",
            "test_subjects": [],
            "use_cache": True,
        },
        "model": {
            "d_behavior": None,  # inferred from data
            "d_max_neurons": None,  # inferred from data
        },
        "training": {
            "batch_size": 128,
            "num_epochs": 15,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "lr_warmup_ratio": 0.1,  # linear warmup fraction of total steps
            "gradient_clip_norm": 1.0,
            # Per-element BCE weighting: loss *= (eps + target)^power
            "element_weighting": {
                "enable": True,
                "eps": 1e-7,
                "power": 1.0,
            },
            "seed": 42,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "use_gpu": True,
            "max_timepoints_per_subject": None,
            "start_timepoint": None,
            "end_timepoint": None,
            "validation_frequency": 2,
        },
        "logging": {
            "log_level": "INFO",
        },
    }


def setup_experiment_dirs(base_dir: Path, name: str) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{name}_{ts}"
    log_dir = exp_dir / "logs"
    plots_dir = log_dir / "plots"
    ckpt_dir = exp_dir / "checkpoints"
    for p in [exp_dir, log_dir, plots_dir, ckpt_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"exp": exp_dir, "logs": log_dir, "plots": plots_dir, "ckpt": ckpt_dir}


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)


def build_logger(log_dir: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("train_bmodel")
    logger.setLevel(getattr(logging, level))
    fh = logging.FileHandler(log_dir / "training.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_architecture_file(
    model: nn.Module, dirs: Dict[str, Path], cfg: Dict[str, Any]
) -> None:
    out_path = dirs["exp"] / "architecture.txt"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params
    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("BMODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for k, v in cfg["model"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nPARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total:        {total_params:,}\n")
        f.write(f"  Trainable:    {trainable_params:,}\n")
        f.write(f"  Non-trainable:{untrainable_params:,}\n\n")
        f.write("MODEL STRUCTURE (repr)\n")
        f.write("-" * 40 + "\n")
        f.write(str(model))
        f.write("\n")


def train_one_epoch(
    model: BModel,
    loader,
    device: torch.device,
    optimizer,
    epoch: int,
    logger: logging.Logger,
    loss_history: list,
    global_step: int,
    scheduler: Optional[optim.lr_scheduler.LambdaLR] = None,
    clip_norm: Optional[float] = None,
    target_dim: int = 0,
    pos_weight: Optional[torch.Tensor] = None,
) -> tuple[float, int]:
    model.train()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    total_batches = 0
    pbar = tqdm(loader, desc=f"Train E{epoch}", mininterval=0.1, smoothing=0.05)
    for batch in pbar:
        spikes = batch["spikes"].to(device)  # (B, L, N) where L=6
        positions = batch["positions"].to(device)  # (B, N, 3)
        behavior = batch["behavior"].to(device)  # (B, K)
        mask_full = batch["neuron_mask"].to(device)  # (B, N)

        # Flatten spikes across neuron dim with positions like model expects
        # BModel expects x: (B, T, N) with T=L (window)
        x = spikes
        # Select target dim and ensure shape (B, 1)
        target = behavior[:, target_dim : target_dim + 1]

        # Prepare balanced views (default to full batch)
        x_use = spikes
        pos_use = positions
        tgt_use = target
        mask_use = mask_full

        # Balance batch by trimming near-mean samples: keep count(near) == count(> mean+std)
        with torch.no_grad():
            mu = target.mean()
            sd = target.std()
            thr = mu + sd
            pos_idx = (target.squeeze(-1) > thr).nonzero(as_tuple=False).squeeze(-1)
            neg_idx = (target.squeeze(-1) <= thr).nonzero(as_tuple=False).squeeze(-1)
            if pos_idx.numel() > 0 and neg_idx.numel() > 0:
                k = min(pos_idx.numel(), neg_idx.numel())
                # Random subset of negatives to match positives
                perm = torch.randperm(neg_idx.numel(), device=device)
                neg_keep = neg_idx[perm[:k]]
                keep_idx = torch.cat([pos_idx, neg_keep], dim=0)
                # Shuffle to avoid ordering bias
                shuffle = torch.randperm(keep_idx.numel(), device=device)
                keep_idx = keep_idx[shuffle]
                # Index tensors along batch dimension
                x_use = spikes.index_select(0, keep_idx)
                pos_use = positions.index_select(0, keep_idx)
                tgt_use = target.index_select(0, keep_idx)
                mask_use = mask_full.index_select(0, keep_idx)
        optimizer.zero_grad()
        preds = model(x_use, pos_use, mask_use, get_logits=True).squeeze(
            1
        )  # (b_bal, 1)
        loss = loss_fn(preds.float(), tgt_use.float())
        # No per-element weighting
        loss.backward()
        if clip_norm is not None and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        # Warmup: step scheduler BEFORE optimizer to apply warmup on the very first batch
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        # Record history (global step, loss)
        loss_history.append((global_step, float(loss.detach().cpu().item())))
        global_step += 1
        pbar.set_postfix({"loss": f"{float(loss.detach().cpu().item()):.6f}"})
    avg = total_loss / max(1, total_batches)
    logger.info(f"Epoch {epoch} train loss: {avg:.6f}")
    return avg, global_step


@torch.no_grad()
def validate(
    model: BModel,
    loader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    *,
    target_dim: int = 0,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
    model.eval()
    loss_fn = nn.L1Loss()
    total_loss = 0.0
    total_batches = 0
    # Aggregate per-file predictions and truths
    per_file: Dict[str, Dict[str, list]] = {}
    for i, batch in enumerate(tqdm(loader, desc=f"Val E{epoch}")):
        spikes = batch["spikes"].to(device)
        positions = batch["positions"].to(device)
        behavior = batch["behavior"].to(device)
        preds = model(
            spikes, positions, batch.get("neuron_mask", None), get_logits=True
        ).squeeze(1)
        target = behavior[:, target_dim : target_dim + 1]
        loss = loss_fn(preds.float(), target.float())
        # No per-element weighting in validation
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        probs = torch.clamp(preds.float(), 0.0, 1.0).detach().cpu().numpy()
        truths = target.detach().cpu().numpy()
        paths = batch["file_path"]
        starts = batch["start_idx"].detach().cpu().numpy()
        for b in range(probs.shape[0]):
            fp = paths[b]
            if fp not in per_file:
                per_file[fp] = {"idx": [], "pred": [], "true": []}
            per_file[fp]["idx"].append(int(starts[b]))
            per_file[fp]["pred"].append(probs[b])
            per_file[fp]["true"].append(truths[b])
    avg = total_loss / max(1, total_batches)

    # Correlation coefficients per validation subject (file)
    corrs: list[float] = []
    for fp, d in per_file.items():
        try:
            order = np.argsort(np.array(d["idx"]))
            pred_fp = np.stack(d["pred"], axis=0)[order][:, 0]
            true_fp = np.stack(d["true"], axis=0)[order][:, 0]
            if np.std(pred_fp) > 0 and np.std(true_fp) > 0:
                corr = float(np.corrcoef(pred_fp, true_fp)[0, 1])
            else:
                corr = float("nan")
            corrs.append(corr)
        except Exception:
            corrs.append(float("nan"))

    if corrs:
        mean_corr = float(np.nanmean(np.array(corrs, dtype=np.float64)))
        logger.info(f"Epoch {epoch} val loss: {avg:.6f} | corr_mean: {mean_corr:.4f}")
    else:
        logger.info(f"Epoch {epoch} val loss: {avg:.6f}")

    # Build a single subject aggregate (first file) for plotting
    if per_file:
        first_fp = sorted(per_file.keys())[0]
        order = np.argsort(np.array(per_file[first_fp]["idx"]))
        pred = np.stack(per_file[first_fp]["pred"], axis=0)[order]  # (T, 1)
        true = np.stack(per_file[first_fp]["true"], axis=0)[order]  # (T, 1)
        return avg, {"file_path": first_fp, "pred": pred, "true": true}
    return avg, None


def plot_pred_vs_true_dim(
    sample: Dict[str, np.ndarray], plots_dir: Path, epoch: int, dim: int
) -> None:
    y_true = sample["true"]  # (T, 1)
    y_pred = sample["pred"]  # (T, 1)
    T = y_true.shape[0]
    time = np.arange(T)
    plt.figure(figsize=(8, 3))
    plt.plot(time, y_true[:, 0], label="true", linewidth=1)
    plt.plot(time, y_pred[:, 0], label="pred", linewidth=1)
    plt.ylim(-0.05, 1.05)
    plt.title(f"Dim {dim}")
    plt.xlabel("time")
    plt.ylabel("0..1")
    plt.legend(fontsize="small")
    plt.tight_layout()
    out_path = plots_dir / f"pred_vs_true_dim{dim}_epoch_{epoch}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_training_loss(loss_history: list, plots_dir: Path, logs_dir: Path) -> None:
    if not loss_history:
        return
    steps = np.array([s for s, _ in loss_history], dtype=np.int64)
    losses = np.array([l for _, l in loss_history], dtype=np.float64)
    # Save CSV
    try:
        import csv

        with open(logs_dir / "train_loss.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss"])
            for s, l in loss_history:
                w.writerow([int(s), float(l)])
    except Exception:
        pass
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("step")
    plt.ylabel("train loss (L1)")
    plt.title("Training loss over time")
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss.png", dpi=120)
    plt.close()


@torch.no_grad()
def evaluate_subjectwise(
    model: BModel, loader, device: torch.device, *, target_dim: int, loss_fn: nn.Module
) -> list[dict]:
    """Evaluate per subject (file_path) over the provided loader.
    Returns a list of dict rows with keys: file_path, num_samples, loss_mean, corr.
    """
    model.eval()
    # Aggregators per file
    per_file = {}
    for batch in tqdm(loader, desc="Per-subject eval"):
        spikes = batch["spikes"].to(device)
        positions = batch["positions"].to(device)
        mask = batch["neuron_mask"].to(device)
        behavior = batch["behavior"].to(device)
        start_idx = batch["start_idx"]
        paths = batch["file_path"]

        preds = model(spikes, positions, mask, get_logits=True).squeeze(1)  # (B,1)
        target = behavior[:, target_dim : target_dim + 1]
        # Per-sample L1 (or other) loss
        sample_loss = (
            torch.mean(torch.abs(preds.float() - target.float()), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        preds_np = preds.detach().cpu().numpy()[:, 0]
        target_np = target.detach().cpu().numpy()[:, 0]
        idx_np = start_idx.detach().cpu().numpy()

        for i in range(len(paths)):
            fp = paths[i]
            d = per_file.setdefault(fp, {"idx": [], "pred": [], "true": [], "loss": []})
            d["idx"].append(int(idx_np[i]))
            d["pred"].append(float(preds_np[i]))
            d["true"].append(float(target_np[i]))
            d["loss"].append(float(sample_loss[i]))

    rows = []
    for fp, d in per_file.items():
        order = np.argsort(np.array(d["idx"]))
        pred = np.array(d["pred"])[order]
        true = np.array(d["true"])[order]
        loss_mean = float(np.mean(np.array(d["loss"]))) if d["loss"] else float("nan")
        if np.std(pred) > 0 and np.std(true) > 0:
            corr = float(np.corrcoef(pred, true)[0, 1])
        else:
            corr = float("nan")
        rows.append(
            {
                "file_path": fp,
                "num_samples": int(len(d["idx"])),
                "loss_mean": loss_mean,
                "corr": corr,
            }
        )
    return rows


def estimate_pos_weight(train_loader, target_dim: int) -> float:
    """Estimate pos_weight for BCE from the mean of the target dim.
    pos_weight = (1 - p) / p where p = mean(target in [0,1]).
    Safeguards zero by clamping p.
    """
    total = 0.0
    count = 0
    for batch in train_loader:
        y = batch["behavior"]  # (B, K)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        # Clamp to [0,1] just in case
        y_dim = torch.clamp(y[:, target_dim].float(), 0.0, 1.0)
        total += float(y_dim.sum().item())
        count += int(y_dim.numel())
    if count == 0:
        return 1.0
    p = max(1e-6, min(1.0 - 1e-6, total / count))
    return float((1.0 - p) / p)


def main():
    parser = argparse.ArgumentParser(
        description="Train behavior predictor (BModel) from neural states"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/train_bmodel.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    cfg = create_default_config()
    if args.config:
        with open(args.config, "r") as f:
            user = yaml.safe_load(f)
        # Shallow update
        for k, v in user.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    # Dirs & logger
    base_dir = Path("experiments/bmodel_behavior")
    dirs = setup_experiment_dirs(base_dir, cfg["experiment"]["name"])
    save_config(cfg, dirs["exp"] / "config.yaml")
    logger = build_logger(dirs["logs"], cfg["logging"].get("log_level", "INFO"))

    # Device & seeds
    device = torch.device(
        "cuda" if (cfg["training"]["use_gpu"] and torch.cuda.is_available()) else "cpu"
    )
    set_seeds(cfg["training"]["seed"])
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")

    # Data
    train_loader, val_loader, _, _ = create_behavior_dataloaders(cfg)

    # Infer dims from data
    data_dir = Path(cfg["data"]["data_dir"])
    all_files = sorted([str(f) for f in data_dir.glob("*.h5")])
    d_behavior = _max_behavior_dim(all_files)
    d_max_neurons = _max_neurons_in_files(all_files)
    # Train a separate model per behavior dimension
    for dim in range(d_behavior):
        logger.info(f"=== Training model for behavior dim {dim}/{d_behavior - 1} ===")
        # Create per-dim subdirs
        dim_ckpt = dirs["ckpt"] / f"dim_{dim}"
        dim_plots = dirs["plots"] / f"dim_{dim}"
        dim_ckpt.mkdir(parents=True, exist_ok=True)
        dim_plots.mkdir(parents=True, exist_ok=True)

        model = BModel(d_behavior=1, d_max_neurons=d_max_neurons).to(device)
        # Save architecture once per dim
        try:
            write_architecture_file(model, dirs, cfg)
        except Exception as e:
            logger.warning(f"Failed to write architecture file: {e}")

        base_lr = float(cfg["training"]["learning_rate"])
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=cfg["training"]["weight_decay"]
        )

        total_train_steps = len(train_loader) * int(cfg["training"]["num_epochs"])
        warmup_steps = int(
            float(cfg["training"].get("lr_warmup_ratio", 0.0))
            * max(1, total_train_steps)
        )

        def lr_lambda(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = (
            optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            if warmup_steps > 0
            else None
        )

        best_val = float("inf")
        # No per-dim weighting: use plain BCE without pos_weight or element weights
        train_history: list[tuple[int, float]] = []
        global_step = 0
        for epoch in range(1, cfg["training"]["num_epochs"] + 1):
            clip = cfg["training"].get("gradient_clip_norm", None)

            # Monkey-patch loss inside train_one_epoch via closure
            def train_with_weighted_bce():
                return train_one_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer,
                    epoch,
                    logger,
                    train_history,
                    global_step,
                    scheduler=scheduler,
                    clip_norm=clip,
                    target_dim=dim,
                    pos_weight=None,
                )

            _, global_step = train_with_weighted_bce()
            # Validate (BCE with logits; same pos_weight)
            val_loss, sample = validate(
                model,
                val_loader,
                device,
                epoch,
                logger,
                target_dim=dim,
                pos_weight=None,
            )
            if sample is not None:
                plot_pred_vs_true_dim(sample, dim_plots, epoch, dim)
            plot_training_loss(train_history, dim_plots, dirs["logs"])
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "config": cfg,
                        "dim": dim,
                    },
                    dim_ckpt / "best.pth",
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": cfg,
                    "dim": dim,
                },
                dim_ckpt / f"epoch_{epoch}.pth",
            )

        # After training this dim model, run subject-wise evaluation over both train and validation loaders and save CSV
        try:
            import csv

            # Train split
            rows_tr = evaluate_subjectwise(
                model, train_loader, device, target_dim=dim, loss_fn=nn.L1Loss()
            )
            csv_path_tr = dim_plots / f"subject_eval_train_dim{dim}.csv"
            with open(csv_path_tr, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["file_path", "num_samples", "loss_mean", "corr"]
                )
                w.writeheader()
                for r in rows_tr:
                    w.writerow(r)
            # Validation split
            rows_va = evaluate_subjectwise(
                model, val_loader, device, target_dim=dim, loss_fn=nn.L1Loss()
            )
            csv_path_va = dim_plots / f"subject_eval_val_dim{dim}.csv"
            with open(csv_path_va, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["file_path", "num_samples", "loss_mean", "corr"]
                )
                w.writeheader()
                for r in rows_va:
                    w.writerow(r)
        except Exception as e:
            logger.warning(f"Subject-wise evaluation failed for dim {dim}: {e}")

    logger.info("Per-dimension training complete.")


if __name__ == "__main__":
    main()
