"""
Lightweight metrics and plotting utilities for neuron-based GBM training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

import torch
import matplotlib.pyplot as plt


class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        return self.value

    def reset(self) -> None:
        self.value = None


class CSVLogger:
    def __init__(self, csv_path: Union[str, Path], fieldnames: List[str]):
        self.csv_path = Path(csv_path)
        self.fieldnames = fieldnames
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Union[float, int, str]]) -> None:
        filtered = {k: v for k, v in row.items() if k in self.fieldnames}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered)


def pr_auc_binned(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    num_bins: int = 1000,
) -> float:
    """
    Compute PR AUC with a memory-efficient binned approach on GPU.
    predictions, targets: flattened tensors on same device in [0,1].
    """
    device = predictions.device
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    binary_targets = (targets >= threshold).float()

    # Handle no positives edge case
    total_positives = binary_targets.sum()
    if total_positives <= 0:
        return 0.0

    bin_indices = torch.searchsorted(bin_edges[1:], predictions, right=False)
    tp_counts = torch.zeros(num_bins, device=device)
    total_counts = torch.zeros(num_bins, device=device)
    tp_counts.scatter_add_(0, bin_indices, binary_targets)
    total_counts.scatter_add_(0, bin_indices, torch.ones_like(binary_targets))

    tp_flip = torch.flip(tp_counts, [0])
    tot_flip = torch.flip(total_counts, [0])
    cum_tp = torch.cumsum(tp_flip, dim=0)
    cum_fp = torch.cumsum(tot_flip - tp_flip, dim=0)
    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / total_positives

    precision = torch.cat([torch.tensor([1.0], device=device), precision])
    recall = torch.cat([torch.tensor([0.0], device=device), recall])
    recall_diff = recall[1:] - recall[:-1]
    auc = torch.sum(recall_diff * precision[:-1]).item()
    return auc


class CombinedMetricsTracker:
    def __init__(
        self,
        log_dir: Union[str, Path],
        ema_alpha: float = 0.1,
        val_threshold: float = 0.5,
        enable_plots: bool = True,
    ):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.train_csv = CSVLogger(
            log_dir / "training.csv", ["epoch", "batch_idx", "loss", "ema_loss", "lr"]
        )
        self.val_csv = CSVLogger(
            log_dir / "validation.csv", ["epoch", "batch_idx", "val_loss", "pr_auc"]
        )
        self.loss_ema = ExponentialMovingAverage(alpha=ema_alpha)
        self.enable_plots = enable_plots
        self.plots_dir = log_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.val_threshold = val_threshold

    def log_train(self, epoch: int, batch_idx: int, loss: float, lr: float) -> None:
        ema = self.loss_ema.update(loss)
        self.train_csv.log(
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "loss": loss,
                "ema_loss": ema,
                "lr": lr,
            }
        )

    def log_validation(
        self,
        epoch: int,
        batch_idx: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        val_loss: float,
        compute_auc: bool = True,
    ) -> Dict[str, float]:
        # predictions/targets may come as batched tensors or already flattened
        if compute_auc:
            preds_flat = predictions.detach().reshape(-1)
            targs_flat = targets.detach().reshape(-1)
            auc = pr_auc_binned(preds_flat, targs_flat, threshold=self.val_threshold)
        else:
            auc = float("nan")
        self.val_csv.log(
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "val_loss": val_loss,
                "pr_auc": auc,
            }
        )
        return {"val_loss": val_loss, "pr_auc": auc}

    def plot_training(self) -> None:
        if not self.enable_plots:
            return
        # Minimal plot from CSV files
        import pandas as pd

        train_path = self.train_csv.csv_path
        val_path = self.val_csv.csv_path
        if not train_path.exists() and not val_path.exists():
            return
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        if train_path.exists():
            df = pd.read_csv(train_path)
            if not df.empty:
                x = (df["epoch"] - 1) * max(
                    1, df[df["epoch"] == 1]["batch_idx"].max() or 1
                ) + df["batch_idx"]
                axes[0].plot(x, df["loss"], alpha=0.3, label="Loss")
                if "ema_loss" in df:
                    axes[0].plot(x, df["ema_loss"], label="EMA Loss")
                axes[0].set_title("Training Loss")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
        if val_path.exists():
            dfv = pd.read_csv(val_path)
            if not dfv.empty:
                x = (dfv["epoch"] - 1) * max(
                    1, dfv[dfv["epoch"] == 1]["batch_idx"].max() or 1
                ) + dfv["batch_idx"]
                axes[1].plot(x, dfv["val_loss"], label="Val Loss")
                if "pr_auc" in dfv:
                    ax2 = axes[1].twinx()
                    ax2.plot(x, dfv["pr_auc"], color="crimson", label="PR AUC")
                    ax2.set_ylim(0, 1)
                axes[1].set_title("Validation Metrics")
                axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.plots_dir / "training_plots.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
