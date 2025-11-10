from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import torch
import yaml

from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.utils.sas import sas_nll


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any], device: torch.device) -> GBM:
    train_loader, _, _, _ = create_dataloaders(cfg)
    sample_batch = next(iter(train_loader))
    d_stimuli = int(sample_batch["stimulus"].shape[-1])
    model_cfg = cfg["model"]
    model = GBM(
        d_model=model_cfg["d_model"],
        d_stimuli=d_stimuli,
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
    ).to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    return model


def normalize_state_dict(sd_in: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def normalize_key(k: str) -> str:
        changed = True
        while changed:
            changed = False
            if k.startswith("module."):
                k = k[len("module.") :]
                changed = True
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod.") :]
                changed = True
        return k

    return {normalize_key(k): v for k, v in sd_in.items()}


def load_state(model: GBM, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd_in = state["model"] if isinstance(state, dict) and "model" in state else state
    if isinstance(sd_in, dict):
        sd = normalize_state_dict(sd_in)
    else:
        sd = sd_in
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model.eval()


def collect_batches(
    loader: torch.utils.data.DataLoader, limit: int
) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(loader, 1):
        stored = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in batch.items()
            if k in {"spikes", "positions", "neuron_mask", "stimulus"}
        }
        batches.append(stored)
        if idx >= limit:
            break
    return batches


def compute_mean_loss(
    model: GBM, batches: List[Dict[str, torch.Tensor]], device: torch.device
) -> float:
    losses = []
    with torch.no_grad():
        for batch in batches:
            spikes = batch["spikes"].to(device)
            positions = batch["positions"].to(device)
            mask = batch["neuron_mask"].to(device)
            stim = batch["stimulus"].to(device)

            if device.type == "cuda":
                spikes = spikes.to(torch.bfloat16)
                stim = stim.to(torch.bfloat16)

            x_in = spikes[:, :-1, :]
            x_tgt = spikes[:, 1:, :].float()
            stim_in = stim[:, :-1, :]

            mu, log_sigma, eta, log_delta = model(
                x_in, stim_in, positions, mask, get_logits=True
            )
            loss = sas_nll(
                x_tgt,
                mu.float(),
                log_sigma.float(),
                eta.float(),
                log_delta.float(),
            )
            losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg_path = Path(
        "/home/user/gbm3/GBM3/experiments/gbm_neural/gbm_neural_training_20250925_224005/config.yaml"
    )
    ckpt_path = Path(
        "/home/user/gbm3/GBM3/experiments/gbm_neural/gbm_neural_training_20250925_224005/checkpoints/best_step_1_76818.pth"
    )

    train_cfg = load_yaml(train_cfg_path)
    train_cfg["training"]["num_workers"] = 0
    train_cfg["training"]["persistent_workers"] = False

    _, val_loader, _, _ = create_dataloaders(train_cfg)
    batches = collect_batches(val_loader, limit=4)
    print(
        f"Collected {len(batches)} batches with batch_size={batches[0]['spikes'].shape[0]}"
    )

    # Eager model
    model_eager = build_model(train_cfg, device)
    load_state(model_eager, ckpt_path, device)
    eager_loss = compute_mean_loss(model_eager, batches, device)
    print("eager_mean_loss", eager_loss)

    # Compiled model (fresh instance)
    model_compiled = build_model(train_cfg, device)
    load_state(model_compiled, ckpt_path, device)
    compiled_model = torch.compile(model_compiled, dynamic=True)
    compiled_loss = compute_mean_loss(compiled_model, batches, device)
    print("compiled_mean_loss", compiled_loss)
    if eager_loss != 0:
        print("compiled_over_eager_ratio", compiled_loss / eager_loss)


if __name__ == "__main__":
    main()
