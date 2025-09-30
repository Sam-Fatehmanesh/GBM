from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from GenerativeBrainModel.dataloaders.neural_dataloader import create_dataloaders
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.utils.sas import sas_nll
from GenerativeBrainModel.scripts.eval_gbm import create_default_config as eval_default_config
from GenerativeBrainModel.scripts.eval_gbm import deep_update as eval_deep_update


def load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> GBM:
    train_loader, _, _, _ = create_dataloaders(cfg)
    sample_batch = next(iter(train_loader))
    d_stimuli = int(sample_batch['stimulus'].shape[-1])
    model_cfg = cfg['model']
    model = GBM(
        d_model=model_cfg['d_model'],
        d_stimuli=d_stimuli,
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
    ).to(device)
    return model


def prepare_eval_cfg(eval_cfg_path: Path) -> dict:
    user_cfg = load_yaml(eval_cfg_path)
    cfg = eval_default_config()
    cfg = eval_deep_update(cfg, user_cfg)

    cfg['training'] = cfg.get('training', {})
    cfg['training']['batch_size'] = cfg['eval']['batch_size']
    cfg['training']['sequence_length'] = cfg['eval']['sequence_length']
    cfg['training']['stride'] = cfg['eval']['stride']
    cfg['training']['num_workers'] = cfg['eval']['num_workers']
    cfg['training']['only_test'] = False
    max_tp = cfg['eval'].get('max_timepoints_per_subject')
    if max_tp is not None:
        cfg['training']['max_timepoints_per_subject'] = max_tp
    test_subjects = cfg['data'].get('test_subjects')
    if isinstance(test_subjects, list) and len(test_subjects) > 0:
        cfg['training']['only_test'] = True
    return cfg


def load_state(model: GBM, ckpt_path: Path, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd_in = state['model'] if isinstance(state, dict) and 'model' in state else state

    def normalize_key(k: str) -> str:
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
        sd = {normalize_key(k): v for k, v in sd_in.items()}
    else:
        sd = sd_in

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)

    model.eval()
    if device.type == 'cuda':
        model.to(dtype=torch.bfloat16)


def compute_mean_loss(model: GBM, loader: torch.utils.data.DataLoader, device: torch.device, limit: int) -> float:
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, 1):
            spikes = batch['spikes'].to(device).to(torch.bfloat16 if device.type == 'cuda' else torch.float32)
            positions = batch['positions'].to(device)
            mask = batch['neuron_mask'].to(device)
            stim = batch['stimulus'].to(device).to(torch.bfloat16 if device.type == 'cuda' else torch.float32)

            x_in = spikes[:, :-1, :]
            x_tgt = spikes[:, 1:, :].float()
            stim_in = stim[:, :-1, :]

            mu, log_sigma, eta, log_delta = model(x_in, stim_in, positions, mask, get_logits=True)
            loss = sas_nll(
                x_tgt.float(),
                mu.float(),
                log_sigma.float(),
                eta.float(),
                log_delta.float(),
            )
            losses.append(float(loss.detach().cpu().item()))
            if limit and idx >= limit:
                break
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_cfg_path = Path('/home/user/gbm3/GBM3/experiments/gbm_neural/gbm_neural_training_20250925_224005/config.yaml')
    eval_cfg_path = Path('/home/user/gbm3/GBM3/experiments/gbm_neural_eval/gbm_neural_evaluation_20250926_103035/config.yaml')
    ckpt_path = Path('/home/user/gbm3/GBM3/experiments/gbm_neural/gbm_neural_training_20250925_224005/checkpoints/best_step_1_76818.pth')

    train_cfg = load_yaml(train_cfg_path)
    model = build_model(train_cfg, device)
    load_state(model, ckpt_path, device)

    # Training-style validation loader (batch size 1)
    _, train_val_loader, _, _ = create_dataloaders(train_cfg)
    train_mean = compute_mean_loss(model, train_val_loader, device, limit=16)

    # Evaluation-style loader (batch size 2)
    eval_cfg = prepare_eval_cfg(eval_cfg_path)
    _, eval_loader, _, _ = create_dataloaders(eval_cfg)
    eval_mean = compute_mean_loss(model, eval_loader, device, limit=16)

    print('device', device)
    print('training_loader_batch_size', train_cfg['training']['batch_size'])
    print('eval_loader_batch_size', eval_cfg['training']['batch_size'])
    print('training_style_mean_loss', train_mean)
    print('eval_style_mean_loss', eval_mean)
    if math.isfinite(train_mean) and math.isfinite(eval_mean) and train_mean != 0:
        print('ratio_eval_over_train', eval_mean / train_mean)


if __name__ == '__main__':
    main()
