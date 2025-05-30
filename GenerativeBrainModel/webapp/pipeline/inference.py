import os
import glob
import torch
import numpy as np
from GenerativeBrainModel.models.gbm import GBM


def generate_predictions(
    baseline_dir: str,
    num_steps: int = None,
    output_dir: str = None
) -> str:
    """
    Load 'baseline_sequence.npy' from baseline_dir, run GBM autoregressive generation,
    and save 'predicted_sequence.npy' and 'predicted_probabilities.npy' under output_dir.

    Returns path to output_dir.
    """
    # Load baseline sequence
    baseline_path = os.path.join(baseline_dir, 'baseline_sequence.npy')
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline sequence not found: {baseline_path}")
    seq = np.load(baseline_path)  # shape: (seq_len, H, W)
    seq_len, H, W = seq.shape

    # Default num_steps = seq_len (predict same number of frames)
    if num_steps is None:
        num_steps = seq_len

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Locate model checkpoint under experiment path (parent of baseline_dir)
    exp_path = os.path.dirname(baseline_dir)
    pt_files = glob.glob(os.path.join(exp_path, '**', '*.pt'), recursive=True)
    if not pt_files:
        raise FileNotFoundError(f'No .pt checkpoint found under {exp_path}')
    # Prefer final_model.pt
    final_ckpt = [p for p in pt_files if os.path.basename(p) == 'final_model.pt']
    if final_ckpt:
        ckpt_path = final_ckpt[0]
    else:
        # Fallback: first checkpoint listing
        ckpt_path = pt_files[0]

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # Instantiate model with parameters from checkpoint
    params = checkpoint.get('params', {})
    mamba_layers = params.get('mamba_layers', 1)
    mamba_dim = params.get('mamba_dim', 1024)
    mamba_state_multiplier = params.get('mamba_state_multiplier', 1)
    model = GBM(
        mamba_layers=mamba_layers,
        mamba_dim=mamba_dim,
        mamba_state_multiplier=mamba_state_multiplier,
        pretrained_ae_path=None
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Prepare input tensor (ensure float dtype)
    x = torch.from_numpy(seq).unsqueeze(0).float().to(device)  # (1, seq_len, H, W)

    # Run autoregressive generation
    with torch.no_grad():
        out_x, probs = model.generate_autoregressive_brain(x, num_steps=num_steps)

    # Convert to numpy
    out_np = out_x.squeeze(0).cpu().numpy()               # shape: (seq_len+num_steps, H, W)
    probs_np = probs.squeeze(0).cpu().numpy()              # shape: (num_steps, H, W)

    # Save results
    if output_dir is None:
        output_dir = os.path.join(baseline_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'predicted_sequence.npy'), out_np)
    np.save(os.path.join(output_dir, 'predicted_probabilities.npy'), probs_np)

    return output_dir 