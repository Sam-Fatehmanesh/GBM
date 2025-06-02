from GenerativeBrainModel.webapp.pipeline.baseline import modified_baseline_sequence
from GenerativeBrainModel.webapp.pipeline.inference import generate_predictions
from GenerativeBrainModel.webapp.pipeline.analysis import analyze_region_differences
from GenerativeBrainModel.webapp.config import settings


def run_simulation(
    regions: list,
    fraction: float,
    sample_idx: int = 0,
    num_steps: int = None,
    mask_loader=None
) -> dict:
    """
    Complete pipeline: apply optogenetic activation to baseline, then run GBM inference.

    Args:
        mask_loader: Optional pre-loaded mask loader to avoid reloading masks

    Returns a dict with keys:
      - baseline_dir: path to modified baseline directory
      - predictions_dir: path to generated predictions directory
      - analysis_dir: path to analysis directory
    """
    # Use configured experiments_root
    exp_path = settings.experiments_root

    # Stage 1: modified baseline
    baseline_dir = modified_baseline_sequence(
        exp_path,
        regions,
        fraction,
        sample_idx,
        mask_loader=mask_loader
    )

    # Stage 2: generate predictions
    predictions_dir = generate_predictions(
        baseline_dir,
        num_steps=num_steps
    )

    # Stage 3: run analysis on all regions
    analysis_dir = analyze_region_differences(baseline_dir, predictions_dir)

    return {
        'baseline_dir': baseline_dir,
        'predictions_dir': predictions_dir,
        'analysis_dir': analysis_dir
    } 