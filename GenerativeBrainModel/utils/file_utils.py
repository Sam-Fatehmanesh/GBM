"""File and directory utilities."""

import os
import pandas as pd


def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)


def create_experiment_dir(base_path, phase_name):
    """Create a phase experiment directory with all necessary subdirectories
    
    Args:
        base_path: Base path for the experiment (e.g., 'experiments/gbm/timestamp')
        phase_name: Name of the phase (e.g., 'pretrain', 'finetune')
        
    Returns:
        phase_dir: Path to the created phase directory
    """
    # Create phase directory
    phase_dir = os.path.join(base_path, phase_name)
    os.makedirs(phase_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    os.makedirs(os.path.join(phase_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(phase_dir, 'logs'), exist_ok=True)
    
    return phase_dir


def ensure_directory_exists(directory_path):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: The directory path
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_checkpoint_path(experiment_dir, phase_name, checkpoint_type='best'):
    """Get the path to a checkpoint file.
    
    Args:
        experiment_dir: Root experiment directory
        phase_name: Name of the phase (e.g., 'pretrain', 'finetune')
        checkpoint_type: Type of checkpoint ('best' or 'final')
        
    Returns:
        str: Path to the checkpoint file
    """
    checkpoint_filename = f'{checkpoint_type}_model.pt'
    return os.path.join(experiment_dir, phase_name, 'checkpoints', checkpoint_filename)


def save_experiment_metadata(experiment_dir, metadata):
    """Save experiment metadata to a text file.
    
    Args:
        experiment_dir: Root experiment directory
        metadata: Dictionary containing experiment metadata
    """
    metadata_path = os.path.join(experiment_dir, "experiment_info.txt")
    
    with open(metadata_path, "w") as f:
        f.write(f"GBM Training Experiment\n")
        f.write(f"=" * 50 + "\n\n")
        
        for key, value in metadata.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
                f.write("\n")
            else:
                f.write(f"{key}: {value}\n")
                
        f.write("\n" + "=" * 50 + "\n") 