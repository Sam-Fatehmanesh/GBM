"""Data processing utilities."""

import os
import h5py


def get_max_z_planes(preaugmented_dir, split='train'):
    """Get the maximum number of z-planes from the preaugmented data.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        split: 'train' or 'test'
    
    Returns:
        max_z_planes: Maximum number of z-planes found across all subjects
    """
    max_z_planes = 0
    
    for subject_dir in os.listdir(preaugmented_dir):
        subject_path = os.path.join(preaugmented_dir, subject_dir)
        if os.path.isdir(subject_path):
            metadata_path = os.path.join(subject_path, 'metadata.h5')
            if os.path.exists(metadata_path):
                with h5py.File(metadata_path, 'r') as f:
                    if 'num_z_planes' in f:
                        max_z_planes = max(max_z_planes, f['num_z_planes'][()])
    
    if max_z_planes == 0:
        raise ValueError("No z-planes found in any of the preaugmented data files!")
    
    return max_z_planes


def validate_subject_directory(preaugmented_dir, subject_name):
    """Validate that a subject directory exists and contains required files.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        subject_name: Name of the subject to validate
        
    Returns:
        bool: True if valid, raises ValueError if not
        
    Raises:
        ValueError: If subject directory or required files are missing
    """
    subject_path = os.path.join(preaugmented_dir, subject_name)
    
    if not os.path.exists(subject_path) or not os.path.isdir(subject_path):
        raise ValueError(f"Subject '{subject_name}' not found in {preaugmented_dir}")
    
    grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
    if not os.path.exists(grid_file):
        raise ValueError(f"Subject '{subject_name}' does not have preaugmented_grids.h5 file")
    
    return True


def get_subject_list(preaugmented_dir):
    """Get list of all valid subjects in the preaugmented directory.
    
    Args:
        preaugmented_dir: Directory containing preaugmented data
        
    Returns:
        list: List of valid subject names
    """
    subjects = []
    
    for subject_dir in os.listdir(preaugmented_dir):
        subject_path = os.path.join(preaugmented_dir, subject_dir)
        if os.path.isdir(subject_path):
            # Check if this is a valid subject directory
            grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
            if os.path.exists(grid_file):
                subjects.append(subject_dir)
    
    return sorted(subjects) 