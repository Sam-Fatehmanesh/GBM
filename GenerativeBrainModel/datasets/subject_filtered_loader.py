"""Subject-filtered data loading utilities."""

import os
import tempfile
from typing import List, Optional
from tqdm import tqdm

from .fast_dali_spike_dataset import FastDALIBrainDataLoader


class SubjectFilteredFastDALIBrainDataLoader(FastDALIBrainDataLoader):
    """FastDALIBrainDataLoader that can filter subjects based on include/exclude lists."""
    
    def __init__(
        self,
        preaugmented_dir: str,
        include_subjects: Optional[List[str]] = None,
        exclude_subjects: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the SubjectFilteredFastDALIBrainDataLoader.
        
        Args:
            preaugmented_dir: Directory containing preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            **kwargs: Additional arguments to pass to FastDALIBrainDataLoader
        """
        self.original_dir = preaugmented_dir
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects if exclude_subjects else []
        
        # Create a temporary directory to store the filtered subject directories
        self.temp_dir = None
        filtered_dir = self._create_filtered_subject_dir(preaugmented_dir, include_subjects, exclude_subjects)
        
        # Initialize the parent class with the filtered directory
        super().__init__(filtered_dir, **kwargs)
        
        # Print subjects being used
        if include_subjects:
            tqdm.write(f"Using only subjects: {include_subjects}")
        if exclude_subjects:
            tqdm.write(f"Excluding subjects: {exclude_subjects}")
    
    def _create_filtered_subject_dir(self, preaugmented_dir, include_subjects, exclude_subjects):
        """Create a temporary directory with symlinks to only the desired subject directories.
        
        Args:
            preaugmented_dir: Original directory containing all preaugmented data
            include_subjects: List of subject names to include. If None, all subjects are included.
            exclude_subjects: List of subject names to exclude. If None, no subjects are excluded.
            
        Returns:
            filtered_dir: Path to a temporary directory containing only the filtered subjects
        """
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="filtered_subjects_")
        
        # Get all subjects in the preaugmented directory
        all_subjects = []
        for subject_dir in os.listdir(preaugmented_dir):
            subject_path = os.path.join(preaugmented_dir, subject_dir)
            if os.path.isdir(subject_path):
                # Check if this is a valid subject directory (contains preaugmented_grids.h5)
                grid_file = os.path.join(subject_path, 'preaugmented_grids.h5')
                if os.path.exists(grid_file):
                    all_subjects.append(subject_dir)
        
        # Filter subjects based on include/exclude lists
        if include_subjects is not None:
            filtered_subjects = [s for s in all_subjects if s in include_subjects]
        elif exclude_subjects is not None:
            filtered_subjects = [s for s in all_subjects if s not in exclude_subjects]
        else:
            # If both include_subjects and exclude_subjects are None, include all subjects
            filtered_subjects = all_subjects
        
        # If no subjects remain after filtering, raise an error
        if not filtered_subjects:
            raise ValueError("No subjects left after filtering!")
        
        # Create symlinks to the filtered subject directories
        for subject in filtered_subjects:
            # Use absolute path for the source to ensure symlinks work correctly
            src_path = os.path.abspath(os.path.join(preaugmented_dir, subject))
            dst_path = os.path.join(self.temp_dir, subject)
            os.symlink(src_path, dst_path)
        
        tqdm.write(f"Created filtered directory with {len(filtered_subjects)} subjects: {filtered_subjects}")
        
        return self.temp_dir
    
    def __del__(self):
        """Clean up temporary directory when the object is deleted."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                # Delete symlinks but not the actual data
                for item in os.listdir(self.temp_dir):
                    os.unlink(os.path.join(self.temp_dir, item))
                # Remove the temporary directory
                os.rmdir(self.temp_dir)
            except Exception as e:
                # Print but don't fail if cleanup encounters an error
                print(f"Warning: Error during cleanup of temporary directory: {e}") 