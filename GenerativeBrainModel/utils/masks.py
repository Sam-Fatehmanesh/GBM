import os
import glob
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
from typing import Dict, Tuple, Optional, Union


class ZebrafishMaskLoader:
    """
    A class to load and manage 3D zebrafish brain masks from TIF files.
    
    This class loads all TIF mask files from a specified directory, downsamples them
    to a target resolution, and stores them as PyTorch tensors on a specified device.
    """
    
    def __init__(
        self, 
        masks_dir: str = "masks",
        target_shape: Tuple[int, int, int] = (30, 256, 128),
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bool
    ):
        """
        Initialize the zebrafish mask loader.
        
        Args:
            masks_dir: Directory containing the TIF mask files
            target_shape: Target shape for downsampling (Z, Y, X). Default: (30, 256, 128)
            device: PyTorch device to load tensors onto. If None, uses CUDA if available
            dtype: Data type for the tensors. Default: torch.bool
        """
        self.masks_dir = masks_dir
        self.target_shape = target_shape
        self.dtype = dtype
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Dictionary to store loaded masks
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Original shape (will be detected from first loaded mask)
        self.original_shape: Optional[Tuple[int, int, int]] = None
        
        # Load all masks
        self._load_all_masks()
    
    def _load_all_masks(self) -> None:
        """Load all TIF mask files from the masks directory."""
        # Find all TIF files in the masks directory
        mask_files = glob.glob(os.path.join(self.masks_dir, "*.tif"))
        
        if not mask_files:
            raise ValueError(f"No TIF files found in directory: {self.masks_dir}")
        
        print(f"Loading {len(mask_files)} mask files from {self.masks_dir}")
        print(f"Target shape: {self.target_shape}")
        print(f"Device: {self.device}")
        
        for mask_file in sorted(mask_files):
            self._load_single_mask(mask_file)
        
        print(f"Successfully loaded {len(self.masks)} masks")
        
        # Print memory usage if on GPU
        if self.device.type == "cuda":
            memory_mb = sum(mask.element_size() * mask.nelement() for mask in self.masks.values()) / 1024**2
            print(f"Total GPU memory used by masks: {memory_mb:.2f} MB")
    
    def _load_single_mask(self, mask_file: str) -> None:
        """
        Load a single TIF mask file, downsample it, and store as tensor.
        
        Args:
            mask_file: Path to the TIF mask file
        """
        # Extract filename without extension for dictionary key
        filename = os.path.basename(mask_file)
        mask_name = os.path.splitext(filename)[0]
        
        try:
            # Load the mask using tifffile
            mask_data = tifffile.imread(mask_file)
            
            # Store original shape from first mask
            if self.original_shape is None:
                self.original_shape = mask_data.shape
                print(f"Original mask shape: {self.original_shape}")
            
            # Verify shape consistency
            if mask_data.shape != self.original_shape:
                print(f"Warning: {mask_name} has different shape {mask_data.shape} vs expected {self.original_shape}")
            
            # Convert to torch tensor
            mask_tensor = torch.from_numpy(mask_data.astype(np.float32))
            
            # Downsample if needed
            if mask_tensor.shape != self.target_shape:
                mask_tensor = self._downsample_mask(mask_tensor)
            
            # Convert to target dtype and move to device
            mask_tensor = mask_tensor.to(dtype=self.dtype, device=self.device)
            
            # Store in dictionary
            self.masks[mask_name] = mask_tensor
            
            # Print progress for large masks
            if mask_tensor.sum() > 0:  # Only print for non-empty masks
                fill_percentage = (mask_tensor.sum().item() / mask_tensor.numel()) * 100
                print(f"  {mask_name}: {fill_percentage:.2f}% filled")
        
        except Exception as e:
            print(f"Error loading {mask_file}: {str(e)}")
    
    def _downsample_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample a mask tensor to the target shape.
        
        Args:
            mask_tensor: Input mask tensor of shape (Z, Y, X)
            
        Returns:
            Downsampled mask tensor of target shape
        """
        # Add batch dimension for interpolation: (1, 1, Z, Y, X)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        
        # Use trilinear interpolation for 3D downsampling
        # Note: using 'nearest' mode to preserve binary nature for boolean masks
        downsampled = F.interpolate(
            mask_tensor, 
            size=self.target_shape,
            mode='nearest'  # Preserves binary nature of masks
        )
        
        # Remove batch dimensions: (Z, Y, X)
        return downsampled.squeeze(0).squeeze(0)
    
    def get_mask(self, mask_name: str) -> torch.Tensor:
        """
        Get a specific mask by name.
        
        Args:
            mask_name: Name of the mask (filename without .tif extension)
            
        Returns:
            The mask tensor
            
        Raises:
            KeyError: If mask name not found
        """
        if mask_name not in self.masks:
            available_masks = list(self.masks.keys())
            raise KeyError(f"Mask '{mask_name}' not found. Available masks: {available_masks[:5]}...")
        
        return self.masks[mask_name]
    
    def get_all_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get all loaded masks.
        
        Returns:
            Dictionary of all mask tensors
        """
        return self.masks
    
    def list_masks(self) -> list:
        """
        Get list of all available mask names.
        
        Returns:
            List of mask names
        """
        return list(self.masks.keys())
    
    def get_combined_mask(self, mask_names: list) -> torch.Tensor:
        """
        Combine multiple masks into a single mask using logical OR.
        
        Args:
            mask_names: List of mask names to combine
            
        Returns:
            Combined mask tensor
        """
        if not mask_names:
            raise ValueError("No mask names provided")
        
        # Start with first mask
        combined = self.get_mask(mask_names[0]).clone()
        
        # OR with remaining masks
        for mask_name in mask_names[1:]:
            combined = combined | self.get_mask(mask_name)
        
        return combined
    
    def get_mask_stats(self, mask_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific mask.
        
        Args:
            mask_name: Name of the mask
            
        Returns:
            Dictionary with mask statistics
        """
        mask = self.get_mask(mask_name)
        total_voxels = mask.numel()
        active_voxels = mask.sum().item()
        
        return {
            'total_voxels': total_voxels,
            'active_voxels': active_voxels,
            'fill_percentage': (active_voxels / total_voxels) * 100,
            'shape': tuple(mask.shape)
        }
    
    def print_summary(self) -> None:
        """Print a summary of all loaded masks."""
        print(f"\nZebrafish Mask Loader Summary:")
        print(f"Original shape: {self.original_shape}")
        print(f"Target shape: {self.target_shape}")
        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")
        print(f"Number of masks: {len(self.masks)}")
        
        # Show top 10 masks by fill percentage
        mask_stats = []
        for name in self.masks.keys():
            stats = self.get_mask_stats(name)
            mask_stats.append((name, stats['fill_percentage']))
        
        mask_stats.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 masks by fill percentage:")
        for i, (name, fill_pct) in enumerate(mask_stats[:10]):
            print(f"  {i+1:2d}. {name:<50} {fill_pct:5.2f}%")
    
    def save_masks(self, output_dir: str) -> None:
        """
        Save all loaded masks to a directory as PyTorch tensors.
        
        Args:
            output_dir: Directory to save masks to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for mask_name, mask_tensor in self.masks.items():
            output_path = os.path.join(output_dir, f"{mask_name}.pt")
            torch.save(mask_tensor.cpu(), output_path)
        
        print(f"Saved {len(self.masks)} masks to {output_dir}")
    
    def __len__(self) -> int:
        """Return the number of loaded masks."""
        return len(self.masks)
    
    def __getitem__(self, mask_name: str) -> torch.Tensor:
        """Allow dictionary-style access to masks."""
        return self.get_mask(mask_name)
    
    def __contains__(self, mask_name: str) -> bool:
        """Check if a mask name exists."""
        return mask_name in self.masks


# Convenience function to create a mask loader
_mask_loader_cache = {}
def load_zebrafish_masks(
    masks_dir: str = "masks",
    target_shape: Tuple[int, int, int] = (30, 256, 128),
    device: Optional[Union[str, torch.device]] = None
) -> ZebrafishMaskLoader:
    """
    Convenience function to create and return a ZebrafishMaskLoader.
    
    Args:
        masks_dir: Directory containing the TIF mask files
        target_shape: Target shape for downsampling (Z, Y, X)
        device: PyTorch device to load tensors onto
        
    Returns:
        ZebrafishMaskLoader instance
    """
    # Cache mask loaders by parameters to avoid reloading
    key = (masks_dir, tuple(target_shape), str(device))
    if key not in _mask_loader_cache:
        _mask_loader_cache[key] = ZebrafishMaskLoader(masks_dir, target_shape, device)
    return _mask_loader_cache[key]


if __name__ == "__main__":
    # Example usage
    print("Loading zebrafish masks...")
    
    # Create mask loader
    mask_loader = load_zebrafish_masks()
    
    # Print summary
    mask_loader.print_summary()
    
    # Example: Get a specific mask
    if "cerebellum" in mask_loader:
        cerebellum_mask = mask_loader["cerebellum"]
        print(f"\nCerebellum mask shape: {cerebellum_mask.shape}")
        print(f"Cerebellum mask device: {cerebellum_mask.device}")
    
    # Example: Combine multiple masks
    brain_regions = ["prosencephalon_(forebrain)", "midbrain", "cerebellum"]
    available_regions = [region for region in brain_regions if region in mask_loader]
    
    if available_regions:
        combined_mask = mask_loader.get_combined_mask(available_regions)
        print(f"\nCombined mask shape: {combined_mask.shape}")
        stats = mask_loader.get_mask_stats(available_regions[0])
        print(f"Example stats: {stats}")