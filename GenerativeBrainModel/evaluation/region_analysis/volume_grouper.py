"""
VolumeGrouper: Group frame sequences into brain volumes.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeGrouper:
    """
    Group frame sequences into brain volumes based on z-plane progression.
    
    Handles:
    - Volume detection: Group frames from z=0 to next z=0
    - Completeness validation: Discard incomplete volumes
    - Volume-level chunking: Ensure proper brain volume boundaries
    """
    
    def __init__(self, 
                 expected_volume_size: Optional[int] = None,
                 tolerance: int = 2):
        """
        Initialize VolumeGrouper.
        
        Args:
            expected_volume_size: Expected number of frames per brain volume
            tolerance: Tolerance for volume size variation (frames)
        """
        self.expected_volume_size = expected_volume_size
        self.tolerance = tolerance
    
    def detect_volume_boundaries(self, 
                               z_one_frames: List[int],
                               total_frames: int) -> List[Tuple[int, int]]:
        """
        Detect volume boundaries from z=1 frame indices.
        
        Args:
            z_one_frames: List of frame indices with z=1 markers (z-index 0)
            total_frames: Total number of available frames
            
        Returns:
            List of (start_idx, end_idx) tuples for complete volumes
        """
        if len(z_one_frames) < 2:
            logger.warning("Need at least 2 z=1 frames to detect volumes")
            return []
        
        volumes = []
        
        for i in range(len(z_one_frames) - 1):
            start_idx = z_one_frames[i]
            end_idx = z_one_frames[i + 1]
            
            # Validate volume size
            volume_size = end_idx - start_idx
            
            if self.expected_volume_size is not None:
                if abs(volume_size - self.expected_volume_size) > self.tolerance:
                    logger.debug(f"Volume {i} size {volume_size} outside expected range "
                               f"{self.expected_volume_size}±{self.tolerance}")
                    continue
            
            volumes.append((start_idx, end_idx))
        
        # Handle the last volume if there are enough frames
        if len(z_one_frames) >= 1:
            last_start = z_one_frames[-1]
            
            if self.expected_volume_size is not None:
                # Check if we have enough frames for a complete last volume
                expected_end = last_start + self.expected_volume_size
                if expected_end <= total_frames:
                    volumes.append((last_start, expected_end))
                else:
                    logger.debug(f"Last volume incomplete: need {expected_end} frames, have {total_frames}")
        
        logger.info(f"Detected {len(volumes)} complete brain volumes")
        return volumes
    
    def group_chunks_into_volumes(self, 
                                chunks: List[Tuple[int, int]],
                                volume_boundaries: List[Tuple[int, int]]) -> List[Dict]:
        """
        Group frame chunks into brain volumes.
        
        Args:
            chunks: List of (start_idx, end_idx) chunk boundaries
            volume_boundaries: List of (start_idx, end_idx) volume boundaries
            
        Returns:
            List of volume dictionaries with chunk information
        """
        volume_chunks = []
        
        for vol_idx, (vol_start, vol_end) in enumerate(volume_boundaries):
            volume_info = {
                'volume_idx': vol_idx,
                'volume_start': vol_start,
                'volume_end': vol_end,
                'volume_size': vol_end - vol_start,
                'chunks': [],
                'total_chunk_frames': 0,
                'complete': True
            }
            
            # Find chunks that overlap with this volume
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
                # Check if chunk overlaps with volume
                if chunk_start < vol_end and chunk_end > vol_start:
                    # Calculate overlap
                    overlap_start = max(chunk_start, vol_start)
                    overlap_end = min(chunk_end, vol_end)
                    overlap_size = overlap_end - overlap_start
                    
                    if overlap_size > 0:
                        chunk_info = {
                            'chunk_idx': chunk_idx,
                            'chunk_start': chunk_start,
                            'chunk_end': chunk_end,
                            'chunk_size': chunk_end - chunk_start,
                            'overlap_start': overlap_start,
                            'overlap_end': overlap_end,
                            'overlap_size': overlap_size
                        }
                        volume_info['chunks'].append(chunk_info)
                        volume_info['total_chunk_frames'] += overlap_size
            
            # Check if volume is complete (has sufficient chunk coverage)
            coverage_ratio = volume_info['total_chunk_frames'] / volume_info['volume_size']
            
            if coverage_ratio < 0.8:  # Need at least 80% coverage
                volume_info['complete'] = False
                logger.debug(f"Volume {vol_idx} incomplete: {coverage_ratio:.1%} coverage")
            
            volume_chunks.append(volume_info)
        
        # Filter to only complete volumes
        complete_volumes = [v for v in volume_chunks if v['complete']]
        
        logger.info(f"Grouped chunks into {len(complete_volumes)} complete volumes "
                   f"(out of {len(volume_chunks)} total)")
        
        return complete_volumes
    
    def extract_volume_data(self, 
                          frames: torch.Tensor,
                          volume_info: Dict) -> torch.Tensor:
        """
        Extract frame data for a specific volume.
        
        Args:
            frames: Full frame tensor
            volume_info: Volume information dictionary
            
        Returns:
            Volume frames tensor
        """
        vol_start = volume_info['volume_start']
        vol_end = volume_info['volume_end']
        
        return frames[vol_start:vol_end]
    
    def extract_volume_chunks(self, 
                            frames: torch.Tensor,
                            volume_info: Dict) -> List[torch.Tensor]:
        """
        Extract chunk data for a specific volume.
        
        Args:
            frames: Full frame tensor
            volume_info: Volume information dictionary
            
        Returns:
            List of chunk tensors that belong to this volume
        """
        chunk_tensors = []
        
        for chunk_info in volume_info['chunks']:
            chunk_start = chunk_info['chunk_start']
            chunk_end = chunk_info['chunk_end']
            
            chunk_tensor = frames[chunk_start:chunk_end]
            chunk_tensors.append(chunk_tensor)
        
        return chunk_tensors
    
    def validate_volume_consistency(self, 
                                  volume_chunks: List[Dict]) -> bool:
        """
        Validate that volumes have consistent properties.
        
        Args:
            volume_chunks: List of volume information dictionaries
            
        Returns:
            True if volumes are consistent
        """
        if not volume_chunks:
            return True
        
        # Check volume sizes
        volume_sizes = [v['volume_size'] for v in volume_chunks]
        
        if self.expected_volume_size is not None:
            # All volumes should be close to expected size
            for size in volume_sizes:
                if abs(size - self.expected_volume_size) > self.tolerance:
                    logger.warning(f"Volume size {size} outside expected range "
                                 f"{self.expected_volume_size}±{self.tolerance}")
                    return False
        else:
            # Infer expected size from first volume
            self.expected_volume_size = volume_sizes[0]
            logger.info(f"Inferred expected volume size: {self.expected_volume_size}")
            
            # Check consistency
            for size in volume_sizes[1:]:
                if abs(size - self.expected_volume_size) > self.tolerance:
                    logger.warning(f"Inconsistent volume sizes: {volume_sizes}")
                    return False
        
        return True
    
    def get_volume_statistics(self, volume_chunks: List[Dict]) -> Dict:
        """Get statistics about extracted volumes."""
        if not volume_chunks:
            return {"num_volumes": 0}
        
        volume_sizes = [v['volume_size'] for v in volume_chunks]
        chunk_counts = [len(v['chunks']) for v in volume_chunks]
        coverage_ratios = [v['total_chunk_frames'] / v['volume_size'] for v in volume_chunks]
        
        return {
            "num_volumes": len(volume_chunks),
            "total_frames": sum(volume_sizes),
            "volume_sizes": {
                "mean": np.mean(volume_sizes),
                "std": np.std(volume_sizes),
                "min": min(volume_sizes),
                "max": max(volume_sizes)
            },
            "chunks_per_volume": {
                "mean": np.mean(chunk_counts),
                "std": np.std(chunk_counts),
                "min": min(chunk_counts),
                "max": max(chunk_counts)
            },
            "coverage_ratios": {
                "mean": np.mean(coverage_ratios),
                "std": np.std(coverage_ratios),
                "min": min(coverage_ratios),
                "max": max(coverage_ratios)
            }
        }
    
    def detect_z_one_frame(self, frames: torch.Tensor) -> int:
        """
        Detect the first z=1 frame (start of brain volume) by looking for z-index 0 markers.
        
        Based on investigation findings:
        - The processed data has shape (T, Z, H, W) where Z is z-plane dimension
        - Z-plane 0 corresponds to original z=1 and contains z-index 0 markers
        - Brain volumes cycle through z-planes: z=1 → z=2 → ... → z=max → z=1 (next volume)
        - Volume boundaries occur when returning from max z-plane back to z-plane 0 (z=1)
        
        Args:
            frames: Input frames tensor of shape (T, Z, H, W) or (T, H, W)
            
        Returns:
            Frame index of first z=1 volume start, or -1 if not found
        """
        # Handle different input shapes
        if frames.dim() == 4:
            # Shape (T, Z, H, W) - this is the processed spike grid format
            T, Z, H, W = frames.shape
            
            # Look for volume boundaries by detecting when we return to z-plane 0
            # In the processed data, each timepoint represents a different z-plane
            # A complete volume goes through all z-planes (0 to Z-1)
            
            # The first frame (t=0) should be z-plane 0 (z=1)
            # Subsequent volumes start every Z frames
            logger.info(f"Detected systematic volume pattern: {Z} z-planes per volume")
            logger.info(f"First z=1 volume starts at frame 0")
            return 0
            
        elif frames.dim() == 3:
            # Shape (T, H, W) - flattened temporal sequence
            T, H, W = frames.shape
            
            # In this case, we need to detect z-plane transitions within the temporal sequence
            # Look for z-index 0 markers (2x2 in top-right corner) that indicate z=1 frames
            for frame_idx in range(T):
                frame = frames[frame_idx]
                
                # Look for z-index 0 markers (which correspond to original z=1)
                if H >= 2 and W >= 4:
                    # Check 2x2 marker region in top-right corner
                    marker_region = frame[:2, -2:]
                    
                    # Check if all 4 pixels in marker are active (close to 1)
                    marker_active = torch.all(marker_region > 0.9)
                    
                    if marker_active:
                        # For z-index 0 markers (z=1), check adjacent pixels should be clear
                        adjacent_left = frame[:2, -4:-2]
                        adjacent_clear = torch.all(adjacent_left < 0.1)
                        
                        if adjacent_clear:
                            logger.info(f"Found first z=1 marker (z-index 0) at frame {frame_idx}")
                            return frame_idx
            
            logger.warning("No z=1 markers (z-index 0) found in temporal sequence")
            return -1
        else:
            raise ValueError(f"Expected 3D or 4D input, got {frames.dim()}D")
        
        logger.warning("No z=1 volume start detected")
        return -1

    def detect_z_zero_frame(self, frames: torch.Tensor) -> int:
        """
        DEPRECATED: Use detect_z_one_frame() instead.
        
        This method is kept for backward compatibility but will always return -1
        because z=0 markers do not exist in the processed data.
        """
        logger.warning("detect_z_zero_frame() is deprecated. No z=0 markers exist in processed data. Use detect_z_one_frame() instead.")
        return -1

    def detect_systematic_volume_boundaries(self, 
                                          frames: torch.Tensor,
                                          z_planes_per_volume: int = None) -> List[Tuple[int, int]]:
        """
        Systematically detect brain volume boundaries using data structure analysis.
        
        Based on investigation findings:
        - Processed data has shape (T, Z, H, W) where T=timepoints, Z=z-planes
        - Each timepoint cycles through all z-planes (z=1 → z=2 → ... → z=max)
        - Brain volumes are complete cycles through all z-planes
        - Volume boundaries occur every Z timepoints
        
        Args:
            frames: Input frames tensor of shape (T, Z, H, W) or (T, H, W)
            z_planes_per_volume: Number of z-planes per volume (if None, will auto-detect from shape)
            
        Returns:
            List of (start_idx, end_idx) tuples for brain volumes
        """
        logger.info("Detecting systematic brain volume boundaries")
        
        if frames.dim() == 4:
            # Shape (T, Z, H, W) - processed spike grid format
            T, Z, H, W = frames.shape
            
            if z_planes_per_volume is None:
                z_planes_per_volume = Z  # Each volume contains all z-planes
                logger.info(f"Auto-detected {z_planes_per_volume} z-planes per volume from data shape")
            
            # In this format, each timepoint represents a complete z-stack
            # So each timepoint is a complete volume
            volumes = []
            for t in range(T):
                volumes.append((t, t + 1))
            
            logger.info(f"Detected {len(volumes)} brain volumes (one per timepoint)")
            return volumes
            
        elif frames.dim() == 3:
            # Shape (T, H, W) - flattened temporal sequence
            T, H, W = frames.shape
            
            # Find the first z=1 frame (z-index 0 marker)
            first_z_one = self.detect_z_one_frame(frames)
            
            if first_z_one == -1:
                logger.warning("No z=1 marker found, using fallback volume detection")
                # Fallback: assume volumes start at frame 0 with default size
                if z_planes_per_volume is None:
                    z_planes_per_volume = 30  # Default assumption
                
                volumes = []
                for start in range(0, T, z_planes_per_volume):
                    end = min(start + z_planes_per_volume, T)
                    if end - start >= z_planes_per_volume // 2:  # At least half a volume
                        volumes.append((start, end))
                
                logger.info(f"Created {len(volumes)} fallback volumes with {z_planes_per_volume} z-planes each")
                return volumes
            
            # If z_planes_per_volume not provided, try to detect it
            if z_planes_per_volume is None:
                # Try to find the next z=1 frame to determine interval
                next_z_one = -1
                for frame_idx in range(first_z_one + 1, min(first_z_one + 50, T)):
                    # Check single frame by getting a single frame slice
                    single_frame = frames[frame_idx:frame_idx+1]
                    # Check if this frame has z=1 marker (z-index 0)
                    if self.detect_z_one_frame(single_frame) == 0:  # Returns 0 if found at index 0
                        next_z_one = frame_idx
                        break
                
                if next_z_one != -1:
                    z_planes_per_volume = next_z_one - first_z_one
                    logger.info(f"Detected {z_planes_per_volume} z-planes per volume from frame interval")
                else:
                    z_planes_per_volume = 30  # Default fallback
                    logger.warning(f"Could not detect interval, using default {z_planes_per_volume} z-planes per volume")
            
            # Systematically calculate all volume boundaries
            volumes = []
            volume_idx = 0
            
            while True:
                start_frame = first_z_one + (volume_idx * z_planes_per_volume)
                end_frame = start_frame + z_planes_per_volume
                
                # Stop if we exceed the available frames
                if start_frame >= T:
                    break
                
                # Clip end frame to available data
                end_frame = min(end_frame, T)
                
                # Only include if we have at least half a volume
                if end_frame - start_frame >= z_planes_per_volume // 2:
                    volumes.append((start_frame, end_frame))
                    logger.debug(f"Volume {volume_idx + 1}: frames {start_frame}-{end_frame} "
                               f"(size: {end_frame - start_frame})")
                
                volume_idx += 1
            
            logger.info(f"Detected {len(volumes)} systematic brain volumes with {z_planes_per_volume} z-planes each")
            logger.info(f"First z=1 at frame {first_z_one}, volumes: {volumes}")
            
            return volumes
        else:
            raise ValueError(f"Expected 3D or 4D input, got {frames.dim()}D")
    
    def process_volumes(self, 
                       frames: torch.Tensor,
                       chunks: List[Tuple[int, int]],
                       z_one_frames: List[int]) -> List[Dict]:
        """
        Complete volume processing pipeline.
        
        Args:
            frames: Input frames tensor
            chunks: List of chunk boundaries
            z_one_frames: List of z=1 frame indices (z-index 0 markers)
            
        Returns:
            List of complete volume information dictionaries
        """
        logger.info("Starting volume processing pipeline")
        
        # Detect volume boundaries
        volume_boundaries = self.detect_systematic_volume_boundaries(frames)
        
        if not volume_boundaries:
            logger.warning("No complete volumes detected")
            return []
        
        # Group chunks into volumes
        volume_chunks = self.group_chunks_into_volumes(chunks, volume_boundaries)
        
        # Validate consistency
        if not self.validate_volume_consistency(volume_chunks):
            logger.warning("Volume consistency validation failed")
        
        logger.info(f"Volume processing complete: {len(volume_chunks)} volumes")
        return volume_chunks 