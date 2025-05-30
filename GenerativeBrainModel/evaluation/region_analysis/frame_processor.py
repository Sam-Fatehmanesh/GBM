"""
FrameProcessor: Detect z-zero frames and extract valid chunks for evaluation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Process frames to detect z-zero markers and extract valid chunks.
    
    Handles:
    - Z-marker detection: 2×2 top-right markers with clear adjacent area
    - Frame validation: Strict requirements for chunk boundaries
    - Chunk extraction: 330-frame sequences for evaluation
    """
    
    def __init__(self, 
                 chunk_size: int = 330,
                 marker_threshold: float = 0.3,  # More lenient threshold
                 clear_area_threshold: float = 0.5):  # More lenient clear area
        """
        Initialize FrameProcessor.
        
        Args:
            chunk_size: Size of frame chunks to extract (default 330)
            marker_threshold: Threshold for detecting active markers (default 0.8)
            clear_area_threshold: Threshold for "clear" adjacent area (default 0.2)
        """
        self.chunk_size = chunk_size
        self.marker_threshold = marker_threshold
        self.clear_area_threshold = clear_area_threshold
    
    def detect_z_one_frames(self, frames: torch.Tensor) -> List[int]:
        """
        Detect frames with valid z=1 markers (z-index 0 markers).
        
        Based on investigation findings:
        - Processed data has shape (T, Z, H, W) where each timepoint contains all z-planes
        - Z-plane 0 corresponds to original z=1 and contains z-index 0 markers
        - Brain volumes are complete z-stacks, so every timepoint is a volume start
        
        Args:
            frames: Tensor of shape (T, Z, H, W) or (T, H, W)
            
        Returns:
            List of frame indices with valid z=1 markers
        """
        # Handle different input shapes
        if frames.dim() == 4:
            # Shape (T, Z, H, W) - processed spike grid format
            T, Z, H, W = frames.shape
            logger.info(f"Detecting z=1 markers in {T} timepoints with {Z} z-planes each")
            
            z_one_frames = []
            
            for t in range(T):
                # Check if this timepoint has z-index 0 marker in z-plane 0
                z_stack = frames[t]  # Shape (Z, H, W)
                
                if self._has_valid_z_one_marker(z_stack):
                    z_one_frames.append(t)
            
            logger.info(f"Found {len(z_one_frames)} timepoints with valid z=1 markers")
            
            # If no markers found, assume all timepoints are valid (each is a complete volume)
            if len(z_one_frames) == 0:
                logger.warning("No z=1 markers detected. Assuming all timepoints are valid volumes.")
                z_one_frames = list(range(T))
                logger.info(f"Created {len(z_one_frames)} fallback z=1 markers (all timepoints)")
            
            return z_one_frames
            
        elif frames.dim() == 3:
            # Shape (T, H, W) - flattened temporal sequence
            T, H, W = frames.shape
            z_one_frames = []
            
            logger.info(f"Detecting z=1 markers (z-index 0) in {T} frames of shape ({H}, {W})")
            
            for t in range(T):
                frame = frames[t]
                
                if self._has_valid_z_one_marker(frame):
                    z_one_frames.append(t)
            
            logger.info(f"Found {len(z_one_frames)} frames with valid z=1 markers")
            
            # If no z=1 markers found, create fallback markers based on chunk size
            if len(z_one_frames) == 0:
                logger.warning("No z=1 markers found. Creating fallback markers based on chunk size.")
                # Create markers at regular intervals based on chunk_size
                fallback_markers = list(range(0, T, self.chunk_size))
                if fallback_markers and fallback_markers[-1] + self.chunk_size <= T:
                    # Add one more marker if there's room for another complete chunk
                    fallback_markers.append(fallback_markers[-1] + self.chunk_size)
                logger.info(f"Created {len(fallback_markers)} fallback z=1 markers: {fallback_markers}")
                return fallback_markers
            
            return z_one_frames
        else:
            raise ValueError(f"Expected 3D or 4D input, got {frames.dim()}D")

    def detect_z_zero_frames(self, frames: torch.Tensor) -> List[int]:
        """
        DEPRECATED: Use detect_z_one_frames() instead.
        
        This method is kept for backward compatibility but will delegate to detect_z_one_frames()
        because z=0 markers do not exist in the processed data.
        """
        logger.warning("detect_z_zero_frames() is deprecated. No z=0 markers exist in processed data. Using detect_z_one_frames() instead.")
        return self.detect_z_one_frames(frames)
    
    def _has_valid_z_one_marker(self, frame: torch.Tensor) -> bool:
        """
        Check if frame has valid z=1 marker (z-index 0: 2×2 top-right + adjacent pixels are zero).
        
        Based on investigation findings:
        - Z-index 0 markers correspond to original z=1 (start of brain volume)
        - Marker pattern: 2×2 pixels in top-right corner with clear adjacent area
        - Markers are spatial markers within z-planes, not temporal markers
        
        Args:
            frame: Single frame tensor of shape (H, W) or (Z, H, W)
            
        Returns:
            True if frame has valid z=1 marker (z-index 0)
        """
        # Handle different input shapes
        if frame.dim() == 3:
            # Shape (Z, H, W) - check z-plane 0 for z-index 0 marker
            Z, H, W = frame.shape
            if Z > 0:
                frame = frame[0]  # Check first z-plane (z=1, z-index 0)
            else:
                return False
        elif frame.dim() == 2:
            # Shape (H, W) - single frame
            H, W = frame.shape
        else:
            return False
        
        # Need at least a 2×4 area to check marker + adjacent pixels
        if H < 2 or W < 4:
            return False
        
        # Define 2×2 marker region in top-right corner
        marker_region = frame[:2, -2:]  # Use top-right corner
        
        # Check if all 4 pixels in marker are above threshold
        marker_active = torch.all(marker_region > self.marker_threshold)
        
        if not marker_active:
            return False
        
        # Check the two pixels immediately to the left of the 2×2 marker
        # These should be zero (below clear_area_threshold)
        adjacent_pixels = frame[:2, -4:-2]  # Two columns to the left of marker
        
        # Check if adjacent pixels are zero/clear
        adjacent_clear = torch.all(adjacent_pixels < self.clear_area_threshold)
        
        return adjacent_clear

    def _has_valid_z_marker(self, frame: torch.Tensor) -> bool:
        """
        DEPRECATED: Use _has_valid_z_one_marker() instead.
        
        This method is kept for backward compatibility.
        """
        return self._has_valid_z_one_marker(frame)
    
    def extract_valid_chunks(self, 
                           frames: torch.Tensor, 
                           z_one_frames: List[int]) -> List[Tuple[int, int]]:
        """
        Extract valid chunk boundaries from z=1 frame indices.
        
        Args:
            frames: Input frames tensor
            z_one_frames: List of z=1 frame indices (z-index 0 markers)
            
        Returns:
            List of (start_idx, end_idx) tuples for valid chunks
        """
        if len(z_one_frames) < 2:
            logger.warning("Need at least 2 z=1 frames to extract chunks")
            return []
        
        valid_chunks = []
        T = frames.shape[0]
        
        for i in range(len(z_one_frames) - 1):
            start_idx = z_one_frames[i]
            
            # Look for the chunk end
            # We need exactly chunk_size frames starting from start_idx
            end_idx = start_idx + self.chunk_size
            
            # Check if we have enough frames
            if end_idx > T:
                logger.debug(f"Chunk starting at {start_idx} would exceed available frames ({T})")
                continue
            
            # Check if the expected end aligns with next z=1 frame (within tolerance)
            next_z_one = z_one_frames[i + 1] if i + 1 < len(z_one_frames) else None
            
            if next_z_one is not None:
                # Allow some tolerance (±2 frames) for z=1 alignment
                if abs(end_idx - next_z_one) <= 2:
                    # Adjust end to align with actual z=1
                    end_idx = next_z_one
                    valid_chunks.append((start_idx, end_idx))
                    logger.debug(f"Valid chunk: frames {start_idx} to {end_idx} (adjusted)")
                else:
                    logger.debug(f"Chunk {start_idx}-{end_idx} doesn't align with next z=1 at {next_z_one}")
            else:
                # Last possible chunk - just check if we have enough frames
                if end_idx <= T:
                    valid_chunks.append((start_idx, end_idx))
                    logger.debug(f"Valid chunk: frames {start_idx} to {end_idx} (final)")
        
        logger.info(f"Extracted {len(valid_chunks)} valid chunks of size ~{self.chunk_size}")
        return valid_chunks
    
    def validate_chunk(self, 
                      frames: torch.Tensor, 
                      start_idx: int, 
                      end_idx: int) -> bool:
        """
        Validate that a chunk meets all requirements.
        
        Args:
            frames: Input frames tensor
            start_idx: Start frame index
            end_idx: End frame index
            
        Returns:
            True if chunk is valid
        """
        chunk_length = end_idx - start_idx
        
        # Check chunk length
        if abs(chunk_length - self.chunk_size) > 2:
            return False
        
        # Check that start and end frames have z=1 markers (z-index 0)
        if not self._has_valid_z_one_marker(frames[start_idx]):
            return False
        
        if end_idx < frames.shape[0] and not self._has_valid_z_one_marker(frames[end_idx]):
            return False
        
        return True
    
    def process_frames(self, frames: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Complete frame processing pipeline.
        
        Args:
            frames: Input frames tensor
            
        Returns:
            List of valid (start_idx, end_idx) chunk boundaries
        """
        logger.info("Starting frame processing pipeline")
        
        # Step 1: Detect z=1 frames (z-index 0 markers)
        z_one_frames = self.detect_z_one_frames(frames)
        
        if len(z_one_frames) < 2:
            logger.warning("Insufficient z=1 frames found for chunk extraction")
            return []
        
        # Step 2: Extract potential chunks
        potential_chunks = self.extract_valid_chunks(frames, z_one_frames)
        
        # Step 3: Validate chunks
        valid_chunks = []
        for start_idx, end_idx in potential_chunks:
            if self.validate_chunk(frames, start_idx, end_idx):
                valid_chunks.append((start_idx, end_idx))
            else:
                logger.debug(f"Chunk {start_idx}-{end_idx} failed validation")
        
        logger.info(f"Frame processing complete: {len(valid_chunks)} valid chunks")
        return valid_chunks
    
    def get_chunk_info(self, chunks: List[Tuple[int, int]]) -> dict:
        """Get information about extracted chunks."""
        if not chunks:
            return {"num_chunks": 0, "total_frames": 0}
        
        chunk_lengths = [end - start for start, end in chunks]
        
        return {
            "num_chunks": len(chunks),
            "total_frames": sum(chunk_lengths),
            "chunk_lengths": chunk_lengths,
            "mean_chunk_length": np.mean(chunk_lengths),
            "std_chunk_length": np.std(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths)
        } 