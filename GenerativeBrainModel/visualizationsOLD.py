import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoVisualizer:
    """
    Generates comparison videos showing original vs predicted volumes from validation data.
    """
    
    def __init__(self, output_dir: Path, fps: int = 1):
        """
        Initialize the video visualizer.
        
        Args:
            output_dir: Directory to save videos
            fps: Frames per second for the output video
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        
    def tensor_to_uint8(self, tensor: torch.Tensor, clip_percentile: float = 99.5) -> np.ndarray:
        """
        Convert tensor to uint8 numpy array for video encoding.
        
        Args:
            tensor: Input tensor with values typically in [0, 1] range
            clip_percentile: Percentile for clipping outliers
            
        Returns:
            Numpy array with values in [0, 255] range
        """
        # Move to CPU and convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        
        # Clip outliers for better visualization
        lower_bound = np.percentile(array, 100 - clip_percentile)
        upper_bound = np.percentile(array, clip_percentile)
        array = np.clip(array, lower_bound, upper_bound)
        
        # Normalize to [0, 1] range
        if upper_bound > lower_bound:
            array = (array - lower_bound) / (upper_bound - lower_bound)
        else:
            array = np.zeros_like(array)
        
        # Convert to uint8 [0, 255]
        return (array * 255).astype(np.uint8)
    
    def create_side_by_side_frame(self, original: np.ndarray, predicted: np.ndarray, 
                                 frame_idx: int, seq_info: dict = None) -> np.ndarray:
        """
        Create a side-by-side comparison frame.
        
        Args:
            original: Original 2D slice (H, W) as uint8
            predicted: Predicted 2D slice (H, W) as uint8
            frame_idx: Frame index for labeling
            seq_info: Optional dict with 'seq_idx', 'time_idx', 'is_seq2seq' for sequence labeling
            
        Returns:
            Side-by-side frame (H, W*2) as uint8, converted to BGR for OpenCV
        """
        H, W = original.shape
        
        # Create side-by-side frame
        frame = np.zeros((H, W * 2), dtype=np.uint8)
        frame[:, :W] = original
        frame[:, W:] = predicted
        
        # Convert grayscale to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add text labels with reduced font size (1/3 of original)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.23  # Reduced to ~1/3 of original (0.7 -> 0.23)
        color = (255, 255, 255)  # White text
        thickness = 1  # Reduced thickness for smaller text
        
        # Add "Original" label
        cv2.putText(frame_bgr, "Original", (10, 30), font, font_scale, color, thickness)
        
        # Add "Predicted" label
        cv2.putText(frame_bgr, "Predicted", (W + 10, 30), font, font_scale, color, thickness)
        
        # Add frame index
        cv2.putText(frame_bgr, f"Frame {frame_idx}", (10, H - 10), font, font_scale, color, thickness)
        
        # Add sequence information for seq2seq videos
        if seq_info and seq_info.get('is_seq2seq', False):
            seq_idx = seq_info.get('seq_idx', 0)
            time_idx = seq_info.get('time_idx', 0)
            seq_text = f"Seq {seq_idx}, Time {time_idx}"
            # Position sequence info in the middle-right area
            cv2.putText(frame_bgr, seq_text, (W + 10, H - 10), font, font_scale, color, thickness)
        
        return frame_bgr
    
    def generate_comparison_video(self, 
                                 original_volumes: torch.Tensor, 
                                 predicted_volumes: torch.Tensor,
                                 video_name: str = "validation_comparison.mp4",
                                 max_frames: Optional[int] = None,
                                 seq2seq: bool = False) -> Path:
        """
        Generate a comparison video from validation volumes.
        
        Args:
            original_volumes: Original volumes tensor (B, T, X, Y, Z)
            predicted_volumes: Predicted volumes tensor (B, T, X, Y, Z) 
            video_name: Name of the output video file
            max_frames: Maximum number of frames to include (None for all)
            seq2seq: If True, add sequence information to video frames
            
        Returns:
            Path to the saved video file
        """
        logger.info(f"Generating comparison video: {video_name}")
        
        # Ensure tensors are on CPU
        if original_volumes.is_cuda:
            original_volumes = original_volumes.cpu()
        if predicted_volumes.is_cuda:
            predicted_volumes = predicted_volumes.cpu()
            
        # Get dimensions
        B, T, X, Y, Z = original_volumes.shape
        middle_z = Z // 2  # Middle z slice
        
        # Determine number of frames to process
        total_frames = B * T
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        logger.info(f"Processing {total_frames} frames from volumes of shape {original_volumes.shape}")
        logger.info(f"Using middle z slice: {middle_z} (out of {Z})")
        if seq2seq:
            logger.info("Adding sequence information to video frames")
        
        # Setup video writer
        video_path = self.output_dir / video_name
        frame_height, frame_width = Y, X * 2  # Side-by-side doubles the width
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), 
            fourcc, 
            self.fps, 
            (frame_width, frame_height)
        )
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
        
        frame_count = 0
        
        try:
            # Process each volume
            for b in tqdm(range(B), desc="Processing volumes"):
                for t in range(T):
                    if frame_count >= total_frames:
                        break
                        
                    # Extract middle z slice: (X, Y, Z) -> (Y, X) for display
                    orig_slice = original_volumes[b, t, :, :, middle_z].transpose(0, 1)  # (Y, X)
                    pred_slice = predicted_volumes[b, t, :, :, middle_z].transpose(0, 1)  # (Y, X)
                    
                    # Convert to uint8
                    orig_uint8 = self.tensor_to_uint8(orig_slice)
                    pred_uint8 = self.tensor_to_uint8(pred_slice)
                    
                    # Prepare sequence info for seq2seq videos
                    seq_info = None
                    if seq2seq:
                        seq_info = {
                            'seq_idx': b,
                            'time_idx': t,
                            'is_seq2seq': True
                        }
                    
                    # Create side-by-side frame
                    frame = self.create_side_by_side_frame(orig_uint8, pred_uint8, frame_count, seq_info)
                    
                    # Write frame to video
                    video_writer.write(frame)
                    frame_count += 1
                
                if frame_count >= total_frames:
                    break
                    
        finally:
            video_writer.release()
        
        logger.info(f"Video saved: {video_path} ({frame_count} frames)")
        return video_path
    
    def generate_validation_comparison_video(self, 
                                           model: torch.nn.Module,
                                           validation_loader: torch.utils.data.DataLoader,
                                           device: torch.device,
                                           video_name: str = "validation_comparison.mp4",
                                           max_batches: int = 5,
                                           seq2seq: bool = False) -> Path:
        """
        Generate comparison video using validation data and model predictions.
        
        Args:
            model: Trained model for making predictions
            validation_loader: DataLoader for validation data
            device: Device to run inference on
            video_name: Name of the output video file
            max_batches: Maximum number of batches to process
            seq2seq: If True, model predicts next frame (shift original frames by 1 for alignment)
            
        Returns:
            Path to the saved video file
        """
        logger.info("Generating validation comparison video...")
        
        model.eval()
        
        original_volumes_list = []
        predicted_volumes_list = []
        
        with torch.no_grad():
            batch_count = 0
            for batch_data in tqdm(validation_loader, desc="Processing validation batches"):
                if batch_count >= max_batches:
                    break
                    
                # Handle different data formats
                if isinstance(batch_data, (list, tuple)):
                    data, target = batch_data
                else:
                    data = batch_data
                    target = data  # Autoencoder target is input
                
                # Move to device
                data = data.to(device)
                target = target.to(device)
                
                # Add sequence dimension if needed: (B, X, Y, Z) -> (B, T=1, X, Y, Z)
                if len(data.shape) == 4:
                    data = data.unsqueeze(1)
                    target = target.unsqueeze(1)
                
                if seq2seq:
                    # For seq2seq models, we need sequences with T > 1 to predict next frames
                    if data.shape[1] < 2:
                        logger.warning(f"Seq2seq mode requires sequences with T >= 2, got T={data.shape[1]}. Skipping batch.")
                        continue
                    
                    # Prepare input (all frames except last) and target (all frames except first)
                    input_seq = data[:, :-1]  # (B, T-1, X, Y, Z)
                    target_seq = target[:, 1:]  # (B, T-1, X, Y, Z) - next frames
                    
                    # Get model predictions for next frames
                    predicted_logits = model(input_seq, get_logits=True)
                    predicted_probs = torch.sigmoid(predicted_logits)
                    
                    # Store aligned volumes: target_seq contains the true "next frames"
                    # that correspond to the predicted "next frames"
                    original_volumes_list.append(target_seq)
                    predicted_volumes_list.append(predicted_probs)
                else:
                    # Standard autoencoder mode: predict same frame
                    predicted_logits = model(data)
                    predicted_probs = torch.sigmoid(predicted_logits)
                    
                    # Store original and predicted volumes (same frame alignment)
                    original_volumes_list.append(target)
                    predicted_volumes_list.append(predicted_probs)
                
                batch_count += 1
        
        if not original_volumes_list:
            raise ValueError("No valid batches processed for video generation")
        
        # Concatenate all batches
        original_volumes = torch.cat(original_volumes_list, dim=0)
        predicted_volumes = torch.cat(predicted_volumes_list, dim=0)
        
        logger.info(f"Collected {original_volumes.shape[0]} volumes for video generation")
        if seq2seq:
            logger.info("Using seq2seq mode: comparing true next frames with predicted next frames")
        
        # Generate the comparison video
        return self.generate_comparison_video(
            original_volumes=original_volumes,
            predicted_volumes=predicted_volumes,
            video_name=video_name,
            max_frames=200,  # Limit to 200 frames for reasonable video length
            seq2seq=seq2seq
        )


def create_validation_video(model: torch.nn.Module,
                          validation_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          experiment_dir: Path,
                          video_name: str = "validation_comparison.mp4",
                          seq2seq: bool = False) -> Path:
    """
    Convenience function to create validation comparison video.
    
    Args:
        model: Trained model
        validation_loader: Validation data loader
        device: Device for inference
        experiment_dir: Experiment directory
        video_name: Name of output video
        seq2seq: If True, model predicts next frame (for seq2seq models)
        
    Returns:
        Path to saved video
    """
    # Create video directory
    video_dir = experiment_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    
    # Create visualizer and generate video
    visualizer = VideoVisualizer(output_dir=video_dir, fps=1)
    
    return visualizer.generate_validation_comparison_video(
        model=model,
        validation_loader=validation_loader,
        device=device,
        video_name=video_name,
        max_batches=5,  # Process 5 batches for video
        seq2seq=seq2seq
    ) 