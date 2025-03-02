import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def create_comparison_video(actual, predicted, output_path, num_frames=8, fps=5):
    """Create video comparing actual brain activity with model predictions.
    
    Args:
        actual: numpy array of shape (batch_size, seq_len, 256, 128) with actual data
        predicted: numpy array of shape (batch_size, seq_len-1, 256, 128) with predictions
        output_path: Path to save the video
        num_frames: Maximum number of frames to include in the video
        fps: Frames per second for the video
    """
    try:
        # Set up video parameters
        width = 256
        height = 128
        scale = 2
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use H264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (scaled_width * 3, scaled_height), 
                            isColor=True)
        
        if not out.isOpened():
            # Try an alternative codec silently
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (scaled_width * 3, scaled_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer. Skipping video creation.")
                return
        
        # Process each sequence in the batch
        batch_size = min(actual.shape[0], 2)  # Limit to at most 2 sequences
        
        for batch_idx in range(batch_size):
            # Get current sequence
            current_seq = actual[batch_idx]
            predicted_seq = predicted[batch_idx]
            
            # Process frames
            seq_len = min(current_seq.shape[0]-1, predicted_seq.shape[0], num_frames)
            
            for t in range(seq_len):
                # Get current frame, prediction, and ground truth next frame
                current = current_seq[t]
                next_frame = current_seq[t+1]  # Ground truth
                prediction = predicted_seq[t]   # Model prediction
                
                # Ensure values are in float [0,1] range
                if current.dtype == np.uint8:
                    current = current.astype(np.float32) / 255.0
                if next_frame.dtype == np.uint8:
                    next_frame = next_frame.astype(np.float32) / 255.0
                if prediction.dtype == np.uint8:
                    prediction = prediction.astype(np.float32) / 255.0
                
                # Convert to uint8 images
                curr_img = (current * 255).astype(np.uint8)
                pred_img = (prediction * 255).astype(np.uint8)
                next_img = (next_frame * 255).astype(np.uint8)
                
                # Scale up images
                curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                next_img = cv2.resize(next_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Convert to RGB
                curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                next_rgb = cv2.cvtColor(next_img, cv2.COLOR_GRAY2BGR)
                
                # Add text labels
                cv2.putText(curr_rgb, 'Current', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(pred_rgb, 'Predicted Next', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(next_rgb, 'Actual Next', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add frame number
                cv2.putText(curr_rgb, f'Seq {batch_idx}, Frame {t}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Combine horizontally
                combined = np.hstack([curr_rgb, pred_rgb, next_rgb])
                out.write(combined)
        
        out.release()
        # Simplify the output message
        print(f"Video saved: {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")

def update_loss_plot(train_losses, test_losses, raw_batch_losses, output_path):
    """Update the training/test loss plot showing both raw batch losses and epoch averages"""
    plt.figure(figsize=(15, 5))
    
    # Plot raw batch losses if available
    if raw_batch_losses:
        plt.subplot(1, 2, 1)
        plt.plot(raw_batch_losses, label='Raw Batch Loss', alpha=0.5)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Raw Training Loss')
        plt.grid(True)
        plt.legend()
    
    # Plot epoch averages
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss (Epoch Avg)')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Progress')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 