import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

def create_comparison_video(actual, predicted, output_path, num_frames=330, fps=1, sampled_predictions=None):
    """Create video comparing actual brain activity with model predictions.
    
    Args:
        actual: numpy array of shape (batch_size, seq_len, 256, 128) with actual data
        predicted: numpy array of shape (batch_size, seq_len-1, 256, 128) with predictions
        output_path: Path to save the video
        num_frames: Maximum sequence length to use (default: 330 for full sequences)
        fps: Frames per second for the video (default: 1)
        sampled_predictions: numpy array of shape (batch_size, seq_len-1, 256, 128) with sampled binary predictions
    """
    try:
        # Derive spatial dimensions from data to avoid aspect distortion
        # actual shape: (batch, seq_len, H, W)
        height = actual.shape[2]
        width = actual.shape[3]
        scale = 2
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if we have sampled predictions (4-panel mode) or just predictions (3-panel mode)
        has_samples = sampled_predictions is not None
        
        # Set video dimensions based on layout (2x2 grid or 3 in a row)
        if has_samples:
            # 2x2 grid layout
            video_width = scaled_width * 2
            video_height = scaled_height * 2
        else:
            # Original 3 in a row layout
            video_width = scaled_width * 3
            video_height = scaled_height
        
        # Use H264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (video_width, video_height), 
                            isColor=True)
        
        if not out.isOpened():
            # Try an alternative codec silently
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (video_width, video_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer. Skipping video creation.")
                return
        
        # Process each sequence in the batch
        batch_size = min(actual.shape[0], 100)  # Limit to at most 100 sequences
        total_frames = 0
        max_total_frames = 1000  # Safety limit to prevent extremely large videos
        
        for batch_idx in range(batch_size):
            # Get current sequence
            current_seq = actual[batch_idx]
            predicted_seq = predicted[batch_idx]
            
            # If we have sampled predictions, get those too
            if has_samples:
                sampled_seq = sampled_predictions[batch_idx]
            
            # Use full sequence length, limited by available frames
            seq_len = min(current_seq.shape[0]-1, predicted_seq.shape[0], num_frames)
            
            # Check if adding this sequence would exceed the max frames limit
            frames_left = max_total_frames - total_frames
            if frames_left <= 0:
                # We've reached the maximum, stop adding sequences
                break
                
            # If this sequence would exceed the limit, truncate it
            if seq_len > frames_left:
                seq_len = frames_left
            
            for t in range(seq_len):
                # Get current frame, prediction, and ground truth next frame
                current = current_seq[t]
                next_frame = current_seq[t+1]  # Ground truth
                prediction = predicted_seq[t]   # Model prediction
                
                # Get sampled prediction if available
                if has_samples:
                    sampled = sampled_seq[t]  # Sampled binary prediction
                
                # Ensure values are in float [0,1] range
                if current.dtype == np.uint8:
                    current = current.astype(np.float32) / 255.0
                if next_frame.dtype == np.uint8:
                    next_frame = next_frame.astype(np.float32) / 255.0
                if prediction.dtype == np.uint8:
                    prediction = prediction.astype(np.float32) / 255.0
                if has_samples and sampled.dtype == np.uint8:
                    sampled = sampled.astype(np.float32) / 255.0
                
                # Convert to uint8 images
                curr_img = (current * 255).astype(np.uint8)
                pred_img = (prediction * 255).astype(np.uint8)
                next_img = (next_frame * 255).astype(np.uint8)
                if has_samples:
                    sample_img = (sampled * 255).astype(np.uint8)
                
                # Scale up images
                curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                next_img = cv2.resize(next_img, (scaled_width, scaled_height), 
                                    interpolation=cv2.INTER_NEAREST)
                if has_samples:
                    sample_img = cv2.resize(sample_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Convert to RGB
                curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                next_rgb = cv2.cvtColor(next_img, cv2.COLOR_GRAY2BGR)
                if has_samples:
                    sample_rgb = cv2.cvtColor(sample_img, cv2.COLOR_GRAY2BGR)
                
                    # Add text labels
                cv2.putText(curr_rgb, 'Previous Spikes', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(pred_rgb, 'Predicted Spike Probabilities', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(next_rgb, 'True Spikes', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if has_samples:
                    cv2.putText(sample_rgb, 'Predicted Spike Samples', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add frame number and sequence info
                cv2.putText(curr_rgb, f'Seq {batch_idx+1}/{batch_size}, Frame {t+1}/{seq_len}', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Combine images based on layout
                if has_samples:
                    # Create 2x2 grid: current | predicted
                    #                  actual  | sampled
                    top_row = np.hstack([curr_rgb, pred_rgb])
                    bottom_row = np.hstack([next_rgb, sample_rgb])
                    combined = np.vstack([top_row, bottom_row])
                else:
                    # Original 3 in a row layout
                    combined = np.hstack([curr_rgb, pred_rgb, next_rgb])
                
                out.write(combined)
                total_frames += 1
            
            # Check if we've reached the maximum frames
            if total_frames >= max_total_frames:
                print(f"Reached maximum frame limit ({max_total_frames}). Truncating video.")
                break
        
        out.release()
        # Simplify the output message
        print(f"Video saved: {os.path.basename(output_path)} ({total_frames} frames, {batch_size} sequences)")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")


def create_color_coded_comparison_video(actual, predicted, sampled_predictions, output_path, num_frames=330, fps=1, threshold_left_panels=False):
    """Create video comparing actual brain activity with model predictions, with color-coded bottom right panel.
    
    Args:
        actual: numpy array of shape (batch_size, seq_len, 256, 128) with actual data
        predicted: numpy array of shape (batch_size, seq_len-1, 256, 128) with predictions
        sampled_predictions: numpy array of shape (batch_size, seq_len-1, 256, 128) with sampled binary predictions
        output_path: Path to save the video
        num_frames: Maximum sequence length to use (default: 330 for full sequences)
        fps: Frames per second for the video (default: 1)
        threshold_left_panels: If True, threshold left panels at 0.5 for binary display (for probability data)
    """
    try:
        # Derive spatial dimensions from data to avoid aspect distortion
        # actual shape: (batch, seq_len, H, W)
        height = actual.shape[2]
        width = actual.shape[3]
        scale = 2
        scaled_width = width * scale
        scaled_height = height * scale
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 2x2 grid layout
        video_width = scaled_width * 2
        video_height = scaled_height * 2
        
        # Use H264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (video_width, video_height), 
                            isColor=True)
        
        if not out.isOpened():
            # Try an alternative codec silently
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (video_width, video_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer. Skipping video creation.")
                return
        
        # Process each sequence in the batch
        batch_size = min(actual.shape[0], 100)  # Limit to at most 100 sequences
        total_frames = 0
        max_total_frames = 1000  # Safety limit to prevent extremely large videos
        
        for batch_idx in range(batch_size):
            # Get current sequence
            current_seq = actual[batch_idx]
            predicted_seq = predicted[batch_idx]
            sampled_seq = sampled_predictions[batch_idx]
            
            # Use full sequence length, limited by available frames
            seq_len = min(current_seq.shape[0]-1, predicted_seq.shape[0], num_frames)
            
            # Check if adding this sequence would exceed the max frames limit
            frames_left = max_total_frames - total_frames
            if frames_left <= 0:
                # We've reached the maximum, stop adding sequences
                break
                
            # If this sequence would exceed the limit, truncate it
            if seq_len > frames_left:
                seq_len = frames_left
            
            for t in range(seq_len):
                try:
                    # Get current frame, prediction, and ground truth next frame
                    current = current_seq[t]
                    next_frame = current_seq[t+1]  # Ground truth
                    prediction = predicted_seq[t]   # Model prediction (probabilities)
                    sampled = sampled_seq[t]  # Sampled binary prediction
                    
                    # Ensure values are in appropriate ranges
                    if current.dtype == np.uint8:
                        current = current.astype(np.float32) / 255.0
                    if next_frame.dtype == np.uint8:
                        next_frame = next_frame.astype(np.float32) / 255.0
                    if prediction.dtype == np.uint8:
                        prediction = prediction.astype(np.float32) / 255.0
                    if sampled.dtype == np.uint8:
                        sampled = sampled.astype(np.float32) / 255.0
                    
                    # Convert binary values to 0/1 for classification
                    sampled_binary = (sampled > 0.5).astype(np.uint8)
                    actual_binary = (next_frame > 0.5).astype(np.uint8)
                    
                    # Debug: ensure we have the right dimensions
                    if current.shape != (height, width):
                        tqdm.write(f"Warning: Unexpected data shape {current.shape}, expected ({height}, {width})")
                    
                    # Create color-coded image for bottom right panel
                    # Green: True positive (predicted=1, actual=1)
                    # Red: False positive (predicted=1, actual=0)  
                    # Black: True negative and False negative (predicted=0 or actual=0)
                    color_coded = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Ensure masks have the same shape as the color_coded array's spatial dimensions
                    assert sampled_binary.shape == (height, width), f"sampled_binary shape {sampled_binary.shape} doesn't match expected ({height}, {width})"
                    assert actual_binary.shape == (height, width), f"actual_binary shape {actual_binary.shape} doesn't match expected ({height}, {width})"
                    
                    # True positives: Green
                    tp_mask = (sampled_binary == 1) & (actual_binary == 1)
                    color_coded[tp_mask] = [0, 255, 0]  # Green in BGR
                    
                    # False positives: Red
                    fp_mask = (sampled_binary == 1) & (actual_binary == 0)
                    color_coded[fp_mask] = [0, 0, 255]  # Red in BGR
                    
                    # True negatives and False negatives: Black (already initialized to black)
                    # No need to set these explicitly as they're already [0, 0, 0]
                    
                    # Threshold left panels if using probability data
                    if threshold_left_panels:
                        # Threshold at 0.5 for binary display
                        current_display = (current > 0.5).astype(np.float32)
                        next_frame_display = (next_frame > 0.5).astype(np.float32)
                    else:
                        current_display = current
                        next_frame_display = next_frame
                    
                    # Convert other images to uint8
                    curr_img = (current_display * 255).astype(np.uint8)
                    pred_img = (prediction * 255).astype(np.uint8)
                    next_img = (next_frame_display * 255).astype(np.uint8)
                    
                    # Scale up images
                    curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    next_img = cv2.resize(next_img, (scaled_width, scaled_height), 
                                        interpolation=cv2.INTER_NEAREST)
                    color_coded_scaled = cv2.resize(color_coded, (scaled_width, scaled_height), 
                                                  interpolation=cv2.INTER_NEAREST)
                    
                    # Convert grayscale images to RGB
                    curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                    next_rgb = cv2.cvtColor(next_img, cv2.COLOR_GRAY2BGR)
                    # color_coded_scaled is already in BGR format
                    
                    # Add text labels based on whether we're thresholding left panels
                    if threshold_left_panels:
                        cv2.putText(curr_rgb, 'Previous Spikes', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(next_rgb, 'True Spikes', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(curr_rgb, 'Previous Spikes', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(next_rgb, 'True Spikes', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.putText(pred_rgb, 'Predicted Spike Probabilities', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(color_coded_scaled, 'Predictions: TP=Green, FP=Red', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
                    
                    # Add frame number and sequence info
                    cv2.putText(curr_rgb, f'Seq {batch_idx+1}/{batch_size}, Frame {t+1}/{seq_len}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Create 2x2 grid: current | predicted
                    #                  actual  | color_coded
                    top_row = np.hstack([curr_rgb, pred_rgb])
                    bottom_row = np.hstack([next_rgb, color_coded_scaled])
                    combined = np.vstack([top_row, bottom_row])
                    
                    out.write(combined)
                    total_frames += 1
                    
                except Exception as frame_error:
                    tqdm.write(f"Error processing frame {t} in sequence {batch_idx}: {str(frame_error)}")
                    # Skip this frame and continue
                    continue
            
            # Check if we've reached the maximum frames
            if total_frames >= max_total_frames:
                tqdm.write(f"Reached maximum frame limit ({max_total_frames}). Truncating video.")
                break
        
        out.release()
        # Simplify the output message
        tqdm.write(f"Color-coded video saved: {os.path.basename(output_path)} ({total_frames} frames, {batch_size} sequences)")
        
        # Verify the video file was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # If file is very small, it likely failed
                tqdm.write(f"Warning: Video file is unusually small ({file_size} bytes), generation may have failed")
        else:
            tqdm.write(f"Warning: Video file was not created at {output_path}")
        
    except Exception as e:
        tqdm.write(f"Error creating color-coded video: {str(e)}")
        import traceback
        traceback.print_exc()


def create_prediction_video(model, data_loader, output_path, num_frames=330, device='cuda', print_batch_metrics=True, sample_bernoulli=False, two_panel=False):
    """Create a video of model predictions vs actual brain activity with color-coded bottom right panel.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for test data
        output_path: Path to save the video
        num_frames: Maximum number of frames to include
        device: Device to run model on
        print_batch_metrics: Whether to print metrics for the batch being visualized
        sample_bernoulli: Whether to use Bernoulli sampling for binary predictions
        two_panel: Whether to create a two-panel video (True Spikes vs Predicted Probabilities)
    """
    # Get a single batch
    data_loader.reset()
    batch = next(iter(data_loader))
    
    # Handle both probability mode (tuple) and binary mode (single tensor)
    if isinstance(batch, tuple):
        # Probability mode: batch is (input_data, target_data)
        input_data, target_data = batch
        # Use input_data for model input (always binary)
        batch_for_model = input_data
        # Use input_data for visualization (binary inputs)
        batch_viz = input_data
        is_probability_mode = True
        tqdm.write("Using probability data loader - feeding binary inputs to model")
    else:
        # Binary mode: batch is single tensor
        batch_for_model = batch
        batch_viz = batch
        is_probability_mode = False
        tqdm.write("Using binary data loader")
    
    # Ensure batch is on CUDA
    if batch_for_model.device.type != 'cuda':
        batch_for_model = batch_for_model.cuda()
    if batch_viz.device.type != 'cuda':
        batch_viz = batch_viz.cuda()
    
    # Convert uint8 to float32 for visualization if needed
    if batch_viz.dtype == torch.uint8:
        batch_viz = batch_viz.float()
    
    # Generate predictions using binary input data
    model.eval()
    with torch.no_grad():
        with autocast():
            # Get probability predictions from the model using binary inputs
            predictions = model.get_predictions(batch_for_model)
            
            if two_panel:
                # In two-panel mode we always threshold true spikes and show probabilities
                sampled_predictions = None  # not used
            elif sample_bernoulli:
                # Bernoulli sampling to generate binary spikes
                sampled_predictions = torch.bernoulli(predictions)
            else:
                # Threshold at 0.5 for deterministic binary predictions
                sampled_predictions = (predictions > 0.5).float()
    
    # Calculate metrics for THIS batch only if requested (always threshold-based)
    if print_batch_metrics:
        with torch.no_grad():
            preds = (predictions > 0.5).bool()
            targets = (batch_for_model[:, 1:] > 0.5).bool()
            
            # Calculate TP, FP, FN for this batch
            tp = (preds & targets).sum().item()
            fp = (preds & ~targets).sum().item()
            fn = (~preds & targets).sum().item()
            
            # Calculate metrics for this batch
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            tqdm.write(f"Video batch metrics - Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")
    
    # Prepare data for visualization
    num_sequences = min(100, batch_viz.size(0))
    rand_indices = torch.randperm(batch_viz.size(0))[:num_sequences]
    batch_np = batch_viz[rand_indices].cpu().numpy()

    if two_panel:
        tqdm.write("Creating two-panel video (True Spikes vs Predicted Probabilities)")
        # dimensions
        height = batch_np.shape[1]
        width = batch_np.shape[2]
        scale = 2
        scaled_w = width*scale
        scaled_h = height*scale
        video_width = scaled_w*2
        video_height = scaled_h
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, 1, (video_width, video_height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, 1, (video_width, video_height))
        total_frames=0
        for seq_idx in range(num_sequences):
            true_seq = batch_np[seq_idx]
            pred_prob_seq = predictions[rand_indices[seq_idx]].cpu().numpy()
            seq_len=min(num_frames, true_seq.shape[0]-1)
            for t in range(seq_len):
                true_img = (true_seq[t+1]>0.5).astype(np.uint8)*255
                pred_img = (pred_prob_seq[t]*255).astype(np.uint8)
                true_rgb=cv2.cvtColor(cv2.resize(true_img,(scaled_w,scaled_h),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                pred_rgb=cv2.cvtColor(cv2.resize(pred_img,(scaled_w,scaled_h),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                cv2.putText(true_rgb,'True Spikes',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                cv2.putText(pred_rgb,'Predicted Spike Probabilities',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                combined=np.hstack([true_rgb,pred_rgb])
                out.write(combined)
                total_frames+=1
                if total_frames>=1000:
                    break
            if total_frames>=1000:
                break
        out.release()
        tqdm.write(f"Two-panel video saved: {os.path.basename(output_path)} ({total_frames} frames)")
    elif sample_bernoulli:
        tqdm.write("Creating video with Bernoulli-sampled predictions")
        create_comparison_video(
            actual=batch_np,
            predicted=predictions[rand_indices].cpu().numpy(),
            sampled_predictions=sampled_predictions[rand_indices].cpu().numpy(),
            output_path=output_path,
            num_frames=num_frames,
            fps=1
        )
    else:
        tqdm.write("Creating video with binary inputs and probability predictions")
        create_color_coded_comparison_video(
            actual=batch_np,
            predicted=predictions[rand_indices].cpu().numpy(),
            sampled_predictions=sampled_predictions[rand_indices].cpu().numpy(),
            output_path=output_path,
            num_frames=num_frames,
            fps=1,
            threshold_left_panels=True
        )


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


def create_autoencoder_comparison_video(model, dataset, output_path, num_frames=100, fps=10):
    """
    Create a video comparing original and reconstructed samples from an autoencoder model.
    
    Args:
        model: The autoencoder model to use for reconstruction
        dataset: The dataset to sample from
        output_path: Path to save the output video
        num_frames: Number of frames to include in the video
        fps: Frames per second for the video
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.animation import FFMpegWriter
        import numpy as np
        import torch
        from tqdm import tqdm
        
        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1 = axes[0]
        ax2 = axes[1] 
        ax3 = axes[2]
        
        # Set up the writer
        writer = FFMpegWriter(fps=fps)
        
        # Move model to evaluation mode
        model.eval()
        
        # Get device
        device = next(model.parameters()).device
        
        # Create the video
        with writer.saving(fig, output_path, dpi=100):
            for i in tqdm(range(min(num_frames, len(dataset))), desc="Creating video"):
                # Get a sample (shape should be [256, 128])
                sample = dataset[i].unsqueeze(0).to(device)  # Add batch dimension: [1, 256, 128]
                
                # Get reconstruction - let the model handle input format
                with torch.no_grad():
                    reconstruction = model(sample)  # Model handles dimension conversion
                    reconstruction = torch.bernoulli(reconstruction)
                
                # Convert to numpy for plotting
                original = sample.cpu().squeeze(0).numpy()  # Should be [256, 128]
                reconstructed = reconstruction.cpu().squeeze(0).numpy()  # Should be [256, 128]
                
                # Ensure proper shape for visualization (in case they're flattened)
                if original.ndim == 1:
                    original = original.reshape(256, 128)
                if reconstructed.ndim == 1:
                    reconstructed = reconstructed.reshape(256, 128)
                
                # Debug: Print data statistics for first frame
                if i == 0:
                    print(f"Original shape: {original.shape}, min: {original.min():.4f}, max: {original.max():.4f}, mean: {original.mean():.4f}")
                    print(f"Reconstructed shape: {reconstructed.shape}, min: {reconstructed.min():.4f}, max: {reconstructed.max():.4f}, mean: {reconstructed.mean():.4f}")
                    print(f"Original non-zero count: {np.count_nonzero(original)}")
                    print(f"Reconstructed non-zero count: {np.count_nonzero(reconstructed)}")
                
                # Auto-adjust the visualization range based on actual data
                orig_min, orig_max = original.min(), original.max()
                recon_min, recon_max = reconstructed.min(), reconstructed.max()
                vmin = min(orig_min, recon_min)
                vmax = max(orig_max, recon_max)
                
                # If all values are the same, add some range for visualization
                if vmax == vmin:
                    vmax = vmin + 1e-6
                
                # Create difference map
                difference = np.abs(original - reconstructed)
                
                # Clear axes
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                # Plot original, reconstruction, and difference
                im1 = ax1.imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
                ax1.set_title(f'Original (Frame {i+1})')
                ax1.axis('off')
                
                im2 = ax2.imshow(reconstructed, cmap='gray', vmin=vmin, vmax=vmax)
                ax2.set_title('Reconstructed')
                ax2.axis('off')
                
                diff_max = difference.max() if difference.max() > 0 else 1e-6
                im3 = ax3.imshow(difference, cmap='hot', vmin=0, vmax=diff_max)
                ax3.set_title('Difference')
                ax3.axis('off')
                
                # Add statistics as text
                mse = np.mean((original - reconstructed) ** 2)
                mae = np.mean(np.abs(original - reconstructed))
                fig.suptitle(f'MSE: {mse:.4f}, MAE: {mae:.4f}', fontsize=12)
                
                # Add to video
                writer.grab_frame()
                
        plt.close()
        print(f"Video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        import traceback
        traceback.print_exc() 