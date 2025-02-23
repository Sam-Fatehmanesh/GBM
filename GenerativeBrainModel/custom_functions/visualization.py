import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def create_comparison_video(model, test_dataset, output_path, max_frames=100, fps=1):
    """Create video comparing original and reconstructed grids"""
    try:
        device = next(model.parameters()).device
        model.eval()
        
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
                            (scaled_width * 2, scaled_height), 
                            isColor=True)
        
        if not out.isOpened():
            print(f"Failed to create video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (scaled_width * 2, scaled_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer with both codecs. Skipping video creation.")
                return
        
        with torch.no_grad():
            for i in range(min(max_frames, len(test_dataset))):
                # Get original grid
                grid = test_dataset[i].to(device)
                grid = grid.unsqueeze(0)
                
                # Get reconstruction
                recon = model(grid)
                
                # Convert to numpy arrays
                original = grid[0].cpu().numpy()
                reconstructed = recon[0].cpu().numpy()
                
                # Convert to uint8 images
                orig_img = (original * 255).astype(np.uint8)
                recon_img = (reconstructed * 255).astype(np.uint8)
                
                # Scale up images
                orig_img = cv2.resize(orig_img, (scaled_width, scaled_height), 
                                   interpolation=cv2.INTER_NEAREST)
                recon_img = cv2.resize(recon_img, (scaled_width, scaled_height), 
                                     interpolation=cv2.INTER_NEAREST)
                
                # Convert to RGB
                orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                recon_rgb = cv2.cvtColor(recon_img, cv2.COLOR_GRAY2BGR)
                
                # Add text labels
                cv2.putText(orig_rgb, 'Original', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(recon_rgb, 'Reconstructed', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combine horizontally
                combined = np.hstack([orig_rgb, recon_rgb])
                out.write(combined)
        
        out.release()
        print(f"\nComparison video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error creating comparison video: {str(e)}")

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