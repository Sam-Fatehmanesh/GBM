# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import cv2
from scipy.interpolate import interp1d

from BrainSimulator.models.spike_set_transformer_vae import SpikeSetTransformerVAE
from BrainSimulator.datasets.spike_dataset import SpikeDataset
from BrainSimulator.custom_functions.adaptive_sliced_wasserstein import AdaptiveSlicedWasserstein

# Initialize ASW distance
asw_distance = AdaptiveSlicedWasserstein(N0=10, s=5, epsilon=0.01, max_projections=100, num_quantile_points=300)

def chamfer_distance(x, y):
    """Compute Chamfer distance between two sets of points supplied in batch.
    Fully vectorized implementation that handles variable-sized point sets.
    Uses sentinel value -1 to identify padded points.
    
    Args:
        x: (B, N, D) tensor of points
        y: (B, M, D) tensor of points
        
    Returns:
        (B,) tensor of Chamfer distances per sample
    """
    # Create masks for valid (non-padded) points using sentinel value
    x_mask = (x[..., 0] != -1) & (x[..., 1] != -1)  # (B, N)
    y_mask = (y[..., 0] != -1) & (y[..., 1] != -1)  # (B, M)
    
    # Count valid points per sample
    x_lengths = x_mask.sum(dim=1)  # (B,)
    y_lengths = y_mask.sum(dim=1)  # (B,)
    
    # Handle empty sets
    valid_samples = (x_lengths > 0) & (y_lengths > 0)
    if not torch.any(valid_samples):
        return torch.zeros(x.size(0), device=x.device)
    
    # Compute squared norms
    x_norm = (x ** 2).sum(2)  # (B, N)
    y_norm = (y ** 2).sum(2)  # (B, M)
    
    # Compute cross term efficiently using batch matrix multiplication
    xy = torch.bmm(x, y.transpose(1, 2))  # (B, N, M)
    
    # Compute pairwise distances using broadcasting
    dist_matrix = x_norm.unsqueeze(2) + y_norm.unsqueeze(1) - 2 * xy  # (B, N, M)
    
    # Apply masks to ignore padded points
    dist_matrix = dist_matrix.masked_fill(~x_mask.unsqueeze(2), float('inf'))
    dist_matrix = dist_matrix.masked_fill(~y_mask.unsqueeze(1), float('inf'))
    
    # Compute minimal distances for valid samples
    min_dist_x = torch.min(dist_matrix, dim=2)[0]  # (B, N)
    min_dist_y = torch.min(dist_matrix, dim=1)[0]  # (B, M)
    
    # Safely compute sum by zeroing out invalid points before summing
    loss_x = torch.where(x_mask, min_dist_x, torch.zeros_like(min_dist_x)).sum(dim=1) / (x_lengths + 1e-8)
    loss_y = torch.where(y_mask, min_dist_y, torch.zeros_like(min_dist_y)).sum(dim=1) / (y_lengths + 1e-8)
    
    # Combine both directions
    chamfer_dist = loss_x + loss_y  # (B,)
    
    # Zero out invalid samples
    chamfer_dist = chamfer_dist.masked_fill(~valid_samples, 0)
    
    return chamfer_dist

def collate_fn(batch):
    """Custom collate function to handle variable-sized point sets.
    Each item in batch is a tuple (points, valid_mask) where:
        points: tensor of shape (num_points, 2)
        valid_mask: tensor of shape (num_points,) indicating valid points
    """
    # Unzip the batch into points and masks
    points, masks = zip(*batch)
    
    # Find maximum number of points in the batch
    max_points = max(p.size(0) for p in points)
    
    # Create padded batch tensors with sentinel value -1
    batch_size = len(points)
    padded_points = torch.full((batch_size, max_points, 2), fill_value=-1.0)
    padded_masks = torch.zeros(batch_size, max_points, dtype=torch.bool)
    
    # Fill in the actual points and masks
    for i, (p, m) in enumerate(zip(points, masks)):
        if p.size(0) > 0:  # Only if there are points
            padded_points[i, :p.size(0)] = p
            padded_masks[i, :p.size(0)] = m
    
    return padded_points, padded_masks

def create_experiment_dir():
    """Create and return path to new experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

def plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader):
    """Plot and save the training and test losses"""
    plt.figure(figsize=(12, 8))
    
    if len(train_losses) > 0:
        # Plot training losses
        iterations = range(1, len(train_losses) + 1)
        plt.plot(iterations, train_losses, label='Train Total Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, recon_losses, label='Train Recon Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, kl_losses, label='Train KL Loss', linewidth=2, alpha=0.7)
        
        # Plot test losses with linear interpolation
        if len(test_losses) > 1:
            steps_per_epoch = len(train_loader)
            test_iterations = [(i + 1) * steps_per_epoch for i in range(len(test_losses))]
            
            f_total = interp1d(test_iterations, [x['test_loss'] for x in test_losses], kind='linear')
            f_recon = interp1d(test_iterations, [x['test_recon_loss'] for x in test_losses], kind='linear')
            f_kl = interp1d(test_iterations, [x['test_kl_loss'] for x in test_losses], kind='linear')
            
            x_smooth = np.linspace(test_iterations[0], test_iterations[-1], len(train_losses))
            
            plt.plot(x_smooth, f_total(x_smooth), '--', label='Test Total Loss', linewidth=2, alpha=0.7)
            plt.plot(x_smooth, f_recon(x_smooth), '--', label='Test Recon Loss', linewidth=2, alpha=0.7)
            plt.plot(x_smooth, f_kl(x_smooth), '--', label='Test KL Loss', linewidth=2, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Set Transformer VAE Training and Test Losses')
        plt.legend()
        plt.grid(True)
        plt.xlim(1, len(train_losses))
        
        plt.savefig(os.path.join(exp_dir, "plots", "training_loss.png"))
        plt.close()

def create_comparison_video(model, test_dataset, output_path, max_frames=100, fps=1):
    """Create video comparing original and reconstructed point sets"""
    try:
        device = next(model.parameters()).device
        model.eval()
        
        # Set up video parameters
        width = 500
        height = 500
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use H264 codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height), isColor=True)
        
        if not out.isOpened():
            print(f"Failed to create video writer. Trying alternative codec...")
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height), isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer with both codecs. Skipping video creation.")
                return
        
        with torch.no_grad():
            for i in range(min(max_frames, len(test_dataset))):
                # Get original point set and ensure proper dimensions
                points, valid_mask = test_dataset[i]
                points = points.to(device)
                valid_mask = valid_mask.to(device)
                
                if points.dim() == 2:
                    points = points.unsqueeze(0)
                    valid_mask = valid_mask.unsqueeze(0)
                
                # Get reconstruction
                recon, _ = model(points)
                
                # Convert to numpy arrays
                original = points[0].cpu().numpy()
                original_mask = valid_mask[0].cpu().numpy()
                reconstructed = recon[0].cpu().numpy()
                
                # Create frames
                orig_frame = np.zeros((height, width, 3), dtype=np.uint8)
                recon_frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Plot original points in white
                if np.any(original_mask):
                    original_valid = original[original_mask]
                    original_norm = (original_valid - original_valid.min(axis=0)) / (original_valid.max(axis=0) - original_valid.min(axis=0) + 1e-8)
                    for p in original_norm:
                        x, y = int(p[0] * (width-1)), int(p[1] * (height-1))
                        cv2.circle(orig_frame, (x, y), 2, (255, 255, 255), -1)
                
                # Plot reconstructed points in white
                recon_mask = ~np.all(reconstructed == 0, axis=1)
                if np.any(recon_mask):
                    reconstructed_valid = reconstructed[recon_mask]
                    reconstructed_norm = (reconstructed_valid - reconstructed_valid.min(axis=0)) / (reconstructed_valid.max(axis=0) - reconstructed_valid.min(axis=0) + 1e-8)
                    for p in reconstructed_norm:
                        x, y = int(p[0] * (width-1)), int(p[1] * (height-1))
                        cv2.circle(recon_frame, (x, y), 2, (255, 255, 255), -1)
                
                # Add labels
                cv2.putText(orig_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(recon_frame, "Reconstructed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combine frames
                comparison = np.hstack([orig_frame, recon_frame])
                
                # Write frame
                try:
                    out.write(comparison)
                except Exception as e:
                    print(f"Error writing video frame: {e}")
                    break
        
        out.release()
        
    except Exception as e:
        print(f"Error in video creation: {e}")
        print("Skipping video creation and continuing with training...")

def evaluate_test_loss(model, test_loader, device, params):
    """Evaluate model on test set and return average losses"""
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for points, valid_mask in test_loader:
            points = points.to(device)
            valid_mask = valid_mask.to(device)
            
            # Forward pass
            recon, (mu, logvar) = model(points)
            
            # Compute losses
            recon_loss = chamfer_distance(recon, points).mean() * 10.0  # Scale reconstruction loss
            kl_loss = model.compute_kl_loss(mu, logvar)
            
            # Combine losses
            loss = recon_loss + params['beta'] * kl_loss
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            n_batches += 1
    
    return {
        'test_loss': total_loss / n_batches,
        'test_recon_loss': total_recon_loss / n_batches,
        'test_kl_loss': total_kl_loss / n_batches,
    }

def main():
    try:
        # Parameters
        params = {
            'batch_size': 200,
            'num_epochs': 8,
            'learning_rate': 1e-3,
            'hidden_dim': 16,
            'latent_dim': 1024,
            'num_heads': 4,
            'num_inds': 16,
            'beta': 0.0,    # KL divergence weight
        }
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        tqdm.write(f"Saving experiment results to: {exp_dir}")
        
        # Find all processed spike data files
        processed_dir = "processed_spikes"
        spike_files = []
        for file in os.listdir(processed_dir):
            if file.endswith("_processed.h5"):
                spike_files.append(os.path.join(processed_dir, file))
        
        if not spike_files:
            raise ValueError("No processed spike data files found!")
        
        tqdm.write(f"Found {len(spike_files)} subject data files: {spike_files}")
        
        # Create combined datasets
        train_dataset = ConcatDataset([SpikeDataset(f, split='train') for f in spike_files])
        test_dataset = ConcatDataset([SpikeDataset(f, split='test') for f in spike_files])
        
        tqdm.write(f"Total training samples: {len(train_dataset)}")
        tqdm.write(f"Total test samples: {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Create model
        model = SpikeSetTransformerVAE(
            input_dim=2,
            hidden_dim=params['hidden_dim'],
            latent_dim=params['latent_dim'],
            num_heads=params['num_heads'],
            num_inds=params['num_inds']
        )
        
        # Save model architecture and parameters
        with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
            f.write("Complete Model Architecture:\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model))
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("Model Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            # Add statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\nModel Statistics:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Input dimension: 2\n")  # Changed to 2D
            f.write(f"Latent dimension: {params['latent_dim']}\n")
        
        # Move to GPU if available
        device = "cuda:1"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        
        # Ensure model parameters are contiguous in memory
        model = model.to(device)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()
        
        # Create optimizer with gradient clipping
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        tqdm.write("Starting training...")
        train_losses = []
        recon_losses = []
        kl_losses = []
        test_losses = []
        best_test_loss = float('inf')
        
        for epoch in range(1, params['num_epochs'] + 1):
            # Create comparison video at start of epoch
            video_path = os.path.join(exp_dir, "videos", f"reconstruction_epoch_{epoch}_start.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            create_comparison_video(model, test_dataset, video_path)
            
            model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for batch_idx, (points, valid_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                points = points.to(device)
                valid_mask = valid_mask.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon, (mu, logvar) = model(points)
                
                # Compute losses
                recon_loss = chamfer_distance(recon, points).mean() * 10.0
                kl_loss = model.compute_kl_loss(mu, logvar) * params['beta']
                loss = recon_loss + kl_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                train_losses.append(loss.item())
                recon_losses.append(recon_loss.item())
                kl_losses.append(kl_loss.item())
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                if batch_idx % 10 == 0:
                    # Get average true count for logging

                    
                    tqdm.write(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]:')
                    tqdm.write(f'  Total Loss: {loss.item():.4f}')
                    tqdm.write(f'  Reconstruction Loss: {recon_loss.item():.4f}')
                    tqdm.write(f'  KL Loss: {kl_loss.item():.4f}')

                    

                    
                    plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader)
            
            # Evaluate on test set
            test_loss_dict = evaluate_test_loss(model, test_loader, device, params)
            test_losses.append(test_loss_dict)
            
            # Save losses
            train_loss_dict = {
                'iteration': list(range(1, len(train_losses) + 1)),
                'total_loss': train_losses,
                'recon_loss': recon_losses,
                'kl_loss': kl_losses,
            }
            save_losses_to_csv(train_loss_dict, os.path.join(exp_dir, "train_losses.csv"))
            
            test_loss_dict_for_csv = {
                'epoch': list(range(1, len(test_losses) + 1)),
                'total_loss': [x['test_loss'] for x in test_losses],
                'recon_loss': [x['test_recon_loss'] for x in test_losses],
                'kl_loss': [x['test_kl_loss'] for x in test_losses],
            }
            save_losses_to_csv(test_loss_dict_for_csv, os.path.join(exp_dir, "test_losses.csv"))
            
            # Plot losses
            plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader)
            
            # Create comparison video at end of epoch
            video_path = os.path.join(exp_dir, "videos", f"reconstruction_epoch_{epoch}_end.mp4")
            create_comparison_video(model, test_dataset, video_path)
            
            # Save checkpoint if best model
            if test_loss_dict['test_loss'] < best_test_loss:
                best_test_loss = test_loss_dict['test_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'recon_losses': recon_losses,
                    'kl_losses': kl_losses,
                    'test_losses': test_losses,
                    'params': params
                }, os.path.join(exp_dir, "checkpoints", "best_model.pt"))
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'recon_losses': recon_losses,
                    'kl_losses': kl_losses,
                    'test_losses': test_losses,
                    'params': params
                }, os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"))

    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user. Saving checkpoint...")
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'recon_losses': recon_losses,
            'kl_losses': kl_losses,
            'test_losses': test_losses,
            'params': params
        }, os.path.join(exp_dir, "checkpoints", "interrupted_checkpoint.pt"))
        tqdm.write("Checkpoint saved successfully.")
    except Exception as e:
        tqdm.write(f"\nAn error occurred: {str(e)}")
        raise e
    finally:
        # Clean up
        if 'train_loader' in locals():
            del train_loader
        if 'test_loader' in locals():
            del test_loader
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 