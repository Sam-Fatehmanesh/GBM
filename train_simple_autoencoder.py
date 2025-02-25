# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import cv2
import h5py

class GridSpikeDataset(Dataset):
    def __init__(self, h5_file, split='train', train_ratio=0.95):
        """Dataset for spike data, converting each z-plane at each timepoint to a 256x128 binary grid.
        
        Args:
            h5_file: Path to the processed spike data h5 file
            split: 'train' or 'test'
            train_ratio: Ratio of data to use for training
        """
        # Load spike data and cell positions
        with h5py.File(h5_file, 'r') as f:
            self.spikes = f['spikes'][:]  # shape: (n_timepoints, n_cells)
            self.cell_positions = f['cell_positions'][:]  # shape: (n_cells, 3)
            
        # Get unique z values (rounded to handle floating point precision)
        self.z_values = np.unique(np.round(self.cell_positions[:, 2], decimals=3))
        self.num_z = len(self.z_values)
        self.num_timepoints = self.spikes.shape[0]
        
        # Pre-compute z-plane masks
        self.z_masks = {}
        cells_per_z = {}
        for z_idx, z_level in enumerate(self.z_values):
            z_mask = (np.round(self.cell_positions[:, 2], decimals=3) == z_level)
            self.z_masks[z_idx] = z_mask
            cells_per_z[z_level] = np.sum(z_mask)
            
        # Normalize cell positions to [0, 1]
        self.cell_positions = (self.cell_positions - self.cell_positions.min(axis=0)) / \
                            (self.cell_positions.max(axis=0) - self.cell_positions.min(axis=0))
        
        # Convert to grid indices
        self.cell_x = np.floor(self.cell_positions[:, 0] * 255).astype(np.int32)  # 0-255
        self.cell_y = np.floor(self.cell_positions[:, 1] * 127).astype(np.int32)  # 0-127
        
        # Create indices for all possible (timepoint, z) combinations
        all_indices = [(t, z) for t in range(self.num_timepoints) 
                      for z in range(self.num_z)]
        
        # Split into train/test
        np.random.seed(42)
        np.random.shuffle(all_indices)
        split_idx = int(len(all_indices) * train_ratio)
        
        if split == 'train':
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]
            
        print(f"\nDataset {h5_file}:")
        print(f"Total z-planes: {self.num_z}")
        print(f"Cells per z-plane: {[cells_per_z[z] for z in self.z_values]}")
        print(f"Total samples ({split}): {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Returns:
            grid: Binary tensor of shape (256, 128) representing active cells in the z-plane
        """
        # Get timepoint and z-level from indices
        timepoint, z_idx = self.indices[idx]
        z_level = self.z_values[z_idx]
        
        # Get mask for cells in this z-plane
        z_mask = self.z_masks[z_idx]
        
        # Get spikes for this timepoint
        spikes_t = self.spikes[timepoint]
        
        # Create empty grid
        grid = np.zeros((256, 128), dtype=np.float32)
        
        # Consider a cell active if it has any non-zero spike value
        active_mask = (np.abs(spikes_t) > 1e-6) & z_mask
        
        # Set active cells to 1 in the grid
        grid[self.cell_x[active_mask], self.cell_y[active_mask]] = 1.0
        
        return torch.FloatTensor(grid)

class SyntheticSpikeDataset(Dataset):
    def __init__(self, num_samples=10000, grid_size=(256, 128)):
        """Dataset that generates synthetic spike data following the empirical distribution.
        
        Args:
            num_samples: Number of synthetic samples to generate
            grid_size: Size of the binary grid (height, width)
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        # Distribution parameters from empirical analysis
        self.mean_spikes = 73.42
        self.std_spikes = 98.25
        self.min_spikes = 0
        self.max_spikes = 2000  # Using 99th percentile instead of max to avoid outliers
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random binary grid with number of spikes following empirical distribution."""
        # Sample number of spikes from truncated normal distribution
        num_spikes = int(np.clip(
            np.random.normal(self.mean_spikes, self.std_spikes),
            self.min_spikes,
            self.max_spikes
        ))
        
        # Create empty grid
        grid = np.zeros(self.grid_size, dtype=np.float32)
        
        if num_spikes > 0:
            # Randomly select positions for spikes
            total_cells = self.grid_size[0] * self.grid_size[1]
            spike_indices = np.random.choice(total_cells, size=num_spikes, replace=False)
            
            # Convert to 2D indices
            spike_rows = spike_indices // self.grid_size[1]
            spike_cols = spike_indices % self.grid_size[1]
            
            # Set spikes
            grid[spike_rows, spike_cols] = 1.0
        
        return torch.FloatTensor(grid)

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size=32768, hidden_size=1024):
        super(SimpleAutoencoder, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = torch.sigmoid(self.decoder(encoded))
        
        # Reshape back to grid
        decoded = decoded.view(batch_size, 256, 128)
        
        return decoded

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

def create_experiment_dir():
    """Create a timestamped experiment directory with all necessary subdirectories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', timestamp)
    
    # Create main experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    return exp_dir
    
def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

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

def main():
    try:
        # Parameters
        params = {
            'batch_size': 64,
            'num_epochs': 1,
            'learning_rate': 1e-3,
            'hidden_size': 1024,
            'train_samples': 64*100000,  # Number of training samples to generate
        }
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        tqdm.write(f"Saving experiment results to: {exp_dir}")
        
        # Create synthetic training dataset
        train_dataset = SyntheticSpikeDataset(num_samples=params['train_samples'])
        
        # Create real test dataset from spike data
        processed_dir = "processed_spikes"
        spike_files = []
        for file in os.listdir(processed_dir):
            if file.endswith("_processed.h5"):
                spike_files.append(os.path.join(processed_dir, file))
        
        if not spike_files:
            raise ValueError("No processed spike data files found!")
        
        # Use real data for testing (5% of the data)
        test_dataset = ConcatDataset([GridSpikeDataset(f, split='test', train_ratio=0.95) for f in spike_files])
        
        tqdm.write(f"Created datasets:")
        tqdm.write(f"Training samples (synthetic): {len(train_dataset)}")
        tqdm.write(f"Test samples (real, 5%): {len(test_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        tqdm.write(f"Number of batches in train_loader: {len(train_loader)}")
        tqdm.write(f"Number of batches in test_loader: {len(test_loader)}")
        
        # Create model
        model = SimpleAutoencoder(
            input_size=256*128,
            hidden_size=params['hidden_size']
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
            f.write(f"Input size: {256*128}\n")
            f.write(f"Hidden size: {params['hidden_size']}\n")
        
        # Move to GPU if available
        device = "cuda:0"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        model = model.to(device)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        tqdm.write("Starting training...")
        train_losses = []
        test_losses = []
        raw_batch_losses = []  # Track individual batch losses
        best_test_loss = float('inf')
        
        for epoch in range(params['num_epochs']):
            # Training
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params["num_epochs"]}')
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                reconstructed = model(batch)
                loss = F.binary_cross_entropy(reconstructed, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                raw_loss = loss.item()
                raw_batch_losses.append(raw_loss)
                epoch_loss += raw_loss
                current_avg_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({'raw_loss': raw_loss, 'avg_loss': current_avg_loss})
                
                # Save intermediate diagnostics
                if batch_idx % 256 == 0:
                    # Log current state
                    log_file = os.path.join(exp_dir, 'logs', 'training_log.txt')
                    with open(log_file, 'a') as f:
                        f.write(f"Epoch {epoch+1}, Batch {batch_idx}, Raw Loss: {raw_loss:.6f}, Avg Loss: {current_avg_loss:.6f}\n")
                    
                    # Update plots with current progress
                    update_loss_plot(train_losses + [current_avg_loss], 
                                   test_losses + [test_losses[-1] if test_losses else 0],
                                   raw_batch_losses,
                                   os.path.join(exp_dir, 'plots', 'loss_plot.png'))
            
            # Append actual epoch average loss
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Testing
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    reconstructed = model(batch)
                    loss = F.binary_cross_entropy(reconstructed, batch)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # Create comparison video after each epoch
            video_path = os.path.join(exp_dir, 'videos', f'reconstruction_epoch_{epoch+1:03d}.mp4')
            create_comparison_video(model, test_dataset, video_path)
            
            # Save best model
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'raw_batch_losses': raw_batch_losses,
                    'params': params,
                    'best_test_loss': best_test_loss
                }, os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            
            # Update plots and save losses
            update_loss_plot(train_losses, test_losses, raw_batch_losses,
                           os.path.join(exp_dir, 'plots', 'loss_plot.png'))
            save_losses_to_csv(
                {'epoch': list(range(1, len(train_losses) + 1)),
                 'train_loss': train_losses,
                 'test_loss': test_losses},
                os.path.join(exp_dir, 'logs', 'losses.csv')
            )
            
            # Save raw batch losses separately
            if raw_batch_losses:
                save_losses_to_csv(
                    {'batch': list(range(1, len(raw_batch_losses) + 1)),
                     'raw_loss': raw_batch_losses},
                    os.path.join(exp_dir, 'logs', 'raw_batch_losses.csv')
                )
            
            # Save current reconstruction samples
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                with torch.no_grad():
                    # Get a batch of test samples
                    test_batch = next(iter(test_loader)).to(device)
                    reconstructed = model(test_batch)
                    
                    # Save a few examples
                    for i in range(min(5, test_batch.size(0))):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                        ax1.imshow(test_batch[i].cpu().numpy(), cmap='gray')
                        ax1.set_title('Original')
                        ax2.imshow(reconstructed[i].cpu().numpy(), cmap='gray')
                        ax2.set_title('Reconstructed')
                        plt.savefig(os.path.join(exp_dir, 'plots', f'reconstruction_epoch_{epoch+1:03d}_sample_{i+1}.png'))
                        plt.close()
            
            tqdm.write(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                      f"Test Loss = {avg_test_loss:.6f}")
            
        tqdm.write("Training complete!")
        
        # Save final model and diagnostics
        torch.save({
            'epoch': params['num_epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'raw_batch_losses': raw_batch_losses,
            'params': params,
            'final_train_loss': avg_train_loss,
            'final_test_loss': avg_test_loss
        }, os.path.join(exp_dir, 'checkpoints', 'final_model.pt'))
        
        # Create final comparison video
        video_path = os.path.join(exp_dir, 'videos', 'final_reconstruction.mp4')
        create_comparison_video(model, test_dataset, video_path, max_frames=200)  # More frames for final video
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 