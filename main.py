import h5py
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from BrainSimulator.models.vae import VariationalAutoEncoder

os.environ["QT_QPA_PLATFORM"] = "offscreen"

class BrainDataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        # Get total number of frames and dimensions
        self.frames_per_file = None
        self.total_frames = 0
        self.file_frame_ranges = []
        self.height = None
        self.width = None
        
        # Get dimensions from first file
        with h5py.File(h5_files[0], 'r') as f:
            data = f['default']
            self.height, self.width = data.shape[1:]
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                if self.frames_per_file is None:
                    self.frames_per_file = f['default'].shape[0]
                self.file_frame_ranges.append((self.total_frames, self.total_frames + self.frames_per_file))
                self.total_frames += self.frames_per_file
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = 0
        while file_idx < len(self.file_frame_ranges) and idx >= self.file_frame_ranges[file_idx][1]:
            file_idx += 1
        
        # Get the frame from the appropriate file
        with h5py.File(self.h5_files[file_idx], 'r') as f:
            frame_idx = idx - self.file_frame_ranges[file_idx][0]
            frame = f['default'][frame_idx].astype(np.float32)
            # Normalize to [0, 1]
            # frame = (frame - frame.min()) / (frame.max() - frame.min())
            # Add channel dimension
            frame = frame[np.newaxis, :, :]
            return torch.from_numpy(frame)

def create_experiment_dir():
    """Create and return path to new experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def plot_losses(train_losses, recon_losses, kl_losses, exp_dir):
    """Plot and save the training losses"""
    plt.figure(figsize=(12, 8))
    
    # Only normalize if we have data
    if len(train_losses) > 0:
        # train_losses_norm = normalize_array(train_losses)
        # recon_losses_norm = normalize_array(recon_losses)
        # kl_losses_norm = normalize_array(kl_losses)
        
        # Plot normalized loss components
        iterations = range(1, len(train_losses) + 1)
        plt.plot(iterations, train_losses, label='Total Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, recon_losses, label='Reconstruction Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, kl_losses, label='KL Loss', linewidth=2, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('VAE Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Set fixed axis limits for normalized values
        plt.xlim(1, len(train_losses))
        # plt.ylim(-0.1, 1.1)  # Slight padding around 0-1 range
        
        # Add actual values in text box
        textstr = f'Current Values:\nTotal Loss: {train_losses[-1]:.4f}\nRecon Loss: {recon_losses[-1]:.4f}\nKL Loss: {kl_losses[-1]:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    # Save the plot
    plt.savefig(os.path.join(exp_dir, "plots", "training_loss.png"))
    plt.close()

def normalize_frame(frame):
    """Normalize frame to 0-255 range"""
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max == frame_min:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)

def create_comparison_video(model, test_dataset, output_path, fps=1, max_frames=None):
    """Create a video comparing original and reconstructed frames"""
    device = next(model.parameters()).device
    
    # Get dimensions from first frame
    first_frame = test_dataset[0].numpy()[0]
    height, width = first_frame.shape
    
    # Initialize video writer for side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Determine how many frames to process
    total_frames = len(test_dataset)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    model.eval()
    print(f"Creating comparison video with first {total_frames} frames...")
    with torch.no_grad():
        for i in tqdm(range(total_frames)):
            # Get original frame
            frame = test_dataset[i].unsqueeze(0).to(device)
            
            # Get reconstruction
            reconstruction, _, _ = model(frame)
            
            # Convert to numpy arrays and normalize to uint8
            original = normalize_frame(frame[0, 0].cpu().numpy())
            reconstructed = normalize_frame(reconstruction[0, 0].cpu().numpy())

            # original = (frame[0, 0].cpu().numpy() * 255).astype(np.uint8)
            # reconstructed = (reconstruction[0, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            comparison = np.hstack([original, reconstructed])
            
            # Convert to RGB
            comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            out.write(comparison_rgb)
    
    out.release()
    print(f"Video saved as {output_path}")

def train_vae(train_loader, model, optimizer, device, epoch, train_losses, recon_losses, kl_losses, exp_dir, test_dataset):
    model.train()
    
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, latent_sample, latent_dist = model(data)
        
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
        # Reconstruction loss (MSE)z
        # recon_loss = F.mse_loss(recon_batch, data, reduction='mean')
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + torch.log(latent_dist + 1e-10) - latent_dist)
        
        # Total loss with beta-VAE weighting
        # Reduce beta since KL loss is much larger than reconstruction loss
        beta = 0.00#1  # Reduced from 0.1 to better balance the losses
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()

        # Record losses
        train_losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
        
        # Plot every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]:')
            print(f'  Total Loss: {loss.item():.4f}')
            print(f'  Reconstruction Loss: {recon_loss.item():.4f}')
            print(f'  KL Loss: {kl_loss.item():.4f}')
            plot_losses(train_losses, recon_losses, kl_losses, exp_dir)
    
    # Generate comparison video for this epoch using first 10% of test data
    video_path = os.path.join(exp_dir, "videos", f"reconstruction_epoch_{epoch}.mp4")
    create_comparison_video(model, test_dataset, video_path, max_frames=len(test_dataset)//10)

def main():
    # Get data dimensions from first file
    first_file = glob(os.path.join("input", "volume*.h5"))[0]
    with h5py.File(first_file, 'r') as f:
        data = f['default']
        height, width = data.shape[1:]
    
    # Parameters
    model_params = {
        'image_height': height,
        'image_width': width,
        'n_distributions': 512,  # Number of categorical distributions
        'n_categories': 8,    # Number of categories per distribution
    }
    
    training_params = {
        'batch_size': 32,
        'epochs': 75,
        'learning_rate': 1e-3,
        'train_split': 0.9,  # 90% for training
    }
    
    # Create experiment directory
    exp_dir = create_experiment_dir()
    print(f"Saving experiment results to: {exp_dir}")
    
    # Initialize model early to save architecture
    model = VariationalAutoEncoder(
        model_params['image_height'],
        model_params['image_width'],
        model_params['n_distributions'],
        model_params['n_categories']
    )
    
    # Save complete model architecture
    with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
        f.write("Complete Model Architecture:\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(model))
        f.write("\n\n" + "=" * 50 + "\n\n")
        f.write("Model Parameters:\n")
        for key, value in model_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nTraining Parameters:\n")
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
        
        # Add some useful statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"\nModel Statistics:\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Input shape: ({1}, {height}, {width})\n")
        f.write(f"Latent space size: {model_params['n_distributions'] * model_params['n_categories']}\n")
    
    # Get and sort all h5 files
    h5_files = sorted(glob(os.path.join("data", "volume*.h5")))
    n_files = len(h5_files)
    n_train = int(n_files * training_params['train_split'])
    
    # Split into train and test sets
    train_files = h5_files[:n_train]
    test_files = h5_files[n_train:]
    
    # Create datasets and dataloaders
    train_dataset = BrainDataset(train_files)
    test_dataset = BrainDataset(test_files)
    
    # Test video creation before training
    print("Testing video creation before training...")
    test_video_path = os.path.join(exp_dir, "videos", "test_video.mp4")
    os.makedirs(os.path.dirname(test_video_path), exist_ok=True)
    
    # Create a simple test video with original frames only
    first_frame = test_dataset[0].numpy()[0]
    height, width = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 1, (width * 2, height))
    
    n_test_frames = min(10, len(test_dataset))  # Use 10 frames for test
    print(f"Creating test video with {n_test_frames} frames...")
    
    for i in tqdm(range(n_test_frames)):
        frame = test_dataset[i].numpy()[0]
        #frame_uint8 = (frame * 255).astype(np.uint8)
        frame_uint8 = normalize_frame(frame)
        # Create side-by-side comparison with the same frame
        comparison = np.hstack([frame_uint8, frame_uint8])
        comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        out.write(comparison_rgb)
    
    out.release()
    print(f"Test video saved as {test_video_path}")
    
    if not os.path.exists(test_video_path) or os.path.getsize(test_video_path) == 0:
        raise Exception("Failed to create test video. Please check video writing capabilities.")
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    
    # Move model to device after architecture is saved
    device = torch.device("cuda:1") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # Training loop
    print("Starting training...")
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(1, training_params['epochs'] + 1):
        train_vae(train_loader, model, optimizer, device, epoch, train_losses, recon_losses, kl_losses, exp_dir, test_dataset)
        
        # Save checkpoint every 10 epochs
        if epoch % 4 == 0:
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"vae_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'recon_losses': recon_losses,
                'kl_losses': kl_losses,
            }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(exp_dir, "checkpoints", "vae_final.pt")
    torch.save(model.state_dict(), final_model_path)

if __name__ == "__main__":
    main()
