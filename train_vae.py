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
import pandas as pd
from scipy.interpolate import interp1d
from BrainSimulator.models.vae import VariationalAutoEncoder

os.environ["QT_QPA_PLATFORM"] = "offscreen"

class BrainDataset(Dataset):
    def __init__(self, h5_files, frame_indices=None, cache_size=4):
        """
        Args:
            h5_files: List of h5 file paths
            frame_indices: Optional list of (file_idx, frame_idx) tuples specifying which frames to use
            cache_size: Number of h5 files to keep in memory (default: 4)
        """
        self.h5_files = h5_files
        self.frames_per_file = None
        self.total_frames = 0
        self.file_frame_ranges = []
        self.height = None
        self.width = None
        self.frame_indices = frame_indices
        
        # Cache for h5 files
        self.cache_size = cache_size
        self.file_cache = {}
        self.cache_order = []
        
        # Get dimensions from first file
        with h5py.File(self.h5_files[0], 'r') as f:
            data = f['default']
            self.height, self.width = data.shape[1:]
            if self.frames_per_file is None:
                self.frames_per_file = data.shape[0]
        
        if frame_indices is None:
            # Use all frames
            for h5_file in self.h5_files:
                with h5py.File(h5_file, 'r') as f:
                    n_frames = f['default'].shape[0]
                    self.file_frame_ranges.append((self.total_frames, self.total_frames + n_frames))
                    self.total_frames += n_frames
        else:
            self.total_frames = len(frame_indices)
            
        # Precompute file indices for faster lookup
        if frame_indices is None:
            self.file_lookup = np.zeros(self.total_frames, dtype=np.int32)
            self.frame_lookup = np.zeros(self.total_frames, dtype=np.int32)
            for idx in range(self.total_frames):
                file_idx = 0
                while file_idx < len(self.file_frame_ranges) and idx >= self.file_frame_ranges[file_idx][1]:
                    file_idx += 1
                self.file_lookup[idx] = file_idx
                self.frame_lookup[idx] = idx - self.file_frame_ranges[file_idx][0]
    
    def _get_cached_file(self, file_idx):
        """Get file from cache or load it if not present"""
        if file_idx in self.file_cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(file_idx)
            self.cache_order.append(file_idx)
            return self.file_cache[file_idx]
        
        # Load file into cache
        data = h5py.File(self.h5_files[file_idx], 'r')['default'][:]
        
        # If cache is full, remove least recently used file
        if len(self.file_cache) >= self.cache_size:
            lru_idx = self.cache_order.pop(0)
            del self.file_cache[lru_idx]
        
        self.file_cache[file_idx] = data
        self.cache_order.append(file_idx)
        return data
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        if self.frame_indices is not None:
            # If using specific frame indices
            file_idx, frame_idx = self.frame_indices[idx]
        else:
            # If using all frames sequentially
            file_idx = self.file_lookup[idx]
            frame_idx = self.frame_lookup[idx]
        
        # Get frame from cache
        data = self._get_cached_file(file_idx)
        frame = data[frame_idx].astype(np.float32)
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

def evaluate_test_loss(model, test_loader, device):
    """Evaluate model on test set and return average losses"""
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, latent_sample, latent_dist = model(data)
            
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + torch.log(latent_dist + 1e-10) - latent_dist)
            
            beta = 0.00  # Same beta as training
            loss = recon_loss + beta * kl_loss
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            n_batches += 1
    
    return {
        'test_loss': total_loss / n_batches,
        'test_recon_loss': total_recon_loss / n_batches,
        'test_kl_loss': total_kl_loss / n_batches
    }

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
        if len(test_losses) > 1:  # Need at least 2 points for interpolation
            # Calculate iterations where test losses were recorded (end of each epoch)
            steps_per_epoch = len(train_loader)
            test_iterations = [(i + 1) * steps_per_epoch for i in range(len(test_losses))]
            
            # Create interpolation functions for each test loss type
            f_total = interp1d(test_iterations, [x['test_loss'] for x in test_losses], kind='linear')
            f_recon = interp1d(test_iterations, [x['test_recon_loss'] for x in test_losses], kind='linear')
            f_kl = interp1d(test_iterations, [x['test_kl_loss'] for x in test_losses], kind='linear')
            
            # Generate points for smooth curve
            x_smooth = np.linspace(test_iterations[0], test_iterations[-1], len(train_losses))
            
            plt.plot(x_smooth, f_total(x_smooth), '--', label='Test Total Loss', linewidth=2, alpha=0.7)
            plt.plot(x_smooth, f_recon(x_smooth), '--', label='Test Recon Loss', linewidth=2, alpha=0.7)
            plt.plot(x_smooth, f_kl(x_smooth), '--', label='Test KL Loss', linewidth=2, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('VAE Training and Test Losses')
        plt.legend()
        plt.grid(True)
        
        plt.xlim(1, len(train_losses))
        
        # Add current values in text box
        textstr = (f'Current Training Values:\n'
                  f'Total Loss: {train_losses[-1]:.4f}\n'
                  f'Recon Loss: {recon_losses[-1]:.4f}\n'
                  f'KL Loss: {kl_losses[-1]:.4f}')
        if test_losses:
            textstr += (f'\n\nLatest Test Values:\n'
                       f'Total Loss: {test_losses[-1]["test_loss"]:.4f}\n'
                       f'Recon Loss: {test_losses[-1]["test_recon_loss"]:.4f}\n'
                       f'KL Loss: {test_losses[-1]["test_kl_loss"]:.4f}')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    plt.savefig(os.path.join(exp_dir, "plots", "training_loss.png"))
    plt.close()

def normalize_frame(frame):
    """Normalize frame to 0-255 range"""
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max == frame_min:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)

def create_comparison_video(model, test_dataset, output_path, recording_dirs, fps=1, max_frames=None, recording_idx=0):
    """Create a video comparing original and reconstructed frames
    
    Args:
        model: The VAE model
        test_dataset: The test dataset
        output_path: Where to save the video
        recording_dirs: List of recording directories
        fps: Frames per second
        max_frames: Maximum number of frames to process
        recording_idx: Which recording to use (index into recording_dirs)
    """
    device = next(model.parameters()).device
    
    # Get dimensions from first frame
    first_frame = test_dataset[0].numpy()[0]
    height, width = first_frame.shape
    
    # Initialize video writer for side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Find consecutive frames from the specified recording
    recording_files = sorted(glob(os.path.join(recording_dirs[recording_idx], "volume*.h5")))
    frames_per_volume = None
    
    # Get frames per volume from first file
    with h5py.File(recording_files[0], 'r') as f:
        frames_per_volume = f['default'].shape[0]
    
    # Create indices for consecutive frames from this recording
    start_file_idx = sum(len(glob(os.path.join(d, "volume*.h5"))) 
                        for d in recording_dirs[:recording_idx])
    consecutive_indices = [(start_file_idx + file_idx, frame_idx)
                         for file_idx in range(len(recording_files))
                         for frame_idx in range(frames_per_volume)]
    
    # Determine how many frames to process
    total_frames = len(consecutive_indices)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    model.eval()
    print(f"Creating comparison video with {total_frames} consecutive frames from recording {recording_idx + 1}...")
    with torch.no_grad():
        for i in tqdm(range(total_frames)):
            # Get original frame using consecutive indices
            file_idx, frame_idx = consecutive_indices[i]
            with h5py.File(test_dataset.h5_files[file_idx], 'r') as f:
                frame = f['default'][frame_idx].astype(np.float32)
            
            # Prepare frame for model
            frame_tensor = torch.from_numpy(frame[np.newaxis, np.newaxis, :, :]).to(device)
            
            # Get reconstruction
            reconstruction, _, _ = model(frame_tensor)
            
            # Convert to numpy arrays and normalize to uint8
            original = normalize_frame(frame)
            reconstructed = normalize_frame(reconstruction[0, 0].cpu().numpy())
            
            # Create side-by-side comparison
            comparison = np.hstack([original, reconstructed])
            
            # Convert to RGB
            comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            out.write(comparison_rgb)
    
    out.release()
    print(f"Video saved as {output_path}")

def train_vae(train_loader, test_loader, model, optimizer, device, epoch, train_losses, 
              recon_losses, kl_losses, test_losses, exp_dir, test_dataset, recording_dirs):
    model.train()
    
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, latent_sample, latent_dist = model(data)
        
        recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + torch.log(latent_dist + 1e-10) - latent_dist)
        
        beta = 0.00
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]:')
            print(f'  Total Loss: {loss.item():.4f}')
            print(f'  Reconstruction Loss: {recon_loss.item():.4f}')
            print(f'  KL Loss: {kl_loss.item():.4f}')
            plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader)
    
    # Evaluate on test set at the end of each epoch
    test_loss_dict = evaluate_test_loss(model, test_loader, device)
    test_losses.append(test_loss_dict)
    
    # Save losses to CSV
    train_loss_dict = {
        'iteration': list(range(1, len(train_losses) + 1)),
        'total_loss': train_losses,
        'recon_loss': recon_losses,
        'kl_loss': kl_losses
    }
    save_losses_to_csv(train_loss_dict, os.path.join(exp_dir, "train_losses.csv"))
    
    test_loss_dict_for_csv = {
        'epoch': list(range(1, len(test_losses) + 1)),
        'total_loss': [x['test_loss'] for x in test_losses],
        'recon_loss': [x['test_recon_loss'] for x in test_losses],
        'kl_loss': [x['test_kl_loss'] for x in test_losses]
    }
    save_losses_to_csv(test_loss_dict_for_csv, os.path.join(exp_dir, "test_losses.csv"))
    
    # Plot losses
    plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader)
    
    # Generate comparison video for each recording
    for recording_idx in range(len(recording_dirs)):
        video_path = os.path.join(exp_dir, "videos", f"reconstruction_recording_{recording_idx + 1}_epoch_{epoch}.mp4")
        create_comparison_video(model, test_dataset, video_path, recording_dirs,
                              max_frames=min(100, len(test_dataset)//len(recording_dirs)),
                              recording_idx=recording_idx)

def cleanup():
    """Clean up temporary directories"""
    import shutil
    if os.path.exists("data/temp_train"):
        shutil.rmtree("data/temp_train")
    if os.path.exists("data/temp_test"):
        shutil.rmtree("data/temp_test")

def main():
    try:
        # Get all recording directories
        data_dir = "data"
        recording_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                         if d.endswith('_preprocessed') and os.path.isdir(os.path.join(data_dir, d))]
        
        if not recording_dirs:
            raise ValueError("No preprocessed recording directories found in data/")
        
        print(f"Found {len(recording_dirs)} recording directories: {recording_dirs}")
        
        # Collect all h5 files from all recordings
        all_h5_files = []
        for recording_dir in recording_dirs:
            recording_files = sorted(glob(os.path.join(recording_dir, "volume*.h5")))
            all_h5_files.extend(recording_files)
        
        if not all_h5_files:
            raise ValueError("No h5 files found in recording directories")
        
        # Get data dimensions and frames per file from first file
        with h5py.File(all_h5_files[0], 'r') as f:
            data = f['default']
            height, width = data.shape[1:]
            frames_per_file = data.shape[0]
        
        # Parameters
        model_params = {
            'image_height': height,
            'image_width': width,
            'n_distributions': 1024,  # Number of categorical distributions
            'n_categories': 8,    # Number of categories per distribution
        }
        
        training_params = {
            'batch_size': 32,
            'epochs': 5,
            'learning_rate': 1e-3,
            'train_split': 0.9,  # 90% for training
        }
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        print(f"Saving experiment results to: {exp_dir}")
        
        # Initialize model
        model = VariationalAutoEncoder(
            model_params['image_height'],
            model_params['image_width'],
            model_params['n_distributions'],
            model_params['n_categories']
        )
        
        # Save model architecture and parameters
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
            f.write("\nRecording Directories:\n")
            for dir_path in recording_dirs:
                f.write(f"- {dir_path}\n")
            
            # Add statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\nModel Statistics:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Input shape: ({1}, {height}, {width})\n")
            f.write(f"Latent space size: {model_params['n_distributions'] * model_params['n_categories']}\n")
        
        # Calculate total number of frames
        total_frames = len(all_h5_files) * frames_per_file
        
        # Create indices for all frames
        all_indices = [(file_idx, frame_idx) 
                      for file_idx in range(len(all_h5_files)) 
                      for frame_idx in range(frames_per_file)]
        
        # Randomly shuffle indices
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_indices)
        
        # Split indices into train and test
        n_train = int(len(all_indices) * training_params['train_split'])
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:]
        
        # Create datasets using the split indices
        train_dataset = BrainDataset(all_h5_files, train_indices)
        test_dataset = BrainDataset(all_h5_files, test_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_params['batch_size'],
            shuffle=True,
            num_workers=4,  # Adjust based on CPU cores
            pin_memory=True,  # Speeds up transfer to GPU
            prefetch_factor=2,  # Number of batches loaded in advance by each worker
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        print(f"Total frames: {total_frames}")
        print(f"Training set size: {len(train_dataset)} frames ({len(train_dataset) / frames_per_file:.1f} volumes)")
        print(f"Test set size: {len(test_dataset)} frames ({len(test_dataset) / frames_per_file:.1f} volumes)")
        
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
        
        # Move model to device after architecture is saved
        device = torch.device("cuda:0") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        
        # Training loop
        print("Starting training...")
        train_losses = []
        recon_losses = []
        kl_losses = []
        test_losses = []  # List to store test losses per epoch
        
        for epoch in range(1, training_params['epochs'] + 1):
            train_vae(train_loader, test_loader, model, optimizer, device, epoch, 
                     train_losses, recon_losses, kl_losses, test_losses, exp_dir, 
                     test_dataset, recording_dirs)
            
            # Save checkpoint every 4 epochs
            if epoch % 4 == 0:
                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"vae_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'recon_losses': recon_losses,
                    'kl_losses': kl_losses,
                    'test_losses': test_losses,
                }, checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(exp_dir, "checkpoints", "vae_final.pt")
        torch.save(model.state_dict(), final_model_path)
        
        # Clean up at the end
        cleanup()
    except Exception as e:
        cleanup()  # Clean up even if there's an error
        raise e

main()