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
from BrainSimulator.models.gbm import GBM
from BrainSimulator.models.vae import VariationalAutoEncoder

os.environ["QT_QPA_PLATFORM"] = "offscreen"

class BrainSequenceDataset(Dataset):
    def __init__(self, h5_files, sequence_length=32, frame_indices=None, cache_size=4):
        """
        Args:
            h5_files: List of h5 file paths
            sequence_length: Number of consecutive frames in each sequence
            frame_indices: Optional list of (file_idx, frame_idx) tuples specifying which frames to use
            cache_size: Number of h5 files to keep in memory (default: 4)
        """
        self.h5_files = h5_files
        self.sequence_length = sequence_length
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
        return max(0, self.total_frames - self.sequence_length)
    
    def __getitem__(self, idx):
        frames = []
        for i in range(self.sequence_length):
            frame_idx = idx + i
            if self.frame_indices is not None:
                # If using specific frame indices
                file_idx, frame_idx = self.frame_indices[frame_idx]
            else:
                # If using all frames sequentially
                file_idx = self.file_lookup[frame_idx]
                frame_idx = self.frame_lookup[frame_idx]
            
            # Get frame from cache
            data = self._get_cached_file(file_idx)
            frame = data[frame_idx].astype(np.float32)
            frame = frame[np.newaxis, :, :]
            frames.append(torch.from_numpy(frame))
        
        # Stack frames into sequence
        sequence = torch.stack(frames)
        return sequence

def create_experiment_dir():
    """Create and return path to new experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def evaluate_test_loss(model, test_loader, device, model_params):
    """Evaluate model on test set and return average loss"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for sequences in test_loader:
            # Reshape sequences to (batch, seq_len, 1, height, width)
            sequences = sequences.permute(1, 0, 2, 3, 4).to(device)
            batch_size = sequences.size(0)
            
            # Get latent sequences from VAE
            latent_sequences = []
            latent_dists = []
            for t in range(sequences.size(1)):
                latent_sample, latent_dist = model.pretrained_vae.encode(sequences[:, t])
                latent_sequences.append(latent_sample)
                latent_dists.append(latent_dist)
            
            # Stack along sequence dimension
            latent_sequences = torch.stack(latent_sequences, dim=1)  # [batch, seq_len, latent_dim]
            latent_dists = torch.stack(latent_dists, dim=1)  # [batch, seq_len, latent_dim]
            
            # Predict next latent distributions
            predicted_dists = model(latent_sequences[:, :-1])  # Input all but last
            target_dists = latent_dists[:, 1:]  # Target is all but first
            
            # Reshape for cross entropy loss
            batch_size, seq_len = predicted_dists.shape[:2]
            predicted_dists = predicted_dists.reshape(-1, model.num_distributions, model_params['n_categories'])
            target_dists = target_dists.reshape(-1, model.num_distributions, model_params['n_categories'])
            
            # Compute cross entropy loss between predicted and target distributions
            # Treat each distribution independently
            loss = F.cross_entropy(
                predicted_dists.transpose(1, 2),  # [batch*seq_len, n_categories, n_distributions]
                target_dists.transpose(1, 2),     # [batch*seq_len, n_categories, n_distributions]
                reduction='mean'
            )
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file"""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

def plot_losses(train_losses, test_losses, exp_dir, train_loader):
    """Plot and save the training and test losses"""
    plt.figure(figsize=(12, 8))
    
    if len(train_losses) > 0:
        # Plot training losses
        iterations = range(1, len(train_losses) + 1)
        plt.plot(iterations, train_losses, label='Train Loss', linewidth=2, alpha=0.7)
        
        # Plot test losses with linear interpolation
        if len(test_losses) > 1:  # Need at least 2 points for interpolation
            # Calculate iterations where test losses were recorded (end of each epoch)
            steps_per_epoch = len(train_loader)
            test_iterations = [(i + 1) * steps_per_epoch for i in range(len(test_losses))]
            
            # Create interpolation function
            f_test = interp1d(test_iterations, test_losses, kind='linear')
            
            # Generate points for smooth curve
            x_smooth = np.linspace(test_iterations[0], test_iterations[-1], len(train_losses))
            plt.plot(x_smooth, f_test(x_smooth), '--', label='Test Loss', linewidth=2, alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('GBM Training and Test Losses')
        plt.legend()
        plt.grid(True)
        
        plt.xlim(1, len(train_losses))
        
        # Add current values in text box
        textstr = f'Current Training Loss: {train_losses[-1]:.4f}'
        if test_losses:
            textstr += f'\nLatest Test Loss: {test_losses[-1]:.4f}'
        
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

def create_prediction_video(model, test_dataset, output_path, recording_dirs, fps=1, max_frames=None, recording_idx=0):
    """Create a video comparing true vs predicted brain activity"""
    device = next(model.parameters()).device
    model.eval()
    
    # Get dimensions from first frame
    first_frame = test_dataset[0][0].numpy()[0]
    height, width = first_frame.shape
    
    # Initialize video writer for side-by-side comparison
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Get a sequence from the test dataset and reshape to (1, seq_len, 1, height, width)
    sequence = test_dataset[0].unsqueeze(1).permute(1, 0, 2, 3, 4).to(device)
    
    with torch.no_grad():
        # Get predictions for the sequence
        predicted_frames = model.predict_image_to_image(sequence)
        
        # Generate comparison frames
        for t in range(sequence.size(1) - 1):  # -1 because we're comparing with next frame
            # Get true next frame
            true_frame = sequence[0, t + 1].cpu().numpy()[0]  # [0] for batch, [0] for channel
            
            # Get predicted frame
            pred_frame = predicted_frames[0, t].cpu().numpy()[0]  # [0] for batch, [0] for channel
            
            # Normalize frames
            true_frame_norm = normalize_frame(true_frame)
            pred_frame_norm = normalize_frame(pred_frame)
            
            # Create side-by-side comparison
            comparison = np.hstack([true_frame_norm, pred_frame_norm])
            comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            out.write(comparison_rgb)
    
    out.release()
    print(f"Video saved as {output_path}")

def train_gbm(train_loader, test_loader, model, optimizer, device, epoch, train_losses, 
              test_losses, exp_dir, test_dataset, recording_dirs, model_params):
    model.train()
    
    for batch_idx, sequences in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # Reshape sequences to (batch, seq_len, 1, height, width)
        sequences = sequences.permute(1, 0, 2, 3, 4).to(device)
        batch_size = sequences.size(0)
        optimizer.zero_grad()
        
        # Get latent sequences from VAE
        latent_sequences = []
        latent_dists = []
        for t in range(sequences.size(1)):
            latent_sample, latent_dist = model.pretrained_vae.encode(sequences[:, t])
            latent_sequences.append(latent_sample)
            latent_dists.append(latent_dist)
        
        # Stack along sequence dimension
        latent_sequences = torch.stack(latent_sequences, dim=1)  # [batch, seq_len, latent_dim]
        latent_dists = torch.stack(latent_dists, dim=1)  # [batch, seq_len, latent_dim]
        
        # Predict next latent distributions
        predicted_dists = model(latent_sequences[:, :-1])  # Input all but last
        target_dists = latent_dists[:, 1:]  # Target is all but first
        
        # Reshape for cross entropy loss
        batch_size, seq_len = predicted_dists.shape[:2]
        predicted_dists = predicted_dists.reshape(-1, model.num_distributions, model_params['n_categories'])
        target_dists = target_dists.reshape(-1, model.num_distributions, model_params['n_categories'])
        
        # Compute cross entropy loss between predicted and target distributions
        # Treat each distribution independently
        loss = F.cross_entropy(
            predicted_dists.transpose(1, 2),  # [batch*seq_len, n_categories, n_distributions]
            target_dists.transpose(1, 2),     # [batch*seq_len, n_categories, n_distributions]
            reduction='mean'
        )
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}]:')
            print(f'  Loss: {loss.item():.4f}')
            plot_losses(train_losses, test_losses, exp_dir, train_loader)
    
    # Evaluate on test set at the end of each epoch
    test_loss = evaluate_test_loss(model, test_loader, device, model_params)
    test_losses.append(test_loss)
    
    # Save losses to CSV
    train_loss_dict = {
        'iteration': list(range(1, len(train_losses) + 1)),
        'loss': train_losses
    }
    save_losses_to_csv(train_loss_dict, os.path.join(exp_dir, "train_losses.csv"))
    
    test_loss_dict = {
        'epoch': list(range(1, len(test_losses) + 1)),
        'loss': test_losses
    }
    save_losses_to_csv(test_loss_dict, os.path.join(exp_dir, "test_losses.csv"))
    
    # Plot losses
    plot_losses(train_losses, test_losses, exp_dir, train_loader)
    
    # Generate prediction video for each recording
    for recording_idx in range(len(recording_dirs)):
        video_path = os.path.join(exp_dir, "videos", f"prediction_recording_{recording_idx + 1}_epoch_{epoch}.mp4")
        create_prediction_video(model, test_dataset, video_path, recording_dirs,
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
            'd_model': 1024,  # Must match VAE latent size
            'num_layers': 8,      # Number of Mamba layers
        }
        
        training_params = {
            'batch_size': 4,
            'sequence_length': 32,
            'epochs': 3,
            'learning_rate': 1e-4,
            'train_split': 0.9,  # 90% for training
        }
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        print(f"Saving experiment results to: {exp_dir}")
        
        # Initialize VAE with same parameters as pretrained
        vae = VariationalAutoEncoder(
            model_params['image_height'],
            model_params['image_width'],
            model_params['n_distributions'],
            model_params['n_categories']
        )
        
        # Load pretrained VAE weights
        vae_checkpoint = torch.load('experiments/20241223_234254/checkpoints/vae_final.pt')
        vae.load_state_dict(vae_checkpoint)
        
        # Initialize GBM with VAE already configured
        model = GBM(
            d_model=model_params['d_model'],
            num_layers=model_params['num_layers'],
            num_distributions=model_params['n_distributions'],
            num_categories=model_params['n_categories']
        )
        model.pretrained_vae = vae  # Replace default VAE with our loaded one
        
        # Set device
        device = torch.device("cuda:0")
        print(f"Using device: {device}")
        model = model.to(device)
        vae = vae.to(device)
        
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
        train_dataset = BrainSequenceDataset(
            all_h5_files, 
            sequence_length=training_params['sequence_length'],
            frame_indices=train_indices
        )
        test_dataset = BrainSequenceDataset(
            all_h5_files, 
            sequence_length=training_params['sequence_length'],
            frame_indices=test_indices
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_params['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
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
        print(f"Training set size: {len(train_dataset)} sequences")
        print(f"Test set size: {len(test_dataset)} sequences")
        
        # Save model architecture and parameters
        with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
            f.write("Complete Model Architecture:\n")
            f.write("=" * 50 + "\n\n")
            f.write("VAE Architecture:\n")
            f.write("-" * 30 + "\n")
            f.write(str(vae))
            f.write("\n\nMamba Core Architecture:\n")
            f.write("-" * 30 + "\n")
            f.write(str(model.mamba_core))
            f.write("\n\nComplete GBM Architecture:\n")
            f.write("-" * 30 + "\n")
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
            vae_params = sum(p.numel() for p in vae.parameters())
            mamba_params = sum(p.numel() for p in model.mamba_core.parameters())
            linear_params = sum(p.numel() for name, p in model.named_parameters() if 'linear' in name)
            
            f.write(f"\nModel Statistics:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"VAE parameters (frozen): {vae_params:,}\n")
            f.write(f"Mamba parameters: {mamba_params:,}\n")
            f.write(f"Linear layer parameters: {linear_params:,}\n")
            f.write(f"Input shape: ({1}, {height}, {width})\n")
            f.write(f"Latent space size: {model_params['n_distributions'] * model_params['n_categories']}\n")
            f.write(f"Sequence length: {training_params['sequence_length']}\n")
            f.write(f"Batch size: {training_params['batch_size']}\n")
            
            # Add model configuration details
            f.write("\nModel Configuration:\n")
            f.write(f"Number of Mamba layers: {model_params['num_layers']}\n")
            f.write(f"Hidden dimension (d_model): {model_params['d_model']}\n")
            f.write(f"Number of categorical distributions: {model_params['n_distributions']}\n")
            f.write(f"Categories per distribution: {model_params['n_categories']}\n")
            f.write(f"Input dimension: {model_params['n_distributions'] * model_params['n_categories']}\n")
            f.write(f"Output dimension: {model_params['n_distributions'] * model_params['n_categories']}\n")
            
            # Add training configuration
            f.write("\nTraining Configuration:\n")
            f.write(f"Learning rate: {training_params['learning_rate']}\n")
            f.write(f"Number of epochs: {training_params['epochs']}\n")
            f.write(f"Train/Test split: {training_params['train_split']}\n")
            f.write(f"Device: {device}\n")
            
            # Add dataset statistics
            f.write("\nDataset Statistics:\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Training sequences: {len(train_dataset)}\n")
            f.write(f"Test sequences: {len(test_dataset)}\n")
            f.write(f"Frame dimensions: {height}x{width}\n")
            f.write(f"Frames per volume: {frames_per_file}\n")
            f.write(f"Number of volumes: {len(all_h5_files)}\n")
        
        # Freeze VAE parameters
        for param in model.pretrained_vae.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=training_params['learning_rate']
        )
        
        # Training loop
        print("Starting training...")
        train_losses = []
        test_losses = []
        
        for epoch in range(1, training_params['epochs'] + 1):
            train_gbm(train_loader, test_loader, model, optimizer, device, epoch, 
                     train_losses, test_losses, exp_dir, test_dataset, recording_dirs,
                     model_params)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"gbm_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                }, checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(exp_dir, "checkpoints", "gbm_final.pt")
        torch.save(model.state_dict(), final_model_path)
        
        # Clean up at the end
        cleanup()
    except Exception as e:
        cleanup()  # Clean up even if there's an error
        raise e

main() 