# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from GenerativeBrainModel.models.simple_autoencoder import SimpleAutoencoder
from GenerativeBrainModel.datasets.spike_datasets import GridSpikeDataset, SyntheticSpikeDataset
from GenerativeBrainModel.custom_functions.visualization import create_comparison_video, update_loss_plot

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