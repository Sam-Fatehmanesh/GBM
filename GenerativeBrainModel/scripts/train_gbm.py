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
import cv2
import h5py

from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.sequential_spike_dataset import SequentialSpikeDataset
from GenerativeBrainModel.custom_functions.visualization import create_comparison_video, update_loss_plot

def get_max_z_planes(spike_files):
    """Get the maximum number of z-planes across all subjects.
    
    Args:
        spike_files: List of paths to processed spike data H5 files
    
    Returns:
        max_z_planes: Maximum number of z-planes found across all subjects
    """
    max_z_planes = 0
    
    for h5_file in spike_files:
        with h5py.File(h5_file, 'r') as f:
            cell_positions = f['cell_positions'][:]
            # Get unique z values (rounded to handle floating point precision)
            z_values = np.unique(np.round(cell_positions[:, 2], decimals=3))
            max_z_planes = max(max_z_planes, len(z_values))
    
    if max_z_planes == 0:
        raise ValueError("No z-planes found in any of the spike files!")
    
    return max_z_planes

def create_experiment_dir():
    """Create a timestamped experiment directory with all necessary subdirectories"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', 'gbm', timestamp)
    
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

def create_prediction_video(model, test_dataset, output_path, max_seqs=10, fps=2):
    """Create video showing model predictions vs actual next frames"""
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
                            (scaled_width * 3, scaled_height), 
                            isColor=True)
        
        if not out.isOpened():
            print(f"Failed to create video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (scaled_width * 3, scaled_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer with both codecs. Skipping video creation.")
                return
        
        with torch.no_grad():
            for i in range(min(max_seqs, len(test_dataset))):
                # Get sequence
                sequence = test_dataset[i].to(device)
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
                # Get predictions
                pred_sequence = model(sequence)
                
                # For each frame in sequence (except last)
                for t in range(sequence.size(1) - 1):
                    # Get current frame, prediction, and target
                    current = sequence[0, t].cpu().numpy()
                    predicted = pred_sequence[0, t].cpu().numpy()
                    target = sequence[0, t + 1].cpu().numpy()
                    
                    # Convert to uint8 images
                    curr_img = (current * 255).astype(np.uint8)
                    pred_img = (predicted * 255).astype(np.uint8)
                    targ_img = (target * 255).astype(np.uint8)
                    
                    # Scale up images
                    curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    targ_img = cv2.resize(targ_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to RGB
                    curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                    targ_rgb = cv2.cvtColor(targ_img, cv2.COLOR_GRAY2BGR)
                    
                    # Add text labels
                    cv2.putText(curr_rgb, 'Current', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(pred_rgb, 'Predicted Next', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(targ_rgb, 'Actual Next', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Combine horizontally
                    combined = np.hstack([curr_rgb, pred_rgb, targ_rgb])
                    out.write(combined)
        
        out.release()
        print(f"\nPrediction video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error creating prediction video: {str(e)}")

def create_data_check_video(model, test_dataset, output_path, max_seqs=5, fps=2):
    """Create video showing raw sequences from dataset to verify data generation."""
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
                            (scaled_width * 3, scaled_height), 
                            isColor=True)
        
        if not out.isOpened():
            print(f"Failed to create video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (scaled_width * 3, scaled_height), 
                                isColor=True)
            
            if not out.isOpened():
                print(f"Failed to create video writer with both codecs. Skipping video creation.")
                return
        
        with torch.no_grad():
            for i in range(min(max_seqs, len(test_dataset))):
                # Get sequence
                sequence = test_dataset[i].to(device)
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
                # Get predictions
                pred_sequence = model(sequence)
                
                # For each frame in sequence (except last)
                for t in range(sequence.size(1) - 1):
                    # Get current frame, prediction, and target
                    current = sequence[0, t].cpu().numpy()
                    predicted = pred_sequence[0, t].cpu().numpy()
                    target = sequence[0, t + 1].cpu().numpy()
                    
                    # Convert to uint8 images
                    curr_img = (current * 255).astype(np.uint8)
                    pred_img = (predicted * 255).astype(np.uint8)
                    targ_img = (target * 255).astype(np.uint8)
                    
                    # Scale up images
                    curr_img = cv2.resize(curr_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    pred_img = cv2.resize(pred_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    targ_img = cv2.resize(targ_img, (scaled_width, scaled_height), 
                                       interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to RGB
                    curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
                    pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                    targ_rgb = cv2.cvtColor(targ_img, cv2.COLOR_GRAY2BGR)
                    
                    # Add text labels
                    cv2.putText(curr_rgb, 'Current', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(pred_rgb, 'Predicted Next', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(targ_rgb, 'Actual Next', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add frame number
                    cv2.putText(curr_rgb, f'Seq {i}, Frame {t}', (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Combine horizontally
                    combined = np.hstack([curr_rgb, pred_rgb, targ_rgb])
                    out.write(combined)
        
        out.release()
        print(f"\nData check video saved as: {output_path}")
        
    except Exception as e:
        print(f"Error creating data check video: {str(e)}")

def main():
    try:
        # Parameters
        params = {
            'batch_size': 32,
            'num_epochs': 4,
            'learning_rate': 1e-4,
            'mamba_layers': 1,
            'mamba_dim': 1024,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95
        }
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        tqdm.write(f"Saving experiment results to: {exp_dir}")
        
        # Create datasets from spike data
        processed_dir = "processed_spikes"
        spike_files = []
        for file in os.listdir(processed_dir):
            if file.endswith("_processed.h5"):
                spike_files.append(os.path.join(processed_dir, file))
        
        if not spike_files:
            raise ValueError("No processed spike data files found!")
        
        # Get maximum number of z-planes across all subjects
        max_z_planes = get_max_z_planes(spike_files)
        params['seq_len'] = params['timesteps_per_sequence'] * max_z_planes
        
        tqdm.write(f"Maximum z-planes across subjects: {max_z_planes}")
        tqdm.write(f"Total sequence length: {params['seq_len']} ({params['timesteps_per_sequence']} timepoints × {max_z_planes} z-planes)")
        
        # Create train and test datasets
        train_datasets = [
            SequentialSpikeDataset(
                f, 
                seq_len=params['seq_len'],
                split='train',
                train_ratio=params['train_ratio']
            ) for f in spike_files
        ]
        test_datasets = [
            SequentialSpikeDataset(
                f,
                seq_len=params['seq_len'],
                split='test',
                train_ratio=params['train_ratio']
            ) for f in spike_files
        ]
        
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
        
        tqdm.write(f"Created datasets:")
        tqdm.write(f"Training sequences: {len(train_dataset)}")
        tqdm.write(f"Test sequences: {len(test_dataset)}")
        
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
        model = GBM(
            mamba_layers=params['mamba_layers'],
            mamba_dim=params['mamba_dim']
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
        
        # Move to GPU if available
        device = "cuda:0"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        model = model.to(device)
        
        # Create data check video with untrained model predictions
        tqdm.write("\nCreating data check video to verify dataset generation and see untrained predictions...")
        video_path = os.path.join(exp_dir, 'videos', 'data_check_untrained.mp4')
        create_data_check_video(model, test_dataset, video_path)
        tqdm.write("Please verify the data check video before proceeding with training.")
        tqdm.write("Press Enter to continue or Ctrl+C to abort...")
        
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
                predictions = model(batch)
                loss = model.compute_loss(predictions, batch[:, 1:])  # Compare with next frames
                
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
                if batch_idx % 32 == 0:
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
                    predictions = model(batch)
                    loss = model.compute_loss(predictions, batch[:, 1:])
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            # Create prediction video after each epoch
            video_path = os.path.join(exp_dir, 'videos', f'predictions_epoch_{epoch+1:03d}.mp4')
            create_prediction_video(model, test_dataset, video_path)
            
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
        
        # Create final prediction video
        video_path = os.path.join(exp_dir, 'videos', 'final_predictions.mp4')
        create_prediction_video(model, test_dataset, video_path, max_seqs=20)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 