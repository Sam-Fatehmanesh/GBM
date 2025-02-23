#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from scipy.interpolate import interp1d

from BrainSimulator.models.point_transformer_vae import create_point_transformer_vae
from BrainSimulator.models.voxel_conv_vae import create_voxel_conv_vae
from BrainSimulator.data.brain_dataset import BrainDataset

def create_experiment_dir(model_type, model_size):
    """Create and return path to new experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{timestamp}_{model_type}_{model_size}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    return exp_dir

def save_losses_to_csv(losses_dict, filepath):
    """Save losses to CSV file."""
    df = pd.DataFrame(losses_dict)
    df.to_csv(filepath, index=False)

def plot_losses(train_losses, recon_losses, kl_losses, test_losses, exp_dir, train_loader):
    """Plot and save the training and test losses."""
    plt.figure(figsize=(12, 8))
    
    if len(train_losses) > 0:
        # Plot training losses
        iterations = range(1, len(train_losses) + 1)
        plt.plot(iterations, train_losses, label='Train Total Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, recon_losses, label='Train Recon Loss', linewidth=2, alpha=0.7)
        plt.plot(iterations, kl_losses, label='Train KL Loss', linewidth=2, alpha=0.7)
        
        # Plot test losses with linear interpolation
        if len(test_losses) > 1:  # Need at least 2 points for interpolation
            # Calculate iterations where test losses were recorded
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

def evaluate_test_loss(model, test_loader, device, beta=1.0):
    """Evaluate model on test set and return average losses."""
    model.eval()
    total_recon_loss = 0
    total_kl_loss = 0
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):  # Point cloud data
                features = batch['features'].to(device)  # Shape: [B, N, 4] (x,y,z,value)
                recon_batch, mu, logvar = model(features)
                
                # MSE loss for both spatial coordinates and values
                recon_loss = F.mse_loss(recon_batch, features, reduction='mean')
                
                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / features.size(0)
            else:  # Voxel data
                voxels = batch['voxels'].to(device)  # Ensure we extract 'voxels' from dict
                recon_voxels, mu, logvar = model(voxels)
                
                # Binary cross entropy for voxel reconstruction
                recon_loss = F.binary_cross_entropy(recon_voxels, voxels, reduction='mean')
                
                # KL divergence
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / voxels.size(0)
            
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

def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """Train for one epoch."""
    model.train()
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        if isinstance(batch, dict):  # Point cloud data
            features = batch['features'].to(device)  # Shape: [B, N, 4] (x,y,z,value)
            recon_batch, mu, logvar = model(features)
            
            # MSE loss for both spatial coordinates and values
            recon_loss = F.mse_loss(recon_batch, features, reduction='mean')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / features.size(0)
            
            # Total loss
            loss = recon_loss + beta * kl_loss
        else:  # Voxel data
            voxels = batch['voxels'].to(device)  # Ensure we extract 'voxels' from dict
            recon_voxels, mu, logvar = model(voxels)
            
            # Binary cross entropy for voxel reconstruction
            recon_loss = F.binary_cross_entropy(recon_voxels, voxels, reduction='mean')
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / voxels.size(0)
            
            # Total loss
            loss = recon_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
    
    return train_losses, recon_losses, kl_losses

def main():
    parser = argparse.ArgumentParser(description='Train Brain VAE models')
    parser.add_argument('--model-type', type=str, required=True, choices=['point', 'voxel'],
                      help='Type of model to train (point or voxel)')
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large'],
                      help='Size of model to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for KL loss weight')
    parser.add_argument('--data-dir', type=str, default='data/processed_subjects',
                      help='Directory containing processed subject data')
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.model_type, args.model_size)
    print(f"Saving experiment results to: {exp_dir}")
    
    # Save experiment configuration
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    if args.model_type == 'point':
        dataset_mode = 'point_cloud'
        model = create_point_transformer_vae(args.model_size).to(device)
    else:
        dataset_mode = 'voxel'
        model = create_voxel_conv_vae(args.model_size).to(device)
    
    train_dataset = BrainDataset(args.data_dir, mode=dataset_mode, split='train')
    test_dataset = BrainDataset(args.data_dir, mode=dataset_mode, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    # Save model architecture
    with open(os.path.join(exp_dir, "model_architecture.txt"), "w") as f:
        f.write(str(model))
        f.write("\n\nModel Parameters:\n")
        f.write(f"Total parameters: {model.get_param_count():,}\n")
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(args.lr)
    
    # Training loop
    print("Starting training...")
    all_train_losses = []
    all_recon_losses = []
    all_kl_losses = []
    test_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_losses, recon_losses, kl_losses = train_epoch(
            model, train_loader, optimizer, device, args.beta)
        
        all_train_losses.extend(train_losses)
        all_recon_losses.extend(recon_losses)
        all_kl_losses.extend(kl_losses)
        
        # Evaluate on test set
        test_loss = evaluate_test_loss(model, test_loader, device, args.beta)
        test_losses.append(test_loss)
        
        # Plot losses
        plot_losses(all_train_losses, all_recon_losses, all_kl_losses,
                   test_losses, exp_dir, train_loader)
        
        # Save losses to CSV
        train_loss_dict = {
            'iteration': list(range(1, len(all_train_losses) + 1)),
            'total_loss': all_train_losses,
            'recon_loss': all_recon_losses,
            'kl_loss': all_kl_losses
        }
        save_losses_to_csv(train_loss_dict, os.path.join(exp_dir, "train_losses.csv"))
        
        test_loss_dict = {
            'epoch': list(range(1, len(test_losses) + 1)),
            'total_loss': [x['test_loss'] for x in test_losses],
            'recon_loss': [x['test_recon_loss'] for x in test_losses],
            'kl_loss': [x['test_kl_loss'] for x in test_losses]
        }
        save_losses_to_csv(test_loss_dict, os.path.join(exp_dir, "test_losses.csv"))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': all_train_losses,
                'recon_losses': all_recon_losses,
                'kl_losses': all_kl_losses,
                'test_losses': test_losses
            }
            torch.save(checkpoint, os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoints", "model_final.pt"))
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 