"""Probability-aware phase runner for two-phase GBM training with probability targets."""

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environment  
import matplotlib.pyplot as plt
import gc
import time
from typing import Dict, Any, Optional, List

# Import our modules
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.probability_data_loader import SubjectFilteredProbabilityDALIBrainDataLoader
from GenerativeBrainModel.training.memory_utils import print_memory_stats
from GenerativeBrainModel.training.schedulers import get_lr_scheduler
from GenerativeBrainModel.evaluation.metrics import track_metrics_during_validation
from GenerativeBrainModel.evaluation.data_saver import save_test_data_and_predictions
from GenerativeBrainModel.custom_functions.visualization import create_prediction_video
from GenerativeBrainModel.utils.file_utils import create_experiment_dir, save_losses_to_csv
from GenerativeBrainModel.utils.data_utils import get_max_z_planes
from .phase_runner import TwoPhaseTrainer


class ProbabilityTwoPhaseTrainer(TwoPhaseTrainer):
    """Orchestrates two-phase GBM training with probability targets: pretrain on multiple subjects, then finetune on target subject."""
    
    def __init__(
        self,
        exp_root: str,
        pretrain_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        target_subject: str,
        skip_pretrain: bool = False,
        pretrain_checkpoint: Optional[str] = None,
        use_probabilities: bool = True
    ):
        """Initialize the probability-aware two-phase trainer.
        
        Args:
            exp_root: Root experiment directory
            pretrain_params: Parameters for pretraining phase
            finetune_params: Parameters for finetuning phase
            target_subject: Name of target subject for finetuning
            skip_pretrain: Whether to skip pretraining phase
            pretrain_checkpoint: Path to existing pretrain checkpoint
            use_probabilities: If True, use probability grids as targets. If False, use binary spikes.
        """
        # Initialize parent class
        super().__init__(
            exp_root=exp_root,
            pretrain_params=pretrain_params,
            finetune_params=finetune_params,
            target_subject=target_subject,
            skip_pretrain=skip_pretrain,
            pretrain_checkpoint=pretrain_checkpoint
        )
        
        # Store probability-specific parameters
        self.use_probabilities = use_probabilities
        
        # Add probability info to both phase parameters
        self.pretrain_params['use_probabilities'] = use_probabilities
        self.finetune_params['use_probabilities'] = use_probabilities
        
        tqdm.write(f"Probability Training Mode: {'ENABLED' if use_probabilities else 'DISABLED (using binary spikes)'}")
        
    def _create_data_loader(self, params, subjects_include, subjects_exclude, split, shuffle):
        """Create a subject-filtered probability data loader."""
        return SubjectFilteredProbabilityDALIBrainDataLoader(
            params['preaugmented_dir'],
            include_subjects=subjects_include,
            exclude_subjects=subjects_exclude,
            batch_size=params['batch_size'],
            seq_len=params['seq_len'],
            split=split,
            device_id=0,
            num_threads=params['dali_num_threads'],
            gpu_prefetch=params['gpu_prefetch'],
            seed=42 if split == 'train' else 43,
            shuffle=shuffle,
            stride=params['seq_stride'],
            use_probabilities=self.use_probabilities
        )
    
    def _save_model_info(self, model, params, phase_dir, phase_name):
        """Save model architecture and parameters to file."""
        with open(os.path.join(phase_dir, "model_architecture.txt"), "w") as f:
            f.write(f"{phase_name.upper()} Phase Model Architecture (Probability Targets):\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model))
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("Model Parameters:\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            
            # Add probability-specific information
            f.write(f"\nProbability Training Configuration:\n")
            f.write(f"use_probabilities: {self.use_probabilities}\n")
            if self.use_probabilities:
                f.write(f"Target type: Continuous probability values (0.0 to ~30.0)\n")
                f.write(f"Loss function: BCE with logits (supports continuous targets)\n")
            else:
                f.write(f"Target type: Binary spike values (0.0 or 1.0)\n")
                f.write(f"Loss function: BCE with logits (binary targets)\n")
            
            # Add statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\nModel Statistics:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
    
    def _run_training_loop(self, model, optimizer, lr_scheduler, train_loader, test_loader,
                          phase_dir, phase_name, params):
        """Run the main training loop for a phase with probability targets."""
        # Training state
        train_losses = []
        test_losses = []
        raw_batch_losses = []
        best_test_loss = float('inf')
        
        # Validation tracking
        validation_step_indices = []
        validation_losses = []
        validation_f1_scores = []
        validation_recall_scores = []
        validation_precision_scores = []
        
        # Test metrics (end of epoch)
        test_f1_scores = []
        test_recall_scores = []
        test_precision_scores = []
        
        # Training setup
        scaler = GradScaler()
        validation_interval = len(train_loader) // params['validation_per_epoch']
        quarter_epoch_size = len(train_loader) // 4
        best_model_checkpoint_path = os.path.join(phase_dir, 'checkpoints', 'best_model.pt')
        
        target_type = "probability" if self.use_probabilities else "binary"
        
        # Debug logging for data loader size
        tqdm.write(f"DEBUG: train_loader length = {len(train_loader)}")
        tqdm.write(f"DEBUG: validation_per_epoch = {params['validation_per_epoch']}")
        tqdm.write(f"DEBUG: calculated validation_interval = {validation_interval}")
        tqdm.write(f"DEBUG: quarter_epoch_size = {quarter_epoch_size}")
        
        tqdm.write(f"Starting {phase_name} training with {target_type} targets...")
        tqdm.write(f"Validating every {validation_interval} batches ({params['validation_per_epoch']} times per epoch)")
        tqdm.write(f"Plots and CSV files will update at each validation step and end of epoch")
        
        # Main training loop
        for epoch in range(params['num_epochs']):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            train_loader.reset()
            train_iter = iter(train_loader)
            
            train_loop = tqdm(range(len(train_loader)), 
                            desc=f"{phase_name.capitalize()} Epoch {epoch+1}/{params['num_epochs']} ({target_type})")
            
            tqdm.write(f"Starting epoch {epoch+1}, expecting {len(train_loader)} batches...")
            
            for batch_idx in train_loop:
                try:
                    # Get batch
                    batch = next(train_iter)
                    
                    # Extract input and target data based on training mode
                    if self.use_probabilities:
                        # Probability mode: batch is tuple of (input_data, target_data)
                        input_data, target_data = batch
                        if input_data.device.type != 'cuda':
                            input_data = input_data.cuda(non_blocking=True)
                        if target_data.device.type != 'cuda':
                            target_data = target_data.cuda(non_blocking=True)
                        # Apply temporal shift for next-frame prediction
                        input_sequences = input_data  # Input: frames 0 to T-1
                        target_sequences = target_data[:, 1:]  # Target: frames 1 to T
                    else:
                        # Binary mode: batch is single tensor
                        if batch.device.type != 'cuda':
                            batch = batch.cuda(non_blocking=True)
                        input_data = batch
                        target_data = batch[:, 1:]
                        input_sequences = input_data
                        target_sequences = target_data
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    with autocast():
                        predictions = model(input_sequences)
                        loss = model.compute_loss(predictions, target_sequences)
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update learning rate
                    lr_scheduler.step()
                    
                    # Track metrics
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    batch_count += 1
                    raw_batch_losses.append(current_loss)
                    
                    # Update progress bar
                    train_loop.set_postfix(loss=current_loss)
                    
                    # Clean up
                    del loss, predictions, batch
                    
                    # Validation
                    if (batch_idx + 1) % validation_interval == 0 or batch_idx == len(train_loader) - 1:
                        global_step = epoch * len(train_loader) + batch_idx
                        val_metrics = track_metrics_during_validation(model, test_loader, 'cuda')
                        
                        # Record validation metrics
                        validation_step_indices.append(global_step)
                        validation_losses.append(val_metrics['loss'])
                        validation_f1_scores.append(val_metrics['f1'])
                        validation_recall_scores.append(val_metrics['recall'])
                        validation_precision_scores.append(val_metrics['precision'])
                        
                        tqdm.write(f"Validation at step {global_step}: Loss={val_metrics['loss']:.6f}, "
                                  f"F1={val_metrics['f1']:.6f}, Recall={val_metrics['recall']:.6f}, "
                                  f"Precision={val_metrics['precision']:.6f}")
                        
                        # Save best model
                        if val_metrics['loss'] < best_test_loss:
                            best_test_loss = val_metrics['loss']
                            self._save_checkpoint(model, optimizer, lr_scheduler, epoch+1,
                                               train_losses, test_losses, validation_losses,
                                               validation_step_indices, raw_batch_losses,
                                               params, best_test_loss, phase_name,
                                               best_model_checkpoint_path)
                        
                        # Update plots and save data at validation time
                        tqdm.write(f"Updating plots and CSV at validation step {global_step}")
                        
                        # Get current epoch losses for incomplete epoch  
                        current_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                        temp_train_losses = train_losses + [current_train_loss] if train_losses or current_train_loss > 0 else []
                        
                        # Update plots and save data with current data
                        self._update_plots_and_save_data(
                            phase_dir, phase_name, temp_train_losses, test_losses, raw_batch_losses,
                            validation_step_indices, validation_losses, validation_f1_scores,
                            validation_recall_scores, validation_precision_scores,
                            test_f1_scores, test_recall_scores, test_precision_scores,
                            len(train_loader)
                        )
                        
                        # Switch back to training mode
                        model.train()
                    
                    # Generate comparison video every quarter epoch
                    if (batch_idx + 1) % quarter_epoch_size == 0:
                        quarter = (batch_idx + 1) // quarter_epoch_size
                        video_path = os.path.join(phase_dir, 'videos', 
                                                f'predictions_epoch_{epoch+1:03d}_quarter_{quarter}.mp4')
                        create_prediction_video(model, test_loader, video_path, num_frames=330)
                    
                    # Memory cleanup
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except StopIteration:
                    break
            
            # End of epoch evaluation
            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            tqdm.write(f"EPOCH {epoch+1} COMPLETED! Processed {batch_count} batches. Starting end-of-epoch evaluation...")
            
            # Test evaluation
            test_metrics = track_metrics_during_validation(model, test_loader, 'cuda')
            test_losses.append(test_metrics['loss'])
            test_f1_scores.append(test_metrics['f1'])
            test_recall_scores.append(test_metrics['recall'])
            test_precision_scores.append(test_metrics['precision'])
            
            tqdm.write(f"{phase_name.capitalize()} Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                      f"Test Loss = {test_metrics['loss']:.6f}, Test F1 = {test_metrics['f1']:.6f}")
            
            # Create epoch video
            tqdm.write(f"Creating epoch {epoch+1} final video...")
            video_path = os.path.join(phase_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_final.mp4')
            create_prediction_video(model, test_loader, video_path, num_frames=330)
            
            # Update plots and save data
            tqdm.write(f"Starting plot and CSV generation for epoch {epoch+1}...")
            self._update_plots_and_save_data(
                phase_dir, phase_name, train_losses, test_losses, raw_batch_losses,
                validation_step_indices, validation_losses, validation_f1_scores,
                validation_recall_scores, validation_precision_scores,
                test_f1_scores, test_recall_scores, test_precision_scores,
                len(train_loader)
            )
        
        # Save final model
        final_model_path = os.path.join(phase_dir, 'checkpoints', 'final_model.pt')
        self._save_checkpoint(model, optimizer, lr_scheduler, params['num_epochs'],
                           train_losses, test_losses, validation_losses,
                           validation_step_indices, raw_batch_losses, params,
                           avg_train_loss, phase_name, final_model_path)
        
        # Create final prediction video
        video_path = os.path.join(phase_dir, 'videos', 'final_predictions.mp4')
        create_prediction_video(model, test_loader, video_path, num_frames=330)
        
        tqdm.write(f"{phase_name.capitalize()} training complete!")
        
        return best_model_checkpoint_path
    
    def _save_checkpoint(self, model, optimizer, lr_scheduler, epoch, train_losses,
                        test_losses, validation_losses, validation_steps, raw_batch_losses,
                        params, loss_value, phase_name, checkpoint_path):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'validation_losses': validation_losses,
            'validation_steps': validation_steps,
            'raw_batch_losses': raw_batch_losses,
            'params': params,
            'loss_value': loss_value,
            'phase': phase_name,
            'use_probabilities': self.use_probabilities,  # Add probability flag
        }, checkpoint_path)
    
    def _update_plots_and_save_data(self, phase_dir, phase_name, train_losses, test_losses,
                                   raw_batch_losses, validation_step_indices, validation_losses,
                                   validation_f1_scores, validation_recall_scores,
                                   validation_precision_scores, test_f1_scores,
                                   test_recall_scores, test_precision_scores, steps_per_epoch):
        """Update plots and save training data to CSV files."""
        target_type = "Probability" if self.use_probabilities else "Binary"
        
        # Helper function to convert numpy arrays to Python floats
        def to_python_floats(data_list):
            """Convert numpy values to Python floats for pandas compatibility."""
            if not data_list:
                return []
            return [float(x) for x in data_list]
        
        # STEP 1: Save plots first (so they don't get lost if CSV fails)
        try:
            plt.figure(figsize=(12, 12))
            
            # Raw batch losses
            plt.subplot(3, 1, 1)
            plt.plot(raw_batch_losses, 'b-', alpha=0.3)
            plt.title(f'{phase_name.capitalize()} Phase - Raw Batch Losses ({target_type} Targets)')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            
            # Training and validation losses
            plt.subplot(3, 1, 2)
            if train_losses:
                epochs = list(range(1, len(train_losses) + 1))
                plt.plot(epochs, train_losses, 'b-', label='Train Loss (epoch)')
            if test_losses:
                epochs = list(range(1, len(test_losses) + 1))
                plt.plot(epochs, test_losses, 'r-', label='Test Loss (epoch)')
            if validation_losses:
                epoch_fractions = [step / steps_per_epoch for step in validation_step_indices]
                plt.plot(epoch_fractions, validation_losses, 'g--', marker='o', label='Validation Loss (frequent)')
            
            plt.title(f'{phase_name.capitalize()} Phase - Training and Validation Losses ({target_type} Targets)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Performance metrics
            plt.subplot(3, 1, 3)
            if validation_f1_scores:
                epoch_fractions = [step / steps_per_epoch for step in validation_step_indices]
                plt.plot(epoch_fractions, validation_f1_scores, 'm--', marker='x', label='Validation F1')
                plt.plot(epoch_fractions, validation_recall_scores, 'g--', marker='o', label='Validation Recall')
                plt.plot(epoch_fractions, validation_precision_scores, 'b--', marker='^', label='Validation Precision')
            
            if test_f1_scores:
                epochs = list(range(1, len(test_f1_scores) + 1))
                plt.plot(epochs, test_f1_scores, 'k-', marker='s', label='Test F1')
                plt.plot(epochs, test_recall_scores, 'r-', marker='d', label='Test Recall')
                plt.plot(epochs, test_precision_scores, 'c-', marker='v', label='Test Precision')
            
            plt.title(f'{phase_name.capitalize()} Phase - Performance Metrics ({target_type} Targets)')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(phase_dir, 'plots', 'loss_plot.png')
            plt.savefig(plot_path)
            plt.close()
            tqdm.write(f"Plot saved: {plot_path}")
        except Exception as e:
            tqdm.write(f"Plot saving failed: {e}")
        
        # STEP 2: Save CSV files with proper error handling and data conversion
        
        # Main losses CSV
        if train_losses and test_losses:
            try:
                losses_data = {
                    'epoch': list(range(1, len(train_losses) + 1)),
                    'train_loss': to_python_floats(train_losses),
                    'test_loss': to_python_floats(test_losses),
                    'test_f1': to_python_floats(test_f1_scores),
                    'test_recall': to_python_floats(test_recall_scores),
                    'test_precision': to_python_floats(test_precision_scores)
                }
                csv_path = os.path.join(phase_dir, 'logs', 'losses.csv')
                save_losses_to_csv(losses_data, csv_path)
                tqdm.write(f"Main losses CSV saved: {csv_path}")
            except Exception as e:
                tqdm.write(f"Main losses CSV failed: {e}")
                tqdm.write(f"   train_losses length: {len(train_losses) if train_losses else 0}")
                tqdm.write(f"   test_losses length: {len(test_losses) if test_losses else 0}")
                tqdm.write(f"   test_f1_scores length: {len(test_f1_scores) if test_f1_scores else 0}")
        
        # Raw batch losses CSV
        if raw_batch_losses:
            try:
                batch_data = {
                    'batch': list(range(1, len(raw_batch_losses) + 1)),
                    'raw_loss': to_python_floats(raw_batch_losses)
                }
                csv_path = os.path.join(phase_dir, 'logs', 'raw_batch_losses.csv')
                save_losses_to_csv(batch_data, csv_path)
                tqdm.write(f"Raw batch losses CSV saved: {csv_path}")
            except Exception as e:
                tqdm.write(f"Raw batch losses CSV failed: {e}")
                tqdm.write(f"   raw_batch_losses length: {len(raw_batch_losses)}")
        
        # Validation losses CSV
        if validation_losses:
            try:
                validation_data = {
                    'step': validation_step_indices,
                    'epoch': [float(step / steps_per_epoch) for step in validation_step_indices],
                    'validation_loss': to_python_floats(validation_losses),
                    'validation_f1': to_python_floats(validation_f1_scores),
                    'validation_recall': to_python_floats(validation_recall_scores),
                    'validation_precision': to_python_floats(validation_precision_scores)
                }
                csv_path = os.path.join(phase_dir, 'logs', 'validation_losses.csv')
                save_losses_to_csv(validation_data, csv_path)
                tqdm.write(f"Validation losses CSV saved: {csv_path}")
            except Exception as e:
                tqdm.write(f"Validation losses CSV failed: {e}")
                tqdm.write(f"   validation_losses length: {len(validation_losses)}")
                tqdm.write(f"   validation_step_indices length: {len(validation_step_indices)}")
        
        tqdm.write(f"Plot and data saving complete for {phase_name} phase ({target_type} targets)") 