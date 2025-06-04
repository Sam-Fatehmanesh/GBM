"""Phase runner for two-phase GBM training."""

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
import time
from typing import Dict, Any, Optional, List

# Import our modules
from GenerativeBrainModel.models.gbm import GBM
from GenerativeBrainModel.datasets.subject_filtered_loader import SubjectFilteredFastDALIBrainDataLoader
from GenerativeBrainModel.training.memory_utils import print_memory_stats
from GenerativeBrainModel.training.schedulers import get_lr_scheduler
from GenerativeBrainModel.evaluation.metrics import track_metrics_during_validation
from GenerativeBrainModel.evaluation.data_saver import save_test_data_and_predictions
from GenerativeBrainModel.custom_functions.visualization import create_prediction_video
from GenerativeBrainModel.utils.file_utils import create_experiment_dir, save_losses_to_csv
from GenerativeBrainModel.utils.data_utils import get_max_z_planes
from GenerativeBrainModel.datasets.probability_data_loader import SubjectFilteredProbabilityDALIBrainDataLoader


class TwoPhaseTrainer:
    """Orchestrates two-phase GBM training: pretrain on multiple subjects, then finetune on target subject."""
    
    def __init__(
        self,
        exp_root: str,
        pretrain_params: Dict[str, Any],
        finetune_params: Dict[str, Any],
        target_subject: Optional[str],
        skip_pretrain: bool = False,
        pretrain_checkpoint: Optional[str] = None,
        pretrain_only_mode: bool = False
    ):
        """Initialize the two-phase trainer.
        
        Args:
            exp_root: Root experiment directory
            pretrain_params: Parameters for pretraining phase
            finetune_params: Parameters for finetuning phase
            target_subject: Name of target subject for finetuning (None for pretrain-only mode)
            skip_pretrain: Whether to skip pretraining phase
            pretrain_checkpoint: Path to existing pretrain checkpoint
            pretrain_only_mode: If True, only run pretraining phase on all subjects
        """
        self.exp_root = exp_root
        self.pretrain_params = pretrain_params
        self.finetune_params = finetune_params
        self.target_subject = target_subject
        self.skip_pretrain = skip_pretrain
        self.pretrain_checkpoint = pretrain_checkpoint
        self.pretrain_only_mode = pretrain_only_mode
        
        # Initialize results
        self.results = {
            'pretrain_checkpoint': None,
            'finetune_checkpoint': None
        }
        
    def run(self) -> Dict[str, Any]:
        """Run the complete two-phase training process.
        
        Returns:
            dict: Training results including checkpoint paths
        """
        # Phase 1: Pretraining
        if not self.skip_pretrain and self.pretrain_checkpoint is None:
            if self.pretrain_only_mode:
                tqdm.write(f"Running pretraining phase on ALL subjects (pretrain-only mode)...")
                self.results['pretrain_checkpoint'] = self._run_phase(
                    phase_name="pretrain",
                    params=self.pretrain_params,
                    subjects_exclude=None,  # Don't exclude any subjects
                    subjects_include=None,
                    init_checkpoint=None
                )
            else:
                tqdm.write(f"Running pretraining phase on all subjects except '{self.target_subject}'...")
                self.results['pretrain_checkpoint'] = self._run_phase(
                    phase_name="pretrain",
                    params=self.pretrain_params,
                    subjects_exclude=[self.target_subject],
                    subjects_include=None,
                    init_checkpoint=None
                )
        elif self.pretrain_checkpoint:
            tqdm.write(f"Skipping pretraining, using provided checkpoint: {self.pretrain_checkpoint}")
            self.results['pretrain_checkpoint'] = self.pretrain_checkpoint
        else:
            tqdm.write(f"Skipping pretraining phase as requested.")
            
        # Phase 2: Finetuning (skip if in pretrain-only mode)
        if not self.pretrain_only_mode:
            tqdm.write(f"Running finetuning phase on target subject '{self.target_subject}'...")
            
            self.results['finetune_checkpoint'] = self._run_phase(
                phase_name="finetune",
                params=self.finetune_params,
                subjects_include=[self.target_subject],
                subjects_exclude=None,
                init_checkpoint=self.results['pretrain_checkpoint']
            )
        else:
            tqdm.write(f"Skipping finetuning phase (pretrain-only mode)")
            self.results['finetune_checkpoint'] = None
        
        return self.results
    
    def _run_phase(
        self,
        phase_name: str,
        params: Dict[str, Any],
        subjects_include: Optional[List[str]] = None,
        subjects_exclude: Optional[List[str]] = None,
        init_checkpoint: Optional[str] = None
    ) -> str:
        """Run a single training phase.
        
        Args:
            phase_name: Name of the phase ('pretrain' or 'finetune')
            params: Parameters for this phase
            subjects_include: List of subjects to include
            subjects_exclude: List of subjects to exclude
            init_checkpoint: Path to checkpoint to initialize from
            
        Returns:
            str: Path to best checkpoint from this phase
        """
        tqdm.write(f"\n{'='*20} Starting {phase_name.upper()} phase {'='*20}\n")
        
        # Create phase directory
        phase_dir = create_experiment_dir(self.exp_root, phase_name)
        tqdm.write(f"Saving {phase_name} results to: {phase_dir}")
        
        print_memory_stats(f"Initial ({phase_name}):")
        
        # Get maximum number of z-planes from preaugmented data
        max_z_planes = get_max_z_planes(params['preaugmented_dir'])
        params['seq_len'] = int(params['timesteps_per_sequence'] * max_z_planes)    
        params['seq_stride'] = int(params['timestep_stride'] * max_z_planes)
        
        tqdm.write(f"Maximum z-planes across subjects: {max_z_planes}")
        tqdm.write(f"Total sequence length: {params['seq_len']} ({params['timesteps_per_sequence']} timepoints × {max_z_planes} z-planes)")
        
        # Create data loaders
        train_loader = self._create_data_loader(
            params, subjects_include, subjects_exclude, split='train', shuffle=True
        )
        test_loader = self._create_data_loader(
            params, subjects_include, subjects_exclude, split='test', shuffle=False
        )
        
        print_memory_stats(f"After data loaders ({phase_name}):")
        
        # Create and initialize model
        model = self._create_model(params, init_checkpoint, phase_dir, phase_name)
        
        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=params['learning_rate'], 
                         weight_decay=params['weight_decay'], betas=(0.9, 0.95))
        
        total_steps = len(train_loader) * params['num_epochs']
        warmup_steps = int(total_steps * params['warmup_ratio'])
        lr_scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=params['min_lr'])
        
        tqdm.write(f"Using AdamW optimizer with weight decay: {params['weight_decay']}")
        tqdm.write(f"Learning rate scheduler: Linear warmup for {warmup_steps} steps, then constant learning rate")
        
        # Create initial prediction video
        tqdm.write(f"Creating initial comparison video for {phase_name}...")
        video_path = os.path.join(phase_dir, 'videos', 'predictions_initial.mp4')
        create_prediction_video(model, test_loader, video_path, num_frames=330)
        
        # Run training
        best_checkpoint_path = self._run_training_loop(
            model, optimizer, lr_scheduler, train_loader, test_loader,
            phase_dir, phase_name, params
        )
        
        # Save test data and predictions if this is the finetuning phase OR pretrain-only mode
        if phase_name == "finetune" or (phase_name == "pretrain" and self.pretrain_only_mode):
            tqdm.write("Saving test data and predictions for analysis...")
            # Directory for test data HDF5
            save_data_dir = os.path.join(phase_dir, 'test_data')
            os.makedirs(save_data_dir, exist_ok=True)
            # Use the original binary data loader to capture sequence z starts
            save_test_data_and_predictions(model, test_loader, save_data_dir, num_samples=100, params=params)
        
        return best_checkpoint_path
    
    def _create_data_loader(self, params, subjects_include, subjects_exclude, split, shuffle):
        """Create a subject-filtered data loader."""
        return SubjectFilteredFastDALIBrainDataLoader(
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
            stride=params['seq_stride']
        )
    
    def _create_model(self, params, init_checkpoint, phase_dir, phase_name):
        """Create and initialize the GBM model."""
        model = GBM(
            mamba_layers=params['mamba_layers'],
            mamba_dim=params['mamba_dim'],
            mamba_state_multiplier=params['mamba_state_multiplier']
        )
        
        # Load from checkpoint if provided
        if init_checkpoint:
            tqdm.write(f"Initializing model from checkpoint: {init_checkpoint}")
            try:
                checkpoint = torch.load(init_checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                tqdm.write("Successfully loaded model weights from checkpoint.")
            except Exception as e:
                tqdm.write(f"Error loading checkpoint: {str(e)}")
                if phase_name == 'finetune':
                    tqdm.write("WARNING: Starting finetuning from scratch!")
        
        # Save model architecture info
        self._save_model_info(model, params, phase_dir, phase_name)
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"Using device: {device}")
        model = model.to(device)
        
        print_memory_stats(f"After model creation ({phase_name}):")
        
        return model
    
    def _save_model_info(self, model, params, phase_dir, phase_name):
        """Save model architecture and parameters to file."""
        with open(os.path.join(phase_dir, "model_architecture.txt"), "w") as f:
            f.write(f"{phase_name.upper()} Phase Model Architecture:\n")
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
    
    def _run_training_loop(self, model, optimizer, lr_scheduler, train_loader, test_loader,
                          phase_dir, phase_name, params):
        """Run the main training loop for a phase."""
        # Scheduled sampling for finetuning: progressively replace ground truth with model predictions in latter half
        scheduled_sampling = params.get('scheduled_sampling', False) and phase_name == 'finetune'
        num_epochs = params.get('num_epochs', 1)
        half_len = params['seq_len'] // 2
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
        
        tqdm.write(f"Starting {phase_name} training...")
        tqdm.write(f"Validating every {validation_interval} batches ({params['validation_per_epoch']} times per epoch)")
        
        # Main training loop
        for epoch in range(params['num_epochs']):
            # Compute replaced_count for scheduled sampling at this epoch
            if scheduled_sampling and num_epochs > 1:
                scheduled_ratio = epoch / float(num_epochs - 1)
                replaced_count = int(scheduled_ratio * half_len)
            else:
                replaced_count = 0
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            train_loader.reset()
            train_iter = iter(train_loader)
            
            train_loop = tqdm(range(len(train_loader)), 
                            desc=f"{phase_name.capitalize()} Epoch {epoch+1}/{params['num_epochs']}")
            
            for batch_idx in train_loop:
                try:
                    # Get batch
                    batch = next(train_iter)
                    if batch.device.type != 'cuda':
                        batch = batch.cuda(non_blocking=True)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    
                    # Scheduled sampling input mixing
                    if scheduled_sampling and replaced_count > 0:
                        # Get model predictions on ground truth without affecting gradients
                        with torch.no_grad():
                            with autocast():
                                logits_gt = model(batch)
                                probs_gt = model.sample_binary_predictions(torch.sigmoid(logits_gt))
                        # Create mixed input: ground truth for first half, predictions for last frames
                        input_batch = batch.clone()
                        seq_len = params['seq_len']
                        tail_start = seq_len - replaced_count
                        # Vectorized replacement: use predictions for the tail portion
                        input_batch[:, tail_start:] = probs_gt[:, tail_start-1:-1]
                        batch_for_model = input_batch
                    else:
                        batch_for_model = batch
                    with autocast():
                        predictions = model(batch_for_model)
                        loss = model.compute_loss(predictions, batch[:, 1:])
                    
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
            
            # Test evaluation
            test_metrics = track_metrics_during_validation(model, test_loader, 'cuda')
            test_losses.append(test_metrics['loss'])
            test_f1_scores.append(test_metrics['f1'])
            test_recall_scores.append(test_metrics['recall'])
            test_precision_scores.append(test_metrics['precision'])
            
            tqdm.write(f"{phase_name.capitalize()} Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                      f"Test Loss = {test_metrics['loss']:.6f}, Test F1 = {test_metrics['f1']:.6f}")
            
            # Create epoch video
            video_path = os.path.join(phase_dir, 'videos', f'predictions_epoch_{epoch+1:03d}_final.mp4')
            create_prediction_video(model, test_loader, video_path, num_frames=330)
            
            # Update plots and save data
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
        }, checkpoint_path)
    
    def _update_plots_and_save_data(self, phase_dir, phase_name, train_losses, test_losses,
                                   raw_batch_losses, validation_step_indices, validation_losses,
                                   validation_f1_scores, validation_recall_scores,
                                   validation_precision_scores, test_f1_scores,
                                   test_recall_scores, test_precision_scores, steps_per_epoch):
        """Update plots and save training data to CSV files."""
        # Create comprehensive plot
        plt.figure(figsize=(12, 12))
        
        # Raw batch losses
        plt.subplot(3, 1, 1)
        plt.plot(raw_batch_losses, 'b-', alpha=0.3)
        plt.title(f'{phase_name.capitalize()} Phase - Raw Batch Losses')
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
        
        plt.title(f'{phase_name.capitalize()} Phase - Training and Validation Losses')
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
        
        plt.title(f'{phase_name.capitalize()} Phase - Performance Metrics (F1, Recall, Precision)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(phase_dir, 'plots', 'loss_plot.png'))
        plt.close()
        
        # Save data to CSV files
        if train_losses and test_losses:
            save_losses_to_csv(
                {'epoch': list(range(1, len(train_losses) + 1)),
                 'train_loss': train_losses,
                 'test_loss': test_losses,
                 'test_f1': test_f1_scores,
                 'test_recall': test_recall_scores,
                 'test_precision': test_precision_scores},
                os.path.join(phase_dir, 'logs', 'losses.csv')
            )
        
        if raw_batch_losses:
            save_losses_to_csv(
                {'batch': list(range(1, len(raw_batch_losses) + 1)),
                 'raw_loss': raw_batch_losses},
                os.path.join(phase_dir, 'logs', 'raw_batch_losses.csv')
            )
        
        if validation_losses:
            save_losses_to_csv(
                {'step': validation_step_indices,
                 'epoch': [step / steps_per_epoch for step in validation_step_indices],
                 'validation_loss': validation_losses,
                 'validation_f1': validation_f1_scores,
                 'validation_recall': validation_recall_scores,
                 'validation_precision': validation_precision_scores},
                os.path.join(phase_dir, 'logs', 'validation_losses.csv')
            ) 