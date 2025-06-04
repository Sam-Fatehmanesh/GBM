#!/usr/bin/env python3
"""
Train GBM model with a two-phase approach: pretrain on all subjects except the target, 
then finetune on the target subject only.

This is a refactored version of the original script with improved modularity.
"""

# Force Qt to use offscreen platform
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Set maximum memory allocations for DALI
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["DALI_PREALLOCATE_WIDTH"] = "4096" 
os.environ["DALI_PREALLOCATE_HEIGHT"] = "4096"
os.environ["DALI_PREALLOCATE_DEPTH"] = "8"
os.environ["DALI_TENSOR_ALLOCATOR_BLOCK_SIZE"] = str(512*1024*1024)  # 512MB blocks

import torch
import torch.multiprocessing as mp
import argparse
from datetime import datetime
from tqdm import tqdm

# Set sharing strategy to file_system to avoid shared memory issues
mp.set_sharing_strategy('file_system')
# Set the start method to spawn to avoid CUDA initialization issues
mp.set_start_method('spawn', force=True)

import numpy as np

# Import our refactored modules
from GenerativeBrainModel.training import print_memory_stats, enable_memory_diagnostics
from GenerativeBrainModel.utils import validate_subject_directory, save_experiment_metadata
from GenerativeBrainModel.training.phase_runner import TwoPhaseTrainer

# Sets torch seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Initialize CUDA after importing DALI
torch.cuda.init()
# Pre-allocate CUDA memory to avoid fragmentation
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()

# Enable tensor cores for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Enable memory optimizations
torch.backends.cudnn.benchmark = True


def main():
    """Main function to run the two-phase training process."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train GBM with a two-phase approach: pretrain on all subjects except target, then finetune on target. If no target subject is specified, only pretrain on all subjects only.")
        parser.add_argument("--preaugmented-dir", type=str, default="preaugmented_training_spike_data_2018", 
                            help="Directory containing preaugmented data")
        parser.add_argument("--target-subject", type=str, default=None,
                            help="Name of the target subject to hold out for finetuning. If not specified, pretrain on all subjects only.")
        parser.add_argument("--num-epochs-pretrain", type=int, default=1,
                            help="Number of epochs for pretraining phase")
        parser.add_argument("--num-epochs-finetune", type=int, default=1, 
                            help="Number of epochs for finetuning phase (ignored if no target subject)")
        parser.add_argument("--batch-size", type=int, default=128,
                            help="Batch size for both phases")
        parser.add_argument("--learning-rate", type=float, default=6e-4,
                            help="Learning rate for both phases")
        parser.add_argument("--skip-pretrain", action="store_true",
                            help="Skip the pretrain phase and go directly to finetuning (requires target subject)")
        parser.add_argument("--scheduled-sampling", dest="scheduled_sampling", action="store_true",
                            help="Enable scheduled sampling during finetuning, replacing more input with model predictions over epochs")
        parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                            help="Path to a pretrained checkpoint to start from (skips pretrain phase)")
        parser.add_argument("--enable-memory-diagnostics", action="store_true",
                            help="Enable memory diagnostics during training")
        args = parser.parse_args()
        
        # Determine training mode
        pretrain_only_mode = args.target_subject is None
        
        # Validation for pretrain-only mode
        if pretrain_only_mode:
            if args.skip_pretrain:
                raise ValueError("Cannot skip pretraining when no target subject is specified (pretrain-only mode)")
            tqdm.write("PRETRAIN-ONLY MODE: Training on all subjects, no finetuning phase")
        else:
            # Validate target subject exists (only if specified)
            validate_subject_directory(args.preaugmented_dir, args.target_subject)
            tqdm.write(f"TWO-PHASE MODE: Target subject validated: {args.target_subject}")
        
        # Enable memory diagnostics if requested
        if args.enable_memory_diagnostics:
            enable_memory_diagnostics(True)
            print_memory_stats("Initial:")
        
        # Base parameters for both phases
        base_params = {
            'preaugmented_dir': args.preaugmented_dir,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': 0.1,
            'warmup_ratio': 0.1,
            'min_lr': 1e-5,
            'mamba_layers': 2,
            'mamba_dim': 1024,
            'mamba_state_multiplier': 8,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95,
            'dali_num_threads': 2,
            'gpu_prefetch': 1,
            'use_float16': False,
            'seed': seed,
            'validation_per_epoch': 8,
            'timestep_stride': 1/3,
        }
        
        # Set phase-specific parameters
        pretrain_params = base_params.copy()
        pretrain_params['num_epochs'] = args.num_epochs_pretrain
        
        finetune_params = base_params.copy()
        finetune_params['num_epochs'] = args.num_epochs_finetune
        finetune_params['scheduled_sampling'] = args.scheduled_sampling
        
        # Create timestamped experiment root directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if pretrain_only_mode:
            exp_root = os.path.join('experiments', 'gbm', f"{timestamp}_pretrain_only")
        else:
            exp_root = os.path.join('experiments', 'gbm', timestamp)
        os.makedirs(exp_root, exist_ok=True)
        
        # Save experiment metadata
        if pretrain_only_mode:
            experiment_metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_type': 'Pretrain-Only GBM Training',
                'target_subject': None,
                'pretrain_epochs': args.num_epochs_pretrain,
                'finetune_epochs': 0,
                'skip_pretrain': False,
                'pretrain_checkpoint': args.pretrain_checkpoint,
                'parameters': base_params
            }
        else:
            experiment_metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_type': 'Two-Phase GBM Training',
                'target_subject': args.target_subject,
                'pretrain_epochs': args.num_epochs_pretrain,
                'finetune_epochs': args.num_epochs_finetune,
                'skip_pretrain': args.skip_pretrain,
                'pretrain_checkpoint': args.pretrain_checkpoint,
                'parameters': base_params
            }
        save_experiment_metadata(exp_root, experiment_metadata)
        
        # Create and run the trainer
        trainer = TwoPhaseTrainer(
            exp_root=exp_root,
            pretrain_params=pretrain_params,
            finetune_params=finetune_params,
            target_subject=args.target_subject,
            skip_pretrain=args.skip_pretrain,
            pretrain_checkpoint=args.pretrain_checkpoint,
            pretrain_only_mode=pretrain_only_mode
        )
        
        # Run the training
        results = trainer.run()
        
        # Print final summary
        tqdm.write("\n" + "="*50)
        if pretrain_only_mode:
            tqdm.write("Pretrain-only training complete!")
            tqdm.write(f"Experiment directory: {exp_root}")
            tqdm.write(f"Best pretrain checkpoint: {results['pretrain_checkpoint']}")
        else:
            tqdm.write("Two-phase training complete!")
            tqdm.write(f"Experiment directory: {exp_root}")
            if results['pretrain_checkpoint']:
                tqdm.write(f"Best pretrain checkpoint: {results['pretrain_checkpoint']}")
            tqdm.write(f"Best finetune checkpoint: {results['finetune_checkpoint']}")
        tqdm.write("="*50)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main() 