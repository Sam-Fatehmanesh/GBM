#!/usr/bin/env python3
"""
Train GBM model with a two-phase approach: pretrain on all subjects except the target, 
then finetune on the target subject only.

This script now uses probability data loaders by default instead of binary spike data.
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
import json

# Import our refactored modules
from GenerativeBrainModel.training import print_memory_stats, enable_memory_diagnostics
from GenerativeBrainModel.utils import validate_subject_directory, save_experiment_metadata
from GenerativeBrainModel.training.phase_runner import MultiPhaseTrainer

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
        parser.add_argument("--preaugmented-dir", type=str, default="processed_spike_grids_2018_aug_prob_cascade", 
                            help="Directory containing preaugmented probability data")
        parser.add_argument("--target-subject", type=str, default=None,
                            help="Name of the target subject to hold out for finetuning. If not specified, pretrain on all subjects only.")
        parser.add_argument("--num-epochs-pretrain", type=int, default=1,
                            help="Number of epochs for pretraining phase")
        parser.add_argument("--num-epochs-finetune", type=int, default=1, 
                            help="Number of epochs for finetuning phase (ignored if no target subject)")
        parser.add_argument("--batch-size", type=int, default=64,
                            help="Batch size for both phases")
        parser.add_argument("--learning-rate", type=float, default=1e-3,
                            help="Learning rate for both phases")
        parser.add_argument("--skip-pretrain", action="store_true",
                            help="Skip the pretrain phase and go directly to finetuning (requires target subject)")
        parser.add_argument("--scheduled-sampling", dest="scheduled_sampling", action="store_true",
                            help="Enable scheduled sampling during finetuning, replacing more input with model predictions over epochs")
        parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                            help="Path to a pretrained checkpoint to start from (skips pretrain phase)")
        parser.add_argument("--enable-memory-diagnostics", action="store_true",
                            help="Enable memory diagnostics during training")
        parser.add_argument("--phase-config", type=str, default=None,
                           help="Path to JSON file defining multiple phases")
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
            'mamba_dim': 2048,
            'mamba_state_multiplier': 8,
            'timesteps_per_sequence': 10,
            'train_ratio': 0.95,
            'dali_num_threads': 12,
            'gpu_prefetch': 4,
            'use_float16': False,
            'seed': seed,
            'validation_per_epoch': 8,
            'timestep_stride': 5/8,
        }
        
        if args.phase_config:
            with open(args.phase_config, 'r') as f:
                phase_configs = json.load(f)
            for pc in phase_configs:
                params = base_params.copy()
                params.update(pc.get('params', {}))
                pc['params'] = params
            experiment_type = 'Multi-Phase GBM Training (Probability Data)'
            tqdm.write(f"Loaded {len(phase_configs)} phases from {args.phase_config}")
        else:
            pretrain_params = base_params.copy()
            pretrain_params['num_epochs'] = args.num_epochs_pretrain
            pretrain_params['scheduled_sampling'] = False
            
            finetune_params = base_params.copy()
            finetune_params['num_epochs'] = args.num_epochs_finetune
            finetune_params['scheduled_sampling'] = args.scheduled_sampling
            
            if pretrain_only_mode:
                phase_configs = [
                    {'name': 'pretrain', 'subjects': None, 'exclude': None, 'params': pretrain_params}
                ]
                experiment_type = 'Pretrain-Only GBM Training (Probability Data)'
            else:
                phase_configs = [
                    {'name': 'pretrain', 'subjects': None, 'exclude': [args.target_subject], 'params': pretrain_params},
                    {'name': 'finetune', 'subjects': [args.target_subject], 'params': finetune_params}
                ]
                experiment_type = 'Two-Phase GBM Training (Probability Data)'
                if args.skip_pretrain:
                    phase_configs = phase_configs[1:]
 
        # Create timestamped experiment root directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if pretrain_only_mode:
            exp_root = os.path.join('experiments', 'gbm', f"{timestamp}_pretrain_only")
        else:
            exp_root = os.path.join('experiments', 'gbm', timestamp)
        os.makedirs(exp_root, exist_ok=True)
        
        # Save experiment metadata
        experiment_metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_type': experiment_type,
            'target_subject': args.target_subject if not args.phase_config else None,
            'skip_pretrain': args.skip_pretrain if not args.phase_config else False,
            'pretrain_checkpoint': args.pretrain_checkpoint,
            'data_type': 'probability',
            'parameters': base_params,
            'phases': [{k: v for k, v in pc.items() if k != 'params'} for pc in phase_configs]
        }
        save_experiment_metadata(exp_root, experiment_metadata)
        
        # Create and run the trainer
        initial_checkpoint = args.pretrain_checkpoint if args.skip_pretrain and args.pretrain_checkpoint else None
        trainer = MultiPhaseTrainer(
           exp_root=exp_root,
           phase_configs=phase_configs,
           initial_checkpoint=initial_checkpoint
        )
        
        # Run the training
        results = trainer.run()
        
        # Print final summary
        tqdm.write("\n" + "="*50)
        tqdm.write(f"{experiment_type} complete!")
        tqdm.write(f"Experiment directory: {exp_root}")
        for phase_name, checkpoint in results.items():
            tqdm.write(f"Best {phase_name}: {checkpoint}")
        tqdm.write("="*50)
        
    except Exception as e:
        tqdm.write(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main() 