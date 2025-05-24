#!/usr/bin/env python3
"""
Train GBM model with probability targets using a two-phase approach: pretrain on all subjects except the target, 
then finetune on the target subject only.

This script extends the standard GBM training to use spike probability grids as targets instead of binary spikes,
providing richer training signals and potentially better model performance.
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
from GenerativeBrainModel.training.probability_phase_runner import ProbabilityTwoPhaseTrainer

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
    """Main function to run the two-phase training process with probability targets."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train GBM with probability targets using a two-phase approach")
        parser.add_argument("--preaugmented-dir", type=str, default="test_probability_grids", 
                            help="Directory containing preaugmented data with probability grids")
        parser.add_argument("--target-subject", type=str, required=True,
                            help="Name of the target subject to hold out for finetuning")
        parser.add_argument("--num-epochs-pretrain", type=int, default=1,
                            help="Number of epochs for pretraining phase")
        parser.add_argument("--num-epochs-finetune", type=int, default=1, 
                            help="Number of epochs for finetuning phase")
        parser.add_argument("--batch-size", type=int, default=128,
                            help="Batch size for both phases")
        parser.add_argument("--learning-rate", type=float, default=1e-3,
                            help="Learning rate for both phases")
        parser.add_argument("--skip-pretrain", action="store_true",
                            help="Skip the pretrain phase and go directly to finetuning")
        parser.add_argument("--pretrain-checkpoint", type=str, default=None,
                            help="Path to a pretrained checkpoint to start from (skips pretrain phase)")
        parser.add_argument("--use-probabilities", action="store_true", default=True,
                            help="Use probability grids as targets (default: True)")
        parser.add_argument("--use-binary", action="store_true",
                            help="Use binary spike grids as targets instead of probabilities")
        parser.add_argument("--enable-memory-diagnostics", action="store_true",
                            help="Enable memory diagnostics during training")
        args = parser.parse_args()
        
        # Handle probability vs binary targets
        if args.use_binary:
            use_probabilities = False
        else:
            use_probabilities = args.use_probabilities
        
        # Enable memory diagnostics if requested
        if args.enable_memory_diagnostics:
            enable_memory_diagnostics(True)
            print_memory_stats("Initial:")
        
        # Validate target subject exists
        validate_subject_directory(args.preaugmented_dir, args.target_subject)
        tqdm.write(f"Target subject validated: {args.target_subject}")
        
        # Base parameters for both phases
        base_params = {
            'preaugmented_dir': args.preaugmented_dir,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': 0.1,
            'warmup_ratio': 0.1,
            'min_lr': 1e-5,
            'mamba_layers': 8,
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
            'use_probabilities': use_probabilities,
        }
        
        # Set phase-specific parameters
        pretrain_params = base_params.copy()
        pretrain_params['num_epochs'] = args.num_epochs_pretrain
        
        finetune_params = base_params.copy()
        finetune_params['num_epochs'] = args.num_epochs_finetune
        
        # Create timestamped experiment root directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_type = "prob" if use_probabilities else "binary"
        exp_root = os.path.join('experiments', 'gbm_probability', f"{timestamp}_{target_type}")
        os.makedirs(exp_root, exist_ok=True)
        
        # Save experiment metadata
        experiment_metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_type': 'Two-Phase GBM Training with Probability Targets',
            'target_subject': args.target_subject,
            'pretrain_epochs': args.num_epochs_pretrain,
            'finetune_epochs': args.num_epochs_finetune,
            'skip_pretrain': args.skip_pretrain,
            'pretrain_checkpoint': args.pretrain_checkpoint,
            'use_probabilities': use_probabilities,
            'target_type': 'Probability grids (continuous)' if use_probabilities else 'Binary spike grids',
            'parameters': base_params
        }
        save_experiment_metadata(exp_root, experiment_metadata)
        
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"PROBABILITY TARGET TRAINING")
        tqdm.write(f"{'='*50}")
        tqdm.write(f"Target type: {'Probability grids (continuous)' if use_probabilities else 'Binary spike grids'}")
        tqdm.write(f"Target subject: {args.target_subject}")
        tqdm.write(f"Experiment directory: {exp_root}")
        tqdm.write(f"{'='*50}\n")
        
        # Create and run the probability-aware two-phase trainer
        trainer = ProbabilityTwoPhaseTrainer(
            exp_root=exp_root,
            pretrain_params=pretrain_params,
            finetune_params=finetune_params,
            target_subject=args.target_subject,
            skip_pretrain=args.skip_pretrain,
            pretrain_checkpoint=args.pretrain_checkpoint,
            use_probabilities=use_probabilities
        )
        
        # Run the training
        results = trainer.run()
        
        # Print final summary
        tqdm.write("\n" + "="*50)
        tqdm.write("PROBABILITY TARGET TRAINING COMPLETE!")
        tqdm.write(f"Target type: {'Probability grids (continuous)' if use_probabilities else 'Binary spike grids'}")
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