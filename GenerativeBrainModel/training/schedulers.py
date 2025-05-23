"""Learning rate scheduling utilities."""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    """Creates a learning rate scheduler with linear warmup and cosine decay to specified min_lr.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate to decay to
    
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        # Get base lr
        base_lr = optimizer.param_groups[0]['lr']
        
        # Convert min_lr to a fraction of base_lr for the lambda function
        # Prevent division by zero by ensuring base_lr is not zero
        min_lr_factor = min_lr / max(base_lr, 1e-8)
        
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return 1.0
        
        # Cosine annealing after warmup with decay to min_lr
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between 1.0 and min_lr_factor instead of between 1.0 and 0.0
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch, min_lr_ratio=0.01):
    """Create a scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        min_lr_ratio: Minimum learning rate as a ratio of initial lr
        
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(warmup_steps)
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda) 