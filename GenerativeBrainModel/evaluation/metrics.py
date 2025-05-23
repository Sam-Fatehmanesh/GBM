"""Metrics calculation utilities for model evaluation."""

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def calculate_binary_metrics(predictions, targets):
    """Calculate binary classification metrics (F1, recall, precision).
    
    Args:
        predictions: Binary predictions tensor
        targets: Binary targets tensor
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Convert to boolean tensors for memory efficiency
    if isinstance(predictions, torch.Tensor):
        preds = (predictions > 0.5).bool()
    else:
        preds = predictions.bool()
        
    if isinstance(targets, torch.Tensor):
        targets = targets.bool()
    else:
        targets = targets.bool()
    
    # Calculate confusion matrix components
    tp = (preds & targets).sum().item()
    fp = (preds & ~targets).sum().item()
    fn = (~preds & targets).sum().item()
    tn = (~preds & ~targets).sum().item()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def track_metrics_during_validation(model, data_loader, device):
    """Run validation and calculate metrics across all batches.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to run evaluation on
        
    Returns:
        dict: Average metrics across all validation batches
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Reset data loader
    data_loader.reset()
    test_iter = iter(data_loader)
    
    with torch.no_grad():
        for batch_idx in range(len(data_loader)):
            try:
                # Get batch with automatic GPU transfer
                batch = next(test_iter)
                
                # Ensure batch is on GPU
                if batch.device.type != 'cuda':
                    batch = batch.cuda(non_blocking=True)
                
                # Get logits from model (no sigmoid)
                with autocast():
                    predictions = model(batch)
                    batch_loss = model.compute_loss(predictions, batch[:, 1:])
                
                total_loss += batch_loss.item()
                batch_count += 1
                del batch_loss
                
                # Calculate metrics using sigmoid on predictions
                probs = torch.sigmoid(predictions)
                del predictions
                # Keep predictions as bool to save memory
                preds = (probs > 0.5)
                del probs
                # Keep targets as bool as well
                targets = batch[:, 1:].bool()
                del batch
                torch.cuda.empty_cache()
                
                # Use boolean operations for memory efficiency
                total_tp += (preds & targets).sum().item()
                total_fp += (preds & ~targets).sum().item()
                total_fn += (~preds & targets).sum().item()
                # Clean up
                del preds, targets
                
            except StopIteration:
                break
    
    # Calculate average metrics
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    
    # Calculate F1, recall, precision
    if total_tp + total_fp + total_fn > 0:
        f1 = 2 * total_tp / float(2 * total_tp + total_fp + total_fn)
    else:
        f1 = 0.0
        
    if total_tp + total_fn > 0:
        recall = total_tp / float(total_tp + total_fn)
    else:
        recall = 0.0
    
    if total_tp + total_fp > 0:
        precision = total_tp / float(total_tp + total_fp)
    else:
        precision = 0.0
    
    return {
        'loss': avg_loss,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_batch_metrics(model, batch, device):
    """Calculate metrics for a single batch (useful for video validation).
    
    Args:
        model: The model to evaluate
        batch: Single batch of data
        device: Device to run evaluation on
        
    Returns:
        dict: Metrics for this batch
    """
    model.eval()
    
    # Ensure batch is on device
    if batch.device != device:
        batch = batch.to(device)
    
    with torch.no_grad():
        # Get predictions
        predictions = model(batch)
        loss = model.compute_loss(predictions, batch[:, 1:])
        
        # Convert to binary predictions
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).bool()
        targets = (batch[:, 1:] > 0.5).bool()
        
        # Calculate metrics
        metrics = calculate_binary_metrics(preds, targets)
        metrics['loss'] = loss.item()
        
        return metrics 