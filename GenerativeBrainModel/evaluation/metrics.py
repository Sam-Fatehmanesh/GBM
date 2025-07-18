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
        targets = (targets > 0.5).bool()
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
    # For Bernoulli-sampled spike counts
    total_pred_spikes_bernoulli = 0
    total_true_spikes = 0
    
    # Reset data loader
    data_loader.reset()
    test_iter = iter(data_loader)
    
    with torch.no_grad():
        for batch_idx in range(len(data_loader)):
            try:
                # Get batch with automatic GPU transfer
                batch = next(test_iter)
                
                # Handle both probability mode (tuple) and binary mode (single tensor)
                if isinstance(batch, tuple):
                    # Probability mode: batch is (input_data, target_data)
                    input_data, target_data = batch
                    # Use input_data for model input (always binary)
                    batch_for_model = input_data
                    # Ensure input is on GPU
                    if batch_for_model.device.type != 'cuda':
                        batch_for_model = batch_for_model.cuda(non_blocking=True)
                else:
                    # Binary mode: batch is single tensor
                    batch_for_model = batch
                    # Ensure batch is on GPU
                    if batch_for_model.device.type != 'cuda':
                        batch_for_model = batch_for_model.cuda(non_blocking=True)
                
                # Get logits from model (no sigmoid) using binary inputs
                with autocast():
                    predictions = model(batch_for_model)
                    batch_loss = model.compute_loss(predictions, batch_for_model[:, 1:])
                
                total_loss += batch_loss.item()
                batch_count += 1
                del batch_loss
                
                # Calculate threshold-based metrics using sigmoid on predictions
                probs = torch.sigmoid(predictions)
                # Keep predictions as bool to save memory
                preds = (probs > 0.5)
                
                # Calculate Bernoulli-sampled spike counts - clean probs for safety
                probs_clean = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs_clean = torch.clamp(probs_clean, min=0.0, max=1.0)
                bernoulli_preds = torch.bernoulli(probs_clean).to(torch.int64)
                total_pred_spikes_bernoulli += bernoulli_preds.sum().item()
                
                del predictions, probs, probs_clean, bernoulli_preds
                
                # Keep targets as bool as well - use 0.5 threshold for consistency with predictions
                targets = (batch_for_model[:, 1:] > 0.5).bool()
                
                # Clean target data before passing to torch.bernoulli() to avoid device-side assertion
                target_data = batch_for_model[:, 1:]
                target_data = torch.nan_to_num(target_data, nan=0.0, posinf=1.0, neginf=0.0)
                target_data = torch.clamp(target_data, min=0.0, max=1.0)
                total_true_spikes += torch.bernoulli(target_data).bool().to(torch.int64).sum().item()
                
                del batch, batch_for_model, target_data
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
        'fn': total_fn,
        'pred_spikes_bernoulli': total_pred_spikes_bernoulli,
        'true_spikes': total_true_spikes
    }


def calculate_batch_metrics(model, batch, device):
    """Calculate metrics for a single batch (useful for video validation).
    
    Args:
        model: The model to evaluate
        batch: Single batch of data (tensor or tuple)
        device: Device to run evaluation on
        
    Returns:
        dict: Metrics for this batch
    """
    model.eval()
    
    # Handle both probability mode (tuple) and binary mode (single tensor)
    if isinstance(batch, tuple):
        # Probability mode: batch is (input_data, target_data)
        input_data, target_data = batch
        # Use input_data for model input (always binary)
        batch_for_model = input_data
    else:
        # Binary mode: batch is single tensor
        batch_for_model = batch
    
    # Ensure batch is on device
    if batch_for_model.device != device:
        batch_for_model = batch_for_model.to(device)
    
    with torch.no_grad():
        # Get predictions using binary inputs
        predictions = model(batch_for_model)
        loss = model.compute_loss(predictions, batch_for_model[:, 1:])
        
        # Convert to binary predictions
        probs = torch.sigmoid(predictions)
        preds = (probs > 0.5).bool()
        targets = (batch_for_model[:, 1:] > 0.5).bool()
        
        # Calculate metrics
        metrics = calculate_binary_metrics(preds, targets)
        metrics['loss'] = loss.item()
        
        return metrics 