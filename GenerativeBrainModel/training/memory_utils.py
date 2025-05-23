"""Memory management utilities for training."""

import torch


# Enable memory diagnostics
MEMORY_DIAGNOSTICS = False


def print_memory_stats(prefix=""):
    """Print GPU memory usage statistics"""
    if MEMORY_DIAGNOSTICS:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f"{prefix} GPU Memory: Total={t/1e9:.2f}GB, Reserved={r/1e9:.2f}GB, Allocated={a/1e9:.2f}GB, Free={f/1e9:.2f}GB")


def enable_memory_diagnostics(enabled=True):
    """Enable or disable memory diagnostics globally."""
    global MEMORY_DIAGNOSTICS
    MEMORY_DIAGNOSTICS = enabled


def get_memory_info():
    """Get current GPU memory usage information.
    
    Returns:
        dict: Memory usage statistics
    """
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a
        
        return {
            'total_gb': t / 1e9,
            'reserved_gb': r / 1e9,
            'allocated_gb': a / 1e9,
            'free_gb': f / 1e9,
            'utilization': a / t if t > 0 else 0
        }
    else:
        return {
            'total_gb': 0,
            'reserved_gb': 0,
            'allocated_gb': 0,
            'free_gb': 0,
            'utilization': 0
        } 