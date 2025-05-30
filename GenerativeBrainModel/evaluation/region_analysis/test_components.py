"""
Test script to verify all region analysis components can be imported and initialized.

This script performs basic import and initialization tests for all components
without requiring actual data files.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from test_data_loader import TestDataLoader
        print("✓ TestDataLoader imported successfully")
    except Exception as e:
        print(f"✗ TestDataLoader import failed: {e}")
    
    try:
        from frame_processor import FrameProcessor
        print("✓ FrameProcessor imported successfully")
    except Exception as e:
        print(f"✗ FrameProcessor import failed: {e}")
    
    try:
        from prediction_runner import PredictionRunner
        print("✓ PredictionRunner imported successfully")
    except Exception as e:
        print(f"✗ PredictionRunner import failed: {e}")
    
    try:
        from volume_grouper import VolumeGrouper
        print("✓ VolumeGrouper imported successfully")
    except Exception as e:
        print(f"✗ VolumeGrouper import failed: {e}")
    
    try:
        from region_performance_evaluator import RegionPerformanceEvaluator
        print("✓ RegionPerformanceEvaluator imported successfully")
    except Exception as e:
        print(f"✗ RegionPerformanceEvaluator import failed: {e}")
    
    try:
        from region_performance_visualizer import RegionPerformanceVisualizer
        print("✓ RegionPerformanceVisualizer imported successfully")
    except Exception as e:
        print(f"✗ RegionPerformanceVisualizer import failed: {e}")
    
    try:
        # Test package-level imports
        from . import (
            TestDataLoader,
            FrameProcessor,
            PredictionRunner,
            VolumeGrouper,
            RegionPerformanceEvaluator,
            RegionPerformanceVisualizer
        )
        print("✓ Package-level imports successful")
    except Exception as e:
        print(f"✗ Package-level imports failed: {e}")


def test_basic_initialization():
    """Test basic initialization of components that don't require data files."""
    print("\nTesting basic initialization...")
    
    try:
        from frame_processor import FrameProcessor
        fp = FrameProcessor()
        print("✓ FrameProcessor initialized successfully")
    except Exception as e:
        print(f"✗ FrameProcessor initialization failed: {e}")
    
    try:
        from volume_grouper import VolumeGrouper
        vg = VolumeGrouper()
        print("✓ VolumeGrouper initialized successfully")
    except Exception as e:
        print(f"✗ VolumeGrouper initialization failed: {e}")
    
    try:
        from region_performance_visualizer import RegionPerformanceVisualizer
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            vis = RegionPerformanceVisualizer(output_dir=temp_dir)
            print("✓ RegionPerformanceVisualizer initialized successfully")
    except Exception as e:
        print(f"✗ RegionPerformanceVisualizer initialization failed: {e}")


def test_torch_import():
    """Test PyTorch availability."""
    print("\nTesting PyTorch availability...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"  Current device: {torch.cuda.get_device_name()}")
        else:
            print("! CUDA not available - will use CPU")
            
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")


def test_visualization_dependencies():
    """Test visualization dependencies."""
    print("\nTesting visualization dependencies...")
    
    dependencies = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("numpy", "numpy"),
        ("pandas", "pandas")
    ]
    
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} not available")


def main():
    """Run all tests."""
    print("=" * 60)
    print("REGION ANALYSIS COMPONENTS TEST")
    print("=" * 60)
    
    test_torch_import()
    test_visualization_dependencies()
    test_imports()
    test_basic_initialization()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
    print("\nIf all tests passed, the region analysis system is ready to use!")
    print("If any tests failed, check the error messages above for troubleshooting.")


if __name__ == "__main__":
    main() 