import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append('/home/user/GBM')

from GenerativeBrainModel.models.convolutional import SpatialDownsampleConv4D

def test_spatial_downsample_basic():
    """Test basic functionality and shape changes"""
    print("Testing SpatialDownsampleConv4D basic functionality...")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    in_channels = 8
    out_channels = 16
    width, height, depth = 16, 12, 8
    stride = (2, 2, 2)
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Volume size: ({width}, {height}, {depth})")
    print(f"  Stride: {stride}")
    
    # Create the layer
    layer = SpatialDownsampleConv4D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3, 3),
        stride=stride
    )
    
    print(f"\nLayer created successfully!")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, in_channels, width, height, depth)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    try:
        output = layer(x)
        print(f"Output shape: {output.shape}")
        
        # Check output shape - spatial dimensions should be downsampled by stride
        expected_shape = (
            batch_size, 
            seq_len, 
            out_channels, 
            width // stride[0], 
            height // stride[1], 
            depth // stride[2]
        )
        
        if output.shape == expected_shape:
            print("✓ Output shape is correct!")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        return False
    
    return True

def test_downsampling_ratios():
    """Test different downsampling ratios"""
    print("\nTesting different downsampling ratios...")
    
    test_configs = [
        ((2, 2, 2), (16, 16, 8)),     # Standard 2x downsampling
        ((1, 1, 1), (16, 16, 8)),     # No downsampling
        ((3, 3, 3), (15, 15, 9)),     # 3x downsampling
        ((2, 1, 2), (16, 16, 8)),     # Mixed downsampling
        ((4, 2, 2), (16, 16, 8)),     # Asymmetric downsampling
    ]
    
    for i, (stride, vol_size) in enumerate(test_configs):
        try:
            layer = SpatialDownsampleConv4D(
                in_channels=4,
                out_channels=8,
                kernel_size=(3, 3, 3),
                stride=stride
            )
            
            W, H, D = vol_size
            x = torch.randn(1, 2, 4, W, H, D)
            output = layer(x)
            
            expected_W = W // stride[0]
            expected_H = H // stride[1] 
            expected_D = D // stride[2]
            
            if output.shape == (1, 2, 8, expected_W, expected_H, expected_D):
                print(f"  Config {i+1}: ✓ Stride {stride} → Output ({expected_W}, {expected_H}, {expected_D})")
            else:
                print(f"  Config {i+1}: ✗ Wrong output shape for stride {stride}")
                return False
                
        except Exception as e:
            print(f"  Config {i+1}: ✗ Error with stride {stride}: {e}")
            return False
    
    return True

def test_temporal_independence():
    """Test that downsampling preserves temporal independence"""
    print("\nTesting temporal independence...")
    
    layer = SpatialDownsampleConv4D(
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2)
    )
    
    # Create input with different patterns at different timesteps
    batch_size, seq_len, channels = 1, 3, 4
    vol_shape = (16, 12, 8)
    
    x = torch.randn(batch_size, seq_len, channels, *vol_shape)
    
    # Process full sequence
    output_full = layer(x)
    
    # Process each timestep individually
    outputs_individual = []
    for t in range(seq_len):
        x_t = x[:, t:t+1, :, :, :, :]  # Keep time dimension
        output_t = layer(x_t)
        outputs_individual.append(output_t)
    
    output_individual = torch.cat(outputs_individual, dim=1)
    
    # Should be exactly the same
    if torch.allclose(output_full, output_individual, atol=1e-6):
        print("✓ Temporal independence verified: processing full sequence vs individual timesteps gives same result")
    else:
        print("✗ Temporal independence failed: results differ between full and individual processing")
        max_diff = (output_full - output_individual).abs().max().item()
        print(f"  Maximum difference: {max_diff}")
        return False
    
    return True

def test_different_kernel_sizes():
    """Test with various kernel sizes"""
    print("\nTesting different kernel sizes...")
    
    kernel_configs = [
        (1, 1, 1),      # 1x1x1 kernel
        (3, 3, 3),      # Standard 3x3x3 kernel
        (5, 5, 5),      # Larger 5x5x5 kernel
        (3, 5, 3),      # Asymmetric kernel
        (1, 3, 1),      # 2D-like kernel
    ]
    
    for i, kernel_size in enumerate(kernel_configs):
        try:
            layer = SpatialDownsampleConv4D(
                in_channels=4,
                out_channels=8,
                kernel_size=kernel_size,
                stride=(2, 2, 2)
            )
            
            x = torch.randn(1, 2, 4, 16, 16, 8)
            output = layer(x)
            
            expected_shape = (1, 2, 8, 8, 8, 4)  # Downsampled by stride (2,2,2)
            
            if output.shape == expected_shape:
                print(f"  Kernel {i+1}: ✓ {kernel_size}")
            else:
                print(f"  Kernel {i+1}: ✗ Wrong output shape for kernel {kernel_size}")
                return False
                
        except Exception as e:
            print(f"  Kernel {i+1}: ✗ Error with kernel {kernel_size}: {e}")
            return False
    
    return True

def test_channel_changes():
    """Test different input/output channel configurations"""
    print("\nTesting different channel configurations...")
    
    channel_configs = [
        (1, 1),      # Same single channel
        (1, 4),      # Single to multi
        (4, 1),      # Multi to single
        (8, 16),     # Standard expansion
        (16, 8),     # Channel reduction
        (32, 32),    # Same channels
    ]
    
    for i, (in_ch, out_ch) in enumerate(channel_configs):
        try:
            layer = SpatialDownsampleConv4D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2)
            )
            
            x = torch.randn(1, 2, in_ch, 16, 16, 8)
            output = layer(x)
            
            expected_shape = (1, 2, out_ch, 8, 8, 4)
            
            if output.shape == expected_shape:
                print(f"  Config {i+1}: ✓ {in_ch} → {out_ch} channels")
            else:
                print(f"  Config {i+1}: ✗ Wrong output shape for {in_ch} → {out_ch}")
                return False
                
        except Exception as e:
            print(f"  Config {i+1}: ✗ Error with channels {in_ch} → {out_ch}: {e}")
            return False
    
    return True

def test_gradient_flow():
    """Test gradient computation and backpropagation"""
    print("\nTesting gradient flow...")
    
    layer = SpatialDownsampleConv4D(
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2)
    )
    
    x = torch.randn(1, 3, 4, 16, 12, 8, requires_grad=True)
    
    try:
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        if x.grad is not None:
            print("✓ Input gradients computed successfully!")
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
        else:
            print("✗ Input gradients not computed!")
            return False
        
        # Check layer parameter gradients
        param_grads_exist = []
        for name, param in layer.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 1e-8:
                param_grads_exist.append(name)
        
        if len(param_grads_exist) > 0:
            print(f"✓ Parameter gradients computed for: {param_grads_exist}")
        else:
            print("✗ No parameter gradients computed!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    return True

def test_dimension_consistency():
    """Test that spatial dimensions are correctly computed"""
    print("\nTesting dimension consistency...")
    
    # Test various input sizes and strides
    test_cases = [
        ((16, 16, 8), (2, 2, 2), (8, 8, 4)),
        ((32, 24, 16), (2, 2, 2), (16, 12, 8)),
        ((15, 15, 9), (3, 3, 3), (5, 5, 3)),
        ((20, 16, 12), (4, 2, 3), (5, 8, 4)),
        ((8, 8, 8), (1, 1, 1), (8, 8, 8)),
    ]
    
    for i, (input_size, stride, expected_output_size) in enumerate(test_cases):
        try:
            layer = SpatialDownsampleConv4D(
                in_channels=4,
                out_channels=4,
                kernel_size=(3, 3, 3),
                stride=stride
            )
            
            W, H, D = input_size
            x = torch.randn(1, 2, 4, W, H, D)
            output = layer(x)
            
            actual_output_size = output.shape[-3:]
            
            if actual_output_size == expected_output_size:
                print(f"  Case {i+1}: ✓ {input_size} → {actual_output_size} with stride {stride}")
            else:
                print(f"  Case {i+1}: ✗ Expected {expected_output_size}, got {actual_output_size}")
                return False
                
        except Exception as e:
            print(f"  Case {i+1}: ✗ Error: {e}")
            return False
    
    return True

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\nTesting edge cases...")
    
    # Test with minimal dimensions
    try:
        layer = SpatialDownsampleConv4D(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1)
        )
        x = torch.randn(1, 1, 1, 2, 2, 2)  # Very small volume
        output = layer(x)
        print("✓ Minimal dimensions handled correctly")
    except Exception as e:
        print(f"✗ Minimal dimensions failed: {e}")
        return False
    
    # Test with large stride
    try:
        layer = SpatialDownsampleConv4D(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3, 3),
            stride=(8, 8, 8)
        )
        x = torch.randn(1, 1, 4, 32, 32, 16)
        output = layer(x)
        print("✓ Large stride handled correctly")
    except Exception as e:
        print(f"✗ Large stride failed: {e}")
        return False
    
    return True

def test_memory_efficiency():
    """Test memory usage for reasonably sized inputs"""
    print("\nTesting memory efficiency...")
    
    try:
        layer = SpatialDownsampleConv4D(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2)
        )
        
        # Test with moderately large input
        x = torch.randn(2, 4, 16, 48, 48, 24)
        
        # Monitor memory before and after
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
        
        output = layer(x)
        
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / 1024**2  # MB
            print(f"✓ Memory usage: {mem_used:.2f} MB for shape {x.shape}")
        else:
            print("✓ CPU execution successful for moderately large input")
        
        print(f"  Input size: {x.numel() * 4 / 1024**2:.2f} MB")
        print(f"  Output size: {output.numel() * 4 / 1024**2:.2f} MB")
        
        # Verify downsampling reduces memory
        if output.numel() < x.numel():
            print("✓ Downsampling reduces tensor size as expected")
        else:
            print("✗ Downsampling did not reduce tensor size")
            return False
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all SpatialDownsampleConv4D tests"""
    print("=" * 60)
    print("RUNNING SPATIALDOWNSAMPLECONV4D TESTS")
    print("=" * 60)
    
    tests = [
        test_spatial_downsample_basic,
        test_downsampling_ratios,
        test_temporal_independence,
        test_different_kernel_sizes,
        test_channel_changes,
        test_gradient_flow,
        test_dimension_consistency,
        test_edge_cases,
        test_memory_efficiency,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} CRASHED: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All SpatialDownsampleConv4D tests passed!")
        return True
    else:
        print("❌ Some SpatialDownsampleConv4D tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
