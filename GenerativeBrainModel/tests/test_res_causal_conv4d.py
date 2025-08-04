import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append('/home/user/GBM')

from GenerativeBrainModel.models.convolutional import ResCausalConv4D

def test_res_causal_conv4d_basic():
    """Test basic functionality and shape preservation"""
    print("Testing ResCausalConv4D basic functionality...")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    in_channels = 8
    out_channels = 16
    width, height, depth = 16, 12, 8
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Volume size: ({width}, {height}, {depth})")
    
    # Create the layer
    layer = ResCausalConv4D(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    print(f"\nLayer created successfully!")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, in_channels, width, height, depth)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    try:
        output = layer(x)
        print(f"Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (batch_size, seq_len, out_channels, width, height, depth)
        if output.shape == expected_shape:
            print("✓ Output shape is correct!")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        return False
    
    return True

def test_residual_connection():
    """Test that the layer implements proper residual connections"""
    print("\nTesting residual connection behavior...")
    
    # Test case 1: Same input/output channels (Identity shortcut)
    in_channels = out_channels = 4
    layer_identity = ResCausalConv4D(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    # Check that shortcut is Identity
    if isinstance(layer_identity.shortcut, nn.Identity):
        print("✓ Identity shortcut used when in_channels == out_channels")
    else:
        print("✗ Identity shortcut not used when in_channels == out_channels")
        return False
    
    # Test case 2: Different input/output channels (Conv shortcut)
    layer_conv = ResCausalConv4D(
        in_channels=4,
        out_channels=8,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    # Check that shortcut is CausalConv4D
    if hasattr(layer_conv.shortcut, 'weight'):  # CausalConv4D has weight parameter
        print("✓ Convolutional shortcut used when in_channels != out_channels")
    else:
        print("✗ Convolutional shortcut not used when in_channels != out_channels")
        return False
    
    # Test forward pass with residual behavior
    x = torch.randn(1, 3, 4, 8, 6, 4)
    
    # For identity case, we can test that output != input (due to nonlinearities)
    with torch.no_grad():
        output_identity = layer_identity(x)
        if not torch.allclose(output_identity, x, atol=1e-3):
            print("✓ Residual layer produces different output from input (as expected)")
        else:
            print("✗ Residual layer produces identical output to input")
            return False
    
    return True

def test_different_configurations():
    """Test with various layer configurations"""
    print("\nTesting different layer configurations...")
    
    test_configs = [
        (4, 4, 3, 2, 1, 1),      # Same channels, standard config
        (2, 8, 3, 2, 1, 1),     # Different channels
        (8, 16, 5, 3, 1, 1),    # Larger kernels
        (4, 4, 3, 2, 1, 2),     # Temporal stride
        (1, 1, 1, 1, 1, 1),     # Minimal config
    ]
    
    for i, (in_ch, out_ch, spat_k, temp_k, spat_s, temp_s) in enumerate(test_configs):
        try:
            layer = ResCausalConv4D(
                in_channels=in_ch,
                out_channels=out_ch,
                spatial_kernel_size=spat_k,
                temporal_kernel_size=temp_k,
                stride=spat_s,
                temporal_stride=temp_s
            )
            
            x = torch.randn(1, 4, in_ch, 8, 8, 4)
            output = layer(x)
            
            print(f"  Config {i+1}: ✓ in_ch={in_ch}, out_ch={out_ch}, spat_k={spat_k}, temp_k={temp_k}")
                
        except Exception as e:
            print(f"  Config {i+1}: ✗ Error: {e}")
            return False
    
    return True

def test_causality():
    """Test that the layer maintains causal properties"""
    print("\nTesting causality...")
    
    layer = ResCausalConv4D(
        in_channels=4,
        out_channels=4,
        spatial_kernel_size=3,
        temporal_kernel_size=3
    )
    
    # Create two inputs: one normal, one with future timesteps modified
    x1 = torch.randn(1, 6, 4, 8, 8, 4)
    x2 = x1.clone()
    
    # Modify future timesteps (t=4 and t=5) in x2
    x2[:, 4:, :, :, :, :] = torch.randn_like(x2[:, 4:, :, :, :, :])
    
    # Forward pass for both
    with torch.no_grad():
        out1 = layer(x1)
        out2 = layer(x2)
    
    # Check if early timesteps are the same (should be due to causality)
    early_timesteps_same = torch.allclose(out1[:, :4, :, :, :, :], out2[:, :4, :, :, :, :], atol=1e-6)
    
    if early_timesteps_same:
        print("✓ Causality test passed! Early timesteps are unaffected by future changes.")
    else:
        print("✗ Causality test failed! Early timesteps are affected by future changes.")
        return False
    
    return True

def test_gradient_flow():
    """Test gradient computation and backpropagation"""
    print("\nTesting gradient flow...")
    
    layer = ResCausalConv4D(
        in_channels=4,
        out_channels=8,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    x = torch.randn(1, 3, 4, 8, 6, 4, requires_grad=True)
    
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
            print(f"✓ Parameter gradients computed for: {len(param_grads_exist)} parameters")
        else:
            print("✗ No parameter gradients computed!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    return True

def test_layer_components():
    """Test that all layer components are working"""
    print("\nTesting layer components...")
    
    layer = ResCausalConv4D(
        in_channels=4,
        out_channels=8,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    # Check that all components exist
    required_components = ['norm0', 'act0', 'conv0', 'norm1', 'act1', 'conv1', 'shortcut']
    
    for component in required_components:
        if hasattr(layer, component):
            print(f"✓ Component '{component}' exists")
        else:
            print(f"✗ Component '{component}' missing")
            return False
    
    # Test intermediate outputs
    x = torch.randn(1, 3, 4, 8, 6, 4)
    
    try:
        with torch.no_grad():
            # Test shortcut
            shortcut = layer.shortcut(x)
            print(f"✓ Shortcut output shape: {shortcut.shape}")
            
            # Test first path
            x_norm0 = layer.norm0(x)
            x_act0 = layer.act0(x_norm0)
            x_conv0 = layer.conv0(x_act0)
            print(f"✓ First conv path output shape: {x_conv0.shape}")
            
            # Test second path
            x_norm1 = layer.norm1(x_conv0)
            x_act1 = layer.act1(x_norm1)
            x_conv1 = layer.conv1(x_act1)
            print(f"✓ Second conv path output shape: {x_conv1.shape}")
            
            # Test final addition
            final_output = x_conv1 + shortcut
            print(f"✓ Final residual addition shape: {final_output.shape}")
            
    except Exception as e:
        print(f"✗ Component testing failed: {e}")
        return False
    
    return True

def test_different_input_sizes():
    """Test with various input dimensions"""
    print("\nTesting different input sizes...")
    
    layer = ResCausalConv4D(
        in_channels=4,
        out_channels=8,
        spatial_kernel_size=3,
        temporal_kernel_size=2
    )
    
    test_shapes = [
        (1, 2, 4, 4, 4, 4),      # Minimal
        (2, 3, 4, 8, 8, 8),     # Small
        (1, 5, 4, 16, 12, 6),   # Standard
        (3, 4, 4, 32, 24, 12),  # Larger
    ]
    
    for i, (B, T, C, W, H, D) in enumerate(test_shapes):
        try:
            x = torch.randn(B, T, C, W, H, D)
            output = layer(x)
            expected_shape = (B, T, 8, W, H, D)  # out_channels = 8
            
            if output.shape == expected_shape:
                print(f"  Size {i+1}: ✓ Shape ({B}, {T}, {C}, {W}, {H}, {D})")
            else:
                print(f"  Size {i+1}: ✗ Wrong output shape. Expected: {expected_shape}, Got: {output.shape}")
                return False
                
        except Exception as e:
            print(f"  Size {i+1}: ✗ Error with shape ({B}, {T}, {C}, {W}, {H}, {D}): {e}")
            return False
    
    return True

def test_memory_efficiency():
    """Test memory usage for reasonably sized inputs"""
    print("\nTesting memory efficiency...")
    
    try:
        layer = ResCausalConv4D(
            in_channels=8,
            out_channels=16,
            spatial_kernel_size=3,
            temporal_kernel_size=2
        )
        
        # Test with moderately large input
        x = torch.randn(2, 4, 8, 24, 24, 12)
        
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
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all ResCausalConv4D tests"""
    print("=" * 60)
    print("RUNNING RESCAUSALCONV4D TESTS")
    print("=" * 60)
    
    tests = [
        test_res_causal_conv4d_basic,
        test_residual_connection,
        test_different_configurations,
        test_causality,
        test_gradient_flow,
        test_layer_components,
        test_different_input_sizes,
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
        print("🎉 All ResCausalConv4D tests passed!")
        return True
    else:
        print("❌ Some ResCausalConv4D tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
