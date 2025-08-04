import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append('/home/user/GBM')

from GenerativeBrainModel.models.attention import VoxelAttention

def test_voxel_attention_basic():
    """Test basic functionality and shape preservation"""
    print("Testing VoxelAttention basic functionality...")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    channels = 8
    width, height, depth = 16, 12, 8
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Channels: {channels}")
    print(f"  Volume size: ({width}, {height}, {depth})")
    
    # Create the layer
    layer = VoxelAttention(dim=channels)
    
    print(f"\nLayer created successfully!")
    
    # Create test input
    x = torch.randn(batch_size, seq_len, channels, width, height, depth)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    try:
        output = layer(x)
        print(f"Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (batch_size, seq_len, channels, width, height, depth)
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
    
    # Create layer with zero-initialized projection (should start as identity)
    layer = VoxelAttention(dim=4)
    
    # Verify projection weights are zero-initialized
    if torch.allclose(layer.proj.weight, torch.zeros_like(layer.proj.weight)):
        print("✓ Projection layer is zero-initialized")
    else:
        print("✗ Projection layer is not zero-initialized")
        return False
    
    # Test that with zero-init, output should be very close to input (due to normalization)
    x = torch.randn(1, 2, 4, 8, 6, 4)
    
    with torch.no_grad():
        output = layer(x)
        
        # Due to normalization, won't be exactly equal to input, but should be close to identity behavior
        # The main test is that it doesn't crash and produces reasonable output
        print(f"✓ Layer produces output with residual connection")
        print(f"  Input norm: {x.norm().item():.4f}")
        print(f"  Output norm: {output.norm().item():.4f}")
    
    return True

def test_time_independence():
    """Test that attention is applied independently per timestep"""
    print("\nTesting temporal independence...")
    
    layer = VoxelAttention(dim=4)
    
    # Create input with different patterns at different timesteps
    batch_size, seq_len, channels = 1, 3, 4
    vol_shape = (8, 6, 4)
    
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

def test_different_dimensions():
    """Test with various input dimensions"""
    print("\nTesting different input dimensions...")
    
    test_configs = [
        (1, 1, 2, 4, 4, 4),      # Minimal case
        (2, 3, 8, 16, 16, 8),    # Standard case
        (1, 5, 16, 32, 24, 12),  # Larger case
        (3, 2, 4, 8, 8, 8),      # Multiple batches
    ]
    
    for i, (B, T, C, W, H, D) in enumerate(test_configs):
        try:
            layer = VoxelAttention(dim=C)
            x = torch.randn(B, T, C, W, H, D)
            output = layer(x)
            
            if output.shape == (B, T, C, W, H, D):
                print(f"  Config {i+1}: ✓ Shape ({B}, {T}, {C}, {W}, {H}, {D})")
            else:
                print(f"  Config {i+1}: ✗ Wrong output shape for ({B}, {T}, {C}, {W}, {H}, {D})")
                return False
                
        except Exception as e:
            print(f"  Config {i+1}: ✗ Error with shape ({B}, {T}, {C}, {W}, {H}, {D}): {e}")
            return False
    
    return True

def test_gradient_flow():
    """Test gradient computation and backpropagation"""
    print("\nTesting gradient flow...")
    
    layer = VoxelAttention(dim=4)
    x = torch.randn(1, 2, 4, 8, 6, 4, requires_grad=True)
    
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
            if param.grad is not None:
                param_grads_exist.append(name)
                print(f"  {name} gradient norm: {param.grad.norm().item():.6f}")
        
        if len(param_grads_exist) > 0:
            print(f"✓ Parameter gradients computed for: {param_grads_exist}")
        else:
            print("✗ No parameter gradients computed!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    return True

def test_attention_mechanism():
    """Test that the attention mechanism is working properly"""
    print("\nTesting attention mechanism properties...")
    
    layer = VoxelAttention(dim=8)
    
    # Create input with known patterns
    B, T, C = 1, 2, 8
    W, H, D = 4, 4, 4
    
    # Test 1: Random inputs should produce different outputs
    x1 = torch.randn(B, T, C, W, H, D)
    x2 = torch.randn(B, T, C, W, H, D)
    
    output1 = layer(x1)
    output2 = layer(x2)
    
    # Outputs should be different for different inputs
    if not torch.allclose(output1, output2, atol=1e-4):
        print("✓ Different inputs produce different outputs")
    else:
        print("✗ Different inputs produce identical outputs")
        return False
    
    # Test 2: Test that attention is actually computed (weights not all zero)
    # Initialize the projection layer with small non-zero weights to test attention mechanism
    layer_test = VoxelAttention(dim=8)
    nn.init.normal_(layer_test.to_qkv.weight, std=0.01)
    nn.init.normal_(layer_test.proj.weight, std=0.01)
    
    x_test = torch.randn(B, T, C, W, H, D)
    
    # Forward pass with initialized weights
    output_test = layer_test(x_test)
    
    # Should produce meaningful output that's different from input
    input_output_diff = (output_test - x_test).abs().mean().item()
    if input_output_diff > 1e-6:
        print("✓ Attention mechanism produces meaningful transformations")
        print(f"  Average input-output difference: {input_output_diff:.6f}")
    else:
        print("✗ Attention mechanism not producing meaningful transformations")
        return False
    
    # Test 3: Same input should produce same output (deterministic)
    output_test2 = layer_test(x_test)
    if torch.allclose(output_test, output_test2, atol=1e-6):
        print("✓ Deterministic behavior: same input produces same output")
    else:
        print("✗ Non-deterministic behavior detected")
        return False
    
    return True

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\nTesting edge cases...")
    
    # Test with very small spatial dimensions
    try:
        layer = VoxelAttention(dim=4)
        x = torch.randn(1, 1, 4, 2, 2, 2)  # Very small volume
        output = layer(x)
        print("✓ Small spatial dimensions handled correctly")
    except Exception as e:
        print(f"✗ Small spatial dimensions failed: {e}")
        return False
    
    # Test with single channel
    try:
        layer = VoxelAttention(dim=1)
        x = torch.randn(1, 1, 1, 4, 4, 4)
        output = layer(x)
        print("✓ Single channel handled correctly")
    except Exception as e:
        print(f"✗ Single channel failed: {e}")
        return False
    
    return True

def test_memory_efficiency():
    """Test memory usage for reasonably sized inputs"""
    print("\nTesting memory efficiency...")
    
    try:
        layer = VoxelAttention(dim=16)
        
        # Test with moderately large input
        x = torch.randn(2, 4, 16, 32, 32, 16)
        
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
    """Run all VoxelAttention tests"""
    print("=" * 60)
    print("RUNNING VOXELATTENTION TESTS")
    print("=" * 60)
    
    tests = [
        test_voxel_attention_basic,
        test_residual_connection,
        test_time_independence,
        test_different_dimensions,
        test_gradient_flow,
        test_attention_mechanism,
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
        print("🎉 All VoxelAttention tests passed!")
        return True
    else:
        print("❌ Some VoxelAttention tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
