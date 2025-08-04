import torch
import sys
import os


from GenerativeBrainModel.models.brainvae import CausalConv4D

def test_causal_conv4d():
    """Test the CausalConv4D implementation"""
    print("Testing CausalConv4D implementation...")
    
    # Test parameters
    batch_size = 2
    sequence_length = 4
    in_channels = 3
    out_channels = 8
    vol_x, vol_y, vol_z = 16, 16, 8
    spatial_kernel_size = 3
    temporal_kernel_size = 3
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Input channels: {in_channels}")
    print(f"  Output channels: {out_channels}")
    print(f"  Volume size: ({vol_x}, {vol_y}, {vol_z})")
    print(f"  Spatial kernel size: {spatial_kernel_size}")
    print(f"  Temporal kernel size: {temporal_kernel_size}")
    
    # Create the layer
    layer = CausalConv4D(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_kernel_size=spatial_kernel_size,
        temporal_kernel_size=temporal_kernel_size,
        stride=1,
        padding='same'
    )
    
    print(f"\nLayer created successfully!")
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Bias shape: {layer.bias.shape}")
    
    # Create test input
    x = torch.randn(batch_size, sequence_length, in_channels, vol_x, vol_y, vol_z)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    try:
        output = layer(x)
        print(f"Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (batch_size, sequence_length, out_channels, vol_x, vol_y, vol_z)
        if output.shape == expected_shape:
            print("✓ Output shape is correct!")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        return False
    
    # Test causality
    print("\nTesting causality...")
    
    # Create two inputs: one normal, one with future timesteps modified
    x1 = torch.randn(1, sequence_length, in_channels, vol_x, vol_y, vol_z)
    x2 = x1.clone()
    
    # Modify future timesteps (t=2 and t=3) in x2
    x2[:, 2:, :, :, :, :] = torch.randn_like(x2[:, 2:, :, :, :, :])
    
    # Forward pass for both
    out1 = layer(x1)
    out2 = layer(x2)
    
    # Check if early timesteps are the same (should be due to causality)
    early_timesteps_same = torch.allclose(out1[:, :2, :, :, :, :], out2[:, :2, :, :, :, :], atol=1e-6)
    
    if early_timesteps_same:
        print("✓ Causality test passed! Early timesteps are unaffected by future changes.")
    else:
        print("✗ Causality test failed! Early timesteps are affected by future changes.")
        return False
    
    # Test with different temporal kernel sizes
    print("\nTesting different temporal kernel sizes...")
    
    for temp_k in [1, 2, 3, 5]:
        try:
            test_layer = CausalConv4D(
                in_channels=2,
                out_channels=4,
                spatial_kernel_size=3,
                temporal_kernel_size=temp_k,
                stride=1,
                temporal_stride=1,
                padding='same'
            )
            
            test_input = torch.randn(1, 6, 2, 8, 8, 4)
            test_output = test_layer(test_input)
            print(f"  Temporal kernel {temp_k}: ✓ (output shape: {test_output.shape})")
            
        except Exception as e:
            print(f"  Temporal kernel {temp_k}: ✗ Error: {e}")
            return False
    
    # Test with different temporal strides
    print("\nTesting different temporal strides...")
    
    for temp_stride in [1, 2, 3, 4]:
        try:
            test_layer = CausalConv4D(
                in_channels=2,
                out_channels=4,
                spatial_kernel_size=3,
                temporal_kernel_size=3,
                stride=1,
                temporal_stride=temp_stride,
                padding='same'
            )
            
            test_input = torch.randn(1, 8, 2, 8, 8, 4)
            test_output = test_layer(test_input)
            expected_t_out = (8 + 3 - 1 - 3) // temp_stride + 1  # (T + padding - kernel) // stride + 1
            print(f"  Temporal stride {temp_stride}: ✓ (output shape: {test_output.shape}, expected T_out: {expected_t_out})")
            
            # Verify temporal dimension is correct
            if test_output.shape[1] == expected_t_out:
                print(f"    ✓ Correct temporal output dimension")
            else:
                print(f"    ✗ Wrong temporal output dimension. Expected: {expected_t_out}, Got: {test_output.shape[1]}")
                return False
            
        except Exception as e:
            print(f"  Temporal stride {temp_stride}: ✗ Error: {e}")
            return False
    
    print("\n✓ All tests passed! CausalConv4D implementation is working correctly.")
    return True

def test_gradient_flow():
    """Test gradient flow through the layer"""
    print("\nTesting gradient flow...")
    
    layer = CausalConv4D(
        in_channels=1,
        out_channels=2,
        spatial_kernel_size=3,
        temporal_kernel_size=2,
        stride=1,
        padding='same'
    )
    
    x = torch.randn(1, 3, 1, 8, 8, 4, requires_grad=True)
    
    try:
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        if x.grad is not None and layer.weight.grad is not None:
            print("✓ Gradients computed successfully!")
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Weight gradient shape: {layer.weight.grad.shape}")
            return True
        else:
            print("✗ Gradients not computed!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_causal_conv4d()
    if success:
        success = test_gradient_flow()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
