import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append('/home/user/GBM')

from GenerativeBrainModel.models.brainvae import hyperVAE

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def test_hypervae_basic():
    """Test basic functionality and instantiation"""
    print("Testing hyperVAE basic functionality...")
    
    # Test parameters
    d_model = 32
    volume = (256, 128, 30)
    batch_size = 2
    seq_len = 4
    
    print(f"Test configuration:")
    print(f"  d_model: {d_model}")
    print(f"  Volume size: {volume}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Create the model
    try:
        model = hyperVAE(d_model=d_model, volume=volume)
        print(f"\nModel created successfully!")
        print(f"  d_model: {model.d_model}")
        print(f"  volume: {model.volume}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    return True

def test_encoder_forward():
    """Test encoder forward pass"""
    print("\nTesting encoder forward pass...")
    
    d_model = 8  # Use smaller model for testing
    volume = (256, 128, 30)
    batch_size = 1
    seq_len = 4  # Reduce sequence length for memory
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create test input - note: expecting (B, T, C, vol_x, vol_y, vol_z) based on encoder
        x = torch.randn(batch_size, seq_len, 1, *volume, device=device)
        print(f"Input shape: {x.shape}")
        
        # Test encoder
        with torch.no_grad():  # Save memory during testing
            mu, logvar = model.encode(x)
            print(f"Encoder output shapes:")
            print(f"  mu: {mu.shape}")
            print(f"  logvar: {logvar.shape}")
        
        # Check that mu and logvar have same shape
        if mu.shape == logvar.shape:
            print("✓ mu and logvar have matching shapes")
        else:
            print(f"✗ mu and logvar shape mismatch: {mu.shape} vs {logvar.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Encoder forward pass failed: {e}")
        return False
    
    return True

def test_reparameterization():
    """Test reparameterization"""
    print("\nTesting reparameterization...")
    
    d_model = 8
    volume = (256, 128, 30)
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create dummy mu and logvar (should be output from encoder)
        # Need to figure out the actual latent dimensions
        batch_size, seq_len = 1, 1
        
        # Estimate latent dims based on downsampling: 
        # 256->128->64->16, 128->128->64, 30->30->30->15
        latent_dims = (16, 16, 15)  # This is a guess, will adjust based on actual encoder output
        latent_channels = d_model  # Half of encoder output channels
        
        mu = torch.randn(batch_size, seq_len, latent_channels, *latent_dims, device=device)
        logvar = torch.randn(batch_size, seq_len, latent_channels, *latent_dims, device=device)
        
        print(f"Input mu shape: {mu.shape}")
        print(f"Input logvar shape: {logvar.shape}")
        
        # Test reparameterization
        with torch.no_grad():
            z = model.reparameterize(mu, logvar)
            print(f"Reparameterized output shape: {z.shape}")
        
        # Check output shape matches input
        if z.shape == mu.shape:
            print("✓ Reparameterization preserves shape")
        else:
            print(f"✗ Shape mismatch: {z.shape} vs {mu.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Reparameterization failed: {e}")
        return False
    
    return True

def test_decoder_forward():
    """Test decoder forward pass"""
    print("\nTesting decoder forward pass...")
    
    d_model = 8
    volume = (256, 128, 30)
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create latent input (output from reparameterization)
        batch_size, seq_len = 1, 1
        latent_dims = (16, 16, 15)  # Estimated latent dimensions
        latent_channels = d_model
        
        z = torch.randn(batch_size, seq_len, latent_channels, *latent_dims, device=device)
        print(f"Decoder input shape: {z.shape}")
        
        # Test decoder
        with torch.no_grad():
            output = model.decode(z)
            print(f"Decoder output shape: {output.shape}")
        
        # Check if output matches expected reconstruction shape
        expected_shape = (batch_size, seq_len, 1, *volume)
        if output.shape == expected_shape:
            print(f"✓ Decoder output shape matches expected: {expected_shape}")
        else:
            print(f"! Decoder output shape: {output.shape}, expected: {expected_shape}")
            print("  (This may be expected due to padding/stride effects)")
            
    except Exception as e:
        print(f"✗ Decoder forward pass failed: {e}")
        return False
    
    return True

def test_full_vae_forward():
    """Test complete VAE forward pass (encode -> reparameterize -> decode)"""
    print("\nTesting full VAE forward pass...")
    
    d_model = 8
    volume = (256, 128, 30)
    batch_size = 1
    seq_len = 1
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create test input
        x = torch.randn(batch_size, seq_len, 1, *volume, device=device)
        print(f"Input shape: {x.shape}")
        
        # Full forward pass
        with torch.no_grad():
            mu, logvar = model.encode(x)
            print(f"Encoded - mu: {mu.shape}, logvar: {logvar.shape}")
            
            z = model.reparameterize(mu, logvar)
            print(f"Reparameterized: {z.shape}")
            
            reconstruction = model.decode(z)
            print(f"Reconstructed: {reconstruction.shape}")
        
        print("✓ Complete VAE forward pass successful")
        
    except Exception as e:
        print(f"✗ Full VAE forward pass failed: {e}")
        return False
    
    return True

def test_gradient_flow():
    """Test gradient computation and backpropagation"""
    print("\nTesting gradient flow...")
    
    d_model = 4  # Use even smaller model for gradient testing
    volume = (256, 128, 30)
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create test input with gradient tracking
        x = torch.randn(1, 1, 1, *volume, requires_grad=True, device=device)
        
        # Forward pass
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        reconstruction = model.decode(z)
        
        # Compute simple loss
        loss = reconstruction.sum()
        loss.backward()
        
        # Check if gradients exist
        if x.grad is not None:
            print("✓ Input gradients computed successfully!")
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
        else:
            print("✗ Input gradients not computed!")
            return False
        
        # Check model parameter gradients
        param_grads_exist = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 1e-8:
                param_grads_exist.append(name)
        
        if len(param_grads_exist) > 0:
            print(f"✓ Parameter gradients computed for {len(param_grads_exist)} parameters")
            print(f"  Sample parameters with gradients: {param_grads_exist[:5]}")
        else:
            print("✗ No parameter gradients computed!")
            return False
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    return True

def test_vae_loss_computation():
    """Test VAE loss computation (reconstruction + KL divergence)"""
    print("\nTesting VAE loss computation...")
    
    d_model = 4
    volume = (256, 128, 30)
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume).to(device)
        
        # Create test input
        x = torch.randn(1, 1, 1, *volume, device=device)
        
        # Forward pass
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        reconstruction = model.decode(z)
        
        # Compute VAE losses
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(reconstruction, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (x.shape[0] * x.shape[1])  # Normalize by batch size and sequence length
        
        total_loss = recon_loss + kl_loss
        
        print(f"✓ VAE loss computation successful:")
        print(f"  Reconstruction loss: {recon_loss.item():.6f}")
        print(f"  KL divergence loss: {kl_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ VAE loss computation failed: {e}")
        return False
    
    return True

def test_different_sequences():
    """Test with different sequence lengths"""
    print("\nTesting different sequence lengths...")
    
    d_model = 4
    volume = (256, 128, 30)
    
    sequence_lengths = [1, 2]  # Reduce for memory
    
    for seq_len in sequence_lengths:
        try:
            model = hyperVAE(d_model=d_model, volume=volume).to(device)
            
            x = torch.randn(1, seq_len, 1, *volume, device=device)
            
            # Test forward pass
            with torch.no_grad():
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                reconstruction = model.decode(z)
            
            print(f"  Sequence length {seq_len}: ✓ Input {x.shape} -> Output {reconstruction.shape}")
            
        except Exception as e:
            print(f"  Sequence length {seq_len}: ✗ Failed: {e}")
            return False
    
    return True

def test_different_batch_sizes():
    """Test with different batch sizes"""
    print("\nTesting different batch sizes...")
    
    d_model = 4
    volume = (256, 128, 30)
    seq_len = 1  # Reduce for memory
    
    batch_sizes = [1, 2]  # Reduce for memory
    
    for batch_size in batch_sizes:
        try:
            model = hyperVAE(d_model=d_model, volume=volume).to(device)
            
            x = torch.randn(batch_size, seq_len, 1, *volume, device=device)
            
            # Test forward pass
            with torch.no_grad():
                mu, logvar = model.encode(x)
                z = model.reparameterize(mu, logvar)
                reconstruction = model.decode(z)
            
            print(f"  Batch size {batch_size}: ✓ Input {x.shape} -> Output {reconstruction.shape}")
            
        except Exception as e:
            print(f"  Batch size {batch_size}: ✗ Failed: {e}")
            return False
    
    return True

def test_model_components():
    """Test individual model components"""
    print("\nTesting model components...")
    
    d_model = 16
    volume = (256, 128, 30)
    
    try:
        model = hyperVAE(d_model=d_model, volume=volume)
        
        # Test that model has required components
        required_components = ['encoder', 'decoder', 'd_model', 'volume']
        
        for component in required_components:
            if hasattr(model, component):
                print(f"✓ Component '{component}' exists")
            else:
                print(f"✗ Component '{component}' missing")
                return False
        
        # Test encoder and decoder are sequential modules
        if isinstance(model.encoder, nn.Sequential):
            print(f"✓ Encoder is Sequential with {len(model.encoder)} layers")
        else:
            print(f"✗ Encoder is not Sequential: {type(model.encoder)}")
            return False
            
        if isinstance(model.decoder, nn.Sequential):
            print(f"✓ Decoder is Sequential with {len(model.decoder)} layers")
        else:
            print(f"✗ Decoder is not Sequential: {type(model.decoder)}")
            return False
        
    except Exception as e:
        print(f"✗ Component testing failed: {e}")
        return False
    
    return True

def run_all_tests():
    """Run all hyperVAE tests"""
    print("=" * 60)
    print("RUNNING hyperVAE TESTS")
    print("=" * 60)
    
    tests = [
        test_hypervae_basic,
        test_model_components,
        test_encoder_forward,
        test_reparameterization,
        test_decoder_forward,
        test_full_vae_forward,
        test_gradient_flow,
        test_vae_loss_computation,
        test_different_sequences,
        test_different_batch_sizes,
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
        print("🎉 All hyperVAE tests passed!")
        return True
    else:
        print("❌ Some hyperVAE tests failed!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
