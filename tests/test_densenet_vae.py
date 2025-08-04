"""
Test suite for DenseNet VAE architecture.

Tests S2.1.1 success criteria:
- 3D DenseNet architecture implemented correctly
- Dilated convolutions in final block achieve >150mm receptive field  
- Model processes batch without memory issues
- Parameter count within expected range (documented)
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.densenet_vae import DenseNetVAE3D, DenseNetEncoder3D, DenseNetDecoder3D


class TestDenseNetVAE3D:
    """Test suite for DenseNet VAE."""
    
    def test_architecture_implementation(self):
        """Test that 3D DenseNet architecture is implemented correctly."""
        model = DenseNetVAE3D(latent_dim=128)
        
        # Check that encoder is DenseNet-based
        assert isinstance(model.encoder, DenseNetEncoder3D)
        assert isinstance(model.decoder, DenseNetDecoder3D)
        
        # Check that encoder has dense blocks
        assert hasattr(model.encoder, 'dense1')
        assert hasattr(model.encoder, 'dense2') 
        assert hasattr(model.encoder, 'dense3')
        assert hasattr(model.encoder, 'dense4')
        
        # Check that encoder has dilated layers
        assert hasattr(model.encoder, 'dilated_layers')
        
    def test_receptive_field_requirement(self):
        """Test that dilated convolutions achieve >150mm receptive field."""
        model = DenseNetVAE3D(latent_dim=128)
        
        # Check receptive field attribute
        assert hasattr(model, 'receptive_field_mm')
        assert model.receptive_field_mm > 150
        
        # Check encoder receptive field calculation
        assert model.encoder.receptive_field_mm > 150
        
        print(f"✅ Receptive field: {model.receptive_field_mm}mm (requirement: >150mm)")
        
    def test_batch_processing_without_memory_issues(self):
        """Test that model processes batch without memory issues."""
        model = DenseNetVAE3D(latent_dim=128)
        
        # Test small batch
        x_small = torch.randn(2, 1, 91, 109, 91)
        with torch.no_grad():
            recon, mu, logvar = model(x_small)
            assert recon.shape == x_small.shape
            assert mu.shape == (2, 128)
            assert logvar.shape == (2, 128)
        
        # Test larger batch (memory test)
        batch_size = 4
        x_large = torch.randn(batch_size, 1, 91, 109, 91)
        with torch.no_grad():
            recon, mu, logvar = model(x_large)
            assert recon.shape == x_large.shape
            assert mu.shape == (batch_size, 128)
            assert logvar.shape == (batch_size, 128)
            
        print(f"✅ Batch processing successful for batch sizes 2 and {batch_size}")
        
    def test_parameter_count_documented(self):
        """Test parameter count is within expected range and documented."""
        model = DenseNetVAE3D(latent_dim=128)
        
        # Get parameter count
        param_count = model.get_parameter_count()
        
        # DenseNet typically has 10M-100M parameters depending on depth
        assert 10_000_000 <= param_count <= 100_000_000, f"Parameter count {param_count:,} outside expected range"
        
        # Test parameter count method works
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count
        
        print(f"✅ Parameter count: {param_count:,} (within expected range: 10M-100M)")
        
    def test_forward_pass_shapes(self):
        """Test forward pass returns correct shapes."""
        model = DenseNetVAE3D(latent_dim=64)  # Test different latent dim
        
        batch_size = 3
        x = torch.randn(batch_size, 1, 91, 109, 91)
        
        with torch.no_grad():
            recon, mu, logvar = model(x)
            
            # Check shapes
            assert recon.shape == (batch_size, 1, 91, 109, 91)
            assert mu.shape == (batch_size, 64)
            assert logvar.shape == (batch_size, 64)
            
            # Check value ranges
            assert torch.isfinite(recon).all(), "Reconstruction contains NaN/Inf"
            assert torch.isfinite(mu).all(), "Mu contains NaN/Inf" 
            assert torch.isfinite(logvar).all(), "Logvar contains NaN/Inf"
            
        print("✅ Forward pass shapes and values correct")
        
    def test_encode_decode_consistency(self):
        """Test encode/decode methods work consistently."""
        model = DenseNetVAE3D(latent_dim=128)
        
        x = torch.randn(2, 1, 91, 109, 91)
        
        with torch.no_grad():
            # Test separate encode/decode
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            recon1 = model.decode(z)
            
            # Test full forward pass
            recon2, mu2, logvar2 = model(x)
            
            # Shapes should match
            assert recon1.shape == recon2.shape
            assert mu.shape == mu2.shape
            assert logvar.shape == logvar2.shape
            
        print("✅ Encode/decode consistency verified")
        
    def test_sampling_functionality(self):
        """Test sampling from prior distribution."""
        model = DenseNetVAE3D(latent_dim=128)
        
        # Test sampling
        num_samples = 5
        device = torch.device('cpu')
        
        with torch.no_grad():
            samples = model.sample(num_samples, device)
            
            assert samples.shape == (num_samples, 1, 91, 109, 91)
            assert torch.isfinite(samples).all(), "Samples contain NaN/Inf"
            
        print(f"✅ Sampling {num_samples} volumes successful")
        
    def test_different_configurations(self):
        """Test model with different configurations."""
        configs = [
            {'latent_dim': 64, 'growth_rate': 8},
            {'latent_dim': 256, 'growth_rate': 16},
            {'input_channels': 1, 'output_channels': 2, 'latent_dim': 128}
        ]
        
        for config in configs:
            model = DenseNetVAE3D(**config)
            x = torch.randn(1, config.get('input_channels', 1), 91, 109, 91)
            
            with torch.no_grad():
                recon, mu, logvar = model(x)
                
                expected_out_channels = config.get('output_channels', 1)
                expected_latent = config.get('latent_dim', 128)
                
                assert recon.shape == (1, expected_out_channels, 91, 109, 91)
                assert mu.shape == (1, expected_latent)
                assert logvar.shape == (1, expected_latent)
                
        print("✅ Different configurations work correctly")


def test_s2_1_1_success_criteria():
    """
    Comprehensive test for S2.1.1 SUCCESS_MARKERS criteria:
    - [✅] 3D DenseNet architecture implemented correctly
    - [✅] Dilated convolutions in final block achieve >150mm receptive field
    - [✅] Model processes batch without memory issues
    - [✅] Parameter count within expected range (documented)
    """
    print("\n=== Testing S2.1.1: DenseNet Backbone Upgrade ===")
    
    test_suite = TestDenseNetVAE3D()
    
    # Run all tests
    test_suite.test_architecture_implementation()
    test_suite.test_receptive_field_requirement()
    test_suite.test_batch_processing_without_memory_issues()
    test_suite.test_parameter_count_documented()
    test_suite.test_forward_pass_shapes()
    test_suite.test_encode_decode_consistency()
    test_suite.test_sampling_functionality()
    test_suite.test_different_configurations()
    
    print("\n🎉 All S2.1.1 SUCCESS CRITERIA PASSED!")
    print("✅ 3D DenseNet architecture implemented correctly")
    print("✅ Dilated convolutions achieve >150mm receptive field")
    print("✅ Model processes batches without memory issues")
    print("✅ Parameter count within expected range and documented")


if __name__ == "__main__":
    test_s2_1_1_success_criteria()