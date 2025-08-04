"""
Test Suite for PointNet++ VAE Architecture (S3.1.2)

Validates all SUCCESS_MARKERS.md criteria for S3.1.2:
- [X] PointNet++ backbone processes padded point clouds
- [X] MLP decoder generates fixed-size point sets (N=30)
- [X] Gaussian Random Fourier Features implemented
- [X] Model instantiation and forward pass successful
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.pointnet_vae import (
    PointNetPlusPlusVAE, 
    create_pointnet_vae,
    GaussianRandomFourierFeatures,
    PointNetPlusPlus,
    PointCloudDecoder
)

class TestPointNetPlusPlusVAE:
    """Test suite for PointNet++ VAE architecture"""
    
    @pytest.fixture
    def model(self):
        """Fixture providing a PointNet++ VAE model"""
        return create_pointnet_vae(latent_dim=128, output_points=30)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample point cloud data"""
        B, N = 4, 50  # Batch size 4, up to 50 points
        
        # Generate brain-like coordinates (MNI152 range)
        points = torch.randn(B, N, 3)
        points[:, :, 0] = points[:, :, 0] * 40  # x: roughly [-80, 80]
        points[:, :, 1] = points[:, :, 1] * 50 - 20  # y: roughly [-70, 30]  
        points[:, :, 2] = points[:, :, 2] * 40 + 20  # z: roughly [-20, 60]
        
        # Create realistic masks (some studies have fewer points)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[1, 30:] = False  # Second sample has only 30 points
        mask[3, 20:] = False  # Fourth sample has only 20 points
        
        return points, mask
    
    def test_model_instantiation_and_forward_pass_successful(self, model, sample_data):
        """SUCCESS CRITERION: Model instantiation and forward pass successful"""
        points, mask = sample_data
        
        # Test that model was created successfully
        assert isinstance(model, PointNetPlusPlusVAE), "Model should be PointNetPlusPlusVAE instance"
        assert model.latent_dim == 128, "Latent dimension should be 128"
        assert model.output_points == 30, "Output points should be 30"
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(points, mask)
        
        # Verify output structure
        required_keys = ['reconstruction', 'mu', 'logvar', 'latent']
        for key in required_keys:
            assert key in output, f"Missing key '{key}' in model output"
        
        # Verify tensor shapes
        B = points.shape[0]
        assert output['reconstruction'].shape == (B, 30, 3), f"Wrong reconstruction shape: {output['reconstruction'].shape}"
        assert output['mu'].shape == (B, 128), f"Wrong mu shape: {output['mu'].shape}"
        assert output['logvar'].shape == (B, 128), f"Wrong logvar shape: {output['logvar'].shape}"
        assert output['latent'].shape == (B, 128), f"Wrong latent shape: {output['latent'].shape}"
        
        print("✅ Model instantiation and forward pass successful")
    
    def test_pointnet_backbone_processes_padded_point_clouds(self, model, sample_data):
        """SUCCESS CRITERION: PointNet++ backbone processes padded point clouds"""
        points, mask = sample_data
        
        # Test with different mask configurations
        test_masks = [
            None,  # No masking
            mask,  # Original mask with variable lengths
            torch.ones_like(mask),  # All valid points
            torch.zeros_like(mask)  # All masked (edge case)
        ]
        
        model.eval()
        for i, test_mask in enumerate(test_masks):
            with torch.no_grad():
                if i == 3:  # Skip all-masked case for encoder (would be invalid)
                    continue
                    
                # Test encoder specifically
                mu, logvar = model.encode(points, test_mask)
                
                assert mu.shape == (points.shape[0], model.latent_dim), f"Wrong mu shape with mask {i}"
                assert logvar.shape == (points.shape[0], model.latent_dim), f"Wrong logvar shape with mask {i}"
                
                # Check that outputs are finite (not NaN/Inf)
                assert torch.all(torch.isfinite(mu)), f"Non-finite mu values with mask {i}"
                assert torch.all(torch.isfinite(logvar)), f"Non-finite logvar values with mask {i}"
        
        # Test that different masks produce different outputs (when appropriate)
        with torch.no_grad():
            mu1, _ = model.encode(points, None)
            mu2, _ = model.encode(points, mask)
            
            # They should be different because masking affects global max pooling
            assert not torch.allclose(mu1, mu2, atol=1e-6), "Masking should affect encoder output"
        
        print("✅ PointNet++ backbone processes padded point clouds correctly")
    
    def test_mlp_decoder_generates_fixed_size_point_sets(self, model):
        """SUCCESS CRITERION: MLP decoder generates fixed-size point sets (N=30)"""
        # Test decoder with various latent inputs
        batch_sizes = [1, 4, 8]
        
        model.eval()
        for B in batch_sizes:
            # Create random latent vectors
            latent = torch.randn(B, model.latent_dim)
            
            with torch.no_grad():
                decoded_points = model.decode(latent)
            
            # Verify fixed output size
            assert decoded_points.shape == (B, 30, 3), f"Wrong decoded shape for batch size {B}: {decoded_points.shape}"
            
            # Verify coordinate ranges are reasonable (brain-like)
            assert torch.all(decoded_points[:, :, 0] >= -100) and torch.all(decoded_points[:, :, 0] <= 100), "X coordinates out of range"
            assert torch.all(decoded_points[:, :, 1] >= -150) and torch.all(decoded_points[:, :, 1] <= 100), "Y coordinates out of range"
            assert torch.all(decoded_points[:, :, 2] >= -100) and torch.all(decoded_points[:, :, 2] <= 120), "Z coordinates out of range"
            
            # Verify all values are finite
            assert torch.all(torch.isfinite(decoded_points)), f"Non-finite decoded points for batch size {B}"
        
        # Test that different latent inputs produce different outputs
        latent1 = torch.randn(2, model.latent_dim)
        latent2 = torch.randn(2, model.latent_dim)
        
        with torch.no_grad():
            decoded1 = model.decode(latent1)
            decoded2 = model.decode(latent2)
        
        assert not torch.allclose(decoded1, decoded2, atol=1e-4), "Different latent inputs should produce different outputs"
        
        print("✅ MLP decoder generates fixed-size point sets (N=30)")
    
    def test_gaussian_random_fourier_features_implemented(self):
        """SUCCESS CRITERION: Gaussian Random Fourier Features implemented"""
        # Test Gaussian Random Fourier Features component
        grff = GaussianRandomFourierFeatures(input_dim=3, feature_dim=256, sigma=1.0)
        
        # Test with sample coordinates
        B, N = 2, 20
        coords = torch.randn(B, N, 3) * 50  # Brain-scale coordinates
        
        features = grff(coords)
        
        # Verify output shape
        assert features.shape == (B, N, 256), f"Wrong GRFF output shape: {features.shape}"
        
        # Verify feature values are in reasonable range (cosine/sine outputs)
        assert torch.all(features >= -1.1) and torch.all(features <= 1.1), "GRFF features out of expected range"
        
        # Test that the random matrix B is correctly sized
        assert grff.B.shape == (3, 128), f"Wrong random matrix B shape: {grff.B.shape}"
        
        # Test that different coordinates produce different features
        coords2 = coords + torch.randn_like(coords) * 10
        features2 = grff(coords2)
        
        assert not torch.allclose(features, features2, atol=1e-4), "Different coordinates should produce different features"
        
        # Test that features are deterministic (same input -> same output)
        features_repeat = grff(coords)
        assert torch.allclose(features, features_repeat), "GRFF should be deterministic"
        
        print("✅ Gaussian Random Fourier Features implemented correctly")
    
    def test_vae_components_working(self, model, sample_data):
        """Test VAE-specific components (reparameterization, latent space)"""
        points, mask = sample_data
        
        model.eval()
        
        # Test reparameterization trick
        with torch.no_grad():
            mu, logvar = model.encode(points, mask)
            
            # Test deterministic mode (eval)
            z1 = model.reparameterize(mu, logvar)
            z2 = model.reparameterize(mu, logvar)
            assert torch.allclose(z1, z2), "Reparameterization should be deterministic in eval mode"
            assert torch.allclose(z1, mu), "In eval mode, reparameterization should return mu"
        
        # Test stochastic mode (training)
        model.train()
        with torch.no_grad():
            z1 = model.reparameterize(mu, logvar)
            z2 = model.reparameterize(mu, logvar)
            assert not torch.allclose(z1, z2, atol=1e-6), "Reparameterization should be stochastic in train mode"
        
        model.eval()
        
        # Test latent space interpolation
        with torch.no_grad():
            # Get latent codes for two different point clouds
            mu1, _ = model.encode(points[:1], mask[:1] if mask is not None else None)
            mu2, _ = model.encode(points[1:2], mask[1:2] if mask is not None else None)
            
            # Interpolate in latent space
            alpha = 0.5
            z_interp = alpha * mu1 + (1 - alpha) * mu2
            
            # Decode interpolated latent
            decoded_interp = model.decode(z_interp)
            assert decoded_interp.shape == (1, 30, 3), "Interpolated decoding should work"
        
        print("✅ VAE components working correctly")
    
    def test_model_with_real_brain_coordinates(self):
        """Test model with realistic brain coordinate ranges"""
        model = create_pointnet_vae()
        
        # Create realistic brain coordinates (MNI152 space)
        B, N = 3, 40
        points = torch.zeros(B, N, 3)
        
        # Sample 1: Visual cortex coordinates
        points[0, :10, :] = torch.tensor([
            [12, -88, -8], [-8, -92, -4], [16, -84, -12], [-12, -96, 0],
            [20, -80, -16], [-16, -92, -8], [8, -84, 4], [-4, -88, 8],
            [24, -76, -20], [-20, -84, -4]
        ], dtype=torch.float32)
        
        # Sample 2: Motor cortex coordinates  
        points[1, :8, :] = torch.tensor([
            [-36, -24, 58], [38, -20, 56], [-40, -28, 62], [42, -16, 54],
            [-32, -20, 54], [34, -24, 60], [-44, -32, 66], [46, -12, 52]
        ], dtype=torch.float32)
        
        # Sample 3: Default mode network coordinates
        points[2, :12, :] = torch.tensor([
            [0, 52, -2], [-46, -66, 36], [46, -62, 32], [0, -54, 28],
            [-8, -56, 6], [8, -52, 10], [-2, 48, 6], [2, 44, 2],
            [-50, -70, 40], [52, -58, 28], [-4, -58, 32], [6, -50, 24]
        ], dtype=torch.float32)
        
        # Create appropriate masks
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, :10] = True
        mask[1, :8] = True  
        mask[2, :12] = True
        
        model.eval()
        with torch.no_grad():
            output = model(points, mask)
            
            # Verify processing works with real coordinates
            assert output['reconstruction'].shape == (B, 30, 3)
            assert torch.all(torch.isfinite(output['reconstruction']))
            assert torch.all(torch.isfinite(output['mu']))
            assert torch.all(torch.isfinite(output['logvar']))
        
        print("✅ Model works with realistic brain coordinates")

def test_model_parameter_count():
    """Test that model has reasonable parameter count"""
    model = create_pointnet_vae()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Should be reasonable for point cloud processing (not too large, not too small)
    assert 100_000 < total_params < 10_000_000, f"Parameter count seems unreasonable: {total_params:,}"
    assert trainable_params == total_params, "All parameters should be trainable"

if __name__ == '__main__':
    # Run tests if called directly
    import subprocess
    import sys
    
    result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    sys.exit(result.returncode)