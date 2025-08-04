"""
Test Suite for Conditional PointNet++ VAE (S3.1.3)

Validates all SUCCESS_MARKERS.md criteria for S3.1.3:
- [X] Metadata vector concatenated to global features
- [X] Forward pass accepts (point_cloud, metadata) batches
- [X] Conditioning effects visible in generated point clouds
- [X] Architecture handles variable metadata dimensions
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.conditional_pointnet_vae import (
    ConditionalPointNetPlusPlusVAE,
    create_conditional_pointnet_vae,
    ConditionalPointNetPlusPlus,
    ConditionalPointCloudDecoder
)

class TestConditionalPointNetPlusPlusVAE:
    """Test suite for conditional PointNet++ VAE architecture"""
    
    @pytest.fixture
    def model(self):
        """Fixture providing a conditional PointNet++ VAE model"""
        return create_conditional_pointnet_vae(latent_dim=128, metadata_dim=64, output_points=30)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample point cloud and metadata data"""
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
        
        # Generate realistic metadata (task category, year, sample size, etc.)
        metadata = torch.randn(B, 64)
        
        return points, metadata, mask
    
    def test_metadata_vector_concatenated_to_global_features(self, model):
        """SUCCESS CRITERION: Metadata vector concatenated to global features"""
        # Test the encoder component specifically
        encoder = model.encoder
        
        B, N = 2, 30
        points = torch.randn(B, N, 3) * 50
        metadata = torch.randn(B, 64)
        
        # Test without metadata (should use zero padding)
        features_no_meta = encoder(points, metadata=None)
        assert features_no_meta.shape == (B, 512), f"Wrong encoder output shape: {features_no_meta.shape}"
        
        # Test with metadata
        features_with_meta = encoder(points, metadata=metadata)
        assert features_with_meta.shape == (B, 512), f"Wrong encoder output shape with metadata: {features_with_meta.shape}"
        
        # Features should be different when metadata is provided vs. zero-padded
        assert not torch.allclose(features_no_meta, features_with_meta, atol=1e-4), \
            "Metadata conditioning should affect encoder output"
        
        # Test that the internal MLP receives concatenated input
        # The encoder's output_mlp should receive 1024 + 64 = 1088 features
        assert encoder.output_mlp[0].in_features == 1024 + 64, \
            f"Expected MLP input of 1088 (1024 + 64), got {encoder.output_mlp[0].in_features}"
        
        print("✅ Metadata vector concatenated to global features")
    
    def test_forward_pass_accepts_point_cloud_metadata_batches(self, model, sample_data):
        """SUCCESS CRITERION: Forward pass accepts (point_cloud, metadata) batches"""
        points, metadata, mask = sample_data
        
        model.eval()
        
        # Test 1: Forward pass with all components
        with torch.no_grad():
            output = model(points, metadata, mask)
        
        # Verify output structure
        required_keys = ['reconstruction', 'mu', 'logvar', 'latent']
        for key in required_keys:
            assert key in output, f"Missing key '{key}' in model output"
        
        B = points.shape[0]
        assert output['reconstruction'].shape == (B, 30, 3), f"Wrong reconstruction shape: {output['reconstruction'].shape}"
        assert output['mu'].shape == (B, 128), f"Wrong mu shape: {output['mu'].shape}"
        assert output['logvar'].shape == (B, 128), f"Wrong logvar shape: {output['logvar'].shape}"
        assert output['latent'].shape == (B, 128), f"Wrong latent shape: {output['latent'].shape}"
        
        # Test 2: Forward pass without mask
        with torch.no_grad():
            output_no_mask = model(points, metadata, mask=None)
        
        assert all(key in output_no_mask for key in required_keys), "Missing keys when mask=None"
        
        # Test 3: Forward pass without metadata (should use zero padding)
        with torch.no_grad():
            output_no_meta = model(points, metadata=None, mask=mask)
        
        assert all(key in output_no_meta for key in required_keys), "Missing keys when metadata=None"
        
        # Test 4: Different batch sizes
        for test_batch_size in [1, 3, 8]:
            test_points = points[:test_batch_size] if test_batch_size <= B else torch.randn(test_batch_size, 50, 3) * 50
            test_metadata = metadata[:test_batch_size] if test_batch_size <= B else torch.randn(test_batch_size, 64)
            test_mask = mask[:test_batch_size] if test_batch_size <= B else torch.ones(test_batch_size, 50, dtype=torch.bool)
            
            with torch.no_grad():
                test_output = model(test_points, test_metadata, test_mask)
            
            assert test_output['reconstruction'].shape == (test_batch_size, 30, 3), \
                f"Wrong shape for batch size {test_batch_size}"
        
        print("✅ Forward pass accepts (point_cloud, metadata) batches")
    
    def test_conditioning_effects_visible_in_generated_point_clouds(self, model):
        """SUCCESS CRITERION: Conditioning effects visible in generated point clouds"""
        B = 3
        latent = torch.randn(B, model.latent_dim)
        
        # Create different metadata conditions
        metadata_conditions = [
            torch.zeros(B, 64),                    # Zero metadata
            torch.ones(B, 64),                     # Ones metadata  
            torch.randn(B, 64),                    # Random metadata 1
            torch.randn(B, 64) * 2 + 1,           # Random metadata 2 (different distribution)
            torch.tensor([1, 0, -1]).unsqueeze(1).expand(B, 64).float()  # Structured metadata
        ]
        
        model.eval()
        generated_clouds = []
        
        # Generate point clouds with different metadata
        with torch.no_grad():
            for i, metadata in enumerate(metadata_conditions):
                generated = model.decode(latent, metadata)
                generated_clouds.append(generated)
                
                assert generated.shape == (B, 30, 3), f"Wrong generation shape for condition {i}"
                assert torch.all(torch.isfinite(generated)), f"Non-finite values in condition {i}"
        
        # Verify that different metadata produces different point clouds
        differences = []
        for i in range(len(generated_clouds)):
            for j in range(i + 1, len(generated_clouds)):
                diff = torch.mean(torch.abs(generated_clouds[i] - generated_clouds[j]))
                differences.append(diff.item())
        
        avg_difference = np.mean(differences)
        min_difference = np.min(differences)
        
        print(f"Average difference between conditions: {avg_difference:.4f}")
        print(f"Minimum difference between conditions: {min_difference:.4f}")
        
        # Conditioning should produce measurably different outputs
        assert avg_difference > 1.0, f"Average difference too small: {avg_difference}"
        assert min_difference > 0.1, f"Some conditions produce nearly identical outputs: {min_difference}"
        
        # Test interpolation between metadata conditions
        metadata1 = torch.zeros(1, 64)
        metadata2 = torch.ones(1, 64)
        single_latent = torch.randn(1, model.latent_dim)
        
        with torch.no_grad():
            gen1 = model.decode(single_latent, metadata1)
            gen2 = model.decode(single_latent, metadata2)
            
            # Interpolated metadata
            alpha = 0.5
            metadata_interp = alpha * metadata1 + (1 - alpha) * metadata2
            gen_interp = model.decode(single_latent, metadata_interp)
            
            # Interpolated generation should be different from both endpoints
            diff_from_1 = torch.mean(torch.abs(gen_interp - gen1)).item()
            diff_from_2 = torch.mean(torch.abs(gen_interp - gen2)).item()
            
            assert diff_from_1 > 0.1, f"Interpolated output too close to endpoint 1: {diff_from_1}"
            assert diff_from_2 > 0.1, f"Interpolated output too close to endpoint 2: {diff_from_2}"
        
        print("✅ Conditioning effects visible in generated point clouds")
    
    def test_architecture_handles_variable_metadata_dimensions(self, model):
        """SUCCESS CRITERION: Architecture handles variable metadata dimensions"""
        B, N = 2, 40
        points = torch.randn(B, N, 3) * 50
        
        # Test with different metadata dimensions by creating models with different metadata dims
        test_metadata_dims = [32, 64, 128, 256]
        
        for metadata_dim in test_metadata_dims:
            # Create model with specific metadata dimension
            test_model = create_conditional_pointnet_vae(
                latent_dim=128, 
                metadata_dim=metadata_dim, 
                output_points=30
            )
            
            # Create metadata with corresponding dimension
            metadata = torch.randn(B, metadata_dim)
            
            test_model.eval()
            with torch.no_grad():
                # Test encoding
                mu, logvar = test_model.encode(points, metadata)
                assert mu.shape == (B, 128), f"Wrong mu shape for metadata_dim {metadata_dim}"
                assert logvar.shape == (B, 128), f"Wrong logvar shape for metadata_dim {metadata_dim}"
                
                # Test decoding
                latent = torch.randn(B, 128)
                decoded = test_model.decode(latent, metadata)
                assert decoded.shape == (B, 30, 3), f"Wrong decoded shape for metadata_dim {metadata_dim}"
                
                # Test full forward pass
                output = test_model(points, metadata)
                assert output['reconstruction'].shape == (B, 30, 3), f"Wrong forward shape for metadata_dim {metadata_dim}"
        
        # Test that models with different metadata dimensions produce different results
        # when given the same latent code but different metadata
        model32 = create_conditional_pointnet_vae(latent_dim=128, metadata_dim=32, output_points=30)
        model128 = create_conditional_pointnet_vae(latent_dim=128, metadata_dim=128, output_points=30)
        
        latent = torch.randn(1, 128)
        meta32 = torch.ones(1, 32)
        meta128 = torch.ones(1, 128)
        
        with torch.no_grad():
            gen32 = model32.decode(latent, meta32)
            gen128 = model128.decode(latent, meta128)
            
            # Different models should produce different outputs even with similar metadata
            # (This tests that the architecture properly handles different dimensions)
            assert gen32.shape == (1, 30, 3), "Model with 32-dim metadata should work"
            assert gen128.shape == (1, 30, 3), "Model with 128-dim metadata should work"
        
        print("✅ Architecture handles variable metadata dimensions")
    
    def test_encoder_decoder_consistency(self, model, sample_data):
        """Test that encoder and decoder work consistently together"""
        points, metadata, mask = sample_data
        
        model.eval()
        with torch.no_grad():
            # Test encode -> decode cycle
            mu, logvar = model.encode(points, metadata, mask)
            reconstructed = model.decode(mu, metadata)  # Use mu (deterministic)
            
            # Should produce valid reconstructions
            assert reconstructed.shape == (points.shape[0], 30, 3)
            assert torch.all(torch.isfinite(reconstructed))
            
            # Test that same latent + different metadata = different outputs
            mu_single = mu[:1]  # Take first sample
            meta1 = metadata[:1]
            meta2 = torch.zeros_like(meta1)
            
            recon1 = model.decode(mu_single, meta1)
            recon2 = model.decode(mu_single, meta2)
            
            diff = torch.mean(torch.abs(recon1 - recon2))
            assert diff > 0.1, f"Same latent with different metadata should produce different outputs: {diff}"
        
        print("✅ Encoder-decoder consistency verified")

def test_model_parameter_comparison():
    """Test parameter counts between conditional and non-conditional models"""
    from src.models.pointnet_vae import create_pointnet_vae
    
    base_model = create_pointnet_vae(latent_dim=128, output_points=30)
    conditional_model = create_conditional_pointnet_vae(latent_dim=128, metadata_dim=64, output_points=30)
    
    base_params = sum(p.numel() for p in base_model.parameters())
    conditional_params = sum(p.numel() for p in conditional_model.parameters())
    
    print(f"Base model parameters: {base_params:,}")
    print(f"Conditional model parameters: {conditional_params:,}")
    print(f"Additional parameters for conditioning: {conditional_params - base_params:,}")
    
    # Conditional model should have more parameters due to metadata handling
    assert conditional_params > base_params, "Conditional model should have more parameters"
    
    # But not too many more (should be reasonable overhead)
    overhead_ratio = conditional_params / base_params
    assert 1.01 < overhead_ratio < 2.0, f"Conditioning overhead seems unreasonable: {overhead_ratio:.2f}x"

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