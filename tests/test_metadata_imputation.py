"""
Test suite for S2.1.2: Metadata Imputation with Amortization Head.

Tests success criteria:
- Amortization head outputs (μ, log σ²) for missing metadata
- Imputation loss term integrated into total loss  
- Forward pass returns imputed values with uncertainty
- Uncertainty propagation via reparameterization trick works
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.metadata_imputation import MetadataImputation, create_default_metadata_config, create_mock_metadata_batch
from src.models.conditional_densenet_vae import ConditionalDenseNetVAE3D


class TestMetadataImputation:
    """Test suite for metadata imputation."""
    
    def test_amortization_head_outputs(self):
        """Test that amortization head outputs (μ, log σ²) for missing metadata."""
        metadata_config = create_default_metadata_config()
        feature_dim = 512
        batch_size = 4
        
        imputer = MetadataImputation(feature_dim, metadata_config)
        features = torch.randn(batch_size, feature_dim)
        
        # Test with no observed metadata (all imputed)
        with torch.no_grad():
            imputed = imputer(features)
            
            # Check that continuous fields have mu, logvar, and uncertainty outputs
            for field_name, config in metadata_config.items():
                if config['type'] == 'continuous':
                    assert f'{field_name}_mu' in imputed, f"Missing mu for {field_name}"
                    assert f'{field_name}_logvar' in imputed, f"Missing logvar for {field_name}"
                    assert f'{field_name}_uncertainty' in imputed, f"Missing uncertainty for {field_name}"
                    
                    mu = imputed[f'{field_name}_mu']
                    logvar = imputed[f'{field_name}_logvar']
                    uncertainty = imputed[f'{field_name}_uncertainty']
                    
                    # Check shapes
                    expected_shape = (batch_size, config['dim'])
                    assert mu.shape == expected_shape, f"Wrong mu shape for {field_name}"
                    assert logvar.shape == expected_shape, f"Wrong logvar shape for {field_name}"
                    assert uncertainty.shape == expected_shape, f"Wrong uncertainty shape for {field_name}"
                    
                    # Check that uncertainty = exp(0.5 * logvar)
                    expected_uncertainty = torch.exp(0.5 * logvar)
                    assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6), f"Uncertainty calculation wrong for {field_name}"
                    
        print("✅ Amortization head outputs (μ, log σ²) correctly")
        
    def test_imputation_loss_integration(self):
        """Test that imputation loss term is integrated into total loss."""
        model = ConditionalDenseNetVAE3D(latent_dim=64)
        batch_size = 2
        
        x = torch.randn(batch_size, 1, 91, 109, 91)
        observed_metadata, missing_mask = create_mock_metadata_batch(
            batch_size, model.metadata_config, missing_rate=0.5
        )
        
        with torch.no_grad():
            recon, mu, logvar, imputed_metadata = model(x, observed_metadata, missing_mask)
            
            # Compute losses
            losses = model.compute_total_loss(x, recon, mu, logvar, imputed_metadata, 
                                            observed_metadata, missing_mask)
            
            # Check that all loss components exist
            assert 'total_loss' in losses, "Missing total_loss"
            assert 'recon_loss' in losses, "Missing recon_loss"
            assert 'kl_loss' in losses, "Missing kl_loss"
            assert 'imputation_loss' in losses, "Missing imputation_loss"
            
            # Check that imputation loss is non-zero when we have observed data
            assert losses['imputation_loss'] > 0, "Imputation loss should be > 0 with observed data"
            
            # Check that total loss includes imputation loss
            expected_total = losses['recon_loss'] + losses['kl_loss'] + losses['imputation_loss']
            assert torch.allclose(losses['total_loss'], expected_total, atol=1e-4), "Total loss doesn't match sum of components"
            
        print("✅ Imputation loss integrated into total loss")
        
    def test_forward_pass_returns_imputed_values_with_uncertainty(self):
        """Test that forward pass returns imputed values with uncertainty."""
        model = ConditionalDenseNetVAE3D(latent_dim=64)
        batch_size = 3
        
        x = torch.randn(batch_size, 1, 91, 109, 91)
        observed_metadata, missing_mask = create_mock_metadata_batch(
            batch_size, model.metadata_config, missing_rate=0.4
        )
        
        with torch.no_grad():
            recon, mu, logvar, imputed_metadata = model(x, observed_metadata, missing_mask)
            
            # Check that we get all expected outputs
            assert recon.shape == x.shape, "Wrong reconstruction shape"
            assert mu.shape == (batch_size, model.latent_dim), "Wrong mu shape"
            assert logvar.shape == (batch_size, model.latent_dim), "Wrong logvar shape"
            assert isinstance(imputed_metadata, dict), "Imputed metadata should be dict"
            
            # Check that imputed metadata has correct structure
            for field_name, config in model.metadata_config.items():
                assert field_name in imputed_metadata, f"Missing imputed field {field_name}"
                
                imputed_value = imputed_metadata[field_name]
                assert imputed_value.shape[0] == batch_size, f"Wrong batch size for {field_name}"
                
                if config['type'] == 'continuous':
                    # Should have uncertainty estimates
                    assert f'{field_name}_mu' in imputed_metadata, f"Missing mu for {field_name}"
                    assert f'{field_name}_logvar' in imputed_metadata, f"Missing logvar for {field_name}"
                    assert f'{field_name}_uncertainty' in imputed_metadata, f"Missing uncertainty for {field_name}"
                    
                    # Check uncertainty is positive
                    uncertainty = imputed_metadata[f'{field_name}_uncertainty']
                    assert (uncertainty > 0).all(), f"Uncertainty should be positive for {field_name}"
                    
        print("✅ Forward pass returns imputed values with uncertainty")
        
    def test_uncertainty_propagation_reparameterization_trick(self):
        """Test that uncertainty propagation via reparameterization trick works."""
        metadata_config = create_default_metadata_config()
        feature_dim = 256
        batch_size = 5
        
        imputer = MetadataImputation(feature_dim, metadata_config)
        features = torch.randn(batch_size, feature_dim)
        
        # Set to eval mode to make mu/logvar deterministic
        imputer.eval()
        
        # Test multiple forward passes to check stochasticity
        outputs1 = imputer(features)
        outputs2 = imputer(features)
        
        # Check that continuous fields use reparameterization trick (stochastic)
        for field_name, config in metadata_config.items():
            if config['type'] == 'continuous':
                value1 = outputs1[field_name]
                value2 = outputs2[field_name]
                
                # Values should be different due to sampling (reparameterization trick)
                assert not torch.allclose(value1, value2, atol=1e-3), f"Values should be stochastic for {field_name}"
                
                # But mu and logvar should be the same (deterministic from features)
                mu1 = outputs1[f'{field_name}_mu']
                mu2 = outputs2[f'{field_name}_mu']
                logvar1 = outputs1[f'{field_name}_logvar']
                logvar2 = outputs2[f'{field_name}_logvar']
                
                assert torch.allclose(mu1, mu2, atol=1e-6), f"Mu should be deterministic for {field_name}"
                assert torch.allclose(logvar1, logvar2, atol=1e-6), f"Logvar should be deterministic for {field_name}"
                
        # Test in eval mode (should be deterministic)
        imputer.eval()
        with torch.no_grad():
            outputs3 = imputer(features)
            outputs4 = imputer(features)
            
            for field_name, config in metadata_config.items():
                if config['type'] == 'continuous':
                    value3 = outputs3[field_name]
                    value4 = outputs4[field_name]
                    
                    # In eval mode, values should still be stochastic due to sampling
                    # (This tests that reparameterization is working)
                    assert not torch.allclose(value3, value4, atol=1e-3), f"Reparameterization should still work in eval mode for {field_name}"
        
        print("✅ Uncertainty propagation via reparameterization trick works")
        
    def test_metadata_conditioning_integration(self):
        """Test that metadata conditioning works in the full model."""
        model = ConditionalDenseNetVAE3D(latent_dim=64)
        batch_size = 2
        
        x = torch.randn(batch_size, 1, 91, 109, 91)
        
        # Test with different metadata configurations
        # Case 1: No metadata (all imputed)
        with torch.no_grad():
            recon1, mu1, logvar1, imputed1 = model(x)
            
        # Case 2: Some observed metadata
        observed_metadata, missing_mask = create_mock_metadata_batch(
            batch_size, model.metadata_config, missing_rate=0.3
        )
        with torch.no_grad():
            recon2, mu2, logvar2, imputed2 = model(x, observed_metadata, missing_mask)
            
        # Case 3: All metadata observed
        observed_metadata_full, missing_mask_full = create_mock_metadata_batch(
            batch_size, model.metadata_config, missing_rate=0.0
        )
        with torch.no_grad():
            recon3, mu3, logvar3, imputed3 = model(x, observed_metadata_full, missing_mask_full)
            
        # Check that outputs have correct shapes
        for recon, mu, logvar in [(recon1, mu1, logvar1), (recon2, mu2, logvar2), (recon3, mu3, logvar3)]:
            assert recon.shape == x.shape
            assert mu.shape == (batch_size, model.latent_dim)
            assert logvar.shape == (batch_size, model.latent_dim)
            
        # Different metadata should produce different reconstructions
        assert not torch.allclose(recon1, recon2, atol=1e-2), "Different metadata should produce different reconstructions"
        assert not torch.allclose(recon2, recon3, atol=1e-2), "Different metadata should produce different reconstructions"
        
        print("✅ Metadata conditioning integration works")


def test_s2_1_2_success_criteria():
    """
    Comprehensive test for S2.1.2 SUCCESS_MARKERS criteria:
    - [✅] Amortization head outputs (μ, log σ²) for missing metadata
    - [✅] Imputation loss term integrated into total loss
    - [✅] Forward pass returns imputed values with uncertainty
    - [✅] Uncertainty propagation via reparameterization trick works
    """
    print("\n=== Testing S2.1.2: Metadata Imputation with Amortization Head ===")
    
    test_suite = TestMetadataImputation()
    
    # Run all tests
    test_suite.test_amortization_head_outputs()
    test_suite.test_imputation_loss_integration()
    test_suite.test_forward_pass_returns_imputed_values_with_uncertainty()
    test_suite.test_uncertainty_propagation_reparameterization_trick()
    test_suite.test_metadata_conditioning_integration()
    
    print("\n🎉 All S2.1.2 SUCCESS CRITERIA PASSED!")
    print("✅ Amortization head outputs (μ, log σ²) for missing metadata")
    print("✅ Imputation loss term integrated into total loss")
    print("✅ Forward pass returns imputed values with uncertainty")
    print("✅ Uncertainty propagation via reparameterization trick works")


if __name__ == "__main__":
    test_s2_1_2_success_criteria()