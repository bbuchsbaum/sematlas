"""
Unit tests for ResNet VAE model.

Tests the S1.2.1 acceptance criteria:
- Model can be instantiated
- Dummy tensor passes through model without error
- Group Normalization with groups=8 implemented
- Encoder outputs μ and log σ² with correct shapes
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.resnet_vae import ResNetVAE3D, create_resnet_vae


class TestResNetVAE(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.batch_size = 2
        self.latent_dim = 32
        self.groups = 8
        self.input_shape = (self.batch_size, 1, 91, 109, 91)
        
    def test_model_instantiation(self):
        """Test that model can be instantiated without errors."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        self.assertIsInstance(model, ResNetVAE3D)
        self.assertEqual(model.latent_dim, self.latent_dim)
        
    def test_dummy_tensor_forward_pass(self):
        """Test that dummy tensor passes through model successfully."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        dummy_input = torch.randn(*self.input_shape)
        
        # Should not raise any exceptions
        with torch.no_grad():
            reconstruction, mu, logvar = model(dummy_input)
            
        # Check output shapes
        self.assertEqual(reconstruction.shape, self.input_shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
    def test_group_normalization_implementation(self):
        """Test that Group Normalization with groups=8 is implemented."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        
        # Count GroupNorm layers and verify groups
        group_norm_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.GroupNorm):
                group_norm_layers.append((name, module))
                self.assertEqual(module.num_groups, self.groups, 
                               f"GroupNorm layer {name} should have {self.groups} groups")
        
        # Should have multiple GroupNorm layers
        self.assertGreater(len(group_norm_layers), 5, 
                          "Model should have multiple GroupNorm layers")
        
    def test_encoder_output_shapes(self):
        """Test that encoder outputs μ and log σ² with correct shapes."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        dummy_input = torch.randn(*self.input_shape)
        
        with torch.no_grad():
            mu, logvar = model.encode(dummy_input)
            
        # Check that outputs are separate tensors with correct shapes
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        
        # Check that mu and logvar are different (not the same tensor)
        self.assertFalse(torch.equal(mu, logvar))
        
    def test_reparameterization_trick(self):
        """Test that reparameterization trick works correctly."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)
        
        with torch.no_grad():
            z = model.reparameterize(mu, logvar)
            
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        
    def test_decoder_functionality(self):
        """Test that decoder produces correct output shape."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        latent_vector = torch.randn(self.batch_size, self.latent_dim)
        
        with torch.no_grad():
            reconstruction = model.decode(latent_vector)
            
        self.assertEqual(reconstruction.shape, self.input_shape)
        
    def test_sampling_functionality(self):
        """Test that sampling from latent space works."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        num_samples = 3
        
        with torch.no_grad():
            samples = model.sample(num_samples, device=self.device)
            
        expected_shape = (num_samples, 1, 91, 109, 91)
        self.assertEqual(samples.shape, expected_shape)
        
    def test_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have parameters and all should be trainable
        self.assertGreater(total_params, 1000000)  # At least 1M parameters
        self.assertEqual(total_params, trainable_params)  # All should be trainable
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model correctly."""
        model = create_resnet_vae(latent_dim=self.latent_dim, groups=self.groups)
        dummy_input = torch.randn(*self.input_shape, requires_grad=True)
        
        reconstruction, mu, logvar = model(dummy_input)
        
        # Simple loss
        loss = torch.mean(reconstruction) + torch.mean(mu) + torch.mean(logvar)
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(dummy_input.grad)
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == '__main__':
    unittest.main()