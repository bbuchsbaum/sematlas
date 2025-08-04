"""
Unit tests for VAE Lightning Module.

Tests the S1.2.3 acceptance criteria:
- Module can be initialized with ResNet model
- VAE loss function works correctly (reconstruction + KL divergence)
- Reparameterization trick is implemented
- Training and validation steps work
- Optimizer configuration works
"""

import unittest
import torch
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.vae_lightning import (
    VAELightningModule,
    create_vae_lightning_module
)


class TestVAELightningModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.batch_size = 2
        self.latent_dim = 16  # Smaller for testing
        self.groups = 8
        
        # Create dummy batch
        self.dummy_batch = {
            'volume': torch.randn(self.batch_size, 1, 91, 109, 91),
            'study_id': [f'test_study_{i}' for i in range(self.batch_size)],
            'kernel_used': ['6mm'] * self.batch_size,
            'metadata': [{'contrast': f'test_{i}'} for i in range(self.batch_size)]
        }
    
    def test_module_initialization(self):
        """Test that module can be initialized with ResNet model."""
        model = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            learning_rate=1e-4,
            beta_vae=0.5
        )
        
        # Check that model is created
        self.assertIsInstance(model, VAELightningModule)
        self.assertIsNotNone(model.vae)
        
        # Check hyperparameters
        self.assertEqual(model.hparams.latent_dim, self.latent_dim)
        self.assertEqual(model.hparams.groups, self.groups)
        self.assertEqual(model.hparams.learning_rate, 1e-4)
        self.assertEqual(model.hparams.beta_vae, 0.5)
    
    def test_factory_function(self):
        """Test factory function creates module correctly."""
        model = create_vae_lightning_module(
            latent_dim=self.latent_dim,
            groups=self.groups,
            beta_vae=0.2
        )
        
        self.assertIsInstance(model, VAELightningModule)
        self.assertEqual(model.hparams.latent_dim, self.latent_dim)
        self.assertEqual(model.hparams.beta_vae, 0.2)
    
    def test_forward_pass(self):
        """Test forward pass through the module."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        x = self.dummy_batch['volume']
        
        with torch.no_grad():
            reconstruction, mu, logvar = model(x)
        
        # Check output shapes
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
    
    def test_encode_decode_methods(self):
        """Test encode and decode methods work correctly."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        x = self.dummy_batch['volume']
        
        with torch.no_grad():
            # Test encode
            mu, logvar = model.encode(x)
            self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
            self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
            
            # Test decode
            z = torch.randn(self.batch_size, self.latent_dim)
            reconstruction = model.decode(z)
            self.assertEqual(reconstruction.shape, x.shape)
    
    def test_sampling_functionality(self):
        """Test sampling from latent space."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        num_samples = 3
        
        with torch.no_grad():
            samples = model.sample(num_samples)
        
        expected_shape = (num_samples, 1, 91, 109, 91)
        self.assertEqual(samples.shape, expected_shape)
    
    def test_vae_loss_function(self):
        """Test VAE loss function (reconstruction + KL divergence)."""
        model = VAELightningModule(
            latent_dim=self.latent_dim, 
            groups=self.groups,
            beta_vae=1.0,
            reconstruction_loss="mse"
        )
        
        x = self.dummy_batch['volume']
        
        with torch.no_grad():
            reconstruction, mu, logvar = model(x)
            
            # Test loss computation
            total_loss, loss_dict = model.vae_loss(reconstruction, x, mu, logvar)
            
            # Check that loss components exist
            self.assertIn('total_loss', loss_dict)
            self.assertIn('recon_loss', loss_dict)
            self.assertIn('kl_loss', loss_dict)
            self.assertIn('beta', loss_dict)
            
            # Check that losses are tensors with correct properties
            self.assertIsInstance(total_loss, torch.Tensor)
            self.assertEqual(total_loss.shape, ())  # Scalar
            self.assertTrue(total_loss.item() >= 0)  # Non-negative
            
            # Check that total loss equals recon + beta * kl
            expected_total = loss_dict['recon_loss'] + loss_dict['beta'] * loss_dict['kl_loss']
            self.assertAlmostEqual(
                total_loss.item(), 
                expected_total.item(), 
                places=5
            )
    
    def test_beta_scheduling(self):
        """Test different beta scheduling strategies."""
        # Test constant beta
        model_const = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            beta_vae=0.5,
            beta_schedule="constant"
        )
        self.assertEqual(model_const.get_current_beta(), 0.5)
        
        # Test linear beta scheduling
        model_linear = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            beta_vae=1.0,
            beta_schedule="linear",
            max_epochs=100
        )
        
        # At epoch 0, should be 0
        model_linear.current_epoch = 0
        self.assertEqual(model_linear.get_current_beta(), 0.0)
        
        # At epoch 50 (half of max), should be 1.0
        model_linear.current_epoch = 50
        self.assertEqual(model_linear.get_current_beta(), 1.0)
        
        # Test cyclical beta scheduling
        model_cyclical = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            beta_vae=1.0,
            beta_schedule="cyclical"
        )
        
        # At epoch 0, should be 0
        model_cyclical.current_epoch = 0
        self.assertAlmostEqual(model_cyclical.get_current_beta(), 0.0, places=5)
        
        # At epoch 5 (middle of cycle), should be 0.5
        model_cyclical.current_epoch = 5
        self.assertAlmostEqual(model_cyclical.get_current_beta(), 0.5, places=5)
    
    def test_training_step(self):
        """Test training step functionality."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        model.train()
        
        # Run training step
        loss = model.training_step(self.dummy_batch, batch_idx=0)
        
        # Check that loss is returned and is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(loss.item() >= 0)  # Non-negative
        
        # Check that outputs were stored
        self.assertEqual(len(model.training_step_outputs), 1)
        self.assertIn('loss', model.training_step_outputs[0])
    
    def test_validation_step(self):
        """Test validation step functionality."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        model.eval()
        
        # Run validation step
        with torch.no_grad():
            loss = model.validation_step(self.dummy_batch, batch_idx=0)
        
        # Check that loss is returned and is a tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(loss.item() >= 0)  # Non-negative
        
        # Check that outputs were stored
        self.assertEqual(len(model.validation_step_outputs), 1)
        self.assertIn('loss', model.validation_step_outputs[0])
    
    def test_epoch_end_methods(self):
        """Test epoch end methods clear outputs correctly."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        # Simulate some training steps
        model.training_step(self.dummy_batch, 0)
        model.training_step(self.dummy_batch, 1)
        self.assertEqual(len(model.training_step_outputs), 2)
        
        # Call epoch end
        model.on_train_epoch_end()
        self.assertEqual(len(model.training_step_outputs), 0)
        
        # Same for validation
        model.validation_step(self.dummy_batch, 0)
        model.validation_step(self.dummy_batch, 1)
        self.assertEqual(len(model.validation_step_outputs), 2)
        
        model.on_validation_epoch_end()
        self.assertEqual(len(model.validation_step_outputs), 0)
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            learning_rate=2e-4,
            weight_decay=1e-5,
            max_epochs=50
        )
        
        optimizer_config = model.configure_optimizers()
        
        # Check structure
        self.assertIn('optimizer', optimizer_config)
        self.assertIn('lr_scheduler', optimizer_config)
        
        optimizer = optimizer_config['optimizer']
        scheduler_config = optimizer_config['lr_scheduler']
        
        # Check optimizer type and parameters
        self.assertEqual(type(optimizer).__name__, 'AdamW')
        self.assertEqual(optimizer.param_groups[0]['lr'], 2e-4)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 1e-5)
        self.assertEqual(optimizer.param_groups[0]['betas'], (0.9, 0.995))  # β₂=0.995
        
        # Check scheduler
        self.assertIn('scheduler', scheduler_config)
        self.assertEqual(scheduler_config['monitor'], 'val/total_loss')
        self.assertEqual(scheduler_config['interval'], 'epoch')
    
    def test_reconstruction_sample(self):
        """Test getting reconstruction samples for visualization."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        x = self.dummy_batch['volume']
        
        sample_dict = model.get_reconstruction_sample(x)
        
        # Check that all expected keys are present
        expected_keys = {'input', 'reconstruction', 'sample', 'mu', 'logvar'}
        self.assertEqual(set(sample_dict.keys()), expected_keys)
        
        # Check shapes
        self.assertEqual(sample_dict['input'].shape, x.shape)
        self.assertEqual(sample_dict['reconstruction'].shape, x.shape)
        self.assertEqual(sample_dict['sample'].shape, x.shape)
        self.assertEqual(sample_dict['mu'].shape, (self.batch_size, self.latent_dim))
        self.assertEqual(sample_dict['logvar'].shape, (self.batch_size, self.latent_dim))
    
    def test_different_reconstruction_losses(self):
        """Test different reconstruction loss types."""
        # Test MSE loss
        model_mse = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            reconstruction_loss="mse"
        )
        
        x = self.dummy_batch['volume']
        with torch.no_grad():
            reconstruction, mu, logvar = model_mse(x)
            loss_mse, _ = model_mse.vae_loss(reconstruction, x, mu, logvar)
        
        self.assertIsInstance(loss_mse, torch.Tensor)
        self.assertTrue(loss_mse.item() >= 0)
        
        # Test BCE loss
        model_bce = VAELightningModule(
            latent_dim=self.latent_dim,
            groups=self.groups,
            reconstruction_loss="bce"
        )
        
        with torch.no_grad():
            reconstruction, mu, logvar = model_bce(x)
            loss_bce, _ = model_bce.vae_loss(reconstruction, x, mu, logvar)
        
        self.assertIsInstance(loss_bce, torch.Tensor)
        self.assertTrue(loss_bce.item() >= 0)
    
    def test_reparameterization_implementation(self):
        """Test that reparameterization trick is properly implemented."""
        model = VAELightningModule(latent_dim=self.latent_dim, groups=self.groups)
        
        # Get latent parameters
        x = self.dummy_batch['volume']
        with torch.no_grad():
            mu, logvar = model.encode(x)
            
            # Test multiple samples with same mu, logvar
            z1 = model.vae.reparameterize(mu, logvar)
            z2 = model.vae.reparameterize(mu, logvar)
            
            # Should have same shape
            self.assertEqual(z1.shape, (self.batch_size, self.latent_dim))
            self.assertEqual(z2.shape, (self.batch_size, self.latent_dim))
            
            # Should be different (due to random sampling)
            self.assertFalse(torch.equal(z1, z2))
            
            # Should be roughly centered around mu
            z_mean = torch.stack([z1, z2]).mean(dim=0)
            # Allow some tolerance due to limited sampling
            self.assertTrue(torch.allclose(z_mean, mu, atol=2.0))


if __name__ == '__main__':
    unittest.main()