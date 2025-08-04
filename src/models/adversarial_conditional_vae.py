"""
Adversarial Conditional DenseNet VAE with GRL de-biasing.

This module extends the conditional DenseNet VAE with adversarial de-biasing
to remove systematic bias (publication year trends) from learned representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .conditional_densenet_vae import (
    ConditionalDenseNetVAE3D, ConditionalDenseNetEncoder3D,
    ConditionalDenseNetDecoder3D
)
from .adversarial_debiasing import (
    AdversarialDebiasing, AdversarialLambdaScheduler, create_mock_year_data
)


class AdversarialConditionalEncoder3D(ConditionalDenseNetEncoder3D):
    """Conditional DenseNet encoder with adversarial de-biasing."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128, 
                 growth_rate: int = 12, groups: int = 8,
                 metadata_config: Optional[Dict[str, Any]] = None,
                 adversarial_lambda: float = 1.0):
        super().__init__(input_channels, latent_dim, growth_rate, groups, metadata_config)
        
        # Adversarial de-biasing module
        # Use features after global pooling (512-dim) for adversarial prediction
        self.adversarial_debiaser = AdversarialDebiasing(
            feature_dim=512, 
            bias_type='year',  # Predict publication year
            lambda_val=adversarial_lambda
        )
        
    def forward(self, x: torch.Tensor, 
                observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                missing_mask: Optional[Dict[str, torch.Tensor]] = None,
                year_targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass with metadata imputation and adversarial de-biasing.
        
        Args:
            x: Input tensor of shape (B, 1, 91, 109, 91)
            observed_metadata: Dict of observed metadata tensors
            missing_mask: Dict of boolean masks for missing values
            year_targets: Publication year targets for adversarial training
            
        Returns:
            Tuple of (mu, logvar, imputed_metadata_dict, adversarial_predictions)
        """
        # Run through DenseNet layers up to dilated layers (same as parent)
        x = F.relu(self.densenet_encoder.norm1(self.densenet_encoder.conv1(x)))
        x = self.densenet_encoder.pool1(x)
        
        x = self.densenet_encoder.dense1(x)
        x = self.densenet_encoder.trans1(x)
        
        x = self.densenet_encoder.dense2(x)
        x = self.densenet_encoder.trans2(x)
        
        x = self.densenet_encoder.dense3(x)
        x = self.densenet_encoder.trans3(x)
        
        x = self.densenet_encoder.dense4(x)
        
        # Extract features after dilated layers (before global pooling)
        image_features = self.densenet_encoder.dilated_layers(x)  # (B, 512, 3, 4, 3)
        
        # Global pooling for image features
        pooled_features = self.densenet_encoder.global_pool(image_features)  # (B, 512, 1, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, 512)
        
        # Adversarial prediction using pooled features (with GRL)
        adversarial_predictions = self.adversarial_debiaser(pooled_features)
        
        # Metadata imputation using image features
        imputed_metadata = self.metadata_imputer(pooled_features, observed_metadata, missing_mask)
        
        # Get metadata vector
        metadata_vector = self.metadata_imputer.get_metadata_vector(imputed_metadata)  # (B, metadata_dim)
        
        # Combine image and metadata features
        combined_features = torch.cat([pooled_features, metadata_vector], dim=1)
        
        # Project to latent space
        mu = self.combined_fc_mu(combined_features)
        logvar = self.combined_fc_logvar(combined_features)
        
        return mu, logvar, imputed_metadata, adversarial_predictions


class AdversarialConditionalVAE3D(nn.Module):
    """Complete adversarial conditional DenseNet VAE with GRL de-biasing."""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 latent_dim: int = 128, growth_rate: int = 12, groups: int = 8,
                 metadata_config: Optional[Dict[str, Any]] = None,
                 adversarial_lambda: float = 1.0):
        super().__init__()
        
        # Use conditional VAE as base but replace encoder with adversarial version
        base_vae = ConditionalDenseNetVAE3D(
            input_channels, output_channels, latent_dim, growth_rate, groups, metadata_config
        )
        
        # Replace encoder with adversarial version
        self.encoder = AdversarialConditionalEncoder3D(
            input_channels, latent_dim, growth_rate, groups, metadata_config, adversarial_lambda
        )
        
        # Use the same decoder
        self.decoder = base_vae.decoder
        
        # Lambda scheduler for adversarial training
        self.lambda_scheduler = AdversarialLambdaScheduler(
            schedule_type='linear_ramp',
            start_epoch=20,
            end_epoch=80,
            lambda_max=1.0
        )
        
        # Store architecture parameters
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.metadata_config = self.encoder.metadata_config
        self.total_metadata_dim = self.encoder.metadata_imputer.total_metadata_dim
        self.receptive_field_mm = self.encoder.densenet_encoder.receptive_field_mm
        
    def encode(self, x: torch.Tensor, 
               observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
               missing_mask: Optional[Dict[str, torch.Tensor]] = None,
               year_targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encode input with metadata imputation and adversarial de-biasing."""
        return self.encoder(x, observed_metadata, missing_mask, year_targets)
    
    def decode(self, z: torch.Tensor, metadata_vector: torch.Tensor) -> torch.Tensor:
        """Decode latent code with metadata conditioning."""
        return self.decoder(z, metadata_vector)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor,
                observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                missing_mask: Optional[Dict[str, torch.Tensor]] = None,
                year_targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through adversarial conditional VAE.
        
        Args:
            x: Input tensor of shape (B, input_channels, 91, 109, 91)
            observed_metadata: Dict of observed metadata tensors
            missing_mask: Dict of boolean masks for missing values
            year_targets: Publication year targets for adversarial training
            
        Returns:
            Tuple of (reconstruction, mu, logvar, imputed_metadata, adversarial_predictions)
        """
        # Encode with metadata imputation and adversarial de-biasing
        mu, logvar, imputed_metadata, adversarial_predictions = self.encode(
            x, observed_metadata, missing_mask, year_targets
        )
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Get metadata vector for conditioning
        metadata_vector = self.encoder.metadata_imputer.get_metadata_vector(imputed_metadata)
        
        # Decode with FiLM conditioning
        reconstruction = self.decode(z, metadata_vector)
        
        return reconstruction, mu, logvar, imputed_metadata, adversarial_predictions
    
    def compute_total_loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
                          mu: torch.Tensor, logvar: torch.Tensor,
                          imputed_metadata: Dict[str, torch.Tensor],
                          adversarial_predictions: torch.Tensor,
                          observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                          missing_mask: Optional[Dict[str, torch.Tensor]] = None,
                          year_targets: Optional[torch.Tensor] = None,
                          beta: float = 1.0, imputation_weight: float = 1.0,
                          adversarial_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute total VAE loss including adversarial de-biasing loss.
        
        Returns:
            Dict containing individual loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Imputation loss
        imputation_loss = torch.tensor(0.0, device=x.device)
        if observed_metadata is not None and missing_mask is not None:
            imputation_loss = self.encoder.metadata_imputer.compute_imputation_loss(
                imputed_metadata, observed_metadata, missing_mask
            )
        
        # Adversarial loss (encourages features that fool the year predictor)
        adversarial_loss = torch.tensor(0.0, device=x.device)
        if year_targets is not None:
            # MSE loss for year prediction (the GRL will reverse gradients)
            adversarial_loss = F.mse_loss(adversarial_predictions.squeeze(), year_targets.float())
        
        # Total loss
        total_loss = (recon_loss + 
                     beta * kl_loss + 
                     imputation_weight * imputation_loss + 
                     adversarial_weight * adversarial_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'imputation_loss': imputation_loss,
            'adversarial_loss': adversarial_loss
        }
    
    def update_adversarial_lambda(self, epoch: int):
        """Update adversarial lambda based on training epoch."""
        new_lambda = self.lambda_scheduler.get_lambda(epoch)
        self.encoder.adversarial_debiaser.set_lambda(new_lambda)
        return new_lambda
    
    def get_adversarial_lambda(self) -> float:
        """Get current adversarial lambda value."""
        return self.encoder.adversarial_debiaser.get_lambda()
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def sample_conditional(self, num_samples: int, metadata_vector: torch.Tensor, 
                          device: torch.device) -> torch.Tensor:
        """Sample from conditional prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, metadata_vector)


def create_adversarial_conditional_vae(config: dict) -> AdversarialConditionalVAE3D:
    """Factory function to create adversarial conditional DenseNet VAE from config."""
    model_config = config.get('model', {})
    
    return AdversarialConditionalVAE3D(
        input_channels=model_config.get('input_channels', 1),
        output_channels=model_config.get('output_channels', 1),
        latent_dim=model_config.get('latent_dim', 128),
        growth_rate=model_config.get('growth_rate', 12),
        groups=model_config.get('group_norm_groups', 8),
        metadata_config=model_config.get('metadata_config', None),
        adversarial_lambda=model_config.get('adversarial_lambda', 1.0)
    )


if __name__ == "__main__":
    # Test the adversarial conditional architecture
    print("Testing Adversarial Conditional DenseNet VAE...")
    
    model = AdversarialConditionalVAE3D(latent_dim=64)  # Smaller for testing
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 1, 91, 109, 91)
    
    # Create mock metadata and year targets
    from src.models.metadata_imputation import create_mock_metadata_batch
    observed_metadata, missing_mask = create_mock_metadata_batch(
        batch_size, model.metadata_config, missing_rate=0.3
    )
    year_targets = create_mock_year_data(batch_size, 'year')
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    print(f"Receptive field: {model.receptive_field_mm}mm")
    print(f"Total metadata dim: {model.total_metadata_dim}")
    print(f"Initial adversarial λ: {model.get_adversarial_lambda():.3f}")
    
    # Test forward pass
    with torch.no_grad():
        recon, mu, logvar, imputed_metadata, adv_pred = model(
            x, observed_metadata, missing_mask, year_targets
        )
        
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {recon.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        print(f"Adversarial predictions shape: {adv_pred.shape}")
        
        # Test loss computation
        losses = model.compute_total_loss(
            x, recon, mu, logvar, imputed_metadata, adv_pred,
            observed_metadata, missing_mask, year_targets
        )
        
        print(f"\nLoss components:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value.item():.4f}")
        
        # Test lambda scheduling
        print(f"\nTesting lambda scheduling:")
        for epoch in [0, 25, 50, 75, 100]:
            new_lambda = model.update_adversarial_lambda(epoch)
            print(f"  Epoch {epoch:3d}: λ = {new_lambda:.3f}")
        
        # Test conditional sampling
        metadata_vector = model.encoder.metadata_imputer.get_metadata_vector(imputed_metadata)
        samples = model.sample_conditional(3, metadata_vector[:1], x.device)
        print(f"\nConditional sample shape: {samples.shape}")
        
    print("\n✅ Adversarial Conditional DenseNet VAE test passed!")