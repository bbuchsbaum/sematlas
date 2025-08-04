"""
Conditional DenseNet VAE with metadata imputation and FiLM conditioning.

This module extends the DenseNet VAE with:
1. Metadata imputation via amortization head
2. FiLM (Feature-wise Linear Modulation) conditioning
3. Integration of imputed metadata into the generative process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .densenet_vae import DenseNetEncoder3D, DenseNetDecoder3D, get_valid_groups
from .metadata_imputation import MetadataImputation, create_default_metadata_config


class ConditionalDenseNetEncoder3D(nn.Module):
    """DenseNet encoder with metadata imputation capability."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128, 
                 growth_rate: int = 12, groups: int = 8,
                 metadata_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Base DenseNet encoder
        self.densenet_encoder = DenseNetEncoder3D(input_channels, latent_dim, growth_rate, groups)
        
        # Metadata imputation head
        if metadata_config is None:
            metadata_config = create_default_metadata_config()
        
        self.metadata_config = metadata_config
        
        # Use features before global pooling for metadata imputation
        # The dilated layers output 512 channels before global pooling
        self.metadata_imputer = MetadataImputation(512, metadata_config)
        
        # Combined latent projection (image + metadata features)
        combined_dim = 512 + self.metadata_imputer.total_metadata_dim
        self.combined_fc_mu = nn.Linear(combined_dim, latent_dim)
        self.combined_fc_logvar = nn.Linear(combined_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, 
                observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                missing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with metadata imputation.
        
        Args:
            x: Input tensor of shape (B, 1, 91, 109, 91)
            observed_metadata: Dict of observed metadata tensors
            missing_mask: Dict of boolean masks for missing values
            
        Returns:
            Tuple of (mu, logvar, imputed_metadata_dict)
        """
        # Run through DenseNet layers up to dilated layers
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
        
        # Metadata imputation using image features
        imputed_metadata = self.metadata_imputer(pooled_features, observed_metadata, missing_mask)
        
        # Get metadata vector
        metadata_vector = self.metadata_imputer.get_metadata_vector(imputed_metadata)  # (B, metadata_dim)
        
        # Combine image and metadata features
        combined_features = torch.cat([pooled_features, metadata_vector], dim=1)
        
        # Project to latent space
        mu = self.combined_fc_mu(combined_features)
        logvar = self.combined_fc_logvar(combined_features)
        
        return mu, logvar, imputed_metadata


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation (FiLM) generator."""
    
    def __init__(self, metadata_dim: int, feature_channels: int, hidden_dim: int = 128):
        super().__init__()
        
        self.metadata_dim = metadata_dim
        self.feature_channels = feature_channels
        
        # MLP to generate γ (scale) and β (bias) parameters
        self.film_mlp = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_channels * 2)  # γ and β
        )
        
    def forward(self, metadata_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate FiLM parameters.
        
        Args:
            metadata_vector: Metadata of shape (B, metadata_dim)
            
        Returns:
            Tuple of (gamma, beta) each of shape (B, feature_channels)
        """
        film_params = self.film_mlp(metadata_vector)  # (B, feature_channels * 2)
        
        gamma = film_params[:, :self.feature_channels]  # (B, feature_channels)
        beta = film_params[:, self.feature_channels:]   # (B, feature_channels)
        
        # Add 1 to gamma for stability (γ = 1 + Δγ)
        gamma = 1.0 + gamma
        
        return gamma, beta


class FiLMLayer(nn.Module):
    """FiLM conditioning layer."""
    
    def __init__(self, feature_channels: int):
        super().__init__()
        self.feature_channels = feature_channels
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: Feature tensor of shape (B, C, D, H, W)
            gamma: Scale parameters of shape (B, C)
            beta: Bias parameters of shape (B, C)
            
        Returns:
            Conditioned features of same shape as x
        """
        # Reshape gamma and beta for broadcasting
        if x.dim() == 5:  # 3D case: (B, C, D, H, W)
            gamma = gamma.view(-1, self.feature_channels, 1, 1, 1)
            beta = beta.view(-1, self.feature_channels, 1, 1, 1)
        elif x.dim() == 4:  # 2D case: (B, C, H, W)
            gamma = gamma.view(-1, self.feature_channels, 1, 1)
            beta = beta.view(-1, self.feature_channels, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")
        
        return gamma * x + beta


class ConditionalDenseNetDecoder3D(nn.Module):
    """DenseNet decoder with FiLM conditioning."""
    
    def __init__(self, latent_dim: int = 128, output_channels: int = 1, 
                 groups: int = 8, metadata_dim: int = 16):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.metadata_dim = metadata_dim
        
        # Base decoder structure
        self.fc = nn.Linear(latent_dim, 512 * 3 * 4 * 3)
        
        # FiLM generators for each layer
        self.film_gen1 = FiLMGenerator(metadata_dim, 512)
        self.film_gen2 = FiLMGenerator(metadata_dim, 256)
        self.film_gen3 = FiLMGenerator(metadata_dim, 128)
        self.film_gen4 = FiLMGenerator(metadata_dim, 64)
        self.film_gen5 = FiLMGenerator(metadata_dim, 32)
        
        # FiLM layers
        self.film1 = FiLMLayer(512)
        self.film2 = FiLMLayer(256)
        self.film3 = FiLMLayer(128)
        self.film4 = FiLMLayer(64)
        self.film5 = FiLMLayer(32)
        
        # Decoder layers
        self.upconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(get_valid_groups(256, groups), 256)
        
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(get_valid_groups(128, groups), 128)
        
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(get_valid_groups(64, groups), 64)
        
        self.upconv4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(get_valid_groups(32, groups), 32)
        
        # Refinement layers
        self.refine1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.refine_norm1 = nn.GroupNorm(get_valid_groups(32, groups), 32)
        
        self.refine2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.refine_norm2 = nn.GroupNorm(get_valid_groups(16, groups), 16)
        
        # Final output layer
        self.final_conv = nn.Conv3d(16, output_channels, kernel_size=3, padding=1)
        
    def forward(self, z: torch.Tensor, metadata_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            z: Latent tensor of shape (B, latent_dim)
            metadata_vector: Metadata vector of shape (B, metadata_dim)
            
        Returns:
            Reconstructed volume of shape (B, output_channels, 91, 109, 91)
        """
        # Project latent to feature map
        x = self.fc(z)
        x = x.view(-1, 512, 3, 4, 3)  # (B, 512, 3, 4, 3)
        
        # Generate FiLM parameters
        gamma1, beta1 = self.film_gen1(metadata_vector)
        gamma2, beta2 = self.film_gen2(metadata_vector)
        gamma3, beta3 = self.film_gen3(metadata_vector)
        gamma4, beta4 = self.film_gen4(metadata_vector)
        gamma5, beta5 = self.film_gen5(metadata_vector)
        
        # Apply FiLM conditioning at each layer
        x = self.film1(x, gamma1, beta1)
        x = F.relu(self.norm1(self.upconv1(x)))  # (B, 256, 6, 8, 6)
        
        x = self.film2(x, gamma2, beta2)
        x = F.relu(self.norm2(self.upconv2(x)))  # (B, 128, 12, 16, 12)
        
        x = self.film3(x, gamma3, beta3)
        x = F.relu(self.norm3(self.upconv3(x)))  # (B, 64, 24, 32, 24)
        
        x = self.film4(x, gamma4, beta4)
        x = F.relu(self.norm4(self.upconv4(x)))  # (B, 32, 48, 64, 48)
        
        # Refinement with FiLM
        x = self.film5(x, gamma5, beta5)
        x = F.relu(self.refine_norm1(self.refine1(x)))  # (B, 32, 48, 64, 48)
        x = F.relu(self.refine_norm2(self.refine2(x)))  # (B, 16, 48, 64, 48)
        
        # Final upsampling to target size
        x = F.interpolate(x, size=(91, 109, 91), mode='trilinear', align_corners=False)
        
        # Final output
        x = self.final_conv(x)  # (B, output_channels, 91, 109, 91)
        
        return x


class ConditionalDenseNetVAE3D(nn.Module):
    """Complete conditional DenseNet VAE with metadata imputation and FiLM conditioning."""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 latent_dim: int = 128, growth_rate: int = 12, groups: int = 8,
                 metadata_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if metadata_config is None:
            metadata_config = create_default_metadata_config()
        
        self.metadata_config = metadata_config
        
        # Calculate total metadata dimension
        total_metadata_dim = sum(config['dim'] for config in metadata_config.values())
        
        self.encoder = ConditionalDenseNetEncoder3D(
            input_channels, latent_dim, growth_rate, groups, metadata_config
        )
        self.decoder = ConditionalDenseNetDecoder3D(
            latent_dim, output_channels, groups, total_metadata_dim
        )
        
        # Store architecture parameters
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.total_metadata_dim = total_metadata_dim
        self.receptive_field_mm = self.encoder.densenet_encoder.receptive_field_mm
        
    def encode(self, x: torch.Tensor, 
               observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
               missing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode input with metadata imputation."""
        return self.encoder(x, observed_metadata, missing_mask)
    
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
                missing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through conditional VAE.
        
        Args:
            x: Input tensor of shape (B, input_channels, 91, 109, 91)
            observed_metadata: Dict of observed metadata tensors
            missing_mask: Dict of boolean masks for missing values
            
        Returns:
            Tuple of (reconstruction, mu, logvar, imputed_metadata)
        """
        # Encode with metadata imputation
        mu, logvar, imputed_metadata = self.encode(x, observed_metadata, missing_mask)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Get metadata vector for conditioning
        metadata_vector = self.encoder.metadata_imputer.get_metadata_vector(imputed_metadata)
        
        # Decode with FiLM conditioning
        reconstruction = self.decode(z, metadata_vector)
        
        return reconstruction, mu, logvar, imputed_metadata
    
    def sample_conditional(self, num_samples: int, metadata_vector: torch.Tensor, 
                          device: torch.device) -> torch.Tensor:
        """Sample from conditional prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, metadata_vector)
    
    def compute_total_loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
                          mu: torch.Tensor, logvar: torch.Tensor,
                          imputed_metadata: Dict[str, torch.Tensor],
                          observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                          missing_mask: Optional[Dict[str, torch.Tensor]] = None,
                          beta: float = 1.0, imputation_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute total VAE loss including imputation loss.
        
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
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + imputation_weight * imputation_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'imputation_loss': imputation_loss
        }
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_conditional_densenet_vae(config: dict) -> ConditionalDenseNetVAE3D:
    """Factory function to create conditional DenseNet VAE from config."""
    model_config = config.get('model', {})
    
    return ConditionalDenseNetVAE3D(
        input_channels=model_config.get('input_channels', 1),
        output_channels=model_config.get('output_channels', 1),
        latent_dim=model_config.get('latent_dim', 128),
        growth_rate=model_config.get('growth_rate', 12),
        groups=model_config.get('group_norm_groups', 8),
        metadata_config=model_config.get('metadata_config', None)
    )


if __name__ == "__main__":
    # Test the conditional architecture
    print("Testing Conditional DenseNet VAE...")
    
    model = ConditionalDenseNetVAE3D(latent_dim=128)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 1, 91, 109, 91)
    
    # Create mock metadata
    from .metadata_imputation import create_mock_metadata_batch
    observed_metadata, missing_mask = create_mock_metadata_batch(
        batch_size, model.metadata_config, missing_rate=0.3
    )
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    print(f"Receptive field: {model.receptive_field_mm}mm")
    print(f"Total metadata dim: {model.total_metadata_dim}")
    
    # Test forward pass
    with torch.no_grad():
        recon, mu, logvar, imputed_metadata = model(x, observed_metadata, missing_mask)
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {recon.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        
        # Test loss computation
        losses = model.compute_total_loss(x, recon, mu, logvar, imputed_metadata, 
                                        observed_metadata, missing_mask)
        print(f"Total loss: {losses['total_loss'].item():.4f}")
        print(f"Reconstruction loss: {losses['recon_loss'].item():.4f}")
        print(f"KL loss: {losses['kl_loss'].item():.4f}")
        print(f"Imputation loss: {losses['imputation_loss'].item():.4f}")
        
        # Test conditional sampling
        metadata_vector = model.encoder.metadata_imputer.get_metadata_vector(imputed_metadata)
        samples = model.sample_conditional(3, metadata_vector[:1], x.device)
        print(f"Conditional sample shape: {samples.shape}")
        
    print("\n✅ Conditional DenseNet VAE test passed!")