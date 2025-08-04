"""
3D ResNet VAE architecture for brain volume generation.

Implements a 3D ResNet-10 Variational Autoencoder with Group Normalization
for generating brain activation maps from latent codes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock3D(nn.Module):
    """3D Residual block with Group Normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class ResNetEncoder3D(nn.Module):
    """3D ResNet encoder for VAE."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 32, groups: int = 8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(groups, 32)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (ResNet-10: 2 blocks per layer)
        self.layer1 = self._make_layer(32, 32, 2, stride=1, groups=groups)
        self.layer2 = self._make_layer(32, 64, 2, stride=2, groups=groups)
        self.layer3 = self._make_layer(64, 128, 2, stride=2, groups=groups)
        
        # Final layers with dilated convolutions for large receptive field
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.GroupNorm(groups, 256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            nn.GroupNorm(groups, 256),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling and latent heads
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, groups: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride, groups))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, 1, 91, 109, 91)
            
        Returns:
            Tuple of (mu, logvar) tensors of shape (B, latent_dim)
        """
        # Initial convolution and pooling
        x = F.relu(self.gn1(self.conv1(x)))  # (B, 32, 46, 55, 46)
        x = self.maxpool(x)  # (B, 32, 23, 28, 23)
        
        # ResNet layers
        x = self.layer1(x)  # (B, 32, 23, 28, 23)
        x = self.layer2(x)  # (B, 64, 12, 14, 12)
        x = self.layer3(x)  # (B, 128, 6, 7, 6)
        x = self.layer4(x)  # (B, 256, 3, 4, 3) with dilated convs
        
        # Global pooling and latent vectors
        x = self.global_pool(x)  # (B, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        mu = self.fc_mu(x)  # (B, latent_dim)
        logvar = self.fc_logvar(x)  # (B, latent_dim)
        
        return mu, logvar


class ResNetDecoder3D(nn.Module):
    """3D ResNet decoder for VAE."""
    
    def __init__(self, latent_dim: int = 32, output_channels: int = 1, groups: int = 8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * 3 * 4 * 3)
        self.initial_size = (3, 4, 3)
        
        # Decoder layers - reverse of encoder
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, 128),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = self._make_decoder_layer(128, 64, groups)
        self.layer3 = self._make_decoder_layer(64, 32, groups)
        self.layer4 = self._make_decoder_layer(32, 32, groups)
        
        # Final upsampling layers to reach target size (91, 109, 91)
        self.upsample1 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.gn_up1 = nn.GroupNorm(groups, 16)
        
        # Final layer with adaptive output size
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output activation for brain maps
        )
    
    def _make_decoder_layer(self, in_channels: int, out_channels: int, groups: int) -> nn.Sequential:
        """Create a decoder layer with upsampling and residual-like structure."""
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (B, latent_dim)
            
        Returns:
            Reconstructed volume of shape (B, 1, 91, 109, 91)
        """
        # Project to initial feature map
        x = self.fc(z)  # (B, 256 * 3 * 4 * 3)
        x = x.view(x.size(0), 256, *self.initial_size)  # (B, 256, 3, 4, 3)
        
        # Decoder layers
        x = self.layer1(x)  # (B, 128, 6, 8, 6)
        x = self.layer2(x)  # (B, 64, 12, 16, 12)
        x = self.layer3(x)  # (B, 32, 24, 32, 24)
        x = self.layer4(x)  # (B, 32, 48, 64, 48)
        
        # Final upsampling
        x = F.relu(self.gn_up1(self.upsample1(x)))  # (B, 16, 96, 128, 96)
        
        # Crop/interpolate to exact target size (91, 109, 91)
        x = F.interpolate(x, size=(91, 109, 91), mode='trilinear', align_corners=False)
        
        # Final convolution
        x = self.final_conv(x)  # (B, 1, 91, 109, 91)
        
        return x


class ResNetVAE3D(nn.Module):
    """3D ResNet Variational Autoencoder for brain volumes."""
    
    def __init__(
        self, 
        input_channels: int = 1,
        latent_dim: int = 32,
        groups: int = 8
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoder3D(input_channels, latent_dim, groups)
        self.decoder = ResNetDecoder3D(latent_dim, input_channels, groups)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (B, 1, 91, 109, 91)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample from the latent space."""
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def create_resnet_vae(latent_dim: int = 32, groups: int = 8) -> ResNetVAE3D:
    """
    Create a 3D ResNet VAE model.
    
    Args:
        latent_dim: Dimensionality of latent space
        groups: Number of groups for Group Normalization
        
    Returns:
        ResNetVAE3D model
    """
    return ResNetVAE3D(
        input_channels=1,
        latent_dim=latent_dim,
        groups=groups
    )


if __name__ == "__main__":
    # Test model instantiation and forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet_vae(latent_dim=32, groups=8).to(device)
    
    # Test with dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 91, 109, 91, device=device)
    
    print(f"Model created successfully on {device}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstruction, mu, logvar = model(dummy_input)
        
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test sampling
    with torch.no_grad():
        samples = model.sample(3, device)
    print(f"Sample shape: {samples.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")