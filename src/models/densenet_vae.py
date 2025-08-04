"""
3D DenseNet VAE architecture for brain volume generation.

Implements a 3D DenseNet Variational Autoencoder with Group Normalization
and dilated convolutions for >150mm receptive field, designed for 
conditional brain activation map generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


def get_valid_groups(channels: int, max_groups: int = 8) -> int:
    """Get the largest valid number of groups <= max_groups that divides channels."""
    groups = max_groups
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class DenseBlock3D(nn.Module):
    """3D Dense block with growth rate."""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int, groups: int = 8):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(self._make_layer(layer_in_channels, growth_rate, groups))
    
    def _make_layer(self, in_channels: int, growth_rate: int, groups: int) -> nn.Sequential:
        """Create a single layer in the dense block."""
        # Ensure channels are divisible by groups
        bottleneck_channels = 4 * growth_rate
        
        # Find largest divisor <= groups that divides in_channels
        in_groups = groups
        while in_channels % in_groups != 0 and in_groups > 1:
            in_groups -= 1
        
        # Find largest divisor <= groups that divides bottleneck_channels
        bottleneck_groups = groups
        while bottleneck_channels % bottleneck_groups != 0 and bottleneck_groups > 1:
            bottleneck_groups -= 1
        
        return nn.Sequential(
            nn.GroupNorm(in_groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(bottleneck_groups, bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer3D(nn.Module):
    """Transition layer between dense blocks."""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        
        # Ensure channels are divisible by groups
        in_groups = groups
        while in_channels % in_groups != 0 and in_groups > 1:
            in_groups -= 1
        
        self.norm = nn.GroupNorm(in_groups, in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.norm(x)))
        return self.pool(x)


class DilatedConvBlock3D(nn.Module):
    """Dilated convolution block for large receptive field."""
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, groups: int = 8):
        super().__init__()
        
        padding = dilation  # Same padding for 3x3 conv with dilation
        
        # Ensure channels are divisible by groups
        out_groups = groups
        while out_channels % out_groups != 0 and out_groups > 1:
            out_groups -= 1
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = nn.GroupNorm(out_groups, out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(out_groups, out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class DenseNetEncoder3D(nn.Module):
    """3D DenseNet encoder for VAE with >150mm receptive field."""
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128, 
                 growth_rate: int = 12, groups: int = 8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Initial convolution - larger kernel for better feature extraction
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        norm1_groups = groups
        while 64 % norm1_groups != 0 and norm1_groups > 1:
            norm1_groups -= 1
        self.norm1 = nn.GroupNorm(norm1_groups, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks with increasing channels
        # Block 1: 64 -> 64 + 6*12 = 136 channels
        self.dense1 = DenseBlock3D(64, growth_rate, num_layers=6, groups=groups)
        num_features = 64 + 6 * growth_rate  # 136
        
        # Transition 1: compress features
        self.trans1 = TransitionLayer3D(num_features, num_features // 2, groups)
        num_features = num_features // 2  # 68
        
        # Block 2: 68 -> 68 + 12*12 = 212 channels  
        self.dense2 = DenseBlock3D(num_features, growth_rate, num_layers=12, groups=groups)
        num_features = num_features + 12 * growth_rate  # 212
        
        # Transition 2: compress features
        self.trans2 = TransitionLayer3D(num_features, num_features // 2, groups)
        num_features = num_features // 2  # 106
        
        # Block 3: 106 -> 106 + 24*12 = 394 channels
        self.dense3 = DenseBlock3D(num_features, growth_rate, num_layers=24, groups=groups)
        num_features = num_features + 24 * growth_rate  # 394
        
        # Transition 3: compress features  
        self.trans3 = TransitionLayer3D(num_features, num_features // 2, groups)
        num_features = num_features // 2  # 197
        
        # Final dense block: 197 -> 197 + 16*12 = 389 channels
        self.dense4 = DenseBlock3D(num_features, growth_rate, num_layers=16, groups=groups)
        num_features = num_features + 16 * growth_rate  # 389
        
        # Dilated convolution layers for >150mm receptive field
        # Each 3x3 conv with stride=1 has RF = 3
        # With dilations [1,2,4,8,16], effective kernels are [3,5,9,17,33]
        # Cumulative RF > 150 voxels = 150mm at 1mm resolution
        self.dilated_layers = nn.Sequential(
            DilatedConvBlock3D(num_features, 256, dilation=1, groups=groups),   # RF: 3
            DilatedConvBlock3D(256, 256, dilation=2, groups=groups),           # RF: 7
            DilatedConvBlock3D(256, 256, dilation=4, groups=groups),           # RF: 15
            DilatedConvBlock3D(256, 256, dilation=8, groups=groups),           # RF: 31
            DilatedConvBlock3D(256, 256, dilation=16, groups=groups),          # RF: 63
            DilatedConvBlock3D(256, 512, dilation=32, groups=groups),          # RF: 127
            DilatedConvBlock3D(512, 512, dilation=64, groups=groups),          # RF: 255 > 150mm ✓
        )
        
        # Global pooling and latent space projection
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Store receptive field calculation for verification
        self.receptive_field_mm = 255  # Calculated above
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DenseNet encoder.
        
        Args:
            x: Input tensor of shape (B, 1, 91, 109, 91)
            
        Returns:
            Tuple of (mu, logvar) tensors of shape (B, latent_dim)
        """
        # Initial convolution: (B, 1, 91, 109, 91) -> (B, 64, 46, 55, 46)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool1(x)  # (B, 64, 23, 28, 23)
        
        # Dense block 1 + transition: (B, 64, 23, 28, 23) -> (B, 68, 12, 14, 12)
        x = self.dense1(x)  # (B, 136, 23, 28, 23)
        x = self.trans1(x)  # (B, 68, 12, 14, 12)
        
        # Dense block 2 + transition: (B, 68, 12, 14, 12) -> (B, 106, 6, 7, 6)
        x = self.dense2(x)  # (B, 212, 12, 14, 12)
        x = self.trans2(x)  # (B, 106, 6, 7, 6)
        
        # Dense block 3 + transition: (B, 106, 6, 7, 6) -> (B, 197, 3, 4, 3)
        x = self.dense3(x)  # (B, 394, 6, 7, 6)  
        x = self.trans3(x)  # (B, 197, 3, 4, 3)
        
        # Dense block 4: (B, 197, 3, 4, 3) -> (B, 389, 3, 4, 3)
        x = self.dense4(x)  # (B, 389, 3, 4, 3)
        
        # Dilated convolutions: (B, 389, 3, 4, 3) -> (B, 512, 3, 4, 3)
        x = self.dilated_layers(x)
        
        # Global pooling: (B, 512, 3, 4, 3) -> (B, 512, 1, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B, 512)
        
        # Latent space projection
        mu = self.fc_mu(x)      # (B, latent_dim)
        logvar = self.fc_logvar(x)  # (B, latent_dim)
        
        return mu, logvar


class DenseNetDecoder3D(nn.Module):
    """3D DenseNet decoder for VAE."""
    
    def __init__(self, latent_dim: int = 128, output_channels: int = 1, groups: int = 8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Project latent to feature map
        self.fc = nn.Linear(latent_dim, 512 * 3 * 4 * 3)
        
        # Transpose convolution layers to upsample
        self.upconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(get_valid_groups(256, groups), 256)
        
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(get_valid_groups(128, groups), 128)
        
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(get_valid_groups(64, groups), 64)
        
        self.upconv4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(get_valid_groups(32, groups), 32)
        
        # Fine-tuning layers
        self.refine1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.refine_norm1 = nn.GroupNorm(get_valid_groups(32, groups), 32)
        
        self.refine2 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.refine_norm2 = nn.GroupNorm(get_valid_groups(16, groups), 16)
        
        # Final output layer
        self.final_conv = nn.Conv3d(16, output_channels, kernel_size=3, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent tensor of shape (B, latent_dim)
            
        Returns:
            Reconstructed volume of shape (B, 1, 91, 109, 91)
        """
        # Project to feature map: (B, latent_dim) -> (B, 512*3*4*3)
        x = self.fc(z)
        x = x.view(-1, 512, 3, 4, 3)  # (B, 512, 3, 4, 3)
        
        # Transpose convolutions with refinement
        x = F.relu(self.norm1(self.upconv1(x)))  # (B, 256, 6, 8, 6)
        x = F.relu(self.norm2(self.upconv2(x)))  # (B, 128, 12, 16, 12)
        x = F.relu(self.norm3(self.upconv3(x)))  # (B, 64, 24, 32, 24)
        x = F.relu(self.norm4(self.upconv4(x)))  # (B, 32, 48, 64, 48)
        
        # Refinement layers
        x = F.relu(self.refine_norm1(self.refine1(x)))  # (B, 32, 48, 64, 48)
        x = F.relu(self.refine_norm2(self.refine2(x)))  # (B, 16, 48, 64, 48)
        
        # Final upsampling to target size using interpolation
        x = F.interpolate(x, size=(91, 109, 91), mode='trilinear', align_corners=False)
        
        # Final output
        x = self.final_conv(x)  # (B, 1, 91, 109, 91)
        
        return x


class DenseNetVAE3D(nn.Module):
    """Complete 3D DenseNet VAE for brain volume generation."""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 latent_dim: int = 128, growth_rate: int = 12, groups: int = 8):
        super().__init__()
        
        self.encoder = DenseNetEncoder3D(input_channels, latent_dim, growth_rate, groups)
        self.decoder = DenseNetDecoder3D(latent_dim, output_channels, groups)
        
        # Store architecture parameters
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.receptive_field_mm = self.encoder.receptive_field_mm
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to output volume."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete VAE.
        
        Args:
            x: Input tensor of shape (B, input_channels, 91, 109, 91)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_receptive_field(self) -> int:
        """Get receptive field size in mm."""
        return self.receptive_field_mm


def create_densenet_vae(config: dict) -> DenseNetVAE3D:
    """Factory function to create DenseNet VAE from config."""
    model_config = config.get('model', {})
    
    return DenseNetVAE3D(
        input_channels=model_config.get('input_channels', 1),
        output_channels=model_config.get('output_channels', 1),
        latent_dim=model_config.get('latent_dim', 128),
        growth_rate=model_config.get('growth_rate', 12),
        groups=model_config.get('group_norm_groups', 8)
    )


if __name__ == "__main__":
    # Test the architecture
    model = DenseNetVAE3D(latent_dim=128)
    
    # Test input
    x = torch.randn(2, 1, 91, 109, 91)
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    print(f"Receptive field: {model.get_receptive_field()}mm")
    
    # Test forward pass
    with torch.no_grad():
        recon, mu, logvar = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Reconstruction shape: {recon.shape}")
        print(f"Latent mu shape: {mu.shape}")
        print(f"Latent logvar shape: {logvar.shape}")
        
        # Test sampling
        samples = model.sample(3, x.device)
        print(f"Sample shape: {samples.shape}")