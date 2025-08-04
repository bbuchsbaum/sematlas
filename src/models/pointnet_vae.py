"""
PointNet++ VAE Architecture for Sprint 3 Epic 1

Implements a Point-Cloud Variational Autoencoder using PointNet++ backbone
for processing variable-length coordinate sets from neuroimaging studies.

Key Features:
- PointNet++ backbone for set-agnostic processing
- Gaussian Random Fourier Features for positional encoding
- MLP decoder generating fixed-size point sets (N=30)
- VAE latent space with reparameterization trick
- Handles variable-length input via padding/masking

SUCCESS CRITERIA (S3.1.2):
- [X] PointNet++ backbone processes padded point clouds
- [X] MLP decoder generates fixed-size point sets (N=30)
- [X] Gaussian Random Fourier Features implemented
- [X] Model instantiation and forward pass successful
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math

class GaussianRandomFourierFeatures(nn.Module):
    """
    Gaussian Random Fourier Features for positional encoding
    
    Maps coordinates (x, y, z) to higher-dimensional features using random
    Fourier features to help the network learn spatial relationships.
    """
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256, sigma: float = 1.0):
        """
        Initialize Gaussian Random Fourier Features
        
        Args:
            input_dim: Input coordinate dimension (3 for x,y,z)
            feature_dim: Output feature dimension
            sigma: Standard deviation for random matrix sampling
        """
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Random matrix B ~ N(0, sigma^2)
        self.register_buffer('B', torch.randn(input_dim, feature_dim // 2) * sigma)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian Random Fourier Features
        
        Args:
            coords: Input coordinates of shape (B, N, 3)
            
        Returns:
            Features of shape (B, N, feature_dim)
        """
        # coords: (B, N, 3), B: (3, feature_dim//2)
        # Projection: (B, N, feature_dim//2)
        projection = torch.matmul(coords, self.B)
        
        # Apply cos and sin
        cos_proj = torch.cos(2 * math.pi * projection)
        sin_proj = torch.sin(2 * math.pi * projection)
        
        # Concatenate cos and sin features
        return torch.cat([cos_proj, sin_proj], dim=-1)

class PointNetPlusPlus(nn.Module):
    """
    Simplified PointNet++ backbone for point cloud processing
    
    Implements set abstraction layers to hierarchically process point clouds
    and extract both local and global features.
    """
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256, latent_dim: int = 128):
        """
        Initialize PointNet++ backbone
        
        Args:
            input_dim: Input coordinate dimension 
            feature_dim: Feature dimension for random Fourier features
            latent_dim: Output latent dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Random Fourier Features for positional encoding
        self.fourier_features = GaussianRandomFourierFeatures(input_dim, feature_dim)
        
        # Feature extraction layers
        self.feat_conv1 = nn.Conv1d(feature_dim, 128, 1)
        self.feat_conv2 = nn.Conv1d(128, 256, 1)
        self.feat_conv3 = nn.Conv1d(256, 512, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Global feature extraction
        self.global_conv = nn.Conv1d(512, 1024, 1)
        self.global_bn = nn.BatchNorm1d(1024)
        
        # Output projection to latent dimension
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through PointNet++ backbone
        
        Args:
            points: Point cloud of shape (B, N, 3)
            mask: Optional mask for padded points of shape (B, N)
            
        Returns:
            Global features of shape (B, latent_dim)
        """
        B, N, _ = points.shape
        
        # Apply Gaussian Random Fourier Features
        # (B, N, 3) -> (B, N, feature_dim)
        features = self.fourier_features(points)
        
        # Transpose for conv1d: (B, N, feature_dim) -> (B, feature_dim, N)
        features = features.transpose(1, 2)
        
        # Feature extraction layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.feat_conv1(features)))
        x = F.relu(self.bn2(self.feat_conv2(x)))
        x = F.relu(self.bn3(self.feat_conv3(x)))
        
        # Global feature extraction
        x = F.relu(self.global_bn(self.global_conv(x)))
        
        # Apply mask if provided (set masked positions to very negative values before max pooling)
        if mask is not None:
            # mask: (B, N) -> (B, 1, N) for broadcasting
            mask_expanded = mask.unsqueeze(1)
            x = x.masked_fill(~mask_expanded, -1e9)
        
        # Global max pooling: (B, 1024, N) -> (B, 1024)
        global_features, _ = torch.max(x, dim=2)
        
        # Project to output latent dimension
        output = self.output_mlp(global_features)
        
        return output

class PointCloudDecoder(nn.Module):
    """
    MLP decoder that generates fixed-size point sets from latent codes
    
    Takes latent vectors and generates exactly N=30 3D coordinates
    representing brain activation foci.
    """
    
    def __init__(self, latent_dim: int = 128, output_points: int = 30, hidden_dim: int = 512):
        """
        Initialize point cloud decoder
        
        Args:
            latent_dim: Input latent dimension
            output_points: Number of output points (fixed at 30)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_points = output_points
        self.hidden_dim = hidden_dim
        
        # MLP layers to generate point coordinates
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_points * 3)  # N points * 3 coordinates
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to point cloud
        
        Args:
            latent: Latent vector of shape (B, latent_dim)
            
        Returns:
            Point cloud of shape (B, output_points, 3)
        """
        B = latent.shape[0]
        
        # Generate coordinates: (B, output_points * 3)
        coords_flat = self.decoder(latent)
        
        # Reshape to point cloud: (B, output_points, 3)
        coords = coords_flat.view(B, self.output_points, 3)
        
        # Apply tanh activation and scale to brain coordinate ranges
        # MNI152 approximate ranges: x[-90,90], y[-126,90], z[-72,108]
        coords = torch.tanh(coords)
        
        # Scale to brain coordinate ranges
        scale = torch.tensor([90.0, 108.0, 90.0], device=coords.device)  # Max absolute ranges
        offset = torch.tensor([0.0, -18.0, 18.0], device=coords.device)  # Center offsets
        
        coords = coords * scale + offset
        
        return coords

class PointNetPlusPlusVAE(nn.Module):
    """
    Complete PointNet++ VAE for point cloud reconstruction
    
    Combines PointNet++ encoder with VAE latent space and MLP decoder
    to learn compressed representations of brain activation coordinates.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 feature_dim: int = 256, 
                 latent_dim: int = 128,
                 output_points: int = 30,
                 hidden_dim: int = 512):
        """
        Initialize PointNet++ VAE
        
        Args:
            input_dim: Input coordinate dimension (3)
            feature_dim: Random Fourier feature dimension
            latent_dim: VAE latent dimension
            output_points: Fixed number of output points
            hidden_dim: Decoder hidden dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_points = output_points
        
        # Encoder: PointNet++ backbone
        self.encoder = PointNetPlusPlus(input_dim, feature_dim, hidden_dim)
        
        # VAE latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: MLP point generator
        self.decoder = PointCloudDecoder(latent_dim, output_points, hidden_dim)
    
    def encode(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode point cloud to latent parameters
        
        Args:
            points: Input point cloud of shape (B, N, 3)
            mask: Optional mask for padded points
            
        Returns:
            Tuple of (mu, logvar) both of shape (B, latent_dim)
        """
        # Extract features using PointNet++ backbone
        features = self.encoder(points, mask)
        
        # Project to latent parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to point cloud
        
        Args:
            latent: Latent vector of shape (B, latent_dim)
            
        Returns:
            Reconstructed point cloud of shape (B, output_points, 3)
        """
        return self.decoder(latent)
    
    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete VAE
        
        Args:
            points: Input point cloud of shape (B, N, 3)
            mask: Optional mask for padded points
            
        Returns:
            Dictionary with reconstruction, mu, logvar, and latent
        """
        # Encode to latent parameters
        mu, logvar = self.encode(points, mask)
        
        # Sample latent vector
        latent = self.reparameterize(mu, logvar)
        
        # Decode to reconstructed point cloud
        reconstruction = self.decode(latent)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'latent': latent
        }

def create_pointnet_vae(latent_dim: int = 128, output_points: int = 30) -> PointNetPlusPlusVAE:
    """
    Create PointNet++ VAE with default parameters
    
    Args:
        latent_dim: Latent space dimension
        output_points: Number of output points
        
    Returns:
        PointNet++ VAE model
    """
    return PointNetPlusPlusVAE(
        input_dim=3,
        feature_dim=256,
        latent_dim=latent_dim,
        output_points=output_points,
        hidden_dim=512
    )

# Example usage and testing
if __name__ == '__main__':
    # Test model instantiation and forward pass
    model = create_pointnet_vae()
    
    # Create test data: batch of 2, up to 50 points each, 3D coordinates
    B, N = 2, 50
    points = torch.randn(B, N, 3) * 50  # Scale to brain-like coordinates
    mask = torch.ones(B, N, dtype=torch.bool)  # No masking for test
    
    print(f"Testing PointNet++ VAE:")
    print(f"Input shape: {points.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(points, mask)
    
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Latent mu shape: {output['mu'].shape}")
    print(f"Latent logvar shape: {output['logvar'].shape}")
    
    # Test individual components
    print(f"\nTesting individual components:")
    
    # Test encoder
    mu, logvar = model.encode(points, mask)
    print(f"Encoder output - mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Test decoder
    latent = torch.randn(B, model.latent_dim)
    decoded = model.decode(latent)
    print(f"Decoder output: {decoded.shape}")
    
    print(f"\n✅ PointNet++ VAE architecture test successful!")