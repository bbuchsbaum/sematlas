"""
Metadata imputation module with amortization head for missing metadata.

This module implements uncertainty-aware imputation of missing metadata
using an amortization head that outputs (μ, log σ²) for each metadata field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
import numpy as np


class MetadataImputation(nn.Module):
    """
    Amortization head for metadata imputation with uncertainty quantification.
    
    This module takes image features and imputes missing metadata values
    with uncertainty estimates using the reparameterization trick.
    """
    
    def __init__(self, feature_dim: int, metadata_config: Dict[str, Any]):
        """
        Initialize metadata imputation module.
        
        Args:
            feature_dim: Dimension of input features (from encoder)
            metadata_config: Configuration for metadata fields
                Format: {
                    'field_name': {
                        'type': 'continuous' | 'categorical',
                        'dim': int,  # For continuous: 1, for categorical: num_classes
                        'missing_rate': float,  # Expected missing rate (0-1)
                        'prior_mean': float,  # Prior mean for continuous variables
                        'prior_std': float   # Prior std for continuous variables
                    }
                }
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.metadata_config = metadata_config
        
        # Build imputation heads for each metadata field
        self.imputation_heads = nn.ModuleDict()
        self.total_metadata_dim = 0
        
        for field_name, config in metadata_config.items():
            field_dim = config['dim']
            self.total_metadata_dim += field_dim
            
            if config['type'] == 'continuous':
                # For continuous: output (μ, log σ²)
                self.imputation_heads[field_name] = self._build_continuous_head(
                    feature_dim, field_dim
                )
            elif config['type'] == 'categorical':
                # For categorical: output logits
                self.imputation_heads[field_name] = self._build_categorical_head(
                    feature_dim, field_dim
                )
            else:
                raise ValueError(f"Unknown metadata type: {config['type']}")
        
        # Store field info for efficient processing
        self.continuous_fields = [name for name, config in metadata_config.items() 
                                 if config['type'] == 'continuous']
        self.categorical_fields = [name for name, config in metadata_config.items() 
                                  if config['type'] == 'categorical']
        
    def _build_continuous_head(self, feature_dim: int, output_dim: int) -> nn.Module:
        """Build amortization head for continuous metadata."""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, output_dim * 2),  # Output (μ, log σ²)
        )
    
    def _build_categorical_head(self, feature_dim: int, num_classes: int) -> nn.Module:
        """Build amortization head for categorical metadata."""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, num_classes),  # Output logits
        )
    
    def forward(self, features: torch.Tensor, 
                observed_metadata: Optional[Dict[str, torch.Tensor]] = None,
                missing_mask: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for metadata imputation.
        
        Args:
            features: Image features of shape (batch_size, feature_dim)
            observed_metadata: Dict of observed metadata tensors (optional)
            missing_mask: Dict of boolean masks indicating missing values (optional)
            
        Returns:
            Dict containing imputed metadata and uncertainty estimates:
            {
                'field_name': tensor,
                'field_name_mu': tensor (for continuous),
                'field_name_logvar': tensor (for continuous),
                'field_name_uncertainty': tensor (for continuous)
            }
        """
        batch_size = features.size(0)
        results = {}
        
        for field_name, head in self.imputation_heads.items():
            config = self.metadata_config[field_name]
            
            if config['type'] == 'continuous':
                # Get (μ, log σ²) from amortization head
                output = head(features)  # Shape: (batch_size, field_dim * 2)
                field_dim = config['dim']
                
                mu = output[:, :field_dim]  # Shape: (batch_size, field_dim)
                logvar = output[:, field_dim:]  # Shape: (batch_size, field_dim)
                
                # Reparameterization trick for imputed values
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                imputed_value = mu + eps * std
                
                # Store all outputs
                results[f'{field_name}_mu'] = mu
                results[f'{field_name}_logvar'] = logvar
                results[f'{field_name}_uncertainty'] = std
                
                # Use observed values where available, imputed values where missing
                if observed_metadata is not None and field_name in observed_metadata:
                    observed = observed_metadata[field_name]
                    if missing_mask is not None and field_name in missing_mask:
                        mask = missing_mask[field_name]  # True = missing
                        final_value = torch.where(mask.unsqueeze(-1), imputed_value, observed)
                    else:
                        final_value = observed
                else:
                    final_value = imputed_value
                
                results[field_name] = final_value
                
            elif config['type'] == 'categorical':
                # Get logits from head
                logits = head(features)  # Shape: (batch_size, num_classes)
                
                # Sample from categorical distribution (Gumbel-Softmax for differentiability)
                if self.training:
                    # Use Gumbel-Softmax during training
                    tau = 1.0  # Temperature parameter
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
                    y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
                else:
                    # Use hard assignment during inference
                    y = F.one_hot(torch.argmax(logits, dim=-1), num_classes=logits.size(-1)).float()
                
                # Use observed values where available
                if observed_metadata is not None and field_name in observed_metadata:
                    observed = observed_metadata[field_name]
                    if missing_mask is not None and field_name in missing_mask:
                        mask = missing_mask[field_name]  # True = missing
                        final_value = torch.where(mask.unsqueeze(-1), y, observed)
                    else:
                        final_value = observed
                else:
                    final_value = y
                
                results[field_name] = final_value
                results[f'{field_name}_logits'] = logits
        
        return results
    
    def compute_imputation_loss(self, imputed_metadata: Dict[str, torch.Tensor],
                               observed_metadata: Dict[str, torch.Tensor],
                               missing_mask: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute imputation loss for observed metadata.
        
        Args:
            imputed_metadata: Output from forward pass
            observed_metadata: Ground truth metadata
            missing_mask: Boolean mask (True = missing, False = observed)
            
        Returns:
            Scalar imputation loss
        """
        total_loss = 0.0
        num_losses = 0
        
        for field_name, config in self.metadata_config.items():
            if field_name not in observed_metadata:
                continue
                
            observed = observed_metadata[field_name]
            mask = missing_mask.get(field_name, torch.zeros_like(observed[:, 0], dtype=torch.bool))
            
            # Only compute loss for observed values (mask = False)
            observed_mask = ~mask
            if not observed_mask.any():
                continue
                
            if config['type'] == 'continuous':
                # Negative log-likelihood loss for continuous variables
                mu = imputed_metadata[f'{field_name}_mu'][observed_mask]
                logvar = imputed_metadata[f'{field_name}_logvar'][observed_mask]
                target = observed[observed_mask]
                
                # NLL loss: -log p(x|μ,σ²) = 0.5 * (log(2π) + logvar + (x-μ)²/σ²)
                mse = (target - mu).pow(2)
                var = torch.exp(logvar)
                nll = 0.5 * (logvar + mse / var)
                loss = nll.mean()
                
            elif config['type'] == 'categorical':
                # Cross-entropy loss for categorical variables
                logits = imputed_metadata[f'{field_name}_logits'][observed_mask]
                target = observed[observed_mask]
                
                # Convert one-hot to class indices if needed
                if target.dim() > 1 and target.size(-1) > 1:
                    target = torch.argmax(target, dim=-1)
                    
                loss = F.cross_entropy(logits, target)
            
            total_loss += loss
            num_losses += 1
        
        return total_loss / max(num_losses, 1)
    
    def get_metadata_vector(self, imputed_metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate all imputed metadata into a single vector.
        
        Args:
            imputed_metadata: Output from forward pass
            
        Returns:
            Concatenated metadata vector of shape (batch_size, total_metadata_dim)
        """
        vectors = []
        
        for field_name in self.metadata_config.keys():
            if field_name in imputed_metadata:
                vectors.append(imputed_metadata[field_name])
                
        return torch.cat(vectors, dim=-1)


def create_default_metadata_config() -> Dict[str, Any]:
    """
    Create default metadata configuration for neuroscience studies.
    
    Returns:
        Default metadata configuration dict
    """
    return {
        'sample_size': {
            'type': 'continuous',
            'dim': 1,
            'missing_rate': 0.2,
            'prior_mean': 50.0,
            'prior_std': 30.0
        },
        'study_year': {
            'type': 'continuous', 
            'dim': 1,
            'missing_rate': 0.1,
            'prior_mean': 2010.0,
            'prior_std': 8.0
        },
        'task_category': {
            'type': 'categorical',
            'dim': 10,  # 10 common task categories
            'missing_rate': 0.3,
        },
        'scanner_field_strength': {
            'type': 'categorical',
            'dim': 3,  # 1.5T, 3T, 7T
            'missing_rate': 0.4,
        },
        'statistical_threshold': {
            'type': 'continuous',
            'dim': 1,
            'missing_rate': 0.5,
            'prior_mean': 3.0,
            'prior_std': 1.0
        }
    }


def create_mock_metadata_batch(batch_size: int, metadata_config: Dict[str, Any], 
                              missing_rate: float = 0.3) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create mock metadata batch for testing.
    
    Args:
        batch_size: Number of samples
        metadata_config: Metadata configuration
        missing_rate: Overall missing rate
        
    Returns:
        Tuple of (metadata_dict, missing_mask_dict)
    """
    metadata = {}
    missing_mask = {}
    
    for field_name, config in metadata_config.items():
        field_dim = config['dim']
        
        if config['type'] == 'continuous':
            # Generate continuous values
            mean = config.get('prior_mean', 0.0)
            std = config.get('prior_std', 1.0)
            values = torch.normal(mean, std, (batch_size, field_dim))
            
        elif config['type'] == 'categorical':
            # Generate categorical values as one-hot
            indices = torch.randint(0, field_dim, (batch_size,))
            values = F.one_hot(indices, num_classes=field_dim).float()
        
        # Generate missing mask
        field_missing_rate = config.get('missing_rate', missing_rate)
        mask = torch.rand(batch_size) < field_missing_rate
        
        metadata[field_name] = values
        missing_mask[field_name] = mask
    
    return metadata, missing_mask


if __name__ == "__main__":
    # Test the metadata imputation module
    print("Testing Metadata Imputation Module...")
    
    # Create test configuration
    metadata_config = create_default_metadata_config()
    feature_dim = 512
    batch_size = 4
    
    # Create imputation module
    imputer = MetadataImputation(feature_dim, metadata_config)
    
    # Create mock data
    features = torch.randn(batch_size, feature_dim)
    observed_metadata, missing_mask = create_mock_metadata_batch(batch_size, metadata_config)
    
    print(f"Feature dim: {feature_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Total metadata dim: {imputer.total_metadata_dim}")
    print(f"Continuous fields: {imputer.continuous_fields}")
    print(f"Categorical fields: {imputer.categorical_fields}")
    
    # Test forward pass
    with torch.no_grad():
        imputed = imputer(features, observed_metadata, missing_mask)
        
        print("\nForward pass results:")
        for key, value in imputed.items():
            print(f"  {key}: {value.shape}")
        
        # Test metadata vector creation
        metadata_vector = imputer.get_metadata_vector(imputed)
        print(f"\nMetadata vector shape: {metadata_vector.shape}")
        
        # Test imputation loss
        loss = imputer.compute_imputation_loss(imputed, observed_metadata, missing_mask)
        print(f"Imputation loss: {loss.item():.4f}")
    
    print("\n✅ Metadata imputation module test passed!")