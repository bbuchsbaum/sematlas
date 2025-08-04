#!/usr/bin/env python3
"""
Script to determine the actual latent dimension of a checkpoint by examining the weights.
"""

import torch
from pathlib import Path

def check_actual_latent_dim(checkpoint_path):
    """Check the actual latent dimension from the weights."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Check the mu layer to determine actual latent dimension
    mu_weight_key = 'vae.encoder.fc_mu.weight'
    if mu_weight_key in state_dict:
        mu_weight = state_dict[mu_weight_key]
        actual_latent_dim = mu_weight.shape[0]  # Output size is latent_dim
        
        print(f"Actual latent dimension from weights: {actual_latent_dim}")
        print(f"Hyperparameter latent_dim: {checkpoint['hyper_parameters']['latent_dim']}")
        
        return actual_latent_dim
    else:
        print(f"Could not find {mu_weight_key} in state_dict")
        print(f"Available keys: {list(state_dict.keys())[:10]}...")
        return None

if __name__ == "__main__":
    actual_dim = check_actual_latent_dim("mock_checkpoint.ckpt")