#!/usr/bin/env python3
"""
Script to validate if a PyTorch Lightning checkpoint contains a genuinely trained model.

This script loads the checkpoint and compares its parameters against a freshly 
initialized model to determine if training actually occurred.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.training.vae_lightning import VAELightningModule

def cosine_similarity(tensor1, tensor2):
    """Compute cosine similarity between two tensors."""
    # Flatten tensors
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    # Compute cosine similarity
    dot_product = torch.dot(flat1, flat2)
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (dot_product / (norm1 * norm2)).item()

def validate_checkpoint(checkpoint_path, verbose=True):
    """
    Validate if a checkpoint contains a trained model.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        verbose: Whether to print detailed information
        
    Returns:
        dict: Validation results
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return {"valid": False, "error": f"Checkpoint not found: {checkpoint_path}"}
    
    try:
        # Load the checkpoint
        if verbose:
            print(f"Loading checkpoint: {checkpoint_path}")
            print(f"Checkpoint size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if verbose:
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if it has the expected structure
        required_keys = ['state_dict', 'hyper_parameters']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            return {
                "valid": False, 
                "error": f"Missing required keys: {missing_keys}"
            }
        
        # Get hyperparameters
        hparams = checkpoint['hyper_parameters'].copy()
        if verbose:
            print(f"Stored hyperparameters: {hparams}")
        
        # Check actual latent dimension from weights (in case hyperparams are wrong)
        state_dict = checkpoint['state_dict']
        mu_weight_key = 'vae.encoder.fc_mu.weight'
        if mu_weight_key in state_dict:
            actual_latent_dim = state_dict[mu_weight_key].shape[0]
            if actual_latent_dim != hparams['latent_dim']:
                if verbose:
                    print(f"WARNING: Hyperparameter latent_dim={hparams['latent_dim']} doesn't match actual weights latent_dim={actual_latent_dim}")
                    print(f"Using actual latent_dim={actual_latent_dim} from weights")
                hparams['latent_dim'] = actual_latent_dim
        
        # Handle parameter name mismatches in older checkpoints
        if 'group_norm_groups' in hparams and 'groups' not in hparams:
            hparams['groups'] = hparams.pop('group_norm_groups')
        
        # Create a fresh model with the corrected hyperparameters
        fresh_model = VAELightningModule(**hparams)
        
        # Create another fresh model to load the checkpoint into
        loaded_model = VAELightningModule(**hparams)
        loaded_model.load_state_dict(checkpoint['state_dict'])
        
        # Compare parameters between fresh and loaded models
        similarities = []
        l2_distances = []
        param_names = []
        
        for name, fresh_param in fresh_model.named_parameters():
            if name in dict(loaded_model.named_parameters()):
                loaded_param = dict(loaded_model.named_parameters())[name]
                
                # Calculate cosine similarity
                sim = cosine_similarity(fresh_param, loaded_param)
                similarities.append(sim)
                
                # Calculate L2 distance
                l2_dist = torch.norm(fresh_param - loaded_param).item()
                l2_distances.append(l2_dist)
                
                param_names.append(name)
                
                if verbose and len(similarities) <= 5:  # Show first 5 parameters
                    print(f"  {name}: similarity={sim:.4f}, L2_distance={l2_dist:.4f}")
        
        avg_similarity = np.mean(similarities)
        avg_l2_distance = np.mean(l2_distances)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        # Training indicators
        has_epoch = 'epoch' in checkpoint and checkpoint['epoch'] > 0
        has_global_step = 'global_step' in checkpoint and checkpoint['global_step'] > 0
        has_optimizer_states = 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0
        
        # Determine if model appears trained
        # A truly trained model should have:
        # 1. Low similarity to fresh initialization (different weights)
        # 2. Evidence of training progress (epoch, global_step)
        # 3. Optimizer states (momentum buffers, etc.)
        
        is_trained = (
            avg_similarity < 0.95 and  # Weights have changed significantly
            min_similarity < 0.9 and   # At least some layers changed substantially
            has_epoch and              # Training occurred
            has_global_step            # Steps were taken
        )
        
        results = {
            "valid": True,
            "is_trained": is_trained,
            "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024*1024),
            "avg_similarity_to_fresh": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "avg_l2_distance": avg_l2_distance,
            "num_parameters": len(similarities),
            "epoch": checkpoint.get('epoch', 0),
            "global_step": checkpoint.get('global_step', 0),
            "has_optimizer_states": has_optimizer_states,
            "hyperparameters": hparams
        }
        
        if verbose:
            print(f"\n=== VALIDATION RESULTS ===")
            print(f"Checkpoint appears trained: {is_trained}")
            print(f"Average similarity to fresh model: {avg_similarity:.4f}")
            print(f"Min/Max similarity: {min_similarity:.4f}/{max_similarity:.4f}")
            print(f"Average L2 distance: {avg_l2_distance:.4f}")
            print(f"Epoch: {results['epoch']}")
            print(f"Global step: {results['global_step']}")
            print(f"Has optimizer states: {has_optimizer_states}")
            print(f"Number of parameters compared: {len(similarities)}")
        
        return results
        
    except Exception as e:
        return {"valid": False, "error": str(e)}

def main():
    """Main function to validate the checkpoint."""
    checkpoint_path = "mock_checkpoint.ckpt"
    
    print("🔍 Validating PyTorch Lightning Checkpoint")
    print("=" * 50)
    
    results = validate_checkpoint(checkpoint_path, verbose=True)
    
    if not results["valid"]:
        print(f"❌ ERROR: {results['error']}")
        return False
    
    print("\n" + "=" * 50)
    if results["is_trained"]:
        print("✅ PASS: Checkpoint contains a trained model")
        print("   - Parameters differ significantly from fresh initialization")
        print("   - Training progress indicators present")
        print("   - Model appears to have learned from data")
    else:
        print("❌ FAIL: Checkpoint appears to contain untrained model")
        print("   - Parameters too similar to fresh initialization")
        print("   - May be a mock or untrained checkpoint")
    
    return results["is_trained"]

if __name__ == "__main__":
    main()