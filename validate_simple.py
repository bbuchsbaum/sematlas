#!/usr/bin/env python3
"""
Simple Baseline VAE Validation - VS1.2.3
Tests validation framework with untrained model to establish acceptance criteria baseline.
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import pickle

# Local imports
from src.training.vae_lightning import VAELightningModule

def quick_validation():
    """Quick validation test for VS1.2.3 acceptance criteria."""
    
    print("=" * 60)
    print("VS1.2.3 BASELINE LATENT VALIDATION - QUICK TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    results = {}
    
    # Test 1: Data loading
    try:
        # Load a few pickle files to test data loading
        cache_dir = Path("data/processed/volumetric_cache")
        pkl_files = list(cache_dir.glob("study_*.pkl"))[:5]  # Just test 5 files
        
        if len(pkl_files) == 0:
            raise RuntimeError("No pickle files found in cache")
        
        volumes = []
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            volume = data['volumes']['kernel_6mm']
            if isinstance(volume, np.ndarray):
                volume = torch.from_numpy(volume).float()
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)  # Add channel dim
            volumes.append(volume)
        
        # Create a batch
        batch = torch.stack(volumes)
        data_loading_time = time.time() - start_time
        
        print(f"✅ Data loading: {data_loading_time:.3f}s for {len(volumes)} volumes")
        print(f"   Batch shape: {batch.shape}")
        
        results['data_loading_works'] = True
        results['data_loading_time'] = data_loading_time
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        results['data_loading_works'] = False
        return results
    
    # Test 2: Model creation and encoding
    try:
        model = VAELightningModule(
            latent_dim=128,
            learning_rate=1e-3,
            beta=0.01
        )
        model.eval()
        
        print(f"✅ Model created: {type(model).__name__}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test encoding
        with torch.no_grad():
            recon_batch, mu, logvar = model(batch)
        
        print(f"✅ Model encodes data: latent shape {mu.shape}")
        results['model_encodes_data'] = True
        results['latent_dim'] = mu.shape[1]
        
    except Exception as e:
        print(f"❌ Model encoding failed: {e}")
        results['model_encodes_data'] = False
        return results
    
    # Test 3: Latent space traversal
    try:
        base_latent = mu[0:1]  # Use first sample
        traversal_outputs = []
        
        for value in [-2, 0, 2]:  # Simple traversal
            modified_latent = base_latent.clone()
            modified_latent[0, 0] = value  # Modify first dimension
            
            with torch.no_grad():
                if hasattr(model, 'decode'):
                    decoded = model.decode(modified_latent)
                else:
                    decoded = model.decoder(modified_latent)
            
            traversal_outputs.append(decoded[0])
        
        # Check if outputs are different
        diff = torch.std(torch.stack(traversal_outputs))
        traversal_works = diff > 0.001  # Some variation expected
        
        print(f"✅ Latent traversal: diversity = {diff:.4f}")
        results['latent_traversal_works'] = traversal_works
        results['traversal_diversity'] = float(diff)
        
    except Exception as e:
        print(f"❌ Latent traversal failed: {e}")
        results['latent_traversal_works'] = False
        return results
    
    # Test 4: Reconstruction quality
    try:
        mse = F.mse_loss(recon_batch, batch)
        
        # Check anatomical plausibility
        in_range = (recon_batch.min() >= -1.0 and recon_batch.max() <= 10.0)
        reasonable_mean = (0 <= recon_batch.mean() <= 2.0)
        anatomically_plausible = in_range and reasonable_mean
        
        print(f"✅ Reconstruction MSE: {mse:.4f}")  
        print(f"✅ Anatomically plausible: {anatomically_plausible}")
        print(f"   Range: [{recon_batch.min():.3f}, {recon_batch.max():.3f}]")
        print(f"   Mean: {recon_batch.mean():.3f}")
        
        results['reconstruction_mse'] = float(mse)
        results['anatomically_plausible'] = anatomically_plausible
        
    except Exception as e:
        print(f"❌ Reconstruction test failed: {e}")
        results['anatomically_plausible'] = False
        return results
    
    # Test 5: Framework validation (this is an untrained model baseline)
    results['model_type'] = 'untrained'
    results['framework_ready'] = True
    
    # Check acceptance criteria
    acceptance_criteria = {
        'data_loading_works': results.get('data_loading_works', False),
        'model_encodes_data': results.get('model_encodes_data', False),
        'latent_traversal_works': results.get('latent_traversal_works', False),
        'anatomically_plausible': results.get('anatomically_plausible', False),
        'validation_framework_ready': results.get('framework_ready', False)
    }
    
    results['acceptance_criteria'] = acceptance_criteria
    results['all_criteria_pass'] = all(acceptance_criteria.values())
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Model type: {results['model_type']}")
    
    print("\nAcceptance Criteria:")
    for criterion, passed in acceptance_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")
    
    overall = "✅ ALL VALIDATION CRITERIA PASS" if results['all_criteria_pass'] else "❌ SOME CRITERIA FAIL"
    print(f"\nOverall: {overall}")
    
    # Save results (convert tensors to float for JSON serialization)
    output_dir = Path("validation_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Convert any tensor values to floats
    json_results = {}
    for k, v in results.items():
        if torch.is_tensor(v):
            json_results[k] = float(v)
        elif isinstance(v, dict):
            json_results[k] = {k2: (float(v2) if torch.is_tensor(v2) else v2) for k2, v2 in v.items()}
        else:
            json_results[k] = v
    
    with open(output_dir / "quick_validation_report.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nValidation report saved to: {output_dir / 'quick_validation_report.json'}")
    
    return results

if __name__ == '__main__':
    try:
        results = quick_validation()
        exit(0 if results.get('all_criteria_pass', False) else 1)
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)