#!/usr/bin/env python3
"""
Validation script for baseline VAE latent representations.

This script validates that the trained VAE model learns meaningful latent representations
by testing encoding, reconstruction, and latent space traversal capabilities.

Usage:
    python validate_baseline_latents.py --checkpoint path/to/checkpoint.ckpt
    python validate_baseline_latents.py --untrained  # Test with untrained model
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import nibabel as nib
# from nilearn import plotting
# import pandas as pd

from src.models.resnet_vae import ResNetVAE3D
from src.data.lightning_datamodule import BrainVolumeDataModule
# from src.inference.model_wrapper import BrainAtlasInference


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model(checkpoint_path: Optional[str] = None) -> ResNetVAE3D:
    """Load trained or untrained model."""
    # Default latent dim (may be overridden if loading checkpoint)
    latent_dim = 128
    
    # If checkpoint provided, check its latent dimension first
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to infer latent_dim from checkpoint
        state_dict = checkpoint.get('state_dict', checkpoint)
        for key, value in state_dict.items():
            if 'fc_mu.weight' in key:
                latent_dim = value.shape[0]
                logging.info(f"Detected latent_dim={latent_dim} from checkpoint")
                break
    
    model = ResNetVAE3D(
        input_channels=1,
        latent_dim=latent_dim,
        groups=8
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle both direct model state dict and Lightning module state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # If keys have 'vae.' prefix (Lightning module), remove it
        if any(key.startswith('vae.') for key in state_dict.keys()):
            state_dict = {key.replace('vae.', ''): value for key, value in state_dict.items() if key.startswith('vae.')}
            logging.info("Detected Lightning checkpoint, removed 'vae.' prefix from keys")
        
        model.load_state_dict(state_dict)
        logging.info("Checkpoint loaded successfully")
    else:
        logging.info("Using untrained model for baseline comparison")
    
    model.eval()
    return model


def compute_reconstruction_metrics(
    model: ResNetVAE3D,
    datamodule: BrainVolumeDataModule,
    num_samples: int = 10
) -> Dict[str, float]:
    """Compute reconstruction quality metrics."""
    logging.info(f"Computing reconstruction metrics on {num_samples} samples...")
    
    dataloader = datamodule.val_dataloader()
    mse_values = []
    correlation_values = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            # Forward pass
            if isinstance(batch, dict):
                x = batch['volume']
            elif isinstance(batch, torch.Tensor):
                x = batch
            else:
                x = batch[0]
            mu, logvar = model.encode(x)
            # Sample z using reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            x_recon = model.decode(z)
            
            # Compute metrics
            mse = F.mse_loss(x_recon, x, reduction='none').mean(dim=(1,2,3,4))
            mse_values.extend(mse.cpu().numpy())
            
            # Compute correlation per sample
            for j in range(x.shape[0]):
                orig = x[j].cpu().numpy().flatten()
                recon = x_recon[j].cpu().numpy().flatten()
                
                # Only compute correlation on non-zero voxels
                mask = orig != 0
                if mask.sum() > 0:
                    corr = np.corrcoef(orig[mask], recon[mask])[0, 1]
                    correlation_values.append(corr)
    
    metrics = {
        'mean_mse': float(np.mean(mse_values)),
        'std_mse': float(np.std(mse_values)),
        'mean_correlation': float(np.mean(correlation_values)),
        'std_correlation': float(np.std(correlation_values)),
        'num_samples': len(mse_values)
    }
    
    return metrics


def test_latent_space_traversal(
    model: ResNetVAE3D,
    datamodule: BrainVolumeDataModule,
    num_dimensions: int = 5,
    num_steps: int = 7,
    output_dir: Path = Path("validation_outputs")
) -> Dict[str, Any]:
    """Test latent space traversal to verify meaningful representations."""
    logging.info("Testing latent space traversal...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Get a sample from the data
    dataloader = datamodule.val_dataloader()
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, dict):
        x = sample_batch['volume']
    elif isinstance(sample_batch, torch.Tensor):
        x = sample_batch
    else:
        x = sample_batch[0]
    x_sample = x[0:1]  # Take first sample
    
    with torch.no_grad():
        # Encode to get base latent code
        mu, logvar = model.encode(x_sample)
        # Use mean for base latent code
        z_base = mu
        
        # Test traversal along first few dimensions
        traversal_results = {}
        
        for dim in range(min(num_dimensions, z_base.shape[1])):
            logging.info(f"Traversing dimension {dim}")
            
            # Create interpolation range
            values = np.linspace(-3, 3, num_steps)
            reconstructions = []
            
            for val in values:
                z_modified = z_base.clone()
                z_modified[0, dim] = val
                x_recon = model.decode(z_modified)
                reconstructions.append(x_recon[0, 0].cpu().numpy())
            
            # Compute diversity metric (std of reconstructions)
            recon_stack = np.stack(reconstructions)
            diversity = float(np.std(recon_stack))
            
            traversal_results[f'dim_{dim}_diversity'] = diversity
            
            # Save visualization
            fig, axes = plt.subplots(1, num_steps, figsize=(20, 4))
            for i, (ax, recon) in enumerate(zip(axes, reconstructions)):
                # Show middle slice
                slice_idx = recon.shape[2] // 2
                ax.imshow(recon[:, :, slice_idx], cmap='hot')
                ax.set_title(f'z[{dim}] = {values[i]:.1f}')
                ax.axis('off')
            
            plt.suptitle(f'Latent Dimension {dim} Traversal')
            plt.tight_layout()
            plt.savefig(output_dir / f'traversal_dim_{dim}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Compute overall traversal quality metric
    diversities = [v for k, v in traversal_results.items() if 'diversity' in k]
    traversal_results['mean_diversity'] = float(np.mean(diversities))
    traversal_results['num_dimensions_tested'] = num_dimensions
    
    return traversal_results


def test_anatomical_plausibility(
    model: ResNetVAE3D,
    datamodule: BrainVolumeDataModule,
    num_samples: int = 10
) -> Dict[str, float]:
    """Test if reconstructions are anatomically plausible."""
    logging.info("Testing anatomical plausibility...")
    
    dataloader = datamodule.val_dataloader()
    
    # Simple brain mask (center of volume should have higher values)
    brain_mask_scores = []
    out_of_brain_ratios = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            if isinstance(batch, dict):
                x = batch['volume']
            elif isinstance(batch, torch.Tensor):
                x = batch
            else:
                x = batch[0]
            mu, logvar = model.encode(x)
            # Sample z using reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            x_recon = model.decode(z)
            
            for j in range(x.shape[0]):
                recon = x_recon[j, 0].cpu().numpy()
                
                # Simple anatomical check: center should have more activation than edges
                center_region = recon[20:60, 24:72, 20:60]
                edge_region = np.concatenate([
                    recon[:10, :, :].flatten(),
                    recon[-10:, :, :].flatten(),
                    recon[:, :10, :].flatten(),
                    recon[:, -10:, :].flatten()
                ])
                
                center_mean = np.mean(np.abs(center_region))
                edge_mean = np.mean(np.abs(edge_region))
                
                if edge_mean > 0:
                    brain_mask_score = center_mean / (edge_mean + 1e-6)
                    brain_mask_scores.append(brain_mask_score)
                
                # Check for out-of-brain activations
                total_activation = np.sum(np.abs(recon))
                edge_activation = np.sum(np.abs(edge_region))
                if total_activation > 0:
                    out_of_brain_ratio = edge_activation / total_activation
                    out_of_brain_ratios.append(out_of_brain_ratio)
    
    return {
        'mean_brain_mask_score': float(np.mean(brain_mask_scores)),
        'std_brain_mask_score': float(np.std(brain_mask_scores)),
        'mean_out_of_brain_ratio': float(np.mean(out_of_brain_ratios)),
        'num_samples': len(brain_mask_scores)
    }


def generate_validation_report(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate comprehensive validation report."""
    logging.info(f"Generating validation report at {output_path}")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'validation_results': results,
        'pass_fail_criteria': {
            'reconstruction_mse_improved': results.get('mse_improvement', 0) >= 50.0,
            'correlation_threshold': results.get('reconstruction_metrics', {}).get('mean_correlation', 0) > 0.3,
            'latent_diversity': results.get('traversal_results', {}).get('mean_diversity', 0) > 0.01,
            'anatomically_plausible': results.get('anatomical_plausibility', {}).get('mean_brain_mask_score', 0) > 2.0
        }
    }
    
    # Overall pass/fail
    report['overall_pass'] = all(report['pass_fail_criteria'].values())
    
    # Save JSON report
    with open(output_path / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    logging.info("\n" + "="*50)
    logging.info("VALIDATION SUMMARY")
    logging.info("="*50)
    
    for criterion, passed in report['pass_fail_criteria'].items():
        status = "PASS" if passed else "FAIL"
        logging.info(f"{criterion}: {status}")
    
    logging.info(f"\nOverall Status: {'PASS' if report['overall_pass'] else 'FAIL'}")
    logging.info("="*50)


def main():
    parser = argparse.ArgumentParser(description='Validate baseline VAE latent representations')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--untrained', action='store_true', help='Test with untrained model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='validation_outputs',
                        help='Directory for validation outputs')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for validation')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    if args.untrained:
        model = load_model(None)
        model_type = "untrained"
    else:
        model = load_model(args.checkpoint)
        model_type = "trained"
    
    # Setup data
    logger.info("Setting up data module...")
    datamodule = BrainVolumeDataModule(
        train_split=f"{args.data_dir}/train_split.csv",
        val_split=f"{args.data_dir}/val_split.csv",
        test_split=f"{args.data_dir}/test_split.csv",
        volumetric_cache_path=args.cache_path,
        batch_size=4,
        num_workers=0,
        include_metadata=False,  # Disable metadata for baseline validation
        kernel_selection="random",
        pin_memory=False
    )
    datamodule.setup('fit')
    
    # Run validation tests
    results = {
        'model_type': model_type,
        'checkpoint': args.checkpoint if not args.untrained else None
    }
    
    # 1. Reconstruction metrics
    logger.info("\n1. Testing reconstruction quality...")
    reconstruction_metrics = compute_reconstruction_metrics(
        model, datamodule, args.num_samples
    )
    results['reconstruction_metrics'] = reconstruction_metrics
    
    # 2. Latent space traversal
    logger.info("\n2. Testing latent space traversal...")
    traversal_results = test_latent_space_traversal(
        model, datamodule, num_dimensions=5, output_dir=output_dir
    )
    results['traversal_results'] = traversal_results
    
    # 3. Anatomical plausibility
    logger.info("\n3. Testing anatomical plausibility...")
    anatomical_results = test_anatomical_plausibility(
        model, datamodule, args.num_samples
    )
    results['anatomical_plausibility'] = anatomical_results
    
    # Compute improvement metrics if we have baseline
    baseline_file = output_dir / 'baseline_metrics.json'
    if args.untrained:
        # Save baseline metrics
        with open(baseline_file, 'w') as f:
            json.dump(reconstruction_metrics, f, indent=2)
        logger.info(f"Saved baseline metrics to {baseline_file}")
    elif baseline_file.exists():
        # Compare with baseline
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)
        
        baseline_mse = baseline_metrics['mean_mse']
        current_mse = reconstruction_metrics['mean_mse']
        
        improvement = ((baseline_mse - current_mse) / baseline_mse) * 100
        results['mse_improvement'] = improvement
        logger.info(f"\nMSE Improvement: {improvement:.1f}%")
    
    # Generate report
    generate_validation_report(results, output_dir)
    
    logger.info(f"\nValidation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()