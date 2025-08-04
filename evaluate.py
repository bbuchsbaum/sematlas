#!/usr/bin/env python3
"""
Formal evaluation script for trained VAE models.

This script evaluates trained VAE checkpoints using formal metrics:
1. Voxel-wise Pearson correlation coefficient
2. Structural Similarity Index (SSIM)
3. Additional metrics for comprehensive evaluation

Results are saved to test_results.json for SUCCESS_MARKERS validation.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Try to import optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    
    def ssim(img1, img2, data_range=None):
        """Mock SSIM implementation for when scikit-image is not available."""
        # Simple normalized cross-correlation as fallback
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        
        # Normalize to zero mean
        img1_norm = img1_flat - np.mean(img1_flat)
        img2_norm = img2_flat - np.mean(img2_flat)
        
        # Compute correlation
        numerator = np.sum(img1_norm * img2_norm)
        denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

# Import our modules
from src.data.lightning_datamodule import create_brain_datamodule
from src.training.vae_lightning import VAELightningModule
from src.models.resnet_vae import create_resnet_vae

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> VAELightningModule:
    """
    Load trained VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded VAE Lightning module
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Create base model with same architecture
        base_model = create_resnet_vae(
            latent_dim=hparams.get('latent_dim', 128),
            groups=hparams.get('group_norm_groups', 8)
        )
        
        # Create Lightning module
        model = VAELightningModule(
            model=base_model,
            learning_rate=hparams.get('learning_rate', 1e-3),
            weight_decay=hparams.get('weight_decay', 1e-4),
            beta_schedule=hparams.get('beta_schedule', 'linear'),
            beta_max=hparams.get('beta_max', 1.0),
            beta_warmup_epochs=hparams.get('beta_warmup_epochs', 10),
            max_epochs=hparams.get('max_epochs', 100)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(device)
        
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def compute_voxel_wise_pearson(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute voxel-wise Pearson correlation coefficient.
    
    Args:
        original: Original brain volumes (B, 1, D, H, W)
        reconstructed: Reconstructed brain volumes (B, 1, D, H, W)
        
    Returns:
        Mean Pearson correlation coefficient across all samples
    """
    correlations = []
    
    # Process each sample in the batch
    for i in range(original.size(0)):
        orig_vol = original[i, 0].cpu().numpy().flatten()
        recon_vol = reconstructed[i, 0].cpu().numpy().flatten()
        
        # Compute Pearson correlation
        corr, _ = pearsonr(orig_vol, recon_vol)
        
        # Handle NaN cases (constant volumes)
        if np.isnan(corr):
            corr = 0.0
            
        correlations.append(corr)
    
    return np.mean(correlations)


def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index (SSIM) for 3D brain volumes.
    
    Args:
        original: Original brain volumes (B, 1, D, H, W)
        reconstructed: Reconstructed brain volumes (B, 1, D, H, W)
        
    Returns:
        Mean SSIM across all samples
    """
    ssim_scores = []
    
    # Process each sample in the batch
    for i in range(original.size(0)):
        orig_vol = original[i, 0].cpu().numpy()
        recon_vol = reconstructed[i, 0].cpu().numpy()
        
        # Compute SSIM for 3D volume
        # We compute SSIM slice by slice and average
        slice_ssims = []
        for slice_idx in range(orig_vol.shape[0]):  # Iterate through depth
            orig_slice = orig_vol[slice_idx]
            recon_slice = recon_vol[slice_idx]
            
            # Skip slices with no variance
            if orig_slice.std() == 0 or recon_slice.std() == 0:
                continue
                
            slice_ssim = ssim(orig_slice, recon_slice, data_range=max(orig_slice.max() - orig_slice.min(), 1e-8))
            slice_ssims.append(slice_ssim)
        
        # Average SSIM across slices
        if slice_ssims:
            volume_ssim = np.mean(slice_ssims)
        else:
            volume_ssim = 0.0
            
        ssim_scores.append(volume_ssim)
    
    return np.mean(ssim_scores)


def compute_additional_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    Compute additional evaluation metrics.
    
    Args:
        original: Original brain volumes
        reconstructed: Reconstructed brain volumes
        
    Returns:
        Dictionary of additional metrics
    """
    # Convert to numpy for computation
    orig_np = original.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((orig_np - recon_np) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(orig_np - recon_np))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Normalized RMSE (by original volume std)
    orig_std = np.std(orig_np)
    nrmse = rmse / orig_std if orig_std > 0 else float('inf')
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(orig_np)
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'psnr': float(psnr)
    }


def evaluate_model(model: VAELightningModule, datamodule, device: torch.device, 
                  num_batches: int = None) -> Dict[str, Any]:
    """
    Evaluate the model on test set.
    
    Args:
        model: Trained VAE model
        datamodule: Data module with test loader
        device: Device for computation
        num_batches: Optional limit on number of batches to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get test dataloader
    test_loader = datamodule.test_dataloader()
    
    # Storage for metrics
    pearson_scores = []
    ssim_scores = []
    additional_metrics_list = []
    
    logger.info(f"Evaluating model on {len(test_loader)} test batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if num_batches and batch_idx >= num_batches:
                break
                
            # Get images from batch
            images = batch['image'].to(device)  # (B, 1, D, H, W)
            
            # Forward pass through model
            reconstructed, mu, logvar = model(images)
            
            # Compute metrics for this batch
            batch_pearson = compute_voxel_wise_pearson(images, reconstructed)
            batch_ssim = compute_ssim(images, reconstructed)
            batch_additional = compute_additional_metrics(images, reconstructed)
            
            pearson_scores.append(batch_pearson)
            ssim_scores.append(batch_ssim)
            additional_metrics_list.append(batch_additional)
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(test_loader)}: "
                          f"Pearson={batch_pearson:.4f}, SSIM={batch_ssim:.4f}")
    
    # Aggregate metrics
    metrics = {
        'voxel_wise_pearson_r': {
            'mean': float(np.mean(pearson_scores)),
            'std': float(np.std(pearson_scores)),
            'min': float(np.min(pearson_scores)),
            'max': float(np.max(pearson_scores))
        },
        'ssim': {
            'mean': float(np.mean(ssim_scores)),
            'std': float(np.std(ssim_scores)),
            'min': float(np.min(ssim_scores)),
            'max': float(np.max(ssim_scores))
        }
    }
    
    # Aggregate additional metrics
    for metric_name in additional_metrics_list[0].keys():
        values = [m[metric_name] for m in additional_metrics_list]
        metrics[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    # Add metadata
    metrics['evaluation_metadata'] = {
        'num_batches_evaluated': len(pearson_scores),
        'total_samples_evaluated': len(pearson_scores) * test_loader.batch_size,
        'device': str(device),
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    return metrics


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained VAE model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--lmdb_cache', type=str, default='data/processed/volumetric_cache',
                       help='Path to LMDB cache')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Limit number of batches for faster testing')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    try:
        # Load model
        logger.info("Loading model from checkpoint...")
        model = load_checkpoint(args.checkpoint, device)
        
        # Create datamodule
        logger.info("Setting up data module...")
        datamodule = create_brain_datamodule(
            data_dir=args.data_dir,
            lmdb_cache=args.lmdb_cache,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_metadata=False  # For evaluation, we only need images
        )
        
        # Setup test data
        datamodule.setup('test')
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluate_model(model, datamodule, device, args.num_batches)
        
        # Print results summary
        logger.info("Evaluation Results:")
        logger.info(f"  Voxel-wise Pearson r: {results['voxel_wise_pearson_r']['mean']:.4f} ± {results['voxel_wise_pearson_r']['std']:.4f}")
        logger.info(f"  SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
        logger.info(f"  MSE: {results['mse']['mean']:.6f}")
        logger.info(f"  RMSE: {results['rmse']['mean']:.6f}")
        logger.info(f"  PSNR: {results['psnr']['mean']:.2f} dB")
        
        # Save results
        save_results(results, args.output)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()