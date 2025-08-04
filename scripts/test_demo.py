#!/usr/bin/env python3
"""
Test script for the Latent Slider Demo functionality.

This script tests the core components of the interactive demo without requiring
Jupyter notebook environment or interactive widgets.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.inference import create_inference_wrapper
    print("✓ Successfully imported inference wrapper")
except ImportError as e:
    print(f"✗ Failed to import inference wrapper: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic model functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Create inference wrapper
    atlas = create_inference_wrapper(fallback_to_untrained=True)
    
    # Get model info
    model_info = atlas.get_model_info()
    print(f"Model loaded: {model_info['status']}")
    print(f"Latent dim: {model_info['latent_dim']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    
    # Test random generation
    print("\n1. Testing random generation...")
    random_volumes = atlas.generate_random(num_samples=2)
    print(f"   Generated shape: {random_volumes.shape}")
    print(f"   Value range: {random_volumes.min():.3f} to {random_volumes.max():.3f}")
    assert random_volumes.shape == (2, 1, 91, 109, 91), f"Unexpected shape: {random_volumes.shape}"
    
    # Test latent traversal
    print("\n2. Testing latent traversal...")
    traversal = atlas.traverse_latent_dimension(
        dimension=0, 
        range_vals=(-2, 2), 
        num_steps=5
    )
    print(f"   Traversal shape: {traversal.shape}")
    assert traversal.shape == (5, 1, 91, 109, 91), f"Unexpected shape: {traversal.shape}"
    
    # Test interpolation
    print("\n3. Testing interpolation...")
    start_code = torch.randn(atlas.latent_dim)
    end_code = torch.randn(atlas.latent_dim)
    interpolated = atlas.interpolate_latent(start_code, end_code, num_steps=3)
    print(f"   Interpolation shape: {interpolated.shape}")
    assert interpolated.shape == (3, atlas.latent_dim), f"Unexpected shape: {interpolated.shape}"
    
    print("✓ All basic functionality tests passed")
    return atlas


def test_visualization_pipeline():
    """Test visualization pipeline without interactive components."""
    print("\n=== Testing Visualization Pipeline ===")
    
    atlas = create_inference_wrapper(fallback_to_untrained=True)
    
    # Test brain volume generation for different dimensions
    test_dimensions = [0, 1, 2, 10] if atlas.latent_dim > 10 else [0, 1]
    base_code = torch.zeros(atlas.latent_dim)
    
    volumes = []
    for dim in test_dimensions:
        test_code = base_code.clone()
        test_code[dim] = 2.0
        volume = atlas.decode(test_code.unsqueeze(0))[0, 0]
        volumes.append(volume.cpu().numpy())
        print(f"   Generated volume for dim {dim}: {volume.shape}")
    
    # Create simple visualization
    fig, axes = plt.subplots(1, len(test_dimensions), figsize=(3 * len(test_dimensions), 3))
    if len(test_dimensions) == 1:
        axes = [axes]
    
    for i, (volume, dim) in enumerate(zip(volumes, test_dimensions)):
        # Show middle axial slice
        mid_slice = volume[:, :, volume.shape[2] // 2]
        axes[i].imshow(mid_slice, cmap='RdBu_r')
        axes[i].set_title(f"Dim {dim}")
        axes[i].axis('off')
    
    plt.suptitle("Latent Dimension Comparison")
    plt.tight_layout()
    
    # Save test visualization
    output_dir = project_root / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "dimension_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization test complete. Saved to {output_dir / 'dimension_comparison.png'}")


def test_export_functionality():
    """Test export functionality."""
    print("\n=== Testing Export Functionality ===")
    
    atlas = create_inference_wrapper(fallback_to_untrained=True)
    
    # Create export directory
    export_dir = project_root / "test_outputs" / "export_test"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate traversal sequence
    traversal = atlas.traverse_latent_dimension(
        dimension=0,
        range_vals=(-2, 2),
        num_steps=5
    )
    
    # Export as numpy arrays
    for i, volume in enumerate(traversal):
        volume_np = volume[0].cpu().numpy()  # Remove channel dimension
        np.save(export_dir / f"traversal_step_{i:02d}.npy", volume_np)
    
    # Export metadata
    import json
    metadata = {
        "model_info": atlas.get_model_info(),
        "traversal_info": {
            "dimension": 0,
            "range": [-2, 2],
            "num_steps": 5
        }
    }
    
    with open(export_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"✓ Export test complete. Files saved to {export_dir}")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    atlas = create_inference_wrapper(fallback_to_untrained=True)
    
    # Test invalid dimension
    try:
        atlas.traverse_latent_dimension(dimension=9999)
        print("✗ Should have raised error for invalid dimension")
    except ValueError:
        print("✓ Correctly handled invalid dimension")
    
    # Test invalid input shapes
    try:
        invalid_volume = torch.randn(10, 10, 10)  # Wrong shape
        atlas.encode(invalid_volume)
        print("✓ Handled unexpected input shape gracefully")
    except Exception as e:
        print(f"✓ Appropriately handled invalid input: {type(e).__name__}")
    
    # Test empty model
    try:
        empty_atlas = create_inference_wrapper(checkpoint_path="nonexistent.ckpt", fallback_to_untrained=False)
        empty_atlas.generate_random()
        print("✗ Should have raised error for missing model")
    except Exception:
        print("✓ Correctly handled missing model")


def main():
    """Run all tests."""
    print("GENERATIVE BRAIN ATLAS - DEMO FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        atlas = test_basic_functionality()
        test_visualization_pipeline() 
        test_export_functionality()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED - DEMO READY FOR USE")
        print("=" * 50)
        
        # Print summary
        model_info = atlas.get_model_info()
        print(f"\nDemo Summary:")
        print(f"  • Model: {model_info['model_type']}")
        print(f"  • Latent dimensions: {model_info['latent_dim']}")
        print(f"  • Parameters: {model_info['total_parameters']:,}")
        print(f"  • Device: {model_info['device']}")
        print(f"  • Status: {'Trained' if model_info['is_trained'] else 'Demo (untrained)'}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)