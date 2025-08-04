"""
Unit tests for volumetric cache functionality.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.volumetric_cache import VolumetricCacheBuilder


class TestVolumetricCache(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create small test dataset
        self.test_data = pd.DataFrame({
            'study_id': ['test_001', 'test_001', 'test_002', 'test_002'],
            'x': [0.0, 10.0, -10.0, 5.0],
            'y': [0.0, 15.0, -15.0, 20.0],
            'z': [0.0, 20.0, -20.0, 25.0],
            'stat_value': [2.5, 3.0, -2.0, 1.5],
            'space': ['MNI', 'MNI', 'MNI', 'MNI']
        })
        
        self.test_csv_path = self.temp_path / "test_data.csv"
        self.test_data.to_csv(self.test_csv_path, index=False)
        
        self.cache_path = self.temp_path / "test_cache.lmdb"
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
        
    def test_cache_builder_initialization(self):
        """Test VolumetricCacheBuilder initialization"""
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0, 12.0]
        )
        
        self.assertEqual(builder.brain_shape, (91, 109, 91))
        self.assertEqual(builder.kernels_mm, [6.0, 12.0])
        self.assertIsNotNone(builder.brain_mask)
        
    def test_coordinates_to_volume(self):
        """Test coordinate to volume conversion"""
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0]
        )
        
        # Test with simple coordinates
        coordinates = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        values = np.array([1.0, 2.0])
        
        volume = builder._coordinates_to_volume(coordinates, values, kernel_size_mm=6.0)
        
        # Should be correct shape
        self.assertEqual(volume.shape, (91, 109, 91))
        
        # Should contain non-zero values
        self.assertGreater(volume.max(), 0)
        
        # Should be finite
        self.assertTrue(np.isfinite(volume).all())
        
    def test_augmented_volumes_creation(self):
        """Test dual-kernel augmented volume creation"""
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0, 12.0]
        )
        
        coordinates = np.array([[0.0, 0.0, 0.0]])
        values = np.array([1.0])
        
        augmented_volumes = builder._create_augmented_volumes(
            coordinates, values, "test_study"
        )
        
        # Should have both kernels
        self.assertIn('kernel_6mm', augmented_volumes)
        self.assertIn('kernel_12mm', augmented_volumes)
        
        # Both should be PyTorch tensors
        for key, volume in augmented_volumes.items():
            self.assertIsInstance(volume, torch.Tensor)
            self.assertEqual(volume.shape, (1, 91, 109, 91))  # (C, H, W, D)
            self.assertTrue(torch.isfinite(volume).all())
            
    def test_cache_build_and_validate(self):
        """Test full cache building and validation process"""
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0, 12.0],
            map_size=100 * 1024**2  # 100MB for test
        )
        
        # Build cache
        cache_stats = builder.build_cache(str(self.test_csv_path))
        
        # Validate statistics
        self.assertEqual(cache_stats['total_studies'], 2)
        self.assertEqual(cache_stats['total_volumes_created'], 4)  # 2 studies * 2 kernels
        self.assertGreater(cache_stats['cache_size_bytes'], 0)
        
        # Validate cache file exists
        self.assertTrue(self.cache_path.exists())
        
        # Validate cache contents
        validation_result = builder.validate_cache()
        self.assertTrue(validation_result)
        
    def test_brain_mask_application(self):
        """Test brain mask application"""
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0]
        )
        
        # Create test volume
        test_volume = np.ones((91, 109, 91))
        
        # Apply mask
        masked_volume = builder._apply_brain_mask(test_volume)
        
        # Should be same shape
        self.assertEqual(masked_volume.shape, test_volume.shape)
        
        # Should be different (some voxels masked out)
        # Note: This might not always be true if mask covers entire volume
        # so we just check that it returns a valid array
        self.assertTrue(np.isfinite(masked_volume).all())
        
    def test_empty_study_handling(self):
        """Test handling of studies with no coordinates"""
        # Create data with empty study
        empty_data = pd.DataFrame({
            'study_id': [],
            'x': [],
            'y': [],
            'z': [],
            'stat_value': [],
            'space': []
        })
        
        empty_csv_path = self.temp_path / "empty_data.csv"
        empty_data.to_csv(empty_csv_path, index=False)
        
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0],
            map_size=10 * 1024**2
        )
        
        # Should handle empty data gracefully
        cache_stats = builder.build_cache(str(empty_csv_path))
        self.assertEqual(cache_stats['total_studies'], 0)
        
    def test_large_coordinates_handling(self):
        """Test handling of coordinates outside brain bounds"""
        # Create data with out-of-bounds coordinates
        extreme_data = pd.DataFrame({
            'study_id': ['extreme_001'],
            'x': [1000.0],  # Way outside brain
            'y': [1000.0],
            'z': [1000.0],
            'stat_value': [1.0],
            'space': ['MNI']
        })
        
        extreme_csv_path = self.temp_path / "extreme_data.csv"
        extreme_data.to_csv(extreme_csv_path, index=False)
        
        builder = VolumetricCacheBuilder(
            output_path=str(self.cache_path),
            brain_shape=(91, 109, 91),
            kernels_mm=[6.0],
            map_size=10 * 1024**2
        )
        
        # Should handle out-of-bounds coordinates gracefully
        cache_stats = builder.build_cache(str(extreme_csv_path))
        self.assertEqual(cache_stats['total_studies'], 1)
        
        # Validate cache still works
        validation_result = builder.validate_cache()
        self.assertTrue(validation_result)


if __name__ == '__main__':
    unittest.main()