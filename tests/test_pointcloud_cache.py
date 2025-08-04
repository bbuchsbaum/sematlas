"""
Test Suite for Point-Cloud Cache (S3.1.1)

Validates all SUCCESS_MARKERS.md criteria for S3.1.1:
- [X] HDF5 file created successfully 
- [X] Random study retrieval returns correct coordinate array
- [X] Coordinate format validation (x,y,z tuples)
- [X] Variable-length coordinate lists properly handled
"""

import pytest
import numpy as np
import h5py
import os
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.pointcloud_cache import PointCloudCache, load_pointcloud_cache, get_study_coordinates

class TestPointCloudCache:
    """Test suite for point-cloud cache functionality"""
    
    @pytest.fixture
    def cache_file(self):
        """Fixture providing path to test cache file"""
        return "data/processed/pointcloud_cache_subset_1k.h5"
    
    @pytest.fixture  
    def cache(self, cache_file):
        """Fixture providing loaded cache instance"""
        if not os.path.exists(cache_file):
            pytest.skip(f"Cache file not found: {cache_file}")
        return PointCloudCache(cache_file)
    
    def test_hdf5_file_created_successfully(self, cache_file):
        """SUCCESS CRITERION: HDF5 file created successfully"""
        assert os.path.exists(cache_file), f"HDF5 cache file not found: {cache_file}"
        
        # Verify it's a valid HDF5 file
        with h5py.File(cache_file, 'r') as hf:
            assert 'coordinates' in hf, "Missing 'coordinates' group"
            assert len(hf['coordinates']) > 0, "No studies in coordinates group"
    
    def test_random_study_retrieval_correct_array(self, cache):
        """SUCCESS CRITERION: Random study retrieval returns correct coordinate array"""
        # Get a few random studies
        random_studies = cache.get_random_studies(5)
        assert len(random_studies) > 0, "No random studies returned"
        
        for study_id in random_studies:
            coords = cache.get_coordinates(study_id)
            
            # Verify it's a numpy array
            assert isinstance(coords, np.ndarray), f"Coordinates not numpy array for study {study_id}"
            
            # Verify shape is (N, 3)
            assert coords.ndim == 2, f"Coordinates not 2D array for study {study_id}"
            assert coords.shape[1] == 3, f"Coordinates not (N, 3) shape for study {study_id}, got {coords.shape}"
            
            # Verify data type
            assert coords.dtype == np.float32, f"Coordinates not float32 for study {study_id}"
            
            print(f"✅ Study {study_id}: {coords.shape} coordinates retrieved successfully")
    
    def test_coordinate_format_validation(self, cache):
        """SUCCESS CRITERION: Coordinate format validation (x,y,z tuples)"""
        # Test multiple studies to ensure consistent format
        test_studies = cache.get_random_studies(10)
        
        for study_id in test_studies:
            coords = cache.get_coordinates(study_id)
            
            # Verify (x,y,z) tuple format
            assert coords.shape[1] == 3, f"Not (x,y,z) format for study {study_id}"
            
            # Verify reasonable brain coordinate ranges (MNI152)
            x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
            
            # MNI152 coordinate ranges
            assert np.all(x_coords >= -90) and np.all(x_coords <= 90), f"X coordinates out of range for study {study_id}"
            assert np.all(y_coords >= -126) and np.all(y_coords <= 90), f"Y coordinates out of range for study {study_id}"
            assert np.all(z_coords >= -72) and np.all(z_coords <= 108), f"Z coordinates out of range for study {study_id}"
            
            # Verify no NaN or infinite values
            assert np.all(np.isfinite(coords)), f"Non-finite coordinates found for study {study_id}"
            
            print(f"✅ Study {study_id}: coordinate format validated")
    
    def test_variable_length_coordinate_lists(self, cache):
        """SUCCESS CRITERION: Variable-length coordinate lists properly handled"""
        # Find studies with different numbers of coordinates
        coord_counts = {}
        
        with h5py.File(cache.cache_file, 'r') as hf:
            coord_group = hf['coordinates']
            
            # Sample studies to check variability
            sample_studies = cache.get_random_studies(20)
            
            for study_id in sample_studies:
                coords = coord_group[study_id]
                coord_counts[study_id] = coords.shape[0]
        
        # Verify we have variable lengths
        counts = list(coord_counts.values())
        assert len(set(counts)) > 1, "All studies have same number of coordinates (no variability)"
        
        # Verify we can handle both small and large coordinate sets
        min_coords = min(counts)
        max_coords = max(counts)
        
        print(f"Coordinate count range: {min_coords} - {max_coords}")
        assert min_coords >= 1, "Found study with no coordinates"
        assert max_coords > min_coords, "No variability in coordinate counts"
        
        # Test retrieval of studies with different sizes
        min_study = min(coord_counts.keys(), key=lambda k: coord_counts[k])
        max_study = max(coord_counts.keys(), key=lambda k: coord_counts[k])
        
        min_coords_data = cache.get_coordinates(min_study)
        max_coords_data = cache.get_coordinates(max_study)
        
        assert min_coords_data.shape[0] == min_coords, "Min coordinate study size mismatch"
        assert max_coords_data.shape[0] == max_coords, "Max coordinate study size mismatch"
        
        print(f"✅ Variable-length handling verified: {min_coords} to {max_coords} coordinates per study")
    
    def test_cache_utility_functions(self, cache):
        """Additional test for cache utility functions"""
        # Test batch loading
        test_studies = cache.get_random_studies(3)
        batch_data = cache.batch_load_coordinates(test_studies)
        
        assert len(batch_data) == len(test_studies), "Batch loading returned wrong number of studies"
        
        for study_id in test_studies:
            assert study_id in batch_data, f"Study {study_id} missing from batch data"
            assert batch_data[study_id].shape[1] == 3, f"Batch data wrong shape for study {study_id}"
        
        # Test metadata access
        if cache.has_metadata:
            metadata = cache.get_metadata(test_studies[0])
            assert metadata is not None, "Metadata should be available"
            assert 'stat_values' in metadata, "Missing stat_values in metadata"
            assert 'sample_sizes' in metadata, "Missing sample_sizes in metadata"
            assert 'num_coordinates' in metadata, "Missing num_coordinates in metadata"
        
        # Test contains operation
        assert test_studies[0] in cache, "Study should be in cache"
        assert "nonexistent_study" not in cache, "Nonexistent study should not be in cache"
        
        print("✅ Cache utility functions working correctly")
    
    def test_convenience_functions(self, cache_file):
        """Test convenience functions for quick access"""
        # Test load_pointcloud_cache
        cache = load_pointcloud_cache(cache_file)
        assert len(cache) > 0, "Cache should contain studies"
        
        # Test get_study_coordinates
        study_id = cache.get_random_studies(1)[0]
        coords = get_study_coordinates(study_id, cache_file)
        assert coords.shape[1] == 3, "Convenience function should return (N, 3) coordinates"
        
        print("✅ Convenience functions working correctly")

def test_cache_creation_stats():
    """Test that cache creation statistics are reasonable"""
    stats_file = "data/processed/pointcloud_cache_subset_1k_creation_stats.json"
    
    if os.path.exists(stats_file):
        import json
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        assert stats['total_studies'] > 900, f"Expected ~1000 studies, got {stats['total_studies']}"
        assert stats['total_coordinates'] > 30000, f"Expected ~30k+ coordinates, got {stats['total_coordinates']}"
        assert stats['mean_coords_per_study'] > 20, f"Expected mean ~30-40 coords/study, got {stats['mean_coords_per_study']}"
        assert stats['validation_failures'] < 50, f"Too many validation failures: {stats['validation_failures']}"
        
        print(f"✅ Cache statistics validated: {stats['total_studies']} studies, {stats['total_coordinates']} coordinates")

if __name__ == '__main__':
    # Run tests if called directly
    import subprocess
    import sys
    
    result = subprocess.run([sys.executable, '-m', 'pytest', __file__, '-v'], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    sys.exit(result.returncode)