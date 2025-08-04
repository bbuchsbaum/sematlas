"""
Unit tests for coordinate space transformation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.coordinate_transform import (
    tal2icbm_transform,
    calculate_rmsd,
    correct_coordinate_space,
    validate_transformation
)


class TestCoordinateTransform(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create test DataFrame with mixed coordinate spaces
        self.test_data = pd.DataFrame({
            'study_id': ['study_tal_001', 'study_tal_001', 'study_mni_001', 'study_mni_001'],
            'x': [10.0, -10.0, 15.0, -15.0],
            'y': [20.0, -20.0, 25.0, -25.0], 
            'z': [30.0, -30.0, 35.0, -35.0],
            'space': ['Talairach', 'Talairach', 'MNI', 'MNI'],
            'stat_value': [2.5, -2.5, 3.0, -3.0]
        })
        
    def test_tal2icbm_transform_basic(self):
        """Test basic Talairach to MNI transformation"""
        x, y, z = tal2icbm_transform(0.0, 0.0, 0.0)
        
        # Should return valid coordinates
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float) 
        self.assertIsInstance(z, float)
        
        # Should be reasonable values (not inf or nan)
        self.assertTrue(np.isfinite(x))
        self.assertTrue(np.isfinite(y))
        self.assertTrue(np.isfinite(z))
        
    def test_tal2icbm_transform_known_point(self):
        """Test transformation of known coordinate"""
        # Test origin transformation
        x, y, z = tal2icbm_transform(0.0, 0.0, 0.0)
        
        # The origin should transform to approximately (0, -4, -5)
        # Using loose tolerance for simplified transformation
        self.assertAlmostEqual(x, 0.0, delta=1.0)
        self.assertAlmostEqual(y, -4.0, delta=2.0)
        self.assertAlmostEqual(z, -5.0, delta=2.0)
        
    def test_calculate_rmsd(self):
        """Test RMSD calculation"""
        original = np.array([[0, 0, 0], [10, 10, 10]])
        transformed = np.array([[1, 1, 1], [11, 11, 11]])
        
        rmsd = calculate_rmsd(original, transformed)
        expected_rmsd = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        
        self.assertAlmostEqual(rmsd, expected_rmsd, places=5)
        
    def test_calculate_rmsd_empty(self):
        """Test RMSD with empty arrays"""
        empty_array = np.array([]).reshape(0, 3)
        rmsd = calculate_rmsd(empty_array, empty_array)
        self.assertEqual(rmsd, 0.0)
        
    def test_correct_coordinate_space(self):
        """Test coordinate space correction function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_mismatch_log.json")
            
            # Run coordinate correction
            corrected_data = correct_coordinate_space(
                self.test_data.copy(), 
                output_log_path=log_path,
                rmsd_threshold=4.0
            )
            
            # Check that all coordinates are now in MNI space
            self.assertTrue(all(corrected_data['space'] == 'MNI'))
            
            # Check that MNI coordinates are unchanged
            mni_mask = self.test_data['space'] == 'MNI'
            original_mni = self.test_data[mni_mask][['x', 'y', 'z']].values
            corrected_mni = corrected_data[corrected_data['study_id'] == 'study_mni_001'][['x', 'y', 'z']].values
            
            np.testing.assert_array_almost_equal(original_mni, corrected_mni, decimal=6)
            
            # Check that Talairach coordinates were transformed
            tal_mask = self.test_data['space'] == 'Talairach'
            original_tal = self.test_data[tal_mask][['x', 'y', 'z']].values
            corrected_tal = corrected_data[corrected_data['study_id'] == 'study_tal_001'][['x', 'y', 'z']].values
            
            # Should be different after transformation
            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal(original_tal, corrected_tal, decimal=1)
            
            # Check log file was created
            self.assertTrue(os.path.exists(log_path))
            
            # Validate log content
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                
            self.assertIn('rmsd_threshold_mm', log_data)
            self.assertIn('transformation_stats', log_data)
            self.assertEqual(log_data['transformation_stats']['total_studies'], 2)
            self.assertEqual(log_data['transformation_stats']['talairach_studies_transformed'], 1)
            self.assertEqual(log_data['transformation_stats']['mni_studies_unchanged'], 1)
            
    def test_mni_coordinates_unchanged(self):
        """Test that MNI coordinates remain unchanged"""
        mni_only_data = pd.DataFrame({
            'study_id': ['study_mni_001'],
            'x': [42.0],
            'y': [-18.0],
            'z': [24.0],
            'space': ['MNI'],
            'stat_value': [3.5]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_log.json")
            corrected_data = correct_coordinate_space(mni_only_data, output_log_path=log_path)
            
            # Coordinates should be exactly the same
            pd.testing.assert_frame_equal(mni_only_data, corrected_data)
            
    def test_validate_transformation(self):
        """Test the validation function"""
        # This should return True or False, not raise an exception
        result = validate_transformation()
        self.assertIsInstance(result, bool)
        
    def test_transformation_preserves_other_columns(self):
        """Test that transformation preserves non-coordinate columns"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "test_log.json")
            corrected_data = correct_coordinate_space(self.test_data.copy(), output_log_path=log_path)
            
            # Check that non-coordinate columns are preserved
            self.assertTrue(all(corrected_data['study_id'] == self.test_data['study_id']))
            self.assertTrue(all(corrected_data['stat_value'] == self.test_data['stat_value']))
            
            # Only 'space' column should be changed to 'MNI'
            self.assertTrue(all(corrected_data['space'] == 'MNI'))


if __name__ == '__main__':
    unittest.main()