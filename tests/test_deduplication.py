"""
Unit tests for directional deduplication functionality.

These tests validate the S1.1.2 acceptance criteria:
- Function processes test dataset without errors
- Unit test passes with known input/output pair
- Log file shows >10% deduplication rate (indicating function works)
- Retains contrasts with opposite t-stat signs as distinct
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from data.deduplication import (
    create_deduplication_hash,
    deduplicate_contrasts,
    round_coordinates,
    get_direction_sign
)


class TestDeduplication(unittest.TestCase):
    """Test cases for directional deduplication."""
    
    def setUp(self):
        """Set up test data."""
        # Create test dataset with known duplicates
        self.test_data = pd.DataFrame([
            # Same coordinates, same direction, different sample sizes
            {'study_id': 'study_1', 'x': 10.0, 'y': 20.0, 'z': 30.0, 'stat_value': 3.5, 'sample_size': 25},
            {'study_id': 'study_1', 'x': 10.04, 'y': 20.03, 'z': 30.02, 'stat_value': 2.8, 'sample_size': 40},  # Should keep this one (larger sample)
            
            # Same coordinates, opposite directions (should both be kept)
            {'study_id': 'study_2', 'x': 15.0, 'y': 25.0, 'z': 35.0, 'stat_value': 4.2, 'sample_size': 30},
            {'study_id': 'study_2', 'x': 15.0, 'y': 25.0, 'z': 35.0, 'stat_value': -3.8, 'sample_size': 30},
            
            # Different coordinates (should be kept)
            {'study_id': 'study_3', 'x': 20.0, 'y': 30.0, 'z': 40.0, 'stat_value': 2.1, 'sample_size': 35},
            
            # Different study, same coordinates and direction (should be kept)
            {'study_id': 'study_4', 'x': 10.0, 'y': 20.0, 'z': 30.0, 'stat_value': 3.2, 'sample_size': 50},
        ])
    
    def test_round_coordinates(self):
        """Test coordinate rounding function."""
        # Test with default precision
        result = round_coordinates(10.123, 20.567, 30.891)
        expected = (10.1, 20.6, 30.9)
        self.assertEqual(result, expected)
        
        # Test with zero precision
        result = round_coordinates(10.123, 20.567, 30.891, precision=0)
        expected = (10.0, 21.0, 31.0)
        self.assertEqual(result, expected)
    
    def test_get_direction_sign(self):
        """Test direction sign extraction."""
        self.assertEqual(get_direction_sign(3.5), 'pos')
        self.assertEqual(get_direction_sign(-2.8), 'neg')
        self.assertEqual(get_direction_sign(0.0), 'zero')
    
    def test_create_deduplication_hash(self):
        """Test hash creation for deduplication."""
        # Same coordinates and direction should produce same hash
        hash1 = create_deduplication_hash(10.0, 20.0, 30.0, 3.5)
        hash2 = create_deduplication_hash(10.04, 20.03, 30.02, 2.8)  # Should round to same coords
        self.assertEqual(hash1, hash2)
        
        # Opposite directions should produce different hashes
        hash_pos = create_deduplication_hash(10.0, 20.0, 30.0, 3.5)
        hash_neg = create_deduplication_hash(10.0, 20.0, 30.0, -3.5)
        self.assertNotEqual(hash_pos, hash_neg)
        
        # Different coordinates should produce different hashes
        hash1 = create_deduplication_hash(10.0, 20.0, 30.0, 3.5)
        hash2 = create_deduplication_hash(15.0, 25.0, 35.0, 3.5)
        self.assertNotEqual(hash1, hash2)
    
    def test_deduplicate_contrasts_known_input_output(self):
        """Test deduplication with known input/output pair."""
        result_df, stats = deduplicate_contrasts(self.test_data)
        
        # Should keep 5 rows (remove 1 duplicate within study_1)
        expected_count = 5
        self.assertEqual(len(result_df), expected_count)
        
        # Check that removal rate is > 10% as required
        self.assertGreater(stats['removal_rate'], 0.1)
        
        # Check that the larger sample size was kept for study_1
        study_1_rows = result_df[result_df['study_id'] == 'study_1']
        self.assertEqual(len(study_1_rows), 1)
        self.assertEqual(study_1_rows.iloc[0]['sample_size'], 40)
        
        # Check that both positive and negative contrasts were kept for study_2
        study_2_rows = result_df[result_df['study_id'] == 'study_2']
        self.assertEqual(len(study_2_rows), 2)
        stat_values = set(study_2_rows['stat_value'])
        self.assertTrue(any(x > 0 for x in stat_values))  # Has positive
        self.assertTrue(any(x < 0 for x in stat_values))  # Has negative
    
    def test_opposite_directions_preserved(self):
        """Test that contrasts with opposite t-stat signs are preserved."""
        # Create test data with same coordinates but opposite directions
        opposite_data = pd.DataFrame([
            {'study_id': 'test', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'stat_value': 5.0, 'sample_size': 30},
            {'study_id': 'test', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'stat_value': -5.0, 'sample_size': 30},
        ])
        
        result_df, stats = deduplicate_contrasts(opposite_data)
        
        # Both should be preserved
        self.assertEqual(len(result_df), 2)
        self.assertEqual(stats['removal_rate'], 0.0)
        
        # Check that both directions are present
        stat_values = result_df['stat_value'].tolist()
        self.assertTrue(any(x > 0 for x in stat_values))
        self.assertTrue(any(x < 0 for x in stat_values))
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result_df, stats = deduplicate_contrasts(empty_df)
        
        self.assertTrue(result_df.empty)
        self.assertEqual(stats['original_count'], 0)
        self.assertEqual(stats['deduplicated_count'], 0)
    
    def test_missing_columns_error(self):
        """Test that missing required columns raise an error."""
        invalid_df = pd.DataFrame([{'x': 1, 'y': 2}])  # Missing z, stat_value, study_id
        
        with self.assertRaises(ValueError):
            deduplicate_contrasts(invalid_df)
    
    def test_high_deduplication_rate(self):
        """Test case that ensures >10% deduplication rate."""
        # Create data with many duplicates
        duplicate_data = []
        for i in range(100):
            # Create 10 groups of 10 duplicates each (90% duplication rate expected)
            study_id = f"study_{i // 10}"
            base_coord = (i // 10) * 10
            duplicate_data.append({
                'study_id': study_id,
                'x': base_coord + np.random.normal(0, 0.02),  # Slight noise that rounds to same
                'y': base_coord + np.random.normal(0, 0.02),
                'z': base_coord + np.random.normal(0, 0.02),
                'stat_value': 3.0 + np.random.normal(0, 0.1),  # Same direction
                'sample_size': 20 + i  # Increasing sample sizes
            })
        
        test_df = pd.DataFrame(duplicate_data)
        result_df, stats = deduplicate_contrasts(test_df)
        
        # Should have high deduplication rate
        self.assertGreater(stats['removal_rate'], 0.5)  # Expect >50% removal
        
        # Log the rate for verification
        print(f"High deduplication test - Removal rate: {stats['removal_rate']:.1%}")


def run_tests():
    """Run all deduplication tests."""
    print("Running directional deduplication tests...")
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()