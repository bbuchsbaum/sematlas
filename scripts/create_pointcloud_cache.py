#!/usr/bin/env python3
"""
Create HDF5 Point-Cloud Cache for Sprint 3 Epic 1

This script exports deduplicated, coordinate-validated data into an HDF5 file
where each study is stored as a variable-length list of (x,y,z) points.

Requirements:
- Input: neurosynth_subset_1k_coordinates.csv (development) or neurosynth_full_12k_coordinates.csv (production)
- Output: HDF5 file with study_id -> coordinate array mapping
- Coordinate format validation: ensure (x,y,z) tuples are correctly formatted
- Variable-length arrays for different numbers of coordinates per study

SUCCESS CRITERIA (S3.1.1):
- [X] HDF5 file created successfully 
- [X] Random study retrieval returns correct coordinate array
- [X] Coordinate format validation (x,y,z tuples)
- [X] Variable-length coordinate lists properly handled
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def validate_coordinates(coords: np.ndarray) -> bool:
    """Validate coordinate array format and values"""
    if coords.shape[1] != 3:
        return False
    
    # Check for reasonable brain coordinate ranges (MNI152 space)
    x_range = (-90, 90)
    y_range = (-126, 90) 
    z_range = (-72, 108)
    
    x_valid = np.all((coords[:, 0] >= x_range[0]) & (coords[:, 0] <= x_range[1]))
    y_valid = np.all((coords[:, 1] >= y_range[0]) & (coords[:, 1] <= y_range[1]))
    z_valid = np.all((coords[:, 2] >= z_range[0]) & (coords[:, 2] <= z_range[1]))
    
    return x_valid and y_valid and z_valid

def create_pointcloud_cache(input_csv: str, output_hdf5: str, include_metadata: bool = True) -> Dict:
    """
    Create HDF5 point-cloud cache from coordinate CSV file
    
    Args:
        input_csv: Path to coordinate CSV file
        output_hdf5: Path to output HDF5 file
        include_metadata: Whether to include stat_value and sample_size in cache
        
    Returns:
        Dictionary with creation statistics
    """
    print(f"Creating point-cloud cache: {input_csv} -> {output_hdf5}")
    
    # Load coordinate data
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} coordinate entries from {df['study_id'].nunique()} studies")
    
    # Group by study_id to create point clouds
    study_groups = df.groupby('study_id')
    
    # Create output directory
    os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_studies': 0,
        'total_coordinates': 0,
        'min_coords_per_study': float('inf'),
        'max_coords_per_study': 0,
        'mean_coords_per_study': 0,
        'validation_failures': 0,
        'created_timestamp': datetime.now().isoformat()
    }
    
    coordinate_counts = []
    
    # Create HDF5 file
    with h5py.File(output_hdf5, 'w') as hf:
        # Create groups for organization
        coord_group = hf.create_group('coordinates')
        if include_metadata:
            meta_group = hf.create_group('metadata')
        
        # Process each study
        for study_id, group in study_groups:
            # Extract coordinates (x, y, z)
            coords = group[['x', 'y', 'z']].values.astype(np.float32)
            
            # Validate coordinates
            if not validate_coordinates(coords):
                print(f"WARNING: Invalid coordinates for study {study_id}, skipping")
                stats['validation_failures'] += 1
                continue
            
            # Store coordinates as variable-length dataset
            coord_group.create_dataset(str(study_id), data=coords, compression='gzip')
            
            # Store metadata if requested
            if include_metadata:
                metadata = {
                    'stat_values': group['stat_value'].values.astype(np.float32),
                    'sample_sizes': group['sample_size'].values.astype(np.int32),
                    'num_coordinates': len(coords)
                }
                
                meta_subgroup = meta_group.create_group(str(study_id))
                for key, value in metadata.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        # Scalar values don't support compression
                        meta_subgroup.create_dataset(key, data=value)
                    else:
                        # Array values can use compression
                        meta_subgroup.create_dataset(key, data=value, compression='gzip')
            
            # Update statistics
            stats['total_studies'] += 1
            stats['total_coordinates'] += len(coords)
            coordinate_counts.append(len(coords))
            
            if len(coords) < stats['min_coords_per_study']:
                stats['min_coords_per_study'] = len(coords)
            if len(coords) > stats['max_coords_per_study']:
                stats['max_coords_per_study'] = len(coords)
        
        # Finalize statistics
        stats['mean_coords_per_study'] = np.mean(coordinate_counts)
        stats['std_coords_per_study'] = np.std(coordinate_counts)
        
        # Store metadata about the cache
        hf.attrs['creation_stats'] = json.dumps(stats)
        hf.attrs['source_file'] = input_csv
        hf.attrs['include_metadata'] = include_metadata
    
    print(f"✅ Point-cloud cache created successfully!")
    print(f"   Studies: {stats['total_studies']}")
    print(f"   Total coordinates: {stats['total_coordinates']}")
    print(f"   Coordinates per study: {stats['min_coords_per_study']:.0f} - {stats['max_coords_per_study']:.0f} (mean: {stats['mean_coords_per_study']:.1f})")
    print(f"   Validation failures: {stats['validation_failures']}")
    
    return stats

def test_pointcloud_cache(hdf5_file: str, test_study_ids: List[str] = None) -> bool:
    """
    Test point-cloud cache by retrieving random studies
    
    Args:
        hdf5_file: Path to HDF5 cache file
        test_study_ids: Optional list of specific study IDs to test
        
    Returns:
        True if all tests pass
    """
    print(f"Testing point-cloud cache: {hdf5_file}")
    
    if not os.path.exists(hdf5_file):
        print(f"❌ Cache file not found: {hdf5_file}")
        return False
    
    try:
        with h5py.File(hdf5_file, 'r') as hf:
            # Check file structure
            if 'coordinates' not in hf:
                print("❌ Missing 'coordinates' group")
                return False
            
            coord_group = hf['coordinates']
            study_ids = list(coord_group.keys())
            
            if len(study_ids) == 0:
                print("❌ No studies found in cache")
                return False
            
            # Test specific studies or random sample
            if test_study_ids is None:
                # Test 5 random studies
                test_ids = np.random.choice(study_ids, min(5, len(study_ids)), replace=False)
            else:
                test_ids = [sid for sid in test_study_ids if sid in study_ids]
            
            print(f"Testing {len(test_ids)} studies...")
            
            for study_id in test_ids:
                # Retrieve coordinates
                coords = coord_group[study_id][:]
                
                # Validate format
                if coords.ndim != 2 or coords.shape[1] != 3:
                    print(f"❌ Invalid coordinate shape for study {study_id}: {coords.shape}")
                    return False
                
                # Validate coordinate values
                if not validate_coordinates(coords):
                    print(f"❌ Invalid coordinate values for study {study_id}")
                    return False
                
                print(f"   ✅ Study {study_id}: {len(coords)} coordinates, shape {coords.shape}")
            
            # Print cache statistics
            if 'creation_stats' in hf.attrs:
                stats = json.loads(hf.attrs['creation_stats'])
                print(f"Cache statistics:")
                print(f"   Total studies: {stats['total_studies']}")
                print(f"   Total coordinates: {stats['total_coordinates']}")
                print(f"   Mean coordinates per study: {stats['mean_coords_per_study']:.1f}")
        
        print("✅ All point-cloud cache tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def main():
    """Main entry point for point-cloud cache creation"""
    parser = argparse.ArgumentParser(description='Create HDF5 point-cloud cache')
    parser.add_argument('--input', type=str, 
                       default='data/processed/neurosynth_subset_1k_coordinates.csv',
                       help='Input coordinate CSV file')
    parser.add_argument('--output', type=str,
                       default='data/processed/pointcloud_cache_subset_1k.h5',
                       help='Output HDF5 cache file')
    parser.add_argument('--include-metadata', action='store_true', default=True,
                       help='Include stat_value and sample_size in cache')
    parser.add_argument('--test', action='store_true',
                       help='Test the created cache')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing cache, do not create')
    
    args = parser.parse_args()
    
    # Create cache unless test-only mode
    if not args.test_only:
        try:
            stats = create_pointcloud_cache(args.input, args.output, args.include_metadata)
            
            # Save stats to JSON file
            stats_file = args.output.replace('.h5', '_creation_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Creation statistics saved to: {stats_file}")
            
        except Exception as e:
            print(f"❌ Failed to create point-cloud cache: {e}")
            return 1
    
    # Test cache if requested
    if args.test or args.test_only:
        success = test_pointcloud_cache(args.output)
        if not success:
            return 1
    
    return 0

if __name__ == '__main__':
    exit(main())