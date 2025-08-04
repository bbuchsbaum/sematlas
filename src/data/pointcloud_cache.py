"""
Point-Cloud Cache Utilities

Provides easy access to HDF5 point-cloud cache created in Sprint 3 Epic 1.
Includes utilities for loading coordinates, metadata, and batch processing.
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path

class PointCloudCache:
    """
    Utility class for accessing HDF5 point-cloud cache
    
    Provides methods to:
    - Load coordinates for specific studies
    - Batch load multiple studies
    - Access metadata (stat_values, sample_sizes)
    - Get cache statistics and information
    """
    
    def __init__(self, cache_file: str):
        """
        Initialize point-cloud cache accessor
        
        Args:
            cache_file: Path to HDF5 cache file
        """
        self.cache_file = Path(cache_file)
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        # Load cache metadata
        with h5py.File(self.cache_file, 'r') as hf:
            self.has_metadata = 'metadata' in hf
            self.creation_stats = json.loads(hf.attrs['creation_stats'])
            self.source_file = hf.attrs['source_file']
            
            # Get list of available study IDs
            self.study_ids = list(hf['coordinates'].keys())
    
    def get_coordinates(self, study_id: Union[str, int]) -> np.ndarray:
        """
        Get coordinates for a specific study
        
        Args:
            study_id: Study ID (string or integer)
            
        Returns:
            Coordinate array of shape (N, 3) where N is number of coordinates
        """
        study_id = str(study_id)
        
        with h5py.File(self.cache_file, 'r') as hf:
            if study_id not in hf['coordinates']:
                raise KeyError(f"Study {study_id} not found in cache")
            
            return hf['coordinates'][study_id][:]
    
    def get_metadata(self, study_id: Union[str, int]) -> Optional[Dict]:
        """
        Get metadata for a specific study
        
        Args:
            study_id: Study ID (string or integer)
            
        Returns:
            Dictionary with metadata or None if metadata not available
        """
        if not self.has_metadata:
            return None
        
        study_id = str(study_id)
        
        with h5py.File(self.cache_file, 'r') as hf:
            if study_id not in hf['metadata']:
                return None
            
            meta_group = hf['metadata'][study_id]
            return {
                'stat_values': meta_group['stat_values'][:],
                'sample_sizes': meta_group['sample_sizes'][:],
                'num_coordinates': meta_group['num_coordinates'][()]
            }
    
    def get_study_data(self, study_id: Union[str, int]) -> Dict:
        """
        Get both coordinates and metadata for a study
        
        Args:
            study_id: Study ID (string or integer)
            
        Returns:
            Dictionary with 'coordinates' and optionally 'metadata'
        """
        data = {
            'coordinates': self.get_coordinates(study_id)
        }
        
        metadata = self.get_metadata(study_id)
        if metadata is not None:
            data['metadata'] = metadata
        
        return data
    
    def batch_load_coordinates(self, study_ids: List[Union[str, int]]) -> Dict[str, np.ndarray]:
        """
        Load coordinates for multiple studies efficiently
        
        Args:
            study_ids: List of study IDs
            
        Returns:
            Dictionary mapping study_id -> coordinate array
        """
        result = {}
        
        with h5py.File(self.cache_file, 'r') as hf:
            coord_group = hf['coordinates']
            
            for study_id in study_ids:
                study_id = str(study_id)
                if study_id in coord_group:
                    result[study_id] = coord_group[study_id][:]
        
        return result
    
    def get_random_studies(self, n: int = 5) -> List[str]:
        """
        Get N random study IDs from the cache
        
        Args:
            n: Number of studies to sample
            
        Returns:
            List of study IDs
        """
        return np.random.choice(self.study_ids, min(n, len(self.study_ids)), replace=False).tolist()
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the cache
        
        Returns:
            Dictionary with cache statistics and metadata
        """
        return {
            'cache_file': str(self.cache_file),
            'source_file': self.source_file,
            'has_metadata': self.has_metadata,
            'num_studies': len(self.study_ids),
            'creation_stats': self.creation_stats
        }
    
    def filter_studies_by_coordinate_count(self, min_coords: int = None, max_coords: int = None) -> List[str]:
        """
        Filter studies by number of coordinates
        
        Args:
            min_coords: Minimum number of coordinates (inclusive)
            max_coords: Maximum number of coordinates (inclusive)
            
        Returns:
            List of study IDs meeting the criteria
        """
        filtered_studies = []
        
        with h5py.File(self.cache_file, 'r') as hf:
            coord_group = hf['coordinates']
            
            for study_id in self.study_ids:
                coords = coord_group[study_id]
                num_coords = coords.shape[0]
                
                if min_coords is not None and num_coords < min_coords:
                    continue
                if max_coords is not None and num_coords > max_coords:
                    continue
                
                filtered_studies.append(study_id)
        
        return filtered_studies
    
    def __len__(self) -> int:
        """Return number of studies in cache"""
        return len(self.study_ids)
    
    def __contains__(self, study_id: Union[str, int]) -> bool:
        """Check if study ID exists in cache"""
        return str(study_id) in self.study_ids
    
    def __iter__(self):
        """Iterate over study IDs"""
        return iter(self.study_ids)

# Convenience functions for quick access
def load_pointcloud_cache(cache_file: str = 'data/processed/pointcloud_cache_subset_1k.h5') -> PointCloudCache:
    """
    Load point-cloud cache with default development file
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        PointCloudCache instance
    """
    return PointCloudCache(cache_file)

def get_study_coordinates(study_id: Union[str, int], 
                         cache_file: str = 'data/processed/pointcloud_cache_subset_1k.h5') -> np.ndarray:
    """
    Quick function to get coordinates for a single study
    
    Args:
        study_id: Study ID
        cache_file: Path to cache file
        
    Returns:
        Coordinate array of shape (N, 3)
    """
    cache = PointCloudCache(cache_file)
    return cache.get_coordinates(study_id)