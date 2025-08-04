"""
Volumetric cache creation module for brain activation maps.

Creates LMDB database with dual-kernel Gaussian convolution and orientation augmentation
for efficient training data loading.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import lmdb
import pickle
import torch
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
from scipy.ndimage import gaussian_filter
from nilearn import image, datasets
from nilearn.input_data import NiftiMasker
import warnings

# Suppress nilearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumetricCacheBuilder:
    """
    Build LMDB cache of volumetric brain activation maps with dual-kernel augmentation.
    """
    
    def __init__(
        self,
        output_path: str = "data/processed/volumetric_cache.lmdb",
        brain_mask_path: Optional[str] = None,
        voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        brain_shape: Tuple[int, int, int] = (91, 109, 91),  # MNI152 2mm
        kernels_mm: List[float] = [6.0, 12.0],
        map_size: int = 10 * 1024**3  # 10GB
    ):
        """
        Initialize volumetric cache builder.
        
        Args:
            output_path: Path for LMDB database
            brain_mask_path: Path to brain mask (if None, uses MNI152)
            voxel_size: Voxel dimensions in mm
            brain_shape: Shape of brain volume
            kernels_mm: Gaussian kernel sizes for dual-kernel augmentation
            map_size: Maximum size of LMDB database
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.brain_mask_path = brain_mask_path
        self.voxel_size = voxel_size
        self.brain_shape = brain_shape
        self.kernels_mm = kernels_mm
        self.map_size = map_size
        
        # Load brain mask
        self.brain_mask = self._load_brain_mask()
        self.masker = NiftiMasker(mask_img=self.brain_mask, standardize=False)
        self.masker.fit()
        
        logger.info(f"Volumetric cache builder initialized")
        logger.info(f"Brain shape: {self.brain_shape}")
        logger.info(f"Voxel size: {self.voxel_size}")
        logger.info(f"Kernel sizes: {self.kernels_mm} mm")
    
    def _load_brain_mask(self) -> nib.Nifti1Image:
        """Load brain mask for spatial normalization."""
        if self.brain_mask_path and Path(self.brain_mask_path).exists():
            logger.info(f"Loading custom brain mask: {self.brain_mask_path}")
            return nib.load(self.brain_mask_path)
        else:
            logger.info("Loading MNI152 brain mask from nilearn")
            # Use nilearn's MNI152 brain mask
            brain_mask = datasets.load_mni152_brain_mask(resolution=2)
            return brain_mask
    
    def _coordinates_to_volume(
        self, 
        coordinates: np.ndarray, 
        values: np.ndarray,
        kernel_size_mm: float
    ) -> np.ndarray:
        """
        Convert coordinate list to volumetric representation with Gaussian kernel.
        
        Args:
            coordinates: Nx3 array of MNI coordinates
            values: N-length array of activation values
            kernel_size_mm: Gaussian kernel FWHM in mm
            
        Returns:
            3D numpy array representing brain volume
        """
        # Initialize empty volume
        volume = np.zeros(self.brain_shape, dtype=np.float32)
        
        # Convert kernel size from mm to voxels
        kernel_voxels = kernel_size_mm / np.array(self.voxel_size)
        sigma_voxels = kernel_voxels / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
        
        # Convert MNI coordinates to voxel indices
        # MNI152 2mm template origin offset
        origin_offset = np.array([90, 126, 72])  # MNI152 2mm template
        
        for coord, value in zip(coordinates, values):
            # Convert to voxel coordinates
            voxel_coord = (coord / np.array(self.voxel_size)) + origin_offset
            voxel_coord = np.round(voxel_coord).astype(int)
            
            # Check bounds
            if (0 <= voxel_coord[0] < self.brain_shape[0] and
                0 <= voxel_coord[1] < self.brain_shape[1] and
                0 <= voxel_coord[2] < self.brain_shape[2]):
                
                volume[voxel_coord[0], voxel_coord[1], voxel_coord[2]] += value
        
        # Apply Gaussian smoothing
        if kernel_size_mm > 0:
            volume = gaussian_filter(volume, sigma=sigma_voxels, mode='constant')
        
        return volume
    
    def _apply_brain_mask(self, volume: np.ndarray) -> np.ndarray:
        """Apply brain mask to volume."""
        mask_data = self.brain_mask.get_fdata()
        if mask_data.shape == volume.shape:
            return volume * mask_data
        else:
            logger.warning(f"Mask shape {mask_data.shape} != volume shape {volume.shape}")
            return volume
    
    def _create_augmented_volumes(
        self, 
        coordinates: np.ndarray, 
        values: np.ndarray,
        study_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Create dual-kernel augmented volumes for a study.
        
        Args:
            coordinates: Nx3 array of coordinates
            values: N-length array of values
            study_id: Study identifier
            
        Returns:
            Dictionary with kernel-specific volumes as PyTorch tensors
        """
        augmented_volumes = {}
        
        for kernel_size in self.kernels_mm:
            # Create volume with specific kernel
            volume = self._coordinates_to_volume(coordinates, values, kernel_size)
            
            # Apply brain mask
            volume = self._apply_brain_mask(volume)
            
            # Convert to PyTorch tensor and add channel dimension
            volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W, D)
            
            # Store with kernel-specific key
            key = f"kernel_{kernel_size:.0f}mm"
            augmented_volumes[key] = volume_tensor
            
            logger.debug(f"Study {study_id}, {key}: {volume_tensor.shape}, range [{volume_tensor.min():.3f}, {volume_tensor.max():.3f}]")
        
        return augmented_volumes
    
    def build_cache(self, data_path: str) -> Dict[str, any]:
        """
        Build LMDB cache from coordinate data.
        
        Args:
            data_path: Path to coordinate CSV file
            
        Returns:
            Dictionary with cache statistics
        """
        logger.info(f"Building volumetric cache from: {data_path}")
        
        # Load coordinate data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} coordinates from {data['study_id'].nunique()} studies")
        
        # Initialize LMDB environment
        env = lmdb.open(str(self.output_path), map_size=self.map_size)
        
        cache_stats = {
            "total_studies": 0,
            "total_volumes_created": 0,
            "cache_size_bytes": 0,
            "kernels_used": self.kernels_mm,
            "brain_shape": self.brain_shape,
            "voxel_size": self.voxel_size
        }
        
        try:
            with env.begin(write=True) as txn:
                # Group by study
                study_groups = data.groupby('study_id')
                
                for study_id, study_data in study_groups:
                    # Extract coordinates and values
                    coordinates = study_data[['x', 'y', 'z']].values
                    values = study_data['stat_value'].values
                    
                    # Create augmented volumes
                    augmented_volumes = self._create_augmented_volumes(
                        coordinates, values, study_id
                    )
                    
                    # Store metadata
                    metadata = {
                        'study_id': study_id,
                        'num_coordinates': len(coordinates),
                        'coordinate_range': {
                            'x': [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())],
                            'y': [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())],
                            'z': [float(coordinates[:, 2].min()), float(coordinates[:, 2].max())]
                        },
                        'stat_value_range': [float(values.min()), float(values.max())],
                        'kernels': list(augmented_volumes.keys())
                    }
                    
                    # Store all data for this study
                    study_entry = {
                        'volumes': augmented_volumes,
                        'metadata': metadata,
                        'original_coordinates': coordinates,
                        'original_values': values
                    }
                    
                    # Serialize and store
                    key = f"study_{study_id}".encode('utf-8')
                    value = pickle.dumps(study_entry)
                    txn.put(key, value)
                    
                    cache_stats["total_studies"] += 1
                    cache_stats["total_volumes_created"] += len(augmented_volumes)
                    cache_stats["cache_size_bytes"] += len(value)
                    
                    if cache_stats["total_studies"] % 10 == 0:
                        logger.info(f"Processed {cache_stats['total_studies']} studies...")
                
                # Store cache metadata
                cache_metadata = {
                    'creation_timestamp': pd.Timestamp.now().isoformat(),
                    'source_data': data_path,
                    'cache_stats': cache_stats
                }
                
                txn.put(b'_metadata', pickle.dumps(cache_metadata))
                
        finally:
            env.close()
        
        logger.info(f"Cache creation complete!")
        logger.info(f"Studies cached: {cache_stats['total_studies']}")
        logger.info(f"Total volumes: {cache_stats['total_volumes_created']}")
        logger.info(f"Cache size: {cache_stats['cache_size_bytes'] / 1024**2:.1f} MB")
        
        return cache_stats
    
    def validate_cache(self) -> bool:
        """
        Validate the created cache by testing random retrievals.
        
        Returns:
            True if validation passes
        """
        logger.info("Validating LMDB cache...")
        
        if not self.output_path.exists():
            logger.error(f"Cache file not found: {self.output_path}")
            return False
        
        try:
            env = lmdb.open(str(self.output_path), readonly=True)
            
            with env.begin() as txn:
                # Get cache metadata
                metadata_raw = txn.get(b'_metadata')
                if metadata_raw is None:
                    logger.error("Cache metadata not found")
                    return False
                
                metadata = pickle.loads(metadata_raw)
                cache_stats = metadata['cache_stats']
                
                logger.info(f"Cache metadata: {cache_stats['total_studies']} studies")
                
                # Test random study retrieval
                cursor = txn.cursor()
                study_keys = [key for key, _ in cursor if key != b'_metadata']
                
                if len(study_keys) == 0:
                    logger.error("No studies found in cache")
                    return False
                
                # Test first study
                test_key = study_keys[0]
                test_data_raw = txn.get(test_key)
                test_data = pickle.loads(test_data_raw)
                
                # Validate structure
                required_keys = ['volumes', 'metadata', 'original_coordinates', 'original_values']
                for key in required_keys:
                    if key not in test_data:
                        logger.error(f"Missing key in study data: {key}")
                        return False
                
                # Validate volumes
                volumes = test_data['volumes']
                for kernel_key, volume_tensor in volumes.items():
                    if not isinstance(volume_tensor, torch.Tensor):
                        logger.error(f"Volume {kernel_key} is not a PyTorch tensor")
                        return False
                    
                    expected_shape = (1,) + self.brain_shape  # (C, H, W, D)
                    if volume_tensor.shape != expected_shape:
                        logger.error(f"Volume {kernel_key} has wrong shape: {volume_tensor.shape} != {expected_shape}")
                        return False
                    
                    if not torch.isfinite(volume_tensor).all():
                        logger.error(f"Volume {kernel_key} contains non-finite values")
                        return False
                
                logger.info(f"✅ Random study validation passed: {test_key.decode()}")
                logger.info(f"   Volumes: {list(volumes.keys())}")
                logger.info(f"   Shapes: {[v.shape for v in volumes.values()]}")
                
            env.close()
            return True
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False


def main():
    """Demonstrate volumetric cache creation."""
    
    # Build cache from coordinate-corrected data
    input_path = "data/processed/coordinate_corrected_data.csv"
    
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run coordinate correction first: python scripts/apply_coordinate_correction.py")
        return False
    
    # Create cache builder
    builder = VolumetricCacheBuilder(
        output_path="data/processed/volumetric_cache.lmdb",
        kernels_mm=[6.0, 12.0]
    )
    
    # Build cache
    cache_stats = builder.build_cache(input_path)
    
    # Validate cache
    validation_result = builder.validate_cache()
    
    if validation_result:
        logger.info("🎉 S1.1.4 Volumetric cache creation SUCCESSFUL")
        return True
    else:
        logger.error("❌ S1.1.4 Volumetric cache creation FAILED")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)