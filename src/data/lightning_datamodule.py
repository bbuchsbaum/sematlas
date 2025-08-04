"""
PyTorch Lightning DataModule for brain activation data.

Handles loading volumetric brain data from LMDB cache with dual-kernel
augmentation for training the VAE models.
"""

import os
import random
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F

from ..models.metadata_imputation import create_default_metadata_config

import pytorch_lightning as pl

import lmdb


logger = logging.getLogger(__name__)


class BrainVolumeDataset(Dataset):
    """Dataset for loading brain activation volumes from LMDB cache with metadata conditioning."""
    
    def __init__(
        self,
        lmdb_path: str,
        split_file: str,
        kernel_selection: str = "random",
        transform: Optional[callable] = None,
        metadata_config: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ):
        """
        Initialize brain volume dataset with metadata conditioning.
        
        Args:
            lmdb_path: Path to LMDB cache directory
            split_file: Path to CSV file with study IDs for this split
            kernel_selection: Strategy for kernel selection ("random", "6mm", "12mm")
            transform: Optional transform to apply to volumes
            metadata_config: Configuration for metadata fields
            include_metadata: Whether to include metadata in batches
        """
        self.lmdb_path = lmdb_path
        self.kernel_selection = kernel_selection
        self.transform = transform
        self.include_metadata = include_metadata
        
        # Initialize metadata configuration
        if metadata_config is None:
            self.metadata_config = create_default_metadata_config()
        else:
            self.metadata_config = metadata_config
        
        # Load study IDs from split file
        if os.path.exists(split_file):
            split_df = pd.read_csv(split_file)
            self.study_ids = split_df['study_id'].tolist()
            # Also load metadata from split file if available
            self.study_metadata_df = split_df
        else:
            raise FileNotFoundError(f"Split file {split_file} not found. Cannot load dataset without proper data splits.")
        
        logger.info(f"Initialized dataset with {len(self.study_ids)} studies")
        logger.info(f"Kernel selection strategy: {kernel_selection}")
        logger.info(f"Metadata conditioning: {include_metadata}")
        logger.info(f"Metadata fields: {list(self.metadata_config.keys())}")
    
    def __len__(self) -> int:
        return len(self.study_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a brain volume and its metadata for conditioning."""
        study_id = self.study_ids[idx]
        
        # Try to load from LMDB cache
        try:
            with lmdb.open(self.lmdb_path, readonly=True) as env:
                with env.begin() as txn:
                    data_bytes = txn.get(str(study_id).encode('utf-8'))
                    
                    if data_bytes is None:
                        raise KeyError(f"Study ID {study_id} not found in cache")
                    
                    data = pickle.loads(data_bytes)
                    
        except Exception as e:
            logger.error(f"Error loading {study_id} from LMDB: {e}")
            raise RuntimeError(f"Failed to load study {study_id} from LMDB cache") from e
        
        # Select kernel based on strategy
        if self.kernel_selection == "random":
            kernel_choice = random.choice(["6mm", "12mm"])
        elif self.kernel_selection in ["6mm", "12mm"]:
            kernel_choice = self.kernel_selection
        else:
            raise ValueError(f"Invalid kernel selection: {self.kernel_selection}")
        
        # Get the appropriate volume
        volume_key = f"kernel_{kernel_choice}"
        volume = data['volumes'][volume_key]
        
        # Ensure correct shape and type
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()
        else:
            volume = volume.float()
        
        # Add channel dimension if needed
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # (1, 91, 109, 91)
        
        # Apply transforms if provided
        if self.transform:
            volume = self.transform(volume)
        
        # Prepare result dict
        result = {
            'volume': volume,  # Expected by VAE Lightning module
            'study_id': study_id,
            'kernel_used': kernel_choice,
        }
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = self._extract_metadata(study_id, data.get('metadata', {}))
            result['metadata'] = metadata
        
        return result
    
    def _extract_metadata(self, study_id: str, raw_metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract and format metadata for conditioning."""
        metadata = {}
        
        # Get study metadata from DataFrame
        study_row = self.study_metadata_df[self.study_metadata_df['study_id'] == study_id]
        
        if len(study_row) == 0:
            # Study not found in metadata - use default values
            study_data = {
                'sample_size': 50.0,
                'study_year': 2010.0,
                'task_category': 0
            }
        else:
            study_data = study_row.iloc[0].to_dict()
        
        # Process each metadata field according to configuration
        for field_name, config in self.metadata_config.items():
            if config['type'] == 'continuous':
                # Extract continuous value
                if field_name == 'sample_size':
                    value = float(study_data.get('sample_size', raw_metadata.get('n_subjects', 50.0)))
                elif field_name == 'study_year':
                    value = float(study_data.get('study_year', 2010.0))
                elif field_name == 'statistical_threshold':
                    value = float(raw_metadata.get('statistical_threshold', 3.0))
                else:
                    # Use prior mean as default
                    value = float(config.get('prior_mean', 0.0))
                
                # Convert to tensor with correct shape
                metadata[field_name] = torch.tensor([value], dtype=torch.float32)
                
            elif config['type'] == 'categorical':
                # Extract categorical value
                if field_name == 'task_category':
                    category_idx = int(study_data.get('task_category', 0)) % config['dim']
                elif field_name == 'scanner_field_strength':
                    # Map field strength to index (1.5T=0, 3T=1, 7T=2)
                    field_strength = raw_metadata.get('scanner_field_strength', '3T')
                    if '1.5' in str(field_strength):
                        category_idx = 0
                    elif '7' in str(field_strength):
                        category_idx = 2
                    else:  # Default to 3T
                        category_idx = 1
                else:
                    category_idx = 0  # Default category
                
                # Convert to one-hot tensor
                one_hot = torch.zeros(config['dim'], dtype=torch.float32)
                one_hot[category_idx] = 1.0
                metadata[field_name] = one_hot
        
        return metadata
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function to handle metadata batching."""
        collated = {
            'image': torch.stack([item['volume'] for item in batch]),
            'volume': torch.stack([item['volume'] for item in batch]),  # Keep both for compatibility
            'study_id': [item['study_id'] for item in batch],
            'kernel_used': [item['kernel_used'] for item in batch]
        }
        
        # Handle metadata if present
        if self.include_metadata and 'metadata' in batch[0]:
            metadata_batch = {}
            
            # Collate each metadata field
            for field_name in self.metadata_config.keys():
                field_values = [item['metadata'][field_name] for item in batch]
                metadata_batch[field_name] = torch.stack(field_values)
            
            collated['metadata'] = metadata_batch
        
        return collated


class BrainVolumeDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for brain activation volumes."""
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        lmdb_cache: str = "data/processed/volumetric_cache",
        batch_size: int = 4,
        num_workers: int = 4,
        kernel_selection: str = "random",
        pin_memory: bool = True,
        include_metadata: bool = True,
        metadata_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the DataModule with metadata conditioning support.
        
        Args:
            data_dir: Directory containing split CSV files
            lmdb_cache: Path to LMDB cache directory
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            kernel_selection: Strategy for kernel selection ("random", "6mm", "12mm")
            pin_memory: Whether to pin memory for faster GPU transfer
            include_metadata: Whether to include metadata in batches
            metadata_config: Configuration for metadata fields
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.lmdb_cache = lmdb_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kernel_selection = kernel_selection
        self.pin_memory = pin_memory
        self.include_metadata = include_metadata
        
        # Initialize metadata configuration
        if metadata_config is None:
            self.metadata_config = create_default_metadata_config()
        else:
            self.metadata_config = metadata_config
        
        # Dataset splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Split file paths
        self.train_split_file = self.data_dir / "train_split.csv"
        self.val_split_file = self.data_dir / "val_split.csv"
        self.test_split_file = self.data_dir / "test_split.csv"
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing."""
        
        # Check if LMDB cache exists
        if not os.path.exists(self.lmdb_cache):
            raise FileNotFoundError(f"LMDB cache not found at {self.lmdb_cache}. Please run data pipeline first.")
        
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = BrainVolumeDataset(
                lmdb_path=self.lmdb_cache,
                split_file=str(self.train_split_file),
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
            
            # Validation dataset  
            self.val_dataset = BrainVolumeDataset(
                lmdb_path=self.lmdb_cache,
                split_file=str(self.val_split_file),
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = BrainVolumeDataset(
                lmdb_path=self.lmdb_cache,
                split_file=str(self.test_split_file),
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
        
        logger.info("DataModule setup completed")
        if self.train_dataset:
            logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation samples: {len(self.val_dataset)}")
        if self.test_dataset:
            logger.info(f"Test samples: {len(self.test_dataset)}")
    
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_dataset.collate_fn if self.include_metadata else None
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.val_dataset.collate_fn if self.include_metadata else None
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.test_dataset.collate_fn if self.include_metadata else None
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader (uses test set)."""
        return self.test_dataloader()




def create_brain_datamodule(
    data_dir: str = "data/processed",
    lmdb_cache: str = "data/processed/volumetric_cache",
    batch_size: int = 4,
    num_workers: int = 4,
    kernel_selection: str = "random",
    include_metadata: bool = True,
    metadata_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BrainVolumeDataModule:
    """
    Factory function to create a BrainVolumeDataModule with metadata conditioning.
    
    Args:
        data_dir: Directory containing split CSV files
        lmdb_cache: Path to LMDB cache directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        kernel_selection: Strategy for kernel selection
        include_metadata: Whether to include metadata in batches
        metadata_config: Configuration for metadata fields
        **kwargs: Additional arguments for DataModule
        
    Returns:
        Configured BrainVolumeDataModule
    """
    return BrainVolumeDataModule(
        data_dir=data_dir,
        lmdb_cache=lmdb_cache,
        batch_size=batch_size,
        num_workers=num_workers,
        kernel_selection=kernel_selection,
        include_metadata=include_metadata,
        metadata_config=metadata_config,
        **kwargs
    )


if __name__ == "__main__":
    # Test the DataModule
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create datamodule
    dm = create_brain_datamodule(batch_size=2, num_workers=0)
    
    # Setup for training
    dm.setup("fit")
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    print(f"Train loader created with {len(train_loader)} batches")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Image dtype: {batch['image'].dtype}")
    print(f"Study IDs: {batch['study_id']}")
    print(f"Kernels used: {batch['kernel_used']}")
    
    # Test metadata if included
    if 'metadata' in batch:
        print(f"Metadata keys: {batch['metadata'].keys()}")
        for field_name, field_tensor in batch['metadata'].items():
            print(f"  {field_name}: shape={field_tensor.shape}, dtype={field_tensor.dtype}")
            print(f"    Sample values: {field_tensor[0]}")
    
    # Test val dataloader
    val_loader = dm.val_dataloader()
    print(f"\nVal loader created with {len(val_loader)} batches")
    
    val_batch = next(iter(val_loader))
    print(f"Val image shape: {val_batch['image'].shape}")
    
    # Test metadata consistency across train/val
    if 'metadata' in val_batch:
        print(f"Val metadata keys: {val_batch['metadata'].keys()}")
        print("Metadata fields match train loader:", 
              batch['metadata'].keys() == val_batch['metadata'].keys())
    
    print("\nDataModule test completed successfully!")