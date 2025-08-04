"""
PyTorch Lightning DataModule for brain activation data - Fixed for multi-worker LMDB access.

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
        
        # LMDB environment and transaction will be initialized per worker
        self.env = None
        self.txn = None
        
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
    
    def _init_lmdb(self):
        """Initialize LMDB environment - called once per worker process."""
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
            self.txn = self.env.begin()
            logger.info(f"Initialized LMDB environment in worker process {os.getpid()}")
    
    def __len__(self) -> int:
        return len(self.study_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a brain volume and its metadata for conditioning."""
        # Initialize LMDB on first access in each worker
        if self.env is None:
            self._init_lmdb()
        
        study_id = self.study_ids[idx]
        
        # Try to load from LMDB cache
        try:
            data_bytes = self.txn.get(str(study_id).encode('utf-8'))
            
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
        kernel_key = f"kernel_{kernel_choice}"
        if kernel_key in data['volumes']:
            volume = data['volumes'][kernel_key]
        else:
            # Fallback to any available kernel
            available_kernels = list(data['volumes'].keys())
            if available_kernels:
                volume = data['volumes'][available_kernels[0]]
                logger.warning(f"Kernel {kernel_key} not found, using {available_kernels[0]}")
            else:
                raise KeyError(f"No volume data found for study {study_id}")
        
        # Apply transform if provided
        if self.transform:
            volume = self.transform(volume)
        
        # Prepare result
        result = {
            'volume': volume,
            'study_id': study_id,
            'kernel': kernel_choice
        }
        
        # Add metadata if requested
        if self.include_metadata:
            # Get metadata from the split dataframe
            study_row = self.study_metadata_df[self.study_metadata_df['study_id'] == study_id]
            
            if len(study_row) > 0:
                study_row = study_row.iloc[0]
                metadata_dict = {}
                
                for field, config in self.metadata_config.items():
                    if field in study_row:
                        value = study_row[field]
                        # Handle missing values
                        if pd.isna(value):
                            value = config.get('default', 0.0)
                        metadata_dict[field] = float(value)
                    else:
                        metadata_dict[field] = config.get('default', 0.0)
                
                result['metadata'] = metadata_dict
            else:
                # Use defaults if study not found in metadata
                result['metadata'] = {
                    field: config.get('default', 0.0)
                    for field, config in self.metadata_config.items()
                }
        
        return result
    
    def __del__(self):
        """Clean up LMDB resources."""
        if self.txn is not None:
            self.txn.abort()
        if self.env is not None:
            self.env.close()


class BrainVolumeDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for brain activation data."""
    
    def __init__(
        self,
        train_split: str,
        val_split: str,
        test_split: str,
        volumetric_cache_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        kernel_selection: str = "random",
        metadata_config: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ):
        """
        Initialize DataModule.
        
        Args:
            train_split: Path to training split CSV
            val_split: Path to validation split CSV
            test_split: Path to test split CSV
            volumetric_cache_path: Path to LMDB cache
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            kernel_selection: Kernel selection strategy
            metadata_config: Configuration for metadata fields
            include_metadata: Whether to include metadata in batches
        """
        super().__init__()
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.volumetric_cache_path = volumetric_cache_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kernel_selection = kernel_selection
        self.metadata_config = metadata_config
        self.include_metadata = include_metadata
        
        # Will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = BrainVolumeDataset(
                lmdb_path=self.volumetric_cache_path,
                split_file=self.train_split,
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
            
            self.val_dataset = BrainVolumeDataset(
                lmdb_path=self.volumetric_cache_path,
                split_file=self.val_split,
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
            
            logger.info("DataModule setup completed")
            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            self.test_dataset = BrainVolumeDataset(
                lmdb_path=self.volumetric_cache_path,
                split_file=self.test_split,
                kernel_selection=self.kernel_selection,
                metadata_config=self.metadata_config,
                include_metadata=self.include_metadata
            )
            logger.info(f"Test samples: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )