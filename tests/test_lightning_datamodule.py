"""
Unit tests for PyTorch Lightning DataModule.

Tests the S1.2.2 acceptance criteria:
- DataModule can be instantiated
- setup() method runs correctly
- train_dataloader() returns correct batch shape and type
- Handles missing files gracefully with mock data
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import pandas as pd
import torch
import sys
import lmdb
import pickle
import numpy as np

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.lightning_datamodule import (
    BrainVolumeDataModule, 
    BrainVolumeDataset,
    create_brain_datamodule
)


class TestBrainVolumeDataModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with temporary directories."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "processed"
        self.lmdb_cache_path = str(Path(self.temp_dir) / "volumetric_cache.lmdb")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock split files
        self.create_mock_split_files()
        
        # Create mock LMDB cache
        self.create_mock_lmdb_cache()
        
        # DataModule parameters
        self.batch_size = 2
        self.num_workers = 0  # Use 0 for testing to avoid multiprocessing issues
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_split_files(self):
        """Create mock CSV split files for testing."""
        # Training split (10 studies)
        train_data = pd.DataFrame({
            'study_id': [f'train_study_{i}' for i in range(10)],
            'contrast': [f'contrast_{i}' for i in range(10)]
        })
        train_data.to_csv(self.data_dir / "train_split.csv", index=False)
        
        # Validation split (3 studies)
        val_data = pd.DataFrame({
            'study_id': [f'val_study_{i}' for i in range(3)],
            'contrast': [f'contrast_{i}' for i in range(3)]
        })
        val_data.to_csv(self.data_dir / "val_split.csv", index=False)
        
        # Test split (3 studies)
        test_data = pd.DataFrame({
            'study_id': [f'test_study_{i}' for i in range(3)],
            'contrast': [f'contrast_{i}' for i in range(3)]
        })
        test_data.to_csv(self.data_dir / "test_split.csv", index=False)
    
    def create_mock_lmdb_cache(self):
        """Create mock LMDB cache with synthetic brain data."""
        # Create LMDB database
        with lmdb.open(self.lmdb_cache_path, map_size=1024*1024*1024) as env:  # 1GB
            with env.begin(write=True) as txn:
                
                # Create data for all study IDs from split files
                all_study_ids = []
                all_study_ids.extend([f'train_study_{i}' for i in range(10)])
                all_study_ids.extend([f'val_study_{i}' for i in range(3)])
                all_study_ids.extend([f'test_study_{i}' for i in range(3)])
                
                for study_id in all_study_ids:
                    # Create mock brain volume data matching the expected format
                    mock_data = {
                        'volumes': {
                            'kernel_6mm': torch.randn(91, 109, 91).numpy(),  # MNI152 standard shape
                            'kernel_12mm': torch.randn(91, 109, 91).numpy()
                        },
                        'metadata': {
                            'study_id': study_id,
                            'num_coordinates': 10,
                            'coordinate_range': {'x': [0, 90], 'y': [0, 108], 'z': [0, 90]},
                            'stat_value_range': [1.0, 5.0],
                            'kernels': ['6mm', '12mm']
                        },
                        'original_coordinates': np.random.rand(10, 3) * 90,  # Random coordinates
                        'original_values': np.random.rand(10) * 5  # Random statistical values
                    }
                    
                    # Store in LMDB with study_id as key
                    key = study_id.encode('utf-8')
                    value = pickle.dumps(mock_data)
                    txn.put(key, value)
    
    def test_datamodule_instantiation(self):
        """Test that DataModule can be instantiated without errors."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        self.assertIsInstance(dm, BrainVolumeDataModule)
        self.assertEqual(dm.batch_size, self.batch_size)
        self.assertEqual(dm.num_workers, self.num_workers)
    
    def test_factory_function(self):
        """Test that factory function creates DataModule correctly."""
        dm = create_brain_datamodule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        self.assertIsInstance(dm, BrainVolumeDataModule)
        self.assertEqual(dm.batch_size, self.batch_size)
    
    def test_setup_method(self):
        """Test that setup() method runs correctly."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        # Should not raise any exceptions
        dm.setup("fit")
        
        # Check that datasets are created
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNotNone(dm.val_dataset)
        
        # Check dataset sizes based on mock data
        self.assertEqual(len(dm.train_dataset), 10)
        self.assertEqual(len(dm.val_dataset), 3)
    
    def test_setup_test_stage(self):
        """Test setup for test stage."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        dm.setup("test")
        
        self.assertIsNotNone(dm.test_dataset)
        self.assertEqual(len(dm.test_dataset), 3)
    
    def test_train_dataloader_returns_correct_batch(self):
        """Test that train_dataloader() returns correct batch shape and type."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        
        # Check that we can get a batch
        batch = next(iter(train_loader))
        
        # Check batch structure
        self.assertIn('image', batch)
        self.assertIn('study_id', batch)
        self.assertIn('kernel_used', batch)
        self.assertIn('metadata', batch)
        
        # Check tensor shapes and types
        images = batch['image']
        self.assertIsInstance(images, torch.Tensor)
        self.assertEqual(images.shape, (self.batch_size, 1, 91, 109, 91))
        self.assertEqual(images.dtype, torch.float32)
        
        # Check study IDs
        study_ids = batch['study_id']
        self.assertEqual(len(study_ids), self.batch_size)
        self.assertTrue(all(isinstance(sid, str) for sid in study_ids))
        
        # Check kernel choices
        kernels = batch['kernel_used']
        self.assertEqual(len(kernels), self.batch_size)
        self.assertTrue(all(k in ['6mm', '12mm'] for k in kernels))
    
    def test_val_dataloader_functionality(self):
        """Test that validation dataloader works correctly."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        dm.setup("fit")
        val_loader = dm.val_dataloader()
        
        batch = next(iter(val_loader))
        
        # Should have same structure as training batch
        self.assertEqual(batch['image'].shape, (self.batch_size, 1, 91, 109, 91))
        self.assertEqual(len(batch['study_id']), self.batch_size)
    
    def test_test_dataloader_functionality(self):
        """Test that test dataloader works correctly."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        dm.setup("test")
        test_loader = dm.test_dataloader()
        
        batch = next(iter(test_loader))
        
        # Should have same structure as training batch
        self.assertEqual(batch['image'].shape, (self.batch_size, 1, 91, 109, 91))
        self.assertEqual(len(batch['study_id']), self.batch_size)
    
    def test_kernel_selection_strategies(self):
        """Test different kernel selection strategies."""
        # Test fixed 6mm kernel
        dm_6mm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            kernel_selection="6mm"
        )
        
        dm_6mm.setup("fit")
        batch_6mm = next(iter(dm_6mm.train_dataloader()))
        
        # All should be 6mm
        self.assertTrue(all(k == '6mm' for k in batch_6mm['kernel_used']))
        
        # Test fixed 12mm kernel
        dm_12mm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            kernel_selection="12mm"
        )
        
        dm_12mm.setup("fit")
        batch_12mm = next(iter(dm_12mm.train_dataloader()))
        
        # All should be 12mm
        self.assertTrue(all(k == '12mm' for k in batch_12mm['kernel_used']))
    
    def test_missing_files_graceful_handling(self):
        """Test that missing files are handled correctly by raising errors."""
        # Create DataModule with non-existent directories
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        nonexistent_cache = Path(self.temp_dir) / "nonexistent_cache"
        
        dm = BrainVolumeDataModule(
            data_dir=str(nonexistent_dir),
            lmdb_cache=str(nonexistent_cache),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        # Should raise FileNotFoundError when LMDB cache doesn't exist
        with self.assertRaises(FileNotFoundError) as cm:
            dm.setup("fit")
        
        # Error message should be informative
        self.assertIn("LMDB cache not found", str(cm.exception))
        self.assertIn("Please run data pipeline first", str(cm.exception))
    
    def test_dataset_individual_functionality(self):
        """Test BrainVolumeDataset class individually."""
        dataset = BrainVolumeDataset(
            lmdb_path=self.lmdb_cache_path,
            split_file=str(self.data_dir / "train_split.csv"),
            kernel_selection="random"
        )
        
        # Test dataset length
        self.assertEqual(len(dataset), 10)
        
        # Test getting an item
        item = dataset[0]
        
        self.assertIn('volume', item)
        self.assertIn('study_id', item)
        self.assertIn('kernel_used', item)
        self.assertIn('metadata', item)
        
        # Check volume properties  
        volume = item['volume']
        self.assertIsInstance(volume, torch.Tensor)
        self.assertEqual(volume.shape, (1, 91, 109, 91))
        self.assertEqual(volume.dtype, torch.float32)
    
    def test_hyperparameter_saving(self):
        """Test that hyperparameters are saved correctly."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            kernel_selection="6mm"
        )
        
        # Check that hyperparameters are accessible
        self.assertEqual(dm.hparams.batch_size, self.batch_size)
        self.assertEqual(dm.hparams.kernel_selection, "6mm")
        self.assertEqual(dm.hparams.num_workers, self.num_workers)
    
    def test_predict_dataloader(self):
        """Test that predict dataloader works (should use test set)."""
        dm = BrainVolumeDataModule(
            data_dir=str(self.data_dir),
            lmdb_cache=self.lmdb_cache_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        dm.setup("test")
        predict_loader = dm.predict_dataloader()
        
        # Should return the same as test dataloader
        test_loader = dm.test_dataloader()
        
        # Both should be iterable and have same structure
        predict_batch = next(iter(predict_loader))
        test_batch = next(iter(test_loader))
        
        self.assertEqual(predict_batch['image'].shape, test_batch['image'].shape)


if __name__ == '__main__':
    unittest.main()