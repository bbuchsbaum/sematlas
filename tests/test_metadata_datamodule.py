"""
Test suite for S2.2.1: Metadata DataModule.

Tests success criteria:
- DataModule yields (image, metadata) batches
- Metadata includes task category, year, sample size  
- Missing metadata handled gracefully
- Batch shapes consistent across epochs
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.lightning_datamodule import (
    BrainVolumeDataModule, BrainVolumeDataset, create_brain_datamodule
)
from src.models.metadata_imputation import create_default_metadata_config


class TestMetadataDataModule:
    """Test suite for metadata-enhanced DataModule."""
    
    def test_datamodule_yields_image_metadata_batches(self):
        """Test that DataModule yields (image, metadata) batches."""
        dm = create_brain_datamodule(batch_size=2, num_workers=0, include_metadata=True)
        
        # Setup for training
        dm.setup("fit")
        
        # Test train dataloader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        # Check batch structure
        assert 'image' in batch, "Batch should contain 'image' key"
        assert 'metadata' in batch, "Batch should contain 'metadata' key"
        assert 'study_id' in batch, "Batch should contain 'study_id' key"
        assert 'kernel_used' in batch, "Batch should contain 'kernel_used' key"
        
        # Check image shape
        assert batch['image'].shape == (2, 1, 91, 109, 91), f"Wrong image shape: {batch['image'].shape}"
        assert batch['image'].dtype == torch.float32, "Image should be float32"
        
        # Check metadata structure
        assert isinstance(batch['metadata'], dict), "Metadata should be dict"
        
        print("✅ DataModule yields (image, metadata) batches")
    
    def test_metadata_includes_required_fields(self):
        """Test that metadata includes task category, year, sample size."""
        dm = create_brain_datamodule(batch_size=3, num_workers=0, include_metadata=True)
        dm.setup("fit")
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        metadata = batch['metadata']
        
        # Check required fields
        required_fields = ['sample_size', 'study_year', 'task_category']
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"
        
        # Check field shapes and types
        batch_size = batch['image'].shape[0]
        
        # Sample size (continuous)
        assert metadata['sample_size'].shape == (batch_size, 1), f"Wrong sample_size shape: {metadata['sample_size'].shape}"
        assert metadata['sample_size'].dtype == torch.float32, "sample_size should be float32"
        assert (metadata['sample_size'] > 0).all(), "Sample sizes should be positive"
        
        # Study year (continuous)
        assert metadata['study_year'].shape == (batch_size, 1), f"Wrong study_year shape: {metadata['study_year'].shape}"
        assert metadata['study_year'].dtype == torch.float32, "study_year should be float32"
        assert (metadata['study_year'] >= 1990).all(), "Study years should be >= 1990"
        assert (metadata['study_year'] <= 2025).all(), "Study years should be <= 2025"
        
        # Task category (categorical/one-hot)
        assert metadata['task_category'].shape == (batch_size, 10), f"Wrong task_category shape: {metadata['task_category'].shape}"
        assert metadata['task_category'].dtype == torch.float32, "task_category should be float32"
        # Check one-hot encoding
        assert torch.allclose(metadata['task_category'].sum(dim=1), torch.ones(batch_size)), "task_category should be one-hot encoded"
        
        print("✅ Metadata includes task category, year, sample size")
    
    def test_missing_metadata_handled_gracefully(self):
        """Test that missing metadata is handled gracefully."""
        # Test with metadata disabled
        dm_no_metadata = create_brain_datamodule(batch_size=2, num_workers=0, include_metadata=False)
        dm_no_metadata.setup("fit")
        
        train_loader = dm_no_metadata.train_dataloader()
        batch = next(iter(train_loader))
        
        # Should not have metadata key
        assert 'metadata' not in batch, "Batch should not contain metadata when disabled"
        assert 'image' in batch, "Batch should still contain image"
        
        # Test with metadata enabled but missing data
        dm_with_metadata = create_brain_datamodule(batch_size=2, num_workers=0, include_metadata=True)
        dm_with_metadata.setup("fit")
        
        train_loader_meta = dm_with_metadata.train_dataloader()
        batch_meta = next(iter(train_loader_meta))
        
        # Should gracefully handle missing LMDB data by using mock values
        assert 'metadata' in batch_meta, "Should gracefully provide metadata even when LMDB missing"
        assert 'image' in batch_meta, "Should still provide image data"
        
        # Check that mock metadata has reasonable values
        metadata = batch_meta['metadata']
        assert (metadata['sample_size'] > 0).all(), "Mock sample sizes should be positive"
        assert (metadata['study_year'] >= 1990).all(), "Mock study years should be reasonable"
        
        print("✅ Missing metadata handled gracefully")
    
    def test_batch_shapes_consistent_across_epochs(self):
        """Test that batch shapes are consistent across epochs."""
        dm = create_brain_datamodule(batch_size=2, num_workers=0, include_metadata=True)
        dm.setup("fit")
        
        train_loader = dm.train_dataloader()
        
        # Test multiple batches from same loader
        shapes_epoch1 = []
        metadata_keys_epoch1 = []
        
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
                
            # Record shapes
            shapes_epoch1.append({
                'image': batch['image'].shape,
                'metadata': {k: v.shape for k, v in batch['metadata'].items()}
            })
            metadata_keys_epoch1.append(set(batch['metadata'].keys()))
        
        # Check shapes are consistent
        for i in range(1, len(shapes_epoch1)):
            assert shapes_epoch1[i]['image'] == shapes_epoch1[0]['image'], \
                f"Image shapes inconsistent: {shapes_epoch1[i]['image']} vs {shapes_epoch1[0]['image']}"
            
            for field_name in shapes_epoch1[0]['metadata']:
                assert shapes_epoch1[i]['metadata'][field_name] == shapes_epoch1[0]['metadata'][field_name], \
                    f"Metadata field {field_name} shapes inconsistent"
        
        # Check metadata keys are consistent
        for keys in metadata_keys_epoch1[1:]:
            assert keys == metadata_keys_epoch1[0], "Metadata keys should be consistent across batches"
        
        # Test second epoch (simulated by creating new iterator)
        train_loader_epoch2 = dm.train_dataloader()
        batch_epoch2 = next(iter(train_loader_epoch2))
        
        # Should have same structure as first epoch
        assert batch_epoch2['image'].shape == shapes_epoch1[0]['image'], \
            "Image shape should be consistent across epochs"
        assert set(batch_epoch2['metadata'].keys()) == metadata_keys_epoch1[0], \
            "Metadata keys should be consistent across epochs"
        
        for field_name in batch_epoch2['metadata']:
            assert batch_epoch2['metadata'][field_name].shape == shapes_epoch1[0]['metadata'][field_name], \
                f"Metadata field {field_name} shape should be consistent across epochs"
        
        print("✅ Batch shapes consistent across epochs")
    
    def test_validation_and_test_loaders(self):
        """Test that validation and test loaders also work correctly."""
        dm = create_brain_datamodule(batch_size=2, num_workers=0, include_metadata=True)
        dm.setup("fit")
        dm.setup("test")
        
        # Test validation loader
        val_loader = dm.val_dataloader()
        val_batch = next(iter(val_loader))
        
        assert 'image' in val_batch, "Val batch should contain image"
        assert 'metadata' in val_batch, "Val batch should contain metadata"
        
        # Test test loader
        test_loader = dm.test_dataloader()
        test_batch = next(iter(test_loader))
        
        assert 'image' in test_batch, "Test batch should contain image"
        assert 'metadata' in test_batch, "Test batch should contain metadata"
        
        # All loaders should have same metadata structure
        train_loader = dm.train_dataloader()
        train_batch = next(iter(train_loader))
        
        assert train_batch['metadata'].keys() == val_batch['metadata'].keys(), \
            "Train/val metadata keys should match"
        assert train_batch['metadata'].keys() == test_batch['metadata'].keys(), \
            "Train/test metadata keys should match"
        
        print("✅ Validation and test loaders work correctly")
    
    def test_metadata_configuration_customization(self):
        """Test that metadata configuration can be customized."""
        # Create custom metadata config
        custom_config = {
            'sample_size': {'type': 'continuous', 'dim': 1, 'missing_rate': 0.1, 'prior_mean': 30.0, 'prior_std': 15.0},
            'custom_field': {'type': 'categorical', 'dim': 5, 'missing_rate': 0.2}
        }
        
        dm = create_brain_datamodule(
            batch_size=2, 
            num_workers=0, 
            include_metadata=True,
            metadata_config=custom_config
        )
        dm.setup("fit")
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        
        metadata = batch['metadata']
        
        # Should have custom fields
        assert 'sample_size' in metadata, "Should have sample_size field"
        assert 'custom_field' in metadata, "Should have custom_field"
        assert 'study_year' not in metadata, "Should not have study_year (not in custom config)"
        
        # Check custom field properties
        assert metadata['custom_field'].shape == (2, 5), f"Wrong custom_field shape: {metadata['custom_field'].shape}"
        assert torch.allclose(metadata['custom_field'].sum(dim=1), torch.ones(2)), \
            "custom_field should be one-hot encoded"
        
        print("✅ Metadata configuration customization works")


def test_s2_2_1_success_criteria():
    """
    Comprehensive test for S2.2.1 SUCCESS_MARKERS criteria:
    - [✅] DataModule yields (image, metadata) batches
    - [✅] Metadata includes task category, year, sample size
    - [✅] Missing metadata handled gracefully
    - [✅] Batch shapes consistent across epochs
    """
    print("\n=== Testing S2.2.1: Metadata DataModule ===")
    
    test_suite = TestMetadataDataModule()
    
    # Run all tests
    test_suite.test_datamodule_yields_image_metadata_batches()
    test_suite.test_metadata_includes_required_fields()
    test_suite.test_missing_metadata_handled_gracefully()
    test_suite.test_batch_shapes_consistent_across_epochs()
    test_suite.test_validation_and_test_loaders()
    test_suite.test_metadata_configuration_customization()
    
    print("\n🎉 All S2.2.1 SUCCESS CRITERIA PASSED!")
    print("✅ DataModule yields (image, metadata) batches")
    print("✅ Metadata includes task category, year, sample size")
    print("✅ Missing metadata handled gracefully")
    print("✅ Batch shapes consistent across epochs")


if __name__ == "__main__":
    test_s2_2_1_success_criteria()