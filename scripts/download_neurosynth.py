#!/usr/bin/env python3
"""
S1.1.1: Setup Neurosynth download with NiMARE

This script downloads the latest Neurosynth database using NiMARE's built-in
functionality. It downloads the core database.json and features.json files 
to the data/raw directory.

Technical Reference: See Appendix1.md A1.2 for Neurosynth specifications 
and A1.3.3 for NiMARE integration details.

Acceptance Criteria:
- Data is downloaded to specified data/raw directory
- Script is runnable via Makefile command  
- Downloaded files match expected checksums/sizes
- Script executes without errors
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    import nimare
    from nimare.io import convert_neurosynth
    from nimare.dataset import Dataset
except ImportError as e:
    print(f"Error importing NiMARE: {e}")
    print("Please ensure NiMARE is installed: pip install nimare>=0.0.14")
    sys.exit(1)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_neurosynth_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_neurosynth_data(data_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Download Neurosynth database using NiMARE.
    
    Returns metadata about the download including file sizes and hashes.
    """
    logger.info("Starting Neurosynth database download...")
    
    # Create data directories
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use NiMARE to download and convert Neurosynth data
        logger.info("Downloading Neurosynth database via NiMARE...")
        
        # This will download the data and create a Dataset object
        dataset = convert_neurosynth(
            text_file=None,  # Use default download
            annotation_file=None,  # Use default download
            data_dir=str(raw_dir)
        )
        
        logger.info(f"Successfully downloaded Neurosynth data to {raw_dir}")
        logger.info(f"Dataset contains {len(dataset.ids)} studies")
        
        # Save the dataset object for later use
        dataset_pkl = raw_dir / "neurosynth_dataset.pkl.gz"
        dataset.save(str(dataset_pkl))
        logger.info(f"Saved NiMARE Dataset object to {dataset_pkl}")
        
        # Collect metadata about downloaded files
        metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "nimare_version": nimare.__version__,
            "n_studies": len(dataset.ids),
            "files": {}
        }
        
        # Calculate hashes and sizes for all downloaded files
        for file_path in raw_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = file_path.relative_to(raw_dir)
                file_hash = calculate_file_hash(file_path)
                file_size = file_path.stat().st_size
                
                metadata["files"][str(rel_path)] = {
                    "size_bytes": file_size,
                    "sha256": file_hash
                }
                logger.info(f"File: {rel_path}, Size: {file_size:,} bytes, SHA256: {file_hash[:16]}...")
        
        # Save metadata
        metadata_file = raw_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved download metadata to {metadata_file}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error downloading Neurosynth data: {e}")
        raise


def validate_download(data_dir: Path, logger: logging.Logger) -> bool:
    """
    Validate that the download completed successfully.
    
    Returns True if validation passes, False otherwise.
    """
    logger.info("Validating Neurosynth download...")
    
    raw_dir = data_dir / "raw"
    
    # Check if dataset object exists and is valid
    dataset_pkl = raw_dir / "neurosynth_dataset.pkl.gz"
    if not dataset_pkl.exists():
        logger.error(f"Dataset pickle file not found: {dataset_pkl}")
        return False
    
    try:
        # Try to load the dataset
        dataset = Dataset.load(str(dataset_pkl))
        n_studies = len(dataset.ids)
        
        if n_studies < 1000:  # Sanity check - should have thousands of studies
            logger.error(f"Dataset seems too small: {n_studies} studies")
            return False
        
        logger.info(f"Validation passed: {n_studies} studies loaded successfully")
        
        # Check for required columns
        required_cols = ['id', 'study_id', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in dataset.coordinates.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        logger.info("All required columns present")
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False


def main():
    """Main function to download Neurosynth data."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    log_dir = project_root / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    try:
        logger.info("=" * 60)
        logger.info("NEUROSYNTH DOWNLOAD SCRIPT - S1.1.1")
        logger.info("=" * 60)
        
        # Download data
        metadata = download_neurosynth_data(data_dir, logger)
        
        # Validate download
        if validate_download(data_dir, logger):
            logger.info("✅ SUCCESS: Neurosynth download completed and validated")
            logger.info(f"Studies downloaded: {metadata['n_studies']:,}")
            logger.info(f"Files created: {len(metadata['files'])}")
        else:
            logger.error("❌ FAILURE: Download validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ FAILURE: Script failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()