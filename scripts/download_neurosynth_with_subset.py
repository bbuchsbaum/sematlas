#!/usr/bin/env python3
"""
Data Strategy Transition: Download Real Neurosynth + Create Subset

This script downloads the complete Neurosynth dataset using NiMARE and creates
the neurosynth_subset_1k for development purposes. It implements the data
strategy transition from mock data to real neuroscientific data.

Key Features:
- Downloads complete ~12,000 study Neurosynth dataset
- Creates representative 1,000 study subset for development
- Maintains metadata distribution representativeness
- Validates pipeline compatibility
- Supports seamless transition to production scale
"""

import os
import sys
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    import nimare
    from nimare.extract import fetch_neurosynth
    from nimare.io import convert_neurosynth_to_dataset
    from nimare.dataset import Dataset
except ImportError as e:
    print(f"Error importing NiMARE: {e}")
    print("Please ensure NiMARE is installed: pip install nimare>=0.0.14")
    sys.exit(1)


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_neurosynth_transition_{timestamp}.log"
    
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


def download_full_neurosynth(data_dir: Path, logger: logging.Logger) -> Dataset:
    """
    Download complete Neurosynth database using NiMARE.
    
    Returns the full Dataset object for subset creation.
    """
    logger.info("Starting complete Neurosynth database download...")
    
    # Create data directories
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Downloading Neurosynth database via NiMARE...")
        
        # First, download the raw Neurosynth files
        logger.info("Step 1: Fetching raw Neurosynth files...")
        found_databases = fetch_neurosynth(
            data_dir=str(raw_dir),
            version='7',
            overwrite=False
        )
        
        logger.info(f"✅ Downloaded raw files: {len(found_databases)} database(s)")
        
        # Get the paths to the downloaded files
        if not found_databases:
            raise ValueError("No databases were downloaded")
        
        database_info = found_databases[0]  # Use the first (and likely only) database
        coordinates_file = database_info['coordinates']
        metadata_file = database_info['metadata']
        
        logger.info(f"Coordinates file: {coordinates_file}")
        logger.info(f"Metadata file: {metadata_file}")
        
        # Convert to NiMARE Dataset
        logger.info("Step 2: Converting to NiMARE Dataset...")
        dataset = convert_neurosynth_to_dataset(
            coordinates_file=coordinates_file,
            metadata_file=metadata_file
        )
        
        logger.info(f"✅ Successfully created NiMARE Dataset")
        logger.info(f"Total studies: {len(dataset.ids)}")
        logger.info(f"Total coordinates: {len(dataset.coordinates)}")
        
        # Save the complete dataset
        full_dataset_pkl = raw_dir / "neurosynth_full_12k.pkl.gz"
        dataset.save(str(full_dataset_pkl))
        logger.info(f"Saved complete dataset to {full_dataset_pkl}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading Neurosynth data: {e}")
        raise


def analyze_dataset_characteristics(dataset: Dataset, logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze dataset characteristics for subset selection.
    
    Returns metadata characteristics for stratified sampling.
    """
    logger.info("Analyzing dataset characteristics for subset selection...")
    
    # Get coordinates and metadata
    coords_df = dataset.coordinates.copy()
    metadata_df = dataset.metadata.copy()
    
    # Merge coordinates with metadata by study_id
    if 'study_id' in coords_df.columns and 'id' in metadata_df.columns:
        merged_df = coords_df.merge(metadata_df, left_on='study_id', right_on='id', how='left')
    else:
        logger.warning("Unable to merge coordinates with metadata - using coordinates only")
        merged_df = coords_df.copy()
    
    characteristics = {
        "total_studies": len(dataset.ids),
        "total_coordinates": len(coords_df),
        "coordinates_per_study": len(coords_df) / len(dataset.ids),
        "coordinate_ranges": {
            "x": {"min": float(coords_df['x'].min()), "max": float(coords_df['x'].max())},
            "y": {"min": float(coords_df['y'].min()), "max": float(coords_df['y'].max())},
            "z": {"min": float(coords_df['z'].min()), "max": float(coords_df['z'].max())}
        }
    }
    
    # Analyze metadata if available
    if 'year' in merged_df.columns:
        years = merged_df['year'].dropna()
        if len(years) > 0:
            characteristics["year_range"] = {
                "min": int(years.min()),
                "max": int(years.max()),
                "distribution": years.value_counts().to_dict()
            }
    
    # Analyze spatial distribution
    study_coords = coords_df.groupby('study_id').agg({
        'x': ['mean', 'std', 'count'],
        'y': ['mean', 'std', 'count'],
        'z': ['mean', 'std', 'count']
    }).reset_index()
    
    characteristics["spatial_distribution"] = {
        "mean_coords_per_study": float(study_coords[('x', 'count')].mean()),
        "centroid_distribution": {
            "x_mean": float(study_coords[('x', 'mean')].mean()),
            "y_mean": float(study_coords[('y', 'mean')].mean()),
            "z_mean": float(study_coords[('z', 'mean')].mean())
        }
    }
    
    logger.info(f"Dataset analysis complete:")
    logger.info(f"  Studies: {characteristics['total_studies']:,}")
    logger.info(f"  Coordinates: {characteristics['total_coordinates']:,}")
    logger.info(f"  Avg coords/study: {characteristics['coordinates_per_study']:.1f}")
    
    return characteristics


def create_representative_subset(dataset: Dataset, target_size: int, logger: logging.Logger) -> Dataset:
    """
    Create a representative subset maintaining key characteristics.
    
    Uses stratified sampling to preserve metadata distributions.
    """
    logger.info(f"Creating representative subset of {target_size} studies...")
    
    # Get study-level metadata for stratification
    metadata_df = dataset.metadata.copy()
    coords_df = dataset.coordinates.copy()
    
    # Calculate coordinates per study for balancing
    coords_per_study = coords_df.groupby('study_id').size().to_dict()
    metadata_df['coord_count'] = metadata_df['id'].map(coords_per_study).fillna(0)
    
    # Create stratification variables
    stratify_vars = []
    
    # Year-based stratification (if available)
    if 'year' in metadata_df.columns:
        years = metadata_df['year'].dropna()
        if len(years) > 0:
            # Create year bins for stratification
            year_bins = pd.cut(metadata_df['year'], bins=5, labels=['early', 'mid_early', 'middle', 'mid_late', 'late'])
            metadata_df['year_bin'] = year_bins
            stratify_vars.append('year_bin')
    
    # Coordinate count stratification
    coord_bins = pd.cut(metadata_df['coord_count'], bins=3, labels=['few', 'medium', 'many'])
    metadata_df['coord_bin'] = coord_bins
    stratify_vars.append('coord_bin')
    
    # Prepare stratification column
    if stratify_vars:
        # Convert categorical columns to string first, then fill missing values
        stratify_df = metadata_df[stratify_vars].copy()
        for col in stratify_vars:
            if stratify_df[col].dtype.name == 'category':
                stratify_df[col] = stratify_df[col].astype(str)
        stratify_col = stratify_df.fillna('unknown').astype(str).agg('_'.join, axis=1)
        
        # Ensure we have enough samples in each stratum
        stratum_counts = stratify_col.value_counts()
        min_stratum_size = 2  # Minimum samples per stratum
        valid_strata = stratum_counts[stratum_counts >= min_stratum_size].index
        
        if len(valid_strata) > 1:
            # Filter to valid strata
            valid_mask = stratify_col.isin(valid_strata)
            subset_df = metadata_df[valid_mask].copy()
            subset_stratify = stratify_col[valid_mask]
            
            # Perform stratified sampling
            try:
                _, selected_df = train_test_split(
                    subset_df,
                    test_size=min(target_size, len(subset_df)),
                    stratify=subset_stratify,
                    random_state=42
                )
                selected_study_ids = selected_df['id'].tolist()
                logger.info(f"✅ Stratified sampling successful: {len(selected_study_ids)} studies")
                
            except ValueError as e:
                logger.warning(f"Stratified sampling failed: {e}, using random sampling")
                selected_study_ids = metadata_df['id'].sample(n=min(target_size, len(metadata_df)), random_state=42).tolist()
        else:
            logger.warning("Insufficient strata for stratified sampling, using random sampling")
            selected_study_ids = metadata_df['id'].sample(n=min(target_size, len(metadata_df)), random_state=42).tolist()
    else:
        logger.warning("No stratification variables available, using random sampling")
        selected_study_ids = metadata_df['id'].sample(n=min(target_size, len(metadata_df)), random_state=42).tolist()
    
    # Create subset dataset
    subset_dataset = dataset.slice(selected_study_ids)
    
    logger.info(f"✅ Created subset with {len(subset_dataset.ids)} studies")
    logger.info(f"   Subset coordinates: {len(subset_dataset.coordinates)}")
    logger.info(f"   Avg coords/study: {len(subset_dataset.coordinates) / len(subset_dataset.ids):.1f}")
    
    return subset_dataset


def save_datasets(full_dataset: Dataset, subset_dataset: Dataset, data_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """
    Save both full and subset datasets with proper naming.
    
    Returns metadata about saved files.
    """
    logger.info("Saving datasets...")
    
    raw_dir = data_dir / "raw"
    
    # Save datasets
    full_path = raw_dir / "neurosynth_full_12k.pkl.gz"
    subset_path = raw_dir / "neurosynth_subset_1k.pkl.gz"
    
    full_dataset.save(str(full_path))
    subset_dataset.save(str(subset_path))
    
    logger.info(f"✅ Saved full dataset: {full_path}")
    logger.info(f"✅ Saved subset dataset: {subset_path}")
    
    # Create metadata
    metadata = {
        "download_timestamp": datetime.now().isoformat(),
        "nimare_version": nimare.__version__,
        "data_strategy": "dual_scale",
        "datasets": {
            "full": {
                "path": str(full_path),
                "n_studies": len(full_dataset.ids),
                "n_coordinates": len(full_dataset.coordinates),
                "purpose": "production_training"
            },
            "subset": {
                "path": str(subset_path),
                "n_studies": len(subset_dataset.ids),
                "n_coordinates": len(subset_dataset.coordinates),
                "purpose": "development_iteration"
            }
        },
        "files": {}
    }
    
    # Calculate file hashes and sizes
    for dataset_path in [full_path, subset_path]:
        if dataset_path.exists():
            rel_path = dataset_path.relative_to(raw_dir)
            file_hash = calculate_file_hash(dataset_path)
            file_size = dataset_path.stat().st_size
            
            metadata["files"][str(rel_path)] = {
                "size_bytes": file_size,
                "sha256": file_hash
            }
            logger.info(f"File: {rel_path}, Size: {file_size:,} bytes")
    
    # Save metadata
    metadata_file = raw_dir / "download_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata: {metadata_file}")
    
    return metadata


def validate_datasets(data_dir: Path, logger: logging.Logger) -> bool:
    """
    Validate both full and subset datasets.
    
    Returns True if validation passes, False otherwise.
    """
    logger.info("Validating downloaded datasets...")
    
    raw_dir = data_dir / "raw"
    
    # Check both datasets
    datasets_to_check = [
        ("neurosynth_full_12k.pkl.gz", "full", 1000),  # Should have >1000 studies
        ("neurosynth_subset_1k.pkl.gz", "subset", 100)   # Should have >100 studies
    ]
    
    for filename, dataset_type, min_studies in datasets_to_check:
        dataset_path = raw_dir / filename
        
        if not dataset_path.exists():
            logger.error(f"❌ {dataset_type} dataset not found: {dataset_path}")
            return False
        
        try:
            # Load and validate dataset
            dataset = Dataset.load(str(dataset_path))
            n_studies = len(dataset.ids)
            n_coords = len(dataset.coordinates)
            
            if n_studies < min_studies:
                logger.error(f"❌ {dataset_type} dataset too small: {n_studies} studies (expected >{min_studies})")
                return False
            
            # Check required columns
            required_cols = ['id', 'study_id', 'x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in dataset.coordinates.columns]
            if missing_cols:
                logger.error(f"❌ {dataset_type} dataset missing columns: {missing_cols}")
                return False
            
            logger.info(f"✅ {dataset_type} dataset valid: {n_studies:,} studies, {n_coords:,} coordinates")
            
        except Exception as e:
            logger.error(f"❌ Error validating {dataset_type} dataset: {e}")
            return False
    
    logger.info("✅ All datasets validated successfully")
    return True


def main():
    """Main function to implement data strategy transition."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    log_dir = project_root / "logs"
    
    # Setup logging
    logger = setup_logging(log_dir)
    
    try:
        logger.info("=" * 80)
        logger.info("DATA STRATEGY TRANSITION: REAL NEUROSYNTH + SUBSET CREATION")
        logger.info("=" * 80)
        
        # Step 1: Download complete Neurosynth dataset
        logger.info("STEP 1: Downloading complete Neurosynth dataset...")
        full_dataset = download_full_neurosynth(data_dir, logger)
        
        # Step 2: Analyze dataset characteristics
        logger.info("STEP 2: Analyzing dataset characteristics...")
        characteristics = analyze_dataset_characteristics(full_dataset, logger)
        
        # Step 3: Create representative subset
        logger.info("STEP 3: Creating representative subset...")
        subset_dataset = create_representative_subset(full_dataset, 1000, logger)
        
        # Step 4: Save both datasets
        logger.info("STEP 4: Saving datasets...")
        metadata = save_datasets(full_dataset, subset_dataset, data_dir, logger)
        
        # Step 5: Validate datasets
        logger.info("STEP 5: Validating datasets...")
        if validate_datasets(data_dir, logger):
            logger.info("🎉 SUCCESS: Data strategy transition completed!")
            logger.info(f"Full dataset: {metadata['datasets']['full']['n_studies']:,} studies")
            logger.info(f"Subset dataset: {metadata['datasets']['subset']['n_studies']:,} studies")
            
            # Create success marker
            success_file = data_dir / "raw" / ".data_strategy_transition_success"
            with open(success_file, 'w') as f:
                f.write(f"Data strategy transition completed at {metadata['download_timestamp']}\n")
                f.write(f"Full dataset: {metadata['datasets']['full']['n_studies']} studies\n")
                f.write(f"Subset dataset: {metadata['datasets']['subset']['n_studies']} studies\n")
                f.write(f"Ready for Sprint 2 development phase\n")
            
            logger.info("✅ Data strategy transition marker created")
            
        else:
            logger.error("❌ FAILURE: Dataset validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ FAILURE: Data strategy transition failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()