#!/usr/bin/env python3
"""
Convert NiMARE Dataset to Pipeline Format

This script converts the downloaded NiMARE datasets (full and subset) to the format
expected by our data pipeline (CSV with coordinates and mock statistical values).
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

try:
    from nimare.dataset import Dataset
except ImportError as e:
    print(f"Error importing NiMARE: {e}")
    print("Please ensure NiMARE is available in your environment")
    sys.exit(1)


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def convert_dataset_to_pipeline_format(
    dataset_path: str, 
    output_path: str, 
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Convert NiMARE Dataset to pipeline CSV format.
    
    Args:
        dataset_path: Path to NiMARE Dataset pickle file
        output_path: Path for output CSV file
        logger: Logger instance
        
    Returns:
        Dictionary with conversion statistics
    """
    logger.info(f"Converting {dataset_path} to pipeline format...")
    
    # Load NiMARE dataset
    dataset = Dataset.load(dataset_path)
    logger.info(f"Loaded dataset: {len(dataset.ids)} studies, {len(dataset.coordinates)} coordinates")
    
    # Get coordinates DataFrame
    coords_df = dataset.coordinates.copy()
    
    # Add mock statistical values (required by pipeline)
    # Use realistic t-statistics drawn from a mixture distribution
    np.random.seed(42)  # For reproducibility
    n_coords = len(coords_df)
    
    # Generate realistic statistical values
    # Mix of positive and negative activations with realistic magnitudes
    pos_mask = np.random.binomial(1, 0.7, n_coords).astype(bool)  # 70% positive
    
    # Positive activations: t ~ Normal(3.5, 1.5) truncated at 1.96
    pos_stats = np.random.normal(3.5, 1.5, pos_mask.sum())
    pos_stats = np.maximum(pos_stats, 1.96)  # Minimum significance threshold
    
    # Negative activations: t ~ Normal(-3.5, 1.5) truncated at -1.96
    neg_stats = np.random.normal(-3.5, 1.5, (~pos_mask).sum())
    neg_stats = np.minimum(neg_stats, -1.96)  # Maximum significance threshold
    
    # Combine
    stat_values = np.zeros(n_coords)
    stat_values[pos_mask] = pos_stats
    stat_values[~pos_mask] = neg_stats
    
    coords_df['stat_value'] = stat_values
    
    # Add sample sizes (required by deduplication)
    # Generate realistic sample sizes (10-200 subjects)
    sample_sizes = np.random.gamma(4, 8, n_coords) + 10
    sample_sizes = np.round(sample_sizes).astype(int)
    sample_sizes = np.minimum(sample_sizes, 200)  # Cap at 200
    coords_df['sample_size'] = sample_sizes
    
    # Ensure required columns are present
    required_cols = ['study_id', 'x', 'y', 'z', 'stat_value', 'sample_size', 'space']
    missing_cols = [col for col in required_cols if col not in coords_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select and reorder columns for pipeline compatibility
    pipeline_df = coords_df[required_cols].copy()
    
    # Convert coordinates to numeric (they should already be)
    for coord_col in ['x', 'y', 'z']:
        pipeline_df[coord_col] = pd.to_numeric(pipeline_df[coord_col], errors='coerce')
    
    # Remove any rows with invalid coordinates
    initial_count = len(pipeline_df)
    pipeline_df = pipeline_df.dropna(subset=['x', 'y', 'z'])
    final_count = len(pipeline_df)
    
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} rows with invalid coordinates")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_df.to_csv(output_path, index=False)
    
    # Generate statistics
    stats = {
        "input_dataset": dataset_path,
        "output_file": str(output_path),
        "studies": len(pipeline_df['study_id'].unique()),
        "coordinates": len(pipeline_df),
        "coordinate_ranges": {
            "x": {"min": float(pipeline_df['x'].min()), "max": float(pipeline_df['x'].max())},
            "y": {"min": float(pipeline_df['y'].min()), "max": float(pipeline_df['y'].max())},
            "z": {"min": float(pipeline_df['z'].min()), "max": float(pipeline_df['z'].max())}
        },
        "stat_value_range": {
            "min": float(pipeline_df['stat_value'].min()), 
            "max": float(pipeline_df['stat_value'].max())
        },
        "sample_size_range": {
            "min": int(pipeline_df['sample_size'].min()),
            "max": int(pipeline_df['sample_size'].max())
        },
        "coordinate_spaces": pipeline_df['space'].value_counts().to_dict()
    }
    
    logger.info(f"✅ Conversion complete:")
    logger.info(f"  Studies: {stats['studies']:,}")
    logger.info(f"  Coordinates: {stats['coordinates']:,}")
    logger.info(f"  Coordinate ranges: X=[{stats['coordinate_ranges']['x']['min']:.1f}, {stats['coordinate_ranges']['x']['max']:.1f}]")
    logger.info(f"  Statistical values: [{stats['stat_value_range']['min']:.2f}, {stats['stat_value_range']['max']:.2f}]")
    logger.info(f"  Saved to: {output_path}")
    
    return stats


def main():
    """Convert both full and subset datasets to pipeline format."""
    logger = setup_logging()
    
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Convert both datasets
    conversions = [
        {
            "name": "subset",
            "input": raw_dir / "neurosynth_subset_1k.pkl.gz",
            "output": processed_dir / "neurosynth_subset_1k_coordinates.csv"
        },
        {
            "name": "full", 
            "input": raw_dir / "neurosynth_full_12k.pkl.gz",
            "output": processed_dir / "neurosynth_full_12k_coordinates.csv"
        }
    ]
    
    conversion_stats = {}
    
    try:
        logger.info("=" * 70)
        logger.info("CONVERTING NIMARE DATASETS TO PIPELINE FORMAT")
        logger.info("=" * 70)
        
        for conversion in conversions:
            if not conversion["input"].exists():
                logger.error(f"Input file not found: {conversion['input']}")
                continue
                
            logger.info(f"\nConverting {conversion['name']} dataset...")
            stats = convert_dataset_to_pipeline_format(
                str(conversion["input"]),
                str(conversion["output"]),
                logger
            )
            conversion_stats[conversion["name"]] = stats
        
        logger.info("\n" + "=" * 70)
        logger.info("CONVERSION SUMMARY")
        logger.info("=" * 70)
        
        for name, stats in conversion_stats.items():
            logger.info(f"{name.upper()} DATASET:")
            logger.info(f"  ✅ {stats['studies']:,} studies, {stats['coordinates']:,} coordinates")
            logger.info(f"  📄 {stats['output_file']}")
        
        if conversion_stats:
            logger.info("🎉 SUCCESS: All conversions completed!")
            return True
        else:
            logger.error("❌ FAILURE: No datasets were converted")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAILURE: Conversion failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)