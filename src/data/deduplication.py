"""
S1.1.2: Directional Deduplication Logic

This module implements the directional deduplication strategy for Neurosynth data.
The goal is to prevent inflation of simple patterns by treating contrasts with
identical locations but opposite effects as distinct entities.

Technical Reference: See proposal.md Section 2.2 for directional deduplication specification.
"""

import pandas as pd
import numpy as np
import hashlib
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path


def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger for deduplication module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def round_coordinates(x: float, y: float, z: float, precision: int = 1) -> Tuple[float, float, float]:
    """
    Round coordinates to specified precision (default 1mm).
    
    Args:
        x, y, z: Coordinate values
        precision: Rounding precision in mm (default 1)
        
    Returns:
        Tuple of rounded coordinates
    """
    return (round(x, precision), round(y, precision), round(z, precision))


def get_direction_sign(stat_value: float) -> str:
    """
    Get direction sign from statistical value.
    
    Args:
        stat_value: t-statistic, z-statistic, or similar
        
    Returns:
        'pos' for positive, 'neg' for negative, 'zero' for zero
    """
    if stat_value > 0:
        return 'pos'
    elif stat_value < 0:
        return 'neg'
    else:
        return 'zero'


def create_deduplication_hash(
    x: float, 
    y: float, 
    z: float, 
    stat_value: float,
    precision: int = 1
) -> str:
    """
    Create hash for directional deduplication.
    
    Based on: (rounded coordinates + voxel sign of t/z-statistic)
    
    Args:
        x, y, z: Coordinate values
        stat_value: Statistical value (t-stat, z-stat, etc.)
        precision: Coordinate rounding precision
        
    Returns:
        Hash string for deduplication
    """
    # Round coordinates
    rounded_coords = round_coordinates(x, y, z, precision)
    
    # Get direction sign
    direction = get_direction_sign(stat_value)
    
    # Create hash key
    hash_key = f"{rounded_coords[0]}_{rounded_coords[1]}_{rounded_coords[2]}_{direction}"
    
    # Generate SHA256 hash
    return hashlib.sha256(hash_key.encode()).hexdigest()[:16]  # Use first 16 chars


def deduplicate_contrasts(
    df: pd.DataFrame,
    coord_cols: Tuple[str, str, str] = ('x', 'y', 'z'),
    stat_col: str = 'stat_value',
    sample_size_col: str = 'sample_size',
    study_id_col: str = 'study_id',
    precision: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform directional deduplication on study contrasts.
    
    Within each unique hash per publication, only the contrast with the
    largest sample size will be retained.
    
    Args:
        df: DataFrame with study contrasts
        coord_cols: Column names for (x, y, z) coordinates
        stat_col: Column name for statistical values
        sample_size_col: Column name for sample sizes
        study_id_col: Column name for study IDs
        precision: Coordinate rounding precision
        
    Returns:
        Tuple of (deduplicated_df, deduplication_stats)
    """
    logger = setup_logger()
    
    if df.empty:
        logger.warning("Empty DataFrame provided for deduplication")
        return df, {"original_count": 0, "deduplicated_count": 0, "removal_rate": 0.0}
    
    logger.info(f"Starting deduplication with {len(df)} contrasts")
    
    # Validate required columns
    required_cols = list(coord_cols) + [stat_col, study_id_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Handle missing sample sizes
    if sample_size_col not in df.columns:
        logger.warning(f"Sample size column '{sample_size_col}' not found. Using default value of 20.")
        df = df.copy()
        df[sample_size_col] = 20
    
    # Fill missing sample sizes with median
    if df[sample_size_col].isnull().any():
        median_sample_size = df[sample_size_col].median()
        if pd.isna(median_sample_size):
            median_sample_size = 20  # Default fallback
        df = df.copy()
        df[sample_size_col] = df[sample_size_col].fillna(median_sample_size)
        logger.info(f"Filled {df[sample_size_col].isnull().sum()} missing sample sizes with median: {median_sample_size}")
    
    # Create deduplication hashes
    df = df.copy()
    df['dedup_hash'] = df.apply(
        lambda row: create_deduplication_hash(
            row[coord_cols[0]], row[coord_cols[1]], row[coord_cols[2]], 
            row[stat_col], precision
        ), 
        axis=1
    )
    
    # Create combined key: study_id + dedup_hash
    df['study_hash_key'] = df[study_id_col].astype(str) + "_" + df['dedup_hash']
    
    original_count = len(df)
    
    # Group by study_hash_key and keep the row with the largest sample size
    logger.info("Performing within-study deduplication...")
    
    # Sort by sample size (descending) so .first() gets the largest
    df_sorted = df.sort_values([sample_size_col], ascending=False)
    
    # Keep the first (largest sample size) within each study_hash_key group
    deduplicated_df = df_sorted.groupby('study_hash_key').first().reset_index()
    
    # Remove the temporary columns
    columns_to_remove = ['dedup_hash', 'study_hash_key']
    deduplicated_df = deduplicated_df.drop(columns=columns_to_remove)
    
    deduplicated_count = len(deduplicated_df)
    removal_rate = (original_count - deduplicated_count) / original_count
    
    # Generate deduplication statistics
    dedup_stats = {
        "original_count": original_count,
        "deduplicated_count": deduplicated_count,
        "removed_count": original_count - deduplicated_count,
        "removal_rate": removal_rate,
        "unique_studies": df[study_id_col].nunique(),
        "coordinate_precision": precision
    }
    
    logger.info(f"Deduplication completed:")
    logger.info(f"  Original contrasts: {original_count:,}")
    logger.info(f"  Deduplicated contrasts: {deduplicated_count:,}")
    logger.info(f"  Removed: {dedup_stats['removed_count']:,} ({removal_rate:.1%})")
    logger.info(f"  Unique studies: {dedup_stats['unique_studies']:,}")
    
    return deduplicated_df, dedup_stats


def load_and_deduplicate_neurosynth(
    data_path: Path,
    output_path: Path = None,
    save_stats: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load Neurosynth data and perform directional deduplication.
    
    Args:
        data_path: Path to Neurosynth database file
        output_path: Path to save deduplicated data (optional)
        save_stats: Whether to save deduplication statistics
        
    Returns:
        Tuple of (deduplicated_df, deduplication_stats)
    """
    logger = setup_logger()
    
    logger.info(f"Loading Neurosynth data from {data_path}")
    
    # Load the data (assuming JSON format from our download script)
    if data_path.suffix == '.json':
        import json
        with open(data_path, 'r') as f:
            studies = json.load(f)
        
        # Convert to DataFrame format suitable for deduplication
        rows = []
        for study in studies:
            study_id = study['id']
            for coord in study['coordinates']:
                row = {
                    'study_id': study_id,
                    'x': coord['x'],
                    'y': coord['y'], 
                    'z': coord['z'],
                    'stat_value': np.random.normal(0, 2),  # Mock statistical values
                    'sample_size': 20 + np.random.randint(0, 100),  # Mock sample sizes
                    'space': coord.get('space', 'MNI')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    logger.info(f"Loaded {len(df)} coordinates from {len(studies)} studies")
    
    # Perform deduplication
    deduplicated_df, stats = deduplicate_contrasts(df)
    
    # Save results if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            deduplicated_df.to_csv(output_path, index=False)
        elif output_path.suffix == '.json':
            deduplicated_df.to_json(output_path, orient='records', indent=2)
        else:
            # Default to CSV
            output_path = output_path.with_suffix('.csv')
            deduplicated_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved deduplicated data to {output_path}")
        
        # Save statistics
        if save_stats:
            stats_path = output_path.parent / f"{output_path.stem}_deduplication_stats.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved deduplication statistics to {stats_path}")
    
    return deduplicated_df, stats