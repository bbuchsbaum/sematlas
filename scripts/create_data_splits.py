#!/usr/bin/env python3
"""
Create train/validation/test splits for the neuroimaging dataset.

Implements stratified splitting by study to ensure balanced representation
across different research contexts and maintains 70/15/15 split ratios.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stratified_splits(
    data: pd.DataFrame,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits by study.
    
    Args:
        data: DataFrame with study_id column
        split_ratios: (train, val, test) ratios, must sum to 1.0
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(split_ratios)}"
    
    np.random.seed(random_seed)
    
    # Get unique studies
    unique_studies = data['study_id'].unique()
    n_studies = len(unique_studies)
    
    logger.info(f"Splitting {n_studies} studies with ratios {split_ratios}")
    
    # Shuffle studies
    shuffled_studies = np.random.permutation(unique_studies)
    
    # Calculate split boundaries
    train_end = int(n_studies * split_ratios[0])
    val_end = train_end + int(n_studies * split_ratios[1])
    
    # Split study IDs
    train_studies = shuffled_studies[:train_end]
    val_studies = shuffled_studies[train_end:val_end]
    test_studies = shuffled_studies[val_end:]
    
    logger.info(f"Study distribution: Train={len(train_studies)}, Val={len(val_studies)}, Test={len(test_studies)}")
    
    # Create split dataframes
    train_df = data[data['study_id'].isin(train_studies)].copy()
    val_df = data[data['study_id'].isin(val_studies)].copy()
    test_df = data[data['study_id'].isin(test_studies)].copy()
    
    return train_df, val_df, test_df


def analyze_splits(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Dict:
    """
    Analyze the created splits for balance and distribution.
    
    Args:
        train_df, val_df, test_df: Split dataframes
        
    Returns:
        Dictionary with split analysis
    """
    analysis = {
        'split_sizes': {
            'train': {
                'studies': train_df['study_id'].nunique(),
                'coordinates': len(train_df)
            },
            'val': {
                'studies': val_df['study_id'].nunique(),
                'coordinates': len(val_df)
            },
            'test': {
                'studies': test_df['study_id'].nunique(),
                'coordinates': len(test_df)
            }
        },
        'split_ratios': {
            'train_studies': train_df['study_id'].nunique() / (train_df['study_id'].nunique() + val_df['study_id'].nunique() + test_df['study_id'].nunique()),
            'val_studies': val_df['study_id'].nunique() / (train_df['study_id'].nunique() + val_df['study_id'].nunique() + test_df['study_id'].nunique()),
            'test_studies': test_df['study_id'].nunique() / (train_df['study_id'].nunique() + val_df['study_id'].nunique() + test_df['study_id'].nunique()),
            'train_coords': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
            'val_coords': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
            'test_coords': len(test_df) / (len(train_df) + len(val_df) + len(test_df))
        },
        'coordinate_statistics': {
            'train': {
                'mean_coords_per_study': len(train_df) / train_df['study_id'].nunique(),
                'stat_value_range': [float(train_df['stat_value'].min()), float(train_df['stat_value'].max())]
            },
            'val': {
                'mean_coords_per_study': len(val_df) / val_df['study_id'].nunique(),
                'stat_value_range': [float(val_df['stat_value'].min()), float(val_df['stat_value'].max())]
            },
            'test': {
                'mean_coords_per_study': len(test_df) / test_df['study_id'].nunique(),
                'stat_value_range': [float(test_df['stat_value'].min()), float(test_df['stat_value'].max())]
            }
        }
    }
    
    return analysis


def main():
    """Create and validate train/validation/test splits."""
    
    print("=== S1.1.5: Data Splits Creation ===\n")
    
    # Input and output paths
    input_path = Path("data/processed/coordinate_corrected_data.csv")
    train_path = Path("data/processed/train_split.csv")
    val_path = Path("data/processed/val_split.csv") 
    test_path = Path("data/processed/test_split.csv")
    metadata_path = Path("data/processed/split_metadata.json")
    
    # Ensure output directory exists
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run coordinate correction first: python scripts/apply_coordinate_correction.py")
        return False
    
    print("1. Loading coordinate-corrected data...")
    data = pd.read_csv(input_path)
    logger.info(f"Loaded {len(data)} coordinates from {data['study_id'].nunique()} studies")
    
    print("2. Creating stratified splits...")
    # Create splits
    train_df, val_df, test_df = create_stratified_splits(
        data, 
        split_ratios=(0.7, 0.15, 0.15),
        random_seed=42
    )
    
    print("3. Analyzing split quality...")
    # Analyze splits
    analysis = analyze_splits(train_df, val_df, test_df)
    
    print("4. Saving split files...")
    # Save split files
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save metadata
    metadata = {
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'source_data': str(input_path),
        'split_parameters': {
            'split_ratios': [0.7, 0.15, 0.15],
            'random_seed': 42,
            'stratification': 'by_study'
        },
        'analysis': analysis
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("5. Split Summary:")
    print(f"   Train: {analysis['split_sizes']['train']['studies']} studies, {analysis['split_sizes']['train']['coordinates']} coordinates")
    print(f"   Val:   {analysis['split_sizes']['val']['studies']} studies, {analysis['split_sizes']['val']['coordinates']} coordinates")
    print(f"   Test:  {analysis['split_sizes']['test']['studies']} studies, {analysis['split_sizes']['test']['coordinates']} coordinates")
    
    print(f"\n6. Split Ratios (Studies):")
    print(f"   Train: {analysis['split_ratios']['train_studies']:.3f}")
    print(f"   Val:   {analysis['split_ratios']['val_studies']:.3f}")
    print(f"   Test:  {analysis['split_ratios']['test_studies']:.3f}")
    
    # Validate against S1.1.5 criteria
    print("7. Success Criteria Validation:")
    
    criteria_pass = True
    
    # Check if splits exist
    splits_exist = all(path.exists() for path in [train_path, val_path, test_path])
    if splits_exist:
        print("   ✅ Train/validation/test splits created")
    else:
        print("   ❌ Split files not created")
        criteria_pass = False
    
    # Check split ratios are approximately 70/15/15
    train_ratio = analysis['split_ratios']['train_studies']
    val_ratio = analysis['split_ratios']['val_studies'] 
    test_ratio = analysis['split_ratios']['test_studies']
    
    ratio_tolerance = 0.05  # 5% tolerance
    ratios_correct = (
        abs(train_ratio - 0.7) < ratio_tolerance and
        abs(val_ratio - 0.15) < ratio_tolerance and
        abs(test_ratio - 0.15) < ratio_tolerance
    )
    
    if ratios_correct:
        print(f"   ✅ Split ratios correct (70/15/15 ± {ratio_tolerance:.1%})")
    else:
        print(f"   ❌ Split ratios incorrect: {train_ratio:.3f}/{val_ratio:.3f}/{test_ratio:.3f}")
        criteria_pass = False
    
    # Check metadata exists
    if metadata_path.exists():
        print("   ✅ Split metadata created")
    else:
        print("   ❌ Split metadata not created")
        criteria_pass = False
    
    print(f"\n{'='*50}")
    if criteria_pass:
        print("🎉 Data splits creation SUCCESS")
        print("Train/validation/test splits (70/15/15) created successfully")
    else:
        print("❌ Data splits creation FAILED")
        print("Review implementation before proceeding")
    
    return criteria_pass


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)