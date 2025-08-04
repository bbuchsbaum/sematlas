#!/usr/bin/env python3
"""
Create volumetric cache directly from subset data.
Bypasses the full pipeline for quick testing on RunPod.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
import logging

# Modify the volumetric cache simple to use our subset data
def main():
    # First, check if we need to create a symlink or copy
    subset_file = Path("data/processed/neurosynth_subset_1k_deduplicated.csv")
    target_file = Path("data/processed/coordinate_corrected_data.csv")
    
    if subset_file.exists() and not target_file.exists():
        print(f"Creating symlink from {subset_file} to {target_file}")
        target_file.symlink_to(subset_file.absolute())
    
    # Now run the volumetric cache creation
    from src.data.volumetric_cache_simple import main as create_cache
    
    print("Creating volumetric cache from subset data...")
    success = create_cache()
    
    if success:
        print("\nCache created successfully!")
        print("Now run: python scripts/convert_pickle_to_lmdb.py")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)