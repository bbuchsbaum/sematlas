#!/usr/bin/env python3
"""
Quick conversion from pickle cache to LMDB for validation.
"""

import lmdb
import pickle
import os
from pathlib import Path
import pandas as pd

def convert_pickle_to_lmdb():
    """Convert existing pickle cache to LMDB format."""
    
    pickle_dir = Path("data/processed/volumetric_cache")
    lmdb_path = "data/processed/volumetric_cache_lmdb"
    
    # Create LMDB environment
    os.makedirs(lmdb_path, exist_ok=True)
    
    # Get list of pickle files
    pickle_files = list(pickle_dir.glob("study_*.pkl"))
    print(f"Converting {len(pickle_files)} pickle files to LMDB...")
    
    # Open LMDB for writing
    env = lmdb.open(lmdb_path, map_size=50 * 10**9)  # 50GB max
    
    with env.begin(write=True) as txn:
        for pkl_file in pickle_files:
            # Extract study ID from filename
            study_id = pkl_file.stem.replace("study_", "")
            
            # Load pickle data
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Store in LMDB
            key = study_id.encode('utf-8')
            value = pickle.dumps(data)
            txn.put(key, value)
            
            if len(pickle_files) < 50 or int(study_id) % 100 == 0:
                print(f"Converted study {study_id}")
    
    env.close()
    print(f"LMDB cache created at {lmdb_path}")
    
    # Update config to use LMDB path
    print("Update your config files to use:")
    print(f"  volumetric_cache_path: {lmdb_path}")

if __name__ == "__main__":
    convert_pickle_to_lmdb()