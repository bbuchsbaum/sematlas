#!/usr/bin/env python3
"""
Script to check the contents of the LMDB database.
"""

import lmdb
import pickle

def check_lmdb_contents(lmdb_path):
    """Check what's in the LMDB database."""
    try:
        with lmdb.open(lmdb_path, readonly=True) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                
                keys = []
                for key, value in cursor:
                    keys.append(key.decode('utf-8'))
                    if len(keys) <= 10:  # Show first 10 keys
                        print(f"Key: {key.decode('utf-8')}")
                        try:
                            data = pickle.loads(value)
                            print(f"  Data type: {type(data)}")
                            if hasattr(data, 'shape'):
                                print(f"  Shape: {data.shape}")
                            elif isinstance(data, dict):
                                print(f"  Dict keys: {list(data.keys())}")
                        except Exception as e:
                            print(f"  Error loading data: {e}")
                
                print(f"\nTotal keys in LMDB: {len(keys)}")
                if len(keys) > 10:
                    print(f"First 10 keys shown above")
                    print(f"Sample of other keys: {keys[10:15]}")
                
                return keys
    except Exception as e:
        print(f"Error opening LMDB at {lmdb_path}: {e}")
        return None

if __name__ == "__main__":
    lmdb_path = "data/processed/volumetric_cache.lmdb"
    print(f"Checking LMDB database: {lmdb_path}")
    keys = check_lmdb_contents(lmdb_path)