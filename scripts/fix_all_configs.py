#!/usr/bin/env python3
"""
Fix metric names in all config files to match actual logged metrics.
"""

import os
import glob

def fix_config_file(filepath):
    """Fix metric names in a single config file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Track if changes were made
    original_content = content
    
    # Fix monitor metrics
    content = content.replace('monitor: "val_loss"', 'monitor: "val/total_loss"')
    
    # Fix filename template
    content = content.replace('{val_loss:', '{val_total_loss:')
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

def main():
    """Fix all config files."""
    config_files = glob.glob('configs/*.yaml')
    fixed_count = 0
    
    print("Fixing metric names in config files...")
    
    for config_file in config_files:
        if fix_config_file(config_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} config files")
    print("All configs now use 'val/total_loss' metric")

if __name__ == "__main__":
    main()