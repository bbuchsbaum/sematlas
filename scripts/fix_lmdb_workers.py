#!/usr/bin/env python3
"""
Quick fix for LMDB multi-worker issue.
Patches the DataModule to work with num_workers > 0.
"""

import shutil
import os

# Backup original
if os.path.exists('src/data/lightning_datamodule.py'):
    shutil.copy('src/data/lightning_datamodule.py', 'src/data/lightning_datamodule_backup.py')
    print("Created backup: src/data/lightning_datamodule_backup.py")

# Copy fixed version
if os.path.exists('src/data/lightning_datamodule_fixed.py'):
    shutil.copy('src/data/lightning_datamodule_fixed.py', 'src/data/lightning_datamodule.py')
    print("Applied fix: LMDB will now work with multiple workers")
    print("Run training with original config: python train.py --config configs/baseline_vae.yaml --test-run")
else:
    print("Fixed file not found. Using sed to patch...")
    # Simple sed fix as fallback
    os.system("sed -i 's/readonly=True/readonly=True, lock=False/g' src/data/lightning_datamodule.py")
    print("Applied lock=False patch")