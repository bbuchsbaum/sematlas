#!/usr/bin/env python3
"""
Environment and code synchronization for Paperspace training.

This script handles:
1. Environment setup and dependency management
2. Code synchronization and workspace preparation
3. Configuration validation
4. Data availability checks

Usage:
    python scripts/sync_environment.py --prepare
    python scripts/sync_environment.py --check
    python scripts/sync_environment.py --sync-code
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def setup_logging() -> logging.Logger:
    """Setup logging for environment synchronization."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('environment_sync.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Get current git commit hash for version tracking."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
        return "unknown"
    except subprocess.TimeoutExpired:
        return "unknown"


def get_file_hash(filepath: str) -> str:
    """Get SHA256 hash of a file for integrity checking."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except FileNotFoundError:
        return "missing"


def check_local_environment() -> Dict[str, Any]:
    """Check local environment and dependencies."""
    logger = logging.getLogger(__name__)
    
    status = {
        'python_version': sys.version_info[:2],
        'git_commit': get_git_commit_hash(),
        'dependencies': {},
        'configs': {},
        'scripts': {},
        'issues': []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        status['issues'].append("Python version < 3.8")
    
    # Check key dependencies
    key_packages = ['torch', 'pytorch_lightning', 'wandb', 'numpy', 'pandas']
    for package in key_packages:
        try:
            import importlib
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                status['dependencies'][package] = module.__version__
            else:
                status['dependencies'][package] = "installed"
        except ImportError:
            status['dependencies'][package] = "missing"
            status['issues'].append(f"Missing package: {package}")
    
    # Check configuration files
    config_files = [
        'configs/baseline_vae.yaml',
        'environment.yml',
        'requirements.txt'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            status['configs'][config_file] = get_file_hash(config_file)
        else:
            status['configs'][config_file] = "missing"
            status['issues'].append(f"Missing config: {config_file}")
    
    # Check key scripts
    key_scripts = [
        'train.py',
        'scripts/train_paperspace.py',
        'src/models/resnet_vae.py',
        'src/data/lightning_datamodule.py'
    ]
    
    for script in key_scripts:
        if os.path.exists(script):
            status['scripts'][script] = get_file_hash(script)
        else:
            status['scripts'][script] = "missing"
            status['issues'].append(f"Missing script: {script}")
    
    # Log status
    logger.info(f"Environment check complete:")
    logger.info(f"  Python: {status['python_version']}")
    logger.info(f"  Git commit: {status['git_commit']}")
    logger.info(f"  Dependencies: {len([v for v in status['dependencies'].values() if v != 'missing'])}/{len(status['dependencies'])}")
    logger.info(f"  Issues found: {len(status['issues'])}")
    
    if status['issues']:
        logger.warning("Issues found:")
        for issue in status['issues']:
            logger.warning(f"  - {issue}")
    
    return status


def create_environment_requirements() -> str:
    """Create requirements.txt for Paperspace environment."""
    
    requirements = [
        "# Core ML packages",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "pytorch-lightning>=1.6.0",
        "",
        "# Data science",
        "numpy>=1.21.0", 
        "pandas>=1.4.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.1.0",
        "",
        "# Neuroimaging",
        "nibabel>=4.0.0",
        "nilearn>=0.9.0",
        "",
        "# Visualization and monitoring",
        "matplotlib>=3.5.0",
        "plotly>=5.8.0",
        "wandb>=0.12.0",
        "",
        "# Configuration and utilities", 
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "h5py>=3.7.0",
        "",
        "# Development tools",
        "pytest>=7.0.0",
        "black>=22.0.0"
    ]
    
    return '\n'.join(requirements)


def create_paperspace_setup_script() -> str:
    """Create setup script for Paperspace environment."""
    
    setup_script = """#!/bin/bash
set -e

echo "=== Paperspace Environment Setup ==="
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo

# Set environment variables
export PYTHONPATH=/notebooks:$PYTHONPATH
export TORCH_HOME=/tmp/torch_cache
export HF_HOME=/tmp/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints wandb_logs data/cache data/processed

# Update pip and install requirements
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify PyTorch CUDA availability
echo "Verifying PyTorch CUDA setup..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No CUDA devices')"

# Set up git (if needed for version tracking)
git config --global user.email "paperspace@sematlas.ai" || true
git config --global user.name "Paperspace Worker" || true

echo "Environment setup complete!"
echo "=== Ready for training ==="
"""
    
    return setup_script


def prepare_sync_manifest() -> Dict[str, Any]:
    """Prepare synchronization manifest for Paperspace."""
    logger = logging.getLogger(__name__)
    
    manifest = {
        'sync_version': '1.0',
        'timestamp': subprocess.run(['date', '-Iseconds'], capture_output=True, text=True).stdout.strip(),
        'git_commit': get_git_commit_hash(),
        'files': {},
        'directories': {},
        'requirements': {}
    }
    
    # Core files to sync
    core_files = [
        'train.py',
        'requirements.txt',
        'environment.yml'
    ]
    
    # Core directories to sync
    core_dirs = [
        'src/',
        'configs/',
        'scripts/'
    ]
    
    # Track file hashes
    for file_path in core_files:
        if os.path.exists(file_path):
            manifest['files'][file_path] = {
                'hash': get_file_hash(file_path),
                'size': os.path.getsize(file_path),
                'modified': os.path.getmtime(file_path)
            }
    
    # Track directory contents
    for dir_path in core_dirs:
        if os.path.exists(dir_path):
            dir_files = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.py') or file.endswith('.yaml') or file.endswith('.yml'):
                        full_path = os.path.join(root, file)
                        dir_files.append({
                            'path': full_path,
                            'hash': get_file_hash(full_path)
                        })
            manifest['directories'][dir_path] = dir_files
    
    # Environment requirements
    env_status = check_local_environment()
    manifest['requirements'] = {
        'python_version': env_status['python_version'],
        'dependencies': env_status['dependencies'],
        'issues': env_status['issues']
    }
    
    logger.info(f"Sync manifest prepared:")
    logger.info(f"  Files: {len(manifest['files'])}")
    logger.info(f"  Directories: {len(manifest['directories'])}")
    logger.info(f"  Git commit: {manifest['git_commit']}")
    
    return manifest


def save_sync_artifacts() -> None:
    """Save synchronization artifacts for Paperspace."""
    logger = logging.getLogger(__name__)
    
    # Create requirements.txt
    requirements_content = create_environment_requirements()
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    logger.info("Created requirements.txt")
    
    # Create setup script
    setup_script = create_paperspace_setup_script()
    with open('paperspace_setup.sh', 'w') as f:
        f.write(setup_script)
    os.chmod('paperspace_setup.sh', 0o755)
    logger.info("Created paperspace_setup.sh")
    
    # Create sync manifest
    manifest = prepare_sync_manifest()
    with open('sync_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info("Created sync_manifest.json")


def validate_sync_readiness() -> Tuple[bool, List[str]]:
    """Validate that environment is ready for synchronization."""
    logger = logging.getLogger(__name__)
    
    issues = []
    
    # Check environment status
    env_status = check_local_environment()
    issues.extend(env_status['issues'])
    
    # Check required files exist
    required_files = [
        'train.py',
        'src/models/resnet_vae.py', 
        'src/data/lightning_datamodule.py',
        'configs/baseline_vae.yaml'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            issues.append(f"Required file missing: {file_path}")
    
    # Check git status
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            logger.warning("Uncommitted changes detected:")
            for line in result.stdout.strip().split('\n'):
                logger.warning(f"  {line}")
    except subprocess.TimeoutExpired:
        issues.append("Could not check git status")
    
    ready = len(issues) == 0
    
    logger.info(f"Sync readiness check: {'READY' if ready else 'NOT READY'}")
    if issues:
        logger.warning("Issues preventing sync:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return ready, issues


def main():
    """Main environment synchronization handler."""
    parser = argparse.ArgumentParser(description='Environment synchronization for Paperspace')
    parser.add_argument('--prepare', action='store_true',
                       help='Prepare synchronization artifacts')
    parser.add_argument('--check', action='store_true', 
                       help='Check environment status')
    parser.add_argument('--validate', action='store_true',
                       help='Validate readiness for sync')
    parser.add_argument('--sync-code', action='store_true',
                       help='Perform code synchronization')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if not any([args.prepare, args.check, args.validate, args.sync_code]):
        logger.error("Please specify an action: --prepare, --check, --validate, or --sync-code")
        return 1
    
    logger.info("=== Environment Synchronization ===")
    
    if args.check:
        logger.info("Checking local environment...")
        env_status = check_local_environment()
        print(json.dumps(env_status, indent=2))
    
    if args.validate:
        logger.info("Validating sync readiness...")
        ready, issues = validate_sync_readiness()
        if not ready:
            return 1
    
    if args.prepare:
        logger.info("Preparing synchronization artifacts...")
        save_sync_artifacts()
        logger.info("Synchronization artifacts prepared successfully!")
    
    if args.sync_code:
        logger.info("Performing code synchronization...")
        ready, issues = validate_sync_readiness()
        if not ready:
            logger.error("Environment not ready for sync. Run --validate first.")
            return 1
        
        save_sync_artifacts()
        logger.info("Code synchronization complete!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)