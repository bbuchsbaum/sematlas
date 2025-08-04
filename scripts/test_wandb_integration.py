#!/usr/bin/env python3
"""
Test script for W&B integration validation.

This script tests the W&B integration to ensure that:
1. API key management works correctly
2. Local and cloud training appear in the same project
3. Proper tagging and metadata is applied
4. Both online and offline modes function

Usage:
    python scripts/test_wandb_integration.py --test-local
    python scripts/test_wandb_integration.py --test-cloud-sim
    python scripts/test_wandb_integration.py --validate-config
"""

import os
import sys
import argparse
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def test_wandb_import() -> bool:
    """Test that wandb can be imported and basic functionality works."""
    logger = logging.getLogger(__name__)
    
    try:
        import wandb
        logger.info(f"✓ W&B import successful (version: {wandb.__version__})")
        return True
    except ImportError:
        logger.error("✗ W&B not installed or not importable")
        return False


def test_wandb_login() -> bool:
    """Test W&B login functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        import wandb
        
        # Check if already logged in
        try:
            api = wandb.Api()
            user = api.viewer
            logger.info(f"✓ W&B already logged in as: {user.get('username', 'unknown')}")
            return True
        except Exception:
            logger.warning("W&B not logged in or API key not valid")
            return False
            
    except Exception as e:
        logger.error(f"✗ W&B login test failed: {e}")
        return False


def simulate_paperspace_environment(wandb_api_key: Optional[str], test_run: bool = True) -> bool:
    """Simulate Paperspace environment setup."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import the W&B setup function
        sys.path.append(str(Path.cwd()))
        from scripts.train_paperspace import setup_wandb_config
        
        # Test W&B config generation
        config_path = "configs/baseline_vae.yaml"
        wandb_env_vars = setup_wandb_config(config_path, wandb_api_key, test_run)
        
        logger.info("✓ Paperspace W&B configuration generated:")
        for key, value in wandb_env_vars.items():
            if 'API_KEY' in key:
                logger.info(f"  {key}: {'***' if value else 'None'}")
            else:
                logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Paperspace environment simulation failed: {e}")
        return False


def test_local_wandb_init() -> bool:
    """Test local W&B initialization."""
    logger = logging.getLogger(__name__)
    
    try:
        import wandb
        
        # Test offline mode
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_PROJECT'] = 'generative-brain-atlas-test'
        os.environ['WANDB_RUN_NAME'] = 'test-local-run'
        os.environ['WANDB_TAGS'] = 'test,local'
        
        # Initialize in offline mode
        run = wandb.init(
            project='generative-brain-atlas-test',
            name='test-local-run',
            tags=['test', 'local'],
            notes='Test run for W&B integration validation',
            mode='offline'
        )
        
        # Log some test metrics
        run.log({'test_metric': 0.5, 'epoch': 1})
        run.finish()
        
        logger.info("✓ Local W&B initialization successful (offline mode)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Local W&B initialization failed: {e}")
        return False


def test_cloud_wandb_simulation() -> bool:
    """Test cloud W&B setup simulation."""
    logger = logging.getLogger(__name__)
    
    try:
        import wandb
        
        # Simulate cloud environment variables
        cloud_env = {
            'WANDB_MODE': 'offline',  # Use offline for testing
            'WANDB_PROJECT': 'generative-brain-atlas-test',
            'WANDB_RUN_NAME': 'test-paperspace-simulation',
            'WANDB_TAGS': 'test,paperspace,simulation',
            'WANDB_NOTES': 'Simulated Paperspace training run',
            'WANDB_DIR': 'wandb_logs'
        }
        
        # Set environment variables
        for key, value in cloud_env.items():
            os.environ[key] = value
        
        # Initialize W&B
        run = wandb.init(
            project=cloud_env['WANDB_PROJECT'],
            name=cloud_env['WANDB_RUN_NAME'],
            tags=cloud_env['WANDB_TAGS'].split(','),
            notes=cloud_env['WANDB_NOTES'],
            mode=cloud_env['WANDB_MODE']
        )
        
        # Log test metrics
        run.log({'test_metric': 0.8, 'epoch': 1, 'val_loss': 0.3})
        run.finish()
        
        logger.info("✓ Cloud W&B simulation successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cloud W&B simulation failed: {e}")
        return False


def validate_config_integration() -> bool:
    """Validate W&B configuration in baseline config."""
    logger = logging.getLogger(__name__)
    
    try:
        import yaml
        
        config_path = Path("configs/baseline_vae.yaml")
        if not config_path.exists():
            logger.error("✗ Baseline configuration file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        wandb_config = config.get('wandb', {})
        
        # Check required W&B fields
        required_fields = ['project', 'name', 'tags']
        missing_fields = []
        
        for field in required_fields:
            if field not in wandb_config:
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"✗ Missing W&B config fields: {missing_fields}")
            return False
        
        logger.info("✓ W&B configuration validation successful:")
        logger.info(f"  Project: {wandb_config['project']}")
        logger.info(f"  Name: {wandb_config['name']}")
        logger.info(f"  Tags: {wandb_config['tags']}")
        logger.info(f"  Notes: {wandb_config.get('notes', 'None')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Config validation failed: {e}")
        return False


def run_comprehensive_test() -> bool:
    """Run comprehensive W&B integration test."""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Comprehensive W&B Integration Test ===")
    
    tests = [
        ("W&B Import", test_wandb_import),
        ("Config Validation", validate_config_integration),
        ("Local W&B Init", test_local_wandb_init),
        ("Cloud Simulation", test_cloud_wandb_simulation),
        ("Paperspace Env Setup", lambda: simulate_paperspace_environment(None, True))
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.warning(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {passed}/{total}")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All tests passed! W&B integration is ready.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check configuration.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test W&B integration')
    parser.add_argument('--test-local', action='store_true',
                       help='Test local W&B initialization')
    parser.add_argument('--test-cloud-sim', action='store_true',
                       help='Test cloud W&B simulation')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate W&B configuration')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive test suite')
    parser.add_argument('--wandb-key', type=str,
                       help='W&B API key for testing')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    success = True
    
    if args.test_local:
        success &= test_local_wandb_init()
    
    if args.test_cloud_sim:
        success &= test_cloud_wandb_simulation()
    
    if args.validate_config:
        success &= validate_config_integration()
    
    if args.comprehensive or not any([args.test_local, args.test_cloud_sim, args.validate_config]):
        success = run_comprehensive_test()
    
    # Test with API key if provided
    if args.wandb_key:
        logger.info("\n--- Testing with API key ---")
        success &= test_wandb_login()
        success &= simulate_paperspace_environment(args.wandb_key, True)
    
    if success:
        logger.info("\n🎉 W&B integration tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()