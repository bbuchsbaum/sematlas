#!/usr/bin/env python3
"""
Paperspace GPU training orchestration script.

This script handles training on Paperspace Gradient with proper environment setup,
data synchronization, and remote execution management.

Usage:
    python scripts/train_paperspace.py --config configs/baseline_vae.yaml
    python scripts/train_paperspace.py --config configs/baseline_vae.yaml --instance-type "P5000"
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List


def setup_logging() -> logging.Logger:
    """Setup logging for Paperspace orchestration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('paperspace_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_paperspace_cli() -> bool:
    """Check if Paperspace CLI is available and authenticated."""
    try:
        result = subprocess.run(['gradient', 'apiKey', 'show'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_paperspace_projects() -> List[Dict[str, Any]]:
    """Get available Paperspace projects."""
    try:
        result = subprocess.run(['gradient', 'projects', 'list', '--json'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def create_training_script(config_path: str, 
                         wandb_api_key: Optional[str] = None,
                         test_run: bool = False) -> str:
    """Create the training script for remote execution."""
    
    # Setup W&B configuration
    wandb_env_vars = setup_wandb_config(config_path, wandb_api_key, test_run)
    
    # Create environment variable exports
    env_exports = []
    for key, value in wandb_env_vars.items():
        env_exports.append(f'export {key}="{value}"')
    
    wandb_setup = '\n'.join(env_exports) if env_exports else '# No W&B configuration'
    
    script_content = f"""#!/bin/bash
set -e

echo "=== Paperspace Training Setup ==="
echo "Config: {config_path}"
echo "Test run: {test_run}"
echo "Timestamp: $(date)"
echo

# Run Paperspace environment setup script
if [ -f "paperspace_setup.sh" ]; then
    echo "Running environment setup script..."
    bash paperspace_setup.sh
else
    echo "WARNING: paperspace_setup.sh not found, using fallback setup..."
    # Fallback setup
    export PYTHONPATH=/notebooks:$PYTHONPATH
    export TORCH_HOME=/tmp/torch_cache
    export HF_HOME=/tmp/hf_cache
    pip install --upgrade pip
    pip install -q torch torchvision pytorch-lightning wandb pyyaml
fi

# Set up W&B configuration
echo "Setting up Weights & Biases..."
{wandb_setup}

# Login to W&B if online mode
if [ "$WANDB_MODE" = "online" ] && [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to W&B..."
    wandb login
else
    echo "W&B running in offline mode or no API key provided"
fi

# Create necessary directories
mkdir -p checkpoints wandb_logs

# Start training
echo "=== Starting Training ==="
"""
    
    if test_run:
        script_content += f"python train.py --config {config_path} --test-run --log-level INFO\n"
    else:
        script_content += f"python train.py --config {config_path} --log-level INFO\n"
    
    script_content += """
echo "=== Training Complete ==="
echo "Timestamp: $(date)"

# Archive outputs
echo "Archiving outputs..."
tar -czf training_outputs.tar.gz checkpoints/ wandb_logs/ training.log || true

echo "Training script finished."
"""
    
    return script_content


def submit_paperspace_job(config_path: str,
                         project_id: str,
                         instance_type: str = "P4000",
                         wandb_api_key: Optional[str] = None,
                         test_run: bool = False) -> Optional[str]:
    """Submit training job to Paperspace."""
    
    logger = logging.getLogger(__name__)
    
    # Create training script
    training_script = create_training_script(config_path, wandb_api_key, test_run)
    script_path = "paperspace_train.sh"
    
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Prepare job command
    job_name = f"brain-atlas-{'test' if test_run else 'train'}-{os.path.basename(config_path).split('.')[0]}"
    
    cmd = [
        'gradient', 'jobs', 'create',
        '--name', job_name,
        '--projectId', project_id,
        '--machineType', instance_type,
        '--container', 'pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel',
        '--command', f'bash /notebooks/{script_path}',
        '--workspace', '.',
        '--json'
    ]
    
    try:
        logger.info(f"Submitting job: {job_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            job_info = json.loads(result.stdout)
            job_id = job_info.get('id')
            logger.info(f"Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            logger.error(f"Job submission failed: {result.stderr}")
            return None
            
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.error(f"Error submitting job: {e}")
        return None
    
    finally:
        # Clean up script file
        if os.path.exists(script_path):
            os.remove(script_path)


def monitor_job(job_id: str) -> None:
    """Monitor Paperspace job status."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Monitoring job {job_id}...")
    logger.info("Use 'gradient jobs logs --id {job_id}' to view logs")
    logger.info("Use 'gradient jobs stop --id {job_id}' to stop the job")
    
    # Show basic job info
    try:
        result = subprocess.run(['gradient', 'jobs', 'show', '--id', job_id, '--json'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            job_info = json.loads(result.stdout)
            logger.info(f"Job status: {job_info.get('state', 'Unknown')}")
            logger.info(f"Machine type: {job_info.get('machineType', 'Unknown')}")
    except Exception as e:
        logger.warning(f"Could not fetch job info: {e}")


def setup_wandb_config(config_path: str, wandb_api_key: Optional[str], test_run: bool) -> Dict[str, str]:
    """Setup W&B configuration for Paperspace training."""
    import yaml
    
    logger = logging.getLogger(__name__)
    
    # Load config to get W&B settings
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}
    
    wandb_config = config.get('wandb', {})
    
    # Environment variables for Paperspace
    env_vars = {}
    
    if wandb_api_key:
        env_vars['WANDB_API_KEY'] = wandb_api_key
        env_vars['WANDB_MODE'] = 'online'
    else:
        logger.warning("No W&B API key provided - running in offline mode")
        env_vars['WANDB_MODE'] = 'offline'
    
    # Set project and run name
    project_name = wandb_config.get('project', 'generative-brain-atlas')
    base_run_name = wandb_config.get('name', 'paperspace-training')
    
    # Add paperspace and test indicators to run name
    run_name_parts = [base_run_name, 'paperspace']
    if test_run:
        run_name_parts.append('test')
    
    env_vars['WANDB_PROJECT'] = project_name
    env_vars['WANDB_RUN_NAME'] = '-'.join(run_name_parts)
    env_vars['WANDB_TAGS'] = ','.join(wandb_config.get('tags', []) + ['paperspace'])
    env_vars['WANDB_NOTES'] = f"Paperspace training - {wandb_config.get('notes', '')}"
    
    # Set save directory
    env_vars['WANDB_DIR'] = wandb_config.get('save_dir', 'wandb_logs')
    
    logger.info(f"W&B Configuration:")
    logger.info(f"  Project: {project_name}")
    logger.info(f"  Run Name: {env_vars['WANDB_RUN_NAME']}")
    logger.info(f"  Mode: {env_vars['WANDB_MODE']}")
    logger.info(f"  Tags: {env_vars['WANDB_TAGS']}")
    
    return env_vars


def prepare_environment_sync() -> bool:
    """Prepare environment synchronization before job submission."""
    logger = logging.getLogger(__name__)
    
    logger.info("Preparing environment synchronization...")
    
    # Run sync environment preparation
    try:
        result = subprocess.run([
            'python', 'scripts/sync_environment.py', '--prepare'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("Environment sync preparation completed")
            return True
        else:
            logger.error(f"Environment sync preparation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Environment sync preparation timed out")
        return False
    except Exception as e:
        logger.error(f"Error preparing environment sync: {e}")
        return False


def main():
    """Main Paperspace training orchestration."""
    parser = argparse.ArgumentParser(description='Run training on Paperspace GPU')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--project-id', type=str,
                       help='Paperspace project ID (will auto-detect if not provided)')
    parser.add_argument('--instance-type', type=str, default='P4000',
                       choices=['P4000', 'P5000', 'P6000', 'V100', 'P100'],
                       help='Paperspace instance type')
    parser.add_argument('--wandb-key', type=str,
                       help='Weights & Biases API key (or set WANDB_API_KEY env var)')
    parser.add_argument('--test-run', action='store_true',
                       help='Run a quick test with 2 epochs')
    parser.add_argument('--no-submit', action='store_true',
                       help='Prepare but don\'t submit the job')
    parser.add_argument('--skip-sync', action='store_true',
                       help='Skip environment synchronization preparation')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    logger.info("=== Paperspace Training Orchestration ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Instance type: {args.instance_type}")
    logger.info(f"Test run: {args.test_run}")
    
    # Check Paperspace CLI
    if not check_paperspace_cli():
        logger.error("Paperspace CLI not available or not authenticated.")
        logger.error("Please install and authenticate: 'pip install gradient' and 'gradient apiKey set <key>'")
        return 1
    
    logger.info("Paperspace CLI available and authenticated")
    
    # Get project ID (skip if doing dry run)
    project_id = args.project_id
    if not project_id and not args.no_submit:
        projects = get_paperspace_projects()
        if projects:
            # Use first available project
            project_id = projects[0]['id']
            logger.info(f"Using project: {projects[0]['name']} ({project_id})")
        else:
            logger.error("No Paperspace projects found. Please create a project first.")
            return 1
    elif not project_id and args.no_submit:
        project_id = "dummy-project-id"  # For dry run
        logger.info("Using dummy project ID for dry run")
    
    # Get W&B API key
    wandb_api_key = args.wandb_key or os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        logger.warning("No W&B API key provided. Training will run without W&B logging.")
    
    # Prepare environment synchronization
    if not args.skip_sync:
        if not prepare_environment_sync():
            logger.error("Environment sync preparation failed. Use --skip-sync to bypass.")
            return 1
    else:
        logger.warning("Skipping environment synchronization preparation")
    
    if args.no_submit:
        logger.info("Preparation complete. Use --submit to actually run the job.")
        return 0
    
    # Submit job
    job_id = submit_paperspace_job(
        config_path=args.config,
        project_id=project_id,
        instance_type=args.instance_type,
        wandb_api_key=wandb_api_key,
        test_run=args.test_run
    )
    
    if job_id:
        monitor_job(job_id)
        return 0
    else:
        logger.error("Failed to submit job")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)