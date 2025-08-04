#!/usr/bin/env python3
"""
RunPod GPU training orchestration script.

This script handles training on RunPod with proper environment setup,
data synchronization, and remote execution management.

Usage:
    python scripts/train_runpod.py --config configs/baseline_vae.yaml
    python scripts/train_runpod.py --config configs/baseline_vae.yaml --gpu-type "RTX 4090"
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import runpod


def setup_logging() -> logging.Logger:
    """Setup logging for RunPod orchestration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('runpod_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_runpod_auth() -> bool:
    """Check if RunPod API is properly authenticated."""
    api_key = os.environ.get('RUNPOD_API_KEY')
    if not api_key:
        return False
    
    # Set the API key for runpod
    runpod.api_key = api_key
    
    try:
        # Test authentication by getting pods
        pods = runpod.get_pods()
        return True
    except Exception as e:
        logging.error(f"RunPod authentication failed: {e}")
        return False


def get_gpu_pricing() -> Dict[str, float]:
    """Get GPU pricing information."""
    return {
        "NVIDIA GeForce RTX 4090": 0.34,
        "NVIDIA RTX A4000": 0.56,
        "NVIDIA RTX A5000": 0.82,
        "NVIDIA RTX A6000": 1.89,
        "NVIDIA A40": 0.76,
        "NVIDIA A100 80GB PCIe": 1.29,
        "NVIDIA A100-SXM4-80GB": 1.89,
        "NVIDIA H100 80GB HBM3": 3.58,
        "NVIDIA V100": 0.79,
        "NVIDIA L4": 0.29,
        "NVIDIA L40": 0.69,
    }


def create_startup_command(config_path: str, max_epochs: Optional[int] = None) -> str:
    """Create the startup command for the RunPod container."""
    # For RunPod, docker_args should be a simple command, not a full script
    # The container will run this command directly
    if max_epochs:
        cmd = f"cd /workspace && pip install -r requirements.txt && pip install lmdb && python train.py --config {config_path} --max-epochs {max_epochs}"
    else:
        cmd = f"cd /workspace && pip install -r requirements.txt && pip install lmdb && python train.py --config {config_path}"
    
    return cmd


def prepare_pod_config(
    config_path: str,
    gpu_type: str,
    test_run: bool,
    volume_size: int = 20
) -> Dict[str, Any]:
    """Prepare the pod configuration."""
    
    # Create startup command
    max_epochs = 2 if test_run else None
    startup_command = create_startup_command(config_path, max_epochs)
    
    # Pod configuration
    pod_config = {
        "name": f"sematlas-{'test' if test_run else 'training'}-{int(time.time())}",
        "image_name": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpu_type_id": gpu_type,
        "cloud_type": "SECURE",  # or "COMMUNITY" for cheaper
        "volume_in_gb": volume_size,  # Correct parameter name
        "volume_mount_path": "/workspace",
        "docker_args": startup_command,
        "support_public_ip": True,
        "data_center_id": None,  # Let RunPod choose
        "country_code": None,
        "min_vcpu_count": 4,
        "min_memory_in_gb": 16,  # Correct parameter name
        "gpu_count": 1,  # Correct parameter name
    }
    
    return pod_config


def upload_project_files(pod_id: str, logger: logging.Logger) -> bool:
    """Upload project files to the pod (simplified for RunPod)."""
    logger.info("Note: RunPod requires manual file upload or git clone inside the pod")
    logger.info("You can access the pod terminal to upload files or clone your repository")
    return True


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description='RunPod GPU training orchestration')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--gpu-type', type=str, default='NVIDIA GeForce RTX 4090',
                        help='GPU type (e.g., "NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe")')
    parser.add_argument('--test-run', action='store_true',
                        help='Run a quick test with 2 epochs')
    parser.add_argument('--no-submit', action='store_true',
                        help='Prepare but do not submit job')
    parser.add_argument('--volume-size', type=int, default=20,
                        help='Volume size in GB (default: 20)')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=== RunPod Training Orchestration ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"GPU type: {args.gpu_type}")
    logger.info(f"Test run: {args.test_run}")
    
    # Check authentication
    if not check_runpod_auth():
        logger.error("RunPod authentication failed. Please set RUNPOD_API_KEY environment variable.")
        sys.exit(1)
    
    logger.info("RunPod API authenticated successfully")
    
    # Get pricing
    pricing = get_gpu_pricing()
    gpu_price = pricing.get(args.gpu_type, 1.0)
    duration_hours = 0.25 if args.test_run else 2.0
    estimated_cost = gpu_price * duration_hours
    
    logger.info(f"Estimated cost: ${estimated_cost:.2f} ({gpu_price}/hour × {duration_hours}h)")
    
    if args.no_submit:
        logger.info("Dry run mode - not submitting job")
        return
    
    # Prepare pod configuration
    pod_config = prepare_pod_config(
        args.config,
        args.gpu_type,
        args.test_run,
        args.volume_size
    )
    
    logger.info("Creating RunPod pod...")
    logger.info(f"Pod name: {pod_config['name']}")
    logger.info(f"Container: {pod_config['image_name']}")
    
    try:
        # Create the pod
        pod = runpod.create_pod(**pod_config)
        pod_id = pod['id']
        
        logger.info(f"Pod created successfully!")
        logger.info(f"Pod ID: {pod_id}")
        logger.info(f"Pod Status: {pod.get('status', 'CREATING')}")
        
        # Wait for pod to be ready
        logger.info("Waiting for pod to be ready...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            pod_info = runpod.get_pod(pod_id)
            status = pod_info.get('status', 'UNKNOWN')
            
            if status == 'RUNNING':
                logger.info("Pod is running!")
                break
            elif status in ['FAILED', 'TERMINATED']:
                logger.error(f"Pod failed to start: {status}")
                sys.exit(1)
            
            time.sleep(10)
        
        # Get SSH access info
        logger.info("\n=== Pod Access Information ===")
        logger.info(f"Pod ID: {pod_id}")
        logger.info(f"SSH Command: ssh root@{pod.get('public_ip', 'N/A')} -p {pod.get('port', 22)}")
        logger.info("\n=== Next Steps ===")
        logger.info("1. SSH into the pod using the command above")
        logger.info("2. Clone your repository: git clone <your-repo-url>")
        logger.info("3. Or upload files using SCP")
        logger.info("4. The startup script will run automatically")
        logger.info("\n=== Monitoring ===")
        logger.info(f"View logs: runpod logs {pod_id}")
        logger.info(f"Stop pod: runpod stop {pod_id}")
        logger.info(f"Terminate pod: runpod terminate {pod_id}")
        
    except Exception as e:
        logger.error(f"Failed to create pod: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()