#!/usr/bin/env python3
"""
Create RunPod pod with compatible CUDA version
"""

import os
import sys
import runpod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API key
runpod.api_key = os.getenv('RUNPOD_API_KEY')

if not runpod.api_key:
    print("Error: RUNPOD_API_KEY not found in environment")
    sys.exit(1)

print("Creating RunPod pod with compatible CUDA version...")

# Use an older, more compatible PyTorch image
compatible_images = [
    "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",  # CUDA 11.8 - most compatible
    "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",  # Also CUDA 11.8
    "runpod/pytorch:2.2.0-py3.10-cuda12.1.0-devel-ubuntu22.04",  # CUDA 12.1 - newer but not 12.8
]

pod_config = {
    "name": "sematlas-gpu-training",
    "image_name": compatible_images[0],  # Using CUDA 11.8 for maximum compatibility
    "gpu_type_id": "NVIDIA GeForce RTX 4090",
    "gpu_count": 1,
    "volume_in_gb": 50,
    "container_disk_in_gb": 50,
    "min_vcpu_count": 8,
    "min_memory_in_gb": 32,
    "support_public_ip": True,
    "start_ssh": True,
    "env": {
        "JUPYTER_PASSWORD": "sematlas2024",
        "WANDB_API_KEY": os.getenv('WANDB_API_KEY', '')
    }
}

try:
    # Create pod
    print(f"\nCreating pod with image: {pod_config['image_name']}")
    print("This uses CUDA 11.8 which is compatible with most GPUs")
    
    new_pod = runpod.create_pod(**pod_config)
    
    print(f"\nSuccess! Pod created:")
    print(f"Pod ID: {new_pod['id']}")
    print(f"Name: {new_pod['name']}")
    print(f"GPU: {pod_config['gpu_count']} x {pod_config['gpu_type_id']}")
    print(f"Image: {pod_config['image_name']}")
    print(f"\nJupyter password: sematlas2024")
    print(f"\nWait for pod to be 'RUNNING' before connecting")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nIf this fails, try creating manually with these settings:")
    print(f"- Image: {compatible_images[0]}")
    print("- GPU: RTX 4090 (1x)")
    print("- Container Disk: 50GB")
    print("- Volume: 50GB")