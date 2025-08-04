#!/usr/bin/env python3
"""
Create a properly configured RunPod pod with GPU
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

# Create pod with proper GPU configuration
print("Creating new RunPod pod with GPU...")

pod_config = {
    "name": "sematlas-gpu-training",
    "image_name": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "gpu_type_id": "NVIDIA GeForce RTX 4090",
    "gpu_count": 1,  # THIS IS CRITICAL - Must be 1 or more!
    "volume_in_gb": 50,
    "container_disk_in_gb": 50,
    "min_vcpu_count": 8,
    "min_memory_in_gb": 32,
    "support_public_ip": True,
    "start_ssh": True,
    "env": {
        "JUPYTER_PASSWORD": "sematlas2024"
    }
}

try:
    # First, list current pods
    print("\nCurrent pods:")
    pods = runpod.get_pods()
    for pod in pods:
        print(f"- {pod['name']}: {pod.get('gpu_count', 0)} x {pod.get('gpu_type_id', 'No GPU')}")
    
    # Terminate the broken pod
    if pods:
        for pod in pods:
            if pod['gpu_count'] == 0:
                print(f"\nTerminating pod without GPU: {pod['id']}")
                runpod.terminate_pod(pod['id'])
    
    # Create new pod
    print("\nCreating new pod with GPU...")
    new_pod = runpod.create_pod(**pod_config)
    
    print(f"\nSuccess! New pod created:")
    print(f"Pod ID: {new_pod['id']}")
    print(f"Name: {new_pod['name']}")
    print(f"GPU: {pod_config['gpu_count']} x {pod_config['gpu_type_id']}")
    print(f"Status: {new_pod.get('desiredStatus', 'Starting')}")
    print(f"\nThe pod should have Jupyter available with password: sematlas2024")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure your RUNPOD_API_KEY is valid and you have credits")