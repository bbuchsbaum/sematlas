#!/bin/bash
#
# Sync code and data to RunPod pod
#
# This script helps transfer the necessary files to a RunPod pod for training.
# It uses rsync to efficiently copy only the required files.
#
# Usage:
#     ./scripts/sync_to_runpod.sh <POD_ID>
#

set -e

# Check if pod ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <POD_ID>"
    echo "Example: $0 y8rgiwnfy0kpua"
    exit 1
fi

POD_ID=$1

# Get SSH details from pod
echo "Getting SSH details for pod: $POD_ID"

SSH_INFO=$(python3 << EOF
import os
import runpod

runpod.api_key = os.environ.get('RUNPOD_API_KEY')
try:
    pod = runpod.get_pod('$POD_ID')
    if pod and pod.get('public_ip'):
        print(f"{pod['public_ip']} {pod.get('port', 22)}")
    else:
        print("ERROR")
except Exception as e:
    print("ERROR")
EOF
)

if [ "$SSH_INFO" = "ERROR" ]; then
    echo "Error: Could not get SSH details for pod $POD_ID"
    echo "Make sure the pod is running and has a public IP"
    exit 1
fi

SSH_IP=$(echo $SSH_INFO | cut -d' ' -f1)
SSH_PORT=$(echo $SSH_INFO | cut -d' ' -f2)

echo "SSH IP: $SSH_IP"
echo "SSH Port: $SSH_PORT"

# Create workspace directory on pod
echo "Creating workspace directory on pod..."
ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_IP "mkdir -p /workspace/sematlas"

# Sync code files (excluding large data and cache)
echo "Syncing code files..."
rsync -avz --progress \
    --exclude='data/processed/volumetric_cache*' \
    --exclude='data/raw/*.pkl.gz' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='lightning_logs' \
    --exclude='wandb' \
    --exclude='*.ckpt' \
    --exclude='validation_outputs' \
    --exclude='test_outputs' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    ./ root@$SSH_IP:/workspace/sematlas/

# Check if we need to sync data
echo ""
echo "Code sync complete!"
echo ""
echo "For data sync, you have two options:"
echo "1. Use the smaller subset data (recommended for testing):"
echo "   ssh -p $SSH_PORT root@$SSH_IP"
echo "   cd /workspace/sematlas"
echo "   python scripts/download_neurosynth_with_subset.py --subset-only"
echo ""
echo "2. Sync the full LMDB cache (20GB+, will take time):"
echo "   rsync -avz --progress -e \"ssh -p $SSH_PORT\" data/processed/volumetric_cache.lmdb root@$SSH_IP:/workspace/sematlas/data/processed/"
echo ""
echo "To connect to the pod:"
echo "   ssh -p $SSH_PORT root@$SSH_IP"
echo ""
echo "To start training after data is ready:"
echo "   cd /workspace/sematlas"
echo "   pip install -r requirements.txt"
echo "   pip install lmdb"
echo "   python train.py --config configs/baseline_vae.yaml --max-epochs 2"