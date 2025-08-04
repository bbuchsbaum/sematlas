#!/bin/bash
#
# Simple sync script for RunPod - requires manual SSH details
#
# Usage:
#     ./scripts/sync_to_runpod_simple.sh <SSH_IP> <SSH_PORT>
#
# Example:
#     ./scripts/sync_to_runpod_simple.sh 123.45.67.89 22222
#

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <SSH_IP> <SSH_PORT>"
    echo ""
    echo "To get SSH details:"
    echo "1. Log into https://www.runpod.io/console/pods"
    echo "2. Find your pod (y8rgiwnfy0kpua)"
    echo "3. Click on it to see SSH connection details"
    echo ""
    echo "Example: $0 123.45.67.89 22222"
    exit 1
fi

SSH_IP=$1
SSH_PORT=$2

echo "Using SSH connection: root@$SSH_IP:$SSH_PORT"

# Test SSH connection
echo "Testing SSH connection..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT root@$SSH_IP "echo 'SSH connection successful'" || {
    echo "Error: Could not connect to pod via SSH"
    echo "Please check:"
    echo "1. The pod is running"
    echo "2. SSH IP and port are correct"
    echo "3. You may need to accept the SSH key on first connection"
    exit 1
}

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
    --exclude='.venv*' \
    --exclude='*.log' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    ./ root@$SSH_IP:/workspace/sematlas/

echo ""
echo "✅ Code sync complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into the pod:"
echo "   ssh -p $SSH_PORT root@$SSH_IP"
echo ""
echo "2. Navigate to workspace:"
echo "   cd /workspace/sematlas"
echo ""
echo "3. Install dependencies:"
echo "   pip install -r requirements.txt"
echo "   pip install lmdb"
echo ""
echo "4. For TESTING (recommended first):"
echo "   # Download subset data only"
echo "   python scripts/download_neurosynth_with_subset.py --subset-only"
echo "   # Create volumetric cache"
echo "   python scripts/create_volumetric_cache.py"
echo "   # Run 2 epoch test"
echo "   python train.py --config configs/baseline_vae.yaml --max-epochs 2"
echo ""
echo "5. For FULL TRAINING (after test succeeds):"
echo "   # Either sync full LMDB from local (20GB+):"
echo "   # From your local machine run:"
echo "   rsync -avz --progress -e \"ssh -p $SSH_PORT\" data/processed/volumetric_cache.lmdb root@$SSH_IP:/workspace/sematlas/data/processed/"
echo "   # OR create it on the pod (slower but self-contained)"
echo ""
echo "6. Monitor training with W&B at: https://wandb.ai"