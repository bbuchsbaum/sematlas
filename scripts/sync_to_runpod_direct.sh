#!/bin/bash
#
# Direct sync script for RunPod using their SSH format
#
# Usage:
#     ./scripts/sync_to_runpod_direct.sh
#

set -e

# RunPod SSH details
SSH_USER="y8rgiwnfy0kpua-64411462"
SSH_HOST="ssh.runpod.io"
SSH_KEY="~/.ssh/runpod_key"

echo "Using RunPod SSH connection: $SSH_USER@$SSH_HOST"

# Test SSH connection
echo "Testing SSH connection..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY $SSH_USER@$SSH_HOST "echo 'SSH connection successful'" || {
    echo "Error: Could not connect to pod via SSH"
    exit 1
}

# Create workspace directory on pod
echo "Creating workspace directory on pod..."
ssh -o StrictHostKeyChecking=no -i $SSH_KEY $SSH_USER@$SSH_HOST "mkdir -p /workspace/sematlas"

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
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    ./ $SSH_USER@$SSH_HOST:/workspace/sematlas/

echo ""
echo "✅ Code sync complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into the pod:"
echo "   ssh -i ~/.ssh/id_ed25519 $SSH_USER@$SSH_HOST"
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
echo "   # Create volumetric cache for subset"
echo "   python scripts/create_volumetric_cache.py --data-source neurosynth_subset_1k"
echo "   # Run 2 epoch test"
echo "   python train.py --config configs/baseline_vae.yaml --max-epochs 2"
echo ""
echo "5. Monitor training with W&B at: https://wandb.ai"