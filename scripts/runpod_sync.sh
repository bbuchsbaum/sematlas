#!/bin/bash
#
# RunPod Sync Script using runpodctl
# 
# This script provides a fully command-line driven workflow for syncing
# code to RunPod pods using runpodctl's send/receive functionality.
#
# Usage:
#     ./scripts/runpod_sync.sh
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if runpodctl is available
check_runpodctl() {
    if ! command -v runpodctl &> /dev/null && ! [ -x "./runpodctl" ]; then
        log_error "runpodctl not found!"
        log_info "Please install runpodctl first:"
        log_info "curl -L https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-arm64 -o runpodctl"
        log_info "chmod +x runpodctl"
        exit 1
    fi
    
    # Use local runpodctl if available
    if [ -x "./runpodctl" ]; then
        RUNPODCTL="./runpodctl"
    else
        RUNPODCTL="runpodctl"
    fi
}

# Create code archive
create_archive() {
    log_info "Creating code archive..."
    
    tar -czf sematlas_code.tar.gz \
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
        --exclude='runpodctl' \
        train.py \
        src/ \
        scripts/ \
        configs/ \
        notebooks/ \
        requirements.txt \
        *.md \
        dvc.yaml 2>/dev/null || true
    
    local size=$(ls -lh sematlas_code.tar.gz | awk '{print $5}')
    log_success "Archive created: sematlas_code.tar.gz (${size})"
}

# Send file using runpodctl
send_code() {
    log_info "Sending code archive to RunPod..."
    
    # Run runpodctl send and capture output
    local output=$($RUNPODCTL send sematlas_code.tar.gz 2>&1)
    
    # Extract the receive code from output
    local receive_code=$(echo "$output" | grep -oE '[0-9]+-[a-z]+-[a-z]+-[a-z]+' | head -1)
    
    if [ -z "$receive_code" ]; then
        log_error "Failed to get receive code from runpodctl"
        echo "$output"
        exit 1
    fi
    
    log_success "File sent successfully!"
    log_info "Receive code: ${GREEN}${receive_code}${NC}"
    
    echo "$receive_code" > .runpod_receive_code
    echo "$receive_code"
}

# Generate pod setup commands
generate_setup_commands() {
    local receive_code=$1
    local epochs=${2:-2}
    
    cat << EOF

${GREEN}=== RunPod Setup Commands ===${NC}

1. Connect to your RunPod pod:
   - Via web console: Click "Connect" -> "Connect via Browser"
   - Via SSH: ssh -i ~/.ssh/runpod_key y8rgiwnfy0kpua-64411462@ssh.runpod.io

2. Once connected, run these commands:

${YELLOW}# Navigate to workspace${NC}
cd /workspace

${YELLOW}# Receive the code archive${NC}
runpodctl receive ${receive_code}

${YELLOW}# Extract the archive${NC}
tar -xzf sematlas_code.tar.gz
cd sematlas

${YELLOW}# Install dependencies${NC}
pip install -r requirements.txt
pip install lmdb

${YELLOW}# Download subset data (for testing)${NC}
python scripts/download_neurosynth_with_subset.py --subset-only

${YELLOW}# Create volumetric cache${NC}
python scripts/create_volumetric_cache.py --data-source neurosynth_subset_1k

${YELLOW}# Start training (${epochs} epochs)${NC}
python train.py --config configs/baseline_vae.yaml --max-epochs ${epochs}

${GREEN}=== Alternative: One-Line Setup ===${NC}

You can run all commands at once:

cd /workspace && runpodctl receive ${receive_code} && tar -xzf sematlas_code.tar.gz && cd sematlas && pip install -r requirements.txt && pip install lmdb && python scripts/download_neurosynth_with_subset.py --subset-only && python scripts/create_volumetric_cache.py --data-source neurosynth_subset_1k && python train.py --config configs/baseline_vae.yaml --max-epochs ${epochs}

${GREEN}=== Monitoring Training ===${NC}

- Training logs will appear in the terminal
- W&B dashboard: https://wandb.ai
- Check GPU usage: nvidia-smi

EOF
}

# Main execution
main() {
    log_info "Starting RunPod sync process..."
    
    # Check runpodctl
    check_runpodctl
    
    # Create archive
    create_archive
    
    # Send code
    local receive_code=$(send_code)
    
    # Generate setup commands
    generate_setup_commands "$receive_code" "${1:-2}"
    
    # Save commands to file for reference
    generate_setup_commands "$receive_code" "${1:-2}" > runpod_setup_commands.txt
    log_info "Setup commands saved to: runpod_setup_commands.txt"
    
    log_success "Sync process complete!"
    log_info "The receive code is valid for ~24 hours"
    log_info "You can re-use it multiple times if needed"
}

# Run main with optional epoch argument
main "$@"