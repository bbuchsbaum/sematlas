#!/bin/bash
#
# Create RunPod deployment package with code and subset data
#

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Creating RunPod deployment package...${NC}"

# Create a temporary directory for the package
mkdir -p runpod_package
cp -r src scripts configs notebooks requirements.txt *.md dvc.yaml train.py runpod_package/ 2>/dev/null || true

# Copy only the subset data (small enough to transfer)
mkdir -p runpod_package/data/raw
cp data/raw/neurosynth_subset_1k.pkl.gz runpod_package/data/raw/

# Copy essential processed data (small files only)
mkdir -p runpod_package/data/processed
cp data/processed/*.csv runpod_package/data/processed/ 2>/dev/null || true
cp data/processed/*.json runpod_package/data/processed/ 2>/dev/null || true

# Create the archive
tar -czf runpod_package.tar.gz -C runpod_package .

# Clean up
rm -rf runpod_package

# Show package info
SIZE=$(ls -lh runpod_package.tar.gz | awk '{print $5}')
echo -e "${GREEN}Package created: runpod_package.tar.gz (${SIZE})${NC}"
echo ""
echo "This package includes:"
echo "- All source code"
echo "- Configuration files"
echo "- Neurosynth subset data (1k studies)"
echo "- Pre-processed CSV files"
echo ""
echo "The pod will only need to:"
echo "1. Install dependencies"
echo "2. Create the volumetric cache locally"
echo "3. Start training"