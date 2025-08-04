#!/bin/bash
#
# Quick start commands for RunPod
# Copy and paste these into your RunPod terminal
#

cat << 'EOF'
# RunPod Quick Start Commands
# ==========================

# 1. Setup workspace
cd /workspace
rm -rf sematlas
mkdir sematlas
cd sematlas

# 2. Create setup script
cat > setup.sh << 'SCRIPT'
#!/bin/bash
echo "Setting up Sematlas environment..."

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning wandb nibabel nilearn matplotlib numpy pandas scipy scikit-learn lmdb h5py

# Create directory structure
mkdir -p src/models src/data src/training src/inference src/evaluation
mkdir -p configs scripts notebooks
mkdir -p data/raw data/processed

# Download subset data directly
echo "Downloading Neurosynth subset data..."
cd data/raw
wget https://raw.githubusercontent.com/neurostuff/NiMARE/main/nimare/tests/data/neurosynth_dset_first500.pkl.gz -O neurosynth_subset.pkl.gz || echo "Using local data"
cd ../..

echo "Setup complete! Ready to receive code files."
SCRIPT

chmod +x setup.sh
./setup.sh

# 3. After setup, you'll need to transfer the code files
# Use runpodctl receive or paste key files manually

EOF

echo ""
echo "Commands saved above. You can also:"
echo "1. Start a local HTTP server: python3 -m http.server 8888"
echo "2. In the pod: wget http://YOUR_LOCAL_IP:8888/runpod_package.tar.gz"
echo "3. Extract: tar -xzf runpod_package.tar.gz"