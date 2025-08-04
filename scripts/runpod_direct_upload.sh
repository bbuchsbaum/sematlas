#!/bin/bash
#
# Direct upload to RunPod using base64 encoding
# Works around SSH PTY limitations
#

set -e

echo "Preparing file upload to RunPod..."

# Create a script that will reconstruct our package on the pod
cat > upload_to_pod.sh << 'EOF'
#!/bin/bash
# This script will be executed on the RunPod pod

cd /workspace

# Create directory
mkdir -p sematlas
cd sematlas

# Receive base64 encoded package
echo "Receiving package..."
cat > package.b64 << 'PACKAGEDATA'
EOF

# Encode our package
base64 runpod_package.tar.gz >> upload_to_pod.sh

cat >> upload_to_pod.sh << 'EOF'
PACKAGEDATA

# Decode and extract
echo "Decoding package..."
base64 -d package.b64 > runpod_package.tar.gz
rm package.b64

echo "Extracting package..."
tar -xzf runpod_package.tar.gz
rm runpod_package.tar.gz

echo "Installing dependencies..."
pip install -r requirements.txt
pip install lmdb

echo "Creating volumetric cache for subset data..."
python scripts/create_volumetric_cache.py --data-source neurosynth_subset_1k

echo ""
echo "Setup complete! To start training:"
echo "cd /workspace/sematlas"
echo "python train.py --config configs/baseline_vae.yaml --max-epochs 2"
EOF

# Upload the script
echo "Uploading to RunPod..."
ssh -i ~/.ssh/runpod_key y8rgiwnfy0kpua-64411462@ssh.runpod.io 'bash -s' < upload_to_pod.sh

echo "Upload complete!"