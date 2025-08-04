# Quick Start Training on RunPod

Your pod is running! Here's how to get training started:

## Step 1: Connect to Pod
Click "Connect" → "Connect via Browser" (Web Terminal)

## Step 2: Setup Commands
Copy and paste these commands into the web terminal:

```bash
# Navigate to workspace
cd /workspace

# Download the package (Option A: Using transfer.sh)
# First, on your local machine, upload the file:
# curl --upload-file runpod_package.tar.gz https://transfer.sh/runpod_package.tar.gz
# Then in RunPod, download it with the URL you get

# Option B: Create setup script
cat > setup_sematlas.sh << 'EOF'
#!/bin/bash
cd /workspace
mkdir -p sematlas && cd sematlas

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning wandb nibabel nilearn matplotlib numpy pandas scipy scikit-learn lmdb h5py dvc

# Create directory structure
mkdir -p src/models src/data src/training src/inference src/evaluation
mkdir -p configs scripts notebooks data/raw data/processed

echo "Environment ready! Upload your code files next."
EOF

chmod +x setup_sematlas.sh
./setup_sematlas.sh
```

## Step 3: Transfer Code
Since runpodctl seems problematic, use one of these:

### Option A: Use transfer.sh (Recommended)
On your local machine:
```bash
curl --upload-file runpod_package.tar.gz https://transfer.sh/runpod_package.tar.gz
```

In RunPod terminal:
```bash
cd /workspace/sematlas
wget [URL_FROM_TRANSFER_SH]
tar -xzf runpod_package.tar.gz
```

### Option B: Python HTTP Server
On your local machine:
```bash
cd /Users/bbuchsbaum/code/sematlas
python3 -m http.server 8888
```

In RunPod terminal:
```bash
cd /workspace/sematlas
wget http://YOUR_LOCAL_IP:8888/runpod_package.tar.gz
tar -xzf runpod_package.tar.gz
```

## Step 4: Start Training
```bash
cd /workspace/sematlas
pip install -r requirements.txt
pip install lmdb

# Create volumetric cache from included subset data
python scripts/create_volumetric_cache.py --data-source neurosynth_subset_1k

# Start 2-epoch test training
python train.py --config configs/baseline_vae.yaml --max-epochs 2

# Monitor GPU usage in another terminal
watch nvidia-smi
```

## Monitoring
- Training progress will show in terminal
- GPU usage: `nvidia-smi`
- W&B dashboard: https://wandb.ai (if configured)

## Tips
- The pod has 41GB RAM and 24GB GPU memory
- Training should take ~15-30 minutes for 2 epochs
- Save checkpoints frequently
- Cost is $0.69/hr - monitor your usage!