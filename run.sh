#!/bin/bash
set -e  # Stop script immediately if any command fails

# --- CONFIGURATION (EDIT THESE) ---
REPO_URL="https://github.com/isamabdullah88/NequIP.git"
WANDB_API_KEY="PASTE_YOUR_WANDB_KEY_HERE"  # Or pass it as env var
DATA_DOWNLOAD_URL="https://drive.google.com/uc?export=download&id=1LyzVyRgdE0H2EFlU6xpOpPscO4GSBLWf"  # Optional: Direct link to your .npz data (Dropbox/Drive)
# ----------------------------------

echo ">>> [1/6] System Update & Essentials..."
apt-get update && apt-get install -y git wget unzip

# echo ">>> [2/6] Cloning Repository..."
# if [ -d "NequIP" ]; then
#     echo "Repo already exists, pulling latest changes..."
#     cd NequIP && git pull && cd ..
# else
#     git clone $REPO_URL
# fi

echo ">>> [3/6] Detecting Environment for PyTorch Geometric..."
# Automatically detect PyTorch and CUDA versions to build the correct wheel URL
PT_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")
# If CUDA is 12.x, PyG wheels usually use 'cu121' convention
if [[ $CUDA_VER == "12"* ]]; then CUDA_SUFFIX="cu121"; else CUDA_SUFFIX="cu${CUDA_VER}"; fi

WHEEL_URL="https://data.pyg.org/whl/torch-${PT_VER}+${CUDA_SUFFIX}.html"
echo "    -> Detected PyTorch: $PT_VER, CUDA: $CUDA_SUFFIX"
echo "    -> Using Wheel URL: $WHEEL_URL"

echo ">>> [4/6] Installing Python Dependencies..."
# Install the difficult GNN libraries using the correct wheel URL
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WHEEL_URL
pip install torch_geometric e3nn wandb

echo ">>> [5/6] Setting up Data..."
cd NequIP
mkdir -p data results

# OPTION A: Download Data (If you have a link)
if [ ! -z "$DATA_DOWNLOAD_URL" ]; then
    echo "Downloading data from URL..."
    wget -O data/dataset.npz "$DATA_DOWNLOAD_URL"
    DATA_PATH="data/dataset.npz"

# OPTION B: Check if you uploaded it manually via SCP
elif [ -f "../dataset.npz" ]; then
    echo "Found uploaded data, moving it..."
    mv "../dataset.npz" data/
    DATA_PATH="data/dataset.npz"

else
    echo "⚠️ WARNING: No data found! Attempting to run with 'md17' placeholder..."
    echo "Please upload your .npz file to the server if needed."
    DATA_PATH="data" # Adjust if your code expects a folder
fi

echo ">>> [6/6] Starting Training..."
# Log in to WandB non-interactively
export WANDB_API_KEY=$WANDB_API_KEY

# Run the training
# Note: Removed --kaggle True, assuming you adjust defaults for Linux paths
python train.py \
    --data_dir "$DATA_PATH" \
    --results_dir "results" \
    --batch_size 32 \
    --epochs 100

echo ">>> Done! Training Complete."