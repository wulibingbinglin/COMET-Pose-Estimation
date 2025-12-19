#!/bin/bash

echo "Starting Installation..."

# 1. Install Basic Requirements
pip install -r requirements.txt

# 2. Install PyTorch3D (Efficiently via Conda if available, otherwise strict pip)
# 尝试使用 conda 安装 PyTorch3D，这通常是最稳的
if command -v conda &> /dev/null; then
    echo "Conda detected, installing PyTorch3D via conda..."
    conda install pytorch3d=0.7.5 -c pytorch3d -y
else
    echo "Conda not found. Installing PyTorch3D via pip (this might take a while)..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# 3. Install LightGlue (Local Clone)
echo "Installing LightGlue..."
if [ ! -d "dependency/LightGlue" ]; then
    mkdir -p dependency
    git clone https://github.com/jytime/LightGlue.git dependency/LightGlue
    cd dependency/LightGlue
    python -m pip install -e .
    cd ../..
else
    echo "LightGlue already exists."
fi

echo "Installation Complete!"