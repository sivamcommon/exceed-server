#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up Jetson Orin NX for YOLO + RF-DETR ==="

# Base packages
pip install --upgrade pip

# Core ML
pip install numpy==1.24.4
pip install torch==2.10.0 torchvision==0.25.0

# Ultralytics YOLO
pip install ultralytics==8.4.41
pip install ultralytics-thop==2.0.19
pip install onnx==1.21.0
pip install onnxruntime==1.23.2
pip install onnxslim==0.1.91

# RF-DETR
pip install rfdetr==1.6.5
pip install transformers==5.6.2
pip install tokenizers==0.22.2
pip install huggingface_hub==1.12.0
pip install peft==0.19.1
pip install accelerate==1.13.0
pip install safetensors==0.7.0

# Supporting
pip install supervision==0.27.0.post2
pip install pydantic==2.13.3
pip install opencv-python==4.11.0.86
pip install scipy==1.15.3
pip install scikit-learn==1.7.2
pip install pandas==1.3.5
pip install matplotlib==3.10.9
pip install tqdm==4.67.3
pip install pillow==12.2.0

# TensorRT (already comes with JetPack)
export PATH=$PATH:/usr/src/tensorrt/bin

echo ""
echo "=== Verifying ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python3 -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python3 -c "from rfdetr import RFDETR; print('RF-DETR OK')"
echo "=== Setup Complete ==="
